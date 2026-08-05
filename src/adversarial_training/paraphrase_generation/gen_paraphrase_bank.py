#!/usr/bin/env python3
"""Generate paraphrased (style-varied) injection training data.

See Appendix "Paraphrase-Generation Prompts".

For each original sample x style: keep the benign units verbatim, replace only
the injection span with an LLM paraphrase in that style, re-segment with the
same split_units() the defense uses at inference, relabel the new span, and
write a file in the same schema as training_data_original.

Reuses AutoDojo's llm_utils.generate (provider layer). Caches by (sample,style)
on disk (skip if output exists) and parallelizes with a thread pool.

  python -m adversarial_training.paraphrase_generation.gen_paraphrase_bank \
    --model google/gemini-2.5-flash --workers 8
"""
import argparse, glob, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import Template

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # anonymous_artifact/

sys.path.insert(0, os.path.join(_REPO, "third_party/AutoDojo", "agentdojo",
                                "variant_generation"))
from llm_utils import generate  # noqa: E402

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize


# ── segmentation: identical to detector.variants.split_units ──────────
def split_units(text):
    out = []
    for block in re.split(r"\n+", text.replace("\\n", "\n")):
        block = block.strip()
        if not block:
            continue
        for s in sent_tokenize(block):
            s = s.strip()
            if s:
                out.append(s)
    return out


def parse_segments(raw):
    m = re.findall(r"\[(\d+)\]\s?(.*?)(?=\n\[\d+\]|$)", raw, re.DOTALL)
    seg = {int(i): c.strip() for i, c in m}
    return [seg[k] for k in sorted(seg)]


def fmt_segments(units):
    return "\n".join(f"[{i}] {u}" for i, u in enumerate(units))


# ── AutoDojo-style Wrapper: parse ───────────────────────────────────────────
def parse_wrapper(out):
    if not out:
        return None
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).replace("**", "")
    m = list(re.finditer(r"(?:^|\n)\s*Wrapper:\s*(.*?)(?=\n\s*Wrapper:|\Z)",
                         out, re.DOTALL | re.IGNORECASE))
    return m[-1].group(1).strip() if m else None


def make_sample(d, jf, in_dir, style, para):
    """Build a new training record with the injection span swapped for `para`.

    Paper's realizable-attack construction (Approach section, adversarial-
    training subsection): benign sentences stay verbatim; only the malicious
    span is replaced by an LLM paraphrase, then re-segmented and relabeled so
    sentence-level labels stay consistent with the swapped span.
    """
    units = parse_segments(d["sentence_segments"])
    s, e = d["injection_span_unit"]["start"], d["injection_span_unit"]["end"]
    pre, post = units[:s], units[e + 1:]
    pu = split_units(para)
    if not pu:
        return None
    new_units = pre + pu + post
    rec = dict(d)
    rec.pop("injected", None)  # stale after swap; embedder ignores it anyway
    rec.update({
        "injection": para,
        "style": style,
        "source_file": os.path.relpath(jf, in_dir),
        "sentence_segments": fmt_segments(new_units),
        "injection_span_unit": {"start": len(pre), "end": len(pre) + len(pu) - 1},
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=os.path.join(_REPO, "data", "training_data_original"))
    ap.add_argument("--out", default=os.path.join(_REPO, "data", "training_data_paraphrased"))
    ap.add_argument("--prompts", default=os.path.join(_HERE, "paraphrase_styles.yaml"))
    ap.add_argument("--styles", nargs="*", default=None, help="subset of styles")
    ap.add_argument("--model", default=os.getenv("PARA_MODEL", "google/gemini-2.5-flash"))
    ap.add_argument("--provider", default=os.getenv("PARA_PROVIDER", "openrouter"))
    ap.add_argument("--n", type=int, default=3, help="variants per (pair, style)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap source files (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # System/user prompt templates and style descriptions are the ones listed
    # in the paper's "Paraphrase-Generation Prompts" appendix.
    cfg = yaml.safe_load(open(a.prompts))
    sys_t, user_t = Template(cfg["system"]), Template(cfg["user"])
    styles = a.styles or list(cfg["styles"])

    files = sorted(glob.glob(os.path.join(a.in_dir, "**/*.json"), recursive=True))
    if a.limit:
        files = files[:a.limit]

    # Dedup by (data, injection): within a scenario these are identical across
    # all user_tasks, so generation is per (scenario, injection_task) — keep one
    # representative file per unique pair (matches AutoDojo: generate per
    # injection_task, not per user_task).
    seen, uniq = set(), []
    for jf in files:
        d = json.load(open(jf))
        key = (d.get("data", ""), d.get("injection", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(jf)
    print(f"{len(files)} files -> {len(uniq)} unique (data,injection) pairs")

    # Build one job per (pair, style) cell; each cell makes up to --n deduped
    # variants. Skip a cell whose first variant already exists (cheap cache).
    jobs = []
    for jf in uniq:
        rel = os.path.relpath(os.path.dirname(jf), a.in_dir)
        stem = os.path.splitext(os.path.basename(jf))[0]
        for style in styles:
            v0 = os.path.join(a.out, rel, f"{stem}__{style}_0.json")
            if os.path.exists(v0) and not a.overwrite:
                continue
            jobs.append((jf, rel, stem, style))
    print(f"{len(files)} files x {len(styles)} styles x n={a.n} -> {len(jobs)} "
          f"cells ({a.model} via {a.provider}, {a.workers} workers)")
    if a.dry_run:
        for jf, rel, stem, style in jobs[:20]:
            print("  ", os.path.join(a.out, rel, f"{stem}__{style}_*.json"))
        return

    def _norm(t):  # for near-duplicate detection within a cell
        return re.sub(r"\s+", " ", t.strip().lower())

    def work(job):
        jf, rel, stem, style = job
        d = json.load(open(jf))
        u = parse_segments(d["sentence_segments"])
        s, e = d["injection_span_unit"]["start"], d["injection_span_unit"]["end"]
        if not (0 <= s <= e < len(u)):
            return ("bad_span", 0)
        benign_doc = d.get("data") or "\n".join(u[:s] + u[e + 1:])
        sys_p = sys_t.safe_substitute()
        user_p = user_t.safe_substitute(style_name=style,
                                        style_desc=cfg["styles"][style],
                                        benign_document=benign_doc,
                                        injection=d["injection"])
        out_dir = os.path.join(a.out, rel)
        seen_norm, written = set(), 0
        # LLM paraphrasing of the malicious span (paper: paraphrase bank for
        # realizable AT); temperature 1.0 for stylistic diversity, near
        # duplicates within a cell are dropped.
        for _ in range(a.n):
            para = parse_wrapper(generate(user_p, a.model, provider=a.provider,
                                          system_prompt=sys_p, temperature=1.0))
            if not para or _norm(para) in seen_norm:
                continue
            rec = make_sample(d, jf, a.in_dir, style, para)
            if rec is None:
                continue
            seen_norm.add(_norm(para))
            os.makedirs(out_dir, exist_ok=True)
            json.dump(rec, open(os.path.join(
                out_dir, f"{stem}__{style}_{written}.json"), "w"), indent=2)
            written += 1
        return ("ok" if written else "empty", written)

    made, total_files = {}, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            status, count = fut.result()
            made[status] = made.get(status, 0) + 1
            total_files += count
            if status != "ok":
                print(f"  [{status}]")
            if i % 50 == 0:
                print(f"  ...{i}/{len(jobs)} cells")
    print(f"done -> {a.out} : cells={made}, files written={total_files}")


if __name__ == "__main__":
    main()
