"""Build Full-CAD embedding tensors from JSON datasets.

Each segment is embedded with the paper's query-aware context-delta representation:
the JSON `user` field is prepended as context before computing the segment delta.
Labels are read from the included JSON sentence spans.

    python -m detector.build_embeddings --input_dir data/training_data_original \
        --output_pt outputs/embeddings/original_fullcad.pt
"""
import os, sys, glob, json, argparse, re
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from detector.defense import QueryContextDeltaEmbedder  # noqa: E402

# 1024-d isolated + 1024-d query-contextual half = 2048-d concatenated feature
# ("Detector Representation and Ablations" appendix).
EXPECT_DIM = 2048


def parse_segments(raw):
    if isinstance(raw, list):
        return raw
    matches = re.findall(r"\[(\d+)\]\s?(.*?)(?=\n\[\d+\]|$)", raw, re.DOTALL)
    segments = {int(idx): content.strip() for idx, content in matches}
    return [segments[k] for k in sorted(segments.keys())]


def process_file(jf):
    with open(jf) as f:
        data = json.load(f)
    sents = parse_segments(data["sentence_segments"])
    # Sentence-level labels (paper, Approach section): each sentence s_j inside the
    # annotated injection span is a positive (malicious) example, all others benign.
    s = data["injection_span_unit"]["start"]
    e = data["injection_span_unit"]["end"]
    labels = [1 if s <= i <= e else 0 for i in range(len(sents))]
    return sents, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, action="append")
    ap.add_argument("--output_pt", required=True)
    a = ap.parse_args()
    out = a.output_pt if os.path.isabs(a.output_pt) else os.path.join(REPO_ROOT, a.output_pt)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    jsons = []
    for d in a.input_dir:
        found = sorted(glob.glob(os.path.join(d, "**/*.json"), recursive=True))
        print(f"Found {len(found)} JSON files in {d}")
        jsons.extend(found)

    # Query- and context-aware embedder f(q, s) (paper, Approach section): each
    # sentence gets [x^iso_j(s) || x_j(q, s)] with the user query q prepended as
    # context (concatenation equation in the "Detector Representation and Ablations" appendix).
    embedder = QueryContextDeltaEmbedder()
    X, y, n_noquery = [], [], 0
    for jf in tqdm(jsons):
        with open(jf) as f:
            data = json.load(f)
        # `user` is the query q that conditions the contextual half x_j(q, s).
        query = (data.get("user") or "").strip()
        if not query:
            n_noquery += 1
        sents, labels = process_file(jf)
        if not sents:
            continue
        with torch.no_grad():
            embs = embedder.encode_batch(sents, query=query)
        X.extend(embs)
        y.extend(labels)

    # Embeddings are cached to a .pt tensor so classifier training / AT sweeps
    # (see the "Adversarial-Training Protocol and Extended Results" appendix) never re-run the encoder.
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)
    torch.save({"X": X, "y": y}, out)
    print(f"Saved {y.numel()} samples (benign {(y==0).sum()}, inj {(y==1).sum()}) -> {out}  shape={tuple(X.shape)}")
    if n_noquery:
        print(f"WARNING: {n_noquery} files had no `user` query field (embedded with empty context)")
    assert X.shape[1] == EXPECT_DIM, f"expected {EXPECT_DIM}-d, got {X.shape[1]}"


if __name__ == "__main__":
    main()
