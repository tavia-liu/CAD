#!/usr/bin/env python3
"""Online feature-space adversarial training with fixed-threshold offline evaluation.

See Appendix "Adversarial Training with Feature-Space Perturbations".

This is standard (online) adversarial training in the embedding space:

* Start every run from the deployed Full-CAD checkpoint g_theta_0 and fine-tune the
  MLP head; the encoder phi is frozen throughout.
* Recompute the FGSM-RS perturbation against the *current* parameters at every
  optimizer step, so the inner maximization tracks g_theta as it is updated.  No
  perturbation is cached or reused across steps or across alpha.
* Alpha is the number of selected perturbed rows divided by the number of rows in the
  original training set.  Injection rows are drawn by cycling through independently
  shuffled permutations, so alpha may exceed N_injection/N_original and every appended
  row still receives its own random start.  Every alpha takes a prefix of that same
  fixed per-seed order, so the selected sets are strictly nested and differences across
  alpha are attributable to the mixing weight alone.
* Alpha zero is the byte-identical deployed classifier and receives no optimizer step.
* Evaluate TPR and FPR at the fixed deployed decision rule p(injection) > 0.5.  No
  threshold matching or recalibration is performed.

The attack is FGSM with random start (single-step PGD) on the sentence embedding,
with delta constrained to the L_inf ball ||delta||_inf <= epsilon and epsilon set
per embedding block.

Offline evaluation sets:

* AgentDyn: cached benign/core-injection embeddings for github, shopping, dailylife.
* AgentDojo: original core-masked test embeddings for banking, slack, travel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "fullcad_adv_training_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:
    plt = None
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
# The 2048-d feature is [x^iso || x_ctx] (paper, Approach section); each 1024-d
# half gets its own perturbation budget epsilon since their scales differ.
BLOCKS = ((0, 1024), (1024, 2048))
AGENTDYN_SUITES = ("github", "shopping", "dailylife")
AGENTDOJO_SUITES = ("banking", "slack", "travel")
# AT ratio grid reported in the "Adversarial-Training Protocol and Extended
# Results" appendix; matches the alphas the evaluation scripts resolve.
DEFAULT_ALPHAS = (0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4)
# Reported numbers average over these five training seeds
# ("Statistical Variation Across Training Seeds" appendix).
REPORTING_SEEDS = (0, 1, 2, 3, 4)


class Classifier(nn.Module):
    def __init__(self, d_in: int, d_hid: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}"


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_frozen_generator(path: Path, device: torch.device) -> Classifier:
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"expected checkpoint dict with model_state_dict: {path}")
    model = Classifier(
        int(ckpt.get("input_dim", 2048)),
        int(ckpt.get("d_hid", 256)),
        float(ckpt.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_student(path: Path, device: torch.device) -> Classifier:
    """Load the deployed head as a trainable student (paper: fine-tune from g_theta_0)."""
    model = load_frozen_generator(path, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Gradients work in eval mode, while the two Dropout layers stay disabled.
    model.eval()
    return model


def save_model(model: Classifier, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": 2048,
        "d_hid": 256,
        "dropout": 0.3,
        "best_threshold": 0.5,
        "online_at_training": metadata,
    }
    torch.save(payload, path)


def per_block_scale(X: torch.Tensor, rel: float) -> torch.Tensor:
    # Perturbation budget for the L_inf ball ||delta||_inf <= epsilon (paper,
    # Approach section, adversarial-training subsection). Epsilon is set relative
    # to the mean absolute activation of each embedding half so the budget is
    # comparable across the isolated and contextual blocks
    # ("Adversarial-Training Protocol and Extended Results" appendix).
    if X.ndim != 2 or X.shape[1] != 2048:
        raise ValueError(f"expected [N, 2048] Full-CAD features, got {tuple(X.shape)}")
    eps = torch.zeros(X.shape[1], dtype=X.dtype)
    for start, end in BLOCKS:
        eps[start:end] = rel * X[:, start:end].abs().mean()
    return eps


def fgsm_rs(
    model: Classifier,
    xb: torch.Tensor,
    yb: torch.Tensor,
    eps_d: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """FGSM with random start, computed against the CURRENT parameters of `model`.

    This is the inner maximization of the paper's adversarial-training objective
    (Approach section, adversarial-training subsection). Because the perturbation is
    recomputed at every optimizer step, the attack tracks g_theta as it is updated;
    nothing is cached across steps or across alpha.
    """
    # Random start: delta_0 ~ Uniform(-epsilon, epsilon), sampled per row on CPU so
    # the draw does not depend on accelerator RNG.
    unit_start = torch.rand(xb.shape, generator=generator, dtype=xb.dtype) * 2.0 - 1.0
    delta = (unit_start.to(xb.device) * eps_d).requires_grad_(True)
    # Ascent direction: gradient of the classification loss w.r.t. delta, taken
    # through the current head (loss-maximizing perturbation).
    loss = nn.functional.cross_entropy(model(xb + delta), yb)
    grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
    with torch.no_grad():
        # Signed-gradient step (step size 1.25*epsilon, standard FGSM-RS), then
        # project back onto the L_inf ball per embedding block.
        adv_delta = delta + 1.25 * eps_d * grad.sign()
        for a, b in BLOCKS:
            adv_delta[:, a:b].clamp_(-eps_d[a], eps_d[a])
    return (xb + adv_delta).detach()


def nested_batch_chunks(
    n_pool: int,
    n_selected: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed + 40_009 + epoch)
    # Shuffle the full fixed pool, then retain the selected prefix.  Therefore rows
    # shared by two alpha values keep the same relative batch order.
    full_order = torch.randperm(n_pool, generator=generator)
    selected_order = full_order[full_order < n_selected]
    return tuple(selected_order.split(batch_size))


def fine_tune_online_at(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    deployed: Path,
    *,
    n_selected: int,
    eps: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[Classifier, list[dict]]:
    """Feature-space AT (paper, Approach section, adversarial-training subsection).

    Fine-tune the deployed head on injection rows whose embeddings are perturbed by
    an FGSM-RS attack recomputed against the current parameters at every step. alpha
    (via n_selected) is the mixing weight controlling how much adversarial data
    enters training, tracing the utility-vs-robustness trade-off
    ("Adversarial-Training Protocol and Extended Results" appendix).
    """
    if not 0 < n_selected <= len(X_pool):
        raise ValueError(f"n_selected must be in [1, {len(X_pool)}], got {n_selected}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = load_student(deployed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    eps_d = eps.to(device)
    attack_generator = torch.Generator().manual_seed(seed + 20_003)
    history = []
    for epoch in range(epochs):
        chunks = nested_batch_chunks(len(X_pool), n_selected, batch_size, seed, epoch)
        total_loss = 0.0
        for indices in chunks:
            xb = X_pool[indices].to(device)
            yb = y_pool[indices].to(device)
            # Inner maximization against the current theta, then the outer step.
            xb_adv = fgsm_rs(model, xb, yb, eps_d, attack_generator)
            loss = nn.functional.cross_entropy(model(xb_adv), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        history.append({"loss": total_loss / max(len(chunks), 1), "steps": len(chunks)})
    return model, history


@torch.inference_mode()
def positive_rate(model: Classifier, X: torch.Tensor, device: torch.device, threshold: float) -> float:
    model.eval()
    positives = 0
    total = 0
    for start in range(0, len(X), 2048):
        xb = X[start : start + 2048].to(device)
        prob = torch.softmax(model(xb), dim=1)[:, 1]
        positives += int((prob > threshold).sum())
        total += len(prob)
    return positives / total if total else float("nan")


def parse_segments(raw: list | str) -> list[str]:
    if isinstance(raw, list):
        return raw
    matches = re.findall(r"\[(\d+)\]\s?(.*?)(?=\n\[\d+\]|$)", raw, re.DOTALL)
    segments = {int(index): content.strip() for index, content in matches}
    return [segments[index] for index in sorted(segments)]


def load_agentdojo_sets(test_path: Path, test_dir: Path) -> dict:
    """Split original AgentDojo test rows by suite using the training-time core mask."""
    data = torch.load(test_path, weights_only=True, map_location="cpu")
    X, y = data["X"].to(torch.float32), data["y"].to(torch.long)
    kept = {suite: {"X": [], "y": []} for suite in AGENTDOJO_SUITES}
    offset = 0
    for path in sorted(test_dir.rglob("*.json")):
        record = json.loads(path.read_text())
        n_rows = len(parse_segments(record["sentence_segments"]))
        end = offset + n_rows
        if end > len(X):
            raise ValueError(f"AgentDojo file rows exceed test tensor at {path}")
        suite = path.relative_to(test_dir).parts[0]
        if suite in kept:
            span_start = int(record["injection_span_unit"]["start"])
            span_end = int(record["injection_span_unit"]["end"])
            core_start, core_end = span_start + 3, span_end - 5
            mask = torch.tensor(
                [
                    not (span_start <= i <= span_end and (i < core_start or i > core_end))
                    for i in range(n_rows)
                ],
                dtype=torch.bool,
            )
            kept[suite]["X"].append(X[offset:end][mask])
            kept[suite]["y"].append(y[offset:end][mask])
        offset = end
    if offset != len(X):
        raise ValueError(f"AgentDojo row mismatch: JSON files={offset}, test tensor={len(X)}")

    result = {}
    for suite, chunks in kept.items():
        X_suite = torch.cat(chunks["X"])
        y_suite = torch.cat(chunks["y"])
        result[suite] = {"neg": X_suite[y_suite == 0], "pos": X_suite[y_suite == 1]}
    return result


def load_eval_sets(
    agentdyn_cache: Path,
    agentdojo_test: Path,
    agentdojo_test_dir: Path,
    benchmarks: tuple[str, ...] = ("agentdyn", "agentdojo"),
) -> dict:
    result = {}
    if "agentdyn" in benchmarks:
        cache = torch.load(agentdyn_cache, weights_only=True, map_location="cpu")
        missing = [s for s in AGENTDYN_SUITES if s not in cache["benign"] or s not in cache["pos"]]
        if missing:
            raise ValueError(f"AgentDyn cache missing suites: {missing}")
        result["agentdyn"] = {
            s: {
                "neg": cache["benign"][s].to(torch.float32),
                "pos": cache["pos"][s].to(torch.float32),
            }
            for s in AGENTDYN_SUITES
        }
    if "agentdojo" in benchmarks:
        result["agentdojo"] = load_agentdojo_sets(agentdojo_test, agentdojo_test_dir)
    return result


def evaluate_offline(
    model: Classifier, eval_sets: dict, device: torch.device, threshold: float
) -> dict:
    result = {}
    for benchmark, suites in eval_sets.items():
        per_suite = {}
        for suite, data in suites.items():
            per_suite[suite] = {
                "tpr": positive_rate(model, data["pos"], device, threshold),
                "fpr": positive_rate(model, data["neg"], device, threshold),
                "n_pos": len(data["pos"]),
                "n_neg": len(data["neg"]),
            }
        result[benchmark] = {
            "per_suite": per_suite,
            "macro_tpr": float(np.mean([x["tpr"] for x in per_suite.values()])),
            "macro_fpr": float(np.mean([x["fpr"] for x in per_suite.values()])),
        }
    return result


def aggregate(rows: list[dict]) -> dict:
    out = {}
    alphas = sorted({float(row["alpha"]) for row in rows})
    for benchmark in rows[0]["offline"]:
        out[benchmark] = []
        for alpha in alphas:
            selected = [row["offline"][benchmark] for row in rows if float(row["alpha"]) == alpha]
            tpr = np.array([x["macro_tpr"] for x in selected], dtype=float)
            fpr = np.array([x["macro_fpr"] for x in selected], dtype=float)
            out[benchmark].append(
                {
                    "alpha": alpha,
                    "n_seeds": len(selected),
                    "tpr_mean": float(tpr.mean()),
                    "tpr_std": float(tpr.std()),
                    "fpr_mean": float(fpr.mean()),
                    "fpr_std": float(fpr.std()),
                }
            )
    return out


def plot_one(
    summary_rows: list[dict],
    benchmark: str,
    reference: dict,
    path: Path,
) -> None:
    titles = {
        "agentdyn": "AgentDyn offline",
        "agentdojo": "AgentDojo original offline",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    alpha = np.array([row["alpha"] for row in summary_rows])
    x = np.arange(len(alpha))
    for metric, color, marker, label in (
        ("tpr", "#15803d", "o", "Macro TPR @ 0.5"),
        ("fpr", "#dc2626", "s", "Macro FPR @ 0.5"),
    ):
        mean = np.array([row[f"{metric}_mean"] for row in summary_rows])
        std = np.array([row[f"{metric}_std"] for row in summary_rows])
        ax.plot(x, mean, marker=marker, color=color, lw=2, ms=5, label=label)
        if np.any(std > 0):
            ax.fill_between(
                x,
                np.maximum(0, mean - std),
                np.minimum(1, mean + std),
                color=color,
                alpha=0.14,
            )
    ax.axhline(
        reference["macro_tpr"],
        color="#15803d",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label="Deployed clf TPR",
    )
    ax.axhline(
        reference["macro_fpr"],
        color="#dc2626",
        ls="--",
        lw=1.2,
        alpha=0.7,
        label="Deployed clf FPR",
    )
    ax.set_title(f"{titles[benchmark]} — online feature-space AT")
    ax.set_xlabel("Selected perturbed injection rows / injection rows (alpha)")
    ax.set_ylabel("Macro rate across suites")
    ax.set_xticks(x, [f"{value:g}" for value in alpha])
    ax.tick_params(axis="x", rotation=45)
    # The alpha grid is intentionally dense near zero, so render sweep points at
    # equal visual spacing and retain their exact numeric values as tick labels.
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_table(rows: list[dict], benchmark: str, reference: dict, path: Path) -> None:
    benchmark_rows = [row for row in rows if benchmark in row["offline"]]
    alphas = sorted({float(row["alpha"]) for row in benchmark_rows})
    suites = list(benchmark_rows[0]["offline"][benchmark]["per_suite"])
    fields = ["model", "alpha", "n_seeds", "n_perturbed"]
    for suite in suites:
        for metric in ("tpr", "fpr"):
            fields += [f"{suite}_{metric}_mean", f"{suite}_{metric}_std"]
    fields += ["macro_tpr_mean", "macro_tpr_std", "macro_fpr_mean", "macro_fpr_std"]

    reference_row = {
        "model": "deployed_clf",
        "alpha": "",
        "n_seeds": 1,
        "n_perturbed": "",
    }
    for suite in suites:
        for metric in ("tpr", "fpr"):
            reference_row[f"{suite}_{metric}_mean"] = reference["per_suite"][suite][metric]
            reference_row[f"{suite}_{metric}_std"] = 0.0
    for metric in ("tpr", "fpr"):
        reference_row[f"macro_{metric}_mean"] = reference[f"macro_{metric}"]
        reference_row[f"macro_{metric}_std"] = 0.0
    table = [reference_row]
    for alpha in alphas:
        selected = [row for row in benchmark_rows if float(row["alpha"]) == alpha]
        output = {
            "model": "online_at",
            "alpha": alpha,
            "n_seeds": len(selected),
            "n_perturbed": selected[0]["n_perturbed"],
        }
        for suite in suites:
            for metric in ("tpr", "fpr"):
                values = np.array(
                    [row["offline"][benchmark]["per_suite"][suite][metric] for row in selected],
                    dtype=float,
                )
                output[f"{suite}_{metric}_mean"] = float(values.mean())
                output[f"{suite}_{metric}_std"] = float(values.std())
        for metric in ("tpr", "fpr"):
            values = np.array(
                [row["offline"][benchmark][f"macro_{metric}"] for row in selected],
                dtype=float,
            )
            output[f"macro_{metric}_mean"] = float(values.mean())
            output[f"macro_{metric}_std"] = float(values.std())
        table.append(output)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=PROJECT / "outputs/embeddings/original_fullcad.pt")
    p.add_argument("--deployed", type=Path, default=PROJECT / "outputs/models/classifier_fullcad.pt")
    p.add_argument("--agentdyn-cache", type=Path, default=PROJECT / "outputs/embeddings/ood_fullcad_eval_cache.pt")
    p.add_argument("--agentdojo-test", type=Path, default=PROJECT / "outputs/embeddings/test_fullcad.pt")
    p.add_argument("--agentdojo-test-dir", type=Path, default=PROJECT / "data/testing_data_original")
    p.add_argument("--out-dir", type=Path, default=PROJECT / "outputs/models/feature_space")
    p.add_argument("--table-dir", type=Path, default=PROJECT / "results/adversarial_training/feature_space")
    p.add_argument("--figure-dir", type=Path, default=PROJECT / "results/adversarial_training/feature_space")
    p.add_argument("--output-prefix", default="online_at_perturbed_only")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("agentdyn", "agentdojo"),
        default=["agentdyn", "agentdojo"],
    )
    p.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(REPORTING_SEEDS))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--rel", type=float, default=0.25)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="skip training and rebuild tables/figures from existing checkpoints",
    )
    return p.parse_args()


def checkpoint_name(alpha: float, seed: int) -> str:
    return f"head_ft_perturbed_only_a{alpha_tag(alpha)}_s{seed}.pt"


def main() -> None:
    args = parse_args()
    if tuple(sorted(args.seeds)) != REPORTING_SEEDS:
        raise SystemExit("reported results require exactly --seeds 0 1 2 3 4")
    alphas = sorted(set(float(a) for a in args.alphas))
    if not alphas or alphas[0] <= 0 or alphas[-1] > 1:
        raise SystemExit("--alphas must be unique values in (0, 1]")
    if any(b <= a for a, b in zip(alphas, alphas[1:])):
        raise SystemExit("--alphas must be strictly increasing after de-duplication")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    device = choose_device(args.device)
    print(f"device={device} threshold={args.threshold} (strict >; no matching)")
    train = torch.load(args.train, weights_only=True, map_location="cpu")
    X, y = train["X"].to(torch.float32), train["y"].to(torch.long)
    injection_indices = torch.where(y == 1)[0]
    eps = per_block_scale(X, args.rel)
    eval_sets = load_eval_sets(
        args.agentdyn_cache,
        args.agentdojo_test,
        args.agentdojo_test_dir,
        tuple(args.benchmarks),
    )
    deployed_model = load_frozen_generator(args.deployed, device)
    deployed_reference = evaluate_offline(deployed_model, eval_sets, device, args.threshold)
    # alpha is defined against the original training rows ("Adversarial-Training
    # Protocol and Extended Results" appendix), not against the injection rows.
    alpha_population = len(X)
    print(f"train={tuple(X.shape)} injections={len(injection_indices)}")
    print(f"deployed checkpoint={args.deployed}")
    print(
        "alpha counts: "
        + " ".join(f"{a:g}:{int(math.floor(a * alpha_population + 0.5))}" for a in alphas)
    )
    print(f"alpha denominator=original training rows population={alpha_population}")
    print("perturbations recomputed online against the current theta at every step")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.evaluate_existing:
        print("evaluation-only mode: loading existing fine-tuned heads")
        for seed in args.seeds:
            for alpha in [0.0] + alphas:
                target_n = int(math.floor(alpha * alpha_population + 0.5))
                ckpt_path = args.out_dir / checkpoint_name(alpha, seed)
                if not ckpt_path.is_file():
                    raise FileNotFoundError(f"missing checkpoint for evaluation-only mode: {ckpt_path}")
                model = load_frozen_generator(ckpt_path, device)
                metrics = evaluate_offline(model, eval_sets, device, args.threshold)
                rows.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "n_perturbed": target_n,
                        "checkpoint": str(ckpt_path),
                        "losses": [],
                        "offline": metrics,
                    }
                )
                metric_text = " ".join(
                    f"{benchmark} TPR/FPR={value['macro_tpr']:.4f}/{value['macro_fpr']:.4f}"
                    for benchmark, value in metrics.items()
                )
                print(f"s={seed} a={alpha:g} n_perturbed={target_n} {metric_text}")

        summary = aggregate(rows)
        print("\noutputs:")
        for benchmark in args.benchmarks:
            table_path = args.table_dir / f"{args.output_prefix}_{benchmark}.csv"
            figure_path = args.figure_dir / f"{args.output_prefix}_{benchmark}.png"
            write_table(rows, benchmark, deployed_reference[benchmark], table_path)
            plot_one(summary[benchmark], benchmark, deployed_reference[benchmark], figure_path)
            print(f"table  -> {table_path}")
            print(f"figure -> {figure_path}")
        return

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        # One fixed per-seed source ordering; every alpha takes a prefix of it, so the
        # selected sets are strictly nested across alpha.  Because alpha is defined
        # against the original training rows, the requested count can exceed the number
        # of injection rows, so cycle through independently shuffled permutations; each
        # occurrence is perturbed separately with its own random start.
        max_rows = int(math.floor(max(alphas) * alpha_population + 0.5))
        order_gen = torch.Generator().manual_seed(seed + 10_003)
        chunks = []
        remaining = max_rows
        while remaining:
            permutation = injection_indices[
                torch.randperm(len(injection_indices), generator=order_gen)
            ]
            take = min(remaining, len(permutation))
            chunks.append(permutation[:take])
            remaining -= take
        order = torch.cat(chunks)
        X_pool = X[order]
        y_pool = y[order]
        print(
            f"source pool={len(order)} rows drawn from {len(injection_indices)} injections "
            f"({len(order) / len(injection_indices):.2f}x coverage)"
        )

        alpha_counts = {}
        for alpha in [0.0] + alphas:
            target_n = int(math.floor(alpha * alpha_population + 0.5))
            if target_n == 0:
                # alpha = 0 is the byte-identical deployed classifier: no optimizer step.
                model = load_student(args.deployed, device)
                losses = []
            else:
                model, losses = fine_tune_online_at(
                    X_pool,
                    y_pool,
                    args.deployed,
                    n_selected=target_n,
                    eps=eps,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                )
            metrics = evaluate_offline(model, eval_sets, device, args.threshold)
            ckpt_path = args.out_dir / checkpoint_name(alpha, seed)
            metadata = {
                "seed": seed,
                "alpha": alpha,
                "n_perturbed": target_n,
                "alpha_denominator": "original training rows",
                "alpha_population": alpha_population,
                "source": "prefix of fixed cycled injection ordering",
                "mode": "online FGSM-RS recomputed against the current theta",
                "attack": "FGSM with random start, single signed-gradient step",
                "step_size": "1.25 * epsilon",
                "initialized_from": str(args.deployed),
                "trained_from_scratch": False,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "rel": args.rel,
                "threshold_evaluation": args.threshold,
                "matched_threshold": False,
            }
            save_model(model, ckpt_path, metadata)
            rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "n_perturbed": target_n,
                    "checkpoint": str(ckpt_path),
                    "losses": losses,
                    "offline": metrics,
                }
            )
            alpha_counts[alpha_tag(alpha)] = target_n
            metric_text = " ".join(
                f"{benchmark} TPR/FPR={value['macro_tpr']:.4f}/{value['macro_fpr']:.4f}"
                for benchmark, value in metrics.items()
            )
            final_loss = f"{losses[-1]['loss']:.5f}" if losses else "n/a"
            print(f"a={alpha:g} n_perturbed={target_n} loss={final_loss} {metric_text}")

        manifest_path = args.out_dir / f"online_at_manifest_s{seed}.json"
        manifest = {
            "seed": seed,
            "n_injection": len(injection_indices),
            "n_original": len(X),
            "alpha_denominator": "original training rows",
            "alpha_counts": alpha_counts,
            "ordered_source_train_indices": order.tolist(),
            "perturbation": "recomputed online; nothing cached across steps or alphas",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = aggregate(rows)
    print("\noutputs:")
    for benchmark in args.benchmarks:
        table_path = args.table_dir / f"{args.output_prefix}_{benchmark}.csv"
        figure_path = args.figure_dir / f"{args.output_prefix}_{benchmark}.png"
        write_table(rows, benchmark, deployed_reference[benchmark], table_path)
        plot_one(summary[benchmark], benchmark, deployed_reference[benchmark], figure_path)
        print(f"table  -> {table_path}")
        print(f"figure -> {figure_path}")


if __name__ == "__main__":
    main()
