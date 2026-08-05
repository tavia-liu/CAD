#!/usr/bin/env python3
"""Nested paraphrased-injection-only fine-tuning with AgentDyn evaluation.

See Appendix "Adversarial Training with LLM Paraphrasing Attacks".

Alpha is the number of selected paraphrased injection rows divided by the number of rows in
the original training set.  Only y=1 rows from the supplied paraphrase datasets
are eligible for optimization. Their Full-CAD embeddings were built from complete
documents, so verbatim benign sentences contribute context during embedding but
are never themselves fine-tuning rows.  The eligible injection rows are shuffled
once with a fixed selection seed.  Every alpha selects a prefix of that same
permutation, so the selected datasets are strictly nested.

For alpha > 0, every classifier seed independently starts from the deployed
checkpoint and fine-tunes only on the selected paraphrased injection rows, all
with label 1.  Original training rows are not replayed.  Dropout is disabled while
gradients remain enabled.  Classifier seeds affect only the batch ordering.  Alpha
zero is the byte-identical deployed classifier and receives no optimizer step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

from adversarial_training import feature_space as base
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
DEFAULT_ALPHAS = (0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4)
DEFAULT_PARAPHRASE_FILES = (
    base.PROJECT / "outputs/embeddings/para_injection_only_combined_fullcad.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=base.PROJECT / "outputs/embeddings/original_fullcad.pt",
        help="Original training tensor; only its row count defines the alpha denominator.",
    )
    parser.add_argument(
        "--deployed", type=Path, default=base.PROJECT / "outputs/models/classifier_fullcad.pt"
    )
    parser.add_argument(
        "--para-files", nargs="+", type=Path, default=list(DEFAULT_PARAPHRASE_FILES)
    )
    parser.add_argument(
        "--agentdyn-cache",
        type=Path,
        default=base.PROJECT / "outputs/embeddings/ood_fullcad_eval_cache.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base.PROJECT / "outputs/models/llm_paraphrase",
    )
    parser.add_argument(
        "--result-dir", type=Path, default=base.PROJECT / "results/adversarial_training/llm_paraphrase"
    )
    parser.add_argument(
        "--figure-dir", type=Path, default=base.PROJECT / "results/adversarial_training/llm_paraphrase"
    )
    parser.add_argument("--output-prefix", default="finetune_paraphrase_injection_only")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(base.REPORTING_SEEDS))
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def rounded_count(alpha: float, denominator: int) -> int:
    return int(math.floor(alpha * denominator + 0.5))


# Paraphrase-bank files are hashed into the manifest so the exact adversarial
# data behind every reported number can be verified.
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_paraphrase_pool(
    paths: list[Path], selection_seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """Load the paraphrase bank (paper, Approach section, adversarial-training
    subsection): malicious sentences rewritten by an LLM to simulate realizable
    evasion attacks. Generation prompts are in the "Paraphrase Prompts"
    appendix; embeddings were precomputed with the Full-CAD embedder so this
    stage never re-runs the encoder or the paraphrasing LLM.
    """
    features = []
    labels = []
    source_ids = []
    sources = []
    feature_dim = None
    for source_id, path in enumerate(paths):
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict) or "X" not in payload or "y" not in payload:
            raise ValueError(f"expected X/y tensor dictionary: {path}")
        X = payload["X"].to(torch.float32)
        y = payload["y"].to(torch.long)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError(f"invalid X/y shapes in {path}: {tuple(X.shape)}, {tuple(y.shape)}")
        if feature_dim is None:
            feature_dim = X.shape[1]
        elif X.shape[1] != feature_dim:
            raise ValueError(f"feature width mismatch in {path}: {X.shape[1]} != {feature_dim}")
        unique_labels = set(int(value) for value in torch.unique(y).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"labels must be binary in {path}, got {sorted(unique_labels)}")
        # Only paraphrased malicious sentences (y=1) enter the adversarial
        # pool; the verbatim benign sentences served solely as embedding
        # context ("Adversarial-Training Protocol and Extended Results" appendix).
        injection_mask = y == 1
        X_injection = X[injection_mask]
        y_injection = y[injection_mask]
        features.append(X_injection)
        labels.append(y_injection)
        source_ids.append(torch.full((len(X_injection),), source_id, dtype=torch.long))
        sources.append(
            {
                "source_id": source_id,
                "name": path.stem.removeprefix("para_").removesuffix("_fullcad"),
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "n_full_document_rows": len(X),
                "n_verbatim_benign_context_rows": int((y == 0).sum()),
                "n_eligible_paraphrased_injection_rows": len(X_injection),
            }
        )

    X_all = torch.cat(features, dim=0)
    y_all = torch.cat(labels, dim=0)
    source_all = torch.cat(source_ids, dim=0)
    # One fixed permutation of the bank; every alpha samples a prefix of it, so
    # the paraphrase subsets used across the alpha sweep are strictly nested
    # (isolates the effect of the mixing weight alpha).
    generator = torch.Generator().manual_seed(selection_seed)
    permutation = torch.randperm(len(X_all), generator=generator)
    return (
        X_all[permutation].contiguous(),
        y_all[permutation].contiguous(),
        source_all[permutation].contiguous(),
        permutation,
        sources,
    )


def load_student(path: Path, device: torch.device) -> base.Classifier:
    model = base.load_frozen_generator(path, device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    # Gradients work in eval mode, while the two Dropout layers stay disabled.
    model.eval()
    return model


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


def fine_tune(
    X_pool: torch.Tensor,
    y_pool: torch.Tensor,
    deployed: Path,
    *,
    n_selected: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[base.Classifier, list[dict]]:
    """Realizable AT via LLM paraphrasing (paper, Approach section,
    adversarial-training subsection): fine-tune the deployed head on
    paraphrase-bank rows only. alpha (via n_selected) is the mixing weight
    controlling how much adversarial data enters training, tracing the
    utility-vs-robustness trade-off ("Adversarial-Training Protocol and Extended Results" appendix).
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
    history = []
    for epoch in range(epochs):
        chunks = nested_batch_chunks(len(X_pool), n_selected, batch_size, seed, epoch)
        total_loss = 0.0
        for indices in chunks:
            logits = model(X_pool[indices].to(device))
            targets = y_pool[indices].to(device)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        history.append({"loss": total_loss / len(chunks), "steps": len(chunks)})
    return model, history


def selected_stats(
    y_pool: torch.Tensor,
    source_pool: torch.Tensor,
    sources: list[dict],
    n_selected: int,
) -> dict:
    labels = y_pool[:n_selected]
    source_ids = source_pool[:n_selected]
    stats = {
        "n_selected": n_selected,
        "n_benign": int((labels == 0).sum()),
        "n_injection": int((labels == 1).sum()),
        "by_source": {},
    }
    for source in sources:
        mask = source_ids == int(source["source_id"])
        source_labels = labels[mask]
        stats["by_source"][source["name"]] = {
            "n_selected": int(mask.sum()),
            "n_benign": int((source_labels == 0).sum()),
            "n_injection": int((source_labels == 1).sum()),
        }
    return stats


def save_manifest(
    path: Path,
    *,
    train_path: Path,
    n_original: int,
    selection_seed: int,
    permutation: torch.Tensor,
    sources: list[dict],
    alpha_stats: dict[str, dict],
) -> None:
    manifest = {
        "protocol": (
            "filter y=1 paraphrased injection rows, make one pooled random permutation, "
            "and use a strict prefix at every alpha"
        ),
        "alpha_definition": "selected paraphrased injection rows / original training rows",
        "benign_role": (
            "verbatim benign text is used only as document context when Full-CAD embeddings "
            "are constructed; benign rows are excluded from fine-tuning"
        ),
        "original_training_file": str(train_path.resolve()),
        "n_original_training_rows": n_original,
        "selection_seed": selection_seed,
        "n_paraphrased_injection_pool_rows": len(permutation),
        "sources": sources,
        "permutation_semantics": (
            "combined input row indices after concatenating sources in listed order"
        ),
        "selection_permutation": permutation.tolist(),
        "alpha_prefixes": alpha_stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def write_outputs(
    rows: list[dict],
    deployed_metrics: dict,
    result_dir: Path,
    figure_dir: Path,
    output_prefix: str,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    suite_fields = [
        field
        for suite in base.AGENTDYN_SUITES
        for field in (f"{suite}_tpr", f"{suite}_fpr")
    ]
    raw_fields = [
        "alpha",
        "seed",
        "n_selected",
        "n_benign",
        "n_injection",
        "tpr",
        "fpr",
        *suite_fields,
        "checkpoint",
        "history_json",
    ]
    raw_path = result_dir / f"{output_prefix}_agentdyn_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        {
            "model": "deployed_clf",
            "alpha": 0.0,
            "n_seeds": 1,
            "n_selected": 0,
            "n_benign": 0,
            "n_injection": 0,
            "tpr": deployed_metrics["macro_tpr"],
            "tpr_std": 0.0,
            "fpr": deployed_metrics["macro_fpr"],
            "fpr_std": 0.0,
            **{
                f"{suite}_{metric}": deployed_metrics["per_suite"][suite][metric]
                for suite in base.AGENTDYN_SUITES
                for metric in ("tpr", "fpr")
            },
            **{
                f"{suite}_{metric}_std": 0.0
                for suite in base.AGENTDYN_SUITES
                for metric in ("tpr", "fpr")
            },
        }
    ]
    for alpha in sorted({float(row["alpha"]) for row in rows if float(row["alpha"]) > 0}):
        selected = [row for row in rows if float(row["alpha"]) == alpha]
        tpr = np.array([float(row["tpr"]) for row in selected])
        fpr = np.array([float(row["fpr"]) for row in selected])
        output = {
            "model": "paraphrased_injection_finetune",
            "alpha": alpha,
            "n_seeds": len(selected),
            "n_selected": selected[0]["n_selected"],
            "n_benign": selected[0]["n_benign"],
            "n_injection": selected[0]["n_injection"],
            "tpr": float(tpr.mean()),
            "tpr_std": float(tpr.std()),
            "fpr": float(fpr.mean()),
            "fpr_std": float(fpr.std()),
        }
        for suite in base.AGENTDYN_SUITES:
            for metric in ("tpr", "fpr"):
                values = np.array([float(row[f"{suite}_{metric}"]) for row in selected])
                output[f"{suite}_{metric}"] = float(values.mean())
                output[f"{suite}_{metric}_std"] = float(values.std())
        summary.append(output)

    summary_path = result_dir / f"{output_prefix}_agentdyn.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    figure_path = figure_dir / f"{output_prefix}_agentdyn.png"
    if plt is not None:
        x = np.arange(len(summary))
        labels = [f"{float(row['alpha']):g}" for row in summary]
        fig, ax = plt.subplots(figsize=(10.5, 5.6))
        for metric, color, marker, label in (
            ("tpr", "#15803d", "o", "TPR @ 0.5"),
            ("fpr", "#dc2626", "s", "FPR @ 0.5"),
        ):
            mean = np.array([float(row[metric]) for row in summary])
            std = np.array([float(row[f"{metric}_std"]) for row in summary])
            ax.plot(x, mean, color=color, marker=marker, lw=2, label=label)
            ax.fill_between(
                x,
                np.maximum(0, mean - std),
                np.minimum(1, mean + std),
                color=color,
                alpha=0.15,
            )
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_xlabel("Selected paraphrased injection rows / original training rows (alpha)")
        ax.set_ylabel("AgentDyn rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title("AgentDyn paraphrased-injection-only fine-tuning")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        print("figure   -> skipped (matplotlib is not installed)")

    print(f"raw      -> {raw_path}")
    print(f"summary  -> {summary_path}")
    if plt is not None:
        print(f"figure   -> {figure_path}")


def main() -> None:
    args = parse_args()
    alphas = sorted(set(float(alpha) for alpha in args.alphas))
    if not alphas or alphas[0] <= 0:
        raise SystemExit("--alphas must be positive")
    if tuple(sorted(args.seeds)) != base.REPORTING_SEEDS:
        raise SystemExit("reported results require exactly --seeds 0 1 2 3 4")

    train = torch.load(args.train, weights_only=True, map_location="cpu")
    if not isinstance(train, dict) or "X" not in train:
        raise ValueError(f"expected X tensor in original training file: {args.train}")
    n_original = len(train["X"])
    X_pool, y_pool, source_pool, permutation, sources = load_paraphrase_pool(
        args.para_files, args.selection_seed
    )
    counts = {alpha: rounded_count(alpha, n_original) for alpha in alphas}
    if min(counts.values()) <= 0:
        raise ValueError("the smallest alpha selects zero rows")
    if max(counts.values()) > len(X_pool):
        raise ValueError(
            f"largest alpha needs {max(counts.values())} rows, but pool has {len(X_pool)}"
        )

    stats_by_alpha = {
        f"{alpha:g}": selected_stats(y_pool, source_pool, sources, n_selected)
        for alpha, n_selected in counts.items()
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "paraphrase_injection_nested_selection_manifest.json"
    save_manifest(
        manifest_path,
        train_path=args.train,
        n_original=n_original,
        selection_seed=args.selection_seed,
        permutation=permutation,
        sources=sources,
        alpha_stats=stats_by_alpha,
    )

    device = base.choose_device(args.device)
    eval_sets = base.load_eval_sets(
        args.agentdyn_cache,
        base.PROJECT / "outputs/embeddings/test_fullcad.pt",
        base.PROJECT / "data/testing_data_original",
        ("agentdyn",),
    )
    deployed = base.load_frozen_generator(args.deployed, device)
    deployed_metrics = base.evaluate_offline(deployed, eval_sets, device, args.threshold)[
        "agentdyn"
    ]
    print(
        f"device={device} original={n_original} injection_only_pool={len(X_pool)} "
        f"labels=0:{int((y_pool == 0).sum())},1:{int((y_pool == 1).sum())} "
        f"deployed TPR/FPR={deployed_metrics['macro_tpr']:.4f}/"
        f"{deployed_metrics['macro_fpr']:.4f}"
    )
    print(f"selection manifest -> {manifest_path}")
    print(
        "alpha prefixes: "
        + " ".join(
            f"{alpha:g}:{counts[alpha]}"
            f"(b={stats_by_alpha[f'{alpha:g}']['n_benign']},"
            f"i={stats_by_alpha[f'{alpha:g}']['n_injection']})"
            for alpha in alphas
        )
    )

    rows = []
    # Seeds follow the five-seed reporting protocol ("Statistical Variation
    # Across Training Seeds" appendix). alpha=0 is the unmodified deployed
    # classifier — the clean end of the alpha trade-off curve.
    for seed in args.seeds:
        rows.append(
            {
                "alpha": 0.0,
                "seed": seed,
                "n_selected": 0,
                "n_benign": 0,
                "n_injection": 0,
                "tpr": deployed_metrics["macro_tpr"],
                "fpr": deployed_metrics["macro_fpr"],
                **{
                    f"{suite}_{metric}": deployed_metrics["per_suite"][suite][metric]
                    for suite in base.AGENTDYN_SUITES
                    for metric in ("tpr", "fpr")
                },
                "checkpoint": str(args.deployed),
                "history_json": "[]",
            }
        )
        for alpha in alphas:
            n_selected = counts[alpha]
            stats = stats_by_alpha[f"{alpha:g}"]
            model, history = fine_tune(
                X_pool,
                y_pool,
                args.deployed,
                n_selected=n_selected,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )
            metrics = base.evaluate_offline(model, eval_sets, device, args.threshold)["agentdyn"]
            checkpoint = (
                args.out_dir
                / f"head_ft_paraphrase_injection_a{base.alpha_tag(alpha)}_s{seed}.pt"
            )
            metadata = {
                "mode": "paraphrased-injection-only fine-tune from deployed checkpoint",
                "deployed_checkpoint": str(args.deployed),
                "selection_manifest": str(manifest_path),
                "paraphrase_files": [str(path) for path in args.para_files],
                "selection_seed": args.selection_seed,
                "alpha": alpha,
                "alpha_denominator": "original training rows",
                "n_original_training_rows": n_original,
                "n_selected": n_selected,
                "n_benign": stats["n_benign"],
                "n_injection": stats["n_injection"],
                "seed": seed,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "threshold": args.threshold,
                "original_training_replay": False,
                "benign_rows_used_for_optimization": False,
                "benign_text_role": "Full-CAD embedding neighbor only",
                "fine_tuning_label": 1,
                "optimizer_steps": args.epochs * math.ceil(n_selected / args.batch_size),
                "dropout_disabled": True,
            }
            base.save_model(model, checkpoint, metadata)
            rows.append(
                {
                    "alpha": alpha,
                    "seed": seed,
                    "n_selected": n_selected,
                    "n_benign": stats["n_benign"],
                    "n_injection": stats["n_injection"],
                    "tpr": metrics["macro_tpr"],
                    "fpr": metrics["macro_fpr"],
                    **{
                        f"{suite}_{metric}": metrics["per_suite"][suite][metric]
                        for suite in base.AGENTDYN_SUITES
                        for metric in ("tpr", "fpr")
                    },
                    "checkpoint": str(checkpoint),
                    "history_json": json.dumps(history),
                }
            )
            print(
                f"s={seed} a={alpha:g} n={n_selected} "
                f"labels=0:{stats['n_benign']},1:{stats['n_injection']} "
                f"TPR/FPR={metrics['macro_tpr']:.4f}/{metrics['macro_fpr']:.4f} "
                f"loss={history[-1]['loss']:.5f} steps={history[-1]['steps']}"
            )

    write_outputs(rows, deployed_metrics, args.result_dir, args.figure_dir, args.output_prefix)


if __name__ == "__main__":
    main()
