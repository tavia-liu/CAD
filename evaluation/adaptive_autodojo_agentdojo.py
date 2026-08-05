#!/usr/bin/env python3
"""Replay adaptive AutoDojo attacks on AgentDojo suites for one Full-CAD head family."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
SUITES = ("banking", "slack", "travel")
DEFAULT_ALPHAS = (0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4)
FAMILIES = {
    "feature_space": (
        ROOT / "outputs/models/feature_space",
        "head_ft_perturbed_only_a{alpha}_s{seed}.pt",
    ),
    "llm_paraphrase": (
        ROOT / "outputs/models/llm_paraphrase",
        "head_ft_paraphrase_injection_a{alpha}_s{seed}.pt",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, choices=SUITES)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--checkpoint-template")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def rate(values: dict) -> float:
    return sum(map(bool, values.values())) / len(values)


def main() -> None:
    args = parse_args()
    if not args.cache.is_file():
        raise FileNotFoundError(args.cache)
    default_dir, default_template = FAMILIES[args.family]
    models_dir = (args.models_dir or default_dir).resolve()
    template = args.checkpoint_template or default_template
    run_root = args.run_root or ROOT / "runs/adaptive_autodojo_agentdojo" / args.family
    result_root = args.result_root or ROOT / "results/adaptive_autodojo_agentdojo" / args.family

    load_dotenv(args.env_file, override=False)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"OPENROUTER_API_KEY is missing (checked {args.env_file})")

    sys.path[:0] = [str(ROOT / "third_party/AutoDojo/agentdojo/src"), str(ROOT / "src"), str(ROOT)]
    os.environ.setdefault("AGENTDOJO_DEFENSE_PLUGINS", "detector")
    os.environ.setdefault("AGENTDOJO_RUN_INJECTION_UTILITY", "0")
    os.environ["CLASSIFIER_THRESHOLD"] = str(args.threshold)
    # Adaptive AutoDojo attack variants are replayed from a fixed cache so every
    # checkpoint faces the identical attack set (paper, Experiments section:
    # adaptive attacks; CU/UA/ASR on AgentDojo suites).
    os.environ["AUTODOJO_CACHE"] = str(args.cache.resolve())
    os.environ.setdefault("AUTODOJO_VARIANT", "0")

    from agentdojo.scripts.benchmark import benchmark_suite
    from agentdojo.task_suite.load_suites import get_suite

    suite = get_suite(args.benchmark_version, args.suite)
    result_root.mkdir(parents=True, exist_ok=True)

    for alpha in sorted(set(args.alphas)):
        label = f"{alpha:g}"
        checkpoint = models_dir / template.format(alpha=label, seed=args.seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = result_root / f"{args.suite}_a{label}_s{args.seed}.json"
        if result.is_file() and not args.force_rerun:
            print(f"SKIP {result}")
            continue

        os.environ["CLASSIFIER_WEIGHT_PATH"] = str(checkpoint)
        cell = run_root / args.suite / f"a{label}_s{args.seed}"
        clean = benchmark_suite(
            suite, args.model, cell / "clean", force_rerun=args.force_rerun,
            benchmark_version=args.benchmark_version, defense="full_cad",
        )
        attacked = benchmark_suite(
            suite, args.model, cell / "autodojo", force_rerun=args.force_rerun,
            benchmark_version=args.benchmark_version, attack="autodojo", defense="full_cad",
        )
        row = {
            "benchmark": "adaptive_autodojo_agentdojo", "family": args.family,
            "suite": args.suite, "alpha": alpha, "seed": args.seed,
            "CU": rate(clean["utility_results"]),
            "UA": rate(attacked["utility_results"]),
            "ASR": rate(attacked["security_results"]),
            "checkpoint": str(checkpoint), "cache": str(args.cache.resolve()),
        }
        result.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(f"DONE {result}")


if __name__ == "__main__":
    main()
