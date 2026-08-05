#!/usr/bin/env python3
"""Aggregate agent-level AT metrics across training seeds; see the "Statistical Variation Across Training Seeds" appendix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev


REPORTING_SEEDS = (0, 1, 2, 3, 4)
REPORTING_ALPHAS = (0.001, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4)
# CU/UA/ASR as defined in the paper's Experiments section; means/stds over the
# available training seeds — REPORTING_SEEDS is the superset of seed ids scanned
# ("Statistical Variation Across Training Seeds" appendix).
METRICS = ("CU", "UA", "ASR")
REQUIRED = {"benchmark", "family", "suite", "alpha", "seed", *METRICS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSON file or directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(REPORTING_ALPHAS))
    return parser.parse_args()


def result_files(inputs: list[Path]) -> list[Path]:
    files = []
    for path in inputs:
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    return files


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, str, str, float], dict[int, dict]] = defaultdict(dict)
    for path in result_files(args.inputs):
        row = json.loads(path.read_text())
        if not isinstance(row, dict):
            continue
        row.setdefault("family", path.parent.name)
        row.setdefault("benchmark", path.parent.parent.name)
        if not REQUIRED.issubset(row):
            continue
        alpha = float(row["alpha"])
        if alpha != 0 and alpha not in args.alphas:
            continue
        key = (row["benchmark"], row["family"], row["suite"], alpha)
        seed = int(row["seed"])
        if seed in groups[key]:
            raise ValueError(f"duplicate seed {seed} for {key}")
        groups[key][seed] = row

    if not groups:
        raise SystemExit("no agent-level result JSON files found")

    summary = []
    for key, by_seed in sorted(groups.items()):
        is_single_baseline = key[3] == 0 and len(by_seed) == 1
        if not is_single_baseline and tuple(sorted(by_seed)) != REPORTING_SEEDS:
            raise ValueError(f"{key} has seeds {sorted(by_seed)}; expected {list(REPORTING_SEEDS)}")
        output = dict(zip(("benchmark", "family", "suite", "alpha"), key))
        output["n_seeds"] = len(by_seed)
        for metric in METRICS:
            values = [float(row[metric]) for row in by_seed.values()]
            output[f"{metric}_mean"] = fmean(values)
            output[f"{metric}_std"] = pstdev(values)
        summary.append(output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {len(summary)} five-seed rows to {args.output}")


if __name__ == "__main__":
    main()
