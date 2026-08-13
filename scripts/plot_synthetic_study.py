#!/usr/bin/env python3
"""Create crossover heat maps and replicate distributions from benchmark logs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def rows(paths: list[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        header: list[str] | None = None
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = next(csv.reader([line]))
                if fields[0] in {"backend", "baseline"}:
                    header = fields
                    continue
                if header is None or len(fields) != len(header):
                    continue
                row = dict(zip(header, fields))
                if row.get("dataset") != "synthetic" or "replicate" not in row:
                    continue
                row["source"] = f"{path}:{line_number}"
                result.append(row)
    return result


def method(row: dict[str, str]) -> str:
    if row.get("backend") in {"cuda", "rocm", "metal"}:
        return row["backend"]
    if row.get("baseline") == "beagle":
        return f"beagle-{row['beagle_resource']}"
    raise ValueError(f"unknown record at {row['source']}")


def elapsed(row: dict[str, str]) -> float:
    field = "end_to_end_ms" if "end_to_end_ms" in row else "beagle_total_ms"
    return float(row[field])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--native", choices=("cuda", "rocm", "metal"), required=True)
    parser.add_argument("--baseline", choices=("beagle-cpu", "beagle-cuda"), required=True)
    parser.add_argument("--precision", choices=("FP32", "FP64"), required=True)
    parser.add_argument(
        "--benchmark-mode",
        choices=("fixed-model", "factor-update", "full-input-update"),
        default="full-input-update",
    )
    parser.add_argument("--beagle-threads", type=int, default=1)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    records = []
    for row in rows(arguments.logs):
        if row["precision"] != arguments.precision:
            continue
        if (
            row.get("benchmark_mode", arguments.benchmark_mode)
            != arguments.benchmark_mode
        ):
            continue
        if (
            method(row) == arguments.baseline
            and int(row.get("threads", "1")) != arguments.beagle_threads
        ):
            continue
        records.append(row)
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    key_fields = ("topology", "seed_base", "seed", "replicate", "leaves", "unique_patterns")
    for row in records:
        indexed[tuple(row[field] for field in key_fields)][method(row)] = row
    paired: list[dict[str, object]] = []
    for key, methods in indexed.items():
        if arguments.native not in methods or arguments.baseline not in methods:
            continue
        native = methods[arguments.native]
        baseline = methods[arguments.baseline]
        paired.append(
            dict(zip(key_fields, key))
            | {
                "precision": arguments.precision,
                "native": arguments.native,
                "baseline": arguments.baseline,
                "benchmark_mode": arguments.benchmark_mode,
                "beagle_threads": arguments.beagle_threads,
                "native_ms": elapsed(native),
                "baseline_ms": elapsed(baseline),
                "speedup": elapsed(baseline) / elapsed(native),
                "tree_height": native["tree_height"],
                "normalized_colless": native["normalized_colless"],
                "primitive_levels": native["primitive_levels"],
            }
        )
    if not paired:
        raise ValueError("no matched native/BEAGLE synthetic records")

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    fields = list(paired[0])
    with (output / "synthetic_paired_replicates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    import matplotlib.pyplot as plt
    import numpy as np

    topologies = sorted({str(row["topology"]) for row in paired})
    leaves = sorted({int(str(row["leaves"])) for row in paired})
    patterns = sorted({int(str(row["unique_patterns"])) for row in paired})
    figure, axes = plt.subplots(1, len(topologies), figsize=(4.3 * len(topologies), 4), squeeze=False)
    for axis, topology in zip(axes[0], topologies):
        matrix = np.full((len(leaves), len(patterns)), np.nan)
        for i, leaf_count in enumerate(leaves):
            for j, pattern_count in enumerate(patterns):
                values = [
                    math.log2(float(row["speedup"]))
                    for row in paired
                    if row["topology"] == topology
                    and int(str(row["leaves"])) == leaf_count
                    and int(str(row["unique_patterns"])) == pattern_count
                ]
                if values:
                    matrix[i, j] = statistics.median(values)
        image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        axis.set_title(topology)
        axis.set_xticks(range(len(patterns)), labels=patterns, rotation=45)
        axis.set_yticks(range(len(leaves)), labels=leaves)
        axis.set_xlabel("unique patterns")
        axis.set_ylabel("taxa")
    figure.colorbar(image, ax=axes.ravel().tolist(), label="median log2 speedup")
    figure.savefig(output / "synthetic_crossover_heatmap.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(6, 1.3 * len(topologies)), 4))
    values = [[float(row["speedup"]) for row in paired if row["topology"] == topology] for topology in topologies]
    axis.violinplot(values, showmedians=True)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xticks(range(1, len(topologies) + 1), labels=topologies, rotation=20)
    axis.set_yscale("log", base=2)
    axis.set_ylabel(f"{arguments.native} end-to-end speedup over {arguments.baseline}")
    figure.savefig(output / "synthetic_speedup_violins.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
