#!/usr/bin/env python3
"""Plot matched complete-likelihood timings for clock-like JC69 simulations."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
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
                if row.get("dataset") != "synthetic-jc69":
                    continue
                required = {
                    "topology",
                    "evolutionary_root_to_tip_distance",
                    "seed_base",
                    "seed",
                    "replicate",
                    "leaves",
                    "sites",
                    "unique_patterns",
                }
                if not required.issubset(row):
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
    parser.add_argument(
        "--baseline", choices=("beagle-cpu", "beagle-cuda"), required=True
    )
    parser.add_argument("--precision", choices=("FP32", "FP64"), required=True)
    parser.add_argument(
        "--benchmark-mode",
        choices=("fixed-model", "factor-update", "full-input-update"),
        default="full-input-update",
    )
    parser.add_argument("--beagle-threads", type=int, default=1)
    parser.add_argument("--distribution-leaves", type=int)
    parser.add_argument("--distribution-raw-sites", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    records = []
    for row in read_rows(arguments.logs):
        if row["precision"] != arguments.precision:
            continue
        row_method = method(row)
        if row_method == arguments.baseline:
            if row.get("benchmark_mode") != arguments.benchmark_mode:
                continue
            if int(row.get("threads", "1")) != arguments.beagle_threads:
                continue
        records.append(row)

    key_fields = (
        "topology",
        "evolutionary_root_to_tip_distance",
        "seed_base",
        "seed",
        "replicate",
        "leaves",
        "sites",
    )
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in records:
        indexed[tuple(row[field] for field in key_fields)][method(row)] = row

    paired: list[dict[str, object]] = []
    for key, methods in indexed.items():
        if arguments.native not in methods or arguments.baseline not in methods:
            continue
        native = methods[arguments.native]
        baseline = methods[arguments.baseline]
        native_unique = int(native["unique_patterns"])
        baseline_unique = int(baseline["unique_patterns"])
        if native_unique != baseline_unique:
            raise ValueError(f"pattern compression mismatch for case {key}")
        raw_sites = int(native["sites"])
        paired.append(
            dict(zip(key_fields, key))
            | {
                "precision": arguments.precision,
                "native": arguments.native,
                "baseline": arguments.baseline,
                "benchmark_mode": arguments.benchmark_mode,
                "beagle_threads": arguments.beagle_threads,
                "unique_patterns": native_unique,
                "unique_pattern_fraction": native_unique / raw_sites,
                "native_ms": elapsed(native),
                "baseline_ms": elapsed(baseline),
                "speedup": elapsed(baseline) / elapsed(native),
                "topological_height_edges": native["tree_height"],
                "normalized_colless": native["normalized_colless"],
                "primitive_levels": native["primitive_levels"],
            }
        )
    if not paired:
        raise ValueError("no matched native/BEAGLE JC69 simulation records")

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    configuration = (
        f"{arguments.precision.lower()}_{arguments.native}_vs_"
        f"{arguments.baseline}_{arguments.benchmark_mode}_t{arguments.beagle_threads}"
    )
    paired.sort(
        key=lambda row: (
            str(row["topology"]),
            float(str(row["evolutionary_root_to_tip_distance"])),
            int(str(row["leaves"])),
            int(str(row["sites"])),
            int(str(row["replicate"])),
        )
    )
    with (output / f"jc69_paired_replicates_{configuration}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    import matplotlib.pyplot as plt
    import numpy as np

    topologies = sorted({str(row["topology"]) for row in paired})
    heights = sorted(
        {float(str(row["evolutionary_root_to_tip_distance"])) for row in paired}
    )
    leaves = sorted({int(str(row["leaves"])) for row in paired})
    raw_sites = sorted({int(str(row["sites"])) for row in paired})
    for topology in topologies:
        figure, axes = plt.subplots(
            1, len(heights), figsize=(4.1 * len(heights), 3.8), squeeze=False
        )
        image = None
        for axis, height in zip(axes[0], heights):
            matrix = np.full((len(leaves), len(raw_sites)), np.nan)
            for i, leaf_count in enumerate(leaves):
                for j, raw_site_count in enumerate(raw_sites):
                    values = [
                        math.log2(float(row["speedup"]))
                        for row in paired
                        if row["topology"] == topology
                        and float(str(row["evolutionary_root_to_tip_distance"]))
                        == height
                        and int(str(row["leaves"])) == leaf_count
                        and int(str(row["sites"])) == raw_site_count
                    ]
                    if values:
                        matrix[i, j] = statistics.median(values)
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-3,
                vmax=3,
            )
            axis.set_title(f"root-to-tip distance {height:g}")
            axis.set_xticks(range(len(raw_sites)), labels=raw_sites, rotation=45)
            axis.set_yticks(range(len(leaves)), labels=leaves)
            axis.set_xlabel("raw sequence length")
            axis.set_ylabel("taxa")
        assert image is not None
        figure.colorbar(image, ax=axes.ravel().tolist(), label="median log2 speedup")
        figure.savefig(
            output / f"jc69_crossover_{topology}_{configuration}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)

    distribution_leaves = arguments.distribution_leaves or max(leaves)
    distribution_raw_sites = arguments.distribution_raw_sites or max(raw_sites)
    selected = [
        row
        for row in paired
        if int(str(row["leaves"])) == distribution_leaves
        and int(str(row["sites"])) == distribution_raw_sites
    ]
    groups = [(topology, height) for topology in topologies for height in heights]
    values = [
        [
            math.log2(float(row["speedup"]))
            for row in selected
            if row["topology"] == topology
            and float(str(row["evolutionary_root_to_tip_distance"])) == height
        ]
        for topology, height in groups
    ]
    if not all(values):
        raise ValueError(
            "the requested violin cell is incomplete across topology and height"
        )
    figure, axis = plt.subplots(figsize=(max(8, 0.9 * len(groups)), 4.2))
    axis.violinplot(values, showmedians=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        range(1, len(groups) + 1),
        labels=[f"{topology}\n{height:g}" for topology, height in groups],
        rotation=25,
    )
    axis.set_ylabel(f"log2 {arguments.native} speedup over {arguments.baseline}")
    axis.set_title(
        f"{distribution_leaves} taxa; {distribution_raw_sites} raw sites"
    )
    figure.savefig(
        output / f"jc69_speedup_violins_{configuration}.pdf", bbox_inches="tight"
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1, len(topologies), figsize=(4.1 * len(topologies), 3.8), squeeze=False
    )
    for axis, topology in zip(axes[0], topologies):
        for raw_site_count in raw_sites:
            medians = []
            for height in heights:
                fractions = [
                    float(row["unique_pattern_fraction"])
                    for row in paired
                    if row["topology"] == topology
                    and int(str(row["leaves"])) == distribution_leaves
                    and int(str(row["sites"])) == raw_site_count
                    and float(str(row["evolutionary_root_to_tip_distance"]))
                    == height
                ]
                medians.append(statistics.median(fractions) if fractions else math.nan)
            axis.plot(heights, medians, marker="o", label=f"{raw_site_count} sites")
        axis.set_xscale("log")
        axis.set_ylim(0.0, 1.02)
        axis.set_title(topology)
        axis.set_xlabel("evolutionary root-to-tip distance")
        axis.set_ylabel("unique patterns / raw sites")
        axis.legend(fontsize="small")
    figure.savefig(
        output / f"jc69_pattern_compression_{configuration}.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
