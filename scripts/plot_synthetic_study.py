#!/usr/bin/env python3
"""Create crossover heat maps and replicate distributions from benchmark logs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def validate_run_identity(paths: list[Path], override: str | None) -> str:
    if override is not None:
        return override
    identities: list[str | None] = []
    prefix = "# cache_identity sha256="
    for path in paths:
        found = {
            line.strip()[len(prefix) :]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(prefix)
        }
        if len(found) > 1:
            raise ValueError(f"{path} contains multiple cache identities")
        identities.append(next(iter(found)) if found else None)
    if len(paths) == 1 and identities[0] is None:
        return str(paths[0].resolve())
    if any(identity is None for identity in identities):
        raise ValueError(
            "multiple logs must carry one common cache identity; otherwise "
            "supply --run-identity after verifying that they came from the "
            "same hardware and benchmark protocol"
        )
    if len(set(identities)) != 1:
        raise ValueError("benchmark logs have different cache identities")
    return str(identities[0])


def study_design(paths: list[Path]) -> tuple[set[str], set[int], set[int], int, str]:
    keys = {
        "topology_distributions",
        "leaf_counts",
        "unique_pattern_counts",
        "topology_replicates",
        "deterministic_seed_base",
    }
    designs: list[dict[str, str]] = []
    for path in paths:
        values: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line.startswith("# ") or "=" not in line:
                continue
            key, value = line[2:].split("=", 1)
            if key in keys:
                values[key] = value
        missing = keys - set(values)
        if missing:
            raise ValueError(
                f"{path} lacks synthetic-study declarations: "
                f"{', '.join(sorted(missing))}"
            )
        designs.append(values)
    if any(design != designs[0] for design in designs[1:]):
        raise ValueError("synthetic logs declare different study designs")
    design = designs[0]
    try:
        return (
            set(design["topology_distributions"].split()),
            {int(value) for value in design["leaf_counts"].split()},
            {int(value) for value in design["unique_pattern_counts"].split()},
            int(design["topology_replicates"]),
            design["deterministic_seed_base"],
        )
    except ValueError as error:
        raise ValueError("invalid synthetic-study declaration") from error


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
                    header = (
                        fields
                        if "measured_total_ms" in fields
                        or "end_to_end_ms" in fields
                        or "beagle_total_ms" in fields
                        else None
                    )
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
    field = (
        "measured_total_ms"
        if "measured_total_ms" in row
        else "end_to_end_ms"
        if "end_to_end_ms" in row
        else "beagle_total_ms"
    )
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
    parser.add_argument(
        "--run-identity",
        help="declared common run identity for logs without cache markers",
    )
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--max-relative-error", type=float)
    parser.add_argument(
        "--distribution-leaves",
        type=int,
        help="taxon count used for the replicate-distribution panel (default: largest matched count)",
    )
    parser.add_argument(
        "--distribution-patterns",
        type=int,
        help="pattern count used for the replicate-distribution panel (default: largest matched count)",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    run_identity = validate_run_identity(arguments.logs, arguments.run_identity)
    expected_topologies, expected_leaves, expected_patterns, expected_count, \
        expected_seed_base = study_design(arguments.logs)
    max_abs_error = (
        arguments.max_abs_error
        if arguments.max_abs_error is not None
        else (0.1 if arguments.precision == "FP32" else 1e-8)
    )
    max_relative_error = (
        arguments.max_relative_error
        if arguments.max_relative_error is not None
        else (1e-3 if arguments.precision == "FP32" else 1e-10)
    )

    selected_methods = {arguments.native, arguments.baseline}
    records = []
    for row in rows(arguments.logs):
        if row["precision"] != arguments.precision:
            continue
        if (
            row.get("benchmark_mode", "full-input-update")
            != arguments.benchmark_mode
        ):
            continue
        row_method = method(row)
        if row_method not in selected_methods:
            continue
        if (
            row_method == arguments.baseline
            and int(row.get("threads", "1")) != arguments.beagle_threads
        ):
            continue
        if (
            arguments.benchmark_mode != "full-input-update"
            and row.get("site_batch") != row.get("unique_patterns")
        ):
            raise ValueError(
                f"{row['source']}: chunked resident timing is a projection, "
                "not a complete-alignment measurement"
            )
        try:
            row_elapsed = elapsed(row)
            absolute_error = float(row["max_abs_error"])
            relative_error = float(row["max_relative_error"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid timing/error record at {row['source']}") from error
        if not all(
            math.isfinite(value)
            for value in (row_elapsed, absolute_error, relative_error)
        ) or row_elapsed <= 0.0:
            raise ValueError(f"nonpositive or nonfinite result at {row['source']}")
        if absolute_error > max_abs_error or relative_error > max_relative_error:
            raise ValueError(
                f"correctness threshold exceeded at {row['source']}: "
                f"abs={absolute_error}, relative={relative_error}"
            )
        records.append(row)
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    key_fields = ("topology", "seed_base", "seed", "replicate", "leaves", "unique_patterns")
    for row in records:
        key = tuple(row[field] for field in key_fields)
        name = method(row)
        if name in indexed[key]:
            raise ValueError(
                f"duplicate {name} record for {dict(zip(key_fields, key))}; "
                "select one run before plotting"
            )
        indexed[key][name] = row
    incomplete = {
        key: selected_methods - set(methods)
        for key, methods in indexed.items()
        if selected_methods - set(methods)
    }
    if incomplete:
        key, missing = next(iter(incomplete.items()))
        raise ValueError(
            f"incomplete native/BEAGLE pair for {dict(zip(key_fields, key))}: "
            f"missing {', '.join(sorted(missing))}"
        )
    paired: list[dict[str, object]] = []
    for key, methods in indexed.items():
        if arguments.native not in methods or arguments.baseline not in methods:
            continue
        native = methods[arguments.native]
        baseline = methods[arguments.baseline]
        for field in (
            "nodes",
            "site_batch",
            "tree_height",
            "normalized_colless",
            "structural_rounds",
            "primitive_levels",
            "primitive_operations",
        ):
            if native[field] != baseline[field]:
                raise ValueError(
                    f"native/BEAGLE {field} mismatch for "
                    f"{dict(zip(key_fields, key))}"
                )
        paired.append(
            dict(zip(key_fields, key))
            | {
                "precision": arguments.precision,
                "native": arguments.native,
                "baseline": arguments.baseline,
                "benchmark_mode": arguments.benchmark_mode,
                "beagle_threads": arguments.beagle_threads,
                "run_identity": run_identity,
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

    topologies_present = {str(row["topology"]) for row in paired}
    leaves_present = {int(str(row["leaves"])) for row in paired}
    patterns_present = {int(str(row["unique_patterns"])) for row in paired}
    if (
        topologies_present != expected_topologies
        or leaves_present != expected_leaves
        or patterns_present != expected_patterns
    ):
        raise ValueError(
            "observed synthetic grid axes do not match the declared study design"
        )
    replicates_by_cell: dict[tuple[str, int, int], set[tuple[str, str]]] = {}
    for topology in topologies_present:
        for leaf_count in leaves_present:
            for pattern_count in patterns_present:
                labels = {
                    (str(row["seed_base"]), str(row["replicate"]))
                    for row in paired
                    if row["topology"] == topology
                    and int(str(row["leaves"])) == leaf_count
                    and int(str(row["unique_patterns"])) == pattern_count
                }
                if not labels:
                    raise ValueError(
                        "incomplete synthetic Cartesian grid: missing "
                        f"{topology}, {leaf_count} taxa, {pattern_count} patterns"
                    )
                replicates_by_cell[(topology, leaf_count, pattern_count)] = labels
    expected_replicates = {
        (expected_seed_base, str(replicate)) for replicate in range(expected_count)
    }
    inconsistent = [
        cell for cell, labels in replicates_by_cell.items()
        if labels != expected_replicates
    ]
    if inconsistent:
        raise ValueError(
            "synthetic grid cells have different seed-base/replicate sets; "
            f"first mismatch: {inconsistent[0]}"
        )

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    fields = list(paired[0])
    configuration = (
        f"{arguments.native}_vs_{arguments.baseline}_"
        f"{arguments.precision.lower()}_{arguments.benchmark_mode}_"
        f"{arguments.beagle_threads}t"
    )
    with (output / f"synthetic_paired_replicates_{configuration}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    import matplotlib.pyplot as plt
    import numpy as np

    topologies = sorted(topologies_present)
    leaves = sorted(leaves_present)
    patterns = sorted(patterns_present)
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
    figure.savefig(
        output / f"synthetic_crossover_heatmap_{configuration}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)

    distribution_leaves = arguments.distribution_leaves or max(leaves)
    distribution_patterns = arguments.distribution_patterns or max(patterns)
    distribution_rows = [
        row
        for row in paired
        if int(str(row["leaves"])) == distribution_leaves
        and int(str(row["unique_patterns"])) == distribution_patterns
    ]
    if not distribution_rows:
        raise ValueError(
            "no matched replicate distribution for "
            f"{distribution_leaves} taxa and {distribution_patterns} patterns"
        )
    distribution_topologies = sorted(
        {str(row["topology"]) for row in distribution_rows}
    )
    figure, axis = plt.subplots(
        figsize=(max(6, 1.3 * len(distribution_topologies)), 4)
    )
    values = [
        [
            math.log2(float(row["speedup"]))
            for row in distribution_rows
            if row["topology"] == topology
        ]
        for topology in distribution_topologies
    ]
    axis.violinplot(values, showmedians=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        range(1, len(distribution_topologies) + 1),
        labels=distribution_topologies,
        rotation=20,
    )
    axis.set_ylabel(
        f"log2 {arguments.native} speedup over {arguments.baseline} "
        f"({arguments.benchmark_mode})"
    )
    axis.set_title(
        f"Replicates conditioned on {distribution_leaves} taxa and "
        f"{distribution_patterns} patterns"
    )
    figure.savefig(
        output / f"synthetic_speedup_violins_{configuration}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
