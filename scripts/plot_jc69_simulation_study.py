#!/usr/bin/env python3
"""Plot matched complete-likelihood timings for clock-like JC69 simulations."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_interleaved_schedule import validate_interleaved_schedule


STUDY = "clock-like-jc69-simulation"


def prescribed_timing_repeats(
    leaves: int,
    raw_sites: int,
    minimum: int,
    maximum: int,
    node_site_budget: int,
) -> int:
    node_sites = (2 * leaves - 1) * raw_sites
    if leaves <= 0 or raw_sites <= 0 or node_sites <= 0:
        raise ValueError("taxon and raw-site counts must be positive")
    if node_site_budget == 0:
        return maximum
    return max(minimum, min(maximum, node_site_budget // node_sites))


def validate_run_identity(paths: list[Path], override: str | None) -> str:
    if override is not None:
        return override
    prefix = "# cache_identity sha256="
    identities: list[str | None] = []
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
            "supply --run-identity after verifying their provenance"
        )
    if len(set(identities)) != 1:
        raise ValueError("benchmark logs have different cache identities")
    return str(identities[0])


def study_design(
    paths: list[Path],
) -> tuple[
    set[str], set[int], set[int], set[float], int, str, str,
    int, int, int, int, int
]:
    keys = {
        "topology_distributions",
        "leaf_counts",
        "raw_sequence_lengths",
        "evolutionary_root_to_tip_distances",
        "topology_and_alignment_replicates",
        "deterministic_seed_base",
        "profile",
        "minimum_timing_repeats",
        "maximum_timing_repeats",
        "timing_work_budget_node_sites",
        "cases_per_method_and_precision",
        "timing_repeat_rule",
        "conditioning_ms",
        "replicate_blocking",
    }
    designs: list[dict[str, str]] = []
    for path in paths:
        values: dict[str, str] = {}
        active_study: str | None = None
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line.startswith("# ") or "=" not in line:
                continue
            key, value = line[2:].split("=", 1)
            if key == "study":
                active_study = value
                continue
            if active_study == STUDY and key in keys:
                if key in values and values[key] != value:
                    raise ValueError(
                        f"{path} contains conflicting {STUDY} {key} declarations"
                    )
                values[key] = value
        missing = keys - set(values)
        if missing:
            raise ValueError(
                f"{path} lacks {STUDY} declarations: "
                f"{', '.join(sorted(missing))}"
            )
        designs.append(values)
    if any(design != designs[0] for design in designs[1:]):
        raise ValueError("JC69 logs declare different study designs")
    design = designs[0]
    try:
        return (
            set(design["topology_distributions"].split()),
            {int(value) for value in design["leaf_counts"].split()},
            {int(value) for value in design["raw_sequence_lengths"].split()},
            {
                float(value)
                for value in design["evolutionary_root_to_tip_distances"].split()
            },
            int(design["topology_and_alignment_replicates"]),
            design["deterministic_seed_base"],
            design["profile"],
            int(design["minimum_timing_repeats"]),
            int(design["maximum_timing_repeats"]),
            int(design["timing_work_budget_node_sites"]),
            int(design["cases_per_method_and_precision"]),
            int(design["conditioning_ms"]),
        )
    except ValueError as error:
        raise ValueError("invalid JC69 study declaration") from error


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
                if (
                    row.get("dataset") != "synthetic-jc69"
                    or row.get("study") != STUDY
                ):
                    continue
                required = {
                    "topology",
                    "sequence_generation",
                    "evolutionary_root_to_tip_distance",
                    "seed_base",
                    "seed",
                    "replicate",
                    "leaves",
                    "sites",
                    "unique_patterns",
                    "structural_rounds",
                    "primitive_levels",
                    "primitive_operations",
                    "repeats",
                    "conditioning_ms",
                    "max_abs_error",
                    "max_relative_error",
                }
                if not required.issubset(row):
                    missing = required - set(row)
                    raise ValueError(
                        f"{path}:{line_number}: JC69 row lacks "
                        f"{', '.join(sorted(missing))}"
                    )
                row["source"] = f"{path}:{line_number}"
                if row["sequence_generation"] != "jc69":
                    raise ValueError(
                        f"{row['source']}: JC69 study row is not a JC69 simulation"
                    )
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
        else "total_accelerator_ms"
        if "total_accelerator_ms" in row
        else "beagle_total_ms"
    )
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
    parser.add_argument(
        "--run-identity",
        help="declared common identity for logs without cache markers",
    )
    parser.add_argument(
        "--max-abs-error", type=float,
        help="optional additional absolute-error guard",
    )
    parser.add_argument(
        "--max-relative-error", type=float,
        help="scale-normalized error bound (default: 2e-3 FP32, 1e-10 FP64)",
    )
    parser.add_argument("--distribution-leaves", type=int)
    parser.add_argument("--distribution-raw-sites", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate and write paired CSV data without rendering figures",
    )
    parser.add_argument(
        "--require-interleaved", action="store_true",
        help="require and validate a complete cyclic case-level method schedule",
    )
    arguments = parser.parse_args()
    run_identity = validate_run_identity(arguments.logs, arguments.run_identity)
    (
        expected_topologies,
        expected_leaves,
        expected_raw_sites,
        expected_heights,
        expected_replicates,
        expected_seed_base,
        expected_profile,
        minimum_timing_repeats,
        maximum_timing_repeats,
        timing_work_budget,
        declared_case_count,
        declared_conditioning_ms,
    ) = study_design(arguments.logs)
    if minimum_timing_repeats <= 0 or (
        maximum_timing_repeats < minimum_timing_repeats
    ) or timing_work_budget < 0:
        raise ValueError("invalid JC69 timing-repeat declaration")
    if arguments.require_interleaved:
        baseline_specification = (
            f"beagle-cpu:{arguments.beagle_threads}"
            if arguments.baseline == "beagle-cpu"
            else "beagle-cuda"
        )
        validate_interleaved_schedule(
            arguments.logs,
            study=STUDY,
            precision=arguments.precision,
            benchmark_mode=arguments.benchmark_mode,
            expected_cases=declared_case_count,
            required_methods={arguments.native, baseline_specification},
        )
    max_relative_error = (
        arguments.max_relative_error
        if arguments.max_relative_error is not None
        else (2e-3 if arguments.precision == "FP32" else 1e-10)
    )

    records = []
    selected_methods = {arguments.native, arguments.baseline}
    for row in read_rows(arguments.logs):
        if row["precision"] != arguments.precision:
            continue
        if row.get("benchmark_mode", "full-input-update") != arguments.benchmark_mode:
            continue
        row_method = method(row)
        if row_method not in selected_methods:
            continue
        if row_method == arguments.baseline:
            if int(row.get("threads", "1")) != arguments.beagle_threads:
                continue
        if (
            arguments.benchmark_mode != "full-input-update"
            and row.get("site_batch") != row.get("unique_patterns")
        ):
            raise ValueError(
                f"{row['source']}: chunked resident timing is a projection"
            )
        try:
            values = (
                elapsed(row),
                float(row["max_abs_error"]),
                float(row["max_relative_error"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid result at {row['source']}") from error
        if not all(math.isfinite(value) for value in values) or values[0] <= 0:
            raise ValueError(f"nonfinite or nonpositive result at {row['source']}")
        if (
            arguments.max_abs_error is not None
            and values[1] > arguments.max_abs_error
        ) or values[2] > max_relative_error:
            raise ValueError(f"correctness threshold exceeded at {row['source']}")
        expected_timing_repeats = prescribed_timing_repeats(
            int(row["leaves"]), int(row["sites"]), minimum_timing_repeats,
            maximum_timing_repeats, timing_work_budget,
        )
        if int(row["repeats"]) != expected_timing_repeats:
            raise ValueError(
                f"{row['source']}: timing repeats do not match the "
                f"declared {expected_profile} profile"
            )
        if int(row["conditioning_ms"]) != declared_conditioning_ms:
            raise ValueError(
                f"{row['source']}: conditioning does not match the declared profile"
            )
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
        key = tuple(row[field] for field in key_fields)
        name = method(row)
        if name in indexed[key]:
            raise ValueError(f"duplicate {name} record for case {key}")
        indexed[key][name] = row

    incomplete = {
        key: selected_methods - set(methods)
        for key, methods in indexed.items()
        if selected_methods - set(methods)
    }
    if incomplete:
        key, missing = next(iter(incomplete.items()))
        raise ValueError(
            f"incomplete native/BEAGLE pair for {key}: "
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
            "sequence_generation",
            "unique_patterns",
            "tree_height",
            "normalized_colless",
            "structural_rounds",
            "primitive_levels",
            "primitive_operations",
        ):
            if native[field] != baseline[field]:
                raise ValueError(f"native/BEAGLE {field} mismatch for case {key}")
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
                "run_identity": run_identity,
                "unique_patterns": native_unique,
                "unique_pattern_fraction": native_unique / raw_sites,
                "native_site_batch": int(native["site_batch"]),
                "baseline_site_batch": int(baseline["site_batch"]),
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

    observed_axes = (
        {str(row["topology"]) for row in paired},
        {int(str(row["leaves"])) for row in paired},
        {int(str(row["sites"])) for row in paired},
        {float(str(row["evolutionary_root_to_tip_distance"])) for row in paired},
    )
    if observed_axes != (
        expected_topologies,
        expected_leaves,
        expected_raw_sites,
        expected_heights,
    ):
        raise ValueError("observed JC69 grid axes do not match the declaration")
    expected_count = (
        len(expected_topologies)
        * len(expected_leaves)
        * len(expected_raw_sites)
        * len(expected_heights)
        * expected_replicates
    )
    if declared_case_count != expected_count:
        raise ValueError(
            f"JC69 design declares {declared_case_count} cases but its axes "
            f"and replicate count imply {expected_count}"
        )
    if len(paired) != expected_count:
        raise ValueError(
            f"JC69 grid has {len(paired)} pairs; expected {expected_count}"
        )
    for row in paired:
        if str(row["seed_base"]) != expected_seed_base or not (
            0 <= int(str(row["replicate"])) < expected_replicates
        ):
            raise ValueError("JC69 replicate or seed lies outside the declaration")
    expected_cells = {
        (topology, leaves, sites, height, replicate)
        for topology in expected_topologies
        for leaves in expected_leaves
        for sites in expected_raw_sites
        for height in expected_heights
        for replicate in range(expected_replicates)
    }
    observed_cells = {
        (
            str(row["topology"]), int(str(row["leaves"])),
            int(str(row["sites"])),
            float(str(row["evolutionary_root_to_tip_distance"])),
            int(str(row["replicate"])),
        )
        for row in paired
    }
    if observed_cells != expected_cells:
        raise ValueError("observed JC69 cases do not equal the declared Cartesian grid")

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

    if arguments.validate_only:
        return

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
