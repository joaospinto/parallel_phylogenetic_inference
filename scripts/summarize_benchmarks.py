#!/usr/bin/env python3
"""Summarize native and BEAGLE benchmark CSV records embedded in logs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


IDENTITY = (
    "precision",
    "benchmark_mode",
    "study",
    "dataset",
    "topology",
    "minimum_branch_length",
    "floored_branch_count",
    "leaves",
    "nodes",
    "sites",
    "unique_patterns",
    "site_batch",
)


def row_identity(row: dict[str, str], include_site_batch: bool = True) -> tuple[str, ...]:
    fields = IDENTITY if include_site_batch else IDENTITY[:-1]
    identity = tuple(row[field] for field in fields)
    if row["dataset"].startswith("synthetic"):
        identity += tuple(
            row.get(field, "")
            for field in (
                "sequence_generation",
                "evolutionary_root_to_tip_distance",
                "seed_base",
                "seed",
                "replicate",
            )
        )
    return identity


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
            "supply --run-identity after verifying that they came from the "
            "same hardware and benchmark protocol"
        )
    if len(set(identities)) != 1:
        raise ValueError("benchmark logs have different cache identities")
    return str(identities[0])


def records(paths: list[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        header: list[str] | None = None
        with path.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = next(csv.reader([line]))
                if fields[0] in {"backend", "baseline"}:
                    # Inference-task records deliberately have a separate
                    # schema and summarizer even though their first column is
                    # also named backend.
                    header = (
                        fields
                        if "measured_total_ms" in fields
                        or "end_to_end_ms" in fields
                        or "total_accelerator_ms" in fields
                        or "beagle_total_ms" in fields
                        else None
                    )
                    continue
                if header is None or fields[0] not in {
                    "cuda",
                    "metal",
                    "rocm",
                    "beagle",
                }:
                    continue
                if len(fields) != len(header):
                    raise ValueError(
                        f"{path}:{line_number}: expected {len(header)} CSV "
                        f"fields, found {len(fields)}"
                    )
                row = dict(zip(header, fields))
                row.setdefault("benchmark_mode", "full-input-update")
                row.setdefault("study", "standard")
                row.setdefault("minimum_branch_length", "0")
                row.setdefault("floored_branch_count", "0")
                if row.get("baseline") == "beagle":
                    row.setdefault("threads", "1")
                missing = [field for field in IDENTITY if field not in row]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: benchmark CSV record is "
                        f"missing required columns {', '.join(missing)}"
                    )
                row["source"] = str(path)
                row["measurement_scope"] = (
                    "complete-alignment-wall-time"
                    if row["benchmark_mode"] == "full-input-update"
                    or row["site_batch"] == row["unique_patterns"]
                    else "sum-of-per-chunk-resident-calls"
                )
                for field in ("max_abs_error", "max_relative_error"):
                    if not math.isfinite(number(row, field)):
                        raise ValueError(
                            f"nonfinite {field} in benchmark record from {path}:{line_number}"
                        )
                total_fields = (
                    "measured_total_ms",
                    "end_to_end_ms",
                    "total_accelerator_ms",
                    "beagle_total_ms",
                )
                total_field = next(
                    (field for field in total_fields if field in row), None
                )
                if total_field is None or not math.isfinite(
                    number(row, total_field)
                ) or number(row, total_field) <= 0.0:
                    raise ValueError(
                        f"nonpositive or nonfinite elapsed time in benchmark "
                        f"record from {path}:{line_number}"
                    )
                result.append(row)
    if not result:
        raise ValueError("no benchmark CSV records found")
    return result


def method(row: dict[str, str]) -> str:
    if row.get("backend") in {"cuda", "metal", "rocm"}:
        return row["backend"]
    if row.get("baseline") == "beagle":
        resource = row.get("beagle_resource")
        if resource not in {"cpu", "cuda"}:
            raise ValueError(
                "BEAGLE records must include beagle_resource=cpu or cuda"
            )
        if resource == "cpu":
            return f"beagle_cpu_{row['threads']}t"
        return "beagle_cuda"
    raise ValueError(f"unrecognized benchmark record from {row['source']}")


def number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"invalid {field!r} in benchmark record from {row['source']}"
        ) from error


def median_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    first = rows[0]
    native = method(first) in {"cuda", "metal", "rocm"}
    total_field = (
        "measured_total_ms"
        if native and "measured_total_ms" in first
        else "end_to_end_ms"
        if native and "end_to_end_ms" in first
        else "total_accelerator_ms"
        if native
        else "beagle_total_ms"
    )
    result: dict[str, object] = {field: first[field] for field in IDENTITY}
    for field in (
        "sequence_generation",
        "evolutionary_root_to_tip_distance",
        "seed_base",
        "seed",
        "replicate",
    ):
        if field in first:
            result[field] = first[field]
    result.update(
        method=method(first),
        measurements=len(rows),
        total_ms=statistics.median(number(row, total_field) for row in rows),
        cpu_ms=statistics.median(
            number(row, "cpu_ms" if native else "sequential_ms")
            for row in rows
        ),
        max_abs_error=max(number(row, "max_abs_error") for row in rows),
        max_relative_error=max(
            number(row, "max_relative_error") for row in rows
        ),
    )
    result["cpu_speedup"] = result["cpu_ms"] / result["total_ms"]
    return result


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(method(row), *row_identity(row))].append(row)
    return [median_rows(group) for group in groups.values()]


def best_batches(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["method"]),
            *row_identity({field: str(value) for field, value in row.items()}, False),
        )
        groups[key].append(row)
    return [
        min(group, key=lambda row: float(row["total_ms"]))
        for group in groups.values()
    ]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def write_best(rows: list[dict[str, object]]) -> None:
    fields = [
        *IDENTITY[:-1],
        "method",
        "site_batch",
        "measurements",
        "cpu_ms",
        "total_ms",
        "cpu_speedup",
        "max_abs_error",
        "max_relative_error",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in sorted(
        rows, key=lambda item: tuple(str(item.get(field, "")) for field in fields[:7])
    ):
        writer.writerow(row)


def write_corpus(rows: list[dict[str, object]], accelerator: str) -> None:
    by_problem: dict[
        tuple[str, ...], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in rows:
        key = row_identity(
            {field: str(value) for field, value in row.items()}, False
        )
        by_problem[key][str(row["method"])] = row
    native = [
        methods[accelerator]
        for methods in by_problem.values()
        if accelerator in methods
    ]
    if not native:
        raise ValueError(
            f"no {accelerator} benchmark records found"
        )
    comparisons: dict[str, list[float]] = {
        f"{accelerator}/conventional": [
            float(row["cpu_ms"]) / float(row["total_ms"]) for row in native
        ]
    }
    beagle_methods = sorted(
        {
            method
            for methods in by_problem.values()
            for method in methods
            if method.startswith("beagle_cpu_") or method == "beagle_cuda"
        }
    )
    for beagle_method in beagle_methods:
        pairs = [
            (methods[accelerator], methods[beagle_method])
            for methods in by_problem.values()
            if accelerator in methods and beagle_method in methods
        ]
        if not pairs:
            continue
        comparisons[f"{beagle_method}/conventional"] = [
            float(beagle["cpu_ms"]) / float(beagle["total_ms"])
            for _, beagle in pairs
        ]
        comparisons[f"{accelerator}/{beagle_method}"] = [
            float(beagle["total_ms"]) / float(native["total_ms"])
            for native, beagle in pairs
        ]
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "comparison",
            "problems",
            "wins",
            "min_speedup",
            "median_speedup",
            "p90_speedup",
            "max_speedup",
        ]
    )
    for comparison, speedups in comparisons.items():
        writer.writerow(
            [
                comparison,
                len(speedups),
                sum(value > 1.0 for value in speedups),
                min(speedups),
                statistics.median(speedups),
                percentile(speedups, 0.9),
                max(speedups),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--corpus", choices=("cuda", "metal", "rocm"))
    parser.add_argument("--dataset-prefix")
    parser.add_argument("--precision", choices=("FP32", "FP64"))
    parser.add_argument(
        "--benchmark-mode",
        choices=("fixed-model", "factor-update", "full-input-update"),
    )
    parser.add_argument(
        "--include-resident-projections",
        action="store_true",
        help="include chunked factor/fixed diagnostic projections",
    )
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--max-relative-error", type=float)
    parser.add_argument(
        "--run-identity",
        help="declared common run identity for logs without cache markers",
    )
    arguments = parser.parse_args()
    validate_run_identity(arguments.logs, arguments.run_identity)
    raw_rows = records(arguments.logs)
    if arguments.dataset_prefix is not None:
        raw_rows = [
            row
            for row in raw_rows
            if row["dataset"].startswith(arguments.dataset_prefix)
        ]
        if not raw_rows:
            raise ValueError(
                "no benchmark records match dataset prefix "
                f"{arguments.dataset_prefix!r}"
            )
    if arguments.precision is not None:
        raw_rows = [
            row for row in raw_rows if row["precision"] == arguments.precision
        ]
        if not raw_rows:
            raise ValueError(
                f"no benchmark records use precision {arguments.precision}"
            )
    if arguments.benchmark_mode is not None:
        raw_rows = [
            row
            for row in raw_rows
            if row["benchmark_mode"] == arguments.benchmark_mode
        ]
        if not raw_rows:
            raise ValueError(
                "no benchmark records use mode "
                f"{arguments.benchmark_mode}"
            )
    if not arguments.include_resident_projections:
        raw_rows = [
            row
            for row in raw_rows
            if row["measurement_scope"] == "complete-alignment-wall-time"
        ]
        if not raw_rows:
            raise ValueError(
                "all selected rows are chunked resident projections; pass "
                "--include-resident-projections to summarize diagnostics"
            )
    if arguments.corpus and (
        arguments.precision is None or arguments.benchmark_mode is None
    ):
        raise ValueError(
            "--corpus requires --precision and --benchmark-mode so distinct "
            "measurement protocols are never pooled"
        )
    if arguments.corpus and (
        arguments.max_abs_error is None
        or arguments.max_relative_error is None
    ):
        raise ValueError(
            "--corpus requires explicit --max-abs-error and "
            "--max-relative-error acceptance thresholds"
        )
    if arguments.max_abs_error is not None:
        rejected = [
            row
            for row in raw_rows
            if number(row, "max_abs_error") > arguments.max_abs_error
        ]
        if rejected:
            raise ValueError(
                f"{len(rejected)} benchmark records exceed --max-abs-error"
            )
    if arguments.max_relative_error is not None:
        rejected = [
            row
            for row in raw_rows
            if number(row, "max_relative_error")
            > arguments.max_relative_error
        ]
        if rejected:
            raise ValueError(
                f"{len(rejected)} benchmark records exceed "
                "--max-relative-error"
            )
    rows = best_batches(aggregate(raw_rows))
    if arguments.corpus:
        write_corpus(rows, arguments.corpus)
    else:
        write_best(rows)


if __name__ == "__main__":
    main()
