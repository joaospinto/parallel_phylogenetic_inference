#!/usr/bin/env python3
"""Summarize CPU-versus-accelerator inference-task benchmark records."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


REQUIRED = (
    "backend",
    "precision",
    "benchmark_mode",
    "task",
    "dataset",
    "topology",
    "leaves",
    "nodes",
    "sites",
    "unique_patterns",
    "repeats",
    "cpu_ms",
    "accelerator_ms",
    "cpu_over_accelerator",
    "max_abs_error",
    "max_relative_error",
    "state_mismatches",
    "cpu_samples_ms",
    "accelerator_samples_ms",
)


def records(paths: list[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        header: list[str] | None = None
        for line_number, text in enumerate(path.read_text().splitlines(), 1):
            if not text or text.startswith("#"):
                continue
            fields = next(csv.reader([text]))
            if fields[0] in {"backend", "baseline"}:
                if len(set(fields)) != len(fields):
                    raise ValueError(f"{path}:{line_number}: duplicate CSV column")
                header = fields if all(field in fields for field in REQUIRED) else None
                continue
            if header is None or fields[0] not in {"cuda", "metal", "rocm"}:
                continue
            if len(fields) != len(header):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(header)} fields, "
                    f"found {len(fields)}"
                )
            row = dict(zip(header, fields))
            row["source"] = f"{path}:{line_number}"
            validate(row)
            result.append(row)
    if not result:
        raise ValueError("no inference-task benchmark CSV records found")
    return result


def samples(text: str, source: str) -> list[float]:
    try:
        values = [float(value) for value in text.split("|")]
    except ValueError as error:
        raise ValueError(f"{source}: invalid timing sample") from error
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f"{source}: timing samples must be finite and positive")
    return values


def validate(row: dict[str, str]) -> None:
    source = row["source"]
    if row["benchmark_mode"] not in {
        "full-input-update",
        "factor-update",
        "fixed-model",
    }:
        raise ValueError(f"{source}: invalid benchmark mode")
    if row["task"] not in {
        "likelihood",
        "joint-map",
        "posterior-sample",
        "all-marginals",
    }:
        raise ValueError(f"{source}: invalid inference task")
    try:
        repeats = int(row["repeats"])
        cpu = float(row["cpu_ms"])
        accelerator = float(row["accelerator_ms"])
        speedup = float(row["cpu_over_accelerator"])
        absolute = float(row["max_abs_error"])
        relative = float(row["max_relative_error"])
        mismatches = int(row["state_mismatches"])
    except ValueError as error:
        raise ValueError(f"{source}: invalid numeric field") from error
    if repeats <= 0 or cpu <= 0 or accelerator <= 0:
        raise ValueError(f"{source}: timing fields must be positive")
    if absolute < 0 or relative < 0 or mismatches < 0:
        raise ValueError(f"{source}: correctness fields must be nonnegative")
    if not math.isclose(speedup, cpu / accelerator, rel_tol=2e-8, abs_tol=1e-10):
        raise ValueError(f"{source}: speedup does not equal cpu_ms/accelerator_ms")
    if len(samples(row["cpu_samples_ms"], source)) != repeats or len(
        samples(row["accelerator_samples_ms"], source)
    ) != repeats:
        raise ValueError(f"{source}: timing sample count does not equal repeats")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--backend", choices=("cuda", "metal", "rocm"))
    parser.add_argument("--precision", choices=("FP32", "FP64"))
    parser.add_argument(
        "--benchmark-mode",
        choices=("full-input-update", "factor-update", "fixed-model"),
    )
    arguments = parser.parse_args()
    rows = records(arguments.logs)
    for field, selected in (
        ("backend", arguments.backend),
        ("precision", arguments.precision),
        ("benchmark_mode", arguments.benchmark_mode),
    ):
        if selected is not None:
            rows = [row for row in rows if row[field] == selected]
    if not rows:
        raise ValueError("no task records satisfy the requested filters")

    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[
            (row["backend"], row["precision"], row["benchmark_mode"], row["task"])
        ].append(row)
    print(
        "backend,precision,benchmark_mode,task,problems,median_speedup,"
        "minimum_speedup,maximum_speedup,max_abs_error,max_relative_error,"
        "state_mismatches"
    )
    for key in sorted(groups):
        group = groups[key]
        speedups = [float(row["cpu_over_accelerator"]) for row in group]
        print(
            ",".join(
                (
                    *key,
                    str(len(group)),
                    f"{statistics.median(speedups):.10g}",
                    f"{min(speedups):.10g}",
                    f"{max(speedups):.10g}",
                    f"{max(float(row['max_abs_error']) for row in group):.10g}",
                    f"{max(float(row['max_relative_error']) for row in group):.10g}",
                    str(sum(int(row["state_mismatches"]) for row in group)),
                )
            )
        )


if __name__ == "__main__":
    main()
