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
    "study",
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


def validate_run_identity(paths: list[Path], override: str | None) -> str:
    if override is not None:
        return override
    prefix = "# cache_identity sha256="
    found_by_path: list[str | None] = []
    for path in paths:
        identities = {
            line.strip()[len(prefix) :]
            for line in path.read_text().splitlines()
            if line.strip().startswith(prefix)
        }
        if len(identities) > 1:
            raise ValueError(f"{path} contains multiple cache identities")
        found_by_path.append(next(iter(identities)) if identities else None)
    if len(paths) == 1 and found_by_path[0] is None:
        return str(paths[0].resolve())
    if any(value is None for value in found_by_path) or len(set(found_by_path)) != 1:
        raise ValueError(
            "task logs must carry one common cache identity; use "
            "--run-identity only after verifying their provenance"
        )
    return str(found_by_path[0])


def study_design(paths: list[Path], study: str) -> dict[str, str]:
    keys = {
        "task_topology", "task_leaves", "task_sites", "task_replicates",
        "task_seed_base",
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
            elif active_study == study and key in keys:
                if key in values and values[key] != value:
                    raise ValueError(f"{path} has conflicting {study} {key}")
                values[key] = value
        missing = keys - set(values)
        if missing:
            raise ValueError(
                f"{path} lacks {study} declarations: {sorted(missing)}"
            )
        designs.append(values)
    if any(design != designs[0] for design in designs[1:]):
        raise ValueError("task logs declare different study designs")
    return designs[0]


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
    parser.add_argument("--study", default="reverse-task-representative")
    parser.add_argument("--run-identity")
    parser.add_argument("--max-abs-error", type=float)
    parser.add_argument("--max-relative-error", type=float)
    arguments = parser.parse_args()
    validate_run_identity(arguments.logs, arguments.run_identity)
    design = study_design(arguments.logs, arguments.study)
    rows = records(arguments.logs)
    for field, selected in (
        ("backend", arguments.backend),
        ("precision", arguments.precision),
        ("benchmark_mode", arguments.benchmark_mode),
        ("study", arguments.study),
    ):
        if selected is not None:
            rows = [row for row in rows if row[field] == selected]
    if not rows:
        raise ValueError("no task records satisfy the requested filters")

    maximum_absolute = (
        arguments.max_abs_error
        if arguments.max_abs_error is not None
        else (0.1 if arguments.precision == "FP32" else 1e-8)
    )
    maximum_relative = (
        arguments.max_relative_error
        if arguments.max_relative_error is not None
        else (1e-3 if arguments.precision == "FP32" else 1e-10)
    )
    expected_tasks = {
        "likelihood", "joint-map", "posterior-sample", "all-marginals"
    }
    cases: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        identity = tuple(
            row[field]
            for field in (
                "backend", "precision", "benchmark_mode", "study", "dataset",
                "topology", "seed_base", "seed", "replicate", "leaves",
                "nodes", "sites", "unique_patterns", "site_batch",
            )
        )
        if row["task"] in cases[identity]:
            raise ValueError(f"duplicate task record for {identity}: {row['task']}")
        cases[identity].add(row["task"])
        if int(row["state_mismatches"]) != 0:
            raise ValueError(f"state mismatch at {row['source']}")
        if (
            float(row["max_abs_error"]) > maximum_absolute
            or float(row["max_relative_error"]) > maximum_relative
        ):
            raise ValueError(f"correctness threshold exceeded at {row['source']}")
    incomplete = {key: expected_tasks - tasks for key, tasks in cases.items() if tasks != expected_tasks}
    if incomplete:
        key, missing = next(iter(incomplete.items()))
        raise ValueError(f"incomplete task set for {key}: {sorted(missing)}")
    try:
        expected_replicates = int(design["task_replicates"])
        expected_leaves = int(design["task_leaves"])
        expected_sites = int(design["task_sites"])
    except ValueError as error:
        raise ValueError("invalid task-study declaration") from error
    expected_replicate_set = set(range(expected_replicates))
    grouped_replicates: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for row in rows:
        if (
            row["topology"] != design["task_topology"]
            or int(row["leaves"]) != expected_leaves
            or int(row["sites"]) != expected_sites
            or row["seed_base"] != design["task_seed_base"]
        ):
            raise ValueError(f"task row lies outside declared design at {row['source']}")
        group = (
            row["backend"], row["precision"], row["benchmark_mode"],
            row["study"], row["dataset"], row["topology"],
        )
        grouped_replicates[group].add(int(row["replicate"]))
    for group, observed in grouped_replicates.items():
        if observed != expected_replicate_set:
            raise ValueError(
                f"task study has replicate set {sorted(observed)} for {group}; "
                f"expected {sorted(expected_replicate_set)}"
            )

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
