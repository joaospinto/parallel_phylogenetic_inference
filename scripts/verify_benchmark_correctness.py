#!/usr/bin/env python3
"""Apply the publication correctness gates to every benchmark row in a log."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import NamedTuple


MAXIMUM_NORMALIZED_ERROR = {"FP32": 2e-3, "FP64": 1e-10}


class CapacityLimit(NamedTuple):
    method: str
    precision: str
    dataset: str
    study: str
    benchmark_mode: str
    threads: str


def marker_values(line: str) -> dict[str, str]:
    return dict(
        field.split("=", 1)
        for field in line.split()[2:]
        if "=" in field
    )


def method(row: dict[str, str]) -> tuple[str, str]:
    if row.get("backend") in {"cuda", "metal", "rocm"}:
        return row["backend"], "none"
    if row.get("baseline") == "beagle":
        resource = row.get("beagle_resource", "")
        threads = row.get("threads", "1")
        return f"beagle-{resource}", threads
    raise ValueError("unrecognized benchmark method")


def verify(
    path: Path, selected_sections: set[str], selected_precisions: set[str]
) -> tuple[int, int]:
    header: list[str] | None = None
    benchmark_rows = 0
    task_rows = 0
    section_rows = {section: 0 for section in selected_sections}
    successful_capacity_protocols: set[CapacityLimit] = set()
    capacity_limits: list[tuple[int, CapacityLimit]] = []
    completed_sections: set[tuple[str, str]] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if line.startswith("# capacity_limit "):
            values = marker_values(line)
            capacity_limits.append(
                (
                    line_number,
                    CapacityLimit(
                        values.get("method", ""),
                        values.get("precision", ""),
                        values.get("dataset", "unknown"),
                        values.get("study", "standard"),
                        values.get("benchmark_mode", "full-input-update"),
                        values.get("threads", "none"),
                    ),
                )
            )
            continue
        if line.startswith("# benchmark_section_complete "):
            values = marker_values(line)
            completed_sections.add(
                (values.get("section", ""), values.get("precision", ""))
            )
            continue
        if line.startswith("# validation_complete "):
            values = marker_values(line)
            completed_sections.add(("validation", values.get("precision", "")))
            continue
        if (
            line.startswith("# jc69_case_unavailable ")
            and "jc69" in selected_sections
        ):
            raise ValueError(
                f"{path}:{line_number}: selected JC69 case is unavailable"
            )
        if not line or line.startswith("#"):
            continue
        fields = next(csv.reader([line]))
        if fields[0] in {"backend", "baseline"}:
            header = fields if "max_relative_error" in fields else None
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
                f"{path}:{line_number}: expected {len(header)} CSV fields, "
                f"found {len(fields)}"
            )
        row = dict(zip(header, fields))
        precision = row.get("precision", "")
        if precision not in MAXIMUM_NORMALIZED_ERROR:
            raise ValueError(
                f"{path}:{line_number}: unsupported benchmark precision "
                f"{precision!r}"
            )
        try:
            absolute_error = float(row["max_abs_error"])
            normalized_error = float(row["max_relative_error"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: invalid benchmark error field"
            ) from error
        if not math.isfinite(absolute_error):
            raise ValueError(
                f"{path}:{line_number}: nonfinite descriptive absolute error"
            )
        limit = MAXIMUM_NORMALIZED_ERROR[precision]
        if not math.isfinite(normalized_error) or normalized_error > limit:
            raise ValueError(
                f"{path}:{line_number}: normalized error {normalized_error} "
                f"exceeds the {precision} limit {limit}"
            )
        benchmark_rows += 1
        row_method, row_threads = method(row)
        successful_capacity_protocols.add(
            CapacityLimit(
                row_method,
                precision,
                row.get("dataset", "unknown"),
                row.get("study", "standard"),
                row.get("benchmark_mode", "full-input-update"),
                row_threads,
            )
        )
        study = row.get("study", "standard")
        dataset = row.get("dataset", "")
        if (
            "synthetic" in selected_sections
            and dataset == "synthetic"
            and study == "standard"
        ):
            section_rows["synthetic"] += 1
        if (
            "distributions" in selected_sections
            and study == "independent-taxa-pattern-grid"
        ):
            section_rows["distributions"] += 1
        if (
            "jc69" in selected_sections
            and study == "clock-like-jc69-simulation"
        ):
            section_rows["jc69"] += 1
        if "fish" in selected_sections and dataset == "actinopt_12k_raxml":
            section_rows["fish"] += 1
        if "pandit" in selected_sections and study.startswith("pandit-"):
            section_rows["pandit"] += 1
        if "empirical" in selected_sections and study.startswith(
            "empirical-manifest-"
        ):
            section_rows["empirical"] += 1
        if "state_mismatches" in row:
            try:
                state_mismatches = int(row["state_mismatches"])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid state_mismatches"
                ) from error
            if state_mismatches != 0:
                raise ValueError(
                    f"{path}:{line_number}: found {state_mismatches} "
                    "discrete-state mismatch(es)"
                )
            task_rows += 1
            if "tasks" in selected_sections:
                section_rows["tasks"] += 1
    for line_number, capacity_limit in capacity_limits:
        matching_protocol = capacity_limit in successful_capacity_protocols
        if (
            not matching_protocol
            and capacity_limit.study == "clock-like-jc69-simulation"
            and capacity_limit.dataset.startswith("synthetic-jc69-")
        ):
            matching_protocol = CapacityLimit(
                capacity_limit.method,
                capacity_limit.precision,
                "synthetic-jc69",
                capacity_limit.study,
                capacity_limit.benchmark_mode,
                capacity_limit.threads,
            ) in successful_capacity_protocols
        if not matching_protocol:
            raise ValueError(
                f"{path}:{line_number}: capacity boundary has no successful "
                "smaller-batch row for the same method/problem/protocol"
            )
    for section, count in section_rows.items():
        if section != "validation" and count == 0:
            raise ValueError(
                f"selected benchmark section {section!r} produced no "
                "accepted timing rows"
            )
    for section in selected_sections:
        for precision in selected_precisions:
            if (section, precision) not in completed_sections:
                raise ValueError(
                    f"selected section {section!r} lacks a completion marker "
                    f"for {precision}"
                )
    return benchmark_rows, task_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--selected-sections",
        default="",
        help="space-separated effective sections that this invocation selected",
    )
    parser.add_argument(
        "--selected-precisions",
        default="",
        help="space-separated precisions selected by this invocation",
    )
    arguments = parser.parse_args()
    selected_sections = set(arguments.selected_sections.split())
    selected_precisions = set(arguments.selected_precisions.split())
    if selected_sections and not selected_precisions:
        raise ValueError("selected sections require selected precisions")
    benchmark_rows, task_rows = verify(
        arguments.report, selected_sections, selected_precisions
    )
    print(
        "# correctness_gate_complete "
        f"benchmark_rows={benchmark_rows} task_rows={task_rows} "
        "fp32_max_normalized_error=0.002 "
        "fp64_max_normalized_error=1e-10 state_mismatches=0"
    )


if __name__ == "__main__":
    main()
