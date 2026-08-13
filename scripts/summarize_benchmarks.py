#!/usr/bin/env python3
"""Summarize native and BEAGLE benchmark CSV records embedded in logs."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


IDENTITY = (
    "precision",
    "dataset",
    "topology",
    "leaves",
    "nodes",
    "sites",
    "site_batch",
)
PROBLEM = IDENTITY[:-1]


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
                    header = fields
                    continue
                if header is None or fields[0] not in {
                    "cuda",
                    "metal",
                    "beagle",
                }:
                    continue
                if len(fields) != len(header):
                    raise ValueError(
                        f"{path}:{line_number}: expected {len(header)} CSV "
                        f"fields, found {len(fields)}"
                    )
                row = dict(zip(header, fields))
                missing = [field for field in IDENTITY if field not in row]
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: benchmark CSV record is "
                        f"missing required columns {', '.join(missing)}"
                    )
                row["source"] = str(path)
                result.append(row)
    if not result:
        raise ValueError("no benchmark CSV records found")
    return result


def method(row: dict[str, str]) -> str:
    if row.get("backend") in {"cuda", "metal"}:
        return row["backend"]
    if row.get("baseline") == "beagle":
        resource = row.get("beagle_resource")
        if resource not in {"cpu", "cuda"}:
            raise ValueError(
                "BEAGLE records must include beagle_resource=cpu or cuda"
            )
        return f"beagle_{resource}"
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
    native = method(first) in {"cuda", "metal"}
    total_field = "total_accelerator_ms" if native else "beagle_total_ms"
    result: dict[str, object] = {field: first[field] for field in IDENTITY}
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
        groups[(method(row), *(row[field] for field in IDENTITY))].append(row)
    return [median_rows(group) for group in groups.values()]


def best_batches(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), *(str(row[field]) for field in PROBLEM))
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
        *PROBLEM,
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
        rows, key=lambda item: tuple(str(item[field]) for field in fields[:7])
    ):
        writer.writerow(row)


def write_corpus(rows: list[dict[str, object]], accelerator: str) -> None:
    by_problem: dict[
        tuple[str, ...], dict[str, dict[str, object]]
    ] = defaultdict(dict)
    for row in rows:
        key = tuple(str(row[field]) for field in PROBLEM)
        by_problem[key][str(row["method"])] = row
    pairs = [
        (methods[accelerator], methods["beagle_cpu"])
        for methods in by_problem.values()
        if accelerator in methods and "beagle_cpu" in methods
    ]
    if not pairs:
        raise ValueError(
            f"no matched {accelerator} and beagle_cpu benchmark records found"
        )
    speedups = [
        float(beagle["total_ms"]) / float(native["total_ms"])
        for native, beagle in pairs
    ]
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "accelerator",
            "problems",
            "wins",
            "min_speedup",
            "median_speedup",
            "p90_speedup",
            "max_speedup",
        ]
    )
    writer.writerow(
        [
            accelerator,
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
    parser.add_argument("--corpus", choices=("cuda", "metal"))
    arguments = parser.parse_args()
    rows = best_batches(aggregate(records(arguments.logs)))
    if arguments.corpus:
        write_corpus(rows, arguments.corpus)
    else:
        write_best(rows)


if __name__ == "__main__":
    main()
