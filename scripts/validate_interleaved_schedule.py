#!/usr/bin/env python3
"""Strictly validate machine-readable cyclic benchmark schedules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[2:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in result:
            raise ValueError(f"duplicate schedule field {key}")
        result[key] = value
    return result


@dataclass
class ScheduleBlock:
    source: str
    study: str
    precision: str
    benchmark_mode: str
    case_count: int
    methods: list[str]
    cases: list[tuple[str, dict[str, str]]] = field(default_factory=list)


def validate_interleaved_schedule(
    paths: list[Path],
    *,
    study: str,
    precision: str,
    benchmark_mode: str,
    expected_cases: int | None = None,
    required_methods: set[str] | None = None,
) -> list[ScheduleBlock]:
    blocks: list[ScheduleBlock] = []
    active: ScheduleBlock | None = None
    for path in paths:
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if line.startswith("# interleaved_schedule "):
                declaration = fields(line)
                if (
                    declaration.get("study") != study
                    or declaration.get("precision") != precision
                    or declaration.get("benchmark_mode") != benchmark_mode
                ):
                    active = None
                    continue
                try:
                    case_count = int(declaration["case_count"])
                    method_count = int(declaration["method_count"])
                    methods = declaration["methods"].split(",")
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid interleaved schedule declaration"
                    ) from error
                if (
                    case_count <= 0
                    or method_count <= 0
                    or len(methods) != method_count
                    or len(set(methods)) != method_count
                    or declaration.get("policy") != "cyclic-rotation-by-case"
                ):
                    raise ValueError(
                        f"{path}:{line_number}: invalid cyclic schedule declaration"
                    )
                active = ScheduleBlock(
                    source=f"{path}:{line_number}",
                    study=study,
                    precision=precision,
                    benchmark_mode=benchmark_mode,
                    case_count=case_count,
                    methods=methods,
                )
                blocks.append(active)
                continue
            if not line.startswith("# interleaved_case "):
                continue
            record = fields(line)
            if (
                record.get("study") != study
                or record.get("precision") != precision
                or record.get("benchmark_mode") != benchmark_mode
            ):
                continue
            if active is None:
                raise ValueError(
                    f"{path}:{line_number}: interleaved case has no matching declaration"
                )
            active.cases.append((f"{path}:{line_number}", record))

    if not blocks:
        raise ValueError(
            f"no interleaved {study} schedule for {precision} {benchmark_mode}"
        )
    complete_blocks: list[ScheduleBlock] = []
    for block in blocks:
        if expected_cases is not None and block.case_count != expected_cases:
            raise ValueError(
                f"{block.source}: declares {block.case_count} cases; "
                f"expected {expected_cases}"
            )
        if required_methods and not required_methods.issubset(block.methods):
            missing = required_methods - set(block.methods)
            raise ValueError(
                f"{block.source}: schedule lacks required methods "
                f"{', '.join(sorted(missing))}"
            )
        expected_records = block.case_count * len(block.methods)
        if len(block.cases) > expected_records:
            raise ValueError(
                f"{block.source}: schedule has {len(block.cases)} case-method "
                f"records; expected at most {expected_records}"
            )
        descriptors: set[tuple[tuple[str, str], ...]] = set()
        ignored = {
            "study", "precision", "benchmark_mode", "case_index",
            "order_index", "specification",
        }
        observed_descriptors: dict[int, set[tuple[tuple[str, str], ...]]] = {}
        for offset, (source, record) in enumerate(block.cases):
            case_index, order_index = divmod(offset, len(block.methods))
            observed_descriptors.setdefault(case_index, set())
            try:
                observed_case = int(record["case_index"])
                observed_order = int(record["order_index"])
                specification = record["specification"]
            except (KeyError, ValueError) as error:
                raise ValueError(f"{source}: malformed interleaved case") from error
            expected_method = block.methods[
                (case_index + order_index) % len(block.methods)
            ]
            if (
                observed_case != case_index
                or observed_order != order_index
                or specification != expected_method
            ):
                raise ValueError(
                    f"{source}: observed ({observed_case}, {observed_order}, "
                    f"{specification}); expected ({case_index}, {order_index}, "
                    f"{expected_method})"
                )
            descriptor = tuple(
                sorted((key, value) for key, value in record.items() if key not in ignored)
            )
            if not descriptor:
                raise ValueError(f"{source}: schedule case lacks an identity")
            observed_descriptors[case_index].add(descriptor)
            if len(observed_descriptors[case_index]) != 1:
                raise ValueError(
                    f"{block.source}: methods disagree on identity for "
                    f"case {case_index}"
                )
            if order_index == len(block.methods) - 1:
                descriptor = next(iter(observed_descriptors[case_index]))
                if descriptor in descriptors:
                    raise ValueError(
                        f"{block.source}: duplicate case identity at index {case_index}"
                    )
                descriptors.add(descriptor)
        if len(block.cases) == expected_records:
            complete_blocks.append(block)
    if not complete_blocks:
        raise ValueError(
            "no complete interleaved schedule block; interrupted prefix blocks "
            "are valid only when followed by a complete resumed traversal"
        )
    return complete_blocks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--study", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--benchmark-mode", required=True)
    parser.add_argument("--expected-cases", type=int)
    parser.add_argument("--require-method", action="append", default=[])
    arguments = parser.parse_args()
    blocks = validate_interleaved_schedule(
        arguments.logs,
        study=arguments.study,
        precision=arguments.precision,
        benchmark_mode=arguments.benchmark_mode,
        expected_cases=arguments.expected_cases,
        required_methods=set(arguments.require_method),
    )
    print(
        f"validated_interleaved_schedules={len(blocks)} "
        f"study={arguments.study} precision={arguments.precision} "
        f"benchmark_mode={arguments.benchmark_mode}"
    )


if __name__ == "__main__":
    main()
