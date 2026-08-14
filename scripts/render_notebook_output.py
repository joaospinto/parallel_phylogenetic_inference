#!/usr/bin/env python3
"""Render a compact live view while the complete notebook log is archived."""

from __future__ import annotations

import argparse
import re
import sys


MILESTONE_PREFIXES = (
    "# benchmark_suite_",
    "# benchmark_precision_complete",
    "# benchmark_section_complete",
    "# portability_compile_",
    "# validation_complete",
    "# validation_incomplete",
    "# validation_sanitizer_skipped",
    "# correctness_gate_",
    "# baseline_unavailable",
    "# capacity_limit",
    "# capacity_failure",
)
ERROR_PATTERN = re.compile(
    r"(?:^|[ :])(critical error|error|failed|failure|fatal)(?:[ :=-]|$)",
    re.IGNORECASE,
)
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_-]*)=([^ ]+)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=("quiet", "compact", "full"), required=True)
    return parser.parse_args()


def progress_identity(line: str) -> tuple[str, ...]:
    fields = dict(KEY_VALUE_PATTERN.findall(line))
    return tuple(
        fields.get(name, "")
        for name in (
            "method",
            "precision",
            "study",
            "benchmark_mode",
            "threads",
        )
    )


def progress_position(line: str) -> tuple[int, int] | None:
    match = re.search(r"(?:case|cases_through)=([0-9]+)/([0-9]+)", line)
    if match is None:
        return None
    current, total = (int(value) for value in match.groups())
    if total <= 0 or current < 0 or current > total:
        return None
    return current, total


def main() -> int:
    arguments = parse_arguments()
    progress_buckets: dict[tuple[str, ...], int] = {}

    for line in sys.stdin:
        render = arguments.level == "full"
        stripped = line.rstrip("\n")

        if not render and (
            stripped.startswith(MILESTONE_PREFIXES)
            or ERROR_PATTERN.search(stripped) is not None
            or "ERROR SUMMARY:" in stripped
        ):
            render = True

        if arguments.level == "compact" and not render:
            if stripped.startswith("===") or stripped.startswith("# task_progress "):
                render = True
            elif stripped.startswith("# progress "):
                position = progress_position(stripped)
                if position is not None:
                    current, total = position
                    identity = progress_identity(stripped) + (str(total),)
                    bucket = min(10, current * 10 // total)
                    previous = progress_buckets.get(identity, -1)
                    if current in (1, total) or bucket > previous:
                        progress_buckets[identity] = bucket
                        render = True

        if render:
            sys.stdout.write(line)
            sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
