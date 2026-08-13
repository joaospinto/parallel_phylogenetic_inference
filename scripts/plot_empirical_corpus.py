#!/usr/bin/env python3
"""Validate and plot matched empirical-corpus native/BEAGLE benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shlex
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import summarize_benchmarks as benchmark_summary


NATIVE_METHODS = {"cuda", "metal", "rocm"}
DEFAULT_TAXA_BIN_EDGES = "0,256,1024,4096,16384,inf"


@dataclass(frozen=True)
class DriverProtocol:
    kind: str
    method: str
    precision: str
    study: str
    benchmark_mode: str
    requested_site_batches: tuple[int, ...]
    planned_cases: int


@dataclass(frozen=True)
class CapacityLimit:
    method: str
    precision: str
    dataset: str
    study: str
    benchmark_mode: str
    threads: str
    first_infeasible_site_batch: int


@dataclass(frozen=True)
class CoverageDeclaration:
    kind: str
    requested_site_batches: tuple[int, ...]
    planned_cases: int


def key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in shlex.split(text):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value
    return values


def positive_integer(value: str, description: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"invalid {description}: {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{description} must be positive")
    return parsed


def parse_protocol(
    paths: list[Path], selected_study: str
) -> tuple[list[DriverProtocol], list[CapacityLimit]]:
    protocols: list[DriverProtocol] = []
    capacity_limits: list[CapacityLimit] = []
    for path in paths:
        pending: dict[str, str] | None = None
        pending_pandit: dict[str, str] | None = None
        pending_line = 0
        for line_number, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = raw.strip()
            if line == "# corpus=PANDIT-17.0":
                if pending is not None or pending_pandit is not None:
                    raise ValueError(
                        f"{path}:{line_number}: a new corpus block begins "
                        "before the preceding block is complete"
                    )
                pending_pandit = {"corpus": "PANDIT-17.0"}
                pending_line = line_number
                continue
            if line.startswith("# empirical_manifest="):
                if pending is not None or pending_pandit is not None:
                    raise ValueError(
                        f"{path}:{line_number}: a new empirical driver block "
                        f"begins before the block at line {pending_line} declares progress"
                    )
                pending = {"empirical_manifest": line.split("=", 1)[1]}
                pending_line = line_number
                continue
            if line.startswith("# capacity_limit "):
                values = key_values(line[len("# capacity_limit ") :])
                required = {
                    "method",
                    "precision",
                    "dataset",
                    "study",
                    "benchmark_mode",
                    "threads",
                    "first_infeasible_site_batch",
                }
                missing = required - set(values)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: capacity marker lacks "
                        f"{', '.join(sorted(missing))}"
                    )
                if values["study"] == selected_study:
                    capacity_limits.append(
                        CapacityLimit(
                            method=values["method"],
                            precision=values["precision"],
                            dataset=values["dataset"],
                            study=values["study"],
                            benchmark_mode=values["benchmark_mode"],
                            threads=values["threads"],
                            first_infeasible_site_batch=positive_integer(
                                values["first_infeasible_site_batch"],
                                "first infeasible site batch",
                            ),
                        )
                    )
                continue
            if pending_pandit is not None and line.startswith("# "):
                body = line[2:]
                if body.startswith("benchmark_start ") or body.startswith(
                    "resume_skip "
                ):
                    values = key_values(body.split(" ", 1)[1])
                    for key in ("method", "precision"):
                        if key in values:
                            previous = pending_pandit.setdefault(key, values[key])
                            if previous != values[key]:
                                raise ValueError(
                                    f"{path}:{line_number}: PANDIT block mixes {key} values"
                                )
                    continue
                if body.startswith("selected_families="):
                    pending_pandit["selected_families"] = body.split("=", 1)[1]
                    pending_pandit.setdefault("study", "standard")
                    pending_pandit.setdefault(
                        "benchmark_mode", "full-input-update"
                    )
                    pending_pandit.setdefault(
                        "method", pending_pandit.get("corpus_backend", "")
                    )
                    required = {
                        "method",
                        "precision",
                        "study",
                        "benchmark_mode",
                        "selected_families",
                    }
                    missing = {
                        key
                        for key in required
                        if not pending_pandit.get(key, "")
                    }
                    if missing:
                        raise ValueError(
                            f"{path}:{pending_line}: PANDIT block lacks "
                            f"{', '.join(sorted(missing))}"
                        )
                    if pending_pandit["study"] == selected_study:
                        protocols.append(
                            DriverProtocol(
                                kind="pandit",
                                method=pending_pandit["method"],
                                precision=pending_pandit["precision"],
                                study=pending_pandit["study"],
                                benchmark_mode=pending_pandit[
                                    "benchmark_mode"
                                ],
                                requested_site_batches=(),
                                planned_cases=positive_integer(
                                    pending_pandit["selected_families"],
                                    "selected PANDIT family count",
                                ),
                            )
                        )
                    pending_pandit = None
                    continue
                if "=" in body:
                    key, value = body.split("=", 1)
                    if key in {
                        "corpus_backend",
                        "benchmark_mode",
                        "study",
                        "precision",
                    }:
                        previous = pending_pandit.setdefault(key, value)
                        if previous != value:
                            raise ValueError(
                                f"{path}:{line_number}: conflicting PANDIT {key} declaration"
                            )
                continue
            if pending is None or not line.startswith("# "):
                continue
            body = line[2:]
            if body.startswith("progress "):
                progress = key_values(body[len("progress ") :])
                required = {
                    "method",
                    "precision",
                    "study",
                    "benchmark_mode",
                    "requested_site_batches",
                    "planned_cases",
                }
                combined = pending | progress
                missing = required - set(combined)
                if missing:
                    raise ValueError(
                        f"{path}:{pending_line}: empirical driver block lacks "
                        f"{', '.join(sorted(missing))}"
                    )
                if combined["study"] == selected_study:
                    requested = tuple(
                        positive_integer(value, "requested site batch")
                        for value in combined["requested_site_batches"].split()
                    )
                    if not requested or any(
                        left >= right for left, right in zip(requested, requested[1:])
                    ):
                        raise ValueError(
                            f"{path}:{pending_line}: requested site batches are "
                            "not strictly increasing"
                        )
                    protocols.append(
                        DriverProtocol(
                            kind="manifest",
                            method=combined["method"],
                            precision=combined["precision"],
                            study=combined["study"],
                            benchmark_mode=combined["benchmark_mode"],
                            requested_site_batches=requested,
                            planned_cases=positive_integer(
                                combined["planned_cases"], "planned case count"
                            ),
                        )
                    )
                pending = None
                continue
            if "=" in body:
                key, value = body.split("=", 1)
                if key not in {
                    "empirical_manifest",
                    "study",
                    "benchmark_mode",
                    "requested_site_batches",
                    "planned_cases",
                }:
                    continue
                if key in pending and pending[key] != value:
                    raise ValueError(
                        f"{path}:{line_number}: conflicting empirical {key} declaration"
                    )
                pending[key] = value
        if pending is not None and pending.get("study") == selected_study:
            raise ValueError(
                f"{path}:{pending_line}: selected empirical driver block is truncated"
            )
        if pending_pandit is not None:
            pending_pandit.setdefault("study", "standard")
            if pending_pandit["study"] == selected_study:
                raise ValueError(
                    f"{path}:{pending_line}: selected PANDIT block is truncated"
                )
    if not protocols:
        raise ValueError(
            "selected study has no empirical-corpus driver declaration; "
            "publication reporting requires declared selection and coverage"
        )
    return protocols, capacity_limits


def exact_method(row: dict[str, str]) -> str:
    return benchmark_summary.method(row)


def coarse_method(method: str) -> str:
    if method in NATIVE_METHODS or method == "beagle_cuda":
        return method.replace("_", "-")
    if re.fullmatch(r"beagle_cpu_[1-9][0-9]*t", method):
        return "beagle-cpu"
    raise ValueError(f"unsupported empirical method {method!r}")


def capacity_exact_method(limit: CapacityLimit) -> str:
    if limit.method in NATIVE_METHODS:
        if limit.threads != "none":
            raise ValueError("native capacity markers must declare threads=none")
        return limit.method
    if limit.method == "beagle-cuda":
        if limit.threads != "1":
            raise ValueError("BEAGLE CUDA capacity markers must declare one thread")
        return "beagle_cuda"
    if limit.method == "beagle-cpu":
        threads = positive_integer(limit.threads, "BEAGLE CPU thread count")
        return f"beagle_cpu_{threads}t"
    raise ValueError(f"unknown method in capacity marker: {limit.method!r}")


def expected_batches(requested: tuple[int, ...], patterns: int) -> tuple[int, ...]:
    batches: list[int] = []
    for candidate in requested:
        capped = min(candidate, patterns)
        if not batches or batches[-1] != capped:
            batches.append(capped)
        if capped == patterns:
            break
    if not batches or batches[-1] != patterns:
        batches.append(patterns)
    return tuple(batches)


def problem_key(row: dict[str, object]) -> tuple[str, ...]:
    return benchmark_summary.row_identity(
        {field: str(value) for field, value in row.items()}, False
    )


def validate_declared_batch_coverage(
    aggregated: list[dict[str, object]],
    selected_methods: tuple[str, str],
    protocols: list[DriverProtocol],
    capacity_limits: list[CapacityLimit],
    precision: str,
    benchmark_mode: str,
    study: str,
) -> tuple[
    dict[tuple[str, tuple[str, ...]], tuple[int, ...]], CoverageDeclaration
]:
    protocol_values = {
        (protocol.kind, protocol.requested_site_batches, protocol.planned_cases)
        for protocol in protocols
        if protocol.method in {coarse_method(method) for method in selected_methods}
        and protocol.precision == precision
        and protocol.benchmark_mode == benchmark_mode
        and protocol.study == study
    }
    methods_with_protocol = {
        protocol.method
        for protocol in protocols
        if protocol.precision == precision
        and protocol.benchmark_mode == benchmark_mode
        and protocol.study == study
    }
    missing_protocols = {
        coarse_method(method) for method in selected_methods
    } - methods_with_protocol
    if missing_protocols:
        raise ValueError(
            "missing empirical driver protocol for "
            + ", ".join(sorted(missing_protocols))
        )
    if len(protocol_values) != 1:
        raise ValueError(
            "selected methods declare different requested batches or planned coverage"
        )
    protocol_kind, requested, planned_cases = next(iter(protocol_values))

    by_method_problem: dict[
        tuple[str, tuple[str, ...]], list[dict[str, object]]
    ] = defaultdict(list)
    for row in aggregated:
        by_method_problem[(str(row["method"]), problem_key(row))].append(row)

    keys_by_method = {
        method: {
            key
            for observed_method, key in by_method_problem
            if observed_method == method
        }
        for method in selected_methods
    }
    if keys_by_method[selected_methods[0]] != keys_by_method[selected_methods[1]]:
        missing = keys_by_method[selected_methods[0]] - keys_by_method[selected_methods[1]]
        extra = keys_by_method[selected_methods[1]] - keys_by_method[selected_methods[0]]
        raise ValueError(
            f"empirical problem coverage differs: {len(missing)} missing baseline "
            f"and {len(extra)} unmatched baseline cases"
        )
    problem_keys = keys_by_method[selected_methods[0]]
    if not problem_keys:
        raise ValueError("no complete empirical problems remain")

    nominal_cases = 0
    for key in problem_keys:
        sample = by_method_problem[(selected_methods[0], key)][0]
        nominal_cases += len(
            expected_batches(
                requested,
                positive_integer(str(sample["unique_patterns"]), "pattern count"),
            )
        )
    if nominal_cases != planned_cases:
        raise ValueError(
            f"selected rows imply {nominal_cases} declared batch cases but the "
            f"driver declared {planned_cases}; the corpus log is incomplete or mixed"
        )
    if protocol_kind == "pandit" and requested:
        raise ValueError("PANDIT protocol unexpectedly declares a batch grid")

    limits: dict[tuple[str, str], int] = {}
    for limit in capacity_limits:
        if (
            limit.precision != precision
            or limit.benchmark_mode != benchmark_mode
            or limit.study != study
        ):
            continue
        method = capacity_exact_method(limit)
        if method not in selected_methods:
            continue
        key = (method, limit.dataset)
        previous = limits.setdefault(key, limit.first_infeasible_site_batch)
        if previous != limit.first_infeasible_site_batch:
            raise ValueError(
                f"conflicting capacity limits for {method} on {limit.dataset}"
            )

    candidates: dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}
    for method in selected_methods:
        for key in problem_keys:
            group = by_method_problem[(method, key)]
            datasets = {str(row["dataset"]) for row in group}
            if len(datasets) != 1:
                raise ValueError(
                    "one empirical problem identity has multiple dataset labels"
                )
            dataset = next(iter(datasets))
            patterns = positive_integer(str(group[0]["unique_patterns"]), "pattern count")
            declared = expected_batches(requested, patterns)
            observed = tuple(
                sorted(
                    positive_integer(str(row["site_batch"]), "site batch")
                    for row in group
                )
            )
            if len(set(observed)) != len(observed):
                raise ValueError(
                    f"duplicate aggregated site batch for {method} on {dataset}"
                )
            infeasible = limits.get((method, dataset))
            if infeasible is None:
                expected_observed = declared
            else:
                if infeasible not in declared:
                    raise ValueError(
                        f"capacity marker for {method} on {dataset} names undeclared "
                        f"batch {infeasible}"
                    )
                expected_observed = tuple(batch for batch in declared if batch < infeasible)
            if observed != expected_observed:
                raise ValueError(
                    f"batch coverage for {method} on {dataset} is {observed}; "
                    f"expected {expected_observed} under the declared capacity policy"
                )
            if not observed:
                raise ValueError(f"{method} has no feasible declared batch for {dataset}")
            candidates[(method, key)] = observed
    return candidates, CoverageDeclaration(protocol_kind, requested, planned_cases)


def parse_taxa_bin_edges(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in text.split(","):
        token = token.strip().lower()
        if token in {"inf", "+inf", "infinity", "+infinity"}:
            values.append(math.inf)
            continue
        try:
            values.append(float(token))
        except ValueError as error:
            raise ValueError(f"invalid taxa-bin edge {token!r}") from error
    if len(values) < 3 or values[0] != 0.0 or not math.isinf(values[-1]):
        raise ValueError(
            "taxa-bin edges must start at 0, end at inf, and define at least two bins"
        )
    if any(not math.isfinite(value) for value in values[:-1]) or any(
        left >= right for left, right in zip(values, values[1:])
    ):
        raise ValueError(
            "taxa-bin edges must be finite and strictly increasing before inf"
        )
    if any(not value.is_integer() for value in values[:-1]):
        raise ValueError("finite taxa-bin edges must be integers")
    return tuple(values)


def taxa_bin_index(leaves: int, edges: tuple[float, ...]) -> int:
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        if lower <= leaves < upper:
            return index
    raise ValueError(f"taxon count {leaves} lies outside declared taxa bins")


def taxa_bin_label(lower: float, upper: float) -> str:
    if math.isinf(upper):
        return f"≥{int(lower):,}"
    if lower == 0:
        return f"1–{int(upper) - 1:,}"
    return f"{int(lower):,}–{int(upper) - 1:,}"


def pair_rows(
    selected: list[dict[str, object]],
    native_method: str,
    baseline_method: str,
    candidates: dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
    run_identity: str,
) -> list[dict[str, object]]:
    indexed: dict[tuple[str, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in selected:
        method = str(row["method"])
        key = problem_key(row)
        if method in indexed[key]:
            raise ValueError(
                f"duplicate selected {method} row for empirical problem {key}"
            )
        indexed[key][method] = row
    paired: list[dict[str, object]] = []
    for key, methods in indexed.items():
        missing = {native_method, baseline_method} - set(methods)
        if missing:
            raise ValueError(
                f"incomplete selected pair for {key}: missing {', '.join(sorted(missing))}"
            )
        native = methods[native_method]
        baseline = methods[baseline_method]
        leaves = positive_integer(str(native["leaves"]), "taxon count")
        nodes = positive_integer(str(native["nodes"]), "node count")
        patterns = positive_integer(str(native["unique_patterns"]), "pattern count")
        native_ms = float(native["total_ms"])
        baseline_ms = float(baseline["total_ms"])
        paired.append(
            {
                "run_identity": run_identity,
                "precision": native["precision"],
                "benchmark_mode": native["benchmark_mode"],
                "study": native["study"],
                "dataset": native["dataset"],
                "taxa": leaves,
                "nodes": nodes,
                "raw_sites": positive_integer(
                    str(native["sites"]), "raw-site count"
                ),
                "unique_patterns": patterns,
                "node_pattern_workload": nodes * patterns,
                "native": native_method,
                "baseline": baseline_method,
                "native_candidate_batches": ";".join(
                    str(value) for value in candidates[(native_method, key)]
                ),
                "baseline_candidate_batches": ";".join(
                    str(value) for value in candidates[(baseline_method, key)]
                ),
                "native_selected_batch": native["site_batch"],
                "baseline_selected_batch": baseline["site_batch"],
                "native_measurements": native["measurements"],
                "baseline_measurements": baseline["measurements"],
                "native_ms": native_ms,
                "baseline_ms": baseline_ms,
                "native_speedup_over_baseline": baseline_ms / native_ms,
                "native_max_abs_error": native["max_abs_error"],
                "baseline_max_abs_error": baseline["max_abs_error"],
                "native_max_normalized_error": native["max_relative_error"],
                "baseline_max_normalized_error": baseline["max_relative_error"],
            }
        )
    paired.sort(
        key=lambda row: (
            int(row["taxa"]),
            int(row["unique_patterns"]),
            str(row["dataset"]),
        )
    )
    return paired


def safe_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def display_method(method: str) -> str:
    names = {
        "cuda": "CUDA",
        "metal": "Metal",
        "rocm": "ROCm",
        "beagle_cuda": "BEAGLE CUDA",
    }
    if method in names:
        return names[method]
    match = re.fullmatch(r"beagle_cpu_([1-9][0-9]*)t", method)
    if match is None:
        raise ValueError(f"cannot display unknown method {method!r}")
    threads = int(match.group(1))
    suffix = "thread" if threads == 1 else "threads"
    return f"BEAGLE CPU ({threads} {suffix})"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty report {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bin_summary(
    paired: list[dict[str, object]], edges: tuple[float, ...]
) -> tuple[list[list[float]], list[dict[str, object]]]:
    groups: list[list[float]] = [[] for _ in range(len(edges) - 1)]
    for row in paired:
        groups[taxa_bin_index(int(row["taxa"]), edges)].append(
            float(row["native_speedup_over_baseline"])
        )
    summary: list[dict[str, object]] = []
    for index, values in enumerate(groups):
        summary.append(
            {
                "taxa_bin": taxa_bin_label(edges[index], edges[index + 1]),
                "lower_inclusive": int(edges[index]),
                "upper_exclusive": (
                    "inf"
                    if math.isinf(edges[index + 1])
                    else int(edges[index + 1])
                ),
                "problems": len(values),
                "native_wins": sum(value > 1.0 for value in values),
                "minimum_speedup": min(values) if values else "",
                "median_speedup": statistics.median(values) if values else "",
                "maximum_speedup": max(values) if values else "",
            }
        )
    return groups, summary


def deterministic_jitter(dataset: str) -> float:
    digest = hashlib.sha256(dataset.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") / (2**32 - 1) - 0.5) * 0.32


def create_plots(
    paired: list[dict[str, object]],
    groups: list[list[float]],
    edges: tuple[float, ...],
    minimum_violin_count: int,
    output: Path,
    configuration: str,
    native_method: str,
    baseline_method: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires matplotlib; use --validate-only to emit strict "
            "CSV reports without figures"
        ) from error

    native_label = display_method(native_method)
    baseline_label = display_method(baseline_method)
    speedup_label = f"$\\log_2$ speedup of {native_label} over {baseline_label}"

    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    speedups = [
        math.log2(float(row["native_speedup_over_baseline"])) for row in paired
    ]
    axes[0].scatter(
        [int(row["taxa"]) for row in paired], speedups, s=18, alpha=0.7
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("taxa")
    axes[1].scatter(
        [int(row["node_pattern_workload"]) for row in paired],
        speedups,
        s=18,
        alpha=0.7,
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("nodes × unique patterns")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(speedup_label)
    figure.suptitle(f"Matched empirical problems (n={len(paired)})")
    figure.tight_layout()
    figure.savefig(
        output / f"empirical_crossover_{configuration}.pdf", bbox_inches="tight"
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for index, values in enumerate(groups, 1):
        if len(values) >= minimum_violin_count:
            violin = axis.violinplot(
                [[math.log2(value) for value in values]],
                positions=[index],
                showmedians=False,
                widths=0.72,
            )
            for body in violin["bodies"]:
                body.set_alpha(0.25)
    for row in paired:
        index = taxa_bin_index(int(row["taxa"]), edges) + 1
        axis.scatter(
            index + deterministic_jitter(str(row["dataset"])),
            math.log2(float(row["native_speedup_over_baseline"])),
            s=14,
            alpha=0.65,
            color="C0",
        )
    for index, values in enumerate(groups, 1):
        if values:
            median = statistics.median(math.log2(value) for value in values)
            axis.plot(
                [index - 0.22, index + 0.22],
                [median, median],
                color="black",
                linewidth=1.5,
            )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        range(1, len(groups) + 1),
        [
            f"{taxa_bin_label(edges[index], edges[index + 1])}\n(n={len(groups[index])})"
            for index in range(len(groups))
        ],
    )
    axis.set_xlabel("taxa (prespecified bins)")
    axis.set_ylabel(speedup_label)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(
        output / f"empirical_taxa_bin_distributions_{configuration}.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--native", choices=sorted(NATIVE_METHODS), required=True)
    parser.add_argument(
        "--baseline", choices=("beagle-cpu", "beagle-cuda"), required=True
    )
    parser.add_argument("--beagle-threads", type=int)
    parser.add_argument("--precision", choices=("FP32", "FP64"), required=True)
    parser.add_argument(
        "--benchmark-mode",
        choices=("fixed-model", "factor-update", "full-input-update"),
        required=True,
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--max-normalized-error",
        type=float,
        required=True,
        help="maximum |result-reference| / max(1, |reference|)",
    )
    parser.add_argument(
        "--max-abs-error",
        type=float,
        help="optional additional absolute-error guard",
    )
    parser.add_argument(
        "--run-identity",
        help="asserted common identity for external logs without cache markers",
    )
    parser.add_argument("--taxa-bin-edges", default=DEFAULT_TAXA_BIN_EDGES)
    parser.add_argument("--minimum-violin-count", type=int, default=5)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="write validated paired/bin CSV reports without importing matplotlib",
    )
    arguments = parser.parse_args()
    thresholds = [arguments.max_normalized_error]
    if arguments.max_abs_error is not None:
        thresholds.append(arguments.max_abs_error)
    if not all(math.isfinite(value) and value >= 0 for value in thresholds):
        raise ValueError("correctness thresholds must be finite and nonnegative")
    if arguments.minimum_violin_count < 5:
        raise ValueError("--minimum-violin-count must be at least five")
    if arguments.baseline == "beagle-cpu":
        if arguments.beagle_threads is None or arguments.beagle_threads <= 0:
            raise ValueError("BEAGLE CPU reporting requires positive --beagle-threads")
        baseline_method = f"beagle_cpu_{arguments.beagle_threads}t"
    else:
        if arguments.beagle_threads not in {None, 1}:
            raise ValueError("BEAGLE CUDA supports only --beagle-threads 1")
        baseline_method = "beagle_cuda"
    if arguments.native == "metal" and arguments.precision != "FP32":
        raise ValueError("Metal empirical reports support only FP32")

    run_identity = benchmark_summary.validate_run_identity(
        arguments.logs, arguments.run_identity
    )
    protocols, capacity_limits = parse_protocol(arguments.logs, arguments.study)
    raw = benchmark_summary.records(arguments.logs)
    selected_methods = (arguments.native, baseline_method)
    selected_raw = [
        row
        for row in raw
        if row["precision"] == arguments.precision
        and row["benchmark_mode"] == arguments.benchmark_mode
        and row["study"] == arguments.study
        and exact_method(row) in selected_methods
    ]
    if not selected_raw:
        raise ValueError("no records match the selected empirical comparison stratum")
    nonempirical = [row for row in selected_raw if row["topology"] != "empirical"]
    if nonempirical:
        raise ValueError("selected study contains non-empirical topology records")
    resident_projections = [
        row
        for row in selected_raw
        if row["measurement_scope"] != "complete-alignment-wall-time"
    ]
    if resident_projections:
        raise ValueError(
            "publication empirical reports reject chunked resident-mode projections"
        )
    correctness_guards = [("max_relative_error", arguments.max_normalized_error)]
    if arguments.max_abs_error is not None:
        correctness_guards.append(("max_abs_error", arguments.max_abs_error))
    for field, threshold in correctness_guards:
        rejected = [
            row
            for row in selected_raw
            if benchmark_summary.number(row, field) > threshold
        ]
        if rejected:
            raise ValueError(
                f"{len(rejected)} selected benchmark records exceed {field}={threshold}"
            )

    aggregated = benchmark_summary.aggregate(selected_raw)
    candidates, coverage = validate_declared_batch_coverage(
        aggregated,
        selected_methods,
        protocols,
        capacity_limits,
        arguments.precision,
        arguments.benchmark_mode,
        arguments.study,
    )
    best = benchmark_summary.best_batches(aggregated)
    paired = pair_rows(best, arguments.native, baseline_method, candidates, run_identity)
    edges = parse_taxa_bin_edges(arguments.taxa_bin_edges)
    groups, summaries = bin_summary(paired, edges)

    output = arguments.output_directory
    output.mkdir(parents=True, exist_ok=True)
    configuration = safe_component(
        f"{arguments.study}_{arguments.precision}_{arguments.native}_vs_"
        f"{baseline_method}_{arguments.benchmark_mode}"
    )
    write_csv(output / f"empirical_paired_cases_{configuration}.csv", paired)
    write_csv(output / f"empirical_taxa_bins_{configuration}.csv", summaries)
    metadata = [
        f"run_identity={run_identity}",
        f"study={arguments.study}",
        f"precision={arguments.precision}",
        f"benchmark_mode={arguments.benchmark_mode}",
        f"native={arguments.native}",
        f"baseline={baseline_method}",
        f"max_normalized_error={arguments.max_normalized_error}",
        "max_abs_error="
        + (
            str(arguments.max_abs_error)
            if arguments.max_abs_error is not None
            else "not-enforced"
        ),
        f"taxa_bin_edges={arguments.taxa_bin_edges}",
        f"minimum_violin_count={arguments.minimum_violin_count}",
        f"coverage_declaration={coverage.kind}",
        "requested_site_batches="
        + (
            " ".join(str(value) for value in coverage.requested_site_batches)
            if coverage.requested_site_batches
            else "full-pattern-only"
        ),
        f"declared_cases_per_method={coverage.planned_cases}",
        "batch_selection=minimum median complete-alignment wall time among all "
        "successfully measured declared batches; ties select the larger batch",
        f"matched_problems={len(paired)}",
    ]
    for index, path in enumerate(arguments.logs, 1):
        metadata.extend(
            (
                f"input_{index}_name={path.name}",
                f"input_{index}_sha256={file_sha256(path)}",
            )
        )
    (output / f"empirical_protocol_{configuration}.txt").write_text(
        "\n".join(metadata) + "\n", encoding="utf-8"
    )
    if not arguments.validate_only:
        create_plots(
            paired,
            groups,
            edges,
            arguments.minimum_violin_count,
            output,
            configuration,
            arguments.native,
            baseline_method,
        )


if __name__ == "__main__":
    main()
