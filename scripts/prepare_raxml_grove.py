#!/usr/bin/env python3
"""Prepare a stratified RAxML-Grove empirical-topology benchmark.

RAxML Grove intentionally withholds the source alignments.  This preparer
therefore uses its empirical trees and dimensions only, and generates a
deterministic JC69 alignment on each selected tree.  The resulting corpus is
always identified as simulated sequence data on empirical topologies.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from corpus_common import (
    GitRepository,
    ParsedNewick,
    compress_patterns,
    dataset_id,
    parse_newick,
    sha256,
    sha256_bytes,
    stable_rank,
    write_fasta,
    write_metadata,
)

SOURCE_URL = "https://github.com/angtft/RAxMLGrove.git"
PINNED_REVISION = "b81faa13a93703fcdfbd6e6fa2ed1bb5b42b76f5"
RANK_NAMESPACE = "parallel-phylogenetics-raxml-grove-v1"
NUCLEOTIDE_MODELS = (
    "JC", "F81", "K80", "K81", "HKY", "TN", "TIM", "TVM", "TPM", "GTR",
)


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & ((1 << 64) - 1)

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
        return value ^ (value >> 31)

    def uniform(self) -> float:
        return (self.next() >> 11) * (1.0 / (1 << 53))

    def below(self, bound: int) -> int:
        if bound < 1:
            raise ValueError("random bound must be positive")
        limit = ((1 << 64) // bound) * bound
        while True:
            value = self.next()
            if value < limit:
                return value % bound


def source_alignment_sites(log: str) -> int:
    patterns = (
        r"Alignment sites:\s*([0-9]+)",
        r"Alignment sites\s*/\s*patterns:\s*([0-9]+)\s*/",
    )
    for pattern in patterns:
        match = re.search(pattern, log, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError("log does not report an alignment-site count")


def is_nucleotide_log(log: str) -> bool:
    data_types = re.findall(r"DataType:\s*([^\s]+)", log, flags=re.IGNORECASE)
    if data_types:
        return all(value.upper() == "DNA" for value in data_types)
    models = re.findall(r"^Model:\s*([^+\s]+)", log, flags=re.IGNORECASE | re.MULTILINE)
    if not models:
        return False
    return all(model.upper().startswith(NUCLEOTIDE_MODELS) for model in models)


def simulated_alignment(
    tree: ParsedNewick, sites: int, seed: int
) -> tuple[list[str], list[str]]:
    children: list[list[int]] = [[] for _ in tree.parents]
    for child, parent in enumerate(tree.parents):
        if parent >= 0:
            children[parent].append(child)
    generator = SplitMix64(seed)
    root_states = bytes(generator.below(4) for _ in range(sites))
    leaves: dict[int, str] = {}
    stack: list[tuple[int, bytes]] = [(0, root_states)]
    alphabet = "ACGT"
    while stack:
        node, parent_states = stack.pop()
        if not children[node]:
            leaves[node] = "".join(alphabet[state] for state in parent_states)
            continue
        for child in reversed(children[node]):
            same_probability = 0.25 + 0.75 * math.exp(
                -4.0 * tree.lengths[child] / 3.0
            )
            child_states = bytearray(sites)
            for site, state in enumerate(parent_states):
                if generator.uniform() < same_probability:
                    child_states[site] = state
                else:
                    child_states[site] = (state + 1 + generator.below(3)) % 4
            stack.append((child, bytes(child_states)))
    leaf_nodes = tree.leaves
    return (
        [tree.labels[node] for node in leaf_nodes],
        [leaves[node] for node in leaf_nodes],
    )


def parse_edges(text: str) -> list[int]:
    edges = [int(value) for value in text.split(",")]
    if len(edges) < 2 or edges != sorted(set(edges)) or edges[0] < 2:
        raise ValueError("taxa-bin-edges must be increasing integers >= 2")
    return edges


def bin_index(taxa: int, edges: list[int]) -> int | None:
    for index in range(len(edges) - 1):
        if edges[index] <= taxa < edges[index + 1]:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--sites", type=int, default=256)
    parser.add_argument("--per-bin", type=int, default=20)
    parser.add_argument(
        "--taxa-bin-edges", default="100,256,1024,4096,16384,65537"
    )
    arguments = parser.parse_args()
    try:
        edges = parse_edges(arguments.taxa_bin_edges)
    except ValueError as error:
        parser.error(str(error))
    if arguments.sites < 1 or arguments.per_bin < 1:
        parser.error("sites and per-bin must be positive")

    source = GitRepository(arguments.repository, arguments.revision)
    arguments.output.mkdir(parents=True, exist_ok=True)
    directories = source.directories("trees")
    ranked = sorted(
        ((stable_rank(RANK_NAMESPACE, directory), directory) for directory in directories)
    )
    selected: list[list[tuple[str, str, str, ParsedNewick, int, bytes, bytes]]] = [
        [] for _ in range(len(edges) - 1)
    ]
    exclusions: list[tuple[str, str, str]] = []
    inspected = 0
    for rank, directory in ranked:
        if all(len(group) == arguments.per_bin for group in selected):
            break
        inspected += 1
        relative = f"trees/{directory}"
        tree_path = f"{relative}/tree_best.newick"
        log_path = f"{relative}/log_0.txt"
        try:
            tree_oid = source.blob_oid_optional(tree_path)
            log_oid = source.blob_oid_optional(log_path)
            missing = [
                path
                for path, oid in ((tree_path, tree_oid), (log_path, log_oid))
                if oid is None
            ]
            if missing:
                raise ValueError(
                    "missing required source entry: " + ", ".join(missing)
                )
            tree_bytes = source.blob(tree_path)
            log_bytes = source.blob(log_path)
            log = log_bytes.decode("utf-8")
            if not is_nucleotide_log(log):
                raise ValueError("source analysis is not nucleotide-valued")
            original_sites = source_alignment_sites(log)
            tree_text = tree_bytes.decode("utf-8").strip().rstrip(";") + ";\n"
            tree = parse_newick(tree_text)
            leaves = tree.leaf_labels
            if len(leaves) != len(set(leaves)):
                raise ValueError("tree has duplicate leaf labels")
            group = bin_index(len(leaves), edges)
            if group is None:
                raise ValueError(
                    f"taxa={len(leaves)} lies outside [{edges[0]}, {edges[-1]})"
                )
            if len(selected[group]) == arguments.per_bin:
                continue
            selected[group].append(
                (rank, directory, tree_text, tree, original_sites, tree_bytes, log_bytes)
            )
        except (OSError, UnicodeError, ValueError) as error:
            exclusions.append((rank, relative, str(error)))

    rows: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for group, candidates in enumerate(selected):
        for rank, directory, tree_text, tree, original_sites, tree_bytes, log_bytes in candidates:
            identifier = dataset_id(directory, "raxml-grove-")
            if identifier in identifiers:
                raise RuntimeError(f"internal dataset-identifier collision: {identifier}")
            identifiers.add(identifier)
            seed = int(stable_rank(RANK_NAMESPACE + "-jc69", directory)[:16], 16)
            names, sequences = simulated_alignment(tree, arguments.sites, seed)
            if set(names) != set(tree.leaf_labels):
                raise RuntimeError("simulated alignment lost a tree taxon")
            patterns, weights = compress_patterns(sequences)
            destination = arguments.output / identifier
            destination.mkdir(exist_ok=True)
            output_alignment = destination / "patterns.fasta"
            output_weights = destination / "pattern_weights.txt"
            output_tree = destination / "tree.nwk"
            write_fasta(output_alignment, names, patterns)
            output_weights.write_text(
                "".join(f"{weight}\n" for weight in weights), encoding="ascii"
            )
            output_tree.write_text(tree_text, encoding="utf-8")
            rows.append(
                {
                    "dataset": identifier,
                    "taxa": len(names),
                    "raw_sites": arguments.sites,
                    "unique_patterns": len(weights),
                    "alignment": str(output_alignment.relative_to(arguments.output)),
                    "pattern_weights": str(output_weights.relative_to(arguments.output)),
                    "tree": str(output_tree.relative_to(arguments.output)),
                    "taxa_bin": f"[{edges[group]},{edges[group + 1]})",
                    "selection_rank": rank,
                    "simulation_seed": seed,
                    "source_alignment_sites": original_sites,
                    "source_relative_directory": f"trees/{directory}",
                    "source_revision": source.revision,
                    "source_tree_blob": source.blob_oid(f"trees/{directory}/tree_best.newick"),
                    "source_tree_sha256": sha256_bytes(tree_bytes),
                    "source_log_sha256": sha256_bytes(log_bytes),
                    "normalized_alignment_sha256": sha256(output_alignment),
                    "pattern_weights_sha256": sha256(output_weights),
                    "normalized_tree_sha256": sha256(output_tree),
                    "selection_rule": (
                        f"first {arguments.per_bin} DNA trees by SHA-256 rank in "
                        f"taxa bin [{edges[group]},{edges[group + 1]}); no timing filter"
                    ),
                }
            )
    rows.sort(key=lambda row: (int(str(row["taxa"])), str(row["selection_rank"])))
    fields = list(rows[0]) if rows else [
        "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
        "pattern_weights", "tree", "taxa_bin", "selection_rank",
        "simulation_seed", "source_alignment_sites",
        "source_relative_directory", "source_revision", "source_tree_blob",
        "source_tree_sha256", "source_log_sha256",
        "normalized_alignment_sha256", "pattern_weights_sha256",
        "normalized_tree_sha256", "selection_rule",
    ]
    with (arguments.output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (arguments.output / "excluded.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("selection_rank", "source_relative_directory", "reason"))
        writer.writerows(exclusions)
    write_metadata(
        arguments.output / "corpus_metadata.txt",
        [
            ("corpus", "raxml-grove"),
            ("corpus_kind", "simulated-JC69-alignments-on-empirical-topologies"),
            ("source_url", SOURCE_URL),
            ("source_revision", source.revision),
            ("source_license", "ODbL-1.0-and-Database-Contents-License"),
            ("source_alignments_available", "no"),
            ("pattern_compression", "exact-duplicate-columns"),
            ("simulated_sites", arguments.sites),
            ("taxa_bin_edges", arguments.taxa_bin_edges),
            ("per_bin", arguments.per_bin),
            ("rank_namespace", RANK_NAMESPACE),
            ("selection_rule", "DNA empirical trees stratified by taxa and SHA-256 rank; no timing filter"),
            ("source_directories", len(directories)),
            ("inspected_directories", inspected),
            ("selected_datasets", len(rows)),
            ("excluded_inspected_datasets", len(exclusions)),
        ],
    )
    counts = ", ".join(
        f"[{edges[index]},{edges[index + 1]}):{len(group)}"
        for index, group in enumerate(selected)
    )
    print(f"selected {len(rows)} RAxML-Grove trees after inspecting {inspected}; {counts}")


if __name__ == "__main__":
    main()
