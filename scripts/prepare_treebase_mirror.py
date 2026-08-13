#!/usr/bin/env python3
"""Prepare a deterministic cohort of public TreeBASE tree/alignment pairs.

The source is the immutable Git revision of ``angtft/TreeBASEMirror`` used by
the LvD study.  Source directories are ordered by a fixed SHA-256 rank and the
first eligible DNA data sets are selected.  Eligibility and ordering never
depend on benchmark measurements.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import tarfile
from pathlib import Path

from corpus_common import (
    GitRepository,
    compress_patterns,
    dataset_id,
    parse_newick,
    read_fasta_bytes,
    sha256,
    sha256_bytes,
    stable_rank,
    write_fasta,
    write_metadata,
)

SOURCE_URL = "https://github.com/angtft/TreeBASEMirror.git"
PINNED_REVISION = "c5bad4a1c3103244bc0d3a21db7a6b9329a9dc13"
RANK_NAMESPACE = "parallel-phylogenetics-treebase-mirror-v1"


def alignment_from_archive(contents: bytes) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and Path(member.name).suffix.lower() in {".fa", ".fas", ".fasta"}
        ]
        if len(members) != 1:
            raise ValueError("alignment archive must contain exactly one FASTA file")
        if members[0].size > 2**34:
            raise ValueError("alignment archive member exceeds the 16-GiB safety limit")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError("failed to read the FASTA archive member")
        return stream.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--maximum-datasets", type=int, default=400)
    parser.add_argument("--minimum-taxa", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.maximum_datasets < 1 or arguments.minimum_taxa < 2:
        parser.error("maximum-datasets must be positive and minimum-taxa at least two")

    source = GitRepository(arguments.repository, arguments.revision)
    if source.revision != PINNED_REVISION and arguments.revision == PINNED_REVISION:
        raise RuntimeError("the pinned TreeBASE Mirror revision resolved unexpectedly")
    arguments.output.mkdir(parents=True, exist_ok=True)
    directories = source.directories("trees")
    ranked = sorted(
        ((stable_rank(RANK_NAMESPACE, directory), directory) for directory in directories)
    )
    rows: list[dict[str, object]] = []
    exclusions: list[tuple[str, str, str]] = []
    identifiers: set[str] = set()
    inspected = 0
    for rank, directory in ranked:
        if len(rows) == arguments.maximum_datasets:
            break
        inspected += 1
        relative = f"trees/{directory}"
        archive_name = f"{directory}.tar.gz"
        archive_path = f"{relative}/{archive_name}"
        tree_path = f"{relative}/tree_best.newick"
        log_path = f"{relative}/log_0.txt"
        model_path = f"{relative}/model_0.txt"
        try:
            archive = source.blob(archive_path)
            tree_bytes = source.blob(tree_path)
            log_bytes = source.blob(log_path)
            try:
                model_bytes = source.blob(model_path)
            except subprocess.CalledProcessError:
                model_bytes = b""
            alignment_bytes = alignment_from_archive(archive)
            names, sequences = read_fasta_bytes(alignment_bytes)
            if len(names) < arguments.minimum_taxa:
                raise ValueError(
                    f"taxa={len(names)} is below minimum-taxa={arguments.minimum_taxa}"
                )
            tree_text = tree_bytes.decode("utf-8").strip().rstrip(";") + ";\n"
            leaves = parse_newick(tree_text).leaf_labels
            if len(leaves) != len(set(leaves)):
                raise ValueError("tree has duplicate leaf labels")
            if set(leaves) != set(names):
                missing = sorted(set(names) - set(leaves))
                extra = sorted(set(leaves) - set(names))
                raise ValueError(
                    f"tree/alignment taxa differ (missing={missing[:3]}, extra={extra[:3]})"
                )
            identifier = dataset_id(directory, "treebase-mirror-")
            if identifier in identifiers:
                raise RuntimeError(f"internal dataset-identifier collision: {identifier}")
            identifiers.add(identifier)
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
                    "raw_sites": len(sequences[0]),
                    "unique_patterns": len(weights),
                    "alignment": str(output_alignment.relative_to(arguments.output)),
                    "pattern_weights": str(output_weights.relative_to(arguments.output)),
                    "tree": str(output_tree.relative_to(arguments.output)),
                    "selection_rank": rank,
                    "source_relative_directory": relative,
                    "source_revision": source.revision,
                    "source_alignment_blob": source.blob_oid(archive_path),
                    "source_alignment_sha256": sha256_bytes(archive),
                    "source_tree_blob": source.blob_oid(tree_path),
                    "source_tree_sha256": sha256_bytes(tree_bytes),
                    "source_log_sha256": sha256_bytes(log_bytes),
                    "source_model_sha256": sha256_bytes(model_bytes),
                    "normalized_alignment_sha256": sha256(output_alignment),
                    "pattern_weights_sha256": sha256(output_weights),
                    "normalized_tree_sha256": sha256(output_tree),
                    "selection_rule": (
                        f"lowest SHA-256 ranks among DNA pairs with taxa >= "
                        f"{arguments.minimum_taxa}; no timing filter"
                    ),
                }
            )
        except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError) as error:
            exclusions.append((rank, relative, str(error)))

    fields = list(rows[0]) if rows else [
        "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
        "pattern_weights", "tree", "selection_rank",
        "source_relative_directory", "source_revision",
        "source_alignment_blob", "source_alignment_sha256", "source_tree_blob",
        "source_tree_sha256", "source_log_sha256", "source_model_sha256",
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
            ("corpus", "treebase-mirror"),
            ("corpus_kind", "empirical-tree-alignment-pairs"),
            ("source_url", SOURCE_URL),
            ("source_revision", source.revision),
            ("source_license", "not-stated-by-upstream-mirror"),
            ("redistribution", "none; outputs are prepared locally from upstream"),
            ("pattern_compression", "exact-duplicate-columns"),
            ("rank_namespace", RANK_NAMESPACE),
            ("selection_rule", f"lowest hash ranks, DNA, taxa>={arguments.minimum_taxa}, no timing filter"),
            ("source_directories", len(directories)),
            ("inspected_directories", inspected),
            ("selected_datasets", len(rows)),
            ("excluded_inspected_datasets", len(exclusions)),
        ],
    )
    print(
        f"selected {len(rows)} TreeBASE pairs after inspecting {inspected} of "
        f"{len(directories)} source directories"
    )


if __name__ == "__main__":
    main()
