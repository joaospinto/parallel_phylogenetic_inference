#!/usr/bin/env python3
"""Build the prespecified Dryad empirical-long DNA likelihood corpus.

Every ``alignment.phy`` in the archive's ``empirical-long/dna_long_empirical``
cohort is considered.  For each data set, the tree is the deterministic
maximum-log-likelihood row among rows whose ``version`` is ``standard`` in
``pars_summary.parquet``.  Eligibility never depends on benchmark timing.
Similarly named records under ``unsuccessful_MSAs`` are a separate cohort.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from corpus_common import (
    DNA,
    compress_patterns,
    dataset_id,
    parse_newick,
    sha256,
    write_fasta,
    write_metadata,
)

DOI = "10.5061/dryad.8gtht76zz"
DRYAD_FILE_ID = "4142269"
ARCHIVE_SHA256 = "06cee5bd75748acf5ba95a10b404b2867dd0b52a3e9e1b9ec357f9d9c7e09f4c"
COHORT_RELATIVE = Path("stopping_criteria_data/empirical-long/dna_long_empirical")
SELECTION_RULE = (
    "all alignment.phy entries in empirical-long/dna_long_empirical; "
    "maximum-logLikelihood version=standard tree"
)


def append_fragment(sequence: str, fragment: str, sites: int) -> str:
    fragment = "".join(fragment.split()).upper()
    if not fragment or not set(fragment) <= DNA:
        raise ValueError("alignment contains a non-nucleotide sequence fragment")
    sequence += fragment
    if len(sequence) > sites:
        raise ValueError("sequence exceeds the PHYLIP header length")
    return sequence


def read_interleaved_phylip(path: Path) -> tuple[list[str], list[str], int]:
    lines = path.read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError("empty PHYLIP file")
    header = lines[0].split()
    if len(header) != 2:
        raise ValueError("PHYLIP header must contain taxon and site counts")
    taxa, sites = map(int, header)
    body = [line.strip() for line in lines[1:] if line.strip()]
    if taxa < 2 or sites < 1 or len(body) < taxa:
        raise ValueError("incomplete PHYLIP first block")
    names: list[str] = []
    sequences: list[str] = []
    for line in body[:taxa]:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError("PHYLIP first-block record has no sequence")
        names.append(fields[0])
        sequences.append(append_fragment("", "".join(fields[1:]), sites))
    if len(set(names)) != taxa:
        raise ValueError("duplicate PHYLIP taxon name")
    for index, line in enumerate(body[taxa:]):
        record = index % taxa
        fields = line.split()
        if fields and fields[0] == names[record]:
            fields = fields[1:]
        sequences[record] = append_fragment(
            sequences[record], "".join(fields), sites
        )
    if any(len(sequence) != sites for sequence in sequences):
        raise ValueError("PHYLIP sequence length does not match its header")
    return names, sequences, sites


def select_tree(parquet: Path) -> tuple[str, float]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required only for corpus preparation; run with "
            "`uv run --with pyarrow scripts/prepare_dryad_corpus.py ...`"
        ) from error
    rows = pq.read_table(
        parquet, columns=["newick", "logLikelihood", "version"]
    ).to_pylist()
    candidates = [
        (float(row["logLikelihood"]), str(row["newick"]))
        for row in rows
        if row["version"] == "standard"
        and row["newick"] is not None
        and row["logLikelihood"] is not None
        and math.isfinite(float(row["logLikelihood"]))
    ]
    if not candidates:
        raise ValueError("pars_summary has no finite version=standard tree")
    likelihood, tree = min(candidates, key=lambda item: (-item[0], item[1]))
    return tree.rstrip(";\n") + ";\n", likelihood


def cohort_alignments(raw: Path) -> list[Path]:
    cohort = raw / COHORT_RELATIVE
    if not cohort.is_dir():
        raise FileNotFoundError(
            f"pinned Dryad cohort directory is missing: {COHORT_RELATIVE}"
        )
    return sorted(cohort.rglob("alignment.phy"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    exclusions: list[tuple[str, str]] = []
    identifiers: set[str] = set()
    alignments = cohort_alignments(arguments.raw)
    for alignment in alignments:
        relative = alignment.parent.relative_to(arguments.raw)
        dataset = dataset_id(relative)
        if dataset in identifiers:
            raise RuntimeError(f"internal dataset-identifier collision: {dataset}")
        identifiers.add(dataset)
        parquet = alignment.parent / "pars_summary.parquet"
        try:
            if not parquet.is_file():
                raise ValueError("missing pars_summary.parquet")
            names, sequences, raw_sites = read_interleaved_phylip(alignment)
            tree, likelihood = select_tree(parquet)
            leaves = parse_newick(tree).leaf_labels
            if len(leaves) != len(set(leaves)):
                raise ValueError("selected tree has duplicate leaf labels")
            if set(leaves) != set(names):
                missing = sorted(set(names) - set(leaves))
                extra = sorted(set(leaves) - set(names))
                raise ValueError(
                    f"tree/alignment taxa differ (missing={missing[:3]}, extra={extra[:3]})"
                )
            patterns, weights = compress_patterns(sequences)
            destination = arguments.output / dataset
            destination.mkdir(exist_ok=True)
            output_alignment = destination / "patterns.fasta"
            output_weights = destination / "pattern_weights.txt"
            output_tree = destination / "tree.nwk"
            write_fasta(output_alignment, names, patterns)
            output_weights.write_text(
                "".join(f"{weight}\n" for weight in weights), encoding="ascii"
            )
            output_tree.write_text(tree, encoding="ascii")
            rows.append(
                {
                    "dataset": dataset,
                    "taxa": len(names),
                    "raw_sites": raw_sites,
                    "unique_patterns": len(weights),
                    "alignment": str(output_alignment.relative_to(arguments.output)),
                    "pattern_weights": str(output_weights.relative_to(arguments.output)),
                    "tree": str(output_tree.relative_to(arguments.output)),
                    "source_alignment_sha256": sha256(alignment),
                    "source_parquet_sha256": sha256(parquet),
                    "normalized_alignment_sha256": sha256(output_alignment),
                    "pattern_weights_sha256": sha256(output_weights),
                    "normalized_tree_sha256": sha256(output_tree),
                    "selected_log_likelihood": likelihood,
                    "source_relative_directory": str(relative),
                    "source_doi": DOI,
                    "source_file_id": DRYAD_FILE_ID,
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "selection_rule": SELECTION_RULE,
                }
            )
        except (OSError, UnicodeError, ValueError) as error:
            exclusions.append((str(relative), str(error)))

    fields = list(rows[0]) if rows else [
        "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
        "pattern_weights", "tree", "source_alignment_sha256",
        "source_parquet_sha256", "normalized_alignment_sha256",
        "pattern_weights_sha256", "normalized_tree_sha256", "selected_log_likelihood",
        "source_relative_directory", "source_doi", "source_file_id",
        "source_archive_sha256", "selection_rule",
    ]
    with (arguments.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (arguments.output / "excluded.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_relative_directory", "reason"))
        writer.writerows(exclusions)
    write_metadata(
        arguments.output / "corpus_metadata.txt",
        [
            ("corpus", "togkousidis-dryad-treebase"),
            ("corpus_kind", "empirical-tree-alignment-pairs"),
            ("corpus_doi", DOI),
            ("corpus_version", 6),
            ("corpus_license", "CC0-1.0"),
            ("source_file_id", DRYAD_FILE_ID),
            ("source_archive_sha256", ARCHIVE_SHA256),
            ("pattern_compression", "exact-duplicate-columns"),
            ("selection_rule", SELECTION_RULE + "; no timing filter"),
            ("selected_datasets", len(rows)),
            ("excluded_datasets", len(exclusions)),
        ],
    )
    print(f"selected {len(rows)} of {len(alignments)} prespecified DNA alignments")


if __name__ == "__main__":
    main()
