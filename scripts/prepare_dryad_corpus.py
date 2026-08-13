#!/usr/bin/env python3
"""Build the prespecified Dryad TreeBASE DNA likelihood corpus.

Every discovered ``alignment.phy`` is considered.  For each data set, the
tree is the deterministic maximum-log-likelihood row among rows whose
``version`` is ``standard`` in ``pars_summary.parquet``.  Eligibility never
depends on a benchmark timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from pathlib import Path

DNA = frozenset("ACGTURYSWKMBDHVN?-.")
DOI = "10.5061/dryad.8gtht76zz"
DRYAD_FILE_ID = "4142269"
ARCHIVE_SHA256 = "06cee5bd75748acf5ba95a10b404b2867dd0b52a3e9e1b9ec357f9d9c7e09f4c"
SELECTION_RULE = "all DNA alignment.phy entries; maximum-logLikelihood version=standard tree"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


class NewickLeaves:
    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def skip(self) -> None:
        while self.position < len(self.text):
            if self.text[self.position].isspace():
                self.position += 1
            elif self.text[self.position] == "[":
                depth = 1
                self.position += 1
                while self.position < len(self.text) and depth:
                    depth += (self.text[self.position] == "[") - (
                        self.text[self.position] == "]"
                    )
                    self.position += 1
                if depth:
                    raise ValueError("unterminated Newick comment")
            else:
                break

    def label(self) -> str:
        self.skip()
        if self.position < len(self.text) and self.text[self.position] == "'":
            self.position += 1
            result = ""
            while self.position < len(self.text):
                character = self.text[self.position]
                self.position += 1
                if character != "'":
                    result += character
                elif self.position < len(self.text) and self.text[self.position] == "'":
                    result += "'"
                    self.position += 1
                else:
                    return result
            raise ValueError("unterminated quoted Newick label")
        begin = self.position
        while self.position < len(self.text) and self.text[self.position] not in "(),:;[] \t\r\n":
            self.position += 1
        return self.text[begin : self.position]

    def suffix(self) -> None:
        self.label()
        self.skip()
        if self.position < len(self.text) and self.text[self.position] == ":":
            self.position += 1
            while self.position < len(self.text) and self.text[self.position] not in ",);[ \t\r\n":
                self.position += 1
        self.skip()

    def subtree(self) -> list[str]:
        self.skip()
        if self.position < len(self.text) and self.text[self.position] == "(":
            self.position += 1
            result = self.subtree()
            while True:
                self.skip()
                if self.position < len(self.text) and self.text[self.position] == ",":
                    self.position += 1
                    result.extend(self.subtree())
                else:
                    break
            if self.position >= len(self.text) or self.text[self.position] != ")":
                raise ValueError("malformed Newick child list")
            self.position += 1
            self.suffix()
            return result
        label = self.label()
        if not label:
            raise ValueError("unlabelled Newick leaf")
        self.skip()
        if self.position < len(self.text) and self.text[self.position] == ":":
            self.position += 1
            while self.position < len(self.text) and self.text[self.position] not in ",);[ \t\r\n":
                self.position += 1
        self.skip()
        return [label]

    def parse(self) -> list[str]:
        result = self.subtree()
        self.skip()
        if self.position >= len(self.text) or self.text[self.position] != ";":
            raise ValueError("Newick tree has no terminator")
        self.position += 1
        self.skip()
        if self.position != len(self.text):
            raise ValueError("text follows Newick terminator")
        return result


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


def compress_patterns(sequences: list[str]) -> tuple[list[str], list[int]]:
    by_pattern: dict[str, int] = {}
    unique: list[str] = []
    weights: list[int] = []
    for pattern in zip(*sequences):
        encoded = "".join(pattern)
        index = by_pattern.get(encoded)
        if index is None:
            by_pattern[encoded] = len(unique)
            unique.append(encoded)
            weights.append(1)
        else:
            weights[index] += 1
    compressed = ["".join(pattern[taxon] for pattern in unique) for taxon in range(len(sequences))]
    return compressed, weights


def write_fasta(path: Path, names: list[str], sequences: list[str]) -> None:
    with path.open("w", encoding="ascii") as stream:
        for name, sequence in zip(names, sequences):
            stream.write(f">{name}\n")
            for start in range(0, len(sequence), 80):
                stream.write(sequence[start : start + 80] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    exclusions: list[tuple[str, str]] = []
    alignments = sorted(arguments.raw.rglob("alignment.phy"))
    for alignment in alignments:
        relative = alignment.parent.relative_to(arguments.raw)
        dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", alignment.parent.name)
        parquet = alignment.parent / "pars_summary.parquet"
        try:
            if not parquet.is_file():
                raise ValueError("missing pars_summary.parquet")
            names, sequences, raw_sites = read_interleaved_phylip(alignment)
            tree, likelihood = select_tree(parquet)
            leaves = NewickLeaves(tree).parse()
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
                    "tree_sha256": sha256(output_tree),
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
        "pattern_weights_sha256", "tree_sha256", "selected_log_likelihood",
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
    print(f"selected {len(rows)} of {len(alignments)} prespecified DNA alignments")


if __name__ == "__main__":
    main()
