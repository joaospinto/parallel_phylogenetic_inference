#!/usr/bin/env python3
"""Extract PANDIT nucleotide alignments and trees into Newick/FASTA pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import TextIO


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="ascii")
    return path.open("r", encoding="ascii")


def write_fasta(path: Path, sequences: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="ascii") as output:
        for name, sequence in sequences:
            output.write(f">{name}\n")
            for begin in range(0, len(sequence), 80):
                output.write(sequence[begin : begin + 80] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract nucleotide trees and alignments from the official "
            "PANDIT 17.0 flat file."
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--min-leaves", type=int, default=2)
    parser.add_argument("--max-leaves", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--include-unreliable",
        action="store_true",
        help="retain columns marked unreliable by the PANDIT HMM mask",
    )
    args = parser.parse_args()
    if args.min_leaves < 2:
        parser.error("--min-leaves must be at least two")
    if args.max_leaves is not None and args.max_leaves < args.min_leaves:
        parser.error("--max-leaves must not be smaller than --min-leaves")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    requested = set(args.family)
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, int, int, int]] = []
    fields: dict[str, str] = {}
    sequences: list[tuple[str, str]] = []
    current_name: str | None = None

    def emit() -> None:
        nonlocal fields, sequences, current_name
        family = fields.get("FAM")
        tree = fields.get("DPH")
        if family is not None and tree is not None and sequences:
            leaves = len(sequences)
            selected = not requested or family in requested
            selected = selected and leaves >= args.min_leaves
            selected = selected and (
                args.max_leaves is None or leaves <= args.max_leaves
            )
            selected = selected and (args.limit == 0 or len(rows) < args.limit)
            lengths = {len(sequence) for _, sequence in sequences}
            if len(lengths) != 1:
                raise ValueError(f"{family}: nucleotide sequences have unequal lengths")
            raw_sites = next(iter(lengths))
            output_sequences = sequences
            mask = fields.get("DMK")
            if not args.include_unreliable and mask is not None:
                if len(mask) != raw_sites:
                    raise ValueError(f"{family}: DNA mask length does not match alignment")
                retained = [index for index, marker in enumerate(mask) if marker == "x"]
                output_sequences = [
                    (name, "".join(sequence[index] for index in retained))
                    for name, sequence in sequences
                ]
            selected_sites = len(output_sequences[0][1])
            if selected and selected_sites != 0:
                (args.output / f"{family}.nwk").write_text(
                    tree.rstrip(";") + ";\n", encoding="ascii"
                )
                write_fasta(args.output / f"{family}.fasta", output_sequences)
                rows.append((family, leaves, raw_sites, selected_sites))
        fields = {}
        sequences = []
        current_name = None

    with open_text(args.archive) as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.rstrip("\r\n")
            if line == "//":
                emit()
                continue
            if len(line) < 3:
                continue
            tag = line[:3]
            value = line[3:].strip()
            if tag == "NAM":
                current_name = value
            elif tag == "DSQ":
                if current_name is None:
                    raise ValueError(
                        f"line {line_number}: DSQ appears before its NAM field"
                    )
                sequences.append((current_name, value))
            elif tag in {"FAM", "DPH", "DMK"}:
                fields[tag] = value
    if fields or sequences:
        raise ValueError("the PANDIT flat file ends before a // family marker")
    missing = requested.difference(row[0] for row in rows)
    if missing:
        raise ValueError("requested PANDIT families were not extracted: " + ", ".join(sorted(missing)))

    with (args.output / "manifest.csv").open("w", newline="", encoding="ascii") as output:
        writer = csv.writer(output)
        writer.writerow(["family", "leaves", "raw_sites", "selected_sites"])
        writer.writerows(rows)
    print(f"extracted {len(rows)} PANDIT families into {args.output}")


if __name__ == "__main__":
    main()
