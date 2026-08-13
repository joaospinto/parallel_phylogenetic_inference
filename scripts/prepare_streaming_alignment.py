#!/usr/bin/env python3
"""Filter a large zipped FASTA without materializing the alignment.

The first pass records, for every coordinate, the number of unambiguous DNA
bases and the set of observed bases.  The second pass writes only coordinates
meeting the prespecified coverage and variability rule.  A completed first
pass is checkpointed and reused only when the input SHA-256 and member name
match exactly.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterator

BASE_BITS = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8}
CHECKPOINT_VERSION = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fasta_records(archive: Path, member: str) -> Iterator[tuple[str, str]]:
    with zipfile.ZipFile(archive) as zipped, zipped.open(member) as binary:
        name: str | None = None
        sequence: list[str] = []
        for raw in binary:
            line = raw.decode("ascii").strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence)
                fields = line[1:].split()
                if not fields:
                    raise ValueError("empty FASTA header")
                name = fields[0]
                sequence = []
            elif line:
                if name is None:
                    raise ValueError("FASTA sequence precedes its header")
                sequence.append("".join(line.split()).upper())
        if name is not None:
            yield name, "".join(sequence)


def quoted_label(text: str, position: int) -> tuple[str, int]:
    if text[position] != "'":
        begin = position
        while position < len(text) and text[position] not in "(),:;[] \t\r\n":
            position += 1
        return text[begin:position], position
    position += 1
    result = ""
    while position < len(text):
        character = text[position]
        position += 1
        if character != "'":
            result += character
        elif position < len(text) and text[position] == "'":
            result += "'"
            position += 1
        else:
            return result, position
    raise ValueError("unterminated quoted Newick label")


def first_newick_tree(text: str) -> int:
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
        elif text[position] == "[":
            depth = 1
            position += 1
            while position < len(text) and depth:
                depth += (text[position] == "[") - (text[position] == "]")
                position += 1
            if depth:
                raise ValueError("unterminated Newick comment")
        elif text[position] == "(":
            return position
        else:
            raise ValueError("unexpected text before Newick tree")
    raise ValueError("tree file contains no Newick tree")


def normalize_arb_tree(path: Path) -> tuple[str, list[str]]:
    source = path.read_text(encoding="utf-8")
    source = source[first_newick_tree(source) :]
    output: list[str] = []
    leaves: list[str] = []
    position = 0
    expect_subtree = True
    while position < len(source):
        character = source[position]
        if character == "[":
            begin = position
            depth = 1
            position += 1
            while position < len(source) and depth:
                depth += (source[position] == "[") - (source[position] == "]")
                position += 1
            if depth:
                raise ValueError("unterminated Newick comment")
            output.append(source[begin:position])
            continue
        if expect_subtree and character.isspace():
            output.append(character)
            position += 1
            continue
        if expect_subtree and character != "(":
            label, position = quoted_label(source, position)
            normalized = label.split(",", 1)[0].strip()
            if not normalized:
                raise ValueError("empty normalized leaf label")
            leaves.append(normalized)
            output.append(normalized)
            expect_subtree = False
            continue
        output.append(character)
        position += 1
        if character in "(,":
            expect_subtree = True
        elif character == ";":
            trailing = source[position:].strip()
            if trailing:
                raise ValueError("text follows Newick terminator")
            break
        elif not character.isspace():
            expect_subtree = False
    if not output or output[-1] != ";":
        raise ValueError("Newick tree has no terminator")
    return "".join(output), leaves


def first_pass(
    archive: Path, member: str
) -> tuple[list[str], object, object, str]:
    try:
        import numpy as np
    except ImportError:
        np = None
    names: list[str] = []
    coverage: object = array.array("I")
    states: object = bytearray()
    width: int | None = None
    if np is not None:
        observed_lookup = np.zeros(256, dtype=np.uint8)
        state_lookup = np.zeros(256, dtype=np.uint8)
        for base, bit in BASE_BITS.items():
            observed_lookup[ord(base)] = 1
            state_lookup[ord(base)] = bit
    fasta_hash = hashlib.sha256()
    for name, sequence in fasta_records(archive, member):
        if width is None:
            width = len(sequence)
            if np is None:
                coverage = array.array("I", [0]) * width
                states = bytearray(width)
            else:
                coverage = np.zeros(width, dtype=np.uint32)
                states = np.zeros(width, dtype=np.uint8)
        elif len(sequence) != width:
            raise ValueError(f"{name}: aligned sequence has a different length")
        names.append(name)
        fasta_hash.update(name.encode("ascii"))
        fasta_hash.update(b"\0")
        fasta_hash.update(sequence.encode("ascii"))
        fasta_hash.update(b"\n")
        if np is None:
            for index, nucleotide in enumerate(sequence):
                bit = BASE_BITS.get(nucleotide)
                if bit is not None:
                    coverage[index] += 1
                    states[index] |= bit
        else:
            encoded = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            coverage += observed_lookup[encoded]
            states |= state_lookup[encoded]
    if width is None or len(names) < 2:
        raise ValueError("alignment must contain at least two records")
    if len(set(names)) != len(names):
        raise ValueError("alignment contains duplicate record names")
    return names, coverage, states, fasta_hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("tree", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--member")
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.5)
    parser.add_argument("--analyze-only", action="store_true")
    arguments = parser.parse_args()
    if not 0 < arguments.minimum_observed_fraction <= 1:
        parser.error("coverage fraction must lie in (0, 1]")
    arguments.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(arguments.archive) as zipped:
        members = [name for name in zipped.namelist() if name.lower().endswith((".fa", ".fasta", ".fas")) and not name.startswith("__MACOSX/")]
    member = arguments.member or (members[0] if len(members) == 1 else None)
    if member is None or member not in members:
        raise ValueError("select exactly one FASTA ZIP member with --member")

    archive_digest = sha256(arguments.archive)
    checkpoint = arguments.output / "column_statistics.json"
    coverage_file = arguments.output / "column_coverage.uint32"
    states_file = arguments.output / "column_states.uint8"
    names_file = arguments.output / "taxa.txt"
    metadata = None
    if checkpoint.is_file():
        metadata = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (
        metadata is None
        or metadata.get("checkpoint_version") != CHECKPOINT_VERSION
        or metadata.get("archive_sha256") != archive_digest
        or metadata.get("member") != member
    ):
        names, coverage, states, fasta_digest = first_pass(arguments.archive, member)
        with coverage_file.open("wb") as stream:
            coverage.tofile(stream)
        states_file.write_bytes(bytes(states))
        names_file.write_text("".join(f"{name}\n" for name in names), encoding="ascii")
        metadata = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "archive_sha256": archive_digest,
            "member": member,
            "taxa": len(names),
            "coordinates": len(states),
            "source_member_canonical_sha256": fasta_digest,
        }
        checkpoint.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        names = names_file.read_text(encoding="ascii").splitlines()
        coverage = array.array("I")
        with coverage_file.open("rb") as stream:
            coverage.fromfile(stream, metadata["coordinates"])
        states = bytearray(states_file.read_bytes())

    threshold = math.ceil(arguments.minimum_observed_fraction * len(names))
    retained = [index for index, (count, mask) in enumerate(zip(coverage, states)) if count >= threshold and mask & (mask - 1)]
    coverage_quantiles = {}
    ordered = sorted(coverage)
    for percentile in (0, 25, 50, 75, 90, 95, 99, 100):
        coverage_quantiles[str(percentile)] = ordered[round((len(ordered) - 1) * percentile / 100)] / len(names)
    summary = {
        **metadata,
        "source_release_tree_endpoint": "https://biocom.uib.es/opucheck-backend/api/releases/02_26_05",
        "source_release_alignment_endpoint": "https://biocom.uib.es/opucheck-backend/api/releases/02_26_07",
        "source_tree_sha256": sha256(arguments.tree),
        "minimum_observed_fraction": arguments.minimum_observed_fraction,
        "minimum_observed_taxa": threshold,
        "variable": True,
        "retained_coordinates": len(retained),
        "coverage_fraction_quantiles": coverage_quantiles,
        "selection_rule": "coverage >= threshold among A/C/G/T/U and at least two observed bases",
        "leaf_label_transform": "exact prefix before first comma, stripped; required unique and complete",
        "nucleotide_transform": "strip alignment whitespace and map U to T; preserve IUPAC ambiguity and gaps",
    }
    (arguments.output / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if arguments.analyze_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    normalized_tree, tree_leaves = normalize_arb_tree(arguments.tree)
    if set(tree_leaves) != set(names) or len(tree_leaves) != len(names):
        raise ValueError("normalized tree leaves and alignment records do not match exactly")
    tree_output = arguments.output / "tree.nwk"
    tree_output.write_text(normalized_tree + ("\n" if not normalized_tree.endswith("\n") else ""), encoding="utf-8")
    alignment_output = arguments.output / "alignment.fasta"
    seen: list[str] = []
    with alignment_output.open("w", encoding="ascii") as output:
        for name, sequence in fasta_records(arguments.archive, member):
            seen.append(name)
            selected = "".join(sequence[index] for index in retained).replace("U", "T")
            output.write(f">{name}\n")
            for start in range(0, len(selected), 80):
                output.write(selected[start : start + 80] + "\n")
    if seen != names:
        raise ValueError("FASTA record order changed between streaming passes")
    summary.update(
        normalized_tree_sha256=sha256(tree_output),
        normalized_alignment_sha256=sha256(alignment_output),
    )
    (arguments.output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"retained {len(retained)} of {len(states)} coordinates for {len(names)} taxa")


if __name__ == "__main__":
    main()
