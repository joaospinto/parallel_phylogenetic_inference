#!/usr/bin/env python3
"""Filter a large zipped FASTA without materializing the alignment.

The first pass records, for every coordinate, the number of unambiguous DNA
bases and the set of observed bases.  The second pass retains every coordinate
meeting the prespecified coverage rule, optionally applying a deterministic
size cap whose seed and selected-index digest are recorded.  A completed first
pass is checkpointed and reused only when the input SHA-256 and member name
match exactly.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterator

BASE_BITS = {"A": 1, "C": 2, "G": 4, "T": 8, "U": 8}
CHECKPOINT_VERSION = 2


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def select_coordinates(
    eligible: list[int], maximum: int | None, seed: int
) -> list[int]:
    if maximum is None or len(eligible) <= maximum:
        return eligible
    selected = sorted(
        eligible, key=lambda index: (splitmix64(index ^ seed), index)
    )[:maximum]
    return sorted(selected)


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
    # The published LTPlus tree contains a small number of non-UTF-8 bytes in
    # the descriptive text following leaf accessions (notably 0xa0 and 0xca).
    # ISO-8859-1 is deliberately byte-preserving.  The normalization below
    # retains only the ASCII accession preceding the first comma, so none of
    # that descriptive text enters the benchmark tree or its taxon mapping.
    source = path.read_text(encoding="iso-8859-1")
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
    parser.add_argument(
        "--maximum-coordinates",
        type=int,
        default=512,
        help="deterministic cap after coverage filtering; zero retains all",
    )
    parser.add_argument("--selection-seed", type=int, default=20260813)
    parser.add_argument("--analyze-only", action="store_true")
    arguments = parser.parse_args()
    if not 0 < arguments.minimum_observed_fraction <= 1:
        parser.error("coverage fraction must lie in (0, 1]")
    if arguments.maximum_coordinates < 0:
        parser.error("maximum-coordinates must be nonnegative")
    if not 0 <= arguments.selection_seed < (1 << 64):
        parser.error("selection-seed must be an unsigned 64-bit integer")
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
    eligible = [
        index for index, count in enumerate(coverage) if count >= threshold
    ]
    maximum = arguments.maximum_coordinates or None
    retained = select_coordinates(eligible, maximum, arguments.selection_seed)
    if not retained:
        raise ValueError("coverage rule retained no alignment coordinates")
    selected_indices_sha256 = hashlib.sha256(
        "".join(f"{index}\n" for index in retained).encode("ascii")
    ).hexdigest()
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
        "eligible_coordinates": len(eligible),
        "eligible_variable_coordinates": sum(
            1 for index in eligible if states[index] & (states[index] - 1)
        ),
        "maximum_coordinates": maximum,
        "selection_seed": arguments.selection_seed,
        "selected_indices_sha256": selected_indices_sha256,
        "retained_coordinates": len(retained),
        "coverage_fraction_quantiles": coverage_quantiles,
        "selection_rule": (
            "all coordinates with A/C/G/T/U coverage >= threshold; if above "
            "maximum_coordinates, lowest SplitMix64(index XOR selection_seed) "
            "ranks, emitted in original coordinate order"
        ),
        "leaf_label_transform": "exact prefix before first comma, stripped; required unique and complete",
        "source_tree_encoding": "ISO-8859-1 byte-preserving decode; retained accession prefixes are ASCII",
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
    alignment_output = arguments.output / "patterns.fasta"
    weights_output = arguments.output / "pattern_weights.txt"
    seen: list[str] = []
    columns = [bytearray() for _ in retained]
    for name, sequence in fasta_records(arguments.archive, member):
        seen.append(name)
        selected = "".join(sequence[index] for index in retained).replace("U", "T")
        for column, character in zip(columns, selected):
            column.append(ord(character))
    if seen != names:
        raise ValueError("FASTA record order changed between streaming passes")
    by_pattern: dict[bytes, int] = {}
    patterns: list[bytes] = []
    weights: list[int] = []
    for column in columns:
        pattern = bytes(column)
        existing = by_pattern.get(pattern)
        if existing is None:
            by_pattern[pattern] = len(patterns)
            patterns.append(pattern)
            weights.append(1)
        else:
            weights[existing] += 1
    with alignment_output.open("w", encoding="ascii") as output:
        for taxon, name in enumerate(names):
            compressed = bytes(pattern[taxon] for pattern in patterns).decode("ascii")
            output.write(f">{name}\n")
            for start in range(0, len(compressed), 80):
                output.write(compressed[start : start + 80] + "\n")
    weights_output.write_text(
        "".join(f"{weight}\n" for weight in weights), encoding="ascii"
    )
    summary.update(
        normalized_tree_sha256=sha256(tree_output),
        normalized_alignment_sha256=sha256(alignment_output),
        pattern_weights_sha256=sha256(weights_output),
    )
    (arguments.output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_fields = (
        "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
        "pattern_weights", "tree", "source_alignment_sha256",
        "source_tree_sha256", "normalized_alignment_sha256",
        "pattern_weights_sha256", "normalized_tree_sha256",
        "selected_indices_sha256", "selection_rule", "source_tree_encoding",
    )
    manifest_row = {
        "dataset": "ltplus-february-2026",
        "taxa": len(names),
        "raw_sites": len(retained),
        "unique_patterns": len(weights),
        "alignment": alignment_output.name,
        "pattern_weights": weights_output.name,
        "tree": tree_output.name,
        "source_alignment_sha256": summary["archive_sha256"],
        "source_tree_sha256": summary["source_tree_sha256"],
        "normalized_alignment_sha256": summary["normalized_alignment_sha256"],
        "pattern_weights_sha256": summary["pattern_weights_sha256"],
        "normalized_tree_sha256": summary["normalized_tree_sha256"],
        "selected_indices_sha256": selected_indices_sha256,
        "selection_rule": summary["selection_rule"],
        "source_tree_encoding": summary["source_tree_encoding"],
    }
    with (arguments.output / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerow(manifest_row)
    (arguments.output / "excluded.csv").write_text(
        "source_relative_directory,reason\n", encoding="utf-8"
    )
    metadata_lines = {
        "corpus": "ltplus-february-2026",
        "corpus_kind": "empirical-tree-alignment-coordinate-subset",
        "source_tree_sha256": summary["source_tree_sha256"],
        "source_alignment_sha256": summary["archive_sha256"],
        "minimum_observed_fraction": arguments.minimum_observed_fraction,
        "eligible_coordinates": len(eligible),
        "maximum_coordinates": maximum if maximum is not None else "all",
        "selection_seed": arguments.selection_seed,
        "selected_indices_sha256": selected_indices_sha256,
        "selection_rule": summary["selection_rule"],
        "source_tree_encoding": summary["source_tree_encoding"],
        "pattern_compression": "exact-duplicate-selected-columns",
        "redistribution": "none; outputs are prepared locally from official endpoints",
    }
    (arguments.output / "corpus_metadata.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in metadata_lines.items()),
        encoding="utf-8",
    )
    print(f"retained {len(retained)} of {len(states)} coordinates for {len(names)} taxa")


if __name__ == "__main__":
    main()
