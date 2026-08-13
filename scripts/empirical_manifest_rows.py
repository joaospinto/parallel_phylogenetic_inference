#!/usr/bin/env python3
"""Validate an empirical-corpus manifest and emit its benchmark fields."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path, PurePosixPath

FIELDS = (
    "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
    "pattern_weights", "tree",
)
HASH_FIELDS = {
    "alignment": "normalized_alignment_sha256",
    "pattern_weights": "pattern_weights_sha256",
    "tree": "normalized_tree_sha256",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} manifest.csv")
    manifest = Path(sys.argv[1]).resolve()
    root = manifest.parent
    identifiers: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or ())]
        if "normalized_tree_sha256" not in (reader.fieldnames or ()) and \
                "tree_sha256" in (reader.fieldnames or ()):
            HASH_FIELDS["tree"] = "tree_sha256"
        missing.extend(
            field for field in HASH_FIELDS.values()
            if field not in (reader.fieldnames or ())
        )
        if missing:
            raise ValueError(f"manifest lacks required fields: {missing}")
        for line, row in enumerate(reader, start=2):
            values = [row[field] for field in FIELDS]
            if any("\t" in value or "\n" in value or "\r" in value for value in values):
                raise ValueError(f"manifest line {line} contains a control character")
            dataset = row["dataset"]
            if not dataset or dataset in identifiers:
                raise ValueError(f"manifest line {line} has an empty or duplicate dataset")
            identifiers.add(dataset)
            for field in ("taxa", "raw_sites", "unique_patterns"):
                if int(row[field]) < 1:
                    raise ValueError(f"manifest line {line} has invalid {field}")
            for field in ("alignment", "pattern_weights", "tree"):
                relative = PurePosixPath(row[field])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"manifest line {line} has unsafe {field}")
                resolved = (root / Path(*relative.parts)).resolve()
                if os.path.commonpath((root, resolved)) != str(root) or not resolved.is_file():
                    raise ValueError(f"manifest line {line} cannot resolve {field}")
                digest_field = HASH_FIELDS[field]
                expected = row[digest_field].lower()
                if len(expected) != 64 or any(
                    character not in "0123456789abcdef" for character in expected
                ):
                    raise ValueError(
                        f"manifest line {line} has invalid {digest_field}"
                    )
                observed = sha256(resolved)
                if observed != expected:
                    raise ValueError(
                        f"manifest line {line} {field} checksum mismatch: "
                        f"expected {expected}, observed {observed}"
                    )
            print("\t".join(values))


if __name__ == "__main__":
    main()
