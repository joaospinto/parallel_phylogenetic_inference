#!/usr/bin/env python3
"""Validate an empirical-corpus manifest and emit its benchmark fields."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path, PurePosixPath

FIELDS = (
    "dataset", "taxa", "raw_sites", "unique_patterns", "alignment",
    "pattern_weights", "tree",
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} manifest.csv")
    manifest = Path(sys.argv[1]).resolve()
    root = manifest.parent
    identifiers: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [field for field in FIELDS if field not in (reader.fieldnames or ())]
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
            print("\t".join(values))


if __name__ == "__main__":
    main()
