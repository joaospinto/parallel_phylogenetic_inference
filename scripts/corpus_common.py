#!/usr/bin/env python3
"""Shared, dependency-free utilities for empirical-corpus preparation."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

DNA = frozenset("ACGTURYSWKMBDHVN?-.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def dataset_id(relative: Path | str, prefix: str = "") -> str:
    source = relative.as_posix() if isinstance(relative, Path) else str(relative)
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_.-")
    if not readable:
        readable = "dataset"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{readable}-{digest}"


def stable_rank(namespace: str, source_id: str) -> str:
    return hashlib.sha256(
        f"{namespace}\0{source_id}".encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ParsedNewick:
    parents: tuple[int, ...]
    labels: tuple[str, ...]
    lengths: tuple[float, ...]

    @property
    def leaves(self) -> tuple[int, ...]:
        child_counts = [0] * len(self.parents)
        for parent in self.parents:
            if parent >= 0:
                child_counts[parent] += 1
        return tuple(index for index, count in enumerate(child_counts) if count == 0)

    @property
    def leaf_labels(self) -> tuple[str, ...]:
        return tuple(self.labels[index] for index in self.leaves)


class NewickParser:
    """Iterative Newick parser suitable for arbitrarily deep trees."""

    def __init__(self, text: str):
        self.text = text
        self.position = 0
        self.parents: list[int] = []
        self.labels: list[str] = []
        self.lengths: list[float] = []
        self.stack: list[int] = []

    def fail(self, message: str) -> None:
        raise ValueError(f"Newick parse error at byte {self.position}: {message}")

    def skip_ignored(self) -> None:
        while self.position < len(self.text):
            if self.text[self.position].isspace():
                self.position += 1
                continue
            if self.text[self.position] != "[":
                return
            depth = 1
            self.position += 1
            while self.position < len(self.text) and depth:
                depth += (self.text[self.position] == "[") - (
                    self.text[self.position] == "]"
                )
                self.position += 1
            if depth:
                self.fail("unterminated comment")

    def optional_label(self) -> str:
        self.skip_ignored()
        if self.position >= len(self.text) or self.text[self.position] in ",):;":
            return ""
        if self.text[self.position] == "'":
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
            self.fail("unterminated quoted label")
        begin = self.position
        while self.position < len(self.text) and self.text[self.position] not in "(),:;[] \t\r\n":
            self.position += 1
        return self.text[begin : self.position]

    def optional_length(self) -> float:
        self.skip_ignored()
        if self.position >= len(self.text) or self.text[self.position] != ":":
            return 0.0
        self.position += 1
        self.skip_ignored()
        begin = self.position
        while self.position < len(self.text) and self.text[self.position] not in ",);[ \t\r\n":
            self.position += 1
        token = self.text[begin : self.position]
        try:
            value = float(token)
        except ValueError:
            self.fail("invalid branch length")
        if not math.isfinite(value) or value < 0:
            self.fail("branch lengths must be finite and nonnegative")
        return value

    def add_node(self, label: str = "") -> int:
        node = len(self.parents)
        self.parents.append(self.stack[-1] if self.stack else -1)
        self.labels.append(label)
        self.lengths.append(0.0)
        return node

    def parse(self) -> ParsedNewick:
        expect_subtree = True
        root: int | None = None
        while True:
            self.skip_ignored()
            if self.position >= len(self.text):
                self.fail("missing Newick terminator")
            character = self.text[self.position]
            if expect_subtree:
                if character == "(":
                    node = self.add_node()
                    if root is None:
                        root = node
                    self.stack.append(node)
                    self.position += 1
                    continue
                label = self.optional_label()
                if not label:
                    self.fail("a leaf must have a label")
                node = self.add_node(label)
                if root is None:
                    root = node
                self.lengths[node] = self.optional_length()
                expect_subtree = False
                continue
            if character == ",":
                if not self.stack:
                    self.fail("comma outside an internal node")
                self.position += 1
                expect_subtree = True
                continue
            if character == ")":
                if not self.stack:
                    self.fail("unmatched closing parenthesis")
                node = self.stack.pop()
                self.position += 1
                self.labels[node] = self.optional_label()
                self.lengths[node] = self.optional_length()
                expect_subtree = False
                continue
            if character == ";":
                if self.stack:
                    self.fail("unterminated internal node")
                self.position += 1
                self.skip_ignored()
                if self.position != len(self.text):
                    self.fail("text follows Newick terminator")
                break
            self.fail(f"expected ',', ')', or ';', found {character!r}")
        if root != 0 or not self.parents:
            self.fail("empty or malformed tree")
        tree = ParsedNewick(
            tuple(self.parents), tuple(self.labels), tuple(self.lengths)
        )
        if any(not tree.labels[node] for node in tree.leaves):
            self.fail("an unlabelled leaf remains")
        return tree


def parse_newick(text: str) -> ParsedNewick:
    return NewickParser(text).parse()


def read_fasta_bytes(contents: bytes) -> tuple[list[str], list[str]]:
    names: list[str] = []
    sequences: list[str] = []
    name: str | None = None
    fragments: list[str] = []
    for raw in contents.decode("ascii").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            if name is not None:
                sequences.append("".join(fragments))
            fields = line[1:].split()
            if not fields:
                raise ValueError("empty FASTA header")
            name = fields[0]
            names.append(name)
            fragments = []
        elif line:
            if name is None:
                raise ValueError("FASTA sequence precedes its header")
            fragment = "".join(line.split()).upper().replace("U", "T")
            if not fragment or not set(fragment) <= DNA:
                raise ValueError("alignment contains a non-nucleotide character")
            fragments.append(fragment)
    if name is not None:
        sequences.append("".join(fragments))
    if len(names) < 2 or len(sequences) != len(names):
        raise ValueError("alignment must contain at least two FASTA records")
    if len(set(names)) != len(names):
        raise ValueError("alignment contains duplicate FASTA names")
    sites = len(sequences[0])
    if sites < 1 or any(len(sequence) != sites for sequence in sequences):
        raise ValueError("FASTA records do not form a nonempty alignment")
    return names, sequences


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
    compressed = [
        "".join(pattern[taxon] for pattern in unique)
        for taxon in range(len(sequences))
    ]
    return compressed, weights


def write_fasta(path: Path, names: list[str], sequences: list[str]) -> None:
    with path.open("w", encoding="ascii") as stream:
        for name, sequence in zip(names, sequences):
            stream.write(f">{name}\n")
            for start in range(0, len(sequence), 80):
                stream.write(sequence[start : start + 80] + "\n")


class GitCommandError(RuntimeError):
    """A Git metadata/object operation failed; never a corpus exclusion."""


class GitRepository:
    """Read immutable blobs from a full or partial Git repository."""

    def __init__(self, path: Path, revision: str):
        self.path = path.resolve()
        self.revision = self.run("rev-parse", f"{revision}^{{commit}}").decode().strip()
        head = self.run("rev-parse", "HEAD", check=False).decode().strip()
        clean = not self.run(
            "status", "--porcelain", "--untracked-files=no", check=False
        ).strip()
        self.use_checkout = head == self.revision and clean
        remotes = self.run("remote", "get-url", "origin", check=False).decode().strip()
        self.remote = remotes or "unknown"

    def run(self, *arguments: str, check: bool = True) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(self.path), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check and result.returncode != 0:
            diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
            command = "git " + " ".join(arguments)
            raise GitCommandError(
                f"Git repository command failed ({command}, exit "
                f"{result.returncode}): {diagnostic or 'no diagnostic'}"
            )
        return result.stdout

    def directories(self, tree: str) -> list[str]:
        output = self.run(
            "ls-tree", "-d", "--name-only", f"{self.revision}:{tree}"
        ).decode("utf-8")
        return sorted(line for line in output.splitlines() if line)

    def blob(self, relative: str) -> bytes:
        checked_out = self.path / relative
        if self.use_checkout and checked_out.is_file():
            return checked_out.read_bytes()
        return self.run("show", f"{self.revision}:{relative}")

    def blob_oid(self, relative: str) -> str:
        result = self.blob_oid_optional(relative)
        if result is None:
            raise FileNotFoundError(relative)
        return result

    def blob_oid_optional(self, relative: str) -> str | None:
        output = self.run("ls-tree", self.revision, "--", relative).decode()
        fields = output.split(None, 3)
        if len(fields) < 3:
            return None
        return fields[2]


def write_metadata(path: Path, entries: list[tuple[str, object]]) -> None:
    lines: list[str] = []
    for key, value in entries:
        text = str(value).replace("\n", " ").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError(f"invalid metadata key {key!r}")
        lines.append(f"{key}={text}\n")
    path.write_text("".join(lines), encoding="utf-8")
