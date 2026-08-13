#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
else
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
work="$(mktemp -d "${TMPDIR:-/tmp}/corpus-preparation-test.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
cat > "${work}/alignment.fasta" <<'EOF'
>A metadata
A C G U A N
>B
A T G T A N
>C
A T G C A N
EOF
python3 - "${work}" <<'PY'
import sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
with zipfile.ZipFile(root / "alignment.zip", "w") as archive:
    archive.write(root / "alignment.fasta", "alignment.fasta")
PY
cat > "${work}/tree.ntree" <<'EOF'
[ARB comment containing a misleading ( parenthesis]
(('A, (metadata)':0.1,B:0.2):0.3,C:0.4);
EOF
python3 "${root}/scripts/prepare_streaming_alignment.py" \
  "${work}/alignment.zip" "${work}/tree.ntree" "${work}/prepared"
grep -Fq '>A' "${work}/prepared/patterns.fasta"
! grep -Fq 'U' "${work}/prepared/patterns.fasta"
grep -Fq '(A:0.1,B:0.2)' "${work}/prepared/tree.nwk"
python3 - "${work}/prepared/manifest.json" <<'PY'
import csv, json, pathlib, sys
summary = json.load(open(sys.argv[1]))
assert summary["taxa"] == 3
assert summary["coordinates"] == 6
assert summary["retained_coordinates"] == 5
assert summary["eligible_variable_coordinates"] == 2
assert summary["minimum_observed_taxa"] == 2
alignment = (pathlib.Path(sys.argv[1]).parent / "patterns.fasta").read_text()
weights = [int(value) for value in (
    pathlib.Path(sys.argv[1]).parent / "pattern_weights.txt"
).read_text().split()]
assert sum(weights) == 5
assert sorted(weights) == [1, 1, 1, 2]  # The two invariant A columns coalesce.
assert "ACGT" in alignment
row = next(csv.DictReader(
    (pathlib.Path(sys.argv[1]).parent / "manifest.csv").open()
))
assert row["raw_sites"] == "5" and row["unique_patterns"] == "4"
PY
python3 - "${root}" "${work}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "scripts" / "prepare_dryad_corpus.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("prepare_dryad_corpus", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.dataset_id(pathlib.Path("first/shared")) != module.dataset_id(
    pathlib.Path("second/shared")
)
assert module.dataset_id(pathlib.Path("first/shared")) == module.dataset_id(
    pathlib.Path("first/shared")
)

raw = pathlib.Path(sys.argv[2]) / "dryad-fixture"
selected = raw / module.COHORT_RELATIVE / "selected"
excluded = raw / "stopping_criteria_data" / "unsuccessful_MSAs" / "duplicate"
selected.mkdir(parents=True)
excluded.mkdir(parents=True)
(selected / "alignment.phy").touch()
(excluded / "alignment.phy").touch()
assert module.cohort_alignments(raw) == [selected / "alignment.phy"]
PY

# A miniature immutable TreeBASE Mirror exercises DNA filtering, exact taxon
# matching, collision-safe identifiers, source hashes, and deterministic rank
# selection without accessing the network.
mirror="${work}/treebase-mirror"
mkdir -p "${mirror}/trees/alpha.phy" "${mirror}/trees/mismatch.phy" \
  "${mirror}/trees/absent.phy"
cat > "${work}/mirror.fasta" <<'EOF'
>taxon1
ACGTACGT
>taxon2
ACGTTCGT
>taxon3
ACGTACGA
EOF
python3 - "${work}/mirror.fasta" "${mirror}" <<'PY'
import io, pathlib, tarfile, sys
alignment = pathlib.Path(sys.argv[1]).read_bytes()
root = pathlib.Path(sys.argv[2])
for name in ("alpha.phy", "mismatch.phy"):
    path = root / "trees" / name / f"{name}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("msa.fasta")
        info.size = len(alignment)
        info.mtime = 0
        archive.addfile(info, io.BytesIO(alignment))
PY
cat > "${mirror}/trees/alpha.phy/tree_best.newick" <<'EOF'
((taxon1:0.1,taxon2:0.2):0.3,taxon3:0.4);
EOF
cat > "${mirror}/trees/mismatch.phy/tree_best.newick" <<'EOF'
((taxon1:0.1,taxon2:0.2):0.3,different:0.4);
EOF
cat > "${mirror}/trees/absent.phy/tree_best.newick" <<'EOF'
((taxon1:0.1,taxon2:0.2):0.3,taxon3:0.4);
EOF
for directory in alpha.phy mismatch.phy absent.phy; do
  printf 'fixture log\n' > "${mirror}/trees/${directory}/log_0.txt"
  printf 'GTR fixture\n' > "${mirror}/trees/${directory}/model_0.txt"
done
git -C "${mirror}" init -q
git -C "${mirror}" config user.name fixture
git -C "${mirror}" config user.email fixture@example.invalid
git -C "${mirror}" add .
git -C "${mirror}" commit -q -m fixture
revision="$(git -C "${mirror}" rev-parse HEAD)"
python3 "${root}/scripts/prepare_treebase_mirror.py" \
  "${mirror}" "${work}/treebase-a" --revision "${revision}" \
  --maximum-datasets 10 --minimum-taxa 2
python3 "${root}/scripts/prepare_treebase_mirror.py" \
  "${mirror}" "${work}/treebase-b" --revision "${revision}" \
  --maximum-datasets 10 --minimum-taxa 2
diff -u "${work}/treebase-a/manifest.csv" "${work}/treebase-b/manifest.csv"
python3 - "${work}/treebase-a" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / "manifest.csv").open()))
excluded = list(csv.DictReader((root / "excluded.csv").open()))
assert len(rows) == 1 and rows[0]["taxa"] == "3"
assert len(excluded) == 2
reasons = [row["reason"] for row in excluded]
assert sum("taxa differ" in reason for reason in reasons) == 1
assert sum(
    "missing required source entry: trees/absent.phy/absent.phy.tar.gz" in reason
    for reason in reasons
) == 1
assert rows[0]["source_revision"]
assert "empirical-tree-alignment-pairs" in (root / "corpus_metadata.txt").read_text()
PY

# Repository/object access failures are infrastructure failures, not reasons
# to exclude a biological record. Simulate an unavailable promisor blob by
# removing one committed object from a disposable repository; preparation
# must fail and preserve Git's diagnostic rather than report zero selections.
archive_relative=trees/alpha.phy/alpha.phy.tar.gz
archive_oid="$(git -C "${mirror}" rev-parse "HEAD:${archive_relative}")"
archive_object="${mirror}/.git/objects/${archive_oid:0:2}/${archive_oid:2}"
mv "${mirror}/${archive_relative}" "${mirror}/${archive_relative}.missing"
mv "${archive_object}" "${archive_object}.missing"
if python3 "${root}/scripts/prepare_treebase_mirror.py" \
  "${mirror}" "${work}/treebase-missing-blob" --revision "${revision}" \
  --maximum-datasets 10 --minimum-taxa 2 \
  >"${work}/missing-blob.stdout" 2>"${work}/missing-blob.stderr"; then
  echo "TreeBASE preparation unexpectedly accepted an unavailable Git blob" >&2
  exit 1
fi
grep -Fq 'Git repository command failed' "${work}/missing-blob.stderr"
grep -Fq 'fatal:' "${work}/missing-blob.stderr"
grep -Fq "${archive_relative}" "${work}/missing-blob.stderr"
[[ ! -f "${work}/treebase-missing-blob/manifest.csv" ]]

# The RAxML-Grove fixture verifies stratified hash sampling and deterministic
# JC69 simulation on empirical-style topologies. The binary source is never
# eligible even if its hash precedes one of the DNA records.
grove="${work}/raxml-grove"
mkdir -p "${grove}/trees/dna-small" "${grove}/trees/dna-large" \
  "${grove}/trees/binary"
cat > "${grove}/trees/dna-small/tree_best.newick" <<'EOF'
((taxon1:0.1,taxon2:0.2):0.3,taxon3:0.4);
EOF
cat > "${grove}/trees/dna-large/tree_best.newick" <<'EOF'
((((taxon1:0.1,taxon2:0.2):0.3,taxon3:0.4):0.2,taxon4:0.1):0.1,taxon5:0.2);
EOF
cat > "${grove}/trees/binary/tree_best.newick" <<'EOF'
((taxon1:0.1,taxon2:0.2):0.3,taxon3:0.4);
EOF
cat > "${grove}/trees/dna-small/log_0.txt" <<'EOF'
Alignment sites: 80
DataType: DNA
EOF
cat > "${grove}/trees/dna-large/log_0.txt" <<'EOF'
Model: GTR+G4m
Alignment sites / patterns: 120 / 70
EOF
cat > "${grove}/trees/binary/log_0.txt" <<'EOF'
Model: BIN+G4m
Alignment sites / patterns: 50 / 20
EOF
git -C "${grove}" init -q
git -C "${grove}" config user.name fixture
git -C "${grove}" config user.email fixture@example.invalid
git -C "${grove}" add .
git -C "${grove}" commit -q -m fixture
revision="$(git -C "${grove}" rev-parse HEAD)"
python3 "${root}/scripts/prepare_raxml_grove.py" \
  "${grove}" "${work}/grove-a" --revision "${revision}" --sites 12 \
  --per-bin 1 --taxa-bin-edges 2,4,10
python3 "${root}/scripts/prepare_raxml_grove.py" \
  "${grove}" "${work}/grove-b" --revision "${revision}" --sites 12 \
  --per-bin 1 --taxa-bin-edges 2,4,10
diff -u "${work}/grove-a/manifest.csv" "${work}/grove-b/manifest.csv"
python3 - "${root}" "${work}/grove-a" <<'PY'
import csv, pathlib, sys
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "scripts"))
from corpus_common import parse_newick, read_fasta_bytes
root = pathlib.Path(sys.argv[2])
rows = list(csv.DictReader((root / "manifest.csv").open()))
assert len(rows) == 2 and {row["taxa_bin"] for row in rows} == {"[2,4)", "[4,10)"}
for row in rows:
    names, sequences = read_fasta_bytes((root / row["alignment"]).read_bytes())
    leaves = parse_newick((root / row["tree"]).read_text()).leaf_labels
    assert set(names) == set(leaves)
    weights = [int(value) for value in (root / row["pattern_weights"]).read_text().split()]
    assert sum(weights) == 12 and all(len(sequence) == len(weights) for sequence in sequences)
metadata = (root / "corpus_metadata.txt").read_text()
assert "simulated-JC69-alignments-on-empirical-topologies" in metadata
assert "source_alignments_available=no" in metadata
PY
python3 "${root}/scripts/empirical_manifest_rows.py" \
  "${work}/grove-a/manifest.csv" > "${work}/grove-rows.tsv"
[[ "$(wc -l < "${work}/grove-rows.tsv" | tr -d ' ')" == 2 ]]
# A second run must reuse the exact-source first-pass checkpoint.
python3 "${root}/scripts/prepare_streaming_alignment.py" \
  "${work}/alignment.zip" "${work}/tree.ntree" "${work}/prepared" >/dev/null
