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
A C G U N -
>B
A T G T N -
>C
A T G C N -
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
grep -Fq '>A' "${work}/prepared/alignment.fasta"
grep -Fq 'CT' "${work}/prepared/alignment.fasta"
! grep -Fq 'U' "${work}/prepared/alignment.fasta"
grep -Fq '(A:0.1,B:0.2)' "${work}/prepared/tree.nwk"
python3 - "${work}/prepared/manifest.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary["taxa"] == 3
assert summary["coordinates"] == 6
assert summary["retained_coordinates"] == 2
assert summary["minimum_observed_taxa"] == 2
PY
python3 - "${root}" <<'PY'
import importlib.util
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / "scripts" / "prepare_dryad_corpus.py"
spec = importlib.util.spec_from_file_location("prepare_dryad_corpus", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.dataset_id(pathlib.Path("first/shared")) != module.dataset_id(
    pathlib.Path("second/shared")
)
assert module.dataset_id(pathlib.Path("first/shared")) == module.dataset_id(
    pathlib.Path("first/shared")
)
PY
# A second run must reuse the exact-source first-pass checkpoint.
python3 "${root}/scripts/prepare_streaming_alignment.py" \
  "${work}/alignment.zip" "${work}/tree.ntree" "${work}/prepared" >/dev/null
