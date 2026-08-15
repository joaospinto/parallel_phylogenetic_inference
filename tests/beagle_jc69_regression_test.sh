#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  binary="${TEST_SRCDIR}/${TEST_WORKSPACE}/beagle_benchmark"
else
  binary="${1:-bazel-bin/beagle_benchmark}"
fi
if [[ -n "${BEAGLE_PREFIX:-}" ]]; then
  beagle_libraries="${BEAGLE_PREFIX}/lib:${BEAGLE_PREFIX}/lib64"
  beagle_libraries+="${beagle_libraries:+:}${BEAGLE_PREFIX}/lib/x86_64-linux-gnu"
  export LD_LIBRARY_PATH="${beagle_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export DYLD_LIBRARY_PATH="${beagle_libraries}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
fi
work_directory="$(mktemp -d "${TEST_TMPDIR:-${TMPDIR:-/tmp}}/beagle-jc69.XXXXXX")"
trap 'rm -rf "${work_directory}"' EXIT
expected_precision="${PRECISION_LABEL:-FP32}"

"${binary}" --beagle-resource cpu --beagle-threads 1 \
  --topology balanced --leaves 64 --sites 64 \
  --synthetic-sequence-model jc69 \
  --evolutionary-root-to-tip-distance 0.00000001 \
  --compress-patterns true --seed 20260814 --replicates 2 --repeats 1 \
  --conditioning-ms 0 --study-label beagle-tiny-branch-regression > \
  "${work_directory}/tiny.txt"

"${binary}" --beagle-resource cpu --beagle-threads 1 \
  --topology caterpillar --leaves 8192 --sites 64 \
  --synthetic-sequence-model jc69 \
  --evolutionary-root-to-tip-distance 0.0001 \
  --compress-patterns true --seed 20260814 --replicates 1 --repeats 1 \
  --conditioning-ms 0 --study-label beagle-deep-tree-regression > \
  "${work_directory}/deep.txt"

"${binary}" --beagle-resource cpu --beagle-threads 1 \
  --topology yule --leaves 128 --sites 73 \
  --nucleotide-model gtr --rate-categories 4 --gamma-shape 0.5 \
  --seed 20260815 --replicates 1 --repeats 1 --conditioning-ms 0 \
  --study-label beagle-gtr-gamma-regression > \
  "${work_directory}/gtr-gamma.txt"

python3 - "${work_directory}/tiny.txt" "${work_directory}/deep.txt" \
  "${work_directory}/gtr-gamma.txt" "${expected_precision}" <<'PY'
import csv
import math
import pathlib
import sys


def rows(path):
    header = None
    result = []
    for raw in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = next(csv.reader([raw]))
        if fields[0] == "baseline":
            header = fields
        elif header is not None and fields[0] == "beagle":
            result.append(dict(zip(header, fields)))
    return result


tiny = rows(sys.argv[1])
deep = rows(sys.argv[2])
gtr_gamma = rows(sys.argv[3])
expected_precision = sys.argv[4]
if len(tiny) != 2 or len(deep) != 1 or len(gtr_gamma) != 1:
    raise SystemExit("BEAGLE reversible-model regression produced an unexpected row count")
for row in tiny + deep:
    if row["precision"] != expected_precision or row["sequence_generation"] != "jc69":
        raise SystemExit("BEAGLE JC69 regression selected the wrong protocol")
    for field in ("max_abs_error", "max_relative_error"):
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise SystemExit(f"invalid {field} in BEAGLE JC69 regression")
row = gtr_gamma[0]
if (
    row["precision"] != expected_precision
    or row["substitution_model"] != "gtr"
    or row["rate_categories"] != "4"
    or float(row["gamma_shape"]) != 0.5
):
    raise SystemExit("BEAGLE GTR+Gamma regression selected the wrong protocol")
threshold = 2e-3 if expected_precision == "FP32" else 1e-10
if float(row["max_relative_error"]) > threshold:
    raise SystemExit(
        "GTR+Gamma error exceeds the publication gate: "
        + row["max_relative_error"]
    )
if float(deep[0]["max_relative_error"]) > threshold:
    raise SystemExit(
        "deep short-edge JC69 error exceeds the publication gate: "
        + deep[0]["max_relative_error"]
    )
# The stable implementation is about 7.5e-8 on this deterministic case,
# whereas BEAGLE's FP32 eigen update before this fix was about 2.4e-4.  Keep a
# deliberately generous cancellation-sensitive bound in addition to the
# general publication gate above.
if expected_precision == "FP32" and float(deep[0]["max_relative_error"]) > 1e-5:
    raise SystemExit(
        "deep short-edge JC69 regression detected transition cancellation: "
        + deep[0]["max_relative_error"]
    )
PY
