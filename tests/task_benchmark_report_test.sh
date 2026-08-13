#!/usr/bin/env bash
set -euo pipefail

summary="$(python3 scripts/summarize_task_benchmarks.py \
  tests/task_benchmark_report_fixture.txt \
  --backend cuda --precision FP32 --benchmark-mode full-input-update)"
grep -q '^cuda,FP32,full-input-update,likelihood,1,3,3,3,1e-06,1e-07,0$' \
  <<<"${summary}"
grep -q '^cuda,FP32,full-input-update,joint-map,1,3,3,3,1e-06,1e-07,0$' \
  <<<"${summary}"

bad_report="${TEST_TMPDIR:-${TMPDIR:-/tmp}}/bad-task-report.txt"
sed 's/,3,1e-6,1e-7,0,/,4,1e-6,1e-7,0,/' \
  tests/task_benchmark_report_fixture.txt >"${bad_report}"
if python3 scripts/summarize_task_benchmarks.py "${bad_report}" >/dev/null 2>&1; then
  echo "task parser accepted a speedup stored in the wrong CSV field" >&2
  exit 1
fi

if python3 scripts/summarize_benchmarks.py \
  tests/task_benchmark_report_fixture.txt >/dev/null 2>&1; then
  echo "likelihood summarizer accepted inference-task CSV rows" >&2
  exit 1
fi
