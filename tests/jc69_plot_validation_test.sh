#!/usr/bin/env bash
set -euo pipefail

output="${TEST_TMPDIR:-${TMPDIR:-/tmp}}/jc69-plot"
python3 scripts/plot_jc69_simulation_study.py \
  tests/jc69_plot_report_fixture.txt --native cuda --baseline beagle-cpu \
  --precision FP32 --benchmark-mode full-input-update --beagle-threads 1 \
  --output-directory "${output}" --validate-only --require-interleaved
test -s "${output}/jc69_paired_replicates_fp32_cuda_vs_beagle-cpu_full-input-update_t1.csv"

bad="${TEST_TMPDIR:-${TMPDIR:-/tmp}}/bad-jc69-plot-report.txt"
sed 's/,5,16,110,3,0,1,1.5,/,5,16,111,3,0,1,1.5,/' \
  tests/jc69_plot_report_fixture.txt >"${bad}"
if python3 scripts/plot_jc69_simulation_study.py "${bad}" \
     --native cuda --baseline beagle-cpu --precision FP32 \
     --benchmark-mode full-input-update --beagle-threads 1 \
     --output-directory "${output}" --validate-only >/dev/null 2>&1; then
  echo "JC69 plot validator accepted mismatched structural metadata" >&2
  exit 1
fi

dry_run="${TEST_TMPDIR:-${TMPDIR:-/tmp}}/jc69-dry-run.txt"
TREE_HMM_DRY_RUN=1 TREE_HMM_JC69_PROFILE=paper \
TREE_HMM_JC69_TOPOLOGIES=yule TREE_HMM_JC69_LEAVES='128 1024' \
TREE_HMM_JC69_RAW_SITES='256 8192' \
TREE_HMM_JC69_EVOLUTIONARY_HEIGHTS=0.001 TREE_HMM_JC69_REPLICATES=2 \
  bash scripts/benchmark_jc69_simulations.sh metal >"${dry_run}"
grep -Fq '# cases_per_method_and_precision=8' "${dry_run}"
grep -Fq 'leaves=128 raw_sites=256 evolutionary_root_to_tip_distance=0.001 replicate_start=0 replicates=2 timing_repeats=5' "${dry_run}"
grep -Fq 'leaves=1024 raw_sites=8192 evolutionary_root_to_tip_distance=0.001 replicate_start=0 replicates=2 timing_repeats=1' "${dry_run}"
[[ "$(grep -c '^# benchmark_start_jc69 ' "${dry_run}")" == 4 ]]

interleaved="${TEST_TMPDIR:-${TMPDIR:-/tmp}}/jc69-interleaved-dry-run.txt"
TREE_HMM_DRY_RUN=1 TREE_HMM_JC69_PROFILE=paper \
TREE_HMM_JC69_TOPOLOGIES=yule TREE_HMM_JC69_LEAVES='128 1024' \
TREE_HMM_JC69_RAW_SITES='256 8192' \
TREE_HMM_JC69_EVOLUTIONARY_HEIGHTS=0.001 TREE_HMM_JC69_REPLICATES=2 \
  bash scripts/benchmark_jc69_simulations.sh --interleave \
    metal beagle-cpu:1 beagle-cpu:10 > "${interleaved}"
grep -Fq '# planned_method_case_runs=24' "${interleaved}"
[[ "$(grep -c '^# interleaved_case ' "${interleaved}")" == 24 ]]
[[ "$(grep -c '^# benchmark_start_jc69 ' "${interleaved}")" == 24 ]]
python3 scripts/validate_interleaved_schedule.py "${interleaved}" \
  --study clock-like-jc69-simulation --precision FP32 \
  --benchmark-mode full-input-update --expected-cases 8 \
  --require-method metal --require-method beagle-cpu:1 \
  --require-method beagle-cpu:10 >/dev/null
grep -Fq 'leaves=128 raw_sites=256 evolutionary_root_to_tip_distance=0.001 replicate_start=0 replicates=1 timing_repeats=5' \
  "${interleaved}"
grep -Fq 'leaves=1024 raw_sites=8192 evolutionary_root_to_tip_distance=0.001 replicate_start=1 replicates=1 timing_repeats=1' \
  "${interleaved}"
