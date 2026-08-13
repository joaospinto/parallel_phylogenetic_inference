#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  root="${TEST_SRCDIR}/${TEST_WORKSPACE}"
else
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

work_directory="$(mktemp -d "${TMPDIR:-/tmp}/empirical-report-test.XXXXXX")"
trap 'rm -rf "${work_directory}"' EXIT
report="${work_directory}/report.txt"
study=empirical-manifest-fixture-minbrlen-0.000001

cat > "${report}" <<EOF
# empirical_manifest=manifest.csv
# study=${study}
# benchmark_mode=full-input-update
# requested_site_batches=64 256
# planned_cases=7
# progress case=1/7 method=cuda precision=FP32 dataset=problem-a site_batch=64
backend,precision,benchmark_mode,study,dataset,topology,minimum_branch_length,floored_branch_count,leaves,nodes,sites,unique_patterns,site_batch,cpu_ms,measured_total_ms,max_abs_error,max_relative_error
cuda,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,64,8,2,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,100,8,1,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,64,12,5,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,256,12,2,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,300,12,3,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,64,30,7,0.001,0.0001
cuda,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,100,30,6,0.001,0.0001
# empirical_manifest=manifest.csv
# study=${study}
# benchmark_mode=full-input-update
# requested_site_batches=64 256
# planned_cases=7
# progress case=1/7 method=beagle-cpu precision=FP32 dataset=problem-a site_batch=64
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,minimum_branch_length,floored_branch_count,leaves,nodes,sites,unique_patterns,site_batch,threads,sequential_ms,beagle_total_ms,max_abs_error,max_relative_error
beagle,cpu,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,64,1,8,2,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,100,1,8,3,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,64,1,12,10,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,256,1,12,5,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,300,1,12,6,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,64,1,30,12,0.001,0.0001
beagle,cpu,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,100,1,30,10,0.001,0.0001
# empirical_manifest=manifest.csv
# study=${study}
# benchmark_mode=full-input-update
# requested_site_batches=64 256
# planned_cases=7
# progress case=1/7 method=beagle-cuda precision=FP32 dataset=problem-a site_batch=64
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,minimum_branch_length,floored_branch_count,leaves,nodes,sites,unique_patterns,site_batch,threads,sequential_ms,beagle_total_ms,max_abs_error,max_relative_error
beagle,cuda,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,64,1,8,1.5,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-a,empirical,0.000001,0,100,199,1000,100,100,1,8,2,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,64,1,12,8,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,256,1,12,4,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-b,empirical,0.000001,0,500,999,2000,300,300,1,12,5,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,64,1,30,9,0.001,0.0001
beagle,cuda,FP32,full-input-update,${study},problem-c,empirical,0.000001,0,5000,9999,4000,100,100,1,30,8,0.001,0.0001
EOF

run_report() {
  local input="$1"
  local native="$2"
  local baseline="$3"
  local output="$4"
  shift 4
  python3 "${root}/scripts/plot_empirical_corpus.py" "${input}" \
    --native "${native}" --baseline "${baseline}" \
    --precision FP32 --benchmark-mode full-input-update \
    --study "${study}" --max-normalized-error 0.001 \
    --run-identity fixture-machine --output-directory "${output}" \
    --validate-only "$@"
}

cpu_output="${work_directory}/cpu"
run_report "${report}" cuda beagle-cpu "${cpu_output}" --beagle-threads 1
paired="$(find "${cpu_output}" -name 'empirical_paired_cases_*.csv')"
[[ "$(wc -l < "${paired}")" == 4 ]]
grep -Fq 'problem-a,100,199,1000,100,19900,cuda,beagle_cpu_1t,64;100,64;100,100,64' \
  "${paired}"
grep -Fq 'problem-b,500,999,2000,300,299700,cuda,beagle_cpu_1t,64;256;300,64;256;300,256,256' \
  "${paired}"
grep -Fq 'batch_selection=minimum median complete-alignment wall time' \
  "$(find "${cpu_output}" -name 'empirical_protocol_*.txt')"
grep -Fq 'declared_cases_per_method=7' \
  "$(find "${cpu_output}" -name 'empirical_protocol_*.txt')"

cuda_output="${work_directory}/beagle-cuda"
run_report "${report}" cuda beagle-cuda "${cuda_output}"
grep -Fq ',beagle_cuda,' \
  "$(find "${cuda_output}" -name 'empirical_paired_cases_*.csv')"

for backend in metal rocm; do
  backend_report="${work_directory}/${backend}.txt"
  sed -e "s/method=cuda/method=${backend}/" \
      -e "s/^cuda,/${backend},/" "${report}" > "${backend_report}"
  run_report "${backend_report}" "${backend}" beagle-cpu \
    "${work_directory}/${backend}" --beagle-threads 1
done

pandit_study=pandit-17.0-minbrlen-0.000001
pandit_report="${work_directory}/pandit.txt"
cat > "${pandit_report}" <<EOF
# corpus=PANDIT-17.0
# corpus_backend=cuda
# benchmark_mode=full-input-update
# study=${pandit_study}
# benchmark_start method=cuda precision=FP32 dataset=PF00001 leaves=100 sites=100
backend,precision,benchmark_mode,study,dataset,topology,minimum_branch_length,floored_branch_count,leaves,nodes,sites,unique_patterns,site_batch,cpu_ms,measured_total_ms,max_abs_error,max_relative_error
cuda,FP32,full-input-update,${pandit_study},PF00001,empirical,0.000001,0,100,199,100,100,100,8,1,0.001,0.0001
cuda,FP32,full-input-update,${pandit_study},PF00002,empirical,0.000001,0,500,999,300,300,300,12,2,0.001,0.0001
cuda,FP32,full-input-update,${pandit_study},PF00003,empirical,0.000001,0,5000,9999,100,100,100,30,6,0.001,0.0001
# selected_families=3
# corpus=PANDIT-17.0
# corpus_backend=beagle-cpu
# benchmark_mode=full-input-update
# study=${pandit_study}
# benchmark_start method=beagle-cpu precision=FP32 dataset=PF00001 leaves=100 sites=100
baseline,beagle_resource,precision,benchmark_mode,study,dataset,topology,minimum_branch_length,floored_branch_count,leaves,nodes,sites,unique_patterns,site_batch,threads,sequential_ms,beagle_total_ms,max_abs_error,max_relative_error
beagle,cpu,FP32,full-input-update,${pandit_study},PF00001,empirical,0.000001,0,100,199,100,100,100,1,8,2,0.001,0.0001
beagle,cpu,FP32,full-input-update,${pandit_study},PF00002,empirical,0.000001,0,500,999,300,300,300,1,12,5,0.001,0.0001
beagle,cpu,FP32,full-input-update,${pandit_study},PF00003,empirical,0.000001,0,5000,9999,100,100,100,1,30,10,0.001,0.0001
# selected_families=3
EOF
python3 "${root}/scripts/plot_empirical_corpus.py" "${pandit_report}" \
  --native cuda --baseline beagle-cpu --beagle-threads 1 \
  --precision FP32 --benchmark-mode full-input-update \
  --study "${pandit_study}" --max-normalized-error 0.001 \
  --run-identity fixture-machine \
  --output-directory "${work_directory}/pandit-output" --validate-only
[[ "$(wc -l < "$(find "${work_directory}/pandit-output" \
  -name 'empirical_paired_cases_*.csv')")" == 4 ]]

missing_batch="${work_directory}/missing-batch.txt"
sed '/^cuda,.*problem-b,.*300,256,/d' "${report}" > "${missing_batch}"
if run_report "${missing_batch}" cuda beagle-cpu \
     "${work_directory}/missing-output" --beagle-threads 1 >/dev/null 2>&1; then
  echo "empirical report accepted a missing feasible batch" >&2
  exit 1
fi

capacity_report="${work_directory}/capacity.txt"
sed -e '/^cuda,.*problem-b,.*300,256,/d' \
    -e '/^cuda,.*problem-b,.*300,300,/d' "${report}" > "${capacity_report}"
printf '%s\n' \
  "# capacity_limit method=cuda precision=FP32 dataset=problem-b study=${study} benchmark_mode=full-input-update threads=none first_infeasible_site_batch=256 reason=allocation-failure" \
  >> "${capacity_report}"
run_report "${capacity_report}" cuda beagle-cpu \
  "${work_directory}/capacity-output" --beagle-threads 1
grep -Fq 'problem-b,500,999,2000,300,299700,cuda,beagle_cpu_1t,64,64;256;300,64,256' \
  "$(find "${work_directory}/capacity-output" -name 'empirical_paired_cases_*.csv')"

incomplete="${work_directory}/incomplete.txt"
sed '/^beagle,cpu,.*problem-c,/d' "${report}" > "${incomplete}"
if run_report "${incomplete}" cuda beagle-cpu \
     "${work_directory}/incomplete-output" --beagle-threads 1 >/dev/null 2>&1; then
  echo "empirical report accepted incomplete baseline coverage" >&2
  exit 1
fi

bad_error="${work_directory}/bad-error.txt"
sed '/^cuda,.*problem-a,/s/0.001,0.0001$/0.1,0.0001/' \
  "${report}" > "${bad_error}"
# Absolute error is descriptive unless the optional extra guard is requested.
run_report "${bad_error}" cuda beagle-cpu \
  "${work_directory}/descriptive-absolute-output" --beagle-threads 1
if run_report "${bad_error}" cuda beagle-cpu \
     "${work_directory}/bad-error-output" --beagle-threads 1 \
     --max-abs-error 0.01 >/dev/null 2>&1; then
  echo "empirical report accepted an optional absolute-error guard failure" >&2
  exit 1
fi

bad_normalized_error="${work_directory}/bad-normalized-error.txt"
sed '/^cuda,.*problem-a,/s/0.001,0.0001$/0.001,0.01/' \
  "${report}" > "${bad_normalized_error}"
if run_report "${bad_normalized_error}" cuda beagle-cpu \
     "${work_directory}/bad-normalized-output" --beagle-threads 1 \
     >/dev/null 2>&1; then
  echo "empirical report accepted a normalized-error threshold failure" >&2
  exit 1
fi

identity_a="${work_directory}/identity-a.txt"
identity_b="${work_directory}/identity-b.txt"
{
  printf '%s\n' '# cache_identity sha256=identity-a'
  cat "${report}"
} > "${identity_a}"
{
  printf '%s\n' '# cache_identity sha256=identity-b'
  cat "${report}"
} > "${identity_b}"
if PYTHONPATH="${root}/scripts" python3 - "${identity_a}" "${identity_b}" <<'PY'
import sys
from pathlib import Path

from summarize_benchmarks import validate_run_identity

try:
    validate_run_identity([Path(value) for value in sys.argv[1:]], "override")
except ValueError:
    raise SystemExit(1)
PY
then
  echo "run-identity override masked conflicting embedded identities" >&2
  exit 1
fi
