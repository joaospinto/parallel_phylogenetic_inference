#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda} manifest.csv" >&2
  exit 2
fi
method="$1"
manifest="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
root="$(dirname "${manifest}")"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="${repository}/scripts"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${script_directory}/capacity_bounded.sh"

precision="${PRECISION:-fp32}"
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
minimum_branch_length="${TREE_HMM_EMPIRICAL_MINIMUM_BRANCH_LENGTH:-0.000001}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
metadata="${root}/corpus_metadata.txt"
read -r -a requested_site_batches <<< \
  "${TREE_HMM_EMPIRICAL_SITE_BATCHES:-256 1024 4096 8192 16384 32768}"
host_memory_guard_percent="${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"
dry_run="${TREE_HMM_DRY_RUN:-0}"

for value in "${repeats}" "${threads}" "${requested_site_batches[@]}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "repeat, thread, and site-batch counts must be positive integers" >&2
    exit 2
  }
done
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "TREE_HMM_DRY_RUN must be 0 or 1" >&2
  exit 2
}
awk -v value="${minimum_branch_length}" \
  'BEGIN { exit !(value + 0 >= 0) }' || {
  echo "TREE_HMM_EMPIRICAL_MINIMUM_BRANCH_LENGTH must be nonnegative" >&2
  exit 2
}
host_memory_guard_kib="${TREE_HMM_HOST_MEMORY_GUARD_KIB:-}"
if [[ -z "${host_memory_guard_kib}" ]]; then
  host_memory_guard_kib="$(benchmark_host_memory_guard_kib \
    "${host_memory_guard_percent}")"
fi
[[ "${host_memory_guard_kib}" =~ ^[1-9][0-9]*$ ]] || {
  echo "TREE_HMM_HOST_MEMORY_GUARD_KIB must be a positive integer" >&2
  exit 2
}
work_directory="$(mktemp -d "${TMPDIR:-/tmp}/empirical-benchmark.XXXXXX")"
rows_file="${work_directory}/manifest-rows.tsv"
trap 'rm -rf "${work_directory}"' EXIT
python3 "${script_directory}/empirical_manifest_rows.py" \
  "${manifest}" > "${rows_file}"

case "${method}" in
  cuda|rocm|metal)
    target="${method}_benchmark"
    extra=()
    resume_threads=""
    ;;
  beagle-cpu)
    target=beagle_benchmark
    extra=(--beagle-resource cpu --beagle-threads "${threads}")
    resume_threads="${threads}"
    ;;
  beagle-cuda)
    target=beagle_benchmark
    extra=(--beagle-resource cuda --beagle-threads 1)
    resume_threads=1
    ;;
  *) echo "unsupported method ${method}" >&2; exit 2 ;;
esac

echo "# empirical_manifest=$(basename "${manifest}")"
echo "# empirical_manifest_sha256=$(shasum -a 256 "${manifest}" | awk '{print $1}')"
if [[ -f "${metadata}" ]]; then
  while IFS= read -r line; do
    [[ -n "${line}" ]] && echo "# ${line}"
  done < "${metadata}"
else
  echo "# corpus_metadata=not supplied; provenance must accompany the manifest"
fi
echo "# pattern_compression=exact duplicate columns, prepared before benchmarking"
echo "# benchmark_mode=${benchmark_mode}"
echo "# minimum_branch_length=${minimum_branch_length}"
echo "# branch_length_policy=max(source length, minimum_branch_length)"
echo "# requested_site_batches=${requested_site_batches[*]}"
echo "# capacity_policy=ascending site batches in isolated, host-memory-bounded children"
echo "# host_memory_guard_kib=${host_memory_guard_kib}"

site_batches_for() {
  local unique_patterns="$1"
  local candidate
  local capped
  local previous=0
  for candidate in "${requested_site_batches[@]}"; do
    capped="${candidate}"
    if [[ "${capped}" -gt "${unique_patterns}" ]]; then
      capped="${unique_patterns}"
    fi
    if [[ "${capped}" -ne "${previous}" ]]; then
      echo "${capped}"
      previous="${capped}"
    fi
    [[ "${capped}" -eq "${unique_patterns}" ]] && break
  done
  if [[ "${previous}" -ne "${unique_patterns}" ]]; then
    echo "${unique_patterns}"
  fi
}

total_cases=0
while IFS=$'\t' read -r dataset taxa raw_sites unique_patterns alignment \
  pattern_weights tree; do
  while IFS= read -r ignored; do
    total_cases=$((total_cases + 1))
  done < <(site_batches_for "${unique_patterns}")
done < "${rows_file}"
echo "# planned_cases=${total_cases}"

completed_cases=0
cd "${repository}"
while IFS=$'\t' read -r dataset taxa raw_sites unique_patterns alignment \
  pattern_weights tree; do
  benchmark_capacity_exhausted=0
  benchmark_capacity_dataset="${dataset}"
  benchmark_capacity_mode="${benchmark_mode}"
  benchmark_capacity_threads="${resume_threads:-none}"
  while IFS= read -r site_batch; do
    completed_cases=$((completed_cases + 1))
    echo "# progress case=${completed_cases}/${total_cases} method=${method}" \
      "precision=${precision_label} dataset=${dataset}" \
      "site_batch=${site_batch}"
    if benchmark_resume_capacity_reached "${resume_report}" "${method}" \
      "${precision_label}" "${site_batch}" "${dataset}" \
      "${benchmark_mode}" "${resume_threads:-none}"; then
      echo "# resume_capacity_limit method=${method}" \
        "precision=${precision_label} dataset=${dataset}" \
        "site_batch=${site_batch}"
      break
    fi
    if benchmark_resume_dataset_batch_completed "${resume_report}" \
      "${method}" "${precision_label}" "${dataset}" "${site_batch}" \
      "${benchmark_mode}" "${resume_threads}"; then
      echo "# resume_skip method=${method} precision=${precision_label}" \
        "dataset=${dataset} site_batch=${site_batch}"
      continue
    fi
    echo "# benchmark_start method=${method} precision=${precision_label}" \
      "dataset=${dataset} taxa=${taxa} raw_sites=${raw_sites}" \
      "unique_patterns=${unique_patterns} site_batch=${site_batch}"
    if [[ "${dry_run}" == 1 ]]; then
      continue
    fi
    benchmark_run_capacity_bounded "${work_directory}" \
      "${host_memory_guard_kib}" "${method}" "${precision_label}" \
      "${site_batch}" "bazel-bin/${target}" "${extra[@]}" \
      --newick "${root}/${tree}" --fasta "${root}/${alignment}" \
      --pattern-weights "${root}/${pattern_weights}" \
      --dataset-label "${dataset}" --site-batch "${site_batch}" \
      --minimum-branch-length "${minimum_branch_length}" \
      --repeats "${repeats}" --benchmark-mode "${benchmark_mode}"
    if [[ "${benchmark_capacity_exhausted}" == 1 ]]; then
      break
    fi
  done < <(site_batches_for "${unique_patterns}")
done < "${rows_file}"
