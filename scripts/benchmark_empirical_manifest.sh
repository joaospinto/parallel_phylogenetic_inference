#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ge 3 && "$1" == --interleave ]]; then
  manifest_argument="$2"
  shift 2
  method_specs=("$@")
elif [[ $# -eq 2 ]]; then
  method_specs=("$1")
  manifest_argument="$2"
else
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda} manifest.csv" >&2
  echo "   or: $0 --interleave manifest.csv METHOD_SPEC..." >&2
  echo "METHOD_SPEC is a native method, beagle-cuda, or beagle-cpu:THREADS" >&2
  exit 2
fi
manifest="$(cd "$(dirname "${manifest_argument}")" && pwd)/$(
  basename "${manifest_argument}")"
root="$(dirname "${manifest}")"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="${repository}/scripts"
if [[ "${method}" == beagle-* ]]; then
  # A parent macOS shell cannot reliably pass DYLD_LIBRARY_PATH through a new
  # /bin/bash process. Configure the runtime in the process that launches the
  # benchmark children.
  # shellcheck source=scripts/beagle_environment.sh
  source "${script_directory}/beagle_environment.sh"
  parallel_phylogenetics_configure_beagle
fi
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${script_directory}/capacity_bounded.sh"

precision="${PRECISION:-fp32}"
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
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
[[ "${conditioning_ms}" =~ ^[0-9]+$ ]] || {
  echo "TREE_HMM_BENCHMARK_CONDITIONING_MS must be nonnegative" >&2
  exit 2
}
previous_site_batch=0
for value in "${requested_site_batches[@]}"; do
  if [[ "${value}" -le "${previous_site_batch}" ]]; then
    echo "TREE_HMM_EMPIRICAL_SITE_BATCHES must be strictly increasing" >&2
    exit 2
  fi
  previous_site_batch="${value}"
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

methods=()
method_threads=()
for specification in "${method_specs[@]}"; do
  case "${specification}" in
    cuda|rocm|metal)
      parsed_method="${specification}"
      parsed_threads=""
      ;;
    beagle-cuda)
      parsed_method=beagle-cuda
      parsed_threads=1
      ;;
    beagle-cpu)
      parsed_method=beagle-cpu
      parsed_threads="${threads}"
      ;;
    beagle-cpu:*)
      parsed_method=beagle-cpu
      parsed_threads="${specification#beagle-cpu:}"
      [[ "${parsed_threads}" =~ ^[1-9][0-9]*$ ]] || {
        echo "invalid BEAGLE CPU method specification ${specification}" >&2
        exit 2
      }
      ;;
    *) echo "unsupported method specification ${specification}" >&2; exit 2 ;;
  esac
  for index in "${!methods[@]}"; do
    if [[ "${methods[index]}" == "${parsed_method}" &&
          "${method_threads[index]}" == "${parsed_threads}" ]]; then
      echo "duplicate method specification ${specification}" >&2
      exit 2
    fi
  done
  methods+=("${parsed_method}")
  method_threads+=("${parsed_threads}")
done

manifest_sha256="$(shasum -a 256 "${manifest}" | awk '{print $1}')"
study_label="empirical-manifest-${manifest_sha256}-minbrlen-${minimum_branch_length}"

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

if [[ "${#methods[@]}" -eq 1 ]]; then
  method_order_policy=single-method
else
  method_order_policy="cyclic rotation by empirical dataset/site-batch case"
fi

emit_protocol() {
  local method_index="$1"
  local protocol_method="${methods[method_index]}"
  local protocol_threads="${method_threads[method_index]}"
  echo "# empirical_manifest=$(basename "${manifest}")"
  echo "# empirical_manifest_sha256=${manifest_sha256}"
  echo "# study=${study_label}"
  if [[ -f "${metadata}" ]]; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "# ${line}"
    done < "${metadata}"
  else
    echo "# corpus_metadata=not supplied; provenance must accompany the manifest"
  fi
  echo "# pattern_compression=exact duplicate columns, prepared before benchmarking"
  echo "# benchmark_mode=${benchmark_mode}"
  echo "# conditioning_ms=${conditioning_ms}"
  echo "# minimum_branch_length=${minimum_branch_length}"
  echo "# branch_length_policy=max(source length, minimum_branch_length)"
  echo "# requested_site_batches=${requested_site_batches[*]}"
  echo "# planned_cases=${total_cases}"
  echo "# capacity_policy=ascending site batches in isolated, host-memory-bounded children"
  echo "# host_memory_guard_kib=${host_memory_guard_kib}"
  echo "# method_order_policy=${method_order_policy}"
  echo "# declared_method_sequence=${method_specs[*]}"
  echo "# progress method=${protocol_method} precision=${precision_label}" \
    "study=${study_label} benchmark_mode=${benchmark_mode}" \
    "requested_site_batches=${requested_site_batches[*]}" \
    "planned_cases=${total_cases}" \
    "threads=${protocol_threads:-none}"
}

completed_cases=()
protocol_emitted=()
for index in "${!methods[@]}"; do
  completed_cases[index]=0
  protocol_emitted[index]=0
done
case_index=0
cd "${repository}"
while IFS=$'\t' read -r dataset taxa raw_sites unique_patterns alignment \
  pattern_weights tree; do
  method_exhausted=()
  for index in "${!methods[@]}"; do
    method_exhausted[index]=0
  done
  while IFS= read -r site_batch; do
    for ((order_index = 0; order_index < ${#methods[@]}; ++order_index)); do
      method_index=$(((case_index + order_index) % ${#methods[@]}))
      [[ "${method_exhausted[method_index]}" == 0 ]] || continue
      method="${methods[method_index]}"
      resume_threads="${method_threads[method_index]}"
      completed_cases[method_index]=$((completed_cases[method_index] + 1))
      if [[ "${protocol_emitted[method_index]}" == 0 ]]; then
        emit_protocol "${method_index}"
        protocol_emitted[method_index]=1
      fi
      echo "# progress case=${completed_cases[method_index]}/${total_cases}" \
        "method=${method} precision=${precision_label} dataset=${dataset}" \
        "site_batch=${site_batch} threads=${resume_threads:-none}"
      if benchmark_resume_capacity_reached "${resume_report}" "${method}" \
        "${precision_label}" "${site_batch}" "${dataset}" \
        "${benchmark_mode}" "${resume_threads:-none}" "${study_label}"; then
        echo "# resume_capacity_limit method=${method}" \
          "precision=${precision_label} dataset=${dataset}" \
          "site_batch=${site_batch} threads=${resume_threads:-none}"
        method_exhausted[method_index]=1
        continue
      fi
      if benchmark_resume_dataset_batch_completed "${resume_report}" \
        "${method}" "${precision_label}" "${dataset}" "${site_batch}" \
        "${benchmark_mode}" "${resume_threads}" "${study_label}"; then
        echo "# resume_skip method=${method} precision=${precision_label}" \
          "dataset=${dataset} site_batch=${site_batch}" \
          "threads=${resume_threads:-none}"
        continue
      fi
      echo "# benchmark_start method=${method} precision=${precision_label}" \
        "dataset=${dataset} taxa=${taxa} raw_sites=${raw_sites}" \
        "unique_patterns=${unique_patterns} site_batch=${site_batch}" \
        "threads=${resume_threads:-none}"
      if [[ "${dry_run}" == 1 ]]; then
        continue
      fi
      case "${method}" in
        cuda|rocm|metal)
          command=("bazel-bin/${method}_benchmark")
          ;;
        beagle-cpu)
          command=(bazel-bin/beagle_benchmark --beagle-resource cpu
                   --beagle-threads "${resume_threads}")
          ;;
        beagle-cuda)
          command=(bazel-bin/beagle_benchmark --beagle-resource cuda
                   --beagle-threads 1)
          ;;
      esac
      benchmark_capacity_exhausted=0
      benchmark_capacity_dataset="${dataset}"
      benchmark_capacity_study="${study_label}"
      benchmark_capacity_mode="${benchmark_mode}"
      benchmark_capacity_threads="${resume_threads:-none}"
      benchmark_run_capacity_bounded "${work_directory}" \
        "${host_memory_guard_kib}" "${method}" "${precision_label}" \
        "${site_batch}" "${command[@]}" \
        --newick "${root}/${tree}" --fasta "${root}/${alignment}" \
        --pattern-weights "${root}/${pattern_weights}" \
        --dataset-label "${dataset}" --site-batch "${site_batch}" \
        --minimum-branch-length "${minimum_branch_length}" \
        --conditioning-ms "${conditioning_ms}" \
        --repeats "${repeats}" --benchmark-mode "${benchmark_mode}" \
        --study-label "${study_label}"
      if [[ "${benchmark_capacity_exhausted}" == 1 ]]; then
        method_exhausted[method_index]=1
      fi
    done
    case_index=$((case_index + 1))
  done < <(site_batches_for "${unique_patterns}")
done < "${rows_file}"
