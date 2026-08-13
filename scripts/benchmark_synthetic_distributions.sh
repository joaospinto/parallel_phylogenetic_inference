#!/usr/bin/env bash
set -euo pipefail

interleave=0
if [[ $# -ge 2 && "$1" == --interleave ]]; then
  interleave=1
  shift
  method_specs=("$@")
elif [[ $# -eq 1 && "$1" != --interleave ]]; then
  method_specs=("$1")
else
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda}" >&2
  echo "   or: $0 --interleave METHOD_SPEC..." >&2
  echo "METHOD_SPEC is a native method, beagle-cuda, or beagle-cpu:THREADS" >&2
  exit 2
fi
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="${repository}/scripts"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${script_directory}/capacity_bounded.sh"
# shellcheck source=scripts/benchmark_method_set.sh
source "${script_directory}/benchmark_method_set.sh"

precision="${PRECISION:-fp32}"
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
read -r -a topologies <<< "${TREE_HMM_DISTRIBUTION_TOPOLOGIES:-yule beta-critical uniform caterpillar}"
read -r -a leaf_counts <<< "${TREE_HMM_DISTRIBUTION_LEAVES:-128 512 2048 8192}"
read -r -a pattern_counts <<< "${TREE_HMM_DISTRIBUTION_PATTERNS:-16 64 256 1024}"
replicates="${TREE_HMM_DISTRIBUTION_REPLICATES:-30}"
seed="${TREE_HMM_DISTRIBUTION_SEED:-20260813}"
repeats="${TREE_HMM_DISTRIBUTION_TIMING_REPEATS:-5}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
dry_run="${TREE_HMM_DRY_RUN:-0}"
capacity_work_directory="${TREE_HMM_CAPACITY_WORK_DIR:-${TMPDIR:-/tmp}}"
host_memory_guard_percent="${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"

for value in "${replicates}" "${repeats}" "${threads}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "replicate, timing-repeat, and thread counts must be positive" >&2
    exit 2
  }
done
[[ "${seed}" =~ ^[0-9]+$ && "${conditioning_ms}" =~ ^[0-9]+$ ]] || {
  echo "seed and conditioning duration must be nonnegative integers" >&2
  exit 2
}
[[ "${precision}" == fp32 || "${precision}" == fp64 ]] || {
  echo "PRECISION must be fp32 or fp64" >&2
  exit 2
}
case "${benchmark_mode}" in
  fixed-model|factor-update|full-input-update) ;;
  *)
    echo "TREE_HMM_BENCHMARK_MODE must be fixed-model, factor-update, or full-input-update" >&2
    exit 2
    ;;
esac
if [[ -n "${resume_report}" && ! -r "${resume_report}" ]]; then
  echo "TREE_HMM_RESUME_REPORT is not readable: ${resume_report}" >&2
  exit 2
fi
[[ "${dry_run}" == 0 || "${dry_run}" == 1 ]] || {
  echo "TREE_HMM_DRY_RUN must be 0 or 1" >&2
  exit 2
}
benchmark_initialize_method_set "${script_directory}" "${threads}" \
  "${dry_run}" "${method_specs[@]}"
for method in "${benchmark_methods[@]}"; do
  if [[ "${method}" == metal && "${precision}" != fp32 ]]; then
    echo "Metal supports only PRECISION=fp32" >&2
    exit 2
  fi
done
if [[ "${dry_run}" == 0 ]]; then
  [[ -d "${capacity_work_directory}" && -w "${capacity_work_directory}" ]] || {
    echo "TREE_HMM_CAPACITY_WORK_DIR must be a writable directory" >&2
    exit 2
  }
fi
host_memory_guard_kib="${TREE_HMM_HOST_MEMORY_GUARD_KIB:-}"
if [[ -z "${host_memory_guard_kib}" ]]; then
  if [[ "${dry_run}" == 0 ]]; then
    host_memory_guard_kib="$(benchmark_host_memory_guard_kib \
      "${host_memory_guard_percent}")"
  else
    host_memory_guard_kib=0
  fi
fi
[[ "${host_memory_guard_kib}" =~ ^[0-9]+$ ]] || {
  echo "TREE_HMM_HOST_MEMORY_GUARD_KIB must be a nonnegative integer" >&2
  exit 2
}

echo "# study=independent-taxa-pattern-grid"
echo "# topology_distributions=${topologies[*]}"
echo "# leaf_counts=${leaf_counts[*]}"
echo "# unique_pattern_counts=${pattern_counts[*]}"
echo "# topology_replicates=${replicates}"
echo "# deterministic_seed_base=${seed}"
echo "# synthetic_patterns=distinct deterministic nucleotide patterns; not JC69 simulations"
echo "# selection_rule=complete Cartesian product; no cell selected from timings"
echo "# benchmark_mode=${benchmark_mode}"
total_cases=$((${#topologies[@]} * ${#leaf_counts[@]} * \
  ${#pattern_counts[@]} * replicates))
echo "# planned_cases=${total_cases}"
if [[ "${#benchmark_methods[@]}" -eq 1 ]]; then
  method_order_policy=single-method
else
  method_order_policy="cyclic rotation by synthetic replicate case"
fi
echo "# method_order_policy=${method_order_policy}"
echo "# declared_method_sequence=${benchmark_method_specs[*]}"
echo "# planned_method_case_runs=$((total_cases * ${#benchmark_methods[@]}))"
if [[ "${interleave}" == 1 ]]; then
  benchmark_emit_interleaved_protocol independent-taxa-pattern-grid \
    "${precision_label}" "${benchmark_mode}" "${total_cases}"
fi

run_grid_batch() {
  local method_index="$1"
  local topology="$2"
  local leaves="$3"
  local patterns="$4"
  local replicate_start="$5"
  local run_count="$6"
  benchmark_select_method "${method_index}" "${benchmark_mode}"
  echo "# benchmark_start_grid_cell method=${benchmark_method}" \
    "precision=${precision_label} benchmark_mode=${benchmark_mode}" \
    "threads=${benchmark_display_threads} topology=${topology}" \
    "leaves=${leaves} unique_patterns=${patterns}" \
    "replicate_start=${replicate_start} replicates=${run_count}"
  if [[ "${dry_run}" == 1 ]]; then
    benchmark_capacity_exhausted=0
    return
  fi
  benchmark_capacity_exhausted=0
  benchmark_capacity_dataset="synthetic-grid-${topology}-${leaves}-${patterns}"
  benchmark_capacity_study=independent-taxa-pattern-grid
  benchmark_capacity_mode="${benchmark_mode}"
  benchmark_capacity_threads="${benchmark_display_threads}"
  benchmark_run_capacity_bounded "${capacity_work_directory}" \
    "${host_memory_guard_kib}" "${benchmark_method}" "${precision_label}" \
    "${patterns}" "${benchmark_command[@]}" \
    --topology "${topology}" --leaves "${leaves}" --sites "${patterns}" \
    --seed "${seed}" --replicate-start "${replicate_start}" \
    --replicates "${run_count}" --repeats "${repeats}" \
    --conditioning-ms "${conditioning_ms}" \
    --study-label independent-taxa-pattern-grid
}

cd "${repository}"
if [[ "${interleave}" == 0 ]]; then
  completed_cases=0
  benchmark_select_method 0 "${benchmark_mode}"
  if [[ "${benchmark_method}" == beagle-* ]]; then
    echo "# beagle_threads=${benchmark_resume_threads}"
  fi
  for topology in "${topologies[@]}"; do
    for leaves in "${leaf_counts[@]}"; do
      for patterns in "${pattern_counts[@]}"; do
        replicate=0
        while [[ "${replicate}" -lt "${replicates}" ]]; do
          if benchmark_resume_synthetic_replicate_completed \
            "${resume_report}" "${benchmark_method}" "${precision_label}" \
            "${topology}" "${leaves}" "${patterns}" "${seed}" \
            "${replicate}" "${benchmark_mode}" \
            "${benchmark_resume_threads}" independent-taxa-pattern-grid; then
            completed_cases=$((completed_cases + 1))
            echo "# progress case=${completed_cases}/${total_cases}" \
              "status=resume-skip method=${benchmark_method}" \
              "precision=${precision_label} topology=${topology}" \
              "leaves=${leaves} unique_patterns=${patterns}" \
              "replicate=${replicate}"
            replicate=$((replicate + 1))
            continue
          fi
          first_missing="${replicate}"
          replicate=$((replicate + 1))
          while [[ "${replicate}" -lt "${replicates}" ]] &&
            ! benchmark_resume_synthetic_replicate_completed \
              "${resume_report}" "${benchmark_method}" "${precision_label}" \
              "${topology}" "${leaves}" "${patterns}" "${seed}" \
              "${replicate}" "${benchmark_mode}" \
              "${benchmark_resume_threads}" independent-taxa-pattern-grid; do
            replicate=$((replicate + 1))
          done
          run_count=$((replicate - first_missing))
          completed_cases=$((completed_cases + run_count))
          echo "# progress cases_through=${completed_cases}/${total_cases}" \
            "status=benchmark method=${benchmark_method}" \
            "precision=${precision_label} topology=${topology}" \
            "leaves=${leaves} unique_patterns=${patterns}" \
            "replicate_start=${first_missing} replicates=${run_count}"
          run_grid_batch 0 "${topology}" "${leaves}" "${patterns}" \
            "${first_missing}" "${run_count}"
          if [[ "${benchmark_capacity_exhausted}" == 1 ]]; then
            echo "# synthetic_grid_cases_unavailable method=${benchmark_method}" \
              "precision=${precision_label} topology=${topology}" \
              "leaves=${leaves} unique_patterns=${patterns}" \
              "replicate_start=${first_missing} replicates=${run_count}"
          fi
        done
      done
    done
  done
else
  completed_cases=()
  for method_index in "${!benchmark_methods[@]}"; do
    completed_cases[method_index]=0
  done
  case_index=0
  for topology in "${topologies[@]}"; do
    for leaves in "${leaf_counts[@]}"; do
      for patterns in "${pattern_counts[@]}"; do
        method_exhausted=()
        for method_index in "${!benchmark_methods[@]}"; do
          method_exhausted[method_index]=0
        done
        for ((replicate = 0; replicate < replicates; ++replicate)); do
          for ((order_index = 0; order_index < ${#benchmark_methods[@]}; ++order_index)); do
            method_index=$(((case_index + order_index) % ${#benchmark_methods[@]}))
            benchmark_select_method "${method_index}" "${benchmark_mode}"
            benchmark_emit_interleaved_case independent-taxa-pattern-grid \
              "${precision_label}" "${benchmark_mode}" "${case_index}" \
              "${order_index}" "${method_index}" \
              "topology=${topology}" "leaves=${leaves}" \
              "unique_patterns=${patterns}" "replicate=${replicate}"
            completed_cases[method_index]=$((completed_cases[method_index] + 1))
            if [[ "${method_exhausted[method_index]}" == 1 ]]; then
              echo "# progress case=${completed_cases[method_index]}/${total_cases}" \
                "status=capacity-skip method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} unique_patterns=${patterns}" \
                "replicate=${replicate} threads=${benchmark_display_threads}"
              continue
            fi
            capacity_dataset="synthetic-grid-${topology}-${leaves}-${patterns}"
            if benchmark_resume_capacity_reached "${resume_report}" \
              "${benchmark_method}" "${precision_label}" "${patterns}" \
              "${capacity_dataset}" "${benchmark_mode}" \
              "${benchmark_display_threads}" independent-taxa-pattern-grid; then
              method_exhausted[method_index]=1
              echo "# progress case=${completed_cases[method_index]}/${total_cases}" \
                "status=resume-capacity-limit method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} unique_patterns=${patterns}" \
                "replicate=${replicate} threads=${benchmark_display_threads}"
              continue
            fi
            if benchmark_resume_synthetic_replicate_completed \
              "${resume_report}" "${benchmark_method}" "${precision_label}" \
              "${topology}" "${leaves}" "${patterns}" "${seed}" \
              "${replicate}" "${benchmark_mode}" \
              "${benchmark_resume_threads}" independent-taxa-pattern-grid; then
              echo "# progress case=${completed_cases[method_index]}/${total_cases}" \
                "status=resume-skip method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} unique_patterns=${patterns}" \
                "replicate=${replicate} threads=${benchmark_display_threads}"
              continue
            fi
            echo "# progress case=${completed_cases[method_index]}/${total_cases}" \
              "status=benchmark method=${benchmark_method}" \
              "precision=${precision_label} topology=${topology}" \
              "leaves=${leaves} unique_patterns=${patterns}" \
              "replicate=${replicate} threads=${benchmark_display_threads}"
            run_grid_batch "${method_index}" "${topology}" "${leaves}" \
              "${patterns}" "${replicate}" 1
            if [[ "${benchmark_capacity_exhausted}" == 1 ]]; then
              method_exhausted[method_index]=1
              echo "# synthetic_grid_case_unavailable method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} unique_patterns=${patterns}" \
                "replicate=${replicate}"
            fi
          done
          case_index=$((case_index + 1))
        done
      done
    done
  done
fi
