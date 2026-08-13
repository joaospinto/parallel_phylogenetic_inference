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
profile="${TREE_HMM_JC69_PROFILE:-smoke}"
case "${profile}" in
  smoke)
    default_leaves="128 512"
    default_sites="256 1024"
    default_heights="0.001 0.1"
    default_replicates=3
    default_minimum_timing_repeats=3
    default_maximum_timing_repeats=3
    default_timing_work_budget=0
    ;;
  paper)
    # A prespecified factorial design spanning two orders of magnitude on
    # every numerical axis. Ten independently seeded replicate blocks expose
    # variability without turning timing repeats into pseudoreplication.
    default_leaves="128 1024 8192"
    default_sites="256 2048 8192"
    default_heights="0.0001 0.001 0.01 0.1"
    default_replicates=10
    default_minimum_timing_repeats=1
    default_maximum_timing_repeats=5
    default_timing_work_budget=20000000
    ;;
  stress)
    # The deliberately expensive stress profile retains the earlier outer
    # scale. It is useful for capacity studies, not the primary paper design.
    default_leaves="100 1000 10000"
    default_sites="1000 10000 100000"
    default_heights="0.0001 0.001 0.01 0.1"
    default_replicates=3
    default_minimum_timing_repeats=1
    default_maximum_timing_repeats=3
    default_timing_work_budget=50000000
    ;;
  *)
    echo "TREE_HMM_JC69_PROFILE must be smoke, paper, or stress" >&2
    exit 2
    ;;
esac

read -r -a topologies <<< \
  "${TREE_HMM_JC69_TOPOLOGIES:-yule beta-critical uniform caterpillar}"
read -r -a leaf_counts <<< "${TREE_HMM_JC69_LEAVES:-${default_leaves}}"
read -r -a raw_site_counts <<< "${TREE_HMM_JC69_RAW_SITES:-${default_sites}}"
read -r -a evolutionary_heights <<< \
  "${TREE_HMM_JC69_EVOLUTIONARY_HEIGHTS:-${default_heights}}"
replicates="${TREE_HMM_JC69_REPLICATES:-${default_replicates}}"
seed="${TREE_HMM_JC69_SEED:-20260814}"
minimum_timing_repeats="${TREE_HMM_JC69_MINIMUM_TIMING_REPEATS:-${default_minimum_timing_repeats}}"
maximum_timing_repeats="${TREE_HMM_JC69_MAXIMUM_TIMING_REPEATS:-${default_maximum_timing_repeats}}"
timing_work_budget="${TREE_HMM_JC69_TIMING_WORK_BUDGET:-${default_timing_work_budget}}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
dry_run="${TREE_HMM_DRY_RUN:-0}"
minimum_site_batch="${TREE_HMM_JC69_MINIMUM_SITE_BATCH:-128}"
capacity_work_directory="${TREE_HMM_CAPACITY_WORK_DIR:-${TMPDIR:-/tmp}}"
host_memory_guard_percent="${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"

for value in "${replicates}" "${minimum_timing_repeats}" \
  "${maximum_timing_repeats}" "${threads}" "${minimum_site_batch}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "replicate, timing-repeat, and thread counts must be positive" >&2
    exit 2
  }
done
[[ "${seed}" =~ ^[0-9]+$ && "${conditioning_ms}" =~ ^[0-9]+$ && \
   "${timing_work_budget}" =~ ^[0-9]+$ ]] || {
  echo "seed, conditioning duration, and timing budget must be nonnegative integers" >&2
  exit 2
}
if (( minimum_timing_repeats > maximum_timing_repeats )); then
  echo "minimum timing repeats may not exceed maximum timing repeats" >&2
  exit 2
fi
for value in "${leaf_counts[@]}" "${raw_site_counts[@]}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "taxon and raw-site counts must be positive integers" >&2
    exit 2
  }
done
for value in "${evolutionary_heights[@]}"; do
  awk -v value="${value}" 'BEGIN { exit !(value + 0 > 0) }' || {
    echo "evolutionary root-to-tip distances must be positive" >&2
    exit 2
  }
done
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
[[ -d "${capacity_work_directory}" && -w "${capacity_work_directory}" ]] || {
  echo "TREE_HMM_CAPACITY_WORK_DIR must be a writable directory" >&2
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
  host_memory_guard_kib="$(benchmark_effective_host_memory_guard_kib \
    "${host_memory_guard_percent}")"
else
  host_memory_guard_kib="${TREE_HMM_HOST_MEMORY_GUARD_KIB:-0}"
fi
[[ "${host_memory_guard_kib}" =~ ^[0-9]+$ ]] || {
  echo "TREE_HMM_HOST_MEMORY_GUARD_KIB must be a nonnegative integer" >&2
  exit 2
}

echo "# study=clock-like-jc69-simulation"
echo "# profile=${profile}"
case_count=$(( ${#topologies[@]} * ${#leaf_counts[@]} * \
  ${#raw_site_counts[@]} * ${#evolutionary_heights[@]} * replicates ))
echo "# cases_per_method_and_precision=${case_count}"
echo "# topology_distributions=${topologies[*]}"
echo "# leaf_counts=${leaf_counts[*]}"
echo "# raw_sequence_lengths=${raw_site_counts[*]}"
echo "# evolutionary_root_to_tip_distances=${evolutionary_heights[*]}"
echo "# topology_and_alignment_replicates=${replicates}"
echo "# deterministic_seed_base=${seed}"
echo "# replicate_blocking=for each topology distribution, taxon count, and replicate index, reuse one topology draw across raw sequence lengths and evolutionary distances; draw a deterministic alignment for each cell"
echo "# timing_repeat_rule=maximum_timing_repeats if timing_work_budget_node_sites is zero; otherwise max(minimum_timing_repeats,min(maximum_timing_repeats,floor(timing_work_budget_node_sites/((2*taxa-1)*raw_sequence_length))))"
echo "# minimum_timing_repeats=${minimum_timing_repeats}"
echo "# maximum_timing_repeats=${maximum_timing_repeats}"
echo "# timing_work_budget_node_sites=${timing_work_budget}"
echo "# conditioning_ms=${conditioning_ms}"
echo "# minimum_site_batch=${minimum_site_batch}"
echo "# capacity_policy=try the complete compressed alignment, then halve the site batch after an isolated capacity failure"
echo "# simulation_model=JC69 with stationary uniform root frequencies"
echo "# branch_model=clock-like; all root-to-tip paths have the specified evolutionary distance"
echo "# preprocessing=exact duplicate-pattern compression before timing"
echo "# reported_sizes=raw sequence length and retained unique-pattern count"
echo "# comparison_scope=complete likelihood evaluation, not site-order incremental updates"
echo "# selection_rule=complete prespecified Cartesian product; no case selected from timings"
echo "# benchmark_mode=${benchmark_mode}"
if [[ "${#benchmark_methods[@]}" -eq 1 ]]; then
  method_order_policy=single-method
else
  method_order_policy="cyclic rotation by JC69 replicate case"
fi
echo "# method_order_policy=${method_order_policy}"
echo "# declared_method_sequence=${benchmark_method_specs[*]}"
echo "# planned_method_case_runs=$((case_count * ${#benchmark_methods[@]}))"
if [[ "${interleave}" == 1 ]]; then
  benchmark_emit_interleaved_protocol clock-like-jc69-simulation \
    "${precision_label}" "${benchmark_mode}" "${case_count}"
fi

timing_repeats_for() {
  local leaves="$1"
  local raw_sites="$2"
  local node_sites=$(( (2 * leaves - 1) * raw_sites ))
  local result="${maximum_timing_repeats}"
  if (( timing_work_budget > 0 )); then
    result=$(( timing_work_budget / node_sites ))
    (( result < minimum_timing_repeats )) && \
      result="${minimum_timing_repeats}"
    (( result > maximum_timing_repeats )) && \
      result="${maximum_timing_repeats}"
  fi
  echo "${result}"
}

run_jc69_batch() {
  local method_index="$1"
  local topology="$2"
  local leaves="$3"
  local raw_sites="$4"
  local evolutionary_height="$5"
  local replicate_start="$6"
  local run_count="$7"
  local repeats="$8"
  local candidate_site_batch="$9"
  local capacity_dataset
  benchmark_select_method "${method_index}" "${benchmark_mode}"
  echo "# benchmark_start_jc69 method=${benchmark_method}" \
    "precision=${precision_label} benchmark_mode=${benchmark_mode}" \
    "threads=${benchmark_display_threads} topology=${topology}" \
    "leaves=${leaves} raw_sites=${raw_sites}" \
    "evolutionary_root_to_tip_distance=${evolutionary_height}" \
    "replicate_start=${replicate_start} replicates=${run_count}" \
    "timing_repeats=${repeats}"
  jc69_case_unavailable=0
  jc69_final_site_batch="${candidate_site_batch}"
  capacity_dataset="synthetic-jc69-${topology}-${leaves}-${raw_sites}-${evolutionary_height}"
  while true; do
    benchmark_capacity_exhausted=0
    if benchmark_resume_capacity_reached "${resume_report}" \
      "${benchmark_method}" "${precision_label}" \
      "${candidate_site_batch}" "${capacity_dataset}" \
      "${benchmark_mode}" "${benchmark_display_threads}" \
      clock-like-jc69-simulation; then
      benchmark_capacity_exhausted=1
      echo "# resume_capacity_limit method=${benchmark_method}" \
        "precision=${precision_label} dataset=${capacity_dataset}" \
        "site_batch=${candidate_site_batch}" \
        "threads=${benchmark_display_threads}"
    elif [[ "${dry_run}" == 1 ]]; then
      echo "# dry_run_capacity_candidate method=${benchmark_method}" \
        "precision=${precision_label} dataset=${capacity_dataset}" \
        "site_batch=${candidate_site_batch}"
      jc69_final_site_batch="${candidate_site_batch}"
      return
    else
      benchmark_capacity_dataset="${capacity_dataset}"
      benchmark_capacity_study=clock-like-jc69-simulation
      benchmark_capacity_mode="${benchmark_mode}"
      benchmark_capacity_threads="${benchmark_display_threads}"
      benchmark_run_capacity_bounded "${capacity_work_directory}" \
        "${host_memory_guard_kib}" "${benchmark_method}" \
        "${precision_label}" "${candidate_site_batch}" \
        "${benchmark_command[@]}" \
        --topology "${topology}" --leaves "${leaves}" \
        --sites "${raw_sites}" --site-batch "${candidate_site_batch}" \
        --study-label clock-like-jc69-simulation \
        --synthetic-sequence-model jc69 \
        --evolutionary-root-to-tip-distance "${evolutionary_height}" \
        --compress-patterns true --seed "${seed}" \
        --replicate-start "${replicate_start}" \
        --replicates "${run_count}" --repeats "${repeats}" \
        --conditioning-ms "${conditioning_ms}"
    fi
    if [[ "${benchmark_capacity_exhausted}" == 0 ]]; then
      jc69_final_site_batch="${candidate_site_batch}"
      return
    fi
    if (( candidate_site_batch <= minimum_site_batch )); then
      jc69_case_unavailable=1
      jc69_final_site_batch="${candidate_site_batch}"
      echo "# jc69_case_unavailable method=${benchmark_method}" \
        "precision=${precision_label} topology=${topology}" \
        "leaves=${leaves} raw_sites=${raw_sites}" \
        "evolutionary_root_to_tip_distance=${evolutionary_height}" \
        "replicate_start=${replicate_start} replicates=${run_count}"
      return
    fi
    candidate_site_batch=$((candidate_site_batch / 2))
    (( candidate_site_batch < minimum_site_batch )) && \
      candidate_site_batch="${minimum_site_batch}"
    echo "# jc69_capacity_retry method=${benchmark_method}" \
      "precision=${precision_label} site_batch=${candidate_site_batch}"
  done
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
      for raw_sites in "${raw_site_counts[@]}"; do
        repeats="$(timing_repeats_for "${leaves}" "${raw_sites}")"
        for evolutionary_height in "${evolutionary_heights[@]}"; do
          replicate=0
          while [[ "${replicate}" -lt "${replicates}" ]]; do
            if benchmark_resume_jc69_replicate_completed \
              "${resume_report}" "${benchmark_method}" "${precision_label}" \
              "${topology}" "${leaves}" "${raw_sites}" \
              "${evolutionary_height}" "${seed}" "${replicate}" \
              "${benchmark_mode}" "${benchmark_resume_threads}" \
              clock-like-jc69-simulation "${repeats}" \
              "${conditioning_ms}"; then
              completed_cases=$((completed_cases + 1))
              echo "# progress case=${completed_cases}/${case_count}" \
                "status=resume-skip method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} raw_sites=${raw_sites}" \
                "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                "replicate=${replicate} timing_repeats=${repeats}"
              replicate=$((replicate + 1))
              continue
            fi
            first_missing="${replicate}"
            replicate=$((replicate + 1))
            while [[ "${replicate}" -lt "${replicates}" ]] &&
              ! benchmark_resume_jc69_replicate_completed \
                "${resume_report}" "${benchmark_method}" "${precision_label}" \
                "${topology}" "${leaves}" "${raw_sites}" \
                "${evolutionary_height}" "${seed}" "${replicate}" \
                "${benchmark_mode}" "${benchmark_resume_threads}" \
                clock-like-jc69-simulation "${repeats}" \
                "${conditioning_ms}"; do
              replicate=$((replicate + 1))
            done
            run_count=$((replicate - first_missing))
            completed_cases=$((completed_cases + run_count))
            echo "# progress cases_through=${completed_cases}/${case_count}" \
              "status=benchmark method=${benchmark_method}" \
              "precision=${precision_label} topology=${topology}" \
              "leaves=${leaves} raw_sites=${raw_sites}" \
              "evolutionary_root_to_tip_distance=${evolutionary_height}" \
              "replicate_start=${first_missing} replicates=${run_count}" \
              "timing_repeats=${repeats}"
            run_jc69_batch 0 "${topology}" "${leaves}" "${raw_sites}" \
              "${evolutionary_height}" "${first_missing}" "${run_count}" \
              "${repeats}" "${raw_sites}"
          done
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
      for raw_sites in "${raw_site_counts[@]}"; do
        repeats="$(timing_repeats_for "${leaves}" "${raw_sites}")"
        for evolutionary_height in "${evolutionary_heights[@]}"; do
          method_site_batch=()
          method_unavailable=()
          for method_index in "${!benchmark_methods[@]}"; do
            method_site_batch[method_index]="${raw_sites}"
            method_unavailable[method_index]=0
          done
          for ((replicate = 0; replicate < replicates; ++replicate)); do
            for ((order_index = 0; order_index < ${#benchmark_methods[@]}; ++order_index)); do
              method_index=$(((case_index + order_index) % ${#benchmark_methods[@]}))
              benchmark_select_method "${method_index}" "${benchmark_mode}"
              benchmark_emit_interleaved_case clock-like-jc69-simulation \
                "${precision_label}" "${benchmark_mode}" "${case_index}" \
                "${order_index}" "${method_index}" \
                "topology=${topology}" "leaves=${leaves}" \
                "raw_sites=${raw_sites}" \
                "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                "replicate=${replicate}" "timing_repeats=${repeats}"
              completed_cases[method_index]=$((completed_cases[method_index] + 1))
              if [[ "${method_unavailable[method_index]}" == 1 ]]; then
                echo "# progress case=${completed_cases[method_index]}/${case_count}" \
                  "status=capacity-skip method=${benchmark_method}" \
                  "precision=${precision_label} topology=${topology}" \
                  "leaves=${leaves} raw_sites=${raw_sites}" \
                  "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                  "replicate=${replicate} timing_repeats=${repeats}" \
                  "threads=${benchmark_display_threads}"
                continue
              fi
              if benchmark_resume_jc69_replicate_completed \
                "${resume_report}" "${benchmark_method}" "${precision_label}" \
                "${topology}" "${leaves}" "${raw_sites}" \
                "${evolutionary_height}" "${seed}" "${replicate}" \
                "${benchmark_mode}" "${benchmark_resume_threads}" \
                clock-like-jc69-simulation "${repeats}" \
                "${conditioning_ms}"; then
                echo "# progress case=${completed_cases[method_index]}/${case_count}" \
                  "status=resume-skip method=${benchmark_method}" \
                  "precision=${precision_label} topology=${topology}" \
                  "leaves=${leaves} raw_sites=${raw_sites}" \
                  "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                  "replicate=${replicate} timing_repeats=${repeats}" \
                  "threads=${benchmark_display_threads}"
                continue
              fi
              echo "# progress case=${completed_cases[method_index]}/${case_count}" \
                "status=benchmark method=${benchmark_method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} raw_sites=${raw_sites}" \
                "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                "replicate=${replicate} timing_repeats=${repeats}" \
                "threads=${benchmark_display_threads}"
              run_jc69_batch "${method_index}" "${topology}" "${leaves}" \
                "${raw_sites}" "${evolutionary_height}" "${replicate}" 1 \
                "${repeats}" "${method_site_batch[method_index]}"
              method_site_batch[method_index]="${jc69_final_site_batch}"
              if [[ "${jc69_case_unavailable}" == 1 ]]; then
                method_unavailable[method_index]=1
              fi
            done
            case_index=$((case_index + 1))
          done
        done
      done
    done
  done
fi
