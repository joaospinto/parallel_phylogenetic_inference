#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda}" >&2
  exit 2
fi
method="$1"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/benchmark_resume.sh
source "${repository}/scripts/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${repository}/scripts/capacity_bounded.sh"

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
if [[ "${method}" == metal && "${precision}" != fp32 ]]; then
  echo "Metal supports only PRECISION=fp32" >&2
  exit 2
fi
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
host_memory_guard_kib=0
if [[ "${dry_run}" == 0 ]]; then
  host_memory_guard_kib="$(benchmark_host_memory_guard_kib \
    "${host_memory_guard_percent}")"
fi

case "${method}" in
  cuda|rocm|metal)
    target="${method}_benchmark"
    resource_arguments=(--benchmark-mode "${benchmark_mode}")
    resume_threads=""
    ;;
  beagle-cpu)
    target=beagle_benchmark
    resource_arguments=(--beagle-resource cpu --beagle-threads "${threads}"
                        --benchmark-mode "${benchmark_mode}")
    resume_threads="${threads}"
    ;;
  beagle-cuda)
    target=beagle_benchmark
    threads=1
    resource_arguments=(--beagle-resource cuda --beagle-threads 1
                        --benchmark-mode "${benchmark_mode}")
    resume_threads=1
    ;;
  *)
    echo "unsupported benchmark method ${method}" >&2
    exit 2
    ;;
esac

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
if [[ "${method}" == beagle-* ]]; then
  echo "# beagle_threads=${threads}"
fi
completed_cases=0

cd "${repository}"
for topology in "${topologies[@]}"; do
  for leaves in "${leaf_counts[@]}"; do
    for raw_sites in "${raw_site_counts[@]}"; do
      for evolutionary_height in "${evolutionary_heights[@]}"; do
        replicate=0
        while [[ "${replicate}" -lt "${replicates}" ]]; do
          node_sites=$(( (2 * leaves - 1) * raw_sites ))
          repeats="${maximum_timing_repeats}"
          if (( timing_work_budget > 0 )); then
            repeats=$(( timing_work_budget / node_sites ))
            (( repeats < minimum_timing_repeats )) && \
              repeats="${minimum_timing_repeats}"
            (( repeats > maximum_timing_repeats )) && \
              repeats="${maximum_timing_repeats}"
          fi
          if benchmark_resume_jc69_replicate_completed \
            "${resume_report}" "${method}" "${precision_label}" \
            "${topology}" "${leaves}" "${raw_sites}" \
            "${evolutionary_height}" "${seed}" "${replicate}" \
            "${benchmark_mode}" "${resume_threads}" \
            clock-like-jc69-simulation "${repeats}" \
            "${conditioning_ms}"; then
            completed_cases=$((completed_cases + 1))
            echo "# progress case=${completed_cases}/${case_count}" \
              "status=resume-skip method=${method}" \
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
              "${resume_report}" "${method}" "${precision_label}" \
              "${topology}" "${leaves}" "${raw_sites}" \
              "${evolutionary_height}" "${seed}" "${replicate}" \
              "${benchmark_mode}" "${resume_threads}" \
              clock-like-jc69-simulation "${repeats}" \
              "${conditioning_ms}"; do
            replicate=$((replicate + 1))
          done
          run_count=$((replicate - first_missing))
          completed_cases=$((completed_cases + run_count))
          echo "# progress cases_through=${completed_cases}/${case_count}" \
            "status=benchmark method=${method}" \
            "precision=${precision_label} topology=${topology}" \
            "leaves=${leaves} raw_sites=${raw_sites}" \
            "evolutionary_root_to_tip_distance=${evolutionary_height}" \
            "replicate_start=${first_missing} replicates=${run_count}" \
            "timing_repeats=${repeats}"
          echo "# benchmark_start_jc69 method=${method} precision=${precision_label} benchmark_mode=${benchmark_mode} threads=${threads} topology=${topology} leaves=${leaves} raw_sites=${raw_sites} evolutionary_root_to_tip_distance=${evolutionary_height} replicate_start=${first_missing} replicates=${run_count} timing_repeats=${repeats}"
          if [[ "${dry_run}" == 1 ]]; then
            continue
          fi
          command=("bazel-bin/${target}")
          if [[ "${#resource_arguments[@]}" -ne 0 ]]; then
            command+=("${resource_arguments[@]}")
          fi
          candidate_site_batch="${raw_sites}"
          while true; do
            benchmark_capacity_exhausted=0
            benchmark_capacity_dataset="synthetic-jc69-${topology}-${leaves}-${raw_sites}-${evolutionary_height}"
            benchmark_capacity_study=clock-like-jc69-simulation
            benchmark_capacity_mode="${benchmark_mode}"
            benchmark_capacity_threads="${resume_threads:-none}"
            benchmark_run_capacity_bounded "${capacity_work_directory}" \
              "${host_memory_guard_kib}" "${method}" "${precision_label}" \
              "${candidate_site_batch}" "${command[@]}" \
              --topology "${topology}" --leaves "${leaves}" \
              --sites "${raw_sites}" --site-batch "${candidate_site_batch}" \
              --study-label clock-like-jc69-simulation \
              --synthetic-sequence-model jc69 \
              --evolutionary-root-to-tip-distance "${evolutionary_height}" \
              --compress-patterns true --seed "${seed}" \
              --replicate-start "${first_missing}" \
              --replicates "${run_count}" --repeats "${repeats}" \
              --conditioning-ms "${conditioning_ms}"
            if [[ "${benchmark_capacity_exhausted}" == 0 ]]; then
              break
            fi
            if (( candidate_site_batch <= minimum_site_batch )); then
              echo "# jc69_case_unavailable method=${method}" \
                "precision=${precision_label} topology=${topology}" \
                "leaves=${leaves} raw_sites=${raw_sites}" \
                "evolutionary_root_to_tip_distance=${evolutionary_height}" \
                "replicate_start=${first_missing} replicates=${run_count}"
              break
            fi
            candidate_site_batch=$((candidate_site_batch / 2))
            (( candidate_site_batch < minimum_site_batch )) && \
              candidate_site_batch="${minimum_site_batch}"
            echo "# jc69_capacity_retry method=${method}" \
              "precision=${precision_label} site_batch=${candidate_site_batch}"
          done
        done
      done
    done
  done
done
