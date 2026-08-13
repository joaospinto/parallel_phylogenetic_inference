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

precision="${PRECISION:-fp32}"
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
profile="${TREE_HMM_JC69_PROFILE:-smoke}"
case "${profile}" in
  smoke)
    default_leaves="128 512"
    default_sites="256 1024"
    default_heights="0.001 0.1"
    default_replicates=3
    ;;
  complete)
    # The complete profile mirrors the principal axes and replicate count of
    # the LvD simulation study, but benchmarks complete JC69 likelihood
    # evaluations after exact pattern compression rather than site-order
    # updates. It is intentionally opt-in because its largest alignments are
    # substantial and the Cartesian product contains 14,400 cases per method.
    default_leaves="100 1000 10000"
    default_sites="1000 10000 100000"
    default_heights="0.0001 0.001 0.01 0.1"
    default_replicates=100
    ;;
  *)
    echo "TREE_HMM_JC69_PROFILE must be smoke or complete" >&2
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
repeats="${TREE_HMM_JC69_TIMING_REPEATS:-5}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"

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
echo "# topology_distributions=${topologies[*]}"
echo "# leaf_counts=${leaf_counts[*]}"
echo "# raw_sequence_lengths=${raw_site_counts[*]}"
echo "# evolutionary_root_to_tip_distances=${evolutionary_heights[*]}"
echo "# topology_and_alignment_replicates=${replicates}"
echo "# deterministic_seed_base=${seed}"
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

cd "${repository}"
for topology in "${topologies[@]}"; do
  for leaves in "${leaf_counts[@]}"; do
    for raw_sites in "${raw_site_counts[@]}"; do
      for evolutionary_height in "${evolutionary_heights[@]}"; do
        replicate=0
        while [[ "${replicate}" -lt "${replicates}" ]]; do
          if benchmark_resume_jc69_replicate_completed \
            "${resume_report}" "${method}" "${precision_label}" \
            "${topology}" "${leaves}" "${raw_sites}" \
            "${evolutionary_height}" "${seed}" "${replicate}" \
            "${benchmark_mode}" "${resume_threads}"; then
            replicate=$((replicate + 1))
            continue
          fi
          echo "# benchmark_start_jc69 method=${method} precision=${precision_label} benchmark_mode=${benchmark_mode} threads=${threads} topology=${topology} leaves=${leaves} raw_sites=${raw_sites} evolutionary_root_to_tip_distance=${evolutionary_height} replicate=${replicate}"
          command=("bazel-bin/${target}")
          if [[ "${#resource_arguments[@]}" -ne 0 ]]; then
            command+=("${resource_arguments[@]}")
          fi
          "${command[@]}" "--topology" "${topology}" \
            --leaves "${leaves}" --sites "${raw_sites}" \
            --study-label clock-like-jc69-simulation \
            --synthetic-sequence-model jc69 \
            --evolutionary-root-to-tip-distance "${evolutionary_height}" \
            --compress-patterns true --seed "${seed}" \
            --replicate-start "${replicate}" --replicates 1 \
            --repeats "${repeats}" --conditioning-ms "${conditioning_ms}"
          replicate=$((replicate + 1))
        done
      done
    done
  done
done
