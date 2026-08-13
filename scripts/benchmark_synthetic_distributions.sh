#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda}" >&2
  exit 2
fi
method="$1"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="${repository}/scripts"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"

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
    resource_arguments=()
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

echo "# study=independent-taxa-pattern-grid"
echo "# topology_distributions=${topologies[*]}"
echo "# leaf_counts=${leaf_counts[*]}"
echo "# unique_pattern_counts=${pattern_counts[*]}"
echo "# topology_replicates=${replicates}"
echo "# deterministic_seed_base=${seed}"
echo "# synthetic_patterns=distinct deterministic nucleotide patterns; not JC69 simulations"
echo "# selection_rule=complete Cartesian product; no cell selected from timings"
echo "# benchmark_mode=${benchmark_mode}"
if [[ "${method}" == beagle-* ]]; then
  echo "# beagle_threads=${threads}"
fi

cd "${repository}"
for topology in "${topologies[@]}"; do
  for leaves in "${leaf_counts[@]}"; do
    for patterns in "${pattern_counts[@]}"; do
      replicate=0
      while [[ "${replicate}" -lt "${replicates}" ]]; do
        if benchmark_resume_synthetic_replicate_completed \
          "${resume_report}" "${method}" "${precision_label}" \
          "${topology}" "${leaves}" "${patterns}" "${seed}" \
          "${replicate}" "${benchmark_mode}" \
          "${resume_threads}"; then
          replicate=$((replicate + 1))
          continue
        fi
        first_missing="${replicate}"
        replicate=$((replicate + 1))
        while [[ "${replicate}" -lt "${replicates}" ]] &&
          ! benchmark_resume_synthetic_replicate_completed \
            "${resume_report}" "${method}" "${precision_label}" \
            "${topology}" "${leaves}" "${patterns}" "${seed}" \
            "${replicate}" "${benchmark_mode}" \
            "${resume_threads}"; do
          replicate=$((replicate + 1))
        done
        run_count=$((replicate - first_missing))
        echo "# benchmark_start_grid_cell method=${method} precision=${precision_label} benchmark_mode=${benchmark_mode} threads=${threads} topology=${topology} leaves=${leaves} unique_patterns=${patterns} replicate_start=${first_missing} replicates=${run_count}"
        command=("bazel-bin/${target}")
        if [[ "${#resource_arguments[@]}" -ne 0 ]]; then
          command+=("${resource_arguments[@]}")
        fi
        "${command[@]}" \
          --topology "${topology}" --leaves "${leaves}" --sites "${patterns}" \
          --seed "${seed}" --replicate-start "${first_missing}" \
          --replicates "${run_count}" --repeats "${repeats}" \
          --conditioning-ms "${conditioning_ms}"
      done
    done
  done
done
