#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repeats="${TREE_HMM_BENCHMARK_REPEATS:-5}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-5000}"
read -r -a benchmark_modes <<< \
  "${TREE_HMM_SYNTHETIC_BENCHMARK_MODES:-full-input-update factor-update fixed-model}"
read -r -a task_modes <<< \
  "${TREE_HMM_TASK_BENCHMARK_MODES:-full-input-update}"
logical_core_count="$(sysctl -n hw.logicalcpu)"
read -r -a beagle_cpu_threads <<< \
  "${TREE_HMM_BEAGLE_CPU_THREADS:-1 ${logical_core_count}}"
if [[ ! "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREE_HMM_BENCHMARK_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${conditioning_ms}" =~ ^[0-9]+$ ]]; then
  echo "TREE_HMM_BENCHMARK_CONDITIONING_MS must be a nonnegative integer" >&2
  exit 2
fi

cd "${repo_dir}"
bazel build //:metal_benchmark --config=fp32
if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != 1 ]]; then
  # shellcheck source=scripts/beagle_environment.sh
  source "${repo_dir}/scripts/beagle_environment.sh"
  parallel_phylogenetics_configure_beagle
  bazel build //:beagle_benchmark --config=fp32
fi

if [[ "${TREE_HMM_RUN_TASKS:-0}" == 1 ]]; then
  bazel build //:metal_tasks_benchmark --config=fp32
  echo "=== Metal representative inference-task timings ==="
  echo "# study=reverse-task-representative"
  echo "# task_topology=yule"
  echo "# task_leaves=2048"
  echo "# task_sites=256"
  echo "# task_replicates=10"
  echo "# task_seed_base=20260813"
  for benchmark_mode in "${task_modes[@]}"; do
    for replicate in {0..9}; do
      task_output="$(mktemp "${TMPDIR:-/tmp}/metal-task-case.XXXXXX")"
      if bazel-bin/metal_tasks_benchmark --topology yule --leaves 2048 \
          --sites 256 --seed 20260813 --replicate-start "${replicate}" \
          --replicates 1 --repeats "${repeats}" \
          --conditioning-ms "${conditioning_ms}" \
          --benchmark-mode "${benchmark_mode}" \
          --study-label reverse-task-representative >"${task_output}"; then
        cat "${task_output}"
        rm -f "${task_output}"
      else
        task_status=$?
        cat "${task_output}" >&2
        rm -f "${task_output}"
        exit "${task_status}"
      fi
    done
  done
fi

if [[ -n "${TREE_HMM_EMPIRICAL_MANIFEST:-}" ]]; then
  for benchmark_mode in "${benchmark_modes[@]}"; do
    empirical_methods=(metal)
    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != 1 ]]; then
      for threads in "${beagle_cpu_threads[@]}"; do
        empirical_methods+=("beagle-cpu:${threads}")
      done
    fi
    PRECISION=fp32 TREE_HMM_BENCHMARK_MODE="${benchmark_mode}" \
      TREE_HMM_BENCHMARK_CONDITIONING_MS="${conditioning_ms}" \
      bash "${repo_dir}/scripts/benchmark_empirical_manifest.sh" \
        --interleave "${TREE_HMM_EMPIRICAL_MANIFEST}" \
        "${empirical_methods[@]}"
  done
fi
echo "=== Apple device ==="
system_profiler SPHardwareDataType SPDisplaysDataType
echo "=== Metal scaling benchmark ==="
cases=(
  "64 256"
  "256 256"
  "1024 256"
  "4096 256"
  "4096 1024"
  "16384 256"
  "65536 64"
)
for benchmark_mode in "${benchmark_modes[@]}"; do
  for topology in balanced caterpillar; do
    for specification in "${cases[@]}"; do
      read -r leaves sites <<< "${specification}"
      bazel-bin/metal_benchmark \
        --topology "${topology}" \
        --leaves "${leaves}" \
        --sites "${sites}" \
        --conditioning-ms "${conditioning_ms}" \
        --repeats "${repeats}" \
        --benchmark-mode "${benchmark_mode}"
      if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != 1 ]]; then
        for threads in "${beagle_cpu_threads[@]}"; do
          bazel-bin/beagle_benchmark --beagle-resource cpu \
            --beagle-threads "${threads}" --topology "${topology}" \
            --leaves "${leaves}" --sites "${sites}" \
            --conditioning-ms "${conditioning_ms}" --repeats "${repeats}" \
            --benchmark-mode "${benchmark_mode}"
        done
      fi
    done
  done
done

if [[ "${TREE_HMM_RUN_DISTRIBUTIONS:-0}" == 1 ]]; then
  for benchmark_mode in "${benchmark_modes[@]}"; do
    PRECISION=fp32 TREE_HMM_BENCHMARK_MODE="${benchmark_mode}" \
      TREE_HMM_BENCHMARK_CONDITIONING_MS="${conditioning_ms}" \
      bash "${repo_dir}/scripts/benchmark_synthetic_distributions.sh" metal
    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != 1 ]]; then
      for threads in "${beagle_cpu_threads[@]}"; do
        PRECISION=fp32 TREE_HMM_BENCHMARK_MODE="${benchmark_mode}" \
          TREE_HMM_BENCHMARK_CONDITIONING_MS="${conditioning_ms}" \
          BEAGLE_THREADS="${threads}" \
          bash "${repo_dir}/scripts/benchmark_synthetic_distributions.sh" \
            beagle-cpu
      done
    fi
  done
fi
