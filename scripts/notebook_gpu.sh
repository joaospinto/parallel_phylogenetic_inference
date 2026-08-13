#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
accelerator_backend="${TREE_HMM_ACCELERATOR_BACKEND:-}"
if [[ -d /kaggle/working ]]; then
  notebook_work_dir=/kaggle/working
else
  notebook_work_dir=/content
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/accelerator_environment.sh
source "${script_dir}/accelerator_environment.sh"
# shellcheck source=scripts/notebook_bazel.sh
source "${script_dir}/notebook_bazel.sh"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_dir}/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${script_dir}/capacity_bounded.sh"
bazel_command="$(tree_hmm_notebook_bazel "${repo_dir}" \
                  "${notebook_work_dir}")"

case "${accelerator_backend}" in
  cuda)
    if ! tree_hmm_nvidia_available; then
      echo "CUDA is unavailable; select an NVIDIA GPU runtime first." >&2
      exit 2
    fi
    tree_hmm_check_cuda_driver_compatibility
    accelerator_title=CUDA
    accelerator_arch="$(tree_hmm_detect_cuda_arch)"
    accelerator_args=(
      --config=cuda
      "--cuda_archs=sm_${accelerator_arch}"
      "--jobs=$(nproc)"
    )
    accelerator_target="sm_${accelerator_arch}"
    accelerator_test_args=()
    other_backend=rocm
    beagle_resources=(cpu cuda)
    ;;
  rocm)
    if ! tree_hmm_amd_available; then
      echo "ROCm is unavailable; select an AMD GPU runtime first." >&2
      exit 2
    fi
    accelerator_title=ROCm
    tree_hmm_resolve_rocm_sdk "${bazel_command}"
    accelerator_arch="$(tree_hmm_detect_rocm_arch)"
    tree_hmm_check_rocm_driver_compatibility \
      "${TREE_HMM_RESOLVED_ROCM_PATH}" "${accelerator_arch}"
    accelerator_args=(
      --config=rocm
      "--repo_env=CC=${TREE_HMM_RESOLVED_AMDCLANG}"
      "--repo_env=CXX=${TREE_HMM_RESOLVED_AMDCLANGXX}"
      "--action_env=ROCM_PATH=${TREE_HMM_RESOLVED_ROCM_PATH}"
      "--rocm_arch=${accelerator_arch}"
      "--jobs=$(nproc)"
    )
    accelerator_target="${accelerator_arch}"
    accelerator_test_args=(
      "--test_env=LD_LIBRARY_PATH=${TREE_HMM_RESOLVED_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    )
    other_backend=cuda
    beagle_resources=(cpu)
    export LD_LIBRARY_PATH="${TREE_HMM_RESOLVED_ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    ;;
  *)
    echo "TREE_HMM_ACCELERATOR_BACKEND must be cuda or rocm" >&2
    exit 2
    ;;
esac
accelerator_test="${accelerator_backend}_test"
accelerator_benchmark="${accelerator_backend}_benchmark"
repeats="${TREE_HMM_BENCHMARK_REPEATS:-5}"
empirical_repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
read -r -a precisions <<< "${TREE_HMM_PRECISIONS:-FP64 FP32}"
read -r -a benchmark_sections <<< \
  "${TREE_HMM_BENCHMARK_SECTIONS:-validation synthetic fish pandit}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
host_memory_guard_percent="${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"
fish_minimum_site_batch="${TREE_HMM_FISH_MINIMUM_SITE_BATCH:-256}"
if [[ ! "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREE_HMM_BENCHMARK_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${empirical_repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREE_HMM_EMPIRICAL_REPEATS must be a positive integer" >&2
  exit 2
fi
if [[ "${#precisions[@]}" -eq 0 ]]; then
  echo "TREE_HMM_PRECISIONS must select FP64, FP32, or both" >&2
  exit 2
fi
for precision in "${precisions[@]}"; do
  if [[ "${precision}" != "FP64" && "${precision}" != "FP32" ]]; then
    echo "TREE_HMM_PRECISIONS contains unsupported precision ${precision}" >&2
    exit 2
  fi
done
if [[ "${#benchmark_sections[@]}" -eq 0 ]]; then
  echo "TREE_HMM_BENCHMARK_SECTIONS must select at least one section" >&2
  exit 2
fi
for section in "${benchmark_sections[@]}"; do
  case "${section}" in
    validation|synthetic|fish|pandit) ;;
    *)
      echo "TREE_HMM_BENCHMARK_SECTIONS contains unsupported section ${section}" >&2
      exit 2
      ;;
  esac
done
if [[ -n "${resume_report}" && ! -r "${resume_report}" ]]; then
  echo "TREE_HMM_RESUME_REPORT is not readable: ${resume_report}" >&2
  exit 2
fi
if [[ ! "${host_memory_guard_percent}" =~ ^[1-9][0-9]?$ ]] ||
   [[ "${host_memory_guard_percent}" -lt 20 ]] ||
   [[ "${host_memory_guard_percent}" -gt 90 ]]; then
  echo "TREE_HMM_HOST_MEMORY_GUARD_PERCENT must be between 20 and 90" >&2
  exit 2
fi
if [[ ! "${fish_minimum_site_batch}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREE_HMM_FISH_MINIMUM_SITE_BATCH must be a positive integer" >&2
  exit 2
fi

if [[ "${TREE_HMM_SKIP_PORTABILITY_COMPILE_CHECK:-0}" != "1" ]]; then
  echo "=== Compile-only ${other_backend} portability check ==="
  if TREE_HMM_BAZEL_COMMAND="${bazel_command}" \
    TREE_HMM_BUILD_BACKENDS="${other_backend}" \
    TREE_HMM_RUN_BACKENDS=none \
    TREE_HMM_PRECISIONS="${precisions[*]}" \
    bash "${repo_dir}/scripts/accelerator_driver.sh"; then
    echo "# portability_compile_complete backend=${other_backend}"
  else
    portability_status=$?
    echo "# portability_compile_failed backend=${other_backend}" \
      "exit_code=${portability_status}"
    echo "The unused-backend portability failure is recorded above;" \
      "continuing with ${accelerator_title} validation and benchmarks."
  fi
fi

detected_available_host_memory_kib() {
  local available
  local candidate
  local current
  available="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"

  if [[ -r /sys/fs/cgroup/memory.max &&
        -r /sys/fs/cgroup/memory.current ]]; then
    candidate="$(</sys/fs/cgroup/memory.max)"
    current="$(</sys/fs/cgroup/memory.current)"
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes &&
          -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    candidate="$(</sys/fs/cgroup/memory/memory.limit_in_bytes)"
    current="$(</sys/fs/cgroup/memory/memory.usage_in_bytes)"
  else
    candidate=""
    current=""
  fi
  if [[ "${candidate}" =~ ^[0-9]+$ && "${current}" =~ ^[0-9]+$ &&
        "${candidate}" -gt "${current}" ]]; then
    candidate=$(((candidate - current) / 1024))
    if [[ "${candidate}" -lt "${available}" ]]; then
      available="${candidate}"
    fi
  fi
  echo "${available}"
}

current_host_memory_guard_kib() {
  local available
  available="$(detected_available_host_memory_kib)"
  echo $((available * host_memory_guard_percent / 100))
}

section_selected() {
  local requested="$1"
  local selected
  for selected in "${benchmark_sections[@]}"; do
    if [[ "${selected}" == "${requested}" ]]; then
      return 0
    fi
  done
  return 1
}

benchmark_section_enabled() {
  [[ "${TREE_HMM_SKIP_BENCHMARKS:-0}" != "1" ]] && section_selected "$1"
}

initial_host_memory_guard_kib="$(current_host_memory_guard_kib)"

echo "=== Source revisions ==="
if [[ -f "$(dirname "${repo_dir}")/SOURCE_REVISIONS.txt" ]]; then
  cat "$(dirname "${repo_dir}")/SOURCE_REVISIONS.txt"
fi
for repository in parallel_phylogenetic_inference parallel_tree_hmm \
                  bidirectional_tree_rake_compress; do
  path="$(dirname "${repo_dir}")/${repository}"
  if [[ -d "${path}/.git" ]]; then
    printf '%s ' "${repository}"
    git -C "${path}" rev-parse HEAD
  else
    echo "${repository} packaged source"
  fi
done

echo "=== Host platform ==="
uname -a
if [[ -r /etc/os-release ]]; then
  cat /etc/os-release
fi
lscpu
free -h

echo "=== Toolchain ==="
c++ --version
"${bazel_command}" --version

if [[ "${accelerator_backend}" == cuda ]]; then
  echo "=== NVIDIA device ==="
  nvidia-smi
  nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,\
clocks.max.graphics,clocks.max.memory --format=csv || true
else
  echo "=== AMD device ==="
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo || true
  fi
  "${TREE_HMM_RESOLVED_AMDCLANGXX}" --version
fi
echo "selected Bazel ${accelerator_title} target: ${accelerator_target}"
echo "selected notebook sections: ${benchmark_sections[*]}"
echo "host-memory guard for CPU capacity probes:" \
  "${initial_host_memory_guard_kib} KiB" \
  "(${host_memory_guard_percent}% of currently available memory)"
if [[ -n "${resume_report}" ]]; then
  echo "resuming completed benchmark rows from ${resume_report}"
fi

cd "${repo_dir}"
if benchmark_section_enabled fish &&
   [[ "${TREE_HMM_SKIP_FISH_TREE:-0}" != "1" ]]; then
  fish_dir="${notebook_work_dir}/fish_tree_of_life"
  bash "${repo_dir}/scripts/fetch_fish_tree.sh" "${fish_dir}"
  fish_sites="$(awk 'NR == 1 { print $2; exit }' \
    "${fish_dir}/final_alignment.phylip")"
  if [[ ! "${fish_sites}" =~ ^[1-9][0-9]*$ ]]; then
    echo "could not read the Fish Tree of Life alignment length" >&2
    exit 2
  fi
  fish_site_batches=()
  site_batch="${fish_minimum_site_batch}"
  while [[ "${site_batch}" -lt "${fish_sites}" ]]; do
    fish_site_batches+=("${site_batch}")
    site_batch=$((site_batch * 2))
  done
  fish_site_batches+=("${fish_sites}")
fi
if benchmark_section_enabled pandit &&
   [[ "${TREE_HMM_SKIP_PANDIT:-0}" != "1" ]]; then
  pandit_dir="${notebook_work_dir}/pandit"
  bash "${repo_dir}/scripts/fetch_pandit.sh" "${pandit_dir}" \
    --min-leaves 100
fi

if [[ "${TREE_HMM_SKIP_BENCHMARKS:-0}" != "1" &&
      "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]] &&
   { section_selected synthetic || section_selected fish ||
     section_selected pandit; }; then
  beagle_build_backend=cpu
  if [[ "${accelerator_backend}" == cuda ]]; then
    beagle_build_backend=cuda
  fi
  beagle_prefix="${notebook_work_dir}/beagle-4.0.1-${beagle_build_backend}"
  BEAGLE_BUILD_BACKEND="${beagle_build_backend}" \
    bash "${repo_dir}/scripts/install_beagle.sh" "${beagle_prefix}"
  export BEAGLE_PREFIX="${beagle_prefix}"
  # shellcheck source=scripts/beagle_environment.sh
  source "${repo_dir}/scripts/beagle_environment.sh"
  parallel_phylogenetics_configure_beagle
fi

for precision in "${precisions[@]}"; do
  precision_config="$(tr '[:upper:]' '[:lower:]' <<< "${precision}")"
  precision_args=("--config=${precision_config}")
  validation_completed=0
  if ! section_selected validation; then
    validation_completed=1
    echo "=== ${precision} validation omitted by section selection ==="
  elif benchmark_resume_validation_completed "${resume_report}" \
       "${precision}" "${accelerator_backend}"; then
    validation_completed=1
    echo "=== ${precision} validation reused from prior report ==="
  else
    echo "=== ${precision} host tests, including device algebra ==="
    "${bazel_command}" test "${precision_args[@]}" --test_output=errors \
      //:likelihood_test \
      @parallel_tree_hmm//:inference_test \
      @parallel_tree_hmm//:gpu_device_algebra_test

    echo "=== ${precision} native ${accelerator_title} correctness test ==="
    "${bazel_command}" test "${precision_args[@]}" \
      "${accelerator_args[@]}" "${accelerator_test_args[@]}" \
      --test_output=errors \
      "//:${accelerator_test}" \
      "@parallel_tree_hmm//:${accelerator_test}"
  fi
  "${bazel_command}" build "${precision_args[@]}" \
    "${accelerator_args[@]}" \
    "//:${accelerator_benchmark}" "//:${accelerator_test}" \
    "@parallel_tree_hmm//:${accelerator_test}"

  if [[ "${accelerator_backend}" == cuda ]] &&
     command -v compute-sanitizer >/dev/null 2>&1 &&
     [[ "${TREE_HMM_SKIP_SANITIZER:-0}" != "1" ]] &&
     [[ "${validation_completed}" == "0" ]]; then
    read -r -a sanitizer_tools <<< \
      "${TREE_HMM_SANITIZER_TOOLS:-memcheck racecheck synccheck}"
    tree_hmm_native_test="$(find -L bazel-bin/external -type f \
      -path '*parallel_tree_hmm*' -name "${accelerator_test}" -perm -111 | \
      head -n 1)"
    native_tests=("bazel-bin/${accelerator_test}" "${tree_hmm_native_test}")
    for native_test in "${native_tests[@]}"; do
      if [[ ! -x "${native_test}" ]]; then
        echo "could not locate ${native_test}" >&2
        exit 2
      fi
      for tool in "${sanitizer_tools[@]}"; do
        echo "=== ${precision} ${accelerator_title} ${tool}: ${native_test} ==="
        compute-sanitizer --tool "${tool}" --error-exitcode 99 \
          "${native_test}"
      done
    done
  fi
  if [[ "${validation_completed}" == "0" ]]; then
    echo "# validation_complete backend=${accelerator_backend}" \
      "precision=${precision}"
  fi

  if benchmark_section_enabled synthetic; then
    echo "=== ${precision} ${accelerator_title} scaling benchmark ==="
    cases=(
      "64 256"
      "256 256"
      "1024 256"
      "4096 256"
      "4096 1024"
      "16384 256"
      "65536 64"
    )
    for topology in balanced caterpillar; do
      for specification in "${cases[@]}"; do
        read -r leaves sites <<< "${specification}"
        if benchmark_resume_case_completed "${resume_report}" \
          "${accelerator_backend}" \
          "${precision}" synthetic "${topology}" "${leaves}" "${sites}" \
          "${sites}"; then
          echo "# resume_skip method=${accelerator_backend}" \
            "precision=${precision}" \
            "dataset=synthetic topology=${topology} leaves=${leaves}" \
            "sites=${sites} site_batch=${sites}"
          continue
        fi
        echo "# benchmark_start method=${accelerator_backend}" \
          "precision=${precision}" \
          "dataset=synthetic topology=${topology} leaves=${leaves}" \
          "sites=${sites} site_batch=${sites}"
        "bazel-bin/${accelerator_benchmark}" \
          --topology "${topology}" \
          --leaves "${leaves}" \
          --sites "${sites}" \
          --repeats "${repeats}"
      done
    done

    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]]; then
      echo "=== ${precision} BEAGLE comparison ==="
      "${bazel_command}" build "${precision_args[@]}" //:beagle_benchmark
      for resource in "${beagle_resources[@]}"; do
        for topology in balanced caterpillar; do
          for specification in "${cases[@]}"; do
            read -r leaves sites <<< "${specification}"
            if benchmark_resume_case_completed "${resume_report}" \
              "beagle-${resource}" "${precision}" synthetic \
              "${topology}" "${leaves}" "${sites}" "${sites}"; then
              echo "# resume_skip method=beagle-${resource}" \
                "precision=${precision} dataset=synthetic" \
                "topology=${topology} leaves=${leaves} sites=${sites}" \
                "site_batch=${sites}"
              continue
            fi
            echo "# benchmark_start method=beagle-${resource}" \
              "precision=${precision} dataset=synthetic" \
              "topology=${topology} leaves=${leaves} sites=${sites}" \
              "site_batch=${sites}"
            bazel-bin/beagle_benchmark \
              --beagle-resource "${resource}" \
              --beagle-threads 1 \
              --topology "${topology}" \
              --leaves "${leaves}" \
              --sites "${sites}" \
              --repeats "${repeats}"
          done
        done
      done
    fi

  fi

  if benchmark_section_enabled fish &&
     [[ "${TREE_HMM_SKIP_FISH_TREE:-0}" != "1" ]]; then
    echo "=== ${precision} Fish Tree of Life public-data benchmark ==="
    benchmark_capacity_exhausted=0
    echo "# adaptive_site_batch_grid=${fish_site_batches[*]}"
    for site_batch in "${fish_site_batches[@]}"; do
      if benchmark_resume_capacity_reached "${resume_report}" \
        "${accelerator_backend}" \
        "${precision}" "${site_batch}"; then
        echo "# resume_capacity_limit method=${accelerator_backend}" \
          "precision=${precision}" \
          "site_batch=${site_batch}"
        break
      fi
      if benchmark_resume_dataset_batch_completed "${resume_report}" \
        "${accelerator_backend}" \
        "${precision}" actinopt_12k_raxml "${site_batch}"; then
        echo "# resume_skip method=${accelerator_backend}" \
          "precision=${precision}" \
          "dataset=actinopt_12k_raxml site_batch=${site_batch}"
        continue
      fi
      echo "# benchmark_start method=${accelerator_backend}" \
        "precision=${precision}" \
        "dataset=actinopt_12k_raxml site_batch=${site_batch}"
      benchmark_run_capacity_bounded "${notebook_work_dir}" \
        "$(current_host_memory_guard_kib)" "${accelerator_backend}" \
        "${precision}" \
        "${site_batch}" \
        "bazel-bin/${accelerator_benchmark}" \
        --newick "${fish_dir}/actinopt_12k_raxml.tre" \
        --phylip "${fish_dir}/final_alignment.phylip" \
        --site-batch "${site_batch}" \
        --repeats "${empirical_repeats}"
      if [[ "${benchmark_capacity_exhausted}" == "1" ]]; then
        break
      fi
    done
    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]]; then
      echo "=== ${precision} Fish Tree of Life BEAGLE comparison ==="
      for resource in "${beagle_resources[@]}"; do
        method="beagle-${resource}"
        benchmark_capacity_exhausted=0
        for site_batch in "${fish_site_batches[@]}"; do
          if benchmark_resume_capacity_reached "${resume_report}" \
            "${method}" "${precision}" "${site_batch}"; then
            echo "# resume_capacity_limit method=${method}" \
              "precision=${precision} site_batch=${site_batch}"
            break
          fi
          if benchmark_resume_dataset_batch_completed "${resume_report}" \
            "${method}" "${precision}" actinopt_12k_raxml \
            "${site_batch}"; then
            echo "# resume_skip method=${method} precision=${precision}" \
              "dataset=actinopt_12k_raxml site_batch=${site_batch}"
            continue
          fi
          echo "# benchmark_start method=${method} precision=${precision}" \
            "dataset=actinopt_12k_raxml site_batch=${site_batch}"
          benchmark_run_capacity_bounded "${notebook_work_dir}" \
            "$(current_host_memory_guard_kib)" "${method}" "${precision}" \
            "${site_batch}" bazel-bin/beagle_benchmark \
            --beagle-resource "${resource}" \
            --beagle-threads 1 \
            --newick "${fish_dir}/actinopt_12k_raxml.tre" \
            --phylip "${fish_dir}/final_alignment.phylip" \
            --site-batch "${site_batch}" \
            --repeats "${empirical_repeats}"
          if [[ "${benchmark_capacity_exhausted}" == "1" ]]; then
            break
          fi
        done
      done
    fi
  fi

  if benchmark_section_enabled pandit &&
     [[ "${TREE_HMM_SKIP_PANDIT:-0}" != "1" ]]; then
    echo "=== ${precision} PANDIT 17.0 corpus benchmark ==="
    PRECISION="${precision_config}" PANDIT_SKIP_BUILD=1 \
      TREE_HMM_RESUME_REPORT="${resume_report}" \
      TREE_HMM_EMPIRICAL_REPEATS="${empirical_repeats}" \
      bash "${repo_dir}/scripts/benchmark_pandit.sh" \
        "${accelerator_backend}" \
        "${pandit_dir}/families"
    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]]; then
      beagle_pandit_resources=(beagle-cpu)
      if [[ "${accelerator_backend}" == cuda ]]; then
        beagle_pandit_resources+=(beagle-cuda)
      fi
      for resource in "${beagle_pandit_resources[@]}"; do
        PRECISION="${precision_config}" PANDIT_SKIP_BUILD=1 \
          TREE_HMM_RESUME_REPORT="${resume_report}" \
          TREE_HMM_EMPIRICAL_REPEATS="${empirical_repeats}" \
          bash "${repo_dir}/scripts/benchmark_pandit.sh" "${resource}" \
            "${pandit_dir}/families"
      done
    fi
  fi
done
