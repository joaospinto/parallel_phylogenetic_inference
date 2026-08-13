#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d /kaggle/working ]]; then
  notebook_work_dir=/kaggle/working
else
  notebook_work_dir=/content
fi

if ! command -v nvcc >/dev/null 2>&1 ||
   ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA is unavailable; select an NVIDIA GPU runtime first." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/notebook_bazel.sh
source "${script_dir}/notebook_bazel.sh"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_dir}/benchmark_resume.sh"
# shellcheck source=scripts/capacity_bounded.sh
source "${script_dir}/capacity_bounded.sh"
bazel_command="$(tree_hmm_notebook_bazel "${repo_dir}" \
                  "${notebook_work_dir}")"

compute_capability="$(nvidia-smi --query-gpu=compute_cap \
  --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:].')"
cuda_arch="${TREE_HMM_CUDA_ARCH:-${compute_capability}}"
repeats="${TREE_HMM_BENCHMARK_REPEATS:-5}"
empirical_repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
read -r -a precisions <<< "${TREE_HMM_PRECISIONS:-FP64 FP32}"
read -r -a benchmark_sections <<< \
  "${TREE_HMM_BENCHMARK_SECTIONS:-validation synthetic fish pandit}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
host_memory_guard_percent="${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"
fish_minimum_site_batch="${TREE_HMM_FISH_MINIMUM_SITE_BATCH:-256}"
if [[ ! "${cuda_arch}" =~ ^[0-9]+$ ]]; then
  echo "could not determine a CUDA architecture; set TREE_HMM_CUDA_ARCH" >&2
  exit 2
fi
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

detected_host_memory_kib() {
  local result
  local candidate
  result="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)"
  for control_file in /sys/fs/cgroup/memory.max \
                      /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ ! -r "${control_file}" ]]; then
      continue
    fi
    candidate="$(<"${control_file}")"
    if [[ "${candidate}" =~ ^[0-9]+$ ]]; then
      candidate=$((candidate / 1024))
      if [[ "${candidate}" -gt 0 && "${candidate}" -lt "${result}" ]]; then
        result="${candidate}"
      fi
    fi
  done
  echo "${result}"
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

host_memory_limit_kib="$(detected_host_memory_kib)"
host_memory_guard_kib=$((
  host_memory_limit_kib * host_memory_guard_percent / 100
))

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
nvcc --version
"${bazel_command}" --version

echo "=== NVIDIA device ==="
nvidia-smi
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total,\
clocks.max.graphics,clocks.max.memory --format=csv || true
echo "selected Bazel CUDA architecture: sm_${cuda_arch}"
echo "selected notebook sections: ${benchmark_sections[*]}"
echo "host-memory guard for CPU capacity probes:" \
  "${host_memory_guard_kib} KiB (${host_memory_guard_percent}% of detected limit)"
if [[ -n "${resume_report}" ]]; then
  echo "resuming completed benchmark rows from ${resume_report}"
fi

cd "${repo_dir}"
cuda_args=(
  --config=cuda
  "--cuda_archs=sm_${cuda_arch}"
  "--jobs=$(nproc)"
)
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
  beagle_prefix="${notebook_work_dir}/beagle-4.0.1"
  bash "${repo_dir}/scripts/install_beagle_cuda.sh" "${beagle_prefix}"
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
       "${precision}"; then
    validation_completed=1
    echo "=== ${precision} validation reused from prior report ==="
  else
    echo "=== ${precision} host tests, including CUDA algebra emulation ==="
    "${bazel_command}" test "${precision_args[@]}" --test_output=errors \
      //:likelihood_test \
      @parallel_tree_hmm//:inference_test \
      @parallel_tree_hmm//:cuda_kernel_emulation_test

    echo "=== ${precision} native CUDA build and correctness test ==="
    "${bazel_command}" test "${precision_args[@]}" "${cuda_args[@]}" \
      --test_output=errors //:cuda_test @parallel_tree_hmm//:cuda_test
  fi
  "${bazel_command}" build "${precision_args[@]}" "${cuda_args[@]}" \
    //:cuda_benchmark //:cuda_test @parallel_tree_hmm//:cuda_test

  if command -v compute-sanitizer >/dev/null 2>&1 &&
     [[ "${TREE_HMM_SKIP_SANITIZER:-0}" != "1" ]] &&
     [[ "${validation_completed}" == "0" ]]; then
    read -r -a sanitizer_tools <<< \
      "${TREE_HMM_SANITIZER_TOOLS:-memcheck racecheck synccheck}"
    tree_hmm_native_test="$(find -L bazel-bin/external -type f \
      -path '*parallel_tree_hmm*' -name cuda_test -perm -111 | head -n 1)"
    native_tests=(bazel-bin/cuda_test "${tree_hmm_native_test}")
    for native_test in "${native_tests[@]}"; do
      if [[ ! -x "${native_test}" ]]; then
        echo "could not locate ${native_test}" >&2
        exit 2
      fi
      for tool in "${sanitizer_tools[@]}"; do
        echo "=== ${precision} CUDA ${tool}: ${native_test} ==="
        compute-sanitizer --tool "${tool}" --error-exitcode 99 \
          "${native_test}"
      done
    done
  fi
  if [[ "${validation_completed}" == "0" ]]; then
    echo "# validation_complete precision=${precision}"
  fi

  if benchmark_section_enabled synthetic; then
    echo "=== ${precision} CUDA scaling benchmark ==="
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
        if benchmark_resume_case_completed "${resume_report}" cuda \
          "${precision}" synthetic "${topology}" "${leaves}" "${sites}" \
          "${sites}"; then
          echo "# resume_skip method=cuda precision=${precision}" \
            "dataset=synthetic topology=${topology} leaves=${leaves}" \
            "sites=${sites} site_batch=${sites}"
          continue
        fi
        bazel-bin/cuda_benchmark \
          --topology "${topology}" \
          --leaves "${leaves}" \
          --sites "${sites}" \
          --repeats "${repeats}"
      done
    done

    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]]; then
      echo "=== ${precision} BEAGLE CPU and CUDA comparison ==="
      "${bazel_command}" build "${precision_args[@]}" //:beagle_benchmark
      for resource in cpu cuda; do
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
      if benchmark_resume_capacity_reached "${resume_report}" cuda \
        "${precision}" "${site_batch}"; then
        echo "# resume_capacity_limit method=cuda precision=${precision}" \
          "site_batch=${site_batch}"
        break
      fi
      if benchmark_resume_dataset_batch_completed "${resume_report}" cuda \
        "${precision}" actinopt_12k_raxml "${site_batch}"; then
        echo "# resume_skip method=cuda precision=${precision}" \
          "dataset=actinopt_12k_raxml site_batch=${site_batch}"
        continue
      fi
      benchmark_run_capacity_bounded "${notebook_work_dir}" \
        "${host_memory_guard_kib}" cuda "${precision}" "${site_batch}" \
        bazel-bin/cuda_benchmark \
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
      for resource in cpu cuda; do
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
          benchmark_run_capacity_bounded "${notebook_work_dir}" \
            "${host_memory_guard_kib}" "${method}" "${precision}" \
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
      bash "${repo_dir}/scripts/benchmark_pandit.sh" cuda \
        "${pandit_dir}/families"
    if [[ "${TREE_HMM_SKIP_BEAGLE:-0}" != "1" ]]; then
      for resource in beagle-cpu beagle-cuda; do
        PRECISION="${precision_config}" PANDIT_SKIP_BUILD=1 \
          TREE_HMM_RESUME_REPORT="${resume_report}" \
          TREE_HMM_EMPIRICAL_REPEATS="${empirical_repeats}" \
          bash "${repo_dir}/scripts/benchmark_pandit.sh" "${resource}" \
            "${pandit_dir}/families"
      done
    fi
  fi
done
