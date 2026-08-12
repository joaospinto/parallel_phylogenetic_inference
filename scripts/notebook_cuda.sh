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
bazel_command="$(tree_hmm_notebook_bazel "${repo_dir}" \
                  "${notebook_work_dir}")"

compute_capability="$(nvidia-smi --query-gpu=compute_cap \
  --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:].')"
cuda_arch="${TREE_HMM_CUDA_ARCH:-${compute_capability}}"
repeats="${TREE_HMM_BENCHMARK_REPEATS:-5}"
empirical_repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
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

cd "${repo_dir}"
echo "=== Host tests, including CUDA algebra emulation ==="
"${bazel_command}" test --test_output=errors \
  //:likelihood_test \
  @parallel_tree_hmm//:inference_test \
  @parallel_tree_hmm//:cuda_kernel_emulation_test

cuda_args=(
  --config=cuda
  "--cuda_archs=sm_${cuda_arch}"
  "--jobs=$(nproc)"
)
echo "=== Native CUDA build and correctness test ==="
"${bazel_command}" test "${cuda_args[@]}" --test_output=errors \
  //:cuda_test \
  @parallel_tree_hmm//:cuda_test
"${bazel_command}" build "${cuda_args[@]}" //:cuda_benchmark //:cuda_test \
  @parallel_tree_hmm//:cuda_test

if command -v compute-sanitizer >/dev/null 2>&1 &&
   [[ "${TREE_HMM_SKIP_SANITIZER:-0}" != "1" ]]; then
  read -r -a sanitizer_tools <<< \
    "${TREE_HMM_SANITIZER_TOOLS:-memcheck racecheck synccheck}"
  tree_hmm_native_test="$(find -L bazel-bin/external -type f \
    -path '*parallel_tree_hmm*' -name cuda_test -perm -111 | head -n 1)"
  native_tests=(
    bazel-bin/cuda_test
    "${tree_hmm_native_test}"
  )
  for native_test in "${native_tests[@]}"; do
    if [[ ! -x "${native_test}" ]]; then
      echo "could not locate ${native_test}" >&2
      exit 2
    fi
    for tool in "${sanitizer_tools[@]}"; do
      echo "=== CUDA ${tool}: ${native_test} ==="
      compute-sanitizer --tool "${tool}" --error-exitcode 99 "${native_test}"
    done
  done
fi

if [[ "${TREE_HMM_SKIP_BENCHMARKS:-0}" != "1" ]]; then
  echo "=== CUDA scaling benchmark ==="
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
      bazel-bin/cuda_benchmark \
        --topology "${topology}" \
        --leaves "${leaves}" \
        --sites "${sites}" \
        --repeats "${repeats}"
    done
  done

  if [[ "${TREE_HMM_SKIP_FISH_TREE:-0}" != "1" ]]; then
    echo "=== Fish Tree of Life public-data benchmark ==="
    fish_dir="${notebook_work_dir}/fish_tree_of_life"
    bash "${repo_dir}/scripts/fetch_fish_tree.sh" "${fish_dir}"
    for site_batch in 256 1024 4096; do
      bazel-bin/cuda_benchmark \
        --newick "${fish_dir}/actinopt_12k_raxml.tre" \
        --phylip "${fish_dir}/final_alignment.phylip" \
        --site-batch "${site_batch}" \
        --repeats "${empirical_repeats}"
    done
  fi
fi
