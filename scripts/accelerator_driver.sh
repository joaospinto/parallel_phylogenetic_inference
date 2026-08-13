#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bazel_command="${TREE_HMM_BAZEL_COMMAND:-bazel}"
read -r -a bazel_common_args <<< "${TREE_HMM_BAZEL_COMMON_ARGS:-}"
operating_system="$(uname -s)"
read -r -a precisions <<< "${TREE_HMM_PRECISIONS:-FP64 FP32}"
requested_build_backends="${TREE_HMM_BUILD_BACKENDS:-auto}"
requested_run_backends="${TREE_HMM_RUN_BACKENDS:-auto}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/accelerator_environment.sh
source "${script_dir}/accelerator_environment.sh"

bazel_run() {
  local operation="$1"
  shift
  "${bazel_command}" "${operation}" "${bazel_common_args[@]}" "$@"
}

status() {
  local backend="$1"
  local state="$2"
  local detail="$3"
  printf 'accelerator_status backend=%s status=%s %s\n' \
    "${backend}" "${state}" "${detail}"
}

contains_word() {
  local words="$1"
  local requested="$2"
  [[ " ${words//,/ } " == *" ${requested} "* ]]
}

build_requested() {
  local backend="$1"
  if [[ "${requested_build_backends}" == "auto" ]]; then
    if [[ "${operating_system}" == "Darwin" ]]; then
      [[ "${backend}" == "metal" ]]
    else
      [[ "${backend}" == "cuda" || "${backend}" == "rocm" ]]
    fi
  else
    contains_word "${requested_build_backends}" "${backend}"
  fi
}

run_requested() {
  local backend="$1"
  local detected="$2"
  if [[ "${requested_run_backends}" == "auto" ]]; then
    [[ "${detected}" == "1" ]]
  elif [[ "${requested_run_backends}" == "none" ]]; then
    return 1
  else
    contains_word "${requested_run_backends}" "${backend}"
  fi
}

validate_precisions() {
  local precision
  if [[ "${#precisions[@]}" -eq 0 ]]; then
    echo "TREE_HMM_PRECISIONS must select FP64, FP32, or both" >&2
    exit 2
  fi
  for precision in "${precisions[@]}"; do
    if [[ "${precision}" != "FP64" && "${precision}" != "FP32" ]]; then
      echo "unsupported precision ${precision}" >&2
      exit 2
    fi
  done
}

nvidia_detected=0
if tree_hmm_nvidia_available; then
  nvidia_detected=1
fi

amd_detected=0
if tree_hmm_amd_available; then
  amd_detected=1
fi

echo "=== Accelerator build policy ==="
echo "host_os=${operating_system} host_arch=$(uname -m)"
echo "build_backends=${requested_build_backends} run_backends=${requested_run_backends}"
echo "precisions=${precisions[*]}"
echo "nvidia_gpu_detected=${nvidia_detected} amd_gpu_detected=${amd_detected}"
for repository in parallel_phylogenetic_inference parallel_tree_hmm \
                  bidirectional_tree_rake_compress; do
  if [[ "${repository}" == "parallel_phylogenetic_inference" ]]; then
    repository_path="${repo_dir}"
  else
    repository_path="$(dirname "${repo_dir}")/${repository}"
  fi
  if git -C "${repository_path}" rev-parse --is-inside-work-tree \
      >/dev/null 2>&1; then
    printf 'source_revision repository=%s revision=%s\n' "${repository}" \
      "$(git -C "${repository_path}" rev-parse HEAD)"
  fi
done

cd "${repo_dir}"
validate_precisions

if [[ "${operating_system}" == "Darwin" ]]; then
  if build_requested metal; then
    bazel_run test --config=fp32 --test_output=errors \
      //:likelihood_test //:metal_test @parallel_tree_hmm//:metal_test
    status metal executed "precision=FP32 sdk=system-Metal"
  else
    status metal unavailable "reason=excluded-by-TREE_HMM_BUILD_BACKENDS"
  fi
  status cuda unavailable "reason=project-policy-is-Linux-only"
  status rocm unavailable "reason=project-policy-is-Linux-only"
  exit 0
fi

if [[ "${operating_system}" != "Linux" ]]; then
  status metal unavailable "reason=unsupported-operating-system"
  status cuda unavailable "reason=unsupported-operating-system"
  status rocm unavailable "reason=unsupported-operating-system"
  exit 2
fi

status metal unavailable "reason=project-policy-is-macOS-only"

cuda_arch="$(tree_hmm_detect_cuda_arch)"

if build_requested cuda; then
  if run_requested cuda "${nvidia_detected}"; then
    tree_hmm_check_cuda_driver_compatibility
  fi
  for precision in "${precisions[@]}"; do
    precision_config="$(tr '[:upper:]' '[:lower:]' <<< "${precision}")"
    cuda_args=(
      "--config=${precision_config}"
      --config=cuda
      "--cuda_archs=sm_${cuda_arch}"
    )
    bazel_run build "${cuda_args[@]}" \
      //:cuda_test //:cuda_benchmark @parallel_tree_hmm//:cuda_test
    if run_requested cuda "${nvidia_detected}"; then
      bazel_run test "${cuda_args[@]}" --test_output=errors \
        //:cuda_test @parallel_tree_hmm//:cuda_test
    fi
  done
  if run_requested cuda "${nvidia_detected}"; then
    status cuda executed \
      "precisions=${precisions[*]} sdk=12.8.1 target=sm_${cuda_arch}"
  else
    status cuda compile-only \
      "precisions=${precisions[*]} sdk=12.8.1 target=sm_${cuda_arch}"
  fi
else
  status cuda unavailable "reason=excluded-by-TREE_HMM_BUILD_BACKENDS"
fi

if build_requested rocm; then
  tree_hmm_resolve_rocm_sdk "${bazel_command}"
  rocm_arch="$(tree_hmm_detect_rocm_arch)"
  amdclang="${TREE_HMM_RESOLVED_AMDCLANG}"
  amdclangxx="${TREE_HMM_RESOLVED_AMDCLANGXX}"
  rocm_path="${TREE_HMM_RESOLVED_ROCM_PATH}"
  if run_requested rocm "${amd_detected}"; then
    tree_hmm_check_rocm_driver_compatibility "${rocm_path}" "${rocm_arch}"
  fi
  rocm_args=(
    --config=rocm
    "--repo_env=CC=${amdclang}"
    "--repo_env=CXX=${amdclangxx}"
    "--action_env=ROCM_PATH=${rocm_path}"
    "--rocm_arch=${rocm_arch}"
  )
  for precision in "${precisions[@]}"; do
    precision_config="$(tr '[:upper:]' '[:lower:]' <<< "${precision}")"
    bazel_run build "--config=${precision_config}" \
      "${rocm_args[@]}" \
      //:rocm_test //:rocm_benchmark @parallel_tree_hmm//:rocm_test
    if run_requested rocm "${amd_detected}"; then
      LD_LIBRARY_PATH="${rocm_path}/lib:${LD_LIBRARY_PATH:-}" \
        bazel_run test "--config=${precision_config}" \
        "${rocm_args[@]}" \
        "--test_env=LD_LIBRARY_PATH=${rocm_path}/lib:${LD_LIBRARY_PATH:-}" \
        --test_output=errors \
        //:rocm_test @parallel_tree_hmm//:rocm_test
    fi
  done
  if run_requested rocm "${amd_detected}"; then
    status rocm executed \
      "precisions=${precisions[*]} sdk=7.2.3 target=${rocm_arch}"
  else
    status rocm compile-only \
      "precisions=${precisions[*]} sdk=7.2.3 target=${rocm_arch}"
  fi
else
  status rocm unavailable "reason=excluded-by-TREE_HMM_BUILD_BACKENDS"
fi
