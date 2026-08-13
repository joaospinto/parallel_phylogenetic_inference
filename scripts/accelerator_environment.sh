#!/usr/bin/env bash

tree_hmm_nvidia_available() {
  command -v nvidia-smi >/dev/null 2>&1 &&
    nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1
}

tree_hmm_amd_available() {
  [[ -e /dev/kfd ]]
}

tree_hmm_require_compute_sanitizer() {
  local precision="$1"
  if command -v compute-sanitizer >/dev/null 2>&1; then
    return 0
  fi
  echo "# validation_incomplete backend=cuda precision=${precision}" \
    "reason=compute-sanitizer-unavailable" >&2
  return 2
}

tree_hmm_detect_cuda_arch() {
  local architecture="${TREE_HMM_CUDA_ARCH:-}"
  if [[ -z "${architecture}" ]] && tree_hmm_nvidia_available; then
    architecture="$(nvidia-smi --query-gpu=compute_cap \
      --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:].')"
  fi
  architecture="${architecture:-80}"
  architecture="${architecture#sm_}"
  if [[ ! "${architecture}" =~ ^[0-9]+$ ]]; then
    echo "TREE_HMM_CUDA_ARCH must look like 60 or sm_60" >&2
    return 2
  fi
  printf '%s\n' "${architecture}"
}

tree_hmm_detect_rocm_arch() {
  local architecture="${TREE_HMM_ROCM_ARCH:-}"
  local library_path="${LD_LIBRARY_PATH:-}"
  local rocminfo="${TREE_HMM_RESOLVED_ROCMINFO:-}"
  if [[ -z "${rocminfo}" ]] && command -v rocminfo >/dev/null 2>&1; then
    rocminfo="$(command -v rocminfo)"
  fi
  if [[ -z "${architecture}" ]] && tree_hmm_amd_available &&
     [[ -n "${rocminfo}" ]]; then
    if [[ -n "${TREE_HMM_RESOLVED_ROCM_PATH:-}" ]]; then
      library_path="${TREE_HMM_RESOLVED_ROCM_PATH}/lib${library_path:+:${library_path}}"
    fi
    architecture="$(
      LD_LIBRARY_PATH="${library_path}" \
        "${rocminfo}" 2>/dev/null | awk \
        '/^[[:space:]]*Name:[[:space:]]+gfx[0-9]+/ { print $2; exit }'
    )"
    if [[ -z "${architecture}" ]]; then
      echo "could not determine the AMD GPU architecture with rocminfo" >&2
      return 2
    fi
  fi
  architecture="${architecture:-gfx942}"
  if [[ ! "${architecture}" =~ ^gfx[0-9]+$ ]]; then
    echo "TREE_HMM_ROCM_ARCH must look like gfx942" >&2
    return 2
  fi
  printf '%s\n' "${architecture}"
}

tree_hmm_version_at_least() {
  local installed="$1"
  local required="$2"
  awk -v installed="${installed}" -v required="${required}" '
    BEGIN {
      installed_count = split(installed, installed_parts, ".")
      required_count = split(required, required_parts, ".")
      count = installed_count > required_count ? installed_count : required_count
      for (position = 1; position <= count; ++position) {
        installed_part = installed_parts[position] + 0
        required_part = required_parts[position] + 0
        if (installed_part > required_part) exit 0
        if (installed_part < required_part) exit 1
      }
      exit 0
    }
  '
}

tree_hmm_cuda_driver_version() {
  if [[ -n "${TREE_HMM_CUDA_DRIVER_VERSION_OVERRIDE:-}" ]]; then
    printf '%s\n' "${TREE_HMM_CUDA_DRIVER_VERSION_OVERRIDE}"
    return
  fi
  nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits |
    head -n 1 | tr -d '[:space:]'
}

# CUDA 12.x minor-version compatibility requires an R525-or-newer Linux
# driver. The toolkit and driver need not have identical versions.
tree_hmm_check_cuda_driver_compatibility() {
  local minimum_driver="${TREE_HMM_CUDA_MINIMUM_DRIVER:-525.60.13}"
  local driver_version
  driver_version="$(tree_hmm_cuda_driver_version)"
  if [[ ! "${driver_version}" =~ ^[0-9]+([.][0-9]+)*$ ]]; then
    echo "could not determine the NVIDIA driver version" >&2
    return 2
  fi
  if ! tree_hmm_version_at_least "${driver_version}" "${minimum_driver}"; then
    echo "CUDA 12.8.1 requires NVIDIA driver ${minimum_driver} or newer;" \
      "found ${driver_version}" >&2
    return 2
  fi
  printf 'accelerator_compatibility backend=cuda sdk=12.8.1 driver=%s minimum_driver=%s status=compatible\n' \
    "${driver_version}" "${minimum_driver}"
}

# Resolves the pinned Bazel ROCm repository. The caller supplies the Bazel or
# Bazelisk executable and receives the resolved paths in exported variables.
tree_hmm_resolve_rocm_sdk() {
  local bazel_command="$1"
  local -a bazel_common_args=()
  local candidate
  local compiler_file
  local compiler_path=""
  local execution_root
  local output_base
  if [[ -n "${TREE_HMM_BAZEL_COMMON_ARGS:-}" ]]; then
    read -r -a bazel_common_args <<< "${TREE_HMM_BAZEL_COMMON_ARGS}"
  fi
  if [[ "${#bazel_common_args[@]}" -eq 0 ]]; then
    compiler_file="$("${bazel_command}" cquery --output=files \
      @rocm_sdk//:amdclang | tail -n 1)"
    execution_root="$("${bazel_command}" info execution_root)"
    output_base="$("${bazel_command}" info output_base)"
  else
    compiler_file="$("${bazel_command}" cquery "${bazel_common_args[@]}" \
      --output=files @rocm_sdk//:amdclang | tail -n 1)"
    execution_root="$("${bazel_command}" info "${bazel_common_args[@]}" \
      execution_root)"
    output_base="$("${bazel_command}" info "${bazel_common_args[@]}" \
      output_base)"
  fi
  for candidate in \
    "${compiler_file}" \
    "${execution_root}/${compiler_file}" \
    "${output_base}/${compiler_file}"; do
    if [[ -n "${compiler_file}" && -f "${candidate}" ]]; then
      compiler_path="${candidate}"
      break
    fi
  done
  if [[ -z "${compiler_path}" ]]; then
    compiler_path="$(
      find -L "${output_base}/external" -type f -name amdclang \
        -path '*/opt/rocm-7.2.3/*' -print -quit 2>/dev/null || true
    )"
  fi
  if [[ -z "${compiler_path}" || ! -f "${compiler_path}" ]]; then
    echo "the pinned ROCm repository contains no amdclang executable" >&2
    echo "Bazel reported artifact: ${compiler_file:-<none>}" >&2
    echo "Bazel execution root: ${execution_root}" >&2
    echo "Bazel output base: ${output_base}" >&2
    return 2
  fi
  # Preserve the amdclang entry-point name. ROCm implements amdclang and
  # amdclang++ as links to the argv[0]-dispatching amdllvm driver; resolving
  # the final link would invoke amdllvm under the wrong name.
  TREE_HMM_RESOLVED_AMDCLANG="$(
    cd "$(dirname "${compiler_path}")" && pwd -P
  )/$(basename "${compiler_path}")"
  case "${TREE_HMM_RESOLVED_AMDCLANG}" in
    */opt/rocm-7.2.3/*)
      TREE_HMM_RESOLVED_ROCM_PATH="${TREE_HMM_RESOLVED_AMDCLANG%%/opt/rocm-7.2.3/*}/opt/rocm-7.2.3"
      ;;
    *)
      echo "amdclang is outside the pinned ROCm 7.2.3 SDK:" \
        "${TREE_HMM_RESOLVED_AMDCLANG}" >&2
      return 2
      ;;
  esac
  TREE_HMM_RESOLVED_AMDCLANGXX="$(
    dirname "${TREE_HMM_RESOLVED_AMDCLANG}"
  )/amdclang++"
  if [[ ! -x "${TREE_HMM_RESOLVED_AMDCLANGXX}" ]]; then
    echo "the pinned ROCm SDK is missing amdclang++" >&2
    return 2
  fi
  TREE_HMM_RESOLVED_ROCMINFO="${TREE_HMM_RESOLVED_ROCM_PATH}/bin/rocminfo"
  if [[ ! -x "${TREE_HMM_RESOLVED_ROCMINFO}" ]]; then
    echo "the pinned ROCm SDK is missing rocminfo" >&2
    return 2
  fi
  export TREE_HMM_RESOLVED_AMDCLANG TREE_HMM_RESOLVED_AMDCLANGXX
  export TREE_HMM_RESOLVED_ROCM_PATH TREE_HMM_RESOLVED_ROCMINFO
}

# Exercising the pinned ROCm user-space runtime against /dev/kfd is a more
# reliable compatibility test than comparing package-version strings. It
# verifies that the host KMD and pinned runtime can enumerate the target GPU.
tree_hmm_check_rocm_driver_compatibility() {
  local rocm_path="$1"
  local expected_arch="$2"
  local rocminfo="${TREE_HMM_RESOLVED_ROCMINFO:-${rocm_path}/bin/rocminfo}"
  local output
  if [[ ! -e /dev/kfd &&
        "${TREE_HMM_ALLOW_MISSING_KFD_FOR_TESTS:-0}" != "1" ]]; then
    echo "ROCm requires the host AMD KMD interface /dev/kfd" >&2
    return 2
  fi
  if [[ ! -x "${rocminfo}" ]]; then
    echo "the pinned ROCm SDK is missing rocminfo at ${rocminfo}" >&2
    return 2
  fi
  if ! output="$(
    LD_LIBRARY_PATH="${rocm_path}/lib:${LD_LIBRARY_PATH:-}" \
      "${rocminfo}" 2>&1
  )"; then
    echo "ROCm 7.2.3 user space could not communicate with the host AMD driver:" \
      >&2
    printf '%s\n' "${output}" >&2
    return 2
  fi
  if ! awk -v expected="${expected_arch}" '
      $1 == "Name:" && $2 == expected { found = 1 }
      END { exit !found }
    ' <<< "${output}"; then
    echo "ROCm enumerated no ${expected_arch} device with the pinned 7.2.3 runtime" \
      >&2
    return 2
  fi
  printf 'accelerator_compatibility backend=rocm sdk=7.2.3 target=%s status=compatible\n' \
    "${expected_arch}"
}
