#!/usr/bin/env bash
set -euo pipefail

version=4.0.1
archive_sha256=9d258cd9bedd86d7c28b91587acd1132f4e01d4f095c657ad4dc93bd83d4f120
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [installation-prefix]" >&2
  exit 2
fi
prefix="${1:-${BEAGLE_PREFIX:-${PWD}/beagle-${version}}}"
prefix="$(mkdir -p "${prefix}" && cd "${prefix}" && pwd)"
marker="${prefix}/.parallel-phylogenetics-beagle-${version}"
if [[ -f "${marker}" ]]; then
  echo "BEAGLE ${version} is already installed in ${prefix}"
  exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake is required to build BEAGLE" >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build the BEAGLE CUDA plugin" >&2
  exit 1
fi

# BEAGLE 4.0.1 links its plugin with -lcuda, but its CMake file does not add
# the directory containing libcuda.so.  Prefer the toolkit stub when present;
# hosted GPU environments may instead expose the unversioned driver link under
# /usr/local/nvidia or the toolkit compatibility directory.  A final fallback
# creates a build-only unversioned link to a discoverable libcuda.so.1.
nvcc_path="$(command -v nvcc)"
cuda_root="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
work_root="${BEAGLE_BUILD_ROOT:-$(dirname "${prefix}")/beagle-build}"
mkdir -p "${work_root}"
cuda_stub_directory="${CUDA_STUB_DIRECTORY:-}"
if [[ -z "${cuda_stub_directory}" ]]; then
  for candidate in \
    "${cuda_root}/lib64/stubs" \
    "${cuda_root}/targets/x86_64-linux/lib/stubs" \
    "${cuda_root}/lib/stubs" \
    /usr/local/nvidia/lib64 \
    /usr/local/nvidia/lib \
    "${cuda_root}/compat" \
    /usr/local/cuda/compat; do
    if [[ -f "${candidate}/libcuda.so" ]]; then
      cuda_stub_directory="${candidate}"
      break
    fi
  done
fi
if [[ -z "${cuda_stub_directory}" ]]; then
  cuda_driver_library="${CUDA_DRIVER_LIBRARY:-}"
  if [[ -z "${cuda_driver_library}" ]]; then
    IFS=: read -r -a cuda_library_paths <<< "${LD_LIBRARY_PATH:-}"
    for directory in "${cuda_library_paths[@]}"; do
      if [[ -f "${directory}/libcuda.so" ]]; then
        cuda_stub_directory="${directory}"
        break
      fi
      if [[ -f "${directory}/libcuda.so.1" ]]; then
        cuda_driver_library="${directory}/libcuda.so.1"
        break
      fi
    done
  fi
  if [[ -z "${cuda_driver_library}" ]] && command -v ldconfig >/dev/null 2>&1; then
    cuda_driver_library="$(
      ldconfig -p 2>/dev/null |
        awk '$1 == "libcuda.so.1" { print $NF; exit }'
    )"
  fi
  if [[ -z "${cuda_driver_library}" ]]; then
    for candidate in \
      /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
      /lib/x86_64-linux-gnu/libcuda.so.1 \
      /usr/lib64/libcuda.so.1 \
      /usr/local/cuda/compat/libcuda.so.1 \
      /usr/local/nvidia/lib64/libcuda.so.1 \
      /usr/local/nvidia/lib/libcuda.so.1 \
      /usr/lib/wsl/lib/libcuda.so.1; do
      if [[ -f "${candidate}" ]]; then
        cuda_driver_library="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${cuda_stub_directory}" && -z "${cuda_driver_library}" ]]; then
    cuda_driver_library="$(
      find -L /usr /lib /opt -type f \
        \( -name 'libcuda.so.1' -o -name 'libcuda.so.[0-9]*' \) \
        -print -quit 2>/dev/null || true
    )"
  fi
  if [[ -z "${cuda_stub_directory}" && -f "${cuda_driver_library}" ]]; then
    cuda_stub_directory="${work_root}/cuda-driver-link"
    mkdir -p "${cuda_stub_directory}"
    ln -sfn "${cuda_driver_library}" "${cuda_stub_directory}/libcuda.so"
  fi
fi
if [[ ! -f "${cuda_stub_directory}/libcuda.so" ]]; then
  echo "could not find libcuda.so or libcuda.so.1" >&2
  echo "CUDA root: ${cuda_root}" >&2
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset>}" >&2
  find -L /usr/local /usr/lib /lib /opt -maxdepth 6 \
    -name 'libcuda.so*' -print 2>/dev/null >&2 || true
  echo "set CUDA_STUB_DIRECTORY or CUDA_DRIVER_LIBRARY explicitly" >&2
  exit 1
fi

archive="${work_root}/beagle-lib-${version}.tar.gz"
source_directory="${work_root}/beagle-lib-${version}"
build_directory="${work_root}/build-${version}"
if [[ ! -f "${archive}" ]] ||
   ! printf '%s  %s\n' "${archive_sha256}" "${archive}" |
     sha256sum --check --status; then
  temporary="${archive}.download"
  curl --fail --location --silent --show-error \
    "https://github.com/beagle-dev/beagle-lib/archive/refs/tags/v${version}.tar.gz" \
    --output "${temporary}"
  printf '%s  %s\n' "${archive_sha256}" "${temporary}" |
    sha256sum --check --status
  mv "${temporary}" "${archive}"
fi

if [[ ! -f "${source_directory}/CMakeLists.txt" ]]; then
  tar -xzf "${archive}" -C "${work_root}"
fi
cmake -S "${source_directory}" -B "${build_directory}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DBUILD_CUDA=ON \
  -DBUILD_OPENCL=OFF \
  -DBUILD_JNI=OFF \
  -DBUILD_OPENMP=OFF \
  -DBUILD_BIT=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS="-L${cuda_stub_directory}"
LIBRARY_PATH="${cuda_stub_directory}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
  cmake --build "${build_directory}" \
    --parallel "${BEAGLE_BUILD_JOBS:-$(nproc)}"
cmake --install "${build_directory}"
touch "${marker}"
echo "Installed BEAGLE ${version} with CUDA support in ${prefix}"
