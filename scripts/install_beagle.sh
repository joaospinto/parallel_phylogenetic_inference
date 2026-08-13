#!/usr/bin/env bash
set -euo pipefail

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    echo "sha256sum or shasum is required" >&2
    return 1
  fi
}

logical_cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
  else
    echo 1
  fi
}

version="${BEAGLE_VERSION_LABEL:-4.1.0-pre-release-d1e9c62}"
source_revision="${BEAGLE_SOURCE_REVISION:-d1e9c62f922cf544fda4555aedf113519367c07a}"
source_url="${BEAGLE_SOURCE_URL:-https://github.com/beagle-dev/beagle-lib/archive/d1e9c62f922cf544fda4555aedf113519367c07a.tar.gz}"
archive_sha256="${BEAGLE_SOURCE_SHA256:-55da832b6cde0e65872926b312fcc9f2b03c719b2ebdaabc309e2581c5725705}"
if [[ ! "${archive_sha256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "BEAGLE_SOURCE_SHA256 must be the exact archive SHA-256" >&2
  exit 2
fi
backend="${BEAGLE_BUILD_BACKEND:-cuda}"
if [[ "${backend}" != cpu && "${backend}" != cuda ]]; then
  echo "BEAGLE_BUILD_BACKEND must be cpu or cuda" >&2
  exit 2
fi
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [installation-prefix]" >&2
  exit 2
fi
prefix="${1:-${BEAGLE_PREFIX:-${PWD}/beagle-${version}}}"
prefix="$(mkdir -p "${prefix}" && cd "${prefix}" && pwd)"
marker="${prefix}/.parallel-phylogenetics-beagle-${archive_sha256}-${backend}"
metadata="${prefix}/BEAGLE_BUILD_METADATA.txt"
if [[ -f "${marker}" ]]; then
  echo "BEAGLE ${version} is already installed in ${prefix}"
  [[ -f "${metadata}" ]] && cat "${metadata}"
  exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "CMake is required to build BEAGLE" >&2
  exit 1
fi
work_root="${BEAGLE_BUILD_ROOT:-$(dirname "${prefix}")/beagle-build}"
mkdir -p "${work_root}"
cuda_stub_directory=""
if [[ "${backend}" == cuda ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "nvcc is required to build the BEAGLE CUDA plugin" >&2
    exit 1
  fi

# BEAGLE's CUDA plugin links with -lcuda, but its CMake file does not add
# the directory containing libcuda.so.  Prefer the toolkit stub when present;
# hosted GPU environments may instead expose the unversioned driver link under
# /usr/local/nvidia or the toolkit compatibility directory.  A final fallback
# creates a build-only unversioned link to a discoverable libcuda.so.1.
nvcc_path="$(command -v nvcc)"
cuda_root="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
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
fi

build_cuda=OFF
shared_linker_flags=""
if [[ "${backend}" == cuda ]]; then
  build_cuda=ON
  shared_linker_flags="-L${cuda_stub_directory}"
fi

archive="${work_root}/beagle-lib-${version}.tar.gz"
build_directory="${work_root}/build-${archive_sha256}-${backend}"
if [[ ! -f "${archive}" ]] ||
   [[ "$(sha256_of "${archive}")" != "${archive_sha256}" ]]; then
  temporary="${archive}.download"
  curl --fail --location --silent --show-error \
    "${source_url}" \
    --output "${temporary}"
  if [[ "$(sha256_of "${temporary}")" != "${archive_sha256}" ]]; then
    echo "BEAGLE archive SHA-256 mismatch" >&2
    exit 1
  fi
  mv "${temporary}" "${archive}"
fi

archive_root="$(
  tar -tzf "${archive}" | awk -F/ '
    NF && $1 != "" {
      if (root == "") root = $1
      else if ($1 != root) exit 2
    }
    END {
      if (root == "") exit 3
      print root
    }
  '
)" || {
  echo "BEAGLE archive must contain exactly one top-level directory" >&2
  exit 1
}
if [[ ! "${archive_root}" =~ ^[A-Za-z0-9._-]+$ ||
      "${archive_root}" == "." || "${archive_root}" == ".." ]]; then
  echo "unsafe BEAGLE archive root: ${archive_root}" >&2
  exit 1
fi
extraction_directory="${work_root}/source-${archive_sha256}"
source_directory="${extraction_directory}/${archive_root}"
if [[ ! -f "${source_directory}/CMakeLists.txt" ]]; then
  mkdir -p "${extraction_directory}"
  tar -xzf "${archive}" -C "${extraction_directory}"
fi
if [[ ! -f "${source_directory}/CMakeLists.txt" ]]; then
  echo "BEAGLE archive root does not contain CMakeLists.txt" >&2
  exit 1
fi
cmake -S "${source_directory}" -B "${build_directory}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${prefix}" \
  "-DBUILD_CUDA=${build_cuda}" \
  -DBUILD_OPENCL=OFF \
  -DBUILD_JNI=OFF \
  -DBUILD_OPENMP=OFF \
  -DBUILD_BIT=OFF \
  "-DCMAKE_SHARED_LINKER_FLAGS=${shared_linker_flags}"
if [[ "${backend}" == cuda ]]; then
  LIBRARY_PATH="${cuda_stub_directory}${LIBRARY_PATH:+:${LIBRARY_PATH}}" \
    cmake --build "${build_directory}" \
      --parallel "${BEAGLE_BUILD_JOBS:-$(logical_cpu_count)}"
else
  cmake --build "${build_directory}" \
    --parallel "${BEAGLE_BUILD_JOBS:-$(logical_cpu_count)}"
fi
cmake --install "${build_directory}"
{
  echo "beagle_version_label=${version}"
  echo "beagle_source_revision=${source_revision}"
  echo "beagle_source_url=${source_url}"
  echo "beagle_source_sha256=${archive_sha256}"
  echo "beagle_build_backend=${backend}"
  echo "beagle_cmake_build_cuda=${build_cuda}"
  echo "beagle_cmake_build_opencl=OFF"
  echo "beagle_cmake_build_jni=OFF"
  echo "beagle_cmake_build_openmp=OFF"
  echo "beagle_cmake_build_bit=OFF"
} > "${metadata}"
touch "${marker}"
echo "Installed BEAGLE ${version} with ${backend} support in ${prefix}"
