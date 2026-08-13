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

work_root="${BEAGLE_BUILD_ROOT:-$(dirname "${prefix}")/beagle-build}"
mkdir -p "${work_root}"
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
  -DBUILD_BIT=OFF
cmake --build "${build_directory}" --parallel "${BEAGLE_BUILD_JOBS:-$(nproc)}"
cmake --install "${build_directory}"
touch "${marker}"
echo "Installed BEAGLE ${version} with CUDA support in ${prefix}"
