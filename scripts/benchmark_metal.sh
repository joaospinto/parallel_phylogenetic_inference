#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repeats="${TREE_HMM_BENCHMARK_REPEATS:-5}"
if [[ ! "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TREE_HMM_BENCHMARK_REPEATS must be a positive integer" >&2
  exit 2
fi

cd "${repo_dir}"
bazel build //:metal_benchmark
echo "=== Apple device ==="
system_profiler SPHardwareDataType
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
for topology in balanced caterpillar; do
  for specification in "${cases[@]}"; do
    read -r leaves sites <<< "${specification}"
    bazel-bin/metal_benchmark \
      --topology "${topology}" \
      --leaves "${leaves}" \
      --sites "${sites}" \
      --repeats "${repeats}"
  done
done
