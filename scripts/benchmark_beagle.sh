#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/beagle_environment.sh
source "${script_directory}/beagle_environment.sh"
parallel_phylogenetics_configure_beagle

precision="${PRECISION:-fp64}"
if [[ "${precision}" != fp32 && "${precision}" != fp64 ]]; then
  echo "PRECISION must be fp32 or fp64" >&2
  exit 1
fi

cd "${repository}"
export BEAGLE_PREFIX
exec bazel run //:beagle_benchmark "--config=${precision}" -- "$@"
