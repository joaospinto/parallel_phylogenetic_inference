#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAGLE_BUILD_BACKEND=cuda exec \
  bash "${repo_dir}/scripts/install_beagle.sh" "$@"
