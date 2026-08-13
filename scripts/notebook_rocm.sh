#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE_HMM_ACCELERATOR_BACKEND=rocm exec \
  bash "${repo_dir}/scripts/notebook_gpu.sh" "$@"
