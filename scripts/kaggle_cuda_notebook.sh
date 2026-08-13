#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TREE_HMM_ACCELERATOR_BACKEND_OVERRIDE=cuda exec \
  bash "${repo_dir}/scripts/kaggle_accelerator_notebook.sh" "$@"
