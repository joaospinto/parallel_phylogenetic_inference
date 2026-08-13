#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent_dir="$(dirname "${repo_dir}")"
worktree_dir="$(dirname "${parent_dir}")/worktrees/parallel_phylogenetics_corpora"
output="${1:-${worktree_dir}/parallel_tree_inference_sources.zip}"
resume_report="${2:-}"
mkdir -p "$(dirname "${output}")"
output="$(cd "$(dirname "${output}")" && pwd)/$(basename "${output}")"
staging="$(mktemp -d "${TMPDIR:-/tmp}/tree-inference-package.XXXXXX")"
trap 'rm -rf "${staging}"' EXIT

for repository in parallel_phylogenetic_inference parallel_tree_hmm \
                  bidirectional_tree_rake_compress; do
  source_dir="${parent_dir}/${repository}"
  if [[ ! -d "${source_dir}/.git" ]]; then
    echo "missing Git repository ${source_dir}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${source_dir}" status --porcelain)" ]]; then
    echo "${source_dir} has uncommitted files; commit them before packaging" >&2
    exit 2
  fi
  git -C "${source_dir}" archive --format=tar \
    --prefix="${repository}/" HEAD | tar -xf - -C "${staging}"
  printf '%s %s\n' "${repository}" \
    "$(git -C "${source_dir}" rev-parse HEAD)" >> \
    "${staging}/SOURCE_REVISIONS.txt"
done

if [[ -n "${resume_report}" ]]; then
  if [[ ! -r "${resume_report}" ]]; then
    echo "resume report is not readable: ${resume_report}" >&2
    exit 2
  fi
  cp "${resume_report}" "${staging}/PREVIOUS_BENCHMARK_REPORT.txt"
  shasum -a 256 "${resume_report}" > \
    "${staging}/PREVIOUS_BENCHMARK_REPORT.sha256"
fi

(
  cd "${staging}"
  zip -q -r "${output}" .
)
echo "${output}"
