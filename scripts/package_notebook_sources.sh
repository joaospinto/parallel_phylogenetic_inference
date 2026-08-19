#!/usr/bin/env bash
set -euo pipefail

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print $1 }'
  else
    shasum -a 256 "$1" | awk '{ print $1 }'
  fi
}

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="${1:-${repo_dir}/parallel_tree_inference_sources.zip}"
resume_report="${2:-}"
mkdir -p "$(dirname "${output}")"
output="$(cd "$(dirname "${output}")" && pwd)/$(basename "${output}")"
staging="$(mktemp -d "${TMPDIR:-/tmp}/tree-inference-package.XXXXXX")"
dependency_staging="$(
  mktemp -d "${TMPDIR:-/tmp}/tree-inference-dependencies.XXXXXX"
)"
archive_staging="$(
  mktemp -d "$(dirname "${output}")/.tree-inference-archive.XXXXXX"
)"
trap 'rm -rf "${staging}" "${dependency_staging}" "${archive_staging}"' EXIT

if [[ -n "$(git -C "${repo_dir}" status --porcelain)" ]]; then
  echo "${repo_dir} has uncommitted files; commit them before packaging" >&2
  exit 2
fi
git -C "${repo_dir}" archive --format=tar \
  --prefix="parallel_phylogenetic_inference/" HEAD | tar -xf - -C "${staging}"
printf '%s %s\n' parallel_phylogenetic_inference \
  "$(git -C "${repo_dir}" rev-parse HEAD)" > \
  "${staging}/SOURCE_REVISIONS.txt"

dependencies=(
  "parallel_tree_hmm|https://github.com/joaospinto/parallel_tree_hmm.git|af7cc473a21451a86c3b868f9525d5210c9b60e8"
  "bidirectional_tree_rake_compress|https://github.com/joaospinto/bidirectional_tree_rake_compress.git|36cfd7592a653011ac36ffd8e3d918acc59a2e05"
)
for dependency in "${dependencies[@]}"; do
  IFS='|' read -r repository remote revision <<< "${dependency}"
  source_dir="${dependency_staging}/${repository}"
  git clone --quiet --filter=blob:none --no-checkout "${remote}" "${source_dir}"
  git -C "${source_dir}" cat-file -e "${revision}^{commit}"
  git -C "${source_dir}" archive --format=tar \
    --prefix="${repository}/" "${revision}" | tar -xf - -C "${staging}"
  printf '%s %s\n' "${repository}" "${revision}" >> \
    "${staging}/SOURCE_REVISIONS.txt"
done

if [[ -n "${resume_report}" ]]; then
  if [[ ! -r "${resume_report}" ]]; then
    echo "resume report is not readable: ${resume_report}" >&2
    exit 2
  fi
  cp "${resume_report}" "${staging}/PREVIOUS_BENCHMARK_REPORT.txt"
  printf '%s  %s\n' "$(sha256_file "${resume_report}")" \
    PREVIOUS_BENCHMARK_REPORT.txt > \
      "${staging}/PREVIOUS_BENCHMARK_REPORT.sha256"
fi

(
  cd "${staging}"
  zip -q -r "${archive_staging}/$(basename "${output}")" .
)
mv "${archive_staging}/$(basename "${output}")" "${output}"
echo "${output}"
