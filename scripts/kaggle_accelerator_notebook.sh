#!/usr/bin/env bash
set -euo pipefail

notebook_input_dir="${TREE_HMM_NOTEBOOK_INPUT_DIR:-/kaggle/input}"
notebook_working_dir="${TREE_HMM_NOTEBOOK_WORKING_DIR:-/kaggle/working}"

sha256_stream() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | awk '{ print $1 }'
  else
    shasum -a 256 | awk '{ print $1 }'
  fi
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{ print $1 }'
  else
    shasum -a 256 "$1" | awk '{ print $1 }'
  fi
}
source_input="${1:-}"
if [[ -z "${source_input}" ]]; then
  source_input="$(find "${notebook_input_dir}" -type f \
    -name 'parallel_tree_inference_sources.zip' -print -quit || true)"
fi
if [[ -z "${source_input}" || ! -r "${source_input}" ]]; then
  echo "attach the parallel_tree_inference_sources input before running" >&2
  find "${notebook_input_dir}" -maxdepth 3 -print >&2
  exit 2
fi

work="${notebook_working_dir}/parallel-tree-inference"
rm -rf "${work}"
mkdir -p "${work}"
if [[ -f "${source_input}" ]]; then
  echo "source bundle SHA-256: $(sha256_file "${source_input}")"
  unzip -q "${source_input}" -d "${work}"
elif [[ -d "${source_input}/parallel_phylogenetic_inference" ]]; then
  echo "using unpacked source input ${source_input}"
  for repository in bidirectional_tree_rake_compress parallel_tree_hmm \
                    parallel_phylogenetic_inference; do
    if [[ ! -d "${source_input}/${repository}" ]]; then
      echo "the attached input is missing ${repository}" >&2
      exit 2
    fi
    cp -R "${source_input}/${repository}" "${work}/${repository}"
  done
  for metadata in SOURCE_REVISIONS.txt PREVIOUS_BENCHMARK_REPORT.txt \
                  PREVIOUS_BENCHMARK_REPORT.sha256; do
    if [[ -f "${source_input}/${metadata}" ]]; then
      cp "${source_input}/${metadata}" "${work}/${metadata}"
    fi
  done
else
  echo "source input must be the source ZIP or its unpacked root" >&2
  exit 2
fi

if [[ ! -r "${work}/SOURCE_REVISIONS.txt" ]]; then
  echo "the source input is missing SOURCE_REVISIONS.txt" >&2
  exit 2
fi
script_dir="${work}/parallel_phylogenetic_inference/scripts"
# shellcheck source=scripts/accelerator_environment.sh
source "${script_dir}/accelerator_environment.sh"
accelerator_backend="${TREE_HMM_ACCELERATOR_BACKEND_OVERRIDE:-}"
if [[ -z "${accelerator_backend}" ]]; then
  if tree_hmm_nvidia_available; then
    accelerator_backend=cuda
  elif tree_hmm_amd_available; then
    accelerator_backend=rocm
  else
    echo "no supported NVIDIA or AMD GPU was detected" >&2
    exit 2
  fi
fi
if [[ "${accelerator_backend}" != cuda &&
      "${accelerator_backend}" != rocm ]]; then
  echo "TREE_HMM_ACCELERATOR_BACKEND_OVERRIDE must be cuda or rocm" >&2
  exit 2
fi
echo "selected notebook accelerator: ${accelerator_backend}"
report="${notebook_working_dir}/parallel_phylogenetics_${accelerator_backend}_report.txt"
cache_identity="$(
  {
    echo "parallel-phylogenetics-benchmark-schema=3"
    echo "accelerator-backend=${accelerator_backend}"
    sort "${work}/SOURCE_REVISIONS.txt"
  } | sha256_stream
)"
cache_marker="# cache_identity sha256=${cache_identity}"

report_matches_sources() {
  local candidate="$1"
  [[ -s "${candidate}" ]] && grep -Fqx "${cache_marker}" "${candidate}"
}

if report_matches_sources "${report}"; then
  echo "resuming from $(wc -l < "${report}") lines in the working report"
else
  if [[ -s "${report}" ]]; then
    stale_report="${report%.txt}.stale.txt"
    cp "${report}" "${stale_report}"
    echo "existing report has a different source identity; preserved as" \
      "${stale_report}"
  fi
  prior_report="$(find "${notebook_input_dir}" -type f \
    -name "parallel_phylogenetics_${accelerator_backend}_report*.txt" \
    -print -quit || true)"
  if [[ -z "${prior_report}" &&
        -f "${work}/PREVIOUS_BENCHMARK_REPORT.txt" ]]; then
    prior_report="${work}/PREVIOUS_BENCHMARK_REPORT.txt"
  fi
  if [[ -n "${prior_report}" ]] && report_matches_sources "${prior_report}"; then
    cp "${prior_report}" "${report}"
    echo "resuming from $(wc -l < "${prior_report}") previously recorded lines"
  else
    if [[ -n "${prior_report}" ]]; then
      echo "ignoring prior report with a different source identity"
    fi
    printf '%s\n' "${cache_marker}" > "${report}"
  fi
fi

cd "${work}/parallel_phylogenetic_inference"
TREE_HMM_PRECISIONS="${TREE_HMM_PRECISIONS_OVERRIDE:-FP64 FP32}" \
  TREE_HMM_RESUME_REPORT="${report}" \
  TREE_HMM_BENCHMARK_REPEATS="${TREE_HMM_BENCHMARK_REPEATS:-15}" \
  TREE_HMM_EMPIRICAL_REPEATS="${TREE_HMM_EMPIRICAL_REPEATS:-3}" \
  bash "scripts/notebook_${accelerator_backend}.sh" 2>&1 | tee -a "${report}"
