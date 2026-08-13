#!/usr/bin/env bash
set -euo pipefail

notebook_input_dir="${TREE_HMM_NOTEBOOK_INPUT_DIR:-/kaggle/input}"
notebook_working_dir="${TREE_HMM_NOTEBOOK_WORKING_DIR:-/kaggle/working}"
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
report="${notebook_working_dir}/parallel_phylogenetics_cuda_report.txt"
rm -rf "${work}"
mkdir -p "${work}"
if [[ -f "${source_input}" ]]; then
  echo "source bundle SHA-256: $(sha256sum "${source_input}" | cut -d' ' -f1)"
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

if [[ -s "${report}" ]]; then
  echo "resuming from $(wc -l < "${report}") lines in the working report"
else
  prior_report="$(find "${notebook_input_dir}" -type f \
    -name 'parallel_phylogenetics_cuda_report*.txt' -print -quit || true)"
  if [[ -z "${prior_report}" &&
        -f "${work}/PREVIOUS_BENCHMARK_REPORT.txt" ]]; then
    prior_report="${work}/PREVIOUS_BENCHMARK_REPORT.txt"
  fi
  if [[ -n "${prior_report}" ]]; then
    cp "${prior_report}" "${report}"
    echo "resuming from $(wc -l < "${prior_report}") previously recorded lines"
  else
    : > "${report}"
  fi
fi

cd "${work}/parallel_phylogenetic_inference"
TREE_HMM_PRECISIONS="${TREE_HMM_PRECISIONS_OVERRIDE:-FP64 FP32}" \
  TREE_HMM_RESUME_REPORT="${report}" \
  TREE_HMM_BENCHMARK_REPEATS="${TREE_HMM_BENCHMARK_REPEATS:-15}" \
  TREE_HMM_EMPIRICAL_REPEATS="${TREE_HMM_EMPIRICAL_REPEATS:-3}" \
  bash scripts/notebook_cuda.sh 2>&1 | tee -a "${report}"
