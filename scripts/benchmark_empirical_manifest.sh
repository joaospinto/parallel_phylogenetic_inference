#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {cuda|rocm|metal|beagle-cpu|beagle-cuda} manifest.csv" >&2
  exit 2
fi
method="$1"
manifest="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
root="$(dirname "${manifest}")"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script_directory="${repository}/scripts"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"

precision="${PRECISION:-fp32}"
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
threads="${BEAGLE_THREADS:-1}"
benchmark_mode="${TREE_HMM_BENCHMARK_MODE:-full-input-update}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"

case "${method}" in
  cuda|rocm|metal) target="${method}_benchmark"; extra=(); resume_threads="" ;;
  beagle-cpu) target=beagle_benchmark; extra=(--beagle-resource cpu --beagle-threads "${threads}" --benchmark-mode "${benchmark_mode}"); resume_threads="${threads}" ;;
  beagle-cuda) target=beagle_benchmark; extra=(--beagle-resource cuda --beagle-threads 1 --benchmark-mode "${benchmark_mode}"); resume_threads=1 ;;
  *) echo "unsupported method ${method}" >&2; exit 2 ;;
esac

echo "# corpus_doi=10.5061/dryad.8gtht76zz"
echo "# corpus_version=6"
echo "# corpus_license=CC0-1.0"
echo "# source_file_id=4142269"
echo "# source_archive_sha256=06cee5bd75748acf5ba95a10b404b2867dd0b52a3e9e1b9ec357f9d9c7e09f4c"
echo "# selection_rule=all 222 DNA alignment.phy entries; maximum-logLikelihood version=standard tree; no timing filter"
echo "# pattern_compression=exact duplicate columns, prepared before benchmarking"
echo "# benchmark_mode=${benchmark_mode}"
cd "${repository}"
while IFS=, read -r dataset taxa raw_sites unique_patterns alignment \
  pattern_weights tree rest; do
  dataset="${dataset%$'\r'}"
  [[ "${dataset}" == dataset ]] && continue
  if benchmark_resume_dataset_completed "${resume_report}" "${method}" \
    "${precision_label}" "${dataset}" "${benchmark_mode}" \
    "${resume_threads}"; then
    echo "# resume_skip method=${method} precision=${precision_label} dataset=${dataset}"
    continue
  fi
  echo "# benchmark_start method=${method} precision=${precision_label} dataset=${dataset} taxa=${taxa} raw_sites=${raw_sites} unique_patterns=${unique_patterns}"
  "bazel-bin/${target}" "${extra[@]}" \
    --newick "${root}/${tree}" --fasta "${root}/${alignment}" \
    --pattern-weights "${root}/${pattern_weights}" --dataset-label "${dataset}" \
    --repeats "${repeats}"
done < "${manifest}"
