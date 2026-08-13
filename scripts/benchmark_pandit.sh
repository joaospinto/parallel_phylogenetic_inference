#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {cuda|metal|beagle-cpu|beagle-cuda} [families-directory]" >&2
  exit 2
fi
backend="$1"
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
families="${2:-${repository}/data/pandit/families}"
manifest="${families}/manifest.csv"
if [[ ! -f "${manifest}" ]]; then
  echo "PANDIT manifest not found at ${manifest}" >&2
  echo "Run scripts/fetch_pandit.sh first or pass its families directory." >&2
  exit 1
fi

precision="${PRECISION:-fp64}"
minimum_leaves="${PANDIT_MIN_LEAVES:-100}"
limit="${PANDIT_LIMIT:-0}"
repeats="${TREE_HMM_EMPIRICAL_REPEATS:-3}"
conditioning_ms="${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
threads="${BEAGLE_THREADS:-1}"
resume_report="${TREE_HMM_RESUME_REPORT:-}"
for value in "${minimum_leaves}" "${repeats}" "${threads}"; do
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "leaf, repeat, and thread counts must be positive integers" >&2
    exit 2
  fi
done
for value in "${limit}" "${conditioning_ms}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "limit and conditioning duration must be nonnegative integers" >&2
    exit 2
  fi
done
if [[ "${precision}" != fp32 && "${precision}" != fp64 ]]; then
  echo "PRECISION must be fp32 or fp64" >&2
  exit 2
fi
if [[ -n "${resume_report}" && ! -r "${resume_report}" ]]; then
  echo "TREE_HMM_RESUME_REPORT is not readable: ${resume_report}" >&2
  exit 2
fi

script_directory="${repository}/scripts"
# shellcheck source=scripts/benchmark_resume.sh
source "${script_directory}/benchmark_resume.sh"

case "${backend}" in
  beagle-cpu|beagle-cuda)
    # shellcheck source=scripts/beagle_environment.sh
    source "${script_directory}/beagle_environment.sh"
    parallel_phylogenetics_configure_beagle
    target=beagle_benchmark
    ;;
  cuda)
    target=cuda_benchmark
    ;;
  metal)
    if [[ "${precision}" != fp32 ]]; then
      echo "Metal only supports PRECISION=fp32" >&2
      exit 2
    fi
    target=metal_benchmark
    ;;
  *)
    echo "unsupported backend ${backend}" >&2
    exit 2
    ;;
esac

cd "${repository}"
if [[ "${PANDIT_SKIP_BUILD:-0}" != 1 ]]; then
  build_arguments=("--config=${precision}")
  if [[ "${backend}" == cuda ]]; then
    if [[ -z "${TREE_HMM_CUDA_ARCH:-}" ]]; then
      echo "TREE_HMM_CUDA_ARCH is required when building the CUDA runner" >&2
      exit 2
    fi
    build_arguments+=(--config=cuda "--cuda_archs=sm_${TREE_HMM_CUDA_ARCH}")
  fi
  bazel build "//:${target}" "${build_arguments[@]}"
fi

echo "# corpus=PANDIT-17.0"
echo "# corpus_backend=${backend}"
echo "# manifest=${manifest}"
echo "# minimum_leaves=${minimum_leaves}"
echo "# family_limit=${limit}"
selected=0
precision_label="$(tr '[:lower:]' '[:upper:]' <<< "${precision}")"
while IFS=, read -r family leaves raw_sites selected_sites; do
  family="${family%$'\r'}"
  leaves="${leaves%$'\r'}"
  if [[ "${family}" == family || "${leaves}" -lt "${minimum_leaves}" ]]; then
    continue
  fi
  if [[ "${limit}" -ne 0 && "${selected}" -ge "${limit}" ]]; then
    break
  fi
  if benchmark_resume_dataset_completed "${resume_report}" "${backend}" \
    "${precision_label}" "${family}"; then
    echo "# resume_skip method=${backend} precision=${precision_label}" \
      "dataset=${family}"
    selected=$((selected + 1))
    continue
  fi
  arguments=(
    --newick "${families}/${family}.nwk"
    --fasta "${families}/${family}.fasta"
    --repeats "${repeats}"
    --conditioning-ms "${conditioning_ms}"
  )
  case "${backend}" in
    beagle-cpu)
      arguments+=(--beagle-resource cpu --beagle-threads "${threads}")
      ;;
    beagle-cuda)
      arguments+=(--beagle-resource cuda --beagle-threads 1)
      ;;
  esac
  "bazel-bin/${target}" "${arguments[@]}"
  selected=$((selected + 1))
done < "${manifest}"
echo "# selected_families=${selected}"
