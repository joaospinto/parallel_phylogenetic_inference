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

if [[ -z "${TREE_HMM_EMPIRICAL_MANIFESTS:-}" ]]; then
  corpus_root="${work}/attached-corpora"
  mkdir -p "${corpus_root}"
  archive_index=0
  while IFS= read -r archive; do
    archive_index=$((archive_index + 1))
    destination="${corpus_root}/archive-${archive_index}"
    mkdir -p "${destination}"
    echo "extracting attached empirical corpus $(basename "${archive}")"
    unzip -q "${archive}" -d "${destination}"
  done < <(find "${notebook_input_dir}" -type f \
    -name 'parallel_phylogenetics_corpus_*.zip' -print | sort)
  empirical_manifests=()
  while IFS= read -r candidate; do
    if [[ -r "$(dirname "${candidate}")/corpus_metadata.txt" ]]; then
      empirical_manifests+=("${candidate}")
    fi
  done < <(find "${notebook_input_dir}" "${corpus_root}" -type f \
    -name manifest.csv -print 2>/dev/null | sort -u)
  if [[ "${#empirical_manifests[@]}" != 0 ]]; then
    export TREE_HMM_EMPIRICAL_MANIFESTS="${empirical_manifests[*]}"
    if [[ -z "${TREE_HMM_BENCHMARK_SECTIONS:-}" ]]; then
      if [[ "${TREE_HMM_BENCHMARK_PROFILE:-curated}" == complete ]]; then
        TREE_HMM_BENCHMARK_SECTIONS="validation synthetic distributions jc69 tasks fish pandit empirical"
      else
        TREE_HMM_BENCHMARK_SECTIONS="validation synthetic fish pandit empirical"
      fi
      export TREE_HMM_BENCHMARK_SECTIONS
    fi
    echo "detected ${#empirical_manifests[@]} attached empirical corpus manifest(s)"
  fi
fi

empirical_manifest_identity=none
if [[ -n "${TREE_HMM_EMPIRICAL_MANIFESTS:-}" ]]; then
  read -r -a empirical_manifests <<< "${TREE_HMM_EMPIRICAL_MANIFESTS}"
  empirical_manifest_identity="$({
    for manifest in "${empirical_manifests[@]}"; do
      if [[ ! -r "${manifest}" ]]; then
        echo "empirical manifest is not readable: ${manifest}" >&2
        exit 2
      fi
      echo "manifest=$(sha256_file "${manifest}")"
      metadata="$(dirname "${manifest}")/corpus_metadata.txt"
      if [[ -r "${metadata}" ]]; then
        echo "metadata=$(sha256_file "${metadata}")"
      else
        echo "metadata=missing"
      fi
    done
  } | sort | sha256_stream)"
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
resume_scope="${TREE_HMM_RESUME_SCOPE:-session}"
case "${resume_scope}" in
  session|hardware-class) ;;
  *)
    echo "TREE_HMM_RESUME_SCOPE must be session or hardware-class" >&2
    exit 2
    ;;
esac
if [[ -n "${TREE_HMM_HARDWARE_IDENTITY_OVERRIDE:-}" ]]; then
  hardware_identity="${TREE_HMM_HARDWARE_IDENTITY_OVERRIDE}"
elif [[ "${accelerator_backend}" == cuda ]] && command -v nvidia-smi >/dev/null; then
  hardware_identity="$(nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader -i 0 | head -1)"
elif [[ "${accelerator_backend}" == rocm ]]; then
  hardware_identity="$(tree_hmm_detect_rocm_arch)"
else
  hardware_identity=unavailable
fi
session_identity="not-included"
if [[ "${resume_scope}" == session ]]; then
  if [[ -n "${TREE_HMM_SESSION_IDENTITY_OVERRIDE:-}" ]]; then
    session_identity="${TREE_HMM_SESSION_IDENTITY_OVERRIDE}"
  else
    host_session="$(hostname)"
    if [[ -r /proc/sys/kernel/random/boot_id ]]; then
      host_session="${host_session},$(</proc/sys/kernel/random/boot_id)"
    fi
    accelerator_session=unavailable
    if [[ "${accelerator_backend}" == cuda ]] &&
       command -v nvidia-smi >/dev/null; then
      accelerator_session="$(nvidia-smi --query-gpu=uuid \
        --format=csv,noheader -i 0 | head -1)"
    elif [[ "${accelerator_backend}" == rocm ]] &&
         command -v rocm-smi >/dev/null; then
      accelerator_session="$(rocm-smi --showuniqueid --csv 2>/dev/null | \
        tail -n +2 | head -1 || true)"
    fi
    session_identity="${host_session},${accelerator_session}"
  fi
fi
echo "benchmark resume scope: ${resume_scope}"
if [[ "${resume_scope}" == hardware-class ]]; then
  echo "hardware-class resume explicitly permits reuse across equivalent workers"
else
  echo "session resume rejects reports from a different host/GPU session"
fi
if command -v nproc >/dev/null 2>&1; then
  logical_core_count="$(nproc)"
else
  logical_core_count="$(getconf _NPROCESSORS_ONLN)"
fi
cache_identity="$(
  {
    benchmark_profile="${TREE_HMM_BENCHMARK_PROFILE:-curated}"
    if [[ "${benchmark_profile}" == complete ]]; then
      default_sections="validation synthetic distributions jc69 tasks fish pandit"
      default_modes="full-input-update factor-update fixed-model"
      default_synthetic_modes="${default_modes}"
      default_task_modes="${default_modes}"
      default_pandit_limit=0
    else
      default_sections="validation synthetic fish pandit"
      default_modes="full-input-update"
      default_synthetic_modes="full-input-update factor-update fixed-model"
      default_task_modes="full-input-update"
      default_pandit_limit=25
    fi
    echo "parallel-phylogenetics-benchmark-schema=8"
    echo "benchmark-profile=${benchmark_profile}"
    echo "accelerator-backend=${accelerator_backend}"
    echo "hardware=${hardware_identity}"
    echo "resume-scope=${resume_scope}"
    echo "session-identity=${session_identity}"
    echo "host-cpu=$(grep -m1 -E 'model name|Hardware' /proc/cpuinfo 2>/dev/null || uname -m)"
    echo "precisions=${TREE_HMM_PRECISIONS_OVERRIDE:-FP64 FP32}"
    echo "sections=${TREE_HMM_BENCHMARK_SECTIONS:-${default_sections}}"
    echo "modes=${TREE_HMM_BENCHMARK_MODES:-${default_modes}}"
    echo "synthetic-modes=${TREE_HMM_SYNTHETIC_BENCHMARK_MODES:-${default_synthetic_modes}}"
    echo "fish-modes=${TREE_HMM_FISH_BENCHMARK_MODES:-${default_modes}}"
    echo "pandit-modes=${TREE_HMM_PANDIT_BENCHMARK_MODES:-${default_modes}}"
    echo "task-modes=${TREE_HMM_TASK_BENCHMARK_MODES:-${default_task_modes}}"
    echo "jc69-modes=${TREE_HMM_JC69_BENCHMARK_MODES:-full-input-update}"
    echo "jc69-profile=${TREE_HMM_JC69_PROFILE:-paper}"
    echo "jc69-topologies-override=${TREE_HMM_JC69_TOPOLOGIES:-unset}"
    echo "jc69-leaves-override=${TREE_HMM_JC69_LEAVES:-unset}"
    echo "jc69-raw-sites-override=${TREE_HMM_JC69_RAW_SITES:-unset}"
    echo "jc69-heights-override=${TREE_HMM_JC69_EVOLUTIONARY_HEIGHTS:-unset}"
    echo "jc69-replicates-override=${TREE_HMM_JC69_REPLICATES:-unset}"
    echo "jc69-seed=${TREE_HMM_JC69_SEED:-20260814}"
    echo "jc69-minimum-repeats-override=${TREE_HMM_JC69_MINIMUM_TIMING_REPEATS:-unset}"
    echo "jc69-maximum-repeats-override=${TREE_HMM_JC69_MAXIMUM_TIMING_REPEATS:-unset}"
    echo "jc69-work-budget-override=${TREE_HMM_JC69_TIMING_WORK_BUDGET:-unset}"
    echo "jc69-minimum-site-batch=${TREE_HMM_JC69_MINIMUM_SITE_BATCH:-128}"
    echo "empirical-modes=${TREE_HMM_EMPIRICAL_BENCHMARK_MODES:-full-input-update}"
    echo "empirical-manifests=${empirical_manifest_identity}"
    echo "pandit-limit=${PANDIT_LIMIT:-${default_pandit_limit}}"
    echo "repeats=${TREE_HMM_BENCHMARK_REPEATS:-15}"
    echo "empirical-repeats=${TREE_HMM_EMPIRICAL_REPEATS:-3}"
    echo "empirical-minimum-branch-length=${TREE_HMM_EMPIRICAL_MINIMUM_BRANCH_LENGTH:-0.000001}"
    echo "conditioning-ms=${TREE_HMM_BENCHMARK_CONDITIONING_MS:-0}"
    echo "beagle-version-label=${BEAGLE_VERSION_LABEL:-4.1.0-pre-release-d1e9c62}"
    echo "beagle-source-revision=${BEAGLE_SOURCE_REVISION:-d1e9c62f922cf544fda4555aedf113519367c07a}"
    echo "beagle-source-url=${BEAGLE_SOURCE_URL:-https://github.com/beagle-dev/beagle-lib/archive/d1e9c62f922cf544fda4555aedf113519367c07a.tar.gz}"
    echo "beagle-source-sha256=${BEAGLE_SOURCE_SHA256:-55da832b6cde0e65872926b312fcc9f2b03c719b2ebdaabc309e2581c5725705}"
    echo "beagle-build-jobs=${BEAGLE_BUILD_JOBS:-${logical_core_count}}"
    echo "beagle-cmake-build-opencl=OFF"
    echo "beagle-cmake-build-jni=OFF"
    echo "beagle-cmake-build-openmp=OFF"
    echo "beagle-cmake-build-bit=OFF"
    echo "beagle-cpu-threads=${TREE_HMM_BEAGLE_CPU_THREADS:-1 ${logical_core_count}}"
    echo "fish-minimum-site-batch=${TREE_HMM_FISH_MINIMUM_SITE_BATCH:-256}"
    echo "host-memory-guard-percent=${TREE_HMM_HOST_MEMORY_GUARD_PERCENT:-75}"
    echo "host-memory-guard-kib=${TREE_HMM_HOST_MEMORY_GUARD_KIB:-automatic}"
    echo "pandit-minimum-leaves=${PANDIT_MIN_LEAVES:-100}"
    echo "distribution-topologies=${TREE_HMM_DISTRIBUTION_TOPOLOGIES:-yule beta-critical uniform caterpillar}"
    echo "distribution-leaves=${TREE_HMM_DISTRIBUTION_LEAVES:-128 512 2048 8192}"
    echo "distribution-patterns=${TREE_HMM_DISTRIBUTION_PATTERNS:-16 64 256 1024}"
    echo "distribution-replicates=${TREE_HMM_DISTRIBUTION_REPLICATES:-30}"
    echo "distribution-seed=${TREE_HMM_DISTRIBUTION_SEED:-20260813}"
    echo "distribution-timing-repeats=${TREE_HMM_DISTRIBUTION_TIMING_REPEATS:-5}"
    echo "sanitizer-tools=${TREE_HMM_SANITIZER_TOOLS:-memcheck racecheck synccheck}"
    echo "skip-sanitizer=${TREE_HMM_SKIP_SANITIZER:-0}"
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
  prior_report=""
  prior_lines=0
  while IFS= read -r candidate; do
    if report_matches_sources "${candidate}"; then
      candidate_lines="$(wc -l < "${candidate}")"
      if [[ "${candidate_lines}" -gt "${prior_lines}" ]]; then
        prior_report="${candidate}"
        prior_lines="${candidate_lines}"
      fi
    fi
  done < <(find "${notebook_input_dir}" -type f \
    -name "parallel_phylogenetics_${accelerator_backend}_report*.txt" \
    -print | sort)
  packaged_report="${work}/PREVIOUS_BENCHMARK_REPORT.txt"
  if report_matches_sources "${packaged_report}"; then
    packaged_lines="$(wc -l < "${packaged_report}")"
    if [[ "${packaged_lines}" -gt "${prior_lines}" ]]; then
      prior_report="${packaged_report}"
    fi
  fi
  if [[ -n "${prior_report}" ]]; then
    cp "${prior_report}" "${report}"
    echo "resuming from $(wc -l < "${prior_report}") previously recorded lines"
  else
    echo "no prior report matches the current source identity"
    printf '%s\n' "${cache_marker}" > "${report}"
  fi
fi

cd "${work}/parallel_phylogenetic_inference"
TREE_HMM_PRECISIONS="${TREE_HMM_PRECISIONS_OVERRIDE:-FP64 FP32}" \
  TREE_HMM_RESUME_REPORT="${report}" \
  TREE_HMM_BENCHMARK_REPEATS="${TREE_HMM_BENCHMARK_REPEATS:-15}" \
  TREE_HMM_EMPIRICAL_REPEATS="${TREE_HMM_EMPIRICAL_REPEATS:-3}" \
  bash "scripts/notebook_${accelerator_backend}.sh" 2>&1 | tee -a "${report}"
