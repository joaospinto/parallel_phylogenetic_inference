#!/usr/bin/env bash

# Shared parsing and command construction for benchmark drivers that can run
# one method or cyclically interleave several methods.  Callers provide the
# default BEAGLE CPU thread count and preserve their existing single-method
# command-line interface.

benchmark_initialize_method_set() {
  local script_directory="$1"
  local default_beagle_threads="$2"
  local dry_run="$3"
  shift 3

  benchmark_methods=()
  benchmark_method_threads=()
  benchmark_method_specs=()
  local specification
  local parsed_method
  local parsed_threads
  local canonical_specification
  local canonical_specs_seen="|"
  local needs_beagle=0
  for specification in "$@"; do
    case "${specification}" in
      cuda|rocm|metal)
        parsed_method="${specification}"
        parsed_threads=""
        canonical_specification="${specification}"
        ;;
      beagle-cuda)
        parsed_method=beagle-cuda
        parsed_threads=1
        canonical_specification=beagle-cuda
        needs_beagle=1
        ;;
      beagle-cpu)
        parsed_method=beagle-cpu
        parsed_threads="${default_beagle_threads}"
        canonical_specification="beagle-cpu:${parsed_threads}"
        needs_beagle=1
        ;;
      beagle-cpu:*)
        parsed_method=beagle-cpu
        parsed_threads="${specification#beagle-cpu:}"
        [[ "${parsed_threads}" =~ ^[1-9][0-9]*$ ]] || {
          echo "invalid BEAGLE CPU method specification ${specification}" >&2
          return 2
        }
        canonical_specification="beagle-cpu:${parsed_threads}"
        needs_beagle=1
        ;;
      *)
        echo "unsupported method specification ${specification}" >&2
        return 2
        ;;
    esac
    if [[ "${canonical_specs_seen}" == *"|${canonical_specification}|"* ]]; then
      echo "duplicate method specification ${specification}" >&2
      return 2
    fi
    canonical_specs_seen+="${canonical_specification}|"
    benchmark_methods+=("${parsed_method}")
    benchmark_method_threads+=("${parsed_threads}")
    benchmark_method_specs+=("${canonical_specification}")
  done
  if [[ "${#benchmark_methods[@]}" -eq 0 ]]; then
    echo "at least one benchmark method is required" >&2
    return 2
  fi

  if [[ "${needs_beagle}" == 1 && "${dry_run}" == 0 ]]; then
    # Configure once in the process that launches every interleaved child.
    # This is required on macOS, where a parent shell cannot reliably pass a
    # newly assigned DYLD_LIBRARY_PATH through another /bin/bash process.
    # shellcheck source=scripts/beagle_environment.sh
    source "${script_directory}/beagle_environment.sh"
    parallel_phylogenetics_configure_beagle
  fi
}

benchmark_select_method() {
  local method_index="$1"
  local benchmark_mode="$2"
  benchmark_method="${benchmark_methods[method_index]}"
  benchmark_resume_threads="${benchmark_method_threads[method_index]}"
  benchmark_display_threads="${benchmark_resume_threads:-none}"
  benchmark_method_specification="${benchmark_method_specs[method_index]}"
  case "${benchmark_method}" in
    cuda|rocm|metal)
      benchmark_command=("bazel-bin/${benchmark_method}_benchmark"
                         --benchmark-mode "${benchmark_mode}")
      ;;
    beagle-cpu)
      benchmark_command=(bazel-bin/beagle_benchmark --beagle-resource cpu
                         --beagle-threads "${benchmark_resume_threads}"
                         --benchmark-mode "${benchmark_mode}")
      ;;
    beagle-cuda)
      benchmark_command=(bazel-bin/beagle_benchmark --beagle-resource cuda
                         --beagle-threads 1 --benchmark-mode "${benchmark_mode}")
      ;;
    *)
      echo "internal error: unsupported parsed method ${benchmark_method}" >&2
      return 2
      ;;
  esac
}

benchmark_emit_interleaved_protocol() {
  local study="$1"
  local precision="$2"
  local benchmark_mode="$3"
  local case_count="$4"
  local joined_specs
  joined_specs="$(IFS=,; echo "${benchmark_method_specs[*]}")"
  echo "# interleaved_schedule study=${study}" \
    "precision=${precision} benchmark_mode=${benchmark_mode}" \
    "case_count=${case_count} method_count=${#benchmark_methods[@]}" \
    "methods=${joined_specs} policy=cyclic-rotation-by-case"
}

benchmark_emit_interleaved_case() {
  local study="$1"
  local precision="$2"
  local benchmark_mode="$3"
  local case_index="$4"
  local order_index="$5"
  local method_index="$6"
  shift 6
  echo "# interleaved_case study=${study} precision=${precision}" \
    "benchmark_mode=${benchmark_mode} case_index=${case_index}" \
    "order_index=${order_index} specification=${benchmark_method_specs[method_index]}" \
    "$@"
}
