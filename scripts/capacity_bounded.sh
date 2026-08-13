#!/usr/bin/env bash

# Run one empirical batch in an isolated process. Recognized capacity failures
# set benchmark_capacity_exhausted and return success so the caller can proceed
# with other methods; unrelated failures retain their original status.
benchmark_run_capacity_bounded() {
  local work_directory="$1"
  local host_memory_guard_kib="$2"
  local method="$3"
  local precision="$4"
  local site_batch="$5"
  shift 5
  local output_file
  local command_status
  local capacity_reason
  output_file="$(mktemp "${work_directory}/batch-run.XXXXXX")"
  if [[ "${method}" == "beagle-cpu" ]]; then
    if (
      ulimit -c 0
      ulimit -v "${host_memory_guard_kib}"
      export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
      exec "$@"
    ) >"${output_file}" 2>&1; then
      command_status=0
    else
      command_status=$?
    fi
  elif "$@" >"${output_file}" 2>&1; then
    command_status=0
  else
    command_status=$?
  fi
  if [[ "${command_status}" -eq 0 ]]; then
    cat "${output_file}"
  else
    cat "${output_file}" >&2
    capacity_reason=""
    if [[ "${command_status}" -eq 137 ]]; then
      capacity_reason="process-killed"
    elif [[ "${method}" == "beagle-cpu" &&
            "${command_status}" -eq 139 ]]; then
      capacity_reason="beagle-segfault-under-memory-limit"
    elif grep -Eqi \
        'out[ _]of[ _]memory|bad_alloc|cannot allocate memory|memory allocation|failed to allocate' \
        "${output_file}"; then
      capacity_reason="allocation-failure"
    fi
    if [[ -n "${capacity_reason}" ]]; then
      echo "# capacity_limit method=${method} precision=${precision}" \
        "first_infeasible_site_batch=${site_batch}" \
        "reason=${capacity_reason}"
      benchmark_capacity_exhausted=1
    else
      rm -f "${output_file}"
      return "${command_status}"
    fi
  fi
  rm -f "${output_file}"
  return 0
}
