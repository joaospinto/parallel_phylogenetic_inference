#!/usr/bin/env bash

# Helpers for recognizing completed CSV records in an earlier notebook report.
# A missing report or record is represented by a nonzero status.

benchmark_resume_case_completed() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local dataset="$4"
  local topology="$5"
  local leaves="$6"
  local sites="$7"
  local site_batch="$8"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v topology="${topology}" \
    -v leaves="${leaves}" -v sites="${sites}" \
    -v site_batch="${site_batch}" '
      method == "cuda" || method == "rocm" || method == "metal" {
        if ($1 == method && $2 == precision && $3 == dataset &&
            $4 == topology && $5 == leaves && $7 == sites &&
            $8 == site_batch) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if ($1 == "beagle" && $2 == resource && $3 == precision &&
            $4 == dataset && $5 == topology && $6 == leaves &&
            $8 == sites && $9 == site_batch) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_dataset_batch_completed() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local dataset="$4"
  local site_batch="$5"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v site_batch="${site_batch}" '
      method == "cuda" || method == "rocm" || method == "metal" {
        if ($1 == method && $2 == precision && $3 == dataset &&
            $8 == site_batch) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if ($1 == "beagle" && $2 == resource && $3 == precision &&
            $4 == dataset && $9 == site_batch) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_dataset_completed() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local dataset="$4"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" '
      method == "cuda" || method == "rocm" || method == "metal" {
        if ($1 == method && $2 == precision && $3 == dataset) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if ($1 == "beagle" && $2 == resource && $3 == precision &&
            $4 == dataset) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_capacity_reached() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local site_batch="$4"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -v prefix="# capacity_limit method=${method} precision=${precision} first_infeasible_site_batch=" \
    -v candidate="${site_batch}" '
      index($0, prefix) == 1 {
        boundary = substr($0, length(prefix) + 1)
        if (boundary ~ /^[0-9]+$/ && boundary <= candidate) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_validation_completed() {
  local report="$1"
  local precision="$2"
  local backend="${3:-}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  if [[ -n "${backend}" ]] &&
     grep -Fq "# validation_complete backend=${backend} precision=${precision}" \
       "${report}"; then
    return 0
  fi
  if grep -Fq "# validation_complete precision=${precision}" "${report}"; then
    [[ -z "${backend}" || "${backend}" == cuda ]]
    return
  fi
  # Reports produced before explicit completion markers are reusable only if
  # both the validation start and the subsequent scaling section are present.
  # A benchmark-only run therefore cannot masquerade as a validated run.
  [[ -z "${backend}" || "${backend}" == cuda ]] &&
    { grep -Fq "=== ${precision} host tests, including device algebra ===" \
      "${report}" ||
    grep -Fq "=== ${precision} host tests, including CUDA algebra emulation ===" \
      "${report}"; } &&
    grep -Fq "=== ${precision} CUDA scaling benchmark ===" "${report}"
}
