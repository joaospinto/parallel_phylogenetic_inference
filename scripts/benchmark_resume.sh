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
  local benchmark_mode="${9:-}"
  local threads="${10:-}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v topology="${topology}" \
    -v leaves="${leaves}" -v sites="${sites}" \
    -v site_batch="${site_batch}" -v benchmark_mode="${benchmark_mode}" \
    -v threads="${threads}" '
      function mode_matches() {
        if (benchmark_mode == "") return 1
        if ("benchmark_mode" in column)
          return $(column["benchmark_mode"]) == benchmark_mode
        return benchmark_mode == "full-input-update"
      }
      function thread_matches() {
        if (threads == "") return 1
        if ("threads" in column) return $(column["threads"]) == threads
        return threads == "1"
      }
      $1 == "backend" && $2 == "precision" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "native"
        next
      }
      $1 == "baseline" && $2 == "beagle_resource" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "beagle"
        next
      }
      method == "cuda" || method == "rocm" || method == "metal" {
        if (kind == "native" && $(column["backend"]) == method &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            $(column["topology"]) == topology &&
            $(column["leaves"]) == leaves &&
            $(column["sites"]) == sites &&
            $(column["site_batch"]) == site_batch &&
            mode_matches() && thread_matches()) found = 1
        else if (kind == "" && $1 == method && $2 == precision &&
                 $3 == dataset && $4 == topology && $5 == leaves &&
                 $7 == sites && $8 == site_batch &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update") &&
                 (threads == "" || threads == "1")) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if (kind == "beagle" && $(column["baseline"]) == "beagle" &&
            $(column["beagle_resource"]) == resource &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            $(column["topology"]) == topology &&
            $(column["leaves"]) == leaves && $(column["sites"]) == sites &&
            $(column["site_batch"]) == site_batch &&
            mode_matches() && thread_matches()) found = 1
        else if (kind == "" && $1 == "beagle" && $2 == resource &&
                 $3 == precision && $4 == dataset && $5 == topology &&
                 $6 == leaves && $8 == sites && $9 == site_batch &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update") &&
                 (threads == "" || threads == "1")) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_synthetic_replicate_completed() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local topology="$4"
  local leaves="$5"
  local patterns="$6"
  local seed_base="$7"
  local replicate="$8"
  local benchmark_mode="${9:-}"
  local threads="${10:-}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v topology="${topology}" -v leaves="${leaves}" \
    -v patterns="${patterns}" -v seed_base="${seed_base}" \
    -v replicate="${replicate}" -v benchmark_mode="${benchmark_mode}" \
    -v threads="${threads}" '
      function mode_matches() {
        if (benchmark_mode == "") return 1
        if ("benchmark_mode" in column)
          return $(column["benchmark_mode"]) == benchmark_mode
        return benchmark_mode == "full-input-update"
      }
      function thread_matches() {
        if (threads == "") return 1
        if ("threads" in column) return $(column["threads"]) == threads
        return threads == "1"
      }
      $1 == "backend" && $2 == "precision" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "native"
        next
      }
      $1 == "baseline" && $2 == "beagle_resource" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "beagle"
        next
      }
      {
        resource = substr(method, 8)
        matching_method = ((kind == "native" &&
                            $(column["backend"]) == method) ||
                           (kind == "beagle" &&
                            $(column["baseline"]) == "beagle" &&
                            $(column["beagle_resource"]) == resource))
        if (matching_method && $(column["precision"]) == precision &&
            $(column["dataset"]) == "synthetic" &&
            $(column["topology"]) == topology &&
            $(column["leaves"]) == leaves &&
            $(column["unique_patterns"]) == patterns &&
            $(column["seed_base"]) == seed_base &&
            $(column["replicate"]) == replicate &&
            mode_matches() && thread_matches()) found = 1
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
  local benchmark_mode="${6:-}"
  local threads="${7:-}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v site_batch="${site_batch}" \
    -v benchmark_mode="${benchmark_mode}" -v threads="${threads}" '
      function mode_matches() {
        if (benchmark_mode == "") return 1
        if ("benchmark_mode" in column)
          return $(column["benchmark_mode"]) == benchmark_mode
        return benchmark_mode == "full-input-update"
      }
      function thread_matches() {
        if (threads == "") return 1
        if ("threads" in column) return $(column["threads"]) == threads
        return threads == "1"
      }
      $1 == "backend" && $2 == "precision" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "native"
        next
      }
      $1 == "baseline" && $2 == "beagle_resource" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "beagle"
        next
      }
      method == "cuda" || method == "rocm" || method == "metal" {
        if (kind == "native" && $(column["backend"]) == method &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            $(column["site_batch"]) == site_batch &&
            mode_matches()) found = 1
        else if (kind == "" && $1 == method && $2 == precision &&
                 $3 == dataset && $8 == site_batch &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update")) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if (kind == "beagle" && $(column["baseline"]) == "beagle" &&
            $(column["beagle_resource"]) == resource &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            $(column["site_batch"]) == site_batch &&
            mode_matches() && thread_matches()) found = 1
        else if (kind == "" && $1 == "beagle" && $2 == resource &&
                 $3 == precision && $4 == dataset && $9 == site_batch &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update") &&
                 (threads == "" || threads == "1")) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_dataset_completed() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local dataset="$4"
  local benchmark_mode="${5:-}"
  local threads="${6:-}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -F, -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v benchmark_mode="${benchmark_mode}" \
    -v threads="${threads}" '
      function mode_matches() {
        if (benchmark_mode == "") return 1
        if ("benchmark_mode" in column)
          return $(column["benchmark_mode"]) == benchmark_mode
        return benchmark_mode == "full-input-update"
      }
      function thread_matches() {
        if (threads == "") return 1
        if ("threads" in column) return $(column["threads"]) == threads
        return threads == "1"
      }
      $1 == "backend" && $2 == "precision" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "native"
        next
      }
      $1 == "baseline" && $2 == "beagle_resource" {
        delete column
        for (field_index = 1; field_index <= NF; ++field_index) column[$(field_index)] = field_index
        kind = "beagle"
        next
      }
      method == "cuda" || method == "rocm" || method == "metal" {
        if (kind == "native" && $(column["backend"]) == method &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            mode_matches()) found = 1
        else if (kind == "" && $1 == method && $2 == precision &&
                 $3 == dataset &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update")) found = 1
        next
      }
      {
        resource = substr(method, 8)
        if (kind == "beagle" && $(column["baseline"]) == "beagle" &&
            $(column["beagle_resource"]) == resource &&
            $(column["precision"]) == precision &&
            $(column["dataset"]) == dataset &&
            mode_matches() && thread_matches()) found = 1
        else if (kind == "" && $1 == "beagle" && $2 == resource &&
                 $3 == precision && $4 == dataset &&
                 (benchmark_mode == "" || benchmark_mode == "full-input-update") &&
                 (threads == "" || threads == "1")) found = 1
      }
      END { exit !found }
    ' "${report}"
}

benchmark_resume_capacity_reached() {
  local report="$1"
  local method="$2"
  local precision="$3"
  local site_batch="$4"
  local dataset="${5:-unknown}"
  local benchmark_mode="${6:-full-input-update}"
  local threads="${7:-none}"
  [[ -n "${report}" && -r "${report}" ]] || return 1
  awk -v method="${method}" -v precision="${precision}" \
    -v dataset="${dataset}" -v benchmark_mode="${benchmark_mode}" \
    -v threads="${threads}" -v candidate="${site_batch}" '
      $1 == "#" && $2 == "capacity_limit" {
        delete value
        for (field_index = 3; field_index <= NF; ++field_index) {
          split($(field_index), part, "=")
          value[part[1]] = part[2]
        }
        legacy = !("dataset" in value)
        metadata_matches = 0
        if (legacy) {
          metadata_matches = dataset == "unknown" &&
                             benchmark_mode == "full-input-update" &&
                             threads == "none"
        } else {
          metadata_matches = value["dataset"] == dataset &&
                             value["benchmark_mode"] == benchmark_mode &&
                             value["threads"] == threads
        }
        if (value["method"] == method && value["precision"] == precision &&
            metadata_matches &&
            value["first_infeasible_site_batch"] ~ /^[0-9]+$/ &&
            value["first_infeasible_site_batch"] <= candidate) found = 1
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
