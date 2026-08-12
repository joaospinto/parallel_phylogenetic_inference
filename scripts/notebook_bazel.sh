#!/usr/bin/env bash

# Bazel/Bazelisk discovery for fresh Kaggle and Colab runtimes.
tree_hmm_notebook_bazel() {
  local repo_dir="$1"
  local notebook_work_dir="$2"
  local bazel_command="${TREE_HMM_BAZEL:-}"
  local required_version
  required_version="$(<"${repo_dir}/.bazelversion")"

  if [[ -z "${bazel_command}" ]] && command -v bazelisk >/dev/null 2>&1; then
    bazel_command="$(command -v bazelisk)"
  fi
  if [[ -z "${bazel_command}" ]] && command -v bazel >/dev/null 2>&1 &&
     [[ "$(bazel --version 2>/dev/null)" == \
        "bazel ${required_version}" ]]; then
    bazel_command="$(command -v bazel)"
  fi
  if [[ -z "${bazel_command}" ]]; then
    local bazelisk_version=1.29.0
    local bazelisk_sha256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
    local tool_dir="${notebook_work_dir}/tree_hmm_tools"
    bazel_command="${tool_dir}/bazelisk-${bazelisk_version}"
    mkdir -p "${tool_dir}"
    if [[ ! -x "${bazel_command}" ]] ||
       ! printf '%s  %s\n' "${bazelisk_sha256}" "${bazel_command}" |
         sha256sum --check --status; then
      local temporary="${bazel_command}.tmp"
      curl --fail --location --silent --show-error \
        "https://github.com/bazelbuild/bazelisk/releases/download/v${bazelisk_version}/bazelisk-linux-amd64" \
        --output "${temporary}"
      printf '%s  %s\n' "${bazelisk_sha256}" "${temporary}" |
        sha256sum --check --status
      chmod +x "${temporary}"
      mv "${temporary}" "${bazel_command}"
    fi
  fi
  printf '%s\n' "${bazel_command}"
}
