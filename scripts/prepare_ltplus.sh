#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 alignment.fasta.zip LTPlus.ntree output-directory [preparer arguments...]" >&2
  exit 2
fi

alignment="$1"
tree="$2"
output="$3"
shift 3
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

expected_alignment_sha256=c6cb5a4be1a29313e4a13904056206187c549600dd4b3057320c1592ba28c44b
expected_tree_sha256=40aa58c94a5be87d46e5c47871f31910e5ef401874fa3f84a08406eced593ceb

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(shasum -a 256 "${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "SHA-256 mismatch for ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
}

verify_sha256 "${alignment}" "${expected_alignment_sha256}"
verify_sha256 "${tree}" "${expected_tree_sha256}"

if python3 -c 'import numpy' 2>/dev/null; then
  python3 "${repository}/scripts/prepare_streaming_alignment.py" \
    "${alignment}" "${tree}" "${output}" "$@"
elif command -v uv >/dev/null; then
  uv run --with numpy python \
    "${repository}/scripts/prepare_streaming_alignment.py" \
    "${alignment}" "${tree}" "${output}" "$@"
else
  echo "NumPy is unavailable; install it or provide uv for the streaming preparation." >&2
  exit 1
fi
