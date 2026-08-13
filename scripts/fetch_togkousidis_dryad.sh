#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 output-directory [downloaded-stopping_criteria_data.tar.gz]" >&2
  exit 2
fi

output="$1"
archive="${2:-${DRYAD_ARCHIVE:-${output}/stopping_criteria_data.tar.gz}}"
expected_sha256=06cee5bd75748acf5ba95a10b404b2867dd0b52a3e9e1b9ec357f9d9c7e09f4c
download_url="${DRYAD_DOWNLOAD_URL:-}"
mkdir -p "${output}"

if [[ ! -f "${archive}" ]]; then
  if [[ -z "${download_url}" ]]; then
    echo "The pinned Dryad API currently requires an authenticated download." >&2
    echo "Download stopping_criteria_data.tar.gz from DOI" \
      "10.5061/dryad.8gtht76zz (version 6) and pass its path, or set" \
      "DRYAD_DOWNLOAD_URL to an authorized direct URL." >&2
    exit 1
  fi
  archive="${output}/stopping_criteria_data.tar.gz"
  curl --fail --location --show-error "${download_url}" --output "${archive}"
fi

printf '%s  %s\n' "${expected_sha256}" "${archive}" |
  shasum -a 256 --check
raw="${output}/raw"
mkdir -p "${raw}"
tar -xzf "${archive}" -C "${raw}"
preparer="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prepare_dryad_corpus.py"
if python3 -c 'import pyarrow' >/dev/null 2>&1; then
  python3 "${preparer}" "${raw}" "${output}/selected"
elif command -v uv >/dev/null 2>&1; then
  uv run --with pyarrow "${preparer}" "${raw}" "${output}/selected"
else
  echo "pyarrow is required only to prepare the corpus; install it or uv" >&2
  exit 1
fi
