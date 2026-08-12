#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 ]]; then
  destination="$1"
  shift
else
  destination="${repo_dir}/data/pandit"
fi
mkdir -p "${destination}"

archive="${destination}/Pandit17.0.gz"
expected_sha256=b7b8c86a83652748438b54395589e9ae1e053d9536d9853437a9adfebb4c21c5
if [[ ! -f "${archive}" ]] ||
   ! printf '%s  %s\n' "${expected_sha256}" "${archive}" |
     shasum --algorithm 256 --check --status; then
  temporary="${archive}.download"
  curl --fail --location \
    https://www.ebi.ac.uk/goldman-srv/pandit/Pandit/data/Pandit17.0.gz \
    --output "${temporary}"
  printf '%s  %s\n' "${expected_sha256}" "${temporary}" |
    shasum --algorithm 256 --check --status
  mv "${temporary}" "${archive}"
fi

python3 "${repo_dir}/scripts/extract_pandit.py" \
  "${archive}" "${destination}/families" "$@"
