#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${repo_dir}/data/fish_tree_of_life}"
mkdir -p "${destination}"

fetch() {
  local name="$1"
  local expected_sha256="$2"
  local url="$3"
  local archive="${destination}/${name}.xz"
  if [[ ! -f "${archive}" ]] ||
     ! printf '%s  %s\n' "${expected_sha256}" "${archive}" |
       shasum --algorithm 256 --check --status; then
    local temporary="${archive}.download"
    curl --fail --location "${url}" --output "${temporary}"
    printf '%s  %s\n' "${expected_sha256}" "${temporary}" |
      shasum --algorithm 256 --check --status
    mv "${temporary}" "${archive}"
  fi
  if [[ ! -f "${destination}/${name}" ]] ||
     [[ "${archive}" -nt "${destination}/${name}" ]]; then
    local temporary="${destination}/${name}.decompressing"
    xz --decompress --stdout "${archive}" > "${temporary}"
    mv "${temporary}" "${destination}/${name}"
  fi
}

fetch actinopt_12k_raxml.tre \
  306f83016a3ba172a8f415549093400ef11b36bb718a5b9f23e264e140681686 \
  https://fishtreeoflife.org/downloads/actinopt_12k_raxml.tre.xz
fetch final_alignment.phylip \
  e3bc380052ee0154bc649d3ba4618b8bd4c7807b01a119b37dec3199dd6e625e \
  https://fishtreeoflife.org/downloads/final_alignment.phylip.xz

echo "${destination}"
