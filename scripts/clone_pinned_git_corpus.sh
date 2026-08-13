#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 repository-url revision destination" >&2
  exit 2
fi
url="$1"
revision="$2"
destination="$3"

if [[ ! -d "${destination}/.git" ]]; then
  if [[ -e "${destination}" ]]; then
    echo "destination exists but is not a Git repository: ${destination}" >&2
    exit 1
  fi
  git clone --filter=blob:none --no-checkout "${url}" "${destination}"
fi
actual_url="$(git -C "${destination}" remote get-url origin)"
if [[ "${actual_url%.git}" != "${url%.git}" ]]; then
  echo "unexpected origin for ${destination}: ${actual_url}" >&2
  exit 1
fi
if ! git -C "${destination}" cat-file -e "${revision}^{commit}" 2>/dev/null; then
  git -C "${destination}" fetch --depth=1 origin "${revision}"
fi
actual_revision="$(git -C "${destination}" rev-parse "${revision}^{commit}")"
if [[ "${actual_revision}" != "${revision}" ]]; then
  echo "revision ${revision} resolved to ${actual_revision}" >&2
  exit 1
fi
echo "prepared blobless corpus repository ${destination} at ${actual_revision}"
