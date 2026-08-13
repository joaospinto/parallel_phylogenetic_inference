#!/usr/bin/env bash

parallel_phylogenetics_configure_beagle() {
  if [[ -z "${BEAGLE_PREFIX:-}" ]]; then
    if command -v brew >/dev/null 2>&1 &&
       brew --prefix beagle >/dev/null 2>&1; then
      BEAGLE_PREFIX="$(brew --prefix beagle)"
    elif [[ -f /usr/include/libhmsbeagle-1/libhmsbeagle/beagle.h ]]; then
      BEAGLE_PREFIX=/usr
    else
      BEAGLE_PREFIX=/usr/local
    fi
  fi

  if [[ ! -f \
          "${BEAGLE_PREFIX}/include/libhmsbeagle-1/libhmsbeagle/beagle.h" &&
        ! -f "${BEAGLE_PREFIX}/include/libhmsbeagle/beagle.h" ]]; then
    echo "BEAGLE header not found under ${BEAGLE_PREFIX}/include" >&2
    echo "Install BEAGLE or set BEAGLE_PREFIX to its installation prefix." >&2
    return 1
  fi
  if [[ ! -f "${BEAGLE_PREFIX}/lib/libhmsbeagle.dylib" &&
        ! -f "${BEAGLE_PREFIX}/lib/libhmsbeagle.so" &&
        ! -f "${BEAGLE_PREFIX}/lib64/libhmsbeagle.so" &&
        ! -f \
          "${BEAGLE_PREFIX}/lib/x86_64-linux-gnu/libhmsbeagle.so" ]]; then
    echo "BEAGLE shared library not found under ${BEAGLE_PREFIX}" >&2
    return 1
  fi
  export BEAGLE_PREFIX
  local libraries="${BEAGLE_PREFIX}/lib:${BEAGLE_PREFIX}/lib64"
  libraries+="${libraries:+:}${BEAGLE_PREFIX}/lib/x86_64-linux-gnu"
  export LD_LIBRARY_PATH="${libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}
