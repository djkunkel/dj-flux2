#!/bin/bash
# ---------------------------------------------------------------------------
# libnuma shim
#
# torch/lib/libtorch_rocshmem.so does dlopen("libnuma.so") - the UNVERSIONED
# name. Most distros only ship libnuma.so.1 (runtime); the bare libnuma.so
# symlink comes from the -devel package (numactl-devel / libnuma-dev). On an
# immutable distro we can't install that, so dlopen fails and torch prints:
#   E-001h rocSHMEM Could not open libnuma. Returning   numa_wrapper.cpp:48
# It's non-fatal (rocSHMEM is unused here) but noisy. We create a private
# libnuma.so symlink inside the venv; run.sh/test_torch.sh add it to
# LD_LIBRARY_PATH so the dlopen succeeds.
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="${SCRIPT_DIR}/.venv/numa-shim"

# Find the system libnuma runtime library.
NUMA_LIB=""
for cand in \
    /lib64/libnuma.so.1 \
    /usr/lib64/libnuma.so.1 \
    /usr/lib/x86_64-linux-gnu/libnuma.so.1 \
    /lib/x86_64-linux-gnu/libnuma.so.1; do
    if [[ -e "${cand}" ]]; then
        # Point at the SONAME path (libnuma.so.1), NOT the resolved versioned
        # file (libnuma.so.1.0.0). The SONAME is kept stable by the package
        # manager, so the shim survives minor libnuma updates.
        NUMA_LIB="${cand}"
        break
    fi
done
# Fall back to ldconfig if the well-known paths missed it.
if [[ -z "${NUMA_LIB}" ]]; then
    NUMA_LIB="$(ldconfig -p 2>/dev/null | awk '/libnuma\.so\.1/ {print $NF; exit}')"
fi

if [[ -z "${NUMA_LIB}" || ! -e "${NUMA_LIB}" ]]; then
    echo "numa shim: WARNING - system libnuma.so.1 not found; skipping." >&2
    echo "           torch will print a harmless 'Could not open libnuma' notice." >&2
    exit 0
fi

mkdir -p "${SHIM_DIR}"
ln -sf "${NUMA_LIB}" "${SHIM_DIR}/libnuma.so"
echo "numa shim: ${SHIM_DIR}/libnuma.so -> ${NUMA_LIB}"
