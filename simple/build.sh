#!/bin/bash
# Build the FA kernel .so and the torch launcher .so. Self-contained: no golden data, no case sweep.
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash build.sh
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[build] ASCEND_HOME_PATH not set. Run: source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
    exit 1
fi

rm -rf build
mkdir build
cd build
cmake .. >/dev/null
make -j16
echo "[build] done -> $(pwd)/lib/libtfa_torch.so"
