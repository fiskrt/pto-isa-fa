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

# Build one .so per TILE_S1 variant into its own dir (build/ = 256, build_tile512/ = 512).
build_one() {
    local tile=$1 dir=$2
    rm -rf "$dir"
    mkdir "$dir"
    ( cd "$dir" && cmake -DFA_TILE_S1="$tile" .. >/dev/null && make -j16 )
    echo "[build] done TILE_S1=$tile -> $dir/lib/$(ls "$dir"/lib/libtfa_torch*.so | xargs -n1 basename)"
}

build_one 256 build
build_one 512 build_tile512
