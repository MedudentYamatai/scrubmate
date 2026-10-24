#!/usr/bin/env bash
# mjpython (MuJoCo の Mac 用 GUI ランチャ) を uv 環境で起動するラッパ。
#
# uv が入れる Python (python-build-standalone) は libpython を @rpath で参照するが、
# mjpython は .app バンドル経由で python を dlopen するので rpath が解決できず
# "Library not loaded: @rpath/libpython3.12.dylib" で落ちる。
# 本体 Python の lib/ を DYLD_LIBRARY_PATH に足して回避する。
#
# 使い方: ./scripts/mjpython.sh -m sim.view_panda
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PREFIX="$(uv run python -c 'import sys; print(sys.base_prefix)')"
export DYLD_LIBRARY_PATH="${BASE_PREFIX}/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec uv run mjpython "$@"
