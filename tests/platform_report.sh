#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# A test report for one platform, in one file: the system, compiler and
# conda packages, the build (pip install -e .), where FLINT / GMP are
# loaded from, then the whole test suite with the C++ engines and with
# pure Python (tests/run_all.sh, tests/run_all.sh --fallback).
#
# Run in the conda environment with Sage, from a clone of the repository
# or from a downloaded release folder (vertexalgebras-<version>):
#   conda activate sage
#   tests/platform_report.sh               # build, then all tests
#   tests/platform_report.sh --no-install  # already installed: tests only
# Writes platform_report_<date>.txt in the top folder (about 15-30
# minutes in all); send that file.  Uses bash 3.2 and POSIX tools only
# (macOS's own bash works).
set -u
cd "$(dirname "$0")/.."
out="platform_report_$(date +%Y%m%d_%H%M%S).txt"
install=1
[ "${1:-}" = "--no-install" ] && install=0

section() {
    printf '\n===== %s =====\n' "$1" >> "$out"
}

run() {   # run a command, its output and exit status into the report
    printf '$ %s\n' "$*" >> "$out"
    "$@" >> "$out" 2>&1
    printf '(exit status %s)\n' "$?" >> "$out"
}

: > "$out"
echo "writing $out"
section "date and repository"
run date
printf 'folder: %s\n' "$(basename "$(pwd)")" >> "$out"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    run git log -1 --format='%H %cd %s'
    run git status --short
else
    echo 'not a git repository (a downloaded release folder)' >> "$out"
fi

section "system"
run uname -a
if [ "$(uname)" = Darwin ]; then
    run sw_vers
    run sysctl -n machdep.cpu.brand_string
    run xcode-select -p
else
    [ -r /etc/os-release ] && run cat /etc/os-release
fi
if [ "$(uname)" = Darwin ]; then run sysctl -n hw.ncpu; else run nproc; fi

section "conda environment"
printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}" >> "$out"
run which python sage
run python --version
run sage --version
if command -v conda > /dev/null; then
    run sh -c "conda list | grep -E '^(sage|sagelib|sagemath|python|flint|libflint|gmp|cython|meson|meson-python|ninja|pkg-config|cxx-compiler|clangxx|clang|gxx|libcxx|singular|pari) '"
fi
for cxx in "${CXX:-}" c++ clang++ g++; do
    if [ -n "$cxx" ] && command -v "$cxx" > /dev/null; then
        run "$cxx" --version
        break
    fi
done

if [ $install = 1 ]; then
    section "build: pip install --no-build-isolation -e ."
    start=$(date +%s)
    run pip install --no-build-isolation -e .
    printf 'build took %s s\n' "$(( $(date +%s) - start ))" >> "$out"
fi

section "import"
run python -c "import vertex_algebras; print(vertex_algebras.__version__)"

section "libraries of the compiled engines"
# the compiled modules that Python imports: in the build folder for an
# editable install, in site-packages otherwise
for m in _universal_engine _linear_algebra_engine; do
    so=$(python -c "import vertex_algebras.$m as m; print(m.__file__)" 2>/dev/null)
    if [ -z "$so" ] || [ ! -e "$so" ]; then
        echo "vertex_algebras.$m cannot be imported" >> "$out"
        continue
    fi
    if command -v otool > /dev/null; then
        run otool -L "$so"
    else
        run ldd "$so"
    fi
done

section "tests/run_all.sh (C++ engines)"
start=$(date +%s)
run tests/run_all.sh
printf 'took %s s\n' "$(( $(date +%s) - start ))" >> "$out"

section "tests/run_all.sh --fallback (pure Python)"
start=$(date +%s)
run tests/run_all.sh --fallback
printf 'took %s s\n' "$(( $(date +%s) - start ))" >> "$out"

section "summary"
summary=$(grep -E 'FAIL|ALL PASSED|SOME CHECKS FAILED|exit status [1-9]' "$out" | tail -40)
printf '%s\n' "$summary" >> "$out"
echo "done: send $out"
tail -20 "$out"
