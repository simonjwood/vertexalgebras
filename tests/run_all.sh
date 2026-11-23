#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Run every doctest (with --long: the engine, thread and Ctrl-C checks are
# tagged '# long time') and test script; run from anywhere inside the
# repository, or inside a folder made by `python -m vertex_algebras.examples`
# (then the doctests of the installed package; the .pyx files are not
# installed, so their doctests are skipped).
#   tests/run_all.sh             the C++ engines (default)
#   tests/run_all.sh --fallback  doctests only, pure Python (no compiled module)
# Needs the Sage conda env active (`conda activate sage`).
set -u
cd "$(dirname "$0")/.."
# needs the package installed: pip install --no-build-isolation -e .
# (an editable install rebuilds changed C++ / Cython files at this import)
if ! err=$(python -c "import vertex_algebras" 2>&1); then
    echo "vertex_algebras cannot be imported (not installed, or the build failed):"
    echo "$err" | tail -30
    echo "install it with: pip install --no-build-isolation -e ."
    exit 1
fi
if [ -d src/vertex_algebras ]; then
    PKG=src/vertex_algebras
else
    PKG=$(python -c 'import vertex_algebras, os; print(os.path.dirname(vertex_algebras.__file__))')
fi
MODULES="__init__.py catalog.py examples.py normally_ordered_expressions.py vertex_algebra_categories.py lie_conformal.py lie_superalgebras.py vertex_algebra.py universal_vertex_algebra.py affine_vertex_algebra.py ghost_systems.py bootstrap.py quantum_hamiltonian_reduction.py quantum_hamiltonian_reduction_modules.py lattice_vertex_algebra.py tensor_product_vertex_algebra.py vertex_subalgebra.py vertex_ideal.py quotient_vertex_algebra.py commutant_vertex_algebra.py weight_spaces.py _grading_groups.py _keyed_spans.py zhu_algebra.py higher_zhu_algebra.py _finite_dimensional_structure.py vertex_algebra_module.py lattice_vertex_algebra_module.py zhu_algebra_module.py vertex_algebra_submodule.py presented_vertex_algebra.py quotient_induced_module.py fusion_product.py vertex_algebra_morphism.py module_twists.py vertex_algebra_module_morphism.py vertex_algebra_module_ext.py _universal_engine.pyx _lattice_engine.pyx _tensor_engine.pyx _module_engine.pyx _linear_algebra_engine.pyx"
fail=0
if [ "${1:-}" = "--fallback" ]; then
    export VERTEX_ALGEBRAS_NO_ENGINE=1
fi
for f in $MODULES; do
    if [ ! -e "$PKG/$f" ] && [ "$PKG" != src/vertex_algebras ]; then
        echo "doctest  $f: skipped (not installed)"; continue
    fi
    out=$(python -m sage.doctest --long --warn-long 30.0 --random-seed=0 "$PKG/$f" 2>&1)
    n=$(echo "$out" | grep -o '\[[0-9]* tests' | head -1)
    if echo "$out" | grep -q "All tests passed"; then
        echo "doctest  $f: ok ${n#[}"
    else
        echo "doctest  $f: FAIL"; echo "$out" | tail -40; fail=1
    fi
done
if [ "${1:-}" != "--fallback" ]; then
    for t in tests/*.sage; do
        start=$(date +%s)
        out=$(sage "$t" 2>&1)
        secs=$(( $(date +%s) - start ))
        if echo "$out" | grep -q "RESULT: PASS"; then
            echo "script   $t: PASS (${secs}s)"
        else
            echo "script   $t: FAIL"; echo "$out" | tail -40; fail=1
        fi
    done
fi
[ $fail = 0 ] && echo "ALL PASSED" || echo "SOME CHECKS FAILED"
exit $fail
