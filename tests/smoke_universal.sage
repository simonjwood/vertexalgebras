# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# UniversalVertexAlgebra: serial-vs-parallel identity, raw-dict caches,
# warm-cache re-runs and axiom checks (n=1 superconformal model).
# Run from the repository root:  sage tests/smoke_universal.sage  -> RESULT: PASS
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import vertex_algebra
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from axioms import (Reporter, skew_ok, commutator_ok, associativity_ok,
                    raw_caches_ok)

report = Reporter()
K.<c0> = Frac(QQ['c0'])


def n1_opes(L, G):
    return {('L', 'L'): [L.derivative(), 2 * L, 0, c0 / 2],
            ('L', 'G'): [G.derivative(), 3 / 2 * G],
            ('G', 'G'): [2 * L, 0, 2 * c0 / 3]}


def n1():
    E = NormallyOrderedExpressions(K, ['L', 'G'], parities={'G': 1})
    return UniversalVertexAlgebra(K, E, n1_opes(*E.gens()))


def canon(m):
    """Letters in canonical order: higher derivative first, then lower index."""
    return tuple(sorted(m, key=lambda t: (t[1], -t[0]), reverse=True))


MONOS = [canon(m) for m in [((0, 0),), ((0, 1),), ((0, 2),), ((1, 0),), ((1, 1),),
         ((1, 0), (1, 0),), ((0, 0), (0, 0),), ((0, 1), (1, 0),),
         ((0, 0), (1, 1),)]]


def coeffs(x):
    return tuple(sorted(x.monomial_coefficients().items()))


def run(V):
    A = sum(V.monomial(m) for m in MONOS)
    t = time.time()
    r = ([coeffs(x) for x in A.ope(A)], coeffs(A.no(A)),
         coeffs(A.derivative(2)), coeffs(A.nproduct(A, -3)))
    return r, time.time() - t, A


# one algebra, serial, then from cold caches in parallel (the number of
# processes is a setting of the session, VertexAlgebraParallelism)
Vs = Vp = n1()
rs, ts, _ = run(Vs)
Vp.clear_caches()
set_nproc(8)
rp, tp, Ap = run(Vp)
report('serial == parallel (ope/no/deriv/nproduct(-3))', rs == rp,
       '(serial %.2fs, nproc=8 %.2fs)' % (ts, tp))
report('engine active', Vp._engine() is not None)
report('caches hold raw dicts after parallel run', raw_caches_ok(Vp))
t = time.time()
rw = ([coeffs(x) for x in Ap.ope(Ap)], coeffs(Ap.no(Ap)),
      coeffs(Ap.derivative(2)), coeffs(Ap.nproduct(Ap, -3)))
report('warm-cache parallel re-run', rw == rp, '(%.2fs)' % (time.time() - t))

# Parallel derivative of an element with many distinct monomials.
L = Vp.gen('L')
pieces = [L]
for i in range(8):
    pieces.append(pieces[-1].no(L))
S = sum(pieces)
Vp.clear_caches()
d_par = S.derivative()
set_nproc(1)
Vs.clear_caches()
d_ser = S.derivative()
report('big parallel derivative (%d monomials)' % len(S.support()),
       coeffs(d_par) == coeffs(d_ser))

# Axioms on random small elements (exercises the generic negative n-products).
random.seed(3r)
L, G = Vs.gen('L'), Vs.gen('G')
pool = [L, G, L.derivative(), G.derivative(), L.no(G), L.no(L)]
trip = [tuple(random.choice(pool) for _ in range(3)) for _ in range(4)]
t = time.time()
report('skew-symmetry', all(skew_ok(Vs, a, b) for a, b, _ in trip),
       '(%.1fs)' % (time.time() - t))
t = time.time()
report('commutator formula',
       all(commutator_ok(Vs, a, b, c, range(-1, 2)) for a, b, c in trip),
       '(%.1fs)' % (time.time() - t))
t = time.time()
report('associativity (m=-1,-2)',
       all(associativity_ok(Vs, a, b, c) for a, b, c in trip[:2]),
       '(%.1fs)' % (time.time() - t))

# Negative control: the same OPEs with G declared EVEN are not skew-symmetric.
Ebad = NormallyOrderedExpressions(K, ['L', 'G'])
Vbad = UniversalVertexAlgebra(K, Ebad, n1_opes(*Ebad.gens()))
Lb, Gb = Vbad.gen('L'), Vbad.gen('G')
report('negative control: even G breaks skew-symmetry', not skew_ok(Vbad, Gb, Gb))
# pure Python (no engine) with 4 threads set: serial, the same results
E = NormallyOrderedExpressions(K, ['L', 'G'], parities={'G': 1})
Vc = UniversalVertexAlgebra(K, E, n1_opes(*E.gens()), implementation='python')
want, _, _ = run(Vc)
Vc.clear_caches()
set_nproc(4)
got, _, _ = run(Vc)
set_nproc(1)
report('pure Python with nproc=4: the same results', got == want)
report.result()
