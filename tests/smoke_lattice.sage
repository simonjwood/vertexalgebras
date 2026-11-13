# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# LatticeVertexAlgebra: axioms, Sugawara/background-charge Virasoro vectors,
# free-field vs recursive kernel, cocycle warning, serial-vs-parallel identity.
# Run from the repository root:  sage tests/smoke_lattice.sage  -> RESULT: PASS
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import vertex_algebra
import random
import time
import warnings

from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
import axioms
from axioms import Reporter, raw_caches_ok

random.seed(1r)
report = Reporter()


def rand_mono(L, maxletters=2, maxder=1, qrange=1):
    n = L.rank()
    letters = [(random.randrange(n), random.randrange(maxder + 1))
               for _ in range(random.randrange(maxletters + 1))]
    letters = tuple(sorted(letters, key=lambda t: (t[1], -t[0]), reverse=True))
    q = tuple(random.randint(-qrange, qrange) for _ in range(n))
    return L.monomial((letters, q))


def nprod(a, n, b):
    return a.nproduct(b, n)


def skew_ok(L, a, b):
    return axioms.skew_ok(L, a, b)


def commutator_ok(L, a, b, c):
    return axioms.commutator_ok(L, a, b, c)


def associativity_ok(L, a, b, c):
    return axioms.associativity_ok(L, a, b, c)


LATTICES = {
    'A2': matrix([[2, -1], [-1, 2]]),
    'U (hyperbolic)': matrix([[0, 1], [1, 0]]),
    'Z (odd)': matrix([[1]]),
    'diag(2,-2)': matrix([[2, 0], [0, -2]]),
    'odd indefinite': matrix([[1, 2], [2, -3]]),
}

for name, B in LATTICES.items():
    L = LatticeVertexAlgebra(QQ, B)
    trip = [(rand_mono(L), rand_mono(L), rand_mono(L)) for _ in range(6)]
    t = time.time()
    good = all(skew_ok(L, a, b) for a, b, _ in trip)
    report('%s skew-symmetry' % name, good, '(%.1fs)' % (time.time() - t))
    t = time.time()
    good = all(commutator_ok(L, a, b, c) for a, b, c in trip)
    report('%s commutator formula' % name, good, '(%.1fs)' % (time.time() - t))
    t = time.time()
    good = all(associativity_ok(L, a, b, c) for a, b, c in trip[:2])
    report('%s associativity (m=-1,-2)' % name, good,
           '(%.1fs)' % (time.time() - t))
    # Conformal vectors are a choice made by the user, not by the class:
    # Sugawara Tsug, and background-charge deformations Tsug + ∂J_alpha.
    Binv = L.gram_matrix().inverse()
    J = L.heisenberg_currents()
    r = L.rank()
    w = sum(Binv[i, j] / 2 * J[i].no(J[j]) for i in range(r) for j in range(r))
    good = w.ope(w) == [w.derivative(), 2 * w, 0, L(r / 2)]
    alpha = tuple(random.randint(-2, 2) for _ in range(r))
    Ja = sum(alpha[i] * J[i] for i in range(r))
    T = w + Ja.derivative()
    c = r - 12 * L._form(alpha, alpha)
    goodT = T.ope(T) == [T.derivative(), 2 * T, 0, L(c / 2)]
    for _ in range(3):
        q = tuple(random.randint(-2, 2) for _ in range(r))
        eq = L.e(q)
        goodT = goodT and nprod(T, 1, eq) == \
            (L._form(q, q) / 2 - L._form(alpha, q)) * eq
    report('%s background charge Tsug + dJ_alpha' % name, goodT,
           '(alpha=%s, c=%s)' % (alpha, c))
    for _ in range(4):
        x = rand_mono(L, 1, 1)
        good = good and nprod(w, 0, x) == x.derivative()
    for _ in range(4):
        q = tuple(random.randint(-2, 2) for _ in range(L.rank()))
        eq = L.e(q)
        good = good and nprod(w, 1, eq) == L._form(q, q) / 2 * eq
    report('%s Sugawara (c = rank)' % name, good)
    # free-field kernel vs the recursive reference implementation
    agree = True
    for a, b, _ in trip:
        ka, kb = a.leading_support(), b.leading_support()
        for n in range(-3, L._nmax(ka, kb) + 1):
            agree = agree and L._nprod(ka, n, kb) == L._nprod_rec(ka, n, kb)
    report('%s free-field == recursive kernel' % name, agree)

# A non-local cocycle must warn and break skew-symmetry.
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter('always')
    T = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]), epsilon='trivial')
report('trivial cocycle on A2 warns', any('not local' in str(x.message)
                                         for x in wlist))
report('trivial cocycle on A2 breaks skew',
       not skew_ok(T, T.e((1, 0)), T.e((0, 1))))

# Serial vs parallel identity.
B = matrix([[2, -1], [-1, 2]])
K = Frac(QQ['c0'])
Ls = Lp = LatticeVertexAlgebra(K, B)     # serial, then (cold) parallel
random.seed(7r)
monos = [rand_mono(Ls, 3, 2, 1) for _ in range(12)]
A = Ap = sum(K(i + 1) * m for i, m in enumerate(monos))
t = time.time()
ope_s, no_s, d_s = A.ope(A), A.no(A), A.derivative(3)
t_s = time.time() - t
Lp.clear_caches()
set_nproc(4)
t = time.time()
ope_p, no_p, d_p = Ap.ope(Ap), Ap.no(Ap), Ap.derivative(3)
t_p = time.time() - t
same = ([x.monomial_coefficients() for x in ope_s]
        == [x.monomial_coefficients() for x in ope_p]
        and no_s.monomial_coefficients() == no_p.monomial_coefficients()
        and d_s.monomial_coefficients() == d_p.monomial_coefficients())
report('serial == parallel (ope/no/deriv)', same,
       '(serial %.2fs, nproc=4 %.2fs, %d poles)' % (t_s, t_p, len(ope_s)))
report('caches hold raw dicts after parallel run', raw_caches_ok(Lp))
# warm-cache re-run must agree
report('warm-cache parallel re-run', [x for x in Ap.ope(Ap)] == ope_p)

report.result()
