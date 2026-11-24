# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# TensorProductVertexAlgebra: the C++ engine (_tensor_engine)
# against the Python recursion (_tprod / _tderiv): universal x lattice with
# odd factors on both sides, three factors, Q(k) (lifted to QQ[k]), QQ[c],
# Q(sqrt 5), Q(sqrt 5)(k) (lifted to Q(sqrt 5)[k]), a tower, QQbar, a
# callable cocycle, a W_3 factor (needs u-variables: no engine, Python path);
# every product kind, key level and element level; threads (identical
# results, dict order too); Ctrl-C.
# Run from the repository root:
#   sage tests/crosscheck_tensor_engine.sage  -> RESULT: PASS
import os
import random
import signal
import sys
import time

from vertex_algebras import tensor_product_vertex_algebra as TPV
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra

if TPV._TVA_ENGINE is None:
    print('engine switched off:', TPV._ENGINE_OFF)
    print('RESULT: PASS (skipped)')
    sys.exit(0)

checked = mismatches = 0


# Ctrl-C timing checks use at most as many threads as there are cores
# (oversubscribed threads notice the interrupt late on small CI runners)
import os as _os
_NCORES = _os.cpu_count() or 1


def _threads(n):
    return min(n, _NCORES)


# Ctrl-C time limits: 2 s on the authors' desktop, longer on slower
# machines (a laptop needed 2.1 s where the desktop needs 0.5 s; user
# report, 2026-11-23).  The limit scales with this machine's time for a
# fixed product, a Virasoro OPE that takes 0.30 s on the desktop.
def _ctrl_c_limit():
    from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    vir = {('T', 'T'): [{(('T', 1),): 1}, {(('T', 0),): 2}, 0, QQ(7) / 12]}
    V = UniversalVertexAlgebra(QQ, ['T'], vir, degrees={'T': 2})
    x = sum(V.monomial(k) for k in V.weight_space(10).basis_keys())
    set_nproc(1)
    best = None
    for _ in range(2):
        V.clear_caches()
        t = time.perf_counter()
        x.ope(x)
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    V.clear_caches()
    return 2 * max(1, best / 0.30)


_CTRL_C_LIMIT = _ctrl_c_limit()
print('Ctrl-C time limit: %.1f s' % _CTRL_C_LIMIT)

def check(ok, what):
    global checked, mismatches
    checked += 1
    if not ok:
        mismatches += 1
        print('MISMATCH', what)


random.seed(5r)


def raw(x):
    if isinstance(x, (list, tuple)):
        return [raw(y) for y in x]
    if isinstance(x, dict):
        return list(x.items())
    return list(x.monomial_coefficients().items())


def srt(x):
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [srt(y) for y in x]
    return sorted(x, key=str)


def batch(V, X, Y):
    """V._nprod_raw_batch, or the single products where it is unavailable
    (no engine): both must agree with every mode and thread count."""
    els = [dict(X._monomial_coefficients), dict(Y._monomial_coefficients)]
    tasks = [(0, -1, 1), (1, -3, 0), (0, 0, 1), (1, 2, 1), (0, -1, 0)]
    r = V._nprod_raw_batch(els, tasks)
    return r if r is not None else [V._nprod_raw(els[i], n, els[j]) for i, n, j in tasks]


def work(T, X, Y):
    kx, ky = sorted(X.support())[0], sorted(Y.support())[-1]
    return raw([X.ope(Y), Y.ope(X), X.no(Y), X.nproduct(Y, -3), X.nproduct(Y, 1),
                X.derivative(2), T._ope_mono(kx, ky), T._nprod_mono(kx, -2, ky),
                T._deriv_mono(ky, 2),
                T._ope_raw_many(dict(X._monomial_coefficients),
                                [dict(Y._monomial_coefficients), dict(X._monomial_coefficients)]),
                batch(T, X, Y)])


def run(T, X, Y, nproc=1):
    """work() in T (X, Y carried over by their basis keys) from cold caches
    with nproc threads; and whether T has an engine."""
    T.clear_caches()
    set_nproc(nproc)
    X, Y = (T._from_dict(dict(Z._monomial_coefficients)) for Z in (X, Y))
    try:
        return work(T, X, Y), T._engine() is not None
    finally:
        set_nproc(1)


def rand_elem(T, gens, n, R, extra=None):
    """A random linear combination of products of the given elements."""
    out = T.zero()
    for t in range(n):
        x = random.choice(gens)
        for _ in range(random.randint(0, 2)):
            y = random.choice(gens)
            x = x.no(y) if random.random() < 0.6 else x.derivative()
        c = R(random.randint(1, 5))
        if extra is not None and t % 2:
            c += extra
        out += c * x
    return out


Kk.<k> = Frac(QQ['k'])
Pc.<c> = QQ['c']
Kc.<cc> = Frac(QQ['cc'])


def n1(R, cv):
    E = NormallyOrderedExpressions(R, ['L', 'G'], parities={'G': 1})
    L, G = E.gens()
    return UniversalVertexAlgebra(
        R, E, {('L', 'L'): [L.derivative(), 2 * L, 0, R(cv) / 2],
               ('L', 'G'): [G.derivative(), R(3) / 2 * G], ('G', 'G'): [2 * L, 0, 2 * R(cv) / 3]})


def bc(R):
    return UniversalVertexAlgebra(R, ['b', 'c'], {('b', 'c'): [1]},
                                  parities={'b': 1, 'c': 1})


def eps_fun(q, p):
    return (-1) ** (q[0] * p[0] * 3)


cases = []
F1 = LatticeVertexAlgebra(QQ, matrix([[1]]))                     # odd
cases.append(('N=1 (x) V_Z(1), QQ', [n1(QQ, QQ(7) / 10), F1], QQ, None, 'q'))
H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
cases.append(('H (x) V_Z(2) (x) bc, QQ', [H, LatticeVertexAlgebra(QQ, matrix([[2]])), bc(QQ)],
              QQ, None, 'q'))
cases.append(('sl2_k (x) bc, Q(k)', [UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, conformal_vector='sugawara'),
                                     bc(Kk)], Kk, k, 'qpoly'))
cases.append(('N=1 (x) V_A2, QQ[c]', [n1(Pc, c), LatticeVertexAlgebra(Pc, matrix([[2, -1], [-1, 2]]))],
              Pc, c, 'qpoly'))
cases.append(('bc (x) V_{A1+A1} callable eps', [bc(QQ),
              LatticeVertexAlgebra(QQ, matrix([[2, 0], [0, 2]]), epsilon=eps_fun)], QQ, None, 'q'))
K5.<s5> = QuadraticField(5)
cases.append(('N=1 (x) V_A2, Q(sqrt 5)', [n1(K5, 134 + 60 * s5),
              LatticeVertexAlgebra(K5, matrix([[2, -1], [-1, 2]]))], K5, s5, 'nf'))
F5 = Frac(K5['k'])
cases.append(('sl2_k (x) bc, Q(sqrt 5)(k)', [UniversalAffineVertexAlgebra(F5, ['A', 1], level=F5.gen(), conformal_vector='sugawara'),
              bc(F5)], F5, F5.gen() + s5, 'nfmpoly'))
K5i.<i5> = K5.extension(QQ['y']([1, 0, 1]))
cases.append(('bc (x) V_A2, Q(sqrt 5)(i) (a tower)', [bc(K5i),
              LatticeVertexAlgebra(K5i, matrix([[2, -1], [-1, 2]]))], K5i, i5 * s5, 'nf'))
cases.append(('N=1 (x) V_Z(1), QQbar', [n1(QQbar, QQbar(2).sqrt()),
              LatticeVertexAlgebra(QQbar, matrix([[1]]))], QQbar, QQbar(3).sqrt(), 'nf'))
for name, factors, R, extra, kind in cases:
    T = factors[0].tensor(*factors[1:])
    TP = factors[0].tensor(*factors[1:], implementation='python')
    gens = []
    for i, V in enumerate(factors):
        gens += [T.embed(i, g) for g in V.gens()]
        if isinstance(V, LatticeVertexAlgebra):
            r = V.rank()
            gens += [T.embed(i, V.e(tuple(random.choice([-1, 1]) if j == 0 else 0 for j in range(r))))]
    X = rand_elem(T, gens, 4, R, extra)
    Y = rand_elem(T, gens, 4, R, extra)
    rp, used_p = run(TP, X, Y)
    re, used = run(T, X, Y)
    rt, _ = run(T, X, Y, 4)
    check(used and not used_p and T._engine().kind() == kind, '%s: engine (not) built' % name)
    check(srt(rp) == srt(re), '%s: engine differs from the Python recursion' % name)
    check(re == rt, '%s: threads change the result or its dict order' % name)
    print('%-34s ok' % name)

# a factor whose OPEs need u-variables (W_3 over Q(c): 1/(5c+22)): no engine
Ew = NormallyOrderedExpressions(Kc, ['L', 'W'])
Lw, Ww = Ew.gens()
Lam = Lw * Lw - 3 / 10 * Lw.derivative(2)
W = UniversalVertexAlgebra(Kc, Ew, {
    ('L', 'L'): [Lw.derivative(), 2 * Lw, 0, cc / 2], ('L', 'W'): [Ww.derivative(), 3 * Ww],
    ('W', 'W'): [Lw.derivative(3) / 15 + 16 / (22 + 5 * cc) * Lam.derivative(),
                 3 / 10 * Lw.derivative(2) + 32 / (22 + 5 * cc) * Lam, Lw.derivative(),
                 2 * Lw, 0, cc / 3]})
TW = W.tensor(bc(Kc))
check(TW._engine() is None, 'engine built although a factor needs u-variables')

# Ctrl-C in a tensor OPE (serial and threads), then consistent results
A = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=QQ(7) / 3, conformal_vector='sugawara')
LZ0 = LatticeVertexAlgebra(QQ, matrix([[2]]), names='K')
LZ = LZ0.with_conformal_vector(LZ0.J(0).no(LZ0.J(0)) / 4)
TB = A.tensor(LZ, degrees='factors')
keys = list(TB.weight_space(5).basis_keys())
big = sum(TB.monomial(m) for m in keys)
small = sum(TB.monomial(m) for m in keys[:6])


def _alarm(*args):
    raise KeyboardInterrupt


old = signal.signal(signal.SIGALRM, _alarm)
ref = None
for n in sorted({1, _threads(4)}):
    TB.clear_caches()
    set_nproc(n)
    t = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, float(0.3r))
    try:
        big.ope(big)
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    dt = time.perf_counter() - t
    check(got == 'interrupted' and dt < _CTRL_C_LIMIT, 'Ctrl-C with %d thread(s): %s after %.2f s' % (n, got, dt))
    r = raw(small.ope(small))
    ref = r if ref is None else ref
    check(r == ref, 'results after Ctrl-C (%d threads)' % n)
    print('Ctrl-C with %d thread(s): %s after %.2f s' % (n, got, dt))
signal.signal(signal.SIGALRM, old)
set_nproc(1)

print('crosscheck tensor engine: %d checks, mismatches: %d' % (checked, mismatches))
print('RESULT:', 'PASS' if mismatches == 0 else 'FAIL')
