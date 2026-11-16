# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# LatticeVertexAlgebra: the C++ engine (_lattice_engine)
# against the Cython core on odd, even, indefinite and rank-3 lattices, a
# callable cocycle, the base rings QQ, QQ[c], Frac(QQ[c]), number fields
# (also towers), polynomial rings over them, Frac(K[c]), QQbar and AA (a
# number field, grown when elements need it) (coefficients with non-constant
# denominators take the generic path), every product kind,
# key level and element level; threads (identical results, dict order too);
# Ctrl-C (KeyboardInterrupt, then consistent caches).
# Run from the repository root:
#   sage tests/crosscheck_lattice_engine.sage  -> RESULT: PASS
import os
import random
import signal
import sys
import time

from vertex_algebras import lattice_vertex_algebra as LV
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra

if LV._LVA_ENGINE is None:
    print('engine switched off:', LV._ENGINE_OFF)
    print('RESULT: PASS (skipped)')
    sys.exit(0)

checked = mismatches = 0


# Ctrl-C timing checks use at most as many threads as there are cores
# (oversubscribed threads notice the interrupt late on small CI runners)
import os as _os
_NCORES = _os.cpu_count() or 1


def _threads(n):
    return min(n, _NCORES)

def check(ok, what):
    global checked, mismatches
    checked += 1
    if not ok:
        mismatches += 1
        print('MISMATCH', what)


random.seed(3r)


def keys_of(L, n, maxlet=3, maxder=2, qrange=1):
    r = L.rank()
    out = set()
    while len(out) < n:
        lets = tuple(sorted(((random.randrange(r), random.randint(0, maxder))
                             for _ in range(random.randint(0, maxlet))),
                            key=lambda t: (t[1], -t[0]), reverse=True))
        out.add((lets, tuple(random.randint(-qrange, qrange) for _ in range(r))))
    return sorted(out)


def raw(x):
    """Ordered item lists: dict order is compared too."""
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


def work(L, X, Y):
    kx, ky = sorted(X.support())[0], sorted(Y.support())[0]
    return raw([X.ope(Y), Y.ope(X), X.no(Y), X.nproduct(Y, -3), X.nproduct(Y, 1),
                X.derivative(3), L._ope_mono(kx, ky), L._nprod_mono(kx, -2, ky),
                L._deriv_mono(kx, 2),
                L._ope_raw_many(dict(X._monomial_coefficients),
                                [dict(Y._monomial_coefficients), dict(X._monomial_coefficients)]),
                batch(L, X, Y)])


def run(L, X, Y, nproc=1):
    """work() in L (X, Y carried over by their basis keys) from cold caches
    with nproc threads; and whether L has an engine."""
    L.clear_caches()
    set_nproc(nproc)
    X, Y = (L._from_dict(dict(Z._monomial_coefficients)) for Z in (X, Y))
    try:
        return work(L, X, Y), L._engine() is not None
    finally:
        set_nproc(1)


def eps_fun(q, p):
    return (-1) ** (q[0] * p[1])


Kc.<c> = Frac(QQ['c'])
Pc = QQ['c']
K5.<s5> = QuadraticField(5)
K3.<z3> = NumberField(QQ['y']([-2, 0, 0, 1]))
P5 = PolynomialRing(K5, ['c', 'd'])
F5 = Frac(K5['c'])
K5i.<i5> = K5.extension(QQ['y']([1, 0, 1]))


def eps_fun1(q, p):
    return (-1) ** (q[0] * p[0])
cases = [('Z, B = 1 (odd)', QQ, matrix([[1]]), None, None),
         ('Z, B = 12', QQ, matrix([[12]]), None, None),
         ('A2', QQ, matrix([[2, -1], [-1, 2]]), None, None),
         ('I_{1,1}', QQ, matrix([[1, 0], [0, -1]]), None, None),
         ('rank 3', QQ, matrix([[2, 1, 0], [1, 4, -1], [0, -1, 2]]), None, None),
         ('A1 + A1, callable cocycle', QQ, matrix([[2, 0], [0, 2]]), eps_fun, None),
         ('A2 over QQ[c]', Pc, matrix([[2, -1], [-1, 2]]), None, Pc.gen()),
         ('A2 over Q(c)', Kc, matrix([[2, -1], [-1, 2]]), None, c),
         ('A2 over Q(c), 1/(c+3)', Kc, matrix([[2, -1], [-1, 2]]), None, 1 / (c + 3)),
         ('A2 over Q(sqrt 5)', K5, matrix([[2, -1], [-1, 2]]), None, s5),
         ('Z, B = 3, over Q(cbrt 2), callable', K3, matrix([[3]]), eps_fun1, z3 + 1),
         ('A2 over Q(sqrt 5)[c, d]', P5, matrix([[2, -1], [-1, 2]]), None, P5.gen(0) - s5 * P5.gen(1)),
         ('A2 over Q(sqrt 5)(c), 1/(c+s)', F5, matrix([[2, -1], [-1, 2]]), None, 1 / (F5.gen() + s5)),
         ('A2 over Q(sqrt 5)(i) (a tower)', K5i, matrix([[2, -1], [-1, 2]]), None, i5 + s5),
         ('A2 over QQbar (field grown)', QQbar, matrix([[2, -1], [-1, 2]]), None, QQbar(2).sqrt()),
         ('Z, B = 3, over AA, callable', AA, matrix([[3]]), eps_fun1, AA(3).sqrt())]
for name, R, Bm, eps, g in cases:
    L = LatticeVertexAlgebra(R, Bm, epsilon=eps)
    ks = keys_of(L, 8)
    X = sum((R(random.randint(1, 5)) + (g if g is not None and i % 2 else 0)) * L.monomial(k)
            for i, k in enumerate(ks[:5]))
    Y = sum((R(random.randint(-5, 5)) + (g ** 2 if g is not None and i % 3 == 0 else 0))
            * L.monomial(k) for i, k in enumerate(ks[3:]))
    LC = LatticeVertexAlgebra(R, Bm, epsilon=eps, implementation='python')
    LE = LatticeVertexAlgebra(R, Bm, epsilon=eps, implementation='engine')
    rc, _ = run(LC, X, Y)
    re, used = run(LE, X, Y)
    rt, _ = run(LE, X, Y, 4)
    check(used, 'engine not built over %s' % R)
    check(srt(rc) == srt(re), '%s: engine differs from the Cython core' % name)
    check(re == rt, '%s: threads change the result or its dict order' % name)
    print('%-28s ok' % name)

# Ctrl-C (SIGALRM raising KeyboardInterrupt), serially and in threads: the
# engine stays usable and gives the same results afterwards
L = LatticeVertexAlgebra(QQ, matrix([[12]]), names='K')
Jk = L.J(0)
L = L.with_conformal_vector(Jk.no(Jk) / 24 + QQ(1) / 12 * Jk.derivative())
L = L.with_horizontal_grading('lattice')
big = sum(L.monomial(k) for k in L.weight_space(16, 1).basis_keys())     # ~1.5 s on 10 threads:
bigm = sum(L.monomial(k) for k in L.weight_space(16, -1).basis_keys())   # never done in 0.3 s
small = sum(L.monomial(k) for k in L.weight_space(7, 1).basis_keys())
smallm = sum(L.monomial(k) for k in L.weight_space(7, -1).basis_keys())


def _alarm(*args):
    raise KeyboardInterrupt


old = signal.signal(signal.SIGALRM, _alarm)
ref = None
for n in sorted({1, _threads(4)}):
    L.clear_caches()
    set_nproc(n)
    t = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, float(0.3r))
    try:
        big.ope(bigm)
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    dt = time.perf_counter() - t
    check(got == 'interrupted' and dt < 2, 'Ctrl-C with %d thread(s): %s after %.2f s' % (n, got, dt))
    r = raw(small.ope(smallm))
    ref = r if ref is None else ref
    check(r == ref, 'results after Ctrl-C (%d threads)' % n)
    print('Ctrl-C with %d thread(s): %s after %.2f s' % (n, got, dt))
# Ctrl-C inside a batch of products (serially and in threads), then the
# same results as an uninterrupted batch
ks1 = list(L.weight_space(14, 1).basis_keys())
ks2 = list(L.weight_space(14, -1).basis_keys())
els = [{k: QQ(1)} for k in ks1 + ks2]
tasks = [(i, n, len(ks1) + j) for i in range(len(ks1)) for j in range(len(ks2))
         for n in (-3, -2, -1, 0, 1, 2)]
ref_b = None
for n in sorted({1, _threads(4)}):
    L.clear_caches()
    set_nproc(n)
    t = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, float(0.3r))
    try:
        L._nprod_raw_batch(els, tasks)
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    dt = time.perf_counter() - t
    check(got == 'interrupted' and dt < 2, 'Ctrl-C in a batch, %d thread(s): %s after %.2f s' % (n, got, dt))
    r = raw(L._nprod_raw_batch(els, tasks[::97]))
    ref_b = r if ref_b is None else ref_b
    check(r == ref_b and r == raw([L._nprod_raw(els[i], m, els[j]) for i, m, j in tasks[::97]]),
          'batch results after Ctrl-C (%d threads)' % n)
    print('Ctrl-C in a batch with %d thread(s): %s after %.2f s' % (n, got, dt))
signal.signal(signal.SIGALRM, old)
set_nproc(1)
LC = LatticeVertexAlgebra(QQ, matrix([[12]]), names='K', implementation='python')
sc, smc = (LC._from_dict(dict(Z._monomial_coefficients)) for Z in (small, smallm))
check(srt(raw(sc.ope(smc))) == srt(ref), 'results after Ctrl-C differ from the core')

print('crosscheck lattice engine: %d checks, mismatches: %d' % (checked, mismatches))
print('RESULT:', 'PASS' if mismatches == 0 else 'FAIL')
