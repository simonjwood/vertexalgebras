# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# UniversalVertexAlgebra: the C++ engine (_universal_engine)
# against the pure-Python recursion on every ring kind it takes (QQ, GF(p),
# QQ[c], QQ[a, b], number fields (also towers), polynomial rings over them,
# Frac(K[c]) with a denominator (u-variable), QQbar and AA through a number
# field), plus what only the engine has: threads
# (identical results, dict order included), Ctrl-C (KeyboardInterrupt, then
# consistent caches), errors in the recursion (GF(p) division by zero),
# a new algebra (with_opes) has its own engine, rings it leaves to pure Python, threads
# inside one OPE (work stealing).
# Run from the repository root:
#   sage tests/crosscheck_universal_engine.sage  -> RESULT: PASS
import os
import signal
import sys
import time

from vertex_algebras import universal_vertex_algebra as U
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions

if U._UVA_ENGINE is None:
    print('engine switched off:', U._ENGINE_OFF)
    print('RESULT: PASS (skipped)')
    sys.exit(0)

checked = mismatches = 0


def check(ok, what):
    global checked, mismatches
    checked += 1
    if not ok:
        mismatches += 1
        print('MISMATCH', what)


def n1_opes(R, cc):
    """The N=1 OPEs over R, central charge cc (G odd)."""
    E = NormallyOrderedExpressions(R, ['L', 'G'], parities={'G': 1})
    L, G = E.gens()
    return {('L', 'L'): [L.derivative(), 2 * L, 0, R(cc) / 2],
            ('L', 'G'): [G.derivative(), R(3) / 2 * G],
            ('G', 'G'): [2 * L, 0, 2 * R(cc) / 3]}


def n1(R, cc, implementation='engine'):
    """N=1 (odd G) over R with this implementation ('engine',
    'python' or None)."""
    return UniversalVertexAlgebra(R, ['L', 'G'], n1_opes(R, cc), parities={'G': 1},
                                  implementation=implementation)


def threaded(V, f, n=4):
    """f(V) from cold caches with n threads (then 1 thread, caches cleared)."""
    V.clear_caches()
    set_nproc(n)
    try:
        return f(V)
    finally:
        set_nproc(1)
        V.clear_caches()


def raw(x):
    """Results as ordered item lists (dict order is compared too)."""
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


def workload(V):
    L, G = V.gens()
    X = L.no(L).no(G) + 3 * G.derivative(2) + L.derivative().no(G.derivative())
    Y = G.no(L) - 5 * L.derivative(3).no(L) + G.derivative()
    Z = V._ope_raw_many(dict(X._monomial_coefficients),
                        [dict(Y._monomial_coefficients), dict(X._monomial_coefficients)])
    return raw([X.ope(Y), Y.ope(X), X.no(Y), X.nproduct(Y, -3), X.nproduct(Y, 2),
                X.derivative(4), L.ope(X), Z,
                V._ope_mono(((0, 0), (1, 0)), ((0, 1), (1, 0))),
                V._no_mono(((1, 1),), ((0, 2), (1, 0))),
                V._deriv_mono(((0, 1), (1, 0)), 3),
                V._nprod_mono(((1, 0),), -4, ((0, 0), (1, 0))), batch(V, X, Y)])


# 1. every ring kind: engine == pure Python; threads keep the order
Rab = PolynomialRing(QQ, ['a', 'b'])
Kn.<s> = NumberField(QQ['z']([-2, 0, 1]))
K3.<r> = NumberField(QQ['z']([-1, -1, 0, 3]))         # 3z^3 - z - 1: not monic
Kc.<kc> = Kn['c']
Kab = PolynomialRing(K3, ['a', 'b'])
Mrel.<t2> = Kn.extension(QQ['y']([1, 0, 1]))          # relative number field (a tower)
# polynomial rings over fraction fields (the bootstrap's algebras), lifted
# to QQ[c, x, y, u0] resp. K[c, x, u0] with u0 = 1/(c + 1) resp. 1/(c + s)
Fxy = PolynomialRing(Frac(QQ['c']), ['x', 'y'])
FKx = PolynomialRing(Frac(Kc), ['x'])
for R, cc, kind in [(QQ, QQ(7) / 3, 'q'), (GF(101), GF(101)(5), 'nmod'),
                    (GF(2^61 - 1), GF(2^61 - 1)(12345), 'nmod'),
                    (QQ['c'], QQ['c'].gen(), 'qpoly'), (Rab, Rab.gen(1) - 2 * Rab.gen(0), 'qmpoly'),
                    (Kn, s + 1, 'nf'), (K3, r^2 / 3 - 1, 'nf'), (Kc, kc + s, 'nfmpoly'),
                    (Kab, Kab.gen(1) - r * Kab.gen(0), 'nfmpoly'),
                    (Frac(Kc), Frac(Kc)(kc) * s, 'nfmpoly'),
                    (Mrel, t2 + s, 'nf'), (Mrel['c'], Mrel['c'].gen() * t2 - 1, 'nfmpoly'),
                    (QQbar, QQbar(2).sqrt() + 1, 'nf'), (AA, AA(3).sqrt() / 2, 'nf'),
                    (Fxy, Fxy.gen(0) / (Fxy.base_ring().gen() + 1) + Fxy.gen(1), 'qmpoly'),
                    (FKx, FKx.gen() * s / (FKx.base_ring().gen() + s) - 1, 'nfmpoly')]:
    VE, VX = (n1(R, cc, m) for m in ('engine', 'python'))
    check(VE._engine() is not None and VE._engine().kind() == kind
          and VX._engine() is None, 'engine kind over %s' % R)
    re, rx = (workload(V) for V in (VE, VX))
    rt = threaded(VE, workload)
    check(srt(re) == srt(rx), 'engine vs pure Python over %s' % R)
    check(re == rt, 'threads change results or dict order over %s' % R)
    print('%-60s ok' % R)

# 1b. Frac(K[c]) with a denominator in the declared OPEs: W_3 over Q(sqrt 2)(c),
# lifted to K[c, u0], u0 = 1/(5c + 22)
FK = Frac(Kc)


def w3(R, cv, implementation):
    E = NormallyOrderedExpressions(R, ['L', 'W'])
    L, W = E.gens()
    Lam = L * L - 3 / 10 * L.derivative(2)
    b = 16 / (22 + 5 * R(cv))
    return UniversalVertexAlgebra(R, E, {
        ('L', 'L'): [L.derivative(), 2 * L, 0, R(cv) / 2],
        ('L', 'W'): [W.derivative(), 3 * W],
        ('W', 'W'): [L.derivative(3) / 15 + b * Lam.derivative(),
                     3 / 10 * L.derivative(2) + 2 * b * Lam, L.derivative(),
                     2 * L, 0, R(cv) / 3]}, implementation=implementation)


def w3work(V):
    L, W = V.gens()
    X = (s + 2) * L.no(W) + W.derivative(2) + 3 * L.no(L.no(L))
    Y = W.no(W) - s * L.derivative(3).no(W)
    return raw([X.ope(Y), X.no(Y), X.nproduct(Y, -3), X.derivative(3), batch(V, X, Y)])


WE, WX = (w3(FK, FK(kc), m) for m in ('engine', 'python'))
check(WE._engine() is not None and WE._engine().kind() == 'nfmpoly'
      and WE._engine().ring().ngens() == 2, 'Frac(K[c]) W_3: engine over K[c, u0]')
re, rx = (w3work(V) for V in (WE, WX))
rt = threaded(WE, w3work)
check(srt(re) == srt(rx), 'W_3 over Frac(K[c]): engine vs pure Python')
check(re == rt, 'W_3 over Frac(K[c]): threads change results or dict order')
print('%-60s ok' % 'W_3 over Frac(Q(sqrt 2)[c]) (u-variable)')

# 1c. QQbar: the engine's number field grows when an element needs it
# (rebuild, then the same results as pure Python); results fed back in
VQ, VQc = n1(QQbar, QQbar(2).sqrt(), 'engine'), n1(QQbar, QQbar(2).sqrt(), 'python')
deg0 = VQ._engine().ring().degree()
out = []
for V in (VQ, VQc):
    L, G = V.gens()
    X = QQbar(3).sqrt() * L.no(G) + QQbar.zeta(3) * G.derivative()      # outside Q(sqrt 2)
    P = X.ope(G.no(L))
    out.append(raw(P + P[0].ope(X)))
check(deg0 == 2 and VQ._engine().ring().degree() == 8 and srt(out[0]) == srt(out[1]),
      'QQbar: field grown from degree %s to %s, same results as pure Python' % (deg0, VQ._engine().ring().degree()))
print('%-60s ok' % 'QQbar: engine field grown by a rebuild')

# 2. rings the engine leaves to pure Python: the default is pure Python
Fp.<t> = Frac(GF(1009)['t'])
for R, cc in [(Fp, t), (QQ['x'].fraction_field(), QQ['x'].fraction_field()(3))]:
    VE, VX = n1(R, cc, None), n1(R, cc, 'python')
    kind = VE._engine() and VE._engine().kind()
    if R is QQ['x'].fraction_field():
        check(kind == 'qpoly', 'Frac(QQ[x]) with constant OPE coefficients: %s' % kind)
    else:
        check(kind is None, 'engine used over %s' % R)
    check(srt(workload(VE)) == srt(workload(VX)), 'default vs pure Python over %s' % R)

# 3. other OPEs make another algebra with its own engine (no stale
# generator OPEs); the first algebra and its engine are untouched
V = n1(QQ, QQ(1))
L, G = V.gens()
E0 = V._engine()
before = G.ope(G)
V2 = V.with_opes(n1_opes(QQ, QQ(5)))
G2 = V2.gen('G')
check(V2 is not V and V2._engine() is not E0 and V._engine() is E0
      and G2.ope(G2)[2] == V2(QQ(10) / 3) and before[2] == V(QQ(2) / 3)
      and G.ope(G)[2] == V(QQ(2) / 3), 'engine of the algebra with other OPEs')

# 4. Ctrl-C (SIGALRM raising KeyboardInterrupt) serially and in threads:
# interrupted, the engine stays usable and gives the same results after
VIR = {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, QQ(7) / 12]}
VQ = UniversalVertexAlgebra(QQ, ['L'], VIR, degrees={'L': 2})
big = sum(VQ.monomial(k) for k in VQ.weight_space(13).basis_keys())
small = sum(VQ.monomial(k) for k in VQ.weight_space(7).basis_keys())


def _alarm(*args):
    raise KeyboardInterrupt


old = signal.signal(signal.SIGALRM, _alarm)
for n in (1, 4):
    VQ.clear_caches()
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
    check(got == 'interrupted' and dt < 2, 'Ctrl-C with %d thread(s): %s after %.2f s' % (n, got, dt))
    sizes = VQ._engine().cache_sizes()
    r1 = raw(small.ope(small.derivative()))
    VQ2 = UniversalVertexAlgebra(QQ, ['L'], VIR)       # another algebra: its own engine
    s2 = VQ2._from_dict(dict(small._monomial_coefficients))
    check(r1 == raw(s2.ope(s2.derivative())), 'results after Ctrl-C (%d threads)' % n)
    print('Ctrl-C with %d thread(s): %s after %.2f s, caches kept %s' % (n, got, dt, sizes))
signal.signal(signal.SIGALRM, old)
set_nproc(1)

# 5. errors inside the recursion: GF(5) needs 1/5! at weight 4 (as pure Python)
after = []
for mode in ('engine', 'python'):
    V5 = n1(GF(5), GF(5)(3), mode)
    L, G = V5.gens()
    X = L.no(L).no(L)
    try:
        X.ope(X)
        got = None
    except ZeroDivisionError:
        got = 'ZeroDivisionError'
    check(got == 'ZeroDivisionError', '%s over GF(5): %s' % (mode, got))
    for n in (1, 4):
        set_nproc(n)
        try:
            X.ope(X)
            got = None
        except ZeroDivisionError:
            got = 'ZeroDivisionError'
        check(got == 'ZeroDivisionError', '%s over GF(5), %d threads: %s' % (mode, n, got))
    set_nproc(1)
    if mode == 'engine':
        # (pure Python divides by 5! in L.no(L).ope(L) too: its recursion
        # order differs from the engine's)
        after.append(raw([L.ope(G.no(G)), L.no(L).ope(L), G.derivative(3).ope(L.no(G))]))
check(after[0][0] == raw([[
    V5.gen(0).derivative(2), 3 * V5.gen(0).derivative(),
    4 * V5.gen(0), V5.zero(), V5(1)]])[0], 'engine unusable after an error')

# 6. threads inside ONE OPE (work stealing, run tokens): single deep pairs
# give identical results (dict order too) for 1, 3, 8 and 16 threads (more
# threads than this test may have cores); Ctrl-C and errors inside stolen jobs
def deep(R, cc, n):
    """N=1 over R from cold caches with n threads, and two deep elements."""
    V = UniversalVertexAlgebra(R, ['L', 'G'], n1_opes(R, cc), parities={'G': 1})
    V.clear_caches()
    set_nproc(n)
    L, G = V.gens()
    X = L.no(L).no(G.derivative()).no(L)
    Y = L.derivative(2).no(G).no(L).no(L)
    return V, X, Y


res = {}
stolen = {}
for n in (1, 3, 8, 16):
    V, X, Y = deep(QQ['c'], QQ['c'].gen(), n)
    res[n] = raw([X.ope(Y), Y.ope(X), X.nproduct(Y, -2)])
    stolen[n] = V._engine().cache_sizes()['stolen']
check(res[1] == res[3] == res[8] == res[16], 'single deep OPE depends on the thread count')
# whether a given run steals is up to the scheduler (a pusher may take its
# jobs back before a woken thread arrives); over the three runs it does
check(stolen[1] == 0 and stolen[3] + stolen[8] + stolen[16] > 0, 'no work stealing: %s' % stolen)
print('single deep OPE, jobs stolen per thread count:', stolen)
old = signal.signal(signal.SIGALRM, _alarm)
V7 = UniversalVertexAlgebra(QQ, ['L'], VIR)
V7.clear_caches()
set_nproc(8)
X7 = V7.monomial(((0, 0),) * 7)
t = time.perf_counter()
signal.setitimer(signal.ITIMER_REAL, float(0.3r))
try:
    X7.ope(X7)
    got = 'finished'
except KeyboardInterrupt:
    got = 'interrupted'
signal.setitimer(signal.ITIMER_REAL, float(0r))
signal.signal(signal.SIGALRM, old)
dt = time.perf_counter() - t
set_nproc(1)
check(got == 'interrupted' and dt < 2, 'Ctrl-C in one deep OPE with 8 threads: %s after %.2f s' % (got, dt))
X4 = V7.monomial(((0, 0),) * 4)
VR = UniversalVertexAlgebra(QQ, ['L'], VIR, implementation='engine')   # not V7: another key
check(raw(X4.ope(X4)) == raw(VR.monomial(((0, 0),) * 4).ope(VR.monomial(((0, 0),) * 4))),
      'results after Ctrl-C in one deep OPE')
print('Ctrl-C in one deep OPE (8 threads): %s after %.2f s' % (got, dt))
try:
    V5, X5, Y5 = deep(GF(5), GF(5)(3), 8)   # building them divides by 5! already
    X5.ope(Y5)
    got = None
except ZeroDivisionError:
    got = 'ZeroDivisionError'
check(got == 'ZeroDivisionError', 'GF(5) error inside stolen jobs: %s' % got)

print('crosscheck engine: %d checks, mismatches: %d' % (checked, mismatches))
print('RESULT:', 'PASS' if mismatches == 0 else 'FAIL')
