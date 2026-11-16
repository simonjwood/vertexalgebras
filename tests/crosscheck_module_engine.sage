# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# InducedModule: the C++ engine of induced modules (_module_engine) against
# the pure-Python recursion (implementation='python') on every key up to a
# depth: generators, normally ordered products and derivatives acting by
# n-th products for a range of n.  Rings: QQ, Frac(QQ[c, h]) and W_3 over
# Q(c) (lifted), Q(k) (affine sl_2, a doublet top: zero-mode matrices), a
# number field; odd generators and half-integer weights (N = 1, βγ at 1/2);
# a proto universal algebra (the W(2) presentation) with a Jordan-block top;
# induction from degree n (higher Zhu algebras: a lowered top).
# Also: batches = single calls, results independent of the thread count
# (dict order too), Ctrl-C (KeyboardInterrupt, then the same results),
# the induced modules of quotients (engine = Python), refusals, pickling.
# Run from the repository root:
#   sage tests/crosscheck_module_engine.sage  -> RESULT: PASS
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.vertex_algebra import _compiled_module, VertexAlgebraParallelism
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from vertex_algebras.vertex_algebra_module import InducedModule
from axioms import Reporter

set_nproc = VertexAlgebraParallelism().set
if _compiled_module('_module_engine')[0] is None:
    print('engine switched off')
    print('RESULT: PASS (skipped)')
    sys.exit(0)

report = Reporter()
t_start = time.time()


def compare(V, U, els, depth, ns=range(-3, 5), cls=InducedModule):
    """Engine and Python agree on every key of depth <= depth, for every
    element of els (raw dicts) and n in ns; a batch of all of them too."""
    Me, Mp = cls(V, U), cls(V, U, implementation='python')
    if Me._engine() is None or Mp._engine() is not None:
        return False, 'engine missing'
    h = Me.top_weight()
    keys = [k for d in range(depth + 1) for k in Me._keys_of_tdeg(d)] if Me._mu is None else \
        [k for d in range(depth + 1) for k in Me._keys_of_tdeg(d)]
    ok, n = True, 0
    vecs, tasks = [], []
    for k in keys:
        vecs.append({k: 1})
        for e, d in enumerate(els):
            for nn in ns:
                tasks.append((e, nn, len(vecs) - 1))
                ok = ok and Me._action_raw(d, nn, {k: 1}) == Mp._action_raw(d, nn, {k: 1})
                n += 1
    batch = Me._action_raw_batch(els, vecs, tasks)
    ok = ok and batch == [Mp._action_raw(els[e], nn, vecs[v]) for e, nn, v in tasks]
    # key level (the quotients' path)
    k = keys[-1]
    a = next(iter(els[-1]))
    ok = ok and Me._action_mono(a, 1, k) == Mp._action_mono(a, 1, k)
    return ok, '%d products, %d keys, engine caches %s' % (n, len(keys), Me._engine().cache_sizes())


def raw(x):
    return dict(x._monomial_coefficients)


# 1. Virasoro: QQ, Frac(QQ[c, h]) (lifted), a number field
E.<L> = NormallyOrderedExpressions(QQ)
Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2 * L, 0, 1 / 4]}, conformal_vector=L)
Lg = Vir.gen('L')
els = [raw(Lg), raw(Lg.no(Lg)), raw(Lg.derivative(2).no(Lg) + 3 * Lg.no(Lg.no(Lg)))]
ok, info = compare(Vir, Vir.zhu_algebra().module({'L': 1 / 16}), els, 5)
report('Virasoro over QQ, h = 1/16', ok, info)
Kch.<c, h> = Frac(QQ['c', 'h'])
Ec.<Lc> = NormallyOrderedExpressions(Kch)
Virc = UniversalVertexAlgebra(Kch, Ec, {('Lc', 'Lc'): [Lc.derivative(), 2 * Lc, 0, c / 2]}, conformal_vector=Lc)
Lcg = Virc.gen('Lc')
ok, info = compare(Virc, Virc.zhu_algebra().module({'Lc': h}),
                   [raw(Lcg), raw(c * Lcg.no(Lcg) - h * Lcg.derivative(2))], 4)
report('Virasoro over Q(c, h) (lifted to QQ[c, h])', ok, info)
K5.<s5> = QuadraticField(5)
E5.<L5> = NormallyOrderedExpressions(K5)
Vir5 = UniversalVertexAlgebra(K5, E5, {('L5', 'L5'): [L5.derivative(), 2 * L5, 0, s5]}, conformal_vector=L5)
L5g = Vir5.gen('L5')
ok, info = compare(Vir5, Vir5.zhu_algebra().module({'L5': 1 + s5}), [raw(L5g), raw(s5 * L5g.no(L5g))], 4)
report('Virasoro over Q(sqrt 5)', ok, info)

# 2. N = 1 (an odd generator of weight 3/2, odd squares) and βγ at weights 1/2
En.<Ln, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
N1 = UniversalVertexAlgebra(QQ, En, {('Ln', 'Ln'): [Ln.derivative(), 2 * Ln, 0, 7 / 20],
                                     ('Ln', 'G'): [G.derivative(), 3 / 2 * G],
                                     ('G', 'G'): [2 * Ln, 0, 7 / 15]}, conformal_vector=Ln)
Lng, Gg = N1.gens()
ok, info = compare(N1, N1.zhu_algebra().module({'Ln': 1 / 10}),
                   [raw(Gg), raw(Lng), raw(Gg.no(Gg.derivative())), raw(Lng.no(Gg))], 4)
report('N = 1 (odd, half-integer weights)', ok, info)
Eb.<beta, gamma> = NormallyOrderedExpressions(QQ)
BG = UniversalVertexAlgebra(QQ, Eb, {('beta', 'gamma'): [1]}, degrees={'beta': 1 / 2, 'gamma': 1 / 2})
bg, gg = BG.gens()
ZB = BG.zhu_algebra()
ok, info = compare(BG, ZB.module({}) if ZB.ngens() == 0 else ZB.module({n: 0 for n in ZB.variable_names()}),
                   [raw(bg), raw(gg), raw(bg.no(gg)), raw(gg.derivative().no(bg))], 4)
report('βγ at weights 1/2 (no zero modes)', ok, info)

# 3. W_3 over QQ and over Q(c) (lifted with u0 = 1/(5c + 22)); affine sl_2 over Q(k)
def w3(R, cv):
    Ew = NormallyOrderedExpressions(R, ['L', 'W'])
    L, W = Ew.gens()
    Lam = L * L - 3 / 10 * L.derivative(2)
    b = 16 / (22 + 5 * R(cv))
    return UniversalVertexAlgebra(R, Ew, {
        ('L', 'L'): [L.derivative(), 2 * L, 0, R(cv) / 2],
        ('L', 'W'): [W.derivative(), 3 * W],
        ('W', 'W'): [L.derivative(3) / 15 + b * Lam.derivative(),
                     3 / 10 * L.derivative(2) + 2 * b * Lam, L.derivative(),
                     2 * L, 0, R(cv) / 3]}, conformal_vector=L)


for R, cv, name in ((QQ, QQ(2), 'W_3 over QQ (c = 2)'),
                    (Frac(QQ['cw']), Frac(QQ['cw']).gen(), 'W_3 over Q(c) (lifted, u0)')):
    W3 = w3(R, cv)
    Lw, Ww = W3.gens()
    U = W3.zhu_algebra().module({'L': R(1) / 3, 'W': R(2)})
    ok, info = compare(W3, U, [raw(Ww), raw(Lw.no(Ww)), raw(Ww.no(Ww))], 3, ns=range(-2, 6))
    report(name, ok, info)
Kk.<k> = Frac(QQ['k'])
Va = UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, names=['E', 'H', 'F'], conformal_vector='sugawara')
Ea, Ha, Fa = Va.gens()
Ud = Va.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
                              'F': matrix([[0, 0], [1, 0]])})
ok, info = compare(Va, Ud, [raw(Ea), raw(Fa), raw(Ea.no(Fa)), raw(Ha.no(Ha.derivative()))], 3)
report('affine sl_2 over Q(k), the doublet top (zero-mode matrices)', ok, info)

# 4. a proto universal algebra: the W(2) presentation, a Jordan-block top
Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm * cp,
                            horizontal_grading={'cp': 1, 'cm': -1})
a, b = SF.gens()
S2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()), a.no(b.derivative()) + b.no(a.derivative()),
                    b.no(b.derivative())], names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient',
                   conformal_vector='ambient')
P2 = S2.presentation()
F2 = P2.ambient()
Z2 = P2.zhu_algebra_up_to(8)
Z0 = matrix(QQ, 2, 2)
Uj = Z2.module({'T': matrix([[0, 1], [0, 0]]), 'Wp': Z0, 'W0': Z0, 'Wm': Z0}, horizontal_grades=[0, 0])
Tg, Wpg, W0g, Wmg = F2.gens()
ok, info = compare(F2, Uj, [raw(Wpg), raw(Wmg), raw(W0g.no(Tg)), raw(Wpg.no(Wmg))], 3)
report('W(2) presentation (proto), Jordan top', ok, info)
dims = {}
for impl in (None, 'python'):
    Mq = P2.induced_module_up_to(Uj, 8, 4, implementation=impl)
    dims[impl] = [[Mq.weight_space(d, horizontal_grade=g).dimension() for g in (-4, -2, 0, 2, 4)] for d in range(5)]
report('induced modules of the quotient: engine = Python (W(2), Jordan top, depth 4)',
       dims[None] == dims['python'], str([sum(r) for r in dims[None]]))

# 4b. induction from degree n (higher Zhu algebras): a lowered top
from vertex_algebras.higher_zhu_algebra import HigherInducedModule
V37 = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2 * L, 0, 3 / 14]}, conformal_vector=L)
L37 = V37.gen('L')
els37 = [raw(L37), raw(L37.no(L37)), raw(L37.derivative(2).no(L37) + 3 * L37.no(L37.no(L37)))]
A1 = V37.zhu_algebra(level=1)
ok1, info1 = compare(V37, A1.module({'L_0': identity_matrix(QQ, 2), 'L_{-1} L_1': matrix([[0, 1], [0, 0]])}),
                     els37, 4, cls=HigherInducedModule)
A2 = V37.zhu_algebra(level=2)
ok2, info2 = compare(V37, A2.module({w: 3 if w == ((0, 0),) else 0 for w in A2._generator_words()}),
                     els37, 4, cls=HigherInducedModule)
report('Virasoro levels 1 (Jordan) and 2: induction from degree n', ok1 and ok2, info1)
Z1 = P2.zhu_algebra_up_to(8, level=1, degree_bound=16)
B1 = list(Z1.basis())
pos1 = {x.leading_support(): i for i, x in enumerate(B1)}


def vec1(x):
    v = vector(QQ, len(B1))
    for k, c in x.monomial_coefficients().items():
        v[pos1[k]] = c
    return v


T1 = Z1(P2.gen('T'))
x_ = polygen(QQ)
mp = x_^2 * (x_ - 1)^2 * (x_ + 1/8) * (x_ - 3/8) * (x_ - 7/8) * (x_ - 11/8) * (x_ - 2)
rest = mp // (x_ - 3/8)
pe = (xgcd(x_ - 3/8, rest)[2] * rest) % mp            # the idempotent of the [T] = 3/8 block
e1 = sum((c * T1^i for i, c in enumerate(pe.list())), Z1.zero())
blk = matrix(QQ, [vec1(b * e1) for b in B1]).row_space().basis_matrix()
els1 = [sum((r[i] * B1[i] for i in range(len(B1))), Z1.zero()) for r in blk.rows()]


def left1(w):
    x = Z1._from_dict(Z1._nf({w: 1}), coerce=False, remove_zeros=True)
    return matrix(QQ, [blk.solve_left(vec1(x * y)) for y in els1]).transpose()


U1 = Z1.module({w: left1(w) for w in Z1._generator_words()})
ok, info = compare(F2, U1, [raw(Wpg), raw(W0g.no(Tg))], 2, cls=HigherInducedModule)
dims = {}
for impl in (None, 'python'):
    Mq = P2.induced_module_up_to(U1, 8, 3, implementation=impl)
    hq = Mq._cover_module.top_weight()
    dims[impl] = [Mq.weight_space(hq + d).dimension() for d in range(4)]
report('W(2) A_1, the [T] = 3/8 block (dim %d): engine = Python, also induced up to depth 3'
       % blk.nrows(), ok and dims[None] == dims['python'], '%s %s' % (dims[None], info))

# 5. threads: identical results, dict order included; Ctrl-C
W3 = w3(QQ, QQ(2))
Lw, Ww = W3.gens()
U = W3.zhu_algebra().module({'L': QQ(1) / 3, 'W': QQ(2)})
keys = [k for d in range(5) for k in InducedModule(W3, U)._keys_of_tdeg(d)]
els = [raw(Ww), raw(Ww.no(Ww)), raw(Lw.no(Lw.no(Ww)))]
vecs = [{k: 1} for k in keys]
tasks = [(e, n, v) for v in range(len(vecs)) for e in range(len(els)) for n in range(-2, 4)]
res = {}
for T in (1, 4):
    M = InducedModule(W3, U)
    M.clear_caches()
    set_nproc(T)
    res[T] = [list(r.items()) for r in M._action_raw_batch(els, vecs, tasks)]
    set_nproc(1)
report('threads: the same results and dict order (1 and 4 threads)', res[1] == res[4],
       '%d tasks' % len(tasks))


def _alarm(*args):
    raise KeyboardInterrupt


old = signal.signal(signal.SIGALRM, _alarm)
M = InducedModule(W3, U)
ok = True
for T in (1, 4):
    M.clear_caches()
    set_nproc(T)
    keys7 = [k for d in range(8) for k in M._keys_of_tdeg(d)]
    signal.setitimer(signal.ITIMER_REAL, float(0.2r))
    try:
        M._action_raw_batch(els, [{k: 1} for k in keys7],
                            [(e, n, v) for v in range(len(keys7)) for e in range(3) for n in range(-2, 4)])
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    set_nproc(1)
    again = [list(r.items()) for r in M._action_raw_batch(els, vecs, tasks)]
    ok = ok and got == 'interrupted' and again == res[1]
    print('Ctrl-C with %d thread(s): %s' % (T, got))
signal.signal(signal.SIGALRM, old)
report('Ctrl-C: KeyboardInterrupt, then the same results', ok)

# 6. refusals, pickling
refused = []
try:
    InducedModule(Va, Va.zhu_algebra().verma_module({'H': 1}), implementation='engine')
    refused.append('infinite top')
except ValueError:
    pass
try:
    InducedModule(Vir, Vir.zhu_algebra().module({'L': 0}), implementation='cython')
    refused.append('cython')
except ValueError:
    pass
Me = InducedModule(Vir, Vir.zhu_algebra().module({'L': 1 / 16}))
report('refusals (infinite top with the engine, cython), pickling',
       not refused and loads(dumps(Me)) is Me and Me is not InducedModule(
           Vir, Vir.zhu_algebra().module({'L': 1 / 16}), implementation='python'), refused)

# 7. the free mode module of the higher Zhu algebras (He's model): the
# engine's free mode (set_free) against the Python recursion, odd
# generators, a lifted ring, levels 1-3; batches and threads
import itertools
from vertex_algebras.higher_zhu_algebra import _FreeModeModule
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra as _UVA
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions as _NOE
import vertex_algebras as _va
_Kc = Frac(QQ['c']); _c = _Kc.gen()


def _free_cases(impl):
    Y = _NOE(QQ, ['cp', 'cm'], parities={'cp': 1, 'cm': 1})
    cp, cm = Y.gens()
    return [_UVA(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm * cp, implementation=impl),
            _va.W3(_Kc, _c, conformal_vector='L', implementation=impl)]


ok = True
for Ve, Vp in zip(_free_cases(None), _free_cases('python')):
    h = tuple(ZZ(Ve.degree(g)) for g in Ve.gens())
    odd = tuple(bool(g.parity()) for g in Ve.gens())
    for n in (1, 2, 3):
        Pe, Pp = _FreeModeModule(Ve, n, h, odd), _FreeModeModule(Vp, n, h, odd)
        modes = sorted([(i, m) for i in range(len(h)) for m in range(-2, n + 1)], key=lambda t: (t[1], t[0]))
        words = [()]
        for L in (1, 2):
            for w in itertools.combinations_with_replacement(modes, L):
                if any(w[k] == w[k + 1] and odd[w[k][0]] for k in range(L - 1)) or Pp._deg(w) < 0:
                    continue
                if all(m > 0 for _, m in w) and sum(m for _, m in w) > n:
                    continue
                words.append(w)
        ok = ok and Pe._engine() is not None and all(
            Pe._gen_act(i, m, w) == Pp._gen_act(i, m, w) for i, m in modes for w in words)
        tasks = [(i, m, {w: 1}) for i, m in modes for w in words[:30]]
        set_nproc(1)
        r1 = Pe.act_many(tasks)
        set_nproc(T)
        rT = Pe.act_many(tasks)
        set_nproc(1)
        ok = ok and r1 == rT == Pp.act_many(tasks)
report('free mode module (higher Zhu algebras): engine = Python, batches, threads', ok)

print('total %.1fs' % (time.time() - t_start))
report.result()
