# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# VertexSubalgebra: strong generation closure, membership, weight spaces
# against independent characters, relations (simple quotients), gradings,
# Virasoro vectors, errors, plumbing (axioms, tensor products, nesting).
# Run from the repository root:  sage tests/smoke_subalgebras.sage  -> RESULT: PASS
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.vertex_subalgebra import StrongGenerationError, VertexSubalgebra
from axioms import (Reporter, skew_ok, commutator_ok, associativity_ok,
                    raw_caches_ok)

report = Reporter()
t_start = time.time()


def raises(exc, f, *args, **kw):
    try:
        f(*args, **kw)
    except exc:
        return True
    return False


def p(n):
    return Partitions(n).cardinality() if n >= 0 else 0


# 1. Virasoro in Heisenberg (background charge, symbolic and a = 0) ---------
K.<a> = Frac(QQ['a'])
for R, aa, label in [(K, a, 'symbolic a'), (QQ, 0, 'a = 0')]:
    H0 = UniversalVertexAlgebra(R, ['J'], {('J', 'J'): [0, 1]})
    J = H0.gen('J')
    H = H0.with_conformal_vector(J.no(J) / 2 + aa * J.derivative())
    J, T = H.gen('J'), H.conformal_vector()
    S = H.subalgebra([T], names=['T'])
    report('Heisenberg Vir (%s): strong generators [T]' % label,
           list(S.variable_names()) == ['T'])
    dims = [S.weight_space(n).dimension() for n in range(9)]
    report('Heisenberg Vir (%s): dim = p(n) - p(n-1)' % label,
           dims == [1] + [p(n) - p(n - 1) for n in range(1, 9)], dims)
    report('Heisenberg Vir (%s): J, :JJ: not in S, :TT: in S' % label,
           J not in S and (aa == 0) == (J.no(J) in S) and T.no(T) in S)
    St = S.with_conformal_vector('ambient')
    report('Heisenberg Vir (%s): inherited c = 1 - 12a^2' % label,
           St.central_charge() == 1 - 12 * aa**2 and S.conformal_vector() is None
           and St._core is S._core)
report('S keeps no caches of its own (delegates to V)',
       not S._ope_cache and not S._no_cache and raw_caches_ok(H))

# 2. Majorana fermion: <T> = L(1/2, 0) = even part ---------------------------
X = NormallyOrderedExpressions(QQ, ['psi'], parities={'psi': 1})
psi = X.gen(0)
F = UniversalVertexAlgebra(QQ, X, {('psi', 'psi'): [1]},
                           conformal_vector=psi.derivative() * psi / 2)
psi, T = F.gen('psi'), F.conformal_vector()
S = F.subalgebra([T], names=['T'])
dims = [S.weight_space(n).dimension() for n in range(9)]
even = [F.weight_space(n).dimension() for n in range(9)]
report('Majorana <T>: dims == even part of the fermion', dims == even, dims)
report('Majorana <T>: dim 3 at weight 6 (universal Vir: 4)', dims[6] == 3)
report('Majorana <T>: psi not in S, :psi d^3psi: in S',
       psi not in S and psi.no(psi.derivative(3)) in S)
c = S.weight_space(6).coordinates(T.no(T.no(T)))
report('Majorana: express handles relations (element round trip)',
       all(S(S.express(x)) == S(x) for x in
           [T.no(T.no(T)), T.derivative(2).no(T.no(T)), T.derivative(6)
            + T.no(T.derivative(2)), T.no(T).nproduct(T, -2)]))
report('coordinates reproduce the element',
       sum(ci * b for ci, b in zip(c, S.weight_space(6).basis()))
       == S(T.no(T.no(T))))

# 3. principal subspace of V_{A1} --------------------------------------------
L0 = LatticeVertexAlgebra(QQ, matrix([[2]]))
Jl = L0.J(0)
L = L0.with_conformal_vector(Jl.no(Jl) / 4)
S = L.subalgebra([L.e((1,))], names=['e'])
q = PowerSeriesRing(QQ, 'q', default_prec=12).gen()
ch = sum(q**(n * n) / prod(1 - q**k for k in range(1, n + 1)) for n in range(4))
dims = [S.weight_space(n).dimension() for n in range(11)]
report('principal subspace: Rogers-Ramanujan character',
       dims == [ch[n] for n in range(11)], dims)
Lg = L.with_horizontal_grading('lattice')
S = Lg.subalgebra([Lg.e((1,))], names=['e'], horizontal_grading='ambient')
ok = all(S.weight_space(W, horizontal_grade=n).dimension()
         == (Partitions(W - n * n, max_length=n).cardinality()
             if W >= n * n else 0)
         for n in range(4) for W in range(11))
report('principal subspace: grade n -> partitions into <= n parts', ok)
S2 = L.subalgebra([L.e((1,)), L.e((-1,))])
report('<e^a, e^-a> = V_A1 (dims up to weight 5)',
       all(S2.weight_space(n).dimension() == L.weight_space(n).dimension()
           for n in range(6)))
# lattice roots compute the monomials (∂^n g)_(-1) b as n! g_(-n-1) b
words = [w for w in S2._vecs if len(w) > 1 and w[0][1]]
report('lattice spans: n! g_(-n-1) b = (∂^n g)_(-1) b (%d words)' % len(words),
       L._deriv_no_by_nproduct and len(words) > 10 and all(
           S2._vecs[w] == S2._deriv(*w[0]).no(S2._vec(w[1:])) for w in words))

# 4. the beta-gamma example (user notebook): <Ts, W> -------------------------
K6.<s6> = QuadraticField(6)
X = NormallyOrderedExpressions(K6, ['b', 'g'])
b, g = X.gens()
B = UniversalVertexAlgebra(K6, X, {('b', 'g'): [-1]}, conformal_vector=-b * g.derivative(),
                           horizontal_grading={'b': 1, 'g': -1})
b, g = B.gens()
Ts = (g.no(g.no(b.no(b))) / 2 + b.derivative().no(g) - b.no(g.derivative()))
W = s6 * (g.no(g.no(g.no(b.no(b.no(b))))) / 9
          - g.no(g.derivative().no(b.no(b))) / 2
          + g.no(g.no(b.no(b.derivative()))) / 2
          + g.derivative(2).no(b) / 6
          - 2 * g.derivative().no(b.derivative()) / 3
          + g.no(b.derivative(2)) / 6)
S = B.subalgebra([Ts, W], names=['T', 'W'], max_closure_steps=1)
report('beta-gamma <Ts, W>: closes in 1 round, 2 strong generators',
       list(S.variable_names()) == ['T', 'W'])
report('beta-gamma: W_(1)W in S, :bg: not in S',
       W.nproduct(W, 1) in S and b.no(g) not in S)
report('beta-gamma: Tbg not in S (conformal_vector=ambient refused)',
       raises(ValueError, S.with_conformal_vector, 'ambient'))
S0 = S
S = S.with_conformal_vector(Ts)
report('beta-gamma: S.with_conformal_vector(Ts), c = -2, closure shared',
       S.central_charge() == -2 and S._core is S0._core
       and S0.conformal_vector() is None)
Tsg, Wsg = S.gens()
els = [Wsg.nproduct(Wsg, k - 1) for k in range(1, 7)] + [Tsg.no(Wsg).derivative(),
       Wsg.no(Tsg) + 2 * Tsg.no(Tsg.no(Tsg)), Tsg.nproduct(Wsg, -3)]
report('beta-gamma: express(x).element() == x',
       all(S(S.express(x)) == x for x in els))
report('beta-gamma: W_(1)W = 8/3 :TT: - 1/2 d^2 T (notebook)',
       repr(S.express(W.nproduct(W, 1))) == '8/3*:T T: - 1/2*d^2T',
       repr(S.express(W.nproduct(W, 1))))
S.print_options(generators='strong')
report("beta-gamma: print mode 'generators': Ts.ope(Ts)",
       repr(Tsg.ope(Tsg)) == '[dT, 2*T, 0, -1]', repr(Tsg.ope(Tsg)))
S.print_options(generators='ambient')
report('beta-gamma: express of a non-member raises',
       raises(ValueError, S.express, b.no(g)))
Sg = S.with_horizontal_grading('ambient')
report('beta-gamma: graded weight spaces (grade 0 = all, grade 1 empty)',
       Sg.weight_space(4, horizontal_grade=0).dimension() == S.weight_space(4).dimension()
       and Sg.weight_space(4, horizontal_grade=1).dimension() == 0,
       S.weight_space(4).dimension())
report('beta-gamma: with_grading adopts the strong generators, keeps T',
       Sg.variable_names() == S.variable_names()
       and Sg.central_charge() == -2 and Sg._core is not S._core)

# 5. rounds: affine sl2 from E, F --------------------------------------------
X = NormallyOrderedExpressions(QQ, ['E', 'H', 'F'])
E, Hc, Fc = X.gens()
A = UniversalVertexAlgebra(QQ, X, {('H', 'H'): [0, 2], ('H', 'E'): [2*E], ('H', 'F'): [-2*Fc],
                                   ('E', 'F'): [Hc, 1]},
                           conformal_vector=(Hc * Hc / 2 + E * Fc + Fc * E) / 6)
E, Hc, Fc = A.gens()
try:
    A.subalgebra([E, Fc], max_closure_steps=1)
    good = False
except StrongGenerationError as err:
    good = len(err.generators) == 3 and err.untested > 0
report('sl2 <E, F>: 1 round raises StrongGenerationError', good)
S = A.subalgebra([E, Fc], names=['E', 'F'], max_closure_steps=2)
report('sl2 <E, F>: 2 rounds, strong generators E, F, H',
       list(S.variable_names()) == ['E', 'F', 'S0'] and S.gen('S0') == S(Hc))
report('sl2 <E, F>: same dimensions as V (weights <= 4)',
       all(S.weight_space(n).dimension() == A.weight_space(n).dimension()
           for n in range(5)))

# 6. errors -------------------------------------------------------------------
H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]},
                           conformal_vector={(('J', 0), ('J', 0)): 1/2})
J = H.gen('J')
report('inhomogeneous generator: ValueError',
       raises(ValueError, H.subalgebra, [J + J.no(J)]))
report('weight-0 generator gamma: NotImplementedError',
       raises(NotImplementedError, B.subalgebra, [g]))
S = H.subalgebra([J.no(J)])
report('S(v) for v not in S: ValueError; v in S False',
       raises(ValueError, S, J) and J not in S)
H2 = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
report('no conformal weights on V: ValueError (H2 is not H: the data differ)',
       H2 is not H and H.conformal_vector() is not None
       and raises(ValueError, H2.subalgebra, [H2.gen('J')]))

# 7. plumbing -----------------------------------------------------------------
S = A.subalgebra([E, Fc], max_closure_steps=2)
x, y, z = [S(v) for v in (E.no(Fc), Hc.derivative(), E)]
report('axioms on subalgebra elements',
       skew_ok(S, x, y) and commutator_ok(S, x, y, z)
       and associativity_ok(S, z, y, x))
TS, TA = S.tensor(H), A.tensor(H)
u, v = TS.pure_tensor(S(E), J), TS.pure_tensor(S(Fc), J.no(J))
uA, vA = TA.pure_tensor(E, J), TA.pure_tensor(Fc, J.no(J))
report('S (x) H products == A (x) H products',
       [d.monomial_coefficients() for d in u.ope(v)]
       == [d.monomial_coefficients() for d in uA.ope(vA)]
       and u.no(v).monomial_coefficients() == uA.no(vA).monomial_coefficients())
S3 = S.subalgebra([S(Hc)], names=['H'])
report('nested subalgebra <H> of <E, F>: Heisenberg dims, coerces',
       S3.ambient() is S and [S3.weight_space(n).dimension() for n in range(6)]
       == [p(n) for n in range(6)] and (S3(Hc) + S(E)).parent() is S)
report('nested: E not in <H>', E not in S3 and S(E) not in S3)

set_nproc(4)
Hp = UniversalVertexAlgebra(QQ, ['Jp'], {('Jp', 'Jp'): [0, 1]},
                            conformal_vector={(('Jp', 0), ('Jp', 0)): 1/2, (('Jp', 1),): 1})
Jp = Hp.gen('Jp')
Sp = Hp.subalgebra([Hp.conformal_vector()])
report('4 processes / threads: same dims',
       [Sp.weight_space(n).dimension() for n in range(10)]
       == [1] + [p(n) - p(n - 1) for n in range(1, 10)])
set_nproc(1)

# 8. non-positive weights with grading='ambient' (beta-gamma) ---------------
for lam in [1, 2]:
    # weights (lam, 1 - lam) for (b, g)
    X = NormallyOrderedExpressions(QQ, ['b', 'g'])
    b, g = X.gens()
    Bl0 = UniversalVertexAlgebra(QQ, X, {('b', 'g'): [-1]},
                                 conformal_vector=-lam * b * g.derivative()
                                 + (1 - lam) * b.derivative() * g)
    b, g = Bl0.gens()
    hg = 1 - lam
    report('beta-gamma lam=%d: <g> without grading refused' % lam,
           raises(NotImplementedError, Bl0.subalgebra, [g]))
    Bl = Bl0.with_horizontal_grading({'b': 1, 'g': -1})
    b, g = Bl.gens()
    G = Bl.subalgebra([g], horizontal_grading='ambient')
    report('beta-gamma lam=%d: express in the graded closure' % lam,
           repr(G.express(g.no(g) + 3 * g.derivative())) == ':S0 S0: + 3*dS0')
    # k letters of weight hg + n_j: partitions of W - k*hg into <= k parts
    ok = all(G.weight_space(W, horizontal_grade=-k).dimension()
             == (Partitions(W - k * hg, max_length=k).cardinality()
                 if W - k * hg >= 0 else 0)
             for k in range(5) for W in range(-4, 8))
    report('beta-gamma lam=%d: <g> graded dims = partitions' % lam, ok)
    report('beta-gamma lam=%d: <g> membership' % lam,
           g.no(g.derivative(2)) in G and b not in G
           and b.no(g) not in G and G(g).horizontal_grade() == -1)
    BG = Bl.subalgebra([b, g], horizontal_grading='ambient')
    ok = all(BG.weight_space(W, horizontal_grade=k).dimension()
             == Bl.weight_space(W, horizontal_grade=k).dimension()
             for W in range(-3, 6) for k in range(-3, 4))
    report('beta-gamma lam=%d: <b, g> = V (graded dims)' % lam, ok)
    report('beta-gamma lam=%d: ungraded weight space refused' % lam,
           raises(ValueError, G.weight_space, 2))
    report('beta-gamma lam=%d: the same grading gives the same algebra and subalgebra'
           % lam, Bl0.with_horizontal_grading({'b': 1, 'g': -1}) is Bl
           and Bl.subalgebra([g], horizontal_grading='ambient') is G and g in G)
    Bz = Bl0.with_horizontal_grading({'b': 1, 'g': 1}, group=Zmod(2))
    report('beta-gamma lam=%d: torsion grading gives no twisted weight' % lam,
           raises(NotImplementedError, Bz.subalgebra, [Bz.gen('g')],
                  horizontal_grading='ambient'))

# with_grading('ambient') gives one graded subalgebra (and its graded spans) --
LP = LatticeVertexAlgebra(QQ, matrix([[2]]))
JP = LP.J(0)
LP = LP.with_conformal_vector(JP.no(JP) / 4).with_horizontal_grading('lattice')
P = LP.subalgebra([LP.e((1,))], names=['e'])
Pg = P.with_horizontal_grading('ambient')
dP = Pg.weight_space(6, horizontal_grade=2).dimension()
spP = Pg._spans[(6, (2,))]
same1 = P.with_horizontal_grading('ambient') is Pg and Pg._spans.get((6, (2,))) is spP
same2 = Pg.with_horizontal_grading(None) is P and P.with_horizontal_grading('ambient') is Pg
report('with_grading twice / after None: the same graded subalgebra and spans',
       same1 and same2 and Pg.weight_space(6, horizontal_grade=2).dimension() == dP == 2
       and Pg is LP.subalgebra([LP.e((1,))], names=['e'], horizontal_grading='ambient'))

# 9. batched normally ordered monomials: the words of a span and their
# missing suffixes, one engine call per length (threads over shared caches)
# = the plain recursive definition (∂^n g)_(-1) b, with 1 and 4 threads
# (results and dict order); roots without an engine (Frac(Q(a)[b]), or
# all with VERTEX_ALGEBRAS_NO_ENGINE=1) refuse, so the serial Python path
# computes their spans


def plain(Sx, w, memo):
    if w not in memo:
        i, n = w[0]
        memo[w] = Sx._deriv(i, n).no(plain(Sx, w[1:], memo) if len(w) > 1 else Sx._root.one())
    return memo[w]


def batched_ok(Sx, W, G=None):
    V = Sx._root
    words = list(Sx._words(W, G))
    if V._engine() is None:              # VERTEX_ALGEBRAS_NO_ENGINE=1: refused
        return not Sx._vectors_batched(words)
    saved, nc0 = (Sx._vecs, Sx._evecs), VertexAlgebraParallelism().get()
    got = []
    try:
        for nc in (1, 4):
            Sx._vecs, Sx._evecs = {(): V.one()}, {}
            set_nproc(nc)
            if not Sx._vectors_batched(words):
                return False
            got.append([list(Sx._vec(w)._monomial_coefficients.items()) for w in words])
    finally:
        Sx._vecs, Sx._evecs = saved
        set_nproc(nc0)
    memo = {}
    return (len(words) > 5 and got[0] == got[1]
            and all(V._from_dict(dict(x)) == plain(Sx, w, memo) for w, x in zip(words, got[0])))


X = NormallyOrderedExpressions(K, ['J'])
Jb = X.gen(0)
Hb = UniversalVertexAlgebra(K, X, {('J', 'J'): [0, 1]},
                            conformal_vector=Jb * Jb / 2 + a * Jb.derivative())
Jb = Hb.gen('J')
report('batched monomials: universal over Q(a) (lifted), weight 9',
       batched_ok(Hb.subalgebra([Hb.conformal_vector()]), 9))
report('batched monomials: universal over QQ, <E, F> in affine sl2, weight 4',
       batched_ok(A.subalgebra([E, Fc], names=['E', 'F'], max_closure_steps=2), 4))
report('batched monomials: lattice <e^a, e^-a>, weight 5', batched_ok(S2, 5))
TT = L.tensor(Hp, conformal_vector='factors')
ST = TT.subalgebra([TT.embed(0, L.e((1,))), TT.embed(1, Hp.conformal_vector())])
report('batched monomials: tensor product (lattice x universal), weight 6',
       batched_ok(ST, 6))
SB = B.subalgebra([Ts, W], names=['T', 'W'], max_closure_steps=1)
report('batched monomials: beta-gamma <Ts, W> over Q(sqrt 6) (number field), weight 8',
       batched_ok(SB, 8))
Krel.<i2> = Frac(Frac(QQ['a2'])['i2'])       # lifts to Q(a2)[i2, ...]: no engine
Hr = UniversalVertexAlgebra(Krel, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
Jr = Hr.gen('J')
Sr = Hr.subalgebra([Jr.no(Jr) / 2 + i2 * Jr.derivative()])
report('no engine (Frac(Q(a)[b])): batched path refused, Python path used',
       Hr._engine() is None and not Sr._vectors_batched(list(Sr._words(6, None)))
       and [Sr.weight_space(n).dimension() for n in range(7)] == [1, 0, 1, 1, 2, 2, 4])

# 10. engine-form spans (the vectors stay in the engine's own form, C++
# elimination over its field) = the Sage path: dims, labels, membership,
# express; Q(sqrt 5), Q(c) with a lifted denominator (u = 1/(c + 22/5)),
# QQbar, a tower
from vertex_algebras import _keyed_spans as SLAs
from vertex_algebras import vertex_subalgebra as VSm


def w3over(R, cv, implementation=None):
    X = NormallyOrderedExpressions(R, ['L', 'W'])
    L, W = X.gens()
    Lam = L * L - 3 / 10 * L.derivative(2)
    b = 16 / (22 + 5 * R(cv))
    return UniversalVertexAlgebra(
        R, X, {('L', 'L'): [L.derivative(), 2 * L, 0, R(cv) / 2],
               ('L', 'W'): [W.derivative(), 3 * W],
               ('W', 'W'): [L.derivative(3) / 15 + b * Lam.derivative(),
                            3 / 10 * L.derivative(2) + 2 * b * Lam, L.derivative(),
                            2 * L, 0, R(cv) / 3]},
        degrees={'L': 2, 'W': 3}, implementation=implementation)


K5s.<s5> = QuadraticField(5)
Fcs.<cs> = Frac(QQ['cs'])
K5i.<i5> = K5s.extension(QQ['y']([1, 0, 1]))
for R, cv, name in [(K5s, 134 + 60 * s5, 'Q(sqrt 5)'), (Fcs, cs, 'Q(c)'),
                    (QQbar, 134 + 60 * QQbar(5).sqrt(), 'QQbar'), (K5i, 3 + i5 * s5, 'a tower')]:
    out = []
    for engine_spans in (True, False):
        old = VSm.NormalOrderedSpans._engine_span_kernel
        if not engine_spans:
            VSm.NormalOrderedSpans._engine_span_kernel = lambda self: None
        try:
            # another algebra for the Sage path (the Cython core: the
            # subalgebra of the first one keeps its engine-form spans)
            V = w3over(R, cv, None if engine_spans else 'python')
            if engine_spans:
                expect_engine = SLAs._frac_engine() is not None and V._engine() is not None
            L, W = V.gens()
            S = V.subalgebra([L.no(L) + W.derivative(), W.no(W)])
            dims = [S.weight_space(w).dimension() for w in range(12)]
            P = (L.no(L) + W.derivative()).ope(W.no(W))
            ex = [S.express(P[k]).monomial_coefficients() for k in (2, 4)]
            used = any(isinstance(sp.echelon.inner, SLAs._EngineSpan) for sp in S._spans.values())
            out.append((dims, [sp.echelon.labels() for _, sp in sorted(S._spans.items())], ex,
                        L.no(W) in S, used))
        finally:
            VSm.NormalOrderedSpans._engine_span_kernel = old
    report('engine-form spans = Sage spans over %s (dims, labels, express, membership)' % name,
           out[0][:4] == out[1][:4] and out[0][4] == expect_engine and not out[1][4])

# a generator with a denominator outside the engine's lift (1/c over QQ[c]):
# the spans move to the Sage path, same results
out = []
for engine_spans in (True, False):
    old = VSm.NormalOrderedSpans._engine_span_kernel
    if not engine_spans:
        VSm.NormalOrderedSpans._engine_span_kernel = lambda self: None
    try:
        Hc = UniversalVertexAlgebra(Fcs, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1},
                                    implementation=None if engine_spans else 'python')
        Jc = Hc.gen('J')
        Tc = Jc.no(Jc) / (2 * cs) + Jc.derivative()
        Sc = Hc.subalgebra([Tc])
        out.append(([Sc.weight_space(w).dimension() for w in range(10)],
                    Sc.express(Tc.no(Tc).derivative(2)).monomial_coefficients(), Jc.no(Jc) in Sc))
    finally:
        VSm.NormalOrderedSpans._engine_span_kernel = old
report('engine-form spans: a vector the engine cannot lift moves the span to the Sage path',
       out[0] == out[1] and out[0][0] == [1, 0, 1, 1, 2, 2, 4, 4, 7, 8])

print('total %.1fs' % (time.time() - t_start))
report.result()
