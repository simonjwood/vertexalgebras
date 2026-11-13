# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Quotients V/I: characters (Lee-Yang, Ising, sl2 level 1, W3 at c = -2,
# minimal models inside the Heisenberg algebra, C[gamma_n]/<gamma^2>),
# well-definedness of the products on classes, vertex algebra axioms, the
# isomorphism sl2 level 1 / <:EE:> = V_A1 (as vertex algebras, up to weight 4),
# nested quotients, subalgebras and tensor products of quotients, serial ==
# parallel, coercion, refusals (zero quotients, parity, stale products).
# Run from the repository root:  sage tests/smoke_quotients.sage  -> RESULT: PASS
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
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.weight_spaces import SubspaceWeightSpace
from axioms import (Reporter, nmax, skew_ok, commutator_ok, associativity_ok,
                    raw_caches_ok)

report = Reporter()
t_start = time.time()
random.seed(1r)
PS = PowerSeriesRing(QQ, 'q', default_prec=20)
q = PS.gen()


def coeffs(f, n):
    return [f[k] for k in range(n)]


def raises(exc, f, *args, **kw):
    try:
        f(*args, **kw)
    except exc:
        return True
    return False


def dims(Q, n, **kw):
    return [Q.weight_space(k, **kw).dimension() for k in range(n)]


def virasoro(R, c, name='L'):
    L = NormallyOrderedExpressions(R, [name]).gen(0)
    Vir = UniversalVertexAlgebra(R, L.parent(), {(name, name): [L.derivative(), 2 * L, 0, c / 2]},
                                 degrees={name: 2})
    return Vir, Vir.gen(name)


def singular_vectors(V, T, W):
    """Basis of the vectors of weight W killed by T_(2), T_(3) (L_1, L_2)."""
    B = V.weight_space(W).basis()
    rows = []
    for k in (2, 3):
        ims = [T.nproduct(b, k) for b in B]
        keys = sorted({m for x in ims for m in x.support()})
        rows += [[x.coefficient(m) for x in ims] for m in keys]
    M = matrix(V.base_ring(), rows)
    return [sum(c * b for c, b in zip(v, B)) for v in M.right_kernel().basis()]


def well_defined(V, Q, xs, ns=range(-2, 3)):
    """Q(u)_(n) Q(v) == Q(u_(n) v) for u, v in xs."""
    return all(Q(u).nproduct(Q(v), n) == Q(u.nproduct(v, n))
               for u in xs for v in xs for n in ns)


def class_ok(V, I, Q, Ws):
    """Q(u + i) == Q(u), Q(i) == 0 for i in I, and Q(u) == 0 iff u in I, on
    random combinations of weight space bases."""
    for W in Ws:
        B = V.weight_space(W).basis()
        IB = I.weight_space(W).basis()
        for _ in range(4):
            u = sum(random.randint(-3, 3) * b for b in B)
            i = sum(random.randint(-3, 3) * b for b in IB)
            if Q(u + i) != Q(u) or not Q(i).is_zero():
                return False
            if Q(u).is_zero() != (u in I):
                return False
    return True


def lift_ok(Q, I, Ws):
    """lift(x) is supported on standard monomials and projects back."""
    for W in Ws:
        lead = set(I.weight_space(W).leading_monomials())
        for x in Q.weight_space(W):
            y = Q.lift(x)
            if Q(y) != x or lead & set(y.support()):
                return False
    return True


def space_ok(Q, Ws):
    """T_(1) x == W x and T_(0) x == ∂x on the weight space bases."""
    T = Q.conformal_vector()
    return all(T.nproduct(x, 1) == W * x and T.nproduct(x, 0) == x.derivative()
               for W in Ws for x in Q.weight_space(W))


def word(key, gens, one):
    """The right-nested normally ordered monomial of a universal key
    (letters (generator index, derivative order)) in the elements gens."""
    if not key:
        return one
    (i, n), rest = key[0], key[1:]
    return gens[i].derivative(n).no(word(rest, gens, one))


rr = prod(1 / (1 - q**n) for n in range(1, 20) if n % 5 in (2, 3))

# 1. Lee-Yang -----------------------------------------------------------------
Vir, L = virasoro(QQ, -22/5)
Lam = L.no(L) - 3/10 * L.derivative(2)
I = Vir.ideal([Lam])
Q = Vir.quotient(I)
Lq = Q.gen('L')
report('Lee-Yang: Vir / <Lambda> = Rogers-Ramanujan (weights < 13)',
       dims(Q, 13) == coeffs(rr, 13), dims(Q, 13))
report('Lee-Yang: same quotient from generators, V / I, V.quo',
       Vir.quotient([Lam]) is Q and Vir / I is Q and Vir.quo([Lam]) is Q
       and Q.defining_ideal() is I)
xs = [b for W in range(0, 5) for b in Vir.weight_space(W).basis()]
report('Lee-Yang: Q(u)_(n) Q(v) == Q(u_(n) v) (weights <= 4, %d elements)'
       % len(xs), well_defined(Vir, Q, xs))
report('Lee-Yang: Q(u + i) == Q(u), Q(u) == 0 iff u in I (weights 4-8)',
       class_ok(Vir, I, Q, range(4, 9)))
report('Lee-Yang: lift = standard monomials, projects back',
       lift_ok(Q, I, range(0, 9)))
els = [Lq, Lq.derivative(), Lq.no(Lq.derivative()), Q(L.no(L.no(L)))]
ok = all(skew_ok(Q, a, b) for a in els for b in els[:3])
ok = ok and all(commutator_ok(Q, a, b, c) for a in els[:2] for b in els[:2]
                for c in els[:3])
ok = ok and all(associativity_ok(Q, a, b, c) for a in els[:2]
                for b in els[:2] for c in els[:2])
report('Lee-Yang: vertex algebra axioms in the quotient', ok)
report('Lee-Yang: raw caches', all(raw_caches_ok(o) for o in Q._cache_owners()))
report('Lee-Yang: no Virasoro vector to take from the ambient algebra',
       raises(ValueError, Q.with_conformal_vector, 'ambient'))
Qt = Q.with_conformal_vector(Lq)
report('Lee-Yang: T = L, c = -22/5, T_(1) = W on the weight spaces',
       Qt.central_charge() == -22/5 and space_ok(Qt, range(0, 9))
       and Q.conformal_vector() is None and Qt._I is Q._I
       and Qt is Vir.quotient([Lam], conformal_vector=L))

# 2. Ising --------------------------------------------------------------------
VirI, LI = virasoro(QQ, 1/2, name='T')
QI = VirI.quotient(singular_vectors(VirI, LI, 6))
psi = NormallyOrderedExpressions(QQ, ['psi'], parities={'psi': 1}).gen(0)
F = UniversalVertexAlgebra(QQ, psi.parent(), {('psi', 'psi'): [1]},
                           conformal_vector=psi.derivative() * psi / 2)
psi = F.gen('psi')
Smaj = F.subalgebra([F.conformal_vector()])
dmaj = [Smaj.weight_space(n).dimension() for n in range(11)]
report('Ising: Vir / <v_6> == <T> in the Majorana fermion (weights < 11)',
       dims(QI, 11) == dmaj, dmaj)

# 3. affine sl2 at level 1 ----------------------------------------------------
X = NormallyOrderedExpressions(QQ, ['E', 'H', 'F'])
E, Hc, Fc = X.gens()
A = UniversalVertexAlgebra(QQ, X, {('H', 'H'): [0, 2], ('H', 'E'): [2*E], ('H', 'F'): [-2*Fc],
                                   ('E', 'F'): [Hc, 1]},
                           conformal_vector=(Hc * Hc / 2 + E * Fc + Fc * E) / 6)
E, Hc, Fc = A.gens()
IA = A.ideal([E.no(E)])
QA = A.quotient(IA)
LA = LatticeVertexAlgebra(QQ, matrix([[2]]))
KA = LA.J(0)
LA = LA.with_conformal_vector(KA.no(KA) / 4)
KA = LA.J(0)
dl = [LA.weight_space(n).dimension() for n in range(7)]
report('sl2 level 1: V / <:EE:> == V_A1 (weights < 7)', dims(QA, 7) == dl, dl)
xs = [b for W in range(0, 3) for b in A.weight_space(W).basis()]
report('sl2 level 1: products well defined on classes (weights <= 2)',
       well_defined(A, QA, xs))
report('sl2 level 1: Q(u + i) == Q(u), Q(u) == 0 iff u in I (weights 2-5)',
       class_ok(A, IA, QA, range(2, 6)))
QAt = QA.with_conformal_vector('ambient')
report('sl2 level 1: Virasoro vector from the ambient algebra, c = 1',
       QAt.central_charge() == 1 and space_ok(QAt, range(0, 5))
       and QAt is A.quotient(IA, conformal_vector='ambient'))

# the isomorphism Q -> V_A1: E -> e^a, H -> J0, F -> s e^-a
ea, fa = LA.e((1,)), LA.e((-1,))
s = 1 / ea.nproduct(fa, 1).coefficient(LA._vacuum_key())
images = [ea, KA, s * fa]
gensQ = QA.gens()
report('word(): universal keys are right-nested NO monomials',
       all(word(k, A.gens(), A.one()) == A.monomial(k)
           for W in range(4) for k in A.weight_space(W).basis_keys()))


def phi(x):
    """The map Q -> V_A1 on normal forms (standard monomials -> words)."""
    return sum((c * word(k, images, LA.one()) for k, c in x
                .monomial_coefficients().items()), LA.zero())


ok = ea.nproduct(s * fa, 0) == KA
for W in range(5):
    keys = QA.weight_space(W).basis_keys()
    ok = ok and all(word(k, gensQ, QA.one()) == QA.monomial(k) for k in keys)
    ims = [word(k, images, LA.one()) for k in keys]
    supp = sorted({m for v in ims for m in v.support()})
    M = matrix(QQ, [[v.coefficient(m) for m in supp] for v in ims])
    ok = ok and M.rank() == len(keys) == LA.weight_space(W).dimension()
report('sl2 level 1: standard monomials map to a basis of V_A1 (weights < 5)',
       ok)
basis = [x for W in range(4) for x in QA.weight_space(W)]
ok = all(phi(g.nproduct(b, n)) == images[i].nproduct(phi(b), n)
         for i, g in enumerate(gensQ) for b in basis for n in range(-2, 3))
report('sl2 level 1: phi(x_(n) b) == phi(x)_(n) phi(b) (%d basis vectors)'
       % len(basis), ok)
report('sl2 level 1: phi(T_sug) == :J0 J0:/4',
       phi(QAt.conformal_vector()) == KA.no(KA) / 4)

LA = LA.with_horizontal_grading('lattice')
Ag = A.with_horizontal_grading({'E': 1, 'H': 0, 'F': -1})
QAg = Ag.quotient([Ag.gen('E').no(Ag.gen('E'))], horizontal_grading='ambient')
ok = all(QAg.weight_space(n, horizontal_grade=k).dimension()
         == LA.weight_space(n, horizontal_grade=k).dimension()
         for n in range(6) for k in range(-2, 3))
report('sl2 level 1, graded by sl2 weight: == V_A1 by lattice grade', ok)
report('sl2 level 1, graded: grades of the standard monomials',
       all(x.horizontal_grade() == QAg.horizontal_grading_group()(k) for n in range(5)
           for k in range(-2, 3) for x in QAg.weight_space(n, horizontal_grade=k)))

# 4. W3 at c = -2 (Wang, q-alg/9708016, Lemma 2.1) ----------------------------
X = NormallyOrderedExpressions(QQ, ['T', 'W'])
T, W = X.gens()
c = -2
beta = 16 / (22 + 5 * c)
Lm = T * T - 3/10 * T.derivative(2)
W3 = UniversalVertexAlgebra(QQ, X, {('T', 'T'): [T.derivative(), 2 * T, 0, c / 2],
                                    ('T', 'W'): [W.derivative(), 3 * W],
                                    ('W', 'W'): [beta * Lm.derivative() + T.derivative(3) / 15,
                                                 2 * beta * Lm + 3/10 * T.derivative(2),
                                                 T.derivative(), 2 * T, 0, c / 3]},
                            degrees={'T': 2, 'W': 3})
T, W = W3.gens()
vs = (3/2 * W.no(W) - 19/36 * T.derivative().no(T.derivative())
      - 8/9 * T.no(T.no(T)) - 14/9 * T.no(T.derivative(2) / 2)
      + 44/9 * T.derivative(4) / 24)
vsp = 9/2 * W.derivative(3) / 6 + 9 * T.derivative().no(W) - 6 * T.no(W.derivative())
QW = W3.quotient([vs, vsp])
report('W3 c=-2: quotient = singlet [1,0,1,2,3,4,6,8,12]',
       dims(QW, 9) == [1, 0, 1, 2, 3, 4, 6, 8, 12], dims(QW, 9))
Tq, Wq = QW.gens()
els = [Tq, Wq, Tq.no(Tq)]
report('W3 c=-2: axioms in the quotient',
       all(skew_ok(QW, a, b) for a in els for b in els)
       and all(commutator_ok(QW, a, b, cc, rng=range(-1, 2))
               for a in els[:2] for b in els[:2] for cc in els[:2]))

# 5. quotients of subalgebras -------------------------------------------------
for name, cc, svw in [('Lee-Yang', -22/5, 4), ('Ising', 1/2, 6)]:
    a2 = (1 - cc) / 12
    K.<r> = QuadraticField(a2.numerator() * a2.denominator())
    a = r / a2.denominator()
    J = NormallyOrderedExpressions(K, ['J']).gen(0)
    H = UniversalVertexAlgebra(K, J.parent(), {('J', 'J'): [0, 1]},
                               conformal_vector=J * J / 2 + a * J.derivative())
    J = H.gen('J')
    S = H.subalgebra([H.conformal_vector()], names=['T'])
    Ts = S.gen('T')
    v = singular_vectors(S, Ts, svw)
    IS = S.ideal(v)
    M = S.quotient(IS)
    target = coeffs(rr, 11) if name == 'Lee-Yang' else dmaj
    report('%s: S / <v> = minimal model (subalgebra path, weights < 11)' % name,
           dims(M, 11) == target[:11] and isinstance(M.weight_space(4),
                                                     SubspaceWeightSpace))
    Tm = M.gen('T')
    ok = all(S.contains(M.lift(x)) for W in range(9) for x in M.weight_space(W))
    xs = [Ts, Ts.derivative(), Ts.no(Ts), Ts.no(Ts.derivative())]
    ok = ok and well_defined(S, M, xs)
    report('%s: normal forms lie in S, products well defined' % name, ok)
    B = M.weight_space(8)
    report('%s: weight space coordinates' % name,
           all(B.coordinates(x) == [int(i == j) for i in range(B.dimension())]
               for j, x in enumerate(B.basis())))
    if name == 'Lee-Yang':
        S.print_options(generators='strong')
        report('Lee-Yang: printed in the generators of S',
               repr(Tm.no(Tm)) == '3/10*d^2T', repr(Tm.no(Tm)))
        S.print_options(generators='ambient')

# beta-gamma, graded: C[gamma_n] / <gamma^2>
X = NormallyOrderedExpressions(QQ, ['b', 'g'])
b, g = X.gens()
B = UniversalVertexAlgebra(QQ, X, {('b', 'g'): [-1]}, conformal_vector=-b * g.derivative(),
                           horizontal_grading={'b': 1, 'g': -1})
b, g = B.gens()
G = B.subalgebra([g], horizontal_grading='ambient')
gs = G(g)
MG = G.quotient([gs.no(gs)], horizontal_grading='ambient')
ok = all(MG.weight_space(W, horizontal_grade=-k).dimension()
         == (Partitions(W - k * (k - 1), max_length=k).cardinality()
             if W >= k * (k - 1) else 0)
         for k in range(5) for W in range(9))
report('beta-gamma: C[gamma_n] / <gamma^2> = difference-2 partitions', ok)
report('beta-gamma: ungraded weight space refused',
       raises(ValueError, MG.weight_space, 2))

# 6. nested quotients, ideals and subalgebras of quotients --------------------
D = UniversalVertexAlgebra(QQ, ['x', 'y'], degrees={'x': 1, 'y': 1})
x, y = D.gens()
Qx = D.quotient([x.no(x)])
xq, yq = Qx.gens()
Qxy = Qx.quotient([yq.no(yq)])
Pxy = D.quotient([x.no(x), y.no(y)])
report('nested: (D/<x^2>)/<y^2> and D/<x^2, y^2> have the same dimensions',
       dims(Qxy, 9) == dims(Pxy, 9), dims(Pxy, 9))
ok = all(Qxy(Qx(D.monomial(k))).monomial_coefficients()
         == Pxy(D.monomial(k)).monomial_coefficients()
         for W in range(7) for k in D.weight_space(W).basis_keys())
report('nested: the same normal forms (all basis monomials, weights < 7)', ok)
report('nested: Q/<1> refused', raises(ValueError, Qx.quotient, [Qx.one()]))
Sx = Qx.subalgebra([xq])
rrd = prod(1 / (1 - q**n) for n in range(1, 20) if n % 5 in (1, 4))
report('subalgebra <x> of D/<x^2> = Rogers-Ramanujan (jets of the double point)',
       [Sx.weight_space(n).dimension() for n in range(10)] == coeffs(rrd, 10)
       and dims(Qx.quotient([yq]), 10) == coeffs(rrd, 10))
Jx = Qx.ideal([xq.no(yq)])
report('ideal of a quotient', xq.no(yq).derivative() in Jx and xq not in Jx
       and Jx.weight_space(3).dimension() > 0)

# 7. tensor products with quotient factors ------------------------------------
Hb = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
Jb = Hb.gen('J')
TP = Q.tensor(Hb)
a0, a1 = TP.embed(0, Lq), TP.embed(1, Jb)
els = [a0, a1, TP.pure_tensor(Lq, Jb), TP.pure_tensor(Lq.no(Lq), Jb.derivative())]
ok = all(skew_ok(TP, u, v) for u in els for v in els)
ok = ok and all(commutator_ok(TP, u, v, w, rng=range(-1, 2))
                for u in els[:3] for v in els[:3] for w in els[:2])
report('tensor Q x H: axioms on mixed elements', ok)
report('tensor Q x H: (L x 1)_(1) (:LL: x 1) == 4 (:LL: x 1)',
       a0.nproduct(TP.embed(0, Lq.no(Lq)), 1) == 4 * TP.embed(0, Lq.no(Lq)))
TPw = TP.with_degrees('factors')
ph = [Partitions(n).cardinality() for n in range(10)]
dq = dims(Q, 10)
conv = [sum(dq[i] * ph[n - i] for i in range(n + 1)) for n in range(10)]
report('tensor Q x H: weight spaces = convolution of the characters',
       dims(TPw, 10) == conv, conv)
report('tensor Q x H: a Virasoro vector not available',
       raises(NotImplementedError, TP.with_conformal_vector,
              a0 + TP.embed(1, Jb.no(Jb) / 2)))
report('tensor Q x H: raw caches',
       all(raw_caches_ok(o) for o in TP._cache_owners()))

# 8. serial == parallel -------------------------------------------------------
# multi-term elements
def mc(xs):
    return [x.monomial_coefficients() for x in xs]


hs = [Jb, Jb.derivative(), Jb.no(Jb)]
us = sum(x for W in range(2, 9) for x in Q.weight_space(W))
TPs = Q.tensor(Hb)
ts = sum(TPs.pure_tensor(x, h) for W in range(2, 7) for x in Q.weight_space(W)
         for h in hs)
serial = (mc(us.ope(us)), mc([us.nproduct(us, -2)]), mc(ts.ope(ts)))
set_nproc(4)
Vir4, L4 = virasoro(QQ, -22/5, name='L4')   # another algebra: a new quotient
Q4 = Vir4.quotient([L4.no(L4) - 3/10 * L4.derivative(2)])
L4q = Q4.gen('L4')
u4 = sum(x for W in range(2, 9) for x in Q4.weight_space(W))
report('parallel: 4 processes, products in the quotient == serial',
       mc(u4.ope(u4)) == serial[0] and mc([u4.nproduct(u4, -2)]) == serial[1])
S4 = Q4.subalgebra([L4q])
report('parallel: subalgebra of a quotient',
       [S4.weight_space(n).dimension() for n in range(12)] == coeffs(rr, 12))
TP4 = Q4.tensor(Hb)
t4 = sum(TP4.pure_tensor(x, h) for W in range(2, 7) for x in Q4.weight_space(W)
         for h in hs)
report('parallel: tensor products with a quotient factor == serial',
       mc(t4.ope(t4)) == serial[2]
       and all(raw_caches_ok(o) for o in TP4._cache_owners())
       and len(Q4._nf_rows) > 0)
set_nproc(1)

# 9. coercion -----------------------------------------------------------------
v = L.no(L.no(L))
report('coercion: V + Q lands in Q, lift / retract',
       (v + Lq).parent() is Q and Q.retract(v) == Q(v)
       and Q.lift(Q(v)).parent() is Vir)
SA = A.subalgebra([Hc])
report('coercion: S + Q lands in Q for S inside V',
       (SA(Hc) + QA.gen('E')).parent() is QA)
report('coercion: no map Q -> V, V / 2 refused',
       raises(TypeError, Vir, Lq) and raises(TypeError, lambda: Vir / 2))
report('zero ideal: V/<0> has the dimensions of V',
       dims(Vir.quotient([0]), 9) == dims(Vir, 9))

# 10. refusals ----------------------------------------------------------------
report('refused: <1>, lattice <e^a>, an ideal of another algebra',
       raises(ValueError, Vir.quotient, [Vir.one()])
       and raises(ValueError, LA.quotient, [LA.e((1,))])
       and raises(ValueError, Vir.quotient, IA))
Fp = UniversalVertexAlgebra(QQ, ['a', 'p'], {('p', 'p'): [1]}, parities={'p': 1},
                            degrees={'a': 1/2, 'p': 1/2})
ap, pp = Fp.gens()
report('refused: generators of mixed parity',
       raises(ValueError, Fp.quotient, [ap + pp]))
report('refused: conformal weights of a quotient (those of the ambient)',
       raises(ValueError, Q.with_degrees, {'L': 2}))
# algebras are fixed by their data: the same OPEs give the same algebra (and
# quotient); other OPEs another algebra, whose elements Q refuses
Vs, Ls = virasoro(QQ, -22/5)
Vc = Vir.with_opes({('L', 'L'): [L.derivative(), 2 * L, 0, -1]})
report('immutable: same data -> same algebra and quotient, other OPEs -> another algebra',
       Vs is Vir and Vs.quotient([Ls.no(Ls) - 3/10 * Ls.derivative(2)]) is Q
       and Vc is not Vir and Vc.degrees() == {'L': 2}
       and Lq.no(Lq) == Q(L.no(L)) and dims(Q, 9) == coeffs(rr, 9)
       and raises((TypeError, ValueError), Q, Vc.gen('L')))

print('total %.1fs' % (time.time() - t_start))
report.result()
