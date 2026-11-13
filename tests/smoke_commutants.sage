# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Commutants (cosets) Com(A, V): dimensions against independent characters
# (free bosons, lattice V_Z inside V_{Z^2}, the GKO coset = Ising, the
# universal sl_2 parafermions and the βγ Heisenberg coset via
# [charge-0 character] * (q)_inf, also with a grading for infinite weight
# spaces), strong generators up to a weight, express, coset Virasoro vectors,
# commutants in subalgebras / quotients / tensor products / commutants,
# S.commutant(), double commutants, subalgebras of commutants, axioms,
# serial == parallel, refusals and stale products.
# Run from the repository root:  sage tests/smoke_commutants.sage  -> RESULT: PASS
import os, random, sys, time
sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import vertex_algebra
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from axioms import (Reporter, skew_ok, commutator_ok, associativity_ok,
                    raw_caches_ok)

report = Reporter()
t_start = time.time()
random.seed(1r)


def raises(exc, f, *args, **kw):
    try:
        f(*args, **kw)
    except exc:
        return True
    return False


def letters_char(letters, N):
    """Character of a free (super)commutative algebra on letters
    (charge, weight in units, odd): {(charge, weight units): count} up to N
    weight units."""
    dp = {(0, 0): 1}
    for c, w, odd in letters:
        new = dict(dp)
        for (ch, wt), m in dp.items():
            k = 1
            while wt + k * w <= N and (k == 1 or not odd):
                key = (ch + k * c, wt + k * w)
                new[key] = new.get(key, 0) + m
                k += 1
        dp = new
    return dp


def times_euler(s):
    """The series s (list) times prod_{n>=1} (1 - q^n)."""
    N = len(s)
    e = [0] * N
    e[0] = 1
    for n in range(1, N):
        e = [e[i] - (e[i - n] if i >= n else 0) for i in range(N)]
    return [sum(s[j] * e[i - j] for j in range(i + 1)) for i in range(N)]


def dims(C, ws, horizontal_grade=None):
    return [C.weight_space(w, horizontal_grade=horizontal_grade).dimension() for w in ws]


# 1. two free bosons -----------------------------------------------------------
t = time.time()
H = UniversalVertexAlgebra(QQ, ['J', 'K'], {('J', 'J'): [0, 1], ('K', 'K'): [0, 1]},
                           degrees={'J': 1, 'K': 1})
J, K = H.gens()
C = H.commutant([J])
report('bosons: dims = p(n)', dims(C, range(11)) ==
       [Partitions(n).cardinality() for n in range(11)])
report('bosons: membership', K in C and K.no(K.derivative()) in C and J not in C
       and J.no(K) not in C and C.contains(H.one()) and not C.contains(J.derivative()))
report('bosons: C(J) refused', raises(ValueError, C, J))
report('bosons: coset alias, S.commutant()', H.coset([J]) is C
       and H.subalgebra([J]).commutant() is C and C.ambient() is H)
gens = C.strong_generators_up_to(7)
report('bosons: strong generators up to 7 = [K]', gens == [C(K)]
       and list(C.variable_names()) == ['C0'])
report('bosons: express', repr(C.express(K.no(K) + 2 * K.derivative())) ==
       ':C0 C0: + 2*dC0')
report('bosons: express beyond the searched weight refused',
       raises(ValueError, C.express, K.no(K.no(K.no(K.no(K.no(K.no(K.no(K)))))))))
D = C.commutant(max_weight=1)
report('bosons: double commutant = <J>', dims(D, range(9)) ==
       [Partitions(n).cardinality() for n in range(9)] and J in D and K not in D)
S2 = C.subalgebra([K])
report('bosons: subalgebra of the commutant', S2.ambient() is C
       and dims(S2, range(7)) == dims(C, range(7))
       and (S2(K) + C(K)).parent() is C and (S2(K) + J).parent() is H)
els = [C(K), C(K.no(K)), C(K.derivative()) + C(K.no(K))]
report('bosons: axioms on commutant elements',
       all(skew_ok(C, a, b) for a in els for b in els)
       and all(commutator_ok(C, a, b, c, rng=range(-1, 2))
               for a in els[:2] for b in els[:2] for c in els[:2])
       and all(associativity_ok(C, a, b, c) for a in els[:2]
               for b in els[:2] for c in els[:1])
       and (els[0].nproduct(els[1], -2)).parent() is C)
report('bosons: raw caches', all(raw_caches_ok(o) for o in C._cache_owners()))
print('  bosons %.1fs' % (time.time() - t))

# 2. tensor product ambient ---------------------------------------------------
t = time.time()
H1 = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
J1 = H1.gen('J')
TP = H1.tensor(H1, degrees='factors')
CT = TP.commutant([TP.embed(0, J1)])
report('tensor: Com(H x 1, H x H) dims = p(n), contains 1 x H',
       dims(CT, range(9)) == [Partitions(n).cardinality() for n in range(9)]
       and TP.embed(1, J1) in CT and TP.embed(0, J1) not in CT)
print('  tensor %.1fs' % (time.time() - t))

# 3. lattice: Com(<J0>, V_{Z^2}) = V_Z ---------------------------------------------
t = time.time()
L = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]]))
J0, J1L = L.heisenberg_currents()
L = L.with_conformal_vector(J0.no(J0) / 2 + J1L.no(J1L) / 2)
J0, J1L = L.heisenberg_currents()
CL = L.commutant([J0])
N = 12
fock = [Partitions(n // 2).cardinality() if n % 2 == 0 else 0
        for n in range(N + 1)]                                # in units 1/2
theta = [0] * (N + 1)
for m in range(-5, 6):
    if m * m <= N:
        theta[m * m] += 1
VZ = [sum(theta[j] * fock[i - j] for j in range(i + 1)) for i in range(N + 1)]
report('lattice: dims = theta_Z / eta (weights 0..6)',
       dims(CL, [n / 2 for n in range(N + 1)]) == VZ)
g = CL.strong_generators_up_to(5)
report('lattice: strong generators e^{+-x_2} (weight 1/2)',
       sorted(x.degree() for x in g) == [1/2, 1/2]
       and set(g) == {CL(L.e((0, 1))), CL(L.e((0, -1)))})
print('  lattice %.1fs' % (time.time() - t))

# 4. GKO: Com(L_2(sl2), L_1(sl2) x L_1(sl2)) = Ising ----------------------------
t = time.time()
LA = LatticeVertexAlgebra(QQ, matrix([[2, 0], [0, 2]]))
a0, a1 = LA.heisenberg_currents()
LA = LA.with_conformal_vector(a0.no(a0) / 4 + a1.no(a1) / 4)
a0, a1 = LA.heisenberg_currents()
e = LA.e((1, 0)) + LA.e((0, 1))
f = LA.e((-1, 0)) + LA.e((0, -1))
h = e.nproduct(f, 0)
lev = e.nproduct(f, 1).coefficient(LA.one().support()[0])
report('GKO: diagonal sl2 at level 2', h == a0 + a1 and lev == 2)
CG = LA.commutant([e, f])
NI = 9
ferm = letters_char([(0, 2 * n - 1, True) for n in range(1, NI + 1)], 2 * NI)
ising = [ferm.get((0, 2 * n), 0) for n in range(NI)]
report('GKO: dims = Ising vacuum character (even part of a Majorana fermion)',
       dims(CG, range(NI)) == ising == [1, 0, 1, 1, 2, 2, 3, 3, 5])
A = LA.subalgebra([e, f],
                  conformal_vector=(h.no(h) / 2 + e.no(f) + f.no(e)) / (2 * (lev + 2)))
CG2 = LA.commutant(A)
report('GKO: commutant of the subalgebra = commutant of its generators',
       CG2 is CG and CG.acting_subalgebra() is A)
CGc = CG.with_conformal_vector('coset')
report('GKO: coset Virasoro vector, c = 1/2', CGc.central_charge() == 1/2
       and CGc.conformal_vector() == CGc(LA.conformal_vector() - A.lift(A.conformal_vector()))
       and CG.conformal_vector() is None and CGc is LA.commutant(A, conformal_vector='coset'))
gs = CGc.strong_generators_up_to(6)
Tc = CGc.conformal_vector()
k0 = Tc.leading_support()
report('GKO: strong generators up to 6 = [T_coset] (up to scaling)',
       len(gs) == 1 and gs[0] * Tc.coefficient(k0) == Tc * gs[0].coefficient(k0))
SG = LA.subalgebra(gs)
report('GKO: <T_coset> has the dims of the commutant up to 8',
       dims(SG, range(NI)) == ising)
report('GKO: conformal vector checks', raises(ValueError, CG.with_conformal_vector, 2 * Tc)
       and raises(ValueError, CG.with_conformal_vector, LA.conformal_vector()))
report('GKO: the commutant with T shares the spaces and generators',
       CGc._core is CG._core and CG.variable_names() == CGc.variable_names())
print('  GKO %.1fs' % (time.time() - t))

# 5. parafermions: Com(<h>, V^k(sl2)) ---------------------------------------------
t = time.time()
kk = 7/3
NP = 8


def affine_sl2(k, names=('E', 'H', 'F')):
    X = NormallyOrderedExpressions(QQ, list(names))
    E, Hc, F = X.gens()
    nE, nH, nF = names
    return UniversalVertexAlgebra(
        QQ, X, {(nH, nH): [0, 2 * k], (nH, nE): [2 * E], (nH, nF): [-2 * F],
                (nE, nF): [Hc, k]},
        degrees={nE: 1, nH: 1, nF: 1}, horizontal_grading={nE: 1, nH: 0, nF: -1})


A2 = affine_sl2(kk)
E2, H2, F2 = A2.gens()
P = A2.commutant([H2], horizontal_grading='ambient')
ch = letters_char([(c, n, False) for n in range(1, NP) for c in (1, 0, -1)], NP - 1)
para = times_euler([ch.get((0, n), 0) for n in range(NP)])
report('parafermions: dims = [charge-0 character] * (q)_inf',
       dims(P, range(NP), horizontal_grade=0) == para)
report('parafermions: other charges are zero',
       dims(P, range(5), horizontal_grade=1) == [0] * 5)
pg = P.strong_generators_up_to(6, horizontal_grades=[0])
report('parafermions: strong generators of weights 2, 3, 4, 5 (none at 6)',
       [x.degree() for x in pg] == [2, 3, 4, 5])
print('  parafermions %.1fs' % (time.time() - t))

# 6. βγ: Com(<:βγ:>) = W_3 at c = -2 ---------------------------------------------
t = time.time()
NB = 8
bg_char = letters_char([(c, 2 * n + 1, False) for n in range(NB) for c in (1, -1)],
                       2 * (NB - 1))
w3 = times_euler([bg_char.get((0, 2 * n), 0) for n in range(NB)])
X = NormallyOrderedExpressions(QQ, ['b', 'g'])
b, gg = X.gens()
B = UniversalVertexAlgebra(QQ, X, {('b', 'g'): [-1]},
                           conformal_vector=(b.derivative() * gg - b * gg.derivative()) / 2)
b, gg = B.gens()
Jbg = b.no(gg)
CB = B.commutant([Jbg])
report('βγ (weights 1/2): dims = [charge-0 character] * (q)_inf',
       dims(CB, range(NB)) == w3 and w3[:7] == [1, 0, 1, 2, 3, 4, 6])
report('βγ: no half-integer weights', dims(CB, [n + 1/2 for n in range(4)]) == [0] * 4)
AB = B.subalgebra([Jbg], conformal_vector=-Jbg.no(Jbg) / 2)
CBc = B.commutant(AB, conformal_vector='coset')
report('βγ: coset Virasoro vector, c = -2', CBc.central_charge() == -2)
report('βγ: strong generators of weights 2, 3 up to 6',
       [x.degree() for x in CBc.strong_generators_up_to(6)] == [2, 3])
b1, g1 = X.gens()
B1 = UniversalVertexAlgebra(QQ, X, {('b', 'g'): [-1]}, conformal_vector=-b1 * g1.derivative(),
                            horizontal_grading={'b': 1, 'g': -1})
b1, g1 = B1.gens()
CB1 = B1.commutant([b1.no(g1)], horizontal_grading='ambient')
report('βγ (weights 1, 0, graded): dims of charge 0 agree',
       dims(CB1, range(NB), horizontal_grade=0) == w3
       and dims(CB1, range(4), horizontal_grade=1) == [0] * 4)
report('βγ (weights 1, 0): ungraded weight space refused',
       raises((ValueError, NotImplementedError), CB1.weight_space, 2))
report('βγ (weights 1, 0): strong generators with grades=[0]',
       [x.degree() for x in
        CB1.strong_generators_up_to(5, horizontal_grades=[0])] == [2, 3])
report('βγ: ungraded search refused without finite spaces',
       raises(ValueError, CB1.strong_generators_up_to, 3))
print('  βγ %.1fs' % (time.time() - t))

# 7. subalgebra and quotient ambients ----------------------------------------------
t = time.time()
H3 = UniversalVertexAlgebra(QQ, ['J', 'K', 'M'],
                            {('J', 'J'): [0, 1], ('K', 'K'): [0, 1], ('M', 'M'): [0, 1]},
                            degrees={'J': 1, 'K': 1, 'M': 1})
x3, y3, z3 = H3.gens()
S3 = H3.subalgebra([x3, y3])
CS = S3.commutant([x3])
report('subalgebra ambient: Com(<J>, <J, K>) = <K>',
       dims(CS, range(8)) == [Partitions(n).cardinality() for n in range(8)]
       and y3 in CS and z3 not in CS and x3 not in CS and CS.ambient() is S3
       and (CS(y3) + S3(x3)).parent() is S3)
X = NormallyOrderedExpressions(QQ, ['E', 'H', 'F'])
Ea, Ha, Fa = X.gens()
A1 = UniversalVertexAlgebra(QQ, X, {('H', 'H'): [0, 2], ('H', 'E'): [2 * Ea],
                                    ('H', 'F'): [-2 * Fa], ('E', 'F'): [Ha, 1]},
                            conformal_vector=(Ha * Ha / 2 + Ea * Fa + Fa * Ea) / 6)
Ea, Ha, Fa = A1.gens()
Q = A1 / [Ea.no(Ea)]
CQ = Q.commutant([Q(Ha)])
report('quotient ambient: Com(<h>, L_1(sl2)) = C (parafermions at level 1)',
       dims(CQ, range(6)) == [1, 0, 0, 0, 0, 0]
       and dims(Q.commutant([Q(Ea), Q(Fa)]), range(4)) == [1, 0, 0, 0])
CU = A1.commutant([Ha])
report('universal level 1 parafermions are not trivial',
       dims(CU, range(4)) == [1, 0, 1, 2])
print('  subalgebra/quotient %.1fs' % (time.time() - t))

# 8. serial == parallel -------------------------------------------------------------
t = time.time()
rows1 = [dict(x._monomial_coefficients) for x in P.weight_space(6, horizontal_grade=0).basis()]
set_nproc(4)
Ap = affine_sl2(kk, names=('Ep', 'Hp', 'Fp'))     # another algebra: a new commutant
Pp = Ap.commutant([Ap.gen('Hp')], horizontal_grading='ambient')
rowsp = [dict(x._monomial_coefficients) for x in Pp.weight_space(6, horizontal_grade=0).basis()]
set_nproc(1)
report('serial == parallel (weight 6, canonical rows)', rows1 == rowsp and rows1)
print('  parallel %.1fs' % (time.time() - t))

# 9. refusals and stale products ---------------------------------------------------
report('refused: inhomogeneous generator',
       raises(ValueError, H.commutant, [J + J.no(J)]))
report('refused: commutant() of a top-level algebra', raises(TypeError, H.commutant))
report('refused: commutant of a commutant without max_weight',
       raises(ValueError, C.commutant) and raises(ValueError, H.commutant, C))
report('refused: max_weight for a list of generators',
       raises(ValueError, H.commutant, [J], max_weight=2))
report('refused: generators outside the algebra', raises(ValueError, H.commutant, [x3]))
report('refused: ideals of a commutant', raises(NotImplementedError, C.ideal, [K]))
report('refused: conformal weights of a commutant (those of the ambient)',
       raises(ValueError, C.with_degrees, {'C0': 1}))
report('refused: T not in the commutant',
       raises(ValueError, CB.with_conformal_vector, B.conformal_vector()))
report('refused: grades without a grading', raises(ValueError, C.strong_generators_up_to,
                                                   3, horizontal_grades=[0]))
# other OPEs give another algebra: the commutant of the old one keeps working
Hs = UniversalVertexAlgebra(QQ, ['J', 'K'], {('J', 'J'): [0, 1], ('K', 'K'): [0, 1]},
                            degrees={'J': 1, 'K': 1})
Js, Ks = Hs.gens()
Cs = Hs.commutant([Js])
Cs.weight_space(2)
Hs2 = Hs.with_opes({('J', 'J'): [0, 2], ('K', 'K'): [0, 1]})
report('immutable: new OPEs = new algebra, the old commutant keeps working',
       Hs2 is not Hs and Cs.weight_space(3).dimension() == 3 and Ks in Cs
       and Hs2.commutant([Hs2.gen('J')]) is not Cs
       and raises((TypeError, ValueError), Cs, Hs2.gen('K')))

print('total %.1fs' % (time.time() - t_start))
report.result()
