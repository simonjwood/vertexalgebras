# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Zhu algebras (zhu_algebra.py): Zhu's identities on random elements (the
# recursion of the Zhu map uses only some elements of J), an independent
# recomputation of the Zhu map by Zhu's product formula, the relations
# (U(sl2), commutative W_3), odd generators (bc: Clifford, symplectic
# fermions: exterior algebra), half-integer weights (N=1, N=2, betagamma),
# confluence (diamond lemma) and a broken Jacobi identity, implementation
# independence, quotients with known Zhu algebras (Lee-Yang, Ising,
# L_k(sl2) for k = 1, 2, nested), lattice vertex algebras (positive
# definite: zero modes on the top levels; otherwise zero), tensor products
# (super tensor products of the factors' Zhu algebras), and the H-twisted
# (Ramond) Zhu algebras (identities, Ramond relations, tricritical Ising,
# Bershadsky-Polyakov = a finite W-algebra).
# Run from the repository root:  sage tests/smoke_zhu.sage  -> RESULT: PASS
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from axioms import Reporter
from itertools import product as _cartesian
from itertools import combinations_with_replacement

report = Reporter()
t_start = time.time()
random.seed(5r)
K.<c> = Frac(QQ['c'])
Kk.<k> = Frac(QQ['k'])


def weight(V, x):
    return x.degree()


def homogeneous(V, W, n=3):
    """A random element of weight W (of the basis keys)."""
    keys = list(V.weight_space(W).basis_keys())
    R = V.base_ring()
    if not keys:
        return V.zero()
    return sum((R(random.randint(-3, 3)) * V.monomial(random.choice(keys))
                for _ in range(n)), V.zero())


def identities_ok(V, weights, trials=6):
    """Zhu's identities in A = V.zhu_algebra() on random homogeneous a, b
    of integer weights: [(∂ + L_0) a] = 0, [a *_{-2} b] = 0, [a * b] =
    [a][b], [a][b] - p [b][a] = Σ C(Δ_a - 1, j) [a_(j) b]."""
    A = V.zhu_algebra()
    z = A
    for _ in range(trials):
        wa, wb = random.choice(weights), random.choice(weights)
        a, b = homogeneous(V, wa), homogeneous(V, wb)
        if a.is_zero() or b.is_zero():
            continue
        try:
            p = (-1) ** (a.parity() * b.parity())
        except ValueError:                        # not homogeneous for the parity
            p = None
        star = sum(binomial(wa, j) * a.nproduct(b, j - 1) for j in range(wa + 1))
        star2 = sum(binomial(wa, j) * a.nproduct(b, j - 2) for j in range(wa + 1))
        # a_(j) b = 0 for j >= wa + wb (non-negative weights)
        comm = sum(binomial(wa - 1, j) * z(a.nproduct(b, j)) for j in range(wa + wb + 1))
        if z(a.derivative() + wa * a) != 0 or z(star2) != 0 or z(star) != z(a) * z(b):
            return False
        if p is not None and z(a) * z(b) - p * z(b) * z(a) != comm:
            return False
    return True


def zhu_product_formula(V, A):
    """The Zhu map recomputed by Zhu's product formula (independent of the
    De Sole-Kac recursion): [:x B:] = [x][B] - Σ_{i>=1} C(Δ_x, i)
    [x_(i-1) B], [∂^k g] = (-1)^k Δ(Δ+1)...(Δ+k-1) [g] (integer weights)."""
    memo = {}

    def z(m):
        if m in memo:
            return memo[m]
        if not m:
            r = A.one()
        elif len(m) == 1:
            g, kk = m[0]
            h = V.degrees()[V.variable_names()[g]]
            r = (-1) ** kk * rising_factorial(h, kk) * A.gen(V.variable_names()[g])
        else:
            x, B = V.monomial((m[0],)), V.monomial(m[1:])
            hx = ZZ(QQ(x.degree()))
            r = z((m[0],)) * z(m[1:])
            for i in range(1, hx + 2):
                r -= binomial(hx, i) * zd(x.nproduct(B, i - 1))
        memo[m] = r
        return r

    def zd(v):
        return sum((cf * z(m) for m, cf in v.monomial_coefficients().items()), A.zero())
    return zd


# 1. Virasoro, W_3, affine sl2: identities, independent Zhu map, relations --
E.<L> = NormallyOrderedExpressions(K)
Vir = UniversalVertexAlgebra(K, E, {('L', 'L'): [L.derivative(), 2 * L, 0, c / 2]},
                             conformal_vector=L)
A = Vir.zhu_algebra()
Lv = Vir.gen('L')
report('Virasoro: A = C[x], [:LL:] = x^2 + 2x, exact',
       A.relations() == {} and A(Lv.no(Lv)) == A.gen(0) ^ 2 + 2 * A.gen(0)
       and A.is_known_exact())
report('A(v) = A.retract(v), a conversion (no coercion from V), scalars',
       A(Lv.no(Lv)) == A.retract(Lv.no(Lv)) and not A.has_coerce_map_from(Vir)
       and A(3) == 3 * A.one() and A(Vir.zero()) == 0 and A(A.gen(0)) == A.gen(0))
report('Virasoro: Zhu identities', identities_ok(Vir, [2, 3, 4, 5]))
zd = zhu_product_formula(Vir, A)
report('Virasoro: Zhu map = Zhu product formula up to weight 8',
       all(A(Vir.monomial(m)) == zd(Vir.monomial(m))
           for W in range(9) for m in Vir.weight_space(W).basis_keys()))

E3.<L3, W3> = NormallyOrderedExpressions(K)
Lam = L3 * L3 - 3 / 10 * L3.derivative(2)


def w3(beta):
    return UniversalVertexAlgebra(K, E3, {
        ('L3', 'L3'): [L3.derivative(), 2 * L3, 0, c / 2], ('L3', 'W3'): [W3.derivative(), 3 * W3],
        ('W3', 'W3'): [beta * Lam.derivative() + L3.derivative(3) / 15,
                       2 * beta * Lam + 3 / 10 * L3.derivative(2), L3.derivative(), 2 * L3, 0, c / 3]},
        conformal_vector=L3)


VW = w3(16 / (22 + 5 * c))
AW = VW.zhu_algebra()
report('W_3: commutative (relations {}), overlaps resolve, exact',
       AW.relations() == {} and AW.check_overlaps() == [] and AW.is_known_exact())
report('W_3: Zhu identities', identities_ok(VW, [2, 3, 4, 5]))
zd = zhu_product_formula(VW, AW)
report('W_3: Zhu map = Zhu product formula up to weight 7',
       all(AW(VW.monomial(m)) == zd(VW.monomial(m))
           for W in range(8) for m in VW.weight_space(W).basis_keys()))
# a broken Jacobi identity is invisible with two even generators (no
# overlaps at all: W_3 with a wrong beta passes); with three it shows
report('W_3 with a wrong beta: no overlaps to check (two even generators)',
       w3(1).zhu_algebra().check_overlaps() == [])
XS.<Es, Hs, Fs> = NormallyOrderedExpressions(QQ)
broken = UniversalVertexAlgebra(QQ, XS, {('Hs', 'Es'): [3 * Es], ('Hs', 'Fs'): [-2 * Fs],
                                         ('Es', 'Fs'): [Hs, 1], ('Hs', 'Hs'): [0, 2]},
                                degrees={'Es': 1, 'Hs': 1, 'Fs': 1})
report('sl2 with [H, E] = 3E (Jacobi fails): an overlap does not resolve',
       len(broken.zhu_algebra().check_overlaps()) == 1)

Aff = UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, names=['E', 'H', 'F'], conformal_vector='sugawara')
AA = Aff.zhu_algebra()
Ez, Hz, Fz = AA.gens()
report('affine sl2: U(sl2), exact, overlaps',
       Ez * Fz - Fz * Ez == Hz and Hz * Ez - Ez * Hz == 2 * Ez and Hz * Fz - Fz * Hz == -2 * Fz
       and AA.is_known_exact() and AA.check_overlaps() == [])
report('affine sl2: Zhu identities', identities_ok(Aff, [1, 2, 3]))
T = Aff.conformal_vector()
report('affine sl2: [T] central', all(AA(T) * x == x * AA(T) for x in AA.gens()))

# 2. odd generators of integer weight -----------------------------------------
X.<b, cc> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'cc': 1})
BC = UniversalVertexAlgebra(QQ, X, {('b', 'cc'): [1]}, conformal_vector=-b * cc.derivative())
ABC = BC.zhu_algebra()
bz, cz = ABC.gens()
report('bc (weights 1, 0): Clifford algebra, overlaps, not claimed exact',
       bz * cz + cz * bz == 1 and bz * bz == 0 and cz * cz == 0
       and ABC.check_overlaps() == [] and not ABC.is_known_exact())
report('bc: Zhu identities', identities_ok(BC, [0, 1, 2, 3]))
Y.<chp, chm> = NormallyOrderedExpressions(QQ, parities={'chp': 1, 'chm': 1})
SF = UniversalVertexAlgebra(QQ, Y, {('chp', 'chm'): [0, 1]},
                            degrees={'chp': 1, 'chm': 1})
ASF = SF.zhu_algebra()
p_, m_ = ASF.gens()
report('symplectic fermions: exterior algebra on [chi+], [chi-], exact',
       p_ * m_ + m_ * p_ == 0 and p_ * p_ == 0 and ASF.is_known_exact()
       and ASF.check_overlaps() == [])
report('symplectic fermions: Zhu identities', identities_ok(SF, [1, 2, 3]))

# 3. half-integer weights: N=1, N=2, betagamma ---------------------------------
N.<Ln, G> = NormallyOrderedExpressions(K, parities={'G': 1})
N1 = UniversalVertexAlgebra(K, N, {('Ln', 'Ln'): [Ln.derivative(), 2 * Ln, 0, c / 2],
                                   ('Ln', 'G'): [G.derivative(), 3 / 2 * G],
                                   ('G', 'G'): [2 * Ln, 0, 2 * c / 3]}, conformal_vector=Ln)
A1 = N1.zhu_algebra()
Gv = N1.gen('G')
report('N=1: one generator [L], not claimed exact, [:G dG:] = 4[L], [G] = 0',
       A1.ngens() == 1 and not A1.is_known_exact()
       and A1(Gv.no(Gv.derivative())) == 4 * A1.gen(0) and A1(Gv) == 0)


def half_identity_ok(V, A, weights, trials=6):
    """Σ_j C(Δ_a - 1/2, j) [a_(j-1) b] = 0 for half-integer a, b (an element
    of J the recursion does not use directly)."""
    for _ in range(trials):
        wa, wb = random.choice(weights), random.choice(weights)
        a, b = homogeneous(V, wa), homogeneous(V, wb)
        if a.is_zero() or b.is_zero():
            continue
        e = sum(binomial(wa - 1 / 2, j) * a.nproduct(b, j - 1) for j in range(ZZ(wa + 1 / 2) + 2))
        if A(e) != 0:
            return False
    return True


report('N=1: Zhu identities (integer) and the half-integer elements of J',
       identities_ok(N1, [2, 3, 4]) and half_identity_ok(N1, A1, [3 / 2, 5 / 2, 7 / 2]))
N2E.<L2, J2, Gp, Gm> = NormallyOrderedExpressions(K, parities={'Gp': 1, 'Gm': 1})
N2 = UniversalVertexAlgebra(K, N2E, {
    ('L2', 'L2'): [L2.derivative(), 2 * L2, 0, c / 2], ('L2', 'J2'): [J2.derivative(), J2],
    ('L2', 'Gp'): [Gp.derivative(), 3 / 2 * Gp], ('L2', 'Gm'): [Gm.derivative(), 3 / 2 * Gm],
    ('J2', 'J2'): [0, c / 3], ('J2', 'Gp'): [Gp], ('J2', 'Gm'): [-Gm],
    ('Gp', 'Gm'): [L2 + J2.derivative() / 2, J2, c / 3]}, conformal_vector=L2)
A2 = N2.zhu_algebra()
report('N=2: generators [L], [J], commutative, overlaps, identities',
       A2.ngens() == 2 and A2.relations() == {} and A2.check_overlaps() == []
       and identities_ok(N2, [1, 2, 3]) and half_identity_ok(N2, A2, [3 / 2, 5 / 2]))
BG = UniversalVertexAlgebra(QQ, ['be', 'ga'], {('be', 'ga'): [1]},
                            degrees={'be': 1 / 2, 'ga': 1 / 2})
ABG = BG.zhu_algebra()
be, ga = BG.gens()
# :be ga: = be_(-1) ga is itself in J (C(Δ_be - 1/2, j) = C(0, j))
report('betagamma (weights 1/2): no generators, [:be ga:] = 0, [1] = 1',
       ABG.ngens() == 0 and ABG(be.no(ga)) == 0 and ABG(BG.one()) == 1)

# 4. implementation independence ------------------------------------------------
Vp = UniversalVertexAlgebra(K, E3, VW.opes() | {}, conformal_vector='L3',
                            implementation='python')
Ap = Vp.zhu_algebra()
keys = [m for W in range(7) for m in VW.weight_space(W).basis_keys()]
report('W_3: the same Zhu map in pure Python',
       all(AW(VW.monomial(m)).monomial_coefficients()
           == Ap(Vp.monomial(m)).monomial_coefficients() for m in keys))

# 5. quotients -----------------------------------------------------------------
EQ.<Lq> = NormallyOrderedExpressions(QQ)


def virasoro_at(cv):
    return UniversalVertexAlgebra(QQ, EQ, {('Lq', 'Lq'): [Lq.derivative(), 2 * Lq, 0, cv / 2]},
                                  degrees={'Lq': 2})


def singular(V, W):
    """The singular vectors of weight W (killed by L_1, L_2)."""
    Lg = V.gen(0)
    S = V.weight_space(W).basis()
    rows = []
    for n in (2, 3):
        ks = list(V.weight_space(W + 1 - n).basis_keys())
        imgs = [Lg.nproduct(s, n).monomial_coefficients() for s in S]
        rows += [[d.get(kk, 0) for d in imgs] for kk in ks]
    return [sum(cf * s for cf, s in zip(r, S)) for r in matrix(QQ, rows).right_kernel_matrix().rows()]


x = polygen(QQ)


def as_poly(f):
    return sum(cf * x ^ len(m) for m, cf in f.monomial_coefficients().items())


for cv, W, roots, name in [(QQ(-22) / 5, 4, {0, QQ(-1) / 5}, 'Lee-Yang'),
                           (QQ(1) / 2, 6, {0, QQ(1) / 16, QQ(1) / 2}, 'Ising')]:
    V = virasoro_at(cv)
    [v] = singular(V, W)
    f = as_poly(V.zhu_algebra()(v))
    Q = V.quotient([v])
    dims = [Q.zhu_algebra_up_to(w).dimension() for w in (W - 1, W, W + 2, W + 4)]
    report('%s: [v] has the roots %s; zhu_algebra_up_to dims %s' % (name, sorted(roots), dims),
           set(r for r, _ in f.roots()) == roots and dims == [Infinity] + [len(roots)] * 3)
Z = Q.zhu_algebra_up_to(8)
Lg = V.gen(0)
report('Ising: cover_ring, Z(x) for x in A(V), V, Q (canonical representative)',
       Z.cover_ring() is V.zhu_algebra() and Z(Z.cover_ring()(v)) == 0 and Z(v) == 0
       and Z(Q(v)) == 0 and Z(Lg.no(Lg)) == Z.retract(Lg.no(Lg))
       and Z(Q(Lg.no(Lg))) == Z(Lg.no(Lg)) and Z(Z.cover_ring().gen(0)) == Z.gen(0))
try:
    TestSuite(Z).run(raise_on_failure=True)
    suite = True
except Exception as err:
    suite = str(err)
report('Ising: TestSuite of the quotient, products commute',
       suite is True and all(a * b == b * a for a in Z.basis() for b in Z.basis()), suite)

for lev, dim in [(1, 5), (2, 14)]:
    Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=lev, names=['E', 'H', 'F'], conformal_vector='sugawara')
    Ea = Va.gen('E')
    v = Ea
    for _ in range(lev):
        v = Ea.no(v)
    Qa = Va.quotient([v])
    dims = [Qa.zhu_algebra_up_to(w).dimension() for w in (lev, lev + 1, lev + 2)]
    report('L_%d(sl2): zhu_algebra_up_to dims %s (U(sl2)/<e^%d>: %d)' % (lev, dims, lev + 1, dim),
           dims == [Infinity, dim, dim])
Va1 = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1, names=['E', 'H', 'F'], conformal_vector='sugawara')
E1, F1 = Va1.gen('E'), Va1.gen('F')
Q1 = Va1.quotient([E1.no(E1)])
Q2 = Q1.quotient([Q1(F1.no(F1))])          # already zero in Q1
report('nested quotient: the same dimension 5', Q2.zhu_algebra_up_to(3).dimension() == 5)

# 6. lattice vertex algebras ---------------------------------------------------
# Positive definite: A(V_L) = ⊕ End(top levels), the Zhu map by zero modes
# (a formula, no products): checked by Zhu's identities computed with the
# lattice products, surjectivity, L_0 on the top levels, and against the
# universal algebras above (L_1(sl2), bc).  Otherwise: zero, with the element
# of O(V) of the proof computed with the products.
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra


def lattice(B, alpha=None, weights=None):
    """V_L with T = T_sug + ∂J_alpha (alpha in coordinates), or with the
    given conformal weights."""
    B = matrix(ZZ, B)
    L = LatticeVertexAlgebra(QQ, B)
    if weights is not None:
        return L.with_degrees(weights)
    J = L.heisenberg_currents()
    n = B.nrows()
    Bi = B.inverse()
    T = sum(Bi[i, j] * J[i].no(J[j]) for i in range(n) for j in range(n)) / 2
    if alpha is not None:
        T += sum(QQ(a) * J[i].derivative() for i, a in enumerate(alpha))
    return L.with_conformal_vector(T)


def lattice_identities_ok(V, weights, trials=10, A=None):
    """Zhu's identities for A (default V.zhu_algebra()) on random homogeneous a, b,
    any weights (the sums end at the OPE length): integer a, b: [(∂ + L_0) a]
    = 0, [a *_{-2} b] = 0, [a * b] = [a][b], the supercommutator formula;
    half-integer a, b: [Σ_j C(Δ_a - 1/2, j) a_(j-1) b] = 0."""
    if A is None:
        A = V.zhu_algebra()
    seen = 0
    for _ in range(trials):
        wa, wb = random.choice(weights), random.choice(weights)
        a, b = homogeneous(V, wa), homogeneous(V, wb)
        if a.is_zero() or b.is_zero():
            continue
        N = len(a.ope(b))                     # a_(n) b = 0 for n >= N
        if wa in ZZ and wb in ZZ:
            star = sum(binomial(wa, j) * a.nproduct(b, j - 1) for j in range(N + 1))
            star2 = sum(binomial(wa, j) * a.nproduct(b, j - 2) for j in range(N + 2))
            if A(a.derivative() + wa * a) != 0 or A(star2) != 0 or A(star) != A(a) * A(b):
                return False
            try:
                p = (-1) ** (a.parity() * b.parity())
            except ValueError:                    # not homogeneous for the parity
                p = None
            comm = sum(binomial(wa - 1, j) * A(a.nproduct(b, j)) for j in range(N))
            if p is not None and A(a) * A(b) - p * A(b) * A(a) != comm:
                return False
            seen += 1
        elif wa not in ZZ and wb not in ZZ:
            el = sum(binomial(wa - 1/2, j) * a.nproduct(b, j - 1) for j in range(N + 1))
            if A(el) != 0:
                return False
            seen += 1
    return seen > 0


def lattice_surjective(V, weights):
    """The images of the basis keys of these weights span A."""
    A = V.zhu_algebra()
    rows = [A(V.monomial(m)).to_vector() for w in weights
            for m in V.weight_space(w).basis_keys()]
    return matrix(QQ, rows).rank() == A.dimension()


def l0_on_top_levels(V):
    """A(T) acts on each top level as its conformal weight."""
    A = V.zhu_algebra()
    mats = A(V.conformal_vector()).matrices()
    return all(M == h for M, (h, _) in zip(mats, A.top_levels()))


VA1 = lattice([[2]])
ZA1 = VA1.zhu_algebra()
report('A1 = L_1(sl2): C + M_2, as the universal quotient above (dimension 5)',
       ZA1.dimension() == 5 and [(h, len(m)) for h, m in ZA1.top_levels()]
       == [(0, 1), (1/4, 2)] and ZA1.is_known_exact())
report('A1: Zhu identities, surjective from weights <= 2, L_0 on the top levels',
       lattice_identities_ok(VA1, [0, 1, 2, 3]) and lattice_surjective(VA1, [0, 1, 2])
       and l0_on_top_levels(VA1))
VA1b = lattice([[2]], alpha=[1/4])            # beta = -1/2: half-integer weights
ZA1b = VA1b.zhu_algebra()
report('A1 with a background charge (weights in 1/2 Z): C + C, identities incl. '
       'half-integer elements, surjective, L_0',
       ZA1b.top_levels() == [(0, [(0,)]), (0, [(1/2,)])]
       and lattice_identities_ok(VA1b, [0, 1/2, 1, 3/2, 2, 5/2], trials=14)
       and lattice_surjective(VA1b, [0, 1/2, 1]) and l0_on_top_levels(VA1b))
VA2 = lattice([[2, -1], [-1, 2]])
ZA2 = VA2.zhu_algebra()
# surjective only from weight 3: below it the diagonal of the 7 top-level
# vectors comes from the 6 polynomials of degree <= 2 in B(x_i, μ)
report('A2 = L_1(sl3): dimension 19 (1 + 9 + 9), identities, surjective from weight 3, L_0',
       ZA2.dimension() == 19 and [len(m) for h, m in ZA2.top_levels()] == [1, 3, 3]
       and lattice_identities_ok(VA2, [0, 1, 2]) and lattice_surjective(VA2, [0, 1, 2, 3])
       and l0_on_top_levels(VA2))
VA2b = lattice([[2, -1], [-1, 2]], alpha=[1/2, 0])
report('A2 with a background charge: identities, surjective, L_0',
       lattice_identities_ok(VA2b, sorted(set(QQ(x) / 2 for x in range(-2, 6))), trials=14)
       and lattice_surjective(VA2b, [QQ(x) / 2 for x in range(-2, 7)]) and l0_on_top_levels(VA2b))
# negative control: the identities see a wrong zero-mode formula (A2 has a
# nontrivial cocycle)
from vertex_algebras import zhu_algebra as _zhu
_orig_zeta = _zhu.LatticeZhuAlgebra._zeta_key


def _broken_zeta(drop):
    def zeta(self, m):
        letters, q = m
        r = {}
        for mu, (t, _, p, Bmu) in self._where.items():
            nu = tuple(x + y for x, y in zip(mu, q))
            if nu in self._where:
                cf = QQ(1) if drop == 'cocycle' else self._V.cocycle(q, p)
                for i, kk in letters:
                    cf *= (1 if drop == 'sign' else (-1) ** kk) * factorial(kk) * Bmu[i]
                if cf:
                    r[(nu, mu)] = cf
        return r
    return zeta


caught = []
for drop in ('cocycle', 'sign'):
    ZA2._zeta.clear()
    _zhu.LatticeZhuAlgebra._zeta_key = _broken_zeta(drop)
    try:
        caught.append(not lattice_identities_ok(VA2, [0, 1, 2, 3], trials=20))
    finally:
        _zhu.LatticeZhuAlgebra._zeta_key = _orig_zeta
        ZA2._zeta.clear()
report('negative control: the identities fail without the cocycle / the sign (-1)^n',
       caught == [True, True], caught)
VD4 = lattice(CartanMatrix(['D', 4]))
ZD4 = VD4.zhu_algebra()
report('D4: dimension 193 (three 8-dimensional top levels of weight 1/2), identities',
       ZD4.dimension() == 193 and [(h, len(m)) for h, m in ZD4.top_levels()]
       == [(0, 1), (1/2, 8), (1/2, 8), (1/2, 8)]
       and lattice_identities_ok(VD4, [0, 1], trials=4) and l0_on_top_levels(VD4))

VZ = lattice([[1]])                           # a complex fermion, weights 1/2
report('V_Z (weights 1/2): A = C; identities incl. half-integer elements',
       VZ.zhu_algebra().dimension() == 1
       and lattice_identities_ok(VZ, [1/2, 1, 3/2, 2], trials=12))
VBC = lattice([[1]], weights={'J0': 1, 'e^(1)': 1, 'e^(-1)': 0})
ZBC = VBC.zhu_algebra()
bl, cl = ZBC(VBC.e((1,))), ZBC(VBC.e((-1,)))
report('V_Z = bc (weights 1, 0): End(C^{1|1}) = the Clifford algebra of the '
       'universal bc above, super, identities, surjective',
       ZBC.dimension() == 4 and bl * cl + cl * bl == ZBC.one() and bl * bl == 0
       and cl * cl == 0 and bl.is_even_odd() == 1 and 'super' in str(ZBC.category())
       and lattice_identities_ok(VBC, [0, 1, 2]) and lattice_surjective(VBC, [0, 1]))
V3 = lattice([[3]])                           # N=2 at c = 1
report('V_{sqrt3 Z} (N=2, c = 1): three modules of weights 0, 1/6, 1/6; identities',
       [(h, len(m)) for h, m in V3.zhu_algebra().top_levels()] == [(0, 1), (1/6, 1), (1/6, 1)]
       and lattice_identities_ok(V3, [1, 3/2, 2, 5/2, 3], trials=12))

zero_cases = []
for B, kw in [([[-2]], {}), ([[1, 0], [0, -1]], {}), ([[2, 0], [0, -2]], {}),
              ([[2, 1], [1, -2]], {}),
              ([[0, 1], [1, 0]], {'weights': {'J0': 1, 'J1': 1, 'e^(1,0)': 0, 'e^(0,1)': 0,
                                               'e^(-1,0)': 0, 'e^(0,-1)': 0}}),
              ([[-1]], {'alpha': [1/2]})]:
    V = lattice(B, **kw)
    Z = V.zhu_algebra()
    Bm = matrix(ZZ, B)
    # a vector of negative norm (doubled if odd), e^{±alpha}, k = -(alpha, alpha)/2
    al = next(v for v in [vector(ZZ, t) for t in _cartesian(*[range(-2, 3)] * Bm.nrows())]
              if v * Bm * v < 0)
    if (al * Bm * al) % 2:
        al = 2 * al
    k = -(al * Bm * al) / 2
    a, b = V.e(tuple(al)), V.e(tuple(-al))
    wa = a.degree()
    el = sum(binomial(wa, i) * a.nproduct(b, i - 2 * k - 1) for i in range(2 * k + 2))
    zero_cases.append(Z.dimension() == 0 and Z.one() == 0 and Z(a) == 0 and Z(V.one()) == 0
                      and el != 0 and el == el.monomial_coefficients()[V.one().leading_support()]
                      * V.one() and 'not positive definite' in repr(Z))
report('not positive definite (negative definite, indefinite, odd, with weights '
       'only, background charge): A = 0, the element of O(V) of the proof is ±1',
       all(zero_cases), zero_cases)
try:
    lattice([[2]], alpha=[1/3]).zhu_algebra()
    refused = False
except ValueError as err:
    refused = 'not an integer or half-integer' in str(err)
report('weights outside 1/2 Z refused; no weights: no zhu_algebra',
       refused and not hasattr(LatticeVertexAlgebra(QQ, matrix([[2]])), 'zhu_algebra'))

# 7. tensor products ------------------------------------------------------------
# A(V_1 ⊗ V_2) = A(V_1) ⊗ A(V_2) (super tensor product), [a ⊗ b] = [a] ⊗ [b]:
# Zhu's identities with the tensor products' products (Koszul signs, mixed
# kinds of factors, half-integer weights, factors graded by the tensor
# product, a logarithmic T), isomorphic lattices, quotient factors.
VirA = virasoro_at(QQ(1) / 2)
VirB = virasoro_at(QQ(-22) / 5)
VAB = VirA.tensor(VirB, degrees='factors')
ZAB = VAB.zhu_algebra()
xA, xB = [ZAB.embed(i, B.gen(0)) for i, B in enumerate(ZAB.factors())]
report('Vir ⊗ Vir: C[x] ⊗ C[y], exact, identities, [L ⊗ L] = [L] ⊗ [L]',
       ZAB.is_known_exact() and xA * xB == xB * xA
       and ZAB(VAB.pure_tensor(VirA.gen(0), VirB.gen(0))) == xA * xB
       and lattice_identities_ok(VAB, [2, 3, 4, 5]))
VL1V = VA1.tensor(VirB, degrees='factors')
report('lattice A1 ⊗ Virasoro: identities, [e ⊗ L] = [e] ⊗ [L]',
       lattice_identities_ok(VL1V, [0, 1, 2, 3, 4], trials=12)
       and VL1V.zhu_algebra()(VL1V.pure_tensor(VA1.e((1,)), VirB.gen(0)))
       == VL1V.zhu_algebra().pure_tensor(ZA1(VA1.e((1,))), VirB.zhu_algebra().gen(0)))
VBB = BC.tensor(BC, conformal_vector='factors')
ZBB = VBB.zhu_algebra()
report('bc ⊗ bc (odd): super, identities (Koszul signs)',
       ZBB in Algebras(QQ).Super() and lattice_identities_ok(VBB, [0, 1, 2], trials=14))
_orig_prod = _zhu.TensorProductZhuAlgebra.product_on_basis


def _unsigned(self, k, l):
    d = self._tensor_dicts([A.product_on_basis(x, y)._monomial_coefficients
                            for A, x, y in zip(self._factors, k, l)])
    return self._from_dict(d, coerce=False, remove_zeros=True)


_zhu.TensorProductZhuAlgebra.product_on_basis = _unsigned
try:
    caught = not lattice_identities_ok(VBB, [0, 1, 2], trials=14)
finally:
    _zhu.TensorProductZhuAlgebra.product_on_basis = _orig_prod
report('negative control: without the Koszul sign the identities fail', caught)
VLB = VBC.tensor(BC, degrees='factors')
report('lattice bc ⊗ universal bc: identities',
       lattice_identities_ok(VLB, [0, 1, 2], trials=14))
VZZ = VZ.tensor(VZ, conformal_vector='factors')      # two complex fermions, weights 1/2
ZZZ = VZZ.zhu_algebra()
VZA = VZ.tensor(VA1, conformal_vector='factors')
report('V_Z ⊗ V_Z = V_{Z^2} and V_Z ⊗ V_A1 = V_diag(1,2): dims 1, 5; half ⊗ half -> 0; '
       'identities incl. half-integer elements',
       ZZZ.dimension() == 1 == lattice([[1, 0], [0, 1]]).zhu_algebra().dimension()
       and VZA.zhu_algebra().dimension() == 5 == lattice([[1, 0], [0, 2]]).zhu_algebra().dimension()
       and ZZZ(VZZ.pure_tensor(VZ.e((1,)), VZ.e((-1,)))) == 0
       and lattice_identities_ok(VZZ, [1/2, 1, 3/2, 2], trials=14))
Hh = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
Jh = Hh.gen('J')
L0p = LatticeVertexAlgebra(QQ, matrix([[2]]))
Jl = L0p.J(0)
Wh = Hh.tensor(L0p)
Wt = Wh.with_conformal_vector(Wh.embed(0, Jh.no(Jh) / 2) + Wh.embed(1, Jl.no(Jl) / 4 + Jl.derivative()))
ZWt = Wt.zhu_algebra()
report('factors graded by the tensor product (no conformal data of their own): '
       'H ⊗ V_A1 with a background charge, identities',
       ZWt.factors()[1].top_levels() == [(-1, [(1,)]), (-3/4, [(1/2,), (3/2,)])]
       and lattice_identities_ok(Wt, [-1, 0, 1, 2], trials=14))
Zp = LatticeVertexAlgebra(QQ, matrix([[1]]))
Ip = Zp.J(0)
Wz = Zp.tensor(Zp)
Tz = Wz.embed(0, Ip.no(Ip) / 2) + Wz.embed(1, Ip.no(Ip) / 2)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Wlog = Wz.with_conformal_vector(Tz + Wz.pure_tensor(Zp.e((1,)), Zp.e((-1,))).derivative())
report('a logarithmic T (V_Z ⊗ V_Z, T + ∂(e^1 ⊗ e^-1)): the grading of its '
       'semisimple part, identities',
       not Wlog.l0_is_diagonalizable() and Wlog.zhu_algebra().dimension() == 1
       and lattice_identities_ok(Wlog, [1/2, 1, 3/2, 2], trials=12))
LY = virasoro_at(QQ(-22) / 5)
[vly] = singular(LY, 4)
QLY = LY.quotient([vly])
VQQ = QLY.tensor(QLY, degrees='factors')
try:
    VQQ.zhu_algebra()
    refused = False
except NotImplementedError as err:
    refused = 'zhu_algebra_up_to' in str(err)
ZQQ = VQQ.zhu_algebra_up_to(8)
VIs = virasoro_at(QQ(1) / 2)
[vis] = singular(VIs, 6)
VQL = VIs.quotient([vis]).tensor(VA1, degrees='factors')
report('quotient factors: zhu_algebra() refused, Lee-Yang ⊗ Lee-Yang dims 4 '
       '(identities up to weight 8), Ising ⊗ L_1(sl2) dims 15, not claimed exact',
       refused and ZQQ.dimension() == 4 and not ZQQ.is_known_exact()
       and VQL.zhu_algebra_up_to(6).dimension() == 15
       and lattice_identities_ok(VQQ, [0, 2, 3, 4], trials=10, A=ZQQ))
report('tensor products: unique, pickling, zhu_algebra_up_to without quotients = '
       'zhu_algebra()',
       ZAB is VAB.zhu_algebra() and loads(dumps(ZAB)) is ZAB
       and VAB.zhu_algebra_up_to(3) is ZAB)

# 8. the H-twisted (Ramond) Zhu algebra ------------------------------------------
# A_H(V) = V / (V *_{-2} V) with Zhu's formulas for every element (any rational
# weight): Zhu's identities on random homogeneous elements of all weights, the
# Ramond relations of N=1 and N=2, Weyl / Clifford algebras of free fields
# (weights 1/2 and 1/3), the Ramond sector of the tricritical Ising model
# (weights 3/80, 7/16), the H-twisted Bershadsky-Polyakov algebra (exact, a
# Smith algebra whose centre in degree <= 3 is spanned by 1 and two Casimirs, as
# the centre of U(sl_3): Premet), tensor products, refusals.
from vertex_algebras.lie_superalgebras import LieSuperalgebra
from vertex_algebras.quantum_hamiltonian_reduction import WAlgebra


def twisted_identities_ok(V, weights, trials=8, A=None):
    """Zhu's identities in A = A_H(V) on random homogeneous a, b of the given
    (rational) weights: [(∂ + L_0) a] = 0, [a *_{-2} b] = 0, [a * b] = [a][b],
    [a][b] - p [b][a] = Σ C(Δ_a - 1, j) [a_(j) b] (a_(n) b = 0 for n >= Δ_a + Δ_b)."""
    z = V.zhu_algebra(twist='H') if A is None else A
    for _ in range(trials):
        wa, wb = QQ(random.choice(weights)), QQ(random.choice(weights))
        a, b = homogeneous(V, wa), homogeneous(V, wb)
        if a.is_zero() or b.is_zero():
            continue
        top = (wa + wb).floor() + 3
        p = (-1) ** (a.parity() * b.parity())
        star = sum(binomial(wa, j) * a.nproduct(b, j - 1) for j in range(top))
        star2 = sum(binomial(wa, j) * a.nproduct(b, j - 2) for j in range(top))
        comm = sum(binomial(wa - 1, j) * z(a.nproduct(b, j)) for j in range(top))
        if z(a.derivative() + wa * a) != 0 or z(star2) != 0 or z(star) != z(a) * z(b):
            return False
        if z(a) * z(b) - p * z(b) * z(a) != comm:
            return False
    return True


A1h = N1.zhu_algebra(twist='ramond')
Lh, Gh = A1h.gens()
report("N=1 twisted: [L], [G], exact, [G]^2 = [L] - c/24 (Ramond), 'H' = 'ramond', "
       'identities (weights 3/2 .. 4); negative control: the untwisted algebra fails them',
       A1h is N1.zhu_algebra(twist='H') and A1h.is_known_exact() and A1h.ngens() == 2
       and Gh * Gh == Lh - c / 24 and A1h.check_overlaps() == []
       and twisted_identities_ok(N1, [3 / 2, 2, 5 / 2, 3, 7 / 2, 4])
       and not twisted_identities_ok(N1, [3 / 2, 5 / 2], A=A1))
A2h = N2.zhu_algebra(twist='H')
L2h, J2h, Gph, Gmh = A2h.gens()
report('N=2 twisted: exact, [J][G±] - [G±][J] = ±[G±], [G±]^2 = 0, '
       '[G+][G-] + [G-][G+] = [L] - c/24, overlaps, identities',
       A2h.is_known_exact() and J2h * Gph - Gph * J2h == Gph and J2h * Gmh - Gmh * J2h == -Gmh
       and Gph * Gph == 0 and Gph * Gmh + Gmh * Gph == L2h - c / 24
       and A2h.check_overlaps() == [] and twisted_identities_ok(N2, [1, 3 / 2, 2, 5 / 2, 3]))
ABGh = BG.zhu_algebra(twist='H')
beh, gah = ABGh.gens()
BG3 = UniversalVertexAlgebra(QQ, ['be', 'ga'], {('be', 'ga'): [1]},
                             degrees={'be': 1 / 3, 'ga': 2 / 3})
try:
    BG3.zhu_algebra()
    refused = False
except ValueError as err:
    refused = "twist='H'" in str(err)
AB3 = BG3.zhu_algebra(twist='H')
b3, g3 = AB3.gens()
Xf.<bf, cf> = NormallyOrderedExpressions(QQ, parities={'bf': 1, 'cf': 1})
BCh = UniversalVertexAlgebra(QQ, Xf, {('bf', 'cf'): [1]},
                             degrees={'bf': 1 / 2, 'cf': 1 / 2})
ABCh = BCh.zhu_algebra(twist='H')
bh, ch = ABCh.gens()
report('free fields twisted: betagamma (weights 1/2 and 1/3, 2/3): the Weyl algebra '
       '[be][ga] - [ga][be] = 1; bc (1/2): Clifford; identities; weights 1/3 refused '
       'untwisted',
       beh * gah - gah * beh == 1 and b3 * g3 - g3 * b3 == 1 and AB3.is_known_exact()
       and bh * ch + ch * bh == 1 and bh * bh == 0 and refused
       and twisted_identities_ok(BG, [1 / 2, 1, 3 / 2, 2])
       and twisted_identities_ok(BG3, [1 / 3, 2 / 3, 1, 4 / 3, 5 / 3])
       and twisted_identities_ok(BCh, [1 / 2, 1, 3 / 2, 2]))
report('integer weights: the twist is trivial (the same algebra)',
       Vir.zhu_algebra(twist='H') is A and VW.zhu_algebra(twist='ramond') is AW
       and Aff.zhu_algebra(twist='H') is AA and BC.zhu_algebra(twist='H') is ABC)
# the tricritical Ising model: N=1 at c = 7/10, singular vector of weight 4
NT.<Lt, Gt> = NormallyOrderedExpressions(QQ, parities={'Gt': 1})
TI = UniversalVertexAlgebra(QQ, NT, {('Lt', 'Lt'): [Lt.derivative(), 2 * Lt, 0, 7 / 20],
                                     ('Lt', 'Gt'): [Gt.derivative(), 3 / 2 * Gt],
                                     ('Gt', 'Gt'): [2 * Lt, 0, 7 / 15]}, conformal_vector=Lt)
Ltv, Gtv = TI.gens()
S4 = TI.weight_space(4).basis()
rows = []
for g, n in [(Ltv, 2), (Ltv, 3), (Gtv, 1), (Gtv, 2)]:
    imgs = [g.nproduct(s, n).monomial_coefficients() for s in S4]
    ks = sorted(set(kk for d in imgs for kk in d))
    rows += [[d.get(kk, 0) for d in imgs] for kk in ks]
[vti] = [sum(cf * s for cf, s in zip(r, S4)) for r in matrix(QQ, rows).right_kernel_matrix().rows()]
QTI = TI.quotient([vti])
ZTI = QTI.zhu_algebra_up_to(6, twist='ramond')
Lz, Gz = ZTI.gens()
report('tricritical Ising (N=1, c = 7/10): Ramond Zhu algebra of dimension 4, [L] with the '
       'Ramond weights 3/80, 7/16, [G]^2 = [L] - 7/240; NS: dimension 2',
       ZTI.dimension() == 4 and (Lz - 3 / 80) * (Lz - 7 / 16) == 0 and Lz != 3 / 80
       and Lz != 7 / 16 and Gz * Gz == Lz - 7 / 240 and QTI.zhu_algebra_up_to(6).dimension() == 2)
sl3 = LieSuperalgebra(QQ, CartanMatrix(['A', 2]))
BPW = WAlgebra(Kk, sl3, 'minimal', k, conformal_vector='reduction')
ABP = BPW.zhu_algebra(twist='H')
Jb, Gpb, Gmb, Lb = ABP.gens()
mons = [m for d in range(4) for m in combinations_with_replacement(range(4), d)
        if ABP.degree_on_basis(m) <= 3]
els = [ABP.monomial(m) for m in mons]
rows = []
for xg in ABP.gens():
    imgs = [(xg * e - e * xg).monomial_coefficients() for e in els]
    ks = sorted(set(kk for d in imgs for kk in d))
    rows += [[d.get(kk, 0) for d in imgs] for kk in ks]
centre = matrix(Kk, rows).right_kernel_matrix().nrows()
report('Bershadsky-Polyakov (sl_3 minimal, Dynkin grading) twisted: exact, 4 generators, '
       '[J][G±] = [G±][J] ± 3[G±], centre of degree <= 3 of dimension 3 (1 and two '
       'Casimirs), overlaps, identities; untwisted: 2 generators',
       ABP.is_known_exact() and ABP.ngens() == 4 and Jb * Gpb - Gpb * Jb == -3 * Gpb
       and Jb * Gmb - Gmb * Jb == 3 * Gmb and centre == 3 and ABP.check_overlaps() == []
       and BPW.zhu_algebra().ngens() == 2
       and twisted_identities_ok(BPW, [1, 3 / 2, 2, 5 / 2], trials=6))
YF.<psi> = NormallyOrderedExpressions(QQ, parities={'psi': 1})
Fm = UniversalVertexAlgebra(QQ, YF, {('psi', 'psi'): [1]}, conformal_vector=psi.derivative() * psi / 2)
VFF = Fm.tensor(Fm, conformal_vector='factors')
ZFF = VFF.zhu_algebra(twist='H')
p0, p1 = [ZFF.embed(i, ZFF.factors()[i].gen(0)) for i in (0, 1)]
report('tensor products twisted: two Majorana fermions (Clifford: [psi_i]^2 = 1/2, '
       'anticommuting), identities with the tensor products\' products',
       p0 * p0 == ZFF.one() / 2 and p0 * p1 + p1 * p0 == 0
       and ZFF is VFF.zhu_algebra_up_to(3, twist='ramond')
       and twisted_identities_ok(VFF, [1 / 2, 1, 3 / 2, 2], A=ZFF))
LZ1 = LatticeVertexAlgebra(QQ, matrix([[1]]))
Jz = LZ1.J(0)
FZ = LZ1.with_conformal_vector(Jz.no(Jz) / 2)                 # V_Z: weights 1/2
try:
    FZ.zhu_algebra(twist='H')
    lat_refused = False
except NotImplementedError:
    lat_refused = True
report('lattice: the twist refused with non-integer weights; trivial with integer ones',
       lat_refused and ZA1 is ZA1.vertex_algebra().zhu_algebra(twist='H'))

# TestSuite (axioms of algebras with basis, super, pickling, overlaps)
suites = []
for name, Z in [('Virasoro', A), ('W_3', AW), ('affine sl2', AA), ('bc', ABC),
                ('symplectic fermions', ASF), ('N=1', A1), ('N=2', A2), ('betagamma', ABG),
                ('lattice A1', ZA1), ('lattice A1, background charge', ZA1b),
                ('lattice bc', ZBC), ('lattice, zero', Z), ('Vir ⊗ Vir', ZAB),
                ('bc ⊗ bc', ZBB), ('V_Z ⊗ V_Z', ZZZ), ('H ⊗ V_A1, background', ZWt),
                ('Lee-Yang ⊗ Lee-Yang', ZQQ), ('N=1 twisted', A1h), ('N=2 twisted', A2h),
                ('betagamma 1/3 twisted', AB3), ('bc 1/2 twisted', ABCh),
                ('tricritical Ising, Ramond', ZTI), ('Bershadsky-Polyakov twisted', ABP),
                ('Majorana ⊗ Majorana twisted', ZFF)]:
    try:
        TestSuite(Z).run(raise_on_failure=True)
    except Exception as err:
        suites.append((name, str(err)[:300]))
report('TestSuite of the Zhu algebras', not suites, suites)

print('total %.1fs' % (time.time() - t_start))
report.result()
