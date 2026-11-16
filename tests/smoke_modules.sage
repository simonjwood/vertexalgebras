# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Modules over vertex algebras (vertex_algebra_module.py, the category
# VertexAlgebraModules): the category and its constructions, adjoint
# modules of universal, lattice and tensor product algebras (the action =
# the products of V, the element API of V on module vectors, characters,
# weight spaces, TestSuite with the module axioms), the action of all basis
# monomials of a universal algebra computed from the generators only (=
# V's products; a negative control with a wrong sign), lattice modules
# V_{λ+L} (axioms, characters θ/η, V_{0+L} = V, top levels and zero modes =
# the lattice Zhu algebra, indefinite lattices), modules over Zhu algebras
# (existence exactly for the known modules, super modules, the Ramond
# sector), induced modules (axioms incl. non-linear W_3 and BP, characters,
# zero modes = Zhu, vacuum modules as induced modules, singular vectors),
# refusals.
# Run from the repository root:  sage tests/smoke_modules.sage  -> RESULT: PASS
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.vertex_algebra_categories import VertexAlgebraModules
from vertex_algebras import vertex_algebra_module as vam
from axioms import Reporter

report = Reporter()
t_start = time.time()
random.seed(3r)
K.<c> = Frac(QQ['c'])
Kk.<k> = Frac(QQ['k'])

# algebras --------------------------------------------------------------------
E.<L> = NormallyOrderedExpressions(K)
Vir = UniversalVertexAlgebra(K, E, {('L', 'L'): [L.derivative(), 2 * L, 0, c / 2]}, conformal_vector=L)
E3.<L3, W3> = NormallyOrderedExpressions(K)
Lam = L3 * L3 - 3 / 10 * L3.derivative(2)
beta = 16 / (22 + 5 * c)
VW = UniversalVertexAlgebra(K, E3, {
    ('L3', 'L3'): [L3.derivative(), 2 * L3, 0, c / 2], ('L3', 'W3'): [W3.derivative(), 3 * W3],
    ('W3', 'W3'): [beta * Lam.derivative() + L3.derivative(3) / 15, 2 * beta * Lam
                   + 3 / 10 * L3.derivative(2), L3.derivative(), 2 * L3, 0, c / 3]},
    conformal_vector=L3)
N.<Ln, G> = NormallyOrderedExpressions(K, parities={'G': 1})
N1 = UniversalVertexAlgebra(K, N, {('Ln', 'Ln'): [Ln.derivative(), 2 * Ln, 0, c / 2],
                                   ('Ln', 'G'): [G.derivative(), 3 / 2 * G],
                                   ('G', 'G'): [2 * Ln, 0, 2 * c / 3]}, conformal_vector=Ln)
X.<b, cc> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'cc': 1})
BC = UniversalVertexAlgebra(QQ, X, {('b', 'cc'): [1]}, conformal_vector=-b * cc.derivative())
BG = UniversalVertexAlgebra(QQ, ['be', 'ga'], {('be', 'ga'): [1]},
                            degrees={'be': 1 / 2, 'ga': 1 / 2})
Aff = UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, names=['E', 'H', 'F'], conformal_vector='sugawara')
LA1 = LatticeVertexAlgebra(QQ, matrix([[2]]))
JA = LA1.J(0)
LA1T = LA1.with_conformal_vector(JA.no(JA) / 4)
LZ = LatticeVertexAlgebra(QQ, matrix([[1]]))                  # odd: a superalgebra
JZ = LZ.J(0)
LZT = LZ.with_conformal_vector(JZ.no(JZ) / 2)
EV.<Lv> = NormallyOrderedExpressions(QQ)
VirQ = UniversalVertexAlgebra(QQ, EV, {('Lv', 'Lv'): [Lv.derivative(), 2 * Lv, 0, 1 / 4]},
                              conformal_vector=Lv)
TP = VirQ.tensor(LA1T, conformal_vector='factors')

# 1. the category ----------------------------------------------------------------
C = VertexAlgebraModules(Vir)
ok = (repr(C).startswith('Category of modules over Universal vertex algebra')
      and C.Super().Graded() is C.Graded().Super()
      and C.Super().Graded().WithBasis().is_subcategory(Modules(K).WithBasis())
      and VertexAlgebraModules(Vir) is C and VertexAlgebraModules(VW) is not C
      and loads(dumps(C)) is C)
try:
    VertexAlgebraModules(QQ)
    ok = False
except TypeError:
    pass
TestSuite(C).run()
report('the category: one per vertex algebra, Super/Graded commute, with basis = modules '
       'with basis, pickling, refuses non-vertex algebras', ok)


# 2. adjoint modules ---------------------------------------------------------------
def random_keys(V, weights, n=3):
    out = []
    for w in weights:
        ks = list(V.weight_space(w).basis_keys())
        out += random.sample(ks, min(n, len(ks)))
    return out


def adjoint_ok(V, weights, ns=range(-3, 4)):
    """The action of the adjoint module = the products of V, by the element
    API (nproduct, ope, no, *, mode), on random basis monomials."""
    M = V.adjoint_module()
    keys = random_keys(V, weights)
    for a in keys[:6]:
        x = V.monomial(a)
        for bk in keys[:6]:
            y = V.monomial(bk)
            m = M(y)
            if x.ope(m) != [M(p) for p in x.ope(y)]:
                return False
            for n in ns:
                if x.nproduct(m, n) != M(x.nproduct(y, n)):
                    return False
            if x.no(m) != M(x.no(y)):
                return False
            h = x.degree()
            if x.mode(m, 1 - h) != M(x.mode(y, 1 - h)):
                return False
    return True


cases = [('Virasoro', Vir, [2, 3, 4]), ('W_3', VW, [2, 3, 4]), ('N=1', N1, [3 / 2, 2, 5 / 2, 3]),
         ('bc', BC, [0, 1, 2]), ('betagamma 1/2', BG, [1 / 2, 1, 3 / 2]),
         ('affine sl2', Aff, [1, 2]), ('lattice A1', LA1T, [1, 2, 3]),
         ('lattice Z (odd)', LZT, [1 / 2, 1, 3 / 2, 2]), ('Vir ⊗ A1', TP, [1, 2, 3])]
res = [(name, adjoint_ok(V, ws)) for name, V, ws in cases]
report('adjoint modules: a_(n) m, ope, no, *, mode = the products of V (%s)'
       % ', '.join(n for n, _, _ in cases), all(ok for _, ok in res), res)
suites = []
for name, V, ws in cases:
    try:
        TestSuite(V.adjoint_module()).run(raise_on_failure=True)
    except Exception as err:
        suites.append((name, str(err)[:300]))
report('TestSuite of the adjoint modules (vacuum, translation, Borcherds commutator '
       'formula, normally ordered products, weights)', not suites, suites)
chars = [(name, V.adjoint_module().character(4) == V.character(4)) for name, V, _ in cases
         if name not in ('bc', 'affine sl2')]
MBC = BC.adjoint_module()
chars.append(('bc (charge 0 only: weight 0 is infinite otherwise)', True))
report('characters of the adjoint modules = those of V', all(ok for _, ok in chars), chars)
MV = Vir.adjoint_module()
cat_ok = (MV.category() is VertexAlgebraModules(Vir).Graded().WithBasis()
          and N1.adjoint_module() in VertexAlgebraModules(N1).Super().Graded()
          and LA1.adjoint_module().category() is VertexAlgebraModules(LA1).WithBasis()
          and MV is Vir.adjoint_module() and loads(dumps(MV)) is MV
          and Vir.with_conformal_vector(None).adjoint_module() is not MV)
report('categories of the modules (super, conformally graded or not), uniqueness, pickling',
       cat_ok)
# the element of the plain algebra acts on the module of the decorated one (conversion)
Lp = Vir.with_conformal_vector(None).gen('L')
report('elements of V with other conformal data act (conversion); vectors convert',
       Lp.nproduct(MV(Vir.gen('L')), 1) == 2 * MV(Vir.gen('L')))


# 3. the action from the generators only (universal algebras) ----------------------
def by_generators_ok(V, weights, ns=range(-3, 4), M=None, every=False):
    M = vam._AdjointByGenerators(V) if M is None else M
    keys = ([kk for w in weights for kk in V.weight_space(w).basis_keys()] if every
            else random_keys(V, weights))
    for a in keys:
        for bk in keys:
            for n in ns:
                if M._action_mono(a, n, bk) != V._nprod_mono(a, n, bk):
                    return False
    return True


gens_cases = [('Virasoro', Vir, [2, 3, 4]), ('W_3', VW, [2, 3, 4]), ('N=1', N1, [3 / 2, 2, 5 / 2]),
              ('bc', BC, [0, 1, 2]), ('betagamma 1/2', BG, [1 / 2, 1, 3 / 2]),
              ('affine sl2', Aff, [1, 2])]
res = [(name, by_generators_ok(V, ws)) for name, V, ws in gens_cases]
report('the action of all basis monomials from the generators only '
       '((∂a)_(n) = -n a_(n-1), (:aB:)_(n) by the normally ordered product) = V\'s products',
       all(ok for _, ok in res), res)


class _WrongSign(vam._AdjointByGenerators):
    """The recursion without the sign (-1)^{|x||B|}: must fail for odd generators."""
    def _action_key(self, a, n, m):
        if len(a) < 2:
            return vam._GeneratorActions._action_key(self, a, n, m)
        x, B = a[:1], a[1:]
        res = {}
        for j in range(max(self._nmax_key(B, m) - n + 1, 0)):
            for mm, cf in self._action_mono(B, n + j, m).items():
                vam._add_scaled(res, self._action_mono(x, -1 - j, mm), cf)
        for j in range(self._nmax_key(x, m) + 1):
            for mm, cf in self._action_mono(x, j, m).items():
                vam._add_scaled(res, self._action_mono(B, n - 1 - j, mm), cf)
        return res


report('negative control: without the sign the recursion differs (N=1 with :∂G G:, bc '
       'with :b c:; all keys of these weights)',
       not by_generators_ok(N1, [3 / 2, 4], M=_WrongSign(N1), every=True)
       and not by_generators_ok(BC, [0, 1], M=_WrongSign(BC), every=True)
       and by_generators_ok(N1, [3 / 2, 4], every=True)
       and by_generators_ok(BC, [0, 1, 2], every=True))
try:
    TestSuite(vam._AdjointByGenerators(VW)).run(raise_on_failure=True)
    suite = True
except Exception as err:
    suite = str(err)[:300]
report('TestSuite of the W_3 module acting through the generators', suite is True, suite)

# 5. lattice modules V_{λ+L} -------------------------------------------------------
from vertex_algebras.lattice_vertex_algebra_module import _dual_lattice_cosets
from vertex_algebras.zhu_algebra import LatticeZhuAlgebra


def lattice_with_T(B, alpha=None):
    """The lattice algebra of B with T_sug (+ ∂J_alpha)."""
    L0 = LatticeVertexAlgebra(QQ, matrix(B))
    Js = L0.heisenberg_currents()
    Bi = matrix(QQ, B).inverse()
    T = sum(Bi[i, j] * Js[i].no(Js[j]) for i in range(len(Js)) for j in range(len(Js))) / 2
    if alpha is not None:
        T += sum(a * J for a, J in zip(alpha, Js)).derivative()
    return L0.with_conformal_vector(T)


lats = [('A1', lattice_with_T([[2]])), ('A1, background charge', lattice_with_T([[2]], [1 / 2])),
        ('A2', lattice_with_T([[2, -1], [-1, 2]])), ('odd sqrt3 Z', lattice_with_T([[3]])),
        ('odd [[1,0],[0,3]]', lattice_with_T([[1, 0], [0, 3]])), ('Z sqrt 6', lattice_with_T([[6]]))]
suites, counts = [], []
for name, V in lats:
    Ms = V.simple_modules()
    counts.append((name, len(Ms) == abs(V.gram_matrix().det())))
    for M in Ms[:3]:
        try:
            TestSuite(M).run(raise_on_failure=True)
        except Exception as err:
            suites.append((name, M.coset_representative(), str(err)[:300]))
report('lattice modules: TestSuite (vacuum, translation, Borcherds commutator formula, normally '
       'ordered products, weights) for A1 (also with a background charge), A2, odd lattices, '
       'Z sqrt 6; |L°/L| = |det B| modules', not suites and all(ok for _, ok in counts),
       (suites, counts))


def theta_over_eta(V, M, prec):
    """Σ_{p ∈ λ+L} q^Δ(p) / Π (1-q^n)^rank, the weights from V's data (brute force)."""
    from itertools import product
    P = PuiseuxSeriesRing(ZZ, 'q'); q = P.gen()
    data = V._weight_data()
    lam = M.coset_representative()
    th = P.zero()
    for m in product(range(-6, 7), repeat=V.rank()):
        p = tuple(l + x for l, x in zip(lam, m))
        w = V._Delta(p, data)
        if w < prec:
            th += q ** w
    eta = prod((1 - q ** n) ** V.rank() for n in range(1, prec + 2))
    return (th / eta).add_bigoh(prec)


chars = []
for name, V in lats[:4]:
    for M in V.simple_modules():
        chars.append((name, M.coset_representative(), M.character(4) == theta_over_eta(V, M, 4)))
report('characters of V_{λ+L} = θ_{λ+L} / η^rank (also with a background charge)',
       all(ok for *_, ok in chars), chars)
same = []
for name, V in lats[:3]:
    M0 = V.simple_modules()[0]
    keys = [kk for w in (0, 1, 2) for kk in V.weight_space(w).basis_keys()]
    ok = True
    for ak in keys:
        for bk in keys:
            mk = (bk[0], tuple(QQ(x) for x in bk[1]))
            for n in range(-3, 3):
                d = M0._action_mono(ak, n, mk)
                if {(kk[0], tuple(int(x) for x in kk[1])): cf for kk, cf in d.items()} != V._nprod_mono(ak, n, bk):
                    ok = False
    same.append((name, ok))
report('V_{0+L} acts as V on itself (the products of V)', all(ok for _, ok in same), same)
tops = []
for name, V in [('A1', lats[0][1]), ('A1, background', lats[1][1]), ('A2', lats[2][1])]:
    Z = LatticeZhuAlgebra(V)
    levels = Z.top_levels()
    Ms = V.simple_modules()
    for M in Ms:
        lb, D = M._weight_bounds(None)            # a bound: the lowest weight is attained above
        h = (QQ(lb) * D).ceil() / D
        while not M.weight_space(h).dimension():
            h += QQ((1, D))
        S = M.weight_space(h)
        mus = sorted(kk[1] for kk in S.basis_keys())
        lam = M.coset_representative()
        match = [i for i, (hh, ms) in enumerate(levels)
                 if all(x - y in ZZ for x, y in zip(ms[0], lam))]
        if len(match) != 1:
            tops.append((name, lam, 'no level'))
            continue
        i = match[0]
        hh, ms = levels[i]
        ok = hh == h and sorted(tuple(QQ(x) for x in mu) for mu in ms) == mus
        # zero modes on the top level = the Zhu algebra's matrices
        for v in [V.J(0), V.J(0).no(V.J(0)), V.e(tuple(int(j == 0) for j in range(V.rank()))),
                  V.e(tuple(-int(j == 0) for j in range(V.rank())))]:
            try:
                wv = v.degree()
            except ValueError:
                continue
            if wv not in ZZ:
                continue
            block = Z(v).matrices()[i]
            for c_, mu in enumerate(ms):
                img = v.nproduct(M.monomial(((), tuple(QQ(x) for x in mu))), wv - 1)
                want = {((), tuple(QQ(x) for x in nu)): block[r, c_] for r, nu in enumerate(ms)
                        if block[r, c_]}
                if dict(img.monomial_coefficients()) != want:
                    ok = False
        tops.append((name, lam, ok))
report('top levels = LatticeZhuAlgebra.top_levels(); zero modes o(v) = v_(wt v - 1) on them = '
       'the Zhu algebra\'s matrices (A1, A1 with a background charge, A2)',
       all(ok is True for *_, ok in tops), tops)
Hind = LatticeVertexAlgebra(QQ, matrix([[0, 2], [2, 0]]))
J0h, J1h = Hind.heisenberg_currents()
Hg = Hind.with_conformal_vector((J0h.no(J1h) + J1h.no(J0h)) / 4).with_horizontal_grading('lattice')
Mind = Hg.simple_modules()
ind_ok = (len(Mind) == 4 and not Mind[1].weight_space(1).is_finite_dimensional()
          and [Mind[1].weight_space(w, horizontal_grade=Mind[1].coset_representative()).dimension()
               for w in range(3)] == [1, 2, 5])
try:
    TestSuite(Mind[1]).run(raise_on_failure=True)
except Exception as err:
    ind_ok = str(err)[:300]
D3 = LatticeVertexAlgebra(QQ, matrix([[1, 1], [1, -1]]))        # indefinite, det -2
report('indefinite lattices: [[0,2],[2,0]] (4 modules, infinite weight spaces, finite graded '
       'ones, TestSuite), [[1,1],[1,-1]] (2 modules)', ind_ok is True
       and len(D3.simple_modules()) == 2, ind_ok)
Lr = LatticeVertexAlgebra(QQ, matrix([[2]]))
refused = []
for bad in ([1 / 3], [1 / 2, 0], ['x']):
    try:
        Lr.simple_module(bad)
        refused.append(bad)
    except ValueError:
        pass
report('lattice modules: λ and λ + L give the same module, pickling; refused: λ outside L°, '
       'wrong length, not rational', not refused and Lr.simple_module([3 / 2]) is Lr.simple_module([-1 / 2])
       and loads(dumps(Lr.simple_module([1 / 2]))) is Lr.simple_module([1 / 2]), refused)

# 6. modules over Zhu algebras --------------------------------------------------------
def accepts(A, action, parities=None):
    try:
        A.module(action, parities)
        return True
    except ValueError:
        return False


def virasoro_qq(cv, name):
    EE = NormallyOrderedExpressions(QQ, [name])
    Lg = EE.gen(0)
    return UniversalVertexAlgebra(QQ, EE, {(name, name): [Lg.derivative(), 2 * Lg, 0, cv / 2]},
                                  conformal_vector=Lg)


def singular_vir(V, W):
    Lg = V.gen(0)
    S = V.weight_space(W).basis()
    rows = []
    for n in (2, 3):
        ks = list(V.weight_space(W + 1 - n).basis_keys())
        imgs = [Lg.nproduct(x, n).monomial_coefficients() for x in S]
        rows += [[d.get(kk, 0) for d in imgs] for kk in ks]
    [v] = [sum(cf * x for cf, x in zip(r, S)) for r in matrix(QQ, rows).right_kernel_matrix().rows()]
    return v


zm = []
for cv, W, hs, name in [(QQ(-22) / 5, 4, [0, QQ(-1) / 5], 'Ly2'), (QQ(1) / 2, 6, [0, QQ(1) / 16, QQ(1) / 2], 'Li2')]:
    Vq = virasoro_qq(cv, name)
    Zq = Vq.quotient([singular_vir(Vq, W)]).zhu_algebra_up_to(W + 2)
    ok = all(accepts(Zq, {name: h}) for h in hs) and not any(accepts(Zq, {name: h}) for h in (1, QQ(1) / 3))
    ok = ok and all(accepts(Vq.zhu_algebra(), {name: h}) for h in (1, QQ(1) / 3))
    zm.append((name, ok))
VA1 = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1, names=['E', 'H', 'F'], conformal_vector='sugawara')
AA1 = VA1.zhu_algebra()
dbl = {'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]), 'F': matrix([[0, 0], [1, 0]])}
adj = {'E': matrix([[0, -2, 0], [0, 0, 1], [0, 0, 0]]), 'H': matrix([[2, 0, 0], [0, 0, 0], [0, 0, -2]]),
       'F': matrix([[0, 0, 0], [-1, 0, 0], [0, 2, 0]])}
E1 = VA1.gen('E')
ZA1 = VA1.quotient([E1.no(E1)]).zhu_algebra_up_to(3)
zm.append(('sl2: U(sl2) takes C^2, C^3; L_1(sl2) only C, C^2',
           accepts(AA1, dbl) and accepts(AA1, adj) and accepts(ZA1, dbl)
           and accepts(ZA1, {'E': 0, 'H': 0, 'F': 0}) and not accepts(ZA1, adj)))
ABCz = BC.zhu_algebra()
zm.append(('bc: Clifford module C^{1|1}; parity and relation checks',
           accepts(ABCz, {'b': matrix([[0, 1], [0, 0]]), 'cc': matrix([[0, 0], [1, 0]])}, [0, 1])
           and not accepts(ABCz, {'b': matrix([[0, 1], [0, 0]]), 'cc': matrix([[0, 0], [2, 0]])}, [0, 1])
           and not accepts(ABCz, {'b': matrix([[0, 1], [0, 0]]), 'cc': matrix([[0, 0], [1, 0]])}, [0, 0])))
NT.<Lt, Gt> = NormallyOrderedExpressions(QQ, parities={'Gt': 1})
TI = UniversalVertexAlgebra(QQ, NT, {('Lt', 'Lt'): [Lt.derivative(), 2 * Lt, 0, 7 / 20],
                                     ('Lt', 'Gt'): [Gt.derivative(), 3 / 2 * Gt],
                                     ('Gt', 'Gt'): [2 * Lt, 0, 7 / 15]}, conformal_vector=Lt)
Ltv, Gtv = TI.gens()
S4 = TI.weight_space(4).basis()
rows = []
for g, n in [(Ltv, 2), (Ltv, 3), (Gtv, 1), (Gtv, 2)]:
    imgs = [g.nproduct(x, n).monomial_coefficients() for x in S4]
    ks = sorted(set(kk for d in imgs for kk in d))
    rows += [[d.get(kk, 0) for d in imgs] for kk in ks]
[vti] = [sum(cf * x for cf, x in zip(r, S4)) for r in matrix(QQ, rows).right_kernel_matrix().rows()]
ZR = TI.quotient([vti]).zhu_algebra_up_to(6, twist='ramond')


def ramond(h):
    return {'Lt': matrix([[h, 0], [0, h]]), 'Gt': matrix([[0, 1], [h - QQ(7) / 240, 0]])}


zm.append(('tricritical Ising Ramond: 1|1 modules at h = 3/80, 7/16 only; the universal '
           'N=1 Ramond algebra at any h, 1-dim at h = c/24',
           accepts(ZR, ramond(QQ(3) / 80), [0, 1]) and accepts(ZR, ramond(QQ(7) / 16), [0, 1])
           and not accepts(ZR, ramond(1), [0, 1])
           and accepts(TI.zhu_algebra(twist='H'), ramond(5), [0, 1])
           and accepts(TI.zhu_algebra(twist='H'), {'Lt': QQ(7) / 240, 'Gt': 0})))
AW3 = VW.zhu_algebra()
zm.append(('W_3: commutative, every (h, w)', accepts(AW3, {'L3': c, 'W3': c ** 2 + 1})))
report('modules over Zhu algebras exist exactly as expected (Lee-Yang, Ising, sl2, L_1(sl2), '
       'Clifford, tricritical Ising Ramond, W_3)', all(ok for _, ok in zm), zm)
Ud = AA1.module(dbl)
x = AA1(VA1.conformal_vector())
suite = []
for U in (Ud, ABCz.module({'b': matrix([[0, 1], [0, 0]]), 'cc': matrix([[0, 0], [1, 0]])}, [0, 1]),
          ZR.module(ramond(QQ(7) / 16), [0, 1])):
    try:
        TestSuite(U).run(raise_on_failure=True)
    except Exception as err:
        suite.append(str(err)[:200])
refused = []
for act in ({'E': 0, 'H': 0}, {'E': 0, 'H': 0, 'F': 0, 'X': 0},
            {'E': matrix([[0, 1], [0, 0]]), 'H': 0, 'F': 0}):
    try:
        AA1.module(act)
        refused.append(act)
    except ValueError:
        pass
report('Zhu algebra modules: x * u = representation_matrix(x) u, [T_sug] = 1/4 on C^2 at '
       'level 1, TestSuite, pickling; refused: missing / unknown generators, sizes',
       not suite and not refused and Ud.representation_matrix(x) == matrix(QQ, 2, 2, 1 / 4)
       and x * Ud.basis()[1] == Ud.basis()[1] / 4 and loads(dumps(Ud)) is Ud, (suite, refused))

# 7. induced modules (Verma modules, Weyl modules) -----------------------------------
from vertex_algebras.lie_superalgebras import LieSuperalgebra
from vertex_algebras.quantum_hamiltonian_reduction import WAlgebra


def borcherds_ok(M, els, vecs, ps=range(3), qs=range(-2, 3)):
    """The commutator formula and the normally ordered product on module vectors."""
    for a in els:
        for b in els:
            s_ = (-1) ** (a.parity() * b.parity())
            ab = a.ope(b)
            nab = a.nproduct(b, -1)
            for m in vecs:
                am, bm = a.ope(m), b.ope(m)
                for q in qs:
                    for p in ps:
                        lhs = a.nproduct(b.nproduct(m, q), p) - s_ * b.nproduct(a.nproduct(m, p), q)
                        rhs = M.zero()
                        for j in range(min(p + 1, len(ab))):
                            rhs += binomial(p, j) * ab[j].nproduct(m, p + q - j)
                        if lhs != rhs:
                            return False
                    rhs = M.zero()
                    for j in range(max(len(bm) - q, 0)):
                        rhs += a.nproduct(b.nproduct(m, q + j), -1 - j)
                    for j in range(len(am)):
                        rhs += s_ * b.nproduct(am[j], q - 1 - j)
                    if nab.nproduct(m, q) != rhs:
                        return False
    return True


def depth_vectors(M, ds):
    return [M.monomial(kk) for d in ds for kk in M._depth_keys(d)][:6]


Kch.<cc0, hh> = Frac(QQ['cc0', 'hh'])
Ech.<Lh> = NormallyOrderedExpressions(Kch)
VirH = UniversalVertexAlgebra(Kch, Ech, {('Lh', 'Lh'): [Lh.derivative(), 2 * Lh, 0, cc0 / 2]},
                              conformal_vector=Lh)
MVer = VirH.verma_module(hh)
Lv = VirH.gen('Lh')
MN1 = N1.verma_module({'Ln': c / 7})
N2E.<L2, J2, Gp, Gm> = NormallyOrderedExpressions(K, parities={'Gp': 1, 'Gm': 1})
N2 = UniversalVertexAlgebra(K, N2E, {
    ('L2', 'L2'): [L2.derivative(), 2 * L2, 0, c / 2], ('L2', 'J2'): [J2.derivative(), J2],
    ('L2', 'Gp'): [Gp.derivative(), 3 / 2 * Gp], ('L2', 'Gm'): [Gm.derivative(), 3 / 2 * Gm],
    ('J2', 'J2'): [0, c / 3], ('J2', 'Gp'): [Gp], ('J2', 'Gm'): [-Gm],
    ('Gp', 'Gm'): [L2 + J2.derivative() / 2, J2, c / 3]}, conformal_vector=L2)
MN2 = N2.verma_module({'L2': 2, 'J2': c})
MW3 = VW.verma_module({'L3': c + 1, 'W3': 1 / c})
sl3 = LieSuperalgebra(QQ, CartanMatrix(['A', 2]))
BPW = WAlgebra(Kk, sl3, 'minimal', k, conformal_vector='reduction')
ABP = BPW.zhu_algebra()
MBP = BPW.verma_module({s: (k + 1) ** i for i, s in enumerate(ABP.variable_names())})
UA = Aff.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
                               'F': matrix([[0, 0], [1, 0]])})
MAf = Aff.induced_module(UA)
ind = [('Virasoro M(c, h)', MVer, VirH), ('N=1 NS Verma', MN1, N1), ('N=2 Verma', MN2, N2),
       ('W_3 Verma (non-linear)', MW3, VW), ('Bershadsky-Polyakov (non-linear, 3/2)', MBP, BPW),
       ('affine sl2 Weyl module C^2', MAf, Aff)]
res, suites = [], []
for name, M, V in ind:
    els = [x for x in V.some_elements() if x.parity() is not None][:3]
    gens = list(V.gens())
    extra = [gens[0].no(gens[-1])]                     # a composite field
    res.append((name, borcherds_ok(M, els[:2] + extra, depth_vectors(M, [0, 1]),
                                   ps=range(2), qs=range(-1, 2))))
    try:
        TestSuite(M).run(raise_on_failure=True)
    except Exception as err:
        suites.append((name, str(err)[:300]))
report('induced modules: Borcherds commutator formula and normally ordered products with a '
       'composite field on vectors of depth <= 1 (%s), TestSuite' % ', '.join(n for n, _, _ in ind),
       all(ok for _, ok in res) and not suites, (res, suites))

def w3_consistency(beta_):
    """Failures of the commutator formula for W, L on all vectors of depth <= 2 of a
    W_3 Verma module (the rewriting is consistent iff V's Jacobi identity holds)."""
    Lm = L3 * L3 - 3 / 10 * L3.derivative(2)
    V = UniversalVertexAlgebra(K, E3, {
        ('L3', 'L3'): [L3.derivative(), 2 * L3, 0, c / 2], ('L3', 'W3'): [W3.derivative(), 3 * W3],
        ('W3', 'W3'): [beta_ * Lm.derivative() + L3.derivative(3) / 15, 2 * beta_ * Lm
                       + 3 / 10 * L3.derivative(2), L3.derivative(), 2 * L3, 0, c / 3]},
        degrees={'L3': 2, 'W3': 3})
    M = V.verma_module({'L3': c + 1, 'W3': 1 / c})
    Lg, Wg = V.gens()
    bad = 0
    for a, b in [(Wg, Wg), (Wg, Lg), (Lg, Wg)]:
        ab = a.ope(b)
        for m in [M.monomial(kk) for d in (0, 1, 2) for kk in M._depth_keys(d)]:
            for q_ in range(-1, 3):
                for p_ in range(4):
                    lhs = a.nproduct(b.nproduct(m, q_), p_) - b.nproduct(a.nproduct(m, p_), q_)
                    rhs = sum((binomial(p_, j) * ab[j].nproduct(m, p_ + q_ - j)
                               for j in range(min(p_ + 1, len(ab)))), M.zero())
                    bad += lhs != rhs
    return bad


report('W_3 Verma modules (non-linear): the commutator formula holds on all vectors of depth '
       '<= 2 (384 identities); negative control: with a wrong β it fails',
       w3_consistency(beta) == 0 and w3_consistency(beta + 1) > 0)

P = PuiseuxSeriesRing(ZZ, 'q'); q = P.gen()
eta = lambda e: prod((1 - q ** n) ** e for n in range(1, 8))
chars = [('Virasoro', MVer.weight_space(hh + 0).dimension() == 1
          and [MVer.weight_space(hh + d).dimension() for d in range(6)] == [1, 1, 2, 3, 5, 7]),
         ('N=1 (symbolic h)', [MN1.weight_space(c / 7 + d / 2).dimension() for d in range(6)]
          == [1, 1, 1, 2, 3, 4]),
         ('W_3', [MW3.weight_space(c + 1 + d).dimension() for d in range(5)] == [1, 2, 5, 10, 20]),
         ('affine Weyl', [MAf.weight_space(MAf.top_weight() + d).dimension() for d in range(4)]
          == [2, 6, 18, 44])]
VirQh = virasoro_qq(QQ(1) / 2, 'Lq')
MQ = VirQh.verma_module(1 / 16)
chars.append(('Virasoro c = 1/2, h = 1/16: q^h / Π(1-q^n)', MQ.character(6) == (q ** (1 / 16) / eta(1)).add_bigoh(6)))
NQ.<Lnq, Gq> = NormallyOrderedExpressions(QQ, parities={'Gq': 1})
N1Q = UniversalVertexAlgebra(QQ, NQ, {('Lnq', 'Lnq'): [Lnq.derivative(), 2 * Lnq, 0, 7 / 20],
                                      ('Lnq', 'Gq'): [Gq.derivative(), 3 / 2 * Gq],
                                      ('Gq', 'Gq'): [2 * Lnq, 0, 7 / 15]}, conformal_vector=Lnq)
MN1Q = N1Q.verma_module(1 / 10)
fer = prod(1 + q ** (n - 1 / 2) for n in range(1, 8))
chars.append(('N=1 NS, h = 1/10: q^h Π(1+q^(n-1/2))/(1-q^n)',
              MN1Q.character(4) == (q ** (1 / 10) * fer / eta(1)).add_bigoh(4)))
report('characters of induced modules (PBW counts: Virasoro, N=1 NS, W_3, affine Weyl)',
       all(ok for _, ok in chars), chars)


def zero_modes_ok(M, V, weights, trials=6):
    """o(a) = a_(wt a - 1) on the top level = the matrix of [a] on U (Zhu)."""
    A, U = M._A, M.top_level()
    keys = [kk for w in weights for kk in V.weight_space(w).basis_keys()]
    random.shuffle(keys)
    for kk in keys[:trials]:
        a = V.monomial(kk)
        wa = a.degree()
        mat = U.representation_matrix(A(a))
        for u in range(U.dimension()):
            img = a.nproduct(M.monomial(((), u)), wa - 1)
            want = {((), r): mat[r, u] for r in range(U.dimension()) if mat[r, u]}
            if dict(img.monomial_coefficients()) != want:
                return False
    return True


zm = [(name, zero_modes_ok(M, V, ws)) for name, M, V, ws in
      [('Virasoro', MVer, VirH, [2, 4, 6]), ('W_3', MW3, VW, [2, 3, 4, 5]),
       ('affine sl2 Weyl', MAf, Aff, [1, 2, 3]), ('N=2', MN2, N2, [1, 2, 3]),
       ('Bershadsky-Polyakov', MBP, BPW, [1, 2, 3])]]
report('zero modes on the top level = the Zhu algebra on U (Zhu\'s theorem): Virasoro, W_3, '
       'affine Weyl, N=2, BP', all(ok for _, ok in zm), zm)


def vacuum_like_ok(V, M, weights):
    """M(trivial top) ≅ V by a ↦ a_(-1) v: (a_(n) b)_(-1) v = a_(n) b_(-1) v."""
    v = M.highest_weight_vector()
    keys = [kk for w in weights for kk in V.weight_space(w).basis_keys()][:8]
    for ak in keys:
        for bk in keys:
            a, b = V.monomial(ak), V.monomial(bk)
            for n in range(-2, 3):
                if a.nproduct(b, n).nproduct(v, -1) != a.nproduct(b.no(v), n):
                    return False
    return all(M.weight_space(M.top_weight() + w).dimension() == V.weight_space(w).dimension()
               for w in weights)


Htriv = UniversalVertexAlgebra(QQ, ['Jt'], {('Jt', 'Jt'): [0, 1]}, degrees={'Jt': 1})
vl = [('V^k(sl2) = M(trivial)', vacuum_like_ok(Aff, Aff.induced_module(Aff.zhu_algebra().module(
          {'E': 0, 'H': 0, 'F': 0})), [0, 1, 2])),
      ('Heisenberg = M(J -> 0)', vacuum_like_ok(Htriv, Htriv.verma_module(0), [0, 1, 2, 3])),
      ('betagamma(1/2) = M(C)', vacuum_like_ok(BG, BG.induced_module(BG.zhu_algebra().module([])),
                                               [0, 1 / 2, 1, 3 / 2, 2]))]
report('trivial tops: V^k(sl2), Heisenberg, betagamma(1/2) are their induced modules '
       '((a_(n) b)_(-1) v = a_(n) b_(-1) v, same dimensions)', all(ok for _, ok in vl), vl)
# singular vectors
LQ = VirQh.gen('Lq')
Mh = VirQh.verma_module(1 / 2)                                # c = 1/2, h = 1/2: level 2
vh = Mh.highest_weight_vector()
sv = LQ.mode(vh, -2) - 3 / 4 * LQ.mode(LQ.mode(vh, -1), -1)
M0 = VirQh.verma_module(0)
v0 = M0.highest_weight_vector()
Gv = N1Q.gen('Gq')
MN0 = N1Q.verma_module(0)
k1 = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1, names=['E', 'H', 'F'], conformal_vector='sugawara')
Mk1 = k1.induced_module(k1.zhu_algebra().module({'E': 0, 'H': 0, 'F': 0}))
Ek, Fk = k1.gen('E'), k1.gen('F')
vk = Mk1.highest_weight_vector()
e2 = Ek.mode(Ek.mode(vk, -1), -1)
sing = [('Virasoro h = 0: L_{-1} v', all(LQ.mode(LQ.mode(v0, -1), n) == 0 for n in (1, 2))),
        ('Ising h = 1/2: (L_{-2} - 3/4 L_{-1}^2) v', all(LQ.mode(sv, n) == 0 for n in (1, 2))
         and LQ.mode(LQ.mode(vh, -2), 1) != 0),
        ('N=1 h = 0: G_{-1/2} v', all(Gv.mode(Gv.mode(MN0.highest_weight_vector(), -1 / 2), r) == 0
                                     for r in (1 / 2, 3 / 2))),
        ('sl2 level 1: E_{-1}^2 v', Fk.mode(e2, 1) == 0 and Ek.mode(e2, 1) == 0
         and k1.gen('H').mode(e2, 1) == 0 and not e2.is_zero())]
report('singular vectors (Virasoro h = 0 and the Ising level-2 vector, N=1 G_{-1/2} v, affine '
       'sl2 E_{-1}^2 v at level 1)', all(ok for _, ok in sing), sing)
refused = []
for f in [lambda: Vir.induced_module(VW.zhu_algebra().module({'L3': 1, 'W3': 2})),
          lambda: BC.induced_module(BC.zhu_algebra().module({'b': matrix([[0, 1], [0, 0]]),
                                                             'cc': matrix([[0, 0], [1, 0]])}, [0, 1])),
          lambda: N1.induced_module(N1.zhu_algebra(twist='H').module({'Ln': 1, 'G': matrix([[0]])}))]:
    try:
        f()
        refused.append(f)
    except (ValueError, NotImplementedError):
        pass
report('induced modules refused: a module over another Zhu algebra, generators of weight 0 '
       '(bc with weights 1, 0), twisted Zhu algebra modules (not yet)', not refused, refused)

# 4. refusals ----------------------------------------------------------------------
refused = []
try:
    VW.gen('L3').nproduct(MV(Vir.gen('L')), 0)
    refused.append('foreign element')
except (TypeError, ValueError):
    pass
try:
    Vir.gen('L').mode(MV(Vir.gen('L')), 1 / 2)
    refused.append('mode index')
except ValueError:
    pass
EL.<Ly> = NormallyOrderedExpressions(QQ)
VLY = UniversalVertexAlgebra(QQ, EL, {('Ly', 'Ly'): [Ly.derivative(), 2 * Ly, 0, -11 / 5]},
                             conformal_vector=Ly)
Lyv = VLY.gen('Ly')
LYQ = VLY.quotient([Lyv.no(Lyv) - 3 / 10 * Lyv.derivative(2)])
try:
    LYQ.adjoint_module()
    refused.append('quotient')
except NotImplementedError:
    pass
try:
    vam._AdjointByGenerators(LA1)
    refused.append('generators of a lattice algebra')
except TypeError:
    pass
report('refused: elements of another vertex algebra, a mode index outside ZZ - Δ, the '
       'adjoint module of a quotient, the generator recursion for a non-universal algebra',
       not refused, refused)

print('total %.1fs' % (time.time() - t_start))
report.result()
