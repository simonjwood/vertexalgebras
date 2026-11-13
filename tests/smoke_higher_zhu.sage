# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Higher level Zhu algebras A_n(V) (higher_zhu_algebra.py): the mode model of
# universal algebras against Dong-Li-Mason's definition ([O_n] = 0, [a *_n b]
# = [a][b]) at levels up to 3, with a negative control; quotients: Lee-Yang
# A_0..A_3 = 2, 3, 5, 7 (Σ dim End of the degree spaces of its simple
# modules); the triplet algebra W(2) at c = -2 from its presentation (no
# modules): A_1 of dimension 24, associative, [T] with minimal polynomial
# x^2 (x-1)^2 (x+1/8)(x-3/8)(x-7/8)(x-11/8)(x-2), products beyond the degree
# bound (regular representation) = those of a larger bound; induction from degree n:
# Virasoro (level 1: Verma modules, shifted Verma modules, the Jordan
# extension; level 2: the degree-2 space of a Verma module gives it back),
# W(2): the [T] = 1 block of A_1 (two copies of the projective
# indecomposable, a homogeneous basis) induces 2 P(1), graded by charge
# (P(1): 2, 4, 6, 12 by degree), with the Loewy layers W(1) | 2 W(0) | W(1)
# in the window; End(P(1)) = 2 (Hom out of \bar M_n(U)); U stays in degree 1; refusals.
# Run from the repository root:  sage tests/smoke_higher_zhu.sage  -> RESULT: PASS
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
import vertex_algebras as va
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from axioms import Reporter

report = Reporter()
t_start = time.time()


# 1. the definition ----------------------------------------------------------------------------
def res(a, b, k, p):
    """Res_z Y(a, z) b (1+z)^p z^{-k}."""
    N = len(a.ope(b))
    return sum((binomial(p, j) * a.nproduct(b, j - k) for j in range(p + 1) if j - k < N),
               b.parent().zero())


def definition(V, n, weights, star_level=None):
    A = V.zhu_algebra(level=n)
    sl = n if star_level is None else star_level
    B = [V.monomial(m) for w in weights for m in V.weight_space(w).basis_keys()]
    bad = 0
    for a in B:
        h = ZZ(a.degree())
        if A(a.derivative() + h * a):
            bad += 1
        for b in B:
            circ = res(a, b, 2 * n + 2, h + n)
            star = sum(((-1)^m * binomial(m + sl, sl) * res(a, b, sl + m + 1, h + sl)
                        for m in range(sl + 1)), V.zero())
            if A(circ) or A(star) != A(a) * A(b):
                bad += 1
    return bad


t = time.time()
K.<c> = Frac(QQ['c'])
Vir = va.Virasoro(K, c, conformal_vector='L')
report('Virasoro over Q(c), levels 1, 2, 3, weights <= 7',
       all(definition(Vir, n, range(8)) == 0 for n in (1, 2, 3)), '%.1fs' % (time.time() - t))
t = time.time()
report('W_3 levels 1, 2; affine sl_2 level 2; symplectic fermions level 2',
       definition(va.W3(QQ, 2, conformal_vector='L'), 1, [0, 2, 3, 4, 5]) == 0
       and definition(va.W3(QQ, 2, conformal_vector='L'), 2, [0, 2, 3, 4]) == 0
       and definition(va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'],
                                conformal_vector='sugawara'), 2, [0, 1, 2, 3]) == 0
       and definition(UniversalVertexAlgebra(QQ, ['chp', 'chm'], {('chp', 'chm'): [0, 1]},
                                             parities={'chp': 1, 'chm': 1},
                                             degrees={'chp': 1, 'chm': 1}), 2, [0, 1, 2, 3]) == 0,
       '%.1fs' % (time.time() - t))
report('negative control: the product *_1 is not that of A_2',
       definition(Vir, 2, range(5), star_level=1) > 0)

# 2. Lee-Yang ----------------------------------------------------------------------------------
t = time.time()
VL = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = VL.gen('L')
LY = VL / [L.no(L) - 3/10 * L.derivative(2)]
dims = [LY.zhu_algebra_up_to(6).dimension()] + [LY.zhu_algebra_up_to(8, level=n).dimension()
                                                 for n in (1, 2, 3)]
report('Lee-Yang A_0..A_3', dims == [2, 3, 5, 7], '%s %.1fs' % (dims, time.time() - t))

# 3. W(2) ---------------------------------------------------------------------------------------
t = time.time()
Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm * cp,
                            horizontal_grading={'cp': 1, 'cm': -1})
a, b = SF.gens()
S2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
                    a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
                   names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient',
                   conformal_vector='ambient')
P = S2.presentation()
Z = P.zhu_algebra_up_to(8, level=1, degree_bound=18)
B = list(Z.basis())
n = len(B)
report('W(2): dim A_1 = 24, no commutator defect', n == 24 and Z._ndefects == 0,
       '%.1fs' % (time.time() - t))
t = time.time()
set_random_seed(1)
ok = True
for _ in range(60):
    x, y, z = [B[randint(0, n - 1)] for _ in range(3)]
    ok = ok and (x * y) * z == x * (y * z)
pos = {k: i for i, k in enumerate(x.leading_support() for x in B)}


def vec(e):
    v = vector(QQ, n)
    for k, cc in e.monomial_coefficients().items():
        v[pos[k]] = cc
    return v


T = Z(P.gen('T'))
powers = [vec(Z.one())]
p = Z.one()
while True:
    p = p * T
    powers.append(vec(p))
    rel = matrix(QQ, powers).left_kernel()
    if rel.dimension():
        break
x_ = polygen(QQ)
mp = sum(cc * x_^i for i, cc in enumerate(rel.basis()[0]))
mp = mp / mp.leading_coefficient()
expected = x_^2 * (x_ - 1)^2 * (x_ + 1/8) * (x_ - 3/8) * (x_ - 7/8) * (x_ - 11/8) * (x_ - 2)
report('W(2): A_1 associative (random triples), [T] minimal polynomial', ok and mp == expected,
       '%.1fs' % (time.time() - t))

# products beyond the degree bound (regular representation) = direct products
t = time.time()
Z16 = P.zhu_algebra_up_to(8, level=1, degree_bound=16)
ws = [x.leading_support() for x in B]
ok = sorted(ws) == sorted(Z16._standard_up_to(16)) and all(
    Z16.product_on_basis(x, y).monomial_coefficients()
    == Z.product_on_basis(x, y).monomial_coefficients() for x in ws for y in ws)
report('W(2): products beyond the degree bound 16 = those at 18', ok,
       '%d beyond, %.1fs' % (sum(1 for x in ws for y in ws if Z16._filt(x) + Z16._filt(y) > 16),
                             time.time() - t))

# the [T] = 1 block: e = p([T]), p = 1 mod (x-1)^2, 0 mod the rest
t = time.time()
other = expected // (x_ - 1)^2
g, s_, t_ = xgcd((x_ - 1)^2, other)
pe = (t_ * other) % expected                   # 1 mod (x-1)^2, 0 mod the rest
e = sum((cc * T^i for i, cc in enumerate(pe.list())), Z.zero())
G, gkey = Z._grade_data()
grades = sorted({gkey(w) for w in ws})
rows, ugrades = [], []
for g in grades:                               # a homogeneous basis of the block
    sp = matrix(QQ, [vec(bb * e) for bb, w in zip(B, ws) if gkey(w) == g]).row_space()
    rows += sp.basis_matrix().rows()
    ugrades += [g] * sp.dimension()
block = matrix(QQ, rows)
elems = [sum((r[i] * B[i] for i in range(n)), Z.zero()) for r in block.rows()]


def left(w):
    x = Z._from_dict(Z._nf({w: 1}), coerce=False, remove_zeros=True)
    return matrix(QQ, [block.solve_left(vec(x * y)) for y in elems]).transpose()


U = Z.module({w: left(w) for w in Z._generator_words()}, horizontal_grades=ugrades)
M = P.induced_module_up_to(U, 8, 3)
h = M._cover_module.top_weight()
dims = [M.weight_space(h + d).dimension() for d in range(4)]
gdims = [[M.weight_space(h + d, horizontal_grade=(g,)).dimension() for g in range(-7, 8)]
         for d in range(4)]
report('W(2): the [T] = 1 block (dim 8) induces 2 P(1), graded by charge', block.nrows() == 8
       and dims == [4, 8, 12, 24] and [sum(r) for r in gdims] == dims
       and all(r == r[::-1] for r in gdims), '%s %s %.1fs' % (dims, gdims, time.time() - t))

# the structure of A_1 by the library: seven simple modules, the projective
# indecomposables (graded by charge), Σ dim S dim P = dim A_1; the projective
# module of W(1)'s top induces one P(1), graded, symmetric about charge -1
# (the grades of A e are those of the words of A: 0 and -2 on the top here;
# a module's grading is fixed up to a shift)
t = time.time()
Ps, Ss = Z.projective_indecomposable_modules(), Z.simple_modules()
TT = P.gen('T')
tw = [Sm.representation_matrix(TT)[0, 0] for Sm in Ss]
P1 = [Mp for Mp, w in zip(Ps, tw) if w == 1][0]
M1 = P.induced_module_up_to(P1, 8, 3)
h1 = M1._cover_module.top_weight()
d1 = [M1.weight_space(h1 + d).dimension() for d in range(4)]
g1 = [[M1.weight_space(h1 + d, horizontal_grade=(g,)).dimension() for g in range(-5, 6)]
      for d in range(4)]
report('W(2): A_1 by structure theory: 7 simples, PIM of W(1) induces P(1) (graded)',
       sorted(tw) == [-1/8, 0, 3/8, 7/8, 1, 11/8, 2]
       and sum(Sm.dimension() * Mp.dimension() for Mp, Sm in zip(Ps, Ss)) == 24
       and Z.cartan_invariants_matrix().diagonal().count(2) == 2
       and P1.dimension() == 4 and d1 == [2, 4, 6, 12] and [sum(r) for r in g1] == d1
       and all(r[4 + j] == r[4 - j] for r in g1 for j in range(5)),
       '%s %.1fs' % (d1, time.time() - t))

# Hom out of \bar M_1(U) (exact, adjunction): End(P(1)) = identity + a
# nilpotent map (W(1) twice in P(1)); End(2 P(1)) = Mat_2 of it
t = time.time()
E1, E2 = End(M1), End(M)
report('W(2): End(P(1)) = 2, End(2 P(1)) = 8, exact',
       E1.dimension() == 2 and E2.dimension() == 8 and E1.checked_up_to_depth() is None
       and E2.checked_up_to_depth() is None, '%.1fs' % (time.time() - t))

# Loewy layers W(1) | 2 W(0) | W(1) of P(1) (twice), in the window: χ_0 = [1, 0, 1, 4],
# χ_1 = [0, 2, 2, 2] (even / odd parts of the symplectic fermion vacuum module)
t = time.time()
N0 = M.submodule(list(M.weight_space(h).basis()))             # the radical: 2 χ_0 + χ_1
img = [x for x in (P.gen('T').nproduct(y, 1) - y for y in M.weight_space(h + 1).basis()) if x]
Soc = M.submodule(img)                                        # the socle (L_0 - 1) M_1: χ_1
nd = [N0.weight_space(h + d).dimension() for d in range(4)]
sd = [Soc.weight_space(h + d).dimension() for d in range(4)]
report('W(2): Loewy layers of 2 P(1): <M_0> = 2(2χ_0 + χ_1), <(L_0 - 1) M_1> = 2χ_1 inside it',
       nd == [4, 4, 8, 20] and sd == [0, 4, 4, 4] and all(N0.contains(x) for x in img),
       '%s %s %.1fs' % (nd, sd, time.time() - t))

# 4. Virasoro inductions --------------------------------------------------------------------------
t = time.time()
V = va.Virasoro(QQ, 3/7, conformal_vector='L'); L = V.gen('L')
A1 = V.zhu_algebra(level=1)


def dims_of(U, D=5):
    M = V.induced_module(U, max_depth=D)
    h = M._cover_module.top_weight()
    return [M.weight_space(h + d).dimension() for d in range(D + 1)]


ok = (dims_of(A1.module({'L_0': 3, 'L_{-1} L_1': 4})) == [1, 1, 2, 3, 5, 7]
      and dims_of(A1.module({'L_0': 3, 'L_{-1} L_1': 0})) == [0, 1, 1, 2, 3, 5]
      and dims_of(A1.module({'L_0': identity_matrix(QQ, 2),
                             'L_{-1} L_1': matrix([[0, 1], [0, 0]])})) == [1, 2, 3, 5, 8, 12])
h0 = 5/3
Mv = V.verma_module(h0)
deg2 = list(Mv.weight_space(h0 + 2).basis())
keys = [x.leading_support() for x in deg2]


def act(w, x):
    for i, m in reversed(w):
        x = L.nproduct(x, m + 1)
    return x


def coords(x):
    d = x.monomial_coefficients()
    return vector(QQ, [d.get(k, 0) for k in keys])


A2 = V.zhu_algebra(level=2)
U2 = A2.module({w: matrix(QQ, [coords(act(w, bb)) for bb in deg2]).transpose()
                for w in A2._generator_words()})
ok = ok and dims_of(U2) == [1, 1, 2, 3, 5, 7]
report('Virasoro: level 1 Verma / shifted / Jordan, level 2 from the degree 2 of M(5/3)', ok,
       '%.1fs' % (time.time() - t))

# 5. refusals ------------------------------------------------------------------------------------
refused = 0
for f in [lambda: V.induced_module(A1.module({'L_0': 3, 'L_{-1} L_1': 4})),
          lambda: A1.module({'L_0': 3, 'L_{-1} L_1': 1}),
          lambda: va.NeveuSchwarz(QQ, 7/10, conformal_vector='L').zhu_algebra(level=1),
          lambda: LY.zhu_algebra_up_to(3, level=1).dimension()]:
    try:
        f()
    except (ValueError, NotImplementedError):
        refused += 1
report('refusals', refused == 4, str(refused))

print('total %.1fs' % (time.time() - t_start))
report.result()
