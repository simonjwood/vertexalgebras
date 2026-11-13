# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Fusion products by the Nahm-Gaberdiel-Kausch algorithm (fusion_product.py):
# depth-d quotients of M ⊠ N against known fusion rules.  The Ising and
# Lee-Yang models (all products, depths 0 and 1), the vacuum as the unit on
# both sides; Kanade-Ridout's c = -2 Virasoro example (Jordan block at depth
# 0, dimension 3 at depth 1, their spurious state); the triplet algebra W(2)
# from its presentation: Gaberdiel-Kausch's table (W(-1/8)^2 and W(3/8)^2 a
# Jordan block at 0 = the top of R_0 = P(0), W(-1/8) x W(3/8) the weights 0,
# 0, 1, 1 of R_1), every depth-0 quotient a graded A(W(2))-module, the fused
# top of W(-1/8)^2 induces P(0); affine sl_2 at level 1 (U(sl_2) acting on
# the depth-0 quotients); odd generators (N = 1 tricritical Ising: τ ⊠ τ =
# 1 + τ, parities; symplectic fermions with the Grassmann top); Δ is a
# homomorphism of mode algebras (composite fields included); the pole
# relations of composite fields follow from those of the generators; the
# pure-Python modules give the same results as the engine; the window only
# shrinks the result.
# Run from the repository root:  sage tests/smoke_fusion.sage  -> RESULT: PASS
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
import vertex_algebras as va
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from axioms import Reporter

report = Reporter()
t_start = time.time()


def weights(M, N, d=0, K=3):
    return M.fusion_product(N).depth_quotient_up_to(d, K).conformal_weights()


# 1. minimal models ------------------------------------------------------------------------------
t = time.time()
V = va.Virasoro(QQ, 1/2, conformal_vector='L')
I = {h: V.verma_module(h).simple_quotient() for h in (0, 1/2, 1/16)}
table = {(0, 0): [0], (0, 1/2): [1/2], (0, 1/16): [1/16], (1/2, 1/2): [0], (1/2, 1/16): [1/16],
         (1/16, 1/16): [0, 1/2]}
ok = all(sorted(w for w, _ in weights(I[a], I[b])) == sorted(r) and
         sorted(w for w, _ in weights(I[b], I[a])) == sorted(r) for (a, b), r in table.items())
ok = ok and weights(I[1/16], I[1/16], 1, 5) == [(0, [1]), (1/2, [1]), (3/2, [1])]
V = va.Virasoro(QQ, -22/5, conformal_vector='L')
LY = {h: V.verma_module(h).simple_quotient() for h in (0, -1/5)}
ok = ok and weights(LY[-1/5], LY[-1/5]) == [(-1/5, [1]), (0, [1])] and weights(LY[0], LY[-1/5]) == [(-1/5, [1])]
ok = ok and weights(LY[-1/5], LY[-1/5], 2, 6) == [(-1/5, [1]), (0, [1]), (4/5, [1]), (9/5, [1]), (2, [1])]
report('Ising and Lee-Yang fusion rules (depths 0-2)', ok, '%.1fs' % (time.time() - t))

# 2. Kanade-Ridout c = -2 -----------------------------------------------------------------------
t = time.time()
V = va.Virasoro(QQ, -2, conformal_vector='L')
M = V.verma_module(-1/8).simple_quotient(); one = V.verma_module(0).simple_quotient()
F0 = M.fusion_product(M).depth_quotient_up_to(0, 3)
F1 = M.fusion_product(M).depth_quotient_up_to(1, 5)
g0, g1 = ((), 0), (((0, -1),), 0)
ok = (F0.conformal_weights() == [(0, [2])] and F1.conformal_weights() == [(0, [2]), (1, [1])]
      and F1._E.normal_form({(g1, g1): 2, (g1, g0): -1/2, (g0, g1): 1/2, (g0, g0): -1/8}) == {}
      and F0.special_subspace_dimensions()[2] == 0
      and all(weights(one, M, d, 5) == weights(M, one, d, 5) for d in (0, 1, 2)))
# the window: the result only shrinks, and is stable from the window where M^ss fits
dims = [M.fusion_product(M).depth_quotient_up_to(1, K).dimension() for K in (1, 2, 3, 4, 5, 6)]
ok = ok and all(x >= y for x, y in zip(dims, dims[1:])) and dims[-3:] == [3, 3, 3]
report('c = -2: Jordan block, spurious state, window', ok, '%s %.1fs' % (dims, time.time() - t))

# 3. the triplet algebra W(2) -------------------------------------------------------------------
t = time.time()
Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp,
                            horizontal_grading={'cp': 1, 'cm': -1})
a, b = SF.gens()
W2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
                    a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
                   names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient', conformal_vector='ambient')
P = W2.presentation()
Z = P.zhu_algebra_up_to(8)
tops = {
    0: Z.module({'T': 0, 'Wp': 0, 'W0': 0, 'Wm': 0}, horizontal_grades=[0]),
    -1/8: Z.module({'T': -1/8, 'Wp': 0, 'W0': 0, 'Wm': 0}, horizontal_grades=[0]),
    1: Z.module({'T': identity_matrix(QQ, 2), 'Wp': matrix([[0, 1], [0, 0]]), 'W0': diagonal_matrix([2, -2]),
                 'Wm': matrix([[0, 0], [-4, 0]])}, horizontal_grades=[1, -1]),
    3/8: Z.module({'T': 3/8*identity_matrix(QQ, 2), 'Wp': matrix([[0, 1], [0, 0]]),
                   'W0': diagonal_matrix([1/2, -1/2]), 'Wm': matrix([[0, 0], [-1/4, 0]])}, horizontal_grades=[1, -1]),
}
Wm = {h: (P.induced_module_up_to(U, 8, 5).simple_quotient() if h == 0 else P.induced_module_up_to(U, 8, 5))
      for h, U in tops.items()}
gk = {(0, 0): [(0, [1])], (0, 1): [(1, [1, 1])], (0, -1/8): [(-1/8, [1])], (0, 3/8): [(3/8, [1, 1])],
      (1, 1): [(0, [1])], (1, -1/8): [(3/8, [1, 1])], (1, 3/8): [(-1/8, [1])],
      (-1/8, -1/8): [(0, [2])], (-1/8, 3/8): [(0, [1, 1]), (1, [1, 1])], (3/8, 3/8): [(0, [2])]}
ok = True
bad = []
for (h1, h2), r in gk.items():
    F = Wm[h1].fusion_product(Wm[h2]).depth_quotient_up_to(0, 3)
    U = F.zhu_algebra_module(Z)
    if F.conformal_weights() != r or F.horizontal_grades() is None or U.dimension() != F.dimension():
        ok = False
        bad.append((h1, h2))
# the fused top of W(-1/8)^2 induces P(0) (= the Jordan top of triplet_modules.ipynb §6)
F = Wm[-1/8].fusion_product(Wm[-1/8]).depth_quotient_up_to(0, 3)
Mu = P.induced_module_up_to(F.zhu_algebra_module(Z), 8, 5)
jordan = Z.module({'T': matrix([[0, 1], [0, 0]]), 'Wp': matrix(QQ, 2), 'W0': matrix(QQ, 2), 'Wm': matrix(QQ, 2)},
                  horizontal_grades=[0, 0])
Mj = P.induced_module_up_to(jordan, 8, 5)
du, dj = [Mu.weight_space(k).dimension() for k in range(6)], [Mj.weight_space(k).dimension() for k in range(6)]
ok = ok and du == dj == [2, 4, 6, 12, 18, 28]
report('W(2): Gaberdiel-Kausch table, A(W(2))-modules, P(0)', ok, '%s %s %.1fs' % (bad, du, time.time() - t))

# engine and pure-Python modules agree
t = time.time()
Mp = {h: P.induced_module_up_to(tops[h], 8, 4, implementation='python') for h in (-1/8, 3/8)}
Fe = Wm[-1/8].fusion_product(Wm[3/8]).depth_quotient_up_to(1, 4)
Fp = Mp[-1/8].fusion_product(Mp[3/8]).depth_quotient_up_to(1, 4)
ok = Fe.dimension() == Fp.dimension() and Fe.conformal_weights() == Fp.conformal_weights() \
    and Fe.l0_matrix() == Fp.l0_matrix()
report('W(2): engine = pure Python (depth 1)', ok, '%s %.1fs' % (Fe.conformal_weights(), time.time() - t))


# 4. Δ is a homomorphism of mode algebras; composite fields ------------------------------------
def gtuple(v):
    dv = v.monomial_coefficients()
    return (('e', tuple(sorted(dv.items(), key=repr))), dv, QQ(v.parent().degree(v)), v.parity())


def delta(F, g, k, vec):
    res = {}
    for x, c in vec.items():
        if x[0] not in F._A.depth or x[1] not in F._B.depth:
            return None
        try:
            r = F._delta_mode(g, k, x)
        except ValueError:            # beyond the depth of a module
            return None
        if r is None:
            return None
        for key, cc in r.items():
            res[key] = res.get(key, 0) + c * cc
    return {k: c for k, c in res.items() if c}


def hom_check(F, elems, modes, ntries):
    random.seed(1r)
    bad = tried = 0
    xs = [x for x, room in F._tensors() if room >= 2]
    for _ in range(ntries):
        u, w = random.choice(elems), random.choice(elems)
        m, n = random.choice(modes), random.choice(modes)
        x = random.choice(xs)
        s = -1 if (u.parity() and w.parity()) else 1
        y1, y2 = delta(F, gtuple(w), n, {x: 1}), delta(F, gtuple(u), m, {x: 1})
        if y1 is None or y2 is None:
            continue
        l1, l2 = delta(F, gtuple(u), m, y1), delta(F, gtuple(w), n, y2)
        if l1 is None or l2 is None:
            continue
        rhs = {}
        good = True
        for j in range(0, 10):
            uw = u.nproduct(w, j)
            if uw == 0:
                continue
            r = delta(F, gtuple(uw), m + n - j, {x: 1})
            if r is None:
                good = False
                break
            for k2, c in r.items():
                rhs[k2] = rhs.get(k2, 0) + binomial(m, j) * c
        if not good:
            continue
        tried += 1
        diff = dict(l1)
        for k2, c in l2.items():
            diff[k2] = diff.get(k2, 0) - s * c
        for k2, c in rhs.items():
            diff[k2] = diff.get(k2, 0) - c
        bad += any(diff.values())
    return tried, bad


def composite_relations(F, elems):
    total = missing = 0
    d = F._d
    for v in elems:
        g = gtuple(v)
        h = g[2]
        for x, room in F._tensors():
            bb = -1
            while h - 1 - bb <= room:
                r = F._delta_pole(g, bb, x)
                for m in range(0, floor(d - (h - 1 - bb)) + 1):
                    if r is None:
                        break
                    rm = F._delta_mode(g, bb - m, x)
                    if rm is None:
                        r = None
                        break
                    for k2, c in rm.items():
                        r[k2] = r.get(k2, 0) - binomial(bb, m) * (-1) ** (m % 2) * c
                if r is not None:
                    r = {k2: c for k2, c in r.items() if c}
                    if r:
                        total += 1
                        missing += bool(F._E.normal_form(r))
                bb -= 1
    return total, missing


t = time.time()
T, Wp, W0, Wmi = [P.gen(n) for n in ('T', 'Wp', 'W0', 'Wm')]
Mh = {h: P.induced_module_up_to(tops[h], 8, 6) for h in (-1/8, 3/8)}
F = Mh[-1/8].fusion_product(Mh[3/8]).depth_quotient_up_to(1, 5)
hom = hom_check(F, [T, Wp, W0, Wmi, T.derivative(), T.no(T)], list(range(-3, 5)), 1000)
comp = composite_relations(F, [T.no(T), T.derivative(), Wp.no(Wmi), T.no(W0), W0.derivative()])
ok = hom[0] >= 30 and hom[1] == 0 and comp[0] >= 10 and comp[1] == 0
report('W(2): Δ homomorphism, composite pole relations', ok, 'hom %s comp %s %.1fs' % (hom, comp, time.time() - t))

# 5. affine sl_2 at level 1: U(sl_2) on the depth-0 quotients ----------------------------------
t = time.time()
V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
A = V.zhu_algebra()
D = V.induced_module(A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
                                'F': matrix([[0, 0], [1, 0]])})).simple_quotient()
O = V.induced_module(A.module({'E': matrix(QQ, 1), 'H': matrix(QQ, 1), 'F': matrix(QQ, 1)})).simple_quotient()
Fdd = D.fusion_product(D).depth_quotient_up_to(0, 3)
Fod = O.fusion_product(D).depth_quotient_up_to(0, 3)
Udo = D.fusion_product(O).depth_quotient_up_to(0, 3).zhu_algebra_module(A)
ok = (Fdd.conformal_weights() == [(0, [1])] and Fod.conformal_weights() == [(1/4, [1, 1])]
      and Udo.dimension() == 2 and Fod.zhu_algebra_module(A).dimension() == 2
      and sorted(Fod.horizontal_grades()) == [(-1,), (1,)])
ok = ok and D.fusion_product(D).depth_quotient_up_to(1, 4).conformal_weights() == [(0, [1]), (1, [1, 1, 1])]
report('affine sl_2 level 1: doublet x doublet = vacuum', ok, '%.1fs' % (time.time() - t))

# 6. odd generators -----------------------------------------------------------------------------
t = time.time()
V = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L')
vac, tau = V.verma_module(0).simple_quotient(), V.verma_module(1/10).simple_quotient()
F = tau.fusion_product(tau)
ok = (F.depth_quotient_up_to(0, 3).conformal_weights() == [(0, [1]), (1/10, [1])]
      and F.depth_quotient_up_to(1, 3).conformal_weights() == [(0, [1]), (1/10, [1]), (3/5, [1]), (11/10, [1])]
      and F.depth_quotient_up_to(1, 3).parities() == [0, 1, 0, 1]
      and all(weights(vac, tau, d, 3) == weights(tau, vac, d, 3) ==
              [(1/10, [1]), (3/5, [1]), (11/10, [1])][:int(2 * d) + 1] for d in (0, 1/2, 1)))
Ep = matrix(QQ, 4, 4, {(1, 0): 1, (3, 2): 1})
Em = matrix(QQ, 4, 4, {(2, 0): 1, (3, 1): -1})
G = SF.induced_module(SF.zhu_algebra().module({'cp': Ep, 'cm': Em}, parities=[0, 1, 1, 0], horizontal_grades=[0, 1, -1, 0]))
Vs = SF.induced_module(SF.zhu_algebra().module({'cp': matrix(QQ, 1), 'cm': matrix(QQ, 1)}))
F1, F2 = Vs.fusion_product(G).depth_quotient_up_to(1, 3), G.fusion_product(Vs).depth_quotient_up_to(1, 3)
ok = ok and F1.dimension() == F2.dimension() == 12 and sorted(F1.parities()) == sorted(F2.parities()) \
    and F1.conformal_weights() == F2.conformal_weights() == [(0, [2, 1, 1]), (1, [2, 2, 1, 1, 1, 1])]
report('odd generators: N = 1 tricritical Ising, symplectic fermions', ok,
       '%s %.1fs' % (F1.conformal_weights(), time.time() - t))

# 7. refusals -----------------------------------------------------------------------------------
refused = 0
for f in (lambda: tau.fusion_product(I[0]),
          lambda: tau.fusion_product(tau).depth_quotient_up_to(1, 1/2),
          lambda: tau.fusion_product(tau).depth_quotient_up_to(1, 3).zhu_algebra_module(None)):
    try:
        f()
    except ValueError:
        refused += 1
report('refusals', refused == 3, str(refused))

print('total %.1fs' % (time.time() - t_start))
report.result()
