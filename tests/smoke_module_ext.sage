# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Ext^1 up to a depth from projective covers (vertex_algebra_module_ext.py):
# Virasoro at c = 3/7, the Jordan block of [L_0] at 2 over A_0 / ((x - 2)^2)
# induces a projective cover of M(2): Ext^1(M(2), M(2)) = 1; the triplet
# algebra W(2) at c = -2 (presentation): P(0) induced from the Jordan top of
# A_0, P(1) from the projective indecomposable of A_1 at [T] = 1:
# Ext^1(W(0), W(1)) = Ext^1(W(1), W(0)) = 2, Ext^1(W(0), W(0)) =
# Ext^1(W(1), W(1)) = 0 (Loewy layers W(0) | 2 W(1) | W(0) and
# W(1) | 2 W(0) | W(1)); a cover that is not onto is refused; the same
# covers from L.projective_cover (levels 0 and 1 chosen automatically); the
# direct route: covers \bar M(C) -> W(0), \bar M_1(U_1) -> W(1) that are not
# projective give the part of Ext^1 with a split top, here all of it (2, 2);
# radical and socle series of P(0), P(1); the four simple modules found
# from A_0 (simple_modules_up_to), the default simples of the series; the
# quotient P(0)/rad^2 = W(0) -> 2W(1) and fusion products of these modules
# (P(0) x P(0) at depth 0 = the tops of 2 P(0) + 2 P(1)); extension modules
# of classes (pushouts), non-split; the image of delta (top-level classes
# that lift: the Jordan block of A_0 for \bar M(C) by W(0), giving P(0)).
# Run from the repository root:  sage tests/smoke_module_ext.sage  -> RESULT: PASS
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

# 1. Virasoro: a Jordan self-extension -----------------------------------------------------------
t = time.time()
V = va.Virasoro(QQ, 3/7, conformal_vector='L')
A0 = V.zhu_algebra()
Zv = A0.quotient([(A0.gen(0) - 2)^2])
Pv = V.induced_module(Zv.module({'L': matrix([[2, 1], [0, 2]])}))
Mv = V.verma_module(2)
piv = Pv.hom([0, Mv.highest_weight_vector()], Mv)
E = Mv.ext1_up_to(Mv, 3, piv)
ok = E.dimension() == 1 and E.kernel_generators() == [Pv.monomial(((), 0))]
try:
    Mv.ext1_up_to(Mv, 3, Hom(Pv, Mv).zero())
    ok = False
except ValueError:
    pass
report('Virasoro: Ext^1(M(2), M(2)) = 1 from the Jordan cover; refusal', ok,
       '%.1fs' % (time.time() - t))

# 2. W(2) -------------------------------------------------------------------------------------
Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm * cp,
                            horizontal_grading={'cp': 1, 'cm': -1})
a, b = SF.gens()
S2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
                    a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
                   names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient',
                   conformal_vector='ambient')
P = S2.presentation()
Z = P.zhu_algebra_up_to(8)
Z0 = matrix(QQ, 2, 2)
D = 4
top0 = Z.module({'T': 0, 'Wp': 0, 'W0': 0, 'Wm': 0}, horizontal_grades=[0])
top1 = Z.module({'T': identity_matrix(QQ, 2), 'Wp': matrix([[0, 1], [0, 0]]), 'W0': diagonal_matrix([2, -2]),
                 'Wm': matrix([[0, 0], [-4, 0]])}, horizontal_grades=[1, -1])
jordan = Z.module({'T': matrix([[0, 1], [0, 0]]), 'Wp': Z0, 'W0': Z0, 'Wm': Z0}, horizontal_grades=[0, 0])
W0 = P.induced_module_up_to(top0, 8, D).simple_quotient()
W1 = P.induced_module_up_to(top1, 8, D)
P0 = P.induced_module_up_to(jordan, 8, D)
t = time.time()
pi0 = Hom(P0, W0).basis()[0]
e01, e00 = [W0.ext1_up_to(N, D - 1, pi0).dimension() for N in (W1, W0)]
report('W(2): Ext^1(W(0), W(1)) = 2, Ext^1(W(0), W(0)) = 0 (cover P(0))', (e01, e00) == (2, 0),
       '%.1fs' % (time.time() - t))

t = time.time()
Z1 = P.zhu_algebra_up_to(8, level=1, degree_bound=18)
Ps, Ss = Z1.projective_indecomposable_modules(), Z1.simple_modules()
tw = [Sm.representation_matrix(P.gen('T'))[0, 0] for Sm in Ss]
P1 = P.induced_module_up_to([Mp for Mp, w in zip(Ps, tw) if w == 1][0], 8, D)
pi1 = Hom(P1, W1).basis()[0]
e10, e11 = [W1.ext1_up_to(N, D - 1, pi1).dimension() for N in (W0, W1)]
report('W(2): Ext^1(W(1), W(0)) = 2, Ext^1(W(1), W(1)) = 0 (cover P(1))', (e10, e11) == (2, 0),
       '%.1fs' % (time.time() - t))

t = time.time()
pc0, pc1 = W0.projective_cover(8, D), W1.projective_cover(8, D)
Q0, Q1 = pc0.domain(), pc1.domain()
ok = ('in degree 0' in repr(Q0) and 'in degree 1' in repr(Q1)
      and all(Q0.weight_space(w).dimension() == P0.weight_space(w).dimension()
              and Q1.weight_space(w).dimension() == P1.weight_space(w).dimension()
              for w in range(D + 1)))
ok = ok and [W0.ext1_up_to(N, D - 1, pc0).dimension() for N in (W1, W0)] == [2, 0]
ok = ok and [W1.ext1_up_to(N, D - 1, pc1).dimension() for N in (W0, W1)] == [2, 0]
report('W(2): projective_cover = P(0), P(1) (levels 0, 1); the same Ext^1', ok,
       '%.1fs' % (time.time() - t))

t = time.time()
M0 = P.induced_module_up_to(top0, 8, D)
U1 = [Sm for Sm in Ss if Sm.dimension() == 2 and Sm.representation_matrix(P.gen('T')) == 1]
M1 = P.induced_module_up_to(U1[0], 8, D)
d01 = W0.ext1_up_to(W1, D - 1, Hom(M0, W0).basis()[0]).dimension()
d10 = W1.ext1_up_to(W0, D - 1, Hom(M1, W1).basis()[0]).dimension()
report('W(2), direct: Ext^1(W(0), W(1)) = Ext^1(W(1), W(0)) = 2 from non-projective covers',
       (len(U1), d01, d10) == (1, 2, 2), '%.1fs' % (time.time() - t))

t = time.time()
W0d = P.induced_module_up_to(top0, 8, D + 1).simple_quotient()
W1d = P.induced_module_up_to(top1, 8, D + 1)
sims = [W0d, W1d]
lay = [Q0.radical_series_up_to(D, sims), Q0.socle_series_up_to(D, sims),
       Q1.radical_series_up_to(D, sims), Q1.socle_series_up_to(D, sims)]
report('W(2): Loewy layers of P(0), P(1) (radical and socle series)',
       lay == [[[1, 0], [0, 2], [1, 0]]] * 2 + [[[0, 1], [2, 0], [0, 1]]] * 2,
       '%s %.1fs' % (lay, time.time() - t))

t = time.time()
Sall = P.simple_modules_up_to(8, D)
ws = [S._weight_base() for S in Sall]
dims = [[S.weight_space(w + d).dimension() for d in range(D + 1)] for S, w in zip(Sall, ws)]
ok = ws == [-1/8, 0, 3/8, 1] and dims[1] == [W0.weight_space(d).dimension() for d in range(D + 1)]
ok = ok and Q1.radical_series_up_to(D) == [[0, 1], [2, 0], [0, 1]]
ok = ok and Q0.socle_series_up_to(D) == [[1, 0], [0, 2], [1, 0]]
report('W(2): simple_modules_up_to (four, by lowest weight); default simples in the series', ok,
       '%s %s %.1fs' % (ws, dims, time.time() - t))

t = time.time()
Wm8 = Sall[0]
subs = Q0.radical_series_up_to(D, submodules=True)[0]
R2 = Q0.quotient(subs[2])
cw = [X.fusion_product(Y).depth_quotient_up_to(0, 3).conformal_weights()
      for X, Y in ((R2, Wm8), (Q0, Wm8), (Q0, W0), (R2, W0), (Q0, Q0))]
ok = ([R2.weight_space(w).dimension() for w in range(5)] == [1, 4, 5, 8, 13]
      and cw == [[(-1/8, [1]), (3/8, [1, 1, 1, 1])], [(-1/8, [1, 1]), (3/8, [1, 1, 1, 1])],
                 [(0, [2])], [(0, [1])], [(0, [2, 2, 1, 1, 1, 1]), (1, [1, 1, 1, 1])]])
report('W(2): P(0)/rad^2 = W(0) -> 2W(1); fusion with W(-1/8), W(0), P(0) (depth 0)', ok,
       '%s %.1fs' % (cw, time.time() - t))

t = time.time()
E01 = W0.ext1_up_to(W1, D - 1, Hom(M0, W0).basis()[0]).extension([1, 0])
E10 = W1.ext1_up_to(W0, D - 1, Hom(M1, W1).basis()[0]).extension([1, 0])
d01 = [E01.weight_space(w).dimension() for w in range(D)]
d10 = [E10.weight_space(w).dimension() for w in range(D)]
s01, s10 = E01.socle_series_up_to(D - 1, [W0, W1]), E10.socle_series_up_to(D - 1, [W0, W1])
ok = (d01 == [W0.weight_space(w).dimension() + W1.weight_space(w).dimension() for w in range(D)]
      and d10 == d01 and s01 == [[0, 1], [1, 0]] and s10 == [[1, 0], [0, 1]])
report('W(2): extensions W(0) by W(1) and W(1) by W(0) (pushouts) are non-split', ok,
       '%s %s %s %s %.1fs' % (d01, d10, s01, s10, time.time() - t))

t = time.time()
Ed = W0.ext1_up_to(W0, D - 1, Hom(M0, W0).basis()[0])
Fd = M0.ext1_up_to(W0, D - 1, End(M0).identity())
Pt = Fd.top_extension([1])
got = (Ed.top_level_dimension(), Ed.delta_image(), Ed.dimension(),
       Fd.top_level_dimension(), Fd.delta_image(), Fd.dimension(),
       [Pt.weight_space(w).dimension() for w in range(D)], Pt.radical_series_up_to(D - 1, [W0, W1]))
report('W(2): delta image: Jordan class of A_0 lifts for M(C) by W(0) (= P(0)), not for W(0)',
       got == (1, [], 0, 1, [(1,)], 1, [2, 4, 6, 12], [[1, 0], [0, 2], [1, 0]]),
       '%s %.1fs' % (got, time.time() - t))

print('total %.1fs' % (time.time() - t_start))
report.result()
