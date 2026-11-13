# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# W-algebras by quantum Hamiltonian reduction (quantum_hamiltonian_reduction.py):
# Kac-Roan-Wakimoto's central charge formula, free generation (the universal
# W against the kernel of d_0 on C^-_0), the Jacobi identity of W, known
# algebras (Virasoro, Zamolodchikov's W_3 with β = 16/(22+5c), N=1, N=2),
# the corrected λ² coefficient of [G^u λ G^v] of Kac-Wakimoto's corrigendum
# (Adv. Math. 193 (2005)) for minimal W-algebras, the ways of giving f (also
# Jordan types), the affine method, rational levels, refusals.
# Run from the repository root:  sage tests/smoke_w_algebras.sage  -> RESULT: PASS
# (VERTEX_ALGEBRAS_SLOW_TESTS=1 adds W(G_2), about 25 s)
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.lie_superalgebras import LieSuperalgebra
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from vertex_algebras.quantum_hamiltonian_reduction import WAlgebra, _nilpotent
from axioms import Reporter

report = Reporter()
t_start = time.time()
K.<k> = Frac(QQ['k'])
L = LieSuperalgebra
A1, A2, A3 = [L(QQ, CartanMatrix(['A', r])) for r in (1, 2, 3)]
C2 = L(QQ, 'spo(4|0)')
B2 = L(QQ, CartanMatrix(['B', 2]))
G2 = L(QQ, CartanMatrix(['G', 2]))


def krw_central_charge(g, f, x):
    """c(g, f, k) = k sdim g/(k + h^∨) - 12k(x|x) - Σ_{α>0} (-1)^{p(α)}(12 m_α^2 -
    12 m_α + 2) - 1/2 sdim g_{1/2}  (Kac-Roan-Wakimoto)."""
    gr = g.ad_eigenvalues(x)
    B = g.basis()
    s = sum((-1) ** B[a].is_even_odd() * (12 * m ^ 2 - 12 * m + 2) for a, m in gr.items() if m > 0)
    sh = sum((-1) ** B[a].is_even_odd() for a, m in gr.items() if m == QQ(1) / 2)
    hv = g.dual_coxeter_number()
    return k * g.superdimension() / (k + hv) - 12 * k * g.invariant_form(x, x) - s - sh / 2


# W(G_2) (principal, 22 s: the kernel of d_0 up to weight 6) only with
# VERTEX_ALGEBRAS_SLOW_TESTS
SLOW = bool(os.environ.get('VERTEX_ALGEBRAS_SLOW_TESTS'))
cases = [(A1, 'principal'), (A2, 'principal'), (A2, 'minimal'), (A3, 'principal'),
         (C2, 'principal')] + ([(G2, 'principal')] if SLOW else []) + [
         (A3, 'minimal'), (A3, [(1, 0, 0), (0, 0, 1)]), (C2, 'minimal'),
         (G2, 'minimal'), (L(QQ, 'spo(2|1)'), 'minimal'), (L(QQ, 'sl(2|1)'), 'minimal'),
         (L(QQ, 'psl(2|2)'), 'minimal'), (L(QQ, 'spo(2|3)'), 'minimal'),
         (L(QQ, 'sl(3|1)'), 'minimal'), (L(QQ, 'D(2,1;a)', a=2), 'minimal'),
         # Jordan types
         (A2, [2, 1]), (A3, [3, 1]), (B2, [3, 1, 1]), (C2, [2, 2]),
         (L(QQ, 'spo(2|3)'), ([1, 1], [3])), (L(QQ, 'sl(3|1)'), ([3], [1])),
         (L(QQ, 'psl(2|2)'), ([2], [2])), (L(QQ, 'osp(4|2)'), ([2, 2], [2]))]
cc, free = [], []
Ws = {}
for g, spec in cases:
    t = time.time()
    W = WAlgebra(K, g, spec, k, conformal_vector='reduction')
    Ws[(str(g), str(spec))] = W
    f = _nilpotent(g, spec)
    x = g.dynkin_grading(f)
    cc.append((str(g).split(' over')[0], spec, W.central_charge() == krw_central_charge(g, f, x)))
    if g.dimension() <= 15:
        Q = UniversalAffineVertexAlgebra(K, g, level=k, conformal_vector='sugawara').quantum_hamiltonian_reduction_complex(spec)
        C, d, Cm, Ker = Q.vertex_algebra(), Q.differential(), Q.subalgebra(), Q.kernel()
        top = 3 if g.dimension() > 8 else 4
        ws = [QQ(n) / 2 for n in range(2 * top + 1)]
        free.append(([W.weight_space(w).dimension() for w in ws]
                     == [Ker.weight_space(w).dimension() for w in ws], str(g).split(' over')[0], spec))
report("central charges of %d reductions = Kac-Roan-Wakimoto's formula (sl_2, W_3, BP, W_4, "
       "W(sp_4), sl_4 minimal and rectangular, sp_4 minimal, G_2 minimal, N=1, N=2, small N=4, "
       "N=3, sl(3|1), D(2,1;2); Jordan types: sl_3 (2,1), sl_4 (3,1), so_5 (3,1,1), sp_4 "
       "(2,2), osp(3|2) ((1,1),(3)), sl(3|1) ((3),(1)), psl(2|2) ((2),(2)), osp(4|2) ((2,2),(2)); "
       "with VERTEX_ALGEBRAS_SLOW_TESTS also W(G_2))"
       % len(cases), all(ok for _, _, ok in cc), cc)
same_alg = [(str(g).split(' over')[0], spec, WAlgebra(K, g, spec, k, algorithm='complex', conformal_vector='reduction') is W)
            for (g, spec), W in zip(cases, Ws.values())
            if not (spec == 'principal' and g.dimension() > 8)]   # (the complex: slow)
report("the Miura map (default) and the complex give the same W (algorithm='complex' "
       "computes the OPEs in C)", all(ok for _, _, ok in same_alg), same_alg)
report('free generation: the universal W has the dimensions of the kernel of d_0 on C^-_0',
       all(ok for ok, _, _ in free), free)
jac = []
for key in [(str(A1), 'principal'), (str(A2), 'principal'), (str(A2), 'minimal'),
            (str(L(QQ, 'spo(2|1)')), 'minimal'), (str(L(QQ, 'sl(2|1)')), 'minimal'),
            (str(C2), str([2, 2])), (str(A3), str([3, 1])),
            (str(L(QQ, 'spo(2|3)')), str(([1, 1], [3])))]:
    jac.append(Ws[key].jacobi_identity_failures() == {})
report('the Jacobi identity holds in W (Virasoro, W_3, BP, N=1, N=2, sp_4 (2,2), sl_4 (3,1), '
       'osp(3|2) ((1,1),(3)))', all(jac), jac)

# known algebras --------------------------------------------------------------
W3 = WAlgebra(K, A2, 'principal', k, names=['L', 'W'], conformal_vector='reduction')
T, Wg = W3.conformal_vector(), W3.gen('W')
c = W3.central_charge()
a = -(T.nproduct(Wg, 2)).monomial_coefficients()[T.leading_support()] / T.nproduct(T.derivative(), 2).monomial_coefficients()[T.leading_support()]
Wp = Wg + a * T.derivative()                      # primary
N = 3 * Wp.nproduct(Wp, 5).monomial_coefficients()[()] / c
beta = 16 / (22 + 5 * c)
Lam = T.no(T) - 3 / 10 * T.derivative(2)
zam = [N * (beta * Lam.derivative() + T.derivative(3) / 15), N * (2 * beta * Lam + 3 / 10 * T.derivative(2)),
       N * T.derivative(), 2 * N * T, 0, N * c / 3 * W3.one()]
report('W_3 from sl_3 is Zamolodchikov\'s W_3: β = 16/(22 + 5c) (W made primary, normalised)',
       T.nproduct(Wp, 2) == 0 and Wp.ope(Wp) == zam)
N2 = Ws[(str(L(QQ, 'sl(2|1)')), 'minimal')]
wts = N2.degrees()
Jn = [s for s, w in wts.items() if w == 1][0]
Gn = [s for s, w in wts.items() if w == QQ(3) / 2]
J = N2.gen(Jn)
q = J.nproduct(N2.gen(Gn[0]), 0).monomial_coefficients()[N2.gen(Gn[0]).leading_support()]
Jnorm = J / q
report('N=2 from sl(2|1): with J normalised to charges ±1, J(z)J(w) ~ (c/3)/(z-w)^2',
       Jnorm.nproduct(Jnorm, 1) == N2.central_charge() / 3 * N2.one()
       and Jnorm.nproduct(N2.gen(Gn[1]), 0) == -N2.gen(Gn[1]))
N1 = Ws[(str(L(QQ, 'spo(2|1)')), 'minimal')]
G = [N1.gen(s) for s, w in N1.degrees().items() if w == QQ(3) / 2][0]
T1 = N1.conformal_vector()
mu = G.nproduct(G, 0).monomial_coefficients()[T1.leading_support()] / T1.monomial_coefficients()[T1.leading_support()]
report('N=1 from osp(1|2): G(z)G(w) ~ (2c/3)/(z-w)^3 + 2T/(z-w) up to normalisation',
       G.nproduct(G, 0) == mu * T1 and G.nproduct(G, 1) == 0
       and G.nproduct(G, 2) == mu * N1.central_charge() / 3 * N1.one())


# the corrigendum: λ² coefficient of [G^u λ G^v] for minimal W-algebras -----------
def corrigendum_ok(g):
    """G^u_(2) G^v = 2 c(u, v)/3 with Kac-Wakimoto's corrected c(u, v) (the Remark of the
    corrigendum: κ_{g_0} = 2 h_0^∨ (|) on g♮), e normalised by (e|f) = 1/2."""
    W = WAlgebra(K, g, 'minimal', k, conformal_vector='reduction')
    f = g.minimal_nilpotent()
    e, h, _ = g.sl2_triple(f)
    x, e = h / 2, e / 2
    gr, B = g.ad_eigenvalues(x), g.basis()
    g0 = [b for b in B.keys() if gr[b] == 0]
    sd0 = sum((-1) ** B[b].is_even_odd() for b in g0)
    sdh = sum((-1) ** B[b].is_even_odd() for b in B.keys() if gr[b] == QQ(1) / 2)
    hv = g.dual_coxeter_number()
    nat = [v for v in g.centralizer_basis(f, x) if gr[v.leading_support()] == 0]

    def kappa0(a, b):
        return sum((-1) ** B[cc].is_even_odd() * a.bracket(b.bracket(B[cc])).monomial_coefficients().get(cc, 0)
                   for cc in g0)
    ratios = set(kappa0(a, b) / (2 * g.invariant_form(a, b)) for a in nat for b in nat
                 if g.invariant_form(a, b))
    h0 = ratios.pop() if ratios else 0
    if ratios:
        return False
    ck = W.central_charge()
    Gs = [(g.basis()[s[2:-1]], W.gen(s)) for s, w in W.degrees().items()
          if w == QQ(3) / 2]
    for u, Gu in Gs:
        for v, Gv in Gs:
            cuv = -g.invariant_form(e, u.bracket(v)) * ((k + hv) * ck - (k + (hv - h0) / 2) * (sd0 + sdh))
            if Gu.nproduct(Gv, 2) != 2 * cuv / 3 * W.one():
                return False
    return True


corr = [(str(g).split(' over')[0], corrigendum_ok(g))
        for g in [L(QQ, 'sl(2|1)'), L(QQ, 'spo(2|1)'), A2, L(QQ, 'psl(2|2)'), L(QQ, 'spo(2|3)')]]
report("the corrected λ² coefficient of [G^u λ G^v] (Kac-Wakimoto's corrigendum, Theorem "
       "5.1(e)): N=2, N=1, BP, small N=4, N=3", all(ok for _, ok in corr), corr)

# the ways of giving f, the affine method, levels, refusals ----------------------
same = [WAlgebra(K, A2, 'principal', k, conformal_vector='reduction') is WAlgebra(K, A2, [(1, 0), (0, 1)], k, conformal_vector='reduction')
        is WAlgebra(K, A2, A2.principal_nilpotent(), k, conformal_vector='reduction'),
        UniversalAffineVertexAlgebra(K, A2, level=k, conformal_vector='sugawara').quantum_hamiltonian_reduction('minimal', conformal_vector='reduction')
        is WAlgebra(K, A2, 'minimal', k, conformal_vector='reduction'),
        UniversalAffineVertexAlgebra(K, ['A', 2], level=k, conformal_vector='sugawara').quantum_hamiltonian_reduction('principal', conformal_vector='reduction')
        is WAlgebra(K, A2, 'principal', k, conformal_vector='reduction')]
# Jordan types: the same f as roots / 'principal'; an orbit of another f (BP:
# (2, 1) is E[-α1], 'minimal' E[-θ]); so_5 ≅ sp_4 ((3,1,1) <-> (2,2))
same.append(WAlgebra(K, A3, [2, 2], k, conformal_vector='reduction') is WAlgebra(K, A3, [(1, 0, 0), (0, 0, 1)], k, conformal_vector='reduction')
            and WAlgebra(K, A3, Partition([4]), k, conformal_vector='reduction') is WAlgebra(K, A3, 'principal', k, conformal_vector='reduction'))
for (g1, s1), (g2, s2) in [((A2, [2, 1]), (A2, 'minimal')), ((B2, [3, 1, 1]), (C2, [2, 2]))]:
    W1, W2 = Ws[(str(g1), str(s1))], Ws[(str(g2), str(s2))]
    same.append(W1.central_charge() == W2.central_charge()
                and sorted(W1.degrees().values()) == sorted(W2.degrees().values())
                and [W1.weight_space(n).dimension() for n in range(5)]
                == [W2.weight_space(n).dimension() for n in range(5)])
BPq = WAlgebra(QQ, A2, 'minimal', QQ(1) / 2, conformal_vector='reduction')
same.append(BPq.base_ring() is QQ and BPq.central_charge() == QQ(-20) / 7)
report("f as 'principal', roots, a Jordan type or an element: the same W; the affine method "
       "(LieSuperalgebra and Cartan type); BP from the Jordan type (2, 1) and 'minimal', "
       "W(so_5, (3,1,1)) and W(sp_4, (2,2)): the same central charges, weights, dimensions; "
       "a rational level", all(same), same)
refused = []
for args, err in [((A2, A2.h(1), k), ValueError), ((A2, A2.root_vector((1, 0), -1)
                    + A2.root_vector((1, 1), -1), k), NotImplementedError),
                  ((L(QQ, 'sl(2|1)'), L(QQ, 'sl(2|1)').f(2), k), ValueError),
                  ((A2, 'other', k), ValueError), ((B2, [2, 2, 1, 1], k), ValueError),
                  ((C2, [3, 1], k), ValueError)]:
    try:
        WAlgebra(K, *args)
        refused.append(str(args[1]))
    except err:
        pass
try:
    WAlgebra(K, A2, 'principal', k, x=A2.h(1) / 2, conformal_vector='reduction')
    refused.append('x')
except ValueError:
    pass
report('refused: not nilpotent, a grading outside the Cartan subalgebra, odd f, unknown '
       'names, Jordan types of the wrong size or not symplectic, a grading that is not good',
       not refused, refused)
report('W pickles, TestSuite', loads(dumps(W3)) is W3 and TestSuite(W3).run() is None)

print('total %.1fs' % (time.time() - t_start))
report.result()
