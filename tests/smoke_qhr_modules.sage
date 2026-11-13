# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Quantum Hamiltonian reduction of highest weight modules
# (quantum_hamiltonian_reduction_modules.py).  The W highest weight from the
# Miura map against Kac-Wakimoto (6.14) (h♮ and L_0) for principal, minimal,
# rectangular and super reductions (non-abelian g_0 included); the '-'
# evaluation against Arakawa (285) and the dot invariance of central
# characters (A_2, B_2, G_2), the '+' one against Arakawa Thm 9.1.4's
# γ_{λ - κρ^∨}; Euler-Poincaré characters (KRW Thm 3.1) = characters of the
# W-Verma modules (sl_2, W_3, BP, N = 1, N = 2); the reduction functor on
# Verma modules (symbolic λ, k) and simple modules: Lee-Yang from the
# admissible sl_2 modules by '+' and '-' (Rogers-Ramanujan), the 3-state
# Potts model from sl_3 at k + 3 = 4/5 by '-' (the six W_3 primaries),
# Bershadsky-Polyakov zeros by Arakawa's criterion; refusals.
# Run from the repository root:  sage tests/smoke_qhr_modules.sage  -> RESULT: PASS
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from vertex_algebras.lie_superalgebras import LieSuperalgebra
from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData, _reduction_is_nonzero
from vertex_algebras.vertex_algebra_submodule import ZeroModule
from axioms import Reporter

report = Reporter()
t_start = time.time()
K.<k, a, b, c> = Frac(QQ['k, a, b, c'])


def affine(g, level=k, R=K):
    if isinstance(g, str):
        return UniversalAffineVertexAlgebra(R, LieSuperalgebra(QQ, g), level=level)
    return UniversalAffineVertexAlgebra(R, g, level=level)


# 1. KW04 (6.14): h♮ and L_0 ---------------------------------------------------------------------
t = time.time()
cases = [(['A', 1], 'principal'), (['A', 2], 'principal'), (['A', 2], 'minimal'),
         (['A', 3], 'minimal'), (['B', 2], 'minimal'), (['A', 3], [2, 2]), (['C', 2], 'principal'),
         ('sl(2|1)', 'minimal'), ('spo(2|1)', 'minimal'), ('spo(2|3)', 'minimal'),
         ('sl(3|1)', 'minimal')]
bad = []
for g, f in cases:
    V = affine(g)
    D = _ReductionData(V, f, None, None)
    lam = D.weight([a, b, c][:D.r])
    values, h = D.top(lam)
    kw = D.ip(lam, lam + 2 * D.rho) / (2 * D.kappa) - D.ip(lam, D.x_form)
    ok = h == kw
    # h♮: the weight-1 generators J{v}, v in the Cartan subalgebra, take λ(v)
    for v in D.g.centralizer_basis(D.f, D.x):
        name = 'J{%s}' % v
        if D.g.is_in_cartan(v) and name in values:
            val = sum(cf * lam[int(str(m)[1:]) - 1] for m, cf in v.monomial_coefficients().items())
            ok = ok and values[name] == val
    if not ok:
        bad.append((g, f))
report('KW04 (6.14) for %d reductions' % len(cases), not bad, '%s %.1fs' % (bad, time.time() - t))

# 2. '-': Arakawa (285) and dot invariance; '+': γ_{λ - κρ^∨} -------------------------------------
t = time.time()
bad = []
for ct in (['A', 2], ['B', 2], ['G', 2]):
    V = affine(ct)
    D = _ReductionData(V, 'principal', None, None)
    lam = D.weight([a, b])
    vm, hm = D.top(D.miura_point(lam, '-'))
    D285 = (D.ip(lam + D.rho, lam + D.rho)
            - D.ip(D.rho - D.kappa * D.x_form, D.rho - D.kappa * D.x_form)) / (2 * D.kappa)
    ok = hm == D285
    simple = [tuple(1 if j == i else 0 for j in range(D.r)) for i in range(D.r)]
    for i, al in enumerate(simple):
        av = D.root_values[al]
        s = lam - (lam[i] + 1) * av                 # s_i ∘ λ (λ(h_i) = ⟨λ, α_i^∨⟩)
        ok = ok and D.top(D.miura_point(s, '-'))[0] == vm
        # '+': the values at λ depend on λ - κρ^∨ up to the dot action
        nu = lam - D.kappa * D.x_form
        nu_s = nu - (nu[i] + 1) * av
        ok = ok and D.top(nu_s + D.kappa * D.x_form)[0] == D.top(lam)[0]
    if not ok:
        bad.append(ct)
report("'-' (285), dot invariance, '+' = γ_{λ-κρ^∨}", not bad, '%s %.1fs' % (bad, time.time() - t))

# 3. Euler-Poincaré characters = W-Verma characters ----------------------------------------------
t = time.time()
bad = []
for g, f, lev, lam in [(['A', 1], 'principal', 1/2, [1/3]), (['A', 2], 'principal', 1/2, [1/3, 2]),
                       (['A', 2], 'minimal', -3/2, [1/2, -1]), ('spo(2|1)', 'minimal', 1/3, [1/2]),
                       ('sl(2|1)', 'minimal', 1/3, [1/2, 1/5])]:
    V = affine(g, lev, QQ)
    ep = V.quantum_hamiltonian_reduction_character(f, lam, 4)
    W = V.quantum_hamiltonian_reduction(f, conformal_vector='reduction')
    ch = W.verma_module(V.quantum_hamiltonian_reduction_highest_weight(f, lam)[0]).character(4)
    if ep != ch:
        bad.append((g, f, ep, ch))
report('Euler-Poincaré = W-Verma characters', not bad, '%s %.1fs' % (bad, time.time() - t))

# 4. the functor on Verma modules (symbolic) -------------------------------------------------------
t = time.time()
K2.<kk, aa> = Frac(QQ['kk, aa'])
V = UniversalAffineVertexAlgebra(K2, ['A', 1], level=kk)
M = V.induced_module(V.zhu_algebra().verma_module({'alphacheck[1]': aa}))
H = M.quantum_hamiltonian_reduction('principal')
ok = H.top_weight() == aa * (aa + 2) / (4 * (kk + 2)) - aa / 2
ok = ok and H.vertex_algebra().central_charge() == 1 - 6 * (kk + 1)^2 / (kk + 2)
Hm = M.quantum_hamiltonian_reduction('principal', reduction='-')
ok = ok and Hm.top_weight() == H.top_weight().subs(aa=-aa - 2 + (kk + 2))  # μ = -λ - 2ρ + κx
report('Verma modules over Q(k, λ)', ok, '%.1fs' % (time.time() - t))

# 5. Lee-Yang from the admissible sl_2 modules --------------------------------------------------
t = time.time()
V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-8/5)
A = V.zhu_algebra()
adm = [0, -2/5, -4/5, -6/5, -8/5]
res = {}
for red in ('+', '-'):
    tops = []
    for l in adm:
        Hl = V.induced_module(A.verma_module({'alphacheck[1]': l})).simple_quotient() \
              .quantum_hamiltonian_reduction('principal', reduction=red)
        tops.append(None if isinstance(Hl, ZeroModule) else Hl.top_weight())
        if tops[-1] is not None:
            res.setdefault(tops[-1], Hl.character(8))
    res[red] = tops
RR = {0: [1, 0, 1, 1, 1, 1, 2, 2], -1/5: [1, 1, 1, 1, 2, 2, 3, 3, 4]}
ok = res['+'] == [0, -1/5, -1/5, 0, None] and res['-'] == [None, 0, -1/5, -1/5, 0]
for h, dims in RR.items():
    ch = dict(zip(res[h].exponents(), res[h].coefficients()))
    ok = ok and all(ch.get(h + n, 0) == dims[n] for n in range(7))
report('Lee-Yang by + and - (Rogers-Ramanujan)', ok, '%s %.1fs' % (res['+'], time.time() - t))

# 6. the 3-state Potts model from sl_3 at k + 3 = 4/5 ('-') -----------------------------------------
t = time.time()
V = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=-3 + 4/5)
A = V.zhu_algebra()
kap = 4/5
tops = set()
D = _ReductionData(V, 'principal', None, None)
for l0 in [(0, 0), (1, 0), (0, 1)]:                          # integrable of level p - 3 = 1
    for m1 in range(5):
        for m2 in range(5 - m1):                              # fractional part, level u - 1 = 4
            lam = [l0[0] - kap * m1, l0[1] - kap * m2]
            if _reduction_is_nonzero(D, D.weight(lam), '-'):
                vals, h = V.quantum_hamiltonian_reduction_highest_weight('principal', lam, reduction='-')
                tops.add(h)
W = V.quantum_hamiltonian_reduction('principal', conformal_vector='reduction')
ok = W.central_charge() == 4/5 and tops == {0, 2/5, 2/3, 1/15}
report('Potts model (W_3, c = 4/5) by -', ok, '%s %.1fs' % (sorted(tops), time.time() - t))

# 7. Bershadsky-Polyakov: Arakawa's zero criterion and the modules ---------------------------------
t = time.time()
V = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=-3/2)
A = V.zhu_algebra()


def L3(l):
    return V.induced_module(A.verma_module({'alphacheck[1]': l[0], 'alphacheck[2]': l[1]})).simple_quotient()


zero = [l for l in [(-1/2, -1/2), (-1/2, -1), (0, -3/2), (-3/2, 0), (0, 0), (1/2, -2)]
        if isinstance(L3(l).quantum_hamiltonian_reduction('minimal'), ZeroModule)]
H = L3((-1/2, -1/2)).quantum_hamiltonian_reduction('minimal')
ok = zero == [(-1/2, -1), (0, -3/2), (-3/2, 0), (1/2, -2)] and H.top_weight() == \
    V.quantum_hamiltonian_reduction_highest_weight('minimal', [-1/2, -1/2])[1]
report("BP: H(L(λ)) = 0 iff k - (λ|θ) ∈ Z_>=0", ok, '%s %.1fs' % (zero, time.time() - t))

# 8. refusals ------------------------------------------------------------------------------------
t = time.time()
refused = []
V3 = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2)
V4 = UniversalAffineVertexAlgebra(QQ, ['A', 3], level=1/2)
Vw = V3.induced_module(V3.zhu_algebra().module({n: matrix(QQ, 1, 1, 0) for n in V3.zhu_algebra().variable_names()}))
M3 = V3.induced_module(V3.zhu_algebra().verma_module({'alphacheck[1]': 0, 'alphacheck[2]': 0}))
M4 = V4.induced_module(V4.zhu_algebra().verma_module({'alphacheck[%d]' % i: 0 for i in (1, 2, 3)}))
Vir = affine(['A', 1]).quantum_hamiltonian_reduction('principal', conformal_vector='reduction')
for name, call, exc in [
        ('Weyl module', lambda: Vw.quantum_hamiltonian_reduction('principal'), NotImplementedError),
        ('non-abelian g♮', lambda: M4.quantum_hamiltonian_reduction('minimal'), NotImplementedError),
        ("'-' minimal", lambda: M3.quantum_hamiltonian_reduction('minimal', reduction='-'), NotImplementedError),
        ("'+' principal off (385)", lambda: M3.simple_quotient().quantum_hamiltonian_reduction('principal'),
         NotImplementedError),
        ('non-affine V', lambda: Vir.verma_module(0).quantum_hamiltonian_reduction('principal'),
         NotImplementedError),
        ('other Borel', lambda: V3.induced_module(V3.zhu_algebra().verma_module(
            {'alphacheck[1]': 0, 'alphacheck[2]': 0}, functional=[1, -2])).quantum_hamiltonian_reduction('minimal'),
         NotImplementedError),
        ('bad reduction', lambda: M3.quantum_hamiltonian_reduction('minimal', reduction='x'), ValueError),
        ('bad weight', lambda: V3.quantum_hamiltonian_reduction_highest_weight('minimal', [1]), ValueError)]:
    try:
        call()
    except exc:
        refused.append(name)
    except Exception as e:
        print(name, 'raised', type(e).__name__, e)
report('refusals', len(refused) == 8, '%s %.1fs' % (refused, time.time() - t))

print('total %.1fs' % (time.time() - t_start))
report.result()
