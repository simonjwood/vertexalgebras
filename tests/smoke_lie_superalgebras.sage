# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Basic classical Lie superalgebras (lie_superalgebras.py): the dual Coxeter
# numbers of Kac-Wakimoto's Tables I-III (the normalisation (θ|θ) = 2 for
# their θ), dimensions and superdimensions, the simple components of the
# even part, the super Jacobi identity, super antisymmetry and the invariance
# of the form on all basis triples, other Dynkin diagrams of the same
# algebras, Lie algebras against Sage, TestSuite, pickling, refusals,
# nilpotents of given Jordan types (centraliser dimensions), and the
# affine vertex superalgebras (Sugawara c = k sdim/(k + h^∨), the
# constructor's own checks).
# Run from the repository root:  sage tests/smoke_lie_superalgebras.sage  -> RESULT: PASS
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.lie_superalgebras import LieSuperalgebra, _from_simple_roots
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
from axioms import Reporter

report = Reporter()
t_start = time.time()
L = LieSuperalgebra


def lsa(spec, **kw):
    return L(QQ, spec, **kw)


# 1. Kac-Wakimoto's dual Coxeter numbers -------------------------------------
table = []                                  # (algebra, h^∨ of the table)
# Table I: simple Lie algebras
table += [(lsa('sl(%d|0)' % n), n) for n in (2, 3, 4, 5)]
table += [(lsa('osp(%d|0)' % n), n - 2) for n in (5, 6, 7, 8)]
table += [(lsa('spo(%d|0)' % n), n / 2 + 1) for n in (2, 4, 6)]
table += [(lsa(CartanMatrix([t, r])), h) for t, r, h in
          [('G', 2, 4), ('F', 4, 9), ('E', 6, 12), ('E', 7, 18)]]
# Table II: θ in an sl_2 (or sp_2)
table += [(lsa('sl(2|%d)' % m), 2 - m) for m in (1, 3, 4)]
table += [(lsa('psl(2|2)'), 0)]
table += [(lsa('spo(2|%d)' % m), 2 - m / 2) for m in (1, 2, 3, 4, 5)]
table += [(lsa('osp(4|%d)' % m), 2 - m) for m in (2, 4)]
table += [(lsa('D(2,1;a)', a=a), 0) for a in (1, 2, QQ(1) / 3, QQ(-1) / 2)]
table += [(lsa('F(4)', theta='sl2'), -2), (lsa('G(3)', theta='sl2'), QQ(-3) / 2)]
# Table III
table += [(lsa('sl(%d|%d)' % (m, n)), m - n) for m in (3, 4) for n in (1, 2, 5)]
table += [(lsa('psl(%d|%d)' % (m, m)), 0) for m in (3, 4)]
table += [(lsa('spo(%d|%d)' % (n, m)), (n - m) / 2 + 1) for n in (4, 6) for m in (1, 2, 3)]
table += [(lsa('osp(%d|%d)' % (m, n)), m - n - 2) for m in (5, 6, 7) for n in (2, 4)]
table += [(lsa('F(4)', theta='so7'), 3), (lsa('G(3)', theta='G2'), 2)]
bad = [(str(g), g.dual_coxeter_number(), h) for g, h in table if g.dual_coxeter_number() != h]
report('h^∨ as in Kac-Wakimoto Tables I-III (%d algebras)' % len(table), not bad, bad)
bad = [str(g) for g, h in table if g.root_norm(g.theta()) != 2]
report('(θ|θ) = 2', not bad, bad)


# 2. dimensions, superdimensions, even parts ---------------------------------
def osp_dims(M, N):
    even = M * (M - 1) // 2 + N * (N + 1) // 2
    return even + M * N, even - M * N


dims = []
for m, n in [(2, 1), (3, 1), (3, 2), (4, 1), (2, 3)]:
    dims.append((lsa('sl(%d|%d)' % (m, n)), (m + n) ^ 2 - 1, (m - n) ^ 2 - 1))
for m in (2, 3, 4):
    dims.append((lsa('psl(%d|%d)' % (m, m)), 4 * m ^ 2 - 2, -2))
for M, N in [(4, 2), (5, 2), (5, 4), (6, 2), (8, 4)]:
    dims.append((lsa('osp(%d|%d)' % (M, N)),) + osp_dims(M, N))
for N, M in [(2, 1), (2, 2), (2, 3), (4, 1), (4, 3)]:
    dims.append((lsa('spo(%d|%d)' % (N, M)),) + osp_dims(M, N))
dims += [(lsa('D(2,1;a)', a=2), 17, 1), (lsa('F(4)', theta='sl2'), 40, 8),
         (lsa('G(3)', theta='sl2'), 31, 3)]
bad = [(str(g), g.dimension(), g.superdimension(), d, s) for g, d, s in dims
       if (g.dimension(), g.superdimension()) != (d, s)]
report('dimensions and superdimensions (%d algebras)' % len(dims), not bad, bad)


def types(g):
    return sorted(str(ct) for ct, _, _ in g.even_part_components())


comps = [(lsa('F(4)', theta='sl2'), ["['A', 1]", "['B', 3]"]),
         (lsa('G(3)', theta='G2'), ["['A', 1]", "['G', 2]"]),
         (lsa('D(2,1;a)', a=3), ["['A', 1]"] * 3),
         (lsa('osp(5|4)'), ["['B', 2]", "['B', 2]"]),        # so_5 = sp_4: Sage says B2
         (lsa('sl(3|2)'), ["['A', 1]", "['A', 2]"]),
         (lsa('psl(3|3)'), ["['A', 2]", "['A', 2]"]),
         (lsa('osp(8|2)'), ["['A', 1]", "['D', 4]"])]
bad = [(str(g), types(g)) for g, t in comps if types(g) != t]
report('simple components of the even part', not bad, bad)


# 3. axioms on all basis triples (raw structure constants) -------------------
def axioms_ok(g):
    sc = g.structure_coefficients()
    keys = list(g.basis().keys())
    par = {k: g.basis()[k].is_even_odd() for k in keys}
    G = g.invariant_form_matrix()
    idx = {k: i for i, k in enumerate(keys)}

    def br(x, y):                       # x, y dicts
        out = {}
        for a, c in x.items():
            for b, d in y.items():
                for z, e in sc.get((a, b), {}).items():
                    out[z] = out.get(z, 0) + c * d * e
        return {k: v for k, v in out.items() if v}
    for a in keys:
        for b in keys:
            s = (-1) ** (par[a] * par[b])
            ab, ba = br({a: 1}, {b: 1}), br({b: 1}, {a: 1})
            if ab != {k: -s * v for k, v in ba.items()}:
                return 'antisymmetry', a, b
            if any(par[z] != (par[a] + par[b]) % 2 for z in ab):
                return 'parity', a, b
            if G[idx[a], idx[b]] != s * G[idx[b], idx[a]]:
                return 'supersymmetric form', a, b
            for c in keys:
                lhs = br({a: 1}, br({b: 1}, {c: 1}))
                r1 = br(ab, {c: 1})
                r2 = br({b: 1}, br({a: 1}, {c: 1}))
                rhs = dict(r1)
                for k, v in r2.items():
                    rhs[k] = rhs.get(k, 0) + s * v
                if lhs != {k: v for k, v in rhs.items() if v}:
                    return 'jacobi', a, b, c
                # invariance ([a, b] | c) = (a | [b, c])
                bc = br({b: 1}, {c: 1})
                if (sum(v * G[idx[k], idx[c]] for k, v in ab.items())
                        != sum(v * G[idx[a], idx[k]] for k, v in bc.items())):
                    return 'invariance', a, b, c
    return True


Ka.<a> = Frac(QQ['a'])
small = [lsa('sl(2|1)'), lsa('spo(2|1)'), lsa('psl(2|2)'), lsa('osp(4|2)'),
         lsa('sl(3|1)'), lsa('spo(2|3)'), lsa('D(2,1;a)', a=2),
         L(Ka, 'D(2,1;a)', a=a), lsa('G(3)', theta='sl2')]
res = [(str(g), axioms_ok(g)) for g in small]
report('super antisymmetry, parity, super Jacobi, invariant supersymmetric form '
       'on all basis triples', all(r is True for _, r in res), res)

# 4. other Dynkin diagrams, D(2,1;1) = osp(4|2), Lie algebras vs Sage --------
H = L(QQ, matrix([[0, 1], [1, 0]]), odd=[1, 2], theta=(1, 1))      # sl(2|1), both odd
A, odd = _from_simple_roots([[-1, 1], [1, 0]], [-1, 1])            # osp(3|2): ε - δ, δ
O = L(QQ, A, odd=[i + 1 for i in odd], theta=(0, 2))
S = lsa('spo(2|3)')
D1, O42 = lsa('D(2,1;a)', a=1), lsa('osp(4|2)')
report('other diagrams: sl(2|1) with two odd roots, osp(3|2) with an odd '
       'non-isotropic root (= spo(2|3)); D(2,1;1) = osp(4|2)',
       (H.dimension(), H.superdimension(), H.dual_coxeter_number()) == (8, 0, 1)
       and O.odd_simple_roots() == (1, 2)
       and (O.dimension(), O.superdimension(), O.dual_coxeter_number())
       == (S.dimension(), S.superdimension(), S.dual_coxeter_number())
       and types(O) == types(S) and axioms_ok(O) is True and axioms_ok(H) is True
       and (D1.dimension(), D1.dual_coxeter_number(), types(D1))
       == (O42.dimension(), O42.dual_coxeter_number(), types(O42)))
sage_ok = []
for t, r in [('A', 2), ('B', 3), ('C', 3), ('G', 2), ('D', 4)]:
    g = lsa(CartanMatrix([t, r]))
    s = LieAlgebra(QQ, cartan_type=[t, r])
    sage_ok.append(g.dimension() == s.dimension()
                   and g.dual_coxeter_number() == CartanType([t, r]).dual_coxeter_number()
                   and len(g.positive_roots()) == len(list(RootSystem([t, r]).root_lattice().positive_roots())))
report('Lie algebras: dimension, h^∨, positive roots as Sage', all(sage_ok), sage_ok)

# 5. TestSuite, pickling, refusals -------------------------------------------
suites = []
for g in [lsa('sl(2|1)'), lsa('psl(2|2)'), lsa('spo(4|1)'), lsa('osp(5|2)'),
          lsa('D(2,1;a)', a=2), L(Ka, 'D(2,1;a)', a=a), lsa('F(4)', theta='so7'),
          lsa('G(3)', theta='sl2'), H, O]:
    try:
        TestSuite(g).run(raise_on_failure=True)
    except Exception as err:
        suites.append((str(g), str(err)[:200]))
report('TestSuite (super antisymmetry, Jacobi, parity, invariant form, pickling)',
       not suites, suites)
refusals = []
for args, kw in [(('F(4)',), {}), (('G(3)',), {'theta': 'so7'}), (('osp(3|2)',), {}),
                 (('sl(1|2)',), {}), (('sl(2|2)',), {}), (('psl(2|3)',), {}),
                 (('D(2,1;a)',), {'a': -1}), ((matrix([[2, -2], [-2, 2]]),), {}),
                 ((matrix([[2, 0], [0, 2]]),), {})]:
    try:
        L(QQ, *args, **kw)
        refusals.append(args)
    except ValueError:
        pass
report('refused: F(4)/G(3) without the choice of θ, osp(3|n), sl(1|n), sl(m|m), '
       'psl(m|n), D(2,1;-1), affine and decomposable Cartan matrices',
       not refusals, refusals)

# 7. nilpotents, sl_2-triples, good gradings, centralizers -------------------
def minimal_data(g):
    """(dim g♮ even|odd, dim g_{-1/2} even|odd, checks) for the minimal grading."""
    f = g.minimal_nilpotent()
    e, h, f = g.sl2_triple(f)
    x = h / 2
    gr = g.ad_eigenvalues(x)
    ok = (e.bracket(f) == h and h.bracket(e) == 2 * e and h.bracket(f) == -2 * f
          and g.is_in_cartan(h) and set(gr.values()) <= {-1, QQ(-1) / 2, 0, QQ(1) / 2, 1}
          and sum(1 for v in gr.values() if v == 1) == 1 and g.is_good_grading(x, f))
    C = g.centralizer_basis(f, x)
    par = g.basis()

    def dims(j, space):
        keys = [k for k, v in gr.items() if v == j]
        return (sum(1 for k in keys if not par[k].is_even_odd()),
                sum(1 for k in keys if par[k].is_even_odd()))
    nat = [c for c in C if gr[c.leading_support()] == 0]
    nat_d = (sum(1 for c in nat if not c.is_even_odd()), sum(1 for c in nat if c.is_even_odd()))
    half = dims(QQ(-1) / 2, None)
    # dim g^f = dim g_0 + dim g_{1/2}, per parity (good grading)
    g0, gh = dims(0, None), dims(QQ(1) / 2, None)
    ok = ok and all(sum(1 for c in C if c.is_even_odd() == q) == g0[q] + gh[q] for q in (0, 1))
    return nat_d, half, ok


def osp_even(M, N):
    return M * (M - 1) // 2 + N * (N + 1) // 2


kw = []   # (algebra, dim g♮ (even, odd), dim g_{-1/2} (even, odd)) from KW Tables I-III
for n in (3, 4, 5):
    kw.append((lsa('sl(%d|0)' % n), ((n - 2) ^ 2, 0), (2 * (n - 2), 0)))
for n in (5, 7, 8):
    kw.append((lsa('osp(%d|0)' % n), (3 + (n - 4) * (n - 5) // 2, 0), (2 * (n - 4), 0)))
for n in (4, 6):
    kw.append((lsa('spo(%d|0)' % n), ((n - 2) * (n - 1) // 2, 0), (n - 2, 0)))
for t, r, nat, half in [('G', 2, 3, 4), ('F', 4, 21, 14), ('E', 6, 35, 20), ('E', 7, 66, 32)]:
    kw.append((lsa(CartanMatrix([t, r])), (nat, 0), (half, 0)))
for m in (1, 3):
    kw.append((lsa('sl(2|%d)' % m), (m ^ 2, 0), (0, 2 * m)))
kw.append((lsa('psl(2|2)'), (3, 0), (0, 4)))
for m in (1, 2, 3):
    kw.append((lsa('spo(2|%d)' % m), (m * (m - 1) // 2, 0), (0, m)))
for m in (2, 4):
    kw.append((lsa('osp(4|%d)' % m), (3 + m * (m + 1) // 2, 0), (0, 2 * m)))
kw += [(lsa('D(2,1;a)', a=3), (6, 0), (0, 4)), (lsa('F(4)', theta='sl2'), (21, 0), (0, 8)),
       (lsa('G(3)', theta='sl2'), (14, 0), (0, 7))]
for m, n in [(3, 1), (3, 2), (4, 1)]:
    kw.append((lsa('sl(%d|%d)' % (m, n)), ((m - 2) ^ 2 + n ^ 2, 2 * (m - 2) * n), (2 * (m - 2), 2 * n)))
kw.append((lsa('psl(3|3)'), (1 + 9 - 1, 2 * 3), (2, 6)))     # sl(1|3)
for n, m in [(4, 1), (4, 2), (6, 1)]:
    kw.append((lsa('spo(%d|%d)' % (n, m)), (osp_even(m, n - 2), m * (n - 2)), (n - 2, m)))
for m, n in [(5, 2), (6, 2), (7, 4)]:
    kw.append((lsa('osp(%d|%d)' % (m, n)), (osp_even(m - 4, n) + 3, (m - 4) * n), (2 * (m - 4), 2 * n)))
kw += [(lsa('F(4)', theta='so7'), (9, 8), (6, 4)), (lsa('G(3)', theta='G2'), (6, 6), (4, 4))]
bad = []
for g, nat, half in kw:
    got = minimal_data(g)
    if got != (nat, half, True):
        bad.append((str(g), got, nat, half))
report('minimal gradings (x = θ/2): sl_2-triples with h in the Cartan subalgebra, good, '
       'dim g^f = dim g_0 + dim g_{1/2}; g♮ and g_{-1/2} as in KW Tables I-III (%d rows)'
       % len(kw), not bad, bad)
prin = []
for t, r, exps in [('A', 3, [1, 2, 3]), ('B', 3, [1, 3, 5]), ('C', 3, [1, 3, 5]),
                   ('G', 2, [1, 5]), ('D', 4, [1, 3, 3, 5]), ('F', 4, [1, 5, 7, 11])]:
    g = lsa(CartanMatrix([t, r]))
    f = g.principal_nilpotent()
    x = g.dynkin_grading(f)
    C = g.centralizer_basis(f, x)
    prin.append(g.is_good_grading(x, f) and g.is_in_cartan(x)
                and sorted(-g.ad_eigenvalues(x)[c.leading_support()] for c in C) == exps)
for spec, kw2 in [('sl(2|1)', {}), ('spo(2|3)', {}), ('osp(5|2)', {}), ('F(4)', {'theta': 'sl2'}),
                  ('D(2,1;a)', {'a': 2})]:
    g = lsa(spec, **kw2)
    f = g.principal_nilpotent()
    x = g.dynkin_grading(f)
    prin.append(g.is_good_grading(x, f) and g.is_in_cartan(x))
report('principal nilpotents: Lie algebras: g^f in the degrees -exponents (A3, B3, C3, G2, '
       'D4, F4); superalgebras: the Dynkin grading of the principal nilpotent of g_0 is good',
       all(prin), prin)
g3 = lsa(CartanMatrix(['A', 2]))
f = g3.root_vector((1, 0), -1) + g3.root_vector((1, 1), -1)        # minimal, h not Cartan
e, h, f = g3.sl2_triple(f)
refused = []
for bad_f in [g3.h(1), g3.zero(), lsa('sl(2|1)').e(2)]:
    try:
        g3.sl2_triple(bad_f) if bad_f.parent() is g3 else lsa('sl(2|1)').sl2_triple(bad_f)
        refused.append(str(bad_f))
    except ValueError:
        pass
report('sl_2-triples outside the Cartan subalgebra (sl_3, f = E[-α1] + E[-α1-α2]: dim g^f = 4); '
       'refused: not nilpotent, zero, odd',
       e.bracket(f) == h and h.bracket(e) == 2 * e and h.bracket(f) == -2 * f
       and not g3.is_in_cartan(h) and len(g3.centralizer_basis(f)) == 4 and not refused, refused)

# 5b. nilpotents of a given Jordan type -----------------------------------------
# every Jordan type of small classical algebras: the internal check (h in the
# Cartan subalgebra with the eigenvalues of the Jordan type) and, independently,
# dim g^f: gl: Σ_{i,j} min(λ_i, λ_j); so / sp: (Σ min(λ_i, λ_j) ∓ #odd parts)/2;
# the odd part of osp: Σ min(λ_i, μ_j) (V_0 ⊗ V_1), of sl(m|n): twice that;
# sl: minus 1 (the supertrace), psl(m|m): minus 2 (also the identity)
from collections import Counter


def summin(a, b):
    return sum(min(x, y) for x in a for y in b)


def jordan_types(kind, size):
    out = []
    for p in Partitions(size):
        c = Counter(p)
        if kind == 'BD' and any(c[x] % 2 for x in c if x % 2 == 0):
            continue
        if kind == 'C' and any(c[x] % 2 for x in c if x % 2):
            continue
        out.append(list(p))
    return out


def cent_dim(kind, lam):
    odd = sum(1 for x in lam if x % 2)
    return {'A': summin(lam, lam), 'BD': (summin(lam, lam) - odd) / 2,
            'C': (summin(lam, lam) + odd) / 2}[kind]


jt_cases = ([(L(QQ, CartanMatrix(['A', r])), [('A', r + 1)], -1) for r in range(1, 6)]
            + [(L(QQ, CartanMatrix(['B', r])), [('BD', 2 * r + 1)], 0) for r in range(2, 5)]
            + [(L(QQ, CartanMatrix(['C', r])), [('C', 2 * r)], 0) for r in range(2, 5)]
            + [(L(QQ, CartanMatrix(['D', r])), [('BD', 2 * r)], 0) for r in range(3, 6)]
            + [(lsa('sl(4|0)'), [('A', 4)], -1), (lsa('osp(9|0)'), [('BD', 9)], 0),
               (lsa('spo(6|0)'), [('C', 6)], 0), (lsa('sl(3|1)'), [('A', 3), ('A', 1)], -1),
               (lsa('sl(3|2)'), [('A', 3), ('A', 2)], -1), (lsa('sl(2|3)'), [('A', 2), ('A', 3)], -1),
               (lsa('psl(2|2)'), [('A', 2), ('A', 2)], -2), (lsa('psl(3|3)'), [('A', 3), ('A', 3)], -2),
               (lsa('osp(4|2)'), [('BD', 4), ('C', 2)], 0), (lsa('osp(5|4)'), [('BD', 5), ('C', 4)], 0),
               (lsa('osp(6|2)'), [('BD', 6), ('C', 2)], 0), (lsa('spo(2|1)'), [('C', 2), ('BD', 1)], 0),
               (lsa('spo(2|3)'), [('C', 2), ('BD', 3)], 0), (lsa('spo(4|3)'), [('C', 4), ('BD', 3)], 0),
               (lsa('spo(4|2)'), [('C', 4), ('BD', 2)], 0)])
bad, count = [], 0
for g, blocks, corr in jt_cases:
    lists = [jordan_types(kind, size) for kind, size in blocks]
    for parts in cartesian_product_iterator(lists):
        if all(x == 1 for p in parts for x in p):
            continue
        try:
            f = g.nilpotent_of_jordan_type(*parts)
        except (ValueError, RuntimeError) as err:
            bad.append((str(g), parts, str(err)))
            continue
        count += 1
        want = sum(cent_dim(kind, p) for (kind, _), p in zip(blocks, parts)) + corr
        if len(parts) == 2:
            want += (2 if blocks[0][0] == 'A' else 1) * summin(*parts)
        x = g.dynkin_grading(f)
        if len(g.centralizer_basis(f)) != want or not g.is_good_grading(x, f):
            bad.append((str(g), parts, len(g.centralizer_basis(f)), want))
report('nilpotents of every Jordan type (%d, in %d algebras: sl_2..sl_6, B_2..B_4, '
       'C_2..C_4, D_3..D_5, sl(m|n), psl(2|2), psl(3|3), osp, spo): h in the Cartan '
       'subalgebra with the eigenvalues of the Jordan type, dim g^f by the formulas, a good '
       'Dynkin grading' % (count, len(jt_cases)), not bad, bad)
prin = []
for ct, lam in [(['A', 4], [5]), (['B', 3], [7]), (['C', 3], [6]), (['D', 4], [7, 1])]:
    g = L(QQ, CartanMatrix(ct))
    prin.append(g.nilpotent_of_jordan_type(lam) == g.principal_nilpotent())


def graded_dims(g, f):
    """The dimensions of the even and odd parts of the g_j of the Dynkin grading
    (an invariant of the orbit)."""
    gr = g.ad_eigenvalues(g.dynkin_grading(f))
    return Counter((j, g.basis()[a].is_even_odd()) for a, j in gr.items())


for spec, lam, mu in [('sl(3|1)', [2, 1], [1]), ('spo(2|3)', [2], [1, 1, 1]),
                      ('osp(5|2)', [2, 2, 1], [1, 1]), ('sl(2|3)', [2], [1, 1, 1])]:
    g = lsa(spec)
    prin.append(graded_dims(g, g.nilpotent_of_jordan_type(lam, mu))
                == graded_dims(g, g.minimal_nilpotent()))
refused = []
B3r = L(QQ, CartanMatrix(['B', 3])).cartan_matrix()[[2, 1, 0], [2, 1, 0]]
for g, args in [(L(QQ, CartanMatrix(['A', 3])), ([3, 2],)), (L(QQ, CartanMatrix(['B', 2])), ([4, 1],)),
                (L(QQ, CartanMatrix(['C', 2])), ([3, 1],)), (L(QQ, CartanMatrix(['A', 2])), ([2, 1], [1])),
                (lsa('sl(2|1)'), ([1, 1], [1])), (lsa('sl(2|1)'), ([2, 0],)),
                (L(QQ, CartanMatrix(['G', 2])), ([2],)), (L(QQ, B3r), ([7],)),
                (lsa('F(4)', theta='so7'), ([3],))]:
    try:
        g.nilpotent_of_jordan_type(*args)
        refused.append((str(g), args))
    except ValueError:
        pass
report('Jordan types (n), (2r+1), (2r), (2r-1, 1) give the principal nilpotents of A, B, C, D; '
       'the minimal gradings of sl(3|1), osp(3|2), osp(5|2), sl(2|3); refused: wrong sizes, not orthogonal / '
       'symplectic, a second partition for a Lie algebra, f = 0, a zero part, G_2, F(4), a '
       'relabelled B_3', all(prin) and not refused, (prin, refused))

# 6. affine vertex superalgebras ------------------------------------------------
K.<k> = Frac(QQ['k'])
cs = []
for g in [lsa('sl(2|1)'), lsa('spo(2|1)'), lsa('spo(2|2)'), lsa('psl(2|2)'),
          lsa('osp(4|2)'), lsa('sl(3|1)'), lsa('F(4)', theta='sl2'),
          lsa('G(3)', theta='G2')]:
    V = UniversalAffineVertexAlgebra(K, g, level=k, conformal_vector='sugawara')
    cs.append((str(g), V.central_charge() == k * g.superdimension() / (k + g.dual_coxeter_number())))
Kak.<aa, kk> = Frac(QQ['aa', 'kk'])
VD = UniversalAffineVertexAlgebra(Kak, L(Frac(QQ['aa']), 'D(2,1;a)', a=Frac(QQ['aa']).gen()),
                                  level=kk, conformal_vector='sugawara')
VS = UniversalAffineVertexAlgebra(K, lsa('spo(2|1)'), level=k, conformal_vector='sugawara')
gS = lsa('spo(2|1)')
x, y = VS.gen('E[alpha[1]]'), VS.gen('E[-alpha[1]]')
xg, yg = gS.e(1), gS.f(1)
report('affine: Sugawara c = k sdim/(k + h^∨) (the constructor checks Jacobi and '
       'invariance), D(2,1;a) over Q(a, k): c = 1, root grading, pickling',
       all(ok for _, ok in cs) and VD.central_charge() == 1
       and x.horizontal_grade() == VS.horizontal_grading_group()((1,)) and loads(dumps(VS)) is VS, cs)
report('affine osp(1|2): x x ~ [x, x]/(z-w), x y ~ k (x|y)/(z-w)^2 + [x, y]/(z-w)',
       x.ope(x) == [VS.current(xg.bracket(xg))]
       and x.ope(y) == [VS.current(xg.bracket(yg)), k * gS.invariant_form(xg, yg) * VS.one()])

print('total %.1fs' % (time.time() - t_start))
report.result()
