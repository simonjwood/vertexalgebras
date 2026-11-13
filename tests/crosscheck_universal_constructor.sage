# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# UniversalVertexAlgebra built from OPEs written as normally ordered
# expressions (letters out of order, lower orientations: normalised by a
# private algebra evaluating them on demand) against references written
# directly in basis monomials and the upper orientation (coefficients that
# need reordering computed inside an algebra that already has the OPEs they
# need, e.g. ∂Λ in the Virasoro algebra; Virasoro vectors computed in the
# reference algebra): the same products and conformal data in every mode
# (C++ engine, Cython core, pure-Python fallback; the normalisation runs in
# the same mode), the same affine algebras, canonical data that normalise
# to themselves, pickles that keep everything (also in a new process),
# verify_jacobi, the cycle error, the refused setters.
# Run from the repository root:
#   sage tests/crosscheck_universal_constructor.sage  -> RESULT: PASS
import os
import subprocess
import sys
import tempfile

from vertex_algebras import universal_vertex_algebra as U
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra

checked = mismatches = 0


def check(ok, what):
    global checked, mismatches
    checked += 1
    if not ok:
        mismatches += 1
        print('MISMATCH', what)


def raw(x):
    if isinstance(x, (list, tuple)):
        return [raw(y) for y in x]
    if isinstance(x, dict):
        return {k: raw(v) for k, v in x.items()}
    return dict(x._monomial_coefficients)


# the implementations (the staging algebra normalising out-of-order
# expressions always runs in pure Python)
MODES = (['engine'] if U._UVA_ENGINE is not None else []) + ['python']


def reference(R, gens, opes, parities=None, mode=None, T=None):
    """The reference: OPEs given in the upper orientation with coefficients
    in basis monomials (``opes``: a dict, no normalisation needed), the
    Virasoro vector ``T(*generators)`` computed in that algebra.  The
    algebra written otherwise must be the SAME object (equal canonical
    data, same implementation)."""
    V = UniversalVertexAlgebra(R, gens, opes, parities=parities,
                               implementation=mode)
    if T is not None:
        V = V.with_conformal_vector(T(*V.gens()))
    return V


def compare(old, new, elements, what):
    """Products of ``elements(algebra)`` (a list) in both algebras."""
    xs, ys = elements(old), elements(new)
    for i, (a, b) in enumerate(zip(xs, ys)):
        for j, (c, d) in enumerate(zip(xs, ys)):
            check(raw(a.ope(c)) == raw(b.ope(d)), '{}: OPE {} {}'.format(what, i, j))
            check(raw(a.no(c)) == raw(b.no(d)), '{}: NO {} {}'.format(what, i, j))
    check(raw(xs[-1].derivative(3)) == raw(ys[-1].derivative(3)), what + ': derivative')


# ---------------------------------------------------------------------------
# W_3 over Q(c): ∂Λ written in the expressions (out of PBW order) against
# the old two-step route (Λ computed after setting the Virasoro OPE)
K.<c> = Frac(QQ['c'])
beta = 16 / (22 + 5 * c)


def w3_reference(mode):
    """∂Λ computed in the Virasoro algebra (only the LL OPE is needed), its
    basis monomials taken as the WW coefficients of W_3."""
    Vir = {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, c / 2],
           ('L', 'W'): [{(('W', 1),): 1}, {(('W', 0),): 3}]}
    V0 = UniversalVertexAlgebra(K, ['L', 'W'], Vir, implementation=mode)
    L, W = V0.gens()
    Lam = L.no(L) - 3 / 10 * L.derivative(2)
    return V0.with_opes(Vir | {('W', 'W'): [L.derivative(3) / 15 + beta * Lam.derivative(),
                                            3 / 10 * L.derivative(2) + 2 * beta * Lam,
                                            L.derivative(), 2 * L, 0, c / 3]})


E.<L, W> = NormallyOrderedExpressions(K)
Lam = L * L - 3 / 10 * L.derivative(2)
w3_new = {('L', 'L'): [L.derivative(), 2 * L, 0, c / 2],
          ('W', 'L'): [2 * W.derivative(), 3 * W],         # the other orientation
          ('W', 'W'): [L.derivative(3) / 15 + beta * Lam.derivative(),
                       3 / 10 * L.derivative(2) + 2 * beta * Lam,
                       L.derivative(), 2 * L, 0, c / 3]}


def w3_elements(V):
    L, W = V.gens()
    return [L, W, W.no(L.derivative()), L.no(W).no(L)]


for mode in MODES:
    old = w3_reference(mode)
    new = UniversalVertexAlgebra(K, E, w3_new, conformal_vector=L, implementation=mode)
    check(raw(old.opes()) == raw(new.opes()), 'W3 declared OPEs, ' + mode)
    check(new.with_conformal_vector(None) is old, 'W3: the same algebra, ' + mode)
    compare(old, new, w3_elements, 'W3 ' + mode)
    check(new.degrees() == {'L': 2, 'W': 3} and new.central_charge() == c,
          'W3 conformal data, ' + mode)
    check(new.satisfies_jacobi_identity(), 'W3 Jacobi, ' + mode)
print('W3: modes', MODES)

# the same algebra however written; canonical data normalise to themselves
V3 = UniversalVertexAlgebra(K, E, w3_new, conformal_vector=L)
d = V3._data
check(UniversalVertexAlgebra(K, ['L', 'W'], d['opes'], conformal_vector=d['T']) is V3,
      'W3 canonical data')
check(UniversalVertexAlgebra(K, ['L', 'W'], old.opes(), conformal_vector='L') is V3,
      'W3 from the elements of the reference algebra')
check(loads(dumps(V3)) is V3, 'W3 pickle')

# verify_jacobi: Zamolodchikov's beta passes, another one fails
try:
    UniversalVertexAlgebra(K, E, w3_new, verify_jacobi=True)
    ok = True
except ValueError:
    ok = False
check(ok, 'W3 verify_jacobi')
bad = dict(w3_new)
bad[('W', 'W')] = [L.derivative(3) / 15 + Lam.derivative(),
                   3 / 10 * L.derivative(2) + 2 * Lam, L.derivative(), 2 * L, 0, c / 3]
try:
    UniversalVertexAlgebra(K, E, bad, verify_jacobi=True)
    got = None
except ValueError as err:
    got = str(err)
check(got is not None and got.startswith('the OPEs violate'), 'W3 wrong beta: %s' % got)

# ---------------------------------------------------------------------------
# N=1 over Q(c) (odd G), betagamma and bc written in the other orientation,
# against the upper orientation (skew-symmetry by hand)


def n1_opes(L, G):
    return {('L', 'L'): [L.derivative(), 2 * L, 0, c / 2],
            ('G', 'L'): [G.derivative() / 2, 3 / 2 * G],
            ('G', 'G'): [2 * L, 0, 2 * c / 3]}


def bg_opes(b, g):
    return {('g', 'b'): [1]}


def bc_opes(b, cc):
    return {('c', 'b'): [1]}


upper = {'N=1': {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, c / 2],
                 ('L', 'G'): [{(('G', 1),): 1}, {(('G', 0),): 3 / 2}],
                 ('G', 'G'): [{(('L', 0),): 2}, 0, 2 * c / 3]},
         'bg': {('b', 'g'): [-1]},
         'bc': {('b', 'c'): [1]}}

cases = [('N=1', K, ['L', 'G'], {'G': 1}, n1_opes,
          lambda V: [V.gen(0), V.gen(1), V.gen(1).no(V.gen(0).derivative())],
          lambda L, G: L),
         ('bg', QQ, ['b', 'g'], None, bg_opes,
          lambda V: [V.gen(0), V.gen(1), V.gen(0).no(V.gen(1).derivative(2))],
          lambda b, g: -b.no(g.derivative())),
         ('bc', QQ, ['b', 'c'], {'b': 1, 'c': 1}, bc_opes,
          lambda V: [V.gen(0), V.gen(1), V.gen(1).derivative().no(V.gen(0))],
          lambda b, cc: -b.no(cc.derivative()))]
for name, R, gens, par, opes, els, T in cases:
    F = NormallyOrderedExpressions(R, gens, parities=par)
    for mode in MODES:
        old = reference(R, gens, upper[name], parities=par, mode=mode, T=T)
        new = UniversalVertexAlgebra(R, F, opes(*F.gens()), implementation=mode,
                                     conformal_vector=T(*F.gens()))
        check(new is old, '{} {}: the same algebra'.format(name, mode))
        compare(old, new, els, '{} {}'.format(name, mode))
        check(raw(old.conformal_vector()) == raw(new.conformal_vector())
              and old.degrees() == new.degrees()
              and old.central_charge() == new.central_charge(),
              '{} {}: conformal data'.format(name, mode))
print('N=1, bg, bc: modes', MODES)

# ---------------------------------------------------------------------------
# affine sl2 (('H', 'E') declared against the generator order) with the
# Sugawara vector written with F*E, and affine osp(1|2) against the affine
# constructor
Q.<k> = Frac(QQ['k'])


def sl2_opes(E_, H, F_):
    return {('H', 'H'): [0, 2 * k], ('H', 'E'): [2 * E_], ('H', 'F'): [-2 * F_],
            ('E', 'F'): [H, k]}


sl2_upper = {('H', 'H'): [0, 2 * k], ('E', 'H'): [{(('E', 0),): -2}],
             ('H', 'F'): [{(('F', 0),): -2}], ('E', 'F'): ['H', k]}


def sug(E_, H, F_):
    return (H * H / 2 + E_ * F_ + F_ * E_) / (2 * (k + 2))


def sug_old(E_, H, F_):
    return (H.no(H) / 2 + E_.no(F_) + F_.no(E_)) / (2 * (k + 2))


G = NormallyOrderedExpressions(Q, ['E', 'H', 'F'])
for mode in MODES:
    old = reference(Q, ['E', 'H', 'F'], sl2_upper, mode=mode, T=sug_old)
    new = UniversalVertexAlgebra(Q, G, sl2_opes(*G.gens()), implementation=mode,
                                 conformal_vector=sug(*G.gens()),
                                 horizontal_grading={'E': 1, 'F': -1})
    check(new.with_horizontal_grading(None) is old, 'sl2: the same algebra, ' + mode)
    compare(old, new, lambda V: [V.gen(0), V.gen(1), V.gen(2),
                                 V.gen(2).no(V.gen(0)).no(V.gen(1))], 'sl2 ' + mode)
    check(raw(old.conformal_vector()) == raw(new.conformal_vector())
          and new.central_charge() == 3 * k / (k + 2), 'sl2 Sugawara, ' + mode)
    check(new.gen(2).no(new.gen(0)).horizontal_grade() == 0, 'sl2 grading, ' + mode)
A = UniversalAffineVertexAlgebra(Q, ['A', 1], level=k, names=['E', 'H', 'F'], conformal_vector='sugawara')
check(isinstance(A, UniversalVertexAlgebra), 'affine is universal')
print('sl2: modes', MODES)

osp = {('h', 'e'): {'e': 2}, ('h', 'f'): {'f': -2}, ('e', 'f'): {'h': 1},
       ('h', 'x'): {'x': 1}, ('h', 'y'): {'y': -1}, ('e', 'y'): {'x': -1},
       ('f', 'x'): {'y': -1}, ('x', 'x'): {'e': 2}, ('x', 'y'): {'h': 1},
       ('y', 'y'): {'f': -2}}
form = {('e', 'f'): 1, ('h', 'h'): 2, ('x', 'y'): 2}
names = ['e', 'h', 'f', 'x', 'y']
Aosp = UniversalAffineVertexAlgebra(QQ, osp, form, level=1, parities={'x': 1, 'y': 1},
                                    names=names, conformal_vector=None)
O = NormallyOrderedExpressions(QQ, names, parities={'x': 1, 'y': 1})
og = dict(zip(names, O.gens()))
osp_opes = {}
for (a, b), br in osp.items():
    p0 = sum(cf * og[t] for t, cf in br.items())
    f = form.get((a, b), 0) or form.get((b, a), 0) * (-1 if a in 'xy' and b in 'xy' else 1)
    osp_opes[(a, b)] = [p0, f] if f else [p0]
osp_opes[('h', 'h')] = [0, 2]
Vosp = UniversalVertexAlgebra(QQ, O, osp_opes, degrees={s: 1 for s in names},
                              verify_jacobi=True)
check(raw(Vosp.opes()) == raw(Aosp.opes()), 'osp(1|2): constructor = affine')
check(Vosp._data['opes'] == Aosp._data['opes'], 'osp(1|2): same canonical data')

# ---------------------------------------------------------------------------
# errors
P.<a> = NormallyOrderedExpressions(QQ)
try:
    UniversalVertexAlgebra(QQ, P, {('a', 'a'): [a * a.derivative()]})
    got = None
except ValueError as err:
    got = str(err)
check(got is not None and 'needed to normally order itself' in got, 'cycle: %s' % got)
Hs = UniversalVertexAlgebra(QQ, ['J'])
check(not any(hasattr(Hs, f) for f in ('set_opes', 'set_virasoro',
                                        'set_conformal_weights', 'set_grading')),
      'no setters')

# ---------------------------------------------------------------------------
# a pickle loads in a new process with OPEs, weights, Virasoro vector, grading
S = UniversalVertexAlgebra(Q, G, sl2_opes(*G.gens()), conformal_vector=sug(*G.gens()),
                           horizontal_grading={'E': 1, 'F': -1})
x = S.gen(2).no(S.gen(0))
want = str((x.ope(x), S.central_charge(), S.degrees(), x.horizontal_grade()))
with tempfile.TemporaryDirectory() as tmp:
    fn = os.path.join(tmp, 'S.sobj')
    save((S, x), fn)
    code = ("S, x = load({!r}); "
            "print(str((x.ope(x), S.central_charge(), S.degrees(), x.horizontal_grade())))"
            .format(fn))
    out = subprocess.run(['sage', '-c', code], capture_output=True, text=True)
got = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:]
check(got == want, 'pickle in a new process: %s' % got)

print('crosscheck constructor: %d checks, mismatches: %d' % (checked, mismatches))
print('RESULT:', 'PASS' if mismatches == 0 else 'FAIL')
