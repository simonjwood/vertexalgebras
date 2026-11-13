# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Jacobi identities of declared OPEs (jacobi_identity_failures against a
# brute-force check, consistent and inconsistent algebras, skew-symmetry,
# the constructor's verify_jacobi=True) and the OPE bootstrap
# (bootstrap.py): free boson, W_3 (generic c and c as a parameter), W(2,4)
# (square-root coupling), W(2,5) (isolated central charges), W(2,3,4), N=1
# and N=2 superconformal, affine sl2 with a grading, ansatz terms, sorted ==
# all triples, serial == parallel, separate internal algebras, a new
# internal algebra per elimination, zero_pairs, save / load (resumed runs,
# refused files), refusals.
# Run from the repository root:  sage tests/smoke_bootstrap.sage  -> RESULT: PASS
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import vertex_algebra
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.bootstrap import OPEBootstrap
from axioms import Reporter

report = Reporter()
t_start = time.time()
Kc.<c> = Frac(QQ['c'])
Pc = PolynomialRing(QQ, 'c')
cp = Pc.gen()


def raises(exc, f, *args, **kw):
    try:
        f(*args, **kw)
    except exc:
        return True
    return False


def brute_failures(V, triples, N):
    """Commutator formula for all 0 <= m, n < N, straight from nproduct."""
    out = {}
    for t in triples:
        a, b, cc = (V.gen(x) for x in t)
        s = (-1) ** (a.parity() * b.parity())
        for m in range(N):
            for n in range(N):
                d = (a.nproduct(b.nproduct(cc, n), m)
                     - s * b.nproduct(a.nproduct(cc, m), n))
                for j in range(m + 1):
                    d -= binomial(m, j) * a.nproduct(b, j).nproduct(cc, m + n - j)
                if d:
                    out[('jacobi',) + tuple(t) + (m, n)] = d
    return out


def same(x, y):
    """Equal coefficients (elements of different algebras)."""
    return dict(x.monomial_coefficients()) == dict(y.monomial_coefficients())


def all_triples(V):
    g = V.variable_names()
    return [(x, y, z) for x in g for y in g for z in g]


def vir_primary(R, names, weights, parities=None, extra=None, cc=None,
                horizontal_grading=None):
    """L Virasoro with central charge c (in R; ``cc`` if given, else 1 over
    QQ), the others primaries; ``extra(*generators)``: more OPEs (written in
    the normally ordered expressions)."""
    E = NormallyOrderedExpressions(R, names, parities=parities)
    g = E.gens()
    L = g[0]
    C = cc if cc is not None else (R.gen() if R is not QQ else R(1))
    opes = {(names[0], names[0]): [L.derivative(), 2 * L, 0, R(C) / 2]}
    for nm, X in zip(names[1:], g[1:]):
        opes[(names[0], nm)] = [X.derivative(), weights[nm] * X]
    if extra is not None:
        opes.update(extra(*g))
    return UniversalVertexAlgebra(R, E, opes, degrees=weights,
                                  horizontal_grading=horizontal_grading)


# ---------------------------------------------------------------------- #
#  Jacobi identities of declared OPEs                                     #
# ---------------------------------------------------------------------- #

# affine sl2: right level, wrong H_(1) H, wrong E_(0) F
Esl = NormallyOrderedExpressions(QQ, ['E', 'H', 'F'])
E, H, F = Esl.gens()


def sl2(hh, ef, he=2):
    return UniversalVertexAlgebra(QQ, Esl, {('H', 'E'): [he * E], ('H', 'F'): [-2 * F],
                                            ('E', 'F'): [ef, 1], ('H', 'H'): [0, hh]})


S2 = sl2(2, H)
report('sl2 level 1 consistent', S2.satisfies_jacobi_identity()
       and S2.jacobi_identity_failures() == {})
for hh, ef, he in [(3, H, 2), (2, 2 * H, 2), (2, H, 3)]:
    S2 = sl2(hh, ef, he)
    ours = {k: v for k, v in S2.jacobi_identity_failures().items()
            if k[0] == 'jacobi'}
    report('box = brute force (sl2 %s, %s, %s)' % (hh, ef, he),
           ours and ours == brute_failures(S2, all_triples(S2), 7))

# W3 with the right / a wrong beta (generic c)
Ew3 = NormallyOrderedExpressions(Kc, ['L', 'W'])
L, W = Ew3.gens()
VIR = {('L', 'L'): [L.derivative(), 2 * L, 0, c / 2],
       ('L', 'W'): [W.derivative(), 3 * W]}
LAM = L * L - 3 / 10 * L.derivative(2)


def w3(beta):
    WW = [L.derivative(3) / 15 + beta * LAM.derivative(),
          3 / 10 * L.derivative(2) + 2 * beta * LAM, L.derivative(), 2 * L, 0,
          c / 3]
    return dict(VIR) | {('W', 'W'): WW}


W3 = UniversalVertexAlgebra(Kc, Ew3, w3(16 / (22 + 5 * c)))
report('W3 consistent', W3.satisfies_jacobi_identity())
W3b = UniversalVertexAlgebra(Kc, Ew3, w3(1))
ours = {k: v for k, v in W3b.jacobi_identity_failures().items()
        if k[0] == 'jacobi'}
report('W3 beta=1: box = brute force',
       ours and ours == brute_failures(W3b, all_triples(W3b), 11))
report('W3 beta=1: no skew failure',
       not any(k[0] == 'skew' for k in W3b.jacobi_identity_failures()))

# verify_jacobi (constructor): refused, accepted with conformal data
old = W3.opes()
ok = raises(ValueError, UniversalVertexAlgebra, Kc, Ew3, w3(2), verify_jacobi=True,
            conformal_vector='L')
report('verify_jacobi refuses', ok)
report('  the algebras built before are untouched', W3.opes() == old
       and W3.conformal_vector() is None and W3.satisfies_jacobi_identity())
W3v = UniversalVertexAlgebra(Kc, Ew3, w3(16 / (22 + 5 * c)), verify_jacobi=True,
                             conformal_vector='L', horizontal_grading={'W': 1},
                             horizontal_grading_group=Zmod(2))
report('verify_jacobi accepts', W3v.conformal_vector() == W3v.gen('L')
       and W3v.horizontal_grading() is not None and W3v._jacobi_verified)

# a pair may be declared in one orientation only (the other follows by
# skew-symmetry); skew-symmetry of a generator with itself is checked
report('both orientations refused', raises(
    ValueError, UniversalVertexAlgebra, QQ, Esl,
    {('H', 'E'): [2 * E], ('E', 'H'): [-2 * E], ('H', 'F'): [-2 * F],
     ('E', 'F'): [H, 1], ('H', 'H'): [0, 2]}))
Sk = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): ['J', 1]})
bad = Sk.jacobi_identity_failures(triples=[])
report('J_(0) J = J: skew failure', list(bad) == [('skew', 'J', 'J', 0)]
       and bad[('skew', 'J', 'J', 0)] == 2 * Sk.gen('J'))
report("'sorted' triples", len(S2._jacobi_triples('sorted')) == 10)

# odd generators: N=1 and bc
N1 = vir_primary(Kc, ['L', 'G'], {'L': 2, 'G': 3 / 2}, {'G': 1},
                 extra=lambda L, G: {('G', 'G'): [2 * L, 0, 2 * c / 3]})
report('N=1 consistent', N1.satisfies_jacobi_identity())
N1 = vir_primary(Kc, ['L', 'G'], {'L': 2, 'G': 3 / 2}, {'G': 1},
                 extra=lambda L, G: {('G', 'G'): [L, 0, 2 * c / 3]})
report('N=1 wrong G G fails', not N1.satisfies_jacobi_identity())
BC = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
                            parities={'b': 1, 'c': 1})
report('bc consistent', BC.satisfies_jacobi_identity())

# ---------------------------------------------------------------------- #
#  Bootstrap                                                              #
# ---------------------------------------------------------------------- #

# free boson
V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
B = OPEBootstrap(V)
B.impose_jacobi()
report('boson: J_(0) J = 0 by skew-symmetry',
       str(B.ansatz()) == "{('J', 'J'): [0, x0]}"
       and str(B.solve()) == '[Bootstrap solution 1 of 1: free [x0]]')

# W3, generic c
V = vir_primary(Kc, ['L', 'W'], {'L': 2, 'W': 3})
B = OPEBootstrap(V)
report('W3: ansatz only on (W, W), V untouched',
       list(B.ansatz()) == [('W', 'W')] and ('W', 'W') not in V.opes()
       and B.algebra() is not V)
# every ansatz term has the right weight and parity
U = B.algebra()
ok = True
for x, ((i, j), n, m) in B._terms.items():
    ok = ok and V._weight_key(m, V._weight_data()) == B._h[i] + B._h[j] - n - 1
    ok = ok and V._parity_key(m) == 0
report('W3: ansatz terms of the right weight and parity',
       ok and len(B._terms) == 11)
B.fix({'x0': c / 3})
B.impose_jacobi()
[S] = B.solve()
A = S.algebra()
report('W3: Zamolodchikov', B.unknowns() == [] and
       all(same(x, W3(y)) for x, y in zip(A.opes()[('W', 'W')],
                                          w3(16 / (22 + 5 * c))[('W', 'W')])))
report('W3: solution algebra', A.satisfies_jacobi_identity()
       and A.degrees() == {'L': 2, 'W': 3} and A.base_ring() is Kc)

# W3 with c a parameter: one family, polynomial in c and x4
V = vir_primary(Pc, ['L', 'W'], {'L': 2, 'W': 3})
B = OPEBootstrap(V)
B.impose_jacobi('sorted')
sols = B.solve()
report('W3 (c parameter): one family', len(sols) == 1
       and str(sols[0].free_variables()) == '[c, x4]')
A = sols[0].algebra()
x0 = B.eliminated()[B.ring()('x0')]
report('W3 (c parameter): x0 = c(5c+22) x4/96, Jacobi',
       x0 == B.ring()('c*(5*c+22)*x4/96') and A.satisfies_jacobi_identity())

# W(2,4): square root
V = vir_primary(Kc, ['L', 'W'], {'L': 2, 'W': 4})
B = OPEBootstrap(V)
B.fix({'x0': c / 4})
B.impose_jacobi()
[S] = B.solve()
p = S.extension().modulus()
report('W(2,4): C_44^4 squared',
       -p(0) == 54 * (c + 24) * (c^2 - 172 * c + 196)
       / ((5 * c + 22) * (7 * c + 68) * (2 * c - 1)))
report('W(2,4): algebra over the extension', S.algebra().satisfies_jacobi_identity())

# W(2,5) with c a parameter: exact solutions only at c = 134 +- 60 sqrt 5
V = vir_primary(Pc, ['L', 'W'], {'L': 2, 'W': 5})
B = OPEBootstrap(V)
B.fix({'x0': cp / 5})
B.impose_jacobi()
sols = B.solve()
cs = sorted(str(S.prime_ideal().gens()[0]) for S in sols)
report('W(2,5): components c = 0, c^2 - 268c - 44 = 0',
       len(sols) == 2 and {str(S.algebraic_variable()) for S in sols} >= {'c'})
S = [S for S in sols if S.algebraic_variable() is not None][0]
report('W(2,5): c^2 - 268c - 44', str(S.extension().polynomial())
       == 'c^2 - 268*c - 44')
report('W(2,5): algebra over Q(sqrt 5)', S.algebra().satisfies_jacobi_identity())
V = vir_primary(QQ, ['L', 'W'], {'L': 2, 'W': 5}, cc=-7)
B = OPEBootstrap(V)
B.fix({'x0': -7 / 5})
B.impose_jacobi()
report('W(2,5) at c = -7: no exact solution', B.solve() == [])

# W(2,3,4): the W_4 family (couplings free up to the two normalisations)
# and a degenerate component; the scale invariant C^2 N_U / N_W^2 of the
# coupling C = coefficient of U in W_(1) W is a regression value (computed
# here, all Jacobi identities checked)
V = vir_primary(Kc, ['L', 'W', 'U'], {'L': 2, 'W': 3, 'U': 4})
B = OPEBootstrap(V)
t0 = time.time()
B.impose_jacobi('sorted')
sols = B.solve()
report('W(2,3,4): 2 components (%.1f s)' % (time.time() - t0),
       len(sols) == 2 and B.equations() == [B.ring()('x7*x32')])
S = [S for S in sols if str(S.free_variables()) == '[x7, x55]'][0]
A = S.algebra()
W4, U4 = A.gen('W'), A.gen('U')
WW, UU = W4.ope(W4), U4.ope(U4)
NW, NU = WW[5].coefficient(()), UU[7].coefficient(())
C = WW[1].coefficient(((2, 0),))
report('W(2,3,4): W_4 invariant',
       C^2 * NU / NW^2 == 12 * (c + 2) * (7 * c + 114) / (c * (5 * c + 22) * (c + 7)))
report('W(2,3,4): all 27 triples hold', A.satisfies_jacobi_identity())

# save / load: an interrupted run (autosaved after each triple) continued by
# load (a new bootstrap) gives the uninterrupted result
from vertex_algebras.bootstrap import _load_saved_bootstrap
fn = tmp_filename(ext='.sobj')
V2 = vir_primary(Kc, ['L', 'W', 'U'], {'L': 2, 'W': 3, 'U': 4})
B1 = OPEBootstrap(V2)
order = sorted(V2._jacobi_triples('sorted'),
               key=lambda t: sum(B1._h[i] for i in t))
names3 = V2.variable_names()
B1.impose_jacobi([tuple(names3[i] for i in t) for t in order[:6]], save=fn)
V3 = vir_primary(Kc, ['L', 'W', 'U'], {'L': 2, 'W': 3, 'U': 4})
B2 = load(fn)
st = B1._saved_state()
report('save/load: state restored', B2 is not B1 and B2.imposed_triples() == B1.imposed_triples()
       and B2.eliminated() == B1.eliminated()
       and B2.equations() == B1.equations()
       and str(B2.ansatz()) == str(B1.ansatz()))
B2.impose_jacobi('sorted')
report('save/load: resumed run = uninterrupted run',
       [str(S) for S in B2.solve()] == [str(S) for S in sols]
       and B2.unknowns() == B.unknowns())
V4 = vir_primary(Kc, ['L', 'W', 'U'], {'L': 2, 'W': 3, 'U': 5})
report('save/load: other weights refused',
       raises(ValueError, _load_saved_bootstrap, V4, 'x', st))
report('save/load: other pairs refused',
       raises(ValueError, OPEBootstrap, V3, pairs=[('W', 'W')], _state=st))
V3g = V3.with_horizontal_grading({'W': 1}, group=Zmod(2))
report('save/load: other grading refused',
       raises(ValueError, _load_saved_bootstrap, V3g, 'x', st))
report('save/load: old formats and other data refused',
       raises(ValueError, _load_saved_bootstrap, V2, 'x', {'kind': 'OPEBootstrap', 'format': 0})
       and raises(ValueError, _load_saved_bootstrap, V2, 'x', [1, 2]))

# N=1: G G fixed up to normalisation
V = vir_primary(Kc, ['L', 'G'], {'L': 2, 'G': 3 / 2}, {'G': 1})
B = OPEBootstrap(V)
B.impose_jacobi()
[S] = B.solve()
A = S.algebra()
G = A.gen('G')
x0 = A.base_ring().gen()
report('N=1: G G = x0/z^3 + 3 x0/c L/z', G.ope(G) == [3 / c * x0 * A.gen('L'), 0, x0 * A.one()]
       and A.satisfies_jacobi_identity())

# N=2 with the U(1) charge grading: 3 components, sorted == all
NAMES = ['L', 'J', 'Gp', 'Gm']
WTS = {'L': 2, 'J': 1, 'Gp': 3 / 2, 'Gm': 3 / 2}
primes = {}
for mode in ['sorted', None]:
    V = vir_primary(Kc, NAMES, WTS, {'Gp': 1, 'Gm': 1},
                    horizontal_grading={'Gp': 1, 'Gm': -1})
    B = OPEBootstrap(V)
    report('N=2 (%s): no Gp Gp, Gm Gm terms' % mode,
           B.ansatz()[('Gp', 'Gp')] == [] and B.ansatz()[('Gm', 'Gm')] == [])
    B.impose_jacobi(mode)
    sols = B.solve()
    primes[mode] = sorted(str(S.prime_ideal()) for S in sols)
    report('N=2 (%s): 3 components, all Jacobi' % mode, len(sols) == 3 and
           all(S.algebra().satisfies_jacobi_identity() for S in sols))
    sc = [S for S in sols if 'x0 = c/3*x2^2' in str(S)]
    report('N=2 (%s): superconformal component' % mode, len(sc) == 1)
report('N=2: sorted == all triples', primes['sorted'] == primes[None])
A = sc[0].algebra()
P2 = A.base_ring()
L2, J2, Gp, Gm = A.gens()
x2, x5 = P2.gens()
report('N=2: standard OPEs at x2 = 1, x5 = 2',
       [p.map_coefficients(lambda t: t(1, 2)) for p in Gp.ope(Gm)]
       == [p.map_coefficients(lambda t: t(1, 2)) for p in
           [2 * L2 + J2.derivative(), 2 * J2, 2 * c / 3 * A.one()]])

# affine sl2 with a grading
V = UniversalVertexAlgebra(QQ, ['E', 'H', 'F'], degrees={'E': 1, 'H': 1, 'F': 1},
                           horizontal_grading={'E': 2, 'F': -2})
B = OPEBootstrap(V)
B.fix({'x0': -2, 'x2': 1})
B.impose_jacobi()
[S] = B.solve()
A = S.algebra()
E, H, F = A.gens()
k = A.base_ring().gen()
report('sl2: level k = x1, H_(1) H = 2k', E.ope(F) == [H, k * A.one()]
       and H.ope(H) == [0, 2 * k * A.one()] and H.ope(F) == [-2 * F]
       and A.horizontal_grading() is not None and A.satisfies_jacobi_identity())

# inconsistent normalisation, restricted pairs, refusals, new algebras
V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
B = OPEBootstrap(V)
B.fix({'x0': 1})
B.fix({'x0': 2})
report('inconsistent fix: no solutions', B.equations() == [B.ring().one()]
       and B.solve() == [])
V = vir_primary(Kc, ['L', 'W', 'X'], {'L': 2, 'W': 3, 'X': 3})
B = OPEBootstrap(V, pairs=[('W', 'X')])
report('pairs=: only (W, X)', list(B.ansatz()) == [('W', 'X')])
report('pairs= declared refused',
       raises(ValueError, OPEBootstrap, V, pairs=[('L', 'W')]))
V1 = UniversalVertexAlgebra(QQ, ['J'])
report('no weights refused', raises(ValueError, OPEBootstrap, V1))
V2 = UniversalVertexAlgebra(ZZ, ['J'], degrees={'J': 1})
report('ZZ refused', raises(ValueError, OPEBootstrap, V2))
B1 = OPEBootstrap(V)
B2 = OPEBootstrap(V)
same = B1.algebra() is B2.algebra()           # equal data: the same algebra
B2.fix({str(B2.unknowns()[-1]): 1})
report('two bootstraps: the same algebra until one eliminates',
       same and B1.algebra() is not B2.algebra())
U = B1.algebra()
before = U.opes()
B1.fix({'x0': 1})
report('an elimination builds a new algebra, the old one is untouched',
       B1.algebra() is not U and U.opes() == before
       and B1.ring()('x0') in B1.eliminated()
       and B1.algebra().opes() != before)
Vz = vir_primary(Kc, ['L', 'W', 'X'], {'L': 2, 'W': 3, 'X': 3})
Bz = OPEBootstrap(Vz, zero_pairs=[('X', 'W'), ('X', 'X')])
report('zero_pairs: no ansatz on pairs known to vanish', list(Bz.ansatz()) == [('W', 'W')])

# serial == parallel (engine threads)
res = []
for nc, names in ((1, ['L', 'W']), (4, ['Lp', 'Wp'])):     # 4: another algebra
    set_nproc(nc)
    V = vir_primary(Kc, names, dict(zip(names, (2, 4))))
    B = OPEBootstrap(V)
    B.impose_jacobi()
    res.append((B.eliminated(), B.equations()))
set_nproc(1)
report('serial == parallel', res[0] == res[1])

print('total %.1f s' % (time.time() - t_start))
report.result()
