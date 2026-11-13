# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# TensorProductVertexAlgebra: lattice factorisations as vertex algebra maps,
# embeddings, axioms on mixed products, serial-vs-parallel identity.
# Run from the repository root:  sage tests/smoke_tensor.sage  -> RESULT: PASS
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import vertex_algebra
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
set_nproc = VertexAlgebraParallelism().set
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras.tensor_product_vertex_algebra import TensorProductVertexAlgebra
from axioms import (Reporter, nmax, skew_ok, commutator_ok, associativity_ok,
                    raw_caches_ok)

random.seed(2r)
report = Reporter()


def rand_lattice_key(L, maxletters=2, maxder=1, qr=1):
    n = L.rank()
    letters = sorted([(random.randrange(n), random.randrange(maxder + 1))
                      for _ in range(random.randrange(maxletters + 1))],
                     key=lambda t: (t[1], -t[0]), reverse=True)
    return (tuple(letters), tuple(random.randint(-qr, qr) for _ in range(n)))


def rand_heis_key(maxletters=2, maxder=1):
    return tuple(sorted([(0, random.randrange(maxder + 1))
                         for _ in range(random.randrange(maxletters + 1))],
                        key=lambda t: (t[1], -t[0]), reverse=True))


def into_big(Lbig, letters, q, offset):
    """The monomial of a sub-lattice (or Heisenberg) factor inside ``Lbig``
    (block-diagonal Gram matrix), shifted to the directions ``offset...``."""
    qb = [0] * Lbig.rank()
    for j, v in enumerate(q):
        qb[offset + j] = v
    return Lbig.monomial((tuple((i + offset, k) for i, k in letters),
                          tuple(qb)))


def morphism_ok(T, phi, keysA, keysB, label):
    """phi(a_(n) b) == phi(a)_(n) phi(b) for all pairs and n in [-3, nmax]."""
    t = time.time()
    good = True
    for ka in keysA:
        for kb in keysB:
            a, b = T.monomial(ka), T.monomial(kb)
            pa, pb = phi(a), phi(b)
            for n in range(-3, nmax(T, a, b) + 1):
                if phi(a.nproduct(b, n)) != pa.nproduct(pb, n):
                    good = False
                    print('   mismatch', label, a, '|', b, 'n =', n)
    report('%s is a vertex algebra map' % label, good,
           '(%d pairs, %.1fs)' % (len(keysA) * len(keysB), time.time() - t))


# --- 1. V_Z ⊗ V_Z ≅ V_diag(1,1): odd factors, Koszul sign vs cocycle -------
VZ = LatticeVertexAlgebra(QQ, matrix([[1]]))
T = VZ.tensor(VZ)
Lbig = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]]))


def phi_ZZ(x):
    res = Lbig.zero()
    for (k1, k2), c in x.monomial_coefficients().items():
        res += c * into_big(Lbig, k1[0], k1[1], 0).no(
            into_big(Lbig, k2[0], k2[1], 1))
    return res


keys = [(rand_lattice_key(VZ), rand_lattice_key(VZ)) for _ in range(5)]
morphism_ok(T, phi_ZZ, keys, keys, 'V_Z ⊗ V_Z -> V_diag(1,1)')

# Negative control: without the Koszul sign the map fails.  (The sign is
# removed from the Python recursion, so the C++ engine is not used.)
Tbad = VZ.tensor(VZ, implementation='python')   # a distinct instance
Tbad.clear_caches()
Tbad._suffix_parity = lambda i, m: 0
fails = any(phi_ZZ(Tbad.monomial(ka).nproduct(Tbad.monomial(kb), n))
            != phi_ZZ(Tbad.monomial(ka)).nproduct(phi_ZZ(Tbad.monomial(kb)), n)
            # (1 ⊗ ψ)_(n) (ψ' ⊗ 1): the only case where the sign is -1
            for ka in [(((), (0,)), ((), (1,))), (((), (0,)), ((), (-1,)))]
            for kb in [(((), (1,)), ((), (0,))), (((), (-1,)), ((), (0,)))]
            for n in range(-2, 1))
report('negative control: no Koszul sign breaks the map', fails)
del Tbad._suffix_parity                         # repaired
Tbad.clear_caches()

# --- 2. V_Z ⊗ V_A1 ≅ V_diag(1,2) --------------------------------------------
VA1 = LatticeVertexAlgebra(QQ, matrix([[2]]))
T2 = VZ.tensor(VA1)
L12 = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 2]]))


def phi_ZA(x):
    res = L12.zero()
    for (k1, k2), c in x.monomial_coefficients().items():
        res += c * into_big(L12, k1[0], k1[1], 0).no(
            into_big(L12, k2[0], k2[1], 1))
    return res


keys = [(rand_lattice_key(VZ), rand_lattice_key(VA1)) for _ in range(5)]
morphism_ok(T2, phi_ZA, keys, keys, 'V_Z ⊗ V_A1 -> V_diag(1,2)')

# --- 3. Heisenberg ⊗ V_A1 -> V_diag(1,2), onto the q_0 = 0 part --------------
H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
T3 = H.tensor(VA1)


def phi_HA(x):
    res = L12.zero()
    for (k1, k2), c in x.monomial_coefficients().items():
        res += c * into_big(L12, k1, (0,), 0).no(
            into_big(L12, k2[0], k2[1], 1))
    return res


keys = [(rand_heis_key(), rand_lattice_key(VA1)) for _ in range(5)]
morphism_ok(T3, phi_HA, keys, keys, 'Heisenberg ⊗ V_A1 -> V_diag(1,2)')

# --- 4. embeddings are morphisms ---------------------------------------------
K.<c0> = Frac(QQ['c0'])
En1 = NormallyOrderedExpressions(K, ['L', 'G'], parities={'G': 1})
L, G = En1.gens()
N1 = UniversalVertexAlgebra(K, En1, {('L', 'L'): [L.derivative(), 2 * L, 0, c0 / 2],
                                     ('L', 'G'): [G.derivative(), 3 / 2 * G],
                                     ('G', 'G'): [2 * L, 0, 2 * c0 / 3]})
L, G = N1.gen('L'), N1.gen('G')
VZK = LatticeVertexAlgebra(K, matrix([[1]]))
M = N1.tensor(VZK)
pool0 = [L, G, L.derivative(), L.no(G)]
pool1 = [VZK.e((1,)), VZK.e((-1,)), VZK.J(0), VZK.J(0).no(VZK.e((1,)))]
good = True
for i, pool in ((0, pool0), (1, pool1)):
    for x in pool:
        for y in pool:
            for n in range(-2, 3):
                good = good and (M.embed(i, x.nproduct(y, n))
                                 == M.embed(i, x).nproduct(M.embed(i, y), n))
report('embeddings are morphisms (N=1 ⊗ V_Z)', good)
report('images of different factors super-commute',
       all(M.embed(0, x).ope(M.embed(1, y)) == [] and
           M.embed(0, x).no(M.embed(1, y))
           == (-1) ** (x.parity() * y.parity()) * M.embed(1, y).no(M.embed(0, x))
           for x in pool0 for y in pool1))

# --- 5. axioms on the mixed product N=1 ⊗ V_Z --------------------------------
elems = ([M.embed(0, x) for x in pool0] + [M.embed(1, y) for y in pool1]
         + [M.pure_tensor(G, VZK.e((1,))), M.pure_tensor(L, VZK.J(0))])
trip = [tuple(random.choice(elems) for _ in range(3)) for _ in range(5)]
t = time.time()
report('N=1 ⊗ V_Z skew-symmetry', all(skew_ok(M, a, b) for a, b, _ in trip),
       '(%.1fs)' % (time.time() - t))
t = time.time()
report('N=1 ⊗ V_Z commutator formula',
       all(commutator_ok(M, a, b, c, range(-1, 2)) for a, b, c in trip),
       '(%.1fs)' % (time.time() - t))
t = time.time()
report('N=1 ⊗ V_Z associativity',
       all(associativity_ok(M, a, b, c, ms=(-1,), ns=range(0, 2))
           for a, b, c in trip[:3]), '(%.1fs)' % (time.time() - t))
cT = M.embed(0, L) + M.embed(1, VZK.J(0).no(VZK.J(0)) / 2)
report('Virasoro central charges add (c0 + 1)',
       cT.ope(cT) == [cT.derivative(), 2 * cT, 0, M((c0 + 1) / 2)])

# --- 6. three factors: Heisenberg ⊗ V_Z ⊗ V_Z ---------------------------------
T3f = H.tensor(VZ, VZ)
x3 = [T3f.embed(0, H.gen('J')), T3f.embed(1, VZ.e((1,))),
      T3f.embed(2, VZ.e((-1,))),
      T3f.pure_tensor(H.gen('J'), VZ.e((1,)), VZ.e((1,)))]
report('3 factors: skew-symmetry', all(skew_ok(T3f, a, b) for a in x3 for b in x3))
report('3 factors: commutator formula',
       all(commutator_ok(T3f, a, b, c, range(-1, 2))
           for a, b, c in [(x3[3], x3[1], x3[2]), (x3[1], x3[2], x3[3]),
                           (x3[0], x3[3], x3[3])]))
report('3 factors: flattening', T3f is H.tensor(VZ).tensor(VZ))

# --- 7. serial vs parallel ----------------------------------------------------
Ts = Tp = N1.tensor(VZK)                  # serial, then (cold) parallel
random.seed(9r)
monos = []
for _ in range(12):
    k1 = tuple(sorted([(random.randrange(2), random.randrange(2))
                       for _ in range(random.randrange(3))],
                      key=lambda t: (t[1], -t[0]), reverse=True))
    monos.append((k1, rand_lattice_key(VZK)))
A = Ap = sum(K(i + 1) * Ts.monomial(m) for i, m in enumerate(monos))


def res(X):
    return ([x.monomial_coefficients() for x in X.ope(X)],
            X.no(X).monomial_coefficients(),
            X.nproduct(X, -2).monomial_coefficients(),
            X.derivative(2).monomial_coefficients())


N1.clear_caches(); VZK.clear_caches(); Ts.clear_caches()
t = time.time(); rs = res(A); ts = time.time() - t
N1.clear_caches(); VZK.clear_caches(); Tp.clear_caches()
set_nproc(4)
t = time.time(); rp = res(Ap); tp = time.time() - t
report('serial == parallel (ope/no/nproduct(-2)/deriv)', rs == rp,
       '(serial %.2fs, nproc=4 %.2fs)' % (ts, tp))
report('tensor and factor caches hold raw dicts',
       all(raw_caches_ok(o) for o in Tp._cache_owners()))
report('warm-cache parallel re-run', res(Ap) == rp)
set_nproc(1)

# universal factors are immutable: other OPEs (or conformal data) make
# another factor and another tensor product; the first one is unchanged
Hs = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
Js = Hs.gen('J')
Bs = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]})
bs, gs = Bs.gens()
Ts = Hs.tensor(Bs)
xs = Ts.pure_tensor(Js.no(Js), bs) + Ts.pure_tensor(Js, bs.no(gs))


def fresh_res(x):
    """The same products in another tensor product (empty caches)."""
    F = Hs.tensor(Bs, implementation='python')
    y = F._from_dict(dict(x.monomial_coefficients()))
    return ([p.monomial_coefficients() for p in y.ope(y)],
            y.nproduct(y, -2).monomial_coefficients())


def own_res(x):
    return ([p.monomial_coefficients() for p in x.ope(x)],
            x.nproduct(x, -2).monomial_coefficients())


r0 = own_res(xs)
Hs3 = Hs.with_opes({('J', 'J'): [0, 3]})
Ts3 = Hs3.tensor(Bs)
xs3 = Ts3._from_dict(dict(xs.monomial_coefficients()))
r3 = own_res(xs3)
report('a factor with other OPEs: another tensor product, the first unchanged',
       Ts3 is not Ts and r3 != r0 and own_res(xs) == r0 and r0 == fresh_res(xs))
Hv = Hs.with_conformal_vector(Js.no(Js) / 2)
Bv = Bs.with_conformal_vector(-bs.no(gs.derivative())).with_horizontal_grading({'b': 1, 'g': -1})
Tv0 = Hv.tensor(Bv)
Tv = Tv0.with_conformal_vector('factors').with_horizontal_grading('factors')
report('factors with conformal data: transferred to their tensor product (explicit)',
       Tv.degrees() == {'J': 1, 'b': 1, 'g': 0} and Tv.central_charge() == 3
       and Ts.conformal_vector() is None and Ts.horizontal_grading() is None
       and Tv0.conformal_vector() is None and Tv0.horizontal_grading() is None
       and Tv is Hv.tensor(Bv, conformal_vector='factors', horizontal_grading='factors'))
def raises_ve(f, *args):
    try:
        f(*args)
    except ValueError:
        return True
    return False


report.result()
