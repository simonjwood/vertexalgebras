# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Tensor products of vertex (super)algebras.

:class:`TensorProductVertexAlgebra` is the (super) tensor product
``V_1 ⊗ ... ⊗ V_r`` of any :class:`vertex_algebra.VertexAlgebra` instances
over a common base ring, e.g. of universal and lattice vertex algebras; it
is also returned by ``V.tensor(W, ...)``.  Nested tensor products are
flattened.

A basis key is the tuple ``(m_1, ..., m_r)`` of factor basis keys.  The
vertex operator is ``Y(a ⊗ b, z) = Y(a, z) ⊗ Y(b, z)`` with the Koszul sign::

    (a ⊗ b)_(n) (c ⊗ d) = (-1)^{|b||c|} Σ_k  a_(k) c ⊗ b_(n-1-k) d,

a finite sum (``n-1-nmax(b,d) <= k <= nmax(a,c)``) that uses negative
``n``-th products of the factors; it is applied recursively for more than
two factors.  Derivatives follow the Leibniz rule.  Caching, linearity and
parallel evaluation come from the base class; each factor keeps its own
caches (and the tensor product's C++ engine builds its own factor engines).

EXAMPLES:

Two free bosons commute with each other::

    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
    sage: T = H.tensor(H)
    sage: T
    Tensor product of vertex algebras [Universal vertex algebra on generators ['J'] over Rational Field, Universal vertex algebra on generators ['J'] over Rational Field]
    sage: T.variable_names()
    ('J_0', 'J_1')
    sage: J0, J1 = T.gens()
    sage: J0, J1
    (J # 1, 1 # J)
    sage: J0.ope(J0), J0.ope(J1), J1.ope(J1)
    ([0, 1], [], [0, 1])
    sage: J0 * J1
    J # J
    sage: (J0 * J1).ope(J1)          # :J1 J2:(z) J2(w) ~ J1(z)/(z-w)^2
    [dJ # 1, J # 1]

Central charges add under the tensor product::

    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: K.<c1, c2> = Frac(QQ['c1', 'c2'])
    sage: N.<L, M> = NormallyOrderedExpressions(K)
    sage: V1 = UniversalVertexAlgebra(K, ['L'], {('L', 'L'): [L.derivative(), 2*L, 0, c1/2]})
    sage: V2 = UniversalVertexAlgebra(K, ['M'], {('M', 'M'): [M.derivative(), 2*M, 0, c2/2]})
    sage: L = V1.gen('L'); M = V2.gen('M')
    sage: V = V1.tensor(V2)
    sage: Ttot = V.embed(0, L) + V.embed(1, M)
    sage: Ttot.ope(Ttot) == [Ttot.derivative(), 2*Ttot, 0, V((c1 + c2)/2)]
    True

Odd elements of different factors anticommute; a universal vertex algebra
times a lattice vertex algebra::

    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
    sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
    ....:                             parities={'b': 1, 'c': 1})
    sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
    sage: W = bc.tensor(F)
    sage: b = W.embed(0, bc.gen('b')); psi = W.embed(1, F.e((1,)))
    sage: b.parity(), psi.parity()
    (1, 1)
    sage: b.ope(psi), b.no(psi) == -psi.no(b)
    ([], True)
    sage: X = W.pure_tensor(bc.gen('c'), F.e((-1,)))
    sage: Y = W.pure_tensor(bc.gen('b'), F.e((1,)))
    sage: X.ope(Y)      # Koszul sign -1, c_(0) b = 1, e^-1_(0) e^1 = 1, e^-1_(-1) e^1 = -J0
    [1 # J0 + :b c: # 1, -1]

Nested products are flattened::

    sage: V1.tensor(V2).tensor(V1) is V1.tensor(V2, V1)
    True
    sage: len(V1.tensor(V2, V1).factors())
    3

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from math import ceil
from math import comb as _comb

from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .vertex_algebra import (
    _TENSOR_SIGN,
    VertexAlgebra,
    _compiled_module,
    _engine_ring,
    _EngineKernel,
    _is_int_like,
    _normalize_implementation,
    _trim,
)

# C++ engine (_tensor_engine.hpp): the _tprod / _tderiv
# recursion on its own universal and lattice factor engines
_TVA_ENGINE, _ENGINE_OFF = _compiled_module('_tensor_engine')


class _TensorEngineKernel(_EngineKernel):
    r"""
    The C++ engine of a tensor product ``T`` of universal and lattice vertex
    algebras (``_tensor_engine``): the recursion over
    the factors and the factors' own recursions in C++, on one coefficient
    ring (see :class:`vertex_algebra._EngineKernel`).  The cocycle values of
    the lattice factors come from the factors for every call; if one cannot
    be lifted, key-level calls use ``T``'s Python recursion and
    element-level calls return ``None`` (generic path).

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
        ....:                             parities={'b': 1, 'c': 1})
        sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
        sage: W = bc.tensor(F)
        sage: E = W._engine()           # None without the compiled engine
        sage: E is None or E.kind() == 'q'
        True
        sage: X = W.pure_tensor(bc.gen('c'), F.e((-1,)))
        sage: Y = W.pure_tensor(bc.gen('b'), F.e((1,)))
        sage: X.ope(Y)
        [1 # J0 + :b c: # 1, -1]

    TESTS:

    The engine against the Python recursion (an odd lattice factor, ``Q(k)``
    lifted to ``QQ[k]``, a tower), then with 2 threads (the same results,
    in the same order)::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
        sage: def n1(R, cv):
        ....:     N = NormallyOrderedExpressions(R, ['L', 'G'], parities={'G': 1})
        ....:     L, G = N.gens()
        ....:     return UniversalVertexAlgebra(R, N, {('L', 'L'): [L.derivative(), 2*L, 0, R(cv)/2],
        ....:         ('L', 'G'): [G.derivative(), R(3)/2*G], ('G', 'G'): [2*L, 0, 2*R(cv)/3]})
        sage: def work(T, g):
        ....:     U, F = T.factors()
        ....:     x, y = (T.embed(0, u) for u in U.gens()[:2])
        ....:     n = F.rank()
        ....:     e, f = T.embed(1, F.e([1] + [0]*(n-1))), T.embed(1, F.e([-1] + [0]*(n-1)))
        ....:     X = x.no(e) + (2 + g)*y.derivative().no(f) + x.no(y)
        ....:     Y = y.no(f) - 3*e.derivative() + g*x.no(x).no(e)
        ....:     return [list(z.monomial_coefficients().items())
        ....:             for z in X.ope(Y) + [X.nproduct(Y, -3), X.derivative(2)]]
        sage: Kk.<k> = Frac(QQ['k'])
        sage: K.<s> = QuadraticField(5)
        sage: M.<i> = K.extension(QQ['y']([1, 0, 1]))
        sage: A2 = matrix([[2, -1], [-1, 2]])
        sage: def bc(R):
        ....:     return UniversalVertexAlgebra(R, ['b', 'c'], {('b', 'c'): [1]}, parities={'b': 1, 'c': 1})
        sage: for U, F, g in [(n1(QQ, 7/10), LatticeVertexAlgebra(QQ, matrix([[1]])), 0),
        ....:                 (n1(Kk, k), LatticeVertexAlgebra(Kk, A2), k),
        ....:                 (bc(M), LatticeVertexAlgebra(M, A2), i*s)]:
        ....:     T = U.tensor(F)
        ....:     res = work(T, g)
        ....:     assert sorted(map(sorted, res)) == sorted(map(sorted, work(U.tensor(F, implementation='python'), g)))
        ....:     T.clear_caches()
        ....:     VertexAlgebraParallelism().set(2)
        ....:     assert work(T, g) == res
        ....:     VertexAlgebraParallelism().reset()

    Ctrl-C (here an alarm) interrupts a long product, serially and with
    threads; afterwards the algebra computes as before::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: A = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=7/3, conformal_vector='sugawara')
        sage: J = LatticeVertexAlgebra(QQ, matrix([[2]])).J(0)
        sage: T = A.tensor(LatticeVertexAlgebra(QQ, matrix([[2]]), conformal_vector=J.no(J)/4),
        ....:              degrees='factors')
        sage: keys = list(T.weight_space(5).basis_keys())
        sage: big, small = (sum(T.monomial(m) for m in ks) for ks in (keys, keys[:6]))
        sage: ref = [list(z.monomial_coefficients().items()) for z in small.ope(small)]
        sage: for n in (1, 2):  # long time
        ....:     T.clear_caches()
        ....:     VertexAlgebraParallelism().set(n)
        ....:     try:
        ....:         alarm(0.3)
        ....:         _ = big.ope(big)
        ....:     except KeyboardInterrupt:
        ....:         pass
        ....:     finally:
        ....:         cancel_alarm()
        ....:     assert [list(z.monomial_coefficients().items()) for z in small.ope(small)] == ref
        sage: VertexAlgebraParallelism().reset()
    """

    def __init__(self, T, F, kind, Rk, specs, up=None, down=None):
        r"""
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or (E._T is W and E._lattice == [False, True])
            True
        """
        _EngineKernel.__init__(self, F, kind, Rk, up, down)
        self._T = T
        E = _TVA_ENGINE.Engine(kind, self.nvars(), self.modulus(), self.minpoly())
        cin = self._cin
        for f, (V, decl) in enumerate(specs):
            if decl is None:
                E.add_lattice(V._B)
            else:
                n = len(V._gens)
                E.add_universal(n, [int(V._parities.get(i, 0)) % 2 for i in range(n)])
                for (i, j), poles in decl.items():
                    E.set_ope(f, i, j, [[(m, cin(c)) for m, c in d.items() if c]
                                        for d in poles])
        self._lattice = [decl is None for _, decl in specs]
        self.attach(E)

    def _eps(self, dA, dB):
        r"""
        Per factor: None, or the lifted cocycle values of a lattice factor
        for the lattice vectors of the keys of dA x dB (may raise), as a
        list of ``(q, p, ε(q, p))``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()
            sage: E is None or E._eps([a], [b]) == [None, [((1,), (-1,), E._lift_scalar(1))]]
            True
        """
        lift = self._lift_scalar
        out = []
        for f, V in enumerate(self._T._factors):
            if not self._lattice[f]:
                out.append(None)
                continue
            qs = {k[f][1] for k in dA}
            ps = {k[f][1] for k in dB}
            out.append([(q, p, lift(V._eps(q, p))) for q in qs for p in ps])
        return out

    def _eps_or_none(self, dA, dB):
        r"""
        :meth:`_eps`, or ``None`` if a cocycle value cannot be lifted to the
        engine's ring (the callers then take the Python or the generic path).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E._eps_or_none([a], [b]) == E._eps([a], [b])
            True
        """
        try:
            return self._eps(dA, dB)
        except (ValueError, ArithmeticError, TypeError, AttributeError):
            return None

    def ope_mono(self, a, b):
        r"""
        The OPE of the basis keys ``a``, ``b`` (a tuple of pole dicts over the
        base ring, trimmed), as ``T._ope_key``; that Python recursion if a
        cocycle value cannot be lifted.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.ope_mono(a, b) == W._ope_key(a, b)
            True
        """
        eps = self._eps_or_none([a], [b])
        if eps is None:
            return TensorProductVertexAlgebra._ope_key(self._T, a, b)
        out = [self._el_down(el) for el in self._E.ope_mono(a, b, eps)]
        while out and not out[-1]:
            out.pop()
        return tuple(out)

    def no_mono(self, a, b):
        r"""
        The normally ordered product ``a_(-1) b`` of basis keys (a raw dict),
        as ``T._no_key``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.no_mono(a, b) == W._no_key(a, b)
            True
        """
        eps = self._eps_or_none([a], [b])
        if eps is None:
            return TensorProductVertexAlgebra._no_key(self._T, a, b)
        return self._el_down(self._E.no_mono(a, b, eps))

    def nprod_mono(self, a, n, b):
        r"""
        ``a_(n) b`` for basis keys and any integer ``n`` (a raw dict), as
        ``T._nprod_key``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or all(E.nprod_mono(a, n, b) == W._nprod_key(a, n, b)
            ....:                    for n in range(-3, 3))
            True
        """
        eps = self._eps_or_none([a], [b])
        if eps is None:
            return TensorProductVertexAlgebra._nprod_key(self._T, a, n, b)
        return self._el_down(self._E.nprod_mono(a, int(n), b, eps))

    def ope(self, dA, dB, nthreads):
        r"""
        The OPE of the elements given as raw dicts ``dA``, ``dB`` (a list of
        raw pole dicts), or ``None`` (a coefficient or cocycle value the engine
        cannot lift: generic path).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.ope({a: 2}, {b: 1}, 1) == [p._monomial_coefficients
            ....:                                         for p in (2*x).ope(y)]
            True
        """
        lifted = self._lift_all(dA, dB)
        eps = self._eps_or_none(dA, dB)
        if lifted is None or eps is None:
            return None
        return [self._el_down(el)
                for el in self._E.ope_el(lifted[0], lifted[1], eps, nthreads)]

    def ope_many(self, dA, dBs, nthreads):
        r"""
        The OPEs of ``dA`` with each of the raw dicts ``dBs`` (one call), or
        ``None`` as :meth:`ope`.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.ope_many({a: 1}, [{b: 1}, {a: 1}], 1) == [
            ....:     [p._monomial_coefficients for p in x.ope(z)] for z in (y, x)]
            True
        """
        lifted = self._lift_all(dA, *dBs)
        eps = self._eps_or_none(dA, [k for dB in dBs for k in dB])
        if lifted is None or eps is None:
            return None
        return [[self._el_down(el) for el in P]
                for P in self._E.ope_el_many(lifted[0], lifted[1:], eps, nthreads)]

    def nprod(self, dA, n, dB, nthreads):
        r"""
        ``A_(n) B`` for elements given as raw dicts, or ``None`` as :meth:`ope`.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.nprod({a: 1}, -2, {b: 1}, 1) == W._nprod_key(a, -2, b)
            True
        """
        lifted = self._lift_all(dA, dB)
        eps = self._eps_or_none(dA, dB)
        if lifted is None or eps is None:
            return None
        return self._el_down(self._E.nprod_el(lifted[0], int(n), lifted[1],
                                              eps, nthreads))

    def nprod_many_engine(self, els, tasks, nthreads):
        r"""
        ``[els[i]_(n) els[j] for (i, n, j) in tasks]`` for elements in the
        engine's form (:meth:`to_engine`), results in that form (no
        conversions), or ``None`` if a cocycle value cannot be lifted.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or [E.from_engine(r) for r in E.nprod_many_engine(
            ....:     [E.to_engine({a: 1}), E.to_engine({b: 1})], [(0, 0, 1)], 1)] == [
            ....:     W._nprod_key(a, 0, b)]
            True
        """
        eps = self._eps_or_none({k for i, _, _ in tasks for k, _ in els[i]},
                                {k for _, _, j in tasks for k, _ in els[j]})
        if eps is None:
            return None
        return self._E.nprod_el_many(
            els, [(int(i), int(n), int(j)) for i, n, j in tasks], eps, nthreads)

    def nprod_many(self, els, tasks, nthreads):
        r"""
        ``[els[i]_(n) els[j] for (i, n, j) in tasks]`` for elements given as
        raw dicts (one engine call), or ``None`` as :meth:`ope`.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: E = W._engine()           # None without the compiled engine
            sage: E is None or E.nprod_many([{a: 1}, {b: 1}], [(0, 0, 1), (1, -1, 0)], 1) == [
            ....:     W._nprod_key(a, 0, b), W._nprod_key(b, -1, a)]
            True
        """
        lifted = self._lift_all(*els)
        eps = self._eps_or_none({k for i, _, _ in tasks for k in els[i]},
                                {k for _, _, j in tasks for k in els[j]})
        if lifted is None or eps is None:
            return None
        return [self._el_down(el) for el in self._E.nprod_el_many(
            lifted, [(int(i), int(n), int(j)) for i, n, j in tasks], eps, nthreads)]


class _TensorKeys:
    r"""
    The basis keys ``(m_1, ..., m_r)`` of a tensor product (tuples of
    basis keys of the factors), the key set of its free module.

    TESTS::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
        sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
        sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
        sage: from vertex_algebras.tensor_product_vertex_algebra import _TensorKeys
        sage: K = _TensorKeys(W.factors()); a in K, b in K
        (True, True)
    """
    def __init__(self, factors):
        r"""
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.tensor_product_vertex_algebra import _TensorKeys
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: K = _TensorKeys([H, H]); K._factors == (H, H)
            True
        """
        self._factors = tuple(factors)

    def __contains__(self, x):
        r"""
        Whether ``x`` is a tuple of basis keys of the factors.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: from vertex_algebras.tensor_product_vertex_algebra import _TensorKeys
            sage: K = _TensorKeys(W.factors())
            sage: a in K, W._vacuum_key() in K, (1, 2) in K, (a[0],) in K
            (True, True, False, False)
        """
        return (isinstance(x, tuple) and len(x) == len(self._factors)
                and all(m in V._indices for m, V in zip(x, self._factors)))

    def an_element(self):
        r"""
        A tuple of an element of each factor's key set.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: from vertex_algebras.tensor_product_vertex_algebra import _TensorKeys
            sage: k = _TensorKeys(W.factors()).an_element(); k in W._indices
            True
        """
        return tuple(V._indices.an_element() for V in self._factors)

    def cardinality(self):
        r"""
        ``+Infinity`` (tensor products are infinite dimensional).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: from vertex_algebras.tensor_product_vertex_algebra import _TensorKeys
            sage: _TensorKeys(W.factors()).cardinality()
            +Infinity
        """
        return Infinity


def _tensor_weight_labels(factors):
    """The labels of :meth:`TensorProductVertexAlgebra._weight_generators`
    (the factors' labels, with the factor index appended when they
    clash), as ``(factor, label, unique label)``.

    TESTS::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: from vertex_algebras.tensor_product_vertex_algebra import _tensor_weight_labels
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
        sage: _tensor_weight_labels([H, F])
        [(0, 'J', 'J'), (1, 'J0', 'J0'), (1, 'e^(1)', 'e^(1)'), (1, 'e^(-1)', 'e^(-1)')]
        sage: _tensor_weight_labels([H, H])
        [(0, 'J', 'J_0'), (1, 'J', 'J_1')]
    """
    items = [(i, label) for i, V in enumerate(factors)
             for label, _ in V._weight_generators()]
    labels = [label for _, label in items]
    unique = len(set(labels)) == len(labels)
    return [(i, label, label if unique else '{}_{}'.format(label, i))
            for i, label in items]


def _tensor_element_data(x, factors, R):
    """
    A conformal vector given to :class:`TensorProductVertexAlgebra` in
    canonical form (a tuple of ``(key, coefficient)`` sorted by key): ``x``
    an element of a tensor product of the same factors (its products do not
    depend on conformal data), a dict ``{key: coefficient}`` or such a
    tuple.  Zero coefficients are dropped; keys sorted by their repr.

    TESTS::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
        sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
        sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
        sage: from vertex_algebras.tensor_product_vertex_algebra import _tensor_element_data
        sage: _tensor_element_data(x + 2*y, [H, F], QQ)
        (((((0, 0),), ((), (-1,))), 2), ((((0, 0),), ((), (1,))), 1))
        sage: _tensor_element_data({a: 1, b: 0}, [H, F], QQ)
        (((((0, 0),), ((), (1,))), 1),)
        sage: _tensor_element_data({(1, 2): 1}, [H, F], QQ)
        Traceback (most recent call last):
        ...
        ValueError: (1, 2) is not a basis key of the tensor product
    """
    P = _parent(x)
    if getattr(P, '_is_tensor_product', False):
        if P._factors != tuple(factors):
            raise ValueError('the conformal vector must be an element of a '
                             'tensor product of the same factors')
        d = x._monomial_coefficients
    elif isinstance(x, (dict, tuple)):
        d = dict(x)
        keys = _TensorKeys(factors)
        for k in d:
            if k not in keys:
                raise ValueError('{!r} is not a basis key of the tensor '
                                 'product'.format(k))
    else:
        raise TypeError('the conformal vector must be an element of the tensor '
                        "product of the factors, 'factors' or a dict of basis "
                        'keys, not {!r}'.format(x))
    out = [(k, R(c)) for k, c in d.items() if R(c)]
    return tuple(sorted(out, key=lambda kc: repr(kc[0])))


def _tensor_embedding(T, i):
    r"""
    ``T.embedding(i)`` (unpickling).

    TESTS::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2); T = V.tensor(V); e = T.embedding(1)
        sage: loads(dumps(e)) == e
        True
    """
    return T.embedding(i)


class TensorProductVertexAlgebra(VertexAlgebra):
    r"""
    The (super) tensor product of vertex algebras ``V_1 ⊗ ... ⊗ V_r``.

    INPUT:

    - ``factors`` -- a list or tuple of vertex algebras over the same base
      ring (tensor products among them are flattened, without their
      conformal data)
    - ``conformal_vector`` -- (default: ``None``) a Virasoro vector (``None``:
      none, even if the factors have one): ``'factors'`` (the sum of the
      factors' Virasoro vectors, every factor needs one) or an element of the
      tensor product of the same factors without conformal data (its products
      do not depend on them); checked as by
      :meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector`, any Virasoro
      vector passing the checks is accepted (see ``_weight_generators()``)
    - ``degrees`` -- (default: ``None``) weights without a Virasoro vector:
      ``'factors'`` (the factors' conformal weights) or a dict over the
      labels of ``_weight_generators()``
    - ``horizontal_grading`` -- (default: ``None``) ``None`` or
      ``'factors'``: the product of the factors' gradings (see
      ``_grading_data()``)
    - ``implementation`` -- (default: ``None``) ``None`` (the C++ engine if it
      takes the factors, else Python), ``'engine'`` (an error if it does not)
      or ``'python'`` (the Python recursion over the factors' products, which
      use their own implementations).  Results do not depend on it.

    These conformal data are part of the algebra: other choices give other
    algebras (``with_conformal_vector`` etc.), whose elements coerce into
    the tensor product without them.  ``V.tensor(W, ...)`` takes the same
    keywords.

    TESTS::

        sage: from vertex_algebras.tensor_product_vertex_algebra import TensorProductVertexAlgebra
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: A = UniversalVertexAlgebra(QQ, ['a'], {('a', 'a'): [0, 1]})
        sage: B = UniversalVertexAlgebra(GF(5), ['b'])
        sage: TensorProductVertexAlgebra([A, B])
        Traceback (most recent call last):
        ...
        ValueError: the factors must have the same base ring
        sage: T = A.tensor(A)
        sage: T is A.tensor(A)
        True
        sage: A.tensor(A, implementation='python') is T
        False
        sage: A.tensor(A, implementation='cython')
        Traceback (most recent call last):
        ...
        ValueError: implementation must be None, 'engine' or 'python' (got 'cython')
        sage: T.one(), T.one().is_one()
        (1, True)
        sage: T(3) == 3 * T.one()
        True
        sage: T.an_element()
        a # 1
        sage: T.embed(0, A.gen('a')).derivative(2)
        d^2a # 1
        sage: x = T.embed(0, A.gen('a')) * T.embed(1, A.gen('a'))
        sage: x.derivative()
        a # da + da # a

    Lattice vertex algebras factorise: ``V_Z ⊗ V_Z ≅ V_{diag(1,1)}`` (the
    Koszul sign of the odd factors becomes the cocycle) and ``V_Z ⊗ V_{A_1}
    ≅ V_{diag(1,2)}``, by ``a ⊗ b ↦ :a b:`` with ``a``, ``b`` in the
    directions ``x_0``, ``x_1``; a vertex algebra map for all ``n``::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: VZ = LatticeVertexAlgebra(QQ, matrix([[1]]))
        sage: def iso(L):
        ....:     def phi(x):
        ....:         return sum((c * L.monomial((k1[0], k1[1] + (0,))).no(L.monomial((
        ....:                    tuple((i + 1, d) for i, d in k2[0]), (0,) + k2[1])))
        ....:                    for (k1, k2), c in x.monomial_coefficients().items()), L.zero())
        ....:     return phi
        sage: for W, B in [(VZ, [[1, 0], [0, 1]]), (LatticeVertexAlgebra(QQ, matrix([[2]])),
        ....:                                       [[1, 0], [0, 2]])]:
        ....:     P = VZ.tensor(W); phi = iso(LatticeVertexAlgebra(QQ, matrix(B)))
        ....:     xs = [P.pure_tensor(VZ.e((1,)), W.e((-1,))), P.pure_tensor(VZ.e((-1,)), W.J(0)),
        ....:           P.pure_tensor(VZ.J(0).derivative(), W.e((1,))),
        ....:           P.pure_tensor(VZ.e((1,)), W.e((1,)))]
        ....:     print(all(phi(x.nproduct(y, n)) == phi(x).nproduct(phi(y), n)
        ....:               for x in xs for y in xs for n in range(-3, 4)))
        True
        True

    A universal factor next to a lattice factor: the Heisenberg algebra
    ``⊗ V_{A_1}`` maps onto the ``q_0 = 0`` part of ``V_{diag(1,2)}``, also a
    vertex algebra map for all ``n``::

        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: A1 = LatticeVertexAlgebra(QQ, matrix([[2]])); P = H.tensor(A1)
        sage: L12 = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 2]]))
        sage: def phi(x):
        ....:     return sum((c * L12.monomial((k1, (0, 0))).no(L12.monomial((
        ....:                tuple((i + 1, d) for i, d in k2[0]), (0,) + k2[1])))
        ....:                for (k1, k2), c in x.monomial_coefficients().items()), L12.zero())
        sage: J = H.gen('J')
        sage: xs = [P.pure_tensor(J, A1.e((-1,))), P.pure_tensor(J.derivative().no(J), A1.J(0)),
        ....:       P.pure_tensor(J, A1.e((1,)).derivative()), P.embed(1, A1.e((1,)))]
        sage: phi(xs[0])
        :J0 e^(0,-1):
        sage: all(phi(x.nproduct(y, n)) == phi(x).nproduct(phi(y), n)
        ....:     for x in xs for y in xs for n in range(-3, 4))
        True

    Negative control: without the Koszul sign (removed from the Python
    recursion of a separate instance) the map fails::

        sage: P = VZ.tensor(VZ, implementation='python')
        sage: phi = iso(LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]])))
        sage: x, y = P.pure_tensor(1, VZ.e((1,))), P.pure_tensor(VZ.e((1,)), 1)
        sage: P.clear_caches(); P._suffix_parity = lambda i, m: 0
        sage: phi(x.no(y)), phi(x).no(phi(y))
        (e^(1,1), -e^(1,1))
        sage: del P._suffix_parity; P.clear_caches()
        sage: phi(x.no(y)) == phi(x).no(phi(y))
        True

    The axioms on a mixed product of a super vertex algebra over ``Q(c)``
    (``N = 1``) and a lattice superalgebra, and on three factors::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: K.<c> = Frac(QQ['c'])
        sage: N.<L, G> = NormallyOrderedExpressions(K, parities={'G': 1})
        sage: N1 = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
        ....:     ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 2*c/3]})
        sage: F = LatticeVertexAlgebra(K, matrix([[1]])); M = N1.tensor(F)
        sage: L, G = N1.gens(); e, f = F.e((1,)), F.e((-1,))
        sage: TestSuite(M).run(elements=[M.pure_tensor(G, e), M.pure_tensor(L, F.J(0)),
        ....:                            M.embed(1, f), M.pure_tensor(G.derivative(), f)])
        sage: TestSuite(A.tensor(VZ, VZ)).run()
    """

    _is_tensor_product = True               # duck typing (doctests)

    @staticmethod
    def __classcall_private__(cls, factors, conformal_vector=None,
                              degrees=None, horizontal_grading=None,
                              implementation=None):
        r"""
        Normalize the arguments (unique representation, pickling).

        EXAMPLES::

            sage: from vertex_algebras import tensor_product_vertex_algebra as tpva
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L0 = LatticeVertexAlgebra(QQ, matrix([[2]])); K = L0.J(0)
            sage: L = L0.with_conformal_vector(K.no(K)/4).with_horizontal_grading('lattice')
            sage: V = tpva.TensorProductVertexAlgebra([L, L], conformal_vector='factors',
            ....:                                     horizontal_grading='factors')
            sage: V is L.tensor(L).with_conformal_vector('factors').with_horizontal_grading('factors')
            True
            sage: loads(dumps(V)) is V, V.central_charge(), V.horizontal_grading_group()
            (True, 2, The Cartesian product of 2 copies of Integer Ring)
            sage: TestSuite(V).run()
        """
        flat = []
        for V in factors:
            if isinstance(V, TensorProductVertexAlgebra):
                flat.extend(V._factors)
            elif isinstance(V, VertexAlgebra):
                flat.append(V)
            else:
                raise TypeError('{!r} is not a vertex algebra'.format(V))
        if not flat:
            raise ValueError('a tensor product needs at least one factor')
        R = flat[0].base_ring()
        if any(V.base_ring() is not R for V in flat):
            raise ValueError('the factors must have the same base ring')
        flat = tuple(flat)
        T = conformal_vector
        if T is not None and not (isinstance(T, str) and T == 'factors'):
            if isinstance(T, str):
                raise ValueError("conformal_vector must be an element, None or "
                                 "'factors'")
            T = _tensor_element_data(T, flat, R)
        weights = degrees
        if weights is not None and not (isinstance(weights, str)
                                        and weights == 'factors'):
            if isinstance(weights, str):
                raise ValueError("conformal_weights must be a dict, None or "
                                 "'factors'")
            labels = [t for _, _, t in _tensor_weight_labels(flat)]
            w = dict(weights)
            if set(w) != set(labels):
                raise ValueError('conformal weights must be given for exactly '
                                 '{} (got {})'.format(labels, sorted(w, key=str)))
            weights = tuple((t, R(w[t])) for t in labels)
        if horizontal_grading is not None and not (isinstance(horizontal_grading, str)
                                        and horizontal_grading == 'factors'):
            raise ValueError("the grading of a tensor product is 'factors' "
                             "(the product of the factors' gradings)")
        return VertexAlgebra.__classcall__(
            cls, flat, conformal_vector=T, degrees=weights,
            horizontal_grading=horizontal_grading,
            implementation=_normalize_implementation(implementation, ('engine', 'python')))

    def __init__(self, factors, conformal_vector=None, degrees=None,
                 horizontal_grading=None, implementation=None):
        r"""
        Initialize ``self``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
            sage: W = H.tensor(F, implementation='python')
            sage: W._r, W._implementation, W.factors() == (H, F)
            (2, 'python', True)
        """
        self._factors = tuple(factors)
        self._implementation = implementation
        self._r = len(self._factors)
        R = self._factors[0].base_ring()
        from .vertex_algebra_categories import (
            _is_finitely_strongly_generated,
            _is_proto,
        )
        VertexAlgebra.__init__(
            self, R, _TensorKeys(self._factors),
            category=self._category_for(
                R, any(V._has_odd() for V in self._factors),
                degrees is not None or conformal_vector is not None,
                conformal_vector is not None,
                fsg=all(_is_finitely_strongly_generated(V)
                        for V in self._factors),
                proto=any(_is_proto(V) for V in self._factors)))
        names = [(i, s) for i, V in enumerate(self._factors)
                 for s in V.variable_names()]
        unique = len({s for _, s in names}) == len(names)
        self._gen_list = [(i, s, s if unique else '{}_{}'.format(s, i))
                          for i, s in names]
        self._gen_idx = {t: k for k, (_, _, t) in enumerate(self._gen_list)}
        self._data = {'T': conformal_vector,
                      'weights': degrees, 'horizontal_grading': horizontal_grading,
                      'implementation': implementation}
        self._check_implementation()
        if degrees is not None:
            self._install_conformal_weights(
                degrees if isinstance(degrees, str)
                else dict(degrees))
        if horizontal_grading is not None:
            self._install_grading(horizontal_grading)
        if conformal_vector is not None:
            self._install_virasoro(
                self._factors_virasoro() if isinstance(conformal_vector, str)
                else self._from_dict(dict(conformal_vector), coerce=False,
                                     remove_zeros=True))

    # conformal data as construction data (see VertexAlgebra._forgets)

    def _structure_key(self):
        r"""
        The construction data without the conformal ones (factors and
        implementation), for the forgetful coercions (see
        ``VertexAlgebra._forgets``).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: P = H.tensor(H); Pw = H.tensor(H, degrees={'J_0': 1, 'J_1': 1})
            sage: Pw._structure_key() == P._structure_key()
            True
            sage: P._structure_key()[0], P._structure_key()[2]
            ('TensorProductVertexAlgebra', None)
            sage: P(Pw.gen(0))
            J # 1
        """
        return ('TensorProductVertexAlgebra', self._factors,
                self._data['implementation'])

    def _conformal_key(self):
        r"""
        The conformal construction data: canonical Virasoro vector, weights and
        grading option (``None`` when not chosen).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: sorted(H.tensor(H, degrees={'J_0': 1, 'J_1': 1})._conformal_key().items())
            [('T', None), ('horizontal_grading', None), ('weights', (('J_0', 1), ('J_1', 1)))]
        """
        d = self._data
        return {'T': d['T'], 'weights': d['weights'], 'horizontal_grading': d['horizontal_grading']}

    def _with(self, **changes):
        r"""
        The tensor product of the same factors (same implementation) with the
        conformal data changed by ``changes`` (``conformal_vector``,
        ``degrees``, ``horizontal_grading``); hook of ``with_conformal_vector``
        etc.  The grading group cannot be chosen (always ``'factors'``).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: Pw = H.tensor(H, degrees={'J_0': 1, 'J_1': 1})
            sage: Pw._with(degrees=None) is H.tensor(H)
            True
            sage: Pw._with(horizontal_grading_group=ZZ)
            Traceback (most recent call last):
            ...
            ValueError: the grading of a tensor product is 'factors' (the product of the factors' gradings)
        """
        d = self._data
        if changes.pop('horizontal_grading_group', None) is not None:
            raise ValueError("the grading of a tensor product is 'factors' "
                             "(the product of the factors' gradings)")
        args = dict(conformal_vector=d['T'], degrees=d['weights'],
                    horizontal_grading=d['horizontal_grading'])
        args.update(changes)
        return TensorProductVertexAlgebra(self._factors,
                                          implementation=d['implementation'],
                                          **args)

    def _has_odd(self):
        r"""
        Whether some factor has odd elements (decides the super category).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: P = H.tensor(bc)
            sage: P._has_odd(), H.tensor(H)._has_odd()
            (True, False)
        """
        return any(V._has_odd() for V in self._factors)

    def _factors_virasoro(self):
        r"""
        ``Σ_i 1 ⊗ ... ⊗ T_i ⊗ ... ⊗ 1`` for the Virasoro vectors ``T_i``
        of the factors (``conformal_vector='factors'``).  A tensor product
        never has a Virasoro vector by default, even if its factors do.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]},
            ....:                            conformal_vector={(('J', 0), ('J', 0)): 1/2})
            sage: L0 = LatticeVertexAlgebra(QQ, matrix([[2]])); K = L0.J(0)
            sage: H.tensor(L0, conformal_vector='factors')
            Traceback (most recent call last):
            ...
            ValueError: factor 1 has no Virasoro vector; choose one for the factor first (with_conformal_vector)
            sage: L = L0.with_conformal_vector(K.no(K) / 4)
            sage: H.tensor(L).conformal_vector() is None                # no default
            True
            sage: V = H.tensor(L, conformal_vector='factors'); V.conformal_vector()
            1/4*1 # :J0 J0: + 1/2*:J J: # 1
            sage: V.central_charge(), V.degrees()
            (2, {'J': 1, 'J0': 1, 'e^(-1)': 1, 'e^(1)': 1})
        """
        for i, V in enumerate(self._factors):
            if V.conformal_vector() is None:
                raise ValueError(
                    'factor {} has no Virasoro vector; choose one for the '
                    'factor first (with_conformal_vector)'.format(i))
        return sum((self.embed(i, V.conformal_vector())
                    for i, V in enumerate(self._factors)), self.zero())

    def _repr_(self):
        r"""
        The list of the factors.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: repr(H.tensor(H))
            "Tensor product of vertex algebras [Universal vertex algebra on generators ['J'] over Rational Field, Universal vertex algebra on generators ['J'] over Rational Field]"
        """
        return 'Tensor product of vertex algebras {}'.format(
            list(self._factors))

    # ------------------------------------------------------------------ #
    #  Factors, embeddings, generators                                    #
    # ------------------------------------------------------------------ #

    def factors(self):
        """
        The tuple of tensor factors.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: P = B.tensor(H); P.factors() == (B, H)
            True
            sage: P.tensor(B).factors() == (B, H, B)    # flattened
            True
        """
        return self._factors

    def factor(self, i):
        """
        The ``i``-th tensor factor (0-based).

        INPUT:

        - ``i`` -- an integer, the index of the factor

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: B.tensor(H).factor(1)
            Universal vertex algebra on generators ['alpha'] over Rational Field
        """
        return self._factors[i]

    def embed(self, i, x):
        r"""
        The image of the element ``x`` of the ``i``-th factor under the
        embedding ``x ↦ 1 ⊗ ... ⊗ x ⊗ ... ⊗ 1`` (a vertex algebra morphism).

        INPUT:

        - ``i`` -- the index of the factor (0-based)
        - ``x`` -- an element of that factor (or convertible into it)

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: P = B.tensor(H)
            sage: b, c = B.gens()
            sage: P.embed(0, b*c)
            :b c: # 1
            sage: P.embed(0, b).ope(P.embed(0, c)) == [P.embed(0, x) for x in b.ope(c)]
            True
        """
        V = self._factors[i]
        x = V(x)
        vac = [W._vacuum_key() for W in self._factors]
        d = {}
        for m, c in x._monomial_coefficients.items():
            key = list(vac)
            key[i] = m
            d[tuple(key)] = c
        return self._el_from_dict(d)

    def embedding(self, i):
        r"""
        The embedding ``x ↦ 1 ⊗ ... ⊗ x ⊗ ... ⊗ 1`` of the ``i``-th factor,
        a morphism of vertex algebras (see :meth:`embed`).

        INPUT:

        - ``i`` -- the index of the factor (0-based)

        OUTPUT:

        A morphism of vertex algebras from the ``i``-th factor to ``self``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: P = B.tensor(H); e = P.embedding(0); e
            Vertex algebra morphism:
              From: Universal vertex algebra on generators ['b', 'c'] over Rational Field
              To:   Tensor product of vertex algebras [Universal vertex algebra on generators ['b', 'c'] over Rational Field, Universal vertex algebra on generators ['alpha'] over Rational Field]
              Defn: b |--> b # 1
                    c |--> c # 1
            sage: e(B.gen(0).no(B.gen(1)))
            :b c: # 1
        """
        from .vertex_algebra_morphism import _function_morphism
        return _function_morphism(self._factors[i], self, lambda x: self.embed(i, x),
                                  reduction=(_tensor_embedding, (self, i)))

    def pure_tensor(self, *xs):
        r"""
        The element ``x_1 ⊗ ... ⊗ x_r`` for elements ``x_i`` of the factors
        (multilinear in the ``x_i``).

        INPUT:

        - ``*xs`` -- one element of each factor, in order

        EXAMPLES:

        With odd ``b``, ``c`` and even ``J``: ``(b ⊗ J)(z) (c ⊗ J)(w) ~
        1/(z-w)^3 + ...``::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: P = B.tensor(H)
            sage: (b, c), J = B.gens(), H.gen(0)
            sage: x, y = P.pure_tensor(b, J), P.pure_tensor(c, J); x
            b # alpha
            sage: x.ope(y)[1:]
            [:b c: # 1, 1]
            sage: x.parity(), x == P.embed(0, b) * P.embed(1, J)
            (1, True)

        TESTS::

            sage: P.pure_tensor(b)
            Traceback (most recent call last):
            ...
            ValueError: 2 factors expected
        """
        if len(xs) != self._r:
            raise ValueError('{} factors expected'.format(self._r))
        terms = [((), self._one_R)]
        for V, x in zip(self._factors, xs):
            x = V(x)
            terms = [(k + (m,), c * v) for k, c in terms
                     for m, v in x._monomial_coefficients.items()]
        d = {}
        for k, c in terms:
            d[k] = d.get(k, self._zero_R) + c
        return self._el_from_dict({k: c for k, c in d.items() if c})

    def zhu_algebra_up_to(self, max_weight, twist=None):
        r"""
        Zhu's algebra of this tensor product when some factors are
        quotients: the tensor product of the factors' Zhu algebras, with
        ``Q.zhu_algebra_up_to(max_weight)`` for the quotient factors ``Q`` (so an
        algebra mapping onto ``A(V)``, equal to it for ``max_weight`` large
        enough); without quotient factors ``A(V)`` itself, as
        ``zhu_algebra()``.  See :class:`zhu_algebra.TensorProductZhuAlgebra`.

        INPUT:

        - ``max_weight`` -- the largest weight of the ideal elements used for
          the quotient factors
        - ``twist`` -- (default: ``None``) ``'H'`` (or ``'ramond'``): the
          H-twisted Zhu algebras

        OUTPUT:

        A :class:`zhu_algebra.TensorProductZhuAlgebra`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
            ....:                              conformal_vector=L)
            sage: L = Vir.gen('L')
            sage: LY = Vir.quotient([L.no(L) - 3/10*L.derivative(2)])     # Lee-Yang
            sage: V = LY.tensor(LY, degrees='factors')
            sage: [V.zhu_algebra_up_to(W).dimension() for W in (3, 4)]
            [+Infinity, 4]
        """
        from .zhu_algebra import TensorProductZhuAlgebra
        return TensorProductZhuAlgebra(self, max_weight, twist)

    def _state_fingerprint(self):
        r"""
        The factors' fingerprints (for saved subalgebra states).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: fp = W._state_fingerprint()
            sage: fp == ('TensorProductVertexAlgebra', (H._state_fingerprint(), F._state_fingerprint()))
            True
        """
        return ('TensorProductVertexAlgebra',
                tuple(V._state_fingerprint() for V in self._factors))

    @property
    def _key_weight_spaces(self):
        r"""
        Whether the key-based weight hooks exist: iff they do for every factor.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._key_weight_spaces
            True
        """
        return all(V._key_weight_spaces for V in self._factors)

    def variable_names(self):
        """
        The generator names of all factors; a name that occurs in several
        factors gets the factor index appended (``'J_0'``, ``'J_1'``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B, H = va.FermionicGhosts(QQ), va.FreeBosons(QQ, 1)
            sage: B.tensor(H).variable_names()
            ('b', 'c', 'alpha')
            sage: H.tensor(H).variable_names()
            ('alpha_0', 'alpha_1')
        """
        return tuple(t for _, _, t in self._gen_list)

    def gen(self, name_or_index):
        """
        The embedded generator with the given name or index.

        INPUT:

        - ``name_or_index`` -- a name in :meth:`variable_names` or its index

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ, 1)
            sage: P = H.tensor(H)
            sage: P.gen('alpha_1'), P.gen(0)
            (1 # alpha, alpha # 1)
            sage: P.gen(1) == P.embed(1, H.gen(0))
            True
        """
        if _is_int_like(name_or_index):
            i, s, _ = self._gen_list[int(name_or_index)]
        else:
            i, s, _ = self._gen_list[self._gen_idx[name_or_index]]
        return self.embed(i, self._factors[i].gen(s))

    # ------------------------------------------------------------------ #
    #  Base-class hooks                                                   #
    # ------------------------------------------------------------------ #

    def _vacuum_key(self):
        r"""
        The tuple of the factors' vacuum keys.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._vacuum_key()
            ((), ((), (0,)))
            sage: W.monomial(W._vacuum_key()) == W.one()
            True
        """
        return tuple(V._vacuum_key() for V in self._factors)

    def _parity_key(self, m):
        r"""
        The parity of the key ``m``: the sum of the factors' parities mod 2.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._parity_key(a), W._parity_key(W._vacuum_key())
            (1, 0)
        """
        return sum(V._parity_key(x) for V, x in zip(self._factors, m)) % 2

    def _term_str(self, m, style='ascii'):
        r"""
        The factors' strings of the key ``m``, joined by the tensor sign of
        the ``style`` (``'ascii'``: `` # ``).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._term_str(a), W._term_str(a, 'latex')
            ('J # e^(1)', 'J \\otimes e^{(1)}')
        """
        return _TENSOR_SIGN[style].join(V._term_str(x, style)
                                        for V, x in zip(self._factors, m))

    def _mode_term(self, m, style, data):
        r"""
        The factors' mode words of the key ``m``, each on its own vacuum,
        joined by the tensor sign (see :meth:`VertexAlgebra._mode_term`;
        the factors' weights are those of the tensor product's data).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ, 1, degrees={'alpha': 1}); F0 = va.Lattice(QQ, matrix([[1]]))
            sage: F = va.Lattice(QQ, matrix([[1]]), conformal_vector=F0.J(0) * F0.J(0) / 2)
            sage: W = H.tensor(F, degrees='factors'); J = H.gen(0)
            sage: x = W.pure_tensor(J.derivative(), F.J(0) * F.e((1,)))
            sage: W.print_options(style='modes'); x
            alpha(-2)|0> # J0(-1)|(1)>
            sage: W.print_options(style='conformal_modes'); x
            alpha_{-2}|0> # J0_{-1}|(1)>
            sage: W.print_options(style='normally_ordered')
        """
        f = 1
        parts = []
        for i, (V, x) in enumerate(zip(self._factors, m)):
            fi, si = V._mode_term(x, style, None if data is None
                                  else data['factors'][i])
            f *= fi
            parts.append(si)
        return f, _TENSOR_SIGN[style].join(parts)

    def clear_caches(self):
        """
        Forget the cached results of the tensor product (the factors
        keep their caches); results are recomputed on demand.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ, 1)
            sage: P = H.tensor(H); x = P.gen(0) * P.gen(1)
            sage: y = x.ope(x); P.clear_caches()
            sage: x.ope(x) == y
            True
        """
        VertexAlgebra.clear_caches(self)
        self._tp_cache = {}

    def _suffix_parity(self, i, m):
        r"""
        The parity of the suffix key ``m`` of the factors ``i, i+1, ...``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: P = H.tensor(bc)
            sage: (m,) = P.pure_tensor(H.gen('J'), bc.gen('b')).monomial_coefficients()
            sage: P._suffix_parity(0, m), P._suffix_parity(1, m[1:])
            (1, 1)
            sage: P._suffix_parity(0, (m[0], P._vacuum_key()[1]))
            0
        """
        return sum(self._factors[j]._parity_key(m[j - i])
                   for j in range(i, self._r)) % 2

    def _suffix_nmax(self, i, a, c):
        r"""
        Upper bound for ``n`` with ``a_(n) c != 0`` on factors ``i..``
        (suffix keys ``a``, ``c``): the factor bounds ``N_j`` give ``Σ N_j
        + (#factors - 1)``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._suffix_nmax(0, a, b), W._suffix_nmax(1, a[1:], b[1:])
            (2, 0)
        """
        V = self._factors[i]
        if i == self._r - 1:
            return V._nmax_key(a[0], c[0])
        return (V._nmax_key(a[0], c[0])
                + self._suffix_nmax(i + 1, a[1:], c[1:]) + 1)

    def _tprod(self, i, a, n, c):
        r"""
        ``a_(n) c`` for suffix keys ``a, c`` of the factors ``i, i+1, ...``,
        as a dict of suffix keys: ``(a_1 ⊗ a')_(n) (c_1 ⊗ c') = (-1)^{|a'||c_1|}
        Σ_k a_1(k) c_1 ⊗ a'_(n-1-k) c'``, cached in ``_tp_cache``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._tprod(1, a[1:], 0, b[1:])
            {(((), (0,)),): 1}
            sage: W._tprod(0, a, 1, b) == x.nproduct(y, 1)._monomial_coefficients
            True
        """
        key = (i, a, n, c)
        r = self._tp_cache.get(key)
        if r is not None:
            return r
        V = self._factors[i]
        if i == self._r - 1:
            r = {(k,): v for k, v in V._nprod_mono(a[0], n, c[0]).items()}
        else:
            a1, c1, ar, cr = a[0], c[0], a[1:], c[1:]
            neg = (self._suffix_parity(i + 1, ar)
                   * V._parity_key(c1)) % 2
            N1 = V._nmax_key(a1, c1)
            N2 = self._suffix_nmax(i + 1, ar, cr)
            r = {}
            for k in range(n - 1 - N2, N1 + 1):
                d1 = V._nprod_mono(a1, k, c1)
                if not d1:
                    continue
                d2 = self._tprod(i + 1, ar, n - 1 - k, cr)
                if not d2:
                    continue
                for k1, v1 in d1.items():
                    if neg:
                        v1 = -v1
                    for k2, v2 in d2.items():
                        kk = (k1,) + k2
                        w = r.get(kk)
                        w = v1 * v2 if w is None else w + v1 * v2
                        if w:
                            r[kk] = w
                        else:
                            del r[kk]
        self._tp_cache[key] = r
        return r

    def _nmax_key(self, a, b):
        r"""
        An upper bound for ``n`` with ``a_(n) b != 0`` (basis keys), from the
        factors' bounds.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._nmax_key(a, b), len(x.ope(y))
            (2, 3)
        """
        return self._suffix_nmax(0, a, b)

    def _ope_key(self, a, b):
        r"""
        The OPE of the basis keys ``a``, ``b``: a tuple of raw pole dicts,
        trimmed (Python recursion :meth:`_tprod`).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._ope_key(a, b) == tuple(p._monomial_coefficients for p in x.ope(y))
            True
        """
        N = self._suffix_nmax(0, a, b)
        return tuple(_trim([self._tprod(0, a, n, b) for n in range(N + 1)]))

    def _no_key(self, a, b):
        r"""
        ``a_(-1) b`` for basis keys (a raw dict).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._no_key(a, b) == x.no(y)._monomial_coefficients
            True
        """
        return self._tprod(0, a, -1, b)

    def _nprod_key(self, a, n, b):
        r"""
        ``a_(n) b`` for basis keys, any integer ``n`` (a raw dict).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._nprod_key(a, -2, b) == x.nproduct(y, -2)._monomial_coefficients
            True
        """
        return self._tprod(0, a, n, b)

    def _tderiv(self, i, m, n):
        r"""
        ``∂^n`` of the suffix key ``m`` of the factors ``i, ...`` (Leibniz
        rule), a dict of suffix keys.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._tderiv(1, a[1:], 1)
            {((((0, 0),), (1,)),): 1}
            sage: W._tderiv(0, a, 1) == x.derivative()._monomial_coefficients
            True
        """
        V = self._factors[i]
        if i == self._r - 1:
            return {(k,): v for k, v in V._deriv_mono(m[0], n).items()}
        r = {}
        for j in range(n + 1):
            d1 = V._deriv_mono(m[0], j)
            d2 = self._tderiv(i + 1, m[1:], n - j)
            if not d1 or not d2:
                continue
            b = _comb(n, j)
            for k1, v1 in d1.items():
                for k2, v2 in d2.items():
                    kk = (k1,) + k2
                    w = r.get(kk)
                    w = b * v1 * v2 if w is None else w + b * v1 * v2
                    if w:
                        r[kk] = w
                    else:
                        del r[kk]
        return r

    def _deriv_key(self, m, n):
        r"""
        ``∂^n`` of the basis key ``m`` (a raw dict), by the Leibniz rule.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._deriv_key(a, 2) == x.derivative(2)._monomial_coefficients
            True
        """
        return self._tderiv(0, m, n)

    # ------------------------------------------------------------------ #
    #  Conformal weights, gradings, weight spaces                         #
    # ------------------------------------------------------------------ #

    def _strong_generators_for_ideals(self):
        r"""
        The embedded strong generators of the factors (for ideals); every factor
        must have a finite set of them.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: P = H.tensor(bc)
            sage: P._strong_generators_for_ideals()
            [J # 1, 1 # b, 1 # c]
        """
        return [self.embed(i, x) for i, V in enumerate(self._factors)
                for x in V._strong_generators_for_ideals()]

    def _key_order(self, m):
        r"""
        The key order of ideals (leading monomials): the sum of the factors'
        degrees (the first entries of their key orders, if integers), then the
        factors' key orders.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: P = H.tensor(H); (m,) = P.pure_tensor(H.gen('J'), H.gen('J').derivative()).monomial_coefficients()
            sage: P._key_order(m)
            (2, ((1, ((0, 0),)), (1, ((0, 1),))))
            sage: P._key_order(P._vacuum_key()) < P._key_order(m)
            True
        """
        ks = tuple(V._key_order(x) for V, x in zip(self._factors, m))
        deg = sum(k[0] if isinstance(k, tuple) and k and isinstance(k[0], int)
                  else 0 for k in ks)
        return (deg, ks)

    def _conformal_weights_data(self, weights):
        r"""
        ``weights``: a dict over the labels of ``_weight_generators()``, or
        ``'factors'``: the factors' conformal weights (from their Virasoro
        vectors or their conformal weights).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: B = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]})
            sage: H.tensor(B, degrees='factors')
            Traceback (most recent call last):
            ...
            ValueError: factor 0 has no conformal weights: no conformal weights chosen; ...
            sage: V = H.with_degrees({'J': 1}).tensor(
            ....:     B.with_degrees({'b': 1/2, 'g': 1/2}),
            ....:     degrees='factors')
            sage: V.degrees()
            {'J': 1, 'b': 1/2, 'g': 1/2}
            sage: V = H.tensor(B, degrees={'J': 1, 'b': 2, 'g': -1})
            sage: V.degrees()
            {'J': 1, 'b': 2, 'g': -1}
        """
        if isinstance(weights, str) and weights == 'factors':
            out = []
            for i, V in enumerate(self._factors):
                try:
                    out.append(V._weight_data())
                except ValueError as err:
                    raise ValueError('factor {} has no conformal weights: {}'
                                     .format(i, err))
            return {'factors': out}
        return VertexAlgebra._conformal_weights_data(self, weights)

    def _weight_generators(self):
        r"""
        The embedded generating sets of the factors (labels get the factor
        index appended when they clash).  Any Virasoro vector passing the
        checks of :meth:`VertexAlgebra.with_conformal_vector` is accepted, including
        ones with mixed terms (several non-vacuum tensor factors), and the
        factors need no Virasoro vector of their own; weights add over the
        factors.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}); J = H.gen('J')
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])); K = L.J(0)
            sage: V = H.tensor(L)
            sage: Vt = V.with_conformal_vector(V.embed(0, J.no(J) / 2)
            ....:                              + V.embed(1, K.no(K) / 4 + K.derivative()))
            sage: Vt.central_charge(), Vt.degrees()
            (-22, {'J': 1, 'J0': 1, 'e^(-1)': 3, 'e^(1)': -1})
            sage: Vt(V.pure_tensor(J.derivative(), L.e((1,)))).degree()
            1
            sage: V.with_conformal_vector(V.embed(0, J.no(J) / 2))
            Traceback (most recent call last):
            ...
            ValueError: T_(0) J0 = 0 is not ∂J0: T does not act as the translation operator on J0

        A mixed term whose ``L_0`` stays triangular: ``V_Z ⊗ V_Z`` with
        ``T + ∂(e^1 ⊗ e^-1)`` (the weight spaces are those of ``T``,
        dimensions ``θ_Z^2 / η^2``)::

            sage: Z = LatticeVertexAlgebra(QQ, matrix([[1]])); I = Z.J(0)
            sage: W = Z.tensor(Z)
            sage: T = W.embed(0, I.no(I) / 2) + W.embed(1, I.no(I) / 2)
            sage: X = W.pure_tensor(Z.e((1,)), Z.e((-1,)))
            sage: Wl = W.with_conformal_vector(T + X.derivative())
            doctest:warning...
            ...LogarithmicWarning: L_0 = T_(1) is not diagonalizable ...
            sage: Wl.l0_jordan_chains()['J0_0']
            [J0 # 1, e^(1) # e^(-1)]
            sage: [Wl.weight_space(n).dimension() for n in range(4)]
            [1, 6, 17, 38]

        A mixed term that makes ``L_0`` diagonalizable only on ``ψ_1 ± i
        ψ_2`` (two Majorana fermions with ``T + ∂(ψ ⊗ ψ)/3``) is refused::

            sage: M = UniversalVertexAlgebra(QQ, ['psi'], {('psi', 'psi'): [1]},
            ....:                            parities={'psi': 1})
            sage: psi = M.gen('psi'); MM = M.tensor(M)
            sage: Tm = sum(MM.embed(i, psi.derivative().no(psi)/2) for i in range(2))
            sage: MM.with_conformal_vector(Tm + MM.pure_tensor(psi, psi).derivative()/3)
            Traceback (most recent call last):
            ...
            ValueError: T_(1) - 1/2 is not nilpotent on psi_0: ... construct the vertex algebra from adapted generators
            sage: MM.with_conformal_vector(Tm).l0_is_diagonalizable()
            True
        """
        vac = self._vacuum_key()
        keys = []
        for i, V in enumerate(self._factors):
            for _, key in V._weight_generators():
                k = list(vac)
                k[i] = key
                keys.append(tuple(k))
        return [(t, k) for (_, _, t), k in
                zip(_tensor_weight_labels(self._factors), keys)]

    def _weight_data_from(self, weights):
        r"""
        The factors' weight data (from their generators' weights, a sequence
        in the order of ``_weight_generators()``).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: HH = H.tensor(H.with_degrees({'J': 1}))
            sage: [label for label, _ in HH._weight_generators()]
            ['J_0', 'J_1']
            sage: HH._weight_data_from((1, 1))
            {'factors': [{'h': [1]}, {'h': [1]}]}
        """
        out, pos = [], 0
        for V in self._factors:
            n = len(V._weight_generators())
            out.append(V._weight_data_from(weights[pos:pos + n]))
            pos += n
        return {'factors': out}

    def _weight_key(self, m, data):
        r"""
        The conformal weight of the key ``m`` for the weight data ``data``
        (``{'factors': [...]}``): the sum of the factors' weights.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: P = H.tensor(H, degrees={'J_0': 1, 'J_1': 1})
            sage: (m,) = P.pure_tensor(H.gen('J'), H.gen('J').derivative()).monomial_coefficients()
            sage: P._weight_key(m, P._weight_data()), P.degree(P.monomial(m))
            (3, 3)
        """
        return sum((V._weight_key(x, d) for V, x, d in
                    zip(self._factors, m, data['factors'])),
                   self.base_ring().zero())

    def _grading_data(self, spec, group):
        r"""
        ``spec``: ``'factors'``, the product of the gradings chosen on the
        factors: the grading group is the cartesian product of the factors'
        grading groups and ``x_1 ⊗ ... ⊗ x_r`` has grade
        ``(horizontal_grade(x_1), ..., horizontal_grade(x_r))``.  A factor without a grading counts
        as graded by the trivial group.  Changing a factor's grading
        afterwards makes this choice void (an error when it is used).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: BG = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]},
            ....:                             horizontal_grading={'b': 1, 'g': -1})
            sage: b, g = BG.gens()
            sage: L = LatticeVertexAlgebra(QQ, matrix([[1]])).with_horizontal_grading([1], group=Zmod(2))
            sage: V = BG.tensor(L, horizontal_grading='factors')
            sage: V.horizontal_grading_group()
            The Cartesian product of (Integer Ring, Ring of integers modulo 2)
            sage: x = V.pure_tensor(b.no(b), L.e((3,))); x.horizontal_grade()
            (2, 1)
            sage: x.horizontal_grade() == V.horizontal_grading_group()((2, 1))
            True
        """
        from sage.groups.additive_abelian.additive_abelian_group import (
            AdditiveAbelianGroup,
        )

        from ._grading_groups import GradingGroup, ProductGradingGroup
        if spec != 'factors' or group is not None:
            raise ValueError("the grading of a tensor product is 'factors' "
                             "(the product of the factors' gradings)")
        grs = [V._grading for V in self._factors]
        trivial = None
        groups = []
        for gd in grs:
            if gd is None:
                if trivial is None:
                    trivial = GradingGroup(AdditiveAbelianGroup([]))
                groups.append(trivial)
            else:
                groups.append(gd['group'])
        return {'group': ProductGradingGroup(groups), 'factors': grs}

    def _grade_key(self, m):
        r"""
        The grade of the key ``m`` (coordinates): the factors' grade
        coordinates concatenated (none for ungraded factors).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: BG = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]},
            ....:     horizontal_grading={'b': 1, 'g': -1}, degrees={'b': 1/2, 'g': 1/2})
            sage: V = BG.tensor(H.with_degrees({'J': 1}), horizontal_grading='factors',
            ....:               degrees='factors')
            sage: (k,) = V.pure_tensor(BG.gen('b'), H.gen('J')).monomial_coefficients()
            sage: V._grade_key(k)
            (1,)
            sage: V.horizontal_grade(V.monomial(k))
            (1, ())
        """
        return sum((V._grade_key(x) if gd is not None else ()
                    for V, x, gd in zip(self._factors, m,
                                        self._grading['factors'])), ())

    def _factor_grades(self, G):
        r"""
        The factor grades of the grade ``G`` (coordinate tuples; ``None``
        for an ungraded factor, all ``None`` for ``G = None``: any grade).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: BG = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]},
            ....:     horizontal_grading={'b': 1, 'g': -1}, degrees={'b': 1/2, 'g': 1/2})
            sage: V = BG.tensor(H.with_degrees({'J': 1}), horizontal_grading='factors',
            ....:               degrees='factors')
            sage: V._factor_grades(None), V._factor_grades((1,))
            ([None, None], [(1,), None])
        """
        if G is None:
            return [None] * self._r
        parts = self._grading['group'].split(G)
        return [None if gd is None else Gi
                for Gi, gd in zip(parts, self._grading['factors'])]

    def _weight_bounds(self, G, data):
        r"""
        ``(lb, D)`` (see ``VertexAlgebra._weight_bounds``): ``D`` the lcm of the
        factors' denominators; ``lb`` the sum of the factors' lower bounds,
        ``None`` if one is uncertified, ``+Infinity`` if a factor has no key of
        its grade.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: BG = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]},
            ....:     horizontal_grading={'b': 1, 'g': -1}, degrees={'b': 1/2, 'g': 1/2})
            sage: V = BG.tensor(H.with_degrees({'J': 1}), horizontal_grading='factors',
            ....:               degrees='factors')
            sage: (k,) = V.pure_tensor(BG.gen('b'), H.gen('J')).monomial_coefficients()
            sage: G = V._grade_key(k)
            sage: V._weight_bounds(G, V._weight_data())
            (0, 2)
            sage: V._weight_bounds(None, V._weight_data())
            (0, 2)
        """
        from sage.arith.functions import lcm
        bounds = [V._weight_bounds(Gi, d) for V, Gi, d in
                  zip(self._factors, self._factor_grades(G), data['factors'])]
        D = int(lcm([d for _, d in bounds]))
        lbs = [lb for lb, _ in bounds]
        if any(lb is Infinity for lb in lbs):
            return Infinity, D
        if any(lb is None for lb in lbs):
            return None, D
        return sum(lbs), D

    def _weight_space_enumeration(self, W, G, data):
        r"""
        The weight space of weight ``W`` and grade ``(G_1, ..., G_r)`` is
        ``⊕ S_1(W_1, G_1) ⊗ ... ⊗ S_r(W_r, G_r)`` over ``W_1 + ... + W_r = W``,
        the ``W_i`` in ``(1/D) ZZ`` for a common denominator ``D``.  If
        every factor has a finiteness certificate for its grade, the factor
        weights are bounded below and the sum is finite; otherwise a
        nonempty space is infinite dimensional (its term for an uncertified
        factor is) and it is enumerated in stages ``s``: the splittings with
        ``max |D W_i| <= s`` combined with the factor stages ``<= s``,
        ``s`` attained.

        EXAMPLES:

        Two free bosons, each ``J`` of weight 1 (dimensions ``p_2(n)``)::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: N.<J> = NormallyOrderedExpressions(QQ)
            sage: H = UniversalVertexAlgebra(QQ, N, {('J', 'J'): [0, 1]}, conformal_vector=J*J/2)
            sage: V = H.tensor(H, conformal_vector='factors')
            sage: [V.weight_space(n).dimension() for n in range(6)]
            [1, 2, 5, 10, 20, 36]
            sage: V.weight_space(2).basis()
            [1 # :J J:, 1 # dJ, J # J, :J J: # 1, dJ # 1]

        βγ with charge ``β ↦ 1, γ ↦ -1`` times ``bc`` (``b`` of weight 2,
        ``c`` of weight -1) with charge ``b ↦ 1, c ↦ -1``: finite once the
        βγ charge is fixed::

            sage: N.<be, ga> = NormallyOrderedExpressions(QQ)
            sage: BG = UniversalVertexAlgebra(QQ, N, {('be', 'ga'): [-1]},
            ....:                             conformal_vector=-be*ga.derivative(),
            ....:                             horizontal_grading={'be': 1, 'ga': -1})
            sage: N.<b, c> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'c': 1})
            sage: bc = UniversalVertexAlgebra(QQ, N, {('b', 'c'): [1]},
            ....:     conformal_vector=-2*b*c.derivative() - b.derivative()*c,
            ....:     horizontal_grading={'b': 1, 'c': -1})
            sage: V = BG.tensor(bc, conformal_vector='factors', horizontal_grading='factors')
            sage: V.central_charge()
            -24
            sage: [V.weight_space(n, horizontal_grade=(0, 0)).dimension() for n in range(6)]
            [1, 2, 6, 14, 32, 66]
            sage: V.weight_space(1, horizontal_grade=(0, 0)).basis()
            [1 # :b c:, :be ga: # 1]
            sage: V.weight_space(1).is_finite_dimensional()
            False

        TESTS:

        ``V_Z ⊗ V_Z ≅ V_{diag(1,1)}``: the same dimensions in every weight
        and lattice grade::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: Z = LatticeVertexAlgebra(QQ, matrix([[1]])); I = Z.J(0)
            sage: Z1 = Z.with_conformal_vector(I.no(I)/2).with_horizontal_grading('lattice')
            sage: Z2 = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]]))
            sage: K0, K1 = Z2.heisenberg_currents()
            sage: Z2 = Z2.with_conformal_vector(K0.no(K0)/2 + K1.no(K1)/2).with_horizontal_grading(
            ....:     'lattice')
            sage: V = Z1.tensor(Z1, conformal_vector='factors', horizontal_grading='factors')
            sage: all(V.weight_space(n, horizontal_grade=(a, b)).dimension()
            ....:     == Z2.weight_space(n, horizontal_grade=(a, b)).dimension()
            ....:     for n in range(4) for a in range(-2, 3) for b in range(-2, 3))
            True

        Graded dimensions are the convolutions of the factors' ones, also
        with negative weights (``A_1`` with a background charge: ``e^q`` of
        weight ``q^2 - 3q``) and with torsion gradings (``V_Z`` with
        ``e^q ↦ q mod 2`` times the ``N = 1`` algebra with ``G ↦ 1 mod 2``)::

            sage: def conv(V, n, G, step=1, lo=-3):
            ....:     F1, F2 = V.factors()
            ....:     return sum(F1.weight_space(w, horizontal_grade=G[0]).dimension()
            ....:                * F2.weight_space(n - w, horizontal_grade=G[1]).dimension()
            ....:                for w in srange(lo, n - lo + 1, step))
            sage: A1 = LatticeVertexAlgebra(QQ, matrix([[2]])); JA = A1.J(0)
            sage: A1 = A1.with_conformal_vector(JA.no(JA)/4 + 3*JA.derivative()/2)
            sage: [A1.degree(A1.e((q,))) for q in range(-1, 4)]
            [4, 0, -2, -2, 0]
            sage: V = BG.tensor(A1.with_horizontal_grading('lattice'),
            ....:               conformal_vector='factors', horizontal_grading='factors')
            sage: all(V.weight_space(n, horizontal_grade=(g1, g2)).dimension() == conv(V, n, (g1, g2))
            ....:     for n in range(-2, 2) for g1 in (-1, 0, 1) for g2 in range(4))
            True
            sage: N.<L, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
            sage: N1 = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 7/20],
            ....:     ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 7/15]},
            ....:     conformal_vector='L', horizontal_grading={'G': 1},
            ....:     horizontal_grading_group=Zmod(2))
            sage: V = Z1.with_horizontal_grading([1], group=Zmod(2)).tensor(
            ....:     N1, conformal_vector='factors', horizontal_grading='factors')
            sage: V.central_charge()
            17/10
            sage: all(V.weight_space(n, horizontal_grade=(a, b)).dimension()
            ....:     == conv(V, n, (a, b), 1/2, 0) for n in srange(0, 4, 1/2)
            ....:     for a in (0, 1) for b in (0, 1))
            True

        An ungraded factor counts as graded by the trivial group::

            sage: N.<J> = NormallyOrderedExpressions(QQ)
            sage: H = UniversalVertexAlgebra(QQ, N, {('J', 'J'): [0, 1]}, conformal_vector=J*J/2)
            sage: V = BG.tensor(H, conformal_vector='factors', horizontal_grading='factors')
            sage: V.horizontal_grading_group()
            The Cartesian product of (Integer Ring, Trivial group)
            sage: all(V.weight_space(n, horizontal_grade=(0, ())).dimension()
            ....:     == conv(V, n, (0, None), 1, 0) for n in range(5))
            True

        Every basis vector of the product-graded spaces of ``βγ ⊗ bc`` has
        the right ``L_0``-eigenvalue, ``T_(0) = ∂`` and the right grade; so
        do the first vectors of the lazily enumerated, infinite ungraded
        space::

            sage: def weight_ok(V, xs, n, G=None):
            ....:     T = V.conformal_vector()
            ....:     return all(T.nproduct(x, 1) == n*x and T.nproduct(x, 0) == x.derivative()
            ....:                and (G is None or x.horizontal_grade() == V.horizontal_grading_group()(G))
            ....:                for x in xs)
            sage: V = BG.tensor(bc, conformal_vector='factors', horizontal_grading='factors')
            sage: all(weight_ok(V, V.weight_space(n, horizontal_grade=G).basis(), n, G)
            ....:     for n in range(-1, 3) for G in cartesian_product([[-1, 0, 1]]*2))
            True
            sage: from itertools import islice
            sage: xs = list(islice(iter(V.weight_space(1)), 20))
            sage: len(set(xs)), weight_ok(V, xs, 1)
            (20, True)

        A factor without finiteness certificate (βγ graded by ``Z/2``
        only): the space is enumerated in stages, without repeats, and the
        stages up to 4 are exactly the keys whose βγ part has even length
        at most 4::

            sage: BGz = BG.with_horizontal_grading({'be': 1, 'ga': 1}, group=Zmod(2))
            sage: V = BGz.tensor(H, conformal_vector='factors', horizontal_grading='factors')
            sage: S = V.weight_space(1, horizontal_grade=(0, ()))
            sage: S.is_finite_dimensional()
            False
            sage: stages = [[V.monomial(k) for k in S._enum._stage(s)] for s in range(5)]; stages
            [[],
             [1 # J],
             [:ga ga: # J, :dga ga: # 1, :be ga: # 1],
             [],
             [:ga ga ga ga: # J, :dga ga ga ga: # 1, :be ga ga ga: # 1]]
            sage: xs = sum(stages, []); len(set(xs)) == len(xs), weight_ok(V, xs, 1, (0, ()))
            (True, True)
        """
        from itertools import product as iproduct

        from sage.arith.functions import lcm

        from .weight_spaces import _KeyEnumeration
        Gs = self._factor_grades(G)
        facs = self._factors
        fdata = data['factors']
        r = self._r
        bounds = [V._weight_bounds(Gi, d) for V, Gi, d in zip(facs, Gs, fdata)]
        empty = _KeyEnumeration(True, factory=lambda: iter(()))
        if any(lb is Infinity for lb, _ in bounds):
            return empty
        D = int(lcm([d for _, d in bounds]))
        N = QQ(W) * D
        if N not in ZZ:
            return empty
        N = int(N)
        lows = [None if lb is None else int(ceil(lb * D)) for lb, _ in bounds]
        enums = {}

        def enum(i, k):
            e = enums.get((i, k))
            if e is None:
                e = enums[(i, k)] = facs[i]._weight_space_enumeration(
                    QQ(k) / D, Gs[i], fdata[i])
            return e

        if all(lo is not None for lo in lows):
            lists = {}

            def keys(i, k):
                ks = lists.get((i, k))
                if ks is None:
                    ks = lists[(i, k)] = list(enum(i, k).factory())
                return ks

            suffix = [0] * (r + 1)
            for i in range(r - 1, -1, -1):
                suffix[i] = suffix[i + 1] + lows[i]

            def splits(i, rest):
                if i == r - 1:
                    if rest >= lows[i] and keys(i, rest):
                        yield (rest,)
                    return
                for k in range(lows[i], rest - suffix[i + 1] + 1):
                    if keys(i, k):
                        for tail in splits(i + 1, rest - k):
                            yield (k,) + tail

            def factory():
                for ks in splits(0, N):
                    yield from iproduct(*[keys(i, k) for i, k in enumerate(ks)])
            return _KeyEnumeration(True, factory=factory)

        stages = {}

        def stage_keys(i, k, t):
            """Keys of stage ``t`` of the factor space ``S_i(k / D)``."""
            ks = stages.get((i, k, t))
            if ks is None:
                e = enum(i, k)
                if e.finite:
                    ks = list(e.factory()) if t == 0 else []
                else:
                    ks = list(e._stage(t))
                stages[(i, k, t)] = ks
            return ks

        def splits_upto(s):
            def rec(i, rest):
                if i == r - 1:
                    if abs(rest) <= s and (lows[i] is None or rest >= lows[i]):
                        yield (rest,)
                    return
                lo = -s if lows[i] is None else max(-s, lows[i])
                for k in range(lo, s + 1):
                    for tail in rec(i + 1, rest - k):
                        yield (k,) + tail
            return rec(0, N)

        def stage(s):
            for ks in splits_upto(s):
                if max(abs(k) for k in ks) == s:
                    lists = [[x for t in range(s + 1)
                              for x in stage_keys(i, k, t)]
                             for i, k in enumerate(ks)]
                    yield from iproduct(*lists)
                    continue
                old = [[x for t in range(s) for x in stage_keys(i, k, t)]
                       for i, k in enumerate(ks)]
                new = [stage_keys(i, k, s) for i, k in enumerate(ks)]
                # at least one factor in its stage s: the first such is j
                for j in range(r):
                    if new[j]:
                        yield from iproduct(*(old[:j] + [new[j]] + [
                            o + n for o, n in zip(old[j + 1:], new[j + 1:])]))
        return _KeyEnumeration(False, stage=stage)

    # ------------------------------------------------------------------ #
    #  Parallel: workers also ship the factors' new cache entries         #
    # ------------------------------------------------------------------ #

    def _build_engine(self):
        """
        The C++ engine (:class:`_TensorEngineKernel`) if it is available and
        not switched off (``implementation='python'``), every factor is a universal or lattice
        vertex algebra not restricted to pure Python, and the
        factors share an engine ring: the base ring itself (``QQ``,
        ``GF(p)``, absolute number fields ``K``, polynomial rings over ``QQ``
        or ``K``) or, for ``Frac(A)``, ``A``
        (if no universal factor's OPEs need denominators: no ``u_j``;
        element coefficients with non-constant denominators take the
        generic path); else ``None`` (the Python recursion).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
            sage: E = H.tensor(F)._build_engine()
            sage: E is None or type(E).__name__ == '_TensorEngineKernel'
            True
            sage: H.tensor(F, implementation='python')._build_engine() is None
            True

        A factor whose OPEs need denominators (``W_3`` over ``Q(c)``:
        ``1/(5c + 22)``): no engine, the Python recursion::

            sage: from vertex_algebras.catalog import W3
            sage: c = Frac(QQ['c']).gen()
            sage: bc = UniversalVertexAlgebra(c.parent(), ['b', 'c'], {('b', 'c'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: W3(c.parent(), c).tensor(bc)._build_engine() is None
            True
        """
        if _TVA_ENGINE is None or self._implementation == 'python':
            return None
        from . import lattice_vertex_algebra as LVA
        from . import universal_vertex_algebra as UVA
        F = self.base_ring()
        try:
            kind = _engine_ring(F)
            Rk, up, down = F, None, None
            alg = None
            if kind is None:
                nums = []
                for V in self._factors:
                    if isinstance(V, UVA.UniversalVertexAlgebra):
                        nums += [c for poles in V._opes.values() for el in poles
                                 for c in el._monomial_coefficients.values()]
                    elif isinstance(V, LVA.LatticeVertexAlgebra):
                        nums += [c for row in (V._eps_mat or ()) for c in row]
                alg = self._algebraic_engine_ring(nums)
            if alg is not None:               # QQbar / AA: in a number field
                kind, alg = alg
                Rk, up, down = alg.K, alg.up, alg.down
            elif kind is None:
                A = F.ring()
                if A is F or F is not A.fraction_field():
                    return None
                kind = _engine_ring(A)
                if kind not in ('qpoly', 'qmpoly', 'nfmpoly'):
                    return None

                def up(x):
                    d = x.denominator()
                    if not d.is_constant():
                        raise ValueError('denominator %s is not a constant' % d)
                    return A(x.numerator()) / d.constant_coefficient()

                Rk, down = A, F
            specs = []
            for V in self._factors:
                if isinstance(V, LVA.LatticeVertexAlgebra):
                    if V._implementation == 'python':
                        return None
                    specs.append((V, None))
                elif isinstance(V, UVA.UniversalVertexAlgebra):
                    if V._implementation == 'python':
                        return None
                    if alg is not None:
                        decl = {key: [{m: alg.up(c) for m, c in el._monomial_coefficients.items()}
                                      for el in poles] for key, poles in V._opes.items()}
                    elif down is None:
                        decl = {key: [dict(el._monomial_coefficients) for el in poles]
                                for key, poles in V._opes.items()}
                    else:
                        lift = UVA._lifted_kernel(V)
                        if lift is None or lift.denominator_factors() or lift.ring() is not Rk:
                            return None
                        decl = lift._decl
                    specs.append((V, decl))
                else:
                    return None
            E = _TensorEngineKernel(self, F, kind, Rk, specs, up=up, down=down)
            E._alg = alg
            return E
        except (AttributeError, TypeError, ValueError, ArithmeticError,
                NotImplementedError):
            return None

    def _cache_owners(self):
        r"""
        The tensor product and the factors' cache owners (without repetitions):
        forked workers ship the factors' new cache entries back too.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._cache_owners() == [W, H, F]
            True
            sage: H.tensor(H)._cache_owners() == [H.tensor(H), H]
            True
        """
        out, seen = [self], {id(self)}
        for V in self._factors:
            for o in V._cache_owners():
                if id(o) not in seen:
                    seen.add(id(o))
                    out.append(o)
        return out

    def _raw_caches(self):
        r"""
        The base caches and ``_tp_cache`` (all hold raw dicts).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); W = H.tensor(F)
            sage: x = W.pure_tensor(H.gen('J'), F.e((1,))); y = W.pure_tensor(H.gen('J'), F.e((-1,)))
            sage: (a,), (b,) = x.monomial_coefficients(), y.monomial_coefficients()
            sage: W._raw_caches()[-1]
            '_tp_cache'
        """
        return VertexAlgebra._raw_caches(self) + ['_tp_cache']
