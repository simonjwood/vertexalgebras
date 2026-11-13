# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Lattice vertex (super)algebras via Heisenberg currents and lattice currents.

A subclass of :class:`vertex_algebra.VertexAlgebra` (linearity, caching,
parallel evaluation and the element API live there).  Products are computed by a C++ engine
(``_lattice_engine``, ``_LatticeEngineKernel``) over
``QQ``, absolute number fields ``K``, polynomial rings over ``QQ`` or ``K``
and their fraction fields (in the threads of
:class:`vertex_algebra.VertexAlgebraParallelism`), else in pure Python;
both give identical results (``implementation=`` chooses one).  The defining data is a coefficient ring (containing ``QQ``),
a symmetric, integral, non-degenerate Gram matrix ``B`` (not necessarily
positive definite) and a lattice 2-cocycle ``epsilon``.

The lattice is ``ZZ^n`` with basis ``x_1, ..., x_n`` (0-based internally:
``x_0, ..., x_{n-1}``).  Every direction gives a Heisenberg current ``J_i``
with ``J_i(z) J_j(w) ~ B[i,j]/(z-w)^2``, and every lattice point ``q`` a
lattice current ``e^q``, ``e^0`` being the vacuum.

A basis element is a normally ordered monomial ``:∂^{n_1} J_{i_1} ...
∂^{n_k} J_{i_k} e^q:`` with the lattice current on the far right and the
Heisenberg letters ordered by derivative (larger to the left), then by index
(smaller to the left, e.g. ``:J0 J1 e^q:``), the same convention as for
universal vertex algebras.  Internally this is the key ``(letters, q)`` with
``letters`` a tuple of ``(index, derivative)`` in that order and ``q`` a
tuple of integers.

Conventions (Kac, *Vertex algebras for beginners*, 5.4-5.5).  The Heisenberg
modes satisfy ``[h_i(m), h_j(k)] = m B[i,j] δ_{m+k,0}``; the letter
``∂^n J_i`` is the state ``n! h_i(-n-1)``.  With ``q(m) = Σ q_i h_i(m)``::

    Y(e^q, z) = e_q z^{q(0)} exp(Σ_{m>0} q(-m) z^m/m) exp(-Σ_{m>0} q(m) z^{-m}/m),

so that ``e^q_(n) e^p = ε(q,p) S_{-n-1-B(q,p)}(q) e^{q+p}`` where ``S_k`` are
the Schur polynomials ``Σ_k S_k z^k = exp(Σ_{m>0} q(-m) z^m/m)``.  Hence the
OPE ``e^q(z) e^p(w)`` starts at ``(z-w)^{B(q,p)}``: it has ``-B(q,p)`` poles
when ``B(q,p) < 0`` and is regular otherwise.  The parity of ``e^q`` is
``B(q,q) mod 2``; locality requires
``ε(q,p)/ε(p,q) = (-1)^{B(q,p) + B(q,q) B(p,p)}``.

EXAMPLES:

The ``A_1`` root lattice (``SU(2)_1``)::

    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
    sage: L = LatticeVertexAlgebra(QQ, matrix([[2]]))
    sage: L
    Lattice vertex algebra of rank 1 with Gram matrix [2] over Rational Field
    sage: J = L.J(0); e = L.e((1,)); f = L.e((-1,))
    sage: J.ope(J)
    [0, 2]
    sage: J.ope(e)
    [2*e^(1)]
    sage: e.ope(J)
    [-2*e^(1)]
    sage: e.derivative()
    :J0 e^(1):
    sage: e.ope(f)
    [J0, 1]
    sage: e.ope(e)
    []
    sage: e.no(e)                  # e^q(z) e^q(w) = O((z-w)^2)
    0
    sage: e.nproduct(e, -3)
    e^(2)
    sage: e.ope(e.derivative())
    []

The Heisenberg current in direction ``i`` acts on ``e^q`` by ``B(x_i, q)``::

    sage: U = LatticeVertexAlgebra(QQ, matrix([[0, 1], [1, 0]]))   # hyperbolic plane
    sage: J0, J1 = U.heisenberg_currents()
    sage: q = U.e((2, -3))
    sage: J0.ope(q), J1.ope(q)
    ([-3*e^(2,-3)], [2*e^(2,-3)])
    sage: q.derivative() == 2*J0.no(q) - 3*J1.no(q)
    True
    sage: U.e((1, 0)).ope(U.e((0, -2)))
    [:J0 e^(1,-2):, e^(1,-2)]

The class does not fix a conformal vector; any choice can be built from the
currents.  For example the Sugawara vector
``Tsug = 1/2 Σ (B^{-1})_{ij} :J_i J_j:`` (central charge ``rank``) and its
background-charge deformation ``Tsug + ∂J_0`` (central charge
``rank - 12 B(x_0,x_0)``)::

    sage: A2 = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
    sage: J = A2.heisenberg_currents()
    sage: Binv = A2.gram_matrix().inverse()
    sage: Tsug = sum(Binv[i, j]/2 * J[i].no(J[j]) for i in range(2) for j in range(2))
    sage: Tsug.ope(Tsug) == [Tsug.derivative(), 2*Tsug, 0, A2(2/2)]
    True
    sage: T = Tsug + J[0].derivative()
    sage: T.ope(T) == [T.derivative(), 2*T, 0, A2((2 - 12*2)/2)]
    True
    sage: q = A2.e((1, 1))
    sage: Tsug.nproduct(q, 1) == q, T.nproduct(q, 1) == (1 - 1)*q
    (True, True)

An odd lattice gives a vertex superalgebra (the charged free fermion)::

    sage: F = LatticeVertexAlgebra(QQ, matrix([[1]]))
    sage: psi = F.e((1,)); psis = F.e((-1,))
    sage: psi.parity(), F.J(0).parity()
    (1, 0)
    sage: psi.ope(psis)
    [1]
    sage: psis.ope(psi)
    [1]
    sage: psi.no(psi)
    0

Compound coefficients are bracketed::

    sage: K.<a> = Frac(QQ['a'])
    sage: LK = LatticeVertexAlgebra(K, matrix([[2]]))
    sage: (a - 1) * LK.J(0) + a * LK.e((1,))
    a*e^(1) + (a - 1)*J0

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import warnings
from math import factorial as _fact

from sage.matrix.constructor import matrix
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .vertex_algebra import (
    VertexAlgebra,
    _compiled_module,
    _derivative_str,
    _engine_ring,
    _EngineKernel,
    _is_int_like,
    _normalize_implementation,
    _trim,
    _word_str,
)


def _letter_key(t):
    """
    Sort key of a letter ``(index, derivative)``: a basis monomial lists
    its letters in DEcreasing key order, i.e. higher derivatives first, then
    lower indices first (``:J0 J1:``, ``:∂ J1 J0:``).

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _letter_key
        sage: _letter_key((1, 2))
        (2, -1)
        sage: sorted([(0, 0), (1, 1), (0, 1)], key=_letter_key, reverse=True)
        [(0, 1), (1, 1), (0, 0)]
    """
    return (t[1], -t[0])


def _insert_letter(letters, x):
    """Insert the letter ``x`` into the weakly descending tuple ``letters``.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _insert_letter
        sage: _insert_letter(((0, 2), (1, 0)), (0, 1)), _insert_letter((), (0, 0))
        (((0, 2), (0, 1), (1, 0)), ((0, 0),))
    """
    kx = _letter_key(x)
    for t, y in enumerate(letters):
        if _letter_key(y) <= kx:
            return letters[:t] + (x,) + letters[t:]
    return letters + (x,)


def _merge_letters(la, lb):
    """Commutative product of two weakly descending letter tuples.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _merge_letters
        sage: _merge_letters(((0, 2), (0, 0)), ((1, 1), (0, 0)))
        ((0, 2), (1, 1), (0, 0), (0, 0))
        sage: _merge_letters((), ((1, 0),)), _merge_letters(((0, 0),), ((1, 1),))
        (((1, 0),), ((1, 1), (0, 0)))
    """
    if not la:
        return lb
    if not lb:
        return la
    if len(lb) == 1:
        return _insert_letter(la, lb[0])
    if len(la) == 1:
        return _insert_letter(lb, la[0])
    return tuple(sorted(la + lb, key=_letter_key, reverse=True))


def _degree(letters):
    """Mode degree ``Σ (n_j + 1)`` of a letter tuple.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _degree
        sage: _degree(((0, 2), (1, 0))), _degree(())
        (4, 0)
    """
    return sum(n + 1 for _, n in letters)


#: Above this many distinct pairs, ``LatticeVertexAlgebra._products()``
#: computes the letter series in batches of this size and drops them after
#: summing (bounded memory; small calls keep every pair's series cached).
LS_BATCH_PAIRS = 2000


# C++ engine (_lattice_engine.hpp on the shared _engine_base.hpp): letter
# series, group sums and E^- in C++; the pure-Python kernels below give
# identical results.
_LVA_ENGINE, _ENGINE_OFF = _compiled_module('_lattice_engine')


class _LatticeEngineKernel(_EngineKernel):
    r"""
    The C++ engine of a lattice vertex algebra ``V``
    (``_lattice_engine``): letter series, their sums per
    lattice group and ``E^-`` in C++, see :class:`vertex_algebra._EngineKernel`.
    The cocycle values ``ε(q, p)`` of every call come from ``V`` (so callable
    cocycles work too); if one cannot be lifted to the engine's ring, the
    key-level calls use ``V``'s own kernel and the element-level calls return
    ``None`` (generic path).

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: from vertex_algebras import lattice_vertex_algebra as LV
        sage: L = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
        sage: E = L._engine()            # None without the compiled engine
        sage: E is None or (E.kind(), E.ring()) == ('q', QQ)
        True
        sage: a, b = (((0, 0),), (1, 0)), ((), (0, 1))
        sage: E is None or E.ope_mono(a, b) == LatticeVertexAlgebra._ope_key(L, a, b)
        True

    TESTS:

    The engine against the pure-Python kernels: an odd, an indefinite
    lattice, a callable cocycle, ``Q(c)`` with a coefficient the lift
    cannot express (``1/(c + 3)``: the generic path) and a tower; then with
    2 threads (the same results, in the same order)::

        sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
        sage: def work(V, g):
        ....:     n = V.rank()
        ....:     J, ep, em = V.J(0), V.e([1] + [0]*(n-1)), V.e([-1] + [0]*(n-1))
        ....:     f = V.e([0]*(n-1) + [1])
        ....:     X = J.no(ep) + (2 + g)*em.derivative(2) + J.derivative().no(f)
        ....:     Y = ep.no(f) - 3*J.no(J).no(em) + g^2*f.derivative()
        ....:     return [list(x.monomial_coefficients().items())
        ....:             for x in X.ope(Y) + [X.nproduct(Y, -3), X.derivative(3)]]
        sage: def eps(q, p):
        ....:     return (-1)^(q[0]*p[1])
        sage: F.<c> = Frac(QQ['c'])
        sage: K.<s> = QuadraticField(5)
        sage: M.<i> = K.extension(QQ['y']([1, 0, 1]))
        sage: A2 = matrix([[2, -1], [-1, 2]])
        sage: for R, B, e, g in [(QQ, matrix([[1]]), None, 0), (QQ, matrix([[1, 0], [0, -1]]), None, 1/2),
        ....:                    (QQ, matrix([[2, 0], [0, 2]]), eps, 3), (F, A2, None, 1/(c + 3)),
        ....:                    (M, A2, None, i + s)]:
        ....:     V = LatticeVertexAlgebra(R, B, epsilon=e)
        ....:     res = work(V, R(g))
        ....:     VP = LatticeVertexAlgebra(R, B, epsilon=e, implementation='python')
        ....:     assert sorted(map(sorted, res)) == sorted(map(sorted, work(VP, R(g)))), R
        ....:     V.clear_caches()
        ....:     VertexAlgebraParallelism().set(2)
        ....:     assert work(V, R(g)) == res, R
        ....:     VertexAlgebraParallelism().reset()

    Ctrl-C (here an alarm) interrupts a long product and a long batch of
    products, serially and with threads; afterwards the algebra computes as
    before, threads giving the serial result in the serial order::

        sage: J = LatticeVertexAlgebra(QQ, matrix([[12]])).J(0)
        sage: L = LatticeVertexAlgebra(QQ, matrix([[12]]), conformal_vector=J.no(J)/24 + J.derivative()/12,
        ....:                          horizontal_grading='lattice')
        sage: def ws(w, g):
        ....:     return sum(L.monomial(k) for k in L.weight_space(w, g).basis_keys())
        sage: big, bigm, small, smallm = ws(16, 1), ws(16, -1), ws(9, 1), ws(9, -1)
        sage: ref = [list(x.monomial_coefficients().items()) for x in small.ope(smallm)]
        sage: ks1, ks2 = (list(L.weight_space(14, g).basis_keys()) for g in (1, -1))
        sage: els = [{k: QQ(1)} for k in ks1 + ks2]
        sage: tasks = [(i, n, len(ks1) + j) for i in range(len(ks1)) for j in range(len(ks2))
        ....:          for n in (-3, -1, 0, 2)]
        sage: few = [dict(small.monomial_coefficients()), dict(smallm.monomial_coefficients())]
        sage: for n in (1, 2):  # long time
        ....:     for f in (lambda: big.ope(bigm), lambda: L._nprod_raw_batch(els, tasks)):
        ....:         L.clear_caches()
        ....:         VertexAlgebraParallelism().set(n)
        ....:         try:
        ....:             alarm(0.3)
        ....:             _ = f()
        ....:         except KeyboardInterrupt:
        ....:             pass
        ....:         finally:
        ....:             cancel_alarm()
        ....:         assert [list(x.monomial_coefficients().items()) for x in small.ope(smallm)] == ref
        ....:         b = L._nprod_raw_batch(few, [(0, -1, 1), (1, 0, 0), (0, 2, 1)])
        ....:         assert b is None or b == [L._nprod_raw(few[i], m, few[j])
        ....:                                   for i, m, j in [(0, -1, 1), (1, 0, 0), (0, 2, 1)]]
        sage: VertexAlgebraParallelism().reset()
    """

    def __init__(self, V, F, kind, Rk, up=None, down=None):
        r"""
        TESTS::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2]]))
            sage: E = L._engine()            # None without the compiled engine
            sage: E is None or E._V is L
            True
        """
        _EngineKernel.__init__(self, F, kind, Rk, up, down)
        self._V = V
        self.attach(_LVA_ENGINE.Engine(kind, V._B, self.nvars(), self.modulus(),
                                       self.minpoly()))

    def _eps(self, qs, ps):
        """The cocycle values for all pairs, lifted (may raise).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E._eps([(1,)], [(-1,), (1,)]) == [      # values in engine form
            ....:     ((1,), (-1,), [((), 1, 1)]), ((1,), (1,), [((), 1, 1)])]
            True
        """
        V, lift = self._V, self._lift_scalar
        return [(q, p, lift(V._eps(q, p))) for q in qs for p in ps]

    def _eps_or_none(self, qs, ps):
        """
        :meth:`_eps`, or ``None`` if a value cannot be lifted to the engine's
        ring.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E._eps_or_none([(1,)], [(-1,)]) == E._eps([(1,)], [(-1,)])
            True
            sage: Lc = LVA(QQ, matrix([[2]]), epsilon=lambda q, p: 'x')     # not a number
            sage: Ec = Lc._engine()
            sage: Ec is None or Ec._eps_or_none([(1,)], [(1,)]) is None
            True
        """
        try:
            return self._eps(qs, ps)
        except (ValueError, ArithmeticError, TypeError, AttributeError):
            return None

    def ope_mono(self, a, b):
        """
        The OPE of two basis keys (tuple of raw pole dicts); ``V``'s Python
        kernel if the cocycle value cannot be lifted.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.ope_mono(a, b) == LVA._ope_key(L, a, b)
            True
        """
        eps = self._eps_or_none([a[1]], [b[1]])
        if eps is None:
            return LatticeVertexAlgebra._ope_key(self._V, a, b)
        out = [self._el_down(el) for el in self._E.ope_mono(a, b, eps)]
        while out and not out[-1]:
            out.pop()
        return tuple(out)

    def no_mono(self, a, b):
        """
        The normally ordered product of two basis keys (raw dict).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.no_mono(a, b) == LVA._no_key(L, a, b)
            True
        """
        eps = self._eps_or_none([a[1]], [b[1]])
        if eps is None:
            return LatticeVertexAlgebra._no_key(self._V, a, b)
        return self._el_down(self._E.no_mono(a, b, eps))

    def nprod_mono(self, a, n, b):
        """
        The ``n``-th product of two basis keys (raw dict).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.nprod_mono(a, -2, b) == LVA._nprod_key(L, a, -2, b)
            True
        """
        eps = self._eps_or_none([a[1]], [b[1]])
        if eps is None:
            return LatticeVertexAlgebra._nprod_key(self._V, a, n, b)
        return self._el_down(self._E.nprod_el([(a, self._one)], int(n),
                                              [(b, self._one)], eps, 1))

    def ope(self, dA, dB, nthreads):
        """
        The OPE of two raw elements (list of pole dicts), or ``None``
        (coefficients or cocycle values the engine cannot take).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.ope({a: 1}, {b: 1}, 1) == list(LVA._ope_key(L, a, b))
            True
        """
        lifted = self._lift_all(dA, dB)
        eps = self._eps_or_none({k[1] for k in dA}, {k[1] for k in dB})
        if lifted is None or eps is None:
            return None
        return [self._el_down(el)
                for el in self._E.ope_el(lifted[0], lifted[1], eps, nthreads)]

    def ope_many(self, dA, dBs, nthreads):
        """
        The OPEs of ``dA`` with every ``dB`` of ``dBs`` in one engine call,
        or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.ope_many({a: 1}, [{b: 1}, {a: 1}], 1) == [
            ....:     list(LVA._ope_key(L, a, b)), list(LVA._ope_key(L, a, a))]
            True
        """
        lifted = self._lift_all(dA, *dBs)
        eps = self._eps_or_none({k[1] for k in dA},
                                {k[1] for dB in dBs for k in dB})
        if lifted is None or eps is None:
            return None
        return [[self._el_down(el) for el in P]
                for P in self._E.ope_el_many(lifted[0], lifted[1:], eps, nthreads)]

    def nprod(self, dA, n, dB, nthreads):
        """
        ``dA_(n) dB`` of raw elements, or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.nprod({a: 1}, -1, {b: 1}, 1) == LVA._no_key(L, a, b)
            True
        """
        lifted = self._lift_all(dA, dB)
        eps = self._eps_or_none({k[1] for k in dA}, {k[1] for k in dB})
        if lifted is None or eps is None:
            return None
        return self._el_down(self._E.nprod_el(lifted[0], int(n), lifted[1],
                                              eps, nthreads))

    def nprod_many_engine(self, els, tasks, nthreads):
        """
        ``els[i]_(n) els[j]`` for the tasks ``(i, n, j)``, elements in the
        engine's form (no conversions; see
        :meth:`vertex_algebra._EngineKernel.nprod_many_engine`), or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or [E.from_engine(x) for x in E.nprod_many_engine(
            ....:     [E.to_engine({a: 1}), E.to_engine({b: 1})], [(0, -1, 1)], 1)] == [
            ....:     LVA._no_key(L, a, b)]
            True
        """
        eps = self._eps_or_none({k[1] for i, _, _ in tasks for k, _ in els[i]},
                                {k[1] for _, _, j in tasks for k, _ in els[j]})
        if eps is None:
            return None
        return self._E.nprod_el_many(
            els, [(int(i), int(n), int(j)) for i, n, j in tasks], eps, nthreads)

    def nprod_many(self, els, tasks, nthreads):
        """
        ``els[i]_(n) els[j]`` for the tasks ``(i, n, j)``, raw elements, in
        one engine call, or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra as LVA
            sage: L = va.Lattice(QQ, matrix([[2]])); E = L._engine()
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: E is None or E.nprod_many([{a: 1}, {b: 1}], [(0, -1, 1), (1, 0, 0)], 1) == [
            ....:     LVA._no_key(L, a, b), LVA._nprod_key(L, b, 0, a)]
            True
        """
        lifted = self._lift_all(*els)
        eps = self._eps_or_none({k[1] for i, _, _ in tasks for k in els[i]},
                                {k[1] for _, _, j in tasks for k in els[j]})
        if lifted is None or eps is None:
            return None
        return [self._el_down(el) for el in self._E.nprod_el_many(
            lifted, [(int(i), int(n), int(j)) for i, n, j in tasks], eps, nthreads)]


class _LatticeMonomialKeys:
    """
    The set of basis keys ``(letters, q)``: a weakly descending tuple of
    Heisenberg letters ``(index, derivative)`` and a lattice vector ``q``.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _LatticeMonomialKeys
        sage: K = _LatticeMonomialKeys(2)
        sage: (((0, 1), (1, 0)), (1, -1)) in K
        True
    """
    def __init__(self, rank):
        """
        TESTS::

            sage: from vertex_algebras.lattice_vertex_algebra import _LatticeMonomialKeys
            sage: _LatticeMonomialKeys(3)._rank
            3
        """
        self._rank = rank

    def __contains__(self, x):
        """
        Whether ``x`` is a key: letters in weakly descending order, indices
        below the rank, an integer vector of the right length.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import _LatticeMonomialKeys
            sage: K = _LatticeMonomialKeys(2)
            sage: (((1, 0), (0, 1)), (0, 0)) in K, (((2, 0),), (0, 0)) in K, ((), (1,)) in K
            (False, False, False)
            sage: ((), (0, 0)) in K
            True
        """
        if not (isinstance(x, tuple) and len(x) == 2):
            return False
        letters, q = x
        if not (isinstance(letters, tuple) and isinstance(q, tuple)
                and len(q) == self._rank
                and all(_is_int_like(v) for v in q)):
            return False
        prev = None
        for t in letters:
            if not (isinstance(t, tuple) and len(t) == 2
                    and _is_int_like(t[0]) and _is_int_like(t[1])
                    and 0 <= t[0] < self._rank and t[1] >= 0):
                return False
            if prev is not None and _letter_key(prev) < _letter_key(t):
                return False
            prev = t
        return True

    def an_element(self):
        """
        A key: ``J_0``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import _LatticeMonomialKeys
            sage: _LatticeMonomialKeys(2).an_element()
            (((0, 0),), (0, 0))
        """
        return (((0, 0),), (0,) * self._rank)

    def cardinality(self):
        """
        ``+Infinity``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import _LatticeMonomialKeys
            sage: _LatticeMonomialKeys(1).cardinality()
            +Infinity
        """
        return Infinity


def _weight_labels(n, prefix):
    """The labels of :meth:`LatticeVertexAlgebra._weight_generators` (``J_i``,
    then ``e^{x_i}``, then ``e^{-x_i}``, as they print).

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _weight_labels
        sage: _weight_labels(2, 'J')
        ['J0', 'J1', 'e^(1,0)', 'e^(0,1)', 'e^(-1,0)', 'e^(0,-1)']
    """
    out = ['{}{}'.format(prefix, i) for i in range(n)]
    for sign in (1, -1):
        for i in range(n):
            out.append('e^({})'.format(','.join(str(sign if j == i else 0)
                                                for j in range(n))))
    return out


def _lattice_element_data(x, R, B, epsilon, names):
    """
    A conformal vector given to :class:`LatticeVertexAlgebra` in canonical
    form (a sorted tuple of ``(key, coefficient)``): ``x`` an element of a
    lattice vertex algebra with the same base ring, Gram matrix, cocycle
    and names (its products do not depend on conformal data), a dict
    ``{key: coefficient}`` or such a tuple.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: L = va.Lattice(QQ, matrix([[2]]))
        sage: from vertex_algebras.lattice_vertex_algebra import _lattice_element_data
        sage: J = L.J(0)
        sage: _lattice_element_data(J.no(J)/4, QQ, L.gram_matrix(), L._eps_spec, 'J')
        (((((0, 0), (0, 0)), (0,)), 1/4),)
        sage: _lattice_element_data({(((0, 0),), (1,)): 2}, QQ, L.gram_matrix(), 'standard', 'J')
        (((((0, 0),), (1,)), 2),)
        sage: _lattice_element_data({'x': 1}, QQ, L.gram_matrix(), 'standard', 'J')
        Traceback (most recent call last):
        ...
        ValueError: 'x' is not a basis key of the lattice vertex algebra
    """
    n = B.nrows()
    P = _parent(x)
    if getattr(P, '_is_lattice_vertex_algebra', False):
        if (P.base_ring() is not R or P._gram != B or P._eps_spec != epsilon
                or P._prefix != names):
            raise ValueError('the conformal vector must be an element of a '
                             'lattice vertex algebra with the same base ring, '
                             'Gram matrix, cocycle and names')
        d = x._monomial_coefficients
    elif isinstance(x, (dict, tuple)):
        d = dict(x)
        keys = _LatticeMonomialKeys(n)
        for k in d:
            if k not in keys:
                raise ValueError('{!r} is not a basis key of the lattice vertex '
                                 'algebra'.format(k))
    else:
        raise TypeError('the conformal vector must be an element of the lattice '
                        'vertex algebra (or a dict of its basis keys), not {!r}'
                        .format(x))
    out = []
    for (letters, q), c in d.items():
        c = R(c)
        if c:
            out.append(((tuple((int(i), int(k)) for i, k in letters),
                         tuple(int(v) for v in q)), c))
    return tuple(sorted(out))


def _canonical_lattice_grading(n, horizontal_grading, group):
    """The grading of :class:`LatticeVertexAlgebra` in canonical form:
    ``'lattice'`` (group ``None``) or the grades of ``e^{x_i}`` as elements of
    the grading group, and that group.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import _canonical_lattice_grading
        sage: _canonical_lattice_grading(2, 'lattice', None), _canonical_lattice_grading(2, None, None)
        (('lattice', None), (None, None))
        sage: _canonical_lattice_grading(2, [1, 1], Zmod(2))
        ((1, 1), Ring of integers modulo 2)
    """
    if horizontal_grading is None:
        if group is not None:
            raise ValueError('grading_group given without a grading')
        return None, None
    if isinstance(horizontal_grading, str):
        if horizontal_grading != 'lattice':
            raise ValueError("the grading must be 'lattice' or a list of the "
                             "grades of the basis vectors")
        if group is not None:
            raise ValueError("the grading 'lattice' has the lattice as its "
                             "group; give the grades of the basis vectors to "
                             "grade by another group")
        return 'lattice', None
    from ._grading_groups import infer_grading_group
    vals = list(horizontal_grading)
    if len(vals) != n:
        raise ValueError('{} grades expected (one per basis vector)'.format(n))
    A = infer_grading_group(vals, group)
    return tuple(A.element(A.coords(v)) for v in vals), A.group


class LatticeVertexAlgebra(VertexAlgebra):
    r"""
    The lattice vertex (super)algebra of the lattice ``ZZ^n`` with Gram
    matrix ``B``.

    INPUT:

    - ``base_ring`` -- (default: ``None``) a ring containing ``QQ``
      (``None``: ``QQ``)
    - ``B`` -- a symmetric, integral, non-degenerate square matrix, or a
      Sage ``IntegralLattice`` (its Gram matrix is used)
    - ``epsilon`` -- (default: ``None``) the 2-cocycle ``ε``; one of

      * ``None`` or ``'standard'`` (default): the bimultiplicative cocycle
        with ``ε(x_i, x_j) = (-1)^{B_ij + B_ii B_jj}`` for ``i > j`` and
        ``1`` otherwise (always satisfies the locality condition);
      * ``'trivial'``: ``ε = 1``;
      * a square matrix ``E`` of units: ``ε(q,p) = ∏ E_ij^{q_i p_j}``;
      * a callable ``(q, p) -> ring element`` on integer tuples (the user is
        responsible for the cocycle identity and ``ε(0,p) = ε(q,0) = 1``).

      A matrix (including the trivial one) that violates
      ``ε(q,p)/ε(p,q) = (-1)^{B(q,p) + B(q,q)B(p,p)}`` triggers a
      ``RuntimeWarning``: the resulting algebra is not local.
    - ``names`` -- (default: ``'J'``) prefix of the Heisenberg current names
    - ``conformal_vector`` -- (default: ``None``) a Virasoro vector (``None``:
      none): an element of the lattice vertex algebra with the same
      ``base_ring``, ``B``, ``epsilon`` and ``names`` and no conformal data
      (its products do not depend on them), checked as by
      :meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector`
    - ``degrees`` -- (default: ``None``) weights without a Virasoro vector: a
      dict over the labels ``J_i``, ``e^(x_i)``, ``e^(-x_i)`` (as they print),
      see ``_weight_data_from()``
    - ``horizontal_grading`` -- (default: ``None``) ``None``, ``'lattice'``
      (``e^q`` has grade ``q``) or the list of the grades of ``e^{x_i}``
      (see ``_grading_data()``)
    - ``horizontal_grading_group`` -- (default: ``None``) the grading group
      of a list of grades (``None``: inferred); not with ``'lattice'``
    - ``implementation`` -- (default: ``None``) ``None`` (the C++ engine when
      it takes the base ring, else pure Python), ``'engine'`` or ``'python'``;
      a ``ValueError`` if the engine is asked for and not available for the
      base ring.  Results do not depend on it.

    These conformal data are part of the algebra: other choices give other
    algebras (see :meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector`),
    whose elements coerce into the algebra without them.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: A2 = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
        sage: A2.cocycle((1, 0), (0, 1)), A2.cocycle((0, 1), (1, 0))
        (1, -1)
        sage: a = A2.e((1, 0)); b = A2.e((0, 1))
        sage: a.ope(b)
        [e^(1,1)]
        sage: b.ope(a)
        [-e^(1,1)]
        sage: A2.e((1, 1)).ope(A2.e((-1, -1)))
        [-J0 - J1, -1]

    The trivial cocycle is rejected (with a warning) when it is not local::

        sage: import warnings
        sage: with warnings.catch_warnings(record=True) as w:
        ....:     warnings.simplefilter('always')
        ....:     T = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]), epsilon='trivial')
        sage: 'not local' in str(w[0].message)
        True

    Indeed, it breaks skew-symmetry, while the standard cocycle satisfies it::

        sage: a, b = T.e((1, 0)), T.e((0, 1))
        sage: a.ope(b), b.ope(a)
        ([e^(1,1)], [e^(1,1)])
        sage: T._test_skew_symmetry(elements=[a, b])
        Traceback (most recent call last):
        ...
        AssertionError: ...
        sage: A2._test_skew_symmetry(elements=[A2.e((1, 0)), A2.e((0, 1))])

    An indefinite lattice ``diag(2, -2)``::

        sage: D = LatticeVertexAlgebra(QQ, matrix([[2, 0], [0, -2]]))
        sage: D.e((0, 1)).ope(D.e((0, 1)))
        [:J1 e^(0,2):, e^(0,2)]
        sage: D.e((1, 1)).ope(D.e((-1, -1)))
        []

    TESTS::

        sage: LatticeVertexAlgebra(QQ, matrix([[1, 2], [0, 1]]))
        Traceback (most recent call last):
        ...
        ValueError: the Gram matrix must be symmetric
        sage: LatticeVertexAlgebra(QQ, matrix([[1, 1], [1, 1]]))
        Traceback (most recent call last):
        ...
        ValueError: the Gram matrix must be non-degenerate
        sage: LatticeVertexAlgebra(ZZ, matrix([[2]]))
        Traceback (most recent call last):
        ...
        ValueError: the base ring must contain QQ
        sage: L = LatticeVertexAlgebra(QQ, matrix([[2]]))
        sage: L is LatticeVertexAlgebra(QQ, matrix([[2]]))
        True
        sage: L(3) == 3 * L.one()
        True
        sage: L(((), (1,))) == L.e((1,))
        True
        sage: L.an_element()
        J0
        sage: P = LatticeVertexAlgebra(QQ, matrix([[2]]), implementation='python')
        sage: P is L, P._engine() is None, P.e((1,)).ope(P.e((-1,)))
        (False, True, [J0, 1])

    The vertex algebra axioms (skew-symmetry, Borcherds' commutator
    formula) for even, odd, definite and indefinite lattices::

        sage: for B in [[[2, -1], [-1, 2]], [[0, 1], [1, 0]], [[1]], [[2, 0], [0, -2]],
        ....:           [[1, 2], [2, -3]]]:
        ....:     TestSuite(LatticeVertexAlgebra(QQ, matrix(B))).run()

    A background charge on an odd indefinite lattice: ``T = Tsug + ∂J_α``
    has central charge ``rank - 12 B(α, α)`` and gives ``e^q`` the weight
    ``B(q, q)/2 - B(α, q)``::

        sage: L = LatticeVertexAlgebra(QQ, matrix([[1, 2], [2, -3]]))
        sage: J = L.heisenberg_currents(); G = L.gram_matrix(); Gi = G.inverse()
        sage: al = vector([1, -2])
        sage: T = (sum(Gi[i, j]/2 * J[i].no(J[j]) for i in range(2) for j in range(2))
        ....:      + (al[0]*J[0] + al[1]*J[1]).derivative())
        sage: T.ope(T) == [T.derivative(), 2*T, 0, L((2 - 12*al*G*al)/2)]
        True
        sage: all(T.nproduct(L.e(q), 1) == (vector(q)*G*vector(q)/2 - al*G*vector(q))*L.e(q)
        ....:     for q in [(1, 0), (0, 1), (1, -1), (2, 1), (-1, -2)])
        True

    The charge-0 sector is the Heisenberg vertex algebra: the universal
    vertex algebra with ``J_i(z) J_j(w) ~ B_ij/(z-w)^2`` maps into it by
    ``m -> (m, 0)``, compatibly with all ``n``-th products::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J0', 'J1'], {('J0', 'J0'): [0, 1],
        ....:     ('J0', 'J1'): [0, 2], ('J1', 'J1'): [0, -3]})
        sage: phi = lambda x: L._from_dict({(m, (0, 0)): c
        ....:                               for m, c in x.monomial_coefficients().items()})
        sage: X = H.gens()
        sage: a, b = X[0].derivative().no(X[1]), X[1].no(X[1]).no(X[0])
        sage: pa, pb = J[0].derivative().no(J[1]), J[1].no(J[1]).no(J[0])
        sage: phi(a) == pa, phi(b) == pb
        (True, True)
        sage: all(phi(a.nproduct(b, n)) == pa.nproduct(pb, n) for n in range(-3, 5))
        True
    """

    _is_lattice_vertex_algebra = True       # duck typing (doctests)

    @staticmethod
    def __classcall_private__(cls, base_ring=None, B=None, epsilon=None,
                              names='J', conformal_vector=None,
                              degrees=None, horizontal_grading=None,
                              horizontal_grading_group=None, implementation=None):
        r"""
        Normalize the arguments so that instance caching works (the
        normalized arguments are also the pickle).

        EXAMPLES::

            sage: from vertex_algebras import lattice_vertex_algebra as lva
            sage: L = lva.LatticeVertexAlgebra(QQ, matrix([[2]]))
            sage: J = L.J(0)
            sage: LT = lva.LatticeVertexAlgebra(QQ, matrix([[2]]),
            ....:     conformal_vector=J.no(J)/4, horizontal_grading='lattice')
            sage: LT is L.with_conformal_vector(J.no(J)/4).with_horizontal_grading('lattice')
            True
            sage: loads(dumps(LT)) is LT, LT.central_charge(), LT.e((1,)).horizontal_grade()
            (True, 1, 1)
            sage: LT.category()
            Category of vertex operator algebras with basis over Rational Field
            sage: TestSuite(LT).run()
            sage: TestSuite(lva.LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, -1]]))).run()
            sage: LT.e((1,)) + L.e((1,))          # forgetful coercion
            2*e^(1)

        A Sage integral lattice in place of its Gram matrix::

            sage: lva.LatticeVertexAlgebra(QQ, IntegralLattice('A2')) is lva.LatticeVertexAlgebra(
            ....:     QQ, matrix([[2, -1], [-1, 2]]))
            True
        """
        if B is None:
            raise TypeError('LatticeVertexAlgebra(base_ring, B): a Gram '
                            'matrix must be given')
        if base_ring is None:
            base_ring = QQ
        if not base_ring.has_coerce_map_from(QQ):
            raise ValueError('the base ring must contain QQ')
        if hasattr(B, 'gram_matrix') and not hasattr(B, 'nrows'):
            B = B.gram_matrix()          # a Sage IntegralLattice (or similar)
        try:
            B = matrix(ZZ, B)
        except (TypeError, ValueError):
            raise ValueError('the Gram matrix must be integral')
        if not B.is_square():
            raise ValueError('the Gram matrix must be square')
        if B != B.transpose():
            raise ValueError('the Gram matrix must be symmetric')
        if B.nrows() == 0 or B.det() == 0:
            raise ValueError('the Gram matrix must be non-degenerate')
        B.set_immutable()
        if epsilon is None:
            epsilon = 'standard'
        if isinstance(epsilon, str):
            if epsilon not in ('standard', 'trivial'):
                raise ValueError("epsilon must be 'standard', 'trivial', a "
                                 "matrix or a callable")
        elif not callable(epsilon):
            E = matrix(base_ring, epsilon)
            if E.dimensions() != B.dimensions():
                raise ValueError('the cocycle matrix must have the shape of B')
            E.set_immutable()
            epsilon = E
        names = str(names)
        n = B.nrows()
        T = None
        if conformal_vector is not None:
            T = _lattice_element_data(conformal_vector, base_ring, B, epsilon,
                                      names)
        weights = None
        if degrees is not None:
            labels = _weight_labels(n, names)
            w = dict(degrees)
            if set(w) != set(labels):
                raise ValueError('conformal weights must be given for exactly '
                                 '{} (got {})'.format(labels, sorted(w, key=str)))
            weights = tuple((s, base_ring(w[s])) for s in labels)
        horizontal_grading, group = _canonical_lattice_grading(n, horizontal_grading, horizontal_grading_group)
        return VertexAlgebra.__classcall__(
            cls, base_ring, B, epsilon, names=names, conformal_vector=T,
            degrees=weights, horizontal_grading=horizontal_grading, horizontal_grading_group=group,
            implementation=_normalize_implementation(implementation))

    # (∂^n a)_(-1) b = n! a_(-n-1) b: a's terms have far fewer creation series
    # than those of ∂^n a (triplet spans, weight 29: 2-10x on words of W's)
    _deriv_no_by_nproduct = True

    def __init__(self, base_ring, B, epsilon, names='J',
                 conformal_vector=None, degrees=None, horizontal_grading=None,
                 horizontal_grading_group=None, implementation=None):
        r"""
        Initialize ``self``.

        TESTS::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]), names='h')
            sage: L._rank, L._B, L._names, L._vac
            (2, [[2, -1], [-1, 2]], ['h0', 'h1'], ((), (0, 0)))
        """
        n = B.nrows()
        self._implementation = implementation
        self._rank = n
        self._gram = B
        self._B = [[int(B[i, j]) for j in range(n)] for i in range(n)]
        self._prefix = names
        self._names = ['{}{}'.format(names, i) for i in range(n)]
        self._idx = {s: i for i, s in enumerate(self._names)}
        self._zero_q = (0,) * n
        self._vac = ((), self._zero_q)
        VertexAlgebra.__init__(
            self, base_ring, _LatticeMonomialKeys(n),
            category=self._category_for(
                base_ring, any(B[i, i] % 2 for i in range(n)),
                degrees is not None or conformal_vector is not None,
                conformal_vector is not None))
        self._schur_one = self._one_R
        self._eps_spec = epsilon
        self._setup_cocycle(epsilon)
        self._data = {'T': conformal_vector,
                      'weights': degrees, 'horizontal_grading': horizontal_grading,
                      'group': horizontal_grading_group, 'implementation': implementation}
        self._check_implementation()
        if degrees is not None:
            self._install_conformal_weights(dict(degrees))
        if horizontal_grading is not None:
            self._install_grading(horizontal_grading if isinstance(horizontal_grading, str)
                                  else list(horizontal_grading), horizontal_grading_group)
        if conformal_vector is not None:
            self._install_virasoro(self._from_dict(
                dict(conformal_vector), coerce=False, remove_zeros=True))

    def _has_odd(self):
        """
        Whether some ``e^q`` is odd, i.e. some ``B_ii`` is odd.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2]]))._has_odd(), va.Lattice(QQ, matrix([[1]]))._has_odd()
            (False, True)
        """
        return any(self._B[i][i] % 2 for i in range(self._rank))

    def _some_extra_elements(self):
        """
        ``e^{±x_0}``, added to the elements ``TestSuite`` tries.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._some_extra_elements()
            [e^(1,0), e^(-1,0)]
        """
        x = tuple(1 if i == 0 else 0 for i in range(self._rank))
        return [self.e(x), self.e(tuple(-v for v in x))]

    # conformal data as construction data (see VertexAlgebra._forgets)

    def _structure_key(self):
        """
        The construction data without the conformal ones: ring, Gram
        matrix, cocycle, names, implementation.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: J = L.J(0)
            sage: L._structure_key() == L.with_conformal_vector(J.no(J)/4)._structure_key()
            True
            sage: L._structure_key()[3:]
            ('standard', 'J', None)
        """
        return ('LatticeVertexAlgebra', self.base_ring(), self._gram,
                self._eps_spec, self._prefix, self._data['implementation'])

    def _conformal_key(self):
        """
        The conformal data: conformal vector, weights, grading.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._conformal_key()
            {'T': None, 'horizontal_grading': None, 'weights': None}
            sage: L.with_horizontal_grading('lattice')._conformal_key()['horizontal_grading']
            ('lattice', None)
        """
        d = self._data
        return {'T': d['T'], 'weights': d['weights'],
                'horizontal_grading': None if d['horizontal_grading'] is None
                else (d['horizontal_grading'], d['group'])}

    def _with(self, **changes):
        """
        The lattice vertex algebra with the same data except ``changes``
        (keywords of the constructor); the base of the ``with_*`` methods.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: J = L.J(0)
            sage: L._with(conformal_vector=J.no(J)/4).central_charge()
            1
        """
        d = self._data
        args = dict(conformal_vector=d['T'], degrees=d['weights'],
                    horizontal_grading=d['horizontal_grading'], horizontal_grading_group=d['group'])
        args.update(changes)
        if args['horizontal_grading'] is None or isinstance(args['horizontal_grading'], str):
            args['horizontal_grading_group'] = changes.get('horizontal_grading_group')
        return LatticeVertexAlgebra(self.base_ring(), self._gram, self._eps_spec,
                                    names=self._prefix,
                                    implementation=d['implementation'], **args)

    # ------------------------------------------------------------------ #
    #  Lattice data                                                       #
    # ------------------------------------------------------------------ #

    def _lift_ring(self):
        """
        The ring the C++ engine computes over for a fraction field: the
        polynomial ring ``A`` if the base ring is ``Frac(A)`` with ``A ⊇
        QQ``; ``ZZ`` for ``QQ``; else ``None``.  Fraction-field arithmetic (a
        gcd per operation) is ~25x slower than arithmetic in ``A``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: B = matrix([[2]])
            sage: va.Lattice(QQ, B)._lift_ring()
            Integer Ring
            sage: va.Lattice(Frac(QQ['c']), B)._lift_ring()
            Univariate Polynomial Ring in c over Rational Field
            sage: va.Lattice(QuadraticField(5), B)._lift_ring() is None
            True
        """
        R = self.base_ring()
        if R is QQ:
            return ZZ
        try:
            A = R.ring()
        except AttributeError:
            return None
        try:
            if (A is R or R is not A.fraction_field()
                    or not A.has_coerce_map_from(QQ)):
                return None
            x = R(A.gen()) / 3 + 1
            if R(x.numerator()) / R(x.denominator()) != x:
                return None
        except Exception:
            return None
        return A

    def _setup_cocycle(self, epsilon):
        """
        Install the cocycle ``epsilon`` (see the class docstring): a callable
        in ``_eps_fun``, else its matrix ``_eps_mat`` (``E_ij``, with
        ``ε(q,p) = Π E_ij^{q_i p_j}``) and ``_eps_sign`` (0 for ``1``, 1 for
        ``-1``, ``None`` for other units); warns if the matrix violates the
        locality condition.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._eps_mat, L._eps_sign
            ([[1, 1], [-1, 1]], [[0, 0], [1, 0]])
            sage: L2 = va.Lattice(QQ, matrix([[2, -1], [-1, 4]]), epsilon='trivial')
            doctest:warning...
            RuntimeWarning: LatticeVertexAlgebra: the cocycle violates ...
        """
        n = self._rank
        R = self.base_ring()
        Bm = self._B
        self._eps_fun = None
        self._eps_values = {}       # values of a callable cocycle, converted
        if isinstance(epsilon, str):
            if epsilon == 'trivial':
                E = [[R.one()] * n for _ in range(n)]
            else:
                E = [[R((-1) ** (Bm[i][j] + Bm[i][i] * Bm[j][j]))
                      if i > j else R.one() for j in range(n)]
                     for i in range(n)]
        elif callable(epsilon):
            self._eps_fun = epsilon
            self._eps_mat = None
            return
        else:
            E = [[R(epsilon[i, j]) for j in range(n)] for i in range(n)]
        self._eps_mat = E
        # 0 for 1, 1 for -1, None otherwise: signs are computed by negation
        # in _eps (powers of R(-1) convert scalars into R, which is slow over
        # a Singular polynomial ring over a number field before Sage 10.10:
        # vertex_algebra._SingularNumberFieldElements)
        one = R.one()
        self._eps_sign = [[0 if e == one else 1 if e == -one else None for e in row]
                          for row in E]
        for i in range(n):
            for j in range(n):
                want = R((-1) ** (Bm[i][j] + Bm[i][i] * Bm[j][j]))
                if E[i][j] != want * E[j][i]:
                    warnings.warn(
                        'LatticeVertexAlgebra: the cocycle violates '
                        'ε(x_%d,x_%d)/ε(x_%d,x_%d) = (-1)^(B_ij + B_ii B_jj); '
                        'the algebra is not local' % (i, j, j, i),
                        RuntimeWarning, stacklevel=4)
                    return

    def cocycle(self, q, p):
        r"""
        The value ``ε(q, p)`` of the lattice cocycle.

        INPUT:

        - ``q``, ``p`` -- lattice vectors (integer coordinates in the basis
          of the lattice)

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]]))
            sage: L.cocycle((0, 1), (1, 0)), L.cocycle((1, 0), (0, 1))
            (-1, 1)
            sage: E = matrix([[1, 1], [-1, 1]])
            sage: LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, 1]]), epsilon=E) is L
            False
        """
        return self._eps(tuple(int(v) for v in q), tuple(int(v) for v in p))

    def _eps(self, q, p):
        """
        The cocycle value ``ε(q, p)`` for integer tuples (cached; callable
        cocycles converted into the base ring once per value).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._eps((1, 0), (0, 1)), L._eps((0, 1), (1, 0)), L._eps((0, 3), (1, 0))
            (1, -1, -1)
            sage: L._eps((1, 1), (2, 1)) == L.cocycle((1, 1), (2, 1))
            True
        """
        key = (q, p)
        c = self._eps_cache.get(key)
        if c is not None:
            return c
        if self._eps_fun is not None:
            v = self._eps_fun(q, p)
            try:                            # one conversion per value
                c = self._eps_values.get(v)
                if c is None:
                    c = self._eps_values[v] = self.base_ring()(v)
            except TypeError:               # not hashable
                c = self.base_ring()(v)
        else:
            c = self._one_R
            E, S = self._eps_mat, self._eps_sign
            for i, qi in enumerate(q):
                if not qi:
                    continue
                for j, pj in enumerate(p):
                    if not pj or S[i][j] == 0:
                        continue
                    if S[i][j] == 1:
                        if (qi * pj) % 2:
                            c = -c
                    else:
                        c = c * E[i][j] ** (qi * pj)
        self._eps_cache[key] = c
        return c

    def _form(self, q, p):
        """``B(q, p)`` for integer tuples.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._form((1, 0), (1, 1)), L._form((1, 1), (1, 1))
            (1, 2)
        """
        Bm = self._B
        return sum(qi * Bm[i][j] * p[j]
                   for i, qi in enumerate(q) if qi
                   for j in range(self._rank))

    def _Bxq(self, i, q):
        """``B(x_i, q)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._Bxq(0, (1, 1)), L._Bxq(1, (2, 0))
            (1, -2)
        """
        row = self._B[i]
        return sum(row[j] * qj for j, qj in enumerate(q) if qj)

    def _parity_q(self, q):
        """
        The parity ``B(q, q) mod 2`` of ``e^q``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2]]))._parity_q((1,)), va.Lattice(QQ, matrix([[1]]))._parity_q((1,))
            (0, 1)
        """
        return self._form(q, q) % 2

    def gram_matrix(self):
        """
        The Gram matrix ``B`` of the lattice form.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2, 1], [1, 2]])).gram_matrix()
            [2 1]
            [1 2]
        """
        return self._gram

    def rank(self):
        """
        The rank ``n`` of the lattice.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2, 1], [1, 2]])).rank()
            2
        """
        return self._rank

    # ------------------------------------------------------------------ #
    #  Basics                                                             #
    # ------------------------------------------------------------------ #

    def _repr_(self):
        """
        The repr: rank, Gram matrix and base ring.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: repr(L)
            'Lattice vertex algebra of rank 2 with Gram matrix [[2, -1], [-1, 2]] over Rational Field'
        """
        B = self._gram
        if B.nrows() == 1:
            s = '[{}]'.format(B[0, 0])
        else:
            s = '[' + ', '.join('[' + ', '.join(str(x) for x in row) + ']'
                                for row in B.rows()) + ']'
        return ('Lattice vertex algebra of rank {} with Gram matrix {} over {}'
                .format(self._rank, s, self.base_ring()))

    def _singular_generators(self):
        """The ``J_i`` and ``e^{±x_i}`` (they generate, not strongly).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._singular_generators()
            (J0, J1, e^(1,0), e^(0,1), e^(-1,0), e^(0,-1))
        """
        return self.gens()

    def _term_str(self, mono, style='ascii'):
        r"""
        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, 1], [1, 2]]))
            sage: L._repr_term((((1, 1), (0, 0)), (1, -1)))
            ':dJ1 J0 e^(1,-1):'
            sage: L._repr_term((((0, 2),), (0, 0))), L._repr_term(((), (0, 0)))
            ('d^2J0', '1')
            sage: L._term_str((((1, 1), (0, 0)), (1, -1)), 'unicode')
            ':∂ J1 J0 e^(1,-1):'
            sage: L._term_str((((1, 1), (0, 0)), (1, -1)), 'latex')
            ':\\partial J1\\, J0\\, e^{(1,-1)}:'
        """
        letters, q = mono
        parts = [_derivative_str(self._names[i], n, style) for (i, n) in letters]
        if any(q):
            parts.append(('e^{{({})}}' if style == 'latex' else 'e^({})').format(
                ','.join(str(v) for v in q)))
        return _word_str(parts, style)

    def _mode_word(self, m, data):
        r"""
        The key ``(letters, q)`` as Heisenberg modes on the Fock vacuum
        `e^q` (see :meth:`VertexAlgebra._mode_word`): the letter
        `\partial^n J_i` is `n!\, J_i(-n-1)`; the currents have conformal
        weight 1.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, 1], [1, 2]]))
            sage: L._mode_word((((1, 2), (0, 0)), (1, -1)), None)
            (2, [('J1', -3, None), ('J0', -1, None)], (1, -1))
        """
        letters, q = m
        f = 1
        out = []
        for i, n in letters:
            f *= _fact(n)
            out.append((self._names[i], -n - 1, None if data is None else 1))
        return f, out, tuple(q)

    def _vacuum_key(self):
        """
        The vacuum key ``((), 0)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L._vacuum_key(), L.monomial(L._vacuum_key()) == L.one()
            (((), (0, 0)), True)
        """
        return self._vac

    def _parity_key(self, m):
        """
        The parity of a key: that of its lattice part.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[1, 0], [0, 2]]))
            sage: L._parity_key((((0, 0),), (1, 0))), L._parity_key(((), (0, 1)))
            (1, 0)
        """
        return self._parity_q(m[1])

    def _state_fingerprint(self):
        """
        The data saved states are checked against: ring, Gram matrix,
        cocycle (a callable by its name only), names.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: f = L._state_fingerprint(); f[0], f[3], f[4]
            ('LatticeVertexAlgebra', [[1, 1], [-1, 1]], 'J')
        """
        eps = self._eps_mat
        if eps is None:              # a callable cocycle: only its name
            f = self._eps_fun
            eps = ('callable', getattr(f, '__module__', None),
                   getattr(f, '__qualname__', repr(f)))
        return ('LatticeVertexAlgebra', self.base_ring(), self._gram, eps,
                self._prefix)

    def variable_names(self):
        """
        The names of :meth:`~vertex_algebra.VertexAlgebra.gens`: the Heisenberg currents ``J_i`` and the
        lattice currents ``e^{x_i}``, ``e^{-x_i}`` of the basis vectors,
        which generate the lattice vertex algebra (not strongly).

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, 1], [1, 2]]))
            sage: L.variable_names()
            ('J0', 'J1', 'e^(1,0)', 'e^(0,1)', 'e^(-1,0)', 'e^(0,-1)')
            sage: L.gens()
            (J0, J1, e^(1,0), e^(0,1), e^(-1,0), e^(0,-1))
            sage: L.ngens(), L.heisenberg_currents()
            (6, (J0, J1))
        """
        return tuple(_weight_labels(self._rank, self._prefix))

    def gen(self, name_or_index):
        r"""
        The generator with index ``i`` or the given name (see
        :meth:`variable_names`), or the lattice current ``e^q`` for a lattice
        vector ``q``.

        INPUT:

        - ``name_or_index`` -- an integer index, a name in
          :meth:`variable_names` (or ``J`` names), or a lattice vector
          (tuple, list or vector)

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, 1], [1, 2]]))
            sage: L.gen(1), L.gen('J0'), L.gen(2), L.gen('e^(0,-1)'), L.gen((1, -1))
            (J1, J0, e^(1,0), e^(0,-1), e^(1,-1))
            sage: L.gen(vector([0, 0])) == L.one()
            True
        """
        if _is_int_like(name_or_index):
            i = int(name_or_index)
            if not 0 <= i < 3 * self._rank:
                raise IndexError('no generator with index {}'.format(i))
            return self.monomial(self._weight_generators()[i][1])
        if isinstance(name_or_index, str):
            if name_or_index in self._idx:
                return self.J(self._idx[name_or_index])
            labels = self.variable_names()
            if name_or_index in labels:
                return self.gen(labels.index(name_or_index))
            raise ValueError('no generator named {!r}'.format(name_or_index))
        return self.e(name_or_index)

    def simple_module(self, lam):
        r"""
        The simple module ``V_{λ+L}`` for ``λ`` in the dual lattice (its
        coordinates in the basis of ``L``; ``B λ`` integral), a
        :class:`lattice_vertex_algebra_module.LatticeVertexAlgebraModule`.

        INPUT:

        - ``lam`` -- the coordinates of ``λ`` in the basis of ``L`` (a
          list, tuple or vector of rationals with ``B λ`` integral); ``λ``
          and ``λ + L`` give the same module

        OUTPUT:

        A :class:`lattice_vertex_algebra_module.LatticeVertexAlgebraModule`.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2]]))
            sage: M = L.simple_module([1/2]); M
            Module V_(λ+L), λ = (1/2), over Lattice vertex algebra of rank 1 with Gram matrix [2] over Rational Field
            sage: L.e((1,)).ope(M.highest_weight_vector())
            []
        """
        from .lattice_vertex_algebra_module import LatticeVertexAlgebraModule
        return LatticeVertexAlgebraModule(self, lam)

    def simple_modules(self):
        r"""
        The simple modules ``V_{λ+L}``, one for each coset of the dual
        lattice ``L°`` modulo ``L`` (``V_L`` itself first).

        OUTPUT:

        A list of
        :class:`lattice_vertex_algebra_module.LatticeVertexAlgebraModule`.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: [M.coset_representative() for M in LatticeVertexAlgebra(QQ, matrix([[4]])).simple_modules()]
            [(0,), (1/4,), (1/2,), (3/4,)]
        """
        from .lattice_vertex_algebra_module import _dual_lattice_cosets
        return [self.simple_module(lam) for lam in _dual_lattice_cosets(self)]

    def J(self, i):
        """
        The Heisenberg current ``J_i`` in the direction ``x_i``.

        INPUT:

        - ``i`` -- an index ``0 <= i < n``

        EXAMPLES:

        ``J_i(z) J_j(w) ~ B_ij/(z-w)^2`` and ``J_i(z) e^q(w) ~ B(x_i,
        q) e^q(w)/(z-w)``::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, 1], [1, 2]]))
            sage: L.J(0).ope(L.J(1)), L.J(1).ope(L.e((1, 0)))
            ([0, 1], [e^(1,0)])

        TESTS::

            sage: L.J(2)
            Traceback (most recent call last):
            ...
            IndexError: no Heisenberg current with index 2
        """
        i = int(i)
        if not 0 <= i < self._rank:
            raise IndexError('no Heisenberg current with index {}'.format(i))
        return self.monomial((((i, 0),), self._zero_q))

    def heisenberg_currents(self):
        """
        The Heisenberg currents ``(J_0, ..., J_{n-1})``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2, 1], [1, 2]])).heisenberg_currents()
            (J0, J1)
        """
        return tuple(self.J(i) for i in range(self._rank))

    def e(self, q):
        """
        The lattice current ``e^q`` (``e^0`` is the vacuum).

        INPUT:

        - ``q`` -- a lattice vector (integer coordinates: tuple, list or vector)

        EXAMPLES:

        ``e^q`` has parity ``B(q, q) mod 2``; ``e^α(z) e^{-α}(w) ~
        1/(z-w)^2 + J_α(w)/(z-w)`` for ``B(α, α) = 2``::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, 1], [1, 2]]))
            sage: L.e((1, 0)).ope(L.e((-1, 0)))
            [J0, 1]
            sage: L.e((0, 0)) == L.one(), va.Lattice(QQ, matrix([[1]])).e([3]).parity()
            (True, 1)

        TESTS::

            sage: L.e((1,))
            Traceback (most recent call last):
            ...
            ValueError: lattice vector of length 2 expected
        """
        q = tuple(int(v) for v in q)
        if len(q) != self._rank:
            raise ValueError('lattice vector of length {} expected'
                             .format(self._rank))
        return self.monomial(((), q))

    def _element_constructor_(self, x):
        """
        Basis keys (with any integer types) as monomials; anything else as
        for :class:`vertex_algebra.VertexAlgebra`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: L(((((0, 0),), (1, 0)))), L(3)
            (:J0 e^(1,0):, 3)
        """
        if isinstance(x, tuple) and x in self._indices:
            letters, q = x
            return self.monomial((tuple((int(i), int(k)) for i, k in letters),
                                  tuple(int(v) for v in q)))
        return VertexAlgebra._element_constructor_(self, x)

    def clear_caches(self):
        """
        Forget all cached intermediate computations (results do not
        change: they are recomputed on demand).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: x = L.e((1,)).ope(L.e((-1,))); L.clear_caches()
            sage: L.e((1,)).ope(L.e((-1,))) == x
            True
        """
        VertexAlgebra.clear_caches(self)
        self._ls_cache = {}
        self._grp_payload = None
        self._rec_cache = {}
        self._eplus_cache = {}
        self._cser_cache = {}
        self._schur_cache = {}
        self._eps_cache = {}

    # ------------------------------------------------------------------ #
    #  Raw-dict kernels.  A dict maps keys ``(letters, q)`` to base ring  #
    #  coefficients; all caches hold raw dicts / tuples of dicts.         #
    # ------------------------------------------------------------------ #

    def _add(self, dst, key, c):
        """
        ``dst[key] += c`` in a raw dict, dropping zeros.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: d = {}; L._add(d, 'k', 1); d
            {'k': 1}
            sage: L._add(d, 'k', -1); d
            {}
        """
        v = dst.get(key)
        v = c if v is None else v + c
        if v:
            dst[key] = v
        else:
            dst.pop(key, None)

    def _addto(self, dst, src, c):
        """``dst += c * src``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: d = {'a': 1}; L._addto(d, {'a': -1, 'b': 2}, 1); d
            {'b': 2}
        """
        if not c:
            return
        for k, v in src.items():
            self._add(dst, k, c * v)

    def _mul_letter(self, x, d):
        """Multiply every monomial of ``d`` by the letter ``x``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._mul_letter((0, 1), {(((0, 0),), (1,)): 2})
            {(((0, 1), (0, 0)), (1,)): 2}
        """
        return {(_insert_letter(k[0], x), k[1]): v for k, v in d.items()}

    def _h_act_mono(self, i, m, key):
        """``h_i(m)`` on a single monomial, as a dict.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._h_act_mono(0, -2, ((), (1,)))       # h(-2) = dJ (letter (0, 1) = 1! h(-2))
            {(((0, 1),), (1,)): 1}
            sage: L._h_act_mono(0, 0, ((), (1,))), L._h_act_mono(0, 1, (((0, 0),), (1,)))
            ({((), (1,)): 2}, {((), (1,)): 2})
        """
        letters, q = key
        R = self.base_ring()
        if m < 0:
            n = -m - 1
            return {(_insert_letter(letters, (i, n)), q):
                    self._one_R / _fact(n)}
        if m == 0:
            c = self._Bxq(i, q)
            return {key: R(c)} if c else {}
        out = {}
        row = self._B[i]
        f = _fact(m)
        for t, (j, n) in enumerate(letters):
            if n == m - 1 and row[j]:
                self._add(out, (letters[:t] + letters[t + 1:], q),
                          R(f * row[j]))
        return out

    def _schur(self, q, k):
        """``S_k(q)`` as a dict ``{letters: coeff}``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: sorted(L._schur((1,), 2).items())        # (h(-1)^2 + h(-2))/2
            [(((0, 0), (0, 0)), 1/2), (((0, 1),), 1/2)]
            sage: L._schur((1,), 0), L._schur((1,), -1)
            ({(): 1}, {})
        """
        key = (q, k)
        r = self._schur_cache.get(key)
        if r is not None:
            return r
        if k < 0:
            r = {}
        elif k == 0:
            r = {(): self._one_R}
        else:
            r = {}
            for m in range(1, k + 1):
                prev = self._schur(q, k - m)
                if not prev:
                    continue
                c0 = self._one_R / (k * _fact(m - 1))
                for i, qi in enumerate(q):
                    if not qi:
                        continue
                    c = c0 * qi
                    for L, v in prev.items():
                        self._add(r, _insert_letter(L, (i, m - 1)), c * v)
        self._schur_cache[key] = r
        return r

    def _nmax(self, a, b):
        """Largest ``n`` with ``a_(n) b`` possibly nonzero.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._nmax(((), (1,)), ((), (-1,))), L._nmax(((), (1,)), ((), (1,)))
            (1, -3)
        """
        return (_degree(a[0]) + _degree(b[0]) - 1
                - self._form(a[1], b[1]))

    def _nprod(self, a, n, b):
        r"""
        The ``n``-th product ``a_(n) b`` of two basis monomials, as a dict
        (the base class's cached :meth:`_nprod_mono`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._from_dict(L._nprod(((), (1,)), -1, ((), (-1,))))
            1/2*:J0 J0: + 1/2*dJ0
        """
        return self._nprod_mono(a, n, b)

    def _eplus(self, q, b):
        r"""
        ``E^+(q,z) b = exp(-Σ_{m>0} q(m) z^{-m}/m) b`` for a monomial ``b``,
        as ``{power of z: {letters: coeff}}`` (powers ``<= 0``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: sorted(L._eplus((1,), (((0, 0),), (0,))).items())    # E^+ J0 = J0 - 2 z^-1
            [(-1, {(): -2}), (0, {((0, 0),): 1})]
        """
        key = (q, b)
        r = self._eplus_cache.get(key)
        if r is not None:
            return r
        R = self.base_ring()
        letters = b[0]
        cur = {0: {letters: self._one_R}}
        r = {0: {letters: self._one_R}}
        step = 0
        while cur:
            step += 1
            new = {}
            for pw, d in cur.items():
                for L, c in d.items():
                    for t, (j, nj) in enumerate(L):
                        bq = self._Bxq(j, q)
                        if not bq:
                            continue
                        m = nj + 1
                        # -q(m)/m on the letter (j, m-1) = m! h_j(-m): -(m-1)! B(q,x_j)
                        coef = c * R(-_fact(m - 1) * bq) / step
                        self._add(new.setdefault(pw - m, {}),
                                  L[:t] + L[t + 1:], coef)
            cur = {pw: d for pw, d in new.items() if d}
            for pw, d in cur.items():
                self._addto(r.setdefault(pw, {}), d, self._one_R)
        r = {pw: d for pw, d in r.items() if d}
        self._eplus_cache[key] = r
        return r

    def _ann(self, i, k, ser, p):
        r"""
        The annihilation part ``Σ_{m>=0} (-1)^k (m+k)!/m! h_i(m) z^{-m-1-k}``
        of ``∂^k J_i(z)`` applied to a series ``{power: {letters: coeff}}``
        of states with lattice part ``p``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._ann(0, 0, {0: {((0, 0),): 1}}, (0,))    # h(1) J0 = 2
            {-2: {(): 2}}
            sage: L._ann(0, 1, {0: {(): 1}}, (1,))           # -h(0) on e^1, z^-2
            {-2: {(): -2}}
        """
        R = self.base_ring()
        out = {}
        sgn = (-1) ** k
        c0 = self._Bxq(i, p)
        row = self._B[i]
        for pw, d in ser.items():
            if c0:
                self._addto(out.setdefault(pw - 1 - k, {}), d,
                            R(sgn * _fact(k) * c0))
            for L, c in d.items():
                for t, (j, nj) in enumerate(L):
                    if not row[j]:
                        continue
                    m = nj + 1
                    self._add(out.setdefault(pw - m - 1 - k, {}),
                              L[:t] + L[t + 1:],
                              c * R(sgn * _fact(m + k) * row[j]))
        return {pw: d for pw, d in out.items() if d}

    def _cseries(self, letters, smax):
        r"""
        The creation part ``Π_t C_t(z)`` truncated at ``z^smax``, where
        ``C_t(z) = Σ_s ∂^{s+k}J_i / s! z^s`` for the letter ``(i,k)``.
        Returns a list (index ``s``) of ``{letters: coeff}``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._cseries(((0, 0),), 2)
            [{((0, 0),): 1}, {((0, 1),): 1}, {((0, 2),): 1/2}]
            sage: L._cseries((), 1)[:2]        # may be longer (cached)
            [{(): 1}, {}]
        """
        hit = self._cser_cache.get(letters)
        if hit is not None and len(hit) > smax:
            return hit
        if not letters:
            ser = [{(): self._one_R}] + [{} for _ in range(smax)]
        else:
            (i, k) = letters[0]
            rest = self._cseries(letters[1:], smax)
            ser = []
            for s in range(smax + 1):
                d = {}
                for s1 in range(s + 1):
                    src = rest[s - s1]
                    if not src:
                        continue
                    c = self._one_R / _fact(s1)
                    for L, v in src.items():
                        self._add(d, _insert_letter(L, (i, s1 + k)), c * v)
                ser.append(d)
        self._cser_cache[letters] = ser
        return ser

    def _letter_series(self, a, b, wmax):
        r"""
        The *letter series* ``W(z) = Σ_S (Π_{t∉S} C_t(z)) (Π_{t∈S} A_t(z))
        E^+(q,z) b`` up to ``z^wmax``, as ``{power: {letters: coeff}}``: the
        part of ``Y(a,z) b`` contributed by the Heisenberg letters of
        ``a = :Π_t ∂^{k_t} J_{i_t} e^q:`` (and ``E^+``).  ``S`` runs over the
        sub-multisets of the letters contributing their annihilation parts
        ``A_t`` (:meth:`_ann`, zero modes included); the others contribute
        their creation parts ``C_t`` (:meth:`_cseries`).  The vertex operator
        is then

            Y(a, z) b = ε(q,p) z^{B(q,p)} E^-(q,z) W(z),

        with ``E^-(q,z) = Σ_s S_s(q) z^s`` applied by :meth:`_eminus`.  The
        factors outside ``W`` depend only on the lattice parts ``(q, p)``,
        so the letter series of a ``(q, p)``-group are summed before ``E^-``
        is applied once (:meth:`_products`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._letter_series(((), (1,)), ((), (-1,)), 2)
            {0: {(): 1}}
            sage: J0 = (((0, 0),), (0,))
            sage: sorted(L._letter_series(J0, J0, 1).items())
            [(-2, {(): 2}), (0, {((0, 0), (0, 0)): 1}), (1, {((0, 1), (0, 0)): 1})]
        """
        return self._letter_series_py(a, b, wmax, self._ls_cache)

    def _letter_series_py(self, a, b, wmax, cache):
        r"""
        :meth:`_letter_series` in Python, cached in ``cache`` (``{(a, b):
        (wmax, W)}``).  The lattice part ``p`` of ``b`` enters only through
        ``B(x_i, p)``, so ``b`` may also be a vector of a module ``V_{λ+L}``
        (``p ∈ λ + L``, :mod:`lattice_vertex_algebra_module`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: J0 = (((0, 0),), (0,))
            sage: L._letter_series_py(J0, J0, 1, {}) == L._letter_series(J0, J0, 1)
            True
        """
        key = (a, b)
        hit = cache.get(key)
        if hit is not None and hit[0] >= wmax:
            return hit[1]
        from math import comb
        (La, q), p = a, b[1]
        # sub-multisets of the letters (identical letters are adjacent),
        # weighted by binomial coefficients; see the core's letter_series
        groups = []
        for x in La:
            if groups and groups[-1][0] == x:
                groups[-1][1] += 1
            else:
                groups.append([x, 1])
        subsets = [((), self._eplus(q, b), 1)]
        for (i, kk), m in groups:
            new = []
            for counts, ser, f in subsets:
                new.append((counts + (0,), ser, f))
                cur = ser
                for j in range(1, m + 1):
                    if not cur:
                        break
                    cur = self._ann(i, kk, cur, p)
                    if not cur:
                        break
                    new.append((counts + (j,), cur, f * comb(m, j)))
            subsets = new
        W = {}
        for counts, ser, f in subsets:
            if not ser:
                continue
            Lc = tuple(x for (x, m), c in zip(groups, counts)
                       for _ in range(m - c))
            smax = wmax - min(ser)
            if smax < 0:
                continue
            C = self._cseries(Lc, smax)
            for pw, dA in ser.items():
                for s in range(wmax - pw + 1):
                    dC = C[s]
                    if not dC:
                        continue
                    dW = W.setdefault(pw + s, {})
                    get = dW.get
                    for LA, cA in dA.items():
                        if f != 1:
                            cA = cA * f
                        for LC, cC in dC.items():
                            key2 = _merge_letters(LC, LA)
                            v = get(key2)
                            dW[key2] = cA * cC if v is None else v + cA * cC
        W = {pw: {L: c for L, c in d.items() if c} for pw, d in W.items()}
        W = {pw: d for pw, d in W.items() if d}
        cache[key] = (wmax, W)
        return W

    def _eminus(self, q, p, W, n):
        r"""
        The coefficient of ``z^{-n-1}`` in ``ε(q,p) z^{B(q,p)} E^-(q,z) W(z)``
        as a dict of monomials with lattice part ``q + p``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: W = L._letter_series(((), (1,)), ((), (-1,)), 2)
            sage: L._from_dict(L._eminus((1,), (-1,), W, -2)) == L.e((1,)).nproduct(L.e((-1,)), -2)
            True
        """
        return self._eminus_py(q, p, W, n, self._form(q, p), self._eps(q, p))

    def _eminus_py(self, q, p, W, n, Bqp, eps):
        r"""
        :meth:`_eminus` in Python with ``B(q, p)`` and the cocycle value
        ``eps`` given (an integer ``Bqp``; for a module ``V_{λ+L}``, ``p ∈ λ +
        L`` and ``eps = ε(q, p - λ)``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: W = L._letter_series(((), (1,)), ((), (-1,)), 2)
            sage: L._eminus_py((1,), (-1,), W, -2, -2, 1) == L._eminus((1,), (-1,), W, -2)
            True
        """
        qp = tuple(x + y for x, y in zip(q, p))
        out = {}
        get = out.get
        for pw, dW in W.items():
            sig = -n - 1 - Bqp - pw
            if sig < 0:
                continue
            Ssig = self._schur(q, sig)
            for LW, cW in dW.items():
                cW = eps * cW
                for LS, cS in Ssig.items():
                    key = (_merge_letters(LS, LW), qp)
                    v = get(key)
                    out[key] = cW * cS if v is None else v + cW * cS
        return {k: c for k, c in out.items() if c}

    def _group_series(self, items, wmax):
        """
        ``Σ c · W_{a,b}`` over the ``(a, b, c)`` of one lattice group, in
        the base ring.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._group_series([(((), (1,)), ((), (-1,)), 2)], 0)
            {0: {(): 2}}
        """
        Wsum = {}
        for a, b, c in items:
            for pw, d in self._letter_series(a, b, wmax).items():
                if pw <= wmax:
                    self._addto(Wsum.setdefault(pw, {}), d, c)
        return {pw: d for pw, d in Wsum.items() if d}

    def _ls_stage(self, plan):
        """Compute (in parallel when worthwhile) the uncached letter series of
        the pairs of ``plan``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: a, b = ((), (1,)), ((), (-1,))
            sage: L._ls_cache.clear(); L._ls_stage([((1,), (-1,), [0], 0, [(a, b, 1)])])
            sage: L._ls_cache
            {(((), (1,)), ((), (-1,))): (0, {0: {(): 1}})}
        """
        jobs = []
        seen = set()
        for q, p, gns, wmax, items in plan:
            for a, b, _ in items:
                hit = self._ls_cache.get((a, b))
                if (hit is None or hit[0] < wmax) and (a, b) not in seen:
                    seen.add((a, b))
                    jobs.append(('ls', a, b, wmax))
        self._run_jobs(jobs)

    def _add_group_series(self, acc, part):
        """The sum of two results of :meth:`_group_series` (``None``/``{}``
        for zero).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._add_group_series({0: {(): 1}}, {0: {(): -1}, 1: {((0, 0),): 2}})
            {1: {((0, 0),): 2}}
            sage: L._add_group_series(None, {0: {(): 1}})
            {0: {(): 1}}
        """
        if not part:
            return acc
        if not acc:
            return part
        out = {pw: dict(d) for pw, d in acc.items()}
        for pw, d in part.items():
            self._addto(out.setdefault(pw, {}), d, self._one_R)
        return {pw: d for pw, d in out.items() if d}

    def _products(self, pairs, ns):
        r"""
        ``Σ c · a_(n) b`` over ``(a, b, c)`` in ``pairs`` for every ``n`` in
        ``ns``, as ``{n: dict}``.

        The pairs are grouped by their lattice parts ``(q, p)``: the
        letter series of a group are summed first so that the (large) factor
        ``E^-(q,z)`` is multiplied in once per group, not once per pair.
        With more
        than ``LS_BATCH_PAIRS`` pairs the letter series are computed in batches
        that are summed and then dropped from the caches, which bounds the
        memory (otherwise every pair's series stays cached).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: a, b = ((), (1,)), ((), (-1,))
            sage: r = L._products([(a, b, QQ(1))], [0, 1]); r
            {0: {(((0, 0),), (0,)): 1}, 1: {((), (0,)): 1}}
            sage: [L._from_dict(r[n]) for n in [0, 1]] == L.e((1,)).ope(L.e((-1,)))
            True
        """
        groups = {}
        for a, b, c in pairs:
            groups.setdefault((a[1], b[1]), []).append((a, b, c))
        plan = []                              # (q, p, [n...], wmax, items)
        for (q, p), items in groups.items():
            top = max(self._nmax(a, b) for a, b, _ in items)
            gns = [n for n in ns if n <= top]
            if not gns:
                continue
            wmax = -min(gns) - 1 - self._form(q, p)
            plan.append((q, p, gns, wmax, items))
        npairs = len({(a, b) for _, _, _, _, items in plan for a, b, _ in items})
        if npairs <= LS_BATCH_PAIRS:
            # stage 1: letter series of every pair (cached), stage 2: per group sum
            self._ls_stage(plan)
            payload = []
            for q, p, gns, wmax, items in plan:
                Wsum = self._group_series(items, wmax)
                if Wsum:
                    for n in gns:
                        payload.append((q, p, Wsum, n))
        else:
            # many pairs: compute the letter series batch by batch, add each batch
            # into the group sum and drop its cache entries (bounded memory)
            payload = []
            for q, p, gns, wmax, items in plan:
                Wsum = None
                for i in range(0, len(items), LS_BATCH_PAIRS):
                    chunk = items[i:i + LS_BATCH_PAIRS]
                    self._ls_stage([(q, p, gns, wmax, chunk)])
                    Wsum = self._add_group_series(
                        Wsum, self._group_series(chunk, wmax))
                    for a, b, _ in chunk:
                        self._ls_cache.pop((a, b), None)
                if Wsum:
                    for n in gns:
                        payload.append((q, p, Wsum, n))
        self._grp_payload = payload
        try:
            parts = self._run_jobs([('grp', i) for i in range(len(payload))])
        finally:
            self._grp_payload = None
        res = {}
        for i, (_, _, _, n) in enumerate(payload):
            self._addto(res.setdefault(n, {}), parts[('grp', i)], self._one_R)
        return res

    def _nprod_rec(self, a, n, b):
        r"""
        Reference implementation of :meth:`_nprod` by recursion on the
        letters (Borcherds' normal-ordering formula), used only to
        cross-check the free-field kernel.  Much slower: its intermediate
        sums cancel heavily.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
            sage: a = (((1, 1), (0, 0)), (1, 0)); b = (((0, 0),), (-1, 1))
            sage: all(L._nprod_rec(a, n, b) == L._nprod(a, n, b)
            ....:     for n in range(-4, 6))
            True
        """
        key = (a, n, b)
        r = self._rec_cache.get(key)
        if r is not None:
            return r
        if n > self._nmax(a, b):
            r = {}
        elif not a[0]:
            q = a[1]
            if not b[0]:
                p = b[1]
                k = -n - 1 - self._form(q, p)
                S = self._schur(q, k)
                if S:
                    eps = self._eps(q, p)
                    qp = tuple(x + y for x, y in zip(q, p))
                    r = {(L, qp): eps * v for L, v in S.items()}
                else:
                    r = {}
            else:
                # e^q_(n) v b' = v e^q_(n) b' - m! B(x_j,q) e^q_(n-m-1) b'
                (j, m) = b[0][0]
                bp = (b[0][1:], b[1])
                r = self._mul_letter((j, m), self._nprod_rec(a, n, bp))
                c = self._Bxq(j, q)
                if c:
                    self._addto(r, self._nprod_rec(a, n - m - 1, bp),
                                self.base_ring()(-_fact(m) * c))
        else:
            # (u_(-1) a')_(n) b = Σ_j u_(-1-j) a'_(n+j) b + a'_(n-1-j) u_(j) b
            (i, k) = a[0][0]
            ap = (a[0][1:], a[1])
            r = {}
            top = self._nmax(ap, b)
            for j in range(top - n + 1):
                t = self._nprod_rec(ap, n + j, b)
                if t:
                    self._addto(r, self._mul_letter((i, j + k), t),
                                self._one_R / _fact(j))
            maxn = max((nb for _, nb in b[0]), default=-1)
            for s in range(maxn + 2):      # s = j - k, h_i(s) on b
                ub = self._h_act_mono(i, s, b)
                if not ub:
                    continue
                j = s + k
                c = (-1) ** k * (_fact(j) // _fact(s))
                for b2, v in ub.items():
                    self._addto(r, self._nprod_rec(ap, n - 1 - j, b2), c * v)
        self._rec_cache[key] = r
        return r

    # ------------------------------------------------------------------ #
    #  Derivative                                                         #
    # ------------------------------------------------------------------ #

    def _d1_mono(self, key):
        """``∂`` of a monomial as a dict.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._d1_mono(((), (1,)))
            {(((0, 0),), (1,)): 1}
            sage: L._from_dict(L._d1_mono((((0, 0),), (1,)))) == L.e((1,)).derivative(2)
            True
        """
        letters, q = key
        out = {}
        R = self.base_ring()
        for t, (i, n) in enumerate(letters):
            nl = _insert_letter(letters[:t] + letters[t + 1:], (i, n + 1))
            self._add(out, (nl, q), self._one_R)
        for i, qi in enumerate(q):
            if qi:
                self._add(out, (_insert_letter(letters, (i, 0)), q), R(qi))
        return out

    # ------------------------------------------------------------------ #
    #  Conformal weights, gradings, weight spaces                         #
    # ------------------------------------------------------------------ #

    def _is_simple(self):
        """Lattice vertex algebras of non-degenerate lattices are simple.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._is_simple()
            True
        """
        return True

    def _key_order(self, m):
        """
        The sort key of basis keys for leading monomials: more letters
        first, then the lattice part.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._key_order((((0, 0),), (1,)))
            (1, (1,), ((0, 0),))
        """
        return (len(m[0]), m[1], m[0])

    def _weight_generators(self):
        """The Heisenberg currents ``J_i`` and ``e^{±x_i}`` (they generate the
        lattice vertex algebra).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._weight_generators()
            [('J0', (((0, 0),), (0,))), ('e^(1)', ((), (1,))), ('e^(-1)', ((), (-1,)))]
        """
        n = self._rank
        zero = (0,) * n
        keys = [(((i, 0),), zero) for i in range(n)]
        for sign in (1, -1):
            for i in range(n):
                keys.append(((), tuple(sign if j == i else 0 for j in range(n))))
        return list(zip(_weight_labels(n, self._prefix), keys))

    def _weight_data_from(self, weights):
        r"""
        The weight data for the weights of ``J_i``, ``e^{x_i}``,
        ``e^{-x_i}``: a grading of the lattice vertex algebra has every
        ``J_i`` of weight 1 (``J_i(0) e^q = B(x_i, q) e^q`` for all ``q``) and

            Δ(q) = wt(e^q) = B(q, q)/2 + β·q,

        since ``e^q_(-1-B(q,p)) e^p = ±e^{q+p}`` forces
        ``Δ(q+p) = Δ(q) + Δ(p) + B(q, p)``; so ``wt(e^{x_i}) + wt(e^{-x_i})``
        must be ``B_ii``.  This is the grading of ``T_sug + ∂J_α`` with
        ``β = -Bα`` (``T_sug = 1/2 Σ (B^{-1})_{ij} :J_i J_j:``, ``J_α = Σ α_i
        J_i``), whatever ``T`` induced it (e.g. ``T_sug + ∂e^q`` with
        ``B(q, q) = 2``, whose ``L_0`` has Jordan blocks).

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
            sage: J0, J1 = L.heisenberg_currents()
            sage: Tsug = (2*J0.no(J0) + J0.no(J1) + J1.no(J0) + 2*J1.no(J1)) / 6
            sage: LT = L.with_conformal_vector(Tsug)
            sage: LT.central_charge(), LT.degrees()
            (2,
             {'J0': 1, 'J1': 1, 'e^(-1,0)': 1, 'e^(0,-1)': 1, 'e^(0,1)': 1, 'e^(1,0)': 1})
            sage: [LT.e(q).degree() for q in [(1, 0), (1, 1), (2, 1)]]
            [1, 1, 3]
            sage: LT = L.with_conformal_vector(Tsug + J0.derivative())     # α = x_0
            sage: LT.central_charge(), LT.e((1, 0)).degree()
            (-22, -1)
            sage: L.with_conformal_vector(J0.no(J0) / 4)
            Traceback (most recent call last):
            ...
            ValueError: T_(0) J1 = ... is not ∂J1: T does not act as the translation operator on J1
            sage: LT = L.with_conformal_vector(Tsug + L.e((1, 1)).derivative())
            doctest:warning...
            ...LogarithmicWarning: L_0 = T_(1) is not diagonalizable ...
            sage: LT.central_charge(), LT.l0_jordan_chains()['J0']
            (2, [J0, e^(1,1)])
        """
        n = self._rank
        R = self.base_ring()
        Bm = self._B
        hJ, wp, wm = weights[:n], weights[n:2 * n], weights[2 * n:]
        for i in range(n):
            if hJ[i] != 1:
                raise ValueError(
                    'the Heisenberg current {} must have weight 1 (J_(0) e^q = '
                    'B(x, q) e^q), but the diagonal coefficient of T_(1) {} is '
                    '{}'.format(self._names[i], self._names[i], hJ[i]))
            if wp[i] + wm[i] != Bm[i][i]:
                raise ValueError(
                    'the weights {} of e^(x_{}) and {} of e^(-x_{}) (diagonal of '
                    'T_(1)) do not add up to B_ii = {}, so they do not define a '
                    'grading'.format(wp[i], i, wm[i], i, Bm[i][i]))
        A = [[R(Bm[i][j]) / 2 for j in range(n)] for i in range(n)]
        beta = [R(wp[i]) - R(Bm[i][i]) / 2 for i in range(n)]
        Binv = self._gram.change_ring(QQ).inverse()
        alpha = [-sum(R(Binv[i, j]) * beta[j] for j in range(n))
                 for i in range(n)]
        return {'hJ': [R.one()] * n, 'A': A, 'beta': beta, 'alpha': alpha}

    @staticmethod
    def _Delta(q, d):
        """The conformal weight ``Δ(q)`` of ``e^q`` for weight data ``d``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: LatticeVertexAlgebra._Delta((1, 2), {'A': [[1, 0], [0, 1/2]], 'beta': [0, 1]})
            5
        """
        A, beta = d['A'], d['beta']
        n = len(q)
        return (sum(A[i][j] * q[i] * q[j] for i in range(n) for j in range(n))
                + sum(beta[i] * q[i] for i in range(n)))

    def _weight_key(self, m, data):
        """
        The weight of a basis key for the weight data ``data``:
        ``Δ(q)`` plus the weights of its letters.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: d = {'hJ': [1], 'A': [[1]], 'beta': [0]}             # T = :J0 J0:/4
            sage: d['hJ'] = [1]
            sage: L._weight_key((((0, 1),), (1,)), d), L._weight_key(((), (2,)), d)
            (3, 4)
        """
        letters, q = m
        hJ = data['hJ']
        return self._Delta(q, data) + sum((hJ[i] + k for i, k in letters),
                                          self.base_ring().zero())

    def _grading_data(self, spec, group):
        r"""
        ``spec``: ``'lattice'`` (``e^q`` has grade ``q``, in ``ZZ^n``; in
        ``ZZ`` for rank 1) or the list of the grades of
        ``e^{x_0}, ..., e^{x_{n-1}}`` (elements of the grading group
        ``group``, or of an inferred one, see
        :meth:`VertexAlgebra.with_horizontal_grading`); then ``e^q`` has grade
        ``Σ q_i horizontal_grade(x_i)``.  The Heisenberg currents have grade 0
        (forced, since ``∂e^q = Σ q_i :J_i e^q:``); such a grading is always
        compatible with the vertex algebra structure.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L0 = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, -1]]))
            sage: L = L0.with_horizontal_grading('lattice')
            sage: L.e((2, -1)).horizontal_grade(), L.J(0).horizontal_grade()
            ((2, -1), (0, 0))
            sage: L = L0.with_horizontal_grading([1, 1])          # the grade q_0 + q_1
            sage: L.J(1).no(L.e((2, -1))).horizontal_grade()
            1
            sage: L = L0.with_horizontal_grading([1, 1], group=Zmod(2))    # q_0 + q_1 mod 2
            sage: L.e((2, -1)).horizontal_grade(), L.e((2, -1)).parity()
            (1, 1)
        """
        from sage.modules.free_module import FreeModule

        from ._grading_groups import GradingGroup, infer_grading_group
        n = self._rank
        if isinstance(spec, str):
            if spec != 'lattice':
                raise ValueError("the grading must be 'lattice' or a list of "
                                 "the grades of the basis vectors")
            if group is not None:
                raise ValueError("the grading 'lattice' has the lattice as its "
                                 "group; give the grades of the basis vectors "
                                 "to grade by another group")
            A = GradingGroup(ZZ if n == 1 else FreeModule(ZZ, n))
            return {'group': A, 'Gamma': matrix.identity(ZZ, n)}
        vals = list(spec)
        if len(vals) != n:
            raise ValueError('{} grades expected (one per basis vector)'.format(n))
        A = infer_grading_group(vals, group)
        cols = [A.coords(v) for v in vals]
        Gamma = matrix(ZZ, A.k, n, lambda a, b: cols[b][a])
        return {'group': A, 'Gamma': Gamma}

    def _grade_key(self, m):
        """
        The grade of a basis key (coordinates in the grading group).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L.with_horizontal_grading('lattice')._grade_key((((0, 0),), (2,)))
            (2,)
            sage: L.with_horizontal_grading([1], group=Zmod(2))._grade_key(((), (3,)))
            (1,)
        """
        gd = self._grading
        Gamma = gd['Gamma']
        q = m[1]
        return gd['group'].reduce(
            sum(Gamma[a, b] * q[b] for b in range(self._rank))
            for a in range(Gamma.nrows()))

    def _lattice_part(self, G, d):
        """The data of the lattice vectors of grade ``G`` (``None``: any):
        ``None`` if there are none, else a dict with ``q = q0 + K m`` and
        ``Δ(q) = m^T Q m + b·m + c`` (and the rational weight data).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: d = {'hJ': [1], 'A': [[1]], 'beta': [0]}             # T = :J0 J0:/4
            sage: d['hJ'] = [1]
            sage: P = L._lattice_part(None, d); P['q0'], P['K'], P['Q'], P['c']
            ((0), [1], [1], 0)
            sage: P = L.with_horizontal_grading('lattice')._lattice_part((3,), d)
            sage: P['q0'], P['dim'], P['c']
            ((3), 0, 9)
        """
        from sage.modules.free_module_element import vector

        from .weight_spaces import _affine_integer_solutions, _to_rational
        n = self._rank
        hJ = [_to_rational(x, 'conformal weight of ' + s)
              for x, s in zip(d['hJ'], self._names)]
        A = matrix(QQ, n, n, [[_to_rational(x, 'lattice weight coefficient')
                               for x in row] for row in d['A']])
        beta = vector(QQ, [_to_rational(x, 'lattice weight coefficient')
                           for x in d['beta']])
        if G is None:
            sol = _affine_integer_solutions(None, None, n)
        else:
            gd = self._grading
            sol = _affine_integer_solutions(gd['Gamma'], G, n,
                                           gd['group'].moduli)
        if sol is None:
            return None
        q0, K = sol
        dim = K.ncols()
        return {'hJ': hJ, 'A': A, 'beta': beta, 'q0': q0, 'K': K, 'dim': dim,
                'Q': K.transpose() * A * K,
                'b': 2 * K.transpose() * A * q0 + K.transpose() * beta,
                'c': q0 * A * q0 + beta * q0}

    @staticmethod
    def _lattice_part_finite(P):
        """
        Whether the weight spaces of :meth:`_lattice_part` ``P`` are finite:
        positive Heisenberg weights and ``Δ`` positive definite on ``q0 + K m``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: d = {'hJ': [1, 1], 'A': [[1/2, 0], [0, -1/2]], 'beta': [0, 0]}
            sage: L = va.Lattice(QQ, matrix([[1, 0], [0, -1]]))
            sage: L._lattice_part_finite(L._lattice_part(None, d))
            False
            sage: A1 = va.Lattice(QQ, matrix([[2]]))
            sage: A1._lattice_part_finite(A1._lattice_part(None, {'hJ': [1], 'A': [[1]], 'beta': [0]}))
            True
        """
        return all(x > 0 for x in P['hJ']) and (P['dim'] == 0 or
                                                 P['Q'].is_positive_definite())

    def _weight_space_enumeration(self, W, G, data):
        r"""
        Solve ``horizontal_grade(q) = G`` over ``ZZ`` (``q = q0 + K m``; torsion
        grades are congruences); the space is finite dimensional iff all
        Heisenberg weights are positive and ``Δ`` restricted to
        ``q0 + K ZZ^d`` is positive definite, and the ``q`` are then the
        points of an ellipsoid.  Otherwise it is enumerated in stages
        ``max(|m|_∞, number of letters) = s``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L0 = LatticeVertexAlgebra(QQ, matrix([[2]]))
            sage: J = L0.J(0); L = L0.with_conformal_vector(J.no(J) / 4)
            sage: [L.weight_space(h).dimension() for h in range(5)]
            [1, 3, 4, 7, 13]
            sage: L.weight_space(1).basis()
            [e^(-1), e^(1), J0]
            sage: Lg = L.with_horizontal_grading('lattice')
            sage: [Lg.weight_space(h, horizontal_grade=0).dimension() for h in range(6)]
            [1, 1, 2, 3, 5, 7]
            sage: Lg.weight_space(4, horizontal_grade=2).basis()
            [e^(2)]
            sage: Lg = L.with_horizontal_grading([1], group=Zmod(2))      # q odd
            sage: [Lg.weight_space(h, horizontal_grade=1).dimension() for h in range(6)]
            [0, 2, 2, 4, 6, 10]
        """
        from sage.modules.free_module_element import vector

        from .weight_spaces import (
            _box_shell,
            _ellipsoid_points,
            _KeyEnumeration,
            _LetterMultisets,
        )
        n = self._rank
        P = self._lattice_part(G, data)
        if P is None:
            return _KeyEnumeration(True, factory=lambda: iter(()))
        hJ, q0, K, dim = P['hJ'], P['q0'], P['K'], P['dim']
        zeros = [()] * n
        evens = [False] * n

        def heis(q):
            return _LetterMultisets(hJ, zeros, evens, W - self._Delta(q, data),
                                   ())

        def q_of(m):
            return tuple(int(x) for x in (q0 + K * vector(ZZ, list(m))))

        if self._lattice_part_finite(P):
            def factory():
                for m in _ellipsoid_points(P['Q'], P['b'], P['c'], W):
                    q = q_of(m)
                    for letters in heis(q):
                        yield (letters, q)
            return _KeyEnumeration(True, factory=factory)

        def stage(s):
            for t in range(s + 1):
                for m in _box_shell(dim, t):
                    q = q_of(m)
                    H = heis(q)
                    for Kl in (range(s + 1) if t == s else [s]):
                        for letters in H.exactly(Kl):
                            yield (letters, q)
        return _KeyEnumeration(False, stage=stage)

    def _weight_bounds(self, G, data):
        """
        ``(lower bound, denominator)`` of the weights of grade ``G`` (see
        :meth:`vertex_algebra.VertexAlgebra._weight_bounds`): ``Infinity`` if no
        lattice vector has grade ``G``, ``None`` if not certified finite.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A1 = va.Lattice(QQ, matrix([[2]]))
            sage: A1._weight_bounds(None, {'hJ': [1], 'A': [[1]], 'beta': [1/2]})
            (-1/16, 2)
            sage: d = {'hJ': [1, 1], 'A': [[1/2, 0], [0, -1/2]], 'beta': [0, 0]}
            sage: va.Lattice(QQ, matrix([[1, 0], [0, -1]]))._weight_bounds(None, d)
            (None, 2)
        """
        from sage.arith.functions import lcm
        d = data
        P = self._lattice_part(G, d)
        den = lcm([QQ(x).denominator() for row in d['A'] for x in row]
                  + [QQ(x).denominator() for x in d['beta']]
                  + [QQ(x).denominator() for x in d['hJ']] + [1])
        if P is None:
            return Infinity, int(den)
        if not self._lattice_part_finite(P):
            return None, int(den)
        if P['dim'] == 0:
            return P['c'], int(den)
        Q, b, c = P['Q'], P['b'], P['c']
        mstar = -Q.inverse() * b / 2
        return mstar * Q * mstar + b * mstar + c, int(den)

    # ------------------------------------------------------------------ #
    #  Base-class hooks                                                   #
    # ------------------------------------------------------------------ #

    def _deriv_key(self, m, n):
        """
        The ``n``-th derivative of a basis key (raw dict), by :meth:`_d1_mono`
        on the cached ``(n-1)``-th.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: sorted(L._deriv_key(((), (1,)), 2).items())
            [((((0, 0), (0, 0)), (1,)), 1), ((((0, 1),), (1,)), 1)]
        """
        r = {}
        for m2, c in self._deriv_mono(m, n - 1).items():
            self._addto(r, self._d1_mono(m2), c)
        return r

    def _nmax_key(self, a, b):
        """
        :meth:`_nmax` (the base-class hook).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._nmax_key(((), (1,)), ((), (-1,)))
            1
        """
        return self._nmax(a, b)

    def _ope_key(self, a, b):
        """
        The OPE of two basis keys (tuple of raw pole dicts, trimmed), by
        :meth:`_products` (the base class calls the engine instead when there
        is one).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._ope_key(((), (1,)), ((), (-1,)))
            ({(((0, 0),), (0,)): 1}, {((), (0,)): 1})
            sage: L._ope_key(((), (1,)), ((), (1,)))
            ()
        """
        N = self._nmax(a, b)
        if N < 0:
            return ()
        res = self._products([(a, b, self._one_R)], list(range(N + 1)))
        return tuple(_trim([res.get(n, {}) for n in range(N + 1)]))

    def _no_key(self, a, b):
        """
        The normally ordered product of two basis keys (raw dict), by
        :meth:`_products`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: sorted(L._no_key(((), (1,)), ((), (-1,))).items())
            [((((0, 0), (0, 0)), (0,)), 1/2), ((((0, 1),), (0,)), 1/2)]
        """
        return self._products([(a, b, self._one_R)], [-1]).get(-1, {})

    def _nprod_key(self, a, n, b):
        """
        The ``n``-th product of two basis keys (raw dict), by
        :meth:`_products`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._nprod_key(((), (1,)), -2, ((), (-1,))) == L._products(
            ....:     [(((), (1,)), ((), (-1,)), 1)], [-2])[-2]
            True
        """
        return self._products([(a, b, self._one_R)], [n]).get(n, {})

    def _build_engine(self):
        """
        The C++ engine (``_LatticeEngineKernel``) if it is available
        and not switched off (``implementation='python'``) and the base ring is ``QQ``, an absolute
        number field ``K``, a polynomial ring over ``QQ`` or ``K`` (computed
        over itself) or ``Frac(A)``, ``A`` such a polynomial ring (computed over
        ``A``; element coefficients with non-constant denominators take the
        generic path); else ``None`` (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: E = L._build_engine()
            sage: E is None or E.kind() == 'q'
            True
            sage: va.Lattice(QQ, matrix([[2]]), implementation='python')._build_engine() is None
            True
        """
        if _LVA_ENGINE is None or self._implementation == 'python':
            return None
        F = self.base_ring()
        try:
            kind = _engine_ring(F)
            if kind is not None:
                return _LatticeEngineKernel(self, F, kind, F)
            alg = self._algebraic_engine_ring(
                [c for row in (self._eps_mat or ()) for c in row])
            if alg is not None:               # QQbar / AA: in a number field
                kind, alg = alg
                E = _LatticeEngineKernel(self, F, kind, alg.K, up=alg.up, down=alg.down)
                E._alg = alg
                return E
            A = self._lift_ring()
            if A is None or A is ZZ:
                return None
            kind = _engine_ring(A)
            if kind not in ('qpoly', 'qmpoly', 'nfmpoly'):
                return None

            def up(x):
                d = x.denominator()
                if not d.is_constant():
                    raise ValueError('denominator %s is not a constant' % d)
                return A(x.numerator()) / d.constant_coefficient()

            return _LatticeEngineKernel(self, F, kind, A, up=up, down=F)
        except (AttributeError, TypeError, ValueError, ArithmeticError,
                NotImplementedError):
            return None

    def _ope_raw(self, dA, dB):
        """All pairs evaluated jointly by :meth:`_products` (grouped by
        lattice parts), or by the C++ engine.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: L._ope_raw({a: 1}, {b: 1}) == list(L._ope_key(a, b))
            True
        """
        r = self._engine_call('ope', dA, dB)
        if r is not None:
            return r
        pairs = self._pairs(dA, dB)
        if not pairs:
            return []
        N = max(self._nmax(a, b) for a, b, _ in pairs)
        res = self._products(pairs, list(range(N + 1)))
        return [res.get(n, {}) for n in range(N + 1)]

    def _ope_raw_many(self, dA, dBs):
        """One :meth:`_ope_raw` per ``dB`` (``_products`` sums over all
        pairs, so the OPEs are not batched; the engine batches).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: a, b = ((), (1,)), (((0, 0),), (-1,))
            sage: L._ope_raw_many({a: 1}, [{b: 1}, {a: 1}]) == [L._ope_raw({a: 1}, {b: 1}), []]
            True
        """
        r = self._engine_call('ope_many', dA, dBs)
        if r is not None:
            return r
        return [self._ope_raw(dA, dB) for dB in dBs]

    def _nprod_raw(self, dA, n, dB):
        """
        ``dA_(n) dB`` of raw elements (raw dict): the engine, or
        :meth:`_products` over all pairs.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: a, b = ((), (1,)), ((), (-1,))
            sage: L._nprod_raw({a: 1}, -1, {b: 2}) == {k: 2*c for k, c in L._no_key(a, b).items()}
            True
        """
        r = self._engine_call('nprod', dA, n, dB)
        if r is not None:
            return r
        return self._products(self._pairs(dA, dB), [n]).get(n, {})

    # ------------------------------------------------------------------ #
    #  Custom jobs: ('ls', a, b, wmax) and ('grp', index)                  #
    # ------------------------------------------------------------------ #

    def _compute_custom_job(self, job):
        """
        The jobs ``('ls', a, b, wmax)`` (a letter series) and ``('grp', i)``
        (:meth:`_eminus` of the ``i``-th group of the current :meth:`_products`
        call).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._compute_custom_job(('ls', ((), (1,)), ((), (-1,)), 0))
            {0: {(): 1}}
        """
        if job[0] == 'ls':
            return self._letter_series(job[1], job[2], job[3])
        if job[0] == 'grp':
            return self._eminus(*self._grp_payload[job[1]])
        return VertexAlgebra._compute_custom_job(self, job)

    def _raw_caches(self):
        """
        The raw caches (checked by the cache-contract tests): the base
        class's and the letter series cache.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]))
            sage: L._raw_caches()
            ['_ope_cache', '_no_cache', '_nprod_cache', '_deriv_cache', '_ls_cache']
        """
        return VertexAlgebra._raw_caches(self) + ['_ls_cache']
