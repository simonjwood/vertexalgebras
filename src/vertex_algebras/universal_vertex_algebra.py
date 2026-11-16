# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Universal vertex algebras: strong generators and OPE relations.

This is a port of K. Thielemans' Mathematica package *OPEdefs* 3.1
[Thi1991]_ (not included in this repository), as a subclass of
:class:`vertex_algebra.VertexAlgebra` (linearity, caching, parallel
evaluation and the element API live there).

A universal vertex algebra is given by its data: the base ring, the
generator names (their order is the *generator order*), the OPEs of the
generators and their parities (omitted names are even); optionally
conformal weights, a Virasoro vector and a grading.  It cannot be changed
afterwards (methods ``with_...`` give new algebras)::

    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: va = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
    sage: va.opes()
    {('J', 'J'): [0, 1]}

The monomial recursion is implemented twice, with identical results, the
compiled one built when the package is installed (``meson.build``;
``vertex_algebra._compiled_module()``):

- a C++ engine with FLINT coefficients (``_universal_engine``
  ``.hpp`` / ``.pyx``, ``_EngineKernel``), used for the base rings
  ``QQ``, ``GF(p)``, absolute number fields ``K``, polynomial rings over
  ``QQ`` or ``K`` and their fraction fields (lifted, ``_LiftedKernel``);
  it computes in the threads of
  :class:`vertex_algebra.VertexAlgebraParallelism` (the result does not
  depend on their number);
- the pure-Python recursion (``_fb_*`` methods) for every other base ring.

The constructor's ``implementation`` (``'engine'``, ``'python'``; default:
the engine when it takes the base ring) chooses one, e.g. to compare them.

Over a fraction field ``Frac(A)`` the engine computes over a polynomial
ring and maps the results back (``_LiftedKernel``): fraction-field
arithmetic, a gcd per operation, otherwise dominates.

The list of generator names fixes the *generator order*.  A basis element
is a monomial of generator derivatives, a tuple of letters
``(generator_index, derivative)`` in canonical order: higher derivatives
first, and among equal derivatives the generator listed earlier first.  (In
terms of ``_cmp_letter`` the tuple is weakly decreasing: ``(a, n)`` is
larger than ``(b, m)`` iff ``n > m``, or ``n == m`` and ``a < b``.)  Such a
tuple is the right-nested normal ordered product, first letter outermost::

    ((0, 0), (1, 0))   ==   :generator[0] generator[1]:
    ((1, 1), (0, 0))   ==   :∂generator[1] generator[0]:

The vacuum is the empty tuple ``()``.

The OPEs are a dict: the value of ``('a', 'b')`` is a *list of poles
whose first entry is the first order pole*: entry ``k`` is the coefficient
of ``(z - w)^(-k-1)`` in the OPE of the generators.  Each pair is given in
one orientation, the other follows by skew-symmetry; the algebra stores
the pairs ``a <= b`` in the generator order.  The coefficients are
elements of the algebra being defined, so they are written as normally
ordered expressions in the generators (see
:mod:`normally_ordered_expressions`: a word ``a b c`` is the right-nested
normally ordered product ``:a :b c::``, in any order of the letters), or
as scalars, generator names or dicts ``{word: coefficient}`` of words of
``(name, derivative)`` letters.  The Virasoro field over
``Frac(QQ['c'])``::

    sage: R = Frac(QQ['c']); c = R.gen()
    sage: E.<T> = NormallyOrderedExpressions(R)
    sage: V = UniversalVertexAlgebra(R, E, {('T', 'T'): [T.derivative(), 2*T, 0, c/2]})
    sage: T = V.gen('T')
    sage: T.ope(T)
    [dT, 2*T, 0, 1/2*c]
    sage: T.derivative().ope(T)
    [0, -dT, -4*T, 0, -2*c]
    sage: T.ope(T.derivative())
    [d^2T, 3*dT, 4*T, 0, 2*c]
    sage: T.nproduct(T, 4)
    0

Coefficients out of the order of the basis monomials are normally ordered
with the OPEs themselves, e.g. ``∂Λ`` of Zamolodchikov's ``W_3`` algebra
[Zam1985]_ (``:L ∂L:`` is not a basis monomial)::

    sage: F.<L, W> = NormallyOrderedExpressions(R)
    sage: Lam = L*L - 3/10*L.derivative(2)
    sage: beta = 16/(22 + 5*c)
    sage: W3 = UniversalVertexAlgebra(R, F,
    ....:     {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
    ....:      ('L', 'W'): [W.derivative(), 3*W],
    ....:      ('W', 'W'): [L.derivative(3)/15 + beta*Lam.derivative(),
    ....:                   3/10*L.derivative(2) + 2*beta*Lam, L.derivative(),
    ....:                   2*L, 0, c/3]},
    ....:     conformal_vector=L, verify_jacobi=True)
    sage: W3.opes()[('W', 'W')][0] == W3(L.derivative(3)/15 + beta*Lam.derivative())
    True
    sage: W3.central_charge(), W3.degrees()
    (c, {'L': 2, 'W': 3})

Odd generators are declared with ``parities``; every coefficient of the
OPE of ``a`` and ``b`` must have the parity ``|a| + |b|``.  A bc-ghost
system (both fields odd)::

    sage: bc = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
    ....:                             parities={'b': 1, 'c': 1})
    sage: b = bc.gen('b'); g = bc.gen('c')
    sage: b.ope(g)
    [1]
    sage: g.ope(b)
    [1]
    sage: b.no(g)
    :b c:
    sage: g.no(b) == -b.no(g)
    True
    sage: b.ope(g.derivative())
    [0, 1]
    sage: g.parity(), b.parity()
    (1, 1)
    sage: UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]}, parities={'c': 1})
    Traceback (most recent call last):
    ...
    ValueError: the OPE of b and c: pole 1 contains 1 of parity 0, expected 1

Conformal weights (without a Virasoro vector), a Virasoro vector and a
grading are part of the data too, and the methods ``with_opes``,
``with_degrees``, ``with_conformal_vector`` and ``with_horizontal_grading``
give the algebra with other data (``with_opes`` keeps the conformal weights
given as such, not a Virasoro vector)::

    sage: W3h = W3.with_conformal_vector(None).with_degrees({'L': 2, 'W': 3})
    sage: W3h.conformal_vector() is None, W3h.degrees()
    (True, {'L': 2, 'W': 3})
    sage: W3h.with_conformal_vector('L').central_charge()
    c
    sage: free = W3h.with_opes({})
    sage: free.opes(), free.degrees()
    ({}, {'L': 2, 'W': 3})

The SU(2)_k Sugawara stress tensor ([Thi1991]_, OPEdefs 3.1 manual, eq. 44,
reproducing ``OPEPole[4][Ts, Ts] == (3 k One)/(2 (2 + k))``)::

    sage: K = Frac(QQ['k']); k = K.gen()
    sage: S = UniversalVertexAlgebra(K, ['J1', 'J2', 'J3'],
    ....:     {('J1', 'J1'): [0, -k/2], ('J2', 'J2'): [0, -k/2], ('J3', 'J3'): [0, -k/2],
    ....:      ('J1', 'J2'): ['J3'], ('J2', 'J3'): ['J1'], ('J1', 'J3'): [{(('J2', 0),): -1}]})
    sage: J1 = S.gen('J1'); J2 = S.gen('J2'); J3 = S.gen('J3')
    sage: Ts = -1/(k+2) * (J1.no(J1) + J2.no(J2) + J3.no(J3))
    sage: Ts.nproduct(Ts, 3) == 3*k/(2*(k+2)) * S.one()
    True
    sage: Ts.nproduct(J1, 1) == J1
    True
    sage: Ts.nproduct(J1, 0) == J1.derivative()
    True
    sage: o = J1.ope(J2.no(J1))
    sage: o[0] == J1.no(J3) + J2.derivative()
    True
    sage: o[1] == (1 - k/2)*J2
    True

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import itertools

from sage.arith.misc import binomial, factorial, rising_factorial
from sage.rings.infinity import Infinity
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .normally_ordered_expressions import (
    NormallyOrderedExpressions,
    _normalize_generators,
)
from .vertex_algebra import (
    VertexAlgebra,
    _compiled_module,
    _derivative_str,
    _engine_ring,
    _EngineKernel,
    _is_int_like,
    _normalize_implementation,
    _word_str,
)


def _cmp_letter(a, b):
    """
    Compare letters: 1 if ``a > b``, 0 if equal, -1 otherwise.  ``(a, n)``
    is larger than ``(b, m)`` iff ``n > m``, or ``n == m`` and ``a < b``
    (higher derivatives first, then earlier generators first).

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import _cmp_letter
        sage: _cmp_letter((0, 1), (1, 1)), _cmp_letter((1, 0), (0, 1)), _cmp_letter((0, 0), (0, 0))
        (1, -1, 0)
    """
    if a[1] != b[1]:
        return 1 if a[1] > b[1] else -1
    if a[0] != b[0]:
        return 1 if a[0] < b[0] else -1
    return 0


def _is_letter(x):
    """Return True if ``x`` is a letter ``(index, derivative)`` of integers.

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import _is_letter
        sage: _is_letter((0, 1)), _is_letter((0, 1, 2)), _is_letter(('L', 1))
        (True, False, False)
    """
    return (isinstance(x, tuple) and len(x) == 2
            and _is_int_like(x[0]) and _is_int_like(x[1]))


def _is_standard_word(word, odd):
    """
    True if the word (a tuple of letters) is a basis key: weakly decreasing
    for ``_cmp_letter``, without a repeated letter of an odd generator
    (``odd``: the set of their indices).  Then the right-nested normally
    ordered product of its letters is the basis monomial of the key.

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import _is_standard_word
        sage: _is_standard_word(((0, 1), (1, 1), (0, 0)), set())
        True
        sage: _is_standard_word(((0, 0), (0, 1)), set())
        False
        sage: _is_standard_word(((1, 0), (1, 0)), set()), _is_standard_word(((1, 0), (1, 0)), {1})
        (True, False)
    """
    prev = None
    for t in word:
        if prev is not None:
            c = _cmp_letter(prev, t)
            if c < 0 or (c == 0 and t[0] in odd):
                return False
        prev = t
    return True


def _expression_of(E, x):
    r"""
    ``x`` as an element of the normally ordered expressions ``E`` (a
    :class:`normally_ordered_expressions.NormallyOrderedExpressions`): an
    expression (fields matched by name), an element of a universal vertex
    algebra (its basis monomials read as words, generators matched by
    name), a dict ``{word: coefficient}``, a word (tuple of letters
    ``(name, derivative)``), a generator name or a scalar.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.universal_vertex_algebra import _expression_of
        sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
        sage: _expression_of(E, {(('b', 0), ('a', 1)): 2, (): 1})
        1 + 2*:b da:
        sage: V = UniversalVertexAlgebra(QQ, ['b', 'a'])
        sage: b, a = V.gens()
        sage: x = b.no(a.derivative()); x
        :da b:
        sage: _expression_of(E, x)
        :da b:
        sage: _expression_of(E, 'b'), _expression_of(E, 0), _expression_of(E, 1/2)
        (b, 0, 1/2)
        sage: _expression_of(E, UniversalVertexAlgebra(QQ, ['z']).gen(0))
        Traceback (most recent call last):
        ...
        ValueError: z is not one of the generators ['a', 'b']
    """
    R = E.base_ring()
    P = _parent(x)
    if (getattr(P, '_is_normally_ordered_expressions', False)
            or getattr(P, '_is_universal_vertex_algebra', False)):
        if P is E:
            return x
        names = P.variable_names()
        idx = E._idx
        d = {}
        for w, c in x.monomial_coefficients().items():
            u = []
            for i, n in w:
                if names[i] not in idx:
                    raise ValueError('{} is not one of the generators {}'.format(
                        names[i], list(E._gens)))
                u.append((idx[names[i]], n))
            d[tuple(u)] = R(c)
        return E._from_dict(d, remove_zeros=True)
    if isinstance(x, dict):
        return E.sum(R(c) * E(w) for w, c in x.items())
    return E(x)


# The C++ engine (built when the package is installed); None when switched
# off by the developer switch (vertex_algebra._compiled_module)
_UVA_ENGINE, _ENGINE_OFF = _compiled_module('_universal_engine')


class _LiftedKernel:
    r"""
    The ring the C++ engine computes over when the base ring is a fraction
    field ``F = Frac(A)`` with ``A ⊇ QQ``: the polynomial ring ``Rk = A``
    (all declared OPE coefficients in ``A``) or ``Rk = A[u_0, ...,
    u_{m-1}]``, ``u_j`` standing for ``1/p_j``, ``p_j`` the irreducible
    factors of the denominators of the declared OPE coefficients, with the
    maps.  Results leave the engine through the ring homomorphism ``phi:
    Rk -> F`` (identity on ``A``, ``u_j -> 1/p_j``, :meth:`_down`).  The
    recursion only adds, multiplies and divides by integers, so ``phi`` of
    the result over ``Rk`` is the result over ``F``; terms that vanish only
    under ``phi`` are carried along and dropped at the end.  Polynomial
    arithmetic needs no gcd per operation (a multiply-add ~0.3 µs in
    ``QQ[c]``, ~1 µs in ``QQ[c, u_0]``, ~4.4 µs in ``Frac(QQ[c])`` with Sage
    elements, measured 2026-10-11).

    EXAMPLES:

    Zamolodchikov's ``W_3`` algebra has the denominator ``5c + 22``::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
        sage: K.<c> = Frac(QQ['c'])
        sage: E.<L, W> = NormallyOrderedExpressions(K)
        sage: Vir = {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
        ....:        ('L', 'W'): [W.derivative(), 3*W]}
        sage: Lam = L*L - 3/10*L.derivative(2)
        sage: beta = 16/(22 + 5*c)
        sage: V = UniversalVertexAlgebra(K, E, dict(Vir) | {('W', 'W'): [
        ....:     L.derivative(3)/15 + beta*Lam.derivative(),
        ....:     3/10*L.derivative(2) + 2*beta*Lam, L.derivative(), 2*L, 0, c/3]})
        sage: KL = _lifted_kernel(V)
        sage: KL.ring(), KL.denominator_factors()
        (Multivariate Polynomial Ring in c, u0 over Rational Field, [c + 22/5])
        sage: KL._decl[(1, 1)][1][((0, 0), (0, 0))]
        32/5*u0

    Without such denominators the kernel ring is ``A`` itself::

        sage: V = V.with_opes(Vir)
        sage: _lifted_kernel(V).ring()
        Univariate Polynomial Ring in c over Rational Field
    """

    def __init__(self, V, A, factors, X=None):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _LiftedKernel
            sage: K.<c> = Frac(QQ['c']); A = K.ring()
            sage: KL = _LiftedKernel(va.W3(K, c), A, [A.gen() + 22/5])
            sage: KL.ring(), KL._F is K, KL._n, KL.denominator_factors()
            (Multivariate Polynomial Ring in c, u0 over Rational Field, True, 1, [c + 22/5])
        """
        B = V.base_ring()
        F = B if X is None else B.base_ring()
        self._B = B                     # the base ring: F, or F[x...] (X)
        self._F = F                     # the fraction field Frac(A)
        self._A = A
        self._factors = factors
        n = A.ngens()
        self._n = n
        self._m = 0 if X is None else B.ngens()
        if factors or X is not None:
            names = list(A.variable_names())
            xnames = [] if X is None else list(B.variable_names())
            prefix = 'u'
            while any(x.startswith(prefix) for x in names + xnames):
                prefix += 'u'
            Rk = PolynomialRing(A.base_ring(), names + xnames
                                + ['{}{}'.format(prefix, j) for j in range(len(factors))])
            self._emb = A.hom(Rk.gens()[:n], Rk)
            self._x = Rk.gens()[n:n + self._m]
            self._u = Rk.gens()[n + self._m:]
            self._powers = {}
            self._xmon = {}
        else:
            Rk = A
        self._Rk = Rk
        self._ring_one = Rk.one()
        try:
            E = F._element_class
            if E(F, A.gen(), A.one(), coerce=False, reduce=False) != F(A.gen()):
                E = None
        except Exception:
            E = None
        self._elt = E
        self._decl = {key: [{m: self._up(x) for m, x in el._monomial_coefficients.items()
                             if x} for el in poles]
                      for key, poles in V._opes.items()}

    def ring(self):
        """The polynomial ring the engine computes over.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL.ring()
            Multivariate Polynomial Ring in c, u0 over Rational Field
        """
        return self._Rk

    def denominator_factors(self):
        """The irreducible (normalised) denominators ``p_j``, ``u_j -> 1/p_j``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL.denominator_factors()
            [c + 22/5]
        """
        return list(self._factors)

    def _up(self, x):
        """An element of ``Rk`` mapped to ``x`` (in the base ring) by ``phi``.

        Raises ``ValueError`` for a denominator outside the ``p_j``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._up(c/(5*c + 22)^2)
            1/25*c*u0^2
            sage: KL._up(1/c)
            Traceback (most recent call last):
            ...
            ValueError: denominator c has a factor outside [c + 22/5]

            Over ``Frac(A)[x]`` (a bootstrap's ring) the ``x`` are variables of ``Rk``::

                sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                sage: B.<x> = K[]
                sage: V = UniversalVertexAlgebra(B, ['J'], {('J', 'J'): [0, x/(c + 1)]})
                sage: KB = _lifted_kernel(V); KB.ring()
                Multivariate Polynomial Ring in c, x, u0 over Rational Field
                sage: KB._up(x^2/(c + 1) + c)
                x^2*u0 + c
        """
        if not self._m:
            return self._up_F(x)
        out = self._Rk.zero()
        for e, c in x.dict().items():
            e = (e,) if self._m == 1 else tuple(e)
            mon = self._xmon.get(e)
            if mon is None:
                mon = self._ring_one
                for v, k in zip(self._x, e):
                    mon *= v ** k
                self._xmon[e] = mon
            out += self._up_F(c) * mon
        return out

    def _up_F(self, x):
        """An element of ``Rk`` (in the variables of ``A`` and the ``u_j``)
        mapped to ``x ∈ F`` by ``phi``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._up_F(c/(5*c + 22)), KL._up_F(K(3/7))
            (1/5*c*u0, 3/7)
        """
        num, den = x.numerator(), x.denominator()
        if den.is_constant():
            if self._factors or self._m:
                num = self._emb(num)
            return num / den.constant_coefficient()
        if not self._factors:
            raise ValueError('denominator {} is not a constant'.format(den))
        u = self._ring_one
        for j, p in enumerate(self._factors):
            while True:
                q, r = den.quo_rem(p)
                if r:
                    break
                den = q
                u *= self._u[j]
        if not den.is_constant():
            raise ValueError('denominator {} has a factor outside {}'
                             .format(x.denominator(), self._factors))
        return self._emb(num) * u / den.constant_coefficient()

    def _power(self, j, k):
        """
        ``p_j^k`` (cached).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._power(0, 2) == KL.denominator_factors()[0]^2
            True
        """
        key = (j, k)
        r = self._powers.get(key)
        if r is None:
            r = self._powers[key] = self._factors[j] ** k
        return r

    def _down(self, v):
        """``phi(v)`` (in the base ring) for ``v ∈ Rk``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._down(KL._up(c/(5*c + 22)^2)) == c/(5*c + 22)^2
            True
            sage: KL._down(KL.ring().gen(1))
            1/(c + 22/5)
        """
        if not self._m:
            if not self._factors:
                F = self._F
                if self._elt is not None:
                    return self._elt(F, v, self._A.one(), coerce=False, reduce=False)
                return F(v)
            return self._down_F([(tuple(e), x) for e, x in v.dict().items()])
        n, m = self._n, self._m
        parts = {}
        for e, x in v.dict().items():
            e = tuple(e)
            parts.setdefault(e[n:n + m], []).append((e[:n] + e[n + m:], x))
        B = self._B
        d = {}
        for ex, items in parts.items():
            c = self._down_F(items)
            if c:
                d[ex[0] if m == 1 else ex] = c
        return B(d)

    def _down_F(self, items):
        """``phi`` of ``Σ x * (A-monomial e[:n]) * (u-monomial e[n:])`` over
        the ``(e, x)`` of ``items``, in ``F``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._down_F([((1, 1), 3)])                 # 3 c u0
            3*c/(c + 22/5)
            sage: KL._down_F([]), KL._down_F([((1, 1), 0)])
            (0, 0)
        """
        F, n, A = self._F, self._n, self._A
        if not self._factors:
            a = A({e[0] if n == 1 else e[:n]: x for e, x in items})
            if self._elt is not None:
                return self._elt(F, a, A.one(), coerce=False, reduce=False)
            return F(a)
        groups = {}
        for e, x in items:
            if not x:
                continue
            g = groups.setdefault(e[n:], {})
            g[e[0] if n == 1 else e[:n]] = x
        if not groups:
            return F.zero()
        top = [max(eu[j] for eu in groups) for j in range(len(self._factors))]
        num = A.zero()
        for eu, g in groups.items():
            a = A(g)
            for j, k in enumerate(top):
                if k > eu[j]:
                    a *= self._power(j, k - eu[j])
            num += a
        den = A.one()
        for j, k in enumerate(top):
            if k:
                den *= self._power(j, k)
        if self._elt is not None:
            return self._elt(F, num, den, coerce=False, reduce=True)
        return F(num) / F(den)

    def _down_dict(self, d):
        """
        ``phi`` on the values of a dict ``{key: element of Rk}``, zeros dropped.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._down_dict({((0, 0),): KL._up(c/(5*c + 22))})
            {((0, 0),): 1/5*c/(c + 22/5)}
        """
        out = {}
        for m, v in d.items():
            x = self._down(v)
            if x:
                out[m] = x
        return out

    def _down_poles(self, poles):
        """
        ``phi`` on a list of pole dicts, as a tuple without trailing zero poles.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
            sage: K.<c> = Frac(QQ['c'])
            sage: KL = _lifted_kernel(va.W3(K, c))
            sage: KL._down_poles([{((0, 0),): KL._up(c)}, {}])
            ({((0, 0),): c},)
        """
        out = [self._down_dict(d) for d in poles]
        while out and not out[-1]:
            out.pop()
        return tuple(out)


def _lifted_kernel(V):
    """
    A ``_LiftedKernel`` for ``V`` if its base ring is ``Frac(A)``, or a
    polynomial ring ``Frac(A)[x...]`` (the bootstrap's algebras), with ``A ⊇
    QQ`` and every denominator of the declared OPE coefficients factors
    (``A`` a polynomial ring over a field, unless all denominators are
    constants); else ``None``.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.universal_vertex_algebra import _lifted_kernel
        sage: _lifted_kernel(va.Virasoro(QQ, 1/2)) is None
        True
        sage: K.<c> = Frac(QQ['c'])
        sage: _lifted_kernel(va.Virasoro(K, c)).ring()
        Univariate Polynomial Ring in c over Rational Field
        sage: _lifted_kernel(va.W3(K, c)).denominator_factors()
        [c + 22/5]
    """
    B = V.base_ring()
    try:
        X = None
        F = B
        if not B.is_field() and hasattr(B, 'gens') and B.base_ring() is not B:
            if B.base_ring().is_field() and B.base_ring().ring() is not B.base_ring():
                X, F = B, B.base_ring()
        A = F.ring()
        if (A is F or F is not A.fraction_field()
                or not A.has_coerce_map_from(QQ)):
            return None

        def coefficients(x):
            return x.coefficients() if X is not None else [x]
        factors = []
        for poles in V._opes.values():
            for el in poles:
                for y in el._monomial_coefficients.values():
                    for x in coefficients(y):
                        d = x.denominator()
                        if d.is_constant():
                            continue
                        for p, _ in d.factor():
                            if p.is_constant():
                                continue
                            p = p / p.lc()
                            if p not in factors:
                                factors.append(p)
        if factors and not A.base_ring().is_field():
            return None
        return _LiftedKernel(V, A, sorted(factors, key=str), X)
    except (AttributeError, TypeError, ValueError, ArithmeticError,
            NotImplementedError):
        return None


def _engine_kernel(V):
    r"""
    The C++ engine (``_universal_engine``) of ``V`` as a
    :class:`vertex_algebra._EngineKernel`, if the engine is available and
    not switched off (``implementation='python'``) and takes ``V``'s base
    ring; else ``None`` (the pure-Python recursion):

    - ``QQ``, ``GF(p)`` (``2 < p < 2^63``), absolute number fields ``K``,
      polynomial rings over ``QQ`` or ``K``: the engine computes over the
      base ring itself (:func:`vertex_algebra._engine_ring`);
    - ``Frac(A)`` with a ``_LiftedKernel`` over ``QQ`` or ``K``: over
      its ring ``A`` or ``A[u_0, ...]``, with its ``_up`` / ``_down``;
    - ``QQbar``, ``AA``: over a number field containing the declared OPE
      coefficients (:class:`vertex_algebra._AlgebraicCoefficients`; rebuilt
      over a larger field when an element needs one).

    Only the declared generator OPEs are passed; the engine derives the
    other orientation.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.universal_vertex_algebra import _engine_kernel
        sage: N.<L> = NormallyOrderedExpressions(QQ)
        sage: V = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2]})
        sage: L = V.gen('L')
        sage: E = _engine_kernel(V)       # None without the compiled engine
        sage: E is None or (E.kind(), E.ring()) == ('q', QQ)
        True
        sage: E is None or E.ope_mono(((0, 0),), ((0, 0),)) == (
        ....:     {((0, 1),): 1}, {((0, 0),): 2}, {}, {(): 1/2})
        True
        sage: E is None or E.ope({((0, 0),): 2}, {((0, 0), (0, 0)): 1}, 1) == [
        ....:     {((0, 3),): 1/3, ((0, 1), (0, 0)): 4}, {((0, 0), (0, 0)): 8},
        ....:     {((0, 1),): 6}, {((0, 0),): 18}, {}, {(): 6}]
        True
        sage: (2*L).ope(L.no(L))
        [4*:dL L: + 1/3*d^3L, 8*:L L:, 6*dL, 18*L, 0, 6]

    Over ``Frac(QQ[c])`` the engine computes over the lifted kernel's ring::

        sage: K.<c> = Frac(QQ['c'])
        sage: N.<L> = NormallyOrderedExpressions(K)
        sage: W = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2]})
        sage: E = _engine_kernel(W)
        sage: E is None or (E.kind(), E.ring()) == ('qpoly', QQ['c'])
        True

    Over a number field (FLINT ``nf_elem``), and over ``Frac(K[c])`` with
    a denominator (lifted to ``K[c, u0]``, ``u0 = 1/(c + s)``)::

        sage: K.<s> = QuadraticField(5)
        sage: N.<L> = NormallyOrderedExpressions(K)
        sage: V5 = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0,
        ....:                                                 (134 + 60*s)/2]})
        sage: L5 = V5.gen('L')
        sage: E = _engine_kernel(V5)
        sage: E is None or E.kind() == 'nf'
        True
        sage: L5.nproduct(L5.no(L5), 3)                    # (c + 8) L
        (60*s + 142)*L
        sage: F.<c> = Frac(K['c'])
        sage: VF = UniversalVertexAlgebra(F, ['J'], {('J', 'J'): [0, 1/(c + s)]})
        sage: J = VF.gen('J')
        sage: E = _engine_kernel(VF)
        sage: E is None or (E.kind(), E.ring().ngens()) == ('nfmpoly', 2)
        True
        sage: J.ope(J.no(J))
        [0, (2/(c + s))*J]

    TESTS:

    The engine against the pure-Python recursion over ring kinds the other
    examples do not compute in (``GF(p)`` with a 61-bit ``p``, ``QQ[a, b]``,
    a number field with a non-monic defining polynomial, a tower,
    ``QQbar``, polynomial rings over fraction fields, lifted with a
    ``u``-variable), on the N=1 algebra (an odd generator); then with 4
    threads (the same results, in the same order)::

        sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
        sage: def n1(R, cc, implementation):
        ....:     N = NormallyOrderedExpressions(R, ['L', 'G'], parities={'G': 1})
        ....:     L, G = N.gens()
        ....:     return UniversalVertexAlgebra(R, N, {('L', 'L'): [L.derivative(), 2*L, 0, R(cc)/2],
        ....:         ('L', 'G'): [G.derivative(), R(3)/2*G], ('G', 'G'): [2*L, 0, 2*R(cc)/3]},
        ....:         implementation=implementation)
        sage: def work(V):
        ....:     L, G = V.gens()
        ....:     X = L.no(L).no(G) + 3*G.derivative(2) + L.derivative().no(G.derivative())
        ....:     Y = G.no(L) - 5*L.derivative(3).no(L) + G.derivative()
        ....:     return [list(x.monomial_coefficients().items())
        ....:             for x in X.ope(Y) + [X.nproduct(Y, -3), X.derivative(4)]]
        sage: K.<s> = QuadraticField(2)
        sage: K3.<r> = NumberField(QQ['z']([-1, -1, 0, 3]))
        sage: M.<t> = K.extension(QQ['y']([1, 0, 1]))
        sage: Rab.<a, b> = QQ[]
        sage: Fxy.<x, y> = Frac(QQ['c'])[]
        sage: FKx.<z> = Frac(K['c'])[]
        sage: cases = [(GF(2^61 - 1), 12345), (Rab, b - 2*a), (K3, r^2/3 - 1), (M, t + s),
        ....:          (QQbar, QQbar(2).sqrt() + 1), (Fxy, x/(Fxy.base_ring().gen() + 1) + y),
        ....:          (FKx, z*s/(FKx.base_ring().gen() + s) - 1)]
        sage: kinds = []
        sage: for R, cc in cases:  # long time
        ....:     V, VP = n1(R, cc, None), n1(R, cc, 'python')
        ....:     kinds.append(None if V._engine() is None else V._engine().kind())
        ....:     res = work(V)
        ....:     assert sorted(map(sorted, res)) == sorted(map(sorted, work(VP))), R
        ....:     V.clear_caches()
        ....:     VertexAlgebraParallelism().set(2)
        ....:     assert work(V) == res, R
        ....:     VertexAlgebraParallelism().reset()
        sage: kinds == [None] * 7 or kinds == ['nmod', 'qmpoly', 'nf', 'nf', 'nf', 'qmpoly', 'nfmpoly']  # long time
        True

    Threads inside one deep OPE (work stealing) give the same result::

        sage: Rc.<c> = QQ[]
        sage: V = n1(Rc, c, None)
        sage: L, G = V.gens()
        sage: X = L.no(L).no(G.derivative()).no(L)
        sage: Y = L.derivative(2).no(G).no(L).no(L)
        sage: res = list(X.ope(Y)[0].monomial_coefficients().items())  # long time
        sage: V.clear_caches()
        sage: VertexAlgebraParallelism().set(2)
        sage: list(X.ope(Y)[0].monomial_coefficients().items()) == res  # long time
        True
        sage: VertexAlgebraParallelism().reset()

    Errors inside the recursion (``GF(5)`` needs ``1/5!`` here) reach the
    caller, with any number of threads, and the algebra stays usable::

        sage: for implementation in (None, 'python'):
        ....:     L, G = n1(GF(5), 3, implementation).gens()
        ....:     for n in (1, 2):
        ....:         VertexAlgebraParallelism().set(n)
        ....:         try:
        ....:             _ = L.no(L).no(L).ope(L.no(L).no(L))
        ....:         except ZeroDivisionError:
        ....:             print('ZeroDivisionError')
        ....:     VertexAlgebraParallelism().reset()
        ....:     print(L.ope(G.no(G)))
        ZeroDivisionError
        ZeroDivisionError
        [d^2L, 3*dL, -L, 0, 1]
        ZeroDivisionError
        ZeroDivisionError
        [d^2L, 3*dL, -L, 0, 1]

    Ctrl-C (here an alarm) interrupts a long product, serially, with
    threads and inside one deep OPE; afterwards the algebra computes as
    before::

        sage: VIR = {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, QQ(7)/12]}
        sage: V = UniversalVertexAlgebra(QQ, ['L'], VIR, degrees={'L': 2})
        sage: big = sum(V.monomial(k) for k in V.weight_space(13).basis_keys())
        sage: small = sum(V.monomial(k) for k in V.weight_space(7).basis_keys())
        sage: ref = small.ope(small.derivative())
        sage: for n, x in [(1, big), (2, big), (2, V.monomial(((0, 0),) * 7))]:  # long time
        ....:     V.clear_caches()
        ....:     VertexAlgebraParallelism().set(n)
        ....:     try:
        ....:         alarm(0.3)
        ....:         _ = x.ope(x)
        ....:     except KeyboardInterrupt:
        ....:         pass
        ....:     finally:
        ....:         cancel_alarm()
        ....:     assert small.ope(small.derivative()) == ref
        sage: VertexAlgebraParallelism().reset()

    Rings the engine does not take are left to pure Python (no engine over
    ``Frac(GF(1009)[t])``); over ``Frac(QQ[x])`` with constant OPE
    coefficients the engine computes over ``QQ[x]``; either way the
    default gives the results of ``implementation='python'``::

        sage: Fp.<t> = Frac(GF(1009)['t']); Fx = Frac(QQ['x'])
        sage: for R, cc in [(Fp, t), (Fx, Fx(3))]:
        ....:     V, VP = n1(R, cc, None), n1(R, cc, 'python')
        ....:     E = V._engine()
        ....:     ok = E is None if R is Fp else (E is None or E.kind() == 'qpoly')
        ....:     print(ok, sorted(map(sorted, work(V))) == sorted(map(sorted, work(VP))))
        True True
        True True
    """
    if _UVA_ENGINE is None or V._implementation == 'python':
        return None
    F = V.base_ring()
    try:
        kind = _engine_ring(F)
        alg = None if kind is not None else V._algebraic_engine_ring(
            [c for poles in V._opes.values() for el in poles
             for c in el._monomial_coefficients.values()])
        if kind is not None:
            K = _EngineKernel(F, kind, F)
            decl = {key: [dict(el._monomial_coefficients) for el in poles]
                    for key, poles in V._opes.items()}
        elif alg is not None:                 # QQbar / AA: in a number field
            kind, alg = alg
            K = _EngineKernel(F, kind, alg.K, up=alg.up, down=alg.down)
            K._alg = alg
            decl = {key: [{m: alg.up(c) for m, c in el._monomial_coefficients.items()}
                          for el in poles] for key, poles in V._opes.items()}
        else:
            lift = _lifted_kernel(V)
            if lift is None:
                return None
            kind = _engine_ring(lift.ring())
            if kind not in ('qpoly', 'qmpoly', 'nfmpoly'):
                return None
            K = _EngineKernel(F, kind, lift.ring(), up=lift._up, down=lift._down)
            K._lift_factors = tuple(lift.denominator_factors())
            decl = lift._decl
        n = len(V._gens)
        K.attach(_UVA_ENGINE.Engine(
            kind, n, [int(V._parities.get(i, 0)) % 2 for i in range(n)],
            K.nvars(), K.modulus(), K.minpoly()))
        for (i, j), poles in decl.items():
            K._E.set_ope(i, j, [[(m, K.coef_in(c)) for m, c in d.items() if c]
                                for d in poles])
        return K
    except (AttributeError, TypeError, ValueError, ArithmeticError,
            NotImplementedError):
        return None


def _canonical_grading(generators, horizontal_grading, group):
    """
    The grading ``{name: horizontal_grade}`` (missing names: grade 0) and its group
    in the form stored by universal vertex algebras: a tuple of ``(name,
    grade)`` in generator order, grades as elements of the grading group
    (see :func:`_grading_groups.infer_grading_group`), and that group.

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import _canonical_grading
        sage: _canonical_grading(('E', 'H', 'F'), {'E': 1, 'F': -1}, None)
        ((('E', 1), ('H', 0), ('F', -1)), Integer Ring)
        sage: _canonical_grading(('a',), {'a': (1, 2)}, None)
        ((('a', (1, 2)),), Ambient free module of rank 2 over the principal ideal domain Integer Ring)
    """
    if horizontal_grading is None:
        if group is not None:
            raise ValueError('grading_group given without a grading')
        return None, None
    from ._grading_groups import infer_grading_group
    spec = dict(horizontal_grading)
    for g in spec:
        if g not in generators:
            raise ValueError('grade given for unknown generator {!r}'.format(g))
    A = infer_grading_group(spec.values(), group)
    return (tuple((g, A.element(A.coords(spec[g]) if g in spec else A.zero))
                  for g in generators), A.group)


def _canonical_opes(E, opes, check=True, implementation=None):
    r"""
    The OPEs ``opes`` (a dict ``(a, b) -> poles``, or the tuple of its
    items; ``a``, ``b`` generator names or indices; entry ``k`` of
    ``poles`` the coefficient of ``(z - w)^(-k-1)``, anything
    :func:`_expression_of` takes) in the canonical form of the data of a
    universal vertex algebra on the generators of ``E``: a tuple of
    ``((a, b), poles)`` for the pairs ``a <= b`` in the generator order
    with a nonzero OPE, ``poles`` a tuple of expressions in standard words
    (basis monomials) without trailing zeros.

    Pairs may be given in either orientation (the other follows by
    skew-symmetry), not both.  Words that are not basis monomials
    (letters out of order, repeated odd letters) and pairs ``b > a`` are
    normally ordered with the OPEs themselves (a private algebra
    evaluating them on demand); this terminates when the OPEs are
    compatible with positive conformal weights.  With ``check``, every
    term must have the parity ``|a| + |b|``.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import _canonical_opes
        sage: E.<L, W> = NormallyOrderedExpressions(QQ)
        sage: _canonical_opes(E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2, 0],
        ....:                     ('W', 'L'): [2*W.derivative(), 3*W], ('W', 'W'): []})
        ((('L', 'L'), (dL, 2*L, 0, 1/2)), (('L', 'W'), (dW, 3*W)))
        sage: Lam = L*L - 3/10*L.derivative(2)
        sage: o = _canonical_opes(E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2],
        ....:                         ('W', 'W'): [Lam.derivative()]})
        sage: o[1]
        (('W', 'W'), (2*:dL L: - 2/15*d^3L,))
        sage: _canonical_opes(E, {('L', 'W'): [W], ('W', 'L'): [W]})
        Traceback (most recent call last):
        ...
        ValueError: the OPE of L and W is given in both orders; give one (the other follows by skew-symmetry)
        sage: F.<b, c> = NormallyOrderedExpressions(QQ, parities={'c': 1})
        sage: _canonical_opes(F, {('b', 'c'): [b]})
        Traceback (most recent call last):
        ...
        ValueError: the OPE of b and c: pole 1 contains b of parity 0, expected 1
        sage: _canonical_opes(F, {('b', 'c'): [b]}, check=False)
        ((('b', 'c'), (b,)),)

    A coefficient that needs its own OPE to be normally ordered::

        sage: G.<a> = NormallyOrderedExpressions(QQ)
        sage: _canonical_opes(G, {('a', 'a'): [a*a.derivative()]})
        Traceback (most recent call last):
        ...
        ValueError: cannot normally order the OPE coefficients: the OPE of a and a is needed to normally order itself (are the OPEs compatible with positive conformal weights?); give its coefficients as basis monomials (standard words)

    TESTS:

    A pair given twice (by name and by index) is refused::

        sage: _canonical_opes(E, {('L', 'W'): [W], (0, 1): [W]})
        Traceback (most recent call last):
        ...
        ValueError: the OPE of L and W is given twice
    """
    gens = E._gens
    idx = E._idx
    odd = {i for i, p in E._parities.items() if p}

    def index(a):
        if isinstance(a, str) and a in idx:
            return idx[a]
        if not isinstance(a, str) and _is_int_like(a) and 0 <= int(a) < len(gens):
            return int(a)
        raise ValueError('unknown generator {!r} in the OPEs'.format(a))

    given = {}
    for pair, poles in (opes.items() if isinstance(opes, dict) else opes):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError('OPEs are keyed by pairs of generators, got {!r}'
                             .format(pair))
        key = (index(pair[0]), index(pair[1]))
        if key in given:
            raise ValueError('the OPE of {} and {} is given twice'.format(
                gens[key[0]], gens[key[1]]))
        if isinstance(poles, (str, dict)) or not hasattr(poles, '__iter__'):
            raise ValueError('the OPE of {} and {} must be a list of poles, got '
                             '{!r}'.format(gens[key[0]], gens[key[1]], poles))
        given[key] = [_expression_of(E, p) for p in poles]
    for i, j in given:
        if i < j and (j, i) in given:
            raise ValueError('the OPE of {} and {} is given in both orders; give '
                             'one (the other follows by skew-symmetry)'.format(
                                 gens[i], gens[j]))
    if all(i <= j for i, j in given) and all(
            _is_standard_word(w, odd) for poles in given.values()
            for p in poles for w in p.support()):
        result = given
    else:
        result = _normal_order_opes(E, given, implementation)
    out = []
    for i, j in sorted(result):
        poles = list(result[(i, j)])
        while poles and poles[-1].is_zero():
            poles.pop()
        if not poles:
            continue
        if check:
            want = (E._parities.get(i, 0) + E._parities.get(j, 0)) % 2
            for n, p in enumerate(poles):
                for w in p.support():
                    if E._parity_key(w) != want:
                        raise ValueError(
                            'the OPE of {} and {}: pole {} contains {} of parity '
                            '{}, expected {}'.format(gens[i], gens[j], n + 1,
                                                     E._repr_term(w),
                                                     E._parity_key(w), want))
        out.append(((gens[i], gens[j]), tuple(poles)))
    return tuple(out)


def _normal_order_opes(E, given, implementation=None):
    """
    The OPEs ``given`` (a dict ``(i, j) -> list of expressions``, at most
    one orientation per pair) as expressions in standard words, for the
    pairs ``i <= j``: evaluated in a private algebra (:class:`_StagingAlgebra`)
    whose generator OPEs are the given expressions, evaluated on demand
    (always the pure-Python recursion).

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import _normal_order_opes
        sage: E.<a> = NormallyOrderedExpressions(QQ)
        sage: _normal_order_opes(E, {(0, 0): [E.zero(), a*a]})
        {(0, 0): [0, :a a:]}
    """
    S = _StagingAlgebra(E.base_ring(), tuple(E._gens),
                        tuple(sorted((E._gens[i], p) for i, p in E._parities.items()
                                     if p)),
                        'python' if implementation == 'python' else None)
    S._expr = given
    out = {}
    for i, j in sorted({(min(i, j), max(i, j)) for i, j in given}):
        out[(i, j)] = [E._from_dict(dict(x._monomial_coefficients),
                                    remove_zeros=True)
                       for x in S._ope_gens(i, j)]
    return out


class _StandardMonomialKeys:
    """The canonically ordered monomial tuples of generator derivatives
    (weakly decreasing for ``_cmp_letter``, odd letters at most once).

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import _StandardMonomialKeys
        sage: S = _StandardMonomialKeys(['a', 'b'], odd=[1])
        sage: ((0, 1), (1, 0)) in S, ((1, 0), (1, 0)) in S
        (True, False)
    """
    def __init__(self, generators, odd=()):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import _StandardMonomialKeys
            sage: S = _StandardMonomialKeys(['a', 'b'], odd=[1])
            sage: S._generators, S._odd
            (('a', 'b'), frozenset({1}))
        """
        self._generators = tuple(generators)
        self._odd = frozenset(odd)

    def __contains__(self, x):
        """
        Whether ``x`` is a standard word in letters of these generators.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import _StandardMonomialKeys
            sage: S = _StandardMonomialKeys(['a', 'b'], odd=[1])
            sage: (), ((0, 0), (0, 0)) in S, ((0, 0), (0, 1)) in S
            ((), True, False)
            sage: ((2, 0),) in S, ((0, -1),) in S, 'x' in S
            (False, False, False)
        """
        if not isinstance(x, tuple):
            return False
        for t in x:
            if not _is_letter(t):
                return False
            if not (0 <= t[0] < len(self._generators) and t[1] >= 0):
                return False
        return _is_standard_word(x, self._odd)

    def an_element(self):
        """
        A key: the first generator.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import _StandardMonomialKeys
            sage: _StandardMonomialKeys(['a']).an_element()
            ((0, 0),)
        """
        return ((0, 0),)

    def cardinality(self):
        """
        ``+Infinity``.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import _StandardMonomialKeys
            sage: _StandardMonomialKeys(['a']).cardinality()
            +Infinity
        """
        return Infinity


class UniversalVertexAlgebra(VertexAlgebra):
    r"""
    The vertex algebra strongly generated by a list of fields with given
    OPEs, with the PBW-type basis of ordered normal ordered monomials.

    INPUT:

    - ``base_ring`` -- (default: ``None``) the coefficient ring (``None``:
      ``QQ``)
    - ``generators`` -- (default: ``None``) the generator names (their
      order is the generator order), or a
      :class:`normally_ordered_expressions.NormallyOrderedExpressions` (its
      names and parities); with ``V.<a, b> = ...`` the names may be given
      that way only
    - ``opes`` -- (default: ``None``) a dict ``(a, b) -> poles``, entry
      ``k`` of ``poles`` the coefficient of ``(z - w)^(-k-1)``: a normally
      ordered expression, a scalar, a generator name, a dict ``{word:
      coefficient}`` or an element of a universal vertex algebra (read by
      generator names); one orientation per pair (``None``: all OPEs zero)
    - ``parities`` -- (default: ``None``) dict ``name -> 0 or 1``; omitted
      names are even
    - ``degrees`` -- (default: ``None``) dict ``name -> weight`` (conformal
      weights without a Virasoro vector, e.g. for :mod:`bootstrap`)
    - ``conformal_vector`` -- (default: ``None``) a Virasoro vector (an
      expression, an element or a generator name), see
      :meth:`vertex_algebra.VertexAlgebra.with_conformal_vector` for the
      checks
    - ``horizontal_grading``, ``horizontal_grading_group`` -- (default:
      ``None``) a grading ``{name: horizontal_grade}`` by an abelian group,
      see :meth:`vertex_algebra.VertexAlgebra.with_horizontal_grading`
    - ``implementation`` -- (default: ``None``) ``None`` (the C++ engine
      when it takes the base ring, else pure Python), ``'engine'`` or
      ``'python'``; a ``ValueError`` if the engine is asked for and not
      available for the base ring.  Results do not depend on it.
    - ``check`` -- (default: ``True``) whether to check that each OPE
      coefficient of ``a`` and ``b`` has the parity ``|a| + |b|``
    - ``verify_jacobi`` -- (default: ``False``) whether to refuse OPEs
      failing the Jacobi identities (see :meth:`jacobi_identity_failures`)
    - ``names`` -- (default: ``None``) the names given by ``V.<a, b> =
      ...``: the generator names when ``generators`` is ``None``;
      otherwise only the Python variables bound to the generators, which
      keep their own names (a generator named for typesetting, typed under
      a short name)
    - ``proto`` -- (default: ``False``) whether the algebra is only a proto
      vertex algebra (no Jacobi identity claimed; the category
      ``ProtoVertexAlgebras``)

    EXAMPLES:

    The free boson -- one bosonic field ``J`` with ``J(z) J(w) ~ 1/(z-w)^2``::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: va = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: J = va.gen('J')
        sage: va.one()
        1
        sage: va.zero()
        0
        sage: J * 2
        2*J
        sage: J.derivative()
        dJ
        sage: J.derivative(2)
        d^2J
        sage: J.no(J)
        :J J:
        sage: J.no(J)
        :J J:
        sage: J * 0
        0

    Generators named for typesetting, typed under short names (the names
    after ``V.<...>`` are only the Python variables)::

        sage: BG.<b, g> = UniversalVertexAlgebra(QQ, ['β', 'γ'], {('β', 'γ'): [-1]})
        sage: BG.variable_names(), b.ope(g), g.no(b)
        (('β', 'γ'), [-1], :β γ:)

    The OPEs of derivatives follow from the basic one::

        sage: J.ope(J)
        [0, 1]
        sage: J.derivative().ope(J)
        [0, 0, -2]
        sage: J.ope(J.derivative())
        [0, 0, 2]
        sage: J.derivative(2).ope(J)
        [0, 0, 0, 6]

    Normal ordering reorders letters into standard form::

        sage: J.no(J.derivative()) == J.derivative().no(J)
        True

    The Virasoro field ``Tb = 1/2 :J J:`` has the expected OPE::

        sage: Tb = (1/2) * J.no(J)
        sage: Tb.ope(Tb)
        [:dJ J:, :J J:, 0, 1/2]

    Negative ``n``-th products come from the base class::

        sage: J.nproduct(J, -2)
        :dJ J:

    The algebra cannot be changed (its OPEs and conformal data are
    construction data); ``with_opes`` and the ``with_...`` methods give
    other algebras::

        sage: va.with_opes({('J', 'J'): [0, 2]}).opes()
        {('J', 'J'): [0, 2]}

    TESTS::

        sage: va = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: va.one().derivative() == va.zero()
        True
        sage: va.one().is_one()
        True
        sage: J = va.gen('J')
        sage: J.derivative() * 3 == 3 * J.derivative()
        True
        sage: va(J) is J
        True
    """

    _is_universal_vertex_algebra = True     # duck typing (doctests)

    @staticmethod
    def __classcall_private__(cls, base_ring=None, generators=None, opes=None,
                              parities=None, degrees=None,
                              conformal_vector=None, horizontal_grading=None,
                              horizontal_grading_group=None,
                              implementation=None, check=True,
                              verify_jacobi=False, names=None, proto=False):
        r"""
        Normalize the construction arguments (unique representation).

        The OPEs are stored canonically: for the pairs ``a <= b`` in the
        generator order, as expressions in basis monomials (normally
        ordered as in the algebra), zero OPEs and trailing zero poles
        dropped.  Equal data give the same algebra, however the OPEs were
        written; another ``implementation`` gives another algebra (with the
        same products).  ``check`` (coefficient parities) and
        ``verify_jacobi`` are not part of the data.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: A = UniversalVertexAlgebra(QQ, ['J'])
            sage: A is UniversalVertexAlgebra(QQ, ['J'])
            True
            sage: B = UniversalVertexAlgebra(QQ, ['J'], implementation='python'); B
            Universal vertex algebra on generators ['J'] over Rational Field
            sage: B is A, B is UniversalVertexAlgebra(QQ, ['J'], implementation='python')
            (False, True)
            sage: UniversalVertexAlgebra(QQ, ['J'], implementation='fast')
            Traceback (most recent call last):
            ...
            ValueError: implementation must be None, 'engine' or 'python' (got 'fast')
            sage: UniversalVertexAlgebra(Frac(GF(5)['t']), ['J'], implementation='engine')
            Traceback (most recent call last):
            ...
            ValueError: implementation='engine': the C++ engine is not available for ... (not compiled, or the base ring is not supported)

        The same OPEs written in other orientations or orders give the
        same algebra::

            sage: E.<b, c> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'c': 1})
            sage: V1 = UniversalVertexAlgebra(QQ, E, {('b', 'c'): [1]})
            sage: V2 = UniversalVertexAlgebra(QQ, ['b', 'c'], {('c', 'b'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: V1 is V2, V1.opes()
            (True, {('b', 'c'): [1]})

        Unpickling calls this method again with the normalized arguments,
        so they must normalize to themselves (``implementation`` and
        ``proto`` are passed by keyword; the doctester runs this file as
        ``__main__``, whose classes cannot be pickled, hence the import)::

            sage: from vertex_algebras import universal_vertex_algebra as uva
            sage: B = uva.UniversalVertexAlgebra(QQ, ['J'], implementation='python')
            sage: C = uva.UniversalVertexAlgebra(QQ, ['b', 'c'], parities={'c': 1})
            sage: loads(dumps(B)) is B, loads(dumps(C)) is C
            (True, True)
            sage: K.<k> = Frac(QQ['k'])
            sage: S = uva.UniversalVertexAlgebra(K, ['E', 'H', 'F'],
            ....:     {('H', 'E'): [{(('E', 0),): 2}],
            ....:      ('H', 'F'): [{(('F', 0),): -2}], ('E', 'F'): ['H', k],
            ....:      ('H', 'H'): [0, 2*k]}, degrees={'E': 1, 'H': 1, 'F': 1},
            ....:     horizontal_grading={'E': 1, 'F': -1})
            sage: loads(dumps(S)) is S
            True
            sage: S.category()
            Category of finitely strongly generated conformally graded vertex algebras with basis over Fraction Field of Univariate Polynomial Ring in k over Rational Field
            sage: TestSuite(S).run(); TestSuite(C).run()
            sage: S.opes()[('E', 'H')]
            [-2*E]

        TESTS:

        ``W_3`` written with ``∂Λ`` out of PBW order and the OPE of ``W``
        and ``L`` in the other orientation is the algebra whose ``WW``
        coefficients were computed in the Virasoro algebra (two steps)::

            sage: K.<c> = Frac(QQ['c']); beta = 16/(22 + 5*c)
            sage: Vir = {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, c/2],
            ....:        ('L', 'W'): [{(('W', 1),): 1}, {(('W', 0),): 3}]}
            sage: V0 = UniversalVertexAlgebra(K, ['L', 'W'], Vir); L, W = V0.gens()
            sage: Lam = L.no(L) - 3/10*L.derivative(2)
            sage: old = V0.with_opes(Vir | {('W', 'W'): [L.derivative(3)/15 + beta*Lam.derivative(),
            ....:     3/10*L.derivative(2) + 2*beta*Lam, L.derivative(), 2*L, 0, c/3]})
            sage: E.<L, W> = NormallyOrderedExpressions(K)
            sage: Lam = L*L - 3/10*L.derivative(2)
            sage: new = UniversalVertexAlgebra(K, E, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:     ('W', 'L'): [2*W.derivative(), 3*W],
            ....:     ('W', 'W'): [L.derivative(3)/15 + beta*Lam.derivative(),
            ....:                  3/10*L.derivative(2) + 2*beta*Lam, L.derivative(), 2*L, 0, c/3]})
            sage: new is old
            True

        With the conformal vector ``L``, the canonical data (OPEs and
        conformal vector) give the same algebra again, and so does a
        pickle::

            sage: from vertex_algebras import universal_vertex_algebra as uva
            sage: T3 = new.with_conformal_vector(L); d = T3._data
            sage: uva.UniversalVertexAlgebra(K, ['L', 'W'], d['opes'], conformal_vector=d['T']) is T3
            True
            sage: loads(dumps(T3)) is T3, T3.central_charge()
            (True, c)

        The same for ``N = 1`` (``G(z) L(w)``) and for affine ``sl_2`` with
        the Sugawara vector written with ``F*E`` (central charge
        ``3k/(k+2)``)::

            sage: E.<L, G> = NormallyOrderedExpressions(K, parities={'G': 1})
            sage: UniversalVertexAlgebra(K, E, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:     ('G', 'L'): [G.derivative()/2, 3/2*G], ('G', 'G'): [2*L, 0, 2*c/3]}) is \
            ....: UniversalVertexAlgebra(K, ['L', 'G'], {('L', 'L'): [{(('L', 1),): 1},
            ....:     {(('L', 0),): 2}, 0, c/2], ('L', 'G'): [{(('G', 1),): 1}, {(('G', 0),): 3/2}],
            ....:     ('G', 'G'): [{(('L', 0),): 2}, 0, 2*c/3]}, parities={'G': 1})
            True
            sage: N.<E, H, F> = NormallyOrderedExpressions(K)
            sage: A = UniversalVertexAlgebra(K, N, {('H', 'H'): [0, 2*c], ('H', 'E'): [2*E],
            ....:     ('H', 'F'): [-2*F], ('E', 'F'): [H, c]},
            ....:     conformal_vector=(H*H/2 + E*F + F*E)/(2*(c + 2)))
            sage: B = UniversalVertexAlgebra(K, ['E', 'H', 'F'], {('H', 'H'): [0, 2*c],
            ....:     ('E', 'H'): [{(('E', 0),): -2}], ('H', 'F'): [{(('F', 0),): -2}],
            ....:     ('E', 'F'): ['H', c]})
            sage: e, h, f = B.gens()
            sage: A is B.with_conformal_vector((h.no(h)/2 + e.no(f) + f.no(e))/(2*(c + 2)))
            True
            sage: A.central_charge()
            3*c/(c + 2)

        Affine ``osp(1|2)`` written by hand has the canonical OPE data of
        the affine constructor::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: osp = {('h', 'e'): {'e': 2}, ('h', 'f'): {'f': -2}, ('e', 'f'): {'h': 1},
            ....:        ('h', 'x'): {'x': 1}, ('h', 'y'): {'y': -1}, ('e', 'y'): {'x': -1},
            ....:        ('f', 'x'): {'y': -1}, ('x', 'x'): {'e': 2}, ('x', 'y'): {'h': 1},
            ....:        ('y', 'y'): {'f': -2}}
            sage: names = ['e', 'h', 'f', 'x', 'y']
            sage: A = UniversalAffineVertexAlgebra(QQ, osp, {('e', 'f'): 1, ('h', 'h'): 2,
            ....:     ('x', 'y'): 2}, level=1, parities={'x': 1, 'y': 1}, names=names)
            sage: O.<e, h, f, x, y> = NormallyOrderedExpressions(QQ, parities={'x': 1, 'y': 1})
            sage: V = UniversalVertexAlgebra(QQ, O, {('h', 'e'): [2*e], ('h', 'f'): [-2*f],
            ....:     ('e', 'f'): [h, 1], ('h', 'h'): [0, 2], ('h', 'x'): [x], ('h', 'y'): [-y],
            ....:     ('e', 'y'): [-x], ('f', 'x'): [-y], ('x', 'x'): [2*e], ('x', 'y'): [h, 2],
            ....:     ('y', 'y'): [-2*f]}, verify_jacobi=True)
            sage: V._data['opes'] == A._data['opes']
            True
        """
        if getattr(generators, '_is_normally_ordered_expressions', False):
            E = generators
            if base_ring is None:
                base_ring = E.base_ring()
            elif base_ring is not E.base_ring():
                raise ValueError('the expressions {} are over another ring than '
                                 '{}'.format(E, base_ring))
            given = E.parities()
            if parities is not None and {g: p for g, p in dict(parities).items() if p} \
                    != {g: p for g, p in given.items() if p}:
                raise ValueError('the parities {} differ from those of {}'.format(
                    dict(parities), E))
            generators, parities = E.variable_names(), given
        if base_ring is None:
            base_ring = QQ
        generators, parity_items = _normalize_generators(
            generators, parities, names, 'UniversalVertexAlgebra')
        E = NormallyOrderedExpressions(base_ring, generators, dict(parity_items))
        implementation = _normalize_implementation(implementation)
        opes = _canonical_opes(E, opes or {}, check, implementation)
        weights = None
        if degrees is not None:
            weights = dict(degrees)
            if set(weights) != set(generators):
                raise ValueError('conformal weights must be given for exactly {} '
                                 '(got {})'.format(list(generators),
                                                   sorted(weights, key=str)))
            weights = tuple((g, base_ring(weights[g])) for g in generators)
        horizontal_grading, group = _canonical_grading(generators, horizontal_grading, horizontal_grading_group)
        T = None
        if conformal_vector is not None:
            # evaluated in the algebra without the conformal data
            V0 = VertexAlgebra.__classcall__(
                UniversalVertexAlgebra, base_ring, generators, opes,
                parity_items, None, None, None, None,
                implementation=implementation, proto=bool(proto))
            T = _expression_of(E, V0._evaluate(_expression_of(E, conformal_vector)))
        V = VertexAlgebra.__classcall__(
            cls, base_ring, generators, opes, parity_items, weights, T,
            horizontal_grading, group, implementation=implementation, proto=bool(proto))
        if verify_jacobi:
            V._verify_jacobi()
        return V

    def __init__(self, base_ring, generators, opes, parity_items,
                 degrees=None, conformal_vector=None, horizontal_grading=None,
                 horizontal_grading_group=None, implementation=None, proto=False,
                 _checked_virasoro=None):
        r"""
        Initialize ``self``.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: UniversalVertexAlgebra(QQ, ['J'])
            Universal vertex algebra on generators ['J'] over Rational Field

            sage: UniversalVertexAlgebra(QQ, ['b', 'c'], parities={'c': 1})
            Universal vertex algebra on generators ['b', 'c'] over Rational Field

        Parallelism is a setting of the session,
        :class:`vertex_algebra.VertexAlgebraParallelism`: over the rings of
        the C++ engine (``QQ``, ``GF(p)``, number fields, polynomial rings
        over them, their fraction fields) the number of threads of each
        product; elsewhere, worker processes.  Results do not depend on it::

            sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
            sage: va = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: J = va.gen('J'); x = J.no(J.no(J))
            sage: serial = x.ope(x)
            sage: VertexAlgebraParallelism().set(2); va.clear_caches()
            sage: x.ope(x) == serial
            True
            sage: VertexAlgebraParallelism().reset()

        Conformal weights, a grading and a Virasoro vector are part of the
        data (checked against the OPEs)::

            sage: E.<b, g> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, E, {('b', 'g'): [-1]},
            ....:                            conformal_vector=-b*g.derivative(),
            ....:                            horizontal_grading={'b': 1, 'g': -1})
            sage: V.central_charge(), V.degrees(), V.conformal_vector()
            (2, {'b': 1, 'g': 0}, -:dg b:)
            sage: b, g = V.gens()
            sage: b.no(b).horizontal_grade(), b.no(g).degree()
            (2, 1)
            sage: UniversalVertexAlgebra(QQ, E, {('b', 'g'): [-1]},
            ....:                        degrees={'b': 1, 'g': 1})
            Traceback (most recent call last):
            ...
            ValueError: the conformal weights {'b': 1, 'g': 1} are not compatible with the OPE of b and g: pole 1 contains 1 of weight 0, expected 1
        """
        self._gens = tuple(generators)
        self._implementation = implementation
        self._idx = {name: k for k, name in enumerate(self._gens)}
        self._names = dict(enumerate(self._gens))
        self._parities = {self._idx[n]: p for n, p in dict(parity_items).items()}
        self._opes = {}
        odd = [k for k, p in self._parities.items() if p]
        VertexAlgebra.__init__(
            self, base_ring, _StandardMonomialKeys(self._gens, odd),
            category=self._category_for(
                base_ring, bool(odd),
                degrees is not None or conformal_vector is not None,
                conformal_vector is not None, fsg=True, proto=proto))
        self._data = {'opes': tuple(opes), 'weights': degrees,
                      'T': conformal_vector, 'horizontal_grading': horizontal_grading,
                      'group': horizontal_grading_group,
                      'implementation': implementation, 'proto': bool(proto)}
        self._opes = {(self._idx[a], self._idx[b]):
                      tuple(self._from_standard(p) for p in poles)
                      for (a, b), poles in opes}
        self._check_implementation()
        if degrees is not None:
            self._install_conformal_weights(dict(degrees))
        if horizontal_grading is not None:
            self._install_grading(dict(horizontal_grading), horizontal_grading_group)
        if conformal_vector is not None:
            T = self._from_standard(conformal_vector)
            if _checked_virasoro is None:
                self._install_virasoro(T)
            else:       # checked by the caller (the affine constructor)
                c, data, chains = _checked_virasoro
                self._conformal = {'T': T, 'c': c, 'data': data,
                                   'jordan': chains}

    # ------------------------------------------------------------------ #
    #  Basics                                                             #
    # ------------------------------------------------------------------ #

    def _repr_(self):
        """
        The repr: generators and base ring.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: repr(V)
            "Universal vertex algebra on generators ['L'] over Rational Field"
        """
        return 'Universal vertex algebra on generators {} over {}'.format(
            list(self._gens), self.base_ring())

    def _vacuum_key(self):
        """
        The vacuum key, the empty word ``()``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._vacuum_key(), V.monomial(V._vacuum_key()) == V.one()
            ((), True)
        """
        return ()

    def _parity_key(self, m):
        """
        The parity of a basis key: the sum of its letters' parities mod 2.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: bc = va.FermionicGhosts(QQ)
            sage: bc._parity_key(((0, 0),)), bc._parity_key(((0, 1), (1, 0))), bc._parity_key(())
            (1, 0, 0)
        """
        return sum(self._parities.get(x[0], 0) for x in m) % 2

    def _term_str(self, mono, style='ascii'):
        r"""
        The notation for a basis monomial (``style``: ``'ascii'``,
        ``'unicode'`` or ``'latex'``).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: va = UniversalVertexAlgebra(QQ, ['J'])
            sage: va._repr_term(((0, 2),)), va._repr_term(((0, 10),)), va._repr_term(((0, 1), (0, 0)))
            ('d^2J', 'd^10J', ':dJ J:')
            sage: va._term_str(((0, 10),), 'unicode'), va._term_str(((0, 1), (0, 0)), 'unicode')
            ('∂¹⁰ J', ':∂ J J:')
        """
        return _word_str([_derivative_str(self._names[i], n, style) for (i, n) in mono],
                         style)

    def _mode_word(self, m, data):
        r"""
        The key ``m`` as a mode word on the vacuum (see
        :meth:`VertexAlgebra._mode_word`): the letter `\partial^n a` is
        `n!\, a_{(-n-1)}` (`(\partial^n a)_{(-1)} = n!\, a_{(-n-1)}`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V._mode_word(((0, 2), (0, 1)), None)
            (2, [('L', -3, None), ('L', -2, None)], None)
            sage: V._mode_word(((0, 0),), V._weight_data())
            (1, [('L', -1, 2)], None)
        """
        f = 1
        letters = []
        for i, n in m:
            f *= factorial(n)
            letters.append((self._names[i], -n - 1,
                            None if data is None else data['h'][i]))
        return f, letters, None

    def _state_fingerprint(self):
        """
        The data saved states are checked against: ring, generators,
        parities and the declared OPEs as raw dicts.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: f = V._state_fingerprint(); f[:4]
            ('UniversalVertexAlgebra', Rational Field, ('L',), ())
            sage: f[4]
            {(0, 0): ({((0, 1),): 1}, {((0, 0),): 2}, {}, {(): 1/4})}
        """
        opes = {k: tuple(dict(p._monomial_coefficients) for p in poles)
                for k, poles in self._opes.items()}
        return ('UniversalVertexAlgebra', self.base_ring(), tuple(self._gens),
                tuple(sorted(self._parities.items())), opes)

    def variable_names(self):
        """
        The list of generator names.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: UniversalVertexAlgebra(QQ, ['a', 'b']).variable_names()
            ('a', 'b')
        """
        return tuple(self._gens)

    def gen(self, name_or_index):
        """
        The generator with the given name or index.  Later generators are
        larger in the standard order.

        INPUT:

        - ``name_or_index`` -- a generator name or its index (0-based, in
          declaration order)

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['a', 'b'])
            sage: V.gen('a') == V.gen(0) and V.gen(0) == V.gen(0)
            True
            sage: V.gens()
            (a, b)
        """
        if _is_int_like(name_or_index):
            k = int(name_or_index)
        else:
            k = self._idx[name_or_index]
        return self.monomial(((k, 0),))

    def _first_ngens(self, n=0):
        r"""
        The first ``n`` generators, enabling ``V.<a, b> = ...``.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V.<a, b> = UniversalVertexAlgebra(QQ, ['a', 'b'])
            sage: V
            Universal vertex algebra on generators ['a', 'b'] over Rational Field
            sage: a.derivative()
            da
            sage: a.no(b)
            :a b:
            sage: V.gen(0) == a and V.gen(1) == b
            True

        If ``names=`` is the only source of generator names they are used
        directly::

            sage: W.<x, y> = UniversalVertexAlgebra(QQ)
            sage: W
            Universal vertex algebra on generators ['x', 'y'] over Rational Field
            sage: x.no(y)
            :x y:
        """
        if n > len(self._gens):
            raise ValueError('requested {} generators but only {} are defined'
                             .format(n, len(self._gens)))
        return tuple(self.gen(self._names[k]) for k in range(n))

    def parities(self):
        """
        A dict mapping generator names to their parity (0 even, 1 odd).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
            ....:                             parities={'b': 1, 'c': 1})
            sage: V.parities()
            {'b': 1, 'c': 1}
            sage: UniversalVertexAlgebra(QQ, ['J']).parities()
            {'J': 0}
        """
        return {self._names[k]: self._parities.get(k, 0)
                for k in range(len(self._gens))}

    def _element_constructor_(self, x):
        r"""
        Convert a scalar, an element (of this algebra, or of the same
        algebra with other conformal data or grading, see
        :meth:`with_conformal_vector`), a basis key, a word (a tuple of
        ``(name, derivative)`` letters in any order: the right-nested
        normally ordered product) or a normally ordered expression (see
        :mod:`normally_ordered_expressions`; fields matched by name).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]})
            sage: V((('L', 1), ('L', 0))), V(('L', 2)), V(3)
            (:dL L:, d^2L, 3)
            sage: V((('L', 0), ('L', 1))) == V(L*L.derivative()) == V.gen(0).no(V.gen(0).derivative())
            True
            sage: V(L*L.derivative())
            :dL L: + 1/6*d^3L
        """
        if getattr(_parent(x), '_is_normally_ordered_expressions', False):
            return self._evaluate(x)
        if isinstance(x, tuple) and x not in self._indices:
            return self._evaluate(self.normally_ordered_expressions()(x))
        return VertexAlgebra._element_constructor_(self, x)

    def _structure_key(self):
        """
        The construction data without the conformal ones (forgetful
        coercions between algebras with equal structure keys).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: T = V.with_conformal_vector('L')
            sage: V._structure_key() == T._structure_key()
            True
            sage: V._structure_key()[:3]
            ('UniversalVertexAlgebra', Rational Field, ('L',))
        """
        d = self._data
        return ('UniversalVertexAlgebra', self.base_ring(), self._gens,
                tuple(sorted(self._parities.items())), d['opes'],
                d['implementation'], d['proto'])

    def _conformal_key(self):
        """
        The conformal data: weights, conformal vector, grading.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: k = V._conformal_key()
            sage: k['weights'], k['T'], k['horizontal_grading']
            ((('L', 2),), None, None)
            sage: V.with_conformal_vector('L')._conformal_key()['T'] is None
            False
        """
        d = self._data
        return {'weights': d['weights'], 'T': d['T'],
                'horizontal_grading': None if d['horizontal_grading'] is None
                else (d['horizontal_grading'], d['group'])}

    def _from_standard(self, p):
        """The element whose basis monomials are the (standard) words of
        the expression ``p`` (the stored data are of this form).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: E = V.normally_ordered_expressions(); l = E.gen(0)
            sage: V._from_standard(3*l.derivative()*l)
            3*:dL L:
        """
        return self._from_dict(dict(p.monomial_coefficients()), coerce=False,
                               remove_zeros=True)

    def _evaluate(self, e):
        r"""
        The element denoted by a normally ordered expression ``e`` (anything
        :func:`_expression_of` takes): each word is the right-nested
        normally ordered product of its letters.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: x = V._evaluate({(('L', 0), ('L', 1)): 1}); x
            :dL L: + 1/6*d^3L
            sage: x == L.no(L.derivative())
            True
        """
        e = _expression_of(self.normally_ordered_expressions(), e)
        odd = {k for k, p in self._parities.items() if p}
        std = {}
        out = self.zero()
        for w, c in e.monomial_coefficients().items():
            if _is_standard_word(w, odd):
                std[w] = c
                continue
            acc = self.monomial(w[-1:])
            for t in reversed(w[:-1]):
                acc = self.monomial((t,)).no(acc)
            out += c * acc
        if std:
            out += self._from_dict(std, coerce=False, remove_zeros=True)
        return out

    def normally_ordered_expressions(self):
        r"""
        The normally ordered expressions in the generators (with their
        parities), in which OPE coefficients and conformal vectors can be
        written before the algebra exists; see
        :mod:`normally_ordered_expressions`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['b', 'c'], parities={'c': 1})
            sage: V.normally_ordered_expressions()
            Normally ordered expressions in the fields ['b', 'c'] over Rational Field
        """
        return NormallyOrderedExpressions(self.base_ring(), self._gens,
                                          self.parities())

    # ------------------------------------------------------------------ #
    #  New algebras from this one                                         #
    # ------------------------------------------------------------------ #

    def _with(self, **changes):
        """
        The universal vertex algebra with the same data except ``changes``
        (keywords of the constructor); the base of the ``with_*`` methods.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._with(conformal_vector='L').central_charge()
            1/2
            sage: V._with(opes={}).opes(), V._with(opes={}).degrees()
            ({}, {'L': 2})
        """
        d = self._data
        T = d['T']
        if T is None and self._conformal is not None:
            T = self._conformal['T']
        args = dict(opes=d['opes'], degrees=d['weights'],
                    conformal_vector=T, horizontal_grading=d['horizontal_grading'],
                    horizontal_grading_group=d['group'])
        args.update(changes)
        if args['horizontal_grading'] is None:
            args['horizontal_grading_group'] = None
        return UniversalVertexAlgebra(self.base_ring(), self._gens,
                                      parities=self.parities(),
                                      implementation=d['implementation'],
                                      proto=d['proto'], **args)

    def with_opes(self, opes, verify_jacobi=False):
        r"""
        The universal vertex algebra on the same generators with the OPEs
        ``opes``, the same conformal weights and grading (checked against
        the new OPEs), and no Virasoro vector.

        INPUT:

        - ``opes`` -- the OPEs, as for the constructor
        - ``verify_jacobi`` -- boolean (default: ``False``); whether to check
          the Jacobi identities (see :meth:`jacobi_identity_failures`)

        OUTPUT:

        A :class:`UniversalVertexAlgebra`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: H = V.with_opes({('J', 'J'): [0, 1]}); H.opes()
            {('J', 'J'): [0, 1]}
            sage: H.degrees(), V.opes()
            ({'J': 1}, {})

        TESTS:

        Other OPEs give another algebra with its own engine (no stale
        generator OPEs); the first algebra and its engine are untouched::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: def n1(cc):
            ....:     N = NormallyOrderedExpressions(QQ, ['L', 'G'], parities={'G': 1}); L, G = N.gens()
            ....:     return {('L', 'L'): [L.derivative(), 2*L, 0, cc/2],
            ....:             ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 2*cc/3]}
            sage: V = UniversalVertexAlgebra(QQ, ['L', 'G'], n1(1), parities={'G': 1})
            sage: G = V.gen('G'); E0 = V._engine(); before = G.ope(G)
            sage: V2 = V.with_opes(n1(5)); G2 = V2.gen('G')
            sage: V2 is not V, V._engine() is E0, E0 is None or V2._engine() is not E0
            (True, True, True)
            sage: G2.ope(G2), G.ope(G) == before, before
            ([2*L, 0, 10/3], True, [2*L, 0, 2/3])
        """
        return self._with(opes=opes, conformal_vector=None,
                          verify_jacobi=verify_jacobi)

    def _verify_jacobi(self):
        """Raise a ``ValueError`` naming a failing Jacobi identity (see
        :meth:`jacobi_identity_failures`); remembered when they hold.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): ['J', 1]}, check=False)
            sage: V._verify_jacobi()
            Traceback (most recent call last):
            ...
            ValueError: the OPEs violate skew-symmetry J_(0) J: 2*J (and at least 2 more)
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: H._verify_jacobi(); H._jacobi_verified
            True
        """
        if getattr(self, '_jacobi_verified', False):
            return
        bad = []
        for item in self._jacobi_failure_iter(None, True):
            bad.append(item)
            if len(bad) == 3:
                break
        if bad:
            label, el = bad[0]
            more = ''
            if len(bad) > 1:
                more = ' (and {}{} more)'.format(
                    'at least ' if len(bad) == 3 else '', len(bad) - 1)
            raise ValueError('the OPEs violate {}: {}{}'.format(
                self._jacobi_label_str(label), el, more))
        self._jacobi_verified = True

    # ------------------------------------------------------------------ #
    #  OPE specification and caches                                       #
    # ------------------------------------------------------------------ #

    def opes(self):
        """
        The stored generator OPEs, keyed by pairs of generator names: entry
        ``k`` of a list is ``a_(k) b``.  Only one orientation of a pair is
        stored, and pairs with zero OPE are omitted.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Virasoro(QQ, 1/2).opes()
            {('L', 'L'): [dL, 2*L, 0, 1/4]}
            sage: va.FermionicGhosts(QQ).opes()
            {('b', 'c'): [1]}
        """
        return {tuple(self._names[k] for k in key): list(poles)
                for key, poles in self._opes.items()}

    def clear_caches(self):
        """
        Forget all cached OPE, normal ordering and derivative results
        (results do not change: they are recomputed on demand).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2)
            sage: L = V.gen('L'); x = L.no(L).ope(L)
            sage: V.clear_caches()
            sage: L.no(L).ope(L) == x
            True
        """
        VertexAlgebra.clear_caches(self)
        self._gen_ope_cache = {}
        # (the base class drops the C++ engine)
        # element-valued caches of the pure-Python recursion
        self._fb_ope_cache = {}
        self._fb_no_cache = {}
        self._fb_deriv_cache = {}

    # ------------------------------------------------------------------ #
    #  Jacobi identities of the declared OPEs                             #
    # ------------------------------------------------------------------ #

    def induced_module(self, U, name=None, implementation=None, max_depth=None,
                       commutators=True):
        r"""
        The induced module ``M(U)``: Zhu's universal positive energy module
        [Zhu1996]_ with top level ``U``, a finite-dimensional module over the
        Zhu algebra (:meth:`~vertex_algebra_categories.VertexAlgebras.Graded.ParentMethods.zhu_algebra`,
        ``A.module(...)``); an :class:`vertex_algebra_module.InducedModule`.
        The generators need positive weights.

        INPUT:

        - ``U`` -- a finite-dimensional module over the Zhu algebra of
          ``self`` (or of a quotient of it)
        - ``name`` -- (default: ``None``) a name used in the repr
        - ``implementation`` -- (default: ``None``) ``None`` (the C++ engine
          of induced modules when available), ``'engine'`` or ``'python'``
        - ``max_depth`` -- (default: ``None``) for ``U`` over a level ``n``
          Zhu algebra (``self.zhu_algebra(level=n).module(...)``) the
          ``\bar M_n(U)`` of [DLM1998]_ with ``U`` in degree ``n``, computed
          in the degrees ``<= max_depth`` (a
          :class:`quotient_induced_module.InducedModuleUpTo` of the
          :class:`higher_zhu_algebra.HigherInducedModule`)
        - ``commutators`` -- (default: ``True``) for ``max_depth``: also
          impose the commutator formula of the generators

        OUTPUT:

        An :class:`vertex_algebra_module.InducedModule`, or with
        ``max_depth`` a :class:`quotient_induced_module.InducedModuleUpTo`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: M = H.induced_module(H.zhu_algebra().module({'J': 2})); J = H.gen('J')
            sage: J.mode(M.highest_weight_vector(), 0), M.character(5)
            (2*v, 1 + q + 2*q^2 + 3*q^3 + 5*q^4 + O(q^5))
        """
        if getattr(U, '_is_higher_zhu_algebra_module', False):
            from .quotient_induced_module import InducedModuleUpTo
            if max_depth is None:
                raise ValueError('a module induced from degree n is computed up to a '
                                 'degree: give max_depth')
            return InducedModuleUpTo(self, U, 0, max_depth, commutators, implementation)
        from .vertex_algebra_module import InducedModule
        return InducedModule(self, U, name, implementation)

    def verma_module(self, weights):
        r"""
        The Verma module: the induced module of the one-dimensional module
        of the Zhu algebra with the given values of its generators, see
        :meth:`induced_module`.

        INPUT:

        - ``weights`` -- the values of the generators of the Zhu algebra on
          the top vector: a dict by name, or a scalar if there is one
          generator

        OUTPUT:

        An :class:`vertex_algebra_module.InducedModule`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: M = Vir.verma_module(1/16); M.top_weight(), M.character(4)
            (1/16, q^(1/16) + q^(17/16) + 2*q^(33/16) + 3*q^(49/16) + O(q^4))
        """
        A = self.zhu_algebra()
        if not isinstance(weights, dict):
            if A.ngens() != 1:
                raise ValueError('give the values of the generators {} of the Zhu algebra '
                                 'as a dict'.format(A.variable_names()))
            weights = {A.variable_names()[0]: weights}
        U = A.module(weights)
        name = 'Verma module of highest weight {{{}}}'.format(
            ', '.join('{!r}: {}'.format(s, weights[s]) for s in A.variable_names()))
        from .vertex_algebra_module import InducedModule
        return InducedModule(self, U, name)

    def jacobi_identity_failures(self, triples=None, skew_symmetry=True):
        r"""
        The failures of the Jacobi identity for the declared OPEs: a dict
        ``label -> nonzero element`` (empty if the OPEs define a vertex
        algebra with the basis of ordered normal ordered monomials).

        Checked are the commutator formulas

        .. MATH::

            a_{(m)} (b_{(n)} c) - (-1)^{|a||b|} b_{(n)} (a_{(m)} c)
            - \sum_{j \ge 0} \binom{m}{j} (a_{(j)} b)_{(m+n-j)} c = 0

        for generators ``a, b, c`` and all ``m, n >= 0`` (only finitely many
        can fail: every term vanishes outside a box computed from the OPEs),
        label ``('jacobi', a, b, c, m, n)``, and skew-symmetry of the OPE of
        a generator with itself (the opposite orientation of the other
        pairs is derived by skew-symmetry), label ``('skew', a, b, n)`` for
        ``a_(n) b``.  With
        coefficients in a polynomial ring or field of fractions the values
        are polynomial in the free constants: the conditions of a conformal
        bootstrap (see :mod:`bootstrap`).

        INPUT:

        - ``triples`` -- (default: ``None``: all ordered triples) a list of
          triples of generator names, or ``'sorted'`` for the triples
          ``a <= b <= c`` in generator order.  The sorted triples are about a
          sixth of the work and gave the same answers in all our tests, but
          only all triples are guaranteed to be enough
        - ``skew_symmetry`` -- boolean (default: ``True``); ``False`` skips
          the skew-symmetry checks

        OUTPUT:

        A dict ``label -> nonzero element``.

        EXAMPLES:

        Affine ``sl_2`` at level 1, with a wrong level in ``H_(1) H``::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: N.<E, H, F> = NormallyOrderedExpressions(QQ)
            sage: sl2 = {('H', 'E'): [2*E], ('H', 'F'): [-2*F], ('E', 'F'): [H, 1]}
            sage: V = UniversalVertexAlgebra(QQ, N, sl2 | {('H', 'H'): [0, 3]})
            sage: bad = V.jacobi_identity_failures()
            sage: len(bad), bad[('jacobi', 'H', 'E', 'F', 1, 0)]
            (8, 1)
            sage: V.jacobi_identity_failures(triples=[('E', 'E', 'F')])
            {}
            sage: V = V.with_opes(sl2 | {('H', 'H'): [0, 2]})
            sage: V.jacobi_identity_failures(), V.satisfies_jacobi_identity()
            ({}, True)

        A Virasoro OPE with ``L_(1) L = 3 L`` is not skew-symmetric::

            sage: N.<L> = NormallyOrderedExpressions(QQ)
            sage: W = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 3*L, 0, 1/2]})
            sage: W.jacobi_identity_failures(triples=[])
            {('skew', 'L', 'L', 0): -dL}

        The ``N = 1`` superconformal algebra (Neveu-Schwarz) satisfies the
        Jacobi identities for generic ``c``; the same OPEs with ``G``
        declared even are not skew-symmetric (negative control)::

            sage: K.<c> = Frac(QQ['c'])
            sage: def n1(parities):
            ....:     N.<L, G> = NormallyOrderedExpressions(K, parities=parities)
            ....:     return UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:         ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 2*c/3]})
            sage: n1({'G': 1}).jacobi_identity_failures()
            {}
            sage: sorted(n1({}).jacobi_identity_failures(triples=[]).items())
            [(('skew', 'G', 'G', 0), 4*L), (('skew', 'G', 'G', 2), 4/3*c)]

        The constructor refuses OPEs failing the Jacobi identities with
        ``verify_jacobi=True``; ``J_(0) J = J`` violates skew-symmetry
        (``J_(0) J = -J``)::

            sage: UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): ['J', 1]}, verify_jacobi=True)
            Traceback (most recent call last):
            ...
            ValueError: the OPEs violate skew-symmetry J_(0) J: 2*J (and at least 2 more)

        TESTS:

        The failures are exactly those of a brute-force check of the
        commutator formula for ``0 <= m, n < 7`` (affine ``sl_2`` with ``[H,
        E] = 3E``): the box bounding ``m, n`` is right::

            sage: N.<E, H, F> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, N, {('H', 'E'): [3*E], ('H', 'F'): [-2*F],
            ....:                                    ('E', 'F'): [H, 1], ('H', 'H'): [0, 2]})
            sage: def brute(V, top):
            ....:     out = {}
            ....:     for a in V.gens():
            ....:         for b in V.gens():
            ....:             for c in V.gens():
            ....:                 for m in range(top):
            ....:                     for n in range(top):
            ....:                         d = (a.nproduct(b.nproduct(c, n), m) - b.nproduct(a.nproduct(c, m), n)
            ....:                              - sum(binomial(m, j) * a.nproduct(b, j).nproduct(c, m + n - j)
            ....:                                    for j in range(m + 1)))
            ....:                         if d:
            ....:                             out['jacobi', str(a), str(b), str(c), m, n] = d
            ....:     return out
            sage: ours = {l: d for l, d in V.jacobi_identity_failures().items() if l[0] == 'jacobi'}
            sage: len(ours), ours == brute(V, 7)
            (14, True)
        """
        return dict(self._jacobi_failure_iter(triples, skew_symmetry))

    def satisfies_jacobi_identity(self, triples=None, skew_symmetry=True):
        r"""
        True if :meth:`jacobi_identity_failures` is empty (stops at the
        first failure).

        INPUT:

        - ``triples`` -- (default: ``None``) as for
          :meth:`jacobi_identity_failures`
        - ``skew_symmetry`` -- (default: ``True``) as for
          :meth:`jacobi_identity_failures`

        OUTPUT:

        A boolean.

        EXAMPLES:

        The ``W_3`` algebra (Zamolodchikov) for generic central charge::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: Vir = {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:        ('L', 'W'): [W.derivative(), 3*W]}
            sage: Lam = L*L - 3/10*L.derivative(2)
            sage: def w3(beta):
            ....:     WW = [L.derivative(3)/15 + beta*Lam.derivative(),
            ....:           3/10*L.derivative(2) + 2*beta*Lam,
            ....:           L.derivative(), 2*L, 0, c/3]
            ....:     return dict(Vir) | {('W', 'W'): WW}
            sage: V = UniversalVertexAlgebra(K, N, w3(16/(22 + 5*c)))
            sage: V.satisfies_jacobi_identity()
            True
            sage: V.opes()[('W', 'W')][1]
            (32/5/(c + 22/5))*:L L: + ((3/10*c - 3/5)/(c + 22/5))*d^2L
            sage: UniversalVertexAlgebra(K, N, w3(1), verify_jacobi=True)
            Traceback (most recent call last):
            ...
            ValueError: the OPEs violate the Jacobi identity [L_(3), W_(0)] W: (c + 6/5)*dL (and at least 2 more)
        """
        for _ in self._jacobi_failure_iter(triples, skew_symmetry):
            return False
        return True

    def _jacobi_triples(self, triples):
        """Generator index triples (default: all ordered triples).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: W = va.W3(QQ, 2)
            sage: W._jacobi_triples(None)[:3], len(W._jacobi_triples(None))
            ([(0, 0, 0), (0, 0, 1), (0, 1, 0)], 8)
            sage: W._jacobi_triples('sorted')
            [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
            sage: W._jacobi_triples([('L', 'W', 'W'), ('W', 'W', 'W')])
            [(0, 1, 1), (1, 1, 1)]
            sage: W._jacobi_triples([(0, 1, 1), (1, int(1), 1)])           # Sage integers too
            [(0, 1, 1), (1, 1, 1)]
        """
        r = len(self._gens)
        if triples is None:
            return [(i, j, k) for i in range(r) for j in range(r)
                    for k in range(r)]
        if isinstance(triples, str) and triples == 'sorted':
            return [(i, j, k) for i in range(r) for j in range(i, r)
                    for k in range(j, r)]
        out = []
        for t in triples:
            if len(t) != 3:
                raise ValueError('a triple of generators expected, got '
                                 '{!r}'.format(t))
            out.append(tuple(int(x) if _is_int_like(x) else self._idx[x]
                             for x in t))
        return out

    @staticmethod
    def _jacobi_label_str(label):
        """
        A failure label as text.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: UniversalVertexAlgebra._jacobi_label_str(('skew', 'a', 'b', 2))
            'skew-symmetry a_(2) b'
            sage: UniversalVertexAlgebra._jacobi_label_str(('jacobi', 'a', 'b', 'c', 1, 0))
            'the Jacobi identity [a_(1), b_(0)] c'
        """
        if label[0] == 'skew':
            return 'skew-symmetry {}_({}) {}'.format(label[1], label[3],
                                                     label[2])
        _, a, b, c, m, n = label
        return 'the Jacobi identity [{}_({}), {}_({})] {}'.format(a, m, b, n, c)

    def _jacobi_failure_iter(self, triples, skew_symmetry):
        """Yield ``(label, nonzero element)`` for each failing identity.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): ['J', 1]}, check=False)
            sage: list(V._jacobi_failure_iter([], True))
            [(('skew', 'J', 'J', 0), 2*J)]
            sage: list(V._jacobi_failure_iter([], False))
            []
        """
        if skew_symmetry:
            yield from self._skew_failure_iter()
        for i, j, k in self._jacobi_triples(triples):
            yield from self._jacobi_triple_iter(i, j, k)

    def _skew_failure_iter(self):
        """Skew-symmetry of the OPEs declared directly: ``i`` with itself,
        or both orientations of ``(i, j)`` (``j_(n) i`` against the
        transform of ``i_(.) j``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): ['J', 1]}, check=False)
            sage: list(V._skew_failure_iter())
            [(('skew', 'J', 'J', 0), 2*J)]
            sage: list(va.Virasoro(QQ, 1/2)._skew_failure_iter())
            []
        """
        for (i, j) in sorted(self._opes):
            if i > j or (i < j and (j, i) not in self._opes):
                continue
            stored = self._opes[(j, i)]
            sign = self._swap((i, 0), (j, 0))
            derived = [sign * x for x in self._commute_list(self._opes[(i, j)])]
            for n in range(max(len(stored), len(derived))):
                d = ((stored[n] if n < len(stored) else self.zero())
                     - (derived[n] if n < len(derived) else self.zero()))
                if d:
                    yield ('skew', self._names[j], self._names[i], n), d

    def _jacobi_triple_iter(self, i, j, k):
        r"""
        The commutator formula for the generators ``a, b, c`` (indices
        ``i, j, k``), all ``m, n >= 0``.  With ``X[n][m] = a_(m) b_(n) c``,
        ``Y[m][n] = b_(n) a_(m) c`` and ``Z[q][p] = (a_(q) b)_(p) c``, every
        term vanishes for ``m >= M`` or ``n >= N`` (below).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: N.<L, W> = NormallyOrderedExpressions(QQ)
            sage: Lam = L*L - 3/10*L.derivative(2)
            sage: def w3(beta):                                   # c = 2
            ....:     return {('L', 'L'): [L.derivative(), 2*L, 0, 1],
            ....:             ('L', 'W'): [W.derivative(), 3*W],
            ....:             ('W', 'W'): [L.derivative(3)/15 + beta*Lam.derivative(),
            ....:                          3/10*L.derivative(2) + 2*beta*Lam,
            ....:                          L.derivative(), 2*L, 0, 2/3]}
            sage: list(UniversalVertexAlgebra(QQ, N, w3(1/2))._jacobi_triple_iter(0, 1, 1))
            []
            sage: f = list(UniversalVertexAlgebra(QQ, N, w3(1))._jacobi_triple_iter(0, 1, 1))
            sage: len(f), f[0]
            (3, (('jacobi', 'L', 'W', 'W', 3, 0), 16/5*dL))
        """
        A, B, C = self.gen(i), self.gen(j), self.gen(k)
        s = self._swap((i, 0), (j, 0))
        AB, AC, BC = A.ope(B), A.ope(C), B.ope(C)
        X = [A.ope(x) for x in BC]
        Y = [B.ope(y) for y in AC]
        Z = [x.ope(C) for x in AB]
        M = max([len(x) for x in X] + [len(AC)]
                + [q + len(z) for q, z in enumerate(Z)])
        N = max([len(BC)] + [len(y) for y in Y] + [len(z) for z in Z])
        zero = self.zero()

        def at(T, a, b):
            if a < len(T) and b < len(T[a]):
                return T[a][b]
            return zero

        names = self._names
        for m in range(M):
            for n in range(N):
                d = at(X, n, m) - s * at(Y, m, n)
                for q in range(min(m, len(Z) - 1) + 1):
                    d -= binomial(m, q) * at(Z, q, m + n - q)
                if d:
                    yield ('jacobi', names[i], names[j], names[k], m, n), d

    # ------------------------------------------------------------------ #
    #  Key-level kernels (base-class hooks)                               #
    # ------------------------------------------------------------------ #

    def _build_engine(self):
        """
        The C++ engine (:func:`_engine_kernel`) this algebra computes with,
        or ``None`` (then the pure-Python recursion).  :class:`vertex_algebra.VertexAlgebra` builds it on first
        use, dispatches to it and drops it in :meth:`clear_caches`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: E = V._build_engine()
            sage: E is None or E.kind() == 'q'
            True
            sage: V.with_conformal_vector('L')._build_engine() is not None or E is None
            True
        """
        return _engine_kernel(self)

    # key-level hooks without an engine (with one, the base class's cached
    # accessors call the engine instead)

    def _ope_key(self, a, b):
        """
        The OPE of two basis keys as a tuple of raw pole dicts, by the
        pure-Python recursion (the base class calls the engine instead when
        there is one).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._ope_key(((0, 0),), ((0, 0),))
            ({((0, 1),): 1}, {((0, 0),): 2}, {}, {(): 1/4})
        """
        return tuple(dict(x._monomial_coefficients)
                     for x in self._fb_ope_mono(a, b))

    def _no_key(self, a, b):
        """
        The normally ordered product of two basis keys as a raw dict
        (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: d = V._no_key(((0, 0),), ((0, 1),)); sorted(d.items())
            [(((0, 1), (0, 0)), 1), (((0, 3),), 1/6)]
            sage: V._from_dict(d) == L.no(L.derivative())
            True
        """
        return dict(self._fb_no_mono(a, b)._monomial_coefficients)

    def _deriv_key(self, m, n):
        """
        The ``n``-th derivative of a basis key as a raw dict (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._deriv_key(((0, 0),), 2)
            {((0, 2),): 1}
            sage: V._from_dict(V._deriv_key(((0, 0), (0, 0)), 1)) == L.no(L).derivative()
            True
        """
        return dict(self._fb_deriv_mono(m, int(n))._monomial_coefficients)

    # ------------------------------------------------------------------ #
    #  Conformal weights, gradings, weight spaces                         #
    # ------------------------------------------------------------------ #

    def _strong_generators_for_ideals(self):
        """
        The generators (the strong generators ideals are closed under).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._strong_generators_for_ideals()
            (L,)
        """
        return self.gens()

    def _key_order(self, m):
        """
        The sort key of basis keys for leading monomials: longer words lead.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._key_order(((0, 1), (0, 0)))
            (2, ((0, 1), (0, 0)))
            sage: max([((0, 3),), ((0, 0), (0, 0))], key=V._key_order)
            ((0, 0), (0, 0))
        """
        return (len(m), m)

    def _weight_generators(self):
        r"""
        The strong generators (a Virasoro vector is checked on them).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: N.<J, X> = NormallyOrderedExpressions(QQ)
            sage: UniversalVertexAlgebra(QQ, N, {('J', 'J'): [0, 1]},
            ....:                        conformal_vector=J*J/2)
            Traceback (most recent call last):
            ...
            ValueError: T_(0) X = 0 is not ∂X: T does not act as the translation operator on X
            sage: N.<J> = NormallyOrderedExpressions(QQ)
            sage: W = UniversalVertexAlgebra(QQ, N, {('J', 'J'): [0, 1]},
            ....:                            conformal_vector=J*J/2 + J.derivative()/3)
            sage: W.central_charge(), W.degrees()
            (-1/3, {'J': 1})
        """
        return [(name, ((k, 0),)) for k, name in enumerate(self._gens)]

    def _weight_data_from(self, weights):
        """Generator weights ``h``; every term of every declared OPE must
        have weight ``h_a + h_b - n - 1`` in its pole ``a_(n) b``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._weight_data_from([2])
            {'h': [2]}
            sage: V._weight_data_from([1])
            Traceback (most recent call last):
            ...
            ValueError: the conformal weights {'L': 1} are not compatible with the OPE of L and L: pole 1 contains dL of weight 2, expected 1
        """
        data = {'h': list(weights)}
        for (i, j), poles in self._opes.items():
            for n, p in enumerate(poles):
                want = weights[i] + weights[j] - n - 1
                for m in p.support():
                    w = self._weight_key(m, data)
                    if w != want:
                        raise ValueError(
                            'the conformal weights {} are not '
                            'compatible with the OPE of {} and {}: pole {} '
                            'contains {} of weight {}, expected {}'.format(
                                dict(zip(self._gens, weights)), self._names[i],
                                self._names[j], n + 1, self._repr_term(m), w,
                                want))
        return data

    def _weight_key(self, m, data):
        """
        The weight of a basis key for the weight data ``data``: the sum of
        ``h_i + n`` over its letters ``(i, n)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._weight_key(((0, 1), (0, 0)), {'h': [2]})
            5
        """
        h = data['h']
        return sum((h[i] + n for i, n in m), self.base_ring().zero())

    def _grading_data(self, spec, group):
        r"""
        ``spec``: a dict ``generator name -> horizontal_grade`` (elements of the
        grading group ``group``, or of an inferred one, see
        :meth:`VertexAlgebra.with_horizontal_grading`; missing generators get grade 0).
        Every term of every declared OPE must have grade ``g_a + g_b``.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: N.<J, E, F> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, N, {('J', 'J'): [0, 2], ('J', 'E'): [2*E],
            ....:                                    ('J', 'F'): [-2*F], ('E', 'F'): [J, 1]})
            sage: Vg = V.with_horizontal_grading({'E': 1, 'F': -1})
            sage: J, E, F = Vg.gens()
            sage: (E.no(F)).horizontal_grade(), E.derivative().no(E).horizontal_grade()
            (0, 2)
            sage: V.with_horizontal_grading({'E': 1, 'F': 1})
            Traceback (most recent call last):
            ...
            ValueError: the grading is not compatible with the OPE of E and F: pole 1 contains J of grade 0, expected 2
            sage: E2 = V.with_horizontal_grading({'E': 1, 'F': 1}, group=Zmod(2)).gen('E')
            sage: E2.no(E2).horizontal_grade()
            0
            sage: Vk = V.with_horizontal_grading({'E': (1, 0), 'F': (-1, 0)})
            sage: Vk.gen('F').horizontal_grade(), Vk.horizontal_grading_group()
            ((-1, 0), Ambient free module of rank 2 over the principal ideal domain Integer Ring)
        """
        from ._grading_groups import infer_grading_group
        spec = dict(spec)
        for s in spec:
            if s not in self._idx:
                raise ValueError('grade given for unknown generator {!r}'.format(s))
        A = infer_grading_group(spec.values(), group)
        g = [A.coords(spec[s]) if s in spec else A.zero for s in self._gens]

        def grade_of(m):
            t = A.zero
            for i, _ in m:
                t = A.add(t, g[i])
            return t

        for (i, j), poles in self._opes.items():
            want = A.add(g[i], g[j])
            for q, p in enumerate(poles):
                for m in p.support():
                    if grade_of(m) != want:
                        raise ValueError(
                            'the grading is not compatible with the OPE of {} '
                            'and {}: pole {} contains {} of grade {}, expected '
                            '{}'.format(self._names[i], self._names[j], q + 1,
                                        self._repr_term(m),
                                        A.element(grade_of(m)), A.element(want)))
        return {'group': A, 'g': g}

    def _grade_key(self, m):
        """
        The grade of a basis key, as coordinates in the grading group.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['E', 'F'], horizontal_grading={'E': 1, 'F': -1})
            sage: V._grade_key(((0, 1), (0, 0), (1, 0)))
            (1,)
        """
        gd = self._grading
        A, g = gd['group'], gd['g']
        t = A.zero
        for i, _ in m:
            t = A.add(t, g[i])
        return t

    def _weight_space_enumeration(self, W, G, data):
        """
        The basis keys of weight ``W`` (and grade ``G``), see
        :meth:`_letter_multisets`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: e = V._weight_space_enumeration(4, None, {'h': [2]})
            sage: e.finite, sorted(e.factory())
            (True, [((0, 0), (0, 0)), ((0, 2),)])
        """
        return self._letter_multisets(W, G, data).enumeration()

    def _letter_multisets(self, W, G, data):
        """
        The multisets of letters of weight ``W`` and grade ``G`` (a
        ``weight_spaces._LetterMultisets``; odd letters at most once).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: S = V._letter_multisets(4, None, {'h': [2]})
            sage: sorted(S.enumeration().factory())
            [((0, 0), (0, 0)), ((0, 2),)]
        """
        from .weight_spaces import _LetterMultisets, _to_rational
        h = [_to_rational(x, 'conformal weight of ' + s)
             for x, s in zip(data['h'], self._gens)]
        r = len(self._gens)
        odd = [self._parities.get(i, 0) == 1 for i in range(r)]
        if G is None:
            return _LetterMultisets(h, [()] * r, odd, W, ())
        gd = self._grading
        return _LetterMultisets(h, gd['g'], odd, W, G, moduli=gd['group'].moduli)

    def _weight_bounds(self, G, data):
        """
        ``(lower bound, denominator)`` of the weights of the keys of grade
        ``G`` (see :meth:`vertex_algebra.VertexAlgebra._weight_bounds`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._weight_bounds(None, {'h': [2]})
            (0, 1)
        """
        S = self._letter_multisets(0, G, data)
        return S.weight_lower_bound(), S.weight_denominator()

    # Element-level products with a lifted kernel: the coefficients are
    # lifted to the kernel ring, the key-level jobs ('kope', 'kno',
    # 'kderiv', 'knprod') computed and summed there, and only the sums mapped back
    # (mapping every cached key result back cost ~30% of a W_3 OPE).
    # Coefficients whose denominators the kernel cannot express (e.g.
    # 1/(k+2) in a Sugawara vector) take the base-class path.

    # ------------------------------------------------------------------ #
    #  Generator OPEs                                                     #
    # ------------------------------------------------------------------ #

    def _swap(self, x, y):
        """The swap sign ``(-1)^{parity(x)*parity(y)}`` in the base ring.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: bc = va.FermionicGhosts(QQ)
            sage: bc._swap((0, 0), (1, 0)), bc._swap((0, 0), ((0, 0), (1, 1)))
            (-1, 1)
        """
        px = self._parity_key(self._as_mono(x))
        py = self._parity_key(self._as_mono(y))
        return self.base_ring()((-1) ** (px * py))

    def _as_mono(self, x):
        """
        A letter as a one-letter key; keys unchanged.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._as_mono((0, 1)), V._as_mono(((0, 1), (0, 0)))
            (((0, 1),), ((0, 1), (0, 0)))
        """
        return x if isinstance(x, tuple) and not _is_letter(x) else (x,)

    def _parity_letter(self, letter):
        """
        The parity of the generator of a letter.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: bc = va.FermionicGhosts(QQ)
            sage: bc._parity_letter((1, 3))
            1
        """
        return self._parities.get(letter[0], 0)

    def _ope_gens(self, i, j):
        """
        OPE of the plain generators ``i`` and ``j`` in the orientation
        ``(i, j)``, as a list of elements.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: bc = va.FermionicGhosts(QQ)
            sage: bc._ope_gens(0, 1), bc._ope_gens(1, 0), bc._ope_gens(0, 0)
            ([1], [1], [])
        """
        key = (i, j)
        if key in self._gen_ope_cache:
            return self._gen_ope_cache[key]
        if (i, j) in self._opes:
            r = list(self._opes[(i, j)])
        elif (j, i) in self._opes:
            stored = self._ope_gens(j, i)
            sign = self._swap((i, 0), (j, 0))
            r = [sign * x for x in self._commute_list(stored)]
        else:
            r = []
        self._gen_ope_cache[key] = r
        return r

    def _ifac(self, k):
        """``1/k!`` in the base ring (multiplying by it also works over
        rings Sage cannot divide elements in, e.g. generic quotient
        rings).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._ifac(3)
            1/6
        """
        return self._one_R / factorial(k)

    def _commute_list(self, AB):
        """[BA]_q from [AB]_l (no swap sign applied).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['a', 'b'], {('a', 'b'): ['b', 'a']})
            sage: a, b = V.gens()
            sage: V._commute_list(V._ope_gens(0, 1)) == b.ope(a)
            True
            sage: b.ope(a)
            [da - b, a]
        """
        L = len(AB)
        out = [self.zero()] * L
        for q in range(1, L + 1):
            tot = self.zero()
            for l in range(q, L + 1):
                tot += ((-1) ** l) * (self._deriv_el_serial(AB[l - 1], l - q)
                                      * self._ifac(l - q))
            out[q - 1] = tot
        return out

    # ------------------------------------------------------------------ #
    #  Pure-Python recursion (element-valued; without the engine)        #
    # ------------------------------------------------------------------ #

    def _fb_deriv(self, el, n=1):
        """
        The ``n``-th derivative of an element (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_deriv(L.no(L), 2) == L.no(L).derivative(2)
            True
        """
        if n == 0 or el.is_zero():
            return el
        res = self.zero()
        for m, c in el._monomial_coefficients.items():
            res += self._rc(c) * self._fb_deriv_mono(m, n)
        return res

    def _fb_deriv_mono(self, mono, n):
        """
        The ``n``-th derivative of a basis key, as an element (pure Python,
        cached in ``_fb_deriv_cache``): Leibniz on the first letter.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_deriv_mono(((0, 0), (0, 0)), 1)
            2*:dL L: + 1/6*d^3L
            sage: _ == L.no(L).derivative()
            True
        """
        n = int(n)
        key = (mono, n)
        if key in self._fb_deriv_cache:
            return self._fb_deriv_cache[key]
        if n == 0:
            res = self.monomial(mono)
        elif not mono:
            res = self.zero()
        elif len(mono) == 1:
            i, k = mono[0]
            res = self.monomial(((i, k + n),))
        else:
            x = mono[0]
            rest = mono[1:]
            res = self.zero()
            for j in range(n + 1):
                term = self._fb_no_el(self.monomial(((x[0], x[1] + j),)),
                                      self._fb_deriv_mono(rest, n - j))
                res += binomial(n, j) * term
        self._fb_deriv_cache[key] = res
        return res

    def _fb_no_el(self, elA, elB):
        """
        The normally ordered product of two elements (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_no_el(L, L.derivative()) == L.no(L.derivative())
            True
        """
        if elA.is_zero() or elB.is_zero():
            return self.zero()
        res = self.zero()
        for mA, cA in elA._monomial_coefficients.items():
            for mB, cB in elB._monomial_coefficients.items():
                res += (self._rc(cA) * self._rc(cB)) * self._fb_no_mono(mA, mB)
        return res

    def _fb_no_mono(self, mA, mB):
        """
        The normally ordered product of two basis keys, as an element
        (pure Python, cached in ``_fb_no_cache``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_no_mono(((0, 0),), ((0, 1),))
            :dL L: + 1/6*d^3L
            sage: x = V.monomial(((0, 1), (0, 0))); x
            :dL L:
            sage: V._fb_no_mono(((0, 1), (0, 0)), ((0, 0),)) == x.no(L)
            True
        """
        key = (mA, mB)
        if key in self._fb_no_cache:
            return self._fb_no_cache[key]
        if not mA:
            res = self.monomial(mB)
        elif not mB:
            res = self.monomial(mA)
        elif len(mA) == 1 and len(mB) == 1:
            res = self._fb_no_letters(mA[0], mB[0])
        elif len(mA) >= 2:
            Al = mA[0]
            rest = mA[1:]
            Cmono = mB
            AC = self._fb_ope_mono((Al,), Cmono)
            BC = self._fb_ope_mono(rest, Cmono)
            if rest == (Al,):
                BC = AC
            sign = self._swap(Al, rest)
            res = self._fb_no_el(self.monomial((Al,)),
                                 self._fb_no_mono(rest, Cmono))
            for l in range(1, len(BC) + 1):
                res += (self._fb_no_el(self.monomial(((Al[0], Al[1] + l),)),
                                       BC[l - 1]) * self._ifac(l))
            for l in range(1, len(AC) + 1):
                res += (sign * self._fb_no_el(self._fb_deriv_mono(rest, l),
                                              AC[l - 1]) * self._ifac(l))
        else:
            res = self._fb_no_insert(mA[0], mB)
        self._fb_no_cache[key] = res
        return res

    def _fb_no_insert(self, x, mB):
        """``NO[x, monomial(mB)]`` with ``x`` a letter.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_no_insert((0, 0), ((0, 1), (0, 0)))
            :dL L L: + 1/6*:d^3L L:
            sage: _ == L.no(L.derivative().no(L))
            True
        """
        key = (x, mB)
        if key in self._fb_no_cache:
            return self._fb_no_cache[key]
        if len(mB) == 1:
            res = self._fb_no_letters(x, mB[0])
        else:
            y = mB[0]
            restB = mB[1:]
            c = _cmp_letter(x, y)
            if c > 0:
                res = self.monomial((x,) + mB)
            elif c == 0:
                if self._parity_letter(x) == 1:
                    res = self._fb_nc(x, x).no(self.monomial(restB)) / 2
                else:
                    res = self.monomial((x,) + mB)
            else:
                sign = self._swap(x, y)
                inner = self._fb_no_insert(x, restB)
                res = (sign * self._fb_no_el(self.monomial((y,)), inner)
                       + self._fb_no_el(self._fb_nc(x, y),
                                        self.monomial(restB)))
        self._fb_no_cache[key] = res
        return res

    def _fb_no_letters(self, x, y):
        """Normal ordering of two letters into standard form.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_no_letters((0, 0), (0, 1)) == L.no(L.derivative())
            True
            sage: V._fb_no_letters((0, 1), (0, 0))
            :dL L:
        """
        key = (x, y)
        if key in self._fb_no_cache:
            return self._fb_no_cache[key]
        c = _cmp_letter(x, y)
        if c > 0:
            res = self.monomial((x, y))
        elif c == 0:
            if self._parity_letter(x) == 1:
                res = self._fb_nc(x, y) / 2
            else:
                res = self.monomial((x, y))
        else:
            sign = self._swap(x, y)
            res = sign * (self.monomial((y, x)) - self._fb_nc(y, x))
        self._fb_no_cache[key] = res
        return res

    def _fb_nc(self, x, y):
        """The contraction part ``NO[x,y] - swap NO[y,x]``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_nc((0, 0), (0, 1))
            1/6*d^3L
            sage: _ == L.no(L.derivative()) - L.derivative().no(L)
            True
        """
        poles = self._fb_ope_letters(x, y)
        res = self.zero()
        for m in range(1, len(poles) + 1):
            res += ((-1) ** (m + 1)) * (self._fb_deriv(poles[m - 1], m)
                                        * self._ifac(m))
        return res

    def _fb_ope_el(self, elA, elB):
        """
        The OPE of two elements as a trimmed list of poles (pure Python).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_ope_el(L, L.no(L)) == L.ope(L.no(L))
            True
        """
        if elA.is_zero() or elB.is_zero():
            return []
        parts = []
        maxq = 0
        for mA, cA in elA._monomial_coefficients.items():
            for mB, cB in elB._monomial_coefficients.items():
                poles = self._fb_ope_mono(mA, mB)
                parts.append((self._rc(cA) * self._rc(cB), poles))
                maxq = max(maxq, len(poles))
        out = [self.zero()] * maxq
        for s, poles in parts:
            for q in range(len(poles)):
                out[q] += s * poles[q]
        return self._fb_trim(out)

    def _fb_ope_mono(self, mA, mB):
        """
        The OPE of two basis keys as a tuple of elements (pure Python,
        cached in ``_fb_ope_cache``): :meth:`_fb_lq` for a composite left key,
        :meth:`_fb_rab` for a composite right key, else the generator OPEs.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_ope_mono(((0, 0),), ((0, 0),))
            (dL, 2*L, 0, 1/4)
            sage: x = V.monomial(((0, 1), (0, 0)))
            sage: list(V._fb_ope_mono(((0, 1), (0, 0)), ((0, 0), (0, 0)))) == x.ope(L.no(L))
            True
        """
        key = (mA, mB)
        if key in self._fb_ope_cache:
            return self._fb_ope_cache[key]
        if not mA or not mB:
            r = ()
        elif len(mA) >= 2:
            r = tuple(self._fb_trim(self._fb_lq(mA[0], mA[1:], mB)))
        elif len(mB) >= 2:
            r = tuple(self._fb_trim(self._fb_rab(mA[0], mB[0], mB[1:])))
        else:
            r = tuple(self._fb_trim(self._fb_ope_letters(mA[0], mB[0])))
        self._fb_ope_cache[key] = r
        return r

    def _fb_ope_letters(self, la, lb):
        """
        The OPE of two letters ``∂^n g_i``, ``∂^m g_j`` (list of poles, not
        trimmed), from the generator OPE by :meth:`_fb_ope_derR` and
        :meth:`_fb_ope_derL`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: dL = L.derivative()
            sage: V._fb_trim(V._fb_ope_letters((0, 1), (0, 2))) == dL.ope(dL.derivative())
            True
        """
        i, n = la
        j, m = lb
        if n == 0 and m == 0:
            return list(self._ope_gens(i, j))
        if m == 0:
            base = self._fb_ope_letters((i, 0), (j, 0))
            return self._fb_ope_derL(base, n)
        if n == 0:
            base = self._fb_ope_letters((i, 0), (j, 0))
            return self._fb_ope_derR(base, m)
        base = self._fb_ope_letters((i, 0), (j, m))
        return self._fb_ope_derL(base, n)

    def _fb_ope_derL(self, base, i):
        """OPE[D^i A, B] from OPE[A, B] (derivative on the left factor).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_ope_derL(V._ope_gens(0, 0), 1)
            [0, -dL, -4*L, 0, -1]
            sage: _ == L.derivative().ope(L)
            True
        """
        if not base:
            return []
        L = len(base)
        out = [self.zero()] * (L + i)
        for q in range(i + 1, L + i + 1):
            k = q - i
            out[q - 1] = ((-1) ** i) * rising_factorial(k, i) * base[k - 1]
        return out

    def _fb_ope_derR(self, base, i):
        """OPE[A, D^i B] from OPE[A, B] (derivative on the right factor).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_ope_derR(V._ope_gens(0, 0), 1)
            [d^2L, 3*dL, 4*L, 0, 1]
            sage: _ == L.ope(L.derivative())
            True
        """
        if not base:
            return []
        L = len(base)
        out = [self.zero()] * (L + i)
        for q in range(1, L + i + 1):
            tot = self.zero()
            kmin = max(0, q - L)
            kmax = min(i, q - 1)
            for k in range(kmin, kmax + 1):
                m = q - k
                if m < 1:
                    continue
                tot += binomial(i, k) * rising_factorial(m, k) * \
                    self._fb_deriv(base[m - 1], i - k)
            out[q - 1] = tot
        return out

    def _fb_rab(self, Al, Bl, Cm):
        """OPE[A, NO[B, C]].

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_trim(V._fb_rab((0, 0), (0, 0), ((0, 0),))) == L.ope(L.no(L))
            True
        """
        AB = self._fb_ope_mono((Al,), (Bl,))
        if Cm == (Bl,):
            AC = AB
        else:
            AC = self._fb_ope_mono((Al,), Cm)
        maxAB = len(AB)
        maxAC = len(AC)
        ABC = [self._fb_ope_el(AB[r - 1], self.monomial(Cm))
               for r in range(1, maxAB + 1)]
        maxABC = max((len(x) for x in ABC), default=0)
        maxq = max([maxAC, 0] + [len(ABC[r - 1]) + r
                                 for r in range(1, maxAB + 1)])
        sign = self._swap(Al, Bl)
        out = [self.zero()] * maxq
        for q in range(1, maxq + 1):
            acq = AC[q - 1] if q <= maxAC else self.zero()
            abq = AB[q - 1] if q <= maxAB else self.zero()
            tot = sign * self._fb_no_el(self.monomial((Bl,)), acq)
            tot += self._fb_no_el(abq, self.monomial(Cm))
            for l in range(max(1, q - maxAB), min(q - 1, maxABC) + 1):
                r = q - l
                abc = ABC[r - 1]
                if l <= len(abc):
                    tot += binomial(q - 1, l) * abc[l - 1]
            out[q - 1] = tot
        return out

    def _fb_lq(self, Al, Bm, Cm):
        """OPE[NO[A, B], C].

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_trim(V._fb_lq((0, 0), ((0, 0),), ((0, 0),))) == L.no(L).ope(L)
            True
        """
        AC = self._fb_ope_mono((Al,), Cm)
        if Bm == (Al,):
            BC = AC
        else:
            BC = self._fb_ope_mono(Bm, Cm)
        maxAC = len(AC)
        maxBC = len(BC)
        BAC = [self._fb_ope_el(self.monomial(Bm), AC[r - 1])
               for r in range(1, maxAC + 1)]
        maxBAC = max((len(x) for x in BAC), default=0)
        maxq = max([maxAC, maxBC, 0] + [len(BAC[r - 1]) + r
                                        for r in range(1, maxAC + 1)])
        sign = self._swap(Al, Bm)
        out = [self.zero()] * maxq
        for q in range(1, maxq + 1):
            tot = self.zero()
            for l in range(maxBC - q + 1):
                if l == 0:
                    dlA = self.monomial((Al,))
                else:
                    dlA = self.monomial(((Al[0], Al[1] + l),))
                tot += self._fb_no_el(dlA, BC[q + l - 1]) * self._ifac(l)
            for l in range(maxAC - q + 1):
                tot += (sign * self._fb_no_el(self._fb_deriv_mono(Bm, l),
                                              AC[q + l - 1]) * self._ifac(l))
            for l in range(max(1, q - maxAC), min(q - 1, maxBAC) + 1):
                r = q - l
                bac = BAC[r - 1]
                if l <= len(bac):
                    tot += sign * bac[l - 1]
            out[q - 1] = tot
        return out

    def _fb_trim(self, poles):
        """
        Drop trailing zero poles (in place) and return the list.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); L = V.gen('L')
            sage: V._fb_trim([L, V.zero(), V.zero()])
            [L]
        """
        while poles and poles[-1].is_zero():
            poles.pop()
        return poles


class _StagingAlgebra(UniversalVertexAlgebra):
    """
    A private universal vertex algebra (never shared: each one gets a new
    token in its key) whose generator OPEs are normally ordered expressions
    (``_expr``, a dict ``(i, j) -> list of expressions``) evaluated on
    demand in the algebra itself; used by :func:`_canonical_opes`.  The
    pure-Python recursion reads the generator OPEs through :meth:`_ope_gens`
    at call time; the C++ engine would take a snapshot of the (empty)
    declared OPEs, so it is switched off.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import _StagingAlgebra
        sage: E.<a> = NormallyOrderedExpressions(QQ)
        sage: S = _StagingAlgebra(QQ, ('a',), ()); S
        Universal vertex algebra on generators ['a'] over Rational Field
        sage: S._expr = {(0, 0): [E.zero(), a*a]}
        sage: S._ope_gens(0, 0)
        [0, :a a:]
    """
    _counter = itertools.count()

    @staticmethod
    def __classcall_private__(cls, base_ring, generators, parity_items,
                              implementation=None):
        """
        A new staging algebra each time (a fresh token in the key).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import _StagingAlgebra
            sage: _StagingAlgebra(QQ, ('a',), ()) is _StagingAlgebra(QQ, ('a',), ())
            False
        """
        return VertexAlgebra.__classcall__(cls, base_ring, generators,
                                           parity_items, implementation,
                                           next(_StagingAlgebra._counter))

    def __init__(self, base_ring, generators, parity_items, implementation,
                 token):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import _StagingAlgebra
            sage: from vertex_algebras.vertex_algebra_categories import _is_proto
            sage: S = _StagingAlgebra(QQ, ('a', 'b'), (('b', 1),))
            sage: S._expr, _is_proto(S), [g.parity() for g in S.gens()]
            ({}, True, [0, 1])
        """
        UniversalVertexAlgebra.__init__(self, base_ring, generators, (),
                                        parity_items,
                                        implementation=implementation,
                                        proto=True)
        self._expr = {}
        self._busy = set()

    def _build_engine(self):
        """
        ``None``: the engine would snapshot the empty declared OPEs.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import _StagingAlgebra
            sage: _StagingAlgebra(QQ, ('a',), ())._build_engine() is None
            True
        """
        return None

    def _ope_gens(self, i, j):
        """
        The generator OPE ``(i, j)``, evaluated from ``_expr`` on demand
        (the other orientation by skew-symmetry); a pair needed to normally
        order itself raises ``ValueError``.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.universal_vertex_algebra import _StagingAlgebra
            sage: E.<a, b> = NormallyOrderedExpressions(QQ)
            sage: S = _StagingAlgebra(QQ, ('a', 'b'), ())
            sage: S._expr = {(0, 1): [b]}
            sage: S._ope_gens(1, 0), S._ope_gens(0, 0)
            ([-b], [])
            sage: T = _StagingAlgebra(QQ, ('a', 'b'), ())
            sage: T._expr = {(0, 0): [a*a.derivative()]}
            sage: T._ope_gens(0, 0)
            Traceback (most recent call last):
            ...
            ValueError: cannot normally order the OPE coefficients: the OPE of a and a is needed to normally order itself ...
        """
        key = (i, j)
        r = self._gen_ope_cache.get(key)
        if r is not None:
            return r
        if key in self._busy:
            raise ValueError(
                'cannot normally order the OPE coefficients: the OPE of {} and {} '
                'is needed to normally order itself (are the OPEs compatible with '
                'positive conformal weights?); give its coefficients as basis '
                'monomials (standard words)'.format(self._names[min(i, j)],
                                                    self._names[max(i, j)]))
        self._busy.add(key)
        try:
            if key in self._expr:
                r = [self._evaluate(p) for p in self._expr[key]]
            elif (j, i) in self._expr:
                stored = self._ope_gens(j, i)
                sign = self._swap((i, 0), (j, 0))
                r = [sign * x for x in self._commute_list(stored)]
            else:
                r = []
        finally:
            self._busy.discard(key)
        self._gen_ope_cache[key] = r
        return r
