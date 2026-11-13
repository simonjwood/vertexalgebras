# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Conformal weight spaces and graded components of vertex algebras.

Machinery behind :meth:`vertex_algebra_categories.ProtoVertexAlgebras.Graded.ParentMethods.weight_space`; the
algebra-specific parts (weights and grades of basis keys, which keys can
occur) live in the subclasses.

The combinatorial core is ``_LetterMultisets``: the multisets of
"letters" ``(i, n)`` (``n``-th derivative of generator ``i``, ``n >= 0``) of
total conformal weight ``W`` and total grade ``G``, where the letter
``(i, n)`` has weight ``h_i + n`` and grade ``g_i`` and odd letters occur at
most once.  Such a set is finite exactly when (it is empty or) there is a
``μ`` with ``h_i + μ·g_i > 0`` for every bosonic generator ``i``:

- if such a ``μ`` exists, every bosonic letter has positive "twisted
  weight" ``φ = weight + μ·horizontal_grade`` and ``φ`` of the total is fixed, which
  bounds everything (``μ`` is found by an exact linear program);
- if not, Farkas' lemma gives a nonzero multiset of bosonic letters of
  weight 0 and grade 0 (derivatives fix the weight), which can be added any
  number of times: the set is infinite as soon as it is nonempty.  It is
  then enumerated lazily, by increasing number of letters.

Grades are coordinate tuples in a decomposition ``Z/n_1 × ... × Z/n_k`` of
the grading group (``n_j = 0`` for ``Z``, see ``_grading_groups``); the
torsion coordinates do not affect finiteness (a weight-0 grade-0 multiset
``c`` for the free part gives ``N c`` of full grade 0, ``N`` the exponent of
the torsion), so ``μ`` only involves the free coordinates and torsion
grades are compared modulo ``n_j``.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from itertools import count
from itertools import product as _iproduct
from math import ceil, floor, sqrt

from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.numerical.mip import MixedIntegerLinearProgram
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

#: In the infinite case, how many stages to search for a first element
#: before giving up on deciding whether the space is empty.
NONEMPTY_SEARCH_STAGES = 40


def _to_rational(x, what='conformal weight'):
    """``x`` as an element of ``QQ`` (weight spaces need rational weights);
    ``what`` names ``x`` in the error.

    TESTS::

        sage: from vertex_algebras.weight_spaces import _to_rational
        sage: _to_rational(3/2), _to_rational(3).parent()
        (3/2, Rational Field)
        sage: _to_rational(QQ['c'].gen())
        Traceback (most recent call last):
        ...
        ValueError: the conformal weight c is not a rational number; weight spaces need
        rational weights
    """
    try:
        return QQ(x)
    except (TypeError, ValueError):
        raise ValueError('the {} {} is not a rational number; weight spaces '
                         'need rational weights'.format(what, x))


def _zero_test(moduli):
    """The test "this integer tuple is the grade 0" for ``moduli`` (``0``
    a free coordinate, ``n`` a coordinate modulo ``n``), as a function.

    TESTS::

        sage: from vertex_algebras.weight_spaces import _zero_test
        sage: z = _zero_test((0,)); z((0,)), z((1,))
        (True, False)
        sage: z = _zero_test((0, 3)); z((0, 3)), z((0, 2)), z((1, 0))
        (True, False, False)
    """
    if not any(moduli):
        return lambda t: not any(t)
    return lambda t: all((x % n == 0) if n else x == 0
                         for x, n in zip(t, moduli))


def _positive_functional(h, g, bosonic):
    r"""
    A ``μ`` in ``QQ^k`` with ``h[i] + μ·g[i] > 0`` for all ``i`` in
    ``bosonic``, or ``None`` if there is none (exact LP over ``QQ``):
    ``h`` the weights, ``g`` the grade tuples (free coordinates).

    TESTS:

    ``βγ`` with weights ``1, 0`` and charges ``1, -1`` or ``1, 1`` (finite
    graded pieces), with charges ``1, 0`` (none: ``γ`` has weight and
    charge 0), and without a grading::

        sage: from vertex_algebras.weight_spaces import _positive_functional
        sage: _positive_functional([1, 0], [(1,), (-1,)], [0, 1])
        (-1/2,)
        sage: _positive_functional([1, 0], [(1,), (1,)], [0, 1])
        (1,)
        sage: _positive_functional([1, 0], [(1,), (0,)], [0, 1]) is None
        True
        sage: _positive_functional([1], [()], [0]), _positive_functional([0], [()], [0])
        ((), None)
        sage: _positive_functional([-1], [(1,)], [])      # no bosonic letters
        (0,)
    """
    k = len(g[0]) if g else 0
    if not bosonic:
        return (QQ(0),) * k
    if k == 0:
        return () if all(h[i] > 0 for i in bosonic) else None
    p = MixedIntegerLinearProgram(maximization=True, solver='ppl')
    x = p.new_variable(real=True, nonnegative=False)
    for i in bosonic:
        p.add_constraint(h[i] + sum(g[i][j] * x[j] for j in range(k)) >= x['t'])
    p.add_constraint(x['t'] <= 1)
    p.set_objective(x['t'])
    if p.solve() <= 0:
        return None
    vals = p.get_values(x)
    return tuple(QQ(vals.get(j, 0)) for j in range(k))


class _LetterMultisets:
    r"""
    The multisets of letters ``(i, n)`` of total weight ``W`` and grade
    ``G`` (see the module documentation), as letter tuples in canonical
    order: higher derivative first, then lower generator index.

    INPUT: ``h`` (rational weights), ``g`` (grade tuples, all of the same
    length ``k``; ``k = 0`` means no grading), ``odd`` (booleans), ``W``,
    ``G`` (tuple of length ``k``), ``moduli`` (optional: the ``n_j`` of the
    grading group, ``0`` for a free coordinate; default all free).

    EXAMPLES:

    βγ with ``β`` of weight 1 and grade 1, ``γ`` of weight 0 and grade -1::

        sage: from vertex_algebras.weight_spaces import _LetterMultisets
        sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
        sage: S.finite
        True
        sage: sorted(S)         # :β β γ γ:, :∂β γ:, :∂γ β:
        [((0, 0), (0, 0), (1, 0), (1, 0)), ((0, 1), (1, 0)), ((1, 1), (0, 0))]

    Without the grading the weight-0 field ``γ`` makes it infinite::

        sage: S = _LetterMultisets([1, 0], [(), ()], [False, False], 2, ())
        sage: S.finite
        False
        sage: from itertools import islice
        sage: list(islice(iter(S), 4))      # by number of letters
        [((1, 2),), ((0, 1),), ((1, 2), (1, 0)), ((0, 1), (1, 0))]

    An odd letter occurs at most once::

        sage: sorted(_LetterMultisets([1/2], [()], [True], 2, ()))
        [((0, 1), (0, 0))]

    A ``Z/2`` grade does not make the βγ space finite, but selects the
    multisets with an even number of letters::

        sage: S = _LetterMultisets([1, 0], [(1,), (1,)], [False, False], 1, (0,), moduli=(2,))
        sage: S.finite
        False
        sage: list(islice(iter(S), 3))
        [((1, 1), (1, 0)), ((0, 0), (1, 0)), ((1, 1), (1, 0), (1, 0), (1, 0))]
    """
    def __init__(self, h, g, odd, W, G, moduli=None):
        """
        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: S.r, S.k, S.free, S.finite
            (2, 1, [0], True)
        """
        self.h = [QQ(x) for x in h]
        self.g = [tuple(x) for x in g]
        self.odd = list(odd)
        self.W = QQ(W)
        self.G = tuple(G)
        self.k = len(self.G)
        self.r = len(self.h)
        self.moduli = (0,) * self.k if moduli is None else tuple(moduli)
        self._is_zero = _zero_test(self.moduli)
        self.free = [j for j in range(self.k) if self.moduli[j] == 0]
        bosonic = [i for i in range(self.r) if not self.odd[i]]
        gfree = [tuple(gr[j] for j in self.free) for gr in self.g]
        mu = _positive_functional(self.h, gfree, bosonic)
        self.mu = mu
        self.finite = mu is not None

    # -- helpers ------------------------------------------------------- #

    def _phi(self, w, gr):
        """
        The twisted weight ``w + μ·gr`` (free coordinates of the grade
        ``gr``); positive on bosonic letters.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: S.mu, S._phi(2, (1,)), S._phi(2, (0,))
            ((-1/2,), 3/2, 2)
        """
        return w + sum(m * gr[j] for m, j in zip(self.mu, self.free))

    def _neg_odd(self):
        """The odd letters of negative twisted weight (finitely many).

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: _LetterMultisets([-1/2], [()], [True], 0, ())._neg_odd()
            [(0, 0)]
            sage: _LetterMultisets([1/2], [()], [True], 0, ())._neg_odd()
            []
        """
        phi_gen = [self._phi(self.h[i], self.g[i]) for i in range(self.r)]
        return [(i, n) for i in range(self.r) if self.odd[i]
                for n in range(max(0, ceil(-phi_gen[i])))
                if phi_gen[i] + n < 0]

    def weight_lower_bound(self):
        r"""
        A lower bound (not necessarily sharp) for the weight of the
        multisets of grade ``G`` (any weight), or ``None`` if there is no finiteness certificate: the
        twisted weight ``φ = weight + μ·horizontal_grade`` of a multiset is at least the
        sum of the negative ``φ`` of odd letters.

        EXAMPLES::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 0, (-3,)).weight_lower_bound()
            -3/2
            sage: _LetterMultisets([-1/2], [()], [True], 0, ()).weight_lower_bound()
            -1/2
        """
        if not self.finite:
            return None
        neg = self._neg_odd()
        tot = sum((self._phi(self.h[i], self.g[i]) + n for i, n in neg), QQ(0))
        return tot - sum((m * self.G[j] for m, j in zip(self.mu, self.free)),
                         QQ(0))

    def weight_denominator(self):
        """A positive integer ``D`` with ``D * weight`` integral for every
        multiset.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: _LetterMultisets([1/2, 3/2], [(), ()], [True, False], 2, ()).weight_denominator()
            2
        """
        from sage.arith.functions import lcm
        return int(lcm([x.denominator() for x in self.h] + [1]))

    def _letters_upto(self, keyfun, bound):
        """All letters ``(i, n)`` with ``keyfun(i, n) <= bound``, in
        canonical order (``n`` descending, then ``i`` ascending);
        ``keyfun`` must grow with ``n``.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: S._letters_upto(lambda i, n: S.h[i] + n, 1)
            [(1, 1), (0, 0), (1, 0)]
        """
        out = []
        for i in range(self.r):
            n = 0
            while keyfun(i, n) <= bound:
                out.append((i, n))
                n += 1
                if n > 10**6:
                    raise RuntimeError('unbounded letter enumeration')
        out.sort(key=lambda t: (-t[1], t[0]))
        return out

    # -- finite case --------------------------------------------------- #

    def _iter_finite(self):
        """
        The multisets of a certified finite space (``self.finite``): a
        branch and bound over the letters of twisted weight at most the
        target, multiplicities from the highest letter down.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: sorted(S._iter_finite())
            [((0, 0), (0, 0), (1, 0), (1, 0)), ((0, 1), (1, 0)), ((1, 1), (0, 0))]
        """
        phi_gen = [self._phi(self.h[i], self.g[i]) for i in range(self.r)]
        # odd letters with negative twisted weight: finitely many
        neg = self._neg_odd()
        neg_total = sum(phi_gen[i] + n for i, n in neg)
        target = self._phi(self.W, self.G)
        phimax = target - neg_total
        if phimax < 0 and not neg:
            if self.W == 0 and self._is_zero(self.G):
                yield ()
            return
        letters = self._letters_upto(lambda i, n: phi_gen[i] + n, phimax)
        info = [(i, n, self.h[i] + n, self.g[i], phi_gen[i] + n, self.odd[i])
                for i, n in letters]
        L = len(info)
        negsuf = [0] * (L + 1)          # sum of negative φ from position on
        for p in range(L - 1, -1, -1):
            f = info[p][4]
            negsuf[p] = negsuf[p + 1] + (f if f < 0 else 0)
        W, G = self.W, self.G
        is_zero = self._is_zero
        chosen = []

        def rec(p, rphi, rw, rg):
            if p == L:
                if rw == 0 and is_zero(rg):
                    yield tuple(chosen)
                return
            if rphi < negsuf[p]:
                return
            i, n, w, gr, f, odd = info[p]
            if odd:
                mmax = 1
            elif f > 0:
                mmax = floor((rphi - negsuf[p + 1]) / f)
            else:
                mmax = 0
            for mult in range(max(0, mmax), -1, -1):
                chosen.extend([(i, n)] * mult)
                yield from rec(p + 1, rphi - mult * f, rw - mult * w,
                               tuple(a - mult * b for a, b in zip(rg, gr)))
                del chosen[len(chosen) - mult:]

        yield from rec(0, target, W, G)

    # -- stage-wise (any case) ----------------------------------------- #

    def exactly(self, K):
        """The multisets with exactly ``K`` letters (always finite), as an
        iterator; the stages of an infinite space.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(), ()], [False, False], 2, ())
            sage: sorted(S.exactly(2))
            [((0, 0), (0, 0)), ((0, 1), (1, 0)), ((1, 1), (0, 0)), ((1, 1), (1, 1)), ((1, 2), (1, 0))]
            sage: list(S.exactly(0))
            []
        """
        is_zero = self._is_zero
        if self.r == 0:
            if K == 0 and self.W == 0 and is_zero(self.G):
                yield ()
            return
        wmin = min(self.h)
        if K == 0:
            if self.W == 0 and is_zero(self.G):
                yield ()
            return
        bound = self.W - (K - 1) * wmin
        letters = self._letters_upto(lambda i, n: self.h[i] + n, bound)
        info = [(i, n, self.h[i] + n, self.g[i], self.odd[i])
                for i, n in letters]
        L = len(info)
        chosen = []

        def rec(p, rk, rw, rg):
            if rk == 0:
                if rw == 0 and is_zero(rg):
                    yield tuple(chosen)
                return
            if p == L or rw < rk * wmin:
                return
            i, n, w, gr, odd = info[p]
            mmax = 1 if odd else rk
            for mult in range(mmax, -1, -1):
                if mult and rw - mult * w < (rk - mult) * wmin:
                    continue
                chosen.extend([(i, n)] * mult)
                yield from rec(p + 1, rk - mult, rw - mult * w,
                               tuple(a - mult * b for a, b in zip(rg, gr)))
                del chosen[len(chosen) - mult:]

        yield from rec(0, K, self.W, self.G)

    def __iter__(self):
        """
        All multisets: by :meth:`_iter_finite` if finite, else stage by
        stage (number of letters), lazily.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: sorted(S) == sorted(S._iter_finite())
            True
        """
        if self.finite:
            return self._iter_finite()
        return (m for K in count() for m in self.exactly(K))

    def enumeration(self):
        """The :class:`_KeyEnumeration` of these multisets.

        TESTS::

            sage: from vertex_algebras.weight_spaces import _LetterMultisets
            sage: S = _LetterMultisets([1, 0], [(1,), (-1,)], [False, False], 2, (0,))
            sage: e = S.enumeration(); e.finite, len(list(e.factory()))
            (True, 3)
            sage: S = _LetterMultisets([1, 0], [(), ()], [False, False], 2, ())
            sage: e = S.enumeration(); e.finite, e.first_within(3)
            (False, ((1, 2),))
        """
        if self.finite:
            return _KeyEnumeration(True, factory=self._iter_finite)
        return _KeyEnumeration(False, stage=self.exactly)


def _affine_integer_solutions(Gamma, G, n, moduli=None):
    r"""
    The integer solutions of ``Gamma q = G`` as ``q0 + K m`` (``m`` in
    ``ZZ^d``): returns ``(q0, K)`` with ``K`` an ``n × d`` integer matrix of
    rank ``d`` (so ``m`` determines ``q``), or ``None`` if there is no
    integer solution.  ``Gamma=None`` means no equations.  With
    ``moduli``, row ``j`` is a congruence modulo ``moduli[j]`` (``0``: an
    equation).

    EXAMPLES::

        sage: from vertex_algebras.weight_spaces import _affine_integer_solutions
        sage: _affine_integer_solutions(matrix(ZZ, [[1, 1]]), (3,), 2)
        (
                [ 1]
        (0, 3), [-1]
        )
        sage: q0, K = _affine_integer_solutions(matrix(ZZ, [[1, 1]]), (1,), 2, moduli=(2,))
        sage: K.transpose(), q0      # q0 + q1 odd
        (
        [1 1]
        [0 2], (0, 1)
        )
    """
    if Gamma is None or Gamma.nrows() == 0:
        return vector(ZZ, [0] * n), matrix.identity(ZZ, n)
    if moduli is not None and any(moduli):
        # q with Gamma_j q ≡ G_j mod n_j: solve [Gamma | diag(n_j)] (q, y) = G
        # and project the solution lattice to q (a basis of the projection)
        tors = [j for j, nj in enumerate(moduli) if nj]
        ext = matrix(ZZ, Gamma.nrows(), len(tors),
                     lambda a, b: moduli[a] if a == tors[b] else 0)
        sol = _affine_integer_solutions(Gamma.augment(ext), G, n + len(tors))
        if sol is None:
            return None
        q0, Kx = sol
        rows = matrix(ZZ, [list(c)[:n] for c in Kx.columns()] or
                      [[0] * n]).echelon_form()
        basis = [r for r in rows.rows() if r]
        K = matrix(ZZ, n, len(basis), lambda a, b: basis[b][a])
        return vector(ZZ, list(q0)[:n]), K
    D, U, V = Gamma.smith_form()          # D = U * Gamma * V
    rhs = U * vector(ZZ, G)
    rnk = sum(1 for i in range(min(D.nrows(), D.ncols())) if D[i, i] != 0)
    y = [0] * n
    for i in range(D.nrows()):
        d = D[i, i] if i < min(D.nrows(), D.ncols()) else 0
        if d:
            if rhs[i] % d:
                return None
            y[i] = rhs[i] // d
        elif rhs[i] != 0:
            return None
    q0 = V * vector(ZZ, y)
    K = V.matrix_from_columns(range(rnk, n))
    return q0, K


def _ellipsoid_points(Q, b, c, bound):
    r"""
    The integer vectors ``m`` with ``m^T Q m + b·m + c <= bound`` for a
    positive definite rational ``Q`` (finite), as an iterator in
    lexicographic order.

    Fincke–Pohst enumeration around the real minimum ``m*``: with ``Q =
    U^T D U`` (``U`` unit upper triangular, :func:`_upper_ldl`) and ``x = m -
    m*``, ``x^T Q x = Σ_i D_i (x_i + Σ_{j>i} U_ij x_j)^2``, so the last
    coordinate is enumerated first and each further one in an interval
    given by the coordinates already chosen.  Exact over ``QQ`` (floats
    only guess the intervals, with a margin; every candidate is tested
    exactly).  Replaces the enumeration of the bounding box, whose size
    grows exponentially faster with the rank (`E_8`, weight 3: more than
    30 s).

    TESTS::

        sage: from vertex_algebras.weight_spaces import _ellipsoid_points
        sage: list(_ellipsoid_points(matrix(QQ, [[1]]), vector(QQ, [0]), 0, 4))
        [(-2), (-1), (0), (1), (2)]
        sage: list(_ellipsoid_points(identity_matrix(QQ, 2), vector(QQ, [0, 0]), 0, 1))
        [(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)]
        sage: list(_ellipsoid_points(matrix(QQ, [[1]]), vector(QQ, [0]), 1, 0))
        []

    A shifted centre (a background charge) and a non-diagonal form, against
    the bounding box::

        sage: from itertools import product
        sage: Q = matrix(QQ, [[1, -1/2, 0], [-1/2, 1, -1/2], [0, -1/2, 1]])
        sage: b = vector(QQ, [1/3, -2, 1/5]); bound = 4
        sage: box = [vector(ZZ, m) for m in product(range(-8, 9), repeat=3)
        ....:        if vector(ZZ, m) * Q * vector(ZZ, m) + b * vector(ZZ, m) <= bound]
        sage: list(_ellipsoid_points(Q, b, 0, bound)) == box
        True

    `E_8` at norm 4 (the 240 roots, the 2160 vectors of norm 4 and 0)::

        sage: B = CartanMatrix(['E', 8])
        sage: len(list(_ellipsoid_points(B / 2, vector(QQ, [0] * 8), 0, 2)))
        2401
    """
    d = Q.nrows()
    if d == 0:
        if c <= bound:
            yield vector(ZZ, [])
        return
    Qi = Q.inverse()
    mstar = -Qi * b / 2
    fmin = mstar * Q * mstar + b * mstar + c
    R = bound - fmin
    if R < 0:
        return
    if all(x in ZZ for x in mstar):
        # centred at a lattice point: PARI's Fincke--Pohst (qfminim)
        yield from _centred_ellipsoid_points(Q, vector(ZZ, mstar), R)
        return
    D, U = _upper_ldl(Q)
    m = [0] * d
    found = []

    def search(i, rem):
        # the coordinates i+1, ..., d-1 are chosen; D_i (m_i - z)^2 <= rem
        z = mstar[i] - sum(U[i][j] * (m[j] - mstar[j]) for j in range(i + 1, d))
        r = sqrt(float(rem / D[i]))
        for v in range(floor(float(z) - r) - 1, ceil(float(z) + r) + 2):
            t = D[i] * (v - z) ** 2
            if t <= rem:
                m[i] = v
                if i:
                    search(i - 1, rem - t)
                else:
                    found.append(tuple(m))

    search(d - 1, R)
    found.sort()
    for p in found:
        mv = vector(ZZ, p)
        if mv * Q * mv + b * mv + c <= bound:
            yield mv


def _centred_ellipsoid_points(Q, m0, R):
    r"""
    The integer vectors ``m`` with ``(m - m0)^T Q (m - m0) <= R`` for a
    positive definite rational ``Q``, an integer vector ``m0`` and
    ``R >= 0``, sorted lexicographically: PARI's ``qfminim`` (exact
    Fincke--Pohst for an integral form; it returns one of ``±x`` and omits
    ``0``) on ``s Q``, ``s`` the common denominator of ``Q``.

    Sage's ``QuadraticForm.short_vector_list_up_to_length`` is the same
    enumeration; it is called directly here to avoid building a
    ``QuadraticForm``.

    TESTS:

    The same points as the general enumeration (shifted centres), and
    `E_8` at norm 4::

        sage: from vertex_algebras.weight_spaces import (_centred_ellipsoid_points,
        ....:                                             _ellipsoid_points)
        sage: Q = matrix(QQ, [[1, -1/2, 0], [-1/2, 1, -1/2], [0, -1/2, 1]])
        sage: m0 = vector(ZZ, [1, -2, 0])
        sage: P = list(_centred_ellipsoid_points(Q, m0, 5/2)); len(P)
        19
        sage: all((m - m0) * Q * (m - m0) <= 5/2 for m in P)
        True
        sage: from itertools import product
        sage: box = sorted(tuple(vector(ZZ, x) + m0) for x in product(range(-4, 5), repeat=3)
        ....:              if vector(ZZ, x) * Q * vector(ZZ, x) <= 5/2)
        sage: [tuple(m) for m in P] == box
        True
        sage: B = CartanMatrix(['E', 8])
        sage: len(list(_centred_ellipsoid_points(B / 2, vector(ZZ, [0] * 8), 2)))
        2401
        sage: list(_centred_ellipsoid_points(matrix(QQ, [[1]]), vector(ZZ, [3]), 0))
        [(3)]
    """
    from sage.arith.functions import lcm
    from sage.libs.pari import pari
    s = lcm([x.denominator() for x in Q.list()])
    M = (s * Q).change_ring(ZZ)
    top = (s * R).floor()
    pts = [tuple(m0)]
    if top > 0:
        for col in pari(M).qfminim(top)[2]:
            x = vector(ZZ, [ZZ(t) for t in col])
            pts.append(tuple(m0 + x))
            pts.append(tuple(m0 - x))
    pts.sort()
    for p in pts:
        yield vector(ZZ, p)


def _upper_ldl(Q):
    r"""
    ``(D, U)`` with ``Q = U^T diag(D) U``, ``U`` unit upper triangular (as
    lists of rows), for a positive definite rational ``Q``.

    TESTS::

        sage: from vertex_algebras.weight_spaces import _upper_ldl
        sage: Q = matrix(QQ, [[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
        sage: D, U = _upper_ldl(Q); D
        [2, 3/2, 4/3]
        sage: Um = matrix(QQ, U); Um.transpose() * diagonal_matrix(D) * Um == Q
        True
    """
    d = Q.nrows()
    D = [QQ.zero()] * d
    U = [[QQ.zero()] * d for _ in range(d)]
    for i in range(d):
        D[i] = Q[i, i] - sum(D[k] * U[k][i] ** 2 for k in range(i))
        U[i][i] = QQ.one()
        for j in range(i + 1, d):
            U[i][j] = (Q[i, j] - sum(D[k] * U[k][i] * U[k][j] for k in range(i))) / D[i]
    return D, U


def _box_shell(d, s):
    """The integer vectors of length ``d`` with max-norm exactly ``s`` (the
    stages of a staged lattice enumeration), as tuples.

    TESTS::

        sage: from vertex_algebras.weight_spaces import _box_shell
        sage: list(_box_shell(2, 1))
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        sage: list(_box_shell(0, 0)), list(_box_shell(0, 1)), list(_box_shell(1, 0))
        ([()], [], [(0,)])
    """
    if d == 0:
        if s == 0:
            yield ()
        return
    for m in _iproduct(range(-s, s + 1), repeat=d):
        if max(abs(x) for x in m) == s:
            yield m


class _KeyEnumeration:
    """
    Basis keys of a weight space.  ``finite`` (bool); for a finite space
    ``factory()`` returns a fresh iterator over all keys, for an infinite one
    ``stage(s)`` returns the (finitely many) keys of stage ``s = 0, 1, ...``,
    every key occurring in exactly one stage.

    TESTS::

        sage: from vertex_algebras.weight_spaces import _KeyEnumeration
        sage: K = _KeyEnumeration(False, stage=lambda s: [s, -s] if s else [0])
        sage: K.finite, K.first_within(2)
        (False, 0)
    """
    def __init__(self, finite, factory=None, stage=None):
        """
        TESTS::

            sage: from vertex_algebras.weight_spaces import _KeyEnumeration
            sage: K = _KeyEnumeration(True, factory=lambda: iter([3, 4]))
            sage: K.finite, K._stage is None
            (True, True)
        """
        self.finite = finite
        self._factory = factory
        self._stage = stage

    def factory(self):
        """
        A fresh iterator over all keys (stage by stage when infinite).

        TESTS::

            sage: from vertex_algebras.weight_spaces import _KeyEnumeration
            sage: from itertools import islice
            sage: K = _KeyEnumeration(False, stage=lambda s: [s, -s] if s else [0])
            sage: list(islice(K.factory(), 5))
            [0, 1, -1, 2, -2]
            sage: list(_KeyEnumeration(True, factory=lambda: iter([3, 4])).factory())
            [3, 4]
        """
        if self._factory is not None:
            return self._factory()
        return (k for s in count() for k in self._stage(s))

    def first_within(self, stages):
        """A key found within the first ``stages`` stages, or ``None`` (a
        finite enumeration: its first key).

        TESTS::

            sage: from vertex_algebras.weight_spaces import _KeyEnumeration
            sage: K = _KeyEnumeration(False, stage=lambda s: [s] if s > 2 else [])
            sage: K.first_within(3), K.first_within(4)
            (None, 3)
            sage: _KeyEnumeration(True, factory=lambda: iter([])).first_within(3) is None
            True
        """
        if self._stage is None:
            for k in self._factory():
                return k
            return None
        for s in range(stages):
            for k in self._stage(s):
                return k
        return None


class WeightSpace:
    r"""
    The subspace of a vertex algebra spanned by the basis monomials of a
    given conformal weight (and, optionally, a given grade), returned by
    :meth:`vertex_algebra_categories.ProtoVertexAlgebras.Graded.ParentMethods.weight_space`.

    ``len``, :meth:`basis_keys`, :meth:`basis` and :meth:`dimension` need a
    finite-dimensional space; iterating over it (``for x in S``) or over
    :meth:`iter_basis_keys` works in both cases (lazily when infinite).

    INPUT:

    - ``V`` -- the vertex algebra
    - ``weight`` -- the conformal weight
    - ``horizontal_grade`` -- a grade (coordinate tuple), or ``None`` for all
      grades
    - ``enum`` -- a ``_KeyEnumeration`` of the basis keys

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
        sage: S = V.weight_space(4); S
        Weight space of conformal weight 4 of Universal vertex algebra on generators ['L']
         over Rational Field (dimension 2)
        sage: S.basis()
        [:L L:, d^2L]
        sage: V.gen('L').no(V.gen('L')) in S, V.gen('L') in S
        (True, False)

    An infinite-dimensional weight space (the ``βγ`` system with ``γ`` of
    weight 0), enumerated lazily, and a finite graded piece of it::

        sage: from vertex_algebras.ghost_systems import BosonicGhosts
        sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
        sage: S = B.weight_space(1); S.dimension()
        +Infinity
        sage: from itertools import islice
        sage: list(islice(S, 4))
        [dgamma, beta, :dgamma gamma:, :beta gamma:]
        sage: B.weight_space(1, horizontal_grade=0).basis()
        [:beta gamma:]
    """
    def __init__(self, V, weight, horizontal_grade, enum):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: S._V is V, S._weight, S._grade
            (True, 4, None)
        """
        self._V = V
        self._weight = weight
        self._grade = horizontal_grade
        self._enum = enum
        self._keys = None

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: S._repr_() == repr(S)
            True
        """
        return repr(self)

    def __repr__(self):
        """
        The dimension is ``+Infinity`` for an infinite space with a key found,
        ``unknown`` if none was found within ``NONEMPTY_SEARCH_STAGES``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: S
            Weight space of conformal weight 4 of Universal vertex algebra on generators ['L']
             over Rational Field (dimension 2)
            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: B.weight_space(1, horizontal_grade=0)
            Weight space of conformal weight 1 and grade 0 of Universal vertex algebra on
             generators ['beta', 'gamma'] over Rational Field (dimension 1)
            sage: B.weight_space(1)
            Weight space of conformal weight 1 of ... (dimension +Infinity)
        """
        g = '' if self._grade is None else ' and grade {}'.format(self._grade)
        if self._enum.finite:
            dim = self.dimension()
        else:
            dim = ('+Infinity' if self._first_key() is not None
                   else 'unknown')
        return 'Weight space of conformal weight {}{} of {} (dimension {})'.format(
            self._weight, g, self._V, dim)

    def ambient(self):
        """
        The vertex algebra.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: V.weight_space(3).ambient() is V
            True
        """
        return self._V

    def weight(self):
        """
        The conformal weight of this space.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: V.weight_space(3).weight()
            3
        """
        return self._weight

    def horizontal_grade(self):
        """
        The grade of this space (``None`` if not restricted to a grade).

        EXAMPLES::

            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: B.weight_space(1, horizontal_grade=0).horizontal_grade()
            0
            sage: B.weight_space(1).horizontal_grade() is None
            True
        """
        return self._grade

    def is_finite_dimensional(self):
        r"""
        True if the space is finite dimensional.

        For a space without a finiteness certificate this searches for an
        element (the space is then infinite dimensional); if none is found
        within the first stages a ``NotImplementedError`` is raised.

        EXAMPLES::

            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: B.weight_space(1).is_finite_dimensional()
            False
            sage: B.weight_space(1, horizontal_grade=0).is_finite_dimensional()
            True
        """
        if self._enum.finite:
            return True
        if self._first_key() is None:
            raise NotImplementedError(
                'could not decide whether this weight space is empty (no '
                'finiteness certificate and no element found)')
        return False

    def _first_key(self):
        """
        A basis key found within ``NONEMPTY_SEARCH_STAGES`` stages, or
        ``None``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: S._first_key() in S.basis_keys()
            True
            sage: V.weight_space(1)._first_key() is None
            True
        """
        return self._enum.first_within(NONEMPTY_SEARCH_STAGES)

    def dimension(self):
        """
        The dimension (an integer, or ``+Infinity``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: [V.weight_space(n).dimension() for n in range(9)]
            [1, 0, 1, 1, 2, 2, 4, 4, 7]
            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: B.weight_space(1).dimension()
            +Infinity
        """
        if self.is_finite_dimensional():
            return len(self.basis_keys())
        return Infinity

    def __len__(self):
        """
        The dimension (finite spaces only).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: len(S), len(V.weight_space(1))
            (2, 0)
            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: len(BosonicGhosts(QQ, beta_weight=1).weight_space(1))
            Traceback (most recent call last):
            ...
            TypeError: the weight space is infinite dimensional
        """
        if not self.is_finite_dimensional():
            raise TypeError('the weight space is infinite dimensional')
        return len(self.basis_keys())

    def basis_keys(self):
        """
        The sorted list of basis keys (finite-dimensional spaces only).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: V.weight_space(4).basis_keys()      # :L L:, d^2L
            [((0, 0), (0, 0)), ((0, 2),)]

        TESTS::

            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: B.weight_space(1).basis_keys()
            Traceback (most recent call last):
            ...
            ValueError: the weight space is infinite dimensional; use iter_basis_keys()
        """
        if not self.is_finite_dimensional():
            raise ValueError('the weight space is infinite dimensional; use '
                             'iter_basis_keys()')
        if self._keys is None:
            self._keys = sorted(set(self._enum.factory()))
        return list(self._keys)

    def iter_basis_keys(self):
        """
        An iterator over the basis keys (lazy if infinite dimensional).

        EXAMPLES::

            sage: from vertex_algebras.ghost_systems import BosonicGhosts
            sage: B = BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')
            sage: it = B.weight_space(1).iter_basis_keys()
            sage: [next(it) for _ in range(3)]       # dgamma, beta, :dgamma gamma:
            [((1, 1),), ((0, 0),), ((1, 1), (1, 0))]
        """
        if self._enum.finite:
            return iter(self.basis_keys())
        return self._enum.factory()

    def basis(self):
        """
        The basis monomials as elements (finite-dimensional spaces only).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: V.weight_space(6).basis()
            [:L L L:, :dL dL:, :d^2L L:, d^4L]
        """
        return [self._V.monomial(k) for k in self.basis_keys()]

    def __iter__(self):
        """
        The basis monomials, lazily.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: list(S)
            [:L L:, d^2L]
        """
        return (self._V.monomial(k) for k in self.iter_basis_keys())

    def __contains__(self, x):
        """
        Whether ``x`` (converted into the vertex algebra) is zero or
        homogeneous of this weight (and grade).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); S = V.weight_space(4)
            sage: L = V.gen('L'); L.no(L) + L.derivative(2) in S, L in S, 0 in S, 'x' in S
            (True, False, True, False)
            sage: L + L.no(L) in S
            False
        """
        V = self._V
        try:
            x = V(x)
        except (TypeError, ValueError):
            return False
        if x.is_zero():
            return True
        try:
            if V.degree(x) != self._weight:
                return False
            if self._grade is not None and V.horizontal_grade(x) != self._grade:
                return False
        except ValueError:
            return False
        return True


class SubspaceWeightSpace:
    r"""
    A finite-dimensional weight space of a graded subspace of a vertex
    algebra that is not spanned by basis keys: of a vertex subalgebra (see
    :meth:`vertex_subalgebra.VertexSubalgebra.weight_space`) or of an ideal
    (see :meth:`vertex_ideal.FinitelyGeneratedVertexIdeal.weight_space`).

    INPUT:

    - ``owner`` -- the subalgebra or ideal (for :meth:`ambient` and printing)
    - ``parent`` -- the vertex algebra the basis elements belong to
    - ``weight`` -- the conformal weight of the space
    - ``horizontal_grade`` -- the grade of the space, or ``None``
    - ``echelon`` -- a ``_keyed_spans.KeyedSpan`` of the
      space (in the coordinates of the root algebra)
    - ``to_dict`` -- a function converting an element to a raw dict in those
      coordinates (``None`` if it cannot be converted)
    - ``element`` -- (default: ``None``) a function mapping a label of
      ``echelon`` to the element it stands for.  With it, the basis consists
      of the labelled vectors (for a subalgebra: normally ordered monomials
      of the strong generators, :meth:`basis_words` their words); without
      it, the basis is the reduced echelon form (canonical when ``echelon``
      has a key order), and :meth:`leading_monomials` are its pivots.

    EXAMPLES:

    The Virasoro subalgebra of the Heisenberg algebra; the basis consists of
    normally ordered monomials in ``T``::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
        sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
        sage: W = S.weight_space(4); W.dimension()
        2
        sage: W.basis()                    # d^2T, :T T:
        [:dJ dJ: + :d^2J J:, 1/4*:J J J J: + 1/2*:d^2J J:]

    A weight space of an ideal; the basis is the reduced echelon form::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
        sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
        sage: B = I.weight_space(5); B.basis()
        [:dL L: - 1/15*d^3L]
        sage: B.leading_monomials()         # :dL L:
        [((0, 1), (0, 0))]
    """
    def __init__(self, owner, parent, weight, horizontal_grade, echelon, to_dict,
                 element=None):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: B = I.weight_space(5)
            sage: B.ambient() is I, B._P is V, B._element is None
            (True, True, True)
        """
        self._owner = owner
        self._P = parent
        self._weight = weight
        self._grade = horizontal_grade
        self._ech = echelon
        self._to_dict = to_dict
        self._element = element

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: W = S.weight_space(4)
            sage: W._repr_() == repr(W)
            True
        """
        return repr(self)

    def __repr__(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: W = S.weight_space(4)
            sage: W
            Weight space of conformal weight 4 of ... (dimension 2)
        """
        g = '' if self._grade is None else ' and grade {}'.format(self._grade)
        return 'Weight space of conformal weight {}{} of {} (dimension {})'.format(
            self._weight, g, self._owner, self.dimension())

    def ambient(self):
        """
        The subalgebra or ideal this is a weight space of.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: S.weight_space(4).ambient() is S
            True
        """
        return self._owner

    def weight(self):
        """
        The conformal weight of this space.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(5).weight()
            5
        """
        return self._weight

    def horizontal_grade(self):
        """
        The grade of this space (``None`` if not restricted to a grade).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(5).horizontal_grade() is None
            True
        """
        return self._grade

    def is_finite_dimensional(self):
        """
        Always True.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(5).is_finite_dimensional()
            True
        """
        return True

    def dimension(self):
        """
        The dimension.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: [I.weight_space(n).dimension() for n in range(9)]
            [0, 0, 0, 0, 1, 1, 2, 2, 4]
        """
        return self._ech.rank()

    def __len__(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: W = S.weight_space(4)
            sage: len(W)
            2
        """
        return self.dimension()

    def basis_words(self):
        """
        The labels of the basis vectors.

        For a subalgebra: the words of the normally ordered monomials,
        letters ``(strong generator index, derivative)``.  Not available for
        ideals (their basis is the reduced echelon form).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: S.weight_space(4).basis_words()     # d^2T, :T T:
            [((0, 2),), ((0, 0), (0, 0))]

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(5).basis_words()
            Traceback (most recent call last):
            ...
            TypeError: this basis is the reduced echelon form; see leading_monomials()
        """
        if self._element is None:
            raise TypeError('this basis is the reduced echelon form; see '
                            'leading_monomials()')
        return self._ech.labels()

    def leading_monomials(self):
        """
        The pivot keys of the reduced echelon form.

        For an ideal: the leading monomials of its elements of this weight.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(6).leading_monomials()    # :L L L:, :d^2L L:
            [((0, 0), (0, 0), (0, 0)), ((0, 2), (0, 0))]
        """
        return self._ech.pivots()

    def basis(self):
        """
        The basis, as elements of the vertex algebra (see the class
        documentation).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(4).basis()
            [:L L: - 3/10*d^2L]
        """
        if self._element is not None:
            return [self._element(w) for w in self.basis_words()]
        return [self._P._el_from_dict(r) for r in self._ech.rows()]

    def basis_keys(self):
        """
        Not available: such a space is not spanned by basis keys.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: I = V.ideal([L.no(L) - 3/10*L.derivative(2)])
            sage: I.weight_space(4).basis_keys()
            Traceback (most recent call last):
            ...
            TypeError: this weight space is not spanned by basis keys; use basis()
        """
        raise TypeError('this weight space is not spanned by basis keys; use '
                        'basis()')

    def __iter__(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: W = S.weight_space(4)
            sage: list(W) == W.basis()
            True
        """
        return iter(self.basis())

    def coordinates(self, x):
        """
        The coefficients of ``x`` in :meth:`basis`, as a list.

        INPUT:

        - ``x`` -- an element of the vertex algebra

        OUTPUT:

        A list of coefficients in the base ring, one per element of
        :meth:`basis`.  A ``ValueError`` if ``x`` is not in this space.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: T = S.gen('T'); W = S.weight_space(4)
            sage: W.coordinates(T.no(T) + 2*T.derivative(2))
            [2, 1]
            sage: W.coordinates(J.no(J.derivative(2)))
            Traceback (most recent call last):
            ...
            ValueError: :d^2J J: is not in Weight space of conformal weight 4 of ...
        """
        d = self._to_dict(x)
        if d is None or not self._ech.contains(d):
            raise ValueError('{} is not in {}'.format(x, self))
        zero = self._P.base_ring().zero()
        if self._element is None:
            return [d.get(p, zero) for p in self._ech.pivots()]
        c = self._ech.coordinates(d)
        return [c.get(w, zero) for w in self.basis_words()]

    def __contains__(self, x):
        """
        Membership in the span (exact linear algebra).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: J = H.gen('J'); S = H.subalgebra([J.no(J)/2], names=['T'])
            sage: W = S.weight_space(4)
            sage: T = S.gen('T'); T.no(T) in W, J.no(J.derivative(2)) in W, T in W
            (True, False, False)
        """
        d = self._to_dict(x)
        return d is not None and self._ech.contains(d)
