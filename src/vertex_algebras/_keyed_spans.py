# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Keyed spans: growing spans of vectors given as raw dicts ``{key: coefficient}``.

:class:`KeyedSpan` maintains the span of a growing list of vectors over a
field, with the vectors given as raw dicts over arbitrary hashable keys
(the basis keys of a vertex algebra): which inserted vectors are
independent (their labels), membership, coordinates in the independent
vectors (certificates), and, with a key order, the reduced row echelon
form with leading monomials as pivots (canonical normal forms).  The set
of keys may grow with every new vector, and the span is kept in blocks (the
connected components of its keys), each eliminated sparsely or densely by
its own fill-in.  It backs the spans of :mod:`vertex_subalgebra`,
:mod:`vertex_ideal`, :mod:`quotient_vertex_algebra` and
:mod:`commutant_vertex_algebra`.

EXAMPLES::

    sage: from vertex_algebras._keyed_spans import KeyedSpan
    sage: E = KeyedSpan(QQ)
    sage: E.insert({'a': 1, 'b': 2}, label='u')
    True
    sage: E.insert({'b': 1, 'c': 1}, label='v')
    True
    sage: E.insert({'a': 1, 'b': 4, 'c': 2}, label='w')     # u + 2 v
    False
    sage: E.rank(), E.labels()
    (2, ['u', 'v'])
    sage: E.contains({'a': 2, 'b': 6, 'c': 2})
    True
    sage: E.reduce({'c': 1})
    {'c': 1}
    sage: E.coordinates({'a': 2, 'b': 6, 'c': 2})
    {'u': 2, 'v': 2}
    sage: E.coordinates({'c': 1}) is None
    True

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""


#: Batches of at least this many vectors (over ``QQ``) may switch a
#: :class:`KeyedSpan` to its dense backend (FLINT reduced echelon forms),
#: see :meth:`KeyedSpan.insert_many`.
DENSE_MIN = 48

#: Such a batch starts with incremental (sparse) elimination and switches to
#: the dense backend once the reduced rows and their combinations fill at
#: least this fraction of a dense matrix (rank x (columns + rank)): fill-in
#: is what makes sparse elimination slow, and dense elimination costs the
#: same whatever the fill.  0 switches at once (always dense), > 1 never.
DENSE_FILL = 0.05

#: A dense block whose span has codimension at most this in its keys (with
#: the new keys of a batch) tests new vectors against an exact basis of the
#: annihilator of the span (see :meth:`_Block._annihilator_insert_many`)
#: instead of eliminating the whole matrix again.  0 turns this off.
#: 512 (2026-11-04, was 128): the level-2 Zhu algebra of the triplet
#: algebra W(3) (spans of codimension 110-180 in up to 7800 keys) 3.5x
#: faster with half the memory (degree bound 24: 399 -> 114 s, 4.4 -> 2.0
#: GB; 1024 / 2048 are slower again); the W(2) induction closures unchanged
#: or faster (P(1) to depth 4: 17.5 -> 15.5 s).
ANNIHILATOR_MAX = 512

#: :meth:`_Block._annihilator_insert_many` multiplies the batch by the
#: annihilator in chunks of rows with about this many entries.
ANNIHILATOR_CHUNK = 1 << 22

#: :meth:`_Block.reduce_many` reduces this many vectors per matrix product.
REDUCE_CHUNK = 256

#: The fill is measured after 4 and 8 vectors of a batch (dense batches
#: switch early) and then every this many.
FILL_CHECK = 16

#: :func:`left_kernel` restarts densely at this (lower) fill: its dense path
#: is one echelon form (no echelon form of the span to build).  Measured on
#: the coset of ``L_2(sl_2)`` in ``V_{A_1 ⊕ A_1}`` at weight 8 (1049 images
#: on 4480 keys): fill 2% after 384 vectors (0.1 s), 5% after ~880 (2-3 s);
#: the dense kernel takes 0.9 s, all sparse 6.8 s.
KERNEL_DENSE_FILL = 0.02

#: Primes (below the limit of Sage's ``Matrix_modn_dense_double``) for the
#: column rank profile of a dense block (see :func:`_column_profile`).
PROFILE_PRIMES = (94906249, 94906247, 94906219)


def _column_profile(M):
    r"""
    Columns ``P`` of the matrix ``M`` over ``QQ`` (independent rows) with
    ``M[:, P]`` invertible: the pivot columns of an echelon form of ``M``
    modulo a prime of :data:`PROFILE_PRIMES`, or ``None`` if every prime
    loses rank.

    Exact: ``M[:, P]`` is invertible mod p (after clearing denominators),
    so its determinant is nonzero.  With high probability ``P`` are also the
    pivots of the reduced echelon form over ``QQ``, but nothing relies on
    that.  Much cheaper than that echelon form, whose entries are huge
    (triplet algebra, weight 29: 964 x 4565, 1.1 s instead of 210 s).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _column_profile
        sage: _column_profile(matrix(QQ, [[1, 2, 3], [2, 4, 7]]))
        [0, 2]
        sage: from vertex_algebras import _keyed_spans as SLA
        sage: old = SLA.PROFILE_PRIMES; SLA.PROFILE_PRIMES = (2,)
        sage: _column_profile(matrix(QQ, [[1, 1], [1, -1]])) is None   # rank 1 mod 2
        True
        sage: SLA.PROFILE_PRIMES = old
    """
    from sage.rings.finite_rings.finite_field_constructor import GF
    n = M.nrows()
    if not n:
        return []
    Z = M._clear_denom()[0]
    for p in PROFILE_PRIMES:
        P = list(Z.change_ring(GF(p)).pivots())
        if len(P) == n:
            return P
    return None


def _pivot_rows(M):
    r"""
    The rows of ``M`` over ``QQ`` independent of the earlier ones (as
    Sage's ``pivot_rows()``), exactly.  A matrix with at least as many rows
    as columns may have full column rank: then the row rank profile ``I``
    modulo a prime of :data:`PROFILE_PRIMES` is the answer (rows
    independent mod p are independent over ``QQ``, and no larger set
    exists), much cheaper than the echelon form of the transpose behind
    ``pivot_rows()`` (a subalgebra span of 2060 vectors on 772 columns over
    QQ(sqrt 5), expanded: 10.1 s -> 1.1 s).  Otherwise ``pivot_rows()``,
    which is fast for wide matrices (the triplet spans: 982 x 4565, 1.5 s).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _pivot_rows
        sage: M = matrix(QQ, [[1, 2, 3], [2, 4, 6], [0, 1, 1], [1, 3, 4]])
        sage: _pivot_rows(M), M.pivot_rows()
        ((0, 2), (0, 2))
        sage: M = matrix(QQ, [[1, 2], [2, 4], [0, 1], [1, 3]])
        sage: _pivot_rows(M), M.pivot_rows()
        ((0, 2), (0, 2))
    """
    from sage.rings.finite_rings.finite_field_constructor import GF
    n, m = M.dimensions()
    if n and n >= m and PROFILE_PRIMES:
        I = M._clear_denom()[0].change_ring(GF(PROFILE_PRIMES[0])).pivot_rows()
        if len(I) == m:
            return tuple(I)
    return M.pivot_rows()


def _kernel_columns(M):
    r"""
    A basis (columns, over ``QQ``) of the right kernel of the matrix ``M``
    over ``QQ`` with independent rows: with columns ``P`` such that
    ``M[:, P]`` is invertible (a rank profile modulo a prime of
    :data:`PROFILE_PRIMES`; exact) and the others ``F``, the columns
    ``(-M[:, P]^{-1} M[:, F]; 1)``, one integral solve (Dixon); FLINT's
    integer kernel if every prime loses rank.  Measured on the dense blocks
    of induced modules and higher Zhu algebras: 348 x 380, 0.48 s
    (``right_kernel_matrix()`` over ``QQ``) / 0.05 s (FLINT) / 0.10 s
    (solve); 250 x 352 of kernel dimension 102: 0.57 / 4.9 / 0.10 s.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _kernel_columns
        sage: M = matrix(QQ, [[1, 1/2, 0], [0, 1, 1]]); K = _kernel_columns(M)
        sage: K.dimensions(), (M * K).is_zero(), K.base_ring()
        ((3, 1), True, Rational Field)
        sage: _kernel_columns(matrix(QQ, 2, 2, 1)).dimensions()
        (2, 0)
        sage: from vertex_algebras import _keyed_spans as SLA
        sage: old = SLA.PROFILE_PRIMES; SLA.PROFILE_PRIMES = (2,)
        sage: M = matrix(QQ, [[2, 1, 0], [0, 2, 1]]); K = _kernel_columns(M)   # FLINT
        sage: SLA.PROFILE_PRIMES = old; K.dimensions(), (M * K).is_zero()
        ((3, 1), True)
    """
    from sage.matrix.constructor import matrix
    from sage.rings.rational_field import QQ
    n, m = M.dimensions()
    if not n:
        return matrix.identity(QQ, m)
    Z = M._clear_denom()[0]
    P = _column_profile(Z)
    if P is None:
        K = Z.right_kernel_matrix(algorithm='flint', basis='computed')
        return K.transpose().change_ring(QQ)
    Ps = set(P)
    F = [j for j in range(m) if j not in Ps]
    X = Z.matrix_from_columns(P).solve_right(Z.matrix_from_columns(F))
    # rows P: -X, rows F: the identity
    K = (-X.change_ring(QQ)).stack(matrix.identity(QQ, len(F)))
    where = {j: i for i, j in enumerate(P + F)}
    return K.matrix_from_rows([where[j] for j in range(m)])


def _addmul(dst, src, c):
    r"""
    ``dst += c * src`` in place, dropping zeros.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _addmul
        sage: d = {'a': 1, 'b': 2}; _addmul(d, {'b': -1, 'c': 3}, 2); d
        {'a': 1, 'c': 6}
    """
    for k, v in src.items():
        w = dst.get(k)
        w = c * v if w is None else w + c * v
        if w:
            dst[k] = w
        else:
            dst.pop(k, None)


def reduce_by_rows(v, rows):
    """
    The remainder of the raw dict ``v`` modulo the rows ``{pivot: row}`` of
    a reduced row echelon form (pivot coefficient 1, no other pivot key in
    any row), as a new dict: one pass suffices.

    INPUT:

    - ``v`` -- a raw dict ``{key: coefficient}``
    - ``rows`` -- a dict ``{pivot: row}`` of raw dicts in reduced row
      echelon form

    OUTPUT:

    A raw dict without zero coefficients and without pivot keys.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import reduce_by_rows
        sage: reduce_by_rows({'a': 2, 'b': 1}, {'a': {'a': 1, 'c': -1}})
        {'b': 1, 'c': 2}
    """
    r = {k: c for k, c in v.items() if c}
    for p in [k for k in r if k in rows]:
        c = r.get(p)
        if c:
            _addmul(r, rows[p], -c)
    return r


class _Block:
    r"""
    One block of a :class:`KeyedSpan`: the reduced row echelon form of
    the span of the inserted vectors whose keys lie in one connected
    component (the former ``KeyedSpan``, unchanged; the blocks of an
    echelon have disjoint key sets).  Rows are ``{pivot: (row, combo)}``,
    ``combo`` keyed by the index into ``_labels``; with ``vector_of`` the
    independent vectors are not stored (the owning echelon passes its own
    lookup, and the labels are its positions).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _Block
        sage: B = _Block(QQ, key_order=lambda k: k)
        sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
        (True, True, False)
        sage: B.rank(), B.labels(), B.pivots(), B.rows()
        (2, ['u', 'v'], [3, 2], [{1: -1, 3: 1}, {1: 1, 2: 1}])
    """
    def __init__(self, R, key_order=None, vector_of=None):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: _Block(QQ)._combos_ok, _Block(QQ['c'].fraction_field())._combos_ok
            (True, False)
            sage: B = _Block(QQ); B.rank(), B._rows, B._dense
            (0, {}, False)
        """
        self._R = R
        self._key_order = key_order
        self._vector_of = vector_of
        self._rows = {}          # pivot key -> (row dict, combo dict or None)
        self._labels = []        # labels of the independent vectors
        self._vecs = []          # the independent vectors (unless vector_of)
        self._dense = False      # dense backend (see insert_many)
        self._K = None           # dense: annihilator (keys, matrix) or None
        self._stale = False      # dense: A, cols, E to be rebuilt (_fresh)
        # the combos (rows as combinations of the independent vectors) are
        # only needed for coordinates: over QQ they are kept (the switch to
        # the dense backend measures them), over other rings computed on the
        # first coordinates() call (_ensure_combos): most of the work of
        # sparse elimination over fraction fields went into them
        from sage.rings.rational_field import QQ
        self._combos_ok = R is QQ

    def rank(self):
        r"""
        The dimension of the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.rank()
            2
        """
        return len(self._labels)

    def _independent(self):
        r"""
        The independent inserted vectors, in order.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._independent()
            [{1: 1, 2: 1}, {2: 1, 3: 1}]
            sage: B2 = _Block(QQ, vector_of={'x': {5: 1}}.get); B2.insert({5: 1}, 'x')
            True
            sage: B2._independent(), B2._vecs
            ([{5: 1}], [])
        """
        if self._vector_of is not None:
            return [self._vector_of(l) for l in self._labels]
        return self._vecs

    def _keep(self, v):
        r"""
        Store the independent vector ``v`` (unless the owner keeps them:
        ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ); B._keep({1: 1}); B._vecs
            [{1: 1}]
            sage: B = _Block(QQ, vector_of=lambda l: {}); B._keep({1: 1}); B._vecs
            []
        """
        if self._vector_of is None:
            self._vecs.append(v)

    # -- dense backend ------------------------------------------------- #

    def _fresh(self):
        r"""
        Rebuild the dense backend (:meth:`_dense_build`) if inserts through
        the annihilator (:meth:`_annihilator_insert_many`) left it stale:
        they need only ``K``, so a closure inserting many batches builds
        the matrix of the independent vectors once, when it is used.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._annihilator_insert_many([{3: 2, 4: 1}], 'x')
            [True]
            sage: D._stale, D.contains({1: 1, 3: 1, 4: 1}), D.contains({4: 1}), D._stale
            (True, True, False, True)
            sage: D.pivots(), D._stale
            ([4, 3, 2], False)
        """
        if self._stale:
            self._stale = False
            self._dense_build(self._independent())

    def _dense_build(self, vecs):
        r"""
        Rebuild the dense backend from the independent vectors ``vecs``
        (raw dicts): columns ordered by decreasing ``key_order`` (so pivots
        are leading monomials), the matrix ``A`` of the vectors and columns
        ``P`` with ``A[:, P]`` invertible (:func:`_column_profile`), which
        is all that membership and coordinates need.  The reduced echelon
        form is computed only when asked for (:meth:`_echelon`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._dense_build([{1: 1, 2: 1}])
            sage: D._cols, D._A, D._pivcols, D._E is None
            ([2, 1], [1 1], [0], True)
        """
        from sage.matrix.constructor import matrix
        keys = set()
        for v in vecs:
            keys.update(v)
        cols = list(keys)
        if self._key_order is not None:
            cols.sort(key=self._key_order, reverse=True)
        else:
            try:
                cols.sort()
            except TypeError:
                pass
        idx = {k: i for i, k in enumerate(cols)}
        A = matrix(self._R, len(vecs), len(cols),
                   {(i, idx[k]): c for i, v in enumerate(vecs)
                    for k, c in v.items() if c}, sparse=False)
        self._cols, self._colidx = cols, idx
        self._A = A              # matrix of the independent vectors
        self._AP = None          # its columns P: invertible (lazy)
        self._E = None           # reduced echelon form (lazy)
        from sage.rings.rational_field import QQ
        self._pivcols = _column_profile(A) if self._R is QQ else None
        if self._pivcols is None:
            self._echelon()

    def _echelon(self):
        r"""
        Dense backend: the reduced echelon form (nonzero rows), computed
        on first use; afterwards ``_pivcols`` are its pivots.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._E is None, D._cols
            (True, [3, 2, 1])
            sage: D._echelon()
            [ 1  0 -1]
            [ 0  1  1]
            sage: D._pivcols
            [0, 1]
        """
        self._fresh()
        if self._E is None:
            self._dense_matrices(solve=False)       # A (after _import)
            E = self._A.echelon_form()
            P = list(E.pivots())
            self._E = E.matrix_from_rows(range(len(P)))
            if P != self._pivcols:
                self._pivcols, self._AP = P, None
        return self._E

    def _dense_vector(self, v):
        r"""
        ``(vector over the columns, extra dict of keys outside)``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._dense_vector({1: 1, 5: 2})
            ((0, 0, 1), {5: 2})
        """
        from sage.modules.free_module_element import vector
        self._fresh()
        idx = self._colidx
        ent, extra = {}, {}
        for k, c in v.items():
            if c:
                j = idx.get(k)
                if j is None:
                    extra[k] = c
                else:
                    ent[j] = c
        return vector(self._R, len(self._cols), ent), extra

    def _dense_reduce(self, v):
        r"""
        Dense backend: :meth:`reduce` by the (lazy) reduced echelon form; keys
        outside the block's columns are kept.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._dense_reduce({3: 1, 7: 1})
            {1: 1, 7: 1}
        """
        x, extra = self._dense_vector(v)
        E = self._echelon()
        if self._pivcols:
            from sage.modules.free_module_element import vector
            coef = vector(self._R, [x[j] for j in self._pivcols])
            x = x - coef * E
        r = {self._cols[j]: c for j, c in x.dict().items() if c}
        r.update(extra)
        return r

    def _key_set(self):
        r"""
        All keys of the block (those of its rows).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._key_set()
            {1, 2, 3}
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._key_set()
            {1, 2, 3}
        """
        if self._dense:
            self._fresh()
            return set(self._cols)
        keys = set()
        for row, _ in self._rows.values():
            keys.update(row)
        return keys

    def insert_many(self, vecs, labels=None, greedy=True):
        r"""
        Insert the vectors ``vecs`` in order (as repeated :meth:`insert`,
        with the given ``labels``); returns the list of booleans.  With
        ``greedy=False`` a dense batch may accept another basis of the same
        span (see :meth:`_dense_insert_many`).

        Over ``QQ`` a large batch (at least ``DENSE_MIN`` vectors) watches
        the fill-in of the incremental elimination and switches to the dense
        backend (FLINT elimination of the independent vectors and the rest
        of the batch) once :meth:`_fill` reaches ``DENSE_FILL``.  Labels,
        and with a ``key_order`` the pivots and rows, do not depend on the
        backend.  A block that switched stays dense.  The fill is only
        meaningful for one connected block: see
        :meth:`KeyedSpan.insert_many`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: from vertex_algebras import _keyed_spans as SLA
            sage: B = _Block(QQ); B.insert_many([{1: 1}, {1: 2}, {2: 1}], 'abc'), B.labels()
            ([True, False, True], ['a', 'c'])
            sage: old = SLA.DENSE_FILL; SLA.DENSE_FILL = 0      # switch at once
            sage: B = _Block(QQ); flags = B.insert_many([{i: 1, i + 1: 1} for i in range(50)])
            sage: all(flags), B._dense, B.rank()
            (True, True, 50)
            sage: SLA.DENSE_FILL = old
        """
        vecs = list(vecs)
        if labels is None:
            labels = [None] * len(vecs)
        else:
            labels = list(labels)
        if self._dense:
            return self._dense_insert_many(vecs, labels, greedy)
        from sage.rings.rational_field import QQ
        if self._R is not QQ or len(vecs) < DENSE_MIN:
            return [self.insert(v, l) for v, l in zip(vecs, labels)]
        keys = set()
        for row, _ in self._rows.values():
            keys.update(row)
        for v in vecs:
            keys.update(v)
        ncols = len(keys)
        out = []
        for i, (v, l) in enumerate(zip(vecs, labels)):
            if ((i % FILL_CHECK == 0 or i in (4, 8))
                    and self._fill(ncols) >= DENSE_FILL):
                return out + self._dense_insert_many(vecs[i:], labels[i:], greedy)
            out.append(self.insert(v, l))
        return out

    def _fill(self, ncols):
        r"""
        The fraction of the dense matrix ``rank x (ncols + rank)`` filled
        by the reduced rows and their combinations (0 for rank 0).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._fill(3)        # 7 entries (rows and combos) in 2 x (3 + 2)
            7/10
            sage: _Block(QQ)._fill(3)
            0
        """
        r = len(self._rows)
        if not r:
            return 0
        n = sum(len(row) + (len(cb) if cb is not None else 0)
                for row, cb in self._rows.values())
        return n / (r * (ncols + r))

    def _dense_insert_many(self, vecs, labels, greedy=True):
        r"""
        :meth:`insert_many` with the dense backend: the pivot rows of
        the matrix of the independent vectors and ``vecs`` are the
        independent ones (in order); then the dense echelon form.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}, {1: 1, 3: -1}, {3: 2}], 'uvwx')
            [True, True, False, True]
            sage: D._dense, D.labels(), D.pivots()
            (True, ['u', 'v', 'x'], [3, 2, 1])
        """
        from sage.matrix.constructor import matrix
        if not self._dense and self._rows:
            self._K = self._annihilator_of_rows()
        out = self._annihilator_insert_many(vecs, labels)
        if out is not None:
            return out
        old = self._independent()
        cand = list(old) + [{k: c for k, c in v.items() if c} for v in vecs]
        keys = set()
        for v in cand:
            keys.update(v)
        cols = sorted(keys, key=self._key_order) if self._key_order else list(keys)
        idx = {k: i for i, k in enumerate(cols)}
        M = matrix(self._R, len(cand), len(cols),
                   {(i, idx[k]): c for i, v in enumerate(cand)
                    for k, c in v.items()}, sparse=False)
        k0 = len(old)
        if not greedy:
            out = self._certified_insert(M, cand, k0, cols, labels)
            if out is not None:
                return out
        # the shortest prefix whose span has codimension <= half of
        # ANNIHILATOR_MAX modulo a prime (so also over QQ, where the rank
        # is not smaller): eliminate it, test the rest against the
        # annihilator (a tall batch on few keys: 1636 x 412 of rank 404,
        # 3.0 s -> 0.6 s)
        L = len(cand)
        from sage.rings.rational_field import QQ
        if (self._R is QQ and ANNIHILATOR_MAX and len(cols) > ANNIHILATOR_MAX
                and len(cand) - k0 >= DENSE_MIN):
            from sage.rings.finite_rings.finite_field_constructor import GF
            I = M._clear_denom()[0].change_ring(GF(PROFILE_PRIMES[0])).pivot_rows()
            t = len(cols) - ANNIHILATOR_MAX // 2
            if len(I) >= t:
                L = max(I[t - 1] + 1, k0 + 1)
        if L < len(cand):
            M = M.matrix_from_rows(range(L))
        prows = set(_pivot_rows(M))
        out = []
        for i, (v, l) in enumerate(zip(cand[k0:L], labels)):
            ok = (k0 + i) in prows
            out.append(ok)
            if ok:
                self._keep(v)
                self._labels.append(l)
        rest = cand[L:]
        del cand, old, M
        self._dense = True
        self._rows = {}
        self._K = None
        self._dense_build(self._independent())
        if rest:
            more = self._annihilator_insert_many(rest, labels[L - k0:])
            if more is None:
                more = self._dense_insert_many(rest, labels[L - k0:])
            out += more
        return out

    def _certified_insert(self, M, cand, k0, cols, labels):
        r"""
        :meth:`_dense_insert_many` without the greedy choice (``greedy=
        False``): ``M`` the dense matrix of ``cand`` (the ``k0`` independent
        vectors of the block, then the batch) on the keys ``cols``.  Accepts
        the batch vectors among the rows ``I`` independent modulo a prime of
        :data:`PROFILE_PRIMES` (independent over ``QQ`` too), once an exact
        basis ``K`` of the annihilator of the rows ``I`` kills every other
        row (so they lie in the span of ``I``).  Returns ``None`` (nothing
        changed) if some prime does not keep the old vectors or the check
        fails.  Then the span is that of all of ``cand``, exactly; only
        which vectors are accepted may differ from the greedy choice (for a
        prime dividing some minors).  The exact greedy choice of a batch
        of nearly full rank costs an echelon form over ``QQ``: the level-2
        Zhu algebra of the triplet algebra W(3), degree bound 26, spent 268
        of 496 s in six of them.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: vs = [{i: 1, i + 1: 1} for i in range(60)] + [{0: 1, 2: -1}, {0: 1, 61: 1}]
            sage: flags = D._dense_insert_many(vs, range(62), greedy=False)
            sage: flags.count(True), D.rank(), D.contains({0: 1, 61: 1}), D._stale
            (61, 61, True, True)
            sage: D.pivots()[:3]
            [61, 60, 59]
        """
        from sage.rings.finite_rings.finite_field_constructor import GF
        from sage.rings.rational_field import QQ
        if self._R is not QQ or len(cand) == k0:
            return None
        Z = M._clear_denom()[0]
        for p in PROFILE_PRIMES:
            I = Z.change_ring(GF(p)).pivot_rows()
            if len(I) >= k0 and tuple(I[:k0]) == tuple(range(k0)):
                break
        else:
            return None
        Iset = set(I)
        K = _kernel_columns(M.matrix_from_rows(I))
        rej = [i for i in range(len(cand)) if i not in Iset]
        step = max(1, ANNIHILATOR_CHUNK // max(1, len(cols)))
        for a in range(0, len(rej), step):
            if not (M.matrix_from_rows(rej[a:a + step]) * K).is_zero():
                return None
        out = []
        for i, l in zip(range(k0, len(cand)), labels):
            ok = i in Iset
            out.append(ok)
            if ok:
                self._keep(cand[i])
                self._labels.append(l)
        self._dense = True
        self._rows = {}
        # K's rows are the columns of M
        self._K = (list(cols), K) if K.ncols() <= ANNIHILATOR_MAX else None
        self._stale = True
        return out

    def _annihilator_of_rows(self):
        r"""
        ``(keys, K)``: an exact basis ``K`` (columns) of the annihilator of
        the span of the sparse reduced rows on their keys, one column
        ``e_j - Σ_p row_p[j] e_p`` per non-pivot key ``j``, or ``None``
        when there are more than :data:`ANNIHILATOR_MAX` of them (or the
        ring is not ``QQ``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v')
            (True, True)
            sage: keys, K = B._annihilator_of_rows(); keys, K.transpose()
            ([2, 3, 1], [-1  1  1])
        """
        from sage.matrix.constructor import matrix
        from sage.rings.rational_field import QQ
        if self._R is not QQ:
            return None
        piv = list(self._rows)
        keys = set()
        for row, _ in self._rows.values():
            keys.update(row)
        free = [k for k in keys if k not in self._rows]
        if len(free) > ANNIHILATOR_MAX:
            return None
        if self._key_order is not None:
            free.sort(key=self._key_order, reverse=True)
        kkeys = piv + free
        np = len(piv)
        K = matrix(QQ, len(kkeys), len(free),
                   {(i, j): -row[f] for i, p in enumerate(piv)
                    for j, f in enumerate(free)
                    for row in (self._rows[p][0],) if f in row}, sparse=False)
        for j in range(len(free)):
            K[np + j, j] = 1
        return (kkeys, K)

    def _annihilator_insert_many(self, vecs, labels):
        r"""
        :meth:`_dense_insert_many` for a dense block of small codimension:
        with ``K`` an exact basis (columns) of the annihilator of the span
        on its keys, extended by the identity on the new keys of ``vecs``,
        ``v -> v K`` has kernel exactly the span, so ``v`` is independent of
        the span and of the earlier accepted vectors iff ``v K`` is
        independent of their images: the pivot rows of the small matrix of
        the images (exact, the same choice as the elimination of the whole
        matrix).  ``K`` is then replaced by ``K`` times the kernel of the
        accepted images.  Returns ``None`` (nothing changed) when the
        codimension exceeds :data:`ANNIHILATOR_MAX` or the ring is not
        ``QQ``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._annihilator_insert_many([{1: 1, 3: -1}, {3: 2, 4: 1}, {4: 1}, {1: 1, 3: 2}], 'wxyz')
            [False, True, True, False]
            sage: D.labels(), D.pivots(), D._K[1].ncols()
            (['u', 'v', 'x', 'y'], [4, 3, 2, 1], 0)

        The same choice as the elimination of the whole matrix::

            sage: from vertex_algebras import _keyed_spans as SLA
            sage: set_random_seed(1)
            sage: vs = [{k: QQ.random_element(3, 3) for k in Subsets(range(12), 4).random_element()}
            ....:       for _ in range(30)]
            sage: B1 = _Block(QQ, key_order=lambda k: k); B2 = _Block(QQ, key_order=lambda k: k)
            sage: B1._dense_insert_many(vs[:9], range(9)) == B2._dense_insert_many(vs[:9], range(9))
            True
            sage: old = SLA.ANNIHILATOR_MAX; SLA.ANNIHILATOR_MAX = 0
            sage: f1 = B1._dense_insert_many(vs[9:], range(9, 30))
            sage: SLA.ANNIHILATOR_MAX = old
            sage: f2 = B2._dense_insert_many(vs[9:], range(9, 30))
            sage: f1 == f2, B1.labels() == B2.labels(), B2._K is not None
            (True, True, True)
        """
        from sage.matrix.constructor import matrix
        from sage.rings.rational_field import QQ
        if self._R is not QQ or not ANNIHILATOR_MAX:
            return None
        vecs = [{k: c for k, c in v.items() if c} for v in vecs]
        if self._K is None:
            if not self._dense:
                return None
            self._fresh()
            cols = list(self._cols)
            if len(cols) - len(self._labels) > ANNIHILATOR_MAX:
                return None
            A = self._A
            if A is None:
                self._dense_build(self._independent())
                A = self._A
            # A's columns are self._cols
            self._K = (cols, _kernel_columns(A))
        kkeys, K = self._K
        kidx = {k: i for i, k in enumerate(kkeys)}
        newkeys = []
        seen = set()
        for v in vecs:
            for k in v:
                if k not in kidx and k not in seen:
                    seen.add(k)
                    newkeys.append(k)
        d, e = K.ncols(), len(newkeys)
        if d + e > ANNIHILATOR_MAX:
            self._K = None
            return None
        nidx = {k: i for i, k in enumerate(newkeys)}
        # images v K: the old keys through K, the new keys by the identity;
        # V K in chunks of rows (a dense V of a whole batch was the memory
        # peak of large closures: ~10^5 vectors on ~10^4 keys)
        step = max(1, ANNIHILATOR_CHUNK // max(1, len(kkeys)))
        parts = []
        for a in range(0, len(vecs), step):
            chunk = vecs[a:a + step]
            V = matrix(QQ, len(chunk), len(kkeys),
                       {(i, kidx[k]): c for i, v in enumerate(chunk)
                        for k, c in v.items() if k in kidx}, sparse=False)
            parts.append(V * K)
        VK = (matrix.block([[P] for P in parts], subdivide=False) if len(parts) > 1
              else parts[0] if parts else matrix(QQ, 0, K.ncols()))
        Phi = VK.augment(matrix(QQ, len(vecs), e,
                   {(i, nidx[k]): c for i, v in enumerate(vecs)
                    for k, c in v.items() if k in nidx}, sparse=False))
        prows = Phi.pivot_rows()
        acc = set(prows)
        # a vector with a new key is independent (its image has a unit
        # coordinate there), so the new keys all belong to accepted vectors
        Knew = K
        if prows:
            Kx = K.block_sum(matrix.identity(QQ, e)) if e else K
            Knew = Kx * _kernel_columns(Phi.matrix_from_rows(prows))
            if Knew.ncols():
                # primitive integral columns (the entries stay small)
                from sage.arith.misc import gcd
                Z = Knew.transpose()._clear_denom()[0]
                Knew = matrix(QQ, [r / gcd(r.list()) for r in Z.rows()]).transpose()
            kkeys = kkeys + newkeys
        out = []
        for i, (v, l) in enumerate(zip(vecs, labels)):
            ok = i in acc
            out.append(ok)
            if ok:
                self._keep(v)
                self._labels.append(l)
        self._K = (kkeys, Knew)
        if prows or not self._dense:
            self._dense = True
            self._rows = {}
            self._stale = True
        return out

    def _export(self):
        r"""
        The state of ``self`` as plain data (for saving to a file; the
        ring, ``key_order`` and ``vector_of`` are not included: the owner
        supplies them again to :meth:`_import`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._export()
            {'dense': False,
             'labels': ['u', 'v'],
             'rows': {2: ({1: 1, 2: 1}, {0: 1}), 3: ({1: -1, 3: 1}, {0: -1, 1: 1})},
             'vecs': [{1: 1, 2: 1}, {2: 1, 3: 1}]}
        """
        self._fresh()
        st = {'rows': self._rows, 'labels': self._labels, 'vecs': self._vecs,
              'dense': self._dense}
        if self._dense:
            # E only if computed (None: rebuilt on demand; pivcols are then
            # columns with A[:, P] invertible, see _dense_build)
            st.update(cols=self._cols, pivcols=self._pivcols, E=self._E)
        return st

    def _import(self, st):
        r"""
        Restore a state returned by :meth:`_export` (into a fresh
        echelon with the same ring, ``key_order`` and ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B2 = _Block(QQ, key_order=lambda k: k); B2._import(B._export())
            sage: B2.rows() == B.rows(), B2.labels(), B2.coordinates({3: 1, 2: 1})
            (True, ['u', 'v'], {'v': 1})
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D2 = _Block(QQ, key_order=lambda k: k); D2._import(D._export())
            sage: D2._dense, D2._A is None, D2.contains({1: 1, 3: -1})
            (True, True, True)
        """
        self._rows = st['rows']
        self._combos_ok = all(cb is not None for _, cb in self._rows.values())
        self._labels = list(st['labels'])
        self._vecs = list(st['vecs'])
        self._dense = st['dense']
        if self._dense:
            self._cols = list(st['cols'])
            self._colidx = {k: i for i, k in enumerate(self._cols)}
            self._pivcols = list(st['pivcols'])
            self._E = st.get('E')
            self._A = self._AP = None
        self._K = None

    def labels(self):
        r"""
        The labels of the independent inserted vectors, in insertion
        order (a basis of the span).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.labels()
            ['u', 'v']
        """
        return list(self._labels)

    def _reduce(self, v):
        r"""
        ``(r, t)``: ``r = v - Σ_p t_p row_p`` reduced (no pivot keys).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._reduce({3: 1, 1: 1})
            ({1: 2}, {3: 1})
        """
        r = {k: c for k, c in v.items() if c}
        t = {}
        rows = self._rows
        for p in [k for k in r if k in rows]:
            c = r.get(p)
            if not c:
                continue
            _addmul(r, rows[p][0], -c)
            t[p] = c
        return r, t

    def row_dict(self):
        r"""
        The reduced echelon form as ``{pivot: row}`` (raw dicts; see
        :meth:`rows`), e.g. for :func:`reduce_by_rows`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.row_dict()
            {2: {1: 1, 2: 1}, 3: {1: -1, 3: 1}}
        """
        return dict(zip(self.pivots(), self.rows()))

    def reduce(self, v):
        r"""
        The remainder of ``v`` modulo the span (a raw dict containing no
        pivot key); zero (``{}``) iff ``v`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.reduce({3: 1}), B.reduce({1: 1, 3: -1})
            ({1: 1}, {})
        """
        if self._dense:
            return self._dense_reduce(v)
        return self._reduce(v)[0]

    def reduce_many(self, vs):
        r"""
        ``[self.reduce(v) for v in vs]``; a dense block reduces
        ``REDUCE_CHUNK`` vectors at a time by one matrix product with the
        echelon form (one vector at a time took seconds each on the
        7700 x 7800 spans of W(3)'s level-2 Zhu algebra).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D.reduce_many([{3: 1, 7: 1}, {1: 1, 2: 1}, {}])
            [{1: 1, 7: 1}, {}, {}]
        """
        if not self._dense:
            return [self._reduce(v)[0] for v in vs]
        from sage.matrix.constructor import matrix
        self._fresh()
        E = self._echelon()
        piv, cols, idx = self._pivcols, self._cols, self._colidx
        out = []
        for s in range(0, len(vs), REDUCE_CHUNK):
            chunk = vs[s:s + REDUCE_CHUNK]
            ent, extras = {}, []
            for i, v in enumerate(chunk):
                extra = {}
                for k, c in v.items():
                    if c:
                        j = idx.get(k)
                        if j is None:
                            extra[k] = c
                        else:
                            ent[i, j] = c
                extras.append(extra)
            X = matrix(self._R, len(chunk), len(cols), ent)
            if piv:
                X = X - self._times_echelon(X.matrix_from_columns(piv), E)
            for i, extra in enumerate(extras):
                r = {cols[j]: c for j, c in X.row(i).dict().items() if c}
                r.update(extra)
                out.append(r)
        return out

    def _times_echelon(self, C, E):
        r"""
        ``C * E`` for the echelon form ``E`` of a dense block; over QQ as
        an integer product with ``E = E_Z / d`` (``E_Z`` cached: converting
        ``E`` for FLINT's rational product took most of the time of each
        product).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1/2}, {2: 1, 3: 1/3}], ['u', 'v'])
            [True, True]
            sage: E = D._echelon(); C = matrix(QQ, [[1, 2/3], [5, 0]])
            sage: D._times_echelon(C, E) == C * E
            True
        """
        from sage.rings.integer_ring import ZZ
        from sage.rings.rational_field import QQ
        if self._R is not QQ:
            return C * E
        ez = getattr(self, '_EZ', None)
        if ez is None or ez[0] is not E:
            d = E.denominator()
            ez = self._EZ = (E, (d * E).change_ring(ZZ), d)
        dc = C.denominator()
        P = (dc * C).change_ring(ZZ) * ez[1]
        return P.change_ring(QQ) / (dc * ez[2])

    def contains(self, v):
        r"""
        True if ``v`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.contains({1: 1, 3: -1}), B.contains({1: 1})
            (True, False)
        """
        if self._dense:
            if self._K is not None:
                # v is in the span iff v K = 0 (and v has no other keys)
                from sage.modules.free_module_element import vector
                kkeys, K = self._K
                if not K.ncols():
                    kset = self._K_keyset()
                    return all(k in kset for k, c in v.items() if c)
                kidx = self._K_index()
                ent = {}
                for k, c in v.items():
                    if c:
                        j = kidx.get(k)
                        if j is None:
                            return False
                        ent[j] = c
                return not (vector(self._R, len(kkeys), ent) * K)
            return self._dense_solve(v) is not None
        return not self.reduce(v)

    def _K_index(self):
        r"""
        ``{key: row of K}`` of the annihilator ``self._K`` (cached with it).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._annihilator_insert_many([{3: 2, 4: 1}], 'x')
            [True]
            sage: sorted(D._K_index())
            [1, 2, 3, 4]
        """
        c = getattr(self, '_kidx', None)
        if c is None or c[0] is not self._K:
            c = self._kidx = (self._K, {k: i for i, k in enumerate(self._K[0])})
        return c[1]

    def _K_keyset(self):
        r"""
        The keys of the annihilator ``self._K`` (a set).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1}], ['u'])
            [True]
            sage: D._annihilator_insert_many([{2: 1}], 'x'); D._K_keyset()
            [True]
            {1, 2}
        """
        return set(self._K_index())

    def _dense_solve(self, v):
        r"""
        Dense backend: the coefficients (a vector) of ``v`` in the
        independent vectors, or ``None`` if ``v`` is not in the span.  With
        ``A`` the matrix of the independent vectors and ``P`` columns with
        ``A[:, P]`` invertible (the pivots of its echelon form, or the
        cheaper :func:`_column_profile`), the only candidate is ``c = v[P]
        A[:, P]^{-1}``; then ``v`` is in the span iff ``c A = v``.  Much
        faster than solving with all of ``A`` or reducing with the echelon
        form, whose entries are much larger than those of ``A`` (weight 29 of
        the triplet algebra: rank 964, 4565 columns, ~2 s instead of 16 s
        (membership) and 215 s (coordinates)).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D._dense_solve({1: 1, 2: 2, 3: 1}), D._dense_solve({1: 1}), D._dense_solve({5: 1})
            ((1, 1), None, None)
        """
        from sage.modules.free_module_element import vector
        x, extra = self._dense_vector(v)
        if extra:
            return None
        if not self._pivcols:
            return None if x else vector(self._R, 0)
        self._dense_matrices()
        c = self._AP.solve_left(vector(self._R, [x[j] for j in self._pivcols]))
        return c if c * self._A == x else None

    def _dense_matrices(self, solve=True):
        r"""
        Dense backend: build ``A`` (the independent vectors; after
        :meth:`_import`) and, with ``solve``, its invertible columns
        ``A[:, P]`` if needed.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D2 = _Block(QQ, key_order=lambda k: k); D2._import(D._export())
            sage: D2._A is None
            True
            sage: D2._dense_matrices(); D2._A, D2._AP
            (
            [0 1 1]  [0 1]
            [1 1 0], [1 1]
            )
        """
        from sage.matrix.constructor import matrix
        self._fresh()
        if self._A is None:
            idx = self._colidx
            vs = self._independent()
            self._A = matrix(self._R, len(vs), len(self._cols),
                             {(i, idx[k]): c for i, u in enumerate(vs)
                              for k, c in u.items()}, sparse=False)
            self._AP = None
        if solve and self._AP is None:
            self._AP = self._A.matrix_from_columns(self._pivcols)

    def _combo(self, t):
        r"""
        The combination ``Σ_p t_p combo_p`` of the combos of the rows (``t``
        as returned by :meth:`_reduce`): keyed by the indices of the labels.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B._combo({3: 1})              # row 3 = v - u
            {0: -1, 1: 1}
        """
        out = {}
        for p, c in t.items():
            _addmul(out, self._rows[p][1], c)
        return out

    def _ensure_combos(self):
        r"""
        Sparse rows without combos: replay the elimination of the
        independent vectors (in order) keeping them.  The rows come out the
        same (combos never influence them), so only the combos are taken.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: F.<x> = Frac(QQ['x'])
            sage: B = _Block(F); B.insert({1: x, 2: 1}, 'p'), B.insert({2: 1}, 'q')
            (True, True)
            sage: B._rows
            {1: ({1: 1}, None), 2: ({2: 1}, None)}
            sage: B._ensure_combos(); B._rows
            {1: ({1: 1}, {0: 1/x, 1: -1/x}), 2: ({2: 1}, {1: 1})}
        """
        if self._combos_ok or self._dense:
            return
        b = _Block(self._R, key_order=self._key_order)
        b._combos_ok = True
        for v in self._independent():
            if not b.insert(v):
                raise RuntimeError('replaying the elimination lost a vector')
        if set(b._rows) != set(self._rows):
            raise RuntimeError('replaying the elimination changed the pivots')
        self._rows = {p: (row, b._rows[p][1]) for p, (row, _) in self._rows.items()}
        self._combos_ok = True

    def insert(self, v, label=None):
        r"""
        Add ``v`` to the span.  Returns True if the span grew (``v`` is then
        recorded under ``label``), False if ``v`` was already in it.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.insert({4: 1}, 'x'), B.insert({4: 2}, 'y'), B.labels()
            (True, False, ['u', 'v', 'x'])
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D.insert({4: 1}, 'x'), D._dense, D.labels()
            (True, True, ['u', 'v', 'x'])
        """
        if self._dense:
            if self.contains(v):
                return False
            self._keep({k: c for k, c in v.items() if c})
            self._labels.append(label)
            self._K = None
            self._dense_build(self._independent())
            return True
        r, t = self._reduce(v)
        if not r:
            return False
        idx = len(self._labels)
        combo = None
        if self._combos_ok:
            combo = {idx: self._R.one()}
            _addmul(combo, self._combo(t), -self._R.one())
        p = (next(iter(r)) if self._key_order is None
             else max(r, key=self._key_order))
        inv = ~self._R(r[p])
        r = {k: c * inv for k, c in r.items()}
        if combo is not None:
            combo = {k: c * inv for k, c in combo.items()}
        # atomic (an interrupt leaves the rows as they were): the new rows
        # first, then the commit
        upd = []
        for q, (row, cb) in self._rows.items():
            c = row.get(p)
            if c:
                row = dict(row)
                _addmul(row, r, -c)
                if combo is not None:
                    cb = dict(cb)
                    _addmul(cb, combo, -c)
                upd.append((q, (row, cb)))
        self._rows.update(upd)
        self._rows[p] = (r, combo)
        self._labels.append(label)
        self._keep(v)
        return True

    def coordinates(self, v):
        r"""
        The coefficients ``{label: c}`` of ``v`` in the independent
        inserted vectors (see :meth:`labels`), or ``None`` if ``v`` is not
        in the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.coordinates({1: 2, 2: 3, 3: 1}), B.coordinates({1: 1})
            ({'u': 2, 'v': 1}, None)
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D.coordinates({1: 2, 2: 3, 3: 1})
            {'u': 2, 'v': 1}
        """
        if self._dense:
            sol = self._dense_solve(v)
            if sol is None:
                return None
            return {self._labels[i]: c for i, c in enumerate(sol) if c}
        r, t = self._reduce(v)
        if r:
            return None
        self._ensure_combos()
        return {self._labels[i]: c for i, c in self._combo(t).items()}

    def pivots(self):
        r"""
        The pivot keys, in decreasing ``key_order`` (insertion order
        without one).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.pivots()
            [3, 2]
        """
        if self._dense:
            self._echelon()
            return [self._cols[j] for j in self._pivcols]
        ps = list(self._rows)
        if self._key_order is not None:
            ps.sort(key=self._key_order, reverse=True)
        return ps

    def rows(self):
        r"""
        The rows of the reduced echelon form (raw dicts, pivot
        coefficient 1, no other pivot keys), in the order of
        :meth:`pivots`: with a ``key_order`` a canonical basis of the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.rows()
            [{1: -1, 3: 1}, {1: 1, 2: 1}]
            sage: D = _Block(QQ, key_order=lambda k: k)
            sage: D._dense_insert_many([{1: 1, 2: 1}, {2: 1, 3: 1}], ['u', 'v'])
            [True, True]
            sage: D.rows() == B.rows()
            True
        """
        if self._dense:
            cols = self._cols
            return [{cols[j]: c for j, c in row.dict().items() if c}
                    for row in self._echelon().rows()]
        return [dict(self._rows[p][0]) for p in self.pivots()]

    def normal_form(self, v):
        r"""
        The remainder of ``v`` modulo the span (as :meth:`reduce`); with a
        ``key_order`` it is canonical: supported on non-pivot keys, and equal
        for vectors that differ by an element of the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: B.insert({1: 1, 2: 1}, 'u'), B.insert({2: 1, 3: 1}, 'v'), B.insert({1: 1, 3: -1}, 'w')
            (True, True, False)
            sage: B.normal_form({3: 1}) == B.normal_form({2: -1})
            True
        """
        return self.reduce(v)


def _find(parent, x):
    r"""
    Union-find root of ``x`` (absent nodes are roots), compressing.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _find, _union
        sage: p = {}; _union(p, 1, 2), _union(p, 3, 2)
        (1, 3)
        sage: _find(p, 2), _find(p, 7), p
        (3, 7, {1: 3, 2: 3})
    """
    root = x
    while True:
        up = parent.get(root, root)
        if up == root:
            break
        root = up
    while x != root:
        up = parent[x]
        parent[x] = root
        x = up
    return root


def _union(parent, x, y):
    r"""
    Join the classes of ``x`` and ``y`` in the union-find forest
    ``parent`` (a dict); returns the root of the joined class.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _find, _union
        sage: p = {}; _union(p, 'a', 'b'), _find(p, 'b')
        ('a', 'a')
    """
    rx, ry = _find(parent, x), _find(parent, y)
    if rx != ry:
        parent[ry] = rx
    return rx


class KeyedSpan:
    r"""
    The span of the inserted vectors (raw dicts) in reduced row echelon
    form over the field ``R``: every row has a pivot key with coefficient
    1 that occurs in no other row.  Each row also records its expression in
    the independent inserted vectors, so :meth:`coordinates` is available.

    INPUT:

    - ``R`` -- a field (the coefficients of the vectors must lie in it)
    - ``key_order`` -- (default: ``None``) a sort key on the keys: the
      pivot of a new row is then its largest key (its leading monomial),
      every row keeps its pivot as its largest key, and the set of pivots
      is the set of leading monomials of the span, independent of the
      insertion order; the non-pivot keys index a canonical complement
      (normal forms modulo the span).  Without it the pivot is an
      arbitrary key.
    - ``vector_of`` -- (default: ``None``) a function returning the vector
      (raw dict) inserted under a label; then the independent vectors are
      not stored a second time (the caller keeps them, e.g. a memo of
      normally ordered monomials), which halves the memory of large spans

    Over number fields (also towers), ``QQbar``, ``AA`` and
    ``Frac(K[x...])`` the constructor returns an :class:`_ExpandedEchelon`
    (exact linear algebra over ``QQ`` / ``Frac(QQ[x...])``).

    The span is kept in BLOCKS: the connected components of its keys (two
    keys are connected if a vector contains both).  Each block is a reduced
    echelon form of its own (:class:`_Block`), sparse or dense, decided by
    its own fill-in (:meth:`insert_many`): the echelon form of a span that
    mixes the grades of a grading is block diagonal, with a tiny overall
    fill even when every block is dense (D4 at level -2, the ideal of its
    weight-2 singular vectors at weight 4: one span 27671 x 43448, fill
    0.16%; per root-lattice grade median 27%).  Vectors joining blocks
    merge them.  Labels, pivots, rows, coordinates and normal forms do not
    depend on the blocks.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import KeyedSpan
        sage: E = KeyedSpan(QQ, key_order=lambda k: k)
        sage: E.insert({1: 1, 2: 1}); E.insert({2: 1, 3: 1})
        True
        True
        sage: E.pivots(), E.rows()
        ([3, 2], [{1: -1, 3: 1}, {1: 1, 2: 1}])
        sage: E.normal_form({3: 1})
        {1: 1}

    TESTS:

    Vectors on three disjoint sets of keys make three blocks, each dense by
    its own fill, while one elimination of everything (:class:`_Block`)
    stays sparse; labels, pivots, rows, coordinates and normal forms are
    the same::

        sage: from vertex_algebras._keyed_spans import _Block
        sage: set_random_seed(2)
        sage: def rvec(keys, p): return {k: QQ.random_element(5, 4) for k in keys if random() < p}
        sage: def combo(vs, cs):
        ....:     out = {}
        ....:     for v, c in zip(vs, cs):
        ....:         for k, a in v.items(): out[k] = out.get(k, 0) + c * a
        ....:     return {k: a for k, a in out.items() if a}
        sage: groups = [range(40 * g, 40 * g + 40) for g in range(3)]
        sage: base = [rvec(groups[g], 0.5) for g in range(3) for _ in range(50)]
        sage: vecs = base + [combo(sample(base[50 * g: 50 * g + 50], 3), [1, 2, -3])
        ....:                for g in range(3) for _ in range(5)]
        sage: shuffle(vecs)
        sage: E = KeyedSpan(QQ, key_order=lambda k: k)
        sage: flags = E.insert_many(vecs, range(len(vecs)))
        sage: B = _Block(QQ, key_order=lambda k: k)
        sage: ref = [B.insert(v, l) for l, v in enumerate(vecs)]
        sage: len(E._blocks), all(b._dense for b in E._blocks), B._dense, flags == ref
        (3, True, False, True)
        sage: E.labels() == B.labels() and E.pivots() == B.pivots() and E.rows() == B.rows()
        True
        sage: tests = [combo(sample(vecs, 2), [1, -1]) for _ in range(4)]
        sage: tests += [rvec(groups[0], 0.5), {}]
        sage: all(E.coordinates(t) == B.coordinates(t) and E.reduce(t) == B.reduce(t)
        ....:     for t in tests)
        True
    """
    def __new__(cls, R, key_order=None, vector_of=None):
        r"""
        Over number fields (also towers), ``QQbar`` and ``AA`` (and
        ``Frac(K[x...])``) an :class:`_ExpandedEchelon` (exact linear algebra over
        ``QQ`` / ``Frac(QQ[x...])``), else a plain :class:`KeyedSpan`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: type(KeyedSpan(QQ)).__name__, type(KeyedSpan(QuadraticField(2))).__name__
            ('KeyedSpan', '_ExpandedEchelon')
        """
        # number fields (also towers), QQbar and AA: exactly over QQ
        if cls is KeyedSpan and _expands_to_QQ(R):
            return object.__new__(_ExpandedEchelon)
        return object.__new__(cls)

    def __init__(self, R, key_order=None, vector_of=None):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E._R, E._blocks, E._block_of, E._pending, E.rank()
            (Rational Field, [], {}, None, 0)
        """
        self._R = R
        self._key_order = key_order
        self._vector_of = vector_of
        self._labels = []        # labels of the independent vectors
        self._vecs = []          # the independent vectors (unless vector_of)
        self._blocks = []        # _Block's, in creation order
        self._block_of = {}      # key -> its block
        self._pending = None     # (first position, batch) during a batch

    @property
    def _dense(self):
        r"""
        True if some block uses the dense backend.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E._dense
            False
            sage: from vertex_algebras import _keyed_spans as SLA
            sage: old = SLA.DENSE_FILL; SLA.DENSE_FILL = 0
            sage: E.insert_many([{10 + i: 1, 11 + i: 1} for i in range(50)]) and E._dense
            True
            sage: SLA.DENSE_FILL = old
        """
        return any(b._dense for b in self._blocks)

    def rank(self):
        r"""
        The dimension of the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.rank()
            2
        """
        return len(self._labels)

    def labels(self):
        r"""
        The labels of the independent inserted vectors, in insertion
        order (a basis of the span).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.labels()
            ['a', 'b']
        """
        return list(self._labels)

    def _vec_at(self, pos):
        r"""
        The independent vector at position ``pos`` (the blocks' labels
        are positions; during a batch, positions from the batch's first one
        on are provisional: the batch index).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E._vec_at(1)
            {3: 1, 4: 1}
            sage: E._pending = (2, [{7: 1}]); E._vec_at(2)       # inside a batch
            {7: 1}
            sage: E._pending = None
        """
        pend = self._pending
        if pend is not None and pos >= pend[0]:
            return pend[1][pos - pend[0]]
        if self._vector_of is not None:
            return self._vector_of(self._labels[pos])
        return self._vecs[pos]

    def _independent(self):
        r"""
        The independent inserted vectors, in order.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E._independent()
            [{1: 1, 2: 1}, {3: 1, 4: 1}]
        """
        return [self._vec_at(i) for i in range(len(self._labels))]

    def _keep(self, v):
        r"""
        Store the independent vector ``v`` (unless the owner keeps them:
        ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ); E._keep({1: 1}); E._vecs
            [{1: 1}]
        """
        if self._vector_of is None:
            self._vecs.append(v)

    def _new_block(self):
        r"""
        A new empty block (a :class:`_FracBlock` over ``Frac(QQ[x...])`` when the
        C++ elimination is available, else a :class:`_Block`), appended to the
        blocks; its key set ``_keys`` is filled by :meth:`_assign`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ); b = E._new_block()
            sage: type(b).__name__, b._keys, E._blocks == [b]
            ('_Block', set(), True)
        """
        cls = _FracBlock if _frac_nvars(self._R) else _Block
        b = cls(self._R, key_order=self._key_order, vector_of=self._vec_at)
        b._keys = set()
        self._blocks.append(b)
        return b

    def _assign(self, block, keys):
        r"""
        Record the keys ``keys`` not yet in a block as keys of ``block``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ); b = E._new_block(); E._assign(b, [7, 1])
            sage: sorted(b._keys), E._block_of[7] is b
            ([1, 7], True)
        """
        bo = self._block_of
        for k in keys:
            if k not in bo:
                bo[k] = block
                block._keys.add(k)

    def _merge(self, blocks):
        r"""
        One block for the (distinct) ``blocks``: sparse blocks are
        joined (their rows stay reduced: the key sets are disjoint), a dense
        one is rebuilt from the independent vectors.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: len(E._blocks)
            2
            sage: b = E._merge(list(E._blocks)); E._blocks == [b], b._labels, sorted(b._keys)
            (True, [0, 1], [1, 2, 3, 4])
            sage: E.insert({2: 1, 3: 1}, 'd'), E.coordinates({1: 1, 4: 1})     # unchanged span
            (True, {'a': 1, 'b': 1, 'd': -1})

        TESTS:

        Sparse blocks joined with a dense one (by a batch) and two dense
        blocks joined (by one vector) give the labels, pivots, rows,
        coordinates and remainders of one incremental elimination::

            sage: from vertex_algebras._keyed_spans import _Block
            sage: set_random_seed(5)
            sage: def rvec(keys, p): return {k: QQ.random_element(5, 4) for k in keys if random() < p}
            sage: groups = [list(range(40*g, 40*g + 40)) for g in range(7)]
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: a = [rvec(sample(groups[0], 4), 1) for _ in range(5)]
            sage: dense = [rvec(groups[3], 0.6) for _ in range(60)]
            sage: _ = E.insert_many(a, range(5)); _ = E.insert_many(dense, range(5, 65))
            sage: any(b._dense for b in E._blocks)
            True
            sage: jv = rvec(groups[3], 0.6); jv[groups[0][1]] = QQ(1)
            sage: more = [rvec(groups[5], 0.6) for _ in range(60)]
            sage: _ = E.insert_many([jv] + more, range(65, 126))
            sage: tie = {groups[3][0]: 1, groups[5][0]: -2, groups[6][0]: 1}; _ = E.insert(tie, 126)
            sage: seq = a + dense + [jv] + more + [tie]
            sage: B = _Block(QQ, key_order=lambda k: k)
            sage: _ = [B.insert(v, l) for l, v in enumerate(seq)]
            sage: def combo(vs):
            ....:     return (sum(vector(QQ, 280, v) * c for v, c in zip(vs, [1, -1, 2]))).dict()
            sage: tests = [combo(sample(seq, 3)) for _ in range(5)] + [rvec(groups[3] + groups[5], 0.3)]
            sage: E.labels() == B.labels(), E.pivots() == B.pivots(), E.rows() == B.rows()
            (True, True, True)
            sage: all(E.coordinates(u) == B.coordinates(u) and E.reduce(u) == B.reduce(u) for u in tests)
            True
        """
        if len(blocks) == 1:
            return blocks[0]
        # the merged block is built on the side and swapped in at the end
        # (an interrupt leaves the blocks as they were)
        if not any(b._dense or isinstance(b, _FracBlock) for b in blocks):
            base = max(blocks, key=lambda b: len(b._labels))
            ok = all(b._combos_ok for b in blocks)
            labels, rows = list(base._labels), dict(base._rows)
            for b in blocks:
                if b is base:
                    continue
                off = len(labels)
                labels.extend(b._labels)
                for p, (row, cb) in b._rows.items():
                    rows[p] = (row, {i + off: c for i, c in cb.items()} if ok else None)
            if not ok:
                rows = {p: (row, None) for p, (row, _) in rows.items()}
            new = False
        else:
            cls = _FracBlock if _frac_nvars(self._R) else _Block
            base = cls(self._R, key_order=self._key_order, vector_of=self._vec_at)
            base._keys = set()
            pos = [p for b in blocks for p in b._labels]
            # independent vectors: any certified choice accepts them all
            base.insert_many([self._vec_at(p) for p in pos], labels=pos, greedy=False)
            new = True
        keys, upd = set(base._keys), {}
        for b in blocks:
            if b is not base:
                keys |= b._keys
                for k in b._keys:
                    upd[k] = base
        ids = {id(b) for b in blocks if b is not base}
        remaining = [b for b in self._blocks if id(b) not in ids] + ([base] if new else [])
        if not new:
            base._labels, base._rows, base._combos_ok = labels, rows, ok and base._combos_ok
        base._keys = keys
        self._block_of.update(upd)
        self._blocks = remaining
        return base

    def _split(self, v):
        r"""
        ``(parts, rest)``: ``v`` cut along the blocks (``[(block,
        part)]``) and the part on keys of no block.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: parts, rest = E._split({1: 2, 3: 3, 5: 1, 6: 0})
            sage: sorted(sorted(part.items()) for _, part in parts), rest
            ([[(1, 2)], [(3, 3)]], {5: 1})
        """
        parts, rest = {}, {}
        bo = self._block_of
        for k, c in v.items():
            if not c:
                continue
            b = bo.get(k)
            if b is None:
                rest[k] = c
            else:
                e = parts.get(id(b))
                if e is None:
                    parts[id(b)] = (b, {k: c})
                else:
                    e[1][k] = c
        return list(parts.values()), rest

    def insert(self, v, label=None):
        r"""
        Add ``v`` to the span.  Returns True if the span grew (``v`` is then
        recorded under ``label``), False if ``v`` was already in it.

        INPUT:

        - ``v`` -- a raw dict ``{key: coefficient}``
        - ``label`` -- (default: ``None``) any object; the label of ``v``
          in :meth:`labels` and :meth:`coordinates`

        OUTPUT:

        A boolean.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.insert({1: 1, 2: 1, 3: 1, 4: 1}, 'x'), E.insert({2: 1, 3: 1}, 'y'), len(E._blocks)
            (False, True, 1)
            sage: E.insert({}, 'z'), E.labels()
            (False, ['a', 'b', 'y'])
        """
        parts, rest = self._split(v)
        if not parts and not rest:
            return False
        if len(parts) > 1 and not rest and self.contains(v):
            return False                    # no merge for dependent vectors
        block = (self._merge([b for b, _ in parts]) if parts
                 else self._new_block())
        pos = len(self._labels)
        self._pending = (pos, [v])
        try:
            ok = block.insert(v, pos)
        finally:
            self._pending = None
        if ok:
            self._labels.append(label)
            self._keep(v)
            self._assign(block, rest)
        return ok

    def insert_many(self, vecs, labels=None, greedy=True):
        r"""
        Insert the vectors ``vecs`` in order (as repeated :meth:`insert`,
        with the given ``labels``); returns the list of booleans.

        The batch is split along the blocks (connected components of the
        keys of the batch and the existing blocks; blocks it joins are
        merged), and each block takes its part as in :meth:`_Block.insert_many`:
        over ``QQ`` a part of at least ``DENSE_MIN`` vectors starts with
        incremental elimination and switches that block to the dense backend
        (FLINT) once its reduced rows and their combinations fill at least
        ``DENSE_FILL`` of a dense matrix (rank x (columns + rank)).  Dense
        problems: the spans of the triplet algebra in a charge sector fill
        35-100% (dense 30-150x faster); the ideal of the weight-2 singular
        vectors of D4 at level -2, weight 4, ungraded: one span of 1681
        root-lattice grades, 0.16% overall, median 27% per grade (228 s
        with one block, never switching; 10 s in blocks).  Labels are
        appended in batch order, and with a ``key_order`` the pivots and
        rows do not depend on the backends.  ``greedy=False``: the accepted
        vectors are a basis of the batch modulo the span, not necessarily
        the first independent ones (QQ, dense blocks: independent modulo a
        prime, checked exactly, see :meth:`_Block._certified_insert`; other
        rings and backends ignore it): for closures that only need the span.

        INPUT:

        - ``vecs`` -- a list of raw dicts
        - ``labels`` -- (default: ``None``) a list of labels, one per
          vector
        - ``greedy`` -- boolean (default: ``True``); whether the accepted
          vectors must be the first independent ones (see above)

        OUTPUT:

        A list of booleans, one per vector: whether it was accepted.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.insert_many([{5: 1}, {4: 1}, {5: 3}], 'xyz'), E.labels(), len(E._blocks)
            ([True, True, False], ['a', 'b', 'x', 'y'], 3)

        TESTS:

        The dense backend (forced) and incremental elimination give the
        same flags, labels, pivots, rows, membership and coordinates (which
        reproduce the vector), also after export and import::

            sage: from vertex_algebras import _keyed_spans as SLA
            sage: set_random_seed(1)
            sage: def rvec(keys, p): return {k: QQ.random_element(5, 4) for k in keys if random() < p}
            sage: def combo(vs, cs):
            ....:     out = {}
            ....:     for v, c in zip(vs, cs):
            ....:         for k, a in v.items(): out[k] = out.get(k, 0) + c * a
            ....:     return {k: a for k, a in out.items() if a}
            sage: base = [rvec(range(40), 0.3) for _ in range(30)]
            sage: vecs = base + [combo(sample(base, 3), [1, -2, 3]) for _ in range(20)]
            sage: shuffle(vecs)
            sage: old = SLA.DENSE_FILL; SLA.DENSE_FILL = 0          # dense at once
            sage: D = KeyedSpan(QQ, key_order=lambda k: k); fD = D.insert_many(vecs, range(50))
            sage: SLA.DENSE_FILL = old
            sage: S = KeyedSpan(QQ, key_order=lambda k: k)
            sage: fS = [S.insert(v, l) for l, v in enumerate(vecs)]
            sage: D._dense, S._dense, fD == fS, D.rank()
            (True, False, True, 30)
            sage: D.labels() == S.labels() and D.pivots() == S.pivots() and D.rows() == S.rows()
            True
            sage: tests = [combo(sample(vecs, 4), [1, 2, -1, 5]) for _ in range(5)]
            sage: tests += [rvec(range(40), 0.3), {'x': 1}, {}]
            sage: all(D.contains(t) == S.contains(t) and D.coordinates(t) == S.coordinates(t)
            ....:     for t in tests)
            True
            sage: all(combo([vecs[l] for l in co], co.values()) == t
            ....:     for t in tests if (co := D.coordinates(t)) is not None)
            True
            sage: F = KeyedSpan(QQ, key_order=lambda k: k); F._import(D._export())
            sage: F.labels() == D.labels() and all(F.coordinates(t) == D.coordinates(t) for t in tests)
            True

        A batch of zero vectors makes no block, also when forced dense, and
        does not switch to the dense backend::

            sage: SLA.DENSE_FILL = 0
            sage: Z = KeyedSpan(QQ); Z.insert_many([{}] * SLA.DENSE_MIN)[:2]
            [False, False]
            sage: SLA.DENSE_FILL = old
            sage: Z._blocks, Z.rank(), Z.contains({}), Z.contains({1: 1}), Z.coordinates({}), Z.coordinates({1: 1})
            ([], 0, True, False, {}, None)
            sage: Z = KeyedSpan(QQ); _ = Z.insert_many([{}] * SLA.DENSE_MIN); Z._dense, Z.rank()
            (False, 0)

        The automatic switch (fill-in): a very sparse batch stays sparse; a
        batch whose vectors fill in switches in its middle, and so does one
        after an earlier sparse batch; labels, flags, pivots, rows,
        membership and coordinates are those of incremental elimination::

            sage: def incremental(vs):
            ....:     S = KeyedSpan(QQ, key_order=lambda k: k)
            ....:     for l, v in enumerate(vs): S.insert(v, l)
            ....:     return S
            sage: def same(E, S):
            ....:     return E.labels() == S.labels() and E.pivots() == S.pivots() and E.rows() == S.rows()
            sage: sp = [rvec(sample(range(3000), 3), 1) for _ in range(150)]
            sage: sp += [combo(sample(sp, 2), [1, 3]) for _ in range(30)]
            sage: E = KeyedSpan(QQ, key_order=lambda k: k); _ = E.insert_many(sp, range(180))
            sage: E._dense, same(E, incremental(sp))
            (False, True)
            sage: dn = [rvec(range(60), 0.5) for _ in range(50)] + [combo(sample(sp[:50], 2), [1, -1]) for _ in range(5)]
            sage: batch = sp[:50] + dn
            sage: E = KeyedSpan(QQ, key_order=lambda k: k); out = E.insert_many(batch, range(len(batch)))
            sage: S = incremental(batch)
            sage: E._dense, same(E, S), out == [l in S.labels() for l in range(len(batch))]
            (True, True, True)
            sage: E = KeyedSpan(QQ, key_order=lambda k: k); _ = E.insert_many(sp[:60], range(60)); E._dense
            False
            sage: _ = E.insert_many(dn, range(60, 60 + len(dn))); S = incremental(sp[:60] + dn)
            sage: tests = [combo(sample(sp[:60] + dn, 3), [1, 2, 3]) for _ in range(5)] + [rvec(range(60), 0.5)]
            sage: E._dense, same(E, S), all(E.contains(u) == S.contains(u) and E.coordinates(u) == S.coordinates(u)
            ....:                           for u in tests)
            (True, True, True)

        Labels relabelled in place after a batch (as ``vertex_ideal`` does)
        are those :meth:`coordinates` returns::

            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: _ = E.insert_many(vecs[:30], range(30)); n0 = E.rank()
            sage: out = E.insert_many(vecs[30:], range(30, 50))
            sage: acc = [30 + i for i, a in enumerate(out) if a]
            sage: E._labels[n0:] = ['L%d' % i for i in acc]
            sage: u = combo([vecs[i] for i in acc[:3]], [1, 2, 3]); cd = E.coordinates(u)
            sage: any(isinstance(l, str) for l in cd)
            True
            sage: combo([vecs[int(l[1:]) if isinstance(l, str) else l] for l in cd], cd.values()) == u
            True

        An interrupted batch (here the 30th insert into a block raises
        ``KeyboardInterrupt``; over ``GF(p)``, i.e. the Python elimination)
        keeps the committed inserts; inserting the rest of the batch gives
        the span of the uninterrupted batch::

            sage: Fp = GF(10007)
            sage: vp = [{k: Fp.random_element() for k in range(50) if random() < 0.9} for _ in range(60)]
            sage: ref = KeyedSpan(Fp, key_order=lambda k: k); _ = ref.insert_many(vp, range(60))
            sage: orig = SLA._Block.insert; calls = [0]
            sage: def flaky(self, v, label=None):
            ....:     calls[0] += 1
            ....:     if calls[0] == 30:
            ....:         raise KeyboardInterrupt
            ....:     return orig(self, v, label)
            sage: SLA._Block.insert = flaky
            sage: P = KeyedSpan(Fp, key_order=lambda k: k)
            sage: try:
            ....:     P.insert_many(vp, range(60))
            ....: except KeyboardInterrupt:
            ....:     print('interrupted')
            ....: finally:
            ....:     SLA._Block.insert = orig
            interrupted
            sage: P.rank(), P.labels()[-1]
            (29, 28)
            sage: _ = P.insert_many(vp[29:], range(29, 60))
            sage: P.labels() == ref.labels(), P.rows() == ref.rows(), ref.rank()
            (True, True, 50)
        """
        vecs = list(vecs)
        n = len(vecs)
        labels = [None] * n if labels is None else list(labels)
        if not n:
            return []
        parent = {}
        first = [None] * n
        seen_blocks = {}
        bo = self._block_of
        for i, v in enumerate(vecs):
            f = None
            for k, c in v.items():
                if not c:
                    continue
                b = bo.get(k)
                if b is None:
                    node = ('k', k)
                else:
                    node = ('b', id(b))
                    seen_blocks[node] = b
                f = node if f is None else _union(parent, f, node)
            first[i] = f
        groups = {}
        for i, f in enumerate(first):
            if f is not None:
                groups.setdefault(_find(parent, f), []).append(i)
        comp_blocks = {}
        for node, b in seen_blocks.items():
            comp_blocks.setdefault(_find(parent, node), []).append(b)
        p0 = len(self._labels)
        flags = [False] * n
        touched = []
        self._pending = (p0, vecs)
        try:
            for root, idx in groups.items():
                bs = comp_blocks.get(root)
                block = self._merge(bs) if bs else self._new_block()
                touched.append(block)
                for i in idx:
                    self._assign(block, [k for k, c in vecs[i].items() if c])
                block.insert_many([vecs[i] for i in idx],
                                  labels=[p0 + i for i in idx], greedy=greedy)
        finally:
            # the accepted vectors are those the blocks recorded (all of an
            # interrupted batch's committed inserts too): positions, labels
            for b in touched:
                for pos in b._labels:
                    if pos >= p0:
                        flags[pos - p0] = True
            acc = [i for i in range(n) if flags[i]]
            remap = {p0 + i: p0 + j for j, i in enumerate(acc)}
            for i in acc:
                self._labels.append(labels[i])
                self._keep(vecs[i])
            for b in touched:
                b._labels = [remap.get(p, p) for p in b._labels]
            self._pending = None
        return flags

    def reduce(self, v):
        r"""
        The remainder of ``v`` modulo the span (a raw dict containing no
        pivot key); zero (``{}``) iff ``v`` lies in the span.

        INPUT:

        - ``v`` -- a raw dict ``{key: coefficient}``

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.reduce({2: 1, 4: 1, 9: 2})
            {1: -1, 3: -1, 9: 2}
        """
        parts, rest = self._split(v)
        out = dict(rest)
        for b, part in parts:
            out.update(b.reduce(part))
        return out

    def reduce_many(self, vs):
        r"""
        ``[self.reduce(v) for v in vs]``, each block reducing its parts in
        one batch (:meth:`_Block.reduce_many`).

        INPUT:

        - ``vs`` -- a list of raw dicts

        OUTPUT:

        A list of raw dicts.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.reduce_many([{2: 1, 4: 1, 9: 2}, {1: 1, 2: 1}])
            [{1: -1, 3: -1, 9: 2}, {}]
        """
        outs, jobs = [], {}
        for i, v in enumerate(vs):
            parts, rest = self._split(v)
            outs.append(dict(rest))
            for b, part in parts:
                jobs.setdefault(id(b), (b, []))[1].append((i, part))
        for b, items in jobs.values():
            red = b.reduce_many([part for _, part in items]) \
                if hasattr(b, 'reduce_many') else [b.reduce(part) for _, part in items]
            for (i, _), r in zip(items, red):
                outs[i].update(r)
        return outs

    def contains(self, v):
        r"""
        True if ``v`` lies in the span.

        INPUT:

        - ``v`` -- a raw dict ``{key: coefficient}``

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.contains({1: 1, 2: 1, 3: 2, 4: 2}), E.contains({1: 1})
            (True, False)
        """
        parts, rest = self._split(v)
        return not rest and all(b.contains(part) for b, part in parts)

    def coordinates(self, v):
        r"""
        The coefficients ``{label: c}`` of ``v`` in the independent
        inserted vectors (see :meth:`labels`), or ``None`` if ``v`` is not
        in the span.

        INPUT:

        - ``v`` -- a raw dict ``{key: coefficient}``

        OUTPUT:

        A dict ``{label: coefficient}``, or ``None``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.coordinates({1: 1, 2: 1, 3: 2, 4: 2}), E.coordinates({1: 1, 9: 1})
            ({'a': 1, 'b': 2}, None)
        """
        parts, rest = self._split(v)
        if rest:
            return None
        out = {}
        labels = self._labels
        for b, part in parts:
            c = b.coordinates(part)
            if c is None:
                return None
            for pos, x in c.items():
                out[labels[pos]] = x
        return out

    def pivots(self):
        r"""
        The pivot keys, in decreasing ``key_order`` (block by block
        without one).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.pivots()
            [4, 2]
        """
        ps = [p for b in self._blocks for p in b.pivots()]
        if self._key_order is not None:
            ps.sort(key=self._key_order, reverse=True)
        return ps

    def rows(self):
        r"""
        The rows of the reduced echelon form (raw dicts, pivot
        coefficient 1, no other pivot keys), in the order of
        :meth:`pivots`: with a ``key_order`` a canonical basis of the span.

        OUTPUT:

        A list of raw dicts.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.rows()
            [{3: 1, 4: 1}, {1: 1, 2: 1}]
        """
        pr = [(p, r) for b in self._blocks for p, r in zip(b.pivots(), b.rows())]
        if self._key_order is not None:
            ko = self._key_order
            pr.sort(key=lambda t: ko(t[0]), reverse=True)
        return [r for _, r in pr]

    def row_dict(self):
        r"""
        The reduced echelon form as ``{pivot: row}`` (raw dicts; see
        :meth:`rows`), e.g. for :func:`reduce_by_rows`.

        OUTPUT:

        A dict ``{pivot: row}``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.row_dict()
            {2: {1: 1, 2: 1}, 4: {3: 1, 4: 1}}
        """
        return {p: r for b in self._blocks for p, r in zip(b.pivots(), b.rows())}

    def normal_form(self, v):
        r"""
        The remainder of ``v`` modulo the span (as :meth:`reduce`); with a
        ``key_order`` it is canonical: supported on non-pivot keys, and equal
        for vectors that differ by an element of the span.

        INPUT:

        - ``v`` -- a raw dict ``{key: coefficient}``

        OUTPUT:

        A raw dict.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E.normal_form({2: 1}), E.normal_form({1: 2, 2: 3})
            ({1: -1}, {1: -1})
        """
        return self.reduce(v)

    def _export(self):
        r"""
        The state of ``self`` as plain data (for saving to a file; the
        ring, ``key_order`` and ``vector_of`` are not included: the owner
        supplies them again to :meth:`_import`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: st = E._export(); sorted(st), len(st['blocks']), st['labels']
            (['blocks', 'dense', 'labels', 'vecs'], 2, ['a', 'b'])
        """
        return {'blocks': [b._export() for b in self._blocks],
                'labels': self._labels, 'vecs': self._vecs,
                'dense': self._dense}

    def _import(self, st):
        r"""
        Restore a state returned by :meth:`_export` (into a fresh
        echelon with the same ring, ``key_order`` and ``vector_of``); also
        the single-block states saved before the blocks (one block).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E.insert_many([{1: 1, 2: 1}, {3: 1, 4: 1}, {1: 2, 2: 2}], 'abc')
            [True, True, False]
            sage: E2 = KeyedSpan(QQ, key_order=lambda k: k); E2._import(E._export())
            sage: E2.rows() == E.rows(), E2.labels(), len(E2._blocks)
            (True, ['a', 'b'], 2)
            sage: B = E._blocks[0]                      # a state of the single-block era
            sage: E3 = KeyedSpan(QQ, key_order=lambda k: k)
            sage: E3._import(dict(B._export(), labels=['a'], vecs=[{1: 1, 2: 1}])); E3.rows(), E3.labels()
            ([{1: 1, 2: 1}], ['a'])
        """
        if not ('blocks' in st or 'rows' in st):
            # a state of an _EngineSpan / _ExpandedEchelon: rebuild from the
            # vectors of its labels
            labels = list(st['labels'])
            self._labels, self._vecs = [], []
            self._blocks, self._block_of = [], {}
            vecs = [self._vector_of(l) for l in labels] if self._vector_of is not None \
                else list(st['vecs'])
            if vecs and not all(self.insert_many(vecs, labels)):
                raise ValueError('the saved vectors of a span are not independent')
            return
        self._labels = list(st['labels'])
        self._vecs = list(st['vecs'])
        self._blocks, self._block_of = [], {}
        if 'blocks' in st:
            bsts = st['blocks']
        else:
            bsts = [dict(st, labels=list(range(len(st['labels']))), vecs=[])]
        for bst in bsts:
            b = self._new_block()
            b._import(bst)
            self._assign(b, b._key_set())


def left_kernel(vecs, R):
    r"""
    A basis of the linear relations among the raw dicts ``vecs`` over the
    field ``R``: dicts ``{j: r_j}`` with ``Σ_j r_j vecs[j] = 0``, one for
    each vector that depends on the earlier ones (coefficient 1 at its own
    index, the others at earlier indices), by increasing index.  This is the
    reduced echelon basis of the space of relations for pivots at the
    largest indices, independent of the method; the caller chooses it by
    ordering the vectors.  The order matters for the cost too: for the
    images of a basis of monomials, ordered by a key order, the relations
    are the canonical basis of the kernel, usually with small entries (the
    Heisenberg coset of ``V^k(sl_2)``, ``k = 7/3``, at weight 12: 1204 images
    on 1754 keys, kernel of dimension 141 with entries of at most 13 digits,
    1.4 s; in the sorted order of the monomials the relations have 211
    digits and take 10.6 s).

    Incremental sparse elimination (:class:`KeyedSpan`); over ``QQ``, a
    batch of at least ``DENSE_MIN`` vectors whose elimination fills in (fill
    as in :meth:`KeyedSpan.insert_many`, threshold
    ``KERNEL_DENSE_FILL``) is restarted densely: one FLINT echelon form of
    the matrix with the vectors as columns (the coset of ``L_2(sl_2)`` in
    ``V_{A_1 ⊕ A_1}`` at weight 8, 1049 images on 4480 keys: 1 s; 8.6 s
    through the dense backend of :class:`KeyedSpan`, which also builds
    the echelon form of the span; 6.8 s all sparse).  (A multimodular
    version, echelon forms mod word-size primes with rational
    reconstruction, was slower in both orders: 4.9 s and 41.6 s.)  The
    vectors are first split into the connected components of their keys
    (relations never mix components), and each component makes this choice
    by its own fill: images of a basis that mixes the grades of a grading
    are block diagonal, with a low overall fill even when every block is
    dense.

    INPUT:

    - ``vecs`` -- a list of raw dicts
    - ``R`` -- a field (the coefficients of the vectors must lie in it)

    OUTPUT:

    A list of dicts ``{j: r_j}`` (indices into ``vecs``), one per
    dependent vector, by increasing index of that vector.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import left_kernel
        sage: left_kernel([{'a': 1}, {'b': 1}, {'a': 2, 'b': 3}, {}], QQ)
        [{0: -2, 1: -3, 2: 1}, {3: 1}]
        sage: left_kernel([{'a': 1}], QQ)
        []

    TESTS:

    The relations are a basis of Sage's left kernel, monic at their largest
    index, and the same as one elimination of all vectors (also for
    vectors that fill in, which take the dense route)::

        sage: from vertex_algebras._keyed_spans import _left_kernel_block
        sage: set_random_seed(3)
        sage: def rvec(keys, p): return {k: QQ.random_element(5, 4) for k in keys if random() < p}
        sage: def combo(vs, cs):
        ....:     out = {}
        ....:     for v, c in zip(vs, cs):
        ....:         for k, a in v.items(): out[k] = out.get(k, 0) + c * a
        ....:     return {k: a for k, a in out.items() if a}
        sage: ok = True
        sage: for n in range(4):
        ....:     ncols = 30 + 10 * n
        ....:     base = [rvec(range(ncols), 0.1 if n % 2 else 0.6) for _ in range(20 + 8 * n)]
        ....:     vs = base + [combo(sample(base, 3), [1, 2, 3]) for _ in range(n + 2)] + [{}]
        ....:     shuffle(vs)
        ....:     M = matrix(QQ, len(vs), ncols, {(i, k): c for i, v in enumerate(vs)
        ....:                                     for k, c in v.items()})
        ....:     rel = left_kernel(vs, QQ)
        ....:     R = matrix(QQ, len(rel), len(vs), {(a, j): c for a, r in enumerate(rel)
        ....:                                         for j, c in r.items()})
        ....:     ok = (ok and R.rank() == len(rel) == M.left_kernel().dimension()
        ....:           and (R * M).is_zero() and all(r[max(r)] == 1 for r in rel)
        ....:           and rel == _left_kernel_block(vs, QQ))
        sage: ok
        True

    Over ``Q(x)`` and ``Q(y_1, y_2)`` the C++ elimination (when built)
    gives the relations of the Python elimination::

        sage: from vertex_algebras import _keyed_spans as SLA
        sage: x = Frac(QQ['x']).gen(); y1, y2 = Frac(QQ['y1, y2']).gens()
        sage: def fvec(p, gens): return {k: sum(QQ.random_element(3, 2) * g for g in gens) for k in range(8) if random() < p}
        sage: ok = True
        sage: for R, gens in [(x.parent(), [x, 1/(x + 2)]), (y1.parent(), [y1, y2/(y1 - 1)])]:
        ....:     base = [fvec(0.5, gens) for _ in range(5)]
        ....:     vs = base + [combo(sample(base, 2), [2, -1]) for _ in range(3)] + [{}]
        ....:     SLA._USE_FRAC_ENGINE = False
        ....:     ref = left_kernel(vs, R)
        ....:     SLA._USE_FRAC_ENGINE = True
        ....:     ok = ok and left_kernel(vs, R) == ref and len(ref) == 4
        sage: ok
        True
    """
    vecs = list(vecs)
    if _expands_to_QQ(R):
        return _expanded_left_kernel(vecs, R)
    one = R.one()
    parent = {}
    first = []
    for v in vecs:
        f = None
        for k, c in v.items():
            if c:
                f = ('k', k) if f is None else _union(parent, f, ('k', k))
        first.append(f)
    out, comps = [], {}
    for j, f in enumerate(first):
        if f is None:
            out.append((j, {j: one}))           # a zero vector
        else:
            comps.setdefault(_find(parent, f), []).append(j)
    for idx in comps.values():
        for r in _left_kernel_block([vecs[j] for j in idx], R):
            out.append((idx[max(r)], {idx[i]: c for i, c in r.items()}))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


def _left_kernel_block(vecs, R):
    r"""
    :func:`left_kernel` of vectors whose keys form one connected
    component (sparse, or restarted densely on fill-in).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _left_kernel_block, left_kernel
        sage: vecs = [{1: 1}, {1: 2}, {1: 1, 2: 1}, {2: 3}]
        sage: _left_kernel_block(vecs, QQ)
        [{0: -2, 1: 1}, {0: 3, 2: -3, 3: 1}]
        sage: _left_kernel_block(vecs, QQ) == left_kernel(vecs, QQ)
        True
    """
    from sage.rings.rational_field import QQ
    if _frac_nvars(R):
        E = _FracBlock(R, combos=True)
    else:
        E = _Block(R)
        E._combos_ok = True    # every dependent vector needs coordinates
    dep = []
    if R is QQ and len(vecs) >= DENSE_MIN:
        keys = set()
        for v in vecs:
            keys.update(v)
        ncols = len(keys)
        for i, v in enumerate(vecs):
            if ((i % FILL_CHECK == 0 or i in (4, 8))
                    and E._fill(ncols) >= KERNEL_DENSE_FILL):
                return _dense_left_kernel(vecs, keys, R)
            if not E.insert(v, i):
                dep.append(i)
    else:
        dep = [i for i, v in enumerate(vecs) if not E.insert(v, i)]
    one = R.one()
    out = []
    for j in dep:
        r = {i: -c for i, c in sorted(E.coordinates(vecs[j]).items()) if c}
        r[j] = one
        out.append(r)
    return out


def _modular_left_kernel(vecs, nthreads=1, max_primes=200, seed=0):
    r"""
    Candidates for a basis of the linear relations among the raw dicts
    ``vecs`` over ``QQ`` when there are few: dicts ``{j: r_j}`` (``r_f = 1``
    at one index ``f`` each), or ``None`` if the C++ engine is not available
    or the reconstruction does not settle within ``max_primes`` primes.

    Modulo primes ``p < 2^31`` the matrix with the vectors as columns is
    projected to ``len(vecs) + 16`` random combinations of its rows (each
    row in four of them, with random multipliers) and its kernel computed by
    one dense elimination (FLINT, ``nthreads``; the reduced echelon basis
    for the non-pivot columns, canonical), checked modulo ``p`` against all
    rows (a projection that lost rank is discarded); kernels with the same
    non-pivot columns are combined by the Chinese remainder theorem and
    rationally reconstructed until two consecutive reconstructions agree.
    The caller must verify the candidates exactly: ``Σ_j r_j vecs[j] = 0``.
    Every relation over ``QQ`` is one modulo ``p`` of the projected matrix,
    so if all candidates pass, they are a basis of the relations (their
    number is at least the nullity over ``QQ``, and they are independent).
    For large spans with few relations (singular vectors): the dense
    elimination over ``QQ`` of :func:`left_kernel` needs memory quadratic in
    the size with large rationals.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _modular_left_kernel, _frac_engine
        sage: vecs = [{'a': 1, 'b': 2}, {'a': 3}, {'a': 1/2, 'b': 1/3}, {'b': 5}]
        sage: r = _modular_left_kernel(vecs)
        sage: _frac_engine() is None or (len(r) == 2 and all(
        ....:     sum(c * vecs[j].get(k, 0) for j, c in x.items()) == 0 for x in r for k in 'ab'))
        True
    """
    import numpy as np
    from sage.arith.misc import random_prime
    from sage.rings.integer import Integer
    from sage.rings.rational_field import QQ
    E = _frac_engine()
    if E is None:
        return None
    keyid = {}
    rl, cl, nums, dens = [], [], [], []
    for j, v in enumerate(vecs):
        for k, c in v.items():
            if not c:
                continue
            i = keyid.get(k)
            if i is None:
                i = keyid[k] = len(keyid)
            c = QQ(c)
            rl.append(i)
            cl.append(j)
            nums.append(int(c.numerator()))
            dens.append(int(c.denominator()))
    N, NR = len(vecs), len(keyid)
    if N == 0:
        return []
    rows = np.array(rl, dtype=np.int64)
    cols = np.array(cl, dtype=np.int64)
    del rl, cl
    dset = sorted(set(dens))
    dpos = {d: t for t, d in enumerate(dset)}
    dix = np.array([dpos[d] for d in dens], dtype=np.int64)
    del dens
    nproj = min(NR, N + 16)
    rng = np.random.default_rng(int(seed))
    profile = None
    acc, modulus, last = None, Integer(1), None
    for _ in range(max_primes):
        p = int(random_prime(2**31, lbound=2**30))
        try:
            dinv = np.array([pow(d, -1, p) for d in dset], dtype=np.int64)
        except ValueError:                  # p divides a denominator
            continue
        res = np.fromiter((n % p for n in nums), dtype=np.int64, count=len(nums))
        res = (res * dinv[dix]) % p
        if NR > nproj:
            prow = np.concatenate([rng.integers(0, nproj, NR)[rows] for _ in range(4)])
            pcol = np.concatenate([cols] * 4)
            vals = np.concatenate([(rng.integers(1, p, NR)[rows] * res) % p
                                   for _ in range(4)])
        else:
            prow, pcol, vals = rows, cols, res
        ker = E.nmod_kernel_triplets(max(nproj, 1), N, p, prow, pcol,
                                     vals.astype(np.uint64), nthreads)
        del prow, pcol, vals
        prof = tuple((f, tuple(c for c, _ in v)) for f, v in ker)
        if profile is not None and len(prof) > len(profile):
            continue                         # an unlucky prime / projection
        if prof != profile and NR > nproj:
            # the kernel of the projection: is it one of the matrix (mod p)?
            lossy = False
            for f, v in ker:
                x = np.zeros(N, dtype=np.int64)
                x[f] = 1
                for c, y in v:
                    x[c] = y
                acc_r = np.zeros(NR, dtype=np.int64)
                np.add.at(acc_r, rows, (x[cols] * res) % p)
                if np.any(acc_r % p):
                    lossy = True
                    break
            if lossy:
                continue
        if prof != profile:
            profile, acc, modulus, last = prof, None, Integer(1), None
        P = Integer(p)
        if acc is None:
            acc = [[Integer(x) for _, x in v] for _, v in ker]
        else:
            # CRT: x = a (mod modulus), x = b (mod p)
            u = modulus.inverse_mod(P)
            acc = [[a + modulus * (((Integer(b) - a) * u) % P)
                    for a, (_, b) in zip(row, v)] for row, (_, v) in zip(acc, ker)]
        modulus *= P
        try:
            rec = [[a.rational_reconstruction(modulus) for a in row] for row in acc]
        except (ArithmeticError, ValueError):
            continue
        if rec == last:
            out = []
            for (f, cs), row in zip(profile, rec):
                r = {c: x for c, x in zip(cs, row) if x}
                r[f] = QQ.one()
                out.append(r)
            return out
        last = rec
    return None


def _dense_left_kernel(vecs, keys, R):
    r"""
    :func:`left_kernel` by one dense elimination: the reduced echelon
    form ``E`` of the matrix with the vectors as columns; a non-pivot column
    ``j`` is ``Σ_i E[i, j]`` times the pivot columns ``p_i < j``.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _dense_left_kernel, _left_kernel_block
        sage: vecs = [{1: 1}, {1: 2}, {1: 1, 2: 1}, {2: 3}]
        sage: _dense_left_kernel(vecs, {1, 2}, QQ)
        [{0: -2, 1: 1}, {0: 3, 2: -3, 3: 1}]
        sage: _dense_left_kernel(vecs, {1, 2}, QQ) == _left_kernel_block(vecs, QQ)
        True
    """
    from sage.matrix.constructor import matrix
    idx = {k: i for i, k in enumerate(keys)}
    MT = matrix(R, len(idx), len(vecs),
                {(idx[k], j): c for j, v in enumerate(vecs)
                 for k, c in v.items() if c}, sparse=False)
    E = MT.echelon_form()
    piv = MT.pivots()
    pset = set(piv)
    one = R.one()
    out = []
    for j in range(len(vecs)):
        if j in pset:
            continue
        r = {}
        for i, p in enumerate(piv):
            if p > j:
                break
            c = E[i, j]
            if c:
                r[p] = -c
        r[j] = one
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
#  Number fields, QQbar, AA: linear algebra over QQ                           #
# --------------------------------------------------------------------------- #

def _expands_to_QQ(R):
    r"""
    True for the fields whose linear algebra is done over ``QQ`` or
    ``QQ(x...)`` (:class:`_ExpandedEchelon`): number fields other than
    ``QQ`` (absolute or towers), ``QQbar``, ``AA``, and ``Frac(K[x...])``
    for such a number field ``K`` (over ``Frac(QQ[x...])``).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _expands_to_QQ
        sage: K = QuadraticField(2)
        sage: [_expands_to_QQ(R) for R in [K, QQ, QQbar, AA, Frac(K['c']), Frac(QQ['c']), GF(5)]]
        [True, False, True, True, True, False, False]
    """
    from sage.rings.number_field.number_field_base import NumberField
    from sage.rings.qqbar import AA, QQbar
    from sage.rings.rational_field import QQ
    if R is QQbar or R is AA or (R is not QQ and isinstance(R, NumberField)):
        return True
    return _function_field_over_nf(R) is not None


def _function_field_over_nf(R):
    r"""
    ``(A, K)`` if ``R = Frac(A)``, ``A`` a polynomial ring over a number
    field ``K != QQ``, else None.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _function_field_over_nf
        sage: K = QuadraticField(2); A = K['c']
        sage: _function_field_over_nf(Frac(A)) == (A, K)
        True
        sage: _function_field_over_nf(K), _function_field_over_nf(Frac(QQ['c']))
        (None, None)
    """
    from sage.rings.number_field.number_field_base import NumberField
    from sage.rings.rational_field import QQ
    try:
        A = R.ring()
        if A is R or R is not A.fraction_field():
            return None
        K = A.base_ring()
        if K is QQ or not isinstance(K, NumberField) or not A.ngens():
            return None
        return A, K
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return None


class _FieldCoordinates:
    r"""
    Coordinates of a field ``R`` of degree ``d`` over ``QQ`` in the power
    basis ``1, θ, ..., θ^(d-1)`` of an absolute number field ``M`` (or
    ``QQ``): ``R -> M`` by ``to_M``, back by ``from_M``.  A vector over
    ``R`` (raw dict) and ``j < d`` give the ``QQ``-vector of ``θ^j v`` on the
    keys ``(key, i)`` (:meth:`expand`); :meth:`collapse` is the inverse
    (``Σ q θ^i`` at ``key``).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _field_coordinates
        sage: K.<s> = QuadraticField(2); F = _field_coordinates(K)
        sage: F.d, F.expand({'a': 1 + s}, 1)                  # s (1 + s) = 2 + s
        (2, {('a', 0): 2, ('a', 1): 1})
        sage: F.collapse(F.expand({'a': 1 + s}, 0))
        {'a': s + 1}
    """

    def __init__(self, M, to_M=None, from_M=None):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import _FieldCoordinates
            sage: K.<s> = QuadraticField(3)
            sage: F = _FieldCoordinates(K); F.d, F.pows
            (2, [1, s])
            sage: F = _FieldCoordinates(QQ); F.d, F.pows
            (1, [1])
        """
        from sage.rings.rational_field import QQ
        self.M = M
        self.d = 1 if M is QQ else M.degree()
        self.to_M = to_M
        self.from_M = from_M
        self.pows = [M.one()] if M is QQ else [M.gen() ** j for j in range(self.d)]

    def coords(self, m):
        r"""
        Power-basis coordinates of an element of ``M``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); F = _field_coordinates(K)
            sage: F.coords(1 + 3*s)
            [1, 3]
            sage: from vertex_algebras._keyed_spans import _FieldCoordinates
            sage: _FieldCoordinates(QQ).coords(QQ(3))
            [3]
        """
        return [m] if self.d == 1 and not hasattr(m, 'list') else m.list()

    def expand(self, v, j, rank=None):
        r"""
        ``θ^j v`` over ``QQ`` (``rank``: a dict assigning every key seen
        a number, filled here).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); F = _field_coordinates(K)
            sage: rank = {}; F.expand({'a': s, 'b': 1, 'c': 0}, 1, rank), rank
            ({('a', 0): 2, ('b', 1): 1}, {'a': 0, 'b': 1})
        """
        to_M, t = self.to_M, self.pows[j]
        out = {}
        for k, c in v.items():
            if not c:
                continue
            if rank is not None and k not in rank:
                rank[k] = len(rank)
            m = (c if to_M is None else to_M(c)) * t
            for i, q in enumerate(self.coords(m)):
                if q:
                    out[(k, i)] = q
        return out

    def element(self, cs):
        r"""
        The element of ``R`` with coordinates ``cs`` (a dict ``{i: q}``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); F = _field_coordinates(K)
            sage: F.element({0: 1, 1: 2}), F.element({})
            (2*s + 1, 0)
        """
        from sage.rings.rational_field import QQ
        M = self.M
        x = sum(QQ(q) * self.pows[i] for i, q in cs.items()) if cs else M.zero()
        return x if self.from_M is None else self.from_M(x)

    def collapse(self, w):
        r"""
        A ``QQ``-vector on the keys ``(key, i)`` as a vector over ``R``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); F = _field_coordinates(K)
            sage: F.collapse({('a', 0): 1, ('a', 1): 1, ('b', 1): 0})
            {'a': s + 1}
        """
        acc = {}
        for (k, i), q in w.items():
            acc.setdefault(k, {})[i] = q
        out = {}
        for k, cs in acc.items():
            x = self.element(cs)
            if x:
                out[k] = x
        return out


class _FunctionFieldCoordinates:
    r"""
    Coordinates of ``R = Frac(K[x...])``, ``K`` a number field of degree
    ``d`` (also a tower, through its absolute field ``M``), over the base
    ``B = Frac(QQ[x...])`` in the basis ``1, θ, ..., θ^(d-1)``: for ``f =
    N/D``, ``f = N D' / Norm(D)`` with ``Norm(D) = Res_t(m(t), D) ∈
    QQ[x...]`` (a multiple of the norm, divisible by ``D`` in ``K[x...]``)
    and ``D' = Norm(D) / D ∈ K[x...]``, cached per denominator; the
    coordinates are those of the ``K``-coefficients of ``N D'``, divided by
    ``Norm(D)``.  Same interface as :class:`_FieldCoordinates`.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _field_coordinates
        sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
        sage: F.d, F.base
        (2, Fraction Field of Multivariate Polynomial Ring in c over Rational Field)
        sage: F.coords(s/(c + s))                  # s (c - s) / (c^2 - 2)
        [(-2)/(c^2 - 2), c/(c^2 - 2)]
        sage: F.collapse(F.expand({'a': s/(c + s)}, 0))
        {'a': s/(c + s)}
    """

    def __init__(self, R):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import _FunctionFieldCoordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c'])
            sage: F = _FunctionFieldCoordinates(R)
            sage: F.d, F.n, F.QA, F.mt
            (2, 1, Multivariate Polynomial Ring in c over Rational Field, t_res^2 - 2)
            sage: F.pows == [1, s]
            True
        """
        from sage.rings.polynomial.polynomial_ring import PolynomialRing_generic
        from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
        from sage.rings.rational_field import QQ

        from .vertex_algebra import _absolute_presentation
        A, K = _function_field_over_nf(R)
        self.R, self.A, self.K = R, A, K
        M, to_abs, from_abs = _absolute_presentation(K)
        self.M, self.to_M, self.from_M = M, to_abs, from_abs
        self.d = M.degree()
        self.n = A.ngens()
        self.alg = None
        self._uni = isinstance(A, PolynomialRing_generic)
        names = [str(v) for v in A.gens()]
        self.QA = PolynomialRing(QQ, names, len(names))          # QQ[x...], multivariate
        self.base = self.QA.fraction_field()
        self.QAt = PolynomialRing(QQ, names + ['t_res'], len(names) + 1)
        z = (0,) * self.n
        self.mt = self.QAt({z + (i,): c for i, c in
                            enumerate(M.absolute_polynomial().list()) if c})
        self.theta = M.gen() if from_abs is None else from_abs(M.gen())
        self.pows = [R(self.theta ** j) for j in range(self.d)]
        # polynomial arithmetic over the absolute field (Singular has no
        # polynomial rings over towers)
        self.AM = A if to_abs is None else A.change_ring(M)
        self._norms = {}

    def _etup(self, e):
        r"""
        An exponent of ``A`` as a tuple (univariate rings give integers).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F._etup(1)
            (1,)
        """
        return (int(e),) if self._uni else tuple(e)

    def _to_AM(self, P):
        r"""
        A polynomial of ``A`` in the polynomial ring over the absolute field
        (the identity unless ``K`` is a tower).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F._to_AM(c + s) == c + s
            True
            sage: L.<t> = K.extension(polygen(K)^2 - 3); A.<d> = L['d']
            sage: G = _field_coordinates(Frac(A)); G._to_AM(d + t).parent()
            Univariate Polynomial Ring in d over Number Field in t with defining polynomial x^4 - 10*x^2 + 1
        """
        if self.to_M is None:
            return P
        return self.AM({e: self.to_M(c) for e, c in P.dict().items()})

    def _poly_A(self, d):
        r"""
        A polynomial of ``A`` from ``{exponent tuple: coefficient}``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F._poly_A({(2,): 3, (0,): s})
            3*c^2 + s
        """
        if self._uni:
            return self.A({e[0]: c for e, c in d.items()})
        return self.A(d)

    def _norm(self, D):
        r"""
        ``(Norm(D) ∈ QQ[x...], D' ∈ K[x...])`` with ``D D' = Norm(D)``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: N, Dp = F._norm(R.ring()(c + s)); N, Dp
            (c^2 - 2, c - s)
            sage: F._norm(R.ring()(2*s))
            (1, 1/4*s)
        """
        r = self._norms.get(D)
        if r is not None:
            return r
        DM = self._to_AM(D)
        if DM.is_constant():
            r = (self.QA.one(), self.AM(~DM.constant_coefficient()))
        else:
            dt = {}
            for e, c in DM.dict().items():
                e = self._etup(e)
                for i, q in enumerate(c.list()):
                    if q:
                        dt[e + (i,)] = dt.get(e + (i,), 0) + q
            Dt = self.QAt(dt)
            N = Dt.resultant(self.mt, self.QAt.gens()[-1])
            NQ = self.QA({e[:-1]: c for e, c in N.dict().items()})
            if self._uni:
                NA = self.AM({e[0]: c for e, c in NQ.dict().items()})
            else:
                NA = self.AM(dict(NQ.dict()))
            Dp = NA // DM
            if Dp * DM != NA:
                raise ArithmeticError('the norm is not divisible by the denominator')
            r = (NQ, Dp)
        self._norms[D] = r
        return r

    def coords(self, f):
        r"""
        The coordinates (elements of ``B``) of ``f ∈ R``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F.coords(1/c + s)
            [1/c, 1]
        """
        f = self.R(f)
        NQ, Dp = self._norm(f.denominator())
        P = self._to_AM(f.numerator()) * Dp
        cs = [{} for _ in range(self.d)]
        for e, c in P.dict().items():
            e = self._etup(e)
            for i, q in enumerate(c.list()):
                if q:
                    cs[i][e] = q
        den = self.base(NQ)
        return [self.base(self.QA(c)) / den if c else self.base.zero() for c in cs]

    def expand(self, v, j, rank=None):
        r"""
        ``θ^j v`` over ``B``: the coordinates of ``θ^j c`` on the keys
        ``(key, i)`` (``rank``: as for :meth:`_FieldCoordinates.expand`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F.expand({'a': s/c, 'b': 0}, 1)
            {('a', 0): 2/c}
        """
        out = {}
        t = self.pows[j]
        for k, c in v.items():
            if not c:
                continue
            if rank is not None and k not in rank:
                rank[k] = len(rank)
            for i, q in enumerate(self.coords(c * t)):
                if q:
                    out[(k, i)] = q
        return out

    def _from_base(self, q):
        r"""
        An element of ``B`` in ``R``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: x = F.base.gen(); F._from_base(1/(1 + x))
            1/(c + 1)
        """
        q = self.base(q)
        num = self._poly_A(dict(q.numerator().dict()))
        den = self._poly_A(dict(q.denominator().dict()))
        return self.R(num) / self.R(den)

    def element(self, cs):
        r"""
        The element ``Σ_i q_i θ^i`` of ``R`` for ``cs = {i: q_i}``, ``q_i`` in
        ``B``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F.element({0: F.base(1), 1: 1/F.base.gen()})
            (c + s)/c
        """
        R = self.R
        return sum((self._from_base(q) * self.pows[i] for i, q in cs.items()), R.zero())

    def collapse(self, w):
        r"""
        A ``B``-vector on the keys ``(key, i)`` as a vector over ``R``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _field_coordinates
            sage: K.<s> = QuadraticField(2); R.<c> = Frac(K['c']); F = _field_coordinates(R)
            sage: F.collapse({('a', 0): 1, ('a', 1): F.base.gen()})
            {'a': s*c + 1}
        """
        acc = {}
        for (k, i), q in w.items():
            acc.setdefault(k, {})[i] = q
        out = {}
        for k, cs in acc.items():
            x = self.element(cs)
            if x:
                out[k] = x
        return out


def _field_coordinates(R, vecs=()):
    r"""
    :class:`_FieldCoordinates` for the field ``R``: a number field
    through its absolute field; ``QQbar`` / ``AA`` through a number field
    containing the coefficients of ``vecs`` (``to_M`` raises ``ValueError``
    for numbers outside it, which are recorded in ``.alg.outside``).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _field_coordinates
        sage: K.<s> = QuadraticField(2)
        sage: type(_field_coordinates(K)).__name__, type(_field_coordinates(Frac(K['c']))).__name__
        ('_FieldCoordinates', '_FunctionFieldCoordinates')
        sage: L.<t> = K.extension(polygen(K)^2 - 3); F = _field_coordinates(L)
        sage: F.d, F.collapse(F.expand({'a': t + s}, 0)) == {'a': t + s}
        (4, True)
        sage: F = _field_coordinates(QQbar, [{'a': QQbar(2).sqrt()}]); F.d, F.alg is not None
        (2, True)
    """
    from sage.rings.qqbar import AA, QQbar

    from .vertex_algebra import _absolute_presentation, _AlgebraicCoefficients
    if _function_field_over_nf(R) is not None:
        return _FunctionFieldCoordinates(R)
    if R is QQbar or R is AA:
        alg = _AlgebraicCoefficients(R, [c for v in vecs for c in v.values() if c])
        F = _FieldCoordinates(alg.K, alg.up, alg.down)
        F.alg = alg
        return F
    M, to_abs, from_abs = _absolute_presentation(R)
    F = _FieldCoordinates(M, to_abs, from_abs)
    F.alg = None
    return F


class _ExpandedEchelon(KeyedSpan):
    r"""
    A :class:`KeyedSpan` over a number field ``K`` of degree ``d``
    (absolute, or a tower through its absolute field), ``QQbar`` or ``AA``
    (through a number field containing the coefficients seen, grown when
    needed), computed EXACTLY over ``QQ``: the ``K``-span of ``v_1, ...`` is
    the ``QQ``-span of all ``θ^j v_i`` (``θ`` a generator, ``j < d``), whose
    coordinates are the power-basis coordinates on the keys ``(key, i)``.
    So the ``QQ`` machinery does the work (dense FLINT backend, lazy
    reduced echelon form).

    - ``v`` is independent of the span iff its ``j = 0`` vector is; then all
      its ``d`` vectors are (the ``QQ``-span is stable under ``θ``).
    - Coordinates: those of the ``j = 0`` vector, ``Σ_j q_(i,j) θ^j`` per
      label.
    - With the ``QQ`` key order ``(key order of key, -i)``, the pivots of a
      ``K``-pivot ``p`` are ``(p, 0), ..., (p, d-1)`` and the ``QQ`` row of
      ``(p, 0)`` is the ``K`` row of ``p`` (coefficient 1 at ``p``, 0 at the
      other pivots): rows, pivots and normal forms come out canonical as
      over ``K``.  Without a key order, keys are ranked when first seen.

    Labels and their in-place relabelling (``_labels``) are as for
    :class:`KeyedSpan`; states are exported over ``K`` and rebuilt on
    import (also states saved before 2026-10-12).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import KeyedSpan, _ExpandedEchelon
        sage: K.<s> = QuadraticField(2)
        sage: E = KeyedSpan(K, key_order=lambda k: k)
        sage: type(E) is _ExpandedEchelon
        True
        sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], labels=['a', 'b', 'c'])
        [True, False, True]
        sage: E.coordinates({1: s + 2, 2: s + 2, 3: s})
        {'a': s + 1, 'c': 1}
        sage: E.pivots(), [sorted(r.items()) for r in E.rows()]
        ([3, 2], [[(1, -1), (3, 1)], [(1, s), (2, 1)]])
        sage: E.normal_form({1: 1, 2: 1})
        {1: -s + 1}

    TESTS:

    Over a quadratic and a cubic field, a tower, ``QQbar``, ``AA`` and
    ``Frac(K[c])`` the expansion to ``QQ`` (or ``QQ(c)``) gives what plain
    elimination over the field (:class:`_Block`) gives: flags, labels,
    pivots, rows, membership, coordinates, normal forms, and the relations
    of :func:`left_kernel`::

        sage: from vertex_algebras._keyed_spans import _Block, left_kernel, _left_kernel_block
        sage: K3.<r> = NumberField(QQ['z']([-1, -1, 0, 3]))
        sage: Kt.<i> = K.extension(QQ['y']([1, 0, 1]))
        sage: Fc.<c> = Frac(K['c'])
        sage: fields = [(K, [s, 1 + s/3]), (K3, [r, r^2 - 1]), (Kt, [i, s*i + 1]),
        ....:           (QQbar, [QQbar(2).sqrt(), QQbar.zeta(3)]),
        ....:           (AA, [AA(3).sqrt(), AA(2)^(1/3)]), (Fc, [s, c/(c + s), 1/(c^2 - 5)])]
        sage: def fvec(keys, p, gens):
        ....:     return {k: ZZ.random_element(-3, 4) + choice(gens) * ZZ.random_element(-2, 3)
        ....:             for k in keys if random() < p}
        sage: def fcombo(vs, gens):
        ....:     out = {}
        ....:     for v in vs:
        ....:         a = ZZ.random_element(-2, 3) + choice(gens)
        ....:         for k, b in v.items(): out[k] = out.get(k, 0) + a * b
        ....:     return {k: b for k, b in out.items() if b}
        sage: set_random_seed(4)
        sage: ok = True
        sage: for R, gens in fields:
        ....:     n = 6 if R is Fc else 14
        ....:     base = [fvec(range(n), 0.4, gens) for _ in range(n // 2 + 2)]
        ....:     vecs = base + [fcombo(sample(base, 2), gens) for _ in range(3)]
        ....:     shuffle(vecs)
        ....:     E = KeyedSpan(R, key_order=lambda k: k)
        ....:     B = _Block(R, key_order=lambda k: k); B._combos_ok = True
        ....:     flags = E.insert_many(vecs, range(len(vecs)))
        ....:     ref = [B.insert(v, l) for l, v in enumerate(vecs)]
        ....:     tests = [fcombo(sample(vecs, 2), gens) for _ in range(2)] + [fvec(range(n), 0.3, gens)]
        ....:     ok = (ok and type(E) is _ExpandedEchelon and flags == ref
        ....:           and E.labels() == B.labels() and E.pivots() == B.pivots()
        ....:           and E.row_dict() == B.row_dict()
        ....:           and all(E.contains(t) == B.contains(t) and E.coordinates(t) == B.coordinates(t)
        ....:                   and E.normal_form(t) == B.reduce(t) for t in tests)
        ....:           and left_kernel(vecs, R) == _left_kernel_block(vecs, R))
        sage: ok
        True

    Over ``QQbar`` a span may contain numbers outside its current field::

        sage: G = KeyedSpan(QQbar); G.insert({(): QQbar(1)}, 'one')
        True
        sage: G.contains({(): QQbar(7).sqrt()}), G.coordinates({(): QQbar(7).sqrt()})
        (True, {'one': 2.645751311064591?})
        sage: G.reduce({(): QQbar(7).sqrt(), 1: 1})
        {1: 1}
    """

    def __init__(self, R, key_order=None, vector_of=None):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E._F.d, E.rank(), E.labels()
            (2, 0, [])
            sage: KeyedSpan(QQbar)._F is None
            True
        """
        self._R = R
        self._key_order = key_order
        self._vector_of = vector_of
        self._labels = []
        self._vecs = []
        self._pending = None
        from sage.rings.qqbar import AA, QQbar
        # QQbar / AA: the field is found from the first vectors (_field_for)
        self._F = None if (R is QQbar or R is AA) else _field_coordinates(R)
        self._reset_inner()

    def _reset_inner(self):
        r"""
        A fresh inner echelon over ``QQ`` (or ``Frac(QQ[x...])``) with the key
        order ``(key order, -i)`` on the keys ``(key, i)``, and fresh serials.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._q.rank()
            4
            sage: E._reset_inner(); E._q.rank(), E._serials
            (0, [])
        """
        from sage.rings.rational_field import QQ
        self._rank = {}          # keys -> number (without a key order)
        self._serials = []       # serial of the vector at each position
        self._pos_of = {}        # serial -> position
        self._nserial = 0
        ko, rank = self._key_order, self._rank
        if ko is None:
            iko = lambda kj: (rank[kj[0]], -kj[1])
        else:
            iko = lambda kj: (ko(kj[0]), -kj[1])
        self._q = KeyedSpan(getattr(self._F, 'base', QQ) if self._F is not None else QQ,
                                key_order=iko, vector_of=self._inner_vec)

    def _inner_vec(self, label):
        r"""
        The inner vector ``θ^j v`` of the label ``(serial, j)`` (the inner
        echelon's ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._q.labels()
            [(0, 0), (0, 1), (2, 0), (2, 1)]
            sage: E._inner_vec((0, 1))           # s (s e_1 + e_2)
            {(1, 0): 2, (2, 1): 1}
        """
        n, j = label
        return self._F.expand(self._outer_vec(self._pos_of[n]), j, self._rank)

    def _outer_vec(self, pos):
        r"""
        The independent vector over ``K`` at position ``pos``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._outer_vec(1)
            {2: 1, 3: s}
        """
        if self._vector_of is not None:
            return self._vector_of(self._labels[pos])
        return self._vecs[pos]

    @property
    def _dense(self):
        r"""
        True if the inner echelon has a dense block.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._dense
            False
        """
        return self._q._dense

    @property
    def _blocks(self):
        r"""
        The blocks of the inner echelon.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: len(E._blocks)
            2
        """
        return self._q._blocks

    def _field_for(self, vecs):
        r"""
        QQbar / AA: make the field contain the coefficients of ``vecs``
        (a larger field rebuilds the span); False if impossible.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: E = KeyedSpan(QQbar); E._F is None
            True
            sage: E._field_for([{1: QQbar(2).sqrt()}]), E._F.d
            (True, 2)
            sage: E.insert({1: QQbar(2).sqrt()}, 'p')
            True
            sage: E._field_for([{1: QQbar(3).sqrt()}]), E._F.d, E.labels()      # grown, rebuilt
            (True, 4, ['p'])
        """
        F = self._F
        if F is not None and F.alg is None:
            return True
        if F is None:
            self._F = _field_coordinates(self._R, vecs)
            return True
        try:
            for v in vecs:
                for c in v.values():
                    if c:
                        F.to_M(c)
            return True
        except ValueError:
            pass
        old = [self._outer_vec(i) for i in range(len(self._labels))]
        labels = list(self._labels)
        keep = dict(self.__dict__)          # restored if the rebuild is interrupted
        try:
            self._F = _field_coordinates(self._R, old + list(vecs))
            self._reset_inner()
            self._labels, self._vecs = [], []
            if old:
                self._insert_many(old, labels)
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(keep)
            raise
        return True

    def _insert_many(self, vecs, labels):
        r"""
        Insert ``vecs`` (over ``K``, the field already containing their
        coefficients) with ``labels``: the ``d`` vectors ``θ^j v`` of each go into
        the inner echelon; ``v`` is accepted iff its ``j = 0`` vector is.  An
        interrupted batch that accepted only some ``θ^j v`` of a vector is
        rebuilt from the accepted vectors.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._insert_many([{1: 1}, {1: s}], ['x', 'y']), E.labels()
            ([True, False], ['a', 'c', 'x'])
        """
        F = self._F
        d = F.d
        n = len(vecs)
        serials = list(range(self._nserial, self._nserial + n))
        self._nserial += n
        ex, exl = [], []
        for s, v in zip(serials, vecs):
            for j in range(d):
                ex.append(F.expand(v, j, self._rank))
                exl.append((s, j))
        q0 = len(self._q._labels)
        try:
            self._q.insert_many(ex, labels=exl)
        finally:
            # the accepted vectors: their j = 0 labels in the inner span (an
            # interrupted batch may stop inside a vector's d inserts: the
            # span over QQ is then not stable under θ, so rebuild)
            got = set(self._q._labels[q0:])
            out = []
            partial = False
            for s, v, l in zip(serials, vecs, labels):
                ok = (s, 0) in got
                if ok != all((s, j) in got for j in range(d)):
                    partial = True
                if ok:
                    self._pos_of[s] = len(self._labels)
                    self._serials.append(s)
                    self._labels.append(l)
                    if self._vector_of is None:
                        self._vecs.append(v)
                out.append(ok)
            if partial:
                kept = [self._outer_vec(i) for i in range(len(self._labels))]
                kl = list(self._labels)
                self._labels, self._vecs = [], []
                self._reset_inner()
                if kept:
                    self._insert_many(kept, kl)
        return out

    def insert_many(self, vecs, labels=None, greedy=True):
        r"""
        As :meth:`KeyedSpan.insert_many` (``QQbar`` / ``AA``: the field
        grown first if needed).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.insert_many([{4: s}, {4: 1}]), E.labels()
            ([True, False], ['a', 'c', None])
        """
        vecs = list(vecs)
        labels = [None] * len(vecs) if labels is None else list(labels)
        if not vecs:
            return []
        self._field_for(vecs)
        return self._insert_many(vecs, labels)

    def insert(self, v, label=None):
        r"""
        As :meth:`KeyedSpan.insert`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.insert({1: 2, 2: s}), E.insert({3: 1}, 'd')
            (False, True)
        """
        return self.insert_many([v], [label])[0]

    def _ex0(self, v):
        r"""
        The ``QQ``-vector of ``v`` (``j = 0``).  QQbar / AA: a coefficient
        outside the field grows the field first (it may still lie in the
        span: the coefficients of a combination are arbitrary).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._ex0({1: s, 2: 1/2})
            {(1, 1): 1, (2, 0): 1/2}
            sage: A = KeyedSpan(QQbar); A.insert({1: QQbar(2).sqrt()})
            True
            sage: w = A._ex0({1: QQbar(3).sqrt()}); A._F.d, len(w)        # the field grew
            (4, 2)
        """
        if self._F is not None:
            try:
                return self._F.expand(v, 0, self._rank)
            except ValueError:
                pass
        self._field_for([v])
        return self._F.expand(v, 0, self._rank)

    def contains(self, v):
        r"""
        As :meth:`KeyedSpan.contains`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.contains({1: 2, 2: s}), E.contains({1: 1})
            (True, False)
        """
        w = self._ex0(v)              # first: it may rebuild self._q
        return self._q.contains(w)

    def coordinates(self, v):
        r"""
        As :meth:`KeyedSpan.coordinates`: those of the ``j = 0`` vector,
        collapsed per label.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.coordinates({1: s + 2, 2: s + 2, 3: s}), E.coordinates({1: 1})
            ({'a': s + 1, 'c': 1}, None)
        """
        w = self._ex0(v)
        c = self._q.coordinates(w)
        if c is None:
            return None
        acc = {}
        for (n, j), q in c.items():
            acc.setdefault(n, {})[j] = q
        out = {}
        for n, cs in acc.items():
            x = self._F.element(cs)
            if x:
                out[self._labels[self._pos_of[n]]] = x
        return out

    def reduce(self, v):
        r"""
        As :meth:`KeyedSpan.reduce`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.reduce({1: 1, 2: 1})
            {1: -s + 1}
        """
        w = self._ex0(v)
        return self._F.collapse(self._q.reduce(w))

    def reduce_many(self, vs):
        r"""
        As :meth:`KeyedSpan.reduce_many`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.reduce_many([{1: 1, 2: 1}, {1: s, 2: 1}])
            [{1: -s + 1}, {}]
        """
        return [self._F.collapse(r) for r in self._q.reduce_many([self._ex0(v) for v in vs])]

    def pivots(self):
        r"""
        As :meth:`KeyedSpan.pivots`: the keys of the inner pivots
        ``(key, 0)``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.pivots(), E._q.pivots()
            ([3, 2], [(3, 0), (3, 1), (2, 0), (2, 1)])
        """
        return [k for k, j in self._q.pivots() if j == 0]

    def rows(self):
        r"""
        As :meth:`KeyedSpan.rows`: the inner rows of the pivots ``(key,
        0)``, collapsed.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.rows()
            [{1: -1, 3: 1}, {1: s, 2: 1}]
        """
        return [self._F.collapse(r) for (k, j), r in zip(self._q.pivots(), self._q.rows()) if j == 0]

    def row_dict(self):
        r"""
        As :meth:`KeyedSpan.row_dict`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E.row_dict()
            {2: {1: s, 2: 1}, 3: {1: -1, 3: 1}}
        """
        return dict(zip(self.pivots(), self.rows()))

    def _export(self):
        r"""
        The state over ``K`` (labels and vectors; rebuilt by :meth:`_import`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E._export()
            {'dense': False,
             'expanded': True,
             'labels': ['a', 'c'],
             'vecs': [{1: s, 2: 1}, {2: 1, 3: s}]}
        """
        return {'expanded': True, 'labels': self._labels, 'vecs': self._vecs,
                'dense': self._dense}

    def _import(self, st):
        r"""
        Rebuild from the independent vectors of a state (also of states
        saved before 2026-10-12, eliminated over ``K``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import KeyedSpan
            sage: K.<s> = QuadraticField(2)
            sage: E = KeyedSpan(K, key_order=lambda k: k)
            sage: E.insert_many([{1: s, 2: 1}, {1: 2, 2: s}, {2: 1, 3: s}], 'abc')
            [True, False, True]
            sage: E2 = KeyedSpan(K, key_order=lambda k: k); E2._import(E._export())
            sage: E2.rows() == E.rows(), E2.labels()
            (True, ['a', 'c'])
        """
        labels = list(st['labels'])
        vecs = list(st['vecs'])
        if self._vector_of is not None:
            vecs = [self._vector_of(l) for l in labels]
        self._labels, self._vecs = [], []
        self._reset_inner()
        if vecs:
            if not all(self.insert_many(vecs, labels)):
                raise ValueError('the saved vectors of a span are not independent')


def _expanded_left_kernel(vecs, R):
    r"""
    :func:`left_kernel` over a number field, ``QQbar`` or ``AA``: the
    relations over ``QQ`` among all ``θ^j v_i`` (in the order ``(i, j)``)
    whose pivot is a ``j = 0`` vector are the ``K``-relations (the others
    are their ``θ``-multiples): the same canonical basis as over ``K``.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _expanded_left_kernel, left_kernel
        sage: K.<s> = QuadraticField(2)
        sage: _expanded_left_kernel([{1: 1}, {1: s}, {2: 1}, {1: 1, 2: s}], K)
        [{0: -s, 1: 1}, {0: -1, 2: -s, 3: 1}]
        sage: left_kernel([{1: 1}, {1: s}, {2: 1}, {1: 1, 2: s}], K)
        [{0: -s, 1: 1}, {0: -1, 2: -s, 3: 1}]
    """
    from sage.rings.rational_field import QQ
    F = _field_coordinates(R, vecs)
    d = F.d
    ex = [F.expand(v, j) for v in vecs for j in range(d)]
    out = []
    for r in left_kernel(ex, getattr(F, 'base', QQ)):
        i0, j0 = divmod(max(r), d)
        if j0:
            continue
        acc = {}
        for idx, q in r.items():
            i, j = divmod(idx, d)
            acc.setdefault(i, {})[j] = q
        rel = {}
        for i, cs in sorted(acc.items()):
            x = F.element(cs)
            if x:
                rel[i] = x
        out.append(rel)
    return out


# --------------------------------------------------------------------------- #
#  Rational function fields: the elimination in C++                           #
# --------------------------------------------------------------------------- #

# Whether blocks over ``Frac(QQ[x_1, ..., x_n])`` eliminate in C++
# (:class:`_FracBlock`, FLINT ``fmpz_mpoly_q``); read when a block is made.
# A developer switch for the tests (C++ against Python elimination), not
# an option: users choose backends with ``VERTEX_ALGEBRAS_NO_ENGINE``.
_USE_FRAC_ENGINE = True

_LAE = None
_LAE_OFF = None
_LAE_TRIED = False


def _frac_engine():
    r"""
    The compiled ``_linear_algebra_engine`` (imported on first use), or
    ``None`` when switched off (``VERTEX_ALGEBRAS_NO_ENGINE`` /
    ``VERTEX_ALGEBRAS_NO_CORE``, see ``vertex_algebra._compiled_module()``).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _frac_engine
        sage: L = _frac_engine()
        sage: L is None or hasattr(L, 'Echelon')
        True

    TESTS:

    The build is all or nothing: this module is present exactly when the
    vertex algebra engines are (a build without it once made every span
    fall back to Sage's arithmetic silently)::

        sage: from vertex_algebras.vertex_algebra import _compiled_module
        sage: names = ['_universal_engine', '_lattice_engine', '_tensor_engine',
        ....:          '_module_engine', '_linear_algebra_engine']
        sage: present = [_compiled_module(n)[0] is not None for n in names]
        sage: all(present) or not any(present)
        True
        sage: (L is None) == (not any(present))
        True
    """
    global _LAE, _LAE_OFF, _LAE_TRIED
    if not _LAE_TRIED:
        from .vertex_algebra import _compiled_module
        _LAE, _LAE_OFF = _compiled_module('_linear_algebra_engine')
        _LAE_TRIED = True
    return _LAE


def _frac_nvars(R):
    r"""
    The number of variables if ``R`` is ``Frac(QQ[x_1, ..., x_n])`` and
    the C++ elimination is available and enabled, else 0.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _frac_nvars, _frac_engine
        sage: n = _frac_nvars(Frac(QQ['c', 'd'])); n == (2 if _frac_engine() is not None else 0)
        True
        sage: _frac_nvars(QQ), _frac_nvars(Frac(QuadraticField(2)['c'])), _frac_nvars(QQ['c'])
        (0, 0, 0)
    """
    if not _USE_FRAC_ENGINE:
        return 0
    from sage.rings.rational_field import QQ
    try:
        A = R.ring()
        if A is R or R is not A.fraction_field() or A.base_ring() is not QQ:
            return 0
        n = A.ngens()
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        return 0
    return n if _frac_engine() is not None else 0


class _KeyIds:
    r"""
    Integer ids for the keys of a C++ echelon (``self._E``) and, with a
    ``key_order``, ranks for its pivot choice: the keys in a sorted list,
    ranks with gaps (all renumbered when there is no room).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
        sage: F.<c> = Frac(QQ['c'])
        sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
        sage: B is None or (B._id('x'), B._id('a'), B._id('x')) == (0, 1, 0)
        True
        sage: B is None or B._sorted == [('a', 1), ('x', 0)] and B._ranks[1] < B._ranks[0]
        True
    """

    _GAP = 1 << 32

    def _init_ids(self):
        r"""
        Empty id tables (called by the constructors).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _KeyIds
            sage: k = _KeyIds(); k._init_ids(); k._ids, k._keys_by_id, k._sorted, k._ranks
            ({}, [], [], {})
        """
        self._ids = {}           # key -> id
        self._keys_by_id = []
        self._sorted = []        # (key order value, id), sorted (with a key order)
        self._ranks = {}

    def _id(self, k):
        r"""
        The id of the key ``k``, a new one (and a rank, with a ``key_order``) if
        ``k`` is new.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or (B._id('x'), B._id('y'), B._id('x'), B._keys_by_id) == (0, 1, 0, ['x', 'y'])
            True
        """
        i = self._ids.get(k)
        if i is None:
            i = self._ids[k] = len(self._keys_by_id)
            self._keys_by_id.append(k)
            if self._key_order is not None:
                self._new_rank(k, i)
        return i

    def _new_rank(self, k, i):
        r"""
        Give the new key ``k`` (id ``i``) a rank between those of its
        neighbours in the key order (gaps of ``_GAP``), renumbering all ranks when
        there is no room; the ranks are passed to the C++ echelon.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or [B._id(k) for k in (10, 30, 20)] == [0, 1, 2]
            True
            sage: B is None or B._ranks == {0: 0, 1: 2^32, 2: 2^31}
            True
            sage: from vertex_algebras._keyed_spans import _KeyIds
            sage: old = _KeyIds._GAP; _KeyIds._GAP = 1                       # no room: renumbered
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or [B._id(k) for k in (10, 30, 20)] == [0, 1, 2] and B._ranks == {0: 0, 2: 1, 1: 2}
            True
            sage: _KeyIds._GAP = old
        """
        import bisect
        val = self._key_order(k)
        s = self._sorted
        pos = bisect.bisect_left(s, (val, -1))
        s.insert(pos, (val, i))
        ranks = self._ranks
        lo = ranks[s[pos - 1][1]] if pos > 0 else None
        hi = ranks[s[pos + 1][1]] if pos + 1 < len(s) else None
        if lo is None and hi is None:
            r = 0
        elif lo is None:
            r = hi - self._GAP
        elif hi is None:
            r = lo + self._GAP
        elif hi - lo > 1:
            r = (lo + hi) // 2
        else:
            for j, (_, kid) in enumerate(s):
                ranks[kid] = j * self._GAP
            self._E.set_ranks(list(ranks.items()))
            return
        ranks[i] = r
        self._E.set_ranks([(i, r)])

    def _take_ids(self, other):
        r"""
        Take over the C++ echelon and the id tables of ``other`` (after a replay
        with combos).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B2 = _FracBlock(F) if B is not None else None
            sage: B is None or B._id('x') == 0 and (B2._take_ids(B) or B2._E is B._E and B2._ids == {'x': 0})
            True
        """
        (self._E, self._ids, self._keys_by_id, self._sorted,
         self._ranks) = (other._E, other._ids, other._keys_by_id, other._sorted, other._ranks)


class _CommittedBatches:
    r"""
    The labels (and kept vectors) of a span eliminated in C++, recorded from
    the C++ echelon's flags of its last batch (``done()``: also the inserts
    committed before an interrupt).

    The C++ echelon commits first and Python records afterwards, so an
    interrupt between the two (in the recording itself) would leave rows
    without labels, and a re-inserted batch would then find them dependent:
    their labels lost for good, wrong dimensions.  So a batch is registered
    as ``_pending`` before the C++ call, and :meth:`_sync` (run by every
    access to ``_labels`` / ``_vecs``) records it, publishing the new lists
    in one assignment: an interrupt anywhere leaves the batch pending, and
    the next access records it.

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
        sage: F.<c> = Frac(QQ['c'])
        sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
        sage: B is None or B.insert_many([{1: c, 2: 1}], 'u') == [True]
        True

    A batch committed in C++ but not yet recorded (as after an interrupt
    right after the C++ call) is recorded by the next access::

        sage: if B is not None:
        ....:     vecs = [{1: c^2, 2: c}, {2: 1, 3: c}]
        ....:     _ = B._E.insert_many([])
        ....:     B._pending = (vecs, ['v', 'w'])
        ....:     flags = B._E.insert_many([B._vec_in(v) for v in vecs])
        sage: B is None or (B.labels(), B.rank()) == (['u', 'w'], 2)
        True
    """

    _pending = None

    @property
    def _labels(self):
        r"""
        The labels of the independent inserted vectors (recorded first).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or (B.insert_many([{1: c}], 'u') and B._labels == ['u'])
            True
        """
        self._sync()
        return self._labels_

    @_labels.setter
    def _labels(self, labels):
        r"""
        Set the labels (the stored list).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _CommittedBatches
            sage: C = _CommittedBatches(); C._labels = ['a']; C._labels_
            ['a']
        """
        self._labels_ = labels

    @property
    def _vecs(self):
        r"""
        The kept independent vectors (recorded first; empty with ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or (B.insert_many([{1: c, 2: 0}], 'u') and B._vecs == [{1: c}])
            True
        """
        self._sync()
        return self._vecs_

    @_vecs.setter
    def _vecs(self, vecs):
        r"""
        Set the kept vectors (the stored list).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _CommittedBatches
            sage: C = _CommittedBatches(); C._vecs = [{1: 2}]; C._vecs_
            [{1: 2}]
        """
        self._vecs_ = vecs

    def _kept(self, v):
        r"""
        The form in which the independent vector ``v`` is kept.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _CommittedBatches
            sage: _CommittedBatches()._kept({1: 2})
            {1: 2}
        """
        return v

    def _sync(self):
        r"""
        Record the committed inserts of the pending batch.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or (B._sync() is None and B._pending is None)
            True
        """
        p = self._pending
        if p is None:
            return
        vecs, labels = p
        new = [(v, l) for v, l, ok in zip(vecs, labels, self._E.done()) if ok]
        kept = self._vecs_
        if self.__dict__.get('_vector_of') is None:
            kept = kept + [self._kept(v) for v, _ in new]
        self._labels_, self._vecs_, self._pending = (
            self._labels_ + [l for _, l in new], kept, None)

    def _insert_batch(self, vecs, labels, evs, insert):
        r"""
        Insert the converted vectors ``evs`` with ``insert`` (the C++ call),
        registering the batch first; the flags.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or B._insert_batch([{1: c}], ['u'], [B._vec_in({1: c})], B._E.insert_many) == [True]
            True
        """
        self._sync()
        insert([])                  # empties done(): no stale flags if interrupted here
        self._pending = (vecs, labels)
        flags = insert(evs)
        self._sync()
        return flags


class _FracBlock(_CommittedBatches, _KeyIds):
    r"""
    A block of a :class:`KeyedSpan` over ``F = Frac(QQ[x_1, ..., x_n])``
    eliminated in C++ (``_linear_algebra_engine``: the algorithm of
    :class:`_Block`, so the same labels, pivots and rows, on FLINT's
    canonical fractions ``fmpz_mpoly_q``).  Keys get integer ids; with a
    ``key_order`` their ranks (for the pivot choice) are kept in a sorted
    list with gaps.  The combos are kept only with ``combos=True`` (else
    built on the first :meth:`coordinates` by replaying the elimination, as
    :class:`_Block` over rings other than ``QQ``).

    EXAMPLES::

        sage: from vertex_algebras._keyed_spans import KeyedSpan, _FracBlock, _frac_engine
        sage: SLA_has_engine = _frac_engine() is not None
        sage: F.<c> = Frac(QQ['c'])
        sage: E = KeyedSpan(F, key_order=lambda k: k)
        sage: E.insert_many([{1: c, 2: 1/(c+1)}, {1: c^2, 2: c/(c+1)}, {2: 1, 3: c}],
        ....:               labels=['u', 'v', 'w'])
        [True, False, True]
        sage: type(E._blocks[0]) is _FracBlock or not SLA_has_engine
        True
        sage: E.coordinates({1: 2*c, 2: 2/(c+1) + 1, 3: c})
        {'u': 2, 'w': 1}
        sage: E.pivots(), [sorted(r.items()) for r in E.rows()]
        ([3, 2], [[(1, -c - 1), (3, 1)], [(1, c^2 + c), (2, 1)]])

    TESTS:

    Over ``QQ(x)`` and ``QQ(y_1, y_2)`` the spans (in C++ when the engine is
    available) are those of the Python elimination :class:`_Block`: flags,
    labels, pivots, rows, membership, coordinates, normal forms, export and
    import, and two blocks merged by a vector joining them::

        sage: from vertex_algebras._keyed_spans import _Block
        sage: F1.<x> = Frac(QQ['x']); F2.<y1, y2> = Frac(QQ['y1, y2'])
        sage: def fvec(keys, p, gens):
        ....:     return {k: ZZ.random_element(-3, 4) + choice(gens) * ZZ.random_element(-2, 3)
        ....:             for k in keys if random() < p}
        sage: def fcombo(vs, gens):
        ....:     out = {}
        ....:     for v in vs:
        ....:         a = ZZ.random_element(-2, 3) + choice(gens)
        ....:         for k, b in v.items(): out[k] = out.get(k, 0) + a * b
        ....:     return {k: b for k, b in out.items() if b}
        sage: set_random_seed(5)
        sage: ok = True
        sage: for R, gens in [(F1, [x, 1/(x + 2), x^2/3]), (F2, [y1, y2/(y1 - 1), y1*y2 + 1])]:
        ....:     order = lambda k: (k % 3, -k)
        ....:     base = [fvec(range(12), 0.4, gens) for _ in range(8)]
        ....:     vecs = base + [fcombo(sample(base, 2), gens) for _ in range(3)]
        ....:     shuffle(vecs)
        ....:     E = KeyedSpan(R, key_order=order)
        ....:     B = _Block(R, key_order=order); B._combos_ok = True
        ....:     flags = E.insert_many(vecs, range(len(vecs)))
        ....:     ref = [B.insert(v, l) for l, v in enumerate(vecs)]
        ....:     tests = [fcombo(sample(vecs, 3), gens) for _ in range(3)] + [fvec(range(12), 0.3, gens)]
        ....:     F = KeyedSpan(R, key_order=order); F._import(E._export())
        ....:     ok = (ok and flags == ref and E.labels() == B.labels() == F.labels()
        ....:           and E.pivots() == sorted(B.pivots(), key=order, reverse=True)
        ....:           and E.row_dict() == B.row_dict()
        ....:           and all(E.contains(t) == B.contains(t)
        ....:                   and E.coordinates(t) == B.coordinates(t) == F.coordinates(t)
        ....:                   and E.normal_form(t) == B.reduce(t) for t in tests))
        ....:     M = KeyedSpan(R, key_order=lambda k: k)
        ....:     a = [fvec(range(5), 0.8, gens) for _ in range(3)]
        ....:     b = [fvec(range(10, 15), 0.8, gens) for _ in range(3)]
        ....:     _ = M.insert_many(a + b); nb = len(M._blocks)
        ....:     j = dict(a[0]); j.update(b[0]); j[20] = R(1)
        ....:     ok = (ok and nb == 2 and M.insert(j) and len(M._blocks) == 1 and M.rank() == 7
        ....:           and M.contains(fcombo(a + b + [j], gens)))
        sage: ok
        True
    """

    _dense = False

    def __init__(self, R, key_order=None, vector_of=None, combos=False):
        r"""
        TESTS::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k, combos=True) if _frac_engine() is not None else None
            sage: B is None or (B.rank(), B.labels(), B._combos_ok, B._uni) == (0, [], True, True)
            True
        """
        self._R = R
        self._P = R.ring()
        self._key_order = key_order
        self._vector_of = vector_of
        self._labels = []
        self._vecs = []
        self._init_ids()
        self._E = _frac_engine().Echelon(self._P.ngens(), bool(combos))
        self._combos_ok = bool(combos)
        from sage.rings.polynomial.polynomial_ring import PolynomialRing_generic
        self._uni = isinstance(self._P, PolynomialRing_generic)

    def rank(self):
        r"""
        The dimension of the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.rank() == 2
            True
        """
        return len(self._labels)

    def _independent(self):
        r"""
        The independent inserted vectors, in order.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B._independent() == [{1: c, 2: 1}, {2: 1, 3: c}]
            True
        """
        if self._vector_of is not None:
            return [self._vector_of(l) for l in self._labels]
        return self._vecs

    def _keep(self, v):
        r"""
        Store the independent vector ``v`` (unless ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or (B._keep({1: 1}) or B._vecs == [{1: 1}])
            True
        """
        if self._vector_of is None:
            self._vecs.append(v)

    def _kept(self, v):
        r"""
        The independent vector ``v`` as kept: without zero entries.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F) if _frac_engine() is not None else None
            sage: B is None or B._kept({1: c, 2: 0}) == {1: c}
            True
        """
        return {k: c for k, c in v.items() if c}

    def _key_set(self):
        r"""
        All keys seen by the block.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B._key_set() == {1, 2, 3}
            True
        """
        return set(self._ids)

    def _fill(self, ncols):
        r"""
        Always 0: the C++ block never switches to a dense backend.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B._fill(3) == 0
            True
        """
        return 0

    # -- conversions ------------------------------------------------------ #

    def _poly_terms(self, p):
        r"""
        The terms ``(exponent tuple, numerator, denominator)`` of a polynomial
        of ``QQ[x...]`` (the engine's input format).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or sorted(B._poly_terms(F.ring()(3*c^2 + 1/2))) == [((0,), 1, 2), ((2,), 3, 1)]
            True
        """
        if self._uni:
            return [((int(e),), int(c.numerator()), int(c.denominator()))
                    for e, c in p.dict().items()]
        return [(tuple(e), int(c.numerator()), int(c.denominator()))
                for e, c in p.dict().items()]

    def _vec_in(self, v):
        r"""
        A raw dict over ``F`` in the engine's form: ``(key id, numerator
        terms, denominator terms)`` per nonzero entry.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or B._vec_in({'k': c/(c + 1), 'z': 0}) == [(0, [((1,), 1, 1)], [((0,), 1, 1), ((1,), 1, 1)])]
            True
        """
        out = []
        for k, c in v.items():
            if c:
                c = self._R(c)
                out.append((self._id(k), self._poly_terms(c.numerator()),
                            self._poly_terms(c.denominator())))
        return out

    def _poly(self, terms):
        r"""
        The polynomial of the engine's terms (integer coefficients: the
        engine's canonical fractions have them, so the denominators are ignored).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or B._poly([((2,), 3, 1), ((0,), 1, 1)]) == 3*c^2 + 1
            True
        """
        if self._uni:
            return self._P({e[0]: a for e, a, _ in terms})
        return self._P({e: a for e, a, _ in terms})

    def _frac(self, num, den):
        r"""
        The fraction of the engine's numerator and denominator terms.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: B is None or B._frac([((1,), 1, 1)], [((0,), 1, 1), ((1,), 1, 1)]) == c/(c + 1)
            True
        """
        return self._R(self._poly(num)) / self._R(self._poly(den))

    def _vec_out(self, d):
        r"""
        A raw dict over ``F`` from the engine's ``[(key id, num, den)]``.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B._vec_out([(0, [((1,), 1, 1)], [((0,), 2, 1)])]) == {1: c/2}
            True
        """
        keys = self._keys_by_id
        return {keys[k]: self._frac(n, m) for k, n, m in d}

    # -- the span --------------------------------------------------------- #

    def insert_many(self, vecs, labels=None, greedy=True):
        r"""
        As :meth:`_Block.insert_many` (in C++; atomic inserts: an interrupted
        batch records its committed inserts).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.insert_many([{4: c}, {4: 1}], 'xy') == [True, False] and B.labels() == ['u', 'w', 'x']
            True
        """
        vecs = list(vecs)
        labels = [None] * len(vecs) if labels is None else list(labels)
        evs = [self._vec_in(v) for v in vecs]
        return self._insert_batch(vecs, labels, evs, self._E.insert_many)

    def insert(self, v, label=None):
        r"""
        As :meth:`_Block.insert`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or (B.insert({1: 1}, 'x'), B.insert({3: c}, 'y')) == (True, False)
            True
        """
        return self.insert_many([v], [label])[0]

    def contains(self, v):
        r"""
        True if ``v`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.contains({1: c, 2: 1}) and not B.contains({1: 1})
            True
        """
        return self._E.contains(self._vec_in(v))

    def reduce(self, v):
        r"""
        The remainder of ``v`` modulo the span (no pivot keys).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.reduce({3: 1}) == {1: 1}
            True
        """
        return self._vec_out(self._E.reduce(self._vec_in(v)))

    def _ensure_combos(self):
        r"""
        Without combos: replay the elimination with combos in a new C++ echelon
        and take it over (as :meth:`_Block._ensure_combos`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or not B._combos_ok and (B._ensure_combos() or B._combos_ok)
            True
        """
        if self._combos_ok:
            return
        b = _FracBlock(self._R, key_order=self._key_order, combos=True)
        flags = b.insert_many(self._independent(), labels=list(range(len(self._labels))))
        if not all(flags):
            raise RuntimeError('replaying the elimination lost a vector')
        self._take_ids(b)
        self._combos_ok = True

    def coordinates(self, v):
        r"""
        As :meth:`_Block.coordinates`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.coordinates({1: 2*c, 2: 3, 3: c}) == {'u': 2, 'w': 1}
            True
            sage: B is None or B.coordinates({1: 1}) is None
            True
        """
        self._ensure_combos()
        d = self._E.coordinates(self._vec_in(v))
        if d is None:
            return None
        return {self._labels[i]: self._frac(n, m) for i, n, m in d}

    def pivots(self):
        r"""
        As :meth:`_Block.pivots`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.pivots() == [3, 2]
            True
        """
        ps = [self._keys_by_id[p] for p in self._E.pivots()]
        if self._key_order is not None:
            ps.sort(key=self._key_order, reverse=True)
        return ps

    def rows(self):
        r"""
        As :meth:`_Block.rows`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.rows() == [{1: -1, 3: 1}, {1: c, 2: 1}]
            True
        """
        return [self._vec_out(self._E.row(self._ids[p])) for p in self.pivots()]

    def normal_form(self, v):
        r"""
        As :meth:`_Block.normal_form`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.normal_form({3: 1}) == B.normal_form({1: 1})
            True
        """
        return self.reduce(v)

    def labels(self):
        r"""
        As :meth:`_Block.labels`.

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B.labels() == ['u', 'w']
            True
        """
        return list(self._labels)

    def _export(self):
        r"""
        The state: the labels and independent vectors (the rows are rebuilt by
        :meth:`_import`).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B is None or B._export() == {'frac': True, 'labels': ['u', 'w'], 'vecs': [{1: c, 2: 1}, {2: 1, 3: c}], 'dense': False, 'rows': {}}
            True
        """
        return {'frac': True, 'labels': self._labels, 'vecs': self._vecs,
                'dense': False, 'rows': {}}

    def _import(self, st):
        r"""
        Rebuild from the independent vectors (also states of a
        :class:`_Block`, saved before 2026-10-12).

        EXAMPLES::

            sage: from vertex_algebras._keyed_spans import _FracBlock, _frac_engine
            sage: F.<c> = Frac(QQ['c'])
            sage: B = _FracBlock(F, key_order=lambda k: k) if _frac_engine() is not None else None
            sage: vecs = [{1: c, 2: 1}, {1: c^2, 2: c}, {2: 1, 3: c}]
            sage: B is None or B.insert_many(vecs, 'uvw') == [True, False, True]
            True
            sage: B2 = _FracBlock(F, key_order=lambda k: k) if B is not None else None
            sage: B is None or (B2._import(B._export()) or B2.rows() == B.rows() and B2.labels() == ['u', 'w'])
            True
        """
        labels = list(st['labels'])
        vecs = list(st['vecs'])
        if self._vector_of is not None:
            vecs = [self._vector_of(l) for l in labels]
        if vecs and not all(self.insert_many(vecs, labels)):
            raise ValueError('the saved vectors of a span are not independent')


class _EngineSpan(_CommittedBatches, _KeyIds):
    r"""
    A span of vectors in a C++ engine's own form (lists of ``(key, terms)``
    over the engine's ring, :meth:`vertex_algebra._EngineKernel.to_engine`),
    eliminated in C++ (``_linear_algebra_engine``) over the engine's field:
    a number field (``NFEchelon_``: kind ``'nf'``, also towers, QQbar, AA)
    or ``Frac(QQ[x...])`` (``Echelon`` on the lifted polynomials, ``u_j =
    1/p_j``).  So span vectors never become Sage elements (the conversions
    cost more than the elimination: a W_3 subalgebra over Q(c)).  Queries
    with raw dicts over the base ring ``F`` (``contains``, ``coordinates``)
    lift them through the kernel; ``ValueError`` if one cannot be lifted (a
    QQbar number outside the field: the owner falls back to the Sage path).
    Coordinates come back over ``F``.  Interface as :class:`KeyedSpan`
    for the subalgebra spans (``insert_many`` takes engine-form vectors).

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
        sage: K.<s> = QuadraticField(5)
        sage: V = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})
        sage: E = V._engine()
        sage: ok = E is None or _frac_engine() is None
        sage: if not ok:
        ....:     S = _EngineSpan(E, E.span_field())
        ....:     vs = [E.to_engine({((0, 1),): s, ((0, 0),): 1}), E.to_engine({((0, 1),): 5, ((0, 0),): s})]
        ....:     ok = S.insert_many(vs, ['a', 'b']) == [True, False]
        ....:     ok = ok and S.coordinates({((0, 1),): 2*s, ((0, 0),): 2}) == {'a': 2}
        sage: ok
        True
    """

    _dense = False

    def __init__(self, kernel, spec, key_order=None, vector_of=None):
        r"""
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: V = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})
            sage: E = V._engine()
            sage: S = None if E is None or _frac_engine() is None else _EngineSpan(E, E.span_field())
            sage: S is None or (S._K is E, S.rank(), S.labels(), S._spec[0]) == (True, 0, [], 'nf')
            True
        """
        self._K = kernel
        self._spec = spec
        self._key_order = key_order
        self._vector_of = vector_of       # label -> engine-form vector
        self._labels = []
        self._vecs = []
        self._combos_ok = False
        self._init_ids()
        self._E = self._new_engine(False)
        if spec[0] == 'frac':
            self._F = kernel._F
            self._P = self._F.ring()
            from sage.rings.polynomial.polynomial_ring import PolynomialRing_generic
            self._uni = isinstance(self._P, PolynomialRing_generic)

    def _new_engine(self, combos):
        r"""
        A new C++ echelon over the engine's field (``combos``: keep the combos).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: S is None or type(S._new_engine(False)).__name__ == 'NFEchelon_'
            True
        """
        L = _frac_engine()
        if self._spec[0] == 'nf':
            return L.NFEchelon_(self._spec[1], combos)
        E = L.Echelon(self._spec[1], combos)
        E.set_subst(self._spec[2])
        return E

    def rank(self):
        r"""
        The dimension of the span.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.rank() == 2
            True
        """
        return len(self._labels)

    def labels(self):
        r"""
        The labels of the independent inserted vectors, in order.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.labels() == ['a', 'c']
            True
        """
        return list(self._labels)

    def _ev(self, ev):
        r"""
        An engine-form vector with key ids instead of keys.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S._ev(E.to_engine(a)) == [(0, [((1,), 1, 1)]), (1, [((0,), 1, 1)])]
            True
        """
        return [(self._id(k), t) for k, t in ev]

    def _insert(self, evs):
        r"""
        Insert vectors of :meth:`_ev` into the C++ echelon (lifted
        polynomials for ``Frac(QQ[x...])``).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S._insert([S._ev(E.to_engine(b))]) == [False]
            True
        """
        if self._spec[0] == 'nf':
            return self._E.insert_many(evs)
        return self._E.insert_many_lifted(evs)

    def insert_many(self, vecs, labels=None, greedy=True):
        r"""
        Insert engine-form vectors.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.insert_many([E.to_engine({((0, 2),): 1})], ['d']) == [True] and S.labels() == ['a', 'c', 'd']
            True
        """
        vecs = list(vecs)
        labels = [None] * len(vecs) if labels is None else list(labels)
        evs = [self._ev(v) for v in vecs]
        return self._insert_batch(vecs, labels, evs, self._insert)

    def insert(self, v, label=None):
        r"""
        Insert one engine-form vector.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.insert(E.to_engine(b), 'x') is False
            True
        """
        return self.insert_many([v], [label])[0]

    def _lift(self, d):
        r"""
        A raw dict over ``F`` in the form of :meth:`_ev` (``ValueError`` if the
        kernel cannot lift it).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S._lift({((0, 0),): s}) == [(1, [((1,), 1, 1)])]
            True
        """
        return self._ev(self._K.to_engine(d))

    def contains(self, d):
        r"""
        ``d`` a raw dict over ``F``.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.contains({((0, 1),): 5*s, ((0, 0),): 5}) and not S.contains({((0, 0),): 1})
            True
        """
        w = self._lift(d)
        if self._spec[0] == 'nf':
            return self._E.contains(w)
        return self._E.contains_lifted(w)

    def contains_engine(self, ev):
        r"""
        True if the engine-form vector ``ev`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.contains_engine(E.to_engine(b))
            True
        """
        w = self._ev(ev)
        if self._spec[0] == 'nf':
            return self._E.contains(w)
        return self._E.contains_lifted(w)

    def _coef(self, c):
        r"""
        A coefficient of the engine (number field terms, or a pair of lifted
        numerator / denominator terms) as an element of ``F``.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S._coef(E.to_engine({((0, 0),): s})[0][1]) == s
            True
            sage: from vertex_algebras import W3
            sage: F.<c> = Frac(QQ['c']); EW = W3(F, c)._engine()
            sage: T = _EngineSpan(EW, EW.span_field()) if EW is not None and _frac_engine() is not None else None
            sage: T is None or T._coef(([((1,), 1, 1)], [((0,), 1, 1)])) == c
            True
        """
        if self._spec[0] == 'nf':
            return self._K.coef_from_engine(c)
        num, den = c
        P = self._P
        if self._uni:
            n, d = P({e[0]: a for e, a, _ in num}), P({e[0]: a for e, a, _ in den})
        else:
            n, d = P({e: a for e, a, _ in num}), P({e: a for e, a, _ in den})
        return self._F(n) / self._F(d)

    def _ensure_combos(self):
        r"""
        Replay the elimination with combos in a new C++ echelon and take it
        over (combos are needed by :meth:`coordinates` only).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or not S._combos_ok and (S._ensure_combos() or S._combos_ok)
            True
        """
        if self._combos_ok:
            return
        b = _EngineSpan(self._K, self._spec, key_order=self._key_order)
        b._E = b._new_engine(True)
        evs = ([self._vector_of(l) for l in self._labels] if self._vector_of is not None
               else self._vecs)
        if not all(b.insert_many(evs, list(range(len(self._labels))))):
            raise RuntimeError('replaying the elimination lost a vector')
        self._take_ids(b)
        self._combos_ok = True

    def coordinates(self, d):
        r"""
        The coefficients ``{label: c}`` (over ``F``) of the raw dict ``d`` over
        ``F`` in the independent vectors, or ``None`` if it is not in the span.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.coordinates({((0, 1),): 2*s, ((0, 0),): 2}) == {'a': 2}
            True
            sage: S is None or S.coordinates({((0, 0),): 1}) is None
            True
            sage: from vertex_algebras import W3
            sage: F.<c> = Frac(QQ['c']); EW = W3(F, c)._engine()
            sage: T = _EngineSpan(EW, EW.span_field()) if EW is not None and _frac_engine() is not None else None
            sage: T is None or (T.insert_many([EW.to_engine({((0, 0),): 1/(5*c + 22)})], ['x']) == [True]
            ....:               and T.coordinates({((0, 0),): c}) == {'x': 5*c^2 + 22*c})
            True
        """
        self._ensure_combos()
        w = self._lift(d)
        if self._spec[0] == 'nf':
            r = self._E.coordinates(w)
            return None if r is None else {self._labels[i]: self._coef(c) for i, c in r}
        r = self._E.coordinates_lifted(w)
        return None if r is None else {self._labels[i]: self._coef((n, m)) for i, n, m in r}

    def pivots(self):
        r"""
        The pivot keys (decreasing ``key_order`` with one).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.pivots() == [((0, 1),), ((0, 0), (0, 0))]
            True
        """
        ps = [self._keys_by_id[p] for p in self._E.pivots()]
        if self._key_order is not None:
            ps.sort(key=self._key_order, reverse=True)
        return ps

    def rows(self):
        r"""
        The rows of the reduced echelon form as raw dicts over ``F``, in the
        order of :meth:`pivots`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.rows() == [{((0, 1),): 1, ((0, 0),): s/5}, {((0, 0), (0, 0)): 1}]
            True
        """
        out = []
        for p in self.pivots():
            r = self._E.row(self._ids[p])
            if self._spec[0] == 'nf':
                out.append({self._keys_by_id[k]: self._coef(c) for k, c in r})
            else:
                out.append({self._keys_by_id[k]: self._coef((n, m)) for k, n, m in r})
        return out

    def row_dict(self):
        r"""
        The reduced echelon form as ``{pivot: row}``.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S.row_dict()[((0, 1),)] == {((0, 1),): 1, ((0, 0),): s/5}
            True
        """
        return dict(zip(self.pivots(), self.rows()))

    def _export(self):
        r"""
        The state: the labels only (the vectors are the owner's: ``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: S is None or S._export() == {'engine': True, 'labels': ['a', 'c'], 'vecs': [], 'dense': False}
            True
        """
        return {'engine': True, 'labels': self._labels, 'vecs': [], 'dense': False}

    def _import(self, st):
        r"""
        Rebuild from the independent vectors (``vector_of``).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine
            sage: K.<s> = QuadraticField(5)
            sage: E = UniversalVertexAlgebra(K, ['J'], {('J', 'J'): [0, 1]})._engine()
            sage: ok = E is not None and _frac_engine() is not None
            sage: S = _EngineSpan(E, E.span_field(), key_order=lambda k: k) if ok else None
            sage: a, b, c = {((0, 1),): s, ((0, 0),): 1}, {((0, 1),): 5, ((0, 0),): s}, {((0, 0), (0, 0)): 1}
            sage: S is None or (S.insert_many([E.to_engine(a), E.to_engine(b), E.to_engine(c)], 'abc')
            ....:               == [True, False, True])
            True
            sage: d = {'a': a, 'c': c}
            sage: S2 = _EngineSpan(E, E.span_field(), key_order=lambda k: k,
            ....:                  vector_of=lambda l: E.to_engine(d[l])) if ok else None
            sage: S2 is None or (S2._import(S._export()) or S2.labels() == ['a', 'c'] and S2.rows() == S.rows())
            True
        """
        labels = list(st['labels'])
        if labels and not all(self.insert_many([self._vector_of(l) for l in labels], labels)):
            raise ValueError('the saved vectors of a span are not independent')
