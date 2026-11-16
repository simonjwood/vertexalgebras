# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of ``_linear_algebra_engine.hpp``: sparse reduced echelon forms
in C++ over rational function fields ``QQ(x_1, ..., x_n)`` (FLINT
``fmpz_mpoly_q``) and number fields (FLINT ``nf_elem``), built when the
package is installed (``meson.build``).  Sage-free: keys are
integer ids, coefficients engine terms ``(exponents, num, den)``:

- :class:`Echelon` (``QQ(x...)``): a coefficient is a fraction ``(numerator,
  denominator)`` of polynomials, or (``*_lifted``) one polynomial in
  ``x_1..x_n, u_1..u_m`` with ``u_j = 1/p_j`` (:meth:`Echelon.set_subst`);
- :class:`NFEchelon`: a polynomial in the generator ``t`` of the field.

Used by ``_keyed_spans._FracBlock`` and ``_EngineSpan``.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"
from libcpp cimport bool as cbool
from libc.stdint cimport int64_t, uint64_t

cdef extern from "_linear_algebra_engine.hpp" namespace "vae":
    ctypedef vector[pair[uint32_t, pair[CoefData, CoefData]]] FracData
    ctypedef vector[pair[uint32_t, CoefData]] PolyData
    cdef cppclass FracEchelon:
        FracEchelon(long, cbool) except +_raise_error
        void set_subst(const vector[CoefData] &) except +_raise_error
        void set_rank(uint32_t, int64_t)
        vector[cbool] insert_frac(const vector[FracData] &, PollFn) except +_raise_error
        vector[cbool] insert_lifted(const vector[PolyData] &, PollFn) except +_raise_error
        cbool contains_frac(const FracData &, PollFn) except +_raise_error
        cbool contains_lifted(const PolyData &, PollFn) except +_raise_error
        FracData reduce_frac(const FracData &, PollFn) except +_raise_error
        FracData coordinates_frac(const FracData &, cbool &, PollFn) except +_raise_error
        FracData coordinates_lifted(const PolyData &, cbool &, PollFn) except +_raise_error
        vector[uint32_t] pivots()
        FracData row(uint32_t) except +_raise_error
        vector[cbool] done
    cdef cppclass NFEchelon:
        NFEchelon(const CoefData &, cbool) except +_raise_error
        void set_rank(uint32_t, int64_t)
        vector[cbool] insert_many(const vector[PolyData] &, PollFn) except +_raise_error
        cbool contains(const PolyData &, PollFn) except +_raise_error
        PolyData reduce(const PolyData &, PollFn) except +_raise_error
        PolyData coordinates(const PolyData &, cbool &, PollFn) except +_raise_error
        vector[uint32_t] pivots()
        PolyData row(uint32_t) except +_raise_error
        vector[cbool] done


cdef extern from "_linear_algebra_engine.hpp" namespace "vae":
    ctypedef vector[pair[uint32_t, pair[Num, Num]]] QData
    cdef cppclass QEchelon:
        QEchelon(cbool) except +_raise_error
        void set_rank(uint32_t, int64_t)
        void set_nlabels(uint32_t)
        void load_row(uint32_t, const QData &, const QData &) except +_raise_error
        vector[cbool] insert_many(const vector[QData] &, PollFn) except +_raise_error
        double fill(int64_t)
        vector[pair[uint32_t, pair[QData, QData]]] rows() except +_raise_error
        vector[cbool] done


cdef QData _q_in(object v) except *:
    cdef QData d
    cdef pair[uint32_t, pair[Num, Num]] t
    for k, num, den in v:
        t.first = k
        _num_in(t.second.first, num)
        _num_in(t.second.second, den)
        d.push_back(t)
    return d


cdef list _q_out(const QData &d):
    cdef list out = []
    cdef size_t i
    for i in range(d.size()):
        out.append((d[i].first, _num_out(d[i].second.first), _num_out(d[i].second.second)))
    return out


cdef class QEchelon_:
    r"""
    A sparse reduced echelon form over `\QQ` (FLINT ``fmpq``): the sparse
    phase of ``_keyed_spans._Block`` over ``QQ`` (same pivots: the key of
    largest rank; combos by label index).  Vectors are lists of
    ``(key id, numerator, denominator)`` (Python integers).

    EXAMPLES::

        sage: from vertex_algebras._linear_algebra_engine import QEchelon_
        sage: E = QEchelon_(True); E.set_ranks([(0, 0), (1, 1), (2, 2)])
        sage: E.insert_many([[(0, 1, 1), (1, 2, 1)], [(0, 2, 1), (1, 4, 1)], [(1, 1, 3), (2, 1, 1)]])
        [True, False, True]
        sage: E.rows()          # e_0/2 + e_1, and v_3 - row_1/3 = -e_0/6 + e_2 (combos: accepted vectors 0, 1)
        [(1, [(0, 1, 2), (1, 1, 1)], [(0, 1, 2)]),
         (2, [(0, -1, 6), (2, 1, 1)], [(0, -1, 6), (1, 1, 1)])]
        sage: E.fill(3)
        0.7
    """
    cdef QEchelon *e

    def __cinit__(self, bint combos):
        """
        An empty echelon form.

        TESTS::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: QEchelon_(False).rows(), QEchelon_(False).fill(5)
            ([], 0.0)
        """
        self.e = new QEchelon(combos)

    def __dealloc__(self):
        del self.e

    def set_ranks(self, ranks):
        """
        ``ranks``: pairs ``(key id, rank)``; the pivot of a new row is its
        key of largest rank.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(False); E.set_ranks([(0, 1), (1, 0)])
            sage: E.insert_many([[(0, 1, 1), (1, 1, 1)]]); E.rows()[0][0]
            [True]
            0
        """
        for k, r in ranks:
            self.e.set_rank(k, r)

    def set_nlabels(self, n):
        """
        The label index of the next inserted vector (combos refer to label
        indices): the number of vectors a loaded block already labelled.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(True); E.set_nlabels(5)
            sage: E.insert_many([[(0, 2, 1)]]); E.rows()
            [True]
            [(0, [(0, 1, 1)], [(5, 1, 2)])]
        """
        self.e.set_nlabels(n)

    def load_row(self, p, row, combo):
        """
        Load an existing row ``(pivot, row, combo)`` (e.g. of a Python block).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(True); E.set_ranks([(0, 0), (1, 1)]); E.set_nlabels(1)
            sage: E.load_row(1, [(0, 1, 1), (1, 1, 1)], [(0, 1, 1)])
            sage: E.insert_many([[(1, 1, 1)], [(0, 1, 1), (1, 1, 1)]])
            [True, False]
            sage: E.load_row(1, [], [])
            Traceback (most recent call last):
            ...
            ValueError: pivot already loaded
        """
        self.e.load_row(p, _q_in(row), _q_in(combo))

    def insert_many(self, vecs):
        """
        Insert the vectors; return for each whether it was independent (an
        interrupt keeps the committed ones: :meth:`done`).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(False)
            sage: E.insert_many([[(0, 1, 2)], [(0, 3, 1)], [(1, 2**70, 3)]])
            [True, False, True]
        """
        cdef vector[QData] vs
        for v in vecs:
            vs.push_back(_q_in(v))
        return list(self.e.insert_many(vs, <PollFn>_poll))

    def done(self):
        """
        The flags of the vectors of the last batch committed so far.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(False); _ = E.insert_many([[(0, 1, 1)], [(0, 2, 1)]]); E.done()
            [True, False]
        """
        return list(self.e.done)

    def fill(self, ncols):
        """
        ``_Block._fill``: entries of the rows and combos over
        ``r (ncols + r)``.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(True); _ = E.insert_many([[(0, 1, 1), (1, 1, 1)]]); E.fill(2)
            1.0
        """
        return self.e.fill(ncols)

    def rows(self):
        """
        All rows ``(pivot, row, combo)``, pivots in insertion order.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import QEchelon_
            sage: E = QEchelon_(False); _ = E.insert_many([[(3, 2, 1), (4, -1, 5)]]); E.rows()
            [(4, [(3, -10, 1), (4, 1, 1)], [])]
        """
        cdef vector[pair[uint32_t, pair[QData, QData]]] r = self.e.rows()
        cdef size_t i
        return [(r[i].first, _q_out(r[i].second.first), _q_out(r[i].second.second))
                for i in range(r.size())]


cdef extern from "_linear_algebra_engine.hpp" namespace "vae":
    ctypedef vector[pair[int64_t, vector[pair[int64_t, uint64_t]]]] NmodKernel
    NmodKernel nmod_kernel(int64_t, int64_t, uint64_t, const int64_t *, const int64_t *,
                           const uint64_t *, size_t, int) except +_raise_error nogil


def nmod_kernel_triplets(int64_t nrows, int64_t ncols, uint64_t p, rows, cols, vals,
                         int nthreads=1):
    """
    The right kernel modulo the prime ``p`` of the ``nrows x ncols`` matrix
    with entries the sums of the triplets ``(rows[t], cols[t], vals[t])``
    (buffers of signed / unsigned 64-bit integers, e.g. ``array('q')`` /
    ``array('Q')``, ``vals`` reduced mod ``p``): one vector per non-pivot
    column ``f``, ``(f, [(pivot column, value), ...])`` (value 1 at ``f``).
    The GIL is released during the elimination (FLINT, ``nthreads``).

    EXAMPLES:

    The kernel of the row `(1, 2, 3)` modulo 7::

        sage: from array import array
        sage: from vertex_algebras._linear_algebra_engine import nmod_kernel_triplets
        sage: nmod_kernel_triplets(1, 3, 7, array('q', [0, 0, 0]), array('q', [0, 1, 2]),
        ....:                      array('Q', [1, 2, 3]))
        [(1, [(0, 5)]), (2, [(0, 4)])]
        sage: nmod_kernel_triplets(1, 3, 7, array('q', [0]), array('q', [0, 1]),
        ....:                      array('Q', [1]))
        Traceback (most recent call last):
        ...
        ValueError: rows, cols and vals differ in length
    """
    cdef const int64_t[:] r = rows
    cdef const int64_t[:] c = cols
    cdef const uint64_t[:] v = vals
    if not (r.shape[0] == c.shape[0] == v.shape[0]):
        raise ValueError('rows, cols and vals differ in length')
    cdef size_t n = r.shape[0]
    cdef NmodKernel res
    cdef const int64_t *rp = &r[0] if n else NULL
    cdef const int64_t *cp = &c[0] if n else NULL
    cdef const uint64_t *vp = &v[0] if n else NULL
    with nogil:
        res = nmod_kernel(nrows, ncols, p, rp, cp, vp, n, nthreads)
    cdef list out = []
    cdef size_t i, j
    for i in range(res.size()):
        out.append((res[i].first, [(res[i].second[j].first, res[i].second[j].second)
                                   for j in range(res[i].second.size())]))
    return out


cdef FracData _frac_in(object vec) except *:
    cdef FracData d
    cdef pair[uint32_t, pair[CoefData, CoefData]] t
    for k, num, den in vec:
        t.first = k
        t.second.first = _coef_in(num)
        t.second.second = _coef_in(den)
        d.push_back(t)
    return d


cdef list _frac_out(const FracData &d):
    cdef list out = []
    cdef size_t i
    for i in range(d.size()):
        out.append((d[i].first, _coef_out(d[i].second.first), _coef_out(d[i].second.second)))
    return out


cdef PolyData _poly_in(object vec) except *:
    cdef PolyData d
    cdef pair[uint32_t, CoefData] t
    for k, terms in vec:
        t.first = k
        t.second = _coef_in(terms)
        d.push_back(t)
    return d


cdef list _poly_out(const PolyData &d):
    cdef list out = []
    cdef size_t i
    for i in range(d.size()):
        out.append((d[i].first, _coef_out(d[i].second)))
    return out


cdef class Echelon:
    r"""
    A sparse reduced echelon form over ``QQ(x_1, ..., x_nvars)``; with
    ``combos`` the rows are also kept as combinations of the inserted
    vectors (for :meth:`coordinates`).

    EXAMPLES:

    Over `\QQ(x)`; a coefficient is a pair (numerator, denominator) of
    polynomials in engine terms::

        sage: from vertex_algebras._linear_algebra_engine import Echelon
        sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
        sage: x = [((1,), 1, 1)]
        sage: E = Echelon(1, True)
        sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
        sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                # e0 + x e1
        sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
        sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
        sage: E.insert_many([v1, v2, v3])
        [True, True, False]
        sage: E.pivots()
        [1, 2]
    """
    cdef FracEchelon *e

    def __cinit__(self, long nvars, bint combos):
        """
        An empty echelon form over ``QQ(x_1, ..., x_nvars)``.

        TESTS::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: E = Echelon(2, False); E.pivots()
            []
        """

        self.e = new FracEchelon(nvars, combos)

    def __dealloc__(self):
        del self.e

    def set_ranks(self, ranks):
        """``ranks``: pairs ``(key id, rank)``; the pivot of a new row is its
        key of largest rank.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: F = Echelon(1, False); F.set_ranks([(0, 1), (1, 0)])
            sage: F.insert_many([v1]), F.pivots()
            ([True], [0])
        """
        for k, r in ranks:
            self.e.set_rank(k, r)

    def set_subst(self, polys):
        """The lifted input: ``u_j = 1/polys[j]`` (polynomials in the
        ``nvars`` variables).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: E = Echelon(1, True)
            sage: E.set_subst([[((1,), 1, 1), ((0,), 1, 1)]])   # u = 1/(x + 1)
            sage: E.insert_many_lifted([[(0, [((0, 1), 1, 1)])],    # u e0
            ....:                       [(0, [((1, 0), 1, 1)])]])   # x e0
            [True, False]
            sage: E.pivots()
            [0]
        """
        cdef vector[CoefData] ps
        for p in polys:
            ps.push_back(_coef_in(p))
        self.e.set_subst(ps)

    def insert_many(self, vecs):
        """
        Insert the vectors ``vecs`` (lists of ``(key id, numerator,
        denominator)``); return for each whether it was independent.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.insert_many([[(0, c(1), c(1))]])
            [True]
        """
        cdef vector[FracData] vs
        for v in vecs:
            vs.push_back(_frac_in(v))
        return list(self.e.insert_frac(vs, <PollFn>_poll))

    def insert_many_lifted(self, vecs):
        """
        :meth:`insert_many` for lifted vectors (lists of ``(key id,
        polynomial in x..., u...)``, see :meth:`set_subst`).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: E = Echelon(1, True)
            sage: E.set_subst([[((1,), 1, 1), ((0,), 1, 1)]])   # u = 1/(x + 1)
            sage: E.insert_many_lifted([[(0, [((0, 1), 1, 1)])],    # u e0
            ....:                       [(0, [((1, 0), 1, 1)])]])   # x e0
            [True, False]
            sage: E.insert_many_lifted([[(1, [((0, 0), 1, 1)])]])
            [True]
        """

        cdef vector[PolyData] vs
        for v in vecs:
            vs.push_back(_poly_in(v))
        return list(self.e.insert_lifted(vs, <PollFn>_poll))

    def contains(self, vec):
        """
        Whether ``vec`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.contains(v3), E.contains([(0, c(1), c(1))])
            (True, False)
        """
        return self.e.contains_frac(_frac_in(vec), <PollFn>_poll)

    def contains_lifted(self, vec):
        """
        :meth:`contains` for a lifted vector.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: E = Echelon(1, True)
            sage: E.set_subst([[((1,), 1, 1), ((0,), 1, 1)]])   # u = 1/(x + 1)
            sage: E.insert_many_lifted([[(0, [((0, 1), 1, 1)])],    # u e0
            ....:                       [(0, [((1, 0), 1, 1)])]])   # x e0
            [True, False]
            sage: E.contains_lifted([(0, [((0, 0), 1, 1)])])
            True
            sage: E.contains_lifted([(1, [((0, 0), 1, 1)])])
            False
        """

        return self.e.contains_lifted(_poly_in(vec), <PollFn>_poll)

    def reduce(self, vec):
        """
        ``vec`` reduced by the rows (zero iff ``vec`` lies in the span).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.reduce([(1, c(1), c(1))])        # e1 = -1/x e0 modulo the span
            [(0, [((0,), -1, 1)], [((1,), 1, 1)])]
            sage: E.reduce(v3)
            []
        """
        return _frac_out(self.e.reduce_frac(_frac_in(vec), <PollFn>_poll))

    def coordinates(self, vec):
        """
        ``vec`` as a combination of the inserted vectors (by their position
        among all inserts), or ``None`` if it is not in the span.

        EXAMPLES:

        `v_3 = v_1 + v_2`::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.coordinates(v3)
            [(0, [((0,), 1, 1)], [((0,), 1, 1)]), (1, [((0,), 1, 1)], [((0,), 1, 1)])]
            sage: E.coordinates([(0, c(1), c(1))]) is None
            True
        """
        cdef cbool ok = False
        cdef FracData r = self.e.coordinates_frac(_frac_in(vec), ok, <PollFn>_poll)
        return _frac_out(r) if ok else None

    def coordinates_lifted(self, vec):
        r"""
        :meth:`coordinates` of a lifted vector.

        EXAMPLES:

        `(x + 1) e_0 = (x + 1)^2 \cdot u e_0`::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: E = Echelon(1, True)
            sage: E.set_subst([[((1,), 1, 1), ((0,), 1, 1)]])   # u = 1/(x + 1)
            sage: E.insert_many_lifted([[(0, [((0, 1), 1, 1)])],    # u e0
            ....:                       [(0, [((1, 0), 1, 1)])]])   # x e0
            [True, False]
            sage: E.coordinates_lifted([(0, [((1, 0), 1, 1), ((0, 0), 1, 1)])])
            [(0, [((2,), 1, 1), ((1,), 2, 1), ((0,), 1, 1)], [((0,), 1, 1)])]
        """

        cdef cbool ok = False
        cdef FracData r = self.e.coordinates_lifted(_poly_in(vec), ok, <PollFn>_poll)
        return _frac_out(r) if ok else None

    def pivots(self):
        """
        The pivots (key ids) of the rows, in the order of their creation.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.pivots()
            [1, 2]
        """
        return list(self.e.pivots())

    def row(self, uint32_t p):
        """
        The reduced row of pivot ``p``.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.row(2)
            [(0, [((0,), 1, 1)], [((0,), 1, 1)]), (2, [((0,), 1, 1)], [((0,), 1, 1)])]
        """
        return _frac_out(self.e.row(p))

    def done(self):
        """The flags of the last batch, also of an interrupted one (its
        inserts are atomic: those listed are committed).

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import Echelon
            sage: c = lambda n, d=1: [((0,), n, d)]     # constants in QQ[x]
            sage: x = [((1,), 1, 1)]
            sage: E = Echelon(1, True)
            sage: E.set_ranks([(0, 0), (1, 1), (2, 2)])
            sage: v1 = [(0, c(1), c(1)), (1, x, c(1))]                    # e0 + x e1
            sage: v2 = [(0, c(2), c(1)), (1, x, c(1)), (2, c(1), c(1))]   # 2 e0 + x e1 + e2
            sage: v3 = [(0, c(3), c(1)), (1, [((1,), 2, 1)], c(1)), (2, c(1), c(1))]
            sage: E.insert_many([v1, v2, v3])
            [True, True, False]
            sage: E.done()
            [True, True, False]
        """
        return list(self.e.done)


cdef class NFEchelon_:
    r"""
    A sparse reduced echelon form over the number field of the defining
    polynomial ``minpoly`` (engine terms of a polynomial in ``t``).

    EXAMPLES:

    Over `\QQ(\sqrt 5)`; the second vector is `t` times the first::

        sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
        sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
        sage: E.set_ranks([(0, 0), (1, 1)])
        sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
        sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
        [True, False]
        sage: E.row(1)
        [(0, [((1,), 1, 5)]), (1, [((0,), 1, 1)])]
    """
    cdef NFEchelon *e

    def __cinit__(self, minpoly, bint combos):
        """
        An empty echelon form over the number field of ``minpoly``.

        TESTS::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: NFEchelon_([((0,), -2, 1), ((2,), 1, 1)], False).pivots()
            []
        """

        self.e = new NFEchelon(_coef_in(minpoly), combos)

    def __dealloc__(self):
        del self.e

    def set_ranks(self, ranks):
        """
        ``ranks``: pairs ``(key id, rank)``; the pivot of a new row is its
        key of largest rank.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.pivots()
            [1]
        """
        for k, r in ranks:
            self.e.set_rank(k, r)

    def insert_many(self, vecs):
        """
        Insert the vectors ``vecs`` (lists of ``(key id, polynomial in
        t)``); return for each whether it was independent.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.insert_many([[(0, one)]])
            [True]
        """
        cdef vector[PolyData] vs
        for v in vecs:
            vs.push_back(_poly_in(v))
        return list(self.e.insert_many(vs, <PollFn>_poll))

    def contains(self, vec):
        """
        Whether ``vec`` lies in the span.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.contains([(0, [((0,), 2, 1)]), (1, [((1,), 2, 1)])])
            True
            sage: E.contains([(1, one)])
            False
        """
        return self.e.contains(_poly_in(vec), <PollFn>_poll)

    def reduce(self, vec):
        """
        ``vec`` reduced by the rows.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.reduce([(1, one)])
            [(0, [((1,), -1, 5)])]
        """
        return _poly_out(self.e.reduce(_poly_in(vec), <PollFn>_poll))

    def coordinates(self, vec):
        """
        ``vec`` as a combination of the inserted vectors, or ``None``.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.coordinates([(0, [((0,), 2, 1)]), (1, [((1,), 2, 1)])])
            [(0, [((0,), 2, 1)])]
            sage: E.coordinates([(1, one)]) is None
            True
        """
        cdef cbool ok = False
        cdef PolyData r = self.e.coordinates(_poly_in(vec), ok, <PollFn>_poll)
        return _poly_out(r) if ok else None

    def pivots(self):
        """
        The pivots (key ids) of the rows.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.pivots()
            [1]
        """
        return list(self.e.pivots())

    def row(self, uint32_t p):
        """
        The reduced row of pivot ``p``.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.row(1)
            [(0, [((1,), 1, 5)]), (1, [((0,), 1, 1)])]
        """
        return _poly_out(self.e.row(p))

    def done(self):
        """
        The flags of the last batch, also of an interrupted one.

        EXAMPLES::

            sage: from vertex_algebras._linear_algebra_engine import NFEchelon_
            sage: E = NFEchelon_([((0,), -5, 1), ((2,), 1, 1)], True)   # t^2 = 5
            sage: E.set_ranks([(0, 0), (1, 1)])
            sage: t, one = [((1,), 1, 1)], [((0,), 1, 1)]
            sage: E.insert_many([[(0, one), (1, t)], [(0, t), (1, [((0,), 5, 1)])]])
            [True, False]
            sage: E.done()
            [True, False]
        """
        return list(self.e.done)
