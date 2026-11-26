# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of the C++ engine of lowering modules (``_lowering_engine.hpp``):
`L(U) = \bar M(U) / J(U)` depth by depth from lowering data, `J(U)` the
largest submodule meeting the top `U` trivially.

Sage-free on purpose: everything crossing this boundary is Python ints,
tuples and lists.  Shared boundary types: ``_engine_base.pxi``.

- A word is a key of the universal vertex algebra: a tuple of letters
  ``(generator index, derivative)``.
- Modes are bracket modes: ``g_(n)`` maps depth ``e`` to depth
  ``e + wt g - n - 1``.
- A coefficient is a list of terms ``(exponents, numerator, denominator)``
  as for the other engines (no exponents: the fields are `\QQ` and
  `\GF{p}`).
- A matrix is a list of ``((row, column), coefficient)`` (its nonzero
  entries).

``nthreads > 1`` releases the GIL and computes in that many threads;
results do not depend on ``nthreads``.  Ctrl-C is polled and raises
``KeyboardInterrupt``; an interrupted :meth:`LoweringEngine.extend` leaves
the engine at the previous depth.  An engine must not be used from two
Python threads at once.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"

cdef extern from "_lowering_engine.hpp" namespace "vae":
    ctypedef vector[pair[int, int]] Word
    ctypedef vector[pair[int, long]] ModeWord
    ctypedef vector[pair[pair[int, int], CoefData]] Entries
    cdef cppclass LoweringEngineBase:
        uint32_t intern_word(const Word &) except +_raise_error
        void set_ope(int, int, const PolesData &) except +_raise_error
        void set_splits(uint32_t, const vector[ElemData] &) except +_raise_error
        vector[uint32_t] missing_splits() except +_raise_error
        void set_top(int, const vector[vector[vector[CoefData]]] &) except +_raise_error
        void set_window(const vector[int] &, const vector[pair[vector[long], Entries]] &) except +_raise_error
        int extend(int, PollFn) except +_raise_error nogil
        int depth()
        vector[int] dims()
        pair[ModeWord, int] basis_word(int, int) except +_raise_error
        Entries gen_matrix(int, long, int) except +_raise_error
        Entries word_matrix(uint32_t, long, int, PollFn) except +_raise_error
        vector[double] stats() except +_raise_error
    LoweringEngineBase *make_lowering_engine(string, int, vector[int], unsigned long) except +_raise_error


cdef class LoweringEngine:
    r"""
    One engine = one lowering module: the strong generators of a universal
    vertex algebra (their positive integer weights), their OPEs
    (:meth:`set_ope`), the split products of the composite words
    (:meth:`set_splits`), the top (:meth:`set_top`); then :meth:`extend`
    adds one depth at a time.  Over `\QQ` (``kind='q'``) or `\GF{p}`
    (``kind='nmod'``, ``modulus=p``).

    EXAMPLES:

    The Virasoro algebra at `c = 1/2` (``L_(n) = L_{n-1}``): the simple
    module of weight `1/16` has the character of the Ising model's `\sigma`,
    `1, 1, 1, 2, 2, 3, 4, 5, ...`::

        sage: from vertex_algebras._lowering_engine import LoweringEngine
        sage: one = lambda n, d=1: [((), n, d)]
        sage: L = ((0, 0),)
        sage: E = LoweringEngine('q', [2])
        sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
        ....:                  [((), one(1, 4))]])
        sage: E.missing_splits()
        []
        sage: E.set_top(1, [[[one(1, 16)]]])
        sage: for d in range(7):
        ....:     _ = E.extend()
        sage: E.dims()
        [1, 1, 1, 2, 2, 3, 4, 5]

    TESTS:

    An empty depth (the vacuum module, ``L_{-1}`` kills the top): the Ising
    model's identity, `1, 0, 1, 1, 2, 2, 3, 3, 5, ...`::

        sage: E = LoweringEngine('q', [2])
        sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
        ....:                  [((), one(1, 4))]])
        sage: E.set_top(1, [[[[]]]])
        sage: for d in range(10):
        ....:     _ = E.extend()
        sage: E.dims()
        [1, 0, 1, 1, 2, 2, 3, 3, 5, 5, 7]
    """
    cdef LoweringEngineBase *e
    cdef dict _wids          # word -> id
    cdef list _words         # id -> word
    cdef readonly str kind
    cdef readonly int ngens

    def __cinit__(self, str kind, weights, unsigned long modulus=0):
        """
        TESTS::

            sage: from vertex_algebras._lowering_engine import LoweringEngine
            sage: LoweringEngine('nmod', [2, 3], 7).kind
            'nmod'
            sage: LoweringEngine('foo', [2])
            Traceback (most recent call last):
            ...
            ValueError: unknown field kind foo
            sage: LoweringEngine('q', [0])
            Traceback (most recent call last):
            ...
            ValueError: generator weights must be positive integers
        """
        self.e = NULL
        self.kind = kind
        self._wids = {}
        self._words = []
        cdef vector[int] w = [int(x) for x in weights]
        self.ngens = len(w)
        self.e = make_lowering_engine(kind.encode(), self.ngens, w, modulus)

    def __dealloc__(self):
        if self.e != NULL:
            del self.e

    def word_id(self, word):
        """
        The id of a word (interned with its suffixes).

        EXAMPLES::

            sage: from vertex_algebras._lowering_engine import LoweringEngine
            sage: E = LoweringEngine('q', [2])
            sage: E.word_id(((0, 0), (0, 1))) == E.word_id(((0, 0), (0, 1)))
            True
            sage: E.word(E.word_id(((0, 1),)))
            ((0, 1),)
        """
        r = self._wids.get(word)
        if r is None:
            r = self.e.intern_word([(int(i), int(k)) for i, k in word])
            self._wids[word] = r
            while len(self._words) <= r:
                self._words.append(None)
            self._words[r] = word
        return r

    def word(self, uint32_t i):
        """
        The word of an id given by :meth:`word_id`.
        """
        return self._words[i]

    cdef ElemData _el_in(self, object el) except *:
        cdef ElemData d
        cdef pair[uint32_t, CoefData] t
        for word, terms in el:
            t.first = self.word_id(word)
            t.second = _coef_in(terms)
            d.push_back(t)
        return d

    def set_ope(self, int i, int j, poles):
        """
        ``poles[k]`` = the element ``g_i(k) g_j`` as a list of ``(word,
        coefficient)``.
        """
        cdef PolesData P
        for el in poles:
            P.push_back(self._el_in(el))
        self.e.set_ope(i, j, P)

    def missing_splits(self):
        """
        The composite words reachable from the OPEs whose split products
        (:meth:`set_splits`) are still missing.
        """
        return [self._words[i] for i in self.e.missing_splits()]

    def set_splits(self, word, products):
        """
        For a composite word ``w = :x B:`` (``x`` its first letter, of
        weight ``h``): ``products[i] = x_(i) B`` for ``i < h``.
        """
        cdef vector[ElemData] P
        for el in products:
            P.push_back(self._el_in(el))
        self.e.set_splits(self.word_id(word), P)

    def set_top(self, int dim, zero_modes):
        """
        The top: ``zero_modes[g][i][j]`` = entry ``(i, j)`` of the zero mode
        of generator ``g`` (columns: images of the basis vectors).
        """
        cdef vector[vector[vector[CoefData]]] Z
        cdef vector[vector[CoefData]] M
        cdef vector[CoefData] row
        for mat in zero_modes:
            M.clear()
            for r in mat:
                row.clear()
                for c in r:
                    row.push_back(_coef_in(c))
                M.push_back(row)
            Z.push_back(M)
        self.e.set_top(dim, Z)

    def set_window(self, dims, modes):
        r"""
        A window instead of a top: depths ``0, ..., n0`` of dimensions
        ``dims`` with every generator mode between them, ``modes[(g, n, e)]``
        = the nonzero entries of ``g_(n)`` on depth ``e`` (as returned by
        :meth:`gen_matrix`; modes not given are zero).  :meth:`extend` then
        continues from depth ``n0 + 1``: the module is generated by the
        window modulo the largest submodule meeting it trivially.  Basis
        vectors of the window are numbered across its depths.

        EXAMPLES:

        The window of depths 0, 1 of the Ising model's `\sigma` (taken from
        an engine with its top) continues as that module::

            sage: from vertex_algebras._lowering_engine import LoweringEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: def vir():
            ....:     E = LoweringEngine('q', [2])
            ....:     E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                      [((), one(1, 4))]])
            ....:     return E
            sage: E = vir(); E.set_top(1, [[[one(1, 16)]]]); _ = E.extend()
            sage: modes = {(0, n, e): E.gen_matrix(0, n, e) for e in range(2)
            ....:          for n in range(e, e + 2)}
            sage: F = vir(); F.set_window(E.dims(), modes); F.depth()
            1
            sage: for d in range(6):
            ....:     _ = E.extend(); _ = F.extend()
            sage: F.dims() == E.dims() == [1, 1, 1, 2, 2, 3, 4, 5]
            True
            sage: [F.basis_word(3, i) for i in range(2)]
            [(((0, -3),), 0), (((0, -2),), 1)]

        TESTS::

            sage: F = vir(); F.set_window([0, 1], {})
            Traceback (most recent call last):
            ...
            ValueError: the lowest depth of the window must not be zero
            sage: F.set_window([1, 1], {(0, 3, 1): []})
            Traceback (most recent call last):
            ...
            RuntimeError: a mode into a depth outside the window
        """
        cdef vector[int] d = [int(x) for x in dims]
        cdef vector[pair[vector[long], Entries]] M
        cdef pair[vector[long], Entries] km
        cdef pair[pair[int, int], CoefData] t
        for (g, n, e), ents in modes.items():
            km.first = [int(g), int(n), int(e)]
            km.second.clear()
            for (i, j), c in ents:
                t.first = pair[int, int](int(i), int(j))
                t.second = _coef_in(c)
                km.second.push_back(t)
            M.push_back(km)
        self.e.set_window(d, M)

    def extend(self, int nthreads=1):
        """
        Add one depth; return its dimension.
        """
        cdef int r
        cdef int n = nthreads
        if n > 1:
            with nogil:
                r = self.e.extend(n, _poll)
        else:
            r = self.e.extend(n, _poll)
        return r

    def depth(self):
        return self.e.depth()

    def dims(self):
        return list(self.e.dims())

    def basis_word(self, int d, int i):
        """
        ``(modes, u)``: basis vector ``i`` of depth ``d`` is
        ``g_{m_1} ... g_{m_k} u`` (physics modes, the first outermost), ``u``
        a basis vector of the top (of the window: numbered across its
        depths).
        """
        cdef pair[ModeWord, int] k = self.e.basis_word(d, i)
        return (tuple([(k.first[j].first, k.first[j].second) for j in range(k.first.size())]),
                k.second)

    cdef list _entries_out(self, const Entries &x):
        cdef size_t i
        return [((x[i].first.first, x[i].first.second), _coef_out(x[i].second))
                for i in range(x.size())]

    def gen_matrix(self, int g, long n, int e):
        """
        The nonzero entries of the generator mode ``g_(n)`` on depth ``e``.
        """
        return self._entries_out(self.e.gen_matrix(g, n, e))

    def word_matrix(self, word, long n, int e):
        """
        The nonzero entries of the mode ``(n)`` of a word on depth ``e``.
        """
        return self._entries_out(self.e.word_matrix(self.word_id(word), n, e, _poll))

    def stats(self):
        """
        A dict: memo entries, nonzero memo matrices, their stored entries,
        approximate bytes, cache misses; the last extension's seconds
        (phase A, elimination, phase B), its candidates and the memo
        entries computed ahead of its phases (in parallel, by word weight).
        """
        s = list(self.e.stats())
        return dict(zip(['memo', 'matrices', 'entries', 'bytes', 'misses', 'phase A',
                         'elimination', 'phase B', 'candidates', 'prefetched'], s))
