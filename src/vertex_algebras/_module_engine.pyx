# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of the C++ engine of induced modules (``_module_engine.hpp``),
built when the package is installed (``meson.build``: C++17, linked with
FLINT and GMP).

Sage-free on purpose: everything crossing this boundary is Python ints,
tuples and lists; ``vertex_algebra_module.InducedModule`` converts to and
from Sage rings (with the conversions of the vertex algebra's
``vertex_algebra._EngineKernel``).  Shared boundary types: ``_engine_base.pxi``.

- A word is a key of the universal vertex algebra: a tuple of letters
  ``(generator index, derivative)`` in canonical order.
- A module key is identified by an int id: :meth:`ModuleEngine.intern_key`
  takes ``(modes, u)``, ``modes`` a tuple of ``(generator index, D m)``
  (physics modes ``m < 0`` in the mode order, scaled by the common
  denominator ``D`` of the weights), ``u`` the index of a top vector;
  :meth:`ModuleEngine.key_of` gives them back.
- A coefficient is a list of terms ``(exponents, numerator, denominator)``
  as for the universal engine.

``nthreads > 1`` releases the GIL and computes the tasks of a call in that
many threads; results do not depend on ``nthreads``.  Ctrl-C is polled and
raises ``KeyboardInterrupt``; the caches stay consistent.  An engine must not
be used from two Python threads at once.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"

cdef extern from "_module_engine.hpp" namespace "vae":
    ctypedef vector[pair[int, int]] WordLetters
    ctypedef vector[pair[int, long]] ModeList
    cdef cppclass ModuleEngineBase:
        uint32_t intern_word(const WordLetters &) except +_raise_error
        WordLetters word_letters(uint32_t) except +_raise_error
        uint32_t intern_key(const ModeList &, int) except +_raise_error
        pair[ModeList, int] key_of(uint32_t) except +_raise_error
        void set_top(int, const vector[int] &) except +_raise_error
        void set_zero_mode(int, const vector[vector[pair[int, CoefData]]] &) except +_raise_error
        void set_top_degrees(const vector[long] &) except +_raise_error
        void set_top_action(int, long, int, const ElemData &) except +_raise_error
        void set_free(long) except +_raise_error
        void set_ope(int, int, const PolesData &) except +_raise_error
        void set_splits(uint32_t, const vector[ElemData] &) except +_raise_error
        vector[ElemData] act_many(const vector[ElemData] &, const vector[ElemData] &,
                                  const vector[int] &, const vector[long] &, const vector[int] &,
                                  int, PollFn) except +_raise_error nogil
        vector[size_t] cache_sizes()
        vector[size_t] word_stats()
        vector[size_t] cache_profile(int)
        vector[size_t] coefficient_stats(int)
    ModuleEngineBase *make_module_engine(string, int, vector[int], vector[long], long, long,
                                         unsigned long, const CoefData &) except +_raise_error


cdef class ModuleEngine:
    r"""
    One engine = one induced module: the generators of the universal vertex
    algebra (count, parities, weights times ``D``), their OPEs
    (:meth:`set_ope`, all ordered pairs), the top level (:meth:`set_top`,
    :meth:`set_zero_mode`; or a lowered top, :meth:`set_top_degrees`,
    :meth:`set_top_action`), over one coefficient ring (``kind``, ``nvars``,
    ``modulus``, ``minpoly`` as for the universal engine).

    EXAMPLES:

    The Verma module of the Virasoro algebra at `c = 1/2`, `h = 1/16` over
    `\QQ` (``D = 1``; ``L_(n) = L_{n-1}``)::

        sage: from vertex_algebras._module_engine import ModuleEngine
        sage: one = lambda n, d=1: [((), n, d)]
        sage: L = ((0, 0),)
        sage: E = ModuleEngine('q', 1, [0], [2], 1)
        sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
        ....:              [((), one(1, 4))]])
        sage: E.set_top(1, [0])
        sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
        sage: v = E.intern_key((), 0)
        sage: r = E.act_many([[(L, one(1))]], [[(v, one(1))]], [(0, 1, 0), (0, 0, 0)])
        sage: [[(E.key_of(i), c) for i, c in x] for x in r]
        [[(((), 0), [((), 1, 16)])], [((((0, -1),), 0), [((), 1, 1)])]]
    """
    cdef ModuleEngineBase *e
    cdef dict _wids          # word -> id
    cdef readonly str kind

    def __cinit__(self, str kind, int ngens, parities, weights2, long D, long nvars=0,
                  unsigned long modulus=0, minpoly=()):
        """
        Build an engine for ``ngens`` generators of the given parities and
        weights ``weights2`` (times ``D``) over the ring ``kind``.

        TESTS::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: ModuleEngine('q', 1, [0], [2], 1).kind
            'q'
            sage: ModuleEngine('nmod', 2, [0, 1], [4, 3], 2, modulus=7).kind
            'nmod'
            sage: ModuleEngine('foo', 1, [0], [2], 1)
            Traceback (most recent call last):
            ...
            ValueError: ...
        """
        self.e = NULL
        self.kind = kind
        self._wids = {}
        cdef vector[int] par = [int(p) for p in parities]
        cdef vector[long] w2 = [int(w) for w in weights2]
        cdef CoefData mp = _coef_in(minpoly)
        self.e = make_module_engine(kind.encode(), ngens, par, w2, D, nvars, modulus, mp)

    def __dealloc__(self):
        if self.e != NULL:
            del self.e

    cdef uint32_t _wid(self, word) except? 0xFFFFFFFF:
        r = self._wids.get(word)
        if r is None:
            r = self.e.intern_word([(int(i), int(n)) for i, n in word])
            self._wids[word] = r
        return r

    cdef ElemData _words_in(self, object el) except *:
        cdef ElemData d
        cdef pair[uint32_t, CoefData] t
        for word, terms in el:
            t.first = self._wid(word)
            t.second = _coef_in(terms)
            d.push_back(t)
        return d

    cdef ElemData _ids_in(self, object el) except *:
        cdef ElemData d
        cdef pair[uint32_t, CoefData] t
        for i, terms in el:
            t.first = i
            t.second = _coef_in(terms)
            d.push_back(t)
        return d

    cdef list _ids_out(self, const ElemData &d):
        cdef list out = []
        cdef size_t i
        for i in range(d.size()):
            out.append((d[i].first, _coef_out(d[i].second)))
        return out

    def intern_key(self, modes, int u):
        """
        The id of the key ``(modes, u)`` (``modes``: ``(i, D m)`` pairs).

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: w = E.intern_key(((0, -2), (0, -1)), 0)
            sage: E.intern_key(((0, -2), (0, -1)), 0) == w != v
            True
        """
        cdef ModeList ms = [(int(i), int(m)) for i, m in modes]
        return self.e.intern_key(ms, u)

    def key_of(self, uint32_t i):
        """
        ``(modes, u)`` of a key id.

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: E.key_of(E.intern_key(((0, -2), (0, -1)), 0))
            (((0, -2), (0, -1)), 0)
            sage: E.key_of(v)
            ((), 0)
        """
        cdef pair[ModeList, int] k = self.e.key_of(i)
        cdef size_t j
        return (tuple([(k.first[j].first, k.first[j].second) for j in range(k.first.size())]),
                k.second)

    def set_top(self, int dim, parities):
        """
        The top level: its dimension and the parities of its basis.

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(2, [0, 0])
            sage: E.key_of(E.intern_key((), 1))
            ((), 1)
        """
        self.e.set_top(dim, [int(p) for p in parities])

    def set_zero_mode(self, int i, columns):
        """The zero mode of generator ``i`` on the top: ``columns[u]`` is a
        list of ``(s, coefficient)`` (the image of ``u``).

        EXAMPLES:

        `L_0 = h` on the top::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: E.act_many([[(L, one(1))]], [[(v, one(3))]], [(0, 1, 0)])
            [[(0, [((), 3, 16)])]]
        """
        cdef vector[vector[pair[int, CoefData]]] cols
        cdef vector[pair[int, CoefData]] col
        cdef pair[int, CoefData] t
        for c in columns:
            col.clear()
            for s, terms in c:
                t.first = s
                t.second = _coef_in(terms)
                col.push_back(t)
            cols.push_back(col)
        self.e.set_zero_mode(i, cols)

    def set_top_degrees(self, degrees):
        """A lowered top (induction from degree ``n``): the degrees of the
        top vectors times ``D``; then the modes ``m >= 0`` on them come from
        :meth:`set_top_action` (absent: zero), not from zero modes.

        EXAMPLES:

        Two top vectors, ``u0`` of degree 1 and ``u1`` of degree 0, with
        `L_1 u_0 = u_1` (`L_0` not given: zero)::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(2, [0, 0])
            sage: E.set_top_degrees([1, 0])
            sage: u0, u1 = E.intern_key((), 0), E.intern_key((), 1)
            sage: E.set_top_action(0, 1, 0, [(u1, one(1))])
            sage: E.act_many([[(L, one(1))]], [[(u0, one(1))]], [(0, 2, 0), (0, 1, 0)])
            [[(1, [((), 1, 1)])], []]
        """
        self.e.set_top_degrees([int(d) for d in degrees])

    def set_top_action(self, int i, long m2, int u, value):
        """The physics mode ``g_{i, m}`` (``m2 = D m >= 0``) on the top vector
        ``u`` of a lowered top: a list of ``(key id, coefficient)``.

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(2, [0, 0])
            sage: E.set_top_degrees([1, 0])
            sage: u0, u1 = E.intern_key((), 0), E.intern_key((), 1)
            sage: E.set_top_action(0, 1, 0, [(u1, one(1))])
            sage: E.act_many([[(L, one(1))]], [[(u0, one(2))]], [(0, 2, 0)])
            [[(1, [((), 2, 1)])]]
        """
        self.e.set_top_action(i, m2, u, self._ids_in(value))

    def set_free(self, long n2):
        """
        Make this the free module ``P`` of the higher Zhu algebra of level
        ``n = n2 / D`` (He's model, ``higher_zhu_algebra._FreeModeModule``):
        one top vector ``v`` of degree ``n``, every mode kept in the words,
        words of negative degree zero, all-lowering words lowering by more
        than ``n`` zero.  After ``set_top(1, [0])``, before computing.

        EXAMPLES:

        The Virasoro algebra at level 1: ``L_1 L_{-1} v = L_{-1} L_1 v + 2 L_0 v``,
        and ``L_2 v = 0`` (it lowers by 2)::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0]); E.set_free(1)
            sage: x = E.intern_key([(0, -1)], 0)
            sage: r = E.act_many([[(L, one(1))]], [[(x, one(1))]], [(0, 2, 0)])     # L_1 = L_(2)
            sage: sorted((E.key_of(k), c) for k, c in r[0])
            [((((0, -1), (0, 1)), 0), [((), 1, 1)]), ((((0, 0),), 0), [((), 2, 1)])]
            sage: E.act_many([[(L, one(1))]], [[(E.intern_key((), 0), one(1))]], [(0, 3, 0)])
            [[]]
        """
        self.e.set_free(n2)

    def set_ope(self, int i, int j, poles):
        """The OPE of generators ``i``, ``j``: a list of poles, each a list
        of ``(word, coefficient)``; before computing.

        EXAMPLES:

        The central term: `L_2 L_{-2} v = (4h + c/2) v`::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: w = E.intern_key(((0, -2),), 0)
            sage: E.act_many([[(L, one(1))]], [[(w, one(1))]], [(0, 3, 0)])
            [[(0, [((), 1, 2)])]]
        """
        cdef PolesData P
        for el in poles:
            P.push_back(self._words_in(el))
        self.e.set_ope(i, j, P)

    def set_splits(self, word, products):
        """The products ``x_(i) B`` (lists of ``(word, coefficient)``) for
        ``i = 0, ..., r - 1`` of a word ``:x B:`` of at least two letters,
        ``r`` the number of raising modes ``x_(j)`` (``wt x - j - 1 > 0``):
        then they act after ``B`` (``_module_engine.hpp``,
        ``word_act_compute``).  Between calls of :meth:`act_many`.

        EXAMPLES:

        ``:L L:_(n) L_{-2} v`` with and without the product ``L_(0) L =
        dL`` (``c = 0``)::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),); LL = ((0, 0), (0, 0))
            sage: def engine():
            ....:     E = ModuleEngine('q', 1, [0], [2], 1)
            ....:     E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [], []])
            ....:     E.set_top(1, [0])
            ....:     E.set_zero_mode(0, [[(0, one(1, 16))]])
            ....:     return E
            sage: E1, E2 = engine(), engine()
            sage: E2.set_splits(LL, [[(((0, 1),), one(1))]])
            sage: w1, w2 = E1.intern_key(((0, -2),), 0), E2.intern_key(((0, -2),), 0)
            sage: T = [(0, n, 0) for n in range(-2, 5)]
            sage: r1 = E1.act_many([[(LL, one(1))]], [[(w1, one(1))]], T)
            sage: r2 = E2.act_many([[(LL, one(1))]], [[(w2, one(1))]], T)
            sage: [sorted((E1.key_of(k), c) for k, c in el) for el in r1] == [sorted((E2.key_of(k), c) for k, c in el) for el in r2]
            True
        """
        cdef vector[ElemData] P
        for el in products:
            P.push_back(self._words_in(el))
        self.e.set_splits(self._wid(word), P)

    def act_many(self, els, vecs, tasks, int nthreads=1):
        """``[els[e]_(n) vecs[v] for (e, n, v) in tasks]``: ``els`` lists of
        ``(word, coefficient)``, ``vecs`` and the results lists of ``(key id,
        coefficient)``; each task in one of ``nthreads`` threads.

        EXAMPLES:

        `L_1 L_{-1} v = 2h v`, and a composite field `:L L:`::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: w = E.intern_key(((0, -1),), 0)
            sage: E.act_many([[(L, one(1))]], [[(w, one(1))]], [(0, 2, 0)])
            [[(0, [((), 1, 8)])]]
            sage: tasks = [(0, 1, 0), (1, 3, 0), (1, 3, 1)]
            sage: els = [[(L, one(1))], [(((0, 0), (0, 0)), one(1))]]
            sage: E.act_many(els, [[(v, one(1))], [(w, one(1))]], tasks) == E.act_many(
            ....:     els, [[(v, one(1))], [(w, one(1))]], tasks, nthreads=2)
            True
        """
        cdef vector[ElemData] E, Vs
        cdef vector[int] ie, iv
        cdef vector[long] ns
        cdef vector[ElemData] r
        cdef size_t k
        for el in els:
            E.push_back(self._words_in(el))
        for v in vecs:
            Vs.push_back(self._ids_in(v))
        for e, n, v in tasks:
            ie.push_back(e)
            ns.push_back(n)
            iv.push_back(v)
        if nthreads > 1:
            with nogil:
                r = self.e.act_many(E, Vs, ie, ns, iv, nthreads, <PollFn>_poll)
        else:
            r = self.e.act_many(E, Vs, ie, ns, iv, 1, <PollFn>_poll)
        return [self._ids_out(r[k]) for k in range(r.size())]

    def cache_sizes(self):
        """Entries of the generator-mode and word-mode caches, the numbers of
        interned keys, words and (word, n) pairs, the terms held by the two
        caches, and the distinct values the caches store their rational
        coefficients as (each once, packed: ``'values'``, ``'value bytes'``;
        0 unless over `\QQ`, or with ``VERTEX_ALGEBRAS_INTERN=0`` when the
        engine was made).

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: _ = E.act_many([[(L, one(1))]], [[(v, one(1))]], [(0, 0, 0)])
            sage: d = E.cache_sizes(); sorted(d)
            ['generator mode terms', 'generator modes', 'keys', 'value bytes', 'values',
             'word mode terms', 'word modes', 'word modes ids', 'words']
            sage: 0 < d['values'] <= d['generator mode terms'] + d['word mode terms']
            True
            sage: d['keys'] >= 2
            True
        """
        s = self.e.cache_sizes()
        return dict(zip(['generator modes', 'word modes', 'keys', 'words', 'word modes ids',
                         'generator mode terms', 'word mode terms', 'values', 'value bytes'], s))

    def word_stats(self):
        """By word length (diagnostics): lookups of word modes, misses
        (computed), terms stored; only lengths that occur.

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key((), 0)
            sage: E.word_stats()
            {}
            sage: _ = E.act_many([[(L, one(1))]], [[(v, one(1))]], [(0, 0, 0)])
            sage: list(E.word_stats())
            [1]
        """
        s = self.e.word_stats()
        return {L: (s[3 * L], s[3 * L + 1], s[3 * L + 2]) for L in range(len(s) // 3) if s[3 * L]}

    def cache_profile(self, cache='word modes'):
        """When the cached lists of ``cache`` (``'word modes'`` or
        ``'generator modes'``) are looked up again (diagnostics, for memory
        offloading): ``None`` unless the environment variable
        ``VERTEX_ALGEBRAS_CACHE_PROFILE=1`` was set when the engine was made.
        The clock ticks once per lookup; a dictionary with ``'clock'`` (the
        lookups so far), ``'reuse'`` (the intervals between two lookups of a
        list, by bit length ``b``: ``2^(b-1) <= interval < 2^b``, as pairs
        (lookups, terms of the list looked up)) and ``'age'`` (the time
        since the last lookup of each stored list, now, by bit length, as
        pairs (lists, terms)).  Only bit lengths that occur.

        EXAMPLES::

            sage: import os
            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: def engine():
            ....:     E = ModuleEngine('q', 1, [0], [2], 1)
            ....:     E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                      [((), one(1, 4))]])
            ....:     E.set_top(1, [0])
            ....:     E.set_zero_mode(0, [[(0, one(1, 16))]])
            ....:     return E
            sage: engine().cache_profile() is None
            True
            sage: os.environ['VERTEX_ALGEBRAS_CACHE_PROFILE'] = '1'
            sage: E = engine()
            sage: del os.environ['VERTEX_ALGEBRAS_CACHE_PROFILE']
            sage: v = E.intern_key(((0, -2), (0, -2)), 0)        # L_{-2} L_{-2} u
            sage: _ = E.act_many([[(L, one(1))]], [[(v, one(1))]], [(0, 2, 0)])
            sage: p = E.cache_profile('generator modes'); sorted(p)
            ['age', 'clock', 'reuse']
            sage: p['clock'] > 0
            True
            sage: sum(n for n, t in p['age'].values()) == E.cache_sizes()['generator modes']
            True
        """
        cdef int which = {'generator modes': 0, 'word modes': 1}[cache]
        s = self.e.cache_profile(which)
        if not s.size():
            return None
        def hist(i):
            return {b: (s[i + b], s[i + 65 + b]) for b in range(65) if s[i + b]}
        return {'clock': s[0], 'reuse': hist(1), 'age': hist(131)}

    def coefficient_stats(self, cache='word modes'):
        """The sizes of the rational coefficients stored in ``cache``
        (``'word modes'`` or ``'generator modes'``; diagnostics, for a more
        compact storage): ``None`` unless the coefficients are rational
        (kind ``'q'``).  A dictionary: ``'terms'``, ``'lists'``,
        ``'big numerators'`` and ``'big denominators'`` (stored as GMP
        integers, beyond FLINT's inline 62 bits), ``'limbs'`` (the 64-bit
        words those allocate), ``'denominator 1'``, ``'distinct values'`` (by
        a 64-bit hash), ``'bits'`` (of all numerators and denominators),
        ``'bits, common denominators'`` (with one denominator per list, their
        lcm, and the numerators scaled to it), ``'lists with one
        denominator'``, and the histograms ``'numerator bits'``,
        ``'denominator bits'`` (bit length: terms; 1023 includes longer).
        Not while :meth:`act_many` runs.

        EXAMPLES::

            sage: from vertex_algebras._module_engine import ModuleEngine
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E = ModuleEngine('q', 1, [0], [2], 1)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:              [((), one(1, 4))]])
            sage: E.set_top(1, [0])
            sage: E.set_zero_mode(0, [[(0, one(1, 16))]])
            sage: v = E.intern_key(((0, -2), (0, -2)), 0)
            sage: _ = E.act_many([[(L, one(1))]], [[(v, one(1))]], [(0, 2, 0)])
            sage: s = E.coefficient_stats('generator modes')
            sage: s['terms'] == E.cache_sizes()['generator mode terms']
            True
            sage: s['terms'] == sum(s['numerator bits'].values()) == sum(s['denominator bits'].values())
            True
            sage: s['big numerators'], s['big denominators']
            (0, 0)
            sage: E2 = ModuleEngine('nmod', 1, [0], [2], 1, modulus=101)
            sage: E2.coefficient_stats() is None
            True
        """
        cdef int which = {'generator modes': 0, 'word modes': 1}[cache]
        s = self.e.coefficient_stats(which)
        if not s.size():
            return None
        names = ['terms', 'lists', 'big numerators', 'big denominators', 'limbs', 'denominator 1',
                 'distinct values', 'bits', 'bits, common denominators', 'lists with one denominator']
        out = dict(zip(names, s[:10]))
        out['numerator bits'] = {b: s[10 + b] for b in range(1024) if s[10 + b]}
        out['denominator bits'] = {b: s[1034 + b] for b in range(1024) if s[1034 + b]}
        return out
