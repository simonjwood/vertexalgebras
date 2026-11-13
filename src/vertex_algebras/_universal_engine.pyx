# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of the C++ engine of universal vertex algebras
(``_universal_engine.hpp``), built when the package is installed
(``meson.build``: C++17, linked with FLINT and GMP).

Sage-free on purpose: everything crossing this boundary is Python ints,
tuples and lists; ``vertex_algebra._EngineKernel`` converts to and from Sage
rings.  The boundary types and conversions shared with other engines are in
``_engine_base.pxi``.

- A key (monomial) is the universal algebra's key: a tuple of letters
  ``(generator index, derivative)`` in canonical order.
- A coefficient is a list of terms ``(exponents, numerator, denominator)``
  (``exponents`` empty for ``kind`` 'q' and 'nmod', one entry for 'qpoly',
  ``nvars`` entries for 'qmpoly').
- An element is a list of ``(key, coefficient)``; poles are a list of
  elements (entry ``k`` = pole ``k + 1``, i.e. ``a_(k) b``).

``nthreads > 1`` releases the GIL and computes the key-level jobs of a call
in that many threads; results do not depend on ``nthreads``.  Ctrl-C is
polled (``PyErr_CheckSignals``) and raises ``KeyboardInterrupt``; the
engine's caches stay consistent.  An engine must not be used from two Python
threads at once.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"

cdef extern from "_universal_engine.hpp" namespace "vae":
    ctypedef vector[pair[int, int]] Letters
    cdef cppclass EngineBase:
        uint32_t intern(const Letters &) except +_raise_error
        Letters letters(uint32_t) except +_raise_error
        void set_ope(int, int, const PolesData &) except +_raise_error
        PolesData ope_mono(uint32_t, uint32_t, PollFn) except +_raise_error
        ElemData no_mono(uint32_t, uint32_t, PollFn) except +_raise_error
        ElemData deriv_mono(uint32_t, int, PollFn) except +_raise_error
        PolesData ope_el(const ElemData &, const ElemData &, int, PollFn) except +_raise_error nogil
        vector[PolesData] ope_el_many(const ElemData &, const vector[ElemData] &, int,
                                      PollFn) except +_raise_error nogil
        ElemData nprod_el(const ElemData &, long, const ElemData &, int,
                          PollFn) except +_raise_error nogil
        ElemData deriv_el(const ElemData &, int, int, PollFn) except +_raise_error nogil
        vector[ElemData] nprod_el_many(const vector[ElemData] &, const vector[int] &,
                                       const vector[long] &, const vector[int] &, int,
                                       PollFn) except +_raise_error nogil
        vector[size_t] cache_sizes()
    EngineBase *make_engine(string, int, vector[int], long, unsigned long,
                            const CoefData &) except +_raise_error


cdef class Engine:
    r"""
    One engine = one universal vertex algebra (generator count, parities,
    declared OPEs) over one coefficient ring: ``kind`` 'q' (QQ), 'qpoly'
    (QQ[x]), 'qmpoly' (QQ[x_1, ..., x_nvars]), 'nmod' (GF(modulus), a prime
    below 2^63), 'nf' (the number field of the defining polynomial
    ``minpoly``, a coefficient in 'qpoly' format) or 'nfmpoly' (polynomials
    in ``nvars`` variables over that field; exponent vectors: the variables,
    then the field generator).

    EXAMPLES:

    The Virasoro algebra at `c = 1/2` over `\QQ` (one even generator `L`,
    letters ``(generator, derivative)``)::

        sage: from vertex_algebras._universal_engine import Engine
        sage: E = Engine('q', 1, [0])
        sage: one = lambda n, d=1: [((), n, d)]
        sage: L = ((0, 0),)
        sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
        ....:              [((), one(1, 4))]])
        sage: E.ope_mono(L, L)
        [[(((0, 1),), [((), 1, 1)])], [(((0, 0),), [((), 2, 1)])], [],
         [((), [((), 1, 4)])]]
    """
    cdef EngineBase *e
    cdef dict _ids          # key -> monomial id
    cdef dict _keys         # monomial id -> key
    cdef readonly str kind
    cdef readonly long nvars
    cdef readonly unsigned long modulus

    def __cinit__(self, str kind, int ngens, parities, long nvars=0, unsigned long modulus=0,
                  minpoly=()):
        """
        Build an engine over the ring ``kind`` (see the class).

        TESTS::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 2, [1, 1])
            sage: E.kind, E.nvars, E.modulus
            ('q', 0, 0)
            sage: Engine('nmod', 1, [0], modulus=7).modulus
            7
            sage: Engine('nf', 1, [0], minpoly=[((0,), -5, 1), ((2,), 1, 1)]).kind
            'nf'
            sage: Engine('foo', 1, [0])
            Traceback (most recent call last):
            ...
            ValueError: ...
        """
        self.e = NULL
        self.kind = kind
        self.nvars = nvars
        self.modulus = modulus
        self._ids = {}
        self._keys = {}
        cdef vector[int] par = [int(p) for p in parities]
        cdef CoefData mp = _coef_in(minpoly)
        self.e = make_engine(kind.encode(), ngens, par, nvars, modulus, mp)

    def __dealloc__(self):
        if self.e != NULL:
            del self.e

    cdef uint32_t _id(self, key) except? 0xFFFFFFFF:
        r = self._ids.get(key)
        if r is None:
            r = self.e.intern([(int(i), int(n)) for i, n in key])
            self._ids[key] = r
            self._keys[r] = key
        return r

    cdef object _key(self, uint32_t i):
        k = self._keys.get(i)
        if k is None:
            k = tuple([(l.first, l.second) for l in self.e.letters(i)])
            self._keys[i] = k
            self._ids[k] = i
        return k

    cdef ElemData _elem_in(self, object el) except *:
        cdef ElemData d
        cdef pair[uint32_t, CoefData] t
        for key, terms in el:
            t.first = self._id(key)
            t.second = _coef_in(terms)
            d.push_back(t)
        return d

    cdef list _elem_out(self, const ElemData &d):
        cdef list out = []
        cdef size_t i
        for i in range(d.size()):
            out.append((self._key(d[i].first), _coef_out(d[i].second)))
        return out

    cdef list _poles_out(self, const PolesData &P):
        cdef list out = []
        cdef size_t i
        for i in range(P.size()):
            out.append(self._elem_out(P[i]))
        return out

    def intern(self, key):
        """
        The engine's id of a key (interned on first use).

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: i = E.intern(((0, 0),))
            sage: E.intern(((0, 0),)) == i, E.intern(()) == i
            (True, False)
        """
        return self._id(key)

    def set_ope(self, int i, int j, poles):
        r"""Declare the OPE of generators ``i``, ``j`` (the other orientation
        is derived by skew-symmetry unless declared too); before computing.

        EXAMPLES:

        The Heisenberg algebra, `J(z) J(w) \sim 1/(z-w)^2`::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: J = ((0, 0),)
            sage: E.set_ope(0, 0, [[], [((), [((), 1, 1)])]])
            sage: E.ope_mono(J, J)
            [[], [((), [((), 1, 1)])]]
        """
        cdef PolesData P
        for el in poles:
            P.push_back(self._elem_in(el))
        self.e.set_ope(i, j, P)

    def ope_mono(self, a, b):
        """
        The poles of ``OPE[a, b]`` for keys ``a``, ``b`` (trimmed).

        EXAMPLES:

        The same as the universal Virasoro algebra gives::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.ope_mono(L, L)
            [[(((0, 1),), [((), 1, 1)])], [(((0, 0),), [((), 2, 1)])], [],
             [((), [((), 1, 4)])]]
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['L'], {('L', 'L'): [
            ....:     {(('L', 1),): 1}, {(('L', 0),): 2}, 0, 1/4]})
            sage: Lv = V.gen('L'); Lv.ope(Lv)
            [dL, 2*L, 0, 1/4]
            sage: E.ope_mono(L, ())
            []
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        return self._poles_out(self.e.ope_mono(ia, ib, <PollFn>_poll))

    def no_mono(self, a, b):
        """
        ``:a b:`` for keys ``a``, ``b``.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.no_mono(L, L)
            [(((0, 0), (0, 0)), [((), 1, 1)])]
            sage: E.no_mono(((0, 1),), L)
            [(((0, 1), (0, 0)), [((), 1, 1)])]
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        return self._elem_out(self.e.no_mono(ia, ib, <PollFn>_poll))

    def deriv_mono(self, m, int n):
        """
        ``∂^n m`` for a key ``m`` (letters carry plain derivatives, not
        divided by factorials).

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.deriv_mono(L, 2)
            [(((0, 2),), [((), 1, 1)])]
            sage: E.deriv_mono((), 1)
            []
        """
        cdef uint32_t im = self._id(m)
        return self._elem_out(self.e.deriv_mono(im, n, <PollFn>_poll))

    def ope_el(self, A, B, int nthreads=1):
        """
        The poles of ``OPE[A, B]`` for elements ``A``, ``B`` (untrimmed).

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.ope_el([(L, one(2))], [(L, one(1)), ((), one(5))])
            [[(((0, 1),), [((), 2, 1)])], [(((0, 0),), [((), 4, 1)])], [],
             [((), [((), 1, 2)])]]
            sage: E.ope_el([(L, one(2))], [(L, one(1))], nthreads=2) == E.ope_el(
            ....:     [(L, one(2))], [(L, one(1))])
            True
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef PolesData r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el(a, b, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el(a, b, 1, <PollFn>_poll)
        return self._poles_out(r)

    def ope_el_many(self, A, Bs, int nthreads=1):
        """
        ``[ope_el(A, B) for B in Bs]``, all key-level jobs in one batch.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: A = [(L, one(1))]
            sage: E.ope_el_many(A, [A, [((), one(3))]]) == [E.ope_el(A, A), []]
            True
        """
        cdef ElemData a = self._elem_in(A)
        cdef vector[ElemData] bs
        for B in Bs:
            bs.push_back(self._elem_in(B))
        cdef vector[PolesData] r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el_many(a, bs, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el_many(a, bs, 1, <PollFn>_poll)
        cdef size_t i
        return [self._poles_out(r[i]) for i in range(r.size())]

    def nprod_el(self, A, long n, B, int nthreads=1):
        """
        ``A_(n) B`` for any integer ``n``.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: A = [(L, one(1))]
            sage: E.nprod_el(A, 3, A)
            [((), [((), 1, 4)])]
            sage: E.nprod_el(A, -2, A)
            [(((0, 1), (0, 0)), [((), 1, 1)])]
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef ElemData r
        if nthreads > 1:
            with nogil:
                r = self.e.nprod_el(a, n, b, nthreads, <PollFn>_poll)
        else:
            r = self.e.nprod_el(a, n, b, 1, <PollFn>_poll)
        return self._elem_out(r)

    def nprod_el_many(self, els, tasks, int nthreads=1):
        """``[els[i]_(n) els[j] for (i, n, j) in tasks]`` in one call: the
        key-level jobs of all of them first, then each sum in one of
        ``nthreads`` threads.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: A = [(L, one(1))]
            sage: E.nprod_el_many([A], [(0, 1, 0), (0, -1, 0)])
            [[(((0, 0),), [((), 2, 1)])], [(((0, 0), (0, 0)), [((), 1, 1)])]]
            sage: _ == E.nprod_el_many([A], [(0, 1, 0), (0, -1, 0)], nthreads=2)
            True
        """
        cdef vector[ElemData] E
        cdef vector[int] ia, ib
        cdef vector[long] ns
        cdef vector[ElemData] r
        cdef size_t k
        for el in els:
            E.push_back(self._elem_in(el))
        for i, n, j in tasks:
            ia.push_back(i)
            ns.push_back(n)
            ib.push_back(j)
        if nthreads > 1:
            with nogil:
                r = self.e.nprod_el_many(E, ia, ns, ib, nthreads, <PollFn>_poll)
        else:
            r = self.e.nprod_el_many(E, ia, ns, ib, 1, <PollFn>_poll)
        return [self._elem_out(r[k]) for k in range(r.size())]

    def deriv_el(self, A, int n, int nthreads=1):
        """
        ``∂^n A``.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: E.deriv_el([(L, one(2)), ((), one(1))], 1)
            [(((0, 1),), [((), 2, 1)])]
        """
        cdef ElemData a = self._elem_in(A)
        cdef ElemData r
        if nthreads > 1:
            with nogil:
                r = self.e.deriv_el(a, n, nthreads, <PollFn>_poll)
        else:
            r = self.e.deriv_el(a, n, 1, <PollFn>_poll)
        return self._elem_out(r)

    def cache_sizes(self):
        """Entries of the OPE, normal ordering, derivative and generator-OPE
        caches, the number of interned monomials, and (cumulative, threaded
        calls) jobs pushed and stolen, waits for entries other threads were
        computing and their total time in ms, workers' idle time in ms.

        EXAMPLES::

            sage: from vertex_algebras._universal_engine import Engine
            sage: E = Engine('q', 1, [0])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: L = ((0, 0),)
            sage: E.set_ope(0, 0, [[(((0, 1),), one(1))], [(L, one(2))], [],
            ....:                  [((), one(1, 4))]])
            sage: d = E.cache_sizes(); sorted(d)
            ['deriv', 'gen', 'idle_ms', 'monomials', 'no', 'ope', 'pushed',
             'stolen', 'wait_ms', 'waits']
            sage: _ = E.ope_mono(L, L); E.cache_sizes()['ope'] >= 1
            True
        """
        s = self.e.cache_sizes()
        return dict(zip(['ope', 'no', 'deriv', 'gen', 'monomials',
                         'pushed', 'stolen', 'waits', 'wait_ms', 'idle_ms'], s))
