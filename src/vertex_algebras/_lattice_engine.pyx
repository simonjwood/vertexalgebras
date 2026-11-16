# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of the C++ engine of lattice vertex algebras
(``_lattice_engine.hpp``), built when the package is installed
(``meson.build``: C++17, linked with FLINT and GMP).

Sage-free on purpose: everything crossing this boundary is Python ints,
tuples and lists; ``lattice_vertex_algebra._LatticeEngineKernel`` (a
``vertex_algebra._EngineKernel``) converts to and from Sage rings and
supplies the cocycle values.  Boundary types and conversions shared with the
other engines: ``_engine_base.pxi``.

- A key is the lattice algebra's key ``(letters, q)``: letters
  ``(index, derivative)`` in canonical order, ``q`` the lattice vector.
- A coefficient is a list of terms ``(exponents, numerator, denominator)``.
- An element is a list of ``(key, coefficient)``; poles are a list of
  elements (entry ``k`` = ``a_(k) b``).
- ``eps``: a list of ``(q, p, coefficient)``, the cocycle values ``ε(q, p)``
  for every pair of lattice vectors of the call.

``nthreads > 1`` releases the GIL; results do not depend on ``nthreads``.
Ctrl-C is polled and raises ``KeyboardInterrupt``.  An engine must not be
used from two Python threads at once.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"

cdef extern from "_lattice_engine.hpp" namespace "vae":
    ctypedef vector[pair[int, int]] Letters
    cdef cppclass EpsData:
        vector[vector[long]] qs
        vector[vector[long]] ps
        vector[CoefData] vals
    cdef cppclass LatticeEngineBase:
        uint32_t intern(const Letters &, const vector[long] &) except +_raise_error
        Letters key_letters(uint32_t) except +_raise_error
        vector[long] key_q(uint32_t) except +_raise_error
        PolesData ope_mono(uint32_t, uint32_t, const EpsData &, PollFn) except +_raise_error
        ElemData no_mono(uint32_t, uint32_t, const EpsData &, PollFn) except +_raise_error
        ElemData deriv_mono(uint32_t, int, PollFn) except +_raise_error
        PolesData ope_el(const ElemData &, const ElemData &, const EpsData &, int,
                         PollFn) except +_raise_error nogil
        vector[PolesData] ope_el_many(const ElemData &, const vector[ElemData] &,
                                      const EpsData &, int, PollFn) except +_raise_error nogil
        ElemData nprod_el(const ElemData &, long, const ElemData &, const EpsData &, int,
                          PollFn) except +_raise_error nogil
        ElemData deriv_el(const ElemData &, int, int, PollFn) except +_raise_error nogil
        vector[ElemData] nprod_el_many(const vector[ElemData] &, const vector[int] &,
                                       const vector[long] &, const vector[int] &, const EpsData &,
                                       int, PollFn) except +_raise_error nogil
        vector[size_t] cache_sizes()
        void set_offset(const vector[long] &) except +_raise_error
    LatticeEngineBase *make_lattice_engine(string, vector[vector[long]], long,
                                           unsigned long, const CoefData &) except +_raise_error


cdef class Engine:
    r"""
    One engine = one lattice vertex algebra (Gram matrix) over one
    coefficient ring: ``kind`` 'q' (QQ), 'qpoly' (QQ[x]), 'qmpoly'
    (QQ[x_1, ..., x_nvars]), 'nmod' (GF(modulus), a prime below 2^63), 'nf'
    (the number field of the defining polynomial ``minpoly``, a coefficient
    in 'qpoly' format) or 'nfmpoly' (polynomials in ``nvars`` variables over
    that field; exponent vectors: the variables, then the field generator).

    EXAMPLES:

    The lattice `\sqrt 2 \ZZ` (`A_1`) over `\QQ`, trivial cocycle values;
    keys ``(letters, q)``::

        sage: from vertex_algebras._lattice_engine import Engine
        sage: E = Engine('q', [[2]])
        sage: one = lambda n, d=1: [((), n, d)]
        sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
        sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
        sage: E.ope_mono(ep, em, eps)
        [[((((0, 0),), (0,)), [((), 1, 1)])], [(((), (0,)), [((), 1, 1)])]]
    """
    cdef LatticeEngineBase *e
    cdef dict _ids          # key -> engine id
    cdef dict _keys         # engine id -> key
    cdef readonly str kind

    def __cinit__(self, str kind, gram, long nvars=0, unsigned long modulus=0, minpoly=()):
        """
        Build an engine for the Gram matrix ``gram`` over the ring ``kind``
        (see the class).

        TESTS::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: Engine('q', [[1, 0], [0, -1]]).kind
            'q'
            sage: Engine('nf', [[2]], minpoly=[((0,), -5, 1), ((2,), 1, 1)]).kind
            'nf'
            sage: Engine('foo', [[2]])
            Traceback (most recent call last):
            ...
            ValueError: ...
        """
        self.e = NULL
        self.kind = kind
        self._ids = {}
        self._keys = {}
        cdef vector[vector[long]] g = [[int(x) for x in row] for row in gram]
        cdef CoefData mp = _coef_in(minpoly)
        self.e = make_lattice_engine(kind.encode(), g, nvars, modulus, mp)

    def __dealloc__(self):
        if self.e != NULL:
            del self.e

    cdef uint32_t _id(self, key) except? 0xFFFFFFFF:
        r = self._ids.get(key)
        if r is None:
            letters, q = key
            r = self.e.intern([(int(i), int(n)) for i, n in letters], [int(x) for x in q])
            self._ids[key] = r
            self._keys[r] = key
        return r

    cdef object _key(self, uint32_t i):
        k = self._keys.get(i)
        if k is None:
            k = (tuple([(l.first, l.second) for l in self.e.key_letters(i)]),
                 tuple(self.e.key_q(i)))
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

    cdef EpsData _eps_in(self, eps) except *:
        cdef EpsData d
        for q, p, terms in eps:
            d.qs.push_back([int(x) for x in q])
            d.ps.push_back([int(x) for x in p])
            d.vals.push_back(_coef_in(terms))
        return d

    def set_offset(self, off):
        """
        Make this an engine of a module ``V_{λ+L}``: the second vector of
        every product is ``e^{λ+s}``, given by its key ``(letters, s)``;
        ``off`` is ``(B(x_i, λ))_i`` (integers).  Call it on a new engine,
        before any product.

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: eps = [((-1,), (0,), [((), 1, 1)])]
            sage: f, v = ((), (-1,)), ((), (0,))     # e^{-α}, and s = 0: e^{α/2}
            sage: Engine('q', [[2]]).ope_mono(f, v, eps)      # V_L: B(-α, 0) = 0, no pole
            []
            sage: E = Engine('q', [[2]]); E.set_offset([1])   # V_{α/2+L}: B(x, α/2) = 1
            sage: E.ope_mono(f, v, eps)                         # a simple pole: e^{-α/2}, s = -1
            [[(((), (-1,)), [((), 1, 1)])]]
            sage: E.set_offset([1, 2])
            Traceback (most recent call last):
            ...
            ValueError: offset of the wrong length
        """
        self.e.set_offset([int(x) for x in off])

    def intern(self, key):
        """
        The engine's id of a key (interned on first use).

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: i = E.intern(((), (1,)))
            sage: E.intern(((), (1,))) == i, E.intern(((), (0,))) == i
            (True, False)
        """
        return self._id(key)

    def ope_mono(self, a, b, eps):
        r"""
        The poles of ``OPE[a, b]`` for keys ``a``, ``b`` (trimmed).

        EXAMPLES:

        `e^{\alpha}(z) e^{-\alpha}(w) \sim 1/(z-w)^2 + J(w)/(z-w)`, as
        the lattice vertex algebra gives::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.ope_mono(ep, em, eps)
            [[((((0, 0),), (0,)), [((), 1, 1)])], [(((), (0,)), [((), 1, 1)])]]
            sage: E.ope_mono(J, ep, eps)
            [[(((), (1,)), [((), 2, 1)])]]
            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: V = LatticeVertexAlgebra(QQ, matrix([[2]]))
            sage: V.e((1,)).ope(V.e((-1,)))
            [J0, 1]
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        cdef EpsData E = self._eps_in(eps)
        return self._poles_out(self.e.ope_mono(ia, ib, E, <PollFn>_poll))

    def no_mono(self, a, b, eps):
        """
        ``:a b:`` for keys ``a``, ``b``.

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.no_mono(J, ep, eps)
            [((((0, 0),), (1,)), [((), 1, 1)])]
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        cdef EpsData E = self._eps_in(eps)
        return self._elem_out(self.e.no_mono(ia, ib, E, <PollFn>_poll))

    def deriv_mono(self, m, int n):
        r"""
        ``∂^n m`` for a key ``m``.

        EXAMPLES:

        `\partial e^{\alpha} = :J e^{\alpha}:`::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.deriv_mono(ep, 1)
            [((((0, 0),), (1,)), [((), 1, 1)])]
        """
        cdef uint32_t im = self._id(m)
        return self._elem_out(self.e.deriv_mono(im, n, <PollFn>_poll))

    def ope_el(self, A, B, eps, int nthreads=1):
        """
        The poles of ``OPE[A, B]`` for elements ``A``, ``B`` (untrimmed).

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.ope_el([(ep, one(3))], [(em, one(1, 3))], eps)
            [[((((0, 0),), (0,)), [((), 1, 1)])], [(((), (0,)), [((), 1, 1)])]]
            sage: _ == E.ope_el([(ep, one(3))], [(em, one(1, 3))], eps, nthreads=2)
            True
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef EpsData E = self._eps_in(eps)
        cdef PolesData r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el(a, b, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el(a, b, E, 1, <PollFn>_poll)
        return self._poles_out(r)

    def ope_el_many(self, A, Bs, eps, int nthreads=1):
        """
        ``[ope_el(A, B) for B in Bs]``, all letter series in one batch.

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: A = [(J, one(1))]
            sage: E.ope_el_many(A, [[(ep, one(1))], A], eps)
            [[[(((), (1,)), [((), 2, 1)])]], [[], [(((), (0,)), [((), 2, 1)])]]]
        """
        cdef ElemData a = self._elem_in(A)
        cdef vector[ElemData] bs
        for B in Bs:
            bs.push_back(self._elem_in(B))
        cdef EpsData E = self._eps_in(eps)
        cdef vector[PolesData] r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el_many(a, bs, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el_many(a, bs, E, 1, <PollFn>_poll)
        cdef size_t i
        return [self._poles_out(r[i]) for i in range(r.size())]

    def nprod_el(self, A, long n, B, eps, int nthreads=1):
        r"""
        ``A_(n) B`` for any integer ``n``.

        EXAMPLES:

        `e^{\alpha}_{(-1)} e^{-\alpha} = (\partial J + :J J:)/2` (letters
        carry plain derivatives)::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.nprod_el([(ep, one(1))], -1, [(em, one(1))], eps)
            [((((0, 1),), (0,)), [((), 1, 2)]), ((((0, 0), (0, 0)), (0,)), [((), 1, 2)])]
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef EpsData E = self._eps_in(eps)
        cdef ElemData r
        if nthreads > 1:
            with nogil:
                r = self.e.nprod_el(a, n, b, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.nprod_el(a, n, b, E, 1, <PollFn>_poll)
        return self._elem_out(r)

    def nprod_el_many(self, els, tasks, eps, int nthreads=1):
        """
        ``[els[i]_(n) els[j] for (i, n, j) in tasks]`` in one call: each
        product computed in one of ``nthreads`` threads.

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: els = [[(ep, one(1))], [(em, one(1))]]
            sage: E.nprod_el_many(els, [(0, 1, 1), (0, -1, 0)], eps)
            [[(((), (0,)), [((), 1, 1)])], []]
            sage: _ == E.nprod_el_many(els, [(0, 1, 1), (0, -1, 0)], eps, nthreads=2)
            True
        """
        cdef vector[ElemData] E
        cdef vector[int] ia, ib
        cdef vector[long] ns
        cdef EpsData ep = self._eps_in(eps)
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
                r = self.e.nprod_el_many(E, ia, ns, ib, ep, nthreads, <PollFn>_poll)
        else:
            r = self.e.nprod_el_many(E, ia, ns, ib, ep, 1, <PollFn>_poll)
        return [self._elem_out(r[k]) for k in range(r.size())]

    def deriv_el(self, A, int n, int nthreads=1):
        """
        ``∂^n A``.

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: E.deriv_el([(em, one(1))], 1)
            [((((0, 0),), (-1,)), [((), -1, 1)])]
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
        """Entries of the E^+, creation series, Schur and derivative
        caches, interned Heisenberg monomials and keys, memoised monomial
        products, and the scheduler statistics (cumulative).

        EXAMPLES::

            sage: from vertex_algebras._lattice_engine import Engine
            sage: E = Engine('q', [[2]])
            sage: one = lambda n, d=1: [((), n, d)]
            sage: J, ep, em = (((0, 0),), (0,)), ((), (1,)), ((), (-1,))
            sage: eps = [((q,), (p,), one(1)) for q in range(-2, 3) for p in range(-2, 3)]
            sage: _ = E.ope_mono(ep, em, eps)
            sage: d = E.cache_sizes(); sorted(d)
            ['cseries', 'deriv', 'eplus', 'heisenberg', 'idle_ms', 'keys',
             'products', 'pushed', 'schur', 'stolen', 'wait_ms', 'waits']
            sage: d['keys'] >= 2
            True
        """
        s = self.e.cache_sizes()
        return dict(zip(['eplus', 'cseries', 'schur', 'deriv',
                         'heisenberg', 'keys', 'products', 'pushed', 'stolen',
                         'waits', 'wait_ms', 'idle_ms'], s))
