# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Cython wrapper of the C++ engine of tensor products of universal and lattice
vertex algebras (``_tensor_engine.hpp``), built when the package is
installed (``meson.build``: C++17, linked with FLINT and GMP).

Sage-free on purpose: everything crossing this boundary is Python ints,
tuples and lists; ``tensor_product_vertex_algebra._TensorEngineKernel``
converts to and from Sage rings.  Boundary types and conversions shared
with the other engines: ``_engine_base.pxi``.

- Factors: ``add_universal(ngens, parities)`` (then ``set_ope``) and
  ``add_lattice(gram)``, in order, before anything else.
- A key is the tensor product's key: a tuple of factor keys (universal:
  letters; lattice: ``(letters, q)``).
- A coefficient is a list of terms ``(exponents, numerator, denominator)``;
  an element a list of ``(key, coefficient)``; poles a list of elements.
- ``eps``: one entry per factor, ``None`` or a list of ``(q, p,
  coefficient)`` (the cocycle values of a lattice factor for the call).

``nthreads > 1`` releases the GIL; results do not depend on ``nthreads``.
An engine must not be used from two Python threads at once.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
include "_engine_base.pxi"

cdef extern from "_tensor_engine.hpp" namespace "vae":
    ctypedef vector[pair[int, int]] Letters
    cdef cppclass EpsData:
        vector[vector[long]] qs
        vector[vector[long]] ps
        vector[CoefData] vals
    cdef cppclass TensorEngineBase:
        int add_universal(int, const vector[int] &) except +_raise_error
        int add_lattice(const vector[vector[long]] &) except +_raise_error
        void set_ope(int, int, int, const PolesData &) except +_raise_error
        uint32_t intern_universal(int, const Letters &) except +_raise_error
        uint32_t intern_lattice(int, const Letters &, const vector[long] &) except +_raise_error
        Letters universal_letters(int, uint32_t) except +_raise_error
        Letters lattice_letters(int, uint32_t) except +_raise_error
        vector[long] lattice_q(int, uint32_t) except +_raise_error
        uint32_t intern(const vector[uint32_t] &) except +_raise_error
        vector[uint32_t] components(uint32_t) except +_raise_error
        PolesData ope_mono(uint32_t, uint32_t, const vector[EpsData] &, PollFn) except +_raise_error
        ElemData no_mono(uint32_t, uint32_t, const vector[EpsData] &, PollFn) except +_raise_error
        ElemData nprod_mono(uint32_t, long, uint32_t, const vector[EpsData] &, PollFn) except +_raise_error
        ElemData deriv_mono(uint32_t, int, PollFn) except +_raise_error
        PolesData ope_el(const ElemData &, const ElemData &, const vector[EpsData] &, int,
                         PollFn) except +_raise_error nogil
        vector[PolesData] ope_el_many(const ElemData &, const vector[ElemData] &,
                                      const vector[EpsData] &, int, PollFn) except +_raise_error nogil
        ElemData nprod_el(const ElemData &, long, const ElemData &, const vector[EpsData] &, int,
                          PollFn) except +_raise_error nogil
        ElemData deriv_el(const ElemData &, int, int, PollFn) except +_raise_error nogil
        vector[ElemData] nprod_el_many(const vector[ElemData] &, const vector[int] &,
                                       const vector[long] &, const vector[int] &,
                                       const vector[EpsData] &, int, PollFn) except +_raise_error nogil
        vector[size_t] cache_sizes()
    TensorEngineBase *make_tensor_engine(string, long, unsigned long,
                                         const CoefData &) except +_raise_error


cdef class Engine:
    r"""
    One engine = one tensor product of universal and lattice vertex algebras
    over one coefficient ring (``kind`` as for the other engines).

    EXAMPLES:

    A free fermion `\psi` (universal, odd) tensor the lattice `\ZZ` (odd
    `e^{\pm 1}`) over `\QQ`; keys are pairs of factor keys::

        sage: from vertex_algebras._tensor_engine import Engine
        sage: E = Engine('q')
        sage: one = lambda n, d=1: [((), n, d)]
        sage: E.add_universal(1, [1]), E.add_lattice([[1]])
        (0, 1)
        sage: E.set_ope(0, 0, 0, [[((), one(1))]])
        sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
        ....:            for p in range(-2, 3)]]
        sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
        sage: E.no_mono(psi, ep, eps), E.no_mono(ep, psi, eps)
        ([((((0, 0),), ((), (1,))), [((), 1, 1)])],
         [((((0, 0),), ((), (1,))), [((), -1, 1)])])
    """
    cdef TensorEngineBase *e
    cdef list _kinds        # per factor: 'universal' / 'lattice'
    cdef dict _ids          # key -> engine id
    cdef dict _keys         # engine id -> key
    cdef list _fids         # per factor: {factor key: factor id}
    cdef list _fkeys        # per factor: {factor id: factor key}
    cdef readonly str kind

    def __cinit__(self, str kind, long nvars=0, unsigned long modulus=0, minpoly=()):
        """
        Build an engine without factors over the ring ``kind``.

        TESTS::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: Engine('q').kind
            'q'
            sage: Engine('nmod', modulus=7).kind
            'nmod'
            sage: Engine('foo')
            Traceback (most recent call last):
            ...
            ValueError: ...
        """
        self.e = NULL
        self.kind = kind
        self._kinds = []
        self._ids = {}
        self._keys = {}
        self._fids = []
        self._fkeys = []
        cdef CoefData mp = _coef_in(minpoly)
        self.e = make_tensor_engine(kind.encode(), nvars, modulus, mp)

    def __dealloc__(self):
        if self.e != NULL:
            del self.e

    def add_universal(self, int ngens, parities):
        """
        Append a universal factor with ``ngens`` generators of the given
        parities; return its index.  Its OPEs follow with :meth:`set_ope`.

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: E.add_universal(2, [0, 1]), E.add_universal(1, [0])
            (0, 1)
        """
        cdef vector[int] par = [int(p) for p in parities]
        f = self.e.add_universal(ngens, par)
        self._kinds.append('universal')
        self._fids.append({})
        self._fkeys.append({})
        return f

    def add_lattice(self, gram):
        """
        Append a lattice factor of Gram matrix ``gram``; return its index.

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: E.add_lattice([[2]]), E.add_lattice([[1, 0], [0, -1]])
            (0, 1)
        """
        cdef vector[vector[long]] g = [[int(x) for x in row] for row in gram]
        f = self.e.add_lattice(g)
        self._kinds.append('lattice')
        self._fids.append({})
        self._fkeys.append({})
        return f

    cdef uint32_t _fid(self, int f, k) except? 0xFFFFFFFF:
        cdef dict d = self._fids[f]
        r = d.get(k)
        if r is None:
            if self._kinds[f] == 'universal':
                r = self.e.intern_universal(f, [(int(i), int(n)) for i, n in k])
            else:
                letters, q = k
                r = self.e.intern_lattice(f, [(int(i), int(n)) for i, n in letters],
                                          [int(x) for x in q])
            d[k] = r
            (<dict>self._fkeys[f])[r] = k
        return r

    cdef object _fkey(self, int f, uint32_t i):
        cdef dict d = self._fkeys[f]
        k = d.get(i)
        if k is None:
            if self._kinds[f] == 'universal':
                k = tuple([(l.first, l.second) for l in self.e.universal_letters(f, i)])
            else:
                k = (tuple([(l.first, l.second) for l in self.e.lattice_letters(f, i)]),
                     tuple(self.e.lattice_q(f, i)))
            d[i] = k
            (<dict>self._fids[f])[k] = i
        return k

    cdef uint32_t _id(self, key) except? 0xFFFFFFFF:
        r = self._ids.get(key)
        if r is None:
            r = self.e.intern([self._fid(f, k) for f, k in enumerate(key)])
            self._ids[key] = r
            self._keys[r] = key
        return r

    cdef object _key(self, uint32_t i):
        k = self._keys.get(i)
        if k is None:
            k = tuple([self._fkey(f, c) for f, c in enumerate(self.e.components(i))])
            self._keys[i] = k
            self._ids[k] = i
        return k

    def set_ope(self, int f, int i, int j, poles):
        """Declare the OPE of generators ``i``, ``j`` of the universal factor
        ``f`` (poles in factor keys).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.ope_mono(psi, psi, eps)
            [[(((), ((), (0,))), [((), 1, 1)])]]
        """
        cdef PolesData P
        cdef ElemData d
        cdef pair[uint32_t, CoefData] t
        for el in poles:
            d.clear()
            for key, terms in el:
                t.first = self._fid(f, key)
                t.second = _coef_in(terms)
                d.push_back(t)
            P.push_back(d)
        self.e.set_ope(f, i, j, P)

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

    cdef vector[EpsData] _eps_in(self, eps) except *:
        cdef vector[EpsData] out
        cdef EpsData d
        for f in range(len(self._kinds)):
            d.qs.clear()
            d.ps.clear()
            d.vals.clear()
            if eps is not None and f < len(eps) and eps[f] is not None:
                for q, p, terms in eps[f]:
                    d.qs.push_back([int(x) for x in q])
                    d.ps.push_back([int(x) for x in p])
                    d.vals.push_back(_coef_in(terms))
            out.push_back(d)
        return out

    def intern(self, key):
        """
        The engine's id of a key (interned on first use).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: i = E.intern(psi)
            sage: E.intern(psi) == i, E.intern(ep) == i
            (True, False)
        """
        return self._id(key)

    def ope_mono(self, a, b, eps):
        """
        The poles of ``OPE[a, b]`` for keys ``a``, ``b`` (trimmed).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.ope_mono(ep, em, eps)
            [[(((), ((), (0,))), [((), 1, 1)])]]
            sage: E.ope_mono(psi, ep, eps)
            []
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        cdef vector[EpsData] E = self._eps_in(eps)
        return self._poles_out(self.e.ope_mono(ia, ib, E, <PollFn>_poll))

    def no_mono(self, a, b, eps):
        """
        ``:a b:`` for keys ``a``, ``b`` (with the Koszul sign).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.no_mono(ep, psi, eps)
            [((((0, 0),), ((), (1,))), [((), -1, 1)])]
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        cdef vector[EpsData] E = self._eps_in(eps)
        return self._elem_out(self.e.no_mono(ia, ib, E, <PollFn>_poll))

    def nprod_mono(self, a, long n, b, eps):
        r"""
        ``a_(n) b`` for keys ``a``, ``b`` and any integer ``n``.

        EXAMPLES:

        `e^{1}_{(-2)} e^{-1} = (\partial J + :J J:)/2` in the lattice
        factor (letters carry plain derivatives)::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.nprod_mono(ep, -2, em, eps)
            [(((), (((0, 1),), (0,))), [((), 1, 2)]),
             (((), (((0, 0), (0, 0)), (0,))), [((), 1, 2)])]
        """
        cdef uint32_t ia = self._id(a), ib = self._id(b)
        cdef vector[EpsData] E = self._eps_in(eps)
        return self._elem_out(self.e.nprod_mono(ia, n, ib, E, <PollFn>_poll))

    def deriv_mono(self, m, int n):
        """
        ``∂^n m`` for a key ``m`` (Leibniz rule over the factors).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.deriv_mono((((0, 0),), ((), (1,))), 1)
            [((((0, 0),), (((0, 0),), (1,))), [((), 1, 1)]),
             ((((0, 1),), ((), (1,))), [((), 1, 1)])]
        """
        cdef uint32_t im = self._id(m)
        return self._elem_out(self.e.deriv_mono(im, n, <PollFn>_poll))

    def ope_el(self, A, B, eps, int nthreads=1):
        """
        The poles of ``OPE[A, B]`` for elements ``A``, ``B`` (untrimmed).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.ope_el([(ep, one(2))], [(em, one(1)), (psi, one(1))], eps)
            [[(((), ((), (0,))), [((), 2, 1)])]]
            sage: _ == E.ope_el([(ep, one(2))], [(em, one(1)), (psi, one(1))], eps,
            ....:               nthreads=2)
            True
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef vector[EpsData] E = self._eps_in(eps)
        cdef PolesData r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el(a, b, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el(a, b, E, 1, <PollFn>_poll)
        return self._poles_out(r)

    def ope_el_many(self, A, Bs, eps, int nthreads=1):
        """
        ``[ope_el(A, B) for B in Bs]`` in one call.

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: A = [(psi, one(1))]
            sage: E.ope_el_many(A, [A, [(ep, one(1))]], eps)
            [[[(((), ((), (0,))), [((), 1, 1)])]], []]
        """
        cdef ElemData a = self._elem_in(A)
        cdef vector[ElemData] bs
        for B in Bs:
            bs.push_back(self._elem_in(B))
        cdef vector[EpsData] E = self._eps_in(eps)
        cdef vector[PolesData] r
        if nthreads > 1:
            with nogil:
                r = self.e.ope_el_many(a, bs, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.ope_el_many(a, bs, E, 1, <PollFn>_poll)
        cdef size_t i
        return [self._poles_out(r[i]) for i in range(r.size())]

    def nprod_el(self, A, long n, B, eps, int nthreads=1):
        """
        ``A_(n) B`` for any integer ``n``.

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.nprod_el([(psi, one(1))], -1, [(ep, one(1))], eps)
            [((((0, 0),), ((), (1,))), [((), 1, 1)])]
        """
        cdef ElemData a = self._elem_in(A), b = self._elem_in(B)
        cdef vector[EpsData] E = self._eps_in(eps)
        cdef ElemData r
        if nthreads > 1:
            with nogil:
                r = self.e.nprod_el(a, n, b, E, nthreads, <PollFn>_poll)
        else:
            r = self.e.nprod_el(a, n, b, E, 1, <PollFn>_poll)
        return self._elem_out(r)

    def nprod_el_many(self, els, tasks, eps, int nthreads=1):
        """``[els[i]_(n) els[j] for (i, n, j) in tasks]`` in one call: the
        key-level products of all of them first, then each sum in one of
        ``nthreads`` threads.

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: els = [[(psi, one(1))], [(ep, one(1))]]
            sage: E.nprod_el_many(els, [(0, -1, 1), (1, -1, 0)], eps)
            [[((((0, 0),), ((), (1,))), [((), 1, 1)])],
             [((((0, 0),), ((), (1,))), [((), -1, 1)])]]
            sage: _ == E.nprod_el_many(els, [(0, -1, 1), (1, -1, 0)], eps, nthreads=2)
            True
        """
        cdef vector[ElemData] E
        cdef vector[int] ia, ib
        cdef vector[long] ns
        cdef vector[EpsData] ep = self._eps_in(eps)
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

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: E.deriv_el([(ep, one(1))], 1)
            [(((), (((0, 0),), (1,))), [((), 1, 1)])]
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
        """Entries of the tensor product, derivative, lattice-factor and
        universal-factor n-product caches, interned keys, and the scheduler
        statistics (cumulative).

        EXAMPLES::

            sage: from vertex_algebras._tensor_engine import Engine
            sage: E = Engine('q')
            sage: one = lambda n, d=1: [((), n, d)]
            sage: E.add_universal(1, [1]), E.add_lattice([[1]])
            (0, 1)
            sage: E.set_ope(0, 0, 0, [[((), one(1))]])
            sage: eps = [None, [((q,), (p,), one(1)) for q in range(-2, 3)
            ....:                for p in range(-2, 3)]]
            sage: psi, ep, em = (((0, 0),), ((), (0,))), ((), ((), (1,))), ((), ((), (-1,)))
            sage: _ = E.ope_mono(ep, em, eps)
            sage: d = E.cache_sizes(); sorted(d)
            ['idle_ms', 'keys', 'lattice', 'pushed', 'stolen', 'tderiv', 'tprod',
             'universal_nprod', 'wait_ms', 'waits']
            sage: d['keys'] >= 2
            True
        """
        s = self.e.cache_sizes()
        return dict(zip(['tprod', 'tderiv', 'lattice', 'universal_nprod', 'keys', 'pushed',
                         'stolen', 'waits', 'wait_ms', 'idle_ms'], s))
