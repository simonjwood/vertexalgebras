# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Top quotients of induced modules, depth by depth from lowering data.

For an induced module ``M`` (Zhu's ``\bar M(U)`` of a module ``U`` over the
Zhu algebra, or Dong--Li--Mason's ``\bar M_n(U)`` of a module over the
level-``n`` Zhu algebra [DLM1998]_, also of a quotient of a universal vertex
algebra), ``M.top_quotient()`` is ``L = M / J``, ``J`` the largest submodule
meeting the WINDOW (the degrees ``0, ..., n`` of ``M``) trivially.  For
``n = 0`` and ``U`` simple, ``L`` is the simple module ``L(U)``; for a
non-simple ``U`` it is the largest quotient of ``M`` in which every nonzero
submodule meets the top, e.g. a projective cover whose socle lies in its top
(``P(0)`` of the triplet algebras).  Projective covers whose socle tops lie
in degrees ``<= n`` are top quotients of ``\bar M_n``.

**The construction.**  A vector of degree ``N > n`` of ``L`` is zero iff
every lowering mode of the strong generators kills it (it would generate a
submodule in the degrees ``>= N``), so ``L_N`` is spanned by the candidates
``g_{-k} b`` (``b`` a basis vector of ``L_{N - k}``) modulo the common
kernel of their lowering data.  The lowering data come from the commutator
formula ``g'_(p) g_(q) b = g_(q) g'_(p) b + [g'_(p), g_(q)] b`` with the
modes of the composite fields of the commutator by the normally ordered
product formula, a non-lowering letter moved to the left so that no
intermediate vector leaves the degrees ``<= N``; the zero modes on ``L_N``
from their lowering data.  Neither the free PBW monomials of ``M`` nor the
relations of a quotient vertex algebra are used: the relations of ``Q =
V/I`` act by zero on ``L`` as soon as they do on the window (``I . L`` is
spanned by the ``r_(p) w``, ``r`` in ``I``, ``w`` in the window, and meets
the window in ``r_(p) w`` with ``r_(p) w`` in the window), so the top
quotient over ``V`` is the one over ``Q``.

The basis vectors are lowering words ``g_{m_1} ... g_{m_k} w`` (physics
modes, ``w`` a basis vector of the window), the first candidates (by word
length, then word) whose lowering data are independent.  Degrees are
computed on demand.  Backends: the C++ engine
(``_lowering_engine``, over `\QQ` and `\GF{p}`, threads
from :class:`~vertex_algebras.vertex_algebra.VertexAlgebraParallelism`) and
a pure-Python version of the same algorithm (any field); both give the same
basis and matrices.  Strong generators of positive integer weight and even
parity, a finite-dimensional top.

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
from sage.arith.misc import binomial
from sage.matrix.constructor import matrix
from sage.matrix.special import block_matrix, identity_matrix
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .vertex_algebra_module import VertexAlgebraModule


def _falling(n, k):
    r"""
    ``n (n - 1) ... (n - k + 1)``.

    TESTS::

        sage: from vertex_algebras.top_quotient_module import _falling
        sage: _falling(5, 2), _falling(-1, 3), _falling(4, 0)
        (20, -6, 1)
    """
    r = ZZ.one()
    for i in range(k):
        r *= n - i
    return r


class _PythonLowering:
    r"""
    The pure-Python lowering construction (the algorithm of the C++ engine
    ``_lowering_engine.hpp``, the same basis and matrices) over a field
    ``R``: matrices of the generator modes by depth, the modes of words of
    the universal algebra memoised.

    INPUT:

    - ``R`` -- the base field
    - ``wt`` -- the generator weights (positive integers)
    - ``nprod`` -- a function ``(x, i, B) -> x_(i) B`` on keys of the
      universal algebra (a dict)

    Modes are bracket modes: ``g_(nn)`` maps depth ``e`` to ``e + wt g - nn
    - 1``; a matrix has the source basis as columns; ``None`` is zero.

    EXAMPLES:

    The Ising model's `\sigma` (Virasoro at `c = 1/2`, `h = 1/16`)::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.top_quotient_module import _PythonLowering
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
        sage: P = _PythonLowering(QQ, [2], V._nprod_mono)
        sage: P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
        sage: for d in range(7):
        ....:     _ = P.extend()
        sage: P.dims()
        [1, 1, 1, 2, 2, 3, 4, 5]
    """

    def __init__(self, R, wt, nprod):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: _PythonLowering(QQ, [2], V._nprod_mono).depth()
            -1
        """
        self._R = R
        self._wt = [ZZ(w) for w in wt]
        self._nprod_fn = nprod
        self._np = {}
        self._D = -1
        self._dims = []
        self._words = []
        self._gen = {}
        self._memo = {}

    def _nprod(self, a, n, b):
        """
        ``a_(n) b`` (cached, zero coefficients dropped).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: _PythonLowering(QQ, [2], V._nprod_mono)._nprod(((0, 0),), 3, ((0, 0),))
            {(): 1/4}
        """
        k = (a, n, b)
        r = self._np.get(k)
        if r is None:
            r = self._np[k] = {m: c for m, c in self._nprod_fn(a, n, b).items() if c}
        return r

    def _kw(self, key):
        """
        The weight of a word.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: _PythonLowering(QQ, [2], V._nprod_mono)._kw(((0, 1), (0, 0)))
            5
        """
        return sum(self._wt[g] + k for g, k in key)

    def set_window(self, dims, modes):
        r"""
        Depths ``0, ..., n0`` of dimensions ``dims`` with every generator mode
        between them: ``modes[(g, nn, e)]`` a matrix (missing: zero).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1, 1], {})
            sage: P.depth(), P.dims(), P.basis_word(1, 0)
            (1, [1, 1], ((), 1))
        """
        R = self._R
        n0 = len(dims) - 1
        self._dims = [int(d) for d in dims]
        self._D = n0
        u = 0
        self._words = []
        for d in range(n0 + 1):
            self._words.append([((), u + j) for j in range(self._dims[d])])
            u += self._dims[d]
        for g in range(len(self._wt)):
            for e in range(n0 + 1):
                for t in range(n0 + 1):
                    nn = e + self._wt[g] - 1 - t
                    A = modes.get((g, nn, e))
                    self._gen[(g, nn, e)] = (None if A is None or A.is_zero()
                                             else matrix(R, A))

    def depth(self):
        """
        The largest known depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {}); P.depth()
            0
        """
        return self._D

    def dims(self):
        """
        The dimensions by depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {}); P.dims()
            [1]
        """
        return list(self._dims)

    def basis_word(self, d, i):
        """
        ``(modes, u)``: basis vector ``i`` of depth ``d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono)
            sage: P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])}); _ = P.extend()
            sage: P.basis_word(1, 0)
            (((0, -1),), 0)
        """
        return self._words[d][i]

    def _target(self, w, nn, e):
        """
        The depth ``e + w - nn - 1``.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: _PythonLowering(QQ, [2], V._nprod_mono)._target(2, 0, 3)
            4
        """
        return e + w - nn - 1

    def gmode(self, g, nn, e):
        """
        The matrix of ``g_(nn)`` on depth ``e`` (``None``: zero).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
            sage: P.gmode(0, 1, 0), P.gmode(0, 2, 0)
            ([1/16], None)
            sage: P.gmode(0, 0, 0)
            Traceback (most recent call last):
            ...
            ValueError: a generator mode outside the known depths
        """
        t = self._target(self._wt[g], nn, e)
        if t < 0 or e < 0:
            return None
        if e > self._D or t > self._D:
            raise ValueError('a generator mode outside the known depths')
        if self._dims[e] == 0 or self._dims[t] == 0:
            return None
        try:
            return self._gen[(g, nn, e)]
        except KeyError:
            raise ValueError('a generator mode that is not known yet')

    def mode(self, key, nn, e):
        r"""
        The matrix of the mode ``(nn)`` of the word ``key`` on depth ``e``
        (``None``: zero), by the normally ordered product formula with the
        mode split (no intermediate depth above the source and target).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
            sage: P.mode(((0, 0), (0, 0)), 3, 0)          # o(:L L:) = h^2 + 2h on the top
            [33/256]
            sage: P.mode((), -1, 0), P.mode(((0, 1),), 2, 0)
            ([1], [-1/8])
        """
        R = self._R
        w = self._kw(key)
        t = self._target(w, nn, e)
        if t < 0 or e < 0:
            return None
        mk = (key, nn, e)
        if mk in self._memo:
            return self._memo[mk]
        if not key:
            A = identity_matrix(R, self._dims[e]) if nn == -1 else None
        elif len(key) == 1:
            g, k = key[0]
            c = (-1)**k * _falling(nn, k)
            A = None
            if c:
                G = self.gmode(g, nn - k, e)
                A = None if G is None else R(c) * G
        else:
            x, B = key[:1], key[1:]
            wx, wB = self._kw(x), self._kw(B)
            A = matrix(R, self._dims[t], self._dims[e])
            # sum_j x_(-1-j) B_(nn+j): B lowers further than the target
            for j in range(max(e + wB - nn, 0)):
                Bm = self.mode(B, nn + j, e)
                if Bm is not None:
                    Xm = self.mode(x, -1 - j, self._target(wB, nn + j, e))
                    if Xm is not None:
                        A += Xm * Bm
            # sum_j B_(nn-1-j) x_(j)
            for j in range(e + wx):
                if wx - j - 1 < 0:             # x_(j) lowers: as it stands
                    Xm = self.mode(x, j, e)
                    if Xm is not None:
                        Bm = self.mode(B, nn - 1 - j, self._target(wx, j, e))
                        if Bm is not None:
                            A += Bm * Xm
                else:                          # x_(j) B_(m) - [x_(j), B_(m)]
                    m = nn - 1 - j
                    Bm = self.mode(B, m, e)
                    if Bm is not None:
                        Xm = self.mode(x, j, self._target(wB, m, e))
                        if Xm is not None:
                            A += Xm * Bm
                    for i in range(j + 1):
                        for kk, c in self._nprod(x, i, B).items():
                            Cm = self.mode(kk, nn - 1 - i, e)
                            if Cm is not None:
                                A -= R(binomial(j, i)) * c * Cm
            if A.is_zero():
                A = None
        self._memo[mk] = A
        return A

    def _commutator(self, g1, p, g2, q, e):
        r"""
        ``[g1_(p), g2_(q)] = Σ_j C(p, j) (g1_(j) g2)_(p+q-j)`` on depth ``e``
        (``None`` if it leaves the depths).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
            sage: _ = P.extend(); P._commutator(0, 2, 0, 0, 0)     # [L_1, L_{-1}] = 2 L_0
            [1/8]
        """
        R = self._R
        wt = self._wt
        t = e + wt[g1] + wt[g2] - p - q - 2
        if t < 0:
            return None
        A = matrix(R, self._dims[t], self._dims[e])
        for j in range(wt[g1] + wt[g2]):
            b = binomial(p, j)
            if not b:
                continue
            for kk, c in self._nprod(((g1, 0),), j, ((g2, 0),)).items():
                Cm = self.mode(kk, p + q - j, e)
                if Cm is not None:
                    A += R(b) * c * Cm
        return A

    def extend(self, nthreads=1):
        r"""
        Add one depth; return its dimension.  An interruption leaves the
        previous depths as they were.

        TESTS:

        Interrupted (a deterministic interrupt in the middle), then the same
        result::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
            sage: for d in range(3):
            ....:     _ = P.extend()
            sage: calls = [0]; orig = P._commutator
            sage: def stop(*args):
            ....:     calls[0] += 1
            ....:     if calls[0] == 3:
            ....:         raise KeyboardInterrupt
            ....:     return orig(*args)
            sage: P._commutator = stop
            sage: P.extend()
            Traceback (most recent call last):
            ...
            KeyboardInterrupt
            sage: P.depth(), P.dims()
            (3, [1, 1, 1, 2])
            sage: del P._commutator
            sage: for d in range(4):
            ....:     _ = P.extend()
            sage: P.dims()
            [1, 1, 1, 2, 2, 3, 4, 5]
        """
        N = self._D + 1
        try:
            return self._extend(N)
        except BaseException:
            # undo depth N: its basis, its generator modes, the word modes
            # into or out of it (the others are determined by the earlier
            # depths)
            if len(self._dims) > N:
                del self._dims[N:]
                del self._words[N:]
            self._D = N - 1
            for k in [k for k in self._gen if k[2] == N
                      or self._target(self._wt[k[0]], k[1], k[2]) == N]:
                del self._gen[k]
            for k in [k for k in self._memo if k[2] == N
                      or self._target(self._kw(k[0]), k[1], k[2]) == N]:
                del self._memo[k]
            raise

    def _extend(self, N):
        """
        The body of :meth:`extend`.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[0]])})
            sage: P._extend(1), P._extend(2)            # the vacuum: L_{-1} v = 0
            (0, 1)
        """
        R = self._R
        wt = self._wt
        G = len(wt)
        dims = self._dims
        # candidates g_{-n} b, by (word length, word)
        cands = []
        for n in range(1, N + 1):
            for g in range(G):
                for b in range(dims[N - n]):
                    modes, u = self._words[N - n][b]
                    cands.append(((((g, -n),) + modes, u), g, n, b))
        cands.sort(key=lambda c: (len(c[0][0]), c[0]))
        dims.append(0)
        self._words.append([])
        self._D = N
        # the lowering data: row blocks (g', m), columns the candidates
        blocks = []
        for g2 in range(G):
            for m in range(1, N + 1):
                p = m + wt[g2] - 1
                cols = {}
                for n in range(1, N + 1):
                    if dims[N - n] == 0:
                        continue
                    for g in range(G):
                        nn = -n + wt[g] - 1
                        A = matrix(R, dims[N - m], dims[N - n])
                        L1 = self.gmode(g2, p, N - n)            # g'_m b, then g_{-n}
                        if L1 is not None:
                            R1 = self.gmode(g, nn, N - n - m)
                            if R1 is not None:
                                A += R1 * L1
                        Cm = self._commutator(g2, p, g, nn, N - n)
                        if Cm is not None:
                            A += Cm
                        cols[(g, n)] = A
                blocks.append(matrix(R, len(cands), dims[N - m],
                                     [cols[(g, n)].column(b) for (_, g, n, b) in cands]
                                     ).transpose() if dims[N - m] and cands
                              else matrix(R, dims[N - m], len(cands)))
        L = block_matrix(R, [[Bk] for Bk in blocks], subdivide=False)
        piv = L.pivots()
        dN = len(piv)
        Lb = L.matrix_from_columns(piv)
        X = Lb.solve_right(L) if dN else matrix(R, 0, len(cands))
        new = {}
        # raising into N: the coordinates of the candidates
        for n in range(1, N + 1):
            if dims[N - n] == 0 or dN == 0:
                continue
            for g in range(G):
                A = matrix(R, dN, dims[N - n])
                for ci, (_, gg, nnn, b) in enumerate(cands):
                    if gg == g and nnn == n:
                        A.set_column(b, X.column(ci))
                new[(g, -n + wt[g] - 1, N - n)] = None if A.is_zero() else A
        # lowering from N: the lowering data of the basis
        r = 0
        for g2 in range(G):
            for m in range(1, N + 1):
                h = dims[N - m]
                A = Lb.matrix_from_rows(range(r, r + h))
                new[(g2, m + wt[g2] - 1, N)] = None if A.is_zero() else A
                r += h
        dims[N] = dN
        self._words[N] = [cands[i][0] for i in piv]
        self._gen.update(new)
        # zero modes on N by their lowering data:
        # g''_(q) g'_(p0) c = g'_(p0) g''_(q) c + [g''_(q), g'_(p0)] c
        for g1 in range(G):
            p0 = wt[g1] - 1
            if dN == 0:
                self._gen[(g1, p0, N)] = None
                continue
            rows = []
            for g2 in range(G):
                for k in range(1, N + 1):
                    q = k + wt[g2] - 1
                    A = matrix(R, dims[N - k], dN)
                    Lk = self.gmode(g2, q, N)
                    if Lk is not None:
                        Z0 = self.gmode(g1, p0, N - k)
                        if Z0 is not None:
                            A += Z0 * Lk
                    Cm = self._commutator(g2, q, g1, p0, N)
                    if Cm is not None:
                        A += Cm
                    rows.append(A)
            data = block_matrix(R, [[Ak] for Ak in rows], subdivide=False)
            try:
                Z = Lb.solve_right(data)
            except ValueError:
                raise ValueError('the zero modes at depth {} are not determined by their '
                                 'lowering data (not a module?)'.format(N))
            self._gen[(g1, p0, N)] = None if Z.is_zero() else Z
        return dN

    def word_matrix(self, key, nn, e):
        """
        The matrix of the mode ``(nn)`` of a word on depth ``e`` (``None``:
        zero); depths must be known.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _PythonLowering
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: P = _PythonLowering(QQ, [2], V._nprod_mono); P.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
            sage: P.word_matrix(((0, 0),), 0, 0)
            Traceback (most recent call last):
            ...
            ValueError: a mode outside the known depths
        """
        t = self._target(self._kw(key), nn, e)
        if e < 0 or e > self._D or t > self._D:
            raise ValueError('a mode outside the known depths')
        return self.mode(key, nn, e)


class _EngineLowering:
    r"""
    The C++ engine (``_lowering_engine.LoweringEngine``)
    behind the interface of :class:`_PythonLowering`: matrices as Sage
    matrices over ``R`` (`\QQ` or `\GF{p}`), split products supplied on
    demand.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.top_quotient_module import _EngineLowering
        sage: from vertex_algebras.vertex_algebra import _compiled_module
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
        sage: mod, _ = _compiled_module('_lowering_engine')
        sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
        sage: if E:
        ....:     E.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])})
        ....:     for d in range(7):
        ....:         _ = E.extend()
        sage: E is None or E.dims() == [1, 1, 1, 2, 2, 3, 4, 5]
        True

    TESTS:

    The same dimensions and matrices as :class:`_PythonLowering`, over
    `\GF{7}` (`c = 4`, `h = 2`)::

        sage: from vertex_algebras.top_quotient_module import _PythonLowering
        sage: F = GF(7); V7 = va.Virasoro(F, F(4), conformal_vector='L')
        sage: B = [_PythonLowering(F, [2], V7._nprod_mono)]
        sage: if mod:
        ....:     B.append(_EngineLowering(mod, F, [2], V7._nprod_mono))
        sage: for X in B:
        ....:     X.set_window([1], {(0, 1, 0): matrix(F, [[2]])})
        ....:     for d in range(6):
        ....:         _ = X.extend()
        sage: len({str(X.dims()) for X in B}), len({str(X.word_matrix(((0, 0), (0, 1)), 4, 5)) for X in B})
        (1, 1)
    """

    def __init__(self, mod, R, wt, nprod):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(GF(7), 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: mod is None or _EngineLowering(mod, GF(7), [2], V._nprod_mono)._E.kind == 'nmod'
            True
            sage: try:
            ....:     r = mod and _EngineLowering(mod, QQ['c'], [2], V._nprod_mono)
            ....: except TypeError as err:
            ....:     r = str(err)
            sage: mod is None or r == 'the lowering engine takes QQ and GF(p) only'
            True
        """
        self._R = R
        self._wt = [int(w) for w in wt]
        self._nprod_fn = nprod
        if R is QQ:
            self._E = mod.LoweringEngine('q', self._wt)
        elif R.is_field() and R.is_finite() and R.is_prime_field() and R.characteristic() < 2**62:
            self._E = mod.LoweringEngine('nmod', self._wt, int(R.characteristic()))
        else:
            raise TypeError('the lowering engine takes QQ and GF(p) only')
        self._split = set()
        G = len(self._wt)
        for g1 in range(G):
            for g2 in range(G):
                a, b = ((g1, 0),), ((g2, 0),)
                self._E.set_ope(g1, g2, [self._el(nprod(a, j, b))
                                         for j in range(self._wt[g1] + self._wt[g2])])
        self._close_splits(self._E.missing_splits())

    def _coef(self, c):
        """
        A coefficient in the engine's format.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: mod is None or _EngineLowering(mod, QQ, [2], V._nprod_mono)._coef(-3/4) == [((), -3, 4)]
            True
        """
        if self._R is QQ:
            c = QQ(c)
            return [((), int(c.numerator()), int(c.denominator()))]
        return [((), int(ZZ(self._R(c))), 1)]

    def _el(self, d):
        """
        An element (a dict) in the engine's format.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: mod is None or _EngineLowering(mod, QQ, [2], V._nprod_mono)._el({(): 1/2, ((0, 0),): 0}) == [((), [((), 1, 2)])]
            True
        """
        return [(k, self._coef(c)) for k, c in d.items() if c]

    def _close_splits(self, words):
        r"""
        Give the engine the split products ``x_(i) B`` of the composite
        words ``:x B:`` among ``words``, their tails and their products.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or (E._close_splits([((0, 0), (0, 0), (0, 0))]), E._E.missing_splits())[1] == []
            True
        """
        todo = list(words)
        while todo:
            w = todo.pop()
            if len(w) < 2 or w in self._split:
                continue
            self._split.add(w)
            x, B = w[:1], w[1:]
            prods = [self._nprod_fn(x, i, B) for i in range(self._wt[x[0][0]] + x[0][1])]
            self._E.set_splits(w, [self._el(d) for d in prods])
            todo.append(B)
            for d in prods:
                todo.extend(d)

    def _matrix(self, ents, r, c):
        """
        A Sage matrix from the engine's entries (``None`` if zero).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or E._matrix([((0, 1), [((), 1, 3)])], 1, 2) == matrix(QQ, [[0, 1/3]])
            True
        """
        if not ents:
            return None
        R = self._R
        A = matrix(R, r, c)
        for (i, j), t in ents:
            A[i, j] = sum(R(n) / R(d) for _, n, d in t)
        return A

    def set_window(self, dims, modes):
        """
        As :meth:`_PythonLowering.set_window`.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or (E.set_window([1, 1], {}), E.dims())[1] == [1, 1]
            True
        """
        out = {}
        for k, A in modes.items():
            if A is None:
                continue
            ents = [((i, j), self._coef(c)) for (i, j), c in A.dict().items() if c]
            if ents:
                out[k] = ents
        self._E.set_window(list(dims), out)

    def depth(self):
        """
        The largest known depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: mod is None or _EngineLowering(mod, QQ, [2], V._nprod_mono).depth() == -1
            True
        """
        return self._E.depth()

    def dims(self):
        """
        The dimensions by depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: mod is None or _EngineLowering(mod, QQ, [2], V._nprod_mono).dims() == []
            True
        """
        return self._E.dims()

    def basis_word(self, d, i):
        """
        ``(modes, u)``: basis vector ``i`` of depth ``d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or (E.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])}), E.extend(), E.basis_word(1, 0))[2] == (((0, -1),), 0)
            True
        """
        return self._E.basis_word(d, i)

    def extend(self, nthreads=1):
        """
        Add one depth (in ``nthreads`` threads); return its dimension.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or (E.set_window([1], {(0, 1, 0): matrix(QQ, [[0]])}), E.extend(2), E.extend())[1:] == (0, 1)
            True
        """
        return self._E.extend(nthreads)

    def word_matrix(self, key, nn, e):
        """
        As :meth:`_PythonLowering.word_matrix`.

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _EngineLowering
            sage: from vertex_algebras.vertex_algebra import _compiled_module
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: mod, _ = _compiled_module('_lowering_engine')
            sage: E = mod and _EngineLowering(mod, QQ, [2], V._nprod_mono)
            sage: E is None or (E.set_window([1], {(0, 1, 0): matrix(QQ, [[1/16]])}),
            ....:               E.word_matrix(((0, 0), (0, 0)), 3, 0))[1] == matrix(QQ, [[33/256]])
            True
        """
        self._close_splits([key])
        t = e + sum(self._wt[g] + k for g, k in key) - nn - 1
        dims = self._E.dims()
        if e < 0 or e >= len(dims) or t >= len(dims):
            raise ValueError('a mode outside the known depths')
        if t < 0:
            return None
        return self._matrix(self._E.word_matrix(key, nn, e), dims[t], dims[e])


class _TopQuotientKeys:
    """
    The basis keys ``(modes, u)`` of a top quotient: the lowering words of
    its basis (``modes`` physics modes ``(i, m)``, the first outermost;
    ``u`` the index of a window vector).  Membership extends the module to
    the key's depth.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
        sage: L = V.verma_module(1/16).top_quotient()
        sage: (((0, -2),), 0) in L._indices, (((0, -1), (0, -1)), 0) in L._indices, 'x' in L._indices
        (True, False, False)
    """

    def __init__(self, L):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.top_quotient_module import _TopQuotientKeys
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient(); _TopQuotientKeys(L)._L is L
            True
        """
        self._L = L

    def __contains__(self, x):
        """
        Whether ``x`` is a basis key.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(0).top_quotient()
            sage: ((), 0) in L._indices, (((0, -1),), 0) in L._indices, ((), 1) in L._indices
            (True, False, False)
        """
        L = self._L
        if not (isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], tuple)):
            return False
        try:
            d = L._key_depth(x)
        except (TypeError, ValueError, IndexError):
            return False
        L._extend_to(d)
        return x in L._index

    def __repr__(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(0).top_quotient()._indices
            Set of elements of Lowering words of the basis of Top quotient of ...
        """
        return 'Lowering words of the basis of {}'.format(self._L)


class TopQuotientModule(VertexAlgebraModule):
    r"""
    The top quotient ``L = M / J`` of an induced module ``M``: ``J`` the
    largest submodule meeting the degrees ``0, ..., n`` of ``M`` (the
    window; ``n`` the level, ``0`` for Zhu's induced modules) trivially;
    ``M.top_quotient()``.  See the module documentation.

    INPUT:

    - ``M`` -- an induced module: ``V.induced_module(U)`` (also Verma
      modules), ``Q.induced_module_up_to(U, max_weight, max_depth)`` for a
      quotient ``Q`` (``max_depth`` at least the level), or a module induced
      from a higher Zhu algebra
    - ``implementation`` -- (default: ``None``) ``None`` (the C++ engine
      when it takes the base ring: `\QQ`, `\GF{p}`), ``'engine'`` or
      ``'python'``

    EXAMPLES:

    The Ising model (Virasoro at `c = 1/2`): the simple modules are the top
    quotients of the Verma modules::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); T = V.gen('L')
        sage: L = V.verma_module(1/16).top_quotient(); L
        Top quotient of Verma module of highest weight {'L': 1/16} over ...
        sage: [L.weight_space(1/16 + d).dimension() for d in range(10)]
        [1, 1, 1, 2, 2, 3, 4, 5, 6, 8]
        sage: L.character(5)
        q^(1/16) + q^(17/16) + q^(33/16) + 2*q^(49/16) + 2*q^(65/16) + O(q^5)
        sage: L.weight_space(1/16 + 2).basis()
        [L_{-2} v]
        sage: v = L.highest_weight_vector(); T.mode(T.mode(v, -1), -1)
        3/4*L_{-2} v
        sage: L0 = V.verma_module(0).top_quotient(); L0.weight_space(1).dimension()
        0
        sage: [L0.weight_space(d).dimension() for d in range(11)]
        [1, 0, 1, 1, 2, 2, 3, 3, 5, 5, 7]

    A quotient vertex algebra: Lee--Yang (`c = -22/5`), from the module of
    its Zhu algebra (the relations need not be used beyond the top)::

        sage: W = va.Virasoro(QQ, -22/5, conformal_vector='L'); Lw = W.gen('L')
        sage: Q = W.quotient([Lw.no(Lw) - 3/10*Lw.derivative(2)])
        sage: U = Q.zhu_algebra_up_to(4).module({'L': -1/5})
        sage: LY = Q.induced_module_up_to(U, 4, 0).top_quotient()
        sage: [LY.weight_space(-1/5 + d).dimension() for d in range(11)]
        [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6]
        sage: LY.vertex_algebra() is Q
        True

    Over a function field (the pure-Python backend): generic `c, h`, so the
    top quotient is the Verma module::

        sage: K.<c, h> = Frac(QQ['c, h'])
        sage: Vc = va.Virasoro(K, c, conformal_vector='L')
        sage: Lc = Vc.verma_module(h).top_quotient(); Lc._backend_name()
        'python'
        sage: [Lc.weight_space(h + d).dimension() for d in range(6)]
        [1, 1, 2, 3, 5, 7]

    TESTS:

    The engine and Python give the same basis and the same products, also
    over `\GF{p}`, also in two threads::

        sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
        sage: def data(L, D):
        ....:     out = []
        ....:     for d in range(D + 1):
        ....:         for x in L.weight_space(L.top_weight() + d).basis():
        ....:             out.append((x.monomial_coefficients(),
        ....:                         [T.mode(x, m).monomial_coefficients() for m in range(-2, 3)]))
        ....:     return out
        sage: Lp = V.verma_module(1/16).top_quotient(implementation='python')
        sage: Lp._backend_name(), data(Lp, 6) == data(L, 6)
        ('python', True)
        sage: VertexAlgebraParallelism().set(2)
        sage: L3 = V.verma_module(1/2).top_quotient()
        sage: d3 = [L3.weight_space(1/2 + d).dimension() for d in range(10)]
        sage: VertexAlgebraParallelism().reset()
        sage: S3 = V.verma_module(1/2).simple_quotient()
        sage: d3 == [S3.weight_space(1/2 + d).dimension() for d in range(10)]
        True

    The category tests::

        sage: TestSuite(L).run()

    Projective covers.  The triplet algebra ``W(2)`` (from its
    presentation): ``P(0)`` is the top quotient of the module induced from
    the Jordan block of ``[T]`` at 0 (its socle lies in its top), the same
    as the library's projective cover (``M_n(A_n e)`` from the relations)
    where both are computed; ``P(1)`` is that of the degrees 0, 1 of the
    library's level-1 projective cover (a window), and continues as ``2
    W(0) + 2 W(1)`` (Loewy layers ``W(1) | 2 W(0) | W(1)``)::

        sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
        sage: Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
        sage: SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp,
        ....:                             horizontal_grading={'cp': 1, 'cm': -1})
        sage: a, b = SF.gens()
        sage: W2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
        ....:     a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
        ....:     names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient', conformal_vector='ambient')
        sage: P = W2.presentation(); Z = P.zhu_algebra_up_to(8)
        sage: z = matrix(QQ, 2)
        sage: P0 = P.induced_module_up_to(Z.module({'T': matrix([[0, 1], [0, 0]]), 'Wp': z,
        ....:                                       'W0': z, 'Wm': z}), 8, 0).top_quotient()
        sage: [P0.weight_space(d).dimension() for d in range(9)]
        [2, 4, 6, 12, 18, 28, 44, 64, 92]
        sage: W0 = P.induced_module_up_to(Z.module({'T': 0, 'Wp': 0, 'W0': 0, 'Wm': 0}), 8, 4).simple_quotient()
        sage: C0 = W0.projective_cover(8, 4).domain()
        sage: [C0.weight_space(d).dimension() for d in range(5)]
        [2, 4, 6, 12, 18]
        sage: W1 = P.induced_module_up_to(Z.module({'T': identity_matrix(QQ, 2),  # long time
        ....:     'Wp': matrix([[0, 1], [0, 0]]), 'W0': diagonal_matrix([2, -2]),
        ....:     'Wm': matrix([[0, 0], [-4, 0]])}), 8, 0).simple_quotient()
        sage: C1 = W1.projective_cover(8, 1).domain(); C1.max_depth()               # long time
        1
        sage: P1 = C1.top_quotient(); P1.level(), len(P1.window())                  # long time
        (1, 6)
        sage: [P1.weight_space(d).dimension() for d in range(9)]                    # long time
        [2, 4, 6, 12, 18, 28, 44, 64, 92]

    Refused: odd generators, an unknown implementation, a module that is
    not induced::

        sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L')
        sage: NS.verma_module(0).top_quotient()
        Traceback (most recent call last):
        ...
        NotImplementedError: top quotients need even strong generators of positive integer weight
        sage: V.verma_module(0).top_quotient(implementation='bad')
        Traceback (most recent call last):
        ...
        ValueError: implementation must be None, 'engine' or 'python', not 'bad'
        sage: from vertex_algebras.top_quotient_module import TopQuotientModule
        sage: TopQuotientModule(V.adjoint_module())
        Traceback (most recent call last):
        ...
        TypeError: ... is not an induced module
    """

    @staticmethod
    def __classcall_private__(cls, M, implementation=None):
        """
        Check ``implementation`` (unique representation on both arguments).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.top_quotient() is M.top_quotient(), M.top_quotient(implementation='python') is M.top_quotient()
            (True, False)
        """
        if implementation not in (None, 'engine', 'python'):
            raise ValueError("implementation must be None, 'engine' or 'python', not "
                             "{!r}".format(implementation))
        return VertexAlgebraModule.__classcall__(cls, M, implementation)

    def __init__(self, M, implementation=None):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: L = M.top_quotient(); L.induced_module() is M, L.level(), L.vertex_algebra() is V
            (True, 0, True)
        """
        from .vertex_algebra_module import InducedModule
        C = getattr(M, '_cover_module', M)
        if not isinstance(C, InducedModule):
            raise TypeError('{} is not an induced module'.format(M))
        if any(x not in ZZ or x <= 0 for x in C._h) or any(C._odd):
            raise NotImplementedError('top quotients need even strong generators of '
                                      'positive integer weight')
        if not C._U._is_finite_top:
            raise NotImplementedError('top quotients need a finite-dimensional top level')
        self._M = M
        self._C = C
        self._implementation = implementation
        self._wt = [ZZ(x) for x in C._h]
        self._n = ZZ(getattr(C, '_n', 0))
        # the window: the degrees 0 .. n of M, from the first nonzero one
        R = M.base_ring()
        base = M._weight_base()
        keys = [list(M.weight_space(base + d).basis_keys()) for d in range(self._n + 1)]
        d0 = 0
        while d0 <= self._n and not keys[d0]:
            d0 += 1
        if d0 > self._n:
            raise ValueError('the window of {} is zero'.format(M))
        keys = keys[d0:]
        self._base = None
        self._wbase = base + d0
        self._window_keys = [k for ks in keys for k in ks]
        self._wdepth = [d for d, ks in enumerate(keys) for _ in ks]
        pos = [{k: i for i, k in enumerate(ks)} for ks in keys]
        n0 = len(keys) - 1
        modes = {}
        one = R.one()
        for g in range(len(self._wt)):
            for e in range(n0 + 1):
                for t in range(n0 + 1):
                    nn = e + self._wt[g] - 1 - t
                    A = matrix(R, len(keys[t]), len(keys[e]))
                    for j, k in enumerate(keys[e]):
                        img = M._action_raw({((g, 0),): one}, nn, {k: one})
                        for kk, c in img.items():
                            if not c:
                                continue
                            if kk not in pos[t]:
                                raise RuntimeError('{} is not a basis key of degree {} of {}'
                                                   .format(kk, d0 + t, M))
                            A[pos[t][kk], j] = c
                    modes[(g, nn, e)] = A
        self._backend = self._make_backend(R)
        self._backend.set_window([len(ks) for ks in keys], modes)
        self._index = {}
        self._basis = []
        self._wm = {}
        self._register()
        VertexAlgebraModule.__init__(self, M.vertex_algebra(), _TopQuotientKeys(self),
                                     super=False, graded=True)

    def _make_backend(self, R):
        """
        The engine (``None``/``'engine'`` when it takes ``R``) or the Python
        backend.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ['c'].fraction_field(), 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient(implementation='engine')
            Traceback (most recent call last):
            ...
            ValueError: implementation='engine': the lowering engine is not available for ...
        """
        from .vertex_algebra import _compiled_module
        Vu = self._C.vertex_algebra()
        nprod = Vu._nprod_mono
        if self._implementation != 'python':
            mod, _ = _compiled_module('_lowering_engine')
            if mod is not None:
                try:
                    return _EngineLowering(mod, R, self._wt, nprod)
                except TypeError:
                    pass
            if self._implementation == 'engine':
                raise ValueError("implementation='engine': the lowering engine is not "
                                 "available for {} (not compiled, or a base ring other "
                                 "than QQ and GF(p))".format(self._M))
        return _PythonLowering(R, self._wt, nprod)

    def _backend_name(self):
        """
        ``'engine'`` or ``'python'``.

        TESTS::

            sage: import vertex_algebras as va, os
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(0).top_quotient()._backend_name() == ('python' if os.environ.get('VERTEX_ALGEBRAS_NO_ENGINE') else 'engine')
            True
        """
        return 'engine' if isinstance(self._backend, _EngineLowering) else 'python'

    def _register(self):
        """
        Index the basis words of the depths known to the backend.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient(); L._extend_to(2); L._basis[:3]
            [[((), 0)], [(((0, -1),), 0)], [(((0, -2),), 0)]]
        """
        B = self._backend
        dims = B.dims()
        for d in range(len(self._basis), len(dims)):
            ks = []
            for i in range(dims[d]):
                k = B.basis_word(d, i)
                self._index[k] = (d, i)
                ks.append(k)
            self._basis.append(ks)

    def _extend_to(self, d):
        """
        Compute the depths up to ``d`` (threads from
        :class:`~vertex_algebras.vertex_algebra.VertexAlgebraParallelism`).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient(); L._extend_to(4); L._backend.dims()[:5]
            [1, 1, 1, 2, 2]
        """
        from .vertex_algebra import _nproc
        B = self._backend
        while B.depth() < d:
            B.extend(_nproc())
        self._register()

    # ------------------------------------------------------------------ #
    #  Data                                                                #
    # ------------------------------------------------------------------ #

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: repr(V.verma_module(0).top_quotient()).startswith('Top quotient of Verma module')
            True
        """
        return 'Top quotient of {}'.format(self._M)

    def induced_module(self):
        """
        The induced module this is a quotient of.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(0)
            sage: M.top_quotient().induced_module() is M
            True
        """
        return self._M

    def level(self):
        """
        The level ``n``: the window is the degrees ``0, ..., n`` (``0`` for
        a module induced from Zhu's algebra).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(0).top_quotient().level()
            0
        """
        return self._n

    def window(self):
        """
        The basis of the window, as vectors of :meth:`induced_module` (the
        index ``u`` of a basis key ``(modes, u)`` refers to this list).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient().window()
            [v]
        """
        M = self._M
        return [M.monomial(k) for k in self._window_keys]

    def top_level(self):
        """
        The module ``U`` this is induced from.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(0)
            sage: M.top_quotient().top_level() is M.top_level()
            True
        """
        return self._M.top_level()

    def top_weight(self):
        """
        The conformal weight of depth 0 (the lowest degree of the window).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient().top_weight()
            1/16
        """
        return self._wbase

    def _weight_base(self):
        """
        The weight of depth 0.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._weight_base()
            1/16
        """
        return self._wbase

    def highest_weight_vector(self):
        r"""
        The image of the induced module's highest weight vector (a vector
        of ``U``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient().highest_weight_vector()
            v
        """
        k = next(iter(self._M.highest_weight_vector().support()))
        if k in self._window_keys:
            return self.monomial(((), self._window_keys.index(k)))
        return self.monomial(((), 0))

    def _l0_grading(self):
        """
        As for the induced module: ``[T]`` a scalar on the top.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._l0_grading()
            True
        """
        return self._C._l0_grading()

    def _generated_by_lowest_weight_space(self):
        """
        True for level 0 (generated by its top); else not known.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._generated_by_lowest_weight_space()
            True
        """
        return self._n == 0

    def _key_order(self, key):
        """
        ``(depth, number of modes, modes, window index)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient()
            sage: L._key_order((((0, -2),), 0)) < L._key_order((((0, -1), (0, -1)), 0))
            True
        """
        return (self._key_depth(key), len(key[0]), key[0], key[1])

    def _key_depth(self, key):
        """
        The depth of a key: that of its window vector minus its modes.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._key_depth((((0, -2), (0, -1)), 0))
            3
        """
        modes, u = key
        return self._wdepth[u] - sum(ZZ(m) for _, m in modes)

    def _repr_term(self, key):
        """
        The modes, then the window vector (as in the induced module).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._repr_term((((0, -2), (0, -1)), 0))
            'L_{-2} L_{-1} v'
        """
        modes, u = key
        names = self._C._V._gens
        w = self._M._repr_term(self._window_keys[u])
        if modes and ' ' in w:
            w = '(' + w + ')'
        return ' '.join(['{}_{{{}}}'.format(names[i], m) for i, m in modes] + [w])

    def _weight_key(self, key):
        """
        The weight of depth 0 plus the depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._weight_key((((0, -1),), 0))
            17/16
        """
        return self._wbase + self._key_depth(key)

    def _weight_bounds(self, G):
        """
        ``(h, D)``: the weight of depth 0 and the denominator of the
        weights.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._weight_bounds(None)
            (1/16, 16)
        """
        try:
            h = QQ(self._wbase)
        except (TypeError, ValueError):
            raise ValueError('the top weight {} is not rational: no character (the '
                             'weight spaces h + d are available)'.format(self._wbase))
        return h, int(h.denominator())

    def _nmax_key(self, a, key):
        """
        ``a_(n) v = 0`` once ``depth(v) + wt a - n - 1`` is negative.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient()._nmax_key(((0, 0),), (((0, -2),), 0))
            3
        """
        return sum(self._wt[g] + k for g, k in a) + self._key_depth(key) - 1

    def _weight_space_enumeration(self, W, G):
        """
        The basis keys of weight ``W``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient()
            sage: sorted(L._weight_space_enumeration(1/16 + 3, None).factory())
            [(((0, -3),), 0), (((0, -2), (0, -1)), 0)]
            sage: L._weight_space_enumeration(1/16, (1,))
            Traceback (most recent call last):
            ...
            NotImplementedError: top quotients are not graded by an abelian group yet
        """
        from .weight_spaces import _KeyEnumeration
        if G is not None:
            raise NotImplementedError('top quotients are not graded by an abelian group yet')
        try:
            d = QQ(W - self._wbase)
        except (TypeError, ValueError):
            d = QQ(-1)
        if d not in ZZ or d < 0:
            keys = []
        else:
            self._extend_to(ZZ(d))
            keys = list(self._basis[ZZ(d)])
        return _KeyEnumeration(True, factory=lambda: iter(keys))

    def weight_space(self, weight, horizontal_grade=None):
        r"""
        The span of the basis vectors of conformal weight ``weight`` (the
        weight of depth 0 plus a depth; it may be symbolic).

        INPUT:

        - ``weight`` -- a conformal weight
        - ``horizontal_grade`` -- (default: ``None``) not supported yet

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/2).top_quotient().weight_space(1/2 + 3).basis()
            [L_{-3} v]
        """
        from .weight_spaces import WeightSpace
        return WeightSpace(self, weight, horizontal_grade,
                           self._weight_space_enumeration(weight, horizontal_grade))

    def degree(self, x):
        r"""
        The conformal weight of a vector homogeneous for it.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); T = V.gen('L')
            sage: L = V.verma_module(1/16).top_quotient(); v = L.highest_weight_vector()
            sage: L.degree(T.mode(v, -3))
            49/16
            sage: L.degree(v + T.mode(v, -1))
            Traceback (most recent call last):
            ...
            ValueError: v + L_{-1} v is not homogeneous of a single conformal weight
        """
        x = self(x)
        ws = {self._weight_key(m) for m in x.support()}
        if len(ws) != 1:
            raise ValueError('{} is not homogeneous of a single conformal weight'.format(x))
        return ws.pop()

    # ------------------------------------------------------------------ #
    #  The action                                                          #
    # ------------------------------------------------------------------ #

    def _word_columns(self, a, nn, e):
        """
        The mode ``(nn)`` of the word ``a`` on depth ``e``, as a dict
        ``column -> {row: coefficient}`` (cached).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient(); L._extend_to(1)
            sage: L._word_columns(((0, 0),), 2, 1)
            {0: {0: 1/8}}
        """
        k = (a, nn, e)
        r = self._wm.get(k)
        if r is None:
            A = self._backend.word_matrix(a, nn, e)
            r = {}
            if A is not None:
                for (i, j), c in A.dict().items():
                    if c:
                        r.setdefault(j, {})[i] = c
            self._wm[k] = r
        return r

    def _action_key(self, a, n, key):
        """
        ``a_(n) key`` for a word ``a`` of the universal algebra, as a dict.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: L = V.verma_module(1/16).top_quotient()
            sage: L._action_key(((0, 0),), 0, ((), 0)), L._action_key(((0, 0),), 1, ((), 0))
            ({(((0, -1),), 0): 1}, {((), 0): 1/16})
            sage: L._action_key(((0, 0), (0, 0)), 5, ((), 0))
            {}
        """
        e = self._key_depth(key)
        t = e + sum(self._wt[g] + k for g, k in a) - n - 1
        if t < 0:
            return {}
        self._extend_to(max(e, t))
        col = self._word_columns(a, n, e).get(self._index[key][1], {})
        ks = self._basis[t]
        return {ks[i]: c for i, c in col.items()}

    def clear_caches(self):
        r"""
        Forget the computed products (the backend keeps its depths).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); T = V.gen('L')
            sage: L = V.verma_module(1/16).top_quotient(); v = L.highest_weight_vector()
            sage: x = T.mode(v, -2); L.clear_caches(); T.mode(v, -2) == x
            True
        """
        VertexAlgebraModule.clear_caches(self)
        self._wm = {}

    def some_elements(self):
        r"""
        Some vectors: the window's basis (up to three) and a vector of the
        next depth.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient().some_elements()
            [v, L_{-1} v]
        """
        out = [self.monomial(((), u)) for u in range(min(3, len(self._window_keys)))]
        D = self._wdepth[-1] + 1
        self._extend_to(D)
        if self._basis[D]:
            out.append(self.monomial(self._basis[D][0]))
        return out

    def _an_element_(self):
        """
        The highest weight vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: V.verma_module(1/16).top_quotient().an_element()
            v
        """
        return self.highest_weight_vector()
