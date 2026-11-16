# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
The structure of a finite-dimensional associative algebra with unit over a
field of characteristic 0, from the matrices of left multiplication by its
basis: radical, primitive idempotents, Cartan invariants, projective
indecomposable and simple modules.

Used by the finite-dimensional Zhu algebras (``higher_zhu_algebra``); the
algebra itself only provides the matrices (and, if graded, the grades of
its basis).

Conventions: the basis is ``b_0, ..., b_{n-1}``, elements are vectors of
coordinates, ``lmats[i]`` is the matrix of ``x -> b_i x`` (column ``j`` =
the coordinates of ``b_i b_j``).

Math (Curtis-Reiner, Section 54; Benson, *Representations and cohomology*
I, 1.7): in characteristic 0 the radical is the kernel of the trace
form ``(a, b) -> tr(L_{ab})`` (Dickson).  An idempotent ``e`` is primitive
iff ``eAe`` is local; when ``A`` is split over the base field (every simple
module absolutely simple) this means ``dim e(A/J)e = 1``, which is the test
used here, so primitivity is certified whenever the algebra is split; an
idempotent that is not primitive is split by the primary decomposition of
the minimal polynomial of some ``x`` in ``eAe`` (the polynomials in ``x``
with unit ``e`` that are ``1`` modulo one primary factor and ``0`` modulo the
others).  Idempotents ``e, f`` are equivalent (``Ae ≅ Af``) iff ``e(A/J)f
≠ 0``.  The projective indecomposable modules are the ``Ae`` (one per class),
their heads ``Ae/Je`` the simple modules, and the Cartan invariants are
``dim e_i A e_j``.  For an algebra graded by a free abelian group (the
grades of a Zhu algebra) elements of degree 0 are tried first: the
idempotents are then homogeneous of degree 0, the radical is graded, and
``Ae``, ``Je`` have homogeneous bases.

EXAMPLES:

The upper triangular 2 x 2 matrices (basis ``E11, E12, E22``): two simple
modules of dimension 1, projective covers of dimensions 1 and 2::

    sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
    sage: def L(i):          # left multiplication by E11, E12, E22 on (E11, E12, E22)
    ....:     return [matrix(QQ, [[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
    ....:             matrix(QQ, [[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    ....:             matrix(QQ, [[0, 0, 0], [0, 0, 0], [0, 0, 1]])][i]
    sage: F = FiniteDimensionalStructure(QQ, [L(i) for i in range(3)], vector(QQ, [1, 0, 1]))
    sage: F.radical()
    [0 1 0]
    sage: F.primitive_idempotents()
    [(1, 0, 0), (0, 0, 1)]
    sage: F.cartan_invariants_matrix()      # E11 A E22 = span(E12)
    [1 1]
    [0 1]
    sage: [F.projective(e)[0].nrows() for e in F.primitive_idempotents()]
    [1, 2]

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ

#: How many combinations of basis elements are tried to split an
#: idempotent that is not primitive (after the basis elements themselves).
SPLIT_TRIES = 40


class FiniteDimensionalStructure:
    r"""
    The structure of the algebra with left multiplication matrices
    ``lmats`` and unit ``one`` over the field ``R`` (characteristic 0);
    ``grades`` (optional) lists a hashable grade per basis element, with
    the zero grade ``zero``, for a grading by a free abelian group.

    INPUT:

    - ``R`` -- a field of characteristic 0
    - ``lmats`` -- a list of ``n x n`` matrices over ``R``: ``lmats[i]`` is
      the matrix of ``x -> b_i x`` (column ``j`` = the coordinates of
      ``b_i b_j``)
    - ``one`` -- a vector over ``R`` of length ``n``: the coordinates of the
      unit
    - ``grades`` -- (default: ``None``) a list of hashable grades, one per
      basis element, or ``None`` if ungraded
    - ``zero`` -- (default: ``None``) the zero grade (used with ``grades``)

    EXAMPLES::

        sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
        sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 1)], vector(QQ, [1]))
        sage: F.radical().nrows(), F.primitive_idempotents()
        (0, [(1)])
    """

    def __init__(self, R, lmats, one, grades=None, zero=None):
        r"""
        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]), grades=[0, 1], zero=0)
            sage: F.n, F.grades, F.zero
            (2, [0, 1], 0)
        """
        self.R = R
        self.L = list(lmats)
        self.n = len(self.L)
        self.one = one
        self.grades = grades
        self.zero = zero
        self._J = None

    def lmat(self, a):
        r"""
        The matrix of left multiplication by ``a``.

        INPUT:

        - ``a`` -- a vector of coordinates

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 1)], vector(QQ, [1]))
            sage: F.lmat(vector(QQ, [3]))
            [3]
        """
        M = matrix(self.R, self.n, self.n)
        for i, c in a.dict().items():
            M += c * self.L[i]
        return M

    def mul(self, a, b):
        r"""
        The product ``a b``.

        INPUT:

        - ``a``, ``b`` -- vectors of coordinates

        OUTPUT:

        The vector of coordinates of ``a b``.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 1)], vector(QQ, [1]))
            sage: F.mul(vector(QQ, [2]), vector(QQ, [3]))
            (6)
        """
        return self.lmat(a) * b

    def _basis(self, i):
        r"""
        The coordinate vector of the basis element ``b_i``.

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F._basis(1)
            (0, 1)
        """
        v = self.one.parent().zero_vector()
        v[i] = 1
        return v

    def radical(self):
        r"""
        A basis (rows, echelon form) of the radical: the kernel of the trace
        form ``(a, b) -> tr(L_a L_b)``; homogeneous rows when graded (the
        radical of an algebra graded by a free abelian group is graded).

        OUTPUT:

        A matrix over ``R`` whose rows are a basis of the radical.

        EXAMPLES:

        The dual numbers ``QQ[x]/x^2``::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F.radical()
            [0 1]
        """
        if self._J is not None:
            return self._J
        n = self.n
        G = matrix(self.R, n, n, lambda i, j: (self.L[i] * self.L[j]).trace())
        K = G.right_kernel_matrix()
        if self.grades is not None and K.nrows():
            rows = []
            for g in sorted(set(self.grades)):
                P = matrix(self.R, [[r[i] if self.grades[i] == g else 0 for i in range(n)]
                                    for r in K.rows()])
                rows += P.row_space().basis_matrix().rows()
            K = matrix(self.R, rows, ncols=n).echelon_form()
        self._J = K
        return self._J

    def _dim_mod_radical(self, vectors):
        r"""
        The dimension of the span of ``vectors`` modulo the radical.

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F._dim_mod_radical([vector(QQ, [1, 1]), vector(QQ, [0, 1])])
            1
            sage: F._dim_mod_radical([])
            0
        """
        J = self.radical()
        vs = list(vectors)
        if not vs:
            return 0
        M = matrix(self.R, vs).stack(J) if J.nrows() else matrix(self.R, vs)
        return M.rank() - J.nrows()

    def _corner(self, e, f=None):
        r"""
        Vectors spanning ``eAf`` (``f = e`` by default): the ``e b_i f``.

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F._corner(F.one)
            [(1, 0), (0, 1)]
        """
        f = e if f is None else f
        Le = self.lmat(e)
        return [Le * self.mul(self._basis(i), f) for i in range(self.n)]

    def _split(self, x, e):
        r"""
        Orthogonal idempotents (sum ``e``) from the primary decomposition of
        the minimal polynomial of ``x`` in ``eAe`` (unit ``e``); ``[e]`` if
        it is a power of one irreducible polynomial.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 1]])],
            ....:                                vector(QQ, [1, 0]))      # basis 1, f with f^2 = f
            sage: F._split(vector(QQ, [0, 1]), F.one)
            [(0, 1), (1, -1)]
        """
        from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
        X = PolynomialRing(self.R, 'x').gen()
        pw = [e]
        Lx = self.lmat(x)
        while True:
            pw.append(Lx * pw[-1])
            K = matrix(self.R, pw).left_kernel_matrix()
            if K.nrows():
                m = sum((c * X ** i for i, c in enumerate(K.row(-1))), X.parent().zero())
                m = m.monic()
                break
        fac = [p ** k for p, k in m.factor()]
        if len(fac) < 2:
            return [e]
        out = []
        for f in fac:
            g = m // f
            _, s, _ = g.xgcd(f)
            q = (g * s) % m
            v = e.parent().zero_vector()
            p = e
            for c in q.list():
                v += c * p
                p = Lx * p
            out.append(v)
        return out

    def _candidates(self, e, homogeneous):
        r"""
        Elements of ``eAe`` to split ``e`` with: ``e b_i e`` for the basis
        elements (of degree 0 if ``homogeneous``), then fixed combinations.

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: from itertools import islice
            sage: list(islice(F._candidates(F.one, False), 3))
            [(1, 0), (0, 1), (-2, 1)]
        """
        idx = [i for i in range(self.n)
               if not homogeneous or self.grades[i] == self.zero]
        Le = self.lmat(e)
        for i in idx:
            yield Le * self.mul(self._basis(i), e)
        for t in range(SPLIT_TRIES):
            v = self.one.parent().zero_vector()
            for k, i in enumerate(idx):
                v[i] = ((7 * t + 3) * (k + 1) + t * t) % 11 - 5
            yield Le * self.mul(v, e)

    def primitive_idempotents(self):
        r"""
        A complete set of orthogonal primitive idempotents (summing to
        the unit); of degree 0 when graded, if the elements of degree 0
        suffice to split.

        OUTPUT:

        A list of coordinate vectors.

        .. WARNING::

            Raises ``NotImplementedError`` if some ``e`` with ``dim
            e(A/J)e > 1`` cannot be split (the algebra may not be split
            over the base field).

        EXAMPLES:

        ``2 x 2`` matrices, with the matrix units as basis::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: units = [(0, 0), (0, 1), (1, 0), (1, 1)]
            sage: def L(a):
            ....:     M = matrix(QQ, 4, 4)
            ....:     for j, b in enumerate(units):
            ....:         if a[1] == b[0]:
            ....:             M[units.index((a[0], b[1])), j] = 1
            ....:     return M
            sage: F = FiniteDimensionalStructure(QQ, [L(a) for a in units], vector(QQ, [1, 0, 0, 1]))
            sage: F.primitive_idempotents()
            [(1, 0, 0, 0), (0, 0, 0, 1)]
            sage: F.cartan_invariants_matrix()
            [1]

        The quaternions over ``QQ`` (not split): refused::

            sage: Q.<i, j, k> = QuaternionAlgebra(QQ, -1, -1)
            sage: B = [Q(1), i, j, k]
            sage: Ls = [matrix(QQ, [list(b * c) for c in B]).transpose() for b in B]
            sage: FiniteDimensionalStructure(QQ, Ls, vector(QQ, [1, 0, 0, 0])).primitive_idempotents()
            Traceback (most recent call last):
            ...
            NotImplementedError: an idempotent with dim e(A/J)e = 4 could not be split: the algebra may not be split over Rational Field
        """
        try:
            return self._idem
        except AttributeError:
            pass
        graded = self.grades is not None
        done, todo = [], [self.one]
        while todo:
            e = todo.pop(0)
            d = self._dim_mod_radical(self._corner(e))
            if d == 1:
                done.append(e)
                continue
            if d == 0:
                continue          # e in J: cannot happen for an idempotent e != 0
            parts = None
            for homogeneous in ((True, False) if graded else (False,)):
                for x in self._candidates(e, homogeneous):
                    s = self._split(x, e)
                    if len(s) > 1:
                        parts = s
                        break
                if parts is not None:
                    break
            if parts is None:
                raise NotImplementedError('an idempotent with dim e(A/J)e = {} could not be '
                                          'split: the algebra may not be split over {}'
                                          .format(d, self.R))
            todo = parts + todo
        self._idem = done
        return done

    def classes(self):
        r"""
        The primitive idempotents grouped by equivalence (``Ae ≅ Af`` iff
        ``e(A/J)f ≠ 0``), in the order found: a list of lists.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 1)], vector(QQ, [1]))
            sage: F.classes()
            [[(1)]]
        """
        out = []
        for e in self.primitive_idempotents():
            for c in out:
                if self._dim_mod_radical(self._corner(c[0], e)):
                    c.append(e)
                    break
            else:
                out.append([e])
        return out

    def cartan_invariants_matrix(self):
        r"""
        The Cartan invariants ``c_{ij} = dim e_i A e_j`` (``e_i`` one
        primitive idempotent per class, :meth:`classes`): the multiplicity
        of the ``i``-th simple module in the ``j``-th projective
        indecomposable module.

        OUTPUT:

        A square matrix over ``ZZ``.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F.cartan_invariants_matrix()
            [2]
        """
        es = [c[0] for c in self.classes()]
        return matrix(ZZ, len(es), len(es),
                      lambda i, j: matrix(self.R, self._corner(es[i], es[j])).rank())

    def _graded_basis(self, vectors):
        r"""
        A basis (rows) of the span of ``vectors`` and the grades of its rows:
        homogeneous rows grade by grade if graded (``vectors`` homogeneous),
        else the echelon form and ``None``.

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]), grades=[0, 1], zero=0)
            sage: F._graded_basis([vector(QQ, [2, 0]), vector(QQ, [0, 3])])
            (
            [1 0]
            [0 1], [0, 1]
            )
            sage: F._graded_basis([vector(QQ, [1, 1])])
            Traceback (most recent call last):
            ...
            ValueError: not homogeneous
        """
        vs = [v for v in vectors if v]
        if self.grades is None:
            if not vs:
                return matrix(self.R, 0, self.n), None
            return matrix(self.R, vs).row_space().basis_matrix(), None
        by = {}
        for v in vs:
            gs = {self.grades[i] for i in v.dict()}
            if len(gs) != 1:
                raise ValueError('not homogeneous')
            by.setdefault(gs.pop(), []).append(v)
        rows, grades = [], []
        for g in sorted(by):
            B = matrix(self.R, by[g]).row_space().basis_matrix()
            rows += B.rows()
            grades += [g] * B.nrows()
        return matrix(self.R, rows, ncols=self.n), grades

    def _homogeneous(self, e):
        r"""
        Whether ``e`` is homogeneous of degree 0 (False when ungraded).

        TESTS::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]), grades=[0, 1], zero=0)
            sage: F._homogeneous(F.one), F._homogeneous(vector(QQ, [1, 1]))
            (True, False)
        """
        return self.grades is not None and len({self.grades[i] for i in e.dict()}) <= 1 \
            and all(self.grades[i] == self.zero for i in e.dict())

    def projective(self, e):
        r"""
        The left ideal ``Ae``: ``(basis, grades)``, the basis as the rows
        of a matrix (homogeneous, with their grades, if ``e`` has degree 0;
        else the echelon form and grades ``None``).

        INPUT:

        - ``e`` -- a coordinate vector: an idempotent

        OUTPUT:

        A pair ``(basis, grades)``: a matrix over ``R`` whose rows are a
        basis of ``Ae``, and the list of their grades or ``None``.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]), grades=[0, 1], zero=0)
            sage: F.projective(F.one)
            (
            [1 0]
            [0 1], [0, 1]
            )
        """
        vs = [self.mul(self._basis(i), e) for i in range(self.n)]
        if self.grades is not None and not self._homogeneous(e):
            return matrix(self.R, [v for v in vs if v]).row_space().basis_matrix(), None
        return self._graded_basis(vs)

    def simple(self, e):
        r"""
        The head ``Ae/Je`` of ``Ae``: ``(basis of Ae, grades, basis of Je)``
        with ``Je`` spanned by the rows of the third matrix and the first
        matrix a basis of ``Ae`` extending it at the END (so its last rows
        give a basis of the quotient).

        INPUT:

        - ``e`` -- a coordinate vector: a primitive idempotent

        OUTPUT:

        A triple ``(B, grades, JB)``: a matrix ``B`` whose rows are a basis
        of ``Ae`` with the rows of ``JB`` (a basis of ``Je``) first, the
        list of the grades of the rows of ``B`` or ``None``, and ``JB``.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: B, g, JB = F.simple(F.one); B, JB
            (
            [0 1]
            [1 0], [0 1]
            )
        """
        J = self.radical()
        A, grades = self.projective(e)
        je = [self.mul(j, e) for j in J.rows()]
        if grades is None:
            JB = matrix(self.R, [v for v in je if v], ncols=self.n)
            JB = JB.row_space().basis_matrix() if JB.nrows() else JB
            rest = []
            M = JB
            for r in A.rows():
                M2 = M.stack(r)
                if M2.rank() > M.rank():
                    rest.append(r)
                    M = M2
            return JB.stack(matrix(self.R, rest, ncols=self.n)), None, JB
        JB, jg = self._graded_basis(je)
        rows, out_g = list(JB.rows()), list(jg)
        M = JB
        for r, g in zip(A.rows(), grades):
            M2 = M.stack(r)
            if M2.rank() > M.rank():
                rows.append(r)
                out_g.append(g)
                M = M2
        return matrix(self.R, rows, ncols=self.n), out_g, JB

    def action(self, L, basis, start=0):
        r"""
        The matrix of left multiplication ``L`` on the span of the rows of
        ``basis`` from row ``start`` on, modulo the earlier rows (which must
        span an ``L``-stable subspace): columns = coordinates of the images.

        INPUT:

        - ``L`` -- an ``n x n`` matrix over ``R``: a left multiplication
        - ``basis`` -- a matrix over ``R`` with ``n`` columns, whose rows
          are linearly independent
        - ``start`` -- (default: ``0``) the index of the first row of the
          quotient

        OUTPUT:

        A square matrix over ``R`` of size ``basis.nrows() - start``.

        EXAMPLES::

            sage: from vertex_algebras._finite_dimensional_structure import FiniteDimensionalStructure
            sage: F = FiniteDimensionalStructure(QQ, [matrix.identity(QQ, 2), matrix(QQ, [[0, 0], [1, 0]])],
            ....:                                vector(QQ, [1, 0]))
            sage: F.action(F.L[1], matrix.identity(QQ, 2))
            [0 0]
            [1 0]
        """
        imgs = (L * basis.matrix_from_rows(range(start, basis.nrows())).transpose()).transpose()
        C = basis.solve_left(imgs)
        return C.matrix_from_columns(range(start, basis.nrows())).transpose()
