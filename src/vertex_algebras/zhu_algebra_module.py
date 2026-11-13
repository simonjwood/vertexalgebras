# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Modules over Zhu algebras.

A finite-dimensional module over a Zhu algebra ``A`` (a
:class:`zhu_algebra.ZhuAlgebra`, also H-twisted, or a
:class:`zhu_algebra.ZhuAlgebraQuotient`), given by the matrices of its
generators ``[g]`` on a basis ``u_0, ..., u_{d-1}``: ``A.module(action)``.
The matrices are checked against the defining relations of ``A`` (the
supercommutators ``[x_a][x_b] - (-1)^{|a||b|}[x_b][x_a]`` and the squares
of odd generators; for a quotient also its two-sided Gröbner basis), and
for a super module against the parities of the basis vectors.  These are
the top levels of positive energy modules of the vertex algebra (Zhu's
correspondence [Zhu1996]_); see :mod:`vertex_algebra_module`.

Elements ``x`` of ``A`` act by ``x * u`` (``representation_matrix(x)``);
elements of the vertex algebra through their image ``A(v)``.

EXAMPLES:

The Virasoro algebra, ``A = C[[L]]``: the one-dimensional module of
conformal weight ``h``::

    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: E.<L> = NormallyOrderedExpressions(QQ)
    sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
    ....:                              conformal_vector=L)
    sage: A = Vir.zhu_algebra(); U = A.module({'L': -1/5}); U
    Module of dimension 1 over Zhu algebra of Universal vertex algebra on generators ['L'] over Rational Field
    sage: L = Vir.gen('L'); x = A(L.no(L)); x * U.basis()[0]
    -9/25*u[0]
    sage: U.representation_matrix(x)
    [-9/25]

The Lee-Yang model: modules of its Zhu algebra exist exactly for the two
conformal weights ``0, -1/5``::

    sage: Z = Vir.quotient([L.no(L) - 3/10*L.derivative(2)]).zhu_algebra_up_to(4)
    sage: Z.module({'L': -1/5}).dimension()
    1
    sage: Z.module({'L': 1})
    Traceback (most recent call last):
    ...
    ValueError: the matrices do not satisfy the relation [L] + 5*[L]^2 = 0 of the Zhu algebra

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import operator

from sage.categories.action import Action
from sage.categories.modules import Modules
from sage.combinat.free_module import CombinatorialFreeModule
from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .vertex_algebra import _ArtFromRepr


def _grading_spec(A, mats, d, horizontal_grades, horizontal_grading_group, embedding):
    r"""
    The grading datum ``(B, horizontal_grades, M)`` of a module over the Zhu algebra
    ``A`` (or ``None``): ``B`` a Sage group, ``horizontal_grades`` the grades of the
    basis vectors (immutable elements of ``B``), ``M`` the matrix of the
    embedding of the grading group of the vertex algebra into ``B``.
    Matrices of the wrong size give ``None`` (refused by the constructor).

    TESTS:

    The default for affine algebras: the Cartan eigenvalues, in ``P ⊗ QQ``;
    given grades in ``ZZ^1``::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.zhu_algebra_module import _grading_spec
        sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
        sage: mats = [matrix(QQ, [[0, 1], [0, 0]]), matrix(QQ, [[1, 0], [0, -1]]),
        ....:         matrix(QQ, [[0, 0], [1, 0]])]
        sage: _grading_spec(A, mats, 2, None, None, None)
        (Vector space of dimension 1 over Rational Field, ((1), (-1)), [2])
        sage: _grading_spec(A, mats, 2, [(2,), (0,)], ZZ^1, None)
        (Ambient free module of rank 1 over the principal ideal domain Integer Ring, ((2), (0)), [1])
        sage: _grading_spec(A, mats, 3, None, None, None) is None
        True
    """
    from sage.matrix.constructor import matrix
    from sage.modules.free_module import FreeModule

    from ._grading_groups import module_grading_group
    cover = A._zhu_cover() if hasattr(A, '_zhu_cover') else A
    V = cover.vertex_algebra()
    gd = V._grading
    R = V.base_ring()
    if any(X.nrows() != d or X.ncols() != d for X in mats):
        return None                     # refused (with the reason) by __init__
    if horizontal_grades is None:
        if horizontal_grading_group is not None or embedding is not None:
            raise ValueError('give the grades of the basis vectors')
        auto = getattr(V, '_cartan_weight_grading', None)
        auto = auto() if (auto is not None and gd is not None) else None
        if auto is None:
            return None
        cartan, M = auto
        zidx = {v: z for z, v in enumerate(cover._vgen)}
        if any(h not in zidx for h in cartan):
            return None
        rows = []
        for u in range(d):
            row = []
            for h in cartan:
                X = mats[zidx[h]]
                if any(X[r, u] for r in range(d) if r != u):
                    return None                 # not diagonal: not graded
                row.append(R(X[u, u]))
            rows.append(row)
        B = FreeModule(R, len(cartan))
        out = []
        for row in rows:
            v = B(row)
            v.set_immutable()
            out.append(v)
        return (B, tuple(out), M)
    if gd is None:
        raise ValueError('grades need a vertex algebra graded by an abelian group')
    AG = gd['group']
    horizontal_grades = list(horizontal_grades)
    if len(horizontal_grades) != d:
        raise ValueError('{} grades expected (one per basis vector)'.format(d))
    auto = getattr(V, '_cartan_weight_grading', None)
    auto = auto() if auto is not None else None
    if horizontal_grading_group is None and embedding is None and auto is not None:
        horizontal_grading_group = FreeModule(R, len(auto[0]))         # weights: P ⊗ R
        embedding = auto[1]
    if horizontal_grading_group is None:
        try:
            [AG.coords(g) for g in horizontal_grades]
            horizontal_grading_group = AG.group
        except ValueError:
            horizontal_grading_group = FreeModule(R, AG.k)
    BG = module_grading_group(horizontal_grading_group)
    if embedding is None:
        if BG.k != AG.k:
            raise ValueError('give the embedding of the grading group {} of the vertex '
                             'algebra into {}'.format(AG.group, BG.group))
        embedding = matrix.identity(BG.R if BG.kind == 'vector' else ZZ, BG.k)
    M = matrix(BG.R if BG.kind == 'vector' else ZZ, BG.k, AG.k, embedding)
    M.set_immutable()
    return (BG.group, tuple(BG.element(BG.coords(g)) for g in horizontal_grades), M)


class _ZhuAction(Action):
    """
    ``x * u`` for ``x`` in the Zhu algebra, ``u`` in the module (a
    :class:`ZhuAlgebraModule` or a :class:`ZhuVermaModule`).

    TESTS::

        sage: from vertex_algebras.zhu_algebra_module import _ZhuAction
        sage: import vertex_algebras as va
        sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
        sage: A = V.zhu_algebra()
        sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:           'F': matrix([[0, 0], [1, 0]])})
        sage: act = _ZhuAction(A, U); act
        Left action by Zhu algebra of ... on Module of dimension 2 over Zhu algebra of ...
    """

    def __init__(self, A, U):
        """
        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _ZhuAction
            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: act = _ZhuAction(A, U); act.domain() is U, act.is_left()
            (True, True)
        """
        Action.__init__(self, A, U, True, operator.mul)

    def _act_(self, x, u):
        """
        ``x`` acting on ``u``: the module's ``_act_element`` if it has one
        (Verma modules), else the representation matrix.

        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _ZhuAction
            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: _ZhuAction(A, U)._act_(A.gen(2), U.basis()[0])
            u[1]
        """
        U = self.codomain()
        if hasattr(U, '_act_element'):
            return U._act_element(x, u)
        M = U.representation_matrix(x)
        return U.from_vector(M * u.to_vector())


class ZhuAlgebraModule(_ArtFromRepr, CombinatorialFreeModule):
    r"""
    A finite-dimensional module over a Zhu algebra ``A`` (``A.module(action,
    parities)``), with basis ``u[0], ..., u[d-1]``.

    INPUT:

    - ``A`` -- a :class:`zhu_algebra.ZhuAlgebra` or
      :class:`zhu_algebra.ZhuAlgebraQuotient`
    - ``action`` -- the matrices of the generators ``A.gens()``: a dict
      ``{name: matrix}`` (all generators) or a list in the order of
      ``A.gens()``; scalars for a one-dimensional module
    - ``parities`` -- (default: ``None``, all even) the parities of the
      basis vectors, for modules over Zhu algebras with odd generators
    - ``horizontal_grades`` -- (default: ``None``) the grades of the basis
      vectors, for a vertex algebra graded by a group ``A``: elements of
      ``horizontal_grading_group``; the matrix of a generator of grade
      ``a`` must map grade ``b`` to ``b + ι(a)``.  For affine vertex
      algebras graded by the root lattice the default grades are the
      eigenvalues of the zero modes of the Cartan currents, in ``P ⊗ R``
      (fundamental weight coordinates; ``R`` the base ring), when they act
      diagonally; otherwise the default is an ungraded module
    - ``horizontal_grading_group`` -- (default: ``None``) the group ``B``
      of the grades: the weights ``P ⊗ R`` for affine vertex algebras,
      else ``A`` if the grades lie in it, else ``R^k``
    - ``embedding`` -- (default: ``None``) the matrix on coordinates of the
      embedding ``ι`` of ``A`` into ``horizontal_grading_group`` (default:
      the identity, ``A ⊂ A ⊗ R``; for affine algebras the map to the
      weights)
    - ``horizontal_grading`` -- (default: ``None``) the normalised grading
      datum; internal (used by unpickling), not to be given by users

    EXAMPLES:

    Affine ``sl_2``: ``A = U(sl_2)`` and its two-dimensional module; wrong
    matrices are refused::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1, names=['E', 'H', 'F'],
        ....:                                  conformal_vector='sugawara')
        sage: A = V.zhu_algebra()
        sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:               'F': matrix([[0, 0], [1, 0]])})
        sage: E, H, F = A.gens(); [x * U.basis()[0] for x in (E, H, F)]
        [0, u[0], u[1]]
        sage: U.representation_matrix(A(V.conformal_vector()))
        [1/4   0]
        [  0 1/4]
        sage: A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[2, 0], [0, -1]]),
        ....:           'F': matrix([[0, 0], [1, 0]])})
        Traceback (most recent call last):
        ...
        ValueError: the matrices do not satisfy the relation [H][E] - [E][H] = 2*[E] of the Zhu algebra

    The Ramond sector of the ``N = 1`` algebra (H-twisted Zhu algebra,
    ``[G]^2 = [L] - c/24``): a super module of dimension ``1|1``::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: Y.<L, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
        sage: NS = UniversalVertexAlgebra(QQ, Y, {('L', 'L'): [L.derivative(), 2*L, 0, 7/20],
        ....:     ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 7/15]}, conformal_vector=L)
        sage: AR = NS.zhu_algebra(twist='ramond')
        sage: UR = AR.module({'L': matrix([[7/16, 0], [0, 7/16]]), 'G': matrix([[0, 1], [49/120, 0]])},
        ....:                parities=[0, 1]); UR.category()
        Category of finite dimensional super modules with basis over Rational Field
        sage: AR.module({'L': matrix([[1, 0], [0, 1]]), 'G': matrix([[0, 1], [1, 0]])}, parities=[0, 1])
        Traceback (most recent call last):
        ...
        ValueError: the matrices do not satisfy the relation [G][G] = -7/240 + [L] of the Zhu algebra
        sage: AR.module({'L': matrix([[7/16, 0], [0, 7/16]]), 'G': matrix([[49/120, 0], [0, 1]])},
        ....:           parities=[0, 1])
        Traceback (most recent call last):
        ...
        ValueError: the matrix of the odd generator [G] does not change the parity

    The grades of the doublet are its ``sl_2``-weights, in ``P ⊗ QQ``::

        sage: U.horizontal_grades(), U.horizontal_grading_group()
        ([(1), (-1)], Vector space of dimension 1 over Rational Field)
        sage: A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:           'F': matrix([[0, 0], [1, 0]])}, horizontal_grades=[(1,), (1,)])
        Traceback (most recent call last):
        ...
        ValueError: the matrix of [E] maps a vector of grade (1) to one of grade (1), not (1) + ι(grade of E) = (3)

    TESTS::

        sage: U is A.module([matrix([[0, 1], [0, 0]]), matrix([[1, 0], [0, -1]]), matrix([[0, 0], [1, 0]])])
        True
        sage: loads(dumps(U)) is U, loads(dumps(UR)) is UR
        (True, True)
        sage: TestSuite(U).run(); TestSuite(UR).run()

    Modules exist exactly for the known representations.  The Ising model
    (``c = 1/2``): one-dimensional modules for ``h = 0, 1/16, 1/2`` only,
    while the universal Virasoro algebra takes any ``h``::

        sage: import vertex_algebras as va
        sage: def accepts(A, action, parities=None):
        ....:     try:
        ....:         A.module(action, parities)
        ....:         return True
        ....:     except ValueError:
        ....:         return False
        sage: Vi = va.Virasoro(QQ, 1/2, conformal_vector='L'); [v] = Vi.find_singular(6)
        sage: Zi = Vi.quotient([v]).zhu_algebra_up_to(8)
        sage: [h for h in (0, 1/16, 1/2, 1, 1/3) if accepts(Zi, {'L': h})]
        [0, 1/16, 1/2]
        sage: accepts(Vi.zhu_algebra(), {'L': 1/3})
        True

    ``U(sl_2)`` takes the doublet and the triplet, ``L_1(sl_2)`` only the
    trivial module and the doublet::

        sage: adj = {'E': matrix([[0, -2, 0], [0, 0, 1], [0, 0, 0]]),
        ....:        'H': matrix([[2, 0, 0], [0, 0, 0], [0, 0, -2]]),
        ....:        'F': matrix([[0, 0, 0], [-1, 0, 0], [0, 2, 0]])}
        sage: dbl = {'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:        'F': matrix([[0, 0], [1, 0]])}
        sage: E1 = V.gen('E'); Z1 = V.quotient([E1.no(E1)]).zhu_algebra_up_to(3)
        sage: (accepts(A, adj), accepts(Z1, {'E': 0, 'H': 0, 'F': 0}), accepts(Z1, dbl),
        ....:  accepts(Z1, adj))
        (True, True, True, False)

    The ``bc`` system (weights 1, 0): the Clifford algebra and its module
    ``C^{1|1}``; wrong scalars or parities are refused::

        sage: X.<b, c> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'c': 1})
        sage: BC = UniversalVertexAlgebra(QQ, X, {('b', 'c'): [1]}, conformal_vector=-b*c.derivative())
        sage: Ab = BC.zhu_algebra(); e, f = matrix([[0, 1], [0, 0]]), matrix([[0, 0], [1, 0]])
        sage: (accepts(Ab, {'b': e, 'c': f}, [0, 1]), accepts(Ab, {'b': e, 'c': 2*f}, [0, 1]),
        ....:  accepts(Ab, {'b': e, 'c': f}, [0, 0]))
        (True, False, False)
        sage: TestSuite(Ab.module({'b': e, 'c': f}, [0, 1])).run()

    The tricritical Ising model (``c = 7/10``): Ramond modules of dimension
    ``1|1`` only at ``h = 3/80``, ``7/16``; the universal ``N = 1`` algebra
    at any ``h``, and one-dimensional at ``h = c/24``::

        sage: [vt] = NS.find_singular(4); ZR = NS.quotient([vt]).zhu_algebra_up_to(6, twist='ramond')
        sage: ramond = lambda h: {'L': matrix([[h, 0], [0, h]]), 'G': matrix([[0, 1], [h - 7/240, 0]])}
        sage: [h for h in (3/80, 7/16, 1) if accepts(ZR, ramond(h), [0, 1])]
        [3/80, 7/16]
        sage: accepts(AR, ramond(5), [0, 1]), accepts(AR, {'L': 7/240, 'G': 0})
        (True, True)
        sage: TestSuite(ZR.module(ramond(7/16), [0, 1])).run()

    ``W_3``: commutative, a module for every ``(h, w)``::

        sage: K.<c> = Frac(QQ['c'])
        sage: accepts(va.W3(K, c, conformal_vector='L').zhu_algebra(), {'L': c, 'W': c^2 + 1})
        True

    Refused: missing or unknown generators, matrices of different sizes::

        sage: [accepts(A, act) for act in ({'E': 0, 'H': 0}, {'E': 0, 'H': 0, 'F': 0, 'X': 0},
        ....:                              {'E': matrix([[0, 1], [0, 0]]), 'H': 0, 'F': 0})]
        [False, False, False]

    A grading group other than the default needs the embedding::

        sage: N2 = va.N2(QQ, 1, conformal_vector='L', horizontal_grading={'G1': 1, 'G2': -1})
        sage: N2.zhu_algebra().module({'L': 1/6, 'J': 1/3}, horizontal_grades=[(1/3,)],
        ....:                         horizontal_grading_group=QQ^2)
        Traceback (most recent call last):
        ...
        ValueError: give the embedding of the grading group Integer Ring of the vertex algebra into Vector space of dimension 2 over Rational Field
    """

    @staticmethod
    def __classcall_private__(cls, A, action, parities=None, horizontal_grades=None, horizontal_grading_group=None,
                              embedding=None, horizontal_grading=None):
        """
        Normalise the input: immutable matrices in the order of the
        generators (scalars as 1 x 1), the parities as a tuple, the grading
        datum (:func:`_grading_spec`, passed by keyword).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U is A.module([matrix([[0, 1], [0, 0]]), matrix([[1, 0], [0, -1]]),
            ....:                matrix([[0, 0], [1, 0]])])
            True
            sage: A.module({'E': 0, 'H': 0})
            Traceback (most recent call last):
            ...
            ValueError: give the matrices of all generators ['E', 'H', 'F']
        """
        R = A.base_ring()
        names = list(A._names)
        n = len(names)
        if isinstance(action, dict):
            unknown = set(map(str, action)) - set(names)
            if unknown:
                raise ValueError('unknown generators {} (the generators are {})'
                                 .format(sorted(unknown), names))
            if len(action) != n:
                raise ValueError('give the matrices of all generators {}'.format(names))
            vals = [action[s] for s in names]
        else:
            vals = list(action)
            if len(vals) != n:
                raise ValueError('{} matrices expected (one per generator)'.format(n))
        mats = []
        for v in vals:
            if isinstance(v, (list, tuple)) or hasattr(v, 'nrows'):
                M = matrix(R, v)
            else:                                  # a scalar: one-dimensional
                M = matrix(R, [[v]])
            M.set_immutable()
            mats.append(M)
        d = mats[0].nrows() if mats else (len(parities) if parities is not None else 1)
        if parities is None:
            parities = [0] * d
        parities = tuple(int(p) % 2 for p in parities)
        if horizontal_grading is None:     # (given: the normalised datum, from unpickling)
            horizontal_grading = _grading_spec(A, mats, d, horizontal_grades, horizontal_grading_group, embedding)
        return CombinatorialFreeModule.__classcall__(cls, A, tuple(mats), parities,
                                                     horizontal_grading=horizontal_grading)

    def __init__(self, A, mats, parities, horizontal_grading=None):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U.dimension(), U._par, U._mats[1].list()
            (2, (0, 0), [1, 0, 0, -1])
        """
        R = A.base_ring()
        d = len(parities)
        for s, M in zip(A._names, mats):
            if M.nrows() != d or M.ncols() != d:
                raise ValueError('the matrix of [{}] is not {} x {}'.format(s, d, d))
        self._A = A
        self._mats = mats
        self._par = parities
        self._cover = A._zhu_cover() if hasattr(A, '_zhu_cover') else A
        self._check()
        self._grading = None
        if horizontal_grading is not None:
            self._install_grading(*horizontal_grading)
        cat = Modules(R).WithBasis().FiniteDimensional()
        if any(parities):
            cat = cat.Super()
        CombinatorialFreeModule.__init__(self, R, list(range(d)), prefix='u', category=cat)

    def _evaluate(self, dct):
        """
        ``Σ c Π ρ(x_i)`` for a dict ``{ordered monomial: c}`` (also words
        of a free quotient).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._evaluate({(0, 2): 1, (): 2})
            [3 0]
            [0 2]
        """
        R = self._A.base_ring()
        d = len(self._par)
        out = matrix(R, d, d)
        for m, c in dct.items():
            P = matrix.identity(R, d)
            for i in m:
                P = P * self._mats[i]
            out += c * P
        return out

    def _check(self):
        """
        Check the parities and the relations of the Zhu algebra (and of the
        quotient: its Gröbner basis, or the defining relations of a free
        quotient); a ``ValueError`` naming the first failure.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._check()
            sage: A.module({'E': 0, 'H': 1, 'F': 0})
            Traceback (most recent call last):
            ...
            ValueError: the matrices do not satisfy the relation [F][E] - [E][F] = -[H] of the Zhu algebra
        """
        A = self._cover
        par = self._par
        for i, M in enumerate(self._mats):
            odd = i in A._odd
            for r in range(M.nrows()):
                for c in range(M.ncols()):
                    if M[r, c] and (par[r] + par[c]) % 2 != int(odd):
                        raise ValueError('the matrix of the {} generator [{}] does not {} '
                                         'the parity'.format('odd' if odd else 'even',
                                                             A._names[i],
                                                             'change' if odd else 'preserve'))
        mats = self._mats
        rels = getattr(self._A, '_relation_dicts', None)
        if rels is not None:
            # a quotient of the free algebra (ZhuAlgebraFreeQuotient): its
            # defining relations, exact whatever the degree of its Gröbner basis
            for d in rels:
                if self._evaluate(d):
                    raise ValueError('the matrices do not satisfy the relation {} = 0 of '
                                     'the Zhu algebra'.format(
                                         self._A._from_dict(d, coerce=False, remove_zeros=True)))
            return
        n = A.ngens()
        for a in range(n):
            for b in range(a + 1):
                if a == b and a not in A._odd:
                    continue
                rel = A._commutator(a, b)
                if a == b:
                    lhs = mats[a] * mats[a]
                else:
                    lhs = mats[a] * mats[b] - A._sign(a, b) * mats[b] * mats[a]
                if lhs != self._evaluate(rel):
                    x = A._from_dict(rel, coerce=False, remove_zeros=True)
                    ga, gb = '[{}]'.format(A._names[a]), '[{}]'.format(A._names[b])
                    if a == b:
                        name = '{}{}'.format(ga, ga)
                    else:
                        name = '{}{} - {}{}{}'.format(ga, gb, '' if A._sign(a, b) == 1 else '-',
                                                     gb, ga)
                        name = name.replace(' - -', ' + ')
                    raise ValueError('the matrices do not satisfy the relation {} = {} of '
                                     'the Zhu algebra'.format(name, x))
        Z = self._A
        if Z is not A:
            for g in Z._G.gens():
                dct = Z._from_P(g)
                if self._evaluate(dct):
                    raise ValueError('the matrices do not satisfy the relation {} = 0 of '
                                     'the Zhu algebra'.format(
                                         A._from_dict(dct, coerce=False, remove_zeros=True)))

    def _install_grading(self, B, horizontal_grades, M):
        """
        Install the grading datum (``B``, the grades of the basis vectors,
        the matrix ``M`` of ``ι``), checking that each generator of grade
        ``a`` maps grade ``b`` to ``b + ι(a)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._grading[1]
            [(1,), (-1,)]
            sage: U._install_grading(U.horizontal_grading_group(), [(1,), (1,)], matrix(QQ, [[2]]))
            Traceback (most recent call last):
            ...
            ValueError: the matrix of [E] maps a vector of grade (1) to one of grade (1), not (1) + ι(grade of E) = (3)
        """
        from ._grading_groups import GradingEmbedding, module_grading_group
        cover = self._cover
        gd = cover.vertex_algebra()._grading
        BG = module_grading_group(B)
        iota = GradingEmbedding(gd['group'], BG, M)
        coords = [BG.coords(g) for g in horizontal_grades]
        for z, M in enumerate(self._mats):
            shift = iota(gd['g'][cover._vgen[z]])
            for r in range(M.nrows()):
                for c in range(M.ncols()):
                    if M[r, c] and coords[r] != BG.add(coords[c], shift):
                        raise ValueError(
                            'the matrix of [{}] maps a vector of grade {} to one of grade {}, '
                            'not {} + ι(grade of {}) = {}'.format(
                                cover._names[z], BG.element(coords[c]), BG.element(coords[r]),
                                BG.element(coords[c]), cover._names[z],
                                BG.element(BG.add(coords[c], shift))))
        self._grading = (iota, coords)

    def horizontal_grades(self):
        """
        The grades of the basis vectors (``None`` if the module is not
        graded).

        OUTPUT:

        A list of elements of :meth:`horizontal_grading_group`, one per
        basis vector, or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U.horizontal_grades()          # sl_2-weights, fundamental weight coordinates
            [(1), (-1)]
            sage: Vir = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: Vir.zhu_algebra().module({'L': 3}).horizontal_grades() is None
            True
        """
        if self._grading is None:
            return None
        iota, coords = self._grading
        return [iota.B.element(t) for t in coords]

    def horizontal_grading_group(self):
        """
        The group of the grades (``None`` if the module is not graded).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U.horizontal_grading_group()
            Vector space of dimension 1 over Rational Field
        """
        return None if self._grading is None else self._grading[0].B.group

    # -- the interface used by induced modules (also of ZhuVermaModule) -- #

    _is_finite_top = True
    _mu = None

    @property
    def _iota(self):
        """
        The embedding ``ι`` of the grading group of the vertex algebra into
        that of the module (``None`` if ungraded).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._iota
            Embedding of Ambient free module of rank 1 ... into Vector space of dimension 1 over Rational Field
            sage: va.Virasoro(QQ, -22/5, conformal_vector='L').zhu_algebra().module({'L': 3})._iota is None
            True
        """
        return None if self._grading is None else self._grading[0]

    def _first_key(self):
        """
        The first basis key, ``0``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._first_key()
            0
        """
        return 0

    def _top_keys(self):
        """
        All basis keys (the top is finite).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._top_keys()
            [0, 1]
        """
        return list(range(len(self._par)))

    def _gen_action_key(self, z, u):
        """
        The ``z``-th generator on the basis vector ``u``, as a dict (the
        column of its matrix).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._gen_action_key(2, 0), U._gen_action_key(1, 1), U._gen_action_key(0, 0)
            ({1: 1}, {1: -1}, {})
        """
        M = self._mats[z]
        return {r: M[r, u] for r in range(M.nrows()) if M[r, u]}

    def _parity_key(self, u):
        """
        The parity of the basis vector ``u``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._parity_key(1)
            0
        """
        return self._par[u]

    def _grade_key(self, u):
        """
        The grade coordinates of the basis vector ``u`` (graded modules
        only).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._grade_key(0), U._grade_key(1)
            ((1,), (-1,))
        """
        return self._grading[1][u]

    def _tdeg(self, u):
        """
        The twisted degree of ``u``: 0 (a finite top).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._tdeg(1)
            0
        """
        return QQ.zero()

    def _keys_of_tdeg(self, t):
        """
        The basis keys of twisted degree ``t``: all of them for ``t = 0``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._keys_of_tdeg(0), U._keys_of_tdeg(1)
            ([0, 1], [])
        """
        return self._top_keys() if t == 0 else []

    def _keys_of_grade(self, G):
        """
        The basis keys of grade coordinates ``G``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._keys_of_grade((-1,)), U._keys_of_grade((3,))
            ([1], [])
        """
        return [u for u in self._top_keys() if self._grade_key(u) == G]

    def _repr_top(self, u):
        """
        The name of the top vector ``u`` in induced modules: ``v`` if the
        module is one-dimensional, else ``u[i]``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._repr_top(1), A.module({'E': 0, 'H': 0, 'F': 0})._repr_top(0)
            ('u[1]', 'v')
        """
        return 'v' if len(self._par) == 1 else 'u[{}]'.format(u)

    def _scalar(self, x):
        """
        The scalar by which ``x`` (an element of the Zhu algebra) acts, or
        ``None`` if it is not a scalar.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U._scalar(A(V.conformal_vector())), U._scalar(A.gen(1))
            (1/4, None)
        """
        m = self.representation_matrix(x)
        return m[0, 0] if m == m[0, 0] * m.parent().identity_matrix() else None

    def _eigenvalue(self, x):
        """
        The eigenvalue of ``x`` if it has only one (``x`` minus it is
        nilpotent), or ``None``.

        TESTS:

        A Jordan block of ``[L]``::

            sage: import vertex_algebras as va
            sage: A = va.Virasoro(QQ, -22/5, conformal_vector='L').zhu_algebra()
            sage: U = A.module({'L': matrix([[1, 1], [0, 1]])})
            sage: U._eigenvalue(A.gen(0)), U._scalar(A.gen(0))
            (1, None)
            sage: A.module({'L': matrix([[1, 0], [0, 2]])})._eigenvalue(A.gen(0)) is None
            True
        """
        m = self.representation_matrix(x)
        d = m.nrows()
        if not d:
            return None
        h = m.trace() / d
        return h if ((m - h) ** d).is_zero() else None

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: repr(U)
            "Module of dimension 2 over Zhu algebra of Universal affine vertex algebra of type ['A', 1] at level 1 over Rational Field"
        """
        return 'Module of dimension {} over {}'.format(len(self._par), self._A)

    def algebra(self):
        """
        The Zhu algebra acting.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: U.algebra() is A
            True
        """
        return self._A

    def generator_matrices(self):
        """
        The matrices of the generators, a dict by name.

        OUTPUT:

        A dictionary ``{name: matrix}`` over the generators of the Zhu
        algebra.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: G = U.generator_matrices(); sorted(G)
            ['E', 'F', 'H']
            sage: G['H']
            [ 1  0]
            [ 0 -1]
        """
        return dict(zip(self._A._names, self._mats))

    def representation_matrix(self, x):
        """
        The matrix of ``x`` on the basis.

        INPUT:

        - ``x`` -- an element of the Zhu algebra, or of the vertex algebra
          (then its image ``A(x)``)

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: E, H, F = A.gens(); U.representation_matrix(E*F)
            [1 0]
            [0 0]
            sage: U.representation_matrix(V.conformal_vector())     # L_0 on the top
            [1/4   0]
            [  0 1/4]
        """
        A = self._A
        P = _parent(x)
        if P is not A:
            x = A(x)
        return self._evaluate(dict(x._monomial_coefficients))

    def _even_odd_on_basis(self, i):
        """
        The parity of the basis vector ``u[i]``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L').zhu_algebra(twist='ramond')
            sage: U = A.module({'L': matrix([[7/16, 0], [0, 7/16]]), 'G': matrix([[0, 1], [49/120, 0]])},
            ....:              parities=[0, 1])
            sage: U._even_odd_on_basis(0), U._even_odd_on_basis(1)
            (0, 1)
        """
        return self._par[i]

    def _get_action_(self, S, op, self_on_left):
        """
        The action ``x * u`` of the Zhu algebra (:class:`_ZhuAction`).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra()
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: import operator
            sage: U._get_action_(A, operator.mul, False)
            Left action by Zhu algebra of ... on Module of dimension 2 over ...
            sage: A.gen(2) * U.basis()[0]
            u[1]
        """
        if op is operator.mul and not self_on_left and S is self._A:
            return _ZhuAction(S, self)
        return super()._get_action_(S, op, self_on_left)


# ---------------------------------------------------------------------- #
#  Verma modules of Zhu algebras (infinite dimensional, graded)          #
# ---------------------------------------------------------------------- #

class _VermaKeys:
    """
    The basis keys of a :class:`ZhuVermaModule`: weakly increasing tuples
    of lowering generators (odd ones at most once; not iterable).

    TESTS::

        sage: from vertex_algebras.zhu_algebra_module import _VermaKeys
        sage: K = _VermaKeys([2], frozenset()); (2, 2) in K, (0,) in K
        (True, False)
    """

    def __init__(self, lowering, odd):
        """
        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _VermaKeys
            sage: K = _VermaKeys([1, 2], frozenset([1])); sorted(K._low), K._odd
            ([1, 2], frozenset({1}))
        """
        self._low = frozenset(lowering)
        self._odd = odd

    def __contains__(self, x):
        """
        Whether ``x`` is a weakly increasing tuple of lowering generators,
        the odd ones at most once.

        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _VermaKeys
            sage: K = _VermaKeys([1, 2], frozenset([1]))
            sage: (1, 2) in K, (1, 1) in K, (2, 1) in K, [1] in K, () in K
            (True, False, False, False, True)
        """
        if not isinstance(x, tuple):
            return False
        for k, z in enumerate(x):
            if z not in self._low:
                return False
            if k and (x[k - 1] > z or (x[k - 1] == z and z in self._odd)):
                return False
        return True

    def an_element(self):
        """
        The empty tuple (the highest weight vector).

        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _VermaKeys
            sage: _VermaKeys([2], frozenset()).an_element()
            ()
        """
        return ()

    def cardinality(self):
        """
        ``+Infinity``.

        TESTS::

            sage: from vertex_algebras.zhu_algebra_module import _VermaKeys
            sage: _VermaKeys([2], frozenset()).cardinality()
            +Infinity
        """
        from sage.rings.infinity import Infinity
        return Infinity


class ZhuVermaModule(_ArtFromRepr, CombinatorialFreeModule):
    r"""
    The Verma module of a Zhu algebra ``A = A(V)`` for a triangular
    decomposition of its generators (``A.verma_module(highest_weight)``):
    ``V`` graded by ``A_V``, a functional ``φ`` on ``A_V`` (default for
    affine vertex algebras: the height) splits the generators ``[g]`` into
    raising (``φ(g) > 0``), Cartan (``g`` of grade 0) and lowering
    (``φ(g) < 0``) ones; ``v`` is killed by the raising ones, the Cartan ones
    act by the scalars ``highest_weight``, and the basis is the ordered
    monomials in the lowering generators applied to ``v``.  A generator acts
    by moving to its place with the commutation relations of ``A``
    (``x l = ± l x + [x, l]``), which terminates by the filtration degree.
    For ``U(g)`` (affine vertex algebras) this is the Verma module ``M(λ)``
    of ``g``; for non-linear ``A`` (finite W-algebras) the rewriting is
    well defined when the PBW theorem holds for the decomposition, which
    ``TestSuite`` checks on low degrees (``_test_relations``).

    The module is infinite dimensional and graded by a group ``B ⊇ A_V``
    (default for affine algebras: the weights ``P ⊗ R``, ``v`` of grade
    ``λ``), with finite-dimensional pieces: an induced module
    ``V.induced_module(U)`` then has infinite-dimensional conformal weight
    spaces and finite graded ones.  It also carries the twisted degree
    ``μ(horizontal_grade - horizontal_grade v)``, ``μ = -ε φ`` with ``ε`` small enough that all
    strong generators of ``V`` keep a positive twisted weight: the induced
    module is bounded below for ``depth + μ``, with finite pieces, which is
    what submodules and maximal submodules are computed by.

    INPUT:

    - ``A`` -- a :class:`zhu_algebra.ZhuAlgebra` of a vertex algebra graded
      by an abelian group (quotients of Zhu algebras are not supported)
    - ``highest_weight`` -- a dict ``{name: scalar}``: the scalars of the
      Cartan generators (those whose grade the functional sends to 0) on
      ``v``
    - ``functional`` -- (default: ``None``) a list of rationals, one per
      free coordinate of the grading group (0 on its torsion): the
      functional ``φ``; the default (affine vertex algebras only) is the
      height
    - ``horizontal_grade`` -- (default: ``None``) the grade of ``v``;
      for affine algebras by default ``λ`` in ``P ⊗ R``
    - ``horizontal_grading_group`` -- (default: ``None``) the group ``B``
      of the grades (default: the weights ``P ⊗ R`` for affine algebras,
      else the grading group of ``V`` if ``horizontal_grade`` lies in it,
      else ``R^k``)
    - ``embedding`` -- (default: ``None``) the matrix on coordinates of the
      embedding of the grading group of ``V`` into ``B`` (default: the
      identity)
    - ``horizontal_grading`` -- (default: ``None``) the normalised grading
      datum; internal (used by unpickling), not to be given by users

    EXAMPLES:

    The Verma module of ``sl_2`` of highest weight ``λ = -1/2`` (an
    admissible weight at level ``-1/2``): one vector of each weight ``λ - 2n``::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
        ....:                                  conformal_vector='sugawara')
        sage: A = V.zhu_algebra(); E, H, F = A.gens()
        sage: M = A.verma_module({'H': -1/2}); M
        Verma module of highest weight {'H': -1/2} over Zhu algebra of Universal affine vertex algebra of type ['A', 1] at level -1/2 over Rational Field
        sage: v = M.highest_weight_vector(); x = F * (F * v); x
        [F]^2 v
        sage: E * x, H * x
        (-3*[F] v, -9/2*[F]^2 v)
        sage: M.horizontal_grade(x), M.horizontal_grading_group(), [len(M.graded_piece((w,))) for w in (-1/2, -5/2, 3/2)]
        ((-9/2), Vector space of dimension 1 over Rational Field, [1, 1, 0])
        sage: M._scalar(A(V.conformal_vector()))          # the Casimir on M(λ): λ(λ + 2)/(4(k + 2))
        -1/8

    TESTS::

        sage: loads(dumps(M)) is M, M is A.verma_module({'H': -1/2})
        (True, True)
        sage: TestSuite(M).run()
        sage: A.verma_module({})
        Traceback (most recent call last):
        ...
        ValueError: give the highest weight of the Cartan generators ['H']

    A symbolic highest weight ``λ = 2k/3`` of ``U(sl_2)``: ``[E] [F]^n v =
    n (λ - n + 1) [F]^{n-1} v``, ``[H]`` and the grade ``λ - 2n``, and the
    Casimir ``[T] = λ(λ + 2)/(4(k + 2))``::

        sage: Kk.<k> = Frac(QQ['k'])
        sage: Vk = UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, names=['E', 'H', 'F'],
        ....:                                   conformal_vector='sugawara')
        sage: Ak = Vk.zhu_algebra(); E, H, F = Ak.gens(); lam = 2*k/3
        sage: U = Ak.verma_module({'H': lam}); w = U.highest_weight_vector(); ok = True
        sage: for n in range(1, 6):
        ....:     w = F * w
        ....:     ok = (ok and E * w == n * (lam - n + 1) * U.monomial((2,) * (n - 1))
        ....:           and H * w == (lam - 2*n) * w and U.horizontal_grade(w) == vector(Kk, [lam - 2*n]))
        sage: ok, U._scalar(Ak(Vk.conformal_vector())) == lam*(lam + 2)/(4*(k + 2))
        (True, True)
        sage: TestSuite(U).run()

    ``U(sl_3)``: the dimensions of the weight spaces ``λ - aα_1 - bα_2`` are
    Kostant's partition function ``min(a, b) + 1``::

        sage: V3 = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1, conformal_vector='sugawara')
        sage: U3 = V3.zhu_algebra().verma_module({'alphacheck[1]': 1/3, 'alphacheck[2]': -2})
        sage: C = CartanMatrix(['A', 2])
        sage: all(len(U3.graded_piece(tuple(vector(QQ, [1/3, -2]) - C * vector([a, b])))) == min(a, b) + 1
        ....:     for a in range(5) for b in range(5))
        True
        sage: TestSuite(U3).run()

    Refused: a functional vanishing on a nonzero grade; no functional (and
    no default) for other gradings::

        sage: V.zhu_algebra().verma_module({'H': 1}, functional=[0])
        Traceback (most recent call last):
        ...
        ValueError: the functional vanishes on the nonzero grade of [E]: choose a generic functional
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: C2 = UniversalVertexAlgebra(QQ, ['a', 'b'], {}, degrees={'a': 1, 'b': 1},
        ....:                             horizontal_grading={'a': 1, 'b': -1})
        sage: C2.zhu_algebra().verma_module({})
        Traceback (most recent call last):
        ...
        ValueError: give the functional on the grading group splitting the generators into raising and lowering ones
    """

    @staticmethod
    def __classcall_private__(cls, A, highest_weight, functional=None, horizontal_grade=None,
                              horizontal_grading_group=None, embedding=None, horizontal_grading=None):
        """
        Normalise the input to ``(A, highest weight items, functional)``
        and the grading datum ``(B, grade of v, M)`` (by keyword): the
        Cartan generators are those on whose grades the functional
        vanishes (default for affine algebras: the height, ``B = P ⊗ R``).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M is A.verma_module({'H': -1/2}, functional=[1]), loads(dumps(M)) is M
            (True, True)
            sage: A.verma_module({'H': 1}, functional=[0])
            Traceback (most recent call last):
            ...
            ValueError: the functional vanishes on the nonzero grade of [E]: choose a generic functional
        """
        from sage.matrix.constructor import matrix
        from sage.modules.free_module import FreeModule

        from ._grading_groups import module_grading_group
        if hasattr(A, 'cover_ring'):
            raise NotImplementedError('Verma modules of quotients of Zhu algebras are not '
                                      'implemented yet')
        V = A.vertex_algebra()
        gd = V._grading
        if gd is None:
            raise ValueError('Verma modules need a vertex algebra graded by an abelian group '
                             '(the triangular decomposition)')
        AG, g = gd['group'], gd['g']
        R = V.base_ring()
        auto = getattr(V, '_cartan_weight_grading', None)
        auto = auto() if auto is not None else None
        if functional is None:
            if auto is None:
                raise ValueError('give the functional on the grading group splitting the '
                                 'generators into raising and lowering ones')
            functional = [1] * AG.k                    # the height
        phi = tuple(QQ(x) for x in functional)
        if len(phi) != AG.k or any(phi[j] for j in AG.torsion):
            raise ValueError('the functional must have one rational value per free coordinate '
                             'of {} (and 0 on its torsion)'.format(AG.group))
        cartan = []
        for z, v in enumerate(A._vgen):
            val = sum(p * x for p, x in zip(phi, g[v]))
            if val == 0:
                if not AG.is_zero(g[v]):
                    raise ValueError('the functional vanishes on the nonzero grade of [{}]: '
                                     'choose a generic functional'.format(A._names[z]))
                cartan.append(z)
        hw = dict(highest_weight)
        names = [A._names[z] for z in cartan]
        if set(map(str, hw)) != set(names):
            raise ValueError('give the highest weight of the Cartan generators {}'.format(names))
        hw = tuple((n, R(hw[n])) for n in names)
        if horizontal_grading is not None:         # the normalised datum (unpickling)
            return CombinatorialFreeModule.__classcall__(cls, A, hw, phi, horizontal_grading=horizontal_grading)
        if horizontal_grade is None and horizontal_grading_group is None and embedding is None and auto is not None:
            B = FreeModule(R, len(auto[0]))
            vals = dict(hw)
            gv = B([vals[V._gens[i]] for i in auto[0]])
            M = auto[1]
        else:
            if horizontal_grade is None:
                raise ValueError('give the grade of the highest weight vector')
            if horizontal_grading_group is None:
                try:
                    AG.coords(horizontal_grade)
                    horizontal_grading_group = AG.group
                except ValueError:
                    horizontal_grading_group = FreeModule(R, AG.k)
            BG = module_grading_group(horizontal_grading_group)
            if embedding is None:
                if BG.k != AG.k:
                    raise ValueError('give the embedding of {} into {}'.format(AG.group, BG.group))
                embedding = matrix.identity(BG.R if BG.kind == 'vector' else ZZ, BG.k)
            B = BG.group
            M = matrix(BG.R if BG.kind == 'vector' else ZZ, BG.k, AG.k, embedding)
            gv = BG.element(BG.coords(horizontal_grade))
        gv.set_immutable() if hasattr(gv, 'set_immutable') else None
        M = matrix(M)
        M.set_immutable()
        return CombinatorialFreeModule.__classcall__(cls, A, hw, phi, horizontal_grading=(B, gv, M))

    def __init__(self, A, hw, phi, horizontal_grading=None):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2})
            sage: M._hw, M._raising, M._lowering, M._mu
            ((('H', -1/2),), frozenset({0}), (2,), (-1/4,))
        """
        from ._grading_groups import GradingEmbedding, module_grading_group
        V = A.vertex_algebra()
        gd = V._grading
        AG, g = gd['group'], gd['g']
        self._A = A
        self._cover = A
        self._hw = hw
        self._phi = phi
        B, gv, M = horizontal_grading
        BG = module_grading_group(B)
        self._iota_ = GradingEmbedding(AG, BG, M)
        self._gv = BG.coords(gv)
        self._g = [g[v] for v in A._vgen]                  # grades of the Zhu generators
        val = [sum(p * x for p, x in zip(phi, t)) for t in self._g]
        self._raising = frozenset(z for z, x in enumerate(val) if x > 0)
        self._lowering = tuple(z for z, x in enumerate(val) if x < 0)
        names = dict(hw)
        self._lam = {z: names[A._names[z]] for z, x in enumerate(val) if x == 0}
        # the twisted degree μ = -ε φ: |μ(g)| <= 1/4 on all generators of V
        top = max([abs(sum(p * x for p, x in zip(phi, t))) for t in g] + [1])
        eps = QQ((1, 4)) / top
        self._mu = tuple(-eps * p for p in phi)
        self._odd = A._odd
        self._cache = {}
        R = A.base_ring()
        cat = Modules(R).WithBasis()
        if any(z in A._odd for z in self._lowering):
            cat = cat.Super()
        CombinatorialFreeModule.__init__(self, R, _VermaKeys(self._lowering, A._odd),
                                         prefix='', category=cat)

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: repr(M)
            "Verma module of highest weight {'H': -1/2} over Zhu algebra of Universal affine vertex algebra of type ['A', 1] at level -1/2 over Rational Field"
        """
        return 'Verma module of highest weight {} over {}'.format(
            dict(self._hw), self._A)

    def _repr_term(self, u):
        """
        The basis vector ``u`` as an ordered monomial applied to ``v``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._repr_term(()), M._repr_term((2, 2))
            ('v', '[F]^2 v')
        """
        return 'v' if not u else '{} v'.format(self._A._repr_term(u))

    def algebra(self):
        """
        The Zhu algebra acting.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: M.algebra() is A
            True
        """
        return self._A

    def highest_weight_vector(self):
        """
        The highest weight vector ``v``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: v = M.highest_weight_vector(); v, E * v, H * v
            (v, 0, -1/2*v)
        """
        return self.monomial(())

    def highest_weight(self):
        """
        The scalars of the Cartan generators on ``v``, a dict by name.

        OUTPUT:

        A dictionary ``{name: scalar}``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: M.highest_weight()
            {'H': -1/2}
        """
        return dict(self._hw)

    # -- grading ---------------------------------------------------------- #

    def horizontal_grading_group(self):
        """
        The group ``B`` of the grades (for affine algebras: the weights
        ``P ⊗ R``, in fundamental weight coordinates).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: M.horizontal_grading_group()
            Vector space of dimension 1 over Rational Field
        """
        return self._iota_.B.group

    def _grade_key(self, u):
        """
        The grade coordinates of the basis vector ``u``: that of ``v`` plus
        ``ι`` of the grades of its lowering generators.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._grade_key(()), M._grade_key((2, 2))
            ((-1/2,), (-9/2,))
        """
        B = self._iota_.B
        t = self._gv
        for z in u:
            t = B.add(t, self._iota_(self._g[z]))
        return t

    def horizontal_grade(self, x):
        """
        The grade of a homogeneous vector ``x``.

        INPUT:

        - ``x`` -- a vector of this module (or anything it converts)

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: v = M.highest_weight_vector()
            sage: M.horizontal_grade(v), M.horizontal_grade(F * (F * v))
            ((-1/2), (-9/2))
            sage: M.horizontal_grade(v + F * v)
            Traceback (most recent call last):
            ...
            ValueError: v + [F] v is not homogeneous for the grading
        """
        x = self(x)
        gs = {self._grade_key(u) for u in x.support()}
        if len(gs) != 1:
            raise ValueError('{} is not homogeneous for the grading'.format(x))
        return self._iota_.B.element(gs.pop())

    def _mu_of(self, t):
        """
        The twisted degree ``μ(t)`` of grade coordinates ``t`` of the
        vertex algebra (``μ = -ε φ``).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._mu, M._mu_of((-2,)), M._mu_of((2,))
            ((-1/4,), 1/2, -1/2)
        """
        return sum((m * x for m, x in zip(self._mu, t)), QQ.zero())

    def _tdeg(self, u):
        """
        The twisted degree of the basis vector ``u`` (the sum of
        ``μ(g) > 0`` over its lowering generators).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._tdeg(()), M._tdeg((2, 2))
            (0, 1/2)
        """
        return sum((self._mu_of(self._g[z]) for z in u), QQ.zero())

    def _monomials(self, ok, budget):
        r"""The keys whose lowering generators satisfy ``ok`` at the end, the
        twisted degree (each lowering generator adds ``μ(g) > 0``) at most
        ``budget``; ``ok(acc, left)`` sees the generators so far and the
        budget left.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._monomials(lambda acc, left: True, 1/2)
            [(), (2,), (2, 2)]
            sage: M._monomials(lambda acc, left: left == 0, 1/2)
            [(2, 2)]
        """
        low = self._lowering
        out = []
        steps = [self._mu_of(self._g[z]) for z in low]

        def grow(start, left, acc):
            if ok(acc, left):
                out.append(tuple(acc))
            for k in range(start, len(low)):
                if steps[k] > left:
                    continue
                z = low[k]
                if z in self._odd and acc and acc[-1] == z:
                    continue
                acc.append(z)
                grow(k + (1 if z in self._odd else 0), left - steps[k], acc)
                acc.pop()
        grow(0, budget, [])
        return out

    def _keys_of_tdeg(self, t):
        """
        The basis keys of twisted degree ``t``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._keys_of_tdeg(0), M._keys_of_tdeg(1/4), M._keys_of_tdeg(1/2)
            ([()], [(2,)], [(2, 2)])
        """
        return self._monomials(lambda acc, left: left == 0, QQ(t))

    def _keys_of_grade(self, G):
        """
        The basis keys of grade coordinates ``G`` (through the twisted
        degree of ``G`` minus the grade of ``v``).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._keys_of_grade((-5/2,)), M._keys_of_grade((3/2,)), M._keys_of_grade((-1,))
            ([(2,)], [], [])
        """
        iota = self._iota_
        B = iota.B
        d = B.add(G, tuple(-x for x in self._gv))
        a = iota.preimage(d)
        if a is None:
            return []
        t = self._mu_of(a)
        if t < 0:
            return []
        return [u for u in self._keys_of_tdeg(t) if self._grade_key(u) == B.reduce(G)]

    def graded_piece(self, horizontal_grade):
        """
        The basis vectors of the given grade (finitely many).

        INPUT:

        - ``horizontal_grade`` -- an element of :meth:`horizontal_grading_group`

        OUTPUT:

        A list of basis vectors of this module.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F']).zhu_algebra()
            sage: E, H, F = A.gens(); M = A.verma_module({'H': -1/2})
            sage: M.graded_piece((-5/2,)), M.graded_piece((3/2,))
            ([[F] v], [])
        """
        G = self._iota_.B.coords(horizontal_grade)
        return [self.monomial(u) for u in self._keys_of_grade(G)]

    # -- the action ------------------------------------------------------- #

    def _comm(self, a, b):
        """
        ``x_a x_b - (-1)^{|a||b|} x_b x_a`` (for ``a == b`` odd: ``x_a^2``;
        ``{}`` for ``a == b`` even) as a dict over ordered monomials, for
        any order of ``a``, ``b``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._comm(0, 2), M._comm(2, 0), M._comm(1, 1)
            ({(1,): 1}, {(1,): -1}, {})
        """
        A = self._A
        if a > b or a == b:
            return A._commutator(a, b) if (a != b or a in self._odd) else {}
        s = A._sign(a, b)
        return {m: -s * c for m, c in A._commutator(b, a).items()}

    def _gen_action_key(self, z, u):
        """
        The ``z``-th generator on the basis vector ``u``, as a dict:
        raising ones kill ``v``, Cartan ones act by the highest weight,
        otherwise moved to their place by the commutation relations
        (memoised in ``_cache``).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._gen_action_key(0, (2, 2)), M._gen_action_key(2, (2,)), M._gen_action_key(1, ())
            ({(2,): -3}, {(2, 2): 1}, {(): -1/2})
        """
        ck = (z, u)
        r = self._cache.get(ck)
        if r is not None:
            return r
        one = self.base_ring().one()
        if not u:
            if z in self._raising:
                r = {}
            elif z in self._lam:
                c = self._lam[z]
                r = {(): c} if c else {}
            else:
                r = {(z,): one}
        else:
            l, rest = u[0], u[1:]
            if z in self._lowering and (z < l or (z == l and z not in self._odd)):
                r = {(z,) + u: one}
            elif z == l:                                 # odd: x x = x^2 (a square)
                r = self._act_dict(self._comm(z, z), {rest: one})
            else:                                        # z l = s l z + [z, l]
                s = self._A._sign(z, l)
                r = {}
                for k2, c2 in self._gen_action_key(z, rest).items():
                    _add(r, self._gen_action_key(l, k2), s * c2)
                _add(r, self._act_dict(self._comm(z, l), {rest: one}), one)
        self._cache[ck] = r
        return r

    def _act_dict(self, x, w):
        """
        The element ``x`` of ``A`` (a dict over ordered monomials) on the
        vector ``w`` (a dict over keys).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._act_dict({(0, 2): 1}, {(): 1}), M._act_dict({(2,): 2}, {(2,): 1})
            ({(): -1/2}, {(2, 2): 2})
        """
        out = {}
        for m, c in x.items():
            cur = dict(w)
            for z in reversed(m):
                nxt = {}
                for k, a in cur.items():
                    _add(nxt, self._gen_action_key(z, k), a)
                cur = nxt
            _add(out, cur, c)
        return out

    def _scalar(self, x):
        """
        The scalar of a central element ``x`` of ``A`` (on ``v``, hence on
        the module, which ``v`` generates), or ``None`` if ``x v`` is not a
        multiple of ``v``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._scalar(A(A.vertex_algebra().conformal_vector())), M._scalar(A.gen(2))
            (-1/8, None)
        """
        x = self._A(x)
        r = self._act_dict(dict(x._monomial_coefficients), {(): self.base_ring().one()})
        if set(r) - {()}:
            return None
        return r.get((), self.base_ring().zero())

    def _get_action_(self, S, op, self_on_left):
        """
        The action ``x * u`` of the Zhu algebra (:class:`_ZhuAction`, which
        calls :meth:`_act_element`).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: import operator
            sage: M._get_action_(A, operator.mul, False)
            Left action by Zhu algebra of ... on Verma module of highest weight {'H': -1/2} over ...
        """
        if op is operator.mul and not self_on_left and S is self._A:
            return _ZhuAction(S, self)
        return super()._get_action_(S, op, self_on_left)

    def _act_element(self, x, u):
        """
        The element ``x`` of ``A`` on the vector ``u``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._act_element(A.gen(2), v), M._act_element(A.gen(0), A.gen(2) * v)
            ([F] v, -1/2*v)
        """
        return self._from_dict(self._act_dict(dict(x._monomial_coefficients),
                                              dict(u._monomial_coefficients)),
                               coerce=False, remove_zeros=True)

    # -- the interface used by induced modules ---------------------------- #

    _is_finite_top = False

    @property
    def _iota(self):
        """
        The embedding ``ι`` of the grading group of the vertex algebra into
        that of the module.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._iota
            Embedding of Ambient free module of rank 1 ... into Vector space of dimension 1 over Rational Field
        """
        return self._iota_

    def _first_key(self):
        """
        The key of ``v``, ``()``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._first_key()
            ()
        """
        return ()

    def _top_keys(self):
        """
        ``None``: the module is infinite dimensional.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._top_keys() is None
            True
        """
        return None

    def _parity_key(self, u):
        """
        The parity of the basis vector ``u``: the number of its odd lowering
        generators mod 2.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._parity_key((2, 2))
            0
        """
        return sum(1 for z in u if z in self._odd) % 2

    def _even_odd_on_basis(self, u):
        """
        The parity of the basis vector ``u`` (see :meth:`_parity_key`).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._even_odd_on_basis(())
            0
        """
        return self._parity_key(u)

    def _repr_top(self, u):
        """
        The name of the top vector ``u`` in induced modules (zero modes
        applied to ``v``).

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._repr_top((2, 2)), M._repr_top(())
            ('F_{0} F_{0} v', 'v')
        """
        names = self._A._names
        return ' '.join(['{}_{{0}}'.format(names[z]) for z in u] + ['v'])

    # -- tests ------------------------------------------------------------ #

    def _test_relations(self, **options):
        r"""
        The relations of ``A`` hold on the vectors of twisted degree ``<= 3/2``
        (twice the largest step): ``x_a (x_b u) - (-1)^{|a||b|} x_b (x_a u) =
        [x_a, x_b] u`` and ``x_a (x_a u) = x_a^2 u`` for odd ``a``.

        TESTS::

            sage: import vertex_algebras as va
            sage: A = va.Affine(QQ, ['A', 1], -1/2, names=['E', 'H', 'F'],
            ....:                conformal_vector='sugawara').zhu_algebra()
            sage: M = A.verma_module({'H': -1/2}); v = M.highest_weight_vector()
            sage: M._test_relations()
        """
        tester = self._tester(**options)
        steps = sorted({self._mu_of(self._g[z]) for z in self._lowering})
        keys = [u for t in {QQ.zero()} | {a + b for a in [0] + steps for b in [0] + steps}
                for u in self._keys_of_tdeg(t)]
        n = self._A.ngens()
        one = self.base_ring().one()
        for u in keys:
            for a in range(n):
                for b in range(a + 1):
                    if a == b and a not in self._odd:
                        continue
                    if a == b:
                        lhs = self._act_dict({(a, a): one}, {u: one})
                    else:
                        lhs = self._act_dict({(a, b): one}, {u: one})
                        s = self._A._sign(a, b)
                        _add(lhs, self._act_dict({(b, a): one}, {u: one}), -s)
                    rhs = self._act_dict(self._comm(a, b), {u: one})
                    tester.assertEqual({k: c for k, c in lhs.items() if c},
                                       {k: c for k, c in rhs.items() if c})


def _add(dst, src, c):
    """
    ``dst += c * src`` for dicts ``{key: coefficient}`` (zeros removed);
    returns ``dst``.

    TESTS::

        sage: from vertex_algebras.zhu_algebra_module import _add
        sage: _add({1: 2}, {1: -2, 3: 1}, 1), _add({1: 2}, {1: 5}, 0)
        ({3: 1}, {1: 2})
    """
    if not c:
        return dst
    for k, v in src.items():
        w = dst.get(k)
        w = c * v if w is None else w + c * v
        if w:
            dst[k] = w
        else:
            dst.pop(k, None)
    return dst
