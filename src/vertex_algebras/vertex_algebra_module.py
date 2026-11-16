# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Modules over vertex algebras.

A (left) module over a vertex (super)algebra ``V`` is a (super) vector space
``M`` with fields ``Y_M(a, z) = Σ_n a_(n) z^{-n-1}`` for ``a ∈ V``, such
that ``a_(n) m = 0`` for ``n`` large, the vacuum acts as the identity and
the Borcherds identity holds.  The elements of ``V`` act through their own
methods, with the conventions of the vertex algebra: ``a.nproduct(m, n) =
a_(n) m``, ``a.ope(m)`` (entry ``k`` is ``a_(k) m``), ``a.no(m) = a * m =
a_(-1) m``, ``a.mode(m, n)`` (physics modes ``a_(n + Δ_a - 1)``).

:class:`VertexAlgebraModule` is the base class of all modules; its
parents are in :class:`vertex_algebra_categories.VertexAlgebraModules`
``(V)`` (``TestSuite`` checks the axioms).  :class:`AdjointModule`
(``V.adjoint_module()``) is ``V`` as a module over itself.

For a universal vertex algebra ``V = V(R)`` a module is determined by the
action of the strong generators: derivatives and normally ordered
products act by::

    (∂a)_(n) = -n a_(n-1),
    (:a B:)_(n) v = Σ_{j>=0} a_(-1-j) B_(n+j) v + (-1)^{|a||B|} Σ_{j>=0} B_(n-1-j) a_(j) v

(both sums finite), which ``_GeneratorActions`` computes for the
basis monomials of ``V`` from the actions of the generators.

EXAMPLES:

The Virasoro algebra acting on itself::

    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: E.<L> = NormallyOrderedExpressions(QQ)
    sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
    ....:                              conformal_vector=L)
    sage: L = Vir.gen('L')
    sage: M = Vir.adjoint_module(); M
    Adjoint module of Universal vertex algebra on generators ['L'] over Rational Field
    sage: M.category()
    Category of conformally graded modules with basis over Universal vertex algebra on generators ['L'] over Rational Field
    sage: v = M(L.no(L)); L.ope(v)
    [2*:dL L: + 1/6*d^3L, 4*:L L:, 3*dL, 17/2*L, 0, 3/2*|0>]
    sage: L.mode(v, 0) == 4*v, L.no(v) == M(L.no(L.no(L))), v.degree()
    (True, True, 4)
    sage: M.character(8) == Vir.character(8)
    True
    sage: TestSuite(M).run()

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import operator

from sage.arith.functions import lcm
from sage.arith.misc import binomial
from sage.categories.action import Action
from sage.categories.homset import Hom
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .vertex_algebra import _add_scaled, _ArtFromRepr
from .vertex_algebra_categories import _vertex_algebra_module_category


def _falling(n, k):
    """
    ``n (n-1) ... (n-k+1)``.

    TESTS::

        sage: from vertex_algebras.vertex_algebra_module import _falling
        sage: _falling(5, 2), _falling(3, 4), _falling(-1, 3), _falling(7, 0)
        (20, 0, -6, 1)
    """
    out = 1
    for i in range(k):
        out *= n - i
    return out


def _is_submodule_of(P, M):
    """
    True if ``P`` is a submodule of a submodule ... of ``M``.

    TESTS::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
        sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
        sage: from vertex_algebras.vertex_algebra_module import _is_submodule_of
        sage: N = M.submodule([L.mode(v, -1)]); N2 = N.submodule([L.mode(v, -1)])
        sage: _is_submodule_of(N, M), _is_submodule_of(N2, M), _is_submodule_of(M, N), _is_submodule_of(M, M)
        (True, True, False, False)
    """
    A = getattr(P, '_ambient_module', None)
    while A is not None:
        if A is M:
            return True
        A = getattr(A, '_ambient_module', None)
    return False


class VertexAlgebraModuleElement(IndexedFreeModuleElement):
    r"""
    A vector of a :class:`VertexAlgebraModule`: a finite linear combination of
    basis vectors.  The elements of the vertex algebra act on it by their
    own methods (``a.nproduct(m, n)``, ``a.ope(m)``, ``a * m``, ...).

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
        sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
        sage: x = L.mode(v, -1); x, x.parent() is M
        (L_{-1} v, True)
        sage: L.mode(x, 1), L.no(v) == L.nproduct(v, -1)
        (1/8*v, True)
    """

    def parity(self):
        r"""
        The parity (0 even, 1 odd) of a homogeneous vector; a ``ValueError``
        for an inhomogeneous one.  The zero vector is even.

        EXAMPLES:

        The adjoint module of the Neveu--Schwarz vertex superalgebra::

            sage: import vertex_algebras as va
            sage: V = va.NeveuSchwarz(QQ, 3/2); L, G = V.gens(); M = V.adjoint_module()
            sage: M(L).parity(), M(G).parity(), M(G.no(L)).parity(), M.zero().parity()
            (0, 1, 1, 0)
            sage: (M(L) + M(G)).parity()
            Traceback (most recent call last):
            ...
            ValueError: the vector is not homogeneous
        """
        P = self.parent()
        pars = {P._parity_key(m) for m in self._monomial_coefficients}
        if not pars:
            return 0
        if len(pars) > 1:
            raise ValueError('the vector is not homogeneous')
        return pars.pop()

    def horizontal_grade(self):
        r"""
        The grade of a homogeneous vector, in the grading group of the
        module (see :meth:`VertexAlgebraModule.horizontal_grading_group`).

        EXAMPLES:

        With the lattice grading, ``e^p`` in a module of the ``A_1`` lattice
        vertex algebra has grade ``p``::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: Lt = LatticeVertexAlgebra(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: M = Lt.simple_module([1/2]); v = M.highest_weight_vector(); e = Lt.e([1])
            sage: v, v.horizontal_grade()
            (e^(1/2), (1/2))
            sage: e.nproduct(v, -2), e.nproduct(v, -2).horizontal_grade()
            (e^(3/2), (3/2))
        """
        return self.parent().horizontal_grade(self)


class _NormallyOrderedAction(Action):
    """``a * m`` for ``a`` in the vertex algebra and ``m`` in the module:
    refused, like ``a * b`` in the vertex algebra (``a * b * m`` would be
    ``(a * b) * m``); use ``a.no(m)``, ``a.nproduct(m, n)`` or
    ``V.normally_ordered(a, b, m)``.

    TESTS::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
        sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
        sage: act = M._get_action_(V, operator.mul, False); act
        Left action by Universal vertex algebra on generators ['L'] over Rational Field on Verma module ...
        sage: L * v
        Traceback (most recent call last):
        ...
        TypeError: * is not the action of vertex algebra elements on module vectors: use a.no(m) = a_(-1) m, a.nproduct(m, n), or V.normally_ordered(a, b, m) (also V.no(a, b, m)) for a_(-1) b_(-1) m
        sage: L.no(v), V.no(L, L, v)
        (L_{-2} v, L_{-2} L_{-2} v)
    """

    def __init__(self, V, M):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.vertex_algebra_module import _NormallyOrderedAction
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: act = _NormallyOrderedAction(V, M)
            sage: act.actor() is V, act.domain() is M, act.is_left()
            (True, True, True)
        """
        Action.__init__(self, V, M, True, operator.mul)

    def _act_(self, a, m):
        """
        Refuse ``a * m`` (see the class).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._get_action_(V, operator.mul, False)._act_(L, v)
            Traceback (most recent call last):
            ...
            TypeError: * is not the action of vertex algebra elements on module vectors: ...
        """
        raise TypeError('* is not the action of vertex algebra elements on '
                        'module vectors: use a.no(m) = a_(-1) m, '
                        'a.nproduct(m, n), or V.normally_ordered(a, b, m) '
                        '(also V.no(a, b, m)) for a_(-1) b_(-1) m')


class VertexAlgebraModule(_ArtFromRepr, CombinatorialFreeModule):
    r"""
    Abstract base class of modules over a vertex (super)algebra ``V``, with a
    basis.

    Subclasses call ``VertexAlgebraModule.__init__(self, V, basis_keys,
    super=..., graded=...)`` and implement, on basis keys (``a`` of ``V``,
    ``m`` of the module):

    - ``_action_key(a, n, m)`` -- ``a_(n) m`` as a raw dict ``{key:
      coefficient}`` (never elements; cached by ``_action_mono()``);
    - ``_nmax_key(a, m)`` -- an upper bound of the ``n`` with ``a_(n) m !=
      0`` (default from the conformal weights, see ``_nmax_key()``);
    - ``_parity_key(m)``, ``_repr_term(m)``, ``vertex_algebra()`` (stored);
    - for conformally graded modules: ``_weight_key(m)``,
      ``_weight_space_enumeration(W, G)`` (a
      ``weight_spaces._KeyEnumeration``), ``_weight_bounds(G)`` (a lower
      bound of the weights or ``None``, and ``D`` with all weights in
      ``(1/D) ZZ``); optionally ``_grade_coords(horizontal_grade)`` and ``horizontal_grade(x)``.

    INPUT (of ``VertexAlgebraModule.__init__``):

    - ``V`` -- the vertex algebra acting
    - ``basis_keys`` -- the basis keys of the module
    - ``super`` -- boolean (default: ``False``); whether the module is a
      super module (parities of the basis keys from ``_parity_key``)
    - ``graded`` -- boolean (default: ``False``); whether the module is
      conformally graded (weights from ``_weight_key``)
    - ``prefix`` -- (default: ``''``) passed to
      :class:`~sage.combinat.free_module.CombinatorialFreeModule`

    Subclasses may set ``self._mgrading`` (a grading embedding, see
    :meth:`horizontal_grading_group`) before calling ``__init__``;
    otherwise the module is not graded by a group.

    EXAMPLES:

    Adjoint modules, Verma (induced) modules and lattice modules are
    instances::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
        sage: L = va.Lattice(QQ, matrix([[2]]))
        sage: [isinstance(M, VertexAlgebraModule) for M in
        ....:  (V.adjoint_module(), V.verma_module(1/16), L.simple_module([1/2]))]
        [True, True, True]
        sage: V.verma_module(1/16).category()
        Category of conformally graded modules with basis over Universal vertex algebra on ...
    """
    _is_vertex_algebra_module = True
    Element = VertexAlgebraModuleElement

    def __init__(self, V, basis_keys, super=False, graded=False, prefix=''):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.vertex_algebra() is V, M._is_super_module, M._is_graded_module, isinstance(M._act_cache, dict)
            (True, False, True, True)
        """
        self._V = V
        self._one_R = V.base_ring().one()
        self._act_cache = {}
        self._is_super_module = bool(super)
        self._is_graded_module = bool(graded)
        if not hasattr(self, '_mgrading'):
            self._mgrading = None
        cat = _vertex_algebra_module_category(V, super=super, graded=graded)
        CombinatorialFreeModule.__init__(self, V.base_ring(), basis_keys,
                                         prefix=prefix, category=cat)

    def vertex_algebra(self):
        r"""
        The vertex algebra acting on this module.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: V.verma_module(0).vertex_algebra() is V, V.adjoint_module().vertex_algebra() is V
            (True, True)
        """
        return self._V

    def clear_caches(self):
        r"""
        Forget the computed actions (for memory; results do not change).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.adjoint_module(); x = L.nproduct(M(L), 0); len(M._act_cache) > 0
            True
            sage: M.clear_caches(); len(M._act_cache), L.nproduct(M(L), 0) == x
            (0, True)
        """
        self._act_cache.clear()

    # ------------------------------------------------------------------ #
    #  Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _action_key(self, a, n, m):
        """
        ``a_(n) m`` on basis keys, as a raw dict (subclasses).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._action_key(M, (), -1, ((), 0))
            Traceback (most recent call last):
            ...
            NotImplementedError
        """
        raise NotImplementedError

    def _parity_key(self, m):
        """
        The parity of the basis vector ``m`` (default: even).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._parity_key(M, ((), 0))
            0
        """
        return 0

    def _nmax_key(self, a, m):
        r"""
        An upper bound of the ``n`` with ``a_(n) m != 0``: from the conformal
        weights, ``wt a + wt m - 1 - h_0`` with ``h_0`` a lower bound of the
        weights of the module.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._nmax_key(M, ((0, 0),), ((), 0)), len(L.ope(v)) - 1
            (1, 1)
        """
        V = self._V
        lb = self._lowest_weight()
        w = V._weight_key(a, V._weight_data()) + self._weight_key(m) - 1 - lb
        return QQ(w).floor()

    def _lowest_weight(self):
        """
        A lower bound of the conformal weights of the module.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._lowest_weight(), V.adjoint_module()._lowest_weight()
            (1/16, 0)
        """
        lb = getattr(self, '_lb', None)
        if lb is None:
            lb = self._weight_bounds(None)[0]
            if lb is None:
                raise NotImplementedError('{} has no lower bound of its conformal '
                                          'weights: give _nmax_key'.format(self))
            self._lb = lb
        return lb

    # ------------------------------------------------------------------ #
    #  Gradings by abelian groups                                          #
    # ------------------------------------------------------------------ #

    def _require_module_grading(self):
        """
        The grading embedding of the module, or a ``ValueError`` if the module
        is not graded by an abelian group.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._require_module_grading()
            Traceback (most recent call last):
            ...
            ValueError: Verma module ... is not graded by an abelian group
            sage: L2 = va.Lattice(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: L2.simple_module([1/2])._require_module_grading()
            Embedding of ...
        """
        if self._mgrading is None:
            raise ValueError('{} is not graded by an abelian group'.format(self))
        return self._mgrading

    def horizontal_grading_group(self):
        r"""
        The group ``B`` of the grades of the module: it contains the grading
        group ``A`` of the vertex algebra (:meth:`horizontal_grading_embedding`), and
        ``a_(n) m`` has grade ``horizontal_grade(a) + horizontal_grade(m)``.  A ``ValueError`` if
        the module is not graded.

        EXAMPLES:

        The lattice modules of ``A_1`` with the lattice grading: ``B`` is the
        dual lattice and ``e^p`` has grade ``p``::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: M = L.simple_module([1/2]); M.horizontal_grading_group()
            Free module of degree 1 and rank 1 over Integer Ring
            Echelon basis matrix:
            [1/2]
            sage: M.horizontal_grade(M.highest_weight_vector()), M.horizontal_grading_embedding()(3)
            ((1/2), (3))
        """
        return self._require_module_grading().B.group

    def horizontal_grading_embedding(self):
        r"""
        The embedding of the grading group of the vertex algebra into
        :meth:`horizontal_grading_group`.  A ``ValueError`` if the module is
        not graded.

        OUTPUT:

        A Sage module morphism (with ``kernel()``, ``image()``, ``lift()``)
        when both groups are ``ZZ``-modules (FGP modules into FGP or free
        ``ZZ``-modules; free ``ZZ``-modules into free ``ZZ``-modules or
        vector spaces), otherwise a map of sets: for ``ZZ``, ``Zmod(n)``,
        multiplicative ``AbelianGroup``, cartesian products, and free into
        FGP modules Sage has no module morphisms (see
        ``_grading_groups.GradingEmbedding.morphism()``).

        EXAMPLES:

        For the lattice grading of ``A_2`` the vertex algebra is graded by
        the lattice ``ZZ^2``, its modules by the dual lattice: a module
        morphism::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L2 = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]])).with_horizontal_grading('lattice')
            sage: iota = L2.simple_module([1/3, 2/3]).horizontal_grading_embedding()
            sage: iota.domain() is L2.horizontal_grading_group(), iota((1, 0))
            (True, (1, 0))
            sage: iota.kernel().rank(), iota.image().index_in(iota.codomain())
            (0, 3)

        The rank-1 lattice grading is by ``ZZ`` itself (a ring for Sage, no
        module morphisms): a map::

            sage: Lt = LatticeVertexAlgebra(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: M = Lt.simple_module([1/2]); e = Lt.e([1])
            sage: iota = M.horizontal_grading_embedding(); iota(2), iota(e.horizontal_grade())
            ((2), (1))
            sage: iota.domain() is Lt.horizontal_grading_group()
            True
        """
        return self._require_module_grading().morphism()

    def _grade_key(self, m):
        """The grade of the basis key ``m``, as coordinates in the grading
        group of the module (subclasses).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._grade_key(M, ((), 0))
            Traceback (most recent call last):
            ...
            NotImplementedError: Verma module ... is not graded by an abelian group
        """
        raise NotImplementedError('{} is not graded by an abelian group'.format(self))

    def _grade_coords(self, horizontal_grade):
        """
        The coordinates of a grade (an element of
        :meth:`horizontal_grading_group`) in the grading group of the module.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._grade_coords((3/2,)), Ma._grade_key(next(iter(Ma.highest_weight_vector().support())))
            ((3/2,), (-1/2,))
        """
        return self._require_module_grading().B.coords(horizontal_grade)

    def horizontal_grade(self, x):
        r"""
        The grade of a homogeneous vector ``x``, in
        :meth:`horizontal_grading_group`; a ``ValueError`` if ``x`` is not
        homogeneous (or the module is not graded).

        INPUT:

        - ``x`` -- a vector of ``self`` (or anything converting into ``self``)

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: Lt = LatticeVertexAlgebra(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: M = Lt.simple_module([1/2]); v = M.highest_weight_vector(); e = Lt.e([1])
            sage: M.horizontal_grade(v), M.horizontal_grade(e.nproduct(v, -2))
            ((1/2), (3/2))
            sage: M.horizontal_grade(v + e.nproduct(v, -2))
            Traceback (most recent call last):
            ...
            ValueError: e^(1/2) + e^(3/2) is not homogeneous for the grading
        """
        B = self._require_module_grading().B
        x = self(x)
        gs = {self._grade_key(m) for m in x.support()}
        if len(gs) != 1:
            raise ValueError('{} is not homogeneous for the grading'.format(x))
        return B.element(gs.pop())

    def _test_module_grading(self, **options):
        r"""
        ``a_(n) m`` has grade ``ι(horizontal_grade a) + horizontal_grade m`` for homogeneous ``a`` in
        ``some_elements()`` of the vertex algebra and ``m`` in those of the
        module (if the module is graded).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._test_module_grading()
            sage: M._test_module_grading()          # not graded: nothing to test
        """
        iota = self._mgrading
        if iota is None:
            return
        from .vertex_algebra_categories import _some_vertex_algebra_elements
        tester = self._tester(**options)
        V = self._V
        B = iota.B

        def key_grade(P, x):
            gs = {P._grade_key(m) for m in x.support()}
            return gs.pop() if len(gs) == 1 else None
        for a in _some_vertex_algebra_elements(V):
            ga = key_grade(V, a)
            if ga is None:
                continue
            for m in tester.some_elements():
                gm = key_grade(self, m)
                if gm is None:
                    continue
                if not self._product_in_window(a, 0, m):
                    continue
                for n in range(-2, len(a.ope(m))):
                    if not self._product_in_window(a, n, m):
                        continue
                    v = a.nproduct(m, n)
                    if not v.is_zero():
                        tester.assertEqual(key_grade(self, v), B.add(iota(ga), gm))

    def _weight_base(self):
        r"""
        The weight the degrees of :mod:`vertex_algebra_submodule` are counted
        from: a lower bound of the conformal weights (default), the top weight
        of an induced module (which may be symbolic).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._weight_base(M), M._weight_base(), V.adjoint_module()._weight_base()
            (1/16, 1/16, 0)
        """
        return self._lowest_weight()

    def _key_order(self, m):
        """A sort key on basis keys for leading monomials (submodules, normal
        forms of quotients); must be injective.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._key_order(M, ((), 0)), M._key_order((((0, -1),), 0))
            (((), 0), (1, ((0, -1),), 0))
        """
        return m

    #: the twisted degree ``depth + μ(horizontal_grade)``: ``μ`` on the coordinates of the
    #: grading group of ``V`` (``None``: ``μ = 0``); see ``_tdeg``
    _mu = None

    def _product_in_window(self, a, n, m):
        r"""
        Whether ``a_(n) m`` can be computed: always, except for modules
        known only in a window of degrees
        (:class:`~vertex_algebras.quotient_induced_module.InducedModuleUpTo`).
        The category tests skip the products outside it.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._product_in_window(L, -5, v)
            True
        """
        return True

    def _mu_of(self, a):
        """
        ``μ(a)`` for grade coordinates ``a`` of the vertex algebra.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: M._mu_of((1,))
            0
            sage: Ma._mu, Ma._mu_of((2,))
            ((-1/4,), -1/2)
        """
        if self._mu is None:
            return QQ.zero()
        return sum((m * x for m, x in zip(self._mu, a)), QQ.zero())

    def _tdeg(self, m):
        r"""
        The twisted degree of the basis key ``m``, a rational ``>= 0``: the
        weight minus :meth:`_weight_base` (plus ``μ`` of the grade for modules
        with infinite weight spaces, e.g. induced from a Verma module of the
        Zhu algebra).  ``a_(n)`` changes it by ``wt a + μ(horizontal_grade a) - n - 1``,
        all its pieces are finite dimensional, and submodules are computed
        piece by piece (:mod:`vertex_algebra_submodule`).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._tdeg(M, (((0, -2),), 0)), M._tdeg((((0, -2), (0, -1)), 0))
            (2, 3)
        """
        return QQ(self._weight_key(m) - self._weight_base())

    def _keys_of_tdeg(self, t):
        """
        The basis keys of twisted degree ``t`` (finitely many).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._keys_of_tdeg(M, 2)
            [(((0, -2),), 0), (((0, -1), (0, -1)), 0)]
        """
        return list(self.weight_space(self._weight_base() + t).basis_keys())

    def _tdeg_basis(self, t):
        """
        A basis of the piece of twisted degree ``t``, as raw dicts.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._tdeg_basis(1)
            [{(((0, -1),), 0): 1}]
        """
        one = self._one_R
        return [{k: one} for k in self._keys_of_tdeg(t)]

    def _piece_keys(self, t, G):
        """The basis keys of twisted degree ``t`` and grade ``G`` (``None``:
        any grade).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._piece_keys(M, 2, None)
            [(((0, -2),), 0), (((0, -1), (0, -1)), 0)]
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: L2 = L0.with_conformal_vector(J.no(J)/4).with_horizontal_grading('lattice'); N = L2.simple_module([1/2])
            sage: VertexAlgebraModule._piece_keys(N, 9/4, N._grade_coords((3/2,)))
            [((), (3/2,))]
        """
        keys = self._keys_of_tdeg(t)
        if G is None:
            return keys
        return [k for k in keys if self._grade_key(k) == G]

    def _piece_basis(self, t, G):
        """
        A basis of the piece ``(t, G)``, as raw dicts.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._piece_basis(1, None)
            [{(((0, -1),), 0): 1}]
        """
        one = self._one_R
        return [{k: one} for k in self._piece_keys(t, G)]

    def _is_simple(self):
        """True if the module is known to be simple (then a nonzero vector
        generates everything); False if not known.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._is_simple(M), va.Lattice(QQ, matrix([[2]])).simple_module([1/2])._is_simple()
            (False, True)
        """
        return False

    def _l0_grading(self):
        r"""
        True if the conformal weights are the generalized eigenvalues of
        ``L_0`` (``V`` has a conformal vector acting accordingly): then
        submodules are graded and the homogeneous components of a vector lie
        in every submodule containing it.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._l0_grading(M), M._l0_grading()
            (False, True)
        """
        return False

    def _generated_by_lowest_weight_space(self):
        """True if the module is known to be generated by its vectors of the
        weight :meth:`_weight_base`.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._generated_by_lowest_weight_space(M), M._generated_by_lowest_weight_space()
            (False, True)
        """
        return False

    def _el_from_dict(self, d):
        """
        The vector of the raw dict ``d`` (no checks).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._el_from_dict({((), 0): 2, (((0, -1),), 0): 0})
            2*v
        """
        return self._from_dict(d, coerce=False, remove_zeros=True)

    def _element_constructor_(self, x):
        """
        Also the vectors of submodules (of submodules ...) of ``self``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: N = M.submodule([L.mode(v, -1)]); x = N(L.mode(v, -1))
            sage: M(x), M(x).parent() is M
            (L_{-1} v, True)
        """
        P = _parent(x)
        if P is not self and _is_submodule_of(P, self):
            return self._from_dict(dict(x._monomial_coefficients), coerce=False)
        return super()._element_constructor_(x)

    def _coerce_map_from_(self, P):
        """
        Submodules (of submodules ...) coerce into the module.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: N = M.submodule([L.mode(v, -1)])
            sage: M.has_coerce_map_from(N), N.has_coerce_map_from(M)
            (True, False)
        """
        if _is_submodule_of(P, self):
            return True
        return super()._coerce_map_from_(P)

    # ------------------------------------------------------------------ #
    #  Submodules, quotients, direct sums                                  #
    # ------------------------------------------------------------------ #

    def submodule(self, gens, names=None):
        r"""
        The submodule generated by the vectors ``gens``: all ``a_(n) v``, ``a``
        in the vertex algebra, ``v`` in ``gens`` (and sums, repeated).  See
        :class:`vertex_algebra_submodule.FinitelyGeneratedSubmodule`.  (This
        is not Sage's linear span of ``gens``.)

        INPUT:

        - ``gens`` -- a list of vectors of ``self`` (inhomogeneous ones are
          split into homogeneous components when the weights are
          ``L_0``-eigenvalues)
        - ``names`` -- (default: ``None``) a list of names of the generators,
          used by :meth:`FinitelyGeneratedSubmodule.express`; ``s0, s1, ...``
          by default

        OUTPUT:

        A :class:`vertex_algebra_submodule.FinitelyGeneratedSubmodule`.

        EXAMPLES:

        The Virasoro Verma module ``M(1/2, 0)``: ``L_{-1} v`` is singular and
        generates a submodule isomorphic to ``M(1/2, 1)``::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: M = Vir.verma_module(0); L = Vir.gen('L')
            sage: N = M.submodule([L.mode(M.highest_weight_vector(), -1)]); N
            Submodule of Verma module of highest weight {'L': 0} over ... generated by [L_{-1} v]
            sage: N.character(7)
            q + q^2 + 2*q^3 + 3*q^4 + 5*q^5 + 7*q^6 + O(q^7)
        """
        from .vertex_algebra_submodule import FinitelyGeneratedSubmodule
        return FinitelyGeneratedSubmodule(self, gens, names)

    def quotient(self, N, names=None):
        r"""
        The quotient module ``self / N`` by a submodule ``N`` (or by the
        submodule generated by a list of vectors); also ``self / N``.  See
        :class:`vertex_algebra_submodule.QuotientVertexAlgebraModule`.

        INPUT:

        - ``N`` -- a submodule of ``self``, or a list of vectors of ``self``
          (then the submodule they generate)
        - ``names`` -- (default: ``None``) names of the generators when ``N``
          is a list, as for :meth:`submodule`

        OUTPUT:

        A :class:`vertex_algebra_submodule.QuotientVertexAlgebraModule`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: M = Vir.verma_module(0); L = Vir.gen('L')
            sage: Q = M / [L.mode(M.highest_weight_vector(), -1)]; Q.character(7)
            1 + q^2 + q^3 + 2*q^4 + 2*q^5 + 4*q^6 + O(q^7)
            sage: Q is M.quotient(M.submodule([L.mode(M.highest_weight_vector(), -1)]))
            True
        """
        from .vertex_algebra_submodule import QuotientVertexAlgebraModule
        if not getattr(N, '_is_vertex_algebra_submodule', False):
            N = self.submodule(N, names)
        return QuotientVertexAlgebraModule(self, N)

    def __truediv__(self, N):
        """
        ``M / N``: the quotient by a submodule or by the submodule generated by
        a list (or tuple) of vectors.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: Q = M / [L.mode(v, -1)]; Q is M.quotient([L.mode(v, -1)])
            True
            sage: M.__truediv__(2)
            NotImplemented
        """
        if getattr(N, '_is_vertex_algebra_submodule', False) or isinstance(N, (list, tuple)):
            return self.quotient(N)
        return NotImplemented

    def maximal_submodule(self):
        r"""
        The maximal proper submodule: the largest (graded) submodule that
        meets the lowest weight space trivially, for a module generated by
        its lowest weight space when that space is known to be simple (one
        dimensional, or the zero modes of the strong generators generate its
        endomorphisms); see :class:`vertex_algebra_submodule.MaximalSubmodule`.
        The zero submodule of a simple module.

        EXAMPLES:

        The Ising model: ``M(1/2, 1/2)`` has the singular vector ``(L_{-2} -
        3/4 L_{-1}^2) v`` at level 2::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: J = Vir.verma_module(1/2).maximal_submodule(); J
            Maximal submodule of Verma module of highest weight {'L': 1/2} over ...
            sage: J.weight_space(5/2).basis()
            [-4/3*L_{-2} v + L_{-1} L_{-1} v]
        """
        if self._is_simple():
            return self.submodule([])
        from .vertex_algebra_submodule import MaximalSubmodule
        return MaximalSubmodule(self)

    def simple_quotient(self):
        r"""
        The simple quotient ``self / self.maximal_submodule()`` (Zhu's
        ``L(U)`` for an induced module with a simple top level ``U``);
        ``self`` if the module is simple.

        EXAMPLES:

        The Ising model ``L(1/2, 1/16)``: the character ``q^{1/16} Π_{n>=1}
        (1 + q^n)``::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: S = Vir.verma_module(1/16).simple_quotient(); S
            Simple quotient of Verma module of highest weight {'L': 1/16} over ...
            sage: S.character(8)
            q^(1/16) + q^(17/16) + q^(33/16) + 2*q^(49/16) + 2*q^(65/16) + 3*q^(81/16) + 4*q^(97/16) + 5*q^(113/16) + O(q^8)
        """
        if self._is_simple():
            return self
        return self.quotient(self.maximal_submodule())

    def direct_sum(self, *others):
        r"""
        The direct sum of ``self`` and ``others`` (modules over the same
        vertex algebra); see
        :class:`vertex_algebra_submodule.DirectSumOfModules`.

        INPUT:

        - ``*others`` -- modules over the vertex algebra of ``self``

        OUTPUT:

        A :class:`vertex_algebra_submodule.DirectSumOfModules` (nested
        direct sums are flattened).

        EXAMPLES:

        The three simple modules of the Ising model; the character of their
        direct sum::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: L0, L1, L2 = [V.verma_module(h).simple_quotient() for h in (0, 1/2, 1/16)]
            sage: S = L0.direct_sum(L1, L2); len(S.summands())
            3
            sage: S.weight_space(2).dimension(), S.weight_space(5/2).dimension()
            (1, 1)
            sage: x = S.embed(2, L2.highest_weight_vector()); L.mode(x, 0)
            1/16*[2] v
        """
        from .vertex_algebra_submodule import DirectSumOfModules
        return DirectSumOfModules((self,) + tuple(others))

    def projective_cover(self, max_weight, max_depth, level=None, degree_bound=None):
        r"""
        The projective cover ``π: P -> self`` of this simple module over a
        quotient of a universal vertex algebra with finite dimensional
        level-``n`` Zhu algebras: ``P`` induced in degree ``n`` from a
        projective indecomposable module of ``A_n``; see
        :func:`~vertex_algebras.vertex_algebra_module_ext.projective_cover`.

        INPUT:

        - ``max_weight`` -- the relations of the quotient of weight at most
          ``max_weight`` give the Zhu algebras and the relations of ``P``
        - ``max_depth`` -- ``P`` is computed in the degrees at most
          ``max_depth``
        - ``level`` -- (default: ``None``) the level ``n``; by default
          from the simple modules of the computed ``A_0``
        - ``degree_bound`` -- (default: ``None``) the degree bound of the
          higher Zhu algebra quotient; by default raised until the
          dimension of ``A_n`` is determined

        OUTPUT:

        The module morphism ``π``, its domain ``P`` an
        :class:`~vertex_algebras.quotient_induced_module.InducedModuleUpTo`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: Q = V / [L.no(L) - 3/10*L.derivative(2)]
            sage: M = Q.induced_module_up_to(Q.zhu_algebra_up_to(6).module({'L': 0}), 6, 4)
            sage: pi = M.simple_quotient().projective_cover(6, 4); pi.codomain() is M.simple_quotient()
            True
        """
        from .vertex_algebra_module_ext import projective_cover
        return projective_cover(self, max_weight, max_depth, level, degree_bound)

    def radical_series_up_to(self, depth, simples=None, submodules=False):
        r"""
        The layers of the radical series of this module up to ``depth``
        (head first), as multiplicities of the modules ``simples`` (default:
        the simple modules that can occur, found from the Zhu algebra of the
        quotient this module is induced over, sorted by lowest weight); with
        ``submodules=True`` also the radicals ``rad^k M`` as submodules.  See
        :func:`~vertex_algebras.vertex_algebra_module_ext.radical_series`.

        INPUT:

        - ``depth`` -- the depth (counted from the base weight of ``self``)
          up to which maps out of the radicals are found
        - ``simples`` -- (default: ``None``) a list of simple modules; by
          default the simple modules that can occur
        - ``submodules`` -- boolean (default: ``False``); whether to return
          the radicals too

        OUTPUT:

        The layers, a list (head first) of lists of multiplicities, one per
        module of ``simples``; with ``submodules=True`` the pair
        ``(submodules, layers)``, ``submodules`` the list of the ``rad^k M``.

        EXAMPLES:

        The projective cover of the trivial module of the triplet algebra
        ``W(2)`` has the Loewy layers ``W(0) | 2 W(1) | W(0)``::

            sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
            sage: Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
            sage: SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp,
            ....:                             horizontal_grading={'cp': 1, 'cm': -1})
            sage: a, b = SF.gens()
            sage: W2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
            ....:     a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
            ....:     names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient', conformal_vector='ambient')
            sage: P = W2.presentation(); Z = P.zhu_algebra_up_to(8)
            sage: W0 = P.induced_module_up_to(Z.module({'T': 0, 'Wp': 0, 'W0': 0, 'Wm': 0}), 8, 5).simple_quotient()
            sage: W1 = P.induced_module_up_to(Z.module({'T': identity_matrix(QQ, 2),
            ....:     'Wp': matrix([[0, 1], [0, 0]]), 'W0': diagonal_matrix([2, -2]),
            ....:     'Wm': matrix([[0, 0], [-4, 0]])}), 8, 5)
            sage: P0 = W0.projective_cover(8, 4).domain()
            sage: P0.radical_series_up_to(4, [W0, W1])
            [[1, 0], [0, 2], [1, 0]]
            sage: P0.socle_series_up_to(4, [W0, W1])
            [[1, 0], [0, 2], [1, 0]]

        The simple modules found by the library (here ``W(0)``, ``W(1)``:
        the composition factors of weights ``0, ..., 4``)::

            sage: P0.radical_series_up_to(4)
            [[1, 0], [0, 2], [1, 0]]

        The radicals as submodules: the radical is generated in weight 1::

            sage: subs, layers = P0.radical_series_up_to(4, [W0, W1], submodules=True)
            sage: [S.weight_space(1).dimension() for S in subs]
            [4, 4, 0]

        Indecomposable modules are quotients of projective ones:
        ``P(0) / rad^2 P(0)`` (here ``rad^2 = soc``) is ``W(0) -> 2 W(1)``
        (weight spaces ``χ_0 + 2 χ_1``); it can be fused.  With
        ``W(-1/8)`` (projective) it gives ``W(-1/8) + 2 W(3/8)``, and ``P(0)``
        gives twice that::

            sage: R = P0.quotient(subs[2]); [R.weight_space(w).dimension() for w in range(5)]
            [1, 4, 5, 8, 13]
            sage: Wm8 = P.simple_modules_up_to(8, 4)[0]; Wm8._weight_base()
            -1/8
            sage: R.fusion_product(Wm8).depth_quotient_up_to(0, 3).conformal_weights()
            [(-1/8, [1]), (3/8, [1, 1, 1, 1])]
            sage: P0.fusion_product(Wm8).depth_quotient_up_to(0, 3).conformal_weights()
            [(-1/8, [1, 1]), (3/8, [1, 1, 1, 1])]
        """
        from .vertex_algebra_module_ext import radical_series
        subs, layers = radical_series(self, depth, simples)
        return (subs, layers) if submodules else layers

    def socle_series_up_to(self, depth, simples=None, submodules=False):
        r"""
        The layers of the socle series of this module up to ``depth``
        (socle first), as multiplicities of the modules ``simples``; with
        ``submodules=True`` also the submodules ``soc^k M``.  See
        :func:`~vertex_algebras.vertex_algebra_module_ext.socle_series` and
        :meth:`radical_series_up_to` (example).

        INPUT:

        - ``depth`` -- the depth up to which the maps from the simple
          modules are checked
        - ``simples`` -- (default: ``None``) a list of simple modules; by
          default as for :meth:`radical_series_up_to`
        - ``submodules`` -- boolean (default: ``False``); whether to return
          the submodules ``soc^k M`` too

        OUTPUT:

        The layers, a list (socle first) of lists of multiplicities, one
        per module of ``simples``; with ``submodules=True`` the pair
        ``(submodules, layers)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: S = V.verma_module(0).simple_quotient()
            sage: S.socle_series_up_to(6, [S]), S.radical_series_up_to(6, [S])
            ([[1]], [[1]])
        """
        from .vertex_algebra_module_ext import socle_series
        subs, layers = socle_series(self, depth, simples)
        return (subs, layers) if submodules else layers

    def ext1_up_to(self, other, depth, cover):
        r"""
        ``Ext^1(self, other)`` up to ``depth``, from a projective cover ``cover:
        P_0 -> self`` (onto; from :meth:`projective_cover`, or ``P_0``
        projective by the user's claim; a cover that is not projective gives
        the extensions whose top splits); see
        :mod:`vertex_algebras.vertex_algebra_module_ext`.

        INPUT:

        - ``other`` -- a module over the vertex algebra of ``self``
        - ``depth`` -- the depth up to which the kernel of ``cover`` is
          generated and its relations imposed
        - ``cover`` -- a module morphism onto ``self`` (its codomain must be
          ``self``, else ``ValueError``)

        OUTPUT:

        An :class:`~vertex_algebras.vertex_algebra_module_ext.Ext1UpTo`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 3/7, conformal_vector='L')
            sage: Z = V.zhu_algebra().quotient([(V.zhu_algebra().gen(0) - 2)^2])
            sage: P = V.induced_module(Z.module({'L': matrix([[2, 1], [0, 2]])}))
            sage: M = V.verma_module(2); pi = P.hom([0, M.highest_weight_vector()], M)
            sage: M.ext1_up_to(M, 3, pi).dimension()
            1
        """
        from .vertex_algebra_module_ext import Ext1UpTo
        if cover.codomain() is not self:
            raise ValueError('{} is not a map onto {}'.format(cover, self))
        return Ext1UpTo(cover, other, depth)

    def fusion_product(self, other):
        r"""
        The fusion (vertex tensor) product ``self ⊠ other`` (modules over
        the same vertex algebra), by the Nahm–Gaberdiel–Kausch algorithm
        [Nah1994]_, [GK1996]_;
        see :mod:`vertex_algebras.fusion_product`.  It computes depth
        quotients (:meth:`~vertex_algebras.fusion_product.FusionProduct.depth_quotient_up_to`).

        INPUT:

        - ``other`` -- a module over the vertex algebra of ``self``

        OUTPUT:

        A :class:`~vertex_algebras.fusion_product.FusionProduct` (cached per
        ``other``).

        EXAMPLES:

        The Ising model, ``σ ⊠ ε = σ``::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: s, e = [V.verma_module(h).simple_quotient() for h in (1/16, 1/2)]
            sage: s.fusion_product(e).depth_quotient_up_to(0, 3).conformal_weights()
            [(1/16, [1])]
            sage: s.fusion_product(e) is s.fusion_product(e)
            True
        """
        from .fusion_product import FusionProduct
        cache = self.__dict__.setdefault('_fusion_products', {})
        P = cache.get(other)
        if P is None:
            P = FusionProduct(self, other)
            cache[other] = P
        return P

    # ------------------------------------------------------------------ #
    #  The action                                                          #
    # ------------------------------------------------------------------ #

    def pullback(self, phi):
        r"""
        The pullback of this ``V``-module along a morphism of vertex
        algebras ``φ: U -> V``: the ``U``-module with ``a_(n) m = φ(a)_(n)
        m`` (e.g. the restriction to a subalgebra along its inclusion); see
        :mod:`module_twists`.

        INPUT:

        - ``phi`` -- a morphism of vertex algebras whose codomain is the
          vertex algebra of ``self``

        OUTPUT:

        A :class:`~vertex_algebras.module_twists.PullbackModule`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
            sage: Vir = va.Virasoro(QQ, 1, conformal_vector='L')
            sage: M = H.adjoint_module().pullback(Vir.hom([H.conformal_vector()], H))
            sage: M.vertex_algebra() is Vir, M.character(4) == H.character(4)
            (True, True)
        """
        from .module_twists import PullbackModule
        return PullbackModule(self, phi)

    def twist(self, sigma):
        r"""
        The twist ``M^σ`` of this module by an endomorphism ``σ`` of its
        vertex algebra: ``a_(n) m = σ(a)_(n) m`` (the pullback along ``σ``).

        INPUT:

        - ``sigma`` -- an endomorphism of the vertex algebra of ``self``
          (else ``ValueError``)

        OUTPUT:

        A :class:`~vertex_algebras.module_twists.PullbackModule`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); J, ep, em = L.gens()
            sage: M = L.simple_module([1/2]); Mt = M.twist(L.hom([em, ep], L))
            sage: v = Mt(M.highest_weight_vector()); J.nproduct(v, 0) == -v
            True
        """
        V = self.vertex_algebra()
        if sigma.domain() is not V or sigma.codomain() is not V:
            raise ValueError('twists are by endomorphisms of {}'.format(V))
        from .module_twists import PullbackModule
        return PullbackModule(self, sigma, name='Twist')

    def spectral_flow(self, h):
        r"""
        The spectral flow of this module by ``h ∈ V`` (H. Li's
        ``Δ``-operator [Li1997]_: ``Y(a, z) = Y_M(Δ(h, z) a, z)``); ``h_(n) h``
        must vanish for ``n >= 0`` except a multiple of the vacuum at ``n =
        1``, and ``h_(0)`` must have integer eigenvalues on ``V``.  See
        :mod:`module_twists`.

        INPUT:

        - ``h`` -- an element of the vertex algebra ``V`` of ``self`` with
          the properties above

        OUTPUT:

        A :class:`~vertex_algebras.module_twists.SpectralFlowModule`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
            sage: S = H.adjoint_module().spectral_flow(3*H.gen(0))
            sage: v = S(H.adjoint_module()(H.one())); H.gen(0).nproduct(v, 0) == 3*v
            True
            sage: S.character(7)
            q^(9/2) + q^(11/2) + 2*q^(13/2) + O(q^7)
        """
        from .module_twists import SpectralFlowModule
        return SpectralFlowModule(self, h)

    def quantum_hamiltonian_reduction(self, f, x=None, reduction='+', names=None):
        r"""
        The quantum Hamiltonian reduction ``H(M)`` of this module, a module
        of the W-algebra ``W^k(g, f)`` (with the conformal vector of the
        reduction), for a Verma module ``M(λ)`` of a universal affine vertex
        algebra (induced from a Verma module of its Zhu algebra) or its
        simple quotient ``L(λ)``.  The answer is given by theorems, not
        computed as cohomology; see
        :mod:`quantum_hamiltonian_reduction_modules` for which ones and for
        the cases that are refused.

        INPUT:

        - ``f`` -- the nilpotent: ``'minimal'``, ``'principal'``, a Jordan
          type, a list of positive roots or an element of ``g``
        - ``x`` -- (default: ``None``) a good grading for ``f`` in the
          Cartan subalgebra; ``None`` for the Dynkin grading
        - ``reduction`` -- (default: ``'+'``) ``'+'`` ([KRW2003]_) or
          ``'-'`` ([Ara2007]_, principal ``f`` of a Lie algebra)
        - ``names`` -- (default: ``None``) names of the generators of the
          W-algebra

        OUTPUT:

        A Verma module of the W-algebra, its simple quotient, or the zero
        module (:class:`vertex_algebra_submodule.ZeroModule`).

        EXAMPLES:

        An ``sl_2`` Verma module reduces to a Virasoro Verma module, here of
        ``c = 0`` and ``h = 1/8``::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2)
            sage: M = V.induced_module(V.zhu_algebra().verma_module({'alphacheck[1]': -1/2}))
            sage: H = M.quantum_hamiltonian_reduction('principal'); H
            Verma module of highest weight {'J{E[-alpha[1]]}': -3/16} over Universal vertex algebra on generators ['J{E[-alpha[1]]}'] over Rational Field
            sage: H.top_weight(), H.vertex_algebra().central_charge()
            (1/8, 0)
        """
        from .quantum_hamiltonian_reduction_modules import reduce_module
        return reduce_module(self, f, x, reduction, names)

    def _Hom_(self, Y, category=None):
        r"""
        The space of module morphisms ``self -> Y`` (Sage's hook for
        ``Hom``); see :mod:`vertex_algebra_module_morphism`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.vertex_algebra_module_morphism import ModuleHomset
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(0)
            sage: isinstance(Hom(M, M), ModuleHomset), End(M) is Hom(M, M)
            (True, True)
        """
        if not getattr(Y, '_is_vertex_algebra_module', False):
            raise TypeError('{} is not a module over a vertex algebra'.format(Y))
        # one homset per codomain (composition asks with the meet of the
        # categories; equal morphisms must have the same parent)
        cache = self.__dict__.setdefault('_homsets', {})
        try:
            return cache[Y]
        except KeyError:
            from .vertex_algebra_module_morphism import ModuleHomset
            H = cache[Y] = ModuleHomset(self, Y)
            return H

    def hom(self, im_gens, codomain=None, check=True, check_depth=None):
        r"""
        The module morphism ``self -> codomain`` sending the generating
        vectors (the top level of an induced module, the vacuum of the
        adjoint module; quotients and twists: those of the module
        underneath) to ``im_gens``; ``check`` verifies that they define a
        morphism (exactly, or up to ``check_depth`` where the relations are
        known degree by degree).  See :mod:`vertex_algebra_module_morphism`.

        INPUT:

        - ``im_gens`` -- the images of the generating vectors (a list or
          tuple, a dict, or a single vector)
        - ``codomain`` -- (default: ``None``) a module; by default the
          parent of the (first) image
        - ``check`` -- boolean (default: ``True``); whether to verify that
          the images define a morphism
        - ``check_depth`` -- (default: ``None``) the depth up to which the
          check runs where it is not exact

        OUTPUT:

        A module morphism, an element of ``Hom(self, codomain)``.

        EXAMPLES:

        The singular vector ``L_{-1} v`` of the Virasoro Verma module ``M(c,
        0)``::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M0 = V.verma_module(0); v = M0.highest_weight_vector()
            sage: f = V.verma_module(1).hom([L.mode(v, -1)], M0); f.im_gens()
            (L_{-1} v,)
        """
        if codomain is None:
            if isinstance(im_gens, (list, tuple)) and im_gens:
                codomain = _parent(im_gens[0])
            elif isinstance(im_gens, dict) and im_gens:
                codomain = _parent(next(iter(im_gens.values())))
            else:
                codomain = _parent(im_gens)
        return Hom(self, codomain)(im_gens, check=check, check_depth=check_depth)

    def find_singular(self, weight, horizontal_grade=None, zero_modes=()):
        r"""
        A basis of the singular vectors of weight ``weight`` (and horizontal
        grade ``horizontal_grade``): the vectors killed by the lowering modes
        ``g_(n)``, ``n > h_g - 1``, of the strong generators ``g`` of the
        vertex algebra (then by all lowering modes), and by the zero modes of
        the elements of ``zero_modes`` (e.g. positive root currents).  The
        space ``Ω`` of the induction adjunction ``Hom(\bar M(U), M) =
        Hom_{A(V)}(U, Ω(M))``.  The weight space must be finite dimensional.

        INPUT:

        - ``weight`` -- a conformal weight of ``self``
        - ``horizontal_grade`` -- (default: ``None``) a grade, or ``None``
          for the whole weight space
        - ``zero_modes`` -- (default: ``()``) elements of the vertex algebra
          whose zero modes must also kill the vectors

        OUTPUT:

        A list of vectors, the reduced echelon form (canonical).

        EXAMPLES:

        The singular vectors of the Virasoro Verma modules ``M(1/2, 0)`` at
        weights 1 and 2 and ``M(1/2, 1/2)`` at weight ``5/2``::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(0)
            sage: M.find_singular(1), M.find_singular(2)
            ([L_{-1} v], [])
            sage: V.verma_module(1/2).find_singular(5/2)
            [-4/3*L_{-2} v + L_{-1} L_{-1} v]

        Affine ``sl_2`` at level 1, the doublet: ``F_0 u`` is killed by the
        lowering modes but not by ``E_0``::

            sage: V = va.Affine(QQ, 'A1', 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: A = V.zhu_algebra(); E = V.gen('E')
            sage: U = A.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: M = V.induced_module(U)
            sage: M.find_singular(1/4), M.find_singular(1/4, zero_modes=[E])
            ([u[0], u[1]], [u[0]])
        """
        from sage.functions.other import floor

        from ._keyed_spans import KeyedSpan, left_kernel
        V = self._V
        R = self.base_ring()
        S = self.weight_space(weight, horizontal_grade)
        if not S.is_finite_dimensional():
            raise ValueError('the weight space of weight {} is infinite: give a horizontal '
                             'grade'.format(weight))
        basis = [dict(b._monomial_coefficients) for b in S.basis()]
        data = V._weight_data()
        ops = []
        for g in V._strong_generators_for_ideals():
            hs = {V._weight_key(k, data) for k in g.support()}
            if len(hs) != 1:
                raise ValueError('the strong generator {} is not homogeneous'.format(g))
            ops.append((dict(g._monomial_coefficients), floor(QQ(hs.pop()) - 1) + 1, None))
        for z in zero_modes:
            z = self._vertex_algebra_element(z)
            hs = {V._weight_key(k, data) for k in z.support()}
            h = QQ(hs.pop()) if len(hs) == 1 else None
            if h is None or not h.is_integer():
                raise ValueError('{} is not homogeneous of integer weight'.format(z))
            ops.append((dict(z._monomial_coefficients), h - 1, h - 1))
        images = []
        for b in basis:
            img = {}
            for j, (a, nmin, nfix) in enumerate(ops):
                nmax = nfix if nfix is not None else max(
                    (self._nmax_key(x, y) for x in a for y in b), default=nmin - 1)
                for n in range(nmin, nmax + 1):
                    for k, c in self._action_raw(a, int(n), b).items():
                        if c:
                            img[(j, n, k)] = c
            images.append(img)
        kernel = []
        for rel in left_kernel(images, R):
            v = {}
            for j, c in rel.items():
                _add_scaled(v, basis[j], c)
            kernel.append({k: c for k, c in v.items() if c})
        E = KeyedSpan(R, key_order=self._key_order)
        if kernel:
            E.insert_many(kernel)
        rows = sorted(E.rows(), key=lambda r: max(self._key_order(k) for k in r))
        return [self._from_dict(r) for r in rows]

    def _action_mono(self, a, n, m):
        """
        Cached :meth:`_action_key`.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._action_mono(((0, 0),), 0, ((), 0)), M._action_mono(((0, 0),), 1, ((), 0))
            ({(((0, -1),), 0): 1}, {((), 0): 1/16})
        """
        key = (a, n, m)
        r = self._act_cache.get(key)
        if r is None:
            r = self._action_key(a, n, m)
            self._act_cache[key] = r
        return r

    def _action_raw(self, dA, n, dM):
        """
        ``A_(n) M`` for raw dicts ``A`` (of ``V``) and ``M``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._action_raw(M, {((0, 0),): 1}, -1, {((), 0): 2})
            {(((0, -2),), 0): 2}
        """
        res = {}
        for a, ca in dA.items():
            for m, cm in dM.items():
                _add_scaled(res, self._action_mono(a, n, m), ca * cm)
        return res

    def _action_raw_batch(self, els, vecs, tasks):
        r"""
        ``[els[e]_(n) vecs[v] for (e, n, v) in tasks]`` (raw dicts) in one
        call of a C++ engine, or ``None`` (no engine for this module; the
        caller computes them with :meth:`_action_raw`).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import VertexAlgebraModule
            sage: VertexAlgebraModule._action_raw_batch(M, [{((0, 0),): 1}], [{((), 0): 1}], [(0, -1, 0)]) is None
            True
        """
        return None

    def _vertex_algebra_element(self, a):
        """``a`` as an element of the vertex algebra (also from the same
        algebra with other conformal data, or a scalar).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: x = M._vertex_algebra_element(3); M._vertex_algebra_element(L) is L, x.parent() is V, x == 3*V.one()
            (True, True, True)
        """
        V = self._V
        if _parent(a) is V:
            return a
        return V(a)

    def _vector(self, m):
        """
        ``m`` as a vector of the module (converted if needed).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._vector(v) is v, M._vector(M.submodule([v])(v)).parent() is M
            (True, True)
        """
        return m if _parent(m) is self else self(m)

    def _action(self, a, n, m):
        r"""
        ``a_(n) m`` for ``a`` in the vertex algebra, ``n`` an integer and
        ``m`` in the module; also ``a.nproduct(m, n)``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])); M = L.adjoint_module()
            sage: M._action(L.J(0), 0, M(L.e((1,)))) == L.J(0).nproduct(M(L.e((1,))), 0)
            True
        """
        a = self._vertex_algebra_element(a)
        m = self._vector(m)
        if a.is_zero() or m.is_zero():
            return self.zero()
        return self._from_dict(self._action_raw(dict(a._monomial_coefficients), int(n),
                                                dict(m._monomial_coefficients)),
                               coerce=False, remove_zeros=True)

    def _ope(self, a, m):
        r"""
        The list whose entry ``k`` is ``a_(k) m``, up to the last nonzero one;
        also ``a.ope(m)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._ope(L, v), L.ope(v)
            ([L_{-1} v, 1/16*v], [L_{-1} v, 1/16*v])
            sage: M._ope(V.zero(), v)
            []
        """
        a = self._vertex_algebra_element(a)
        m = self._vector(m)
        if a.is_zero() or m.is_zero():
            return []
        dA, dM = dict(a._monomial_coefficients), dict(m._monomial_coefficients)
        N = max(self._nmax_key(x, y) for x in dA for y in dM)
        out = [self._from_dict(self._action_raw(dA, k, dM), coerce=False, remove_zeros=True)
               for k in range(N + 1)]
        while out and out[-1].is_zero():
            out.pop()
        return out

    def _mode(self, n, a, m):
        r"""
        The mode ``a_n m`` in the physics convention, ``a_(n + Δ_a - 1) m``
        (``Δ_a`` the conformal weight of ``a``, ``n ∈ ZZ - Δ_a``); also
        ``a.mode(m, n)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._mode(-2, L, v), M._mode(1, L, L.mode(v, -1))
            (L_{-2} v, 1/8*v)
            sage: M._mode(1/2, L, v)
            Traceback (most recent call last):
            ...
            ValueError: the modes a_n of a = L (conformal weight 2) need n in ZZ, not n = 1/2
        """
        V = self._V
        a = self._vertex_algebra_element(a)
        if a.is_zero():
            return self.zero()
        h = QQ(V.degree(a))
        k = QQ(n) + h - 1
        if not k.is_integer():
            raise ValueError('the modes a_n of a = {} (conformal weight {}) need n in '
                             'ZZ{}, not n = {}'.format(
                                 a, h, '' if h.is_integer()
                                 else ' + {}'.format(-h - (-h).floor()), n))
        return self._action(a, int(k), m)

    def _get_action_(self, S, op, self_on_left):
        """
        The action of ``*`` of the vertex algebra on the left, which
        refuses ``a * m`` with a pointer to ``a.no(m)`` (a Sage error
        would only say that ``*`` is not defined).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._get_action_(V, operator.mul, False)
            Left action by ...
            sage: M._get_action_(V, operator.mul, True) is None
            True
        """
        if op is operator.mul and not self_on_left and S is self._V:
            return _NormallyOrderedAction(S, self)
        return super()._get_action_(S, op, self_on_left)


class _GeneratorActions:
    r"""
    Mixin for modules over a universal vertex algebra: ``_action_key`` for
    all basis monomials of ``V`` from ``_generator_action_key(i, n, m)`` (the
    action of the ``i``-th strong generator), by ``(∂^k g)_(n) = (-1)^k n
    (n-1) ... (n-k+1) g_(n-k)`` and the normally ordered product (a basis
    monomial ``(x, B...)`` of ``V`` is ``:x B:``, ``x`` a letter)::

        (:x B:)_(n) v = Σ_{j>=0} x_(-1-j) B_(n+j) v + (-1)^{|x||B|} Σ_{j>=0} B_(n-1-j) x_(j) v,

    the sums cut off by ``_nmax_key``.

    TESTS::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
        sage: A = V.adjoint_module()
        sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
        sage: B = _AdjointByGenerators(V)
        sage: keys = [k for w in range(5) for k in V.weight_space(w).basis_keys()]
        sage: nz = lambda d: {k: c for k, c in d.items() if c}
        sage: all(nz(B._action_key(a, n, m)) == nz(A._action_key(a, n, m))
        ....:     for a in keys for m in keys for n in range(-2, 4))
        True

    The same for odd generators, half-integer weights, a weight-0 generator
    and non-linear OPEs (``N = 1``, ``bc``, ``βγ``, ``W_3``, affine
    ``sl_2``)::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: def by_generators(V, weights, M=None):
        ....:     M = _AdjointByGenerators(V) if M is None else M
        ....:     keys = [k for w in weights for k in V.weight_space(w).basis_keys()]
        ....:     return all(M._action_mono(a, n, b) == V._nprod_mono(a, n, b)
        ....:                for a in keys for b in keys for n in range(-3, 4))
        sage: N1 = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L')
        sage: BC = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
        ....:                             parities={'b': 1, 'c': 1}, degrees={'b': 1, 'c': 0})
        sage: BG = UniversalVertexAlgebra(QQ, ['be', 'ga'], {('be', 'ga'): [1]},
        ....:                             degrees={'be': 1/2, 'ga': 1/2})
        sage: (by_generators(N1, [3/2, 2, 5/2, 4]), by_generators(BC, [0, 1, 2]),
        ....:  by_generators(BG, [1/2, 1, 3/2]), by_generators(va.W3(QQ, 2, conformal_vector='L'), [2, 3, 4]),
        ....:  by_generators(va.Affine(QQ, ['A', 1], 1), [1, 2]))
        (True, True, True, True, True)

    A negative control: without the sign ``(-1)^{|x||B|}`` the recursion
    differs (``:∂G G:`` of ``N = 1``, ``:b c:``)::

        sage: from vertex_algebras.vertex_algebra_module import _GeneratorActions, _add_scaled
        sage: class WrongSign(_AdjointByGenerators):
        ....:     def _action_key(self, a, n, m):
        ....:         if len(a) < 2:
        ....:             return _GeneratorActions._action_key(self, a, n, m)
        ....:         x, B = a[:1], a[1:]; res = {}
        ....:         for j in range(max(self._nmax_key(B, m) - n + 1, 0)):
        ....:             for mm, c in self._action_mono(B, n + j, m).items():
        ....:                 _add_scaled(res, self._action_mono(x, -1 - j, mm), c)
        ....:         for j in range(self._nmax_key(x, m) + 1):
        ....:             for mm, c in self._action_mono(x, j, m).items():
        ....:                 _add_scaled(res, self._action_mono(B, n - 1 - j, mm), c)
        ....:         return res
        sage: by_generators(N1, [3/2, 4], WrongSign(N1)), by_generators(BC, [0, 1], WrongSign(BC))
        (False, False)

    The module of ``W_3`` acting through its generators passes the module
    tests (the Borcherds identities with a non-linear OPE)::

        sage: TestSuite(_AdjointByGenerators(va.W3(QQ, 2, conformal_vector='L'))).run()

    Only for universal vertex algebras::

        sage: _AdjointByGenerators(va.Lattice(QQ, matrix([[2]])))
        Traceback (most recent call last):
        ...
        TypeError: Lattice vertex algebra of rank 1 with Gram matrix [2] over Rational Field is not a universal vertex algebra
    """

    def _action_key(self, a, n, m):
        """
        ``a_(n) m`` for a basis monomial ``a`` of ``V`` (raw dict), from the
        generator actions by the rules of the class.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
            sage: B = _AdjointByGenerators(V); a = ((0, 1), (0, 0))      # :dL L:
            sage: B._action_key(a, 1, ((0, 0),)) == V._nprod_mono(a, 1, ((0, 0),))
            True
            sage: B._action_key((), -1, ((0, 0),)), B._action_key(((0, 2),), 0, ((0, 0),))
            ({((0, 0),): 1}, {})
        """
        if not a:
            return {m: self._one_R} if n == -1 else {}
        if len(a) == 1:
            g, k = a[0]
            c = (-1) ** k * _falling(n, k)
            if not c:
                return {}
            return {mm: c * v for mm, v in self._generator_action_key(g, n - k, m).items()}
        V = self._V
        x, B = a[:1], a[1:]
        s = (-1) ** (V._parity_key(x) * V._parity_key(B))
        res = {}
        for j in range(max(self._nmax_key(B, m) - n + 1, 0)):
            for mm, c in self._action_mono(B, n + j, m).items():
                _add_scaled(res, self._action_mono(x, -1 - j, mm), c)
        for j in range(self._nmax_key(x, m) + 1):
            for mm, c in self._action_mono(x, j, m).items():
                _add_scaled(res, self._action_mono(B, n - 1 - j, mm), s * c)
        return res


class AdjointModule(VertexAlgebraModule):
    r"""
    A vertex algebra ``V`` as a module over itself (``V.adjoint_module()``):
    the same basis, ``a_(n) m`` the ``n``-th product of ``V``, the same
    conformal weights and grading.  ``M(v)`` is the vector of ``v ∈ V``.

    INPUT:

    - ``V`` -- a vertex algebra spanned by its basis keys (not a
      subalgebra or a quotient)

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: X.<b, c> = NormallyOrderedExpressions(QQ, parities={'b': 1, 'c': 1})
        sage: BC = UniversalVertexAlgebra(QQ, X, {('b', 'c'): [1]}, degrees={'b': 1, 'c': 0})
        sage: M = BC.adjoint_module(); M.category()
        Category of conformally graded super modules with basis over Universal vertex algebra on generators ['b', 'c'] over Rational Field
        sage: b, c = BC.gens()
        sage: m = M(c); b.ope(m), b.no(m).parity(), M(b) is b
        ([|0>], 0, False)
        sage: M is BC.adjoint_module(), loads(dumps(M)) is M
        (True, True)
        sage: TestSuite(M).run()

    Subalgebras and quotients are not spanned by basis keys::

        sage: E.<L> = NormallyOrderedExpressions(QQ)
        sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
        ....:                              conformal_vector=L)
        sage: L = Vir.gen('L')
        sage: (Vir / [L.no(L) - 3/10*L.derivative(2)]).adjoint_module()
        Traceback (most recent call last):
        ...
        NotImplementedError: the adjoint module needs a vertex algebra spanned by its basis keys (not a subalgebra or a quotient)

    TESTS:

    The action (by ``ope``, ``*``, ``mode``) is the products of ``V``; the
    module axioms (``TestSuite``) and the characters, for ``N = 1``, the odd
    lattice ``Z`` and ``Vir ⊗ V_{A_1}``::

        sage: import vertex_algebras as va
        sage: def acts_as_V(V, weights):
        ....:     M = V.adjoint_module()
        ....:     B = [V.monomial(m) for w in weights for m in V.weight_space(w).basis_keys()][:8]
        ....:     return all(x.ope(M(y)) == [M(p) for p in x.ope(y)] and x.no(M(y)) == M(x.no(y))
        ....:                and x.mode(M(y), 1 - x.degree()) == M(x.mode(y, 1 - x.degree()))
        ....:                for x in B for y in B)
        sage: N1 = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L')
        sage: LZ = va.Lattice(QQ, matrix([[1]])); J = LZ.J(0); VZ = LZ.with_conformal_vector(J.no(J)/2)
        sage: LA = va.Lattice(QQ, matrix([[2]])); J = LA.J(0); VA = LA.with_conformal_vector(J.no(J)/4)
        sage: TP = va.Virasoro(QQ, 1/2, conformal_vector='L').tensor(VA, conformal_vector='factors')
        sage: acts_as_V(N1, [3/2, 2, 5/2]), acts_as_V(VZ, [1/2, 1, 3/2]), acts_as_V(TP, [1, 2])
        (True, True, True)
        sage: for V in (N1, VZ, TP):
        ....:     TestSuite(V.adjoint_module()).run()
        sage: [V.adjoint_module().character(4) == V.character(4) for V in (N1, VZ, TP)]
        [True, True, True]

    Elements of ``V`` with other conformal data act by conversion; elements
    of another vertex algebra and mode indices outside ``ZZ - Δ`` are
    refused::

        sage: Vir = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = Vir.adjoint_module()
        sage: m = M(Vir.gen('L')); Vir.with_conformal_vector(None).gen('L').nproduct(m, 1) == 2*m
        True
        sage: va.W3(QQ, 2, conformal_vector='L').gen('L').nproduct(m, 0)
        Traceback (most recent call last):
        ...
        TypeError: do not know how to make x (= L) an element of self (=Universal vertex algebra on generators ['L'] over Rational Field)
        sage: Vir.gen('L').mode(m, 1/2)
        Traceback (most recent call last):
        ...
        ValueError: the modes a_n of a = L (conformal weight 2) need n in ZZ, not n = 1/2
    """

    def __init__(self, V):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); A = V.adjoint_module()
            sage: A._indices is V._indices, A._is_graded_module, A._mgrading
            (True, True, None)
            sage: L = V.gen('L'); V.subalgebra([L.no(L)]).adjoint_module()
            Traceback (most recent call last):
            ...
            NotImplementedError: the adjoint module needs a vertex algebra spanned by its basis keys (not a subalgebra or a quotient)
        """
        if (getattr(V, '_is_subspace_algebra', False)
                or getattr(V, '_is_quotient_vertex_algebra', False)):
            raise NotImplementedError('the adjoint module needs a vertex algebra spanned '
                                      'by its basis keys (not a subalgebra or a quotient)')
        try:
            V._weight_data()
            graded = True
        except ValueError:
            graded = False
        if V._grading is not None:
            from sage.matrix.constructor import matrix

            from ._grading_groups import GradingEmbedding
            A = V._grading['group']
            self._mgrading = GradingEmbedding(A, A, matrix.identity(ZZ, A.k))
        VertexAlgebraModule.__init__(self, V, V._indices, super=V._has_odd(), graded=graded)

    def _repr_(self):
        """
        The string representation.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: repr(A)
            "Adjoint module of Universal vertex algebra on generators ['L'] over Rational Field"
        """
        return 'Adjoint module of {}'.format(self._V)

    def _repr_term(self, m):
        """
        The string of a basis key: ``V``'s, the vacuum as ``|0>``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._repr_term(()), A._repr_term(((0, 1), (0, 0)))
            ('|0>', ':dL L:')
        """
        V = self._V
        return '|0>' if m == V._vacuum_key() else V._repr_term(m)

    def _ascii_art_term(self, m):
        """
        The ascii art of a basis key (its string).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._ascii_art_term(((0, 1),)), ascii_art(A(L.derivative()))
            (dL, dL)
        """
        from sage.typeset.ascii_art import AsciiArt
        return AsciiArt([self._repr_term(m)])

    def _unicode_art_term(self, m):
        """
        The unicode art of a basis key: ``V``'s, the vacuum as ``|0⟩``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._unicode_art_term(()), unicode_art(A(L.derivative()))
            (|0⟩, ∂ L)
        """
        from sage.typeset.unicode_art import UnicodeArt
        V = self._V
        return UnicodeArt(['|0⟩' if m == V._vacuum_key() else V._term_str(m, 'unicode')])

    def _latex_term(self, m):
        r"""
        The LaTeX of a basis key: ``V``'s, the vacuum as ``|0\rangle``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._latex_term(()), latex(A(L.derivative()))
            ('|0\\rangle', \partial L)
        """
        V = self._V
        return '|0\\rangle' if m == V._vacuum_key() else V._term_str(m, 'latex')

    def _action_key(self, a, n, m):
        """
        ``a_(n) m``: the ``n``-th product of ``V``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._action_key(((0, 0),), 1, ((0, 0),))
            {((0, 0),): 2}
        """
        return self._V._nprod_mono(a, n, m)

    def _nmax_key(self, a, m):
        """
        ``V``'s bound.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._nmax_key(((0, 0),), ((0, 0),)), len(L.ope(L)) - 1
            (3, 3)
        """
        return self._V._nmax_key(a, m)

    def _parity_key(self, m):
        """
        ``V``'s parity.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._parity_key(((0, 0),))
            0
            sage: NS = va.NeveuSchwarz(QQ, 3/2); NS.adjoint_module()._parity_key(((1, 0),))
            1
        """
        return self._V._parity_key(m)

    def _weight_key(self, m):
        """
        ``V``'s conformal weight.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._weight_key(((0, 1), (0, 0)))
            5
        """
        V = self._V
        return V._weight_key(m, V._weight_data())

    def _weight_space_enumeration(self, W, G):
        """
        ``V``'s enumeration of the keys of weight ``W`` and grade ``G``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: sorted(A._weight_space_enumeration(4, None).factory())
            [((0, 0), (0, 0)), ((0, 2),)]
        """
        V = self._V
        return V._weight_space_enumeration(W, G, V._weight_data())

    def _weight_bounds(self, G):
        """
        ``V``'s bounds ``(lb, D)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._weight_bounds(None)
            (0, 1)
        """
        V = self._V
        return V._character_bounds(G, V._weight_data())

    def _grade_key(self, m):
        """
        ``V``'s grade of a key.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: L2 = va.Lattice(QQ, matrix([[2]])).with_horizontal_grading('lattice')
            sage: L2.adjoint_module()._grade_key(((), (3,)))
            (3,)
        """
        return self._V._grade_key(m)

    def _key_order(self, m):
        """
        ``V``'s key order.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._key_order(((0, 1),)) == V._key_order(((0, 1),))
            True
        """
        return self._V._key_order(m)

    def _is_simple(self):
        """
        Whether ``V`` is known to be simple.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._is_simple(), va.Lattice(QQ, matrix([[2]])).adjoint_module()._is_simple()
            (False, True)
        """
        # a simple V containing a conformal vector (T_(0) = ∂, so left ideals
        # are two-sided) is simple as a module over itself; the lattice
        # vertex algebras contain the Sugawara vector
        return self._V._is_simple()

    def _l0_grading(self):
        """
        True iff ``V`` has a conformal vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._l0_grading(), va.Virasoro(QQ, 1/2).with_degrees({'L': 2}).adjoint_module()._l0_grading()
            (True, False)
        """
        return self._V.conformal_vector() is not None

    def _generated_by_lowest_weight_space(self):
        """
        True iff the lowest weight is ``0`` (the vacuum generates ``V``).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A._generated_by_lowest_weight_space()
            True
        """
        # V is generated by the vacuum, of weight 0
        return self._weight_base() == 0

    def _element_constructor_(self, x):
        """``M(v)`` for ``v`` in ``V`` (or converting into ``V``); otherwise as
        for any ``CombinatorialFreeModule``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A(L), A(L).parent() is A, A(2)
            (L, True, 2*|0>)
        """
        V = self._V
        P = _parent(x)
        if P is not self and (P is V or V.has_coerce_map_from(P)
                              or getattr(P, '_is_vertex_algebra', False)):
            v = V(x)
            return self._from_dict(dict(v._monomial_coefficients), coerce=False)
        return super()._element_constructor_(x)

    def some_elements(self):
        r"""
        Some vectors: the images of ``some_elements()`` of the vertex algebra.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: V.adjoint_module().some_elements()
            [|0>, L, dL]
        """
        return [self(x) for x in self._V.some_elements()]

    def _an_element_(self):
        """
        The image of ``V``'s element.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: A.an_element()
            L
        """
        return self(self._V.an_element())


class _AdjointByGenerators(_GeneratorActions, AdjointModule):
    r"""
    The adjoint module of a universal vertex algebra with the action
    computed by ``_GeneratorActions`` from the products of the
    generators only (a check of the recursion against ``V``'s products;
    the bounds from the conformal weights).

    EXAMPLES::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
        sage: J = H.gen('J'); M = _AdjointByGenerators(H)
        sage: x = J.no(J.derivative()); x.ope(M(x.no(J))) == [M(p) for p in x.ope(x.no(J))]
        True
    """

    def __init__(self, V):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
            sage: _AdjointByGenerators(H).vertex_algebra() is H
            True
            sage: import vertex_algebras as va
            sage: _AdjointByGenerators(va.Lattice(QQ, matrix([[2]])))
            Traceback (most recent call last):
            ...
            TypeError: Lattice vertex algebra ... is not a universal vertex algebra
        """
        if not getattr(V, '_is_universal_vertex_algebra', False):
            raise TypeError('{} is not a universal vertex algebra'.format(V))
        AdjointModule.__init__(self, V)

    def _repr_(self):
        """
        The string representation.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
            sage: _AdjointByGenerators(V)
            Adjoint module, acting through the generators, of Universal vertex algebra on generators ['L'] over Rational Field
        """
        return 'Adjoint module, acting through the generators, of {}'.format(self._V)

    def _generator_action_key(self, i, n, m):
        """
        The ``i``-th generator acting by ``V``'s products.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: A = V.adjoint_module()
            sage: from vertex_algebras.vertex_algebra_module import _AdjointByGenerators
            sage: _AdjointByGenerators(V)._generator_action_key(0, 1, ((0, 0),))
            {((0, 0),): 2}
        """
        return self._V._nprod_mono(((i, 0),), n, m)

    _nmax_key = VertexAlgebraModule._nmax_key


# ---------------------------------------------------------------------- #
#  Induced modules (Zhu's universal positive energy modules)             #
# ---------------------------------------------------------------------- #

_NOT_BUILT = object()


class _InducedModuleEngine:
    r"""
    The C++ engine (``_module_engine``) of an :class:`InducedModule` with a
    finite top level, over the coefficient ring of the vertex algebra's
    engine (its :class:`vertex_algebra._EngineKernel` converts the
    coefficients, also through a lift such as ``Frac(A) -> A``): the
    generators (parities, weights times ``D``), all ordered generator OPEs,
    the top (parities, the zero modes of the integer-weight generators).
    Module keys cross as engine ids (both directions cached here); a
    coefficient the engine's ring cannot hold makes :meth:`act_many` return
    ``None`` (the caller takes the Python path).

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
        sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
        sage: E = M._engine()
        sage: E is None or isinstance(E._E, object) and E.act_many([{((0, 0),): 1}], [{((), 0): 1}], [(0, 1, 0)], 1) == [{((), 0): 1/16}]
        True

    TESTS:

    The engine against the Python recursion on every key up to a depth,
    one product at a time and as a batch: Virasoro over ``QQ`` and over
    ``Q(c, h)`` (lifted), N=1 (an odd generator, half-integer weights),
    ``W_3`` over ``Q(c)`` (lifted with ``u_0 = 1/(5c + 22)``) and affine
    `\mathfrak{sl}_2` over ``Q(k)`` with the doublet top (zero modes by
    matrices)::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.vertex_algebra_module import InducedModule
        sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
        sage: def compare(V, U, els, depth):
        ....:     M, MP = InducedModule(V, U), InducedModule(V, U, implementation='python')
        ....:     vecs = [{k: 1} for d in range(depth + 1) for k in M._keys_of_tdeg(d)]
        ....:     els = [dict(x.monomial_coefficients()) for x in els]
        ....:     tasks = [(e, n, v) for v in range(len(vecs)) for e in range(len(els))
        ....:              for n in range(-2, 4)]
        ....:     single = [MP._action_raw(els[e], n, vecs[v]) for e, n, v in tasks]
        ....:     assert [M._action_raw(els[e], n, vecs[v]) for e, n, v in tasks] == single
        ....:     b = M._action_raw_batch(els, vecs, tasks)
        ....:     assert b is None or b == single
        ....:     return M._engine() is not None
        sage: used = []
        sage: L = va.Virasoro(QQ, 1/2, conformal_vector='L').gen('L')
        sage: used.append(compare(L.parent(), L.parent().zhu_algebra().module({'L': 1/16}),
        ....:                     [L, L.no(L), L.derivative(2).no(L) + 3*L.no(L.no(L))], 4))
        sage: K.<c, h> = Frac(QQ['c', 'h'])
        sage: L = va.Virasoro(K, c, conformal_vector='L').gen('L')
        sage: used.append(compare(L.parent(), L.parent().zhu_algebra().module({'L': h}),
        ....:                     [L, c*L.no(L) - h*L.derivative(2)], 3))
        sage: N.<L, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
        sage: N1 = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 7/20],
        ....:     ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 7/15]}, conformal_vector=L)
        sage: L, G = N1.gens()
        sage: used.append(compare(N1, N1.zhu_algebra().module({'L': 1/10}),
        ....:                     [G, L, G.no(G.derivative()), L.no(G)], 3))
        sage: c = Frac(QQ['c']).gen()
        sage: W = va.W3(c.parent(), c, conformal_vector='L')
        sage: L, Wg = W.gens()
        sage: used.append(compare(W, W.zhu_algebra().module({'L': 1/3, 'W': 2}),
        ....:                     [Wg, L.no(Wg), Wg.no(Wg)], 2))
        sage: k = Frac(QQ['k']).gen()
        sage: A = UniversalAffineVertexAlgebra(k.parent(), ['A', 1], level=k, names=['E', 'H', 'F'],
        ....:                                  conformal_vector='sugawara')
        sage: e, hh, f = A.gens()
        sage: U = A.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:                             'F': matrix([[0, 0], [1, 0]])})
        sage: used.append(compare(A, U, [e, f, e.no(f), hh.no(hh.derivative())], 2))
        sage: from vertex_algebras.vertex_algebra import _compiled_module
        sage: used == [_compiled_module('_module_engine')[0] is not None] * 5
        True

    Threads give the serial results in the serial order; Ctrl-C (here an
    alarm) interrupts a long batch, serially and with threads, and the
    module computes as before afterwards::

        sage: W = va.W3(QQ, 2, conformal_vector='L')
        sage: L, Wg = W.gens()
        sage: M = InducedModule(W, W.zhu_algebra().module({'L': 1/3, 'W': 2}))
        sage: els = [dict(x.monomial_coefficients()) for x in (Wg, Wg.no(Wg), L.no(L.no(Wg)))]
        sage: def batch(depth):
        ....:     vecs = [{k: 1} for d in range(depth + 1) for k in M._keys_of_tdeg(d)]
        ....:     tasks = [(e, n, v) for v in range(len(vecs)) for e in range(3) for n in range(-2, 4)]
        ....:     r = M._action_raw_batch(els, vecs, tasks)
        ....:     return None if r is None else [list(x.items()) for x in r]
        sage: ref = batch(3)
        sage: for n in (1, 2):  # long time
        ....:     M.clear_caches()
        ....:     VertexAlgebraParallelism().set(n)
        ....:     try:
        ....:         alarm(0.3)
        ....:         _ = batch(7)
        ....:     except KeyboardInterrupt:
        ....:         pass
        ....:     finally:
        ....:         cancel_alarm()
        ....:     assert batch(3) == ref
        sage: VertexAlgebraParallelism().reset()

    Pickling, and the engine refused for an infinite top::

        sage: M = InducedModule(W, W.zhu_algebra().module({'L': 1/3, 'W': 2}))
        sage: loads(dumps(M)) is M
        True
        sage: InducedModule(A, A.zhu_algebra().verma_module({'H': 1}), implementation='engine')
        Traceback (most recent call last):
        ...
        ValueError: ...
    """

    def __init__(self, M, K, mod):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: M = V.verma_module(1/16); E = M._engine()
            sage: E is None or (E._D, E._top, E._F is QQ) == (1, [0], True)
            True
        """
        V = M._V
        G = len(M._h)
        D = int(M._D)
        self._D = D
        self._K = K
        self._F = M.base_ring()
        E = mod.ModuleEngine(K.kind(), G, list(M._odd), [int(h * D) for h in M._h], D,
                             K.nvars(), K.modulus(), K.minpoly())
        U = M._U
        lowered = getattr(M, '_engine_lowered_top', None)
        top = list(U._top_keys()) if lowered is None else lowered()
        self._top = top
        self._top_index = {u: k for k, u in enumerate(top)}
        E.set_top(len(top), [U._parity_key(u) for u in top])
        self._E = E
        self._ids = {}
        self._keys = {}
        cin, up = K._cin, K._up
        if lowered is None:
            for i, z in M._zidx.items():
                E.set_zero_mode(i, [[(self._top_index[s], cin(up(c)))
                                     for s, c in U._gen_action_key(z, u).items() if c]
                                    for u in top])
        else:
            # a lowered top (HigherInducedModule): its degrees and the modes
            # m >= 0 on it, from the Python recursion (finitely many)
            E.set_top_degrees([int(U._tdeg(u) * D) for u in top])
            for k, u in enumerate(top):
                for i in range(G):          # integer weights (higher Zhu algebras)
                    for m in range(int(U._tdeg(u)) + 1):
                        d = M._gen_act(i, m, ((), u))
                        if d:
                            E.set_top_action(i, m * D, k, [(self.key_id(kk), cin(up(c)))
                                                           for kk, c in d.items() if c])
        for i in range(G):
            for j in range(G):
                E.set_ope(i, j, [K._el_up(d) for d in V._ope_mono(((i, 0),), ((j, 0),))])

    def key_id(self, key):
        """
        The engine id of a module key (interned on first use).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: E = M._engine(); k = (((0, -2), (0, -1)), 0)
            sage: E is None or E.key(E.key_id(k)) == k and E.key_id(k) == E.key_id(k)
            True
        """
        r = self._ids.get(key)
        if r is None:
            modes, u = key
            r = self._E.intern_key([(i, int(m * self._D)) for i, m in modes], self._top_index[u])
            self._ids[key] = r
            self._keys[r] = key
        return r

    def key(self, r):
        """
        The module key of an engine id.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: E = M._engine()
            sage: E is None or E.key(E.key_id(((), 0))) == ((), 0)
            True
        """
        k = self._keys.get(r)
        if k is None:
            modes, u = self._E.key_of(r)
            k = (tuple((i, QQ(m) / self._D) for i, m in modes), self._top[u])
            self._keys[r] = k
            self._ids[k] = r
        return k

    def act_many(self, els, vecs, tasks, nthreads):
        """``[els[e]_(n) vecs[v] for (e, n, v) in tasks]`` (raw dicts), or
        ``None`` if a coefficient cannot be lifted.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: E = M._engine()
            sage: E is None or E.act_many([{((0, 0),): 1}], [{((), 0): 2}], [(0, -1, 0), (0, 1, 0)], 1) == [
            ....:     {(((0, -2),), 0): 2}, {((), 0): 1/8}]
            True
        """
        K, F = self._K, self._F
        cin, up = K._cin, K._up
        try:
            E = [[(w, cin(up(F(c)))) for w, c in d.items() if c] for d in els]
            Vs = [[(self.key_id(k), cin(up(F(c)))) for k, c in d.items() if c] for d in vecs]
        except (ValueError, ArithmeticError, TypeError, AttributeError):
            return None
        out = self._E.act_many(E, Vs, [(int(e), int(n), int(v)) for e, n, v in tasks],
                               int(nthreads))
        cout, down = K._cout, K._down
        key = self.key
        res = []
        for el in out:
            d = {}
            for i, t in el:
                x = cout(t) if down is None else down(cout(t))
                if x:
                    d[key(i)] = x
            res.append(d)
        return res

    def cache_sizes(self):
        """
        Entries of the engine's caches (see ``_module_engine``).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: E = M._engine(); _ = L.mode(L.mode(v, -2), 1)
            sage: E is None or 'keys' in E.cache_sizes()
            True
        """
        return self._E.cache_sizes()


class _PBWModeKeys:
    """The basis keys ``(modes, u)`` of an induced module: an ordered tuple
    of creation modes ``(i, m)`` (``m < 0``, ``m ∈ ZZ - Δ_i``; sorted by
    ``(m, i)``, odd modes at most once) and the index ``u`` of a basis
    vector of the top level.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
        sage: from vertex_algebras.vertex_algebra_module import _PBWModeKeys
        sage: K = _PBWModeKeys(MN); (((1, -3/2), (0, -1)), 0) in K, (((0, -1), (1, -3/2)), 0) in K
        (True, False)
    """

    def __init__(self, M):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.vertex_algebra_module import _PBWModeKeys
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: _PBWModeKeys(MN)._M is MN
            True
        """
        self._M = M

    def __contains__(self, x):
        """
        Whether ``x`` is a key: modes in order, of the right weights, odd ones
        at most once, and a top key.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: from vertex_algebras.vertex_algebra_module import _PBWModeKeys
            sage: K = _PBWModeKeys(MN)
            sage: (((1, -1/2),), 0) in K, (((1, -1/2), (1, -1/2)), 0) in K, (((1, -1),), 0) in K, 'x' in K
            (True, False, False, False)
        """
        M = self._M
        if not (isinstance(x, tuple) and len(x) == 2):
            return False
        modes, u = x
        if not (isinstance(modes, tuple) and u in M._U._indices):
            return False
        prev = None
        for t in modes:
            if not (isinstance(t, tuple) and len(t) == 2 and t[0] in range(len(M._h))):
                return False
            i, m = t
            try:
                m = QQ(m)
            except (TypeError, ValueError):
                return False
            if m >= 0 or (m + M._h[i]) not in ZZ:
                return False
            if prev is not None and (M._mode_order(prev) > M._mode_order(t)
                                     or (prev == t and M._odd[i])):
                return False
            prev = t
        return True

    def an_element(self):
        """
        The top key ``0`` without modes.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: from vertex_algebras.vertex_algebra_module import _PBWModeKeys
            sage: _PBWModeKeys(MN).an_element()
            ((), 0)
        """
        return ((), 0)

    def cardinality(self):
        """
        Infinite.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: from vertex_algebras.vertex_algebra_module import _PBWModeKeys
            sage: _PBWModeKeys(MN).cardinality()
            +Infinity
        """
        return Infinity


class InducedModule(_GeneratorActions, VertexAlgebraModule):
    r"""
    The induced module ``M(U)`` of a universal vertex algebra ``V`` with
    generators of positive weight, for a finite-dimensional module ``U``
    over its Zhu algebra (``V.induced_module(U)``; ``V.verma_module(h)`` for
    a one-dimensional ``U``): Zhu's universal positive energy module with
    top level ``U`` (Dong--Li--Mason's ``\bar M(U)`` [DLM1998]_).

    Basis: the ordered monomials ``g_{i_1, m_1} ... g_{i_k, m_k} u`` of the
    negative (physics) modes of the strong generators (``m_1 <= ... <=
    m_k < 0``, generators in order for equal modes, odd modes at most once)
    on the basis ``u`` of ``U``.  A generator mode ``g_m`` acts by moving to
    its place, ``g_m x = (-1)^{|g||x|} x g_m + [g_m, x]`` with the
    commutator formula ``[a_m, b_n] = Σ_j C(m + Δ_a - 1, j) (a_(j) b)_{m+n}``
    (modes of normally ordered polynomials acting through
    ``_GeneratorActions``); on ``U`` positive modes act by ``0`` and
    zero modes by the matrices of ``U``.  The depth of a monomial is
    ``-Σ m_j``; the conformal weight is ``h`` plus the depth, ``h`` the
    eigenvalue of ``[T]`` on ``U`` if ``V`` has a conformal vector and
    ``[T]`` has a single eigenvalue (else ``0``, the depth grading).

    Why this is ``\bar M(U)``: ``\bar M(U)`` is spanned by these PBW
    monomials (modes of all fields reduce to those of the generators, and
    the commutator formula reorders them), so whenever the rules above
    define a module on the free span of the PBW monomials, it is
    ``\bar M(U)``.  For linear OPEs (the modes form a Lie superalgebra:
    affine, Virasoro, ``N = 1, 2``, free fields) this is the PBW theorem;
    for non-linear ones (W-algebras) the consistency of the rewriting is
    what the Borcherds identities in ``TestSuite`` and in the tests check
    (it holds for ``W_3`` on all vectors of depth ``<= 2`` and fails for
    ``W_3`` with a wrong ``β``); De Sole--Kac's induction (Lemma 2.28)
    instead takes the quotient of the module induced over the Lie algebra
    of all modes by its maximal submodule meeting the top trivially.

    INPUT:

    - ``V`` -- a universal vertex algebra whose generators have positive
      conformal weight
    - ``U`` -- a finite-dimensional module over the (untwisted) Zhu
      algebra of ``V`` or a quotient of it, or a Verma module of the Zhu
      algebra (an infinite top level)
    - ``name`` -- (default: ``None``) a string; if given, the repr is
      ``name over V``
    - ``implementation`` -- (default: ``None``) ``None``, ``'engine'`` or
      ``'python'``: the C++ module engine (``None``: when available) or
      the Python recursion

    EXAMPLES:

    The Virasoro Verma module ``M(c, h)``: at ``h = 0`` the vector ``L_{-1}
    v`` is singular, at level 2 the Kac determinant vanishes for ``c =
    2h(5-8h)/(2h+1)``::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: K.<c, h> = Frac(QQ['c', 'h'])
        sage: E.<L> = NormallyOrderedExpressions(K)
        sage: Vir = UniversalVertexAlgebra(K, E, {('L', 'L'): [L.derivative(), 2*L, 0, c/2]},
        ....:                              conformal_vector=L)
        sage: M = Vir.verma_module({'L': h}); M
        Verma module of highest weight {'L': h} over Universal vertex algebra on generators ['L'] over Fraction Field of Multivariate Polynomial Ring in c, h over Rational Field
        sage: L = Vir.gen('L'); v = M.highest_weight_vector()
        sage: w = L.mode(L.mode(v, -1), -1); w
        L_{-1} L_{-1} v
        sage: L.mode(w, 1), L.mode(w, 2)
        ((4*h+2)*L_{-1} v, 6*h*v)
        sage: L.mode(w, 0) == (h + 2) * w, M.weight_space(h + 3).dimension()
        (True, 3)
        sage: TestSuite(M).run()

    Affine ``sl_2`` at level ``k``: the Weyl module of the doublet, of top
    weight ``3/(4(k+2))`` (the Sugawara vector on the doublet)::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: Kk.<k> = Frac(QQ['k'])
        sage: V = UniversalAffineVertexAlgebra(Kk, ['A', 1], level=k, names=['E', 'H', 'F'],
        ....:                                  conformal_vector='sugawara')
        sage: U = V.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
        ....:                             'F': matrix([[0, 0], [1, 0]])})
        sage: W = V.induced_module(U); W.top_weight()
        3/4/(k + 2)
        sage: [W.weight_space(W.top_weight() + d).dimension() for d in range(3)]
        [2, 6, 18]
        sage: E = V.gen('E'); E.mode(W.monomial(((), 1)), 0)
        u[0]

    An infinite-dimensional top level, a Verma module of the Zhu algebra
    (:class:`zhu_algebra_module.ZhuVermaModule`): the affine Verma module of
    the admissible weight ``λ = -1/2`` at level ``-1/2``.  Its conformal
    weight spaces are infinite dimensional, those of a grade (an
    ``sl_2``-weight) finite; its simple quotient ``L(λ)`` (half of the
    Ramond sector of the ``βγ`` system)::

        sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
        ....:                                   conformal_vector='sugawara')
        sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
        sage: h = Ma.top_weight(); h
        -1/8
        sage: [[Ma.weight_space(h + d, horizontal_grade=(-1/2 + m,)).dimension() for m in (-4, -2, 0, 2)]
        ....:  for d in range(3)]
        [[1, 1, 1, 0], [3, 3, 2, 1], [9, 8, 6, 3]]
        sage: La = Ma.simple_quotient()
        sage: [[La.weight_space(h + d, horizontal_grade=(-1/2 + m,)).dimension() for m in (-4, -2, 0, 2)]
        ....:  for d in range(3)]
        [[1, 1, 1, 0], [2, 2, 1, 0], [5, 5, 3, 1]]
        sage: Ma.weight_space(h)
        Traceback (most recent call last):
        ...
        ValueError: the conformal weight spaces of ... are infinite dimensional: give a grade

    TESTS:

    Non-linear OPEs: the commutator formula ``[a_(p), b_(q)] = Σ_j C(p, j)
    (a_(j) b)_(p+q-j)`` for ``W`` and ``L`` holds on all vectors of depth
    ``<= 2`` of a ``W_3`` Verma module; with a wrong ``β`` (the Jacobi
    identity of ``V`` fails) it does not::

        sage: Kc.<c> = Frac(QQ['c'])
        sage: def failures(beta):
        ....:     E.<L, W> = NormallyOrderedExpressions(Kc)
        ....:     Lam = L*L - 3/10*L.derivative(2)
        ....:     V = UniversalVertexAlgebra(Kc, E, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
        ....:         ('L', 'W'): [W.derivative(), 3*W], ('W', 'W'): [beta*Lam.derivative()
        ....:         + L.derivative(3)/15, 2*beta*Lam + 3/10*L.derivative(2), L.derivative(), 2*L, 0, c/3]},
        ....:         degrees={'L': 2, 'W': 3})
        ....:     M = V.verma_module({'L': c + 1, 'W': 1/c}); Lg, Wg = V.gens(); bad = 0
        ....:     for a, b in [(Wg, Wg), (Wg, Lg)]:
        ....:         ab = a.ope(b)
        ....:         for m in [M.monomial(key) for d in (0, 1, 2) for key in M._depth_keys(d)]:
        ....:             for q in range(-1, 3):
        ....:                 for p in range(4):
        ....:                     lhs = a.nproduct(b.nproduct(m, q), p) - b.nproduct(a.nproduct(m, p), q)
        ....:                     bad += lhs != sum((binomial(p, j) * ab[j].nproduct(m, p + q - j)
        ....:                                        for j in range(min(p + 1, len(ab)))), M.zero())
        ....:     return bad
        sage: failures(16/(22 + 5*c)), failures(1) > 0
        (0, True)

    The Borcherds identities with a composite field (commutator formula and
    the normally ordered product on module vectors) on vectors of depth
    ``<= 1``: ``N = 1`` (symbolic weight), ``N = 2``, Bershadsky--Polyakov
    (non-linear, weights ``3/2``)::

        sage: import vertex_algebras as va
        sage: def borcherds(V, M):
        ....:     g = list(V.gens()); els = g[:2] + [g[0].no(g[-1])]
        ....:     vecs = [M.monomial(key) for d in (0, 1) for key in M._depth_keys(d)][:6]
        ....:     for a in els:
        ....:         for b in els:
        ....:             s = (-1)^(a.parity()*b.parity()); ab = a.ope(b); nab = a.nproduct(b, -1)
        ....:             for m in vecs:
        ....:                 am, bm = a.ope(m), b.ope(m)
        ....:                 for q in range(-1, 2):
        ....:                     for p in range(2):
        ....:                         if (a.nproduct(b.nproduct(m, q), p) - s*b.nproduct(a.nproduct(m, p), q)
        ....:                                 != sum((binomial(p, j) * ab[j].nproduct(m, p + q - j)
        ....:                                         for j in range(min(p + 1, len(ab)))), M.zero())):
        ....:                             return False
        ....:                     rhs = sum((a.nproduct(b.nproduct(m, q + j), -1 - j)
        ....:                                for j in range(max(len(bm) - q, 0))), M.zero())
        ....:                     rhs += sum((s * b.nproduct(am[j], q - 1 - j) for j in range(len(am))), M.zero())
        ....:                     if nab.nproduct(m, q) != rhs:
        ....:                         return False
        ....:     return True
        sage: N1 = va.NeveuSchwarz(Kc, c, conformal_vector='L'); MN1 = N1.verma_module({'L': c/7})
        sage: N2 = va.N2(Kc, c, conformal_vector='L'); MN2 = N2.verma_module({'L': 2, 'J': c})
        sage: from vertex_algebras.lie_superalgebras import LieSuperalgebra
        sage: BP = va.W(Kk, LieSuperalgebra(QQ, CartanMatrix(['A', 2])), 'minimal', k,
        ....:           conformal_vector='reduction')
        sage: MBP = BP.verma_module({s: (k + 1)^i for i, s in enumerate(BP.zhu_algebra().variable_names())})
        sage: borcherds(N1, MN1), borcherds(N2, MN2), borcherds(BP, MBP)
        (True, True, True)
        sage: TestSuite(MBP).run()

    Characters (PBW counts): ``N = 1`` with a symbolic weight, ``W_3``,
    ``q^{1/16}/Π(1-q^n)`` for ``c = 1/2``, and the ``N = 1`` (Neveu--Schwarz)
    ``q^h Π(1+q^{n-1/2})/(1-q^n)``::

        sage: [MN1.weight_space(c/7 + d/2).dimension() for d in range(6)]
        [1, 1, 1, 2, 3, 4]
        sage: W3 = va.W3(Kc, c, conformal_vector='L'); MW3 = W3.verma_module({'L': c + 1, 'W': 1/c})
        sage: [MW3.weight_space(c + 1 + d).dimension() for d in range(5)]
        [1, 2, 5, 10, 20]
        sage: q = PuiseuxSeriesRing(ZZ, 'q').gen(); eta = prod(1 - q^n for n in range(1, 8))
        sage: va.Virasoro(QQ, 1/2, conformal_vector='L').verma_module(1/16).character(6) == (q^(1/16)/eta).add_bigoh(6)
        True
        sage: NS = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L')
        sage: NS.verma_module(1/10).character(4) == (q^(1/10) * prod(1 + q^(n - 1/2) for n in range(1, 8))
        ....:                                       / eta).add_bigoh(4)
        True

    Zhu's theorem [Zhu1996]_: the zero modes ``o(a) = a_(wt a - 1)`` on the top level
    are the matrices of ``[a]`` on ``U`` (``W_3``, Bershadsky--Polyakov, the
    affine Weyl module above)::

        sage: def zero_modes(M, weights):
        ....:     V, A, U = M.vertex_algebra(), M._A, M.top_level()
        ....:     for key in [key for w in weights for key in V.weight_space(w).basis_keys()]:
        ....:         a = V.monomial(key); mat = U.representation_matrix(A(a))
        ....:         for u in range(U.dimension()):
        ....:             if (dict(a.nproduct(M.monomial(((), u)), a.degree() - 1).monomial_coefficients())
        ....:                     != {((), r): mat[r, u] for r in range(U.dimension()) if mat[r, u]}):
        ....:                 return False
        ....:     return True
        sage: zero_modes(MW3, [2, 3, 4]), zero_modes(MBP, [1, 2]), zero_modes(W, [1, 2])
        (True, True, True)

    Trivial tops give ``V`` itself, by ``a ↦ a_(-1) v``: ``(a_(n) b)_(-1) v =
    a_(n) b_(-1) v`` and the same dimensions (Heisenberg, ``βγ`` at weights
    1/2, ``V^1(sl_2)``)::

        sage: def vacuum_like(V, M, weights):
        ....:     v = M.highest_weight_vector()
        ....:     B = [V.monomial(key) for w in weights for key in V.weight_space(w).basis_keys()][:8]
        ....:     return (all(a.nproduct(b, n).nproduct(v, -1) == a.nproduct(b.no(v), n)
        ....:                 for a in B for b in B for n in range(-2, 3))
        ....:             and all(M.weight_space(M.top_weight() + w).dimension()
        ....:                     == V.weight_space(w).dimension() for w in weights))
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, degrees={'J': 1})
        sage: BG = UniversalVertexAlgebra(QQ, ['be', 'ga'], {('be', 'ga'): [1]},
        ....:                             degrees={'be': 1/2, 'ga': 1/2})
        sage: A1 = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
        sage: Mk = A1.induced_module(A1.zhu_algebra().module({'E': 0, 'H': 0, 'F': 0}))
        sage: (vacuum_like(H, H.verma_module(0), [0, 1, 2, 3]),
        ....:  vacuum_like(BG, BG.induced_module(BG.zhu_algebra().module([])), [0, 1/2, 1, 3/2, 2]),
        ....:  vacuum_like(A1, Mk, [0, 1, 2]))
        (True, True, True)

    Singular vectors: the Ising level-2 vector ``(L_{-2} - 3/4 L_{-1}^2) v``
    at ``h = 1/2``, ``G_{-1/2} v`` at ``h = 0`` (``N = 1``), ``E_{-1}^2 v``
    of ``sl_2`` at level 1::

        sage: Vi = va.Virasoro(QQ, 1/2, conformal_vector='L'); Li = Vi.gen('L')
        sage: v = Vi.verma_module(1/2).highest_weight_vector()
        sage: s = Li.mode(v, -2) - 3/4*Li.mode(Li.mode(v, -1), -1)
        sage: [Li.mode(s, n) for n in (1, 2)], Li.mode(Li.mode(v, -2), 1)
        ([0, 0], 3*L_{-1} v)
        sage: G = NS.gen('G'); v = NS.verma_module(0).highest_weight_vector()
        sage: [G.mode(G.mode(v, -1/2), r) for r in (1/2, 3/2)]
        [0, 0]
        sage: v = Mk.highest_weight_vector(); Ek = A1.gen('E'); e2 = Ek.mode(Ek.mode(v, -1), -1)
        sage: e2, [x.mode(e2, 1) for x in A1.gens()]
        (E_{-1} E_{-1} v, [0, 0, 0])

    Refused: a module over another Zhu algebra, a generator of weight 0,
    twisted tops::

        sage: Vi.induced_module(W3.zhu_algebra().module({'L': 1, 'W': 2}))
        Traceback (most recent call last):
        ...
        ValueError: Module of dimension 1 over Zhu algebra of ... is not a module over the Zhu algebra of ...
        sage: BC = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
        ....:                             parities={'b': 1, 'c': 1}, degrees={'b': 1, 'c': 0})
        sage: BC.induced_module(BC.zhu_algebra().module({'b': matrix([[0, 1], [0, 0]]),
        ....:                                            'c': matrix([[0, 0], [1, 0]])}, [0, 1]))
        Traceback (most recent call last):
        ...
        NotImplementedError: induced modules need generators of positive weight
        sage: NS.induced_module(NS.zhu_algebra(twist='H').module({'L': 7/240, 'G': 0}))
        Traceback (most recent call last):
        ...
        NotImplementedError: twisted induced modules are not implemented yet

    Gradings by weights.  The affine Weyl module of the triplet over
    ``Q(k)``: the graded dimensions are ``χ_U(z) Π_{n>=1} 1/((1 - q^n)(1 -
    z^2 q^n)(1 - z^{-2} q^n))``::

        sage: R.<z, q> = LaurentPolynomialRing(QQ)
        sage: def weyl(weights, depth):
        ....:     f = sum(z^w for w in weights)
        ....:     for n in range(1, depth + 1):
        ....:         for a in (0, 2, -2):
        ....:             f = R({e: c for e, c in (f * sum((z^a*q^n)^j for j in range(depth//n + 1))).dict().items()
        ....:                    if e[1] <= depth})
        ....:     return {tuple(e): c for e, c in f.dict().items()}
        sage: W3 = V.induced_module(V.zhu_algebra().module({'E': matrix([[0, -2, 0], [0, 0, 1], [0, 0, 0]]),
        ....:     'H': matrix([[2, 0, 0], [0, 0, 0], [0, 0, -2]]), 'F': matrix([[0, 0, 0], [-1, 0, 0], [0, 2, 0]])}))
        sage: f = weyl([2, 0, -2], 3); h = W3.top_weight()
        sage: all(W3.weight_space(h + d, horizontal_grade=(m,)).dimension() == f.get((m, d), 0)
        ....:     for d in range(4) for m in range(-2 - 2*d, 3 + 2*d))
        True

    At level 1 the simple quotient of the doublet's Weyl module is the
    lattice module ``V_{1/2+A_1}``, grade by grade (``J_(0) e^p = 2p e^p``)::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: dbl = {'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]), 'F': matrix([[0, 0], [1, 0]])}
        sage: S1 = A1.induced_module(A1.zhu_algebra().module(dbl)).simple_quotient()
        sage: LA = LatticeVertexAlgebra(QQ, matrix([[2]])); J = LA.J(0)
        sage: M1 = LA.with_conformal_vector(J.no(J)/4).with_horizontal_grading('lattice').simple_module([1/2])
        sage: all(S1.weight_space(1/4 + d, horizontal_grade=(m,)).dimension()
        ....:     == M1.weight_space(1/4 + d, horizontal_grade=(m/2,)).dimension() for d in range(4) for m in range(-5, 6))
        True

    The ``N = 2`` Verma module of charge ``1/3`` (grades in ``QQ ⊃ ZZ``):
    graded dimensions ``Π (1 + y q^{n-1/2})(1 + y^{-1} q^{n-1/2})/(1 - q^n)^2``::

        sage: N2g = va.N2(QQ, 1, conformal_vector='L', horizontal_grading={'G1': 1, 'G2': -1})
        sage: MN = N2g.induced_module(N2g.zhu_algebra().module({'L': 1/6, 'J': 1/3}, horizontal_grades=[(1/3,)]))
        sage: MN.horizontal_grading_group(), MN.horizontal_grade(MN.highest_weight_vector())
        (Vector space of dimension 1 over Rational Field, (1/3))
        sage: R.<y, t> = LaurentPolynomialRing(QQ); f = R.one()       # t = q^(1/2)
        sage: for n in range(1, 9):
        ....:     f = f * (1 + y*t^(2*n - 1)) * (1 + y^-1*t^(2*n - 1)) * sum(t^(2*n*j) for j in range(5))^2
        ....:     f = R({e: c for e, c in f.dict().items() if e[1] <= 8})
        sage: f = {tuple(e): c for e, c in f.dict().items()}
        sage: all(MN.weight_space(1/6 + e/2, horizontal_grade=(1/3 + m,)).dimension() == f.get((m, e), 0)
        ....:     for e in range(9) for m in range(-3, 4))
        True
        sage: TestSuite(MN).run()
    """

    @staticmethod
    def __classcall_private__(cls, V, U, name=None, implementation=None):
        """
        Check ``implementation`` (unique representation on all arguments).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: U = V.zhu_algebra().module({'L': 1/16})
            sage: V.induced_module(U, implementation='bad')
            Traceback (most recent call last):
            ...
            ValueError: implementation must be None, 'engine' or 'python', not 'bad'
            sage: V.induced_module(U) is V.induced_module(U), V.induced_module(U, implementation='python') is V.induced_module(U)
            (True, False)
        """
        if implementation not in (None, 'engine', 'python'):
            raise ValueError("implementation must be None, 'engine' or 'python', not "
                             "{!r}".format(implementation))
        return VertexAlgebraModule.__classcall__(cls, V, U, name, implementation)

    def __init__(self, V, U, name=None, implementation=None):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: U = V.zhu_algebra().module({'L': 1/16}); M = V.induced_module(U, name='M')
            sage: M._U is U, M._h, M._odd, M._name
            (True, (2,), (0,), 'M')
            sage: W = va.Virasoro(QQ, 1/3, conformal_vector='L')
            sage: W.induced_module(U)
            Traceback (most recent call last):
            ...
            ValueError: ... is not a module over the Zhu algebra of ...
        """
        self._implementation = implementation
        if not getattr(V, '_is_universal_vertex_algebra', False):
            raise TypeError('induced modules need a universal vertex algebra, not {}'.format(V))
        A = U.algebra()
        cover = A._zhu_cover() if hasattr(A, '_zhu_cover') else A
        # a Zhu algebra of a quotient computed in the free algebra
        # (ZhuAlgebraFreeQuotient) refers to the universal algebra under it
        base = cover._A0._V if hasattr(cover, '_A0') else cover.vertex_algebra()
        if base is not V:
            raise ValueError('{} is not a module over the Zhu algebra of {}'.format(U, V))
        if cover.twist() is not None:
            raise NotImplementedError('twisted induced modules are not implemented yet')
        data = V._weight_data()
        self._h = tuple(QQ(x) for x in data['h'])
        if any(x <= 0 for x in self._h):
            raise NotImplementedError('induced modules need generators of positive weight')
        self._U = U
        self._A = cover
        self._name = name
        self._odd = tuple(V._parities.get(i, 0) for i in range(len(self._h)))
        #: V generator index -> Zhu generator index (those with zero modes)
        self._zidx = dict(cover._zidx)
        self._gen_cache = {}
        self._D = lcm([x.denominator() for x in self._h] + [1])
        iota = U._iota
        if iota is not None:
            self._mgrading = iota
            g = V._grading['g']
            self._ggrade = [iota(g[i]) for i in range(len(self._h))]
        self._mu = U._mu
        g = V._grading['g'] if self._mu is not None else None
        self._gmu = [self._mu_of(g[i]) if g is not None else QQ.zero()
                     for i in range(len(self._h))]
        self._top = None
        T = V.conformal_vector()
        if T is None and hasattr(cover, '_A0'):
            # the conformal vector of the quotient, lifted (presentations)
            Q = cover.vertex_algebra()
            if Q.conformal_vector() is not None:
                T = Q.lift(Q.conformal_vector())
        self._top_scalar = False
        if T is not None:
            t = cover(T)
            self._top = U._scalar(t)
            self._top_scalar = self._top is not None
            if self._top is None and hasattr(U, '_eigenvalue'):
                # a single eigenvalue (a Jordan block of L_0): the weight
                self._top = U._eigenvalue(t)
        if U._is_finite_top:
            usuper = any(U._par)
        else:
            usuper = any(z in U._odd for z in U._lowering)
        VertexAlgebraModule.__init__(self, V, _PBWModeKeys(self), super=V._has_odd()
                                     or usuper, graded=True)
        if implementation == 'engine' and self._engine() is None:
            raise ValueError("implementation='engine': the C++ engine of induced modules "
                             "is not available for {} (not compiled, an infinite or "
                             "twisted top level, or a base ring the engine does not "
                             "take)".format(self))

    def _repr_(self):
        """
        The string representation (the name if one was given).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: U = V.zhu_algebra().module({'L': 1/16}); repr(V.induced_module(U, name='M'))
            "M over Universal vertex algebra on generators ['L'] over Rational Field"
            sage: repr(V.induced_module(U)).startswith('Induced module of Module of dimension 1')
            True
        """
        if self._name is not None:
            return '{} over {}'.format(self._name, self._V)
        return 'Induced module of {} over {}'.format(self._U, self._V)

    # ------------------------------------------------------------------ #
    #  The C++ engine (_module_engine): the same recursions               #
    # ------------------------------------------------------------------ #

    def _engine(self):
        """
        The (lazily built) :class:`_InducedModuleEngine`, or ``None``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._engine() is M._engine()
            True
            sage: U = V.zhu_algebra().module({'L': 1/16}); V.induced_module(U, implementation='python')._engine() is None
            True
        """
        E = self.__dict__.get('_mengine', _NOT_BUILT)
        if E is _NOT_BUILT:
            # None while building: a lowered top (HigherInducedModule) fills
            # its table with the Python recursion, which asks for the engine
            self._mengine = None
            E = self._mengine = self._build_engine()
        return E

    def _build_engine(self):
        """
        Build the :class:`_InducedModuleEngine`: ``None`` for
        ``implementation='python'``, without the compiled module, for infinite
        or twisted top levels, without an engine of ``V`` or for coefficients its
        ring cannot hold.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: U = V.zhu_algebra().module({'L': 1/16}); V.induced_module(U, implementation='python')._build_engine() is None
            True
            sage: E = M._build_engine(); E is None or E.act_many([{((0, 0),): 1}], [{((), 0): 1}], [(0, 1, 0)], 1) == [{((), 0): 1/16}]
            True
        """
        from .vertex_algebra import _compiled_module
        if self._implementation == 'python':
            return None
        mod, _ = _compiled_module('_module_engine')
        if mod is None or not self._U._is_finite_top or self._A.twist() is not None:
            return None
        K = self._V._engine()
        if K is None:
            return None
        try:
            return _InducedModuleEngine(self, K, mod)
        except (ValueError, ArithmeticError, TypeError, AttributeError):
            return None             # e.g. a coefficient the engine's ring cannot hold

    def clear_caches(self):
        r"""
        Forget the computed actions (and the engine with its caches); for
        memory, results do not change.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); x = L.mode(L.mode(M.highest_weight_vector(), -1), 1)
            sage: M.clear_caches(); L.mode(L.mode(M.highest_weight_vector(), -1), 1) == x
            True
        """
        VertexAlgebraModule.clear_caches(self)
        self._gen_cache.clear()
        self.__dict__.pop('_mengine', None)

    def _action_mono(self, a, n, m):
        """
        Cached ``a_(n) m`` on keys: the engine if there is one, else
        :meth:`_action_key`.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._action_mono(((0, 1),), 1, ((), 0)), M._action_key(((0, 1),), 1, ((), 0))     # (dL)_(1) = -L_(0)
            ({(((0, -1),), 0): -1}, {(((0, -1),), 0): -1})
        """
        key = (a, n, m)
        r = self._act_cache.get(key)
        if r is None:
            E = self._engine()
            if E is not None:
                one = self._one_R
                r = E.act_many([{a: one}], [{m: one}], [(0, n, 0)], 1)
                r = None if r is None else r[0]
            if r is None:
                r = self._action_key(a, n, m)
            self._act_cache[key] = r
        return r

    def _action_raw(self, dA, n, dM):
        """
        ``A_(n) M`` for raw dicts, by one engine call if there is an engine.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._action_raw({((0, 0),): 1, ((0, 1),): 2}, 1, {((), 0): 1}) == {((), 0): 1/16, (((0, -1),), 0): -2}
            True
        """
        E = self._engine()
        if E is not None and dA and dM:
            r = E.act_many([dA], [dM], [(0, n, 0)], self._V._nthreads())
            if r is not None:
                return r[0]
        return VertexAlgebraModule._action_raw(self, dA, n, dM)

    def _action_raw_batch(self, els, vecs, tasks):
        """
        A batch of actions in one engine call, or ``None`` without an engine.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: r = M._action_raw_batch([{((0, 0),): 1}], [{((), 0): 1}], [(0, -1, 0), (0, 1, 0)])
            sage: r is None or r == [{(((0, -2),), 0): 1}, {((), 0): 1/16}]
            True
        """
        E = self._engine()
        if E is None:
            return None
        return E.act_many(els, vecs, tasks, self._V._nthreads())

    def top_level(self):
        r"""
        The module ``U`` over the Zhu algebra (the top level).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: U = V.zhu_algebra().module({'L': matrix([[1/16]])})
            sage: M = V.induced_module(U); M.top_level() is U
            True
            sage: V.verma_module(1/16).top_level()
            Module of dimension 1 over Zhu algebra of Universal vertex algebra on generators ['L'] ...
        """
        return self._U

    def top_weight(self):
        """The conformal weight ``h`` of the top level: the eigenvalue of
        ``[T]`` on ``U`` if it has only one (``L_0`` may have Jordan blocks),
        else ``0``, as when ``V`` has no conformal vector.

        EXAMPLES:

        A Jordan block of ``L_0`` on the top level of a Virasoro module::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
            ....:                              conformal_vector=L)
            sage: U = Vir.zhu_algebra().module({'L': matrix([[1/3, 1], [0, 1/3]])})
            sage: M = Vir.induced_module(U); M.top_weight(), M.weight_space(1/3).dimension()
            (1/3, 2)
            sage: L = Vir.gen('L'); [L.mode(x, 0) for x in M.weight_space(1/3).basis()]
            [1/3*u[0], u[0] + 1/3*u[1]]
        """
        return self._V.base_ring().zero() if self._top is None else self._top

    def highest_weight_vector(self):
        r"""
        The first basis vector of the top level (``v`` for a Verma module of
        the Zhu algebra).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector(); v
            v
            sage: L.mode(v, 0), L.mode(v, 1), L.mode(v, 2)
            (1/16*v, 0, 0)
            sage: U = V.zhu_algebra().module({'L': matrix([[1/3, 1], [0, 1/3]])})
            sage: V.induced_module(U).highest_weight_vector()
            u[0]
        """
        return self.monomial(((), self._U._first_key()))

    def _weight_base(self):
        """
        The top weight.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._weight_base()
            1/16
        """
        return self.top_weight()

    def _key_order(self, key):
        """
        Longer monomials are leading: ``(number of modes, modes, top key)``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._key_order((((0, -2),), 0)) < M._key_order((((0, -1), (0, -1)), 0))
            True
        """
        return (len(key[0]), key[0], key[1])

    def _l0_grading(self):
        """
        True iff ``[T]`` acts on the top level by a scalar.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: U = V.zhu_algebra().module({'L': matrix([[1/3, 1], [0, 1/3]])})
            sage: M._l0_grading(), V.induced_module(U)._l0_grading()
            (True, False)
        """
        return self._top_scalar

    def _generated_by_lowest_weight_space(self):
        """
        Always true: the top level generates.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._generated_by_lowest_weight_space()
            True
        """
        return True

    def _mode_order(self, t):
        """
        The order of creation modes ``(i, m)`` in a monomial: by ``m``, then
        ``i``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: M._mode_order((0, -2)) < M._mode_order((0, -1))
            True
            sage: MN._mode_order((1, -3/2)) < MN._mode_order((0, -1))
            True
        """
        return (QQ(t[1]), t[0])

    def _repr_term(self, key):
        """
        The string of a key: the modes, then the top vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._repr_term((((0, -2), (0, -1)), 0))
            'L_{-2} L_{-1} v'
        """
        modes, u = key
        names = self._V._gens
        return ' '.join(['{}_{{{}}}'.format(names[i], m) for i, m in modes]
                        + [self._U._repr_top(u)])

    def _depth(self, key):
        """
        The depth ``-Σ m`` of a key.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: M._depth((((0, -2), (0, -1)), 0)), MN._depth((((1, -3/2), (0, -1)), 0))
            (3, 5/2)
        """
        return -sum((QQ(m) for _, m in key[0]), QQ.zero())

    # ------------------------------------------------------------------ #
    #  The action                                                          #
    # ------------------------------------------------------------------ #

    def _parity_key(self, key):
        """
        The parity of a key: odd modes plus the parity of the top vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: MN._parity_key((((1, -3/2), (0, -1)), 0)), MN._parity_key((((1, -3/2), (1, -1/2)), 0))
            (1, 0)
        """
        modes, u = key
        return (sum(self._odd[i] for i, _ in modes) + self._U._parity_key(u)) % 2

    def _nmax_key(self, a, key):
        """``a_(n) v = 0`` once its depth ``depth(v) + wt a - n - 1`` is
        negative.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._nmax_key(((0, 0),), ((), 0)), M._nmax_key(((0, 0),), (((0, -2),), 0))
            (1, 3)
        """
        V = self._V
        return QQ(V._weight_key(a, V._weight_data()) + self._depth(key) - 1).floor()

    def _generator_action_key(self, i, n, key):
        """
        The ``i``-th generator's ``(n)``-product, i.e. the physics mode ``n -
        Δ_i + 1``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._generator_action_key(0, 2, (((0, -1),), 0))
            {((), 0): 1/8}
        """
        return self._gen_act(i, QQ(n) - self._h[i] + 1, key)

    def _gen_act(self, i, m, key):
        """
        The physics mode ``g_{i,m}`` on the basis vector ``key``, as a dict.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._gen_act(0, 1, (((0, -1),), 0)), M._gen_act(0, -2, (((0, -1),), 0))
            ({((), 0): 1/8}, {(((0, -2), (0, -1)), 0): 1})
            sage: M._gen_act(0, -1, (((0, -2),), 0)) == {(((0, -2), (0, -1)), 0): 1, (((0, -3),), 0): 1}
            True
        """
        ck = (i, m, key)
        r = self._gen_cache.get(ck)
        if r is not None:
            return r
        modes, u = key
        one = self._one_R
        if not modes:
            if m > 0:
                r = {}
            elif m == 0:
                z = self._zidx.get(i)
                r = {} if z is None else {((), s): c for s, c in
                                          self._U._gen_action_key(z, u).items()}
            else:
                r = {(((i, m),), u): one}
        else:
            x, rest = modes[0], modes[1:]
            t = (i, m)
            ot, ox = self._mode_order(t), self._mode_order(x)
            if ot < ox or (t == x and not self._odd[i]):
                r = {((t,) + modes, u): one}               # already in place
            elif t == x:
                # an odd mode squared: x x = 1/2 [x, x]
                r = {}
                _add_scaled(r, self._commutator_on(i, m, x, (rest, u)), one / 2)
            else:
                # g_m x rest = s x (g_m rest) + [g_m, x] rest
                s = (-1) ** (self._odd[i] * self._odd[x[0]])
                r = {}
                for k2, c2 in self._gen_act(i, m, (rest, u)).items():
                    _add_scaled(r, self._gen_act(x[0], x[1], k2), s * c2)
                _add_scaled(r, self._commutator_on(i, m, x, (rest, u)), one)
        self._gen_cache[ck] = r
        return r

    def _commutator_on(self, i, m, x, key):
        r"""``[g_{i,m}, x] key`` for a mode ``x = (j, m')``: ``Σ_p C(m + Δ_i - 1,
        p) (g_i(p) g_j)_{m + m'} key``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._commutator_on(0, 1, (0, -1), ((), 0))       # [L_1, L_{-1}] v = 2 L_0 v
            {((), 0): 1/8}
        """
        V = self._V
        j, mx = x
        poles = V._ope_mono(((i, 0),), ((j, 0),))
        hi = self._h[i]
        data = V._weight_data()
        out = {}
        for p, d in enumerate(poles):
            if not d:
                continue
            c = binomial(m + hi - 1, p)
            if not c:
                continue
            for a, ca in d.items():
                wa = V._weight_key(a, data)
                n = m + mx + wa - 1                  # (n)-notation of the mode m + m'
                _add_scaled(out, self._action_mono(a, int(n), key), c * ca)
        return out

    # ------------------------------------------------------------------ #
    #  Weights                                                             #
    # ------------------------------------------------------------------ #

    def _weight_key(self, key):
        """
        The top weight plus the depth.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._weight_key((((0, -1),), 0))
            17/16
        """
        return self.top_weight() + self._depth(key)

    def _weight_bounds(self, G):
        """
        ``(h, D)``: the top weight and the denominator of the weights; a
        ``ValueError`` for a non-rational top weight.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M._weight_bounds(None)
            (1/16, 16)
        """
        try:
            h = QQ(self.top_weight())              # weights in h + (1/D) ZZ
        except (TypeError, ValueError):
            raise ValueError('the top weight {} is not rational: no character (the '
                             'weight spaces h + d are available)'.format(self.top_weight()))
        return h, int(lcm(self._D, h.denominator()))

    def _modes_up_to(self, d):
        """
        The creation modes ``(i, m)`` with ``-m <= d``, in mode order.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: MN._modes_up_to(2)
            [(0, -2), (1, -3/2), (0, -1), (1, -1/2)]
        """
        out = []
        for i, h in enumerate(self._h):
            t = h - h.floor() if h not in ZZ else QQ.one()     # the smallest -m > 0
            if t == 0:
                t = QQ.one()
            while t <= d:
                out.append((i, -t))
                t += 1
        out.sort(key=self._mode_order)
        return out

    def _mode_monomials(self, d):
        """
        The ordered monomials of creation modes of depth ``d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: NS = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); MN = NS.verma_module(0)
            sage: MN._mode_monomials(5/2)
            [((1, -5/2),), ((0, -2), (1, -1/2)), ((1, -3/2), (0, -1)), ((0, -1), (0, -1), (1, -1/2))]
            sage: MN._mode_monomials(1/3), MN._mode_monomials(-1)
            ([], [])
        """
        d = QQ(d)
        if d < 0 or (d * self._D) not in ZZ:
            return []
        modes = self._modes_up_to(d)
        res = []

        def grow(start, left, acc):
            if left == 0:
                res.append(tuple(acc))
                return
            for k in range(start, len(modes)):
                i, m = modes[k]
                if -m > left:
                    continue
                if self._odd[i] and acc and acc[-1] == (i, m):
                    continue
                acc.append((i, m))
                grow(k if not self._odd[i] else k + 1, left + m, acc)
                acc.pop()
        grow(0, d, [])
        return res

    def _depth_keys(self, d):
        """
        The basis keys of depth ``d`` (a finite top level).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: M._depth_keys(2)
            [(((0, -2),), 0), (((0, -1), (0, -1)), 0)]
            sage: Ma._depth_keys(0)
            Traceback (most recent call last):
            ...
            ValueError: the conformal weight spaces of ... are infinite dimensional: give a grade
        """
        top = self._U._top_keys()
        if top is None:
            raise ValueError('the conformal weight spaces of {} are infinite dimensional: '
                             'give a grade'.format(self))
        return [(ms, u) for ms in self._mode_monomials(d) for u in top]

    def _grade_key(self, key):
        """
        The grade of a key: that of the top vector plus the grades of the
        modes.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._grade_key(((), ())), Ma._grade_key((((0, -1),), ()))
            ((-1/2,), (3/2,))
        """
        B = self._require_module_grading().B
        modes, u = key
        t = self._U._grade_key(u)
        for i, _ in modes:
            t = B.add(t, self._ggrade[i])
        return t

    def _tdeg(self, key):
        """
        The twisted degree: ``Σ (-m + μ(grade g_i))`` over the modes plus that of
        the top key.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._tdeg(((), ())), Ma._tdeg((((0, -1),), ())), Ma._tdeg(((), (2,)))
            (0, 3/4, 1/4)
        """
        modes, u = key
        return (sum((-m + self._gmu[i] for i, m in modes), QQ.zero())
                + self._U._tdeg(u))

    def _keys_of_tdeg(self, t):
        r"""The keys of twisted degree ``t``: creation modes (each of twisted
        degree ``-m + μ(horizontal_grade) > 0``) and top keys of the rest.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._keys_of_tdeg(1)
            [((), (2, 2, 2, 2)), (((0, -1),), (2,)), (((1, -1),), ())]
        """
        t = QQ(t)
        if t < 0:
            return []
        U = self._U
        bound = t + max([abs(x) for x in self._gmu] + [0])
        modes = [(i, m) for i, m in self._modes_up_to(bound) if -m + self._gmu[i] <= t]
        out = []

        def grow(start, left, acc):
            for u in U._keys_of_tdeg(left):
                out.append((tuple(acc), u))
            for k in range(start, len(modes)):
                i, m = modes[k]
                c = -m + self._gmu[i]
                if c > left:
                    continue
                if self._odd[i] and acc and acc[-1] == (i, m):
                    continue
                acc.append((i, m))
                grow(k if not self._odd[i] else k + 1, left - c, acc)
                acc.pop()
        grow(0, t, [])
        return out

    def _piece_keys(self, t, G):
        r"""The keys of twisted degree ``t`` and grade ``G``: creation modes and
        the top keys of the remaining grade (finitely many) of the remaining
        twisted degree.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: Ma._piece_keys(1, Ma._grade_coords((-1/2,)))
            [(((0, -1),), (2,)), (((1, -1),), ())]
            sage: Ma._piece_keys(1, None) == Ma._keys_of_tdeg(1)
            True
        """
        if G is None:
            return self._keys_of_tdeg(t)
        t = QQ(t)
        if t < 0:
            return []
        U = self._U
        B = self._require_module_grading().B
        bound = t + max([abs(x) for x in self._gmu] + [0])
        modes = [(i, m) for i, m in self._modes_up_to(bound) if -m + self._gmu[i] <= t]
        out = []

        def grow(start, left, acc, Gl):
            out.extend((tuple(acc), u) for u in U._keys_of_grade(Gl) if U._tdeg(u) == left)
            for k in range(start, len(modes)):
                i, m = modes[k]
                c = -m + self._gmu[i]
                if c > left:
                    continue
                if self._odd[i] and acc and acc[-1] == (i, m):
                    continue
                acc.append((i, m))
                grow(k if not self._odd[i] else k + 1, left - c, acc,
                     B.add(Gl, tuple(-x for x in self._ggrade[i])))
                acc.pop()
        grow(0, t, [], B.reduce(G))
        return out

    def _weight_space_enumeration(self, W, G):
        """
        The keys of weight ``W`` (and grade ``G``): mode monomials of depth
        ``W - h`` times the top keys (of the remaining grade).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: Va = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-1/2, names=['E', 'H', 'F'],
            ....:                                   conformal_vector='sugawara')
            sage: Ma = Va.induced_module(Va.zhu_algebra().verma_module({'H': -1/2}))
            sage: sorted(Ma._weight_space_enumeration(Ma.top_weight() + 1, Ma._grade_coords((-1/2,))).factory())
            [(((0, -1),), (2,)), (((1, -1),), ())]
        """
        from .weight_spaces import _KeyEnumeration
        d = QQ(W - self.top_weight())
        if G is None:
            keys = self._depth_keys(d)
        else:
            B = self._require_module_grading().B
            keys = []
            for ms in self._mode_monomials(d):
                t = G
                for i, _ in ms:
                    t = B.add(t, tuple(-x for x in self._ggrade[i]))
                keys.extend((ms, u) for u in self._U._keys_of_grade(B.reduce(t)))
        return _KeyEnumeration(True, factory=lambda: iter(keys))

    def weight_space(self, weight, horizontal_grade=None):
        r"""
        The span of the basis vectors of conformal weight ``weight`` (``h``
        plus a depth; ``h`` may be symbolic), and of the grade ``horizontal_grade`` if
        given (if the top level is graded, see
        :class:`zhu_algebra_module.ZhuAlgebraModule`).

        INPUT:

        - ``weight`` -- the conformal weight, the top weight plus a
          nonnegative rational depth
        - ``horizontal_grade`` -- (default: ``None``) a grade in
          :meth:`horizontal_grading_group`, or ``None`` for all grades

        EXAMPLES:

        The Weyl module of the doublet of ``sl_2`` at level ``k``, graded by
        the weights ``P ⊗ Q(k)``::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: K.<k> = Frac(QQ['k'])
            sage: V = UniversalAffineVertexAlgebra(K, ['A', 1], level=k, names=['E', 'H', 'F'])
            sage: U = V.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]),
            ....:     'H': matrix([[1, 0], [0, -1]]), 'F': matrix([[0, 0], [1, 0]])})
            sage: W = V.induced_module(U); h = W.top_weight()
            sage: [[W.weight_space(h + d, horizontal_grade=(j,)).dimension() for j in range(-5, 6, 2)]
            ....:  for d in range(3)]
            [[0, 0, 1, 1, 0, 0], [0, 1, 2, 2, 1, 0], [1, 3, 5, 5, 3, 1]]
            sage: W.horizontal_grade(V.gen('E').mode(W.highest_weight_vector(), -1))
            (3)
        """
        from .weight_spaces import WeightSpace
        G = None
        if horizontal_grade is not None:
            G = self._grade_coords(horizontal_grade)
        QQ(weight - self.top_weight())
        return WeightSpace(self, weight, horizontal_grade, self._weight_space_enumeration(weight, G))

    def degree(self, x):
        r"""
        The conformal weight of a vector ``x`` homogeneous for it (a
        ``ValueError`` otherwise): the top weight plus the depth.

        INPUT:

        - ``x`` -- a vector of ``self``

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M.degree(v), M.degree(L.mode(L.mode(v, -1), -2))
            (1/16, 49/16)
            sage: M.degree(v + L.mode(v, -1))
            Traceback (most recent call last):
            ...
            ValueError: v + L_{-1} v is not homogeneous of a single conformal weight
        """
        x = self(x)
        ws = {self._weight_key(m) for m in x.support()}
        if len(ws) != 1:
            raise ValueError('{} is not homogeneous of a single conformal weight'.format(x))
        return ws.pop()

    def some_elements(self):
        r"""
        Some vectors: the highest weight vector, one creation mode of each of
        the first generators applied to it, and another vector of the top
        level if there is one.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: V.verma_module(1/16).some_elements()
            [v, L_{-1} v]
            sage: U = V.zhu_algebra().module({'L': matrix([[1/3, 1], [0, 1/3]])})
            sage: V.induced_module(U).some_elements()
            [u[0], L_{-1} u[0], u[1]]
        """
        v = self.highest_weight_vector()
        out = [v]
        for i in range(len(self._h)):
            m = -(self._h[i] - self._h[i].floor()) if self._h[i] not in ZZ else -QQ.one()
            if m == 0:
                m = -QQ.one()
            out.append(self.monomial((((i, m),), self._U._first_key())))
            if len(out) >= 4:
                break
        U = self._U
        top = U._top_keys()
        if top is not None and len(top) > 1:
            out.append(self.monomial(((), top[-1])))
        if top is None and U._lowering:
            out.append(self.monomial(((), (U._lowering[0],))))
        return out

    def _an_element_(self):
        """
        The highest weight vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(1/16); v = M.highest_weight_vector()
            sage: M.an_element()
            v
        """
        return self.highest_weight_vector()
