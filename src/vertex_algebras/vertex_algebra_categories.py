# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Categories of vertex algebras.

:class:`VertexAlgebras` is the category of vertex (super)algebras over a
commutative ring, a subcategory of modules only (not of Sage's Lie
conformal algebras: a design decision of this library).  Its
functorial constructions follow Sage's conventions:

- ``.Super()`` -- vertex superalgebras (parities of basis vectors,
  ``x.is_even_odd()``);
- ``.Graded()`` -- *conformally graded* vertex algebras: the conformal
  weight grading, from a Virasoro vector or from conformal weights chosen
  without one (``x.degree()`` is the conformal weight of a homogeneous
  ``x``);
- ``.WithBasis()`` -- with a distinguished basis (all algebras of this
  library).

:class:`VertexOperatorAlgebras` are the conformally graded vertex algebras
with a chosen Virasoro vector (logarithmic ones included).  An additional
grading by an abelian group (``horizontal_grading=``) is data of the algebra, not a
category.

The categories run vertex algebra axiom checks in ``TestSuite`` (vacuum,
translation, skew-symmetry, Borcherds' commutator formula, weights of the
products, the Virasoro vector) on ``some_elements()``.

:class:`VertexAlgebraModules` ``(V)`` is the category of modules over a
vertex algebra ``V`` (a category over ``V``, with the same constructions).

EXAMPLES::

    sage: from vertex_algebras import vertex_algebra_categories as vac
    sage: vac.VertexAlgebras(QQ)
    Category of vertex algebras over Rational Field
    sage: vac.VertexAlgebras(QQ).Super().Graded().WithBasis()
    Category of conformally graded super vertex algebras with basis over Rational Field
    sage: vac.VertexAlgebras(QQ).Graded().Super() is vac.VertexAlgebras(QQ).Super().Graded()
    True
    sage: vac.VertexOperatorAlgebras(QQ).Super().WithBasis()
    Category of super vertex operator algebras with basis over Rational Field
    sage: vac.VertexOperatorAlgebras(QQ).is_subcategory(vac.VertexAlgebras(QQ).Graded())
    True

    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: V = UniversalVertexAlgebra(QQ, ['b', 'c'], {('b', 'c'): [1]},
    ....:                            parities={'b': 1, 'c': 1})
    sage: V.category()
    Category of finitely strongly generated super vertex algebras with basis over Rational Field
    sage: TestSuite(V).run()

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import sage.categories.category_with_axiom
from sage.arith.misc import binomial, factorial
from sage.categories.category_types import Category_over_base, Category_over_base_ring
from sage.categories.category_with_axiom import CategoryWithAxiom_over_base_ring
from sage.categories.graded_modules import GradedModulesCategory
from sage.categories.modules import Modules
from sage.categories.super_modules import SuperModulesCategory
from sage.misc.cachefunc import cached_method

# the axiom of this module (Sage's documented way for axioms defined outside
# its library: append the name to the global container)
if 'FinitelyStronglyGenerated' not in sage.categories.category_with_axiom.all_axioms:
    sage.categories.category_with_axiom.all_axioms += ('FinitelyStronglyGenerated',)


def _vertex_algebra_category(R, super=False, graded=False, conformal=False,
                            finitely_strongly_generated=False, proto=False):
    """
    The category of a vertex algebra over ``R`` with basis: super if it
    has odd basis vectors, conformally graded if it has conformal weights,
    a vertex operator algebra if it has a Virasoro vector; with ``proto``
    a proto vertex algebra (the Jacobi identity is not claimed; then a
    Virasoro vector only makes it conformally graded).

    EXAMPLES::

        sage: from vertex_algebras.vertex_algebra_categories import _vertex_algebra_category
        sage: _vertex_algebra_category(QQ)
        Category of vertex algebras with basis over Rational Field
        sage: _vertex_algebra_category(QQ, super=True, graded=True)
        Category of conformally graded super vertex algebras with basis over Rational Field
        sage: _vertex_algebra_category(QQ, graded=True, conformal=True)
        Category of vertex operator algebras with basis over Rational Field
        sage: _vertex_algebra_category(QQ, graded=True, conformal=True, proto=True)
        Category of conformally graded proto vertex algebras with basis over Rational Field

    TESTS:

    Every construction is in the category its data say (super with odd
    vectors, conformally graded with weights, a vertex operator algebra
    with a Virasoro vector; finitely strongly generated or not), and passes
    ``TestSuite``::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: X.<L> = NormallyOrderedExpressions(QQ)
        sage: Vir = UniversalVertexAlgebra(QQ, X, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
        ....:                              conformal_vector='L')
        sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); J = F.J(0)
        sage: FT = F.with_conformal_vector(J.no(J)/2)
        sage: L = Vir.gen(0)
        sage: P.<p> = NormallyOrderedExpressions(QQ, parities={'p': 1})
        sage: Maj = UniversalVertexAlgebra(QQ, P, {('p', 'p'): [1]},
        ....:                              conformal_vector=p.derivative()*p/2)
        sage: H = UniversalVertexAlgebra(QQ, ['J', 'K'], {('J', 'J'): [0, 1], ('K', 'K'): [0, 1]},
        ....:                            degrees={'J': 1, 'K': 1})
        sage: cases = [(Vir.tensor(F), (True, False, False, False)),
        ....:     (Vir.tensor(FT, conformal_vector='factors'), (True, True, True, False)),
        ....:     (Vir.tensor(Vir), (False, False, False, True)),
        ....:     (va.Affine(QQ['k'], ['A', 1], QQ['k'].gen()), (False, True, False, True)),
        ....:     (va.BosonicGhosts(QQ), (False, False, False, True)),
        ....:     (Vir.quotient([L.no(L) - 3/10*L.derivative(2)], conformal_vector='ambient'),
        ....:      (False, True, True, True)),
        ....:     (Maj.subalgebra([Maj.conformal_vector()], conformal_vector='ambient'),
        ....:      (True, True, True, True)),
        ....:     (H.commutant([H.gen('J')]), (False, True, False, False))]
        sage: all(V.category() is _vertex_algebra_category(V.base_ring(), super=s, graded=g,
        ....:     conformal=t, finitely_strongly_generated=f) for V, (s, g, t, f) in cases)
        True
        sage: for V, _ in cases:
        ....:     TestSuite(V).run()

    Methods exist only where the category provides them::

        sage: [hasattr(V, 'weight_space') for V, _ in cases[:3]]
        [False, True, False]
        sage: hasattr(F, 'strong_generators'), hasattr(cases[-1][0], 'strong_generators')
        (False, False)
        sage: list(cases[5][0].strong_generators())
        [L]
    """
    if proto:
        C = ProtoVertexAlgebras(R)
        conformal = False
    else:
        C = VertexOperatorAlgebras(R) if conformal else VertexAlgebras(R)
    if graded and not conformal:
        C = C.Graded()
    if super:
        C = C.Super()
    if finitely_strongly_generated:
        C = C.FinitelyStronglyGenerated()
    return C.WithBasis()


def _is_finitely_strongly_generated(V):
    r"""
    Whether the vertex algebra ``V`` is in a finitely strongly
    generated category.

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: import vertex_algebras as va
        sage: vac._is_finitely_strongly_generated(va.Virasoro(QQ, 1/2))
        True
        sage: vac._is_finitely_strongly_generated(va.Lattice(QQ, matrix([[2]])))
        False
    """
    C = ProtoVertexAlgebras(V.base_ring()).FinitelyStronglyGenerated()
    return V.category().is_subcategory(C)


def _is_proto(V):
    r"""
    Whether the (proto) vertex algebra ``V`` is not in
    :class:`VertexAlgebras` (its Jacobi identity is not claimed).

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: P = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]}, proto=True)
        sage: vac._is_proto(H), vac._is_proto(P)
        (False, True)
    """
    return not V.category().is_subcategory(VertexAlgebras(V.base_ring()))


def _parity(x):
    r"""
    The parity of a homogeneous element (``None`` if inhomogeneous).

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: import vertex_algebras as va
        sage: F = va.FreeFermions(QQ); p = F.gen(0)
        sage: vac._parity(p), vac._parity(p.no(p.derivative())), vac._parity(p + F.one())
        (1, 0, None)
    """
    try:
        return x.parity()
    except ValueError:
        return None


class ProtoVertexAlgebras(Category_over_base_ring):
    r"""
    The category of proto vertex (super)algebras over a commutative ring:
    a vacuum, a translation ``∂`` and ``n``-th products for all integers
    ``n`` with the vacuum and translation axioms, but no locality (the
    Jacobi identity is not claimed): e.g. universal vertex algebras of OPE
    ansätze with unknown coefficients (``proto=True``, the bootstrap's
    algebras).  :class:`VertexAlgebras` is a subcategory.

    INPUT:

    - ``base_ring`` -- a commutative ring

    EXAMPLES::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: C = vac.ProtoVertexAlgebras(QQ); C
        Category of proto vertex algebras over Rational Field
        sage: vac.VertexAlgebras(QQ).is_subcategory(C)
        True
        sage: C.Super().Graded().FinitelyStronglyGenerated().WithBasis()
        Category of finitely strongly generated conformally graded super proto vertex algebras with basis over Rational Field
    """

    @cached_method
    def super_categories(self):
        r"""
        The super categories: modules over the base ring.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).super_categories()
            [Category of vector spaces over Rational Field]
        """
        return [Modules(self.base_ring())]

    class SubcategoryMethods:
        r"""
        Methods of every subcategory of proto vertex algebras: the axiom
        :meth:`FinitelyStronglyGenerated`.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).Graded().FinitelyStronglyGenerated()
            Category of finitely strongly generated conformally graded proto vertex algebras over Rational Field
        """

        @cached_method
        def FinitelyStronglyGenerated(self):
            r"""
            The subcategory of the vertex algebras strongly generated by
            finitely many fields: the normally ordered monomials of the
            derivatives of :meth:`strong_generators` span the algebra.

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexAlgebras(QQ).FinitelyStronglyGenerated()
                Category of finitely strongly generated vertex algebras over Rational Field
                sage: vac.VertexOperatorAlgebras(QQ).Super().FinitelyStronglyGenerated().WithBasis()
                Category of finitely strongly generated super vertex operator algebras with basis over Rational Field
            """
            return self._with_axiom('FinitelyStronglyGenerated')

    class FinitelyStronglyGenerated(CategoryWithAxiom_over_base_ring):
        r"""
        Vertex algebras strongly generated by finitely many fields
        (universal vertex algebras, their quotients, the subalgebras whose
        closure is complete, tensor products of such).

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).FinitelyStronglyGenerated()
            Category of finitely strongly generated proto vertex algebras over Rational Field
            sage: import vertex_algebras as va
            sage: va.Virasoro(QQ, 1/2) in vac.VertexAlgebras(QQ).FinitelyStronglyGenerated()
            True
            sage: va.Lattice(QQ, matrix([[2]])) in vac.VertexAlgebras(QQ).FinitelyStronglyGenerated()
            False
        """

        class ParentMethods:
            r"""
            Methods of finitely strongly generated vertex algebras.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: va.N2(QQ, 1).strong_generators()
                (L, J, G1, G2)
            """

            def strong_generators(self):
                r"""
                The strong generators: finitely many elements whose
                derivatives span the algebra by normally ordered monomials
                (the nonzero :meth:`~vertex_algebra.VertexAlgebra.gens`).

                EXAMPLES::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: V = UniversalVertexAlgebra(QQ, ['b', 'g'], {('b', 'g'): [-1]})
                    sage: V.strong_generators(), V.ngens()
                    ((b, g), 2)
                """
                return tuple(x for x in self.gens() if not x.is_zero())

    class ParentMethods:
        r"""
        Methods of all proto vertex algebras: the vacuum and translation
        axiom checks of ``TestSuite``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.FermionicGhosts(QQ); V._test_vacuum(); V._test_translation()
        """

        def _test_vacuum(self, **options):
            r"""
            The vacuum axioms on ``some_elements()``: ``1_(n) x = δ_{n,-1}
            x``, ``x_(-1) 1 = x``, ``x_(-2) 1 = ∂x``, ``x_(n) 1 = 0`` for
            ``n >= 0``, and ``∂1 = 0``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[2]]))._test_vacuum()
            """
            tester = self._tester(**options)
            one = self.one()
            tester.assertTrue(one.derivative().is_zero())
            for x in tester.some_elements():
                tester.assertEqual(one.ope(x), [])
                tester.assertEqual(one.no(x), x)
                tester.assertEqual(x.ope(one), [])
                tester.assertEqual(x.no(one), x)
                tester.assertEqual(x.nproduct(one, -2), x.derivative())

        def _test_raw_caches(self, **options):
            r"""
            The cache contract: after products of pairs of
            ``some_elements()`` (element and key level), every cache of
            raw products of this algebra and of the algebras it computes in
            (``_cache_owners()``, ``_raw_caches()``) holds raw dicts
            ``{key: coefficient}`` or tuples of them (the poles of an OPE),
            never elements.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: L = LatticeVertexAlgebra(QQ, matrix([[2]]))
                sage: L._test_raw_caches()
                sage: L._ope_cache[next(iter(L._ope_cache))] = L.one()    # breaks the contract
                sage: L._test_raw_caches()
                Traceback (most recent call last):
                ...
                AssertionError: _ope_cache of ... holds ...
                sage: L.clear_caches()
            """
            tester = self._tester(**options)
            els = list(tester.some_elements())
            for x in els:
                for y in els:
                    x.ope(y)
                    x.no(y)
                    for a in list(x.support())[:2]:
                        for b in list(y.support())[:2]:
                            self._ope_mono(a, b)
                            self._no_mono(a, b)
                            self._nprod_mono(a, -2, b)
                x.derivative(2)
                for a in list(x.support())[:2]:
                    self._deriv_mono(a, 2)

            def raw(v):
                return isinstance(v, dict) or (
                    isinstance(v, tuple) and all(isinstance(p, dict) for p in v))
            for owner in self._cache_owners():
                for name in owner._raw_caches():
                    for v in getattr(owner, name, {}).values():
                        if name == '_ls_cache' and isinstance(v, tuple):
                            v = v[1:]           # (packed series, raw dicts...)
                        tester.assertTrue(raw(v), '%s of %s holds %s' % (name, owner, type(v)))

        def _test_translation(self, **options):
            r"""
            Translation covariance on pairs of ``some_elements()``:
            ``(∂x)_(n) y = -n x_(n-1) y`` and ``∂(x_(n) y) = (∂x)_(n) y +
            x_(n) ∂y``, for ``-2 <= n <= 2``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[2]]))._test_translation()
            """
            from sage.misc.misc import some_tuples
            tester = self._tester(**options)
            for x, y in some_tuples(tester.some_elements(), 2, tester._max_runs):
                dx = x.derivative()
                for n in range(-2, 3):
                    tester.assertEqual(dx.nproduct(y, n), -n * x.nproduct(y, n - 1))
                    tester.assertEqual(x.nproduct(y, n).derivative(),
                                       dx.nproduct(y, n) + x.nproduct(y.derivative(), n))

    class WithBasis(CategoryWithAxiom_over_base_ring):
        r"""
        Proto vertex algebras with a distinguished basis.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).WithBasis()
            Category of proto vertex algebras with basis over Rational Field
        """

    class Super(SuperModulesCategory):
        r"""
        Proto vertex superalgebras.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).Super()
            Category of super proto vertex algebras over Rational Field
        """
        def extra_super_categories(self):
            r"""
            The base category: a super proto vertex algebra is a proto
            vertex algebra.

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.ProtoVertexAlgebras(QQ).Super().extra_super_categories()
                [Category of proto vertex algebras over Rational Field]
            """
            return [self.base_category()]

        class WithBasis(CategoryWithAxiom_over_base_ring):
            r"""
            Proto vertex superalgebras with a distinguished basis: also
            super modules with basis, so that elements have Sage's
            ``is_even_odd`` and ``is_super_homogeneous`` (from the parity
            of the basis vectors, ``_even_odd_on_basis``).

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.NeveuSchwarz(QQ, 1); L, G = V.gens()
                sage: G.is_even_odd(), L.no(G).is_even_odd(), L.is_even_odd()
                (1, 1, 0)
                sage: (L + G).is_super_homogeneous(), (L + L.no(L)).is_super_homogeneous()
                (False, True)
                sage: va.FermionicGhosts(QQ).gen('b').is_even_odd()       # no degrees
                1
            """
            def extra_super_categories(self):
                r"""
                Super modules with basis.

                EXAMPLES::

                    sage: from vertex_algebras import vertex_algebra_categories as vac
                    sage: vac.ProtoVertexAlgebras(QQ).Super().WithBasis().extra_super_categories()
                    [Category of super modules with basis over Rational Field]
                """
                from sage.categories.modules_with_basis import ModulesWithBasis
                return [ModulesWithBasis(self.base_ring()).Super()]

            class ParentMethods:
                r"""
                The parity of basis vectors (before the default of super
                modules with basis, which reads it off the degree).

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.NeveuSchwarz(QQ, 1)
                    sage: [V._even_odd_on_basis(x.leading_support()) for x in V.gens()]
                    [0, 1]
                """
                def _even_odd_on_basis(self, m):
                    r"""
                    The parity of the basis vector ``m``.

                    TESTS::

                        sage: import vertex_algebras as va
                        sage: V = va.NeveuSchwarz(QQ, 1); L, G = V.gens()
                        sage: x = G.no(G.derivative()) + L.no(G)
                        sage: sorted(set(V._even_odd_on_basis(m) for m in x.support()))
                        [0, 1]
                    """
                    return self._parity_key(m)

        class ParentMethods:
            r"""
            Methods of proto vertex superalgebras: the parity of basis
            vectors (``_even_odd_on_basis``, Sage's super module interface).

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.NeveuSchwarz(QQ, 1); L, G = V.gens()
                sage: [V._even_odd_on_basis(x.leading_support()) for x in (L, G, L.no(G))]
                [0, 1, 1]
            """
            def _even_odd_on_basis(self, m):
                r"""
                The parity of the basis vector ``m``.

                TESTS::

                    sage: import vertex_algebras as va
                    sage: V = va.NeveuSchwarz(QQ, 1); L, G = V.gens()
                    sage: V._even_odd_on_basis(G.leading_support()), V._even_odd_on_basis(
                    ....:     V.one().leading_support())
                    (1, 0)
                """
                return self._parity_key(m)

        class Graded(GradedModulesCategory):
            r"""
            Conformally graded proto vertex superalgebras (the same category as
            ``Graded().Super()``).

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.ProtoVertexAlgebras(QQ).Super().Graded()
                Category of conformally graded super proto vertex algebras over Rational Field
            """
            def _repr_object_names(self):
                r"""
                TESTS::

                    sage: from vertex_algebras import vertex_algebra_categories as vac
                    sage: vac.ProtoVertexAlgebras(QQ).Super().Graded()._repr_object_names()
                    'conformally graded super proto vertex algebras over Rational Field'
                """
                return 'conformally graded {}'.format(
                    self.base_category()._repr_object_names())

    class Graded(GradedModulesCategory):
        r"""
        Conformally graded proto vertex algebras: graded by conformal weights
        (``wt(x_(n) y) = wt x + wt y - n - 1``), from a Virasoro vector or
        chosen without one.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.ProtoVertexAlgebras(QQ).Graded()
            Category of conformally graded proto vertex algebras over Rational Field
        """
        def Super(self, base_ring=None):
            r"""
            The super subcategory: ``Super`` and ``Graded`` commute
            (``Graded().Super()`` is ``Super().Graded()``).

            INPUT:

            - ``base_ring`` -- (default: ``None``) ignored (the signature of
              Sage's ``Super``)

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: C = vac.ProtoVertexAlgebras(QQ)
                sage: C.Graded().Super() is C.Super().Graded()
                True
            """
            return self.base_category().Super().Graded()

        def _repr_object_names(self):
            r"""
            TESTS::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.ProtoVertexAlgebras(QQ).Graded()._repr_object_names()
                'conformally graded proto vertex algebras over Rational Field'
            """
            return 'conformally graded {}'.format(
                self.base_category()._repr_object_names())

        class ParentMethods:
            r"""
            Methods of conformally graded proto vertex algebras: degrees
            (conformal weights), weight spaces, characters, modes in the
            physics convention, singular vectors.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
                sage: V.degrees(), V.character(6)
                ({'L': 2}, 1 + q^2 + q^3 + 2*q^4 + 2*q^5 + O(q^6))
            """
            def degree_on_basis(self, m):
                r"""
                The conformal weight of the basis vector (key) ``m``.

                INPUT:

                - ``m`` -- a basis key

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
                    sage: {V.degree_on_basis(m) for m in L.no(L.derivative()).support()}
                    {5}
                    sage: V.degree_on_basis(V.one().leading_support())
                    0
                """
                return self._weight_key(m, self._weight_data())

            def degrees(self):
                r"""
                The conformal weights of the strong generators, by name.

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: va.N2(QQ, 1).degrees()
                    {'G1': 3/2, 'G2': 3/2, 'J': 1, 'L': 2}
                    sage: va.Lattice(QQ, matrix([[2]])).with_degrees(
                    ....:     {'J0': 1, 'e^(1)': 1, 'e^(-1)': 1}).degrees()
                    {'J0': 1, 'e^(-1)': 1, 'e^(1)': 1}
                """
                self._weight_data()
                return {s: self.degree(self.gen(s))
                        for s in self.variable_names()}

            def degree(self, x):
                r"""
                The conformal weight of a homogeneous element ``x`` (a
                ``ValueError`` if ``x`` is not homogeneous).

                INPUT:

                - ``x`` -- an element (anything the parent converts)

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.NeveuSchwarz(QQ, 1); L, G = V.gens()
                    sage: V.degree(L.no(G.derivative())), V.degree(V.one())
                    (9/2, 0)
                    sage: V.degree(L + G)
                    Traceback (most recent call last):
                    ...
                    ValueError: L + G is not homogeneous of a single conformal weight
                """
                data = self._weight_data()
                x = self(x)
                ws = {self._weight_key(m, data) for m in x.support()}
                if len(ws) != 1:
                    raise ValueError('{} is not homogeneous of a single conformal '
                                     'weight'.format(x))
                return ws.pop()

            def weight_space(self, weight, horizontal_grade=None):
                r"""
                The span of the basis monomials of conformal weight ``weight`` and,
                if ``horizontal_grade`` is given, of that grade.

                INPUT:

                - ``weight`` -- a conformal weight (a rational number)
                - ``horizontal_grade`` -- (default: ``None``) an element of
                  :meth:`~vertex_algebra.VertexAlgebra.horizontal_grading_group`,
                  or ``None`` for all grades

                OUTPUT:

                A :class:`weight_spaces.WeightSpace` with ``dimension()``,
                ``basis_keys()``, ``basis()`` and lazy iteration.

                Finite dimensionality is decided exactly: the space is finite
                dimensional when some ``μ`` makes ``h_i + μ·g_i`` positive for all
                bosonic generators (weights ``h_i``, grades ``g_i``); otherwise it is
                infinite dimensional as soon as it is nonempty and only lazy
                iteration is available.

                EXAMPLES:

                In the βγ system (``β`` weight 1, ``γ`` weight 0) the weight 3 space
                is infinite dimensional (multiply by ``γ``), but its grade 0 part
                for the grading ``β ↦ 1``, ``γ ↦ -1`` is finite::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
                    sage: N.<b, g> = NormallyOrderedExpressions(QQ)
                    sage: V = UniversalVertexAlgebra(QQ, N, {('b', 'g'): [-1]},
                    ....:                            conformal_vector=-b*g.derivative(),
                    ....:                            horizontal_grading={'b': 1, 'g': -1})
                    sage: b, g = V.gens()
                    sage: V.weight_space(3).is_finite_dimensional()
                    False
                    sage: S = V.weight_space(3, horizontal_grade=0); S
                    Weight space of conformal weight 3 and grade 0 of Universal vertex algebra on generators ['b', 'g'] over Rational Field (dimension 6)
                    sage: S.basis()
                    [:b b b g g g:, :db b g g:, :db dg:, :d^2b g:, :dg b b g:, :d^2g b:]
                    sage: S.basis_keys()[0]
                    ((0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0))
                    sage: all(x.degree() == 3 and x.horizontal_grade() == 0 for x in S)
                    True

                TESTS:

                The basis vectors are eigenvectors of ``T_(1)``, with ``T_(0) =
                ∂``, of the given grade; the charge-0 dimensions are the
                coefficients of ``z^0`` in ``Π_{n >= 1} 1/((1 - z q^n)(1 - z^{-1}
                q^{n-1}))``::

                    sage: T = V.conformal_vector()
                    sage: all(T.nproduct(x, 1) == n*x and T.nproduct(x, 0) == x.derivative()
                    ....:     and x.horizontal_grade() == G for n in range(5) for G in (-1, 0, 1, 2)
                    ....:     for x in V.weight_space(n, horizontal_grade=G))
                    True
                    sage: R.<z> = LaurentPolynomialRing(QQ); P.<q> = PowerSeriesRing(R, default_prec=6)
                    sage: f = prod(sum((z*q^n)^k for k in range(6))
                    ....:          * sum((q^(n-1)*z^-1)^k for k in range(6)) for n in range(1, 7))
                    sage: [V.weight_space(n, horizontal_grade=0).dimension() for n in range(6)]
                    [1, 1, 3, 6, 12, 21]
                    sage: [f[n].constant_coefficient() for n in range(6)]
                    [1, 1, 3, 6, 12, 21]

                Gradings with torsion: by ``Z × Z/2`` (``β ↦ (1, 1)``, ``γ ↦ (-1,
                1)``: the ``Z/2`` part counts the letters mod 2), by a
                multiplicative ``Z × C_3`` (the same spaces as the ``Z``-grading)
                and by ``Z/2`` alone (infinite dimensional spaces)::

                    sage: A = AdditiveAbelianGroup([0, 2]); u, v = A.gens()
                    sage: Vx = V.with_horizontal_grading({'b': u + v, 'g': -u + v})
                    sage: all(set(Vx.weight_space(n, horizontal_grade=G*u + e*v).basis_keys())
                    ....:     == {m for m in V.weight_space(n, horizontal_grade=G).basis_keys()
                    ....:         if len(m) % 2 == e}
                    ....:     for n in range(5) for G in (-1, 0, 1, 2) for e in (0, 1))
                    True
                    sage: M = AbelianGroup([0, 3]); a1, a2 = M.gens()
                    sage: Vm = V.with_horizontal_grading({'b': a1*a2, 'g': a1^-1*a2^-1})
                    sage: all(Vm.weight_space(n, horizontal_grade=a1^G*a2^G).basis_keys()
                    ....:     == V.weight_space(n, horizontal_grade=G).basis_keys()
                    ....:     for n in range(4) for G in (-1, 0, 1))
                    True
                    sage: from itertools import islice
                    sage: S = V.with_horizontal_grading({'b': 1, 'g': 1}, group=Zmod(2)).weight_space(
                    ....:     2, horizontal_grade=0)
                    sage: S.is_finite_dimensional()
                    False
                    sage: all(x.degree() == 2 and x.horizontal_grade() == 0 for x in islice(S, 20))
                    True

                The indefinite lattice ``I_{1,1}`` with a background charge (``c =
                2 - 12 B(α, α)``): infinite weight spaces, finite ones for the
                lattice grading, infinite again for the grade ``q_0 + q_1`` (the
                direction ``(1, -1)`` is null)::

                    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                    sage: I11 = LatticeVertexAlgebra(QQ, matrix([[1, 0], [0, -1]]))
                    sage: K0, K1 = I11.heisenberg_currents()
                    sage: I11 = I11.with_conformal_vector(K0.no(K0)/2 - K1.no(K1)/2
                    ....:                                 + K0.derivative()/2 + K1.derivative()/3)
                    sage: I11.central_charge() == 2 - 12*(1/4 - 1/9)
                    True
                    sage: I11.weight_space(1).is_finite_dimensional()
                    False
                    sage: Ig = I11.with_horizontal_grading('lattice'); T = Ig.conformal_vector()
                    sage: grades = [(0, 0), (1, 0), (0, 1), (1, 1), (2, -1)]
                    sage: [[Ig.weight_space(n, horizontal_grade=G).dimension() for G in grades]
                    ....:  for n in (-1, 0, 1, 2, 5/2)]
                    [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [5, 5, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
                    sage: all(T.nproduct(x, 1) == n*x for n in (-1, 0, 1, 2, 5/2) for G in grades
                    ....:     for x in Ig.weight_space(n, horizontal_grade=G))
                    True
                    sage: I11.with_horizontal_grading([1, 1]).weight_space(
                    ....:     0, horizontal_grade=0).is_finite_dimensional()
                    False

                A symbolic background charge: the weights of a free boson stay
                rational, those of ``e^q`` do not, and weight spaces need
                rational weights::

                    sage: K.<a> = Frac(QQ['a'])
                    sage: N.<J> = NormallyOrderedExpressions(K)
                    sage: H = UniversalVertexAlgebra(K, N, {('J', 'J'): [0, 1]},
                    ....:                            conformal_vector=J*J/2 + a*J.derivative())
                    sage: H.central_charge(), H.weight_space(2).basis()
                    (-12*a^2 + 1, [:J J:, dJ])
                    sage: LK = LatticeVertexAlgebra(K, matrix([[2]])); JK = LK.J(0)
                    sage: LK = LK.with_conformal_vector(JK.no(JK)/4 + a*JK.derivative())
                    sage: LK.e((1,)).degree()
                    -2*a + 1
                    sage: LK.weight_space(1)
                    Traceback (most recent call last):
                    ...
                    ValueError: the lattice weight coefficient -2*a is not a rational number; weight spaces need rational weights
                """
                from .weight_spaces import WeightSpace, _to_rational
                data = self._weight_data()
                W = _to_rational(weight)
                G = None
                if horizontal_grade is not None:
                    G = self._require_grading()['group'].coords(horizontal_grade)
                enum = self._weight_space_enumeration(W, G, data)
                return WeightSpace(self, W, None if G is None else self._grade_out(G),
                                   enum)

            def _mode(self, n, u, v):
                r"""
                The mode ``u_n v`` in the physics convention: for ``u`` homogeneous
                of conformal weight ``h``, the coefficient of ``z^{-n-h}`` in
                ``Y(u, z) v``, i.e. ``u_(n+h-1) v`` (:meth:`~vertex_algebra.VertexAlgebra.nproduct`).  ``n`` must
                lie in ``ZZ - h`` (half-integers for ``h = 3/2``).  Needs
                conformal weights (:meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector` or
                :meth:`~vertex_algebra.VertexAlgebra.with_degrees`); ``u = 0`` gives ``0``.

                EXAMPLES:

                The ``N = 1`` superconformal algebra at ``c = 3/2``: ``L_n`` and the
                Neveu–Schwarz modes ``G_r``, ``r`` in ``ZZ + 1/2``::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
                    sage: N.<L, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
                    sage: V = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 3/4],
                    ....:                                    ('L', 'G'): [G.derivative(), 3/2*G],
                    ....:                                    ('G', 'G'): [2*L, 0, 1]},
                    ....:                            degrees={'L': 2, 'G': 3/2})
                    sage: L, G = V.gens()
                    sage: L.mode(G, -1) == G.derivative(), L.mode(G, 0), L.mode(L, 2)
                    (True, 3/2*G, 3/4)
                    sage: G.mode(G, -1/2), G.mode(G, 3/2), G.mode(G, 1/2)
                    (2*L, 1, 0)
                    sage: L.mode(L, -2) == L.no(L) == L.nproduct(L, -1)
                    True
                    sage: G.mode(G, 1)
                    Traceback (most recent call last):
                    ...
                    ValueError: the modes u_n of u = G (conformal weight 3/2) need n in ZZ + 1/2, not n = 1
                    sage: (L + G).mode(G, 0)
                    Traceback (most recent call last):
                    ...
                    ValueError: L + G is not homogeneous of a single conformal weight
                    sage: V.zero().mode(G, 1/3)
                    0
                """
                from sage.rings.rational_field import QQ

                from .vertex_algebra import _module_of
                M = _module_of(v)
                if M is not None:
                    return M._mode(n, u, v)
                if not u:
                    return self.zero()
                h = self.degree(u)
                try:
                    h, n = QQ(h), QQ(n)
                except (TypeError, ValueError):
                    raise ValueError('modes need a rational conformal weight and '
                                     'index (u = {} has weight {}, n = {})'.format(
                                         u, h, n))
                k = n + h - 1
                if not k.is_integer():
                    raise ValueError('the modes u_n of u = {} (conformal weight {}) '
                                     'need n in ZZ{}, not n = {}'.format(
                                         u, h, '' if h.is_integer()
                                         else ' + {}'.format(-h - (-h).floor()), n))
                return self._nprod_el(u, int(k), v)

            def character(self, prec, horizontal_grade=None):
                r"""
                The character ``Σ_W dim V_W q^W`` over the conformal
                weights ``W < prec`` (of the grade ``horizontal_grade`` if given).

                INPUT:

                - ``prec`` -- the precision: weights ``W < prec`` are counted
                - ``horizontal_grade`` -- (default: ``None``) a grade, or
                  ``None`` for all grades

                OUTPUT:

                A Puiseux series in ``q`` (weights may be fractional) with
                precision ``prec``.  The weight spaces must be finite
                dimensional (e.g. positive weights; or give a grade), else
                a ``ValueError``.

                EXAMPLES:

                The Lee–Yang minimal model ``L(-22/5, 0)`` (the
                Rogers–Ramanujan series ``Π 1/((1-q^{5n+2})(1-q^{5n+3}))``)
                and the free fermion (weights ``1/2``)::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
                    sage: X.<L> = NormallyOrderedExpressions(QQ)
                    sage: Vir = UniversalVertexAlgebra(QQ, X, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
                    ....:                              conformal_vector='L')
                    sage: L = Vir.gen(0)
                    sage: Vir.character(8)
                    1 + q^2 + q^3 + 2*q^4 + 2*q^5 + 4*q^6 + 4*q^7 + O(q^8)
                    sage: Q = Vir.quotient([L.no(L) - 3/10*L.derivative(2)])
                    sage: Q.character(12)
                    1 + q^2 + q^3 + q^4 + q^5 + 2*q^6 + 2*q^7 + 3*q^8 + 3*q^9 + 4*q^10 + 4*q^11 + O(q^12)
                    sage: F.<p> = NormallyOrderedExpressions(QQ, parities={'p': 1})
                    sage: Maj = UniversalVertexAlgebra(QQ, F, {('p', 'p'): [1]},
                    ....:                              conformal_vector=p.derivative()*p/2)
                    sage: Maj.character(4)
                    1 + q^(1/2) + q^(3/2) + q^2 + q^(5/2) + q^3 + q^(7/2) + O(q^4)
                    sage: Maj.hilbert_series(2) == Maj.character(2)
                    True

                TESTS:

                The ``N = 1`` algebra has the PBW character ``Π_{n >= 0} (1 +
                q^{n+3/2}) / (1 - q^{n+2})``; graded by ``G ↦ 1 mod 2`` its
                grade-1 part is the part of half-integer weight::

                    sage: N.<L, G> = NormallyOrderedExpressions(QQ, parities={'G': 1})
                    sage: N1 = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 7/20],
                    ....:     ('L', 'G'): [G.derivative(), 3/2*G], ('G', 'G'): [2*L, 0, 7/15]},
                    ....:     conformal_vector='L')
                    sage: ch = N1.character(15/2); q = ch.parent().gen()
                    sage: ch == (prod(1 + q^(n + 3/2) for n in range(7))
                    ....:        / prod(1 - q^(n + 2) for n in range(7))).add_bigoh(15/2)
                    True
                    sage: N1g = N1.with_horizontal_grading({'G': 1}, group=Zmod(2))
                    sage: c0, c1 = [N1g.character(15/2, horizontal_grade=e) for e in (0, 1)]
                    sage: c1
                    q^(3/2) + q^(5/2) + 2*q^(7/2) + 3*q^(9/2) + 5*q^(11/2) + 7*q^(13/2) + O(q^(15/2))
                    sage: c0 + c1 == ch, all(e in ZZ for e in c0.exponents())
                    (True, True)

                The ``A_2`` lattice with the Sugawara vector has the character
                ``θ_{A_2} / η^2``; its classes for torsion gradings (``q_0 - q_1
                mod 3``, and ``ZZ^2 / 2 ZZ^2``) add up to it::

                    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                    sage: A2 = LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]]))
                    sage: J0, J1 = A2.heisenberg_currents()
                    sage: A2 = A2.with_conformal_vector((2*J0.no(J0) + J0.no(J1) + J1.no(J0)
                    ....:                                + 2*J1.no(J1))/6)
                    sage: A2.character(6)
                    1 + 8*q + 17*q^2 + 46*q^3 + 98*q^4 + 198*q^5 + O(q^6)
                    sage: P.<q> = PowerSeriesRing(QQ, default_prec=6)
                    sage: (sum(q^(a*a - a*b + b*b) for a in range(-4, 5) for b in range(-4, 5))
                    ....:  / prod((1 - q^n)^2 for n in range(1, 6)))
                    1 + 8*q + 17*q^2 + 46*q^3 + 98*q^4 + 198*q^5 + O(q^6)
                    sage: A2t = A2.with_horizontal_grading([1, -1], group=Zmod(3))
                    sage: sum(A2t.character(5, horizontal_grade=e) for e in range(3)) == A2.character(5)
                    True
                    sage: Q22 = (ZZ^2) / (ZZ^2).span([[2, 0], [0, 2]])
                    sage: A2q = A2.with_horizontal_grading(list(Q22.gens()), group=Q22)
                    sage: sum(A2q.character(5, horizontal_grade=Q22((a, b)))
                    ....:     for a in range(2) for b in range(2)) == A2.character(5)
                    True
                    sage: all(m[1][0] % 2 == 1 and m[1][1] % 2 == 0 for n in range(4)
                    ....:     for m in A2q.weight_space(n, horizontal_grade=Q22((1, 0))).basis_keys())
                    True

                ``V_Z`` graded by ``q mod 2``: the even part is ``θ_{2Z} / η``::

                    sage: Z = LatticeVertexAlgebra(QQ, matrix([[1]])); I = Z.J(0)
                    sage: Z = Z.with_conformal_vector(I.no(I)/2).with_horizontal_grading(
                    ....:     [1], group=Zmod(2))
                    sage: Z.character(8, horizontal_grade=0)
                    1 + q + 4*q^2 + 5*q^3 + 9*q^4 + 13*q^5 + 21*q^6 + 29*q^7 + O(q^8)
                    sage: P.<q> = PowerSeriesRing(QQ, default_prec=8)
                    sage: sum(q^(2*a*a) for a in range(-3, 4)) / prod(1 - q^n for n in range(1, 8))
                    1 + q + 4*q^2 + 5*q^3 + 9*q^4 + 13*q^5 + 21*q^6 + 29*q^7 + O(q^8)

                The character of a tensor product is the product of the
                characters (the Virasoro algebra at ``c = -22/5`` times ``V_Z``)::

                    sage: F = LatticeVertexAlgebra(QQ, matrix([[1]])); I = F.J(0)
                    sage: F = F.with_conformal_vector(I.no(I)/2)
                    sage: P = Vir.tensor(F, conformal_vector='factors')
                    sage: P.character(5) == (Vir.character(5) * F.character(5)).add_bigoh(5)
                    True
                """
                from sage.rings.infinity import Infinity
                from sage.rings.integer_ring import ZZ
                from sage.rings.puiseux_series_ring import PuiseuxSeriesRing
                from sage.rings.rational_field import QQ
                data = self._weight_data()
                G = None
                if horizontal_grade is not None:
                    G = self._require_grading()['group'].coords(horizontal_grade)
                lb, D = self._character_bounds(G, data)
                if lb is None:
                    raise ValueError('the weight spaces{} are not certified to be '
                                     'finite dimensional: give a grade'.format(
                                         '' if G is None else ' of this grade'))
                P = PuiseuxSeriesRing(ZZ, 'q')
                q = P.gen()
                out = P.zero()
                if lb != Infinity:
                    prec = QQ(prec)
                    k = (QQ(lb) * D).ceil()
                    while QQ(k) / D < prec:
                        W = QQ(k) / D
                        n = self.weight_space(W, horizontal_grade).dimension()
                        if n:
                            out += n * q ** W
                        k += 1
                return out.add_bigoh(prec)

            def hilbert_series(self, prec, horizontal_grade=None):
                r"""
                Alias of :meth:`character`.

                INPUT:

                - ``prec`` -- the precision: weights ``W < prec`` are counted
                - ``horizontal_grade`` -- (default: ``None``) a grade, or
                  ``None`` for all grades

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: va.FreeBosons(QQ).hilbert_series(6)
                    1 + q + 2*q^2 + 3*q^3 + 5*q^4 + 7*q^5 + O(q^6)
                """
                return self.character(prec, horizontal_grade)

            def homogeneous_component_basis(self, d):
                r"""
                The basis of the weight space of weight ``d`` (see
                :meth:`weight_space`), Sage's graded modules interface.

                INPUT:

                - ``d`` -- a conformal weight

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
                    sage: V.homogeneous_component_basis(4)
                    [:L L:, d^2L]
                """
                return self.weight_space(d).basis()

            def _test_degrees(self, **options):
                r"""
                Weights of products on pairs of homogeneous
                ``some_elements()``: ``x_(n) y`` has weight ``wt x + wt y -
                n - 1`` and ``∂x`` weight ``wt x + 1``.

                EXAMPLES::

                    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                    sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])); J = L.J(0)
                    sage: L.with_conformal_vector(J.no(J)/4)._test_degrees()
                """
                from sage.misc.misc import some_tuples
                tester = self._tester(**options)

                def weight(x):
                    try:
                        return x.degree()
                    except ValueError:
                        return None
                for x in tester.some_elements():
                    w = weight(x)
                    if w is not None and not x.derivative().is_zero():
                        tester.assertEqual(weight(x.derivative()), w + 1)
                for x, y in some_tuples(tester.some_elements(), 2, tester._max_runs):
                    wx, wy = weight(x), weight(y)
                    if wx is None or wy is None:
                        continue
                    for n in range(-2, len(x.ope(y))):
                        p = x.nproduct(y, n)
                        if not p.is_zero():
                            tester.assertEqual(weight(p), wx + wy - n - 1)

            def find_singular(self, weight, horizontal_grade=None, zero_modes=()):
                r"""
                A basis of the singular vectors of degree ``weight`` (and
                horizontal grade ``horizontal_grade``): the elements ``v``
                killed by the positive modes ``g_(n) v = 0``, ``n > h_g -
                1``, of the strong generators ``g`` (``h_g`` their
                degrees), and by the zero modes ``z_(h_z - 1) v = 0`` of the
                elements ``z`` of ``zero_modes`` (e.g. the positive root
                currents of an affine algebra, for singular vectors of
                highest weight).  Positive modes of the strong generators
                suffice: those of their normally ordered products are sums
                of products of modes ending in one that kills ``v``.  Also in
                quotients and subalgebras (their strong generators); the
                weight space must be finite dimensional.

                INPUT:

                - ``weight`` -- the degree (conformal weight) of the singular
                  vectors
                - ``horizontal_grade`` -- (default: ``None``) their grade, or
                  ``None`` for all grades
                - ``zero_modes`` -- (default: ``()``) elements of integer
                  degree whose zero modes must also kill the vectors

                OUTPUT:

                A list of elements, a basis of the singular vectors, each
                monic in its leading monomial.

                EXAMPLES:

                The Virasoro algebra at ``c = -22/5`` (Lee--Yang): ``Λ =
                :LL: - 3/10 ∂²L`` at degree 4, which vanishes in the
                quotient::

                    sage: import vertex_algebras as va
                    sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
                    sage: L = V.gen('L'); V.find_singular(4)
                    [:L L: - 3/10*d^2L]
                    sage: V.find_singular(2), V.find_singular(0)
                    ([], [1])
                    sage: Q = V / V.find_singular(4); Q.find_singular(4)
                    []
                    sage: (L.no(L) - 3/10*L.derivative(2)).is_singular(), L.is_singular()
                    (True, False)

                Affine ``sl_2`` at level 1: ``E_{-1}^2 |0>`` (degree 2, root
                lattice grade 2) is singular of highest weight::

                    sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'])
                    sage: E = A.gen('E'); A.find_singular(2, horizontal_grade=(2,), zero_modes=[E])
                    [:E E:]
                    sage: A.find_singular(2, horizontal_grade=(0,), zero_modes=[E])
                    []

                Without the zero modes, also the ``F_0``-descendants of
                ``:E E:`` count (``F_0`` commutes with the positive modes)::

                    sage: F = A.gen('F'); A.find_singular(2, horizontal_grade=(0,))
                    [-2*:E F: + :H H: + dH]
                    sage: s = F.mode(F.mode(E.no(E), 0), 0); s, s.is_singular()
                    (-4*:E F: + 2*:H H: + 2*dH, True)

                TESTS:

                In the minimal models ``M(p, q)`` the first singular vector
                has degree ``(p-1)(q-1)``; the results agree with the kernel
                of ``L_1, L_2`` computed directly::

                    sage: import vertex_algebras as va
                    sage: def kernel_dim(V, W):         # of L_1, L_2 on the weight space
                    ....:     L = V.gen('L'); B = V.weight_space(W).basis()
                    ....:     ims = [[L.mode(b, 1), L.mode(b, 2)] for b in B]
                    ....:     keys = sorted({(m, k) for im in ims for m in (0, 1)
                    ....:                    for k in im[m].support()})
                    ....:     return matrix(QQ, len(B), len(keys),
                    ....:                   [[im[m].coefficient(k) for m, k in keys]
                    ....:                    for im in ims]).left_nullity()
                    sage: for p, q in [(2, 5), (3, 4), (2, 7)]:
                    ....:     V = va.Virasoro(QQ, 1 - 6*(p - q)^2/(p*q), conformal_vector='L')
                    ....:     w = (p - 1)*(q - 1)
                    ....:     print([len(V.find_singular(W)) for W in range(w + 1)],
                    ....:           all(len(V.find_singular(W)) == kernel_dim(V, W)
                    ....:               for W in range(1, w + 1)))
                    [1, 0, 0, 0, 1] True
                    [1, 0, 0, 0, 0, 0, 1] True
                    [1, 0, 0, 0, 0, 0, 1] True
                    sage: K.<c> = Frac(QQ['c'])
                    sage: V = va.Virasoro(K, c, conformal_vector='L')
                    sage: [len(V.find_singular(W)) for W in range(9)]
                    [1, 0, 0, 0, 0, 0, 0, 0, 0]

                Affine ``sl_2`` at level 2: ``E_{-1}^3 |0>``, nothing below;
                none at a generic level::

                    sage: A = va.Affine(QQ, ['A', 1], 2, names=['E', 'H', 'F'])
                    sage: E = A.gen('E')
                    sage: A.find_singular(3, horizontal_grade=(3,), zero_modes=[E])
                    [:E E E:]
                    sage: all(A.find_singular(W, horizontal_grade=(g,), zero_modes=[E]) == []
                    ....:     for W in range(1, 3) for g in range(-W, W + 1))
                    True
                    sage: Kk.<k> = Frac(QQ['k'])
                    sage: Ag = va.Affine(Kk, ['A', 1], k, names=['E', 'H', 'F'])
                    sage: all(Ag.find_singular(W, horizontal_grade=(g,), zero_modes=[Ag.gen('E')])
                    ....:     == [] for W in (1, 2) for g in range(-W, W + 1))
                    True

                Simple algebras have none of positive degree: ``<T>`` in the
                Majorana fermion (``L(1/2, 0)``) and ``V_{A_1}``::

                    sage: F = va.FreeFermions(QQ); psi = F.gen(0)
                    sage: S = F.subalgebra([psi.derivative().no(psi)/2], names=['T'])
                    sage: all(S.find_singular(W) == [] for W in range(2, 9))
                    True
                    sage: LA = va.Lattice(QQ, matrix([[2]])); J = LA.J(0)
                    sage: LA = LA.with_conformal_vector(J.no(J)/4)
                    sage: LA = LA.with_horizontal_grading('lattice')
                    sage: all(LA.find_singular(W, horizontal_grade=g) == []
                    ....:     for W in range(1, 5) for g in range(-2, 3))
                    True

                The zero mode of a field of non-integer degree is refused::

                    sage: NS = va.NeveuSchwarz(QQ, 1, conformal_vector='L')
                    sage: NS.find_singular(2, zero_modes=[NS.gen('G')])
                    Traceback (most recent call last):
                    ...
                    ValueError: G has no zero mode (degree 3/2)
                """
                from sage.rings.rational_field import QQ

                from ._keyed_spans import _modular_left_kernel, left_kernel
                from .vertex_algebra import _nproc
                basis = list(self.weight_space(weight, horizontal_grade).basis())
                conds = self._singular_conditions(zero_modes)
                vecs = _singular_image_vectors(self, basis, conds)
                rels = None
                if self.base_ring() is QQ and len(basis) >= SINGULAR_MODULAR_MIN:
                    # few relations among many vectors: modulo primes, then
                    # checked exactly (see _modular_left_kernel)
                    rels = _modular_left_kernel(vecs, nthreads=_nproc())
                    if rels is not None and not all(_is_relation(vecs, r) for r in rels):
                        rels = None
                if rels is None:
                    rels = left_kernel(vecs, self.base_ring())
                del vecs
                out = []
                order = getattr(self, '_key_order', None)
                for r in rels:
                    v = sum((c * basis[j] for j, c in r.items()), self.zero())
                    d = v.monomial_coefficients()
                    lead = max(d, key=order) if order is not None else max(d)
                    out.append(v * ~d[lead])          # monic in the leading monomial
                return out

            def _singular_conditions(self, zero_modes=()):
                r"""
                ``[(a, n_min, n_max)]``: the modes ``a_(n)``, ``n_min <= n
                <= n_max`` (``None``: no bound), that must kill singular
                vectors (see :meth:`find_singular`).

                If ``self`` has a conformal vector ``T`` and every strong
                generator ``g`` is ``T`` or primary of positive degree ``h``
                (``T_(1) g = h g``, ``T_(n) g = 0`` for ``n >= 2``),
                ``L_1``, ``L_2`` and the first positive mode ``g_{m_0}`` of
                each generator suffice (also ``g_{m_0 + 1}`` if ``h - 1 =
                m_0``): ``[L_n, g_m] = ((h - 1) n - m) g_{n+m}``, so
                ``g_{m+1}`` is a multiple of ``[L_1, g_m]`` unless ``m = h -
                1``, and ``g_h`` of ``[L_2, g_{h-2}]`` (coefficient ``h``)
                when ``h - 2 >= m_0``; ``L_1``, ``L_2`` generate all
                ``L_n``, ``n >= 1``.  (W_3: ``L_1``, ``L_2``, ``W_1``.)

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.W3(QQ, 2, conformal_vector='L')
                    sage: [(str(a), lo, hi) for a, lo, hi in V._singular_conditions()]
                    [('L', 2, 3), ('W', 3, 3)]
                    sage: N = va.NeveuSchwarz(QQ, 1, conformal_vector='L')    # G of weight 3/2
                    sage: [(str(a), lo, hi) for a, lo, hi in N._singular_conditions()]
                    [('L', 2, 3), ('G', 1, 2)]
                    sage: [(str(a), lo, hi) for a, lo, hi in va.W3(QQ, 2)._singular_conditions()]   # no T
                    [('L', 2, None), ('W', 3, None)]
                    sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'])
                    sage: [(str(a), lo, hi) for a, lo, hi in A._singular_conditions()]
                    [('E', 1, None), ('H', 1, None), ('F', 1, None)]
                """
                from sage.functions.other import floor
                from sage.rings.integer_ring import ZZ
                gens = [self(g) for g in self._singular_generators()]
                T = self._singular_reduction(gens)
                conds = []
                for g in gens:
                    h = self.degree(g)
                    n0 = floor(h - 1) + 1               # g_(n0) = g_{m0}, m0 = n0 - h + 1
                    if T is None:
                        conds.append((g, n0, None))
                    elif g == T or h - 1 == n0 - h + 1:
                        conds.append((g, n0, n0 + 1))  # L_1, L_2; or g_{m0}, g_{m0+1}
                    else:
                        conds.append((g, n0, n0))
                if T is not None and all(g != T for g in gens):
                    conds.append((T, ZZ(2), ZZ(3)))
                for z in zero_modes:
                    z = self(z)
                    n = self.degree(z) - 1
                    if n not in ZZ:
                        raise ValueError('{} has no zero mode (degree {})'.format(
                            z, self.degree(z)))
                    conds.append((z, ZZ(n), ZZ(n)))
                return conds

            def _singular_generators(self):
                r"""
                The strong generators whose positive modes define singular
                vectors.

                TESTS::

                    sage: import vertex_algebras as va
                    sage: va.Virasoro(QQ, 1/2, conformal_vector='L')._singular_generators()
                    (L,)
                """
                return self._strong_generators_for_ideals()

            def _singular_reduction(self, gens):
                r"""
                The conformal vector ``T`` if every element of ``gens`` is
                ``T`` or primary of positive degree for it (then two modes
                per generator suffice, see :meth:`_singular_conditions`),
                else ``None``.

                TESTS::

                    sage: import vertex_algebras as va
                    sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
                    sage: V._singular_reduction([L])
                    L
                    sage: V._singular_reduction([L, L.no(L)]) is None       # :LL: is not primary
                    True
                    sage: va.Virasoro(QQ, 1/2)._singular_reduction([L]) is None   # no conformal vector
                    True
                """
                T = self.conformal_vector()
                if T is None:
                    return None
                for g in gens:
                    if g == T:
                        continue
                    h = self.degree(g)
                    if not h > 0:
                        return None
                    poles = T.ope(g)
                    if len(poles) > 2 or (len(poles) == 2 and poles[1] != h * g):
                        return None
                return T

        class ElementMethods:
            r"""
            Methods of the elements of conformally graded proto vertex
            algebras: :meth:`degree`, :meth:`mode`, :meth:`is_singular`.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
                sage: L.degree(), L.mode(L, 2)
                (2, 1/4)
            """
            def degree(self):
                r"""
                The conformal weight of a homogeneous element (with respect
                to the chosen Virasoro vector or conformal weights, see
                :meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector`).

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.FreeFermions(QQ); p = V.gen(0)
                    sage: p.degree(), p.no(p.derivative()).degree()
                    (1/2, 2)
                """
                return self.parent().degree(self)

            def is_singular(self, zero_modes=()):
                r"""
                Whether this homogeneous element is a singular vector: killed
                by the positive modes of the strong generators (and the zero
                modes of ``zero_modes``), see
                :meth:`~vertex_algebra_categories.ProtoVertexAlgebras.Graded.ParentMethods.find_singular`.

                INPUT:

                - ``zero_modes`` -- (default: ``()``) elements of integer
                  degree whose zero modes must also kill ``self``

                EXAMPLES:

                ``Λ = :LL: - 3/10 ∂²L`` is singular exactly at ``c =
                -22/5``::

                    sage: import vertex_algebras as va
                    sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
                    sage: (L.no(L) - 3/10*L.derivative(2)).is_singular(), L.is_singular()
                    (True, False)
                    sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
                    sage: (L.no(L) - 3/10*L.derivative(2)).is_singular()
                    False

                Affine ``sl_2`` at level 1: ``:E E:`` is also killed by the
                zero mode of ``E``, while ``:H H:`` is not even singular::

                    sage: A = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'])
                    sage: E, H, F = A.gens()
                    sage: E.no(E).is_singular(zero_modes=[E]), H.no(H).is_singular()
                    (True, False)
                """
                P = self.parent()
                self.degree()           # homogeneous
                return not _singular_images(self, P._singular_conditions(zero_modes))

            def mode(self, right, n):
                r"""
                Return the mode ``self_n right`` in the physics convention: the
                coefficient of ``z^{-n-h}`` in ``Y(self, z) right``, where ``h`` is
                the conformal weight of ``self``, i.e. ``self_(n+h-1) right``.

                INPUT:

                - ``right`` -- an element of the vertex algebra, or a vector
                  of a module over it
                - ``n`` -- the mode index, in ``ZZ - h``

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L'); L, G = V.gens()
                    sage: L.mode(G, 0), G.mode(G, 3/2), L.mode(L, -2) == L.no(L)
                    (3/2*G, 1, True)
                    sage: M = V.adjoint_module(); G.mode(M(G), -1/2)
                    2*L
                    sage: G.mode(G, 1)
                    Traceback (most recent call last):
                    ...
                    ValueError: the modes u_n of u = G (conformal weight 3/2) need n in ZZ + 1/2, not n = 1
                """
                from .vertex_algebra import _common, _module_of
                M = _module_of(right)
                if M is not None:
                    return M._mode(n, self, right)
                a, b = _common(self, right)
                return a.parent()._mode(n, a, b)


#: Spans of at least this many vectors: :meth:`find_singular` over ``QQ``
#: computes the relations modulo primes (and checks them exactly).


SINGULAR_MODULAR_MIN = 2000

#: Batches of at least this many basis vectors: the images are computed in
#: chunks of ``SINGULAR_CHUNK``, the root algebra's caches emptied between
#: them (bounded memory; m = 4 of the W_3 test, degree 32: 22 GB peak in
#: chunks, out of memory beyond 45 GB in one batch).
SINGULAR_CHUNK_MIN = 4000
SINGULAR_CHUNK = 1000


def _is_relation(vecs, r):
    r"""
    Whether ``Σ_j r[j] vecs[j] = 0`` (exactly).

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: vac._is_relation([{'a': 1}, {'a': 2, 'b': 1}], {0: 2, 1: -1})
        False
        sage: vac._is_relation([{'a': 1}, {'a': 2}], {0: 2, 1: -1})
        True
    """
    tot = {}
    for j, c in r.items():
        for k, x in vecs[j].items():
            tot[k] = tot.get(k, 0) + c * x
    return not any(tot.values())


def _release_memory():
    r"""
    Give freed heap memory back to the system (glibc ``malloc_trim``;
    nothing elsewhere).

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: vac._release_memory()
    """
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(ctypes.c_size_t(0))
    except (OSError, AttributeError):
        pass


def _singular_image_vectors(P, basis, conds):
    r"""
    The images of the elements ``basis`` of ``P`` under the modes of
    ``conds`` (see ``_singular_conditions``), raw dicts ``{(k, n, key):
    coefficient}``: with finitely many modes in batches of the C++ engine
    (of the root algebra for subalgebras and commutants; large ones in
    chunks, see ``SINGULAR_CHUNK_MIN``), else element by element.

    TESTS:

    Virasoro at ``c = -22/5``: ``L_1, L_2`` on the weight-4 space; the two
    image vectors are proportional (``Λ = :LL: - 3/10 ∂²L`` is singular), and
    the batched path agrees with :func:`_singular_images`::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
        sage: conds = V._singular_conditions(()); conds
        [(L, 2, 3)]
        sage: B = list(V.weight_space(4).basis()); B
        [:L L:, d^2L]
        sage: vecs = vac._singular_image_vectors(V, B, conds)
        sage: sorted(vecs[0].items()), sorted(vecs[1].items())
        ([((0, 2, ((0, 1),)), 3), ((0, 3, ((0, 0),)), 18/5)],
         [((0, 2, ((0, 1),)), 10), ((0, 3, ((0, 0),)), 12)])
        sage: vecs == [vac._singular_images(x, conds) for x in B]
        True
        sage: vac._is_relation(vecs, {0: 10, 1: -3})
        True
    """
    if all(hi is not None for _, _, hi in conds):
        root = P._root if getattr(P, '_is_subspace_algebra', False) else P
        gens = [dict(a._monomial_coefficients) for a, _, _ in conds]
        k0 = len(conds)
        step = SINGULAR_CHUNK if len(basis) >= SINGULAR_CHUNK_MIN else max(len(basis), 1)
        vecs = []
        for start in range(0, len(basis), step):
            part = basis[start:start + step]
            els = gens + [dict(x._monomial_coefficients) for x in part]
            tasks = [(k, int(n), k0 + j) for j in range(len(part))
                     for k, (_, lo, hi) in enumerate(conds) for n in range(lo, hi + 1)]
            res = root._nprod_raw_batch(els, tasks) if tasks else []
            if res is None:
                break
            out = [{} for _ in part]
            for (k, n, j), d in zip(tasks, res):
                v = out[j - k0]
                for key, c in d.items():
                    if c:
                        v[(k, n, key)] = c
            vecs.extend(out)
            del res
            if step < len(basis):
                root.clear_caches()
                _release_memory()
        else:
            return vecs
    return [_singular_images(x, conds) for x in basis]


def _singular_images(x, conds):
    r"""
    The images of ``x`` under the modes of ``conds`` (see
    ``_singular_conditions``), a raw dict ``{(k, n, key): coefficient}``.

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
        sage: conds = V._singular_conditions(())
        sage: vac._singular_images(L.no(L) - 3/10*L.derivative(2), conds)
        {}
        sage: sorted(vac._singular_images(L, conds).items())
        [((0, 3, ()), -11/5)]
    """
    out = {}
    for k, (a, nmin, nmax) in enumerate(conds):
        ope = a.ope(x)
        top = len(ope) - 1 if nmax is None else nmax
        for n in range(nmin, top + 1):
            p = ope[n] if 0 <= n < len(ope) else (a.nproduct(x, n) if n < 0 else None)
            if p is None:
                continue
            for key, c in p.monomial_coefficients().items():
                if c:
                    out[(k, n, key)] = c
    return out


class VertexAlgebras(Category_over_base_ring):
    r"""
    The category of vertex (super)algebras over a commutative ring.

    Parents implement the vertex algebra API of
    :class:`vertex_algebra.VertexAlgebra` (``one``, ``x.nproduct(y, n)``,
    ``x.ope(y)``, ``x.derivative(n)``, ``x.parity()``, ...); the category
    adds the axiom checks run by ``TestSuite``.

    INPUT:

    - ``base_ring`` -- a commutative ring

    EXAMPLES::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: C = vac.VertexAlgebras(QQ); C
        Category of vertex algebras over Rational Field
        sage: C.super_categories()
        [Category of proto vertex algebras over Rational Field]
        sage: C.WithBasis()
        Category of vertex algebras with basis over Rational Field
    """

    @cached_method
    def super_categories(self):
        r"""
        The super categories: proto vertex algebras.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.VertexAlgebras(QQ).super_categories()
            [Category of proto vertex algebras over Rational Field]
        """
        return [ProtoVertexAlgebras(self.base_ring())]

    class ParentMethods:
        r"""
        Methods of all vertex algebras: :meth:`lie_conformal_algebra` and
        the skew-symmetry and Borcherds checks of ``TestSuite``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Weyl(QQ); V._test_skew_symmetry(); V._test_borcherds()
            sage: V.lie_conformal_algebra()
            Lie conformal algebra of Universal vertex algebra on generators ['alpha0', 'alpha1'] over Rational Field
        """

        def lie_conformal_algebra(self):
            r"""
            The Lie conformal algebra underlying ``self`` (the same space,
            ``T = ∂``, the ``λ``-bracket of the products ``a_(n) b``, ``n >=
            0``), in Sage's category of Lie conformal algebras; see
            :class:`lie_conformal.LieConformalAlgebraOfVertexAlgebra`.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: L = LatticeVertexAlgebra(QQ, matrix([[1]]))
                sage: P = L.lie_conformal_algebra(); P.category()
                Category of super Lie conformal algebras with basis over Rational Field
                sage: P(L.e((1,))).bracket(P(L.e((-1,))))
                {0: 1}
            """
            from .lie_conformal import LieConformalAlgebraOfVertexAlgebra
            return LieConformalAlgebraOfVertexAlgebra(self)

        def _test_skew_symmetry(self, **options):
            r"""
            Skew-symmetry on pairs of homogeneous ``some_elements()``:
            ``y_(n) x = (-1)^{|x||y|} Σ_j (-1)^{n+j+1} ∂^j (x_(n+j) y)/j!``
            for ``-2 <= n`` up to the last pole.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[1]]))._test_skew_symmetry()
            """
            from sage.misc.misc import some_tuples
            tester = self._tester(**options)
            for x, y in some_tuples(tester.some_elements(), 2, tester._max_runs):
                px, py = _parity(x), _parity(y)
                if px is None or py is None:
                    continue
                s = (-1) ** (px * py)
                xy = x.ope(y)
                for n in range(-2, len(y.ope(x)) + 1):
                    rhs = self.zero()
                    for j in range(max(len(xy) - n, 0) + 1):
                        rhs += ((-1) ** (n + j + 1) / factorial(j)
                                * x.nproduct(y, n + j).derivative(j))
                    tester.assertEqual(y.nproduct(x, n), s * rhs)

        def _test_borcherds(self, **options):
            r"""
            Borcherds' commutator formula on triples of homogeneous
            ``some_elements()``: ``[x_(m), y_(n)] z = Σ_j C(m, j)
            (x_(j) y)_(m+n-j) z`` for ``0 <= m, n <= 2`` and the
            normal ordered product ``m = -1``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[2]]))._test_borcherds()

            TESTS:

            Negative control: ``W_3`` with a wrong ``β`` (``1`` instead of
            ``16/(22 + 5c)``) fails; declared ``proto``, the same OPEs give a
            proto vertex algebra, without the Jacobi tests and without a Lie
            conformal algebra::

                sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
                sage: from vertex_algebras.vertex_algebra_categories import _is_proto
                sage: K.<c> = Frac(QQ['c']); N.<L, W> = NormallyOrderedExpressions(K)
                sage: Lam = L*L - 3/10*L.derivative(2)
                sage: def w3(beta, **kw):
                ....:     return UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
                ....:         ('L', 'W'): [W.derivative(), 3*W],
                ....:         ('W', 'W'): [L.derivative(3)/15 + beta*Lam.derivative(),
                ....:                      3/10*L.derivative(2) + 2*beta*Lam, L.derivative(), 2*L, 0,
                ....:                      c/3]}, **kw)
                sage: w3(16/(22 + 5*c))._test_borcherds()
                sage: w3(1)._test_borcherds()
                Traceback (most recent call last):
                ...
                AssertionError: ...
                sage: P = w3(1, proto=True, conformal_vector=L)
                sage: _is_proto(P), hasattr(P, 'lie_conformal_algebra')
                (True, False)
                sage: TestSuite(P).run()
            """
            from sage.misc.misc import some_tuples
            tester = self._tester(**options)
            for x, y, z in some_tuples(tester.some_elements(), 3, tester._max_runs):
                px, py = _parity(x), _parity(y)
                if px is None or py is None:
                    continue
                s = (-1) ** (px * py)
                xy = x.ope(y)
                for m in range(-1, 3):
                    for n in range(3):
                        lhs = (x.nproduct(y.nproduct(z, n), m)
                               - s * y.nproduct(x.nproduct(z, m), n))
                        rhs = self.zero()
                        for j in range(len(xy)):    # C(-1, j) = (-1)^j
                            rhs += binomial(m, j) * xy[j].nproduct(z, m + n - j)
                        tester.assertEqual(lhs, rhs)

    class WithBasis(CategoryWithAxiom_over_base_ring):
        r"""
        Vertex algebras with a distinguished basis (all vertex algebras of
        this library: the basis keys are their normally ordered monomials).

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2]])) in vac.VertexAlgebras(QQ).WithBasis()
            True
        """

    class Super(SuperModulesCategory):
        r"""
        Vertex superalgebras.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.VertexAlgebras(QQ).Super()
            Category of super vertex algebras over Rational Field
        """
        def extra_super_categories(self):
            r"""
            The base category: a vertex superalgebra is a vertex algebra.

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexAlgebras(QQ).Super().extra_super_categories()
                [Category of vertex algebras over Rational Field]
            """
            return [self.base_category()]

        class Graded(GradedModulesCategory):
            r"""
            Conformally graded vertex superalgebras (the same category as
            ``Graded().Super()``).

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexAlgebras(QQ).Super().Graded()
                Category of conformally graded super vertex algebras over Rational Field
            """
            def _repr_object_names(self):
                r"""
                TESTS::

                    sage: from vertex_algebras import vertex_algebra_categories as vac
                    sage: vac.VertexAlgebras(QQ).Super().Graded()._repr_object_names()
                    'conformally graded super vertex algebras over Rational Field'
                """
                return 'conformally graded {}'.format(
                    self.base_category()._repr_object_names())

    class Graded(GradedModulesCategory):
        r"""
        Conformally graded vertex algebras: graded by conformal weights
        (``wt(x_(n) y) = wt x + wt y - n - 1``), from a Virasoro vector or
        chosen without one.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.VertexAlgebras(QQ).Graded()
            Category of conformally graded vertex algebras over Rational Field
        """
        def Super(self, base_ring=None):
            r"""
            The super subcategory: ``Super`` and ``Graded`` commute
            (``Graded().Super()`` is ``Super().Graded()``).

            INPUT:

            - ``base_ring`` -- (default: ``None``) ignored (the signature of
              Sage's ``Super``)

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexAlgebras(QQ).Graded().Super()
                Category of conformally graded super vertex algebras over Rational Field
            """
            return self.base_category().Super().Graded()

        def _repr_object_names(self):
            r"""
            TESTS::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexAlgebras(QQ).Graded()._repr_object_names()
                'conformally graded vertex algebras over Rational Field'
            """
            return 'conformally graded {}'.format(
                self.base_category()._repr_object_names())

        class ParentMethods:
            r"""
            Methods of conformally graded vertex algebras: :meth:`zhu_algebra`.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: va.Virasoro(QQ, 1/2, degrees={'L': 2}).zhu_algebra()
                Zhu algebra of Universal vertex algebra on generators ['L'] over Rational Field
            """
            def zhu_algebra(self, twist=None, level=0):
                r"""
                Zhu's associative algebra ``A(V)`` [Zhu1996]_ (integer and
                half-integer conformal weights): a
                :class:`zhu_algebra.ZhuAlgebra` for universal vertex
                algebras, a :class:`zhu_algebra.LatticeZhuAlgebra` for
                lattice vertex algebras, a
                :class:`zhu_algebra.TensorProductZhuAlgebra` for tensor
                products of these.  For a quotient ``V/I`` (also as a
                tensor factor) see ``zhu_algebra_up_to``.

                INPUT:

                - ``twist`` -- (default: ``None``) ``None`` for the
                  untwisted Zhu algebra, or ``'H'`` (or ``'ramond'``): the
                  H-twisted Zhu algebra (the modules twisted by ``e^{2πi
                  L_0}``: Ramond sectors), any rational weights; see
                  :mod:`zhu_algebra`.  Not yet for lattice vertex algebras
                  with non-integer weights.
                - ``level`` -- (default: ``0``) ``n >= 1``: the level ``n`` Zhu
                  algebra of Dong--Li--Mason [DLM1998]_ ``A_n(V)``, a
                  :class:`higher_zhu_algebra.HigherZhuAlgebra` (universal
                  vertex algebras with positive integer weights)

                EXAMPLES::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]},
                    ....:                            degrees={'J': 1})
                    sage: A = H.zhu_algebra(); A, A.relations()
                    (Zhu algebra of Universal vertex algebra on generators ['J'] over Rational Field, {})
                    sage: J = H.gen('J'); A(J.no(J.derivative()))
                    -[J]^2
                    sage: Vir = UniversalVertexAlgebra(QQ, ['L'], {('L', 'L'): [
                    ....:     {(('L', 1),): 1}, {(('L', 0),): 2}, 0, -11/5]}, degrees={'L': 2})
                    sage: L = Vir.gen('L')
                    sage: (Vir / [L.no(L) - 3/10*L.derivative(2)]).zhu_algebra()
                    Traceback (most recent call last):
                    ...
                    NotImplementedError: the Zhu algebra of a quotient is computed up to a weight: use zhu_algebra_up_to(max_weight)

                The Ramond sector of the ``N = 1`` algebra: ``[G]^2 = [L] -
                c/24``::

                    sage: import vertex_algebras as va
                    sage: A = va.NeveuSchwarz(QQ, 3/2, conformal_vector='L').zhu_algebra(twist='ramond')
                    sage: A
                    H-twisted Zhu algebra of Universal vertex algebra on generators ['L', 'G'] over Rational Field

                Level 1 (Dong--Li--Mason)::

                    sage: va.Virasoro(QQ, 1/2, conformal_vector='L').zhu_algebra(level=1).gens()
                    (L_0, L_{-1} L_1)
                """
                from .zhu_algebra import _zhu_algebra
                return _zhu_algebra(self, twist, level)


class VertexOperatorAlgebras(Category_over_base_ring):
    r"""
    The category of vertex operator (super)algebras: conformally graded
    vertex algebras with a chosen Virasoro vector ``T`` (``conformal_vector()``,
    ``central_charge()``), ``T_(0) = ∂``, the conformal weights those of
    ``L_0 = T_(1)`` (generalized eigenvalues for logarithmic ones).

    INPUT:

    - ``base_ring`` -- a commutative ring

    EXAMPLES::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: vac.VertexOperatorAlgebras(QQ).super_categories()
        [Category of conformally graded vertex algebras over Rational Field]
    """

    @cached_method
    def super_categories(self):
        r"""
        The super categories: conformally graded vertex algebras.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.VertexOperatorAlgebras(QQ).super_categories()
            [Category of conformally graded vertex algebras over Rational Field]
        """
        return [VertexAlgebras(self.base_ring()).Graded()]

    class ParentMethods:
        r"""
        Methods of vertex operator algebras: the central charge and the
        Jordan structure of ``L_0``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, conformal_vector='sugawara')
            sage: V.central_charge(), V.l0_is_diagonalizable()
            (1, True)
        """
        def central_charge(self):
            r"""
            The central charge of the chosen Virasoro vector.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: va.Virasoro(QQ, 1/2, conformal_vector='L').central_charge()
                1/2
                sage: va.FermionicGhosts(QQ, b_weight=2).central_charge()
                -26
            """
            return self._require_conformal()['c']

        def l0_is_diagonalizable(self):
            r"""
            True if ``L_0 = T_(1)`` of the chosen Virasoro vector is
            diagonalizable (no Jordan blocks); see :meth:`~vertex_algebra.VertexAlgebra.with_conformal_vector`.

            EXAMPLES:

            The lattice vertex algebra of ``A_1`` with the Sugawara vector,
            and with ``T + ∂e^α`` (logarithmic)::

                sage: import vertex_algebras as va, warnings
                sage: L = va.Lattice(QQ, matrix([[2]])); J = L.J(0)
                sage: L.with_conformal_vector(J.no(J)/4).l0_is_diagonalizable()
                True
                sage: with warnings.catch_warnings():
                ....:     warnings.simplefilter('ignore')
                ....:     LT = L.with_conformal_vector(J.no(J)/4 + L.e((1,)).derivative())
                sage: LT.l0_is_diagonalizable()
                False
            """
            return not self._require_conformal()['jordan']

        def l0_jordan_chains(self):
            r"""
            The Jordan chains ``[x, D x, D^2 x, ...]`` (up to the last nonzero
            term) of the nilpotent part ``D = T_(1) - L_0'`` of ``L_0``, for the
            generators ``x`` with ``D x != 0``, by generator label; empty iff
            ``L_0`` is diagonalizable.  ``D`` is a derivation, so these
            determine it.

            EXAMPLES:

            The lattice vertex algebra of ``A_1`` with ``T = :JJ:/4 +
            ∂e^α``::

                sage: import vertex_algebras as va, warnings
                sage: L = va.Lattice(QQ, matrix([[2]])); J = L.J(0)
                sage: with warnings.catch_warnings():
                ....:     warnings.simplefilter('ignore')
                ....:     LT = L.with_conformal_vector(J.no(J)/4 + L.e((1,)).derivative())
                sage: LT.l0_jordan_chains()
                {'J0': [J0, 2*e^(1)], 'e^(-1)': [e^(-1), -J0, -2*e^(1)]}
                sage: L.with_conformal_vector(J.no(J)/4).l0_jordan_chains()
                {}
            """
            return {s: list(ch)
                    for s, ch in self._require_conformal()['jordan'].items()}

        def _test_conformal_vector(self, **options):
            r"""
            The Virasoro vector: ``T(z)T(w) ~ (c/2)/(z-w)^4 + 2T/(z-w)^2 +
            ∂T/(z-w)``, and on ``some_elements()`` ``T_(0) x = ∂x`` and
            ``(T_(1) - wt x)`` nilpotent on homogeneous ``x``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])); J = L.J(0)
                sage: L.with_conformal_vector(J.no(J)/4)._test_conformal_vector()
            """
            tester = self._tester(**options)
            T = self.conformal_vector()
            tester.assertIsNotNone(T)
            c = self.central_charge()
            poles = T.ope(T)
            full = poles + [self.zero()] * max(0, 4 - len(poles))
            tester.assertTrue(len(poles) <= 4)
            tester.assertEqual(full[0], T.derivative())
            tester.assertEqual(full[1], 2 * T)
            tester.assertTrue(full[2].is_zero())
            if c is not None:
                tester.assertEqual(full[3], c / 2 * self.one())
            for x in tester.some_elements():
                tester.assertEqual(T.nproduct(x, 0), x.derivative())
                try:
                    w = x.degree()
                except ValueError:
                    continue
                y = x
                for _ in range(20):
                    y = T.nproduct(y, 1) - w * y
                    if y.is_zero():
                        break
                tester.assertTrue(y.is_zero())

    class WithBasis(CategoryWithAxiom_over_base_ring):
        r"""
        Vertex operator algebras with a distinguished basis.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: va.Virasoro(QQ, 1/2, conformal_vector='L').category()
            Category of finitely strongly generated vertex operator algebras with basis over Rational Field
            sage: vac.VertexOperatorAlgebras(QQ).WithBasis()
            Category of vertex operator algebras with basis over Rational Field
        """

    class Super(SuperModulesCategory):
        r"""
        Vertex operator superalgebras.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: vac.VertexOperatorAlgebras(QQ).Super()
            Category of super vertex operator algebras over Rational Field
        """
        def extra_super_categories(self):
            r"""
            The base category: a vertex operator superalgebra is a vertex
            operator algebra.

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: vac.VertexOperatorAlgebras(QQ).Super().extra_super_categories()
                [Category of vertex operator algebras over Rational Field]
            """
            return [self.base_category()]


# ---------------------------------------------------------------------- #
#  Modules over a vertex algebra                                         #
# ---------------------------------------------------------------------- #

def _vertex_algebra_module_category(V, super=False, graded=False):
    """
    The category of a module over the vertex algebra ``V`` with basis:
    super if it has odd basis vectors, conformally graded if it has
    conformal weights.

    EXAMPLES::

        sage: from vertex_algebras.vertex_algebra_categories import _vertex_algebra_module_category
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: _vertex_algebra_module_category(H, super=True, graded=True)
        Category of conformally graded super modules with basis over Universal vertex algebra on generators ['J'] over Rational Field
    """
    C = VertexAlgebraModules(V)
    if graded:
        C = C.Graded()
    if super:
        C = C.Super()
    return C.WithBasis()


def _some_vertex_algebra_elements(V, k=4):
    r"""
    At most ``k`` homogeneous (for the parity) elements of ``V``.

    TESTS::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: import vertex_algebras as va
        sage: vac._some_vertex_algebra_elements(va.FreeFermions(QQ))
        [1, psi, dpsi]
        sage: vac._some_vertex_algebra_elements(va.Virasoro(QQ, 1/2), 2)
        [1, L]
    """
    return [x for x in V.some_elements() if _parity(x) is not None][:k]


class VertexAlgebraModules(Category_over_base):
    r"""
    The category of (left) modules over a vertex (super)algebra ``V``: (super)
    modules ``M`` over the base ring with fields ``Y_M(a, z) = Σ_n a_(n)
    z^{-n-1}`` (``a ∈ V``), ``a_(n) m = 0`` for ``n`` large, the vacuum
    acting as the identity, and the Borcherds identity.  A category over
    ``V`` (as Sage's ``AlgebraModules(A)`` over an algebra), contained in the
    modules over the base ring of ``V``; nested ``WithBasis``, ``Super`` and
    ``Graded`` (conformally graded: by the eigenvalues of ``L_0``, or by
    conformal weights chosen without a conformal vector) as for
    :class:`VertexAlgebras`.

    Parents implement the API of :class:`vertex_algebra_module.VertexAlgebraModule`;
    the elements of ``V`` act through their own methods (``a.nproduct(m, n)``,
    ``a.ope(m)``, ``a.mode(m, n)``, ``a * m``).  ``TestSuite`` checks the
    axioms on ``some_elements()`` of ``V`` and of the module.

    INPUT:

    - ``V`` -- a vertex (super)algebra

    EXAMPLES::

        sage: from vertex_algebras import vertex_algebra_categories as vac
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
        sage: C = vac.VertexAlgebraModules(H); C
        Category of modules over Universal vertex algebra on generators ['J'] over Rational Field
        sage: C.super_categories()
        [Category of vector spaces over Rational Field]
        sage: C.Graded().Super() is C.Super().Graded()
        True
        sage: C.Super().Graded().WithBasis().is_subcategory(Modules(QQ).WithBasis())
        True
        sage: TestSuite(C).run(); loads(dumps(C)) is C
        True
    """

    def __init__(self, V):
        r"""
        TESTS::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2); vac.VertexAlgebraModules(V).base() is V
            True
            sage: vac.VertexAlgebraModules(QQ)
            Traceback (most recent call last):
            ...
            TypeError: Rational Field is not a vertex algebra
        """
        if not getattr(V, '_is_vertex_algebra', False):
            raise TypeError('{} is not a vertex algebra'.format(V))
        Category_over_base.__init__(self, V)

    def base_ring(self):
        r"""
        The base ring of the vertex algebra.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: K.<k> = Frac(QQ['k'])
            sage: vac.VertexAlgebraModules(va.Affine(K, ['A', 1], k)).base_ring()
            Fraction Field of Univariate Polynomial Ring in k over Rational Field
        """
        return self.base().base_ring()

    @cached_method
    def super_categories(self):
        r"""
        The super categories: modules over the base ring of the vertex
        algebra.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: vac.VertexAlgebraModules(va.FreeBosons(QQ)).super_categories()
            [Category of vector spaces over Rational Field]
        """
        return [Modules(self.base_ring())]

    def _repr_object_names(self):
        r"""
        TESTS::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: vac.VertexAlgebraModules(va.FreeBosons(QQ))._repr_object_names()
            "modules over Universal vertex algebra on generators ['alpha'] over Rational Field"
        """
        return 'modules over {}'.format(self.base())

    class ParentMethods:
        r"""
        Methods of all modules over a vertex algebra: the vacuum,
        translation and Borcherds checks of ``TestSuite``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: M = va.FreeBosons(QQ).adjoint_module()
            sage: M._test_module_vacuum(); M._test_module_borcherds()
        """

        def _test_module_vacuum(self, **options):
            r"""
            The vacuum acts as the identity: ``1_(n) m = δ_{n,-1} m`` on
            ``some_elements()``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[2]])).adjoint_module()._test_module_vacuum()
            """
            tester = self._tester(**options)
            one = self.vertex_algebra().one()
            for m in tester.some_elements():
                tester.assertEqual(one.ope(m), [])
                tester.assertEqual(one.no(m), m)
                for n in (-3, -2):
                    tester.assertTrue(one.nproduct(m, n).is_zero())

        def _test_module_translation(self, **options):
            r"""
            ``(∂a)_(n) m = -n a_(n-1) m`` for ``a`` in ``some_elements()``
            of the vertex algebra, ``m`` in those of the module, ``-2 <= n
            <= 2``.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[2]])).adjoint_module()._test_module_translation()
            """
            tester = self._tester(**options)
            for a in _some_vertex_algebra_elements(self.vertex_algebra()):
                da = a.derivative()
                for m in tester.some_elements():
                    for n in range(-2, 3):
                        if not (self._product_in_window(da, n, m)
                                and self._product_in_window(a, n - 1, m)):
                            continue
                        tester.assertEqual(da.nproduct(m, n), -n * a.nproduct(m, n - 1))

        def _test_module_borcherds(self, **options):
            r"""
            Borcherds' commutator formula ``a_(p) b_(q) m - (-1)^{|a||b|}
            b_(q) a_(p) m = Σ_j C(p, j) (a_(j) b)_(p+q-j) m`` (``0 <= p <=
            2``, ``-2 <= q <= 2``) and the normally ordered product ``(a_(-1)
            b)_(q) m = Σ_j a_(-1-j) b_(q+j) m + (-1)^{|a||b|} Σ_j b_(q-1-j)
            a_(j) m`` (``-2 <= q <= 2``) for ``a``, ``b`` in
            ``some_elements()`` of the vertex algebra and ``m`` in those of
            the module.

            EXAMPLES::

                sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                sage: LatticeVertexAlgebra(QQ, matrix([[1]])).adjoint_module()._test_module_borcherds()
            """
            from sage.misc.misc import some_tuples
            tester = self._tester(**options)
            els = _some_vertex_algebra_elements(self.vertex_algebra())
            vecs = list(tester.some_elements())[:4]
            for a, b in some_tuples(els, 2, tester._max_runs):
                s = (-1) ** (a.parity() * b.parity())
                ab = a.ope(b)
                nab = a.nproduct(b, -1)
                for m in vecs:
                    am, bm = a.ope(m), b.ope(m)
                    for q in range(-2, 3):
                        for p in range(3):
                            lhs = a.nproduct(b.nproduct(m, q), p) - s * b.nproduct(a.nproduct(m, p), q)
                            rhs = self.zero()
                            for j in range(min(p + 1, len(ab))):
                                rhs += binomial(p, j) * ab[j].nproduct(m, p + q - j)
                            tester.assertEqual(lhs, rhs)
                        rhs = self.zero()
                        for j in range(max(len(bm) - q, 0)):
                            rhs += a.nproduct(b.nproduct(m, q + j), -1 - j)
                        for j in range(len(am)):
                            rhs += s * b.nproduct(am[j], q - 1 - j)
                        tester.assertEqual(nab.nproduct(m, q), rhs)

    class WithBasis(CategoryWithAxiom_over_base_ring):
        r"""
        Modules over a vertex algebra with a distinguished basis.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: import vertex_algebras as va
            sage: V = va.FreeBosons(QQ); vac.VertexAlgebraModules(V).WithBasis()
            Category of modules with basis over Universal vertex algebra on generators ['alpha'] over Rational Field
            sage: V.adjoint_module() in vac.VertexAlgebraModules(V).WithBasis()
            True
        """

    class Super(SuperModulesCategory):
        r"""
        Super modules over a vertex (super)algebra.

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: vac.VertexAlgebraModules(H).Super()
            Category of super modules over Universal vertex algebra on generators ['J'] over Rational Field
        """
        def extra_super_categories(self):
            r"""
            The base category: a super module over ``V`` is a module over
            ``V``.

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: import vertex_algebras as va
                sage: C = vac.VertexAlgebraModules(va.FreeBosons(QQ))
                sage: C.Super().extra_super_categories()
                [Category of modules over Universal vertex algebra on generators ['alpha'] over Rational Field]
            """
            return [self.base_category()]

        class ElementMethods:
            r"""
            Parities of vectors from the basis (Sage's methods of super
            modules with basis, taken over: the category of super modules
            with basis cannot join that of graded modules over a vertex
            algebra, and the generic super module methods read the parity
            off the degree).

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.NeveuSchwarz(QQ, 3/2); L, G = V.gens(); M = V.adjoint_module()
                sage: M(G).is_even_odd(), M(G.no(L)).is_even_odd(), M(L).is_even_odd()
                (1, 1, 0)
                sage: (M(L) + M(G)).is_super_homogeneous(), (M(L) + M(G)).odd_component() == M(G)
                (False, True)
            """
            from sage.categories.super_modules_with_basis import (
                SuperModulesWithBasis as _S,
            )
            is_super_homogeneous = _S.ElementMethods.is_super_homogeneous
            is_even_odd = _S.ElementMethods.is_even_odd
            even_component = _S.ElementMethods.even_component
            odd_component = _S.ElementMethods.odd_component
            del _S

        class ParentMethods:
            r"""
            Methods of super modules: the parity of basis vectors
            (``_even_odd_on_basis``, Sage's super module interface).

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.FreeFermions(QQ); M = V.adjoint_module(); p = V.gen(0)
                sage: [M._even_odd_on_basis(M(x).leading_support()) for x in (p, p.no(p.derivative()))]
                [1, 0]
            """
            def _even_odd_on_basis(self, m):
                r"""
                The parity of the basis vector ``m``.

                TESTS::

                    sage: import vertex_algebras as va
                    sage: V = va.FreeFermions(QQ); M = V.adjoint_module(); p = V.gen(0)
                    sage: M._even_odd_on_basis(M(p).leading_support()), M._even_odd_on_basis(
                    ....:     M(V.one()).leading_support())
                    (1, 0)
                """
                return self._parity_key(m)

        class Graded(GradedModulesCategory):
            r"""
            Conformally graded super modules over a vertex algebra (the same
            category as ``Graded().Super()``).

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: import vertex_algebras as va
                sage: C = vac.VertexAlgebraModules(va.FreeFermions(QQ)); C.Super().Graded()
                Category of conformally graded super modules over Universal vertex algebra on generators ['psi'] over Rational Field
                sage: C.Super().Graded() is C.Graded().Super()
                True
            """
            def _repr_object_names(self):
                r"""
                TESTS::

                    sage: from vertex_algebras import vertex_algebra_categories as vac
                    sage: import vertex_algebras as va
                    sage: C = vac.VertexAlgebraModules(va.FreeFermions(QQ)).Super().Graded()
                    sage: C._repr_object_names()
                    "conformally graded super modules over Universal vertex algebra on generators ['psi'] over ..."
                """
                return 'conformally graded {}'.format(
                    self.base_category()._repr_object_names())

    class Graded(GradedModulesCategory):
        r"""
        Conformally graded modules: ``wt(a_(n) m) = wt a + wt m - n - 1``
        (the generalized eigenvalues of ``L_0``, or weights chosen without a
        conformal vector).

        EXAMPLES::

            sage: from vertex_algebras import vertex_algebra_categories as vac
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: vac.VertexAlgebraModules(H).Graded()
            Category of conformally graded modules over Universal vertex algebra on generators ['J'] over Rational Field
        """
        def Super(self, base_ring=None):
            r"""
            The super subcategory: ``Super`` and ``Graded`` commute
            (``Graded().Super()`` is ``Super().Graded()``).

            INPUT:

            - ``base_ring`` -- (default: ``None``) ignored (the signature of
              Sage's ``Super``)

            EXAMPLES::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: import vertex_algebras as va
                sage: C = vac.VertexAlgebraModules(va.FreeBosons(QQ))
                sage: C.Graded().Super() is C.Super().Graded()
                True
            """
            return self.base_category().Super().Graded()

        def _repr_object_names(self):
            r"""
            TESTS::

                sage: from vertex_algebras import vertex_algebra_categories as vac
                sage: import vertex_algebras as va
                sage: vac.VertexAlgebraModules(va.FreeBosons(QQ)).Graded()._repr_object_names()
                "conformally graded modules over Universal vertex algebra on generators ['alpha'] over ..."
            """
            return 'conformally graded {}'.format(
                self.base_category()._repr_object_names())

        class ParentMethods:
            r"""
            Methods of conformally graded modules: degrees, weight spaces,
            characters.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: M = va.FreeFermions(QQ).adjoint_module(); M.character(3)
                1 + q^(1/2) + q^(3/2) + q^2 + q^(5/2) + O(q^3)
            """
            def degree_on_basis(self, m):
                r"""
                The conformal weight of the basis vector (key) ``m``.

                INPUT:

                - ``m`` -- a basis key

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.FreeBosons(QQ); J = V.gen(0); M = V.adjoint_module()
                    sage: M.degree_on_basis(M(J.no(J.derivative())).leading_support())
                    3
                """
                return self._weight_key(m)

            def degree(self, x):
                r"""
                The conformal weight of a homogeneous element ``x`` (a
                ``ValueError`` if ``x`` is not homogeneous).

                INPUT:

                - ``x`` -- an element (anything the parent converts)

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.FreeBosons(QQ); J = V.gen(0); M = V.adjoint_module()
                    sage: M.degree(M(J.no(J))), M.degree(M(V.one()))
                    (2, 0)
                    sage: M.degree(M(J + J.no(J)))
                    Traceback (most recent call last):
                    ...
                    ValueError: ... is not homogeneous of a single conformal weight
                """
                x = self(x)
                ws = {self._weight_key(m) for m in x.support()}
                if len(ws) != 1:
                    raise ValueError('{} is not homogeneous of a single conformal '
                                     'weight'.format(x))
                return ws.pop()

            def weight_space(self, weight, horizontal_grade=None):
                r"""
                The span of the basis vectors of conformal weight ``weight``
                (and of the grade ``horizontal_grade``, if the module is graded).

                INPUT:

                - ``weight`` -- a conformal weight
                - ``horizontal_grade`` -- (default: ``None``) an element of
                  the module's grading group, or ``None`` for all grades

                OUTPUT:

                A :class:`weight_spaces.WeightSpace` with ``dimension()``,
                ``basis_keys()``, ``basis()`` and lazy iteration.

                EXAMPLES::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]},
                    ....:                            degrees={'J': 1})
                    sage: M = H.adjoint_module(); S = M.weight_space(3); S.basis()
                    [:J J J:, :dJ J:, d^2J]
                """
                from .weight_spaces import WeightSpace, _to_rational
                W = _to_rational(weight)
                G = None if horizontal_grade is None else self._grade_coords(horizontal_grade)
                return WeightSpace(self, W, horizontal_grade, self._weight_space_enumeration(W, G))

            def character(self, prec, horizontal_grade=None):
                r"""
                The character ``Σ_W dim M_W q^W`` over the conformal weights
                ``W < prec`` (of the grade ``horizontal_grade`` if given).

                INPUT:

                - ``prec`` -- the precision: weights ``W < prec`` are counted
                - ``horizontal_grade`` -- (default: ``None``) a grade, or
                  ``None`` for all grades

                OUTPUT:

                A Puiseux series in ``q``; the weight spaces must be finite
                dimensional (else a ``ValueError``).

                EXAMPLES::

                    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
                    sage: H = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]},
                    ....:                            degrees={'J': 1})
                    sage: H.adjoint_module().character(6)
                    1 + q + 2*q^2 + 3*q^3 + 5*q^4 + 7*q^5 + O(q^6)
                """
                from sage.rings.infinity import Infinity
                from sage.rings.integer_ring import ZZ
                from sage.rings.puiseux_series_ring import PuiseuxSeriesRing
                from sage.rings.rational_field import QQ
                G = None if horizontal_grade is None else self._grade_coords(horizontal_grade)
                lb, D = self._weight_bounds(G)
                if lb is None:
                    raise ValueError('the weight spaces{} are not certified to be '
                                     'finite dimensional: give a grade'.format(
                                         '' if G is None else ' of this grade'))
                P = PuiseuxSeriesRing(ZZ, 'q')
                q = P.gen()
                out = P.zero()
                if lb != Infinity:
                    prec = QQ(prec)
                    k = (QQ(lb) * D).ceil()
                    while QQ(k) / D < prec:
                        W = QQ(k) / D
                        n = self.weight_space(W, horizontal_grade).dimension()
                        if n:
                            out += n * q ** W
                        k += 1
                return out.add_bigoh(prec)

            def _test_module_weights(self, **options):
                r"""
                ``a_(n) m`` has weight ``wt a + wt m - n - 1`` for homogeneous
                ``a`` in ``some_elements()`` of the vertex algebra and ``m``
                in those of the module.

                EXAMPLES::

                    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
                    sage: L = LatticeVertexAlgebra(QQ, matrix([[2]])); J = L.J(0)
                    sage: L.with_conformal_vector(J.no(J)/4).adjoint_module()._test_module_weights()
                """
                tester = self._tester(**options)

                def weight(x):
                    try:
                        return x.degree()
                    except ValueError:
                        return None
                for a in _some_vertex_algebra_elements(self.vertex_algebra()):
                    wa = weight(a)
                    if wa is None:
                        continue
                    for m in tester.some_elements():
                        wm = weight(m)
                        if wm is None or not self._product_in_window(a, 0, m):
                            continue
                        for n in range(-2, len(a.ope(m))):
                            if not self._product_in_window(a, n, m):
                                continue
                            v = a.nproduct(m, n)
                            if not v.is_zero():
                                tester.assertEqual(weight(v), wa + wm - n - 1)

        class ElementMethods:
            r"""
            Methods of the vectors of conformally graded modules:
            :meth:`degree`.

            EXAMPLES::

                sage: import vertex_algebras as va
                sage: V = va.FreeBosons(QQ); M = V.adjoint_module()
                sage: M(V.gen(0).derivative()).degree()
                2
            """
            def degree(self):
                r"""
                The conformal weight of a homogeneous module vector.

                EXAMPLES::

                    sage: import vertex_algebras as va
                    sage: V = va.FreeFermions(QQ); p = V.gen(0); M = V.adjoint_module()
                    sage: M(p).degree(), M(p.no(p.derivative())).degree()
                    (1/2, 2)
                """
                return self.parent().degree(self)
