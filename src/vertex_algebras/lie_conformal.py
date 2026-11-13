# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Vertex algebras and Sage's Lie conformal algebras.

Two constructions linking :mod:`vertex_algebra_categories` with Sage's
Lie conformal algebras (``sage.algebras.lie_conformal_algebras``); vertex
algebras are not a subcategory of Lie conformal algebras in this library
(a design decision), so both directions are explicit:

- :class:`LieConformalAlgebraOfVertexAlgebra` (``V.lie_conformal_algebra()``):
  the Lie conformal algebra underlying a vertex algebra ``V`` -- the same
  space, the translation ``T`` and the ``λ``-bracket ``[a_λ b] = Σ_{n >= 0}
  λ^n/n! a_(n) b`` (the forgetful functor);
- :func:`universal_enveloping_vertex_algebra`: the universal enveloping
  vertex algebra of a finitely generated Lie conformal algebra of Sage (with
  structure coefficients: ``lie_conformal_algebras.Virasoro``, ``Affine``,
  ``NeveuSchwarz``, ``N2``, ``Weyl``, ``FreeBosons``, ``FreeFermions``,
  ``BosonicGhosts``, ``FermionicGhosts``, ``Abelian``, ...), with values
  for its central elements, as a
  :class:`universal_vertex_algebra.UniversalVertexAlgebra`.

EXAMPLES::

    sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
    sage: Vir = lie_conformal_algebras.Virasoro(QQ)
    sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2}); V
    Universal vertex algebra on generators ['L'] over Rational Field
    sage: L = V.gen('L'); L.ope(L)
    [dL, 2*L, 0, 1/4]
    sage: V.degrees(), V.category()
    ({'L': 2}, Category of finitely strongly generated conformally graded vertex algebras with basis over Rational Field)
    sage: P = V.lie_conformal_algebra(); P
    Lie conformal algebra of Universal vertex algebra on generators ['L'] over Rational Field
    sage: b = P(L.no(L)).bracket(P(L)); sorted(b), b[3], b[5]
    ([0, 1, 2, 3, 5], 17/2*L, 3/2)
    sage: TestSuite(P).run()          # includes Sage's Jacobi test of the λ-brackets

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.arith.misc import factorial
from sage.categories.lie_conformal_algebras import LieConformalAlgebras
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.structure.element import parent as _parent

from .vertex_algebra import _ArtFromRepr


class LieConformalAlgebraElement(IndexedFreeModuleElement):
    r"""
    An element of :class:`LieConformalAlgebraOfVertexAlgebra`: an element
    of the vertex algebra, with its ``λ``-bracket.

    EXAMPLES::

        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
        sage: Vir = lie_conformal_algebras.Virasoro(QQ)
        sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
        sage: x = V.lie_conformal_algebra()(V.gen('L')); x
        L
        sage: x.bracket(x)
        {0: dL, 1: 2*L, 3: 1/4}
    """

    def lift(self):
        """
        The element of the vertex algebra.

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: x = P(L.no(L)); x.lift() == L.no(L), x.lift().parent() is V
            (True, True)
        """
        V = self.parent()._V
        return V._from_dict(dict(self._monomial_coefficients), coerce=False,
                            remove_zeros=True)

    def _bracket_(self, rhs):
        r"""
        The ``λ``-bracket: the dict ``{n: self_(n) rhs}`` of the nonzero
        ``n``-th products, ``n >= 0``.

        EXAMPLES::

            sage: from vertex_algebras import lie_conformal
            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: V = UniversalVertexAlgebra(QQ, ['J'], {('J', 'J'): [0, 1]})
            sage: P = V.lie_conformal_algebra(); J = P(V.gen(0))
            sage: J.bracket(J), J.bracket(J.T()), J.nproduct(J, 1)
            ({1: 1}, {2: 2}, 1)
        """
        P = self.parent()
        poles = self.lift().ope(rhs.lift())
        return {n: P(p) for n, p in enumerate(poles) if p}

    def is_even_odd(self):
        """
        The parity (0 even, 1 odd) of a homogeneous element.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.FermionicGhosts(QQ); P = V.lie_conformal_algebra()
            sage: P(V.gen('b')).is_even_odd(), P(V.gen('b').no(V.gen('c'))).is_even_odd()
            (1, 0)
        """
        return self.lift().parity()

    def degree(self):
        """
        The conformal weight of a homogeneous element (H-graded).

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P(L).degree(), P(L.no(L).derivative()).degree()
            (2, 5)
        """
        return self.lift().degree()

    def T(self, n=1):
        r"""
        ``T^(n) self = ∂^n self / n!`` (divided powers, Sage's convention
        for Lie conformal algebras; the vertex algebra's ``derivative`` is
        ``∂^n``).

        INPUT:

        - ``n`` -- (default: ``1``) a nonnegative integer

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: x = P(L); x.T(), x.T(2), x.T(3)
            (dL, 1/2*d^2L, 1/6*d^3L)
            sage: x.T(2).lift() == L.derivative(2) / 2
            True
        """
        n = int(n)
        P = self.parent()
        return P(self.lift().derivative(n) / factorial(n))

    def _repr_(self):
        """
        The element prints as its lift to the vertex algebra.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: x = P(L.no(L)); repr(x), repr(x) == repr(x.lift())
            (':L L:', True)
        """
        return repr(self.lift())


class LieConformalAlgebraOfVertexAlgebra(_ArtFromRepr, CombinatorialFreeModule):
    r"""
    The Lie conformal algebra underlying a vertex algebra ``V``: the space
    of ``V`` (same basis) with ``T = ∂`` and the ``λ``-bracket ``[a_λ b] =
    Σ_{n >= 0} λ^n/n! a_(n) b`` (``x.bracket(y)`` gives ``{n: x_(n) y}``),
    in Sage's category of Lie conformal algebras (super, H-graded by the
    conformal weights when ``V`` is).  Use ``V.lie_conformal_algebra()``;
    ``P(v)`` converts an element of ``V``, ``x.lift()`` gives it back.

    INPUT:

    - ``V`` -- a vertex algebra

    EXAMPLES::

        sage: from vertex_algebras import lie_conformal
        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: V = UniversalVertexAlgebra(QQ, ['L', 'G'], {('L', 'L'): [{(('L', 1),): 1}, {(('L', 0),): 2}, 0, 3/4],
        ....:     ('L', 'G'): [{(('G', 1),): 1}, {(('G', 0),): 3/2}], ('G', 'G'): [{(('L', 0),): 2}, 0, 1]},
        ....:     parities={'G': 1}, degrees={'L': 2, 'G': 3/2})
        sage: P = V.lie_conformal_algebra(); P.category()
        Category of H-graded super Lie conformal algebras with basis over Rational Field
        sage: G = P(V.gen('G')); G.bracket(G), G.is_even_odd(), G.degree()
        ({0: 2*L, 2: 1}, 1, 3/2)
        sage: TestSuite(P).run()
    """
    Element = LieConformalAlgebraElement

    @staticmethod
    def __classcall_private__(cls, V):
        """
        Unique representation keyed on the vertex algebra ``V``.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: from vertex_algebras.lie_conformal import LieConformalAlgebraOfVertexAlgebra
            sage: LieConformalAlgebraOfVertexAlgebra(V) is P
            True
        """
        return super().__classcall__(cls, V)

    def __init__(self, V):
        """
        Initialize ``self``.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); P._V is V, P._graded
            (True, True)
            sage: P.category()
            Category of H-graded Lie conformal algebras with basis over Rational Field
        """
        from .vertex_algebra_categories import VertexAlgebras
        R = V.base_ring()
        C = LieConformalAlgebras(R)
        if V._has_odd():
            C = C.Super()
        graded = V.category().is_subcategory(VertexAlgebras(R).Graded())
        if graded:
            C = C.Graded()
        self._V = V
        self._graded = graded
        CombinatorialFreeModule.__init__(self, R, V._indices, prefix='',
                                         category=C.WithBasis())

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P
            Lie conformal algebra of Universal vertex algebra on generators ['L'] over Rational Field
        """
        return 'Lie conformal algebra of {}'.format(self._V)

    def vertex_algebra(self):
        """
        The vertex algebra this is the Lie conformal algebra of.

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P.vertex_algebra() is V
            True
        """
        return self._V

    def _repr_term(self, m):
        """
        Basis keys print as in the vertex algebra.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P._repr_term(L.derivative().no(L).leading_support())
            ':dL L:'
        """
        return self._V._repr_term(m)

    def _element_constructor_(self, x):
        """
        Convert ``x``: an element of this algebra, or anything ``V``
        converts (its coefficients are taken over as they are).

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: x = P(L.derivative()); x, P(x) is x, P(3)
            (dL, True, 3)
        """
        V = self._V
        if _parent(x) is self:
            return x
        v = V(x)
        return self._from_dict(dict(v._monomial_coefficients), coerce=False,
                               remove_zeros=True)

    def _even_odd_on_basis(self, m):
        """
        The parity of a basis key: that of the vertex algebra.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(
            ....:     lie_conformal_algebras.NeveuSchwarz(QQ), {'C': 1})
            sage: P = V.lie_conformal_algebra(); G = V.gen('G')
            sage: [P._even_odd_on_basis(m) for m in [G.leading_support(),
            ....:                                     G.no(G.derivative()).leading_support()]]
            [1, 0]
        """
        return self._V._parity_key(m)

    def degree_on_basis(self, m):
        """
        The conformal weight of the basis vector with key ``m`` (a basis key
        of the vertex algebra).

        INPUT:

        - ``m`` -- a basis key of the vertex algebra

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: x = P(L.no(L).derivative()); [P.degree_on_basis(m) for m in x.support()]
            [5, 5]
        """
        V = self._V
        return V._weight_key(m, V._weight_data())

    def some_elements(self):
        """
        Some elements, for ``TestSuite``: those of the vertex algebra.

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P.some_elements()
            [1, L, dL]
        """
        return [self(x) for x in self._V.some_elements()]

    def _an_element_(self):
        """
        The image of the vertex algebra's ``an_element()``.

        TESTS::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: V = universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ),
            ....:                                         {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P.an_element(), P.an_element().lift() == V.an_element()
            (L, True)
        """
        return self(self._V.an_element())

    def lie_conformal_algebra_generators(self):
        """
        The (strong) generators of the vertex algebra; they generate the
        vertex algebra, not this Lie conformal algebra.

        EXAMPLES::

            sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
            sage: Vir = lie_conformal_algebras.Virasoro(QQ)
            sage: V = universal_enveloping_vertex_algebra(Vir, {'C': 1/2})
            sage: P = V.lie_conformal_algebra(); L = V.gen('L')
            sage: P.lie_conformal_algebra_generators()
            (L,)
        """
        return tuple(self(x) for x in self._V.gens())


def universal_enveloping_vertex_algebra(L, central_parameters=None,
                                        degrees=True, **options):
    r"""
    The universal enveloping vertex algebra of the finitely generated Lie
    conformal algebra ``L`` of Sage (with structure coefficients), with the
    central elements ``C`` set to ``central_parameters[C]`` times the
    vacuum, as a :class:`universal_vertex_algebra.UniversalVertexAlgebra`
    on the non-central generators (named by the strings of their indices;
    Sage's divided powers ``T^(j) a`` become ``∂^j a / j!``).

    INPUT:

    - ``L`` -- a Lie conformal algebra of Sage with
      ``structure_coefficients()`` (e.g. ``lie_conformal_algebras.Virasoro(R)``)
    - ``central_parameters`` -- (default: ``None``) a dict ``{central
      element or its name: value}`` for all central elements of ``L``
      (``None``: no values, for ``L`` without central elements)
    - ``degrees`` -- (default: ``True``) ``True``: if ``L`` is H-graded,
      the degrees of its generators become the conformal weights;
      ``False``: none; or a dict as for the universal constructor
    - ``**options`` -- further keywords (``conformal_vector``,
      ``horizontal_grading``, ``implementation``, ...) go to
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`; no
      Virasoro vector by default (a choice, e.g. ``conformal_vector='L'``)

    OUTPUT:

    A :class:`universal_vertex_algebra.UniversalVertexAlgebra`.

    EXAMPLES::

        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
        sage: NS = lie_conformal_algebras.NeveuSchwarz(QQ)
        sage: V = universal_enveloping_vertex_algebra(NS, {'C': 3/2}, conformal_vector='L')
        sage: V.parities(), V.central_charge(), V.gen('G').ope(V.gen('G'))
        ({'G': 1, 'L': 0}, 3/2, [2*L, 0, 1])
        sage: K.<k> = Frac(QQ['k'])
        sage: A = universal_enveloping_vertex_algebra(lie_conformal_algebras.Affine(K, 'A1'), {'K': k})
        sage: A.variable_names()
        ('alpha[1]', 'alphacheck[1]', '-alpha[1]')
        sage: e, h, f = A.gens(); e.ope(f), h.ope(h)
        ([alphacheck[1], k], [0, 2*k])
        sage: W = universal_enveloping_vertex_algebra(lie_conformal_algebras.Weyl(QQ, 2), {'K': 1})
        sage: [x.ope(y) for x in W.gens() for y in W.gens()]
        [[], [1], [-1], []]
        sage: universal_enveloping_vertex_algebra(lie_conformal_algebras.Virasoro(QQ))
        Traceback (most recent call last):
        ...
        ValueError: give the values of the central elements ['C'] (central_parameters)

    TESTS:

    For every Lie conformal algebra of Sage the generator OPEs are the
    λ-brackets (``T^{(j)} = ∂^j/j!``, central elements replaced by their
    values), and both the vertex algebra and its Lie conformal algebra pass
    their test suites (ours: the vertex algebra axioms; Sage's: the Jacobi
    identity of the λ-brackets)::

        sage: lca = lie_conformal_algebras; K.<c> = Frac(QQ['c'])
        sage: def check(L, params):
        ....:     V = universal_enveloping_vertex_algebra(L, params); R = L.base_ring()
        ....:     central = [next(iter(z.monomial_coefficients()))[0] for z in L.central_elements()]
        ....:     image = lambda x: V.sum(cf * R(params[str(a)]) * V.one() if a in central
        ....:                             else cf / factorial(j) * V.gen(str(a)).derivative(j)
        ....:                             for (a, j), cf in x.monomial_coefficients().items())
        ....:     gens = [g for g in L.gens() if next(iter(g.monomial_coefficients()))[0] not in central]
        ....:     TestSuite(V).run(); TestSuite(V.lie_conformal_algebra()).run()
        ....:     return all(image(a).ope(image(b))
        ....:                == [image(a.bracket(b).get(n, L.zero()))
        ....:                    for n in range(max(a.bracket(b), default=-1) + 1)]
        ....:                for a in gens for b in gens)
        sage: [check(L, p) for L, p in [(lca.Virasoro(K), {'C': c}), (lca.NeveuSchwarz(K), {'C': c}),
        ....:  (lca.N2(K), {'C': c}), (lca.Affine(K, 'A1'), {'K': c}), (lca.Affine(QQ, 'B2'), {'K': 3}),
        ....:  (lca.Weyl(QQ, 4), {'K': 1}), (lca.FreeBosons(QQ, 2), {'K': 1}),
        ....:  (lca.FreeFermions(QQ, 2), {'K': 1}), (lca.BosonicGhosts(QQ, 2), {'K': 1}),
        ....:  (lca.FermionicGhosts(QQ, 2), {'K': 1}), (lca.Abelian(QQ, 2), {})]]
        [True, True, True, True, True, True, True, True, True, True, True]

    Affine ``sl_2`` has the OPEs of
    :class:`~vertex_algebras.affine_vertex_algebra.UniversalAffineVertexAlgebra`::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: Q.<k> = Frac(QQ['k'])
        sage: VA = universal_enveloping_vertex_algebra(lca.Affine(Q, 'A1'), {'K': k}, degrees=False)
        sage: A = UniversalAffineVertexAlgebra(Q, ['A', 1], level=k)
        sage: VA.variable_names() == A.variable_names()
        True
        sage: {p: [VA(x) for x in v] for p, v in A.opes().items()} == VA.opes()
        True
    """
    from .normally_ordered_expressions import NormallyOrderedExpressions
    from .universal_vertex_algebra import UniversalVertexAlgebra
    R = L.base_ring()
    central = [c.monomial_coefficients() for c in L.central_elements()]
    cnames = [next(iter(d))[0] for d in central]
    vals = {}
    for k, v in dict(central_parameters or {}).items():
        if not isinstance(k, str) and hasattr(k, 'monomial_coefficients'):
            k = next(iter(k.monomial_coefficients()))[0]
        vals[k] = R(v)
    missing = [str(c) for c in cnames if c not in vals and str(c) not in vals]
    if missing:
        raise ValueError('give the values of the central elements {} '
                         '(central_parameters)'.format(missing))

    def cval(c):
        return vals[c] if c in vals else vals[str(c)]

    gens = []
    parities = {}
    weights = {}
    for g in L.gens():
        (index, _), = g.monomial_coefficients().items()   # (a, 0): T^(0) a
        a = index[0]
        if a in cnames:
            continue
        name = str(a)
        gens.append((a, name))
        if g.is_even_odd() if hasattr(g, 'is_even_odd') else 0:
            parities[name] = 1
        if hasattr(g, 'degree'):
            try:
                weights[name] = g.degree()
            except (AttributeError, ValueError, NotImplementedError):
                pass
    names = [n for _, n in gens]
    tonames = dict(gens)
    E = NormallyOrderedExpressions(R, names, parities)

    def expression(x):
        out = E.zero()
        for (k, j), c in x.monomial_coefficients().items():
            if k in cnames:
                out += c * cval(k)
            else:
                out += c / factorial(j) * E.gen(tonames[k]).derivative(j)
        return out

    opes = {}
    for (a, b), poles in L.structure_coefficients().items():
        if a in cnames or b in cnames:
            continue
        na, nb = tonames[a], tonames[b]
        if (nb, na) in opes:            # the other orientation is derived
            continue
        top = max((n for n, _ in poles), default=-1)
        lst = [E.zero()] * (top + 1)
        for n, x in poles:
            lst[n] = expression(x)
        opes[(na, nb)] = lst
    if degrees is True:
        degrees = weights if len(weights) == len(names) else None
    elif degrees is False:
        degrees = None
    return UniversalVertexAlgebra(R, E, opes, degrees=degrees,
                                  **options)
