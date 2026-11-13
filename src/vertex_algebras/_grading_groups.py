# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Grading groups: finitely generated abelian groups in which the grades of a
graded vertex algebra live (see :meth:`vertex_algebra.VertexAlgebra.with_horizontal_grading`).

A :class:`GradingGroup` wraps a Sage group ``A`` together with a
decomposition ``A ≅ Z/n_1 × ... × Z/n_k`` into cyclic factors (``n_j = 0``
for a free factor ``Z``; the decomposition need not be the Smith form).
Internally a grade is the tuple of its coordinates in this decomposition,
torsion coordinates reduced to ``0 <= x < n_j``: adding grades along
normal ordered products is then tuple arithmetic, and only the values
handed to the user are converted back to elements of ``A``.

Supported groups ``A``:

- ``ZZ``, ``Zmod(n)`` (``Z/n`` as an additive group),
- free ``ZZ``-modules such as ``ZZ^k`` or a submodule ``(ZZ^k).span(...)``
  (coordinates with respect to the module basis),
- finitely generated ``ZZ``-modules ``M/N`` (``FGP_Module``), in particular
  ``AdditiveAbelianGroup([0, 3])`` (coordinates with respect to
  ``smith_form_gens()``),
- multiplicative ``AbelianGroup([0, 2])`` (exponents; the "sum" of grades
  is their product).

A :class:`GradingEmbedding` ``ι: A -> B`` (the grades of a module lie in
``B``) is a coordinate matrix for speed, checked and inverted by a Sage
module morphism between models of ``A`` and ``B`` on these coordinates
(``ZZ^k / (n_j)``, FGP modules; a vector space for
:class:`VectorSpaceGradingGroup`): Sage refuses maps that are not well
defined on torsion, and its ``kernel`` and ``lift`` give injectivity and
preimages.  :meth:`GradingEmbedding.morphism` is ``ι`` on the user's
groups: a Sage module morphism when both are ``ZZ``-modules (or the
target is a vector space) and Sage has the morphism, else a map.

EXAMPLES::

    sage: from vertex_algebras._grading_groups import GradingGroup
    sage: A = AdditiveAbelianGroup([0, 3])
    sage: GA = GradingGroup(A); GA
    Grading group Additive abelian group isomorphic to Z + Z/3 (coordinates in Z/3 x Z)
    sage: g, h = A.gens()
    sage: t = GA.coords(5*g + 4*h); t
    (1, 5)
    sage: GA.element(GA.add(t, t)) == 10*g + 8*h
    True
    sage: GradingGroup(ZZ^2).element((1, -1))
    (1, -1)
    sage: G2 = GradingGroup(Zmod(2)); G2.coords(3), G2.element((1,)).parent()
    ((1,), Ring of integers modulo 2)

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.rings.integer_ring import ZZ
from sage.structure.element import Element


def _modules():
    r"""
    The classes ``(FGP_Module_class, FreeModule_generic)``, imported late
    (module import order of Sage).

    TESTS::

        sage: from vertex_algebras._grading_groups import _modules
        sage: FGP, Free = _modules(); isinstance(ZZ^2, Free), isinstance(ZZ^2, FGP)
        (True, False)
    """
    from sage.modules.fg_pid.fgp_module import FGP_Module_class
    from sage.modules.free_module import FreeModule_generic
    return FGP_Module_class, FreeModule_generic


class GradingGroup:
    r"""
    A finitely generated abelian group with integer coordinates (see
    the module documentation).

    INPUT:

    - ``A`` -- a supported Sage group: ``ZZ``, ``Zmod(n)``, a free
      ``ZZ``-module, a finitely generated ``ZZ``-module (``FGP_Module``)
      or a multiplicative ``AbelianGroup``

    The attributes: ``group`` (the Sage group), ``kind`` (``'ZZ'``, ``'Zmod'``,
    ``'fgp'``, ``'free'``, ``'mult'``), ``moduli`` (``0`` = free), ``k``,
    ``free`` / ``torsion`` (indices of the coordinates), ``zero``.

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import GradingGroup
        sage: G = GradingGroup(AbelianGroup([0, 2]))
        sage: G.kind, G.moduli, G.free, G.torsion, G.zero
        ('mult', (0, 2), (0,), (1,), (0, 0))
        sage: GradingGroup(QQ)
        Traceback (most recent call last):
        ...
        TypeError: Rational Field is not a supported grading group (use ZZ, Zmod(n), ZZ^k, AdditiveAbelianGroup or AbelianGroup)
    """

    def __init__(self, A):
        """
        TESTS::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: G = GradingGroup(AdditiveAbelianGroup([0, 3]))
            sage: G.kind, G.moduli, G.k
            ('fgp', (3, 0), 2)
        """
        from sage.groups.abelian_gps.abelian_group import AbelianGroup_class
        from sage.rings.finite_rings.integer_mod_ring import IntegerModRing_generic
        FGP_Module_class, FreeModule_generic = _modules()
        self.group = A
        if A is ZZ:
            self.kind, moduli = 'ZZ', (0,)
        elif isinstance(A, IntegerModRing_generic):
            self.kind, moduli = 'Zmod', (int(A.order()),)
        elif isinstance(A, FGP_Module_class):
            if A.base_ring() is not ZZ:
                raise ValueError('a grading group must be a ZZ-module')
            self.kind, moduli = 'fgp', tuple(int(n) for n in A.invariants())
            self._gens = A.smith_form_gens()
        elif isinstance(A, FreeModule_generic):
            if A.base_ring() is not ZZ:
                raise ValueError('a grading group must be a ZZ-module '
                                 '(e.g. ZZ^k), not a {}-module'.format(A.base_ring()))
            self.kind, moduli = 'free', (0,) * A.rank()
        elif isinstance(A, AbelianGroup_class):
            self.kind, moduli = 'mult', tuple(int(n) for n in A.gens_orders())
        else:
            raise TypeError('{} is not a supported grading group (use ZZ, '
                            'Zmod(n), ZZ^k, AdditiveAbelianGroup or '
                            'AbelianGroup)'.format(A))
        self.moduli = moduli
        self.k = len(moduli)
        self.free = tuple(j for j, n in enumerate(moduli) if n == 0)
        self.torsion = tuple(j for j, n in enumerate(moduli) if n != 0)
        self.zero = (0,) * self.k

    def __repr__(self):
        r"""
        The group and its cyclic coordinates.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: GradingGroup(Zmod(3))
            Grading group Ring of integers modulo 3 (coordinates in Z/3)
            sage: GradingGroup(ZZ^0)
            Grading group Ambient free module of rank 0 over the principal ideal domain Integer Ring (coordinates in 0)
        """
        parts = ['Z' if n == 0 else 'Z/{}'.format(n) for n in self.moduli]
        return 'Grading group {} (coordinates in {})'.format(
            self.group, ' x '.join(parts) if parts else '0')

    def _model(self):
        r"""
        The group as a module on its coordinates: ``ZZ^k / (n_j e_j)`` (an
        FGP module; the free module ``ZZ^k`` modulo nothing for a free
        group).  Cached.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: GradingGroup(AbelianGroup([0, 2]))._model()
            Finitely generated module V/W over Integer Ring with invariants (2, 0)
        """
        M = self.__dict__.get('_model_cache')
        if M is None:
            from sage.modules.free_module import FreeModule
            V = FreeModule(ZZ, self.k)
            M = V / V.span([n * V.gen(j) for j, n in enumerate(self.moduli) if n])
            self._model_cache = M
        return M

    def _model_element(self, t):
        r"""
        The element of :meth:`_model` with coordinate tuple ``t``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: G = GradingGroup(AbelianGroup([0, 2])); x = G._model_element((3, 5))
            sage: G._model_coords(x)
            (3, 1)
        """
        return self._model()(list(t))

    def _model_coords(self, x):
        r"""
        The reduced coordinate tuple of an element ``x`` of :meth:`_model`.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: G = GradingGroup(Zmod(3)); G._model_coords(G._model_element((4,)))
            (1,)
        """
        return self.reduce(tuple(ZZ(c) for c in x.lift()))

    def __getstate__(self):
        r"""
        The attributes without the cached model (rebuilt on demand).

        TESTS::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: G = GradingGroup(Zmod(3)); _ = G._model()
            sage: '_model_cache' in G.__getstate__(), loads(dumps(G)).moduli
            (False, (3,))
        """
        d = dict(self.__dict__)
        d.pop('_model_cache', None)
        return d

    def __eq__(self, other):
        r"""
        Equal iff the wrapped groups are the same object.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: GradingGroup(Zmod(3)) == GradingGroup(Zmod(3)), GradingGroup(ZZ) == ZZ
            (True, False)
        """
        return isinstance(other, GradingGroup) and other.group is self.group

    def __hash__(self):
        r"""
        Hash of the identity of the wrapped group (consistent with ``==``).

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: hash(GradingGroup(Zmod(3))) == hash(GradingGroup(Zmod(3)))
            True
        """
        return hash(id(self.group))

    # -- coordinates ---------------------------------------------------- #

    def reduce(self, t):
        r"""
        Reduce the torsion coordinates of the integer tuple ``t``.

        INPUT:

        - ``t`` -- a tuple of integers of length ``k``

        OUTPUT:

        A tuple of Python integers, the torsion coordinates reduced to
        ``0 <= x < n_j``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: GradingGroup(AbelianGroup([0, 2])).reduce((5, -3))
            (5, 1)
        """
        return tuple(int(x) % n if n else int(x) for x, n in zip(t, self.moduli))

    def coords(self, g):
        r"""
        The coordinate tuple of ``g`` (an element of the group, or
        anything the group converts, e.g. an integer or a tuple).

        INPUT:

        - ``g`` -- an element of the group, or anything the group converts

        OUTPUT:

        The reduced coordinate tuple of ``g``.  A ``ValueError`` if the
        group does not convert ``g``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: A = AbelianGroup([0, 2]); a, b = A.gens()
            sage: GradingGroup(A).coords(a^2*b^3)
            (2, 1)
            sage: GradingGroup((ZZ^2).span([(2, 0), (0, 1)])).coords((4, 3))
            (2, 3)
            sage: GradingGroup(ZZ).coords(1/2)
            Traceback (most recent call last):
            ...
            ValueError: 1/2 is not an element of the grading group Integer Ring
        """
        A = self.group
        try:
            g = A(g)
        except (TypeError, ValueError, ArithmeticError):
            raise ValueError('{} is not an element of the grading group '
                             '{}'.format(g, A))
        if self.kind == 'ZZ':
            t = (g,)
        elif self.kind == 'Zmod':
            t = (g.lift(),)
        elif self.kind == 'fgp':
            t = tuple(g.vector())
        elif self.kind == 'free':
            t = tuple(A.coordinate_vector(g))
        else:
            t = tuple(g.exponents())
        return self.reduce(t)

    def element(self, t):
        r"""
        The group element with coordinate tuple ``t``.

        INPUT:

        - ``t`` -- a coordinate tuple

        OUTPUT:

        An element of the wrapped group (an immutable vector for free
        ``ZZ``-modules).

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: A = AbelianGroup([0, 2]); a, b = A.gens()
            sage: GradingGroup(A).element((2, 1)) == a^2*b
            True
            sage: v = GradingGroup(ZZ^2).element((1, 2)); v, v.is_immutable()
            ((1, 2), True)
        """
        A = self.group
        if self.kind == 'ZZ':
            return ZZ(t[0])
        if self.kind == 'Zmod':
            return A(t[0])
        if self.kind == 'fgp':
            return A.linear_combination_of_smith_form_gens(list(t))
        if self.kind == 'free':
            v = A.linear_combination_of_basis(list(t))
            v.set_immutable()
            return v
        out = A.one()
        for x, g in zip(t, A.gens()):
            out *= g ** x
        return out

    def add(self, a, b):
        r"""
        The sum of two reduced coordinate tuples, reduced (the group law; for a
        multiplicative group, the product).

        INPUT:

        - ``a``, ``b`` -- reduced coordinate tuples

        OUTPUT:

        The reduced coordinate tuple of the sum.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: GradingGroup(AbelianGroup([0, 2])).add((1, 1), (2, 1))
            (3, 0)
        """
        return tuple((x + y) % n if n else x + y
                     for x, y, n in zip(a, b, self.moduli))

    def is_zero(self, t):
        r"""
        True if the integer tuple ``t`` (not necessarily reduced) is the
        coordinate tuple of 0.

        INPUT:

        - ``t`` -- a tuple of integers, not necessarily reduced

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup
            sage: G = GradingGroup(AbelianGroup([0, 2])); G.is_zero((0, 4)), G.is_zero((1, 0))
            (True, False)
        """
        return all((x % n == 0) if n else x == 0 for x, n in zip(t, self.moduli))


def _int_like(v):
    r"""
    True if ``v`` is an integer (a Sage or Python integer, or a rational
    equal to one), not a tuple or list.

    TESTS::

        sage: from vertex_algebras._grading_groups import _int_like
        sage: _int_like(3), _int_like(QQ(4)), _int_like(1/2), _int_like((1, 2)), _int_like('a')
        (True, True, False, False, False)
    """
    try:
        return ZZ(v) == v and not isinstance(v, (tuple, list))
    except (TypeError, ValueError):
        return False


def infer_grading_group(values, group=None):
    r"""
    The :class:`GradingGroup` for grades ``values`` when the user gave the
    group ``group`` (``None``: infer it).  Inference: integers give ``ZZ``,
    elements of one supported group give that group, tuples of length ``k``
    give ``ZZ^k``.

    INPUT:

    - ``values`` -- an iterable of grades (integers, group elements or
      tuples of integers)
    - ``group`` -- (default: ``None``) a Sage group or a
      :class:`GradingGroup`, or ``None`` to infer it from ``values``

    OUTPUT:

    A :class:`GradingGroup` (``group`` itself if it is one).  A
    ``ValueError`` if the group cannot be inferred.

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import infer_grading_group
        sage: infer_grading_group([1, -1]).group
        Integer Ring
        sage: infer_grading_group([(1, 0), (0, 1)]).group
        Ambient free module of rank 2 over the principal ideal domain Integer Ring
        sage: A = AdditiveAbelianGroup([2]); infer_grading_group([A.gen(0)]).group is A
        True
        sage: infer_grading_group([Zmod(2)(1), 0]).group
        Ring of integers modulo 2
        sage: infer_grading_group([1, (1, 2)])
        Traceback (most recent call last):
        ...
        ValueError: cannot infer the grading group from the grades [1, (1, 2)]; pass group=...
    """
    from sage.modules.free_module import FreeModule
    if group is not None:
        return group if isinstance(group, GradingGroup) else GradingGroup(group)
    values = list(values)
    parents = {v.parent() for v in values
               if isinstance(v, Element) and v.parent() is not ZZ}
    if len(parents) == 1:
        P = parents.pop()
        try:
            G = GradingGroup(P)
        except (TypeError, ValueError):
            G = None
        if G is not None and all(
                (isinstance(v, Element) and v.parent() is P) or _int_like(v)
                for v in values):
            return G
    if all(_int_like(v) for v in values):
        return GradingGroup(ZZ)
    try:
        lens = {len(tuple(v)) for v in values}
    except TypeError:
        lens = set()
    if len(lens) == 1 and all(_int_like(x) for v in values for x in v):
        return GradingGroup(FreeModule(ZZ, lens.pop()))
    raise ValueError('cannot infer the grading group from the grades {}; '
                     'pass group=...'.format(values))


class ProductGradingGroup(GradingGroup):
    r"""
    The direct product ``A_1 × ... × A_r`` of grading groups (the grading
    group of a tensor product), realised as Sage's ``cartesian_product``;
    the coordinates are the concatenated coordinates of the factors.

    INPUT:

    - ``factors`` -- a list of :class:`GradingGroup` objects

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import GradingGroup, ProductGradingGroup
        sage: P = ProductGradingGroup([GradingGroup(ZZ), GradingGroup(Zmod(2))])
        sage: P.moduli, P.coords((3, 1)), P.element((3, 1))
        ((0, 2), (3, 1), (3, 1))
        sage: P.split(P.coords((3, 1)))
        [(3,), (1,)]
    """

    def __init__(self, factors):
        """
        TESTS::

            sage: from vertex_algebras._grading_groups import GradingGroup, ProductGradingGroup
            sage: P = ProductGradingGroup([GradingGroup(ZZ^2), GradingGroup(Zmod(2))])
            sage: P.kind, P.moduli, len(P.factors)
            ('product', (0, 0, 2), 2)
        """
        from sage.categories.cartesian_product import cartesian_product
        self.factors = tuple(factors)
        self.group = cartesian_product([F.group for F in self.factors])
        self.kind = 'product'
        self.moduli = sum((F.moduli for F in self.factors), ())
        self.k = len(self.moduli)
        self.free = tuple(j for j, n in enumerate(self.moduli) if n == 0)
        self.torsion = tuple(j for j, n in enumerate(self.moduli) if n != 0)
        self.zero = (0,) * self.k
        self.offsets = [0]
        for F in self.factors:
            self.offsets.append(self.offsets[-1] + F.k)

    def split(self, t):
        r"""
        The factor coordinate tuples of the coordinate tuple ``t``.

        INPUT:

        - ``t`` -- a coordinate tuple of the product

        OUTPUT:

        A list of tuples, one per factor.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup, ProductGradingGroup
            sage: P = ProductGradingGroup([GradingGroup(ZZ^2), GradingGroup(Zmod(2))])
            sage: P.split((1, 2, 1))
            [(1, 2), (1,)]
        """
        o = self.offsets
        return [tuple(t[o[i]:o[i + 1]]) for i in range(len(self.factors))]

    def coords(self, g):
        r"""
        The concatenated coordinates of the factors of ``g`` (an element of the
        cartesian product, or a tuple of factor elements).

        INPUT:

        - ``g`` -- an element of the cartesian product, or a tuple of
          factor elements

        OUTPUT:

        A tuple of coordinates.  A ``ValueError`` if ``g`` is not an
        element of the product.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup, ProductGradingGroup
            sage: P = ProductGradingGroup([GradingGroup(ZZ^2), GradingGroup(Zmod(2))])
            sage: P.coords(((1, 2), 3))
            (1, 2, 1)
        """
        try:
            g = self.group(g)
        except (TypeError, ValueError, ArithmeticError):
            raise ValueError('{} is not an element of the grading group '
                             '{}'.format(g, self.group))
        return sum((F.coords(x) for F, x in zip(self.factors,
                                                 g.cartesian_factors())), ())

    def element(self, t):
        r"""
        The element of the cartesian product with coordinate tuple ``t``.

        INPUT:

        - ``t`` -- a coordinate tuple of the product

        OUTPUT:

        An element of Sage's ``cartesian_product`` of the factor groups.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import GradingGroup, ProductGradingGroup
            sage: P = ProductGradingGroup([GradingGroup(ZZ^2), GradingGroup(Zmod(2))])
            sage: x = P.element((1, 2, 1)); x, P.coords(x)
            (((1, 2), 1), (1, 2, 1))
        """
        return self.group(tuple(F.element(x)
                                for F, x in zip(self.factors, self.split(t))))


class VectorSpaceGradingGroup(GradingGroup):
    r"""
    A free module ``R^n`` over a field ``R`` of characteristic 0 as a
    grading group of *modules*: the grades of a module may lie in a group
    containing the grading group of the vertex algebra, e.g. the weights
    ``P ⊗ R`` of an affine vertex algebra (graded by the root lattice
    ``Q ⊂ P``), whose highest weights may have rational or symbolic
    coordinates.  Coordinates are tuples of elements of ``R`` (no torsion).

    INPUT:

    - ``A`` -- a free module over a field of characteristic 0

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
        sage: K.<k> = Frac(QQ['k'])
        sage: B = VectorSpaceGradingGroup(K^2); B
        Grading group Vector space of dimension 2 over Fraction Field of Univariate Polynomial Ring in k over Rational Field (coordinates in K^2)
        sage: t = B.coords((k, -1/2)); t, B.add(t, (1, 1)), B.element(t)
        ((k, -1/2), (k + 1, 1/2), (k, -1/2))
    """

    def __init__(self, A):
        """
        TESTS::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: B = VectorSpaceGradingGroup(QQ^2); B.kind, B.k, B.torsion
            ('vector', 2, ())
            sage: VectorSpaceGradingGroup(ZZ^2)
            Traceback (most recent call last):
            ...
            TypeError: Ambient free module of rank 2 over the principal ideal domain Integer Ring is not a free module over a field of characteristic 0
        """
        from sage.modules.free_module import FreeModule_generic
        R = getattr(A, 'base_ring', lambda: None)()
        if (not isinstance(A, FreeModule_generic) or R is None or not R.is_field()
                or R.characteristic() != 0):
            raise TypeError('{} is not a free module over a field of characteristic 0'
                            .format(A))
        self.group = A
        self.kind = 'vector'
        self.R = R
        self.moduli = (0,) * A.rank()
        self.k = len(self.moduli)
        self.free = tuple(range(self.k))
        self.torsion = ()
        self.zero = (R.zero(),) * self.k

    def __repr__(self):
        r"""
        The module and the coordinate space.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: VectorSpaceGradingGroup(QQ^2)
            Grading group Vector space of dimension 2 over Rational Field (coordinates in K^2)
        """
        return 'Grading group {} (coordinates in K^{})'.format(self.group, self.k)

    def reduce(self, t):
        r"""
        The tuple ``t`` with entries converted into the base field (no torsion
        to reduce).

        INPUT:

        - ``t`` -- a tuple of elements convertible into the base field

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: t = VectorSpaceGradingGroup(QQ^2).reduce((1, 2)); t, t[0].parent()
            ((1, 2), Rational Field)
        """
        return tuple(self.R(x) for x in t)

    def coords(self, g):
        r"""
        The coordinates of ``g`` with respect to the basis of the module.

        INPUT:

        - ``g`` -- an element of the module, or anything it converts

        OUTPUT:

        A tuple of elements of the base field.  A ``ValueError`` if the
        module does not convert ``g``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: B = VectorSpaceGradingGroup((QQ^2).span_of_basis([(2, 0), (0, 1)])); B.coords((1, 3))
            (1/2, 3)
        """
        A = self.group
        try:
            g = A(g)
        except (TypeError, ValueError, ArithmeticError):
            raise ValueError('{} is not an element of the grading group '
                             '{}'.format(g, A))
        return self.reduce(A.coordinate_vector(g))

    def element(self, t):
        r"""
        The (immutable) vector with coordinates ``t``.

        INPUT:

        - ``t`` -- a coordinate tuple

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: v = VectorSpaceGradingGroup(QQ^2).element((1, 2)); v, v.is_immutable()
            ((1, 2), True)
        """
        v = self.group.linear_combination_of_basis(list(t))
        v.set_immutable()
        return v

    def add(self, a, b):
        r"""
        The sum of coordinate tuples.

        INPUT:

        - ``a``, ``b`` -- coordinate tuples

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: VectorSpaceGradingGroup(QQ^2).add((1, 2), (3, 1/2))
            (4, 5/2)
        """
        return tuple(x + y for x, y in zip(a, b))

    def is_zero(self, t):
        r"""
        True if every coordinate is zero.

        INPUT:

        - ``t`` -- a coordinate tuple

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: B = VectorSpaceGradingGroup(QQ^2); B.is_zero((0, 0)), B.is_zero((0, 1))
            (True, False)
        """
        return all(x == 0 for x in t)

    def _model(self):
        r"""
        The coordinate space ``R^k`` (the vector space on the coordinates).

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: VectorSpaceGradingGroup((QQ^2).span_of_basis([(2, 0), (0, 1)]))._model()
            Vector space of dimension 2 over Rational Field
        """
        from sage.modules.free_module import FreeModule
        return FreeModule(self.R, self.k)

    def _model_element(self, t):
        r"""
        The vector with coordinates ``t``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import VectorSpaceGradingGroup
            sage: VectorSpaceGradingGroup(QQ^2)._model_element((1, 1/2))
            (1, 1/2)
        """
        return self._model()(list(t))


def module_grading_group(B):
    r"""
    The grading group object of a Sage group ``B`` for module gradings:
    :class:`GradingGroup` for finitely generated abelian groups,
    :class:`VectorSpaceGradingGroup` for free modules over fields of
    characteristic 0.

    INPUT:

    - ``B`` -- a Sage group (as for :class:`GradingGroup`), a free module
      over a field of characteristic 0, or a :class:`GradingGroup`
      (returned as it is)

    OUTPUT:

    A :class:`GradingGroup` or a :class:`VectorSpaceGradingGroup`.

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import GradingGroup, module_grading_group
        sage: module_grading_group(QQ^2)
        Grading group Vector space of dimension 2 over Rational Field (coordinates in K^2)
        sage: module_grading_group(ZZ^2)
        Grading group Ambient free module of rank 2 over the principal ideal domain Integer Ring (coordinates in Z x Z)
        sage: G = GradingGroup(ZZ); module_grading_group(G) is G
        True
    """
    if isinstance(B, GradingGroup):
        return B
    from sage.modules.free_module import FreeModule_generic
    if isinstance(B, FreeModule_generic) and B.base_ring() is not ZZ:
        return VectorSpaceGradingGroup(B)
    return GradingGroup(B)


class GradingEmbedding:
    r"""
    A homomorphism ``ι: A -> B`` of grading groups on coordinates, ``ι(a) =
    M a`` (reduced in ``B``) for a matrix ``M`` (rows: coordinates of ``B``,
    columns: those of ``A``): the grades of a module over a vertex algebra
    graded by ``A`` lie in ``B``, and ``grade(a_(n) m) = ι(grade a) +
    grade m``.  Checked: ``ι`` is well defined on the torsion of ``A`` and
    injective on its free part.  Both checks and :meth:`preimage` are done
    by a Sage module morphism of the coordinate models of ``A`` and ``B``
    (``A._model() -> B._model()``, attribute ``hom``); evaluation uses the
    matrix (faster).  :meth:`morphism` is ``ι`` on the groups themselves.

    INPUT:

    - ``A`` -- a :class:`GradingGroup`, the source
    - ``B`` -- a :class:`GradingGroup` or :class:`VectorSpaceGradingGroup`,
      the target
    - ``M`` -- a matrix, or anything ``matrix`` converts, with one row per
      coordinate of ``B`` and one column per coordinate of ``A`` (entries
      in ``B``'s base field for a :class:`VectorSpaceGradingGroup`, else
      integers)

    EXAMPLES::

        sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
        ....:                                             VectorSpaceGradingGroup)
        sage: A = GradingGroup(ZZ^1); B = VectorSpaceGradingGroup(QQ^1)
        sage: iota = GradingEmbedding(A, B, matrix([[2]])); iota((3,))
        (6,)
        sage: GradingEmbedding(GradingGroup(Zmod(2)), GradingGroup(ZZ), matrix([[1]]))
        Traceback (most recent call last):
        ...
        ValueError: the map of grading groups is not well defined on the torsion of Ring of integers modulo 2
    """

    def __init__(self, A, B, M):
        """
        TESTS::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: iota = GradingEmbedding(GradingGroup(ZZ^1), VectorSpaceGradingGroup(QQ^1), [[2]])
            sage: iota.M, iota.M.is_immutable()
            ([2], True)
        """
        from sage.matrix.constructor import matrix
        from sage.modules.free_module import FreeModule
        R = B.R if B.kind == 'vector' else ZZ
        M = matrix(R, B.k, A.k, M)
        M.set_immutable()
        self.A, self.B, self.M = A, B, M
        if B.kind == 'vector':
            # a free ZZ-module into a vector space (torsion cannot embed)
            if A.torsion:
                raise ValueError('the map of grading groups {} -> {} is not injective '
                                 '(torsion)'.format(A.group, B.group))
            VB = B._model()
            hom = FreeModule(ZZ, A.k).hom([VB(M.column(j)) for j in range(A.k)], VB)
            injective = hom.kernel().rank() == 0
        else:
            MA, MB = A._model(), B._model()
            try:
                hom = MA.hom([MB(M * g.lift()) for g in MA.gens()], MB)
            except ValueError:
                raise ValueError('the map of grading groups is not well defined on the '
                                 'torsion of {}'.format(A.group))
            injective = 0 not in hom.kernel().invariants()      # finite kernel
        if not injective:
            raise ValueError('the map of grading groups {} -> {} is not injective'
                             .format(A.group, B.group))
        self.hom = hom

    def __call__(self, t):
        r"""
        ``ι(t) = M t``, reduced in ``B`` (``t`` a coordinate tuple of ``A``).

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: GradingEmbedding(GradingGroup(ZZ), VectorSpaceGradingGroup(QQ^1), matrix([[1/2]]))((3,))
            (3/2,)
            sage: GradingEmbedding(GradingGroup(Zmod(2)), GradingGroup(Zmod(4)), matrix([[2]]))((1,))
            (2,)
        """
        M = self.M
        return self.B.reduce(tuple(sum(M[i, j] * t[j] for j in range(len(t)))
                                   for i in range(M.nrows())))

    def preimage(self, t):
        r"""
        The coordinates ``a`` (in ``A``) with ``ι(a) = t``, or ``None`` if
        ``t`` is not in the image.

        INPUT:

        - ``t`` -- a coordinate tuple of ``B``

        OUTPUT:

        A reduced coordinate tuple of ``A``, or ``None``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: iota = GradingEmbedding(GradingGroup(ZZ^2), VectorSpaceGradingGroup(QQ^2),
            ....:                         matrix([[2, -1], [-1, 2]]))
            sage: iota.preimage((3, 0)), iota.preimage((1, 0))
            ((2, 1), None)
            sage: iota2 = GradingEmbedding(GradingGroup(Zmod(2)), GradingGroup(Zmod(4)), matrix([[2]]))
            sage: iota2.preimage((2,)), iota2.preimage((1,))
            ((1,), None)
        """
        A, B = self.A, self.B
        t = B.reduce(t)
        try:
            x = self.hom.lift(B._model_element(t))
        except ValueError:                  # not in the image
            return None
        a = (A.reduce(tuple(ZZ(c) for c in x)) if B.kind == 'vector'
             else A._model_coords(x))
        return a if self(a) == t else None

    def morphism(self):
        r"""
        ``ι`` as a map from ``A`` to ``B`` (the Sage groups).  A Sage module
        morphism (with ``kernel()``, ``image()``, ``lift()``) when ``A`` is
        an FGP module and ``B`` an FGP or free ``ZZ``-module, or ``A`` is a
        free ``ZZ``-module and ``B`` a free ``ZZ``-module or a vector space;
        otherwise (``ZZ``, ``Zmod(n)``, multiplicative ``AbelianGroup``,
        cartesian products, and free into FGP modules, which Sage does not
        provide) a map of sets.  Cached.

        OUTPUT:

        A morphism with domain ``A.group`` and codomain ``B.group``.

        EXAMPLES:

        Free ``ZZ``-modules: a Sage module morphism::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: iota = GradingEmbedding(GradingGroup(ZZ^2), VectorSpaceGradingGroup(QQ^2),
            ....:                         matrix([[2, -1], [-1, 2]]))
            sage: f = iota.morphism(); f((1, 1)), f.kernel().rank(), f.lift(vector([3, 0]))
            ((1, 1), 0, (2, 1))

        ``ZZ``: a map::

            sage: iota = GradingEmbedding(GradingGroup(ZZ), VectorSpaceGradingGroup(QQ^1), [[1/2]])
            sage: g = iota.morphism(); g(3), g.domain(), g is iota.morphism()
            ((3/2), Integer Ring, True)
        """
        f = self.__dict__.get('_morphism')
        if f is not None:
            return f
        A, B = self.A, self.B

        def image(g):
            return B.element(self(A.coords(g)))

        if A.kind == 'fgp' and B.kind in ('fgp', 'free'):
            f = A.group.hom([image(g) for g in A.group.gens()], B.group)
        elif A.kind == 'free' and B.kind in ('free', 'vector'):
            f = A.group.hom([image(g) for g in A.group.basis()], B.group)
        else:
            from sage.categories.homset import Hom
            from sage.categories.morphism import SetMorphism
            from sage.categories.sets_cat import Sets
            f = SetMorphism(Hom(A.group, B.group, Sets()), image)
        self._morphism = f
        return f

    def __reduce__(self):
        r"""
        Pickled by ``(A, B, M)`` (the morphisms are rebuilt and the checks
        rerun).

        TESTS::

            sage: from vertex_algebras._grading_groups import GradingGroup, GradingEmbedding
            sage: iota = GradingEmbedding(GradingGroup(Zmod(2)), GradingGroup(Zmod(4)), [[2]])
            sage: loads(dumps(iota)) == iota
            True
        """
        return GradingEmbedding, (self.A, self.B, self.M)

    def __eq__(self, other):
        r"""
        Equal iff source, target and matrix agree.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: A, B = GradingGroup(ZZ), VectorSpaceGradingGroup(QQ^1)
            sage: GradingEmbedding(A, B, matrix([[1]])) == GradingEmbedding(A, B, matrix([[1]]))
            True
            sage: GradingEmbedding(A, B, matrix([[1]])) == GradingEmbedding(A, B, matrix([[2]]))
            False
        """
        return (isinstance(other, GradingEmbedding) and other.A == self.A
                and other.B == self.B and other.M == self.M)

    def __hash__(self):
        r"""
        Hash of ``(A, B, M)``.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: A, B = GradingGroup(ZZ), VectorSpaceGradingGroup(QQ^1)
            sage: hash(GradingEmbedding(A, B, matrix([[1]]))) == hash(GradingEmbedding(A, B, matrix([[1]])))
            True
        """
        return hash((self.A, self.B, self.M))

    def __repr__(self):
        r"""
        The source and target groups.

        EXAMPLES::

            sage: from vertex_algebras._grading_groups import (GradingGroup, GradingEmbedding,
            ....:                                             VectorSpaceGradingGroup)
            sage: GradingEmbedding(GradingGroup(ZZ), VectorSpaceGradingGroup(QQ^1), matrix([[1]]))
            Embedding of Integer Ring into Vector space of dimension 1 over Rational Field
        """
        return 'Embedding of {} into {}'.format(self.A.group, self.B.group)
