# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Ghost systems and free superfermions.

Constructors of named free field vertex (super)algebras.  Each returns a
:class:`universal_vertex_algebra.UniversalVertexAlgebra` (so equal data
give the same algebra, as for every universal vertex algebra), with the
λ-brackets of Sage's Lie conformal algebras of the same names
(``lie_conformal_algebras.BosonicGhosts``, ``FermionicGhosts``,
``Weyl``, ``FreeFermions``):

- :func:`BosonicGhosts` (``vertex_algebras.BosonicGhosts``): even
  ``β_i``, ``γ_i`` with ``β_i(z) γ_j(w) ~ δ_ij/(z-w)``;
- :func:`FermionicGhosts` (``vertex_algebras.FermionicGhosts``): odd ``b_i``,
  ``c_i`` with ``b_i(z) c_j(w) ~ δ_ij/(z-w)``;
- :func:`FreeSuperFermions`: the fields ``Φ_i`` of a basis of a super
  vector space with a non-degenerate even form, skew-symmetric on the
  even part and symmetric on the odd part, ``Φ_i(z) Φ_j(w) ~
  M_ij/(z-w)``: symplectic bosons and free fermions (the neutral free
  superfermions of [KRW2003]_); given by the Gram matrix ``M`` (as for
  lattice vertex algebras) or by a basis and the form.

No conformal vector is chosen by default.  The common ones by name
(``conformal_vector=``):

- ``'symmetric'``: every field of weight ``1/2`` (for ``βγ`` and ``bc``:
  ``λ = 1/2``);
- ``'b1c0'``, ``'b0c1'`` (``βγ`` and ``bc``): ``β``, ``b`` of weight 1 and
  ``γ``, ``c`` of weight 0, resp. the other way round;
- ``'degrees'`` with ``degrees=``: the free field conformal
  vector with exactly these weights (the weights of paired fields add up
  to 1).

Also ``beta_weight=λ`` / ``b_weight=λ`` (any ``λ``, or a list, one per
pair; ``λ = 2`` for ``bc``: the ghosts of the bosonic string, ``c =
-26``).  Any other conformal vector: an element of the algebra without
conformal data (as for lattice vertex algebras), or an expression.

EXAMPLES::

    sage: from vertex_algebras.ghost_systems import BosonicGhosts, FermionicGhosts, FreeSuperFermions
    sage: V = BosonicGhosts(QQ, conformal_vector='symmetric'); V
    Universal vertex algebra on generators ['beta', 'gamma'] over Rational Field
    sage: V.central_charge(), V.degrees()
    (-1, {'beta': 1/2, 'gamma': 1/2})
    sage: beta, gamma = V.gens()
    sage: beta.ope(gamma), gamma.ope(beta)
    ([1], [-1])
    sage: [FermionicGhosts(QQ, conformal_vector=c).central_charge() for c in ('symmetric', 'b1c0', 'b0c1')]
    [1, -2, -2]
    sage: [FermionicGhosts(QQ, b_weight=l).central_charge() for l in (3/2, 2)]
    [-11, -26]
    sage: F = FreeSuperFermions(QQ, matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    ....:                       conformal_vector='symmetric')
    sage: F.parities(), F.central_charge()
    ({'Phi0': 0, 'Phi1': 0, 'Phi2': 1}, -1/2)

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.matrix.constructor import matrix
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .normally_ordered_expressions import NormallyOrderedExpressions
from .universal_vertex_algebra import UniversalVertexAlgebra

#: the named conformal vectors of a pair: the weight of β, resp. b, as
#: (numerator, denominator) (no Sage arithmetic at import time)
_PAIR_WEIGHTS = {'symmetric': (1, 2), 'b1c0': (1, 1), 'b0c1': (0, 1)}


def _pair_names(first, second, rank, names):
    """The names of the ``rank`` pairs (Sage's: ``beta, gamma`` for rank
    1, ``beta0, ..., gamma0, ...`` otherwise; ``names``: the first
    generators of all pairs, then the second ones).

    TESTS::

        sage: from vertex_algebras.ghost_systems import _pair_names
        sage: _pair_names('b', 'c', 1, None), _pair_names('beta', 'gamma', 2, None)
        ((['b'], ['c']), (['beta0', 'beta1'], ['gamma0', 'gamma1']))
        sage: _pair_names('b', 'c', 2, ['x', 'y', 'u', 'v'])
        (['x', 'y'], ['u', 'v'])
        sage: _pair_names('b', 'c', 2, ['x'])
        Traceback (most recent call last):
        ...
        ValueError: 4 names expected (the first generator of each pair, then the second ones)
    """
    if names is not None:
        names = [str(s) for s in names]
        if len(names) != 2 * rank:
            raise ValueError('{} names expected (the first generator of each pair, '
                             'then the second ones)'.format(2 * rank))
        return names[:rank], names[rank:]
    if rank == 1:
        return [first], [second]
    return (['{}{}'.format(first, i) for i in range(rank)],
            ['{}{}'.format(second, i) for i in range(rank)])


def _pair_virasoro(x, y, lam, odd):
    r"""
    The conformal vector of one pair ``x(z) y(w) ~ 1/(z-w)`` with ``x`` of
    weight ``λ`` and ``y`` of weight ``1 - λ``::

        βγ:  T = (λ - 1) :∂β γ: + λ :β ∂γ:
        bc:  T = (1 - λ) :∂b c: - λ :b ∂c:

    (central charges ``2(6λ^2 - 6λ + 1)`` and ``-2(6λ^2 - 6λ + 1)``).
    ``x``, ``y`` are elements of a :class:`NormallyOrderedExpressions` (or
    of a vertex algebra), ``odd`` says whether the pair is ``bc``.

    TESTS::

        sage: from vertex_algebras.ghost_systems import _pair_virasoro
        sage: from vertex_algebras import NormallyOrderedExpressions
        sage: E.<b, c> = NormallyOrderedExpressions(QQ, ['b', 'c'], {'b': 1, 'c': 1})
        sage: _pair_virasoro(b, c, 2, True)
        -2*:b dc: - :db c:
    """
    s = 1 if odd else -1
    return s * (1 - lam) * (x.derivative() * y) - s * lam * (x * y.derivative())


def _ghost_system(R, rank, odd, weight, names, conformal_vector,
                  degrees, horizontal_grading, horizontal_grading_group, implementation):
    r"""
    The common body of :func:`BosonicGhosts` (``odd=False``) and
    :func:`FermionicGhosts` (``odd=True``): ``rank`` pairs ``x_i(z) y_i(w) ~
    1/(z-w)``; ``weight`` = the weight ``λ`` of the first field of each pair
    (a value or a list), or a named ``conformal_vector`` (``'symmetric'``,
    ``'b1c0'``, ``'b0c1'``, ``'degrees'`` with ``degrees``), or any other
    conformal vector; ``horizontal_grading='charge'`` = the ghost number.

    TESTS::

        sage: from vertex_algebras.ghost_systems import _ghost_system
        sage: V = _ghost_system(QQ, 1, True, 2, None, None, None, None, None, None)
        sage: V.gens(), V.central_charge()
        ((b, c), -26)
        sage: V = _ghost_system(QQ, 1, False, None, None, 'symmetric', None, 'charge',
        ....:                   None, None)
        sage: V.central_charge(), V.degrees(), V.horizontal_grade(V.gen('gamma'))
        (-1, {'beta': 1/2, 'gamma': 1/2}, 1)
        sage: _ghost_system(QQ, 0, True, None, None, None, None, None, None, None)
        Traceback (most recent call last):
        ...
        ValueError: the rank must be positive
    """
    rank = ZZ(rank)
    if rank < 1:
        raise ValueError('the rank must be positive')
    first, second = ('b', 'c') if odd else ('beta', 'gamma')
    X, Y = _pair_names(first, second, rank, names)
    E = NormallyOrderedExpressions(R, X + Y, dict.fromkeys(X + Y, 1) if odd else {})
    opes = {(x, y): [1] for x, y in zip(X, Y)}
    if isinstance(conformal_vector, str):
        if conformal_vector == 'degrees':
            if degrees is None:
                raise ValueError("conformal_vector='degrees' needs degrees=")
            w = dict(degrees)
            if set(w) != set(X + Y):
                raise ValueError('conformal weights must be given for exactly {}'
                                 .format(X + Y))
            if any(R(w[x]) + R(w[y]) != 1 for x, y in zip(X, Y)):
                raise ValueError('the weights of the two fields of a pair must add '
                                 'up to 1')
            weight = [R(w[x]) for x in X]
            degrees = None
        elif conformal_vector in _PAIR_WEIGHTS:
            if weight is not None:
                raise ValueError('give the weight of {} or a conformal vector, not '
                                 'both'.format(first))
            weight = QQ(_PAIR_WEIGHTS[conformal_vector])
        else:
            raise ValueError("conformal_vector: 'symmetric', 'b1c0', 'b0c1', "
                             "'degrees', an element or None")
        conformal_vector = None
    if weight is not None:
        if conformal_vector is not None or degrees is not None:
            raise ValueError('give the weight of {}, or a conformal vector or '
                             'conformal weights, not both'.format(first))
        lams = list(weight) if isinstance(weight, (list, tuple)) else [weight] * rank
        if len(lams) != rank:
            raise ValueError('{} weights expected, one per pair'.format(rank))
        g = E.gens()
        conformal_vector = sum((_pair_virasoro(g[i], g[rank + i], R(lam), odd)
                                for i, lam in enumerate(lams)), E.zero())
    if isinstance(horizontal_grading, str):
        if horizontal_grading != 'charge':
            raise ValueError("horizontal_grading: 'charge', a dict or None")
        # ghost number: c, γ have charge 1, b, β charge -1
        horizontal_grading = dict([(s, -1) for s in X] + [(s, 1) for s in Y])
    return UniversalVertexAlgebra(R, E, opes, conformal_vector=conformal_vector,
                                  degrees=degrees,
                                  horizontal_grading=horizontal_grading, horizontal_grading_group=horizontal_grading_group,
                                  implementation=implementation)


def BosonicGhosts(base_ring, rank=1, beta_weight=None, names=None,
                  conformal_vector=None, degrees=None, horizontal_grading=None,
                  horizontal_grading_group=None, implementation=None):
    r"""
    The ``βγ`` system (Weyl vertex algebra) of rank ``n``: the universal
    vertex algebra of even ``β_i``, ``γ_i`` (``i = 0, ..., n-1``) with::

        β_i(z) γ_j(w) ~ δ_ij/(z-w),   γ_i(z) β_j(w) ~ -δ_ij/(z-w),

    the λ-brackets of Sage's ``lie_conformal_algebras.BosonicGhosts``.

    INPUT:

    - ``base_ring`` -- the ring of coefficients
    - ``rank`` -- (default: 1) the number of ``βγ`` pairs (tensor factors)
    - ``beta_weight`` -- (default: ``None``) ``λ`` (or a list, one per
      pair): the conformal vector ``Σ_i (λ - 1) :∂β_i γ_i: + λ :β_i
      ∂γ_i:``, ``β_i`` of weight ``λ``, ``γ_i`` of weight ``1 - λ``, central
      charge ``2(6λ^2 - 6λ + 1)`` per pair (``-1`` for ``λ = 1/2``, ``2``
      for ``λ = 0, 1``)
    - ``names`` -- (default: ``None``) the ``β``'s, then the ``γ``'s
      (``None``: Sage's ``beta, gamma``, resp. ``beta0, ..., gamma0, ...``)
    - ``conformal_vector`` -- (default: ``None``) ``None``;
      ``'symmetric'``, ``'b1c0'``, ``'b0c1'``: ``β_i`` of weight ``1/2``,
      ``1``, ``0`` and ``γ_i`` of weight ``1/2``, ``0``, ``1``;
      ``'degrees'`` with ``degrees=`` (a weight per pair); or any conformal
      vector (an element of the algebra without conformal data, or an
      expression)
    - ``degrees`` -- (default: ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`
    - ``horizontal_grading`` -- (default: ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`, or
      ``'charge'``: ``γ_i`` of charge 1, ``β_i`` of charge -1 (needed for
      finite weight spaces with ``'b1c0'`` or ``'b0c1'``)
    - ``horizontal_grading_group``, ``implementation`` -- (default:
      ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`

    OUTPUT:

    A :class:`universal_vertex_algebra.UniversalVertexAlgebra`.

    EXAMPLES::

        sage: from vertex_algebras.ghost_systems import BosonicGhosts
        sage: [BosonicGhosts(QQ, conformal_vector=c).degrees()
        ....:  for c in ('symmetric', 'b1c0', 'b0c1')]
        [{'beta': 1/2, 'gamma': 1/2}, {'beta': 1, 'gamma': 0}, {'beta': 0, 'gamma': 1}]
        sage: [BosonicGhosts(QQ, beta_weight=l).central_charge() for l in (0, 1/2, 1, 3/2)]
        [2, -1, 2, 11]
        sage: V = BosonicGhosts(QQ, rank=2, beta_weight=[1, 1/2], horizontal_grading='charge')
        sage: V.variable_names(), V.degrees()
        (('beta0', 'beta1', 'gamma0', 'gamma1'),
         {'beta0': 1, 'beta1': 1/2, 'gamma0': 0, 'gamma1': 1/2})
        sage: V.central_charge(), V.weight_space(1, horizontal_grade=0).basis()     # γ0 has weight 0
        (1, [:beta0 gamma0:, :beta1 beta1 gamma0 gamma0:, :beta1 gamma1:])
        sage: V is BosonicGhosts(QQ, rank=2, horizontal_grading='charge', conformal_vector='degrees',
        ....:     degrees={'beta0': 1, 'beta1': 1/2, 'gamma0': 0, 'gamma1': 1/2})
        True

    The same algebra as the universal enveloping vertex algebra of Sage's
    bosonic ghosts (whose degrees give ``β`` weight 1)::

        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
        sage: L = lie_conformal_algebras.BosonicGhosts(QQ)
        sage: universal_enveloping_vertex_algebra(L, central_parameters={'K': 1}) is \
        ....:     BosonicGhosts(QQ, degrees={'beta': 1, 'gamma': 0})
        True

    A general conformal vector, an element of the algebra without one::

        sage: B = BosonicGhosts(QQ); b, g = B.gens()
        sage: T = -b.derivative().no(g) / 2 + b.no(g.derivative()) / 2 + b.no(g).derivative()
        sage: BosonicGhosts(QQ, conformal_vector=T) is BosonicGhosts(QQ, beta_weight=3/2)
        True

    TESTS::

        sage: BosonicGhosts(QQ, beta_weight=1, degrees={'beta': 1, 'gamma': 0})
        Traceback (most recent call last):
        ...
        ValueError: give the weight of beta, or a conformal vector or conformal weights, not both
        sage: BosonicGhosts(QQ, conformal_vector='degrees', degrees={'beta': 1, 'gamma': 1})
        Traceback (most recent call last):
        ...
        ValueError: the weights of the two fields of a pair must add up to 1
        sage: TestSuite(BosonicGhosts(QQ, conformal_vector='symmetric')).run()

    No conformal data by default; with ``beta_weight=λ`` the weights are
    ``(λ, 1 - λ)`` and ``c = 2(6λ^2 - 6λ + 1)``, also for a symbolic ``λ``::

        sage: V = BosonicGhosts(QQ); b, g = V.gens()
        sage: b.ope(g), g.ope(b), b.ope(b), V.conformal_vector(), V.parities()
        ([1], [-1], [], None, {'beta': 0, 'gamma': 0})
        sage: all(BosonicGhosts(QQ, beta_weight=l).central_charge() == 2*(6*l^2 - 6*l + 1)
        ....:     and BosonicGhosts(QQ, beta_weight=l).degrees() == {'beta': l, 'gamma': 1 - l}
        ....:     for l in [0, 1/2, 1, 3/2, 2, -1/3])
        True
        sage: K.<l> = Frac(QQ['l'])
        sage: BosonicGhosts(K, beta_weight=l).central_charge() == 2*(6*l^2 - 6*l + 1)
        True

    Several pairs (Sage's names), weights per pair, ``c`` additive; the
    catalog names; ``rank=2`` is ``U(L)`` of Sage's ``BosonicGhosts`` with
    four generators::

        sage: R3 = BosonicGhosts(QQ, rank=3, beta_weight=[0, 1/2, 1])
        sage: R3.variable_names(), R3.central_charge(), R3.degrees()['gamma1']
        (('beta0', 'beta1', 'beta2', 'gamma0', 'gamma1', 'gamma2'), 3, 1/2)
        sage: import vertex_algebras as va
        sage: va.BosonicGhosts(QQ, 2) is BosonicGhosts(QQ, 2)
        True
        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
        sage: L = lie_conformal_algebras.BosonicGhosts(QQ, ngens=4)
        sage: universal_enveloping_vertex_algebra(L, central_parameters={'K': 1}) is \
        ....:     BosonicGhosts(QQ, rank=2, degrees={'beta0': 1, 'beta1': 1, 'gamma0': 0,
        ....:                                          'gamma1': 0})
        True

    The character at ``λ = 1/2`` is ``Π_n (1 - q^{n-1/2})^{-2}``::

        sage: P.<t> = PowerSeriesRing(QQ, default_prec=10)        # t = q^(1/2)
        sage: V12 = BosonicGhosts(QQ, beta_weight=1/2)
        sage: ([V12.weight_space(k/2).dimension() for k in range(10)]
        ....:  == list(prod((1 - t^(2*n - 1))^-2 for n in range(1, 6)))[:10])
        True

    The named conformal vectors are those of ``beta_weight`` 1/2, 1, 0; at
    charge 0 the weight-space dimensions do not depend on ``λ`` (finite for
    ``λ = 0, 1`` only with the charge grading)::

        sage: [BosonicGhosts(QQ, conformal_vector=c) is BosonicGhosts(QQ, beta_weight=l)
        ....:  for c, l in [('symmetric', 1/2), ('b1c0', 1), ('b0c1', 0)]]
        [True, True, True]
        sage: BosonicGhosts(QQ, rank=2, conformal_vector='b1c0').degrees()
        {'beta0': 1, 'beta1': 1, 'gamma0': 0, 'gamma1': 0}
        sage: [[BosonicGhosts(QQ, horizontal_grading='charge', beta_weight=l)
        ....:   .weight_space(n, horizontal_grade=0).dimension() for n in range(6)]
        ....:  for l in (0, 1/2, 1, 2)]
        [[1, 1, 3, 6, 12, 21], [1, 1, 3, 6, 12, 21], [1, 1, 3, 6, 12, 21], [1, 1, 3, 6, 12, 21]]
        sage: BosonicGhosts(QQ, horizontal_grading='charge', beta_weight=1).gen('beta').horizontal_grade()
        -1
        sage: TestSuite(BosonicGhosts(QQ, beta_weight=1, horizontal_grading='charge')).run()

    Refusals::

        sage: BosonicGhosts(QQ, rank=2, beta_weight=[1])
        Traceback (most recent call last):
        ...
        ValueError: 2 weights expected, one per pair
    """
    return _ghost_system(base_ring, rank, False, beta_weight, names, conformal_vector,
                         degrees, horizontal_grading, horizontal_grading_group, implementation)


def FermionicGhosts(base_ring, rank=1, b_weight=None, names=None,
                    conformal_vector=None, degrees=None, horizontal_grading=None,
                    horizontal_grading_group=None, implementation=None):
    r"""
    The ``bc`` system (charged free fermions, Clifford vertex algebra) of
    rank ``n``: the universal vertex superalgebra of odd ``b_i``, ``c_i``
    with ``b_i(z) c_j(w) ~ δ_ij/(z-w) ~ c_j(z) b_i(w)``, the λ-brackets of
    Sage's ``lie_conformal_algebras.FermionicGhosts``.

    INPUT:

    - ``base_ring`` -- the ring of coefficients
    - ``rank`` -- (default: 1) the number of ``bc`` pairs
    - ``b_weight`` -- (default: ``None``) ``λ`` (or a list, one per pair):
      the conformal vector ``Σ_i (1 - λ) :∂b_i c_i: - λ :b_i ∂c_i:``,
      ``b_i`` of weight ``λ``, ``c_i`` of weight ``1 - λ``, central charge
      ``-2(6λ^2 - 6λ + 1)`` per pair (``1`` for ``λ = 1/2``, ``-2`` for
      ``λ = 0, 1``, ``-26`` for ``λ = 2``)
    - ``names`` -- (default: ``None``) the ``b``'s, then the ``c``'s
      (``None``: Sage's ``b, c``, resp. ``b0, ..., c0, ...``)
    - ``conformal_vector`` -- (default: ``None``) as for
      :func:`BosonicGhosts` (``'symmetric'``, ``'b1c0'``, ``'b0c1'``,
      ``'degrees'``, an element or an expression)
    - ``degrees`` -- (default: ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`
    - ``horizontal_grading`` -- (default: ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`, or
      ``'charge'``: the ghost number (``c_i`` 1, ``b_i`` -1)
    - ``horizontal_grading_group``, ``implementation`` -- (default:
      ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`

    OUTPUT:

    A :class:`universal_vertex_algebra.UniversalVertexAlgebra`.

    EXAMPLES::

        sage: from vertex_algebras.ghost_systems import FermionicGhosts
        sage: V = FermionicGhosts(QQ, b_weight=2, horizontal_grading='charge'); b, c = V.gens()
        sage: b.ope(c), c.ope(b), b.ope(b), V.central_charge()
        ([1], [1], [], -26)
        sage: V.degrees(), c.horizontal_grade()
        ({'b': 2, 'c': -1}, 1)
        sage: V.character(6, horizontal_grade=0)          # 1/(q)_∞, as for any b_weight
        1 + q + 2*q^2 + 3*q^3 + 5*q^4 + 7*q^5 + O(q^6)
        sage: FermionicGhosts(QQ, conformal_vector='b1c0') is FermionicGhosts(QQ, b_weight=1)
        True
        sage: FermionicGhosts(QQ, rank=3).variable_names()
        ('b0', 'b1', 'b2', 'c0', 'c1', 'c2')

    TESTS:

    ``b_weight=λ``: weights ``(λ, 1 - λ)``, ``c = -2(6λ^2 - 6λ + 1)``, also
    for a symbolic ``λ``; the named conformal vectors::

        sage: all(FermionicGhosts(QQ, b_weight=l).central_charge() == -2*(6*l^2 - 6*l + 1)
        ....:     and FermionicGhosts(QQ, b_weight=l).degrees() == {'b': l, 'c': 1 - l}
        ....:     for l in [0, 1/2, 1, 3/2, 2, 3, -1/3])
        True
        sage: K.<l> = Frac(QQ['l'])
        sage: FermionicGhosts(K, b_weight=l).central_charge() == -2*(6*l^2 - 6*l + 1)
        True
        sage: [FermionicGhosts(QQ, conformal_vector=c) is FermionicGhosts(QQ, b_weight=l)
        ....:  for c, l in [('symmetric', 1/2), ('b1c0', 1), ('b0c1', 0)]]
        [True, True, True]
        sage: FermionicGhosts(QQ, rank=2, conformal_vector='degrees',
        ....:          degrees={'b0': 3/2, 'b1': -1/3, 'c0': -1/2, 'c1': 4/3}) \
        ....:     is FermionicGhosts(QQ, rank=2, b_weight=[3/2, -1/3])
        True

    Two pairs with given names; the catalog name; the same object as
    ``U(L)`` of Sage's ``FermionicGhosts``::

        sage: B2 = FermionicGhosts(QQ, rank=2, b_weight=2, names=['b', 'bp', 'c', 'cp'])
        sage: B2.central_charge(), B2.gen('bp').ope(B2.gen('cp')), B2.gen('b').ope(B2.gen('cp'))
        (-52, [1], [])
        sage: import vertex_algebras as va
        sage: va.FermionicGhosts(QQ, b_weight=2) is FermionicGhosts(QQ, b_weight=2)
        True
        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra
        sage: L = lie_conformal_algebras.FermionicGhosts(QQ)
        sage: universal_enveloping_vertex_algebra(L, central_parameters={'K': 1}) is \
        ....:     FermionicGhosts(QQ, degrees={'b': 1, 'c': 0})
        True

    The character at ``λ = 1/2`` is ``Π_n (1 + q^{n-1/2})^2``; the Zhu
    algebra for weights ``(1, 0)`` is the Clifford algebra::

        sage: P.<t> = PowerSeriesRing(QQ, default_prec=10)        # t = q^(1/2)
        sage: B12 = FermionicGhosts(QQ, b_weight=1/2)
        sage: ([B12.weight_space(k/2).dimension() for k in range(10)]
        ....:  == list(prod((1 + t^(2*n - 1))^2 for n in range(1, 6)))[:10])
        True
        sage: A = FermionicGhosts(QQ, b_weight=1).zhu_algebra(); zb, zc = A.gens()
        sage: zb * zc + zc * zb
        1
        sage: TestSuite(B12).run(); TestSuite(B2).run()

    Refusals::

        sage: FermionicGhosts(QQ, names=['b'])
        Traceback (most recent call last):
        ...
        ValueError: 2 names expected (the first generator of each pair, then the second ones)
        sage: FermionicGhosts(QQ, conformal_vector='b1c0', b_weight=1)
        Traceback (most recent call last):
        ...
        ValueError: give the weight of b or a conformal vector, not both
        sage: FermionicGhosts(QQ, conformal_vector='degrees')
        Traceback (most recent call last):
        ...
        ValueError: conformal_vector='degrees' needs degrees=
        sage: FermionicGhosts(QQ, horizontal_grading='other')
        Traceback (most recent call last):
        ...
        ValueError: horizontal_grading: 'charge', a dict or None
    """
    return _ghost_system(base_ring, rank, True, b_weight, names, conformal_vector,
                         degrees, horizontal_grading, horizontal_grading_group, implementation)


def _parities_of_form(M):
    """The parities of the basis vectors of a non-degenerate form that is
    skew-symmetric on the even part and symmetric on the odd part,
    determined component by component (connected by nonzero entries).
    Returns the list of parities (0 even, 1 odd).

    TESTS::

        sage: from vertex_algebras.ghost_systems import _parities_of_form
        sage: _parities_of_form(matrix(QQ, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
        [0, 0, 1]
        sage: _parities_of_form(matrix(QQ, [[0, 1], [2, 0]]))
        Traceback (most recent call last):
        ...
        ValueError: the form must be, on each block of basis vectors it connects,
        skew-symmetric (even) or symmetric (odd)
    """
    n = M.nrows()
    seen = [None] * n
    for start in range(n):
        if seen[start] is not None:
            continue
        comp, stack = [], [start]
        seen[start] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in range(n):
                if (M[i, j] or M[j, i]) and seen[j] is None:
                    seen[j] = True
                    stack.append(j)
        sym = all(M[i, j] == M[j, i] for i in comp for j in comp)
        skew = all(M[i, j] == -M[j, i] for i in comp for j in comp)
        if sym == skew:
            raise ValueError('the form must be, on each block of basis vectors it '
                             'connects, skew-symmetric (even) or symmetric (odd)')
        for i in comp:
            seen[i] = 1 if sym else 0
    return seen


def _check_parities(M, par):
    """The form is even and skew-symmetric on the even, symmetric on the
    odd basis vectors (``M_ji = -(-1)^{p_i p_j} M_ij``): returns ``None``,
    raises ``ValueError`` otherwise (given parities ``par`` of a basis).

    TESTS::

        sage: from vertex_algebras.ghost_systems import _check_parities
        sage: M = matrix(QQ, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        sage: _check_parities(M, [0, 0, 1])
        sage: _check_parities(M, [1, 1, 1])
        Traceback (most recent call last):
        ...
        ValueError: the form must be skew-symmetric on the even and symmetric on the odd vectors
        sage: _check_parities(M, [0, 1, 1])
        Traceback (most recent call last):
        ...
        ValueError: the form pairs an even and an odd vector
    """
    n = M.nrows()
    for i in range(n):
        for j in range(n):
            if par[i] != par[j]:
                if M[i, j]:
                    raise ValueError('the form pairs an even and an odd vector')
            elif M[j, i] != (M[i, j] if par[i] else -M[i, j]):
                raise ValueError('the form must be skew-symmetric on the even and '
                                 'symmetric on the odd vectors')


def _weights_virasoro(R, E, opes, M, par, weights):
    r"""
    The conformal vector of the free superfermions with the conformal
    weights ``weights`` (a dict over the names): ``T = T_sym + ∂J``,
    ``T_sym`` the symmetric one (all weights ``1/2``) and ``J`` the
    quadratic field with ``J_(0) Φ_k = (1/2 - h_k) Φ_k`` (so that ``T_(1)
    = T_sym_(1) - J_(0)``), solved for among the ``:Φ_a Φ_b:``.  ``E`` the
    expressions on the fields, ``opes`` their OPEs, ``M`` the Gram matrix,
    ``par`` the parities; the result lies in the plain universal algebra.

    TESTS:

    A symplectic boson pair with weights ``1`` and ``0`` is ``βγ`` at
    ``λ = 1`` (central charge 2)::

        sage: from vertex_algebras.ghost_systems import _weights_virasoro
        sage: from vertex_algebras import NormallyOrderedExpressions
        sage: E.<x, y> = NormallyOrderedExpressions(QQ, ['x', 'y'])
        sage: M = matrix(QQ, [[0, 1], [-1, 0]])
        sage: T = _weights_virasoro(QQ, E, {('x', 'y'): [1]}, M, [0, 0], {'x': 1, 'y': 0})
        sage: T
        :dy x:
        sage: T.ope(T)
        [:dx dy: + :d^2y x:, 2*:dy x:, 0, 1]
        sage: _weights_virasoro(QQ, E, {('x', 'y'): [1]}, M, [0, 0], {'x': 1, 'y': 1})
        Traceback (most recent call last):
        ...
        ValueError: the weights of paired fields (M_ij ≠ 0) must add up to 1
    """
    names = E.variable_names()
    n = len(names)
    w = dict(weights)
    if set(w) != set(names):
        raise ValueError('conformal weights must be given for exactly {}'.format(names))
    h = [R(w[s]) for s in names]
    if any(M[i, j] and h[i] + h[j] != 1 for i in range(n) for j in range(n)):
        raise ValueError('the weights of paired fields (M_ij ≠ 0) must add up to 1')
    P = UniversalVertexAlgebra(R, E, opes)
    g = P.gens()
    N = M.inverse()
    T = sum((-N[k, i] / 2 * g[k].no(g[i].derivative())
             for i in range(n) for k in range(n) if N[k, i]), P.zero())
    if all(x == R((1, 2)) for x in h):
        return T
    cands = [(a, b) for a in range(n) for b in range(a, n) if not (a == b and par[a])]
    Js = [g[a].no(g[b]) for a, b in cands]
    keys = [x.leading_support() for x in g]
    rows, rhs = [], []
    for l in range(n):
        acts = [J.nproduct(g[l], 0).monomial_coefficients() for J in Js]
        for k in range(n):
            rows.append([d.get(keys[k], 0) for d in acts])
            rhs.append((R((1, 2)) - h[l]) if k == l else R.zero())
    A = matrix(R, rows)
    try:
        s = A.solve_right(matrix(R, len(rhs), 1, rhs)).column(0)
    except ValueError:
        raise ValueError('no conformal vector of these free superfermions has the '
                         'weights {}'.format(w))
    return T + sum((c * J.derivative() for c, J in zip(s, Js) if c), P.zero())


def FreeSuperFermions(base_ring, A, form=None, names=None, conformal_vector=None,
                      degrees=None, horizontal_grading=None, horizontal_grading_group=None,
                      implementation=None):
    r"""
    The free superfermions of a super vector space ``A`` with a
    non-degenerate even form, skew-symmetric on ``A_0`` and symmetric on
    ``A_1``: the universal vertex superalgebra of the fields ``Φ_i`` of a
    basis, of the parity of the basis vector, with::

        Φ_i(z) Φ_j(w) ~ M_ij/(z-w)

    (``M`` the Gram matrix; symplectic bosons on ``A_0``, free fermions on
    ``A_1``).

    INPUT:

    - ``base_ring`` -- the coefficients
    - ``A`` -- the Gram matrix ``M`` of the form, or a basis of ``A`` (a
      list of elements of a super module, e.g. root vectors of a
      :class:`lie_superalgebras.LieSuperalgebra`), then with ``form``
    - ``form`` -- (default: ``None``) with a basis, the form, a function of
      two elements; with a Gram matrix, ``None``
    - ``names`` -- (default: ``None``) a prefix (``None``: ``Phi``, i.e.
      ``Phi0, Phi1, ...``) or a list; with a basis the default is the
      elements as they print
    - ``conformal_vector`` -- (default: ``None``) ``None``; ``'symmetric'``
      (every ``Φ_i`` of weight ``1/2``: ``T = -1/2 Σ_i :Φ^i ∂Φ_i:``, ``Φ^i``
      the dual basis, ``c = -sdim A / 2``); ``'degrees'`` with ``degrees=``:
      the conformal vector with these weights (``T_sym + ∂J``, ``J`` quadratic;
      the weights of paired fields add up to 1); or any conformal vector, an
      element of the algebra without conformal data (as for lattice vertex
      algebras) or an expression
    - ``degrees``, ``horizontal_grading``, ``horizontal_grading_group``,
      ``implementation`` -- (default: ``None``) as for
      :class:`universal_vertex_algebra.UniversalVertexAlgebra`

    OUTPUT:

    A :class:`universal_vertex_algebra.UniversalVertexAlgebra`.

    The parities: with a Gram matrix they are read off ``M``, as for
    lattice vertex algebras (a block of basis vectors connected by nonzero
    entries is even if ``M`` is skew-symmetric on it, odd if symmetric);
    with a basis of a super module they are those of the elements (checked
    against the form).

    EXAMPLES:

    A Majorana fermion (Ising), a symplectic boson pair, a ``bc`` pair::

        sage: from vertex_algebras.ghost_systems import FreeSuperFermions
        sage: [FreeSuperFermions(QQ, M, conformal_vector='symmetric').central_charge()
        ....:  for M in (matrix([[1]]), matrix([[0, 1], [-1, 0]]), matrix([[0, 1], [1, 0]]))]
        [1/2, -1, 1]
        sage: F = FreeSuperFermions(QQ, matrix([[0, 2], [-2, 0]]), names=['x', 'y'],
        ....:                       conformal_vector='symmetric')
        sage: x, y = F.gens(); x.ope(y), F.degrees(), F.parities()
        ([2], {'x': 1/2, 'y': 1/2}, {'x': 0, 'y': 0})

    Prescribed weights; for a symplectic pair the ``βγ`` systems::

        sage: from vertex_algebras.ghost_systems import BosonicGhosts
        sage: M = matrix([[0, 1], [-1, 0]])
        sage: G = FreeSuperFermions(QQ, M, names=['beta', 'gamma'], conformal_vector='degrees',
        ....:                       degrees={'beta': 3/2, 'gamma': -1/2})
        sage: G is BosonicGhosts(QQ, beta_weight=3/2), G.central_charge()
        (True, 11)
        sage: W = FreeSuperFermions(QQ, matrix([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
        ....:     conformal_vector='degrees', degrees={'Phi0': 2, 'Phi1': 1/2, 'Phi2': -1})
        sage: W.degrees(), W.central_charge()
        ({'Phi0': 2, 'Phi1': 1/2, 'Phi2': -1}, 53/2)

    A general conformal vector, as for lattice vertex algebras: built in
    the algebra without conformal data::

        sage: P = FreeSuperFermions(QQ, M, names=['beta', 'gamma']); b, g = P.gens()
        sage: FreeSuperFermions(QQ, M, names=['beta', 'gamma'], conformal_vector=b.no(g.derivative())) \
        ....:     is BosonicGhosts(QQ, conformal_vector='b1c0')
        True

    From a basis and a form: the neutral free superfermions of the minimal
    reduction of ``sl(2|1)`` (``θ = α_1``, ``f = f_1``, ``x = θ/2``), on
    ``g_{1/2}``, spanned by the odd root vectors of ``α_1 + α_2`` and
    ``-α_2`` (``(α_1|α_2) = -1``), with ``<a, b> = (f | [a, b])``::

        sage: from vertex_algebras.lie_superalgebras import LieSuperalgebra
        sage: g = LieSuperalgebra(QQ, 'sl(2|1)'); f = g.f(1)
        sage: half = [g.root_vector((1, 1)), g.root_vector((0, 1), -1)]
        sage: Ne = FreeSuperFermions(QQ, half, lambda a, b: g.invariant_form(f, a.bracket(b)),
        ....:                        conformal_vector='symmetric')
        sage: Ne.variable_names(), Ne.central_charge()
        (('E[alpha[1] + alpha[2]]', 'E[-alpha[2]]'), 1)
        sage: Ne.parities() == {'E[alpha[1] + alpha[2]]': 1, 'E[-alpha[2]]': 1}
        True
        sage: [x.ope(y) for x in Ne.gens() for y in Ne.gens()]   # symmetric: odd
        [[], [-1], [-1], []]

    TESTS::

        sage: FreeSuperFermions(QQ, matrix([[0, 1], [2, 0]]))
        Traceback (most recent call last):
        ...
        ValueError: the form must be, on each block of basis vectors it connects, skew-symmetric (even) or symmetric (odd)
        sage: FreeSuperFermions(QQ, matrix([[1, 1], [1, 1]]))
        Traceback (most recent call last):
        ...
        ValueError: the form must be non-degenerate
        sage: FreeSuperFermions(QQ, M, conformal_vector='degrees', degrees={'Phi0': 1, 'Phi1': 1})
        Traceback (most recent call last):
        ...
        ValueError: the weights of paired fields (M_ij ≠ 0) must add up to 1
        sage: FreeSuperFermions(QQ, [g.e(1), g.f(1)], lambda a, b: g.invariant_form(a, b))
        Traceback (most recent call last):
        ...
        ValueError: the form must be skew-symmetric on the even and symmetric on the odd vectors
        sage: TestSuite(F).run(); TestSuite(Ne).run()

    Parities are read off the form in any basis order (blocks: skew =
    even, symmetric = odd); the symmetric conformal vector has all weights
    1/2 and ``c = -sdim/2``, also for congruent and non-diagonal forms::

        sage: J, S = matrix([[0, 1], [-1, 0]]), matrix([[0, 1], [1, 0]])
        sage: mixed = matrix([[0, 0, 1, 0], [0, 2, 0, 0], [-1, 0, 0, 0], [0, 0, 0, -3]])
        sage: Fm = FreeSuperFermions(QQ, mixed, conformal_vector='symmetric')
        sage: Fm.parities(), set(Fm.degrees().values()), Fm.central_charge()
        ({'Phi0': 0, 'Phi1': 1, 'Phi2': 0, 'Phi3': 1}, {1/2}, 0)
        sage: Pm = matrix(QQ, [[1, 2, 0, 1], [0, 1, -1, 3], [2, 0, 1, 1], [1, 1, 1, -1]])
        sage: [FreeSuperFermions(QQ, X, conformal_vector='symmetric').central_charge() for X in
        ....:  [block_diagonal_matrix([J, J, matrix([[1]])]),
        ....:   Pm.T * block_diagonal_matrix([J, J]) * Pm,
        ....:   matrix(QQ, [[2, 1, 0], [1, 3, 1], [0, 1, 1]])]]
        [-3/2, -2, 3/2]
        sage: TestSuite(Fm).run()

    The symmetric conformal vector of a symplectic pair is that of ``βγ`` at
    ``λ = 1/2``, of an odd pair that of ``bc`` (the same objects); the
    ``βγ`` systems from conformal vectors given as elements::

        sage: FreeSuperFermions(QQ, J, names=['beta', 'gamma'], conformal_vector='symmetric') \
        ....:     is BosonicGhosts(QQ, beta_weight=1/2)
        True
        sage: from vertex_algebras.ghost_systems import FermionicGhosts
        sage: FreeSuperFermions(QQ, S, names=['b', 'c'], conformal_vector='symmetric') \
        ....:     is FermionicGhosts(QQ, b_weight=1/2)
        True
        sage: P = FreeSuperFermions(QQ, J, names=['beta', 'gamma']); pb, pg = P.gens()
        sage: all(FreeSuperFermions(QQ, J, names=['beta', 'gamma'], conformal_vector=(l - 1)
        ....:         * pb.derivative().no(pg) + l * pb.no(pg.derivative()))
        ....:     is BosonicGhosts(QQ, beta_weight=l) for l in (0, 1, 3/2))
        True

    Prescribed weights for a ``βγ`` pair, a ``bc`` pair and a Majorana
    fermion: ``c`` is the sum of theirs::

        sage: Mb = block_diagonal_matrix([J, S, matrix([[1]])])
        sage: w = {'Phi0': 3/2, 'Phi1': -1/2, 'Phi2': 2, 'Phi3': -1, 'Phi4': 1/2}
        sage: Wv = FreeSuperFermions(QQ, Mb, conformal_vector='degrees', degrees=w)
        sage: Wv.degrees() == w, Wv.central_charge() == 11 - 26 + 1/2
        (True, True)

    The same objects as ``U(L)`` of Sage's ``Weyl`` and ``FreeFermions``::

        sage: from vertex_algebras.lie_conformal import universal_enveloping_vertex_algebra as U
        sage: U(lie_conformal_algebras.Weyl(QQ), central_parameters={'K': 1}) \
        ....:     is FreeSuperFermions(QQ, J, names=['alpha0', 'alpha1'])
        True
        sage: L = lie_conformal_algebras.FreeFermions(QQ, gram_matrix=identity_matrix(3))
        sage: U(L, central_parameters={'K': 1}) is FreeSuperFermions(
        ....:     QQ, identity_matrix(3), names=['psi_0', 'psi_1', 'psi_2'],
        ....:     degrees={'psi_0': 1/2, 'psi_1': 1/2, 'psi_2': 1/2})
        True

    The neutral free superfermions ``F(g_{1/2})``, ``<a, b> = (f|[a, b])``,
    of minimal reductions (Kac--Wakimoto): 2 bosons for ``sl_3``
    (Bershadsky--Polyakov), 4, 8, 7 fermions for ``D(2,1;a)``, ``F(4)``,
    ``G(3)``; ``c = -sdim/2``::

        sage: def neutral(g):
        ....:     th = g.theta(); f = g.root_vector(th, -1)
        ....:     half = [g.basis()[k] for k in g.basis().keys()
        ....:             if any(g.root_on_basis(k)) and g.root_norm(g.root_on_basis(k), th) == 1]
        ....:     N = FreeSuperFermions(QQ, half, lambda a, b: g.invariant_form(f, a.bracket(b)),
        ....:                           conformal_vector='symmetric')
        ....:     assert all(N.parities()[str(x)] == x.is_even_odd() for x in half)
        ....:     return sum(1 - x.is_even_odd() for x in half), N.central_charge()
        sage: [neutral(LieSuperalgebra(QQ, n, **kw)) for n, kw in [(CartanMatrix(['A', 2]), {}),
        ....:  ('D(2,1;a)', {'a': 2}), ('F(4)', {'theta': 'sl2'}), ('G(3)', {'theta': 'sl2'})]]
        [(2, -1), (0, 2), (0, 4), (0, 7/2)]

    Refusals::

        sage: FreeSuperFermions(QQ, matrix([[0, 1, 1], [-1, 0, 0], [1, 0, 1]]))
        Traceback (most recent call last):
        ...
        ValueError: the form must be, on each block of basis vectors it connects, skew-symmetric (even) or symmetric (odd)
        sage: FreeSuperFermions(QQ, J, names=['x'])
        Traceback (most recent call last):
        ...
        ValueError: 2 names expected
        sage: FreeSuperFermions(QQ, J, conformal_vector='b1c0')
        Traceback (most recent call last):
        ...
        ValueError: conformal_vector: 'symmetric', 'degrees', an element or None
        sage: FreeSuperFermions(QQ, [1, 2])
        Traceback (most recent call last):
        ...
        ValueError: with a basis, give the form (a function of two elements)
        sage: FreeSuperFermions(QQ, J, form=lambda x, y: 1)
        Traceback (most recent call last):
        ...
        ValueError: a Gram matrix needs no form
    """
    R = base_ring
    if isinstance(A, (list, tuple)):
        basis = list(A)
        if form is None:
            raise ValueError('with a basis, give the form (a function of two elements)')
        n = len(basis)
        M = matrix(R, n, n, [[R(form(x, y)) for y in basis] for x in basis])
        try:
            par = [int(x.is_even_odd()) for x in basis]
        except (AttributeError, ValueError):
            par = None
        if names is None:
            names = [str(x) for x in basis]
    else:
        if form is not None:
            raise ValueError('a Gram matrix needs no form')
        M = matrix(R, A)
        par = None
    if not M.is_square():
        raise ValueError('the form must be a square matrix')
    if M.det() == 0:
        raise ValueError('the form must be non-degenerate')
    n = M.nrows()
    if par is None:
        par = _parities_of_form(M)
    else:
        _check_parities(M, par)
    if names is None:
        names = 'Phi'
    if isinstance(names, str):
        names = ['{}{}'.format(names, i) for i in range(n)]
    names = [str(s) for s in names]
    if len(names) != n:
        raise ValueError('{} names expected'.format(n))
    E = NormallyOrderedExpressions(R, names, {s: p for s, p in zip(names, par) if p})
    opes = {(names[i], names[j]): [M[i, j]] for i in range(n) for j in range(i, n)
            if M[i, j]}
    if isinstance(conformal_vector, str):
        if conformal_vector == 'symmetric':
            weights = {s: R((1, 2)) for s in names}
        elif conformal_vector == 'degrees':
            if degrees is None:
                raise ValueError("conformal_vector='degrees' needs degrees=")
            weights, degrees = degrees, None
        else:
            raise ValueError("conformal_vector: 'symmetric', 'degrees', an element "
                             "or None")
        conformal_vector = _weights_virasoro(R, E, opes, M, par, weights)
    return UniversalVertexAlgebra(R, E, opes, conformal_vector=conformal_vector,
                                  degrees=degrees,
                                  horizontal_grading=horizontal_grading, horizontal_grading_group=horizontal_grading_group,
                                  implementation=implementation)
