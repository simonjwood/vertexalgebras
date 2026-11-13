# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Conformal bootstrap of OPEs in universal vertex algebras.

Start from a :class:`universal_vertex_algebra.UniversalVertexAlgebra` ``V``
with conformal weights (``degrees=`` or a ``conformal_vector=``),
optionally a grading, and the OPEs that are known (e.g. Virasoro and
primary fields).  :class:`OPEBootstrap` puts the most general OPE on every
pair of generators whose OPE in ``V`` is zero (or on chosen pairs): pole
``a_(n) b`` is an arbitrary combination, with unknown coefficients, of the
basis monomials of weight ``h_a + h_b - n - 1``, parity ``|a| + |b|`` (and
grade ``g_a + g_b``).  The Jacobi identities
(:meth:`~universal_vertex_algebra.UniversalVertexAlgebra.jacobi_identity_failures`)
of this ansatz are polynomial equations in the unknowns; they are imposed
triple of generators by triple, and every equation that determines an
unknown (``u x + g`` with ``u`` a nonzero constant, ``g`` free of ``x``) is
used at once to eliminate it, so the expensive triples are computed with
few unknowns left.  :meth:`OPEBootstrap.solve` decomposes what remains
(minimal associated primes, Singular) and each :class:`BootstrapSolution`
gives the vertex algebra of a component.

Coefficients: the unknowns live in ``R = K[parameters, x0, x1, ...]``.  If
``V`` is defined over a field ``K`` (e.g. ``QQ`` or ``Frac(QQ['c'])``) there
are no parameters and ``c`` is generic: dividing by ``22 + 5c`` is allowed.
If ``V`` is defined over a polynomial ring ``K[c, ...]``, its variables
become parameters of ``R`` and are solved for like the unknowns, so special
values (e.g. ``c = -22/5``) appear as separate components.  Normalisations
of the generators are not fixed automatically; :meth:`OPEBootstrap.fix`
sets unknowns to chosen values.

EXAMPLES:

The ``W_3`` algebra: a Virasoro field ``L`` and a primary field ``W`` of
weight 3 ([Zam1985]_).  The ``W W`` OPE is unknown::

    sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: from vertex_algebras.bootstrap import OPEBootstrap
    sage: K.<c> = Frac(QQ['c'])
    sage: N.<L, W> = NormallyOrderedExpressions(K)
    sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
    ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
    ....:                            degrees={'L': 2, 'W': 3})
    sage: B = OPEBootstrap(V); B
    OPE bootstrap of Universal vertex algebra on generators ['L', 'W'] over Fraction Field of Univariate Polynomial Ring in c over Rational Field: 11 unknowns in the OPEs of [('W', 'W')] (5 left), 0 equations, Jacobi identities imposed for 0 of 8 triples
    sage: B.describe('x0'), B.describe('x7')
    ('coefficient of 1 in W_(5) W', 'coefficient of :L W: in W_(0) W')

Skew-symmetry of ``W W`` was imposed at once (6 unknowns eliminated).
Virasoro covariance (the triples containing ``L``) then fixes everything
up to the normalisation ``x0``; we choose ``x0 = c/3``::

    sage: B.fix({'x0': c/3})
    sage: B.impose_jacobi()
    sage: B.unknowns(), B.equations()
    ([], [])
    sage: B.ansatz()[('W', 'W')]
    [(32/(5*c + 22))*:dL L: + ((c - 2)/(15*c + 66))*d^3L,
     (32/(5*c + 22))*:L L: + ((3*c - 6)/(10*c + 44))*d^2L,
     dL,
     2*L,
     0,
     c/3]
    sage: [S] = B.solve(); S
    Bootstrap solution 1 of 1: no free unknowns, all determined
    sage: A = S.algebra(); A
    Universal vertex algebra on generators ['L', 'W'] over Fraction Field of Univariate Polynomial Ring in c over Rational Field
    sage: A.satisfies_jacobi_identity(), A.degrees()
    (True, {'L': 2, 'W': 3})

This is Zamolodchikov's algebra [Zam1985]_: ``W_(1) W = 2βΛ + 3/10 ∂²L`` with
``Λ = :LL: - 3/10 ∂²L`` and ``β = 16/(22 + 5c)`` (equal data give the same
algebra)::

    sage: Lam = L*L - 3/10*L.derivative(2); beta = 16/(22 + 5*c)
    sage: W3 = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
    ....:     ('L', 'W'): [W.derivative(), 3*W],
    ....:     ('W', 'W'): [beta*Lam.derivative() + L.derivative(3)/15,
    ....:                  2*beta*Lam + 3/10*L.derivative(2), L.derivative(), 2*L, 0, c/3]},
    ....:     degrees={'L': 2, 'W': 3})
    sage: A is W3
    True

With ``c`` a parameter (``V`` over ``QQ['c']``) and no normalisation, the
unknowns are eliminated in favour of the ``:LL:`` coefficient ``x4`` and
the solutions are one family, polynomial in ``c`` and ``x4``; the
two-point function ``x0 = c(5c + 22) x4/96`` vanishes at ``c = -22/5``::

    sage: P.<c> = QQ['c']
    sage: N.<L, W> = NormallyOrderedExpressions(P)
    sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
    ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
    ....:                            degrees={'L': 2, 'W': 3})
    sage: B = OPEBootstrap(V)
    sage: B.impose_jacobi()
    sage: B.parameters(), B.unknowns(), B.equations()
    ([c], [x4], [])
    sage: B.describe('x4'), B.eliminated()[B.ring()('x0')]
    ('coefficient of :L L: in W_(1) W', 5/96*c^2*x4 + 11/48*c*x4)
    sage: B.solve()
    [Bootstrap solution 1 of 1: free [c, x4]]

A primary of weight 5 (normalised to ``W_(9) W = c/5``) gives a universal
vertex algebra only for ``c^2 - 268c - 44 = 0``, i.e. ``c = 134 ± 60√5``
(and ``c = 0``, where the normalisation is degenerate).  Other central
charges reported for ``W(2,5)`` in the literature (e.g. ``c = -7``) do not
appear: there the Jacobi identities only hold modulo null fields, which a
universal vertex algebra does not allow::

    sage: N.<L, X> = NormallyOrderedExpressions(P)
    sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
    ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
    ....:                            degrees={'L': 2, 'X': 5})
    sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
    sage: B.unknowns(), B.equations()
    ([], [c^3 - 268*c^2 - 44*c])
    sage: S, S0 = B.solve(); S, S0
    (Bootstrap solution 1 of 2: no free unknowns, c^2 - 268*c - 44 = 0,
     Bootstrap solution 2 of 2: no free unknowns, c = 0)
    sage: S.extension()
    Number Field in c with defining polynomial c^2 - 268*c - 44
    sage: S.algebra().satisfies_jacobi_identity()
    True

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""


from sage.matrix.constructor import matrix
from sage.rings.infinity import Infinity
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.structure.sage_object import SageObject

from .normally_ordered_expressions import NormallyOrderedExpressions


def _implementation_of(V):
    """The ``implementation`` of the bootstrap's algebras: ``'python'`` if
    ``V``'s is, else ``None`` (the engine when it takes their ring: not
    always, e.g. the solution algebras over algebraic extensions).

    TESTS::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.bootstrap import _implementation_of
        sage: _implementation_of(UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1}))
        sage: _implementation_of(UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1},
        ....:                                           implementation='python'))
        'python'
    """
    return 'python' if V._implementation == 'python' else None


#: Interreduce the remaining equations (Singular ``interred``) once there
#: are more than this many (W(2,3,4): 429 equations -> 1).
INTERRED_MIN = 20

#: Format of the state written by :meth:`OPEBootstrap.save`; bump it when
#: the meaning of the saved data changes, so old files are refused.
STATE_FORMAT = 1


def _load_saved_bootstrap(V, names, st):
    """Unpickling of :meth:`OPEBootstrap.save`: a bootstrap of ``V`` on the
    saved pairs, continuing from the saved state ``st``.

    TESTS::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.bootstrap import OPEBootstrap, _load_saved_bootstrap
        sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
        sage: B = OPEBootstrap(V); B.fix({'x0': 3})
        sage: _load_saved_bootstrap(V, 'x', B._saved_state()).ansatz()
        {('J', 'J'): [0, 3]}
        sage: _load_saved_bootstrap(V, 'x', {'kind': 'something else'})
        Traceback (most recent call last):
        ...
        ValueError: not a saved bootstrap
    """
    if not isinstance(st, dict) or st.get('kind') != 'OPEBootstrap':
        raise ValueError('not a saved bootstrap')
    if st.get('format') != STATE_FORMAT:
        raise ValueError('the saved bootstrap has state format {}, this version '
                         'reads format {}'.format(st.get('format'), STATE_FORMAT))
    return OPEBootstrap(V, pairs=st['pairs'], names=names, _state=st)


def _coefficient_field(F):
    """
    ``(K, parameter names)`` for the base ring ``F`` of the algebra: a field
    ``K`` with no parameters, or a polynomial ring over a field ``K`` whose
    variables become parameters.

    TESTS::

        sage: from vertex_algebras.bootstrap import _coefficient_field
        sage: _coefficient_field(QQ), _coefficient_field(QQ['a, b'])
        ((Rational Field, []), (Rational Field, ['a', 'b']))
        sage: _coefficient_field(Frac(QQ['c']))
        (Fraction Field of Univariate Polynomial Ring in c over Rational Field, [])
        sage: _coefficient_field(ZZ)
        Traceback (most recent call last):
        ...
        ValueError: the bootstrap needs an algebra over a field or a polynomial ring over a field, not over Integer Ring
    """
    from sage.rings.polynomial.multi_polynomial_ring_base import MPolynomialRing_base
    from sage.rings.polynomial.polynomial_ring import PolynomialRing_generic
    if isinstance(F, (PolynomialRing_generic, MPolynomialRing_base)):
        K = F.base_ring()
        if K.is_field():
            return K, list(F.variable_names())
    elif F.is_field():
        return F, []
    raise ValueError('the bootstrap needs an algebra over a field or a '
                     'polynomial ring over a field, not over {}'.format(F))


def _monic(f):
    """``f`` divided by its leading coefficient (``0`` stays ``0``).

    TESTS::

        sage: from vertex_algebras.bootstrap import _monic
        sage: R.<x, y> = QQ[]
        sage: _monic(3*x + 6*y), _monic(R.zero())
        (x + 2*y, 0)
    """
    return f if not f else f / f.lc()


class OPEBootstrap(SageObject):
    r"""
    OPE ansätze with unknown coefficients for the generator pairs of a
    universal vertex algebra whose OPE is zero (or chosen pairs), and their
    Jacobi identities.

    INPUT:

    - ``V`` -- a universal vertex algebra over a field or a polynomial ring
      over a field, with conformal weights (``degrees=`` or a
      ``conformal_vector=``), possibly a grading, and the known (nonzero)
      OPEs

    - ``pairs`` -- (default: ``None``) the generator pairs ``(a, b)`` to
      put an ansatz on; ``None`` for every pair whose OPE in ``V`` is zero,
      except ``zero_pairs`` (pairs not listed have vanishing OPE)

    - ``names`` -- (default: ``'x'``) the prefix of the unknowns

    - ``zero_pairs`` -- (default: ``None``) pairs whose OPE is known to
      vanish (no ansatz on them by default)

    Weight spaces of the poles must be finite (positive weights, or a
    grading making them finite, see
    :meth:`~vertex_algebra_categories.ProtoVertexAlgebras.Graded.ParentMethods.weight_space`).  Skew-symmetry of
    ansätze of a generator with itself is imposed at once.

    EXAMPLES:

    A free boson ``J`` of weight 1 needs ``J_(0) J = 0``::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.bootstrap import OPEBootstrap
        sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
        sage: B = OPEBootstrap(V)
        sage: B.ansatz()
        {('J', 'J'): [0, x0]}
        sage: B.impose_jacobi(); B.solve()
        [Bootstrap solution 1 of 1: free [x0]]

    The ``N = 1`` superconformal algebra: an odd primary ``G`` of weight
    3/2; the ``G G`` OPE is fixed up to normalisation::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: K.<c> = Frac(QQ['c'])
        sage: N.<L, G> = NormallyOrderedExpressions(K, parities={'G': 1})
        sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
        ....:                                   ('L', 'G'): [G.derivative(), 3/2*G]},
        ....:                            degrees={'L': 2, 'G': 3/2})
        sage: B = OPEBootstrap(V); B.ansatz()
        {('G', 'G'): [x1*L, 0, x0]}
        sage: B.impose_jacobi(); B.ansatz()
        {('G', 'G'): [3/c*x0*L, 0, x0]}
        sage: [S] = B.solve(); S.free_variables()
        [x0]
        sage: A = S.algebra(); A.base_ring()
        Univariate Polynomial Ring in x0 over Fraction Field of Univariate Polynomial Ring in c over Rational Field
        sage: A.satisfies_jacobi_identity()
        True

    Affine ``sl_2``: currents ``E, H, F`` of weight 1 and charges ``2, 0,
    -2`` (a grading, so the ansatz only has terms of the right charge).
    With ``H_(0) E = 2E`` and ``E_(0) F = H`` fixed, the Jacobi identities
    give ``H_(0) F = -2F`` and the level ``k = x1 = E_(1) F`` with
    ``H_(1) H = 2k``::

        sage: V = UniversalVertexAlgebra(QQ, ['E', 'H', 'F'],
        ....:                            degrees={'E': 1, 'H': 1, 'F': 1},
        ....:                            horizontal_grading={'E': 2, 'F': -2})
        sage: B = OPEBootstrap(V); B.ansatz()
        {('E', 'E'): [],
         ('E', 'F'): [x2*H, x1],
         ('E', 'H'): [x0*E],
         ('F', 'F'): [],
         ('H', 'F'): [x5*F],
         ('H', 'H'): [0, x3]}
        sage: B.fix({'x0': -2, 'x2': 1})
        sage: B.impose_jacobi(); B.ansatz()
        {('E', 'E'): [],
         ('E', 'F'): [H, x1],
         ('E', 'H'): [-2*E],
         ('F', 'F'): [],
         ('H', 'F'): [-2*F],
         ('H', 'H'): [0, 2*x1]}
        sage: B.solve()
        [Bootstrap solution 1 of 1: free [x1]]

    TESTS:

    ``W(2,5)`` at ``c = -7`` (a value reported in the literature, where the
    Jacobi identities hold only modulo null fields) has no exact solution::

        sage: N.<L, X> = NormallyOrderedExpressions(QQ)
        sage: V = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, -7/2],
        ....:                                    ('L', 'X'): [X.derivative(), 5*X]},
        ....:                            degrees={'L': 2, 'X': 5})
        sage: B = OPEBootstrap(V); B.fix({'x0': -7/5}); B.impose_jacobi(); B.solve()
        []

    ``W(2,3,4)``: the ``W_4`` family (couplings free up to the two
    normalisations) and a degenerate component; the scale invariant ``C^2
    N_U / N_W^2`` of the coupling ``C`` of ``U`` in ``W_(1) W`` (a regression
    value: all Jacobi identities hold)::

        sage: N.<L, W, U> = NormallyOrderedExpressions(K)
        sage: vir = {('L', 'L'): [L.derivative(), 2*L, 0, c/2]}
        sage: V = UniversalVertexAlgebra(K, N, vir | {('L', 'W'): [W.derivative(), 3*W],
        ....:     ('L', 'U'): [U.derivative(), 4*U]}, degrees={'L': 2, 'W': 3, 'U': 4})
        sage: B = OPEBootstrap(V); B.impose_jacobi('sorted'); sols = B.solve()   # long time
        sage: sols, B.equations()                                                 # long time
        ([Bootstrap solution 1 of 2: free [x7, x55], x32 = 0,
          Bootstrap solution 2 of 2: free [x32, x55], x7 = 0],
         [x7*x32])
        sage: A = sols[0].algebra(); WW, UU = A.gen('W').ope(A.gen('W')), A.gen('U').ope(A.gen('U'))  # long time
        sage: NW, NU, C = WW[5].coefficient(()), UU[7].coefficient(()), WW[1].coefficient(((2, 0),))  # long time
        sage: C^2 * NU / NW^2 == 12*(c + 2)*(7*c + 114) / (c*(5*c + 22)*(c + 7))  # long time
        True

    ``N = 2`` with the ``U(1)`` charge grading (no ``G^+ G^+`` terms): three
    components, the same for all and for sorted triples; the
    superconformal one has the standard OPEs, e.g. at ``x2 = 1``, ``x5 =
    2``::

        sage: N.<L, J, Gp, Gm> = NormallyOrderedExpressions(K, parities={'Gp': 1, 'Gm': 1})
        sage: V = UniversalVertexAlgebra(K, N, vir | {('L', 'J'): [J.derivative(), J],
        ....:     ('L', 'Gp'): [Gp.derivative(), 3/2*Gp], ('L', 'Gm'): [Gm.derivative(), 3/2*Gm]},
        ....:     degrees={'L': 2, 'J': 1, 'Gp': 3/2, 'Gm': 3/2},
        ....:     horizontal_grading={'Gp': 1, 'Gm': -1})
        sage: B = OPEBootstrap(V); B.ansatz()[('Gp', 'Gp')], B.ansatz()[('Gm', 'Gm')]
        ([], [])
        sage: B.impose_jacobi(); sols = B.solve(); sols
        [Bootstrap solution 1 of 3: free [x0, x2, x3], x5 = 0, x7 = 0,
         Bootstrap solution 2 of 3: free [x2, x5], x0 = c/3*x2^2, x3 = -x2, x7 = 0,
         Bootstrap solution 3 of 3: free [x5, x7], x0 = 0, x2 = 0, x3 = 0]
        sage: B2 = OPEBootstrap(V); B2.impose_jacobi('sorted')
        sage: [S.prime_ideal() for S in B2.solve()] == [S.prime_ideal() for S in sols]
        True
        sage: A = sols[1].algebra(); L2, J2, Gp2, Gm2 = A.gens()
        sage: all(S.algebra().satisfies_jacobi_identity() for S in sols)
        True
        sage: ([p.map_coefficients(lambda t: t(1, 2)) for p in Gp2.ope(Gm2)]
        ....:  == [p.map_coefficients(lambda t: t(1, 2))
        ....:      for p in [2*L2 + J2.derivative(), 2*J2, 2*c/3*A.one()]])
        True

    ``pairs`` and ``zero_pairs`` restrict the ansatz; two bootstraps of one
    algebra share their algebras until one eliminates an unknown, and an
    elimination builds a new algebra::

        sage: N.<L, W, X> = NormallyOrderedExpressions(K)
        sage: V = UniversalVertexAlgebra(K, N, vir | {('L', 'W'): [W.derivative(), 3*W],
        ....:     ('L', 'X'): [X.derivative(), 3*X]}, degrees={'L': 2, 'W': 3, 'X': 3})
        sage: list(OPEBootstrap(V, pairs=[('W', 'X')]).ansatz())
        [('W', 'X')]
        sage: list(OPEBootstrap(V, zero_pairs=[('X', 'W'), ('X', 'X')]).ansatz())
        [('W', 'W')]
        sage: B1, B2 = OPEBootstrap(V), OPEBootstrap(V); B1.algebra() is B2.algebra()
        True
        sage: U = B1.algebra(); before = U.opes(); B1.fix({'x0': 1})
        sage: B1.algebra() is not U, U.opes() == before, B1.algebra() is B2.algebra()
        (True, True, False)

    Refusals::

        sage: OPEBootstrap(V, pairs=[('L', 'W')])
        Traceback (most recent call last):
        ...
        ValueError: the OPE of L and W is known; it cannot get an ansatz
        sage: OPEBootstrap(V, pairs=[('W', 'X'), ('X', 'W')])
        Traceback (most recent call last):
        ...
        ValueError: the pair (X, W) is listed twice
        sage: OPEBootstrap(UniversalVertexAlgebra(QQ, ['J']))
        Traceback (most recent call last):
        ...
        ValueError: ...
        sage: OPEBootstrap(UniversalVertexAlgebra(ZZ, ['J'], degrees={'J': 1}))
        Traceback (most recent call last):
        ...
        ValueError: ...
        sage: OPEBootstrap(QQ)
        Traceback (most recent call last):
        ...
        TypeError: the bootstrap needs a universal vertex algebra, not Rational Field
    """

    def __init__(self, V, pairs=None, names='x', zero_pairs=None, _state=None):
        r"""
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: N.<L, W> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2],
            ....:                                    ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B._V is V, B._h, list(B.ansatz()), B.unknowns()
            (True, [2, 3], [('W', 'W')], [x0, x1, x4, x5, x6])
        """
        from .weight_spaces import _to_rational
        if not hasattr(V, '_jacobi_triple_iter'):
            raise TypeError('the bootstrap needs a universal vertex algebra, '
                            'not {}'.format(V))
        self._V = V
        data = V._weight_data()
        gens = list(V.variable_names())
        r = len(gens)
        self._h = [_to_rational(x, 'conformal weight of ' + s)
                   for x, s in zip(data['h'], gens)]
        par = [V._parities.get(i, 0) for i in range(r)]
        gd = V._grading
        idx = V._idx
        # what the ansatz depends on (checked when a saved state is loaded)
        self._fingerprint = V._state_fingerprint()
        self._gkey = None if gd is None else (tuple(gd['group'].moduli),
                                              tuple(gd['g']))

        known = set(V._opes)
        for a, b in zero_pairs or ():
            i, j = idx[a], idx[b]
            known.add((min(i, j), max(i, j)))
        if pairs is None:
            plist = [(i, j) for i in range(r) for j in range(i, r)
                     if (i, j) not in known and (j, i) not in known]
        else:
            plist = []
            for a, b in pairs:
                i, j = idx[a], idx[b]
                if (i, j) in known or (j, i) in known:
                    raise ValueError('the OPE of {} and {} is known; it '
                                     'cannot get an ansatz'.format(a, b))
                if (i, j) in plist or (j, i) in plist:
                    raise ValueError('the pair ({}, {}) is listed twice'
                                     .format(a, b))
                plist.append((i, j))

        # the terms of the ansatz: (pair, n, key), most singular poles first
        terms, nmax = [], {}
        for (i, j) in plist:
            G = None if gd is None else gd['group'].add(gd['g'][i], gd['g'][j])
            lb, _ = V._weight_bounds(G, data)
            if lb is None:
                raise ValueError(
                    'the weight spaces of the OPE of {} and {} are not '
                    'certified finite dimensional (choose a grading that '
                    'makes them finite)'.format(gens[i], gens[j]))
            top = self._h[i] + self._h[j] - 1        # weight of a_(0) b
            nmax[(i, j)] = -1 if lb == Infinity or top < lb else \
                int((top - lb).floor())
            horizontal_grade = None if G is None else gd['group'].element(G)
            p = (par[i] + par[j]) % 2
            for n in range(nmax[(i, j)], -1, -1):
                keys = [m for m in V.weight_space(top - n, horizontal_grade).basis_keys()
                        if V._parity_key(m) == p]
                terms.extend(((i, j), n, m) for m in keys)

        K, params = _coefficient_field(V.base_ring())
        unames = ['{}{}'.format(names, t) for t in range(len(terms))]
        if set(unames) & set(params):
            raise ValueError('the unknowns {}0, {}1, ... clash with the '
                             'parameters {}; choose other names'.format(
                                 names, names, params))
        if not unames and not params:
            raise ValueError('nothing to determine: no ansatz and no '
                             'parameters (use jacobi_identity_failures())')
        allnames = params + unames
        R = PolynomialRing(K, len(allnames), allnames)
        self._K, self._R = K, R
        self._params = list(R.gens()[:len(params)])
        self._unknowns = list(R.gens()[len(params):])
        self._terms = dict(zip(self._unknowns, terms))
        self._gens = gens
        # OPE data over R, as expressions in basis monomials: the declared
        # ones, then the ansätze; the algebra over R is rebuilt from them
        # (eliminated variables substituted) after every elimination
        E = self._E = NormallyOrderedExpressions(R, gens, V.parities())
        self._known = {k: [self._to_R(p) for p in poles]
                       for k, poles in V._opes.items()}
        self._pairs = plist
        self._ansatz0 = {}
        for (i, j) in plist:
            poles = [{} for _ in range(nmax[(i, j)] + 1)]
            for x, (pair, n, m) in self._terms.items():
                if pair == (i, j):
                    poles[n][m] = poles[n].get(m, 0) + x
            poles = [E._from_dict(d, remove_zeros=True) for d in poles]
            while poles and poles[-1].is_zero():
                poles.pop()
            self._ansatz0[(i, j)] = poles
        self._subs = {}               # eliminated variable -> value
        self._eqs = set()             # remaining equations (monic)
        self._done = []               # triples imposed (indices)
        self._update_algebra()
        self._names = names
        if _state is None:
            self._add(el for _, el in self._U._skew_failure_iter())
        else:
            self._restore(_state)

    # ------------------------------------------------------------------ #
    #  Basics                                                             #
    # ------------------------------------------------------------------ #

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: OPEBootstrap(V)
            OPE bootstrap of Universal vertex algebra on generators ['J'] over Rational Field: 2 unknowns in the OPEs of [('J', 'J')] (1 left), 0 equations, Jacobi identities imposed for 0 of 1 triples
        """
        return ('OPE bootstrap of {}: {} unknowns in the OPEs of {} ({} '
                'left), {} equations, Jacobi identities imposed for {} of {} '
                'triples'.format(self._V, len(self._unknowns),
                                 [self._pair_names(p) for p in self._pairs],
                                 len(self.unknowns()), len(self._eqs),
                                 len(self._done), len(self._gens) ** 3))

    def _pair_names(self, p):
        """
        The generator names of a pair ``p`` of generator indices.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: OPEBootstrap(V)._pair_names((0, 0))
            ('J', 'J')
        """
        return (self._gens[p[0]], self._gens[p[1]])

    def _to_R(self, el):
        """An element of ``V`` as an expression over ``R`` (its basis
        monomials as words).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); J = V.gen('J')
            sage: e = B._to_R(3*J.no(J)); e
            3*:J J:
            sage: e.parent()
            Normally ordered expressions in the fields ['J'] over Multivariate Polynomial Ring in x0, x1 over Rational Field
        """
        R = self._R
        return self._E._from_dict({m: R(c) for m, c in
                                   el._monomial_coefficients.items()},
                                  remove_zeros=True)

    def _var(self, x):
        """The variable of ``R`` named or given by ``x``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V)
            sage: B._var('x0'), B._var(B.ring().gen(1))
            (x0, x1)
            sage: B._var('y')
            Traceback (most recent call last):
            ...
            ValueError: 'y' is not an unknown or parameter
        """
        if isinstance(x, str):
            try:
                return self._R(x)
            except (TypeError, ValueError):
                raise ValueError('{!r} is not an unknown or parameter'.format(x))
        x = self._R(x)
        if x not in self._R.gens():
            raise ValueError('{} is not an unknown or parameter'.format(x))
        return x

    def original_algebra(self):
        """
        The algebra ``V`` the bootstrap was started from.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.original_algebra() is V
            True
        """
        return self._V

    def ring(self):
        """
        The polynomial ring of the parameters and unknowns.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.ring()
            Multivariate Polynomial Ring in c, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
             over Rational Field
        """
        return self._R

    def algebra(self):
        """
        The universal vertex algebra over :meth:`ring` with the current
        ansatz (eliminated unknowns substituted); a new algebra after every
        elimination.  A proto vertex algebra.

        OUTPUT: a :class:`universal_vertex_algebra.UniversalVertexAlgebra`

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: U = B.algebra(); U.base_ring() is B.ring()
            True
            sage: U.gen('W').ope(U.gen('W'))[-1]
            x0
            sage: B.fix({'x0': c/3}); B.algebra() is U
            False
            sage: B.algebra().gen('W').ope(B.algebra().gen('W'))[-1]
            1/3*c
        """
        return self._U

    def unknowns(self):
        """
        The unknowns not eliminated yet.

        OUTPUT: a list of generators of :meth:`ring`

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.unknowns()          # after skew-symmetry
            [x0, x1, x4, x5, x6]
            sage: B.impose_jacobi(); B.unknowns()
            [x4]
        """
        return [x for x in self._unknowns if x not in self._subs]

    def parameters(self):
        """
        The parameters (variables of the base ring of ``V``) not
        eliminated yet.

        OUTPUT: a list of generators of :meth:`ring`

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.parameters()
            [c]
        """
        return [x for x in self._params if x not in self._subs]

    def eliminated(self):
        """
        A dict: eliminated unknown or parameter -> its value (in the
        remaining ones).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi()
            sage: x = B.eliminated(); len(x), x[B.ring()('x0')]
            (10, 5/96*c^2*x4 + 11/48*c*x4)
        """
        return dict(self._subs)

    def describe(self, x):
        """
        Which coefficient of which OPE the unknown ``x`` is.

        INPUT:

        - ``x`` -- an unknown or parameter (or its name)

        OUTPUT:

        A string: ``'coefficient of m in a_(n) b'`` for an unknown,
        ``'parameter x'`` for a parameter.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.describe('x0'), B.describe('x4'), B.describe('c')
            ('coefficient of 1 in W_(5) W', 'coefficient of :L L: in W_(1) W', 'parameter c')
        """
        x = self._var(x)
        if x not in self._terms:
            return 'parameter {}'.format(x)
        (i, j), n, m = self._terms[x]
        return 'coefficient of {} in {}_({}) {}'.format(
            self._V._repr_term(m), self._gens[i], n, self._gens[j])

    def ansatz(self):
        """
        The current OPEs of the ansatz pairs (in the orientation of the
        pairs), as lists of poles.

        OUTPUT:

        A dictionary ``{(a, b): [a_(0) b, a_(1) b, ...]}`` keyed by pairs of
        generator names, the poles elements of :meth:`algebra`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi(); B.ansatz()[('W', 'W')][3:]
            [(5/16*c*x4 + 11/8*x4)*L, 0, 5/96*c^2*x4 + 11/48*c*x4]
        """
        U = self._U
        return {self._pair_names(p): U.gen(p[0]).ope(U.gen(p[1]))
                for p in self._pairs}

    def equations(self):
        """
        The remaining equations (monic polynomials, sorted), after the
        eliminations.

        OUTPUT: a sorted list of elements of :meth:`ring`

        EXAMPLES:

        A primary of weight 5 exists only for special central charges::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: B.equations()
            [c^3 - 268*c^2 - 44*c]
        """
        return sorted(self._eqs)

    def imposed_triples(self):
        """
        The generator triples whose Jacobi identities were imposed.

        OUTPUT: a list of triples of generator names

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi([('L', 'W', 'W'), ('W', 'L', 'W')])
            sage: B.imposed_triples()
            [('L', 'W', 'W'), ('W', 'L', 'W')]
        """
        return [tuple(self._gens[i] for i in t) for t in self._done]

    # ------------------------------------------------------------------ #
    #  Equations and eliminations                                         #
    # ------------------------------------------------------------------ #

    def _update_algebra(self):
        """Build the internal algebra: the declared OPEs and the ansätze,
        eliminated variables substituted.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); R = B.ring()
            sage: B._subs = {R('x1'): R(0), R('x0'): R(7)}
            sage: B._update_algebra(); U = B._U
            sage: U.base_ring() is R, U.gen('J').ope(U.gen('J'))
            (True, [0, 7])
        """
        from .universal_vertex_algebra import UniversalVertexAlgebra
        R, E, V = self._R, self._E, self._V
        phi = R.hom([self._subs.get(g, g) for g in R.gens()]) \
            if self._subs else None

        def sub(p):
            if phi is None:
                return p
            return E._from_dict({m: phi(c) for m, c in
                                 p.monomial_coefficients().items()},
                                remove_zeros=True)

        data = {self._pair_names(k): [sub(p) for p in poles]
                for k, poles in list(self._known.items())
                + list(self._ansatz0.items())}
        U = UniversalVertexAlgebra(R, self._gens, data, parities=V.parities(),
                                   degrees=dict(zip(self._gens, self._h)),
                                   implementation=_implementation_of(V),
                                   proto=True)
        self._U = U

    def _add(self, elements):
        """Add the coefficients of ``elements`` as equations and
        eliminate.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); x0 = B.ring()('x0')
            sage: B._add([(x0 - 1)*B._U.one()]); B.ansatz()
            {('J', 'J'): [0, 1]}
        """
        self._add_polys(c for el in elements
                        for c in el._monomial_coefficients.values())

    def _priority(self, v):
        """Elimination priority: unknowns before parameters, later
        unknowns (less singular poles) first.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ['c'], ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V)
            sage: sorted(B.ring().gens(), key=B._priority, reverse=True)
            [x1, x0, c]
        """
        return self._R.gens().index(v)

    def _reduce(self):
        """Eliminate variables while some equation determines one: first
        all linear equations at once (echelon form over ``K``), then one
        ``u x + g`` (``u`` a nonzero constant, ``x`` not in ``g``) at a
        time; then update the algebra.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V); B.unknowns()
            [x0, x1, x4, x5, x6]
            sage: x0, x1, x4, x5 = [B.ring()(s) for s in ['x0', 'x1', 'x4', 'x5']]
            sage: B._eqs = {x0 + x1*x4, x1 - 2}; B._reduce()
            sage: B.unknowns(), B.equations()
            ([x0, x5, x6], [])
            sage: B.eliminated()[x1], B.eliminated()[x4]
            (2, -1/2*x0)
        """
        changed = False
        while self._eqs and self._R.one() not in self._eqs:
            lin = [f for f in self._eqs if f.degree() == 1]
            sol = self._solve_linear(lin) if lin else None
            if sol is None:
                sol = self._nonlinear_elimination()
            if not sol:
                break
            self._eliminate(sol)
            changed = True
        if changed:
            self._update_algebra()

    def _solve_linear(self, lin):
        """Pivot variables of the linear equations -> values; ``None`` if
        there are none (``{1}`` stored if inconsistent).  Nothing is
        substituted.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V); B.unknowns()
            [x0, x1, x4, x5, x6]
            sage: x0, x1, x4, x5 = [B.ring()(s) for s in ['x0', 'x1', 'x4', 'x5']]
            sage: B._solve_linear([x0 - 2*x1, x1 - x4 + 1])
            {x4: 1/2*x0 + 1, x1: 1/2*x0}
            sage: B._solve_linear([x0 - x1, x0 - x1 + 1]), B.equations()
            (None, [1])
        """
        R, K = self._R, self._K
        cols = sorted({v for f in lin for v in f.variables()},
                      key=self._priority, reverse=True)
        M = matrix(K, [[f.monomial_coefficient(v) for v in cols]
                       + [f.constant_coefficient()] for f in lin]).rref()
        piv = M.pivots()
        if len(cols) in piv:
            self._eqs = {R.one()}
            return None
        out = {}
        for i, p in enumerate(piv):
            out[cols[p]] = -sum((M[i, q] * cols[q]
                                 for q in range(p + 1, len(cols))
                                 if q not in piv), R(M[i, len(cols)]))
        return out or None

    def _nonlinear_elimination(self):
        """One elimination ``{x: -g/u}`` from some ``u x + g``, ``None`` if
        no equation has that form.  Nothing is substituted.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V); B.unknowns()
            [x0, x1, x4, x5, x6]
            sage: x0, x1, x4, x5 = [B.ring()(s) for s in ['x0', 'x1', 'x4', 'x5']]
            sage: B._eqs = {x0*x1 - x0*x4^2 + 2*x5}; B._nonlinear_elimination()
            {x5: 1/2*x0*x4^2 - 1/2*x0*x1}
            sage: B._eqs = {x0*x1 - 1}; B._nonlinear_elimination() is None
            True
        """
        best = None
        for f in self._eqs:
            for v in f.variables():
                if f.degree(v) != 1:
                    continue
                u = f.coefficient({v: 1})
                if not u.is_constant():
                    continue
                key = (self._priority(v), -len(f.monomials()))
                if best is None or key > best[0]:
                    best = (key, v, f, u.constant_coefficient())
        if best is None:
            return None
        _, v, f, u = best
        return {v: -(f - u * v) / u}

    def _eliminate(self, sol):
        """Substitute ``sol`` (variable -> value free of eliminated
        variables) everywhere (the algebra is not updated: see
        :meth:`_reduce`).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); B._eliminate({B.ring()('x0'): 5}); B._subs
            {x1: 0, x0: 5}
        """
        R = self._R
        phi = R.hom([sol.get(g, g) for g in R.gens()])
        self._subs = {x: phi(val) for x, val in self._subs.items()}
        self._subs.update(sol)
        eqs = {_monic(phi(f)) for f in self._eqs}
        eqs.discard(R.zero())
        self._eqs = eqs

    def fix(self, values):
        r"""
        Set unknowns (or parameters) to values, e.g. to normalise
        generators.

        INPUT:

        - ``values`` -- a dict ``unknown or name -> value``

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V)
            sage: B.fix({'x0': 1}); B.ansatz()
            {('J', 'J'): [0, 1]}
            sage: B.fix({'x0': 2}); B.equations()
            [1]
            sage: B.solve()
            []
        """
        R = self._R
        sub = R.hom([self._subs.get(g, g) for g in R.gens()])
        self._add_polys([sub(self._var(x) - R(val))
                         for x, val in values.items()])

    def _add_polys(self, polys):
        """Add equations (polynomials free of eliminated variables) and
        eliminate.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); x0 = B.ring()('x0')
            sage: B._add_polys([x0^2 - 2]); B.equations()
            [x0^2 - 2]
            sage: B._add_polys([2*x0 - 2]); B.equations()
            [1]
        """
        new = {_monic(f) for f in polys if f}
        if new - self._eqs:
            self._eqs |= new
            self._reduce()

    def _interreduce(self, minimum=INTERRED_MIN):
        """Replace the equations by an interreduced basis of their ideal
        (it may contain new eliminable equations) if there are more than
        ``minimum``.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V); B.unknowns()
            [x0, x1, x4, x5, x6]
            sage: x0, x1, x4, x5 = [B.ring()(s) for s in ['x0', 'x1', 'x4', 'x5']]
            sage: B._eqs = {x0*x1 - 1, x0*x1 + x5 - 1}
            sage: B._interreduce(); len(B.equations())
            2
            sage: B._interreduce(minimum=0)
            sage: B.equations(), B.eliminated()[x5]
            ([x0*x1 - 1], 0)
        """
        if len(self._eqs) > minimum and self._R.one() not in self._eqs:
            basis = self._R.ideal(sorted(self._eqs)).interreduced_basis()
            self._eqs = {_monic(f) for f in basis if f}
            self._reduce()

    def impose_jacobi(self, triples=None, verbose=False, save=None):
        r"""
        Impose the Jacobi identities of the given generator triples
        (default: all ordered triples; ``'sorted'``: those with ``a <= b <=
        c``, see
        :meth:`~universal_vertex_algebra.UniversalVertexAlgebra.jacobi_identity_failures`),
        cheapest (lowest total weight) first, eliminating unknowns after
        each triple.  Triples imposed before are skipped.

        INPUT:

        - ``triples`` -- (default: ``None``, all ordered triples) a list of
          triples of generator names, or ``'sorted'`` (those with ``a <= b
          <= c``)

        - ``verbose`` -- (default: ``False``) print one line per triple

        - ``save`` -- (default: ``None``) a file name: ``self`` is saved
          there after every triple (:meth:`save`), so a long run can be
          resumed with ``load(save)``

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi([('L', 'W', 'W')], verbose=True)
            Jacobi (L, W, W): 9 failing identities, 1 unknowns left, 0 equations, ... s
            sage: B.imposed_triples(), B.unknowns()
            ([('L', 'W', 'W')], [x0])

        TESTS:

        The ``W(2, 4)`` bootstrap gives the same eliminations (also in their
        order) and equations with one and with two engine threads (the
        threaded run on generators with other names: an algebra of its own,
        with cold caches)::

            sage: from vertex_algebras.vertex_algebra import VertexAlgebraParallelism
            sage: res = []
            sage: for n, (a, b) in [(1, ('L4', 'W4')), (2, ('L4p', 'W4p'))]:
            ....:     VertexAlgebraParallelism().set(n)
            ....:     N4 = NormallyOrderedExpressions(K, [a, b]); L4, W4 = N4.gens()
            ....:     V4 = UniversalVertexAlgebra(K, N4, {(a, a): [L4.derivative(), 2*L4, 0, c/2],
            ....:                                         (a, b): [W4.derivative(), 4*W4]},
            ....:                                 degrees={a: 2, b: 4})
            ....:     B4 = OPEBootstrap(V4); B4.impose_jacobi()
            ....:     res.append((list(B4.eliminated().items()), B4.equations()))
            sage: VertexAlgebraParallelism().reset()
            sage: res[0] == res[1], len(res[0][0]), res[0][1]
            (True, 21, [])
        """
        import time
        trs = self._U._jacobi_triples(triples)
        h = self._h
        trs = sorted(dict.fromkeys(trs), key=lambda t: sum(h[i] for i in t))
        for t in trs:
            if t in self._done:
                continue
            t0 = time.time()
            els = [el for _, el in self._U._jacobi_triple_iter(*t)]
            self._add(els)
            self._interreduce()
            self._done.append(t)
            if save is not None:
                self.save(save)
            if verbose:
                print('Jacobi ({}): {} failing identities, {} unknowns left, '
                      '{} equations, {:.1f} s'.format(
                          ', '.join(self._gens[i] for i in t), len(els),
                          len(self.unknowns()), len(self._eqs),
                          time.time() - t0), flush=True)
        self._interreduce(1)

    # ------------------------------------------------------------------ #
    #  Saving and loading                                                 #
    # ------------------------------------------------------------------ #

    def save(self, filename=None, compress=True):
        r"""
        Save ``self`` (the algebra, the ansatz, the eliminations, the
        remaining equations and the imposed triples) to ``filename``;
        Sage's ``save(B, filename)`` calls this.  ``load(filename)`` returns
        a bootstrap continuing from there, also in another session or on
        another machine (the file does not depend on the parallelism or on
        the engines).  A compressed Sage pickle, written atomically;
        only load files from trusted sources.  Loading checks that the
        state belongs to the same ansatz: the base ring, generators,
        parities and declared OPEs of ``V``, its conformal weights and
        grading, the pairs and the names of the unknowns.

        INPUT:

        - ``filename`` -- (default: ``None``) the file name (Sage's ``save``
          adds ``.sobj``)

        - ``compress`` -- (default: ``True``) compress the pickle

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: K.<c> = Frac(QQ['c'])
            sage: N.<L, W> = NormallyOrderedExpressions(K)
            sage: V = UniversalVertexAlgebra(
            ....:     K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:            ('L', 'W'): [W.derivative(), 3*W]},
            ....:     degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi([('L', 'W', 'W')]); B.unknowns()
            [x0]
            sage: fn = tmp_filename(ext='.sobj')
            sage: save(B, fn)

        The loaded bootstrap continues with the remaining triples::

            sage: B2 = load(fn)
            sage: B2.imposed_triples(), B2.unknowns(), B2.eliminated() == B.eliminated()
            ([('L', 'W', 'W')], [x0], True)
            sage: B2.impose_jacobi(); len(B2.imposed_triples()), B2.solve()
            (8, [Bootstrap solution 1 of 1: free [x0]])

        States of another ansatz are refused::

            sage: from vertex_algebras.bootstrap import _load_saved_bootstrap
            sage: _load_saved_bootstrap(V, 'y', B._saved_state())
            Traceback (most recent call last):
            ...
            ValueError: the saved bootstrap state was saved for another ansatz (other pairs or names of the unknowns)
            sage: V3 = UniversalVertexAlgebra(K, ['L', 'W'], degrees={'L': 2, 'W': 3})
            sage: _load_saved_bootstrap(V3, 'x', B._saved_state())
            Traceback (most recent call last):
            ...
            ValueError: the saved bootstrap state was saved for a different vertex algebra (base ring, generators, parities or declared OPEs)
        """
        from .vertex_algebra import _save_with_state
        _save_with_state(self, filename, compress, _load_saved_bootstrap,
                         (self._V, self._names, self._saved_state()))

    def _saved_state(self):
        """The state (raw data) saved by :meth:`save`.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); B.fix({'x0': 3}); st = B._saved_state()
            sage: st['pairs'], st['variables'], st['subs'], st['eqs'], st['done']
            ([('J', 'J')], ['x0', 'x1'], {'x0': 3, 'x1': 0}, [], [])
        """
        st = {
            'kind': 'OPEBootstrap',
            'format': STATE_FORMAT,
            'fingerprint': self._fingerprint,
            'weights': list(self._h),
            'grading': self._gkey,
            'pairs': [self._pair_names(p) for p in self._pairs],
            'variables': list(self._R.variable_names()),
            'terms': [self._terms[x] for x in self._unknowns],
            'subs': {str(x): v for x, v in self._subs.items()},
            'eqs': sorted(self._eqs),
            'done': [tuple(self._gens[i] for i in t) for t in self._done],
        }
        return st

    def _restore(self, st):
        """Continue from a saved state (see :meth:`save`).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); B.fix({'x0': 3})
            sage: B2 = OPEBootstrap(V); B2._restore(B._saved_state()); B2.ansatz()
            {('J', 'J'): [0, 3]}
            sage: st = B._saved_state(); st['weights'] = [2]
            sage: B2._restore(st)
            Traceback (most recent call last):
            ...
            ValueError: the saved bootstrap state was saved for other conformal weights or another grading
        """
        what = 'the saved bootstrap state'
        if st['fingerprint'] != self._fingerprint:
            raise ValueError('{} was saved for a different vertex algebra '
                             '(base ring, generators, parities or declared '
                             'OPEs)'.format(what))
        if st['weights'] != list(self._h) or st['grading'] != self._gkey:
            raise ValueError('{} was saved for other conformal weights or '
                             'another grading'.format(what))
        if (st['pairs'] != [self._pair_names(p) for p in self._pairs]
                or st['variables'] != list(self._R.variable_names())
                or st['terms'] != [self._terms[x] for x in self._unknowns]):
            raise ValueError('{} was saved for another ansatz (other pairs or '
                             'names of the unknowns)'.format(what))
        R, idx = self._R, self._V._idx
        self._subs = {R(x): R(v) for x, v in st['subs'].items()}
        self._eqs = {R(f) for f in st['eqs']}
        self._done = [tuple(idx[a] for a in t) for t in st['done']]
        self._update_algebra()

    # ------------------------------------------------------------------ #
    #  Solving                                                            #
    # ------------------------------------------------------------------ #

    def ideal(self):
        """
        The ideal of the remaining equations in :meth:`ring`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: B.ideal().gens()
            [c^3 - 268*c^2 - 44*c]
            sage: len(B.ideal().minimal_associated_primes())     # see solve()
            2
        """
        return self._R.ideal(sorted(self._eqs) or [self._R.zero()])

    def solve(self):
        r"""
        The irreducible components of the solutions of the remaining
        equations (minimal associated primes), as
        :class:`BootstrapSolution` objects (empty if the equations are
        inconsistent).

        Only the Jacobi identities imposed so far are taken into account;
        impose all triples (:meth:`impose_jacobi`) for vertex algebras.

        OUTPUT: a list of :class:`BootstrapSolution` objects

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: S, S0 = B.solve()
            sage: S, S0
            (Bootstrap solution 1 of 2: no free unknowns, c^2 - 268*c - 44 = 0,
             Bootstrap solution 2 of 2: no free unknowns, c = 0)

        Before any Jacobi identity, the ansatz is one component::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.solve()
            [Bootstrap solution 1 of 1: free [c, x0, x1, x4, x5, x6]]
        """
        if self._R.one() in self._eqs:
            return []
        if not self._eqs:
            primes = [self.ideal()]
        else:
            primes = self.ideal().minimal_associated_primes()
        n = len(primes)
        return [BootstrapSolution(self, P, k + 1, n)
                for k, P in enumerate(primes)]


def _extension(E, q, name):
    """The field ``E[t]/(q)`` (``q`` monic irreducible over ``E``), a
    number field when ``E`` is ``QQ`` or a number field.

    TESTS::

        sage: from vertex_algebras.bootstrap import _extension
        sage: _extension(QQ, QQ['s']('s^2 - 2'), 'r')
        Number Field in r with defining polynomial s^2 - 2
        sage: F = Frac(QQ['c'])
        sage: _extension(F, F['s']('s^2 - c'), 'r')
        Univariate Quotient Polynomial Ring in r over Fraction Field of Univariate Polynomial Ring in c over Rational Field with modulus s^2 - c
    """
    from sage.categories.number_fields import NumberFields
    if E in NumberFields():
        return E.extension(q.change_ring(E), name)
    return q.parent().quotient(q, name)


class BootstrapSolution(SageObject):
    r"""
    An irreducible component of the solutions of an :class:`OPEBootstrap`.

    The remaining variables split into ``free_variables()`` ones (a transcendence
    basis of the component, preferring parameters and earlier unknowns) and
    the others, which take ``values()`` in the field ``L`` of rational
    functions of the free ones, or in an extension ``L[x]/(p(x))`` when one
    of them is algebraic of degree ``> 1`` over ``L`` (``extension()``; a
    number field over ``QQ``).  Components needing more than one algebraic
    variable only give ``prime_ideal()``.

    Returned by :meth:`OPEBootstrap.solve`.

    INPUT:

    - ``B`` -- the :class:`OPEBootstrap`

    - ``P`` -- the prime ideal of the component, in ``B.ring()``

    - ``k``, ``n`` -- integers: this is component ``k`` of ``n`` (for the
      representation)

    EXAMPLES:

    The ``W(2, 4)`` algebra: the self-coupling of the weight-4 primary
    ``W`` (normalised to ``W_(7) W = c/4``) is the square root of a
    rational function of ``c`` (Kausch–Watts; Blumenhagen et al.)::

        sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
        sage: from vertex_algebras.bootstrap import OPEBootstrap
        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: K.<c> = Frac(QQ['c'])
        sage: N.<L, W> = NormallyOrderedExpressions(K)
        sage: V = UniversalVertexAlgebra(K, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
        ....:                                   ('L', 'W'): [W.derivative(), 4*W]},
        ....:                            degrees={'L': 2, 'W': 4})
        sage: B = OPEBootstrap(V); B.fix({'x0': c/4}); B.impose_jacobi()
        sage: B.unknowns(), B.describe('x5')
        ([x5], 'coefficient of W in W_(3) W')
        sage: [S] = B.solve()
        sage: p = S.extension().modulus(); p
        x5^2 + (-27/35*c^3 + 3996/35*c^2 + 106164/35*c - 18144/5)/(c^3 + 953/70*c^2 + 1249/35*c - 748/35)
        sage: -p(0) == 54*(c + 24)*(c^2 - 172*c + 196)/((5*c + 22)*(7*c + 68)*(2*c - 1))
        True
        sage: A = S.algebra(); A.base_ring()
        Univariate Quotient Polynomial Ring in x5 over Fraction Field of Univariate Polynomial Ring in c over Rational Field with modulus x5^2 + ...
        sage: A.satisfies_jacobi_identity()
        True

    At ``c = 1`` (over ``QQ``) the coupling lies in a number field::

        sage: N.<L, W> = NormallyOrderedExpressions(QQ)
        sage: V1 = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2],
        ....:                                     ('L', 'W'): [W.derivative(), 4*W]},
        ....:                             degrees={'L': 2, 'W': 4})
        sage: B = OPEBootstrap(V1); B.fix({'x0': 1/4}); B.impose_jacobi()
        sage: [S] = B.solve(); S
        Bootstrap solution 1 of 1: no free unknowns, x5^2 - 50/3 = 0
        sage: S.extension()
        Number Field in x5 with defining polynomial x5^2 - 50/3
        sage: S.algebra().satisfies_jacobi_identity()
        True
    """

    def __init__(self, B, P, k, n):
        r"""
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: N.<L, W> = NormallyOrderedExpressions(QQ)
            sage: V = UniversalVertexAlgebra(QQ, N, {('L', 'L'): [L.derivative(), 2*L, 0, 1/2],
            ....:                                    ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.fix({'x0': 1}); B.impose_jacobi(); [S] = B.solve(); S
            Bootstrap solution 1 of 1: no free unknowns, all determined
            sage: S._B is B, (S._k, S._n), S.free_variables(), S._L
            (True, (1, 1), [], Rational Field)
        """
        self._B, self._P, self._k, self._n = B, P, k, n
        R, K = B._R, B._K
        rem = [g for g in R.gens() if g not in B._subs]
        self._free, self._values, self._alg = [], {}, None
        self._L = self._E = K
        self._Pf = None
        if not rem:
            return
        # lex order (largest first): eliminate later unknowns first,
        # parameters last; if a variable is algebraic over the free ones,
        # retry with it made smallest among the unknowns (x^2 = a y may be
        # parametrised by x)
        order = sorted(rem, key=B._priority, reverse=True)
        params = set(B._params)
        best, moved = None, set()
        while True:
            res = self._parametrise(order)
            if best is None or res['score'] < best['score']:
                best = res
            v = res['alg'] if res['values'] is not None else res['stuck']
            if v is None or v in moved or v in params:
                break
            moved.add(v)
            order.remove(v)
            pos = next((t for t, w in enumerate(order) if w in params),
                       len(order))
            order.insert(pos, v)
        self._free, self._values, self._alg = (best['free'], best['values'],
                                               best['alg'])
        self._L, self._E, self._Pf = best['L'], best['E'], best['Pf']

    def _parametrise(self, order):
        """Free variables and values for the lex order ``order`` (largest
        first), allowing one algebraic variable: a dict with ``'free'``,
        ``'values'`` (``None`` if not parametrised), ``'alg'`` (the
        algebraic variable), ``'stuck'``, the fields ``'L'`` (rational
        functions of the free variables), ``'E'`` (``L`` or its algebraic
        extension), ``'Pf'`` and ``'score'`` (smaller is better).

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); x0 = B.ring()('x0')
            sage: B._add_polys([x0^2 - 2]); [S] = B.solve()
            sage: res = S._parametrise([x0])
            sage: res['free'], res['values'], res['alg'], res['score']
            ([], {x0: x0}, x0, (False, True))
            sage: res['E']
            Number Field in x0 with defining polynomial x0^2 - 2
        """
        B, P = self._B, self._P
        R, K = B._R, B._K
        Rl = PolynomialRing(K, len(order), [str(v) for v in order],
                            order='lex')
        to_l = R.hom([Rl(str(g)) if g in order else Rl.zero()
                      for g in R.gens()])
        G = list(Rl.ideal([to_l(f) for f in P.gens()]).groebner_basis()) \
            if not P.is_zero() else []
        lead = {}
        for g in G:
            v = min(g.lm().variables(), key=Rl.gens().index)
            lead.setdefault(v, []).append(g)
        free = [v for v in Rl.gens() if v not in lead]
        out = {'free': sorted((R(str(v)) for v in free), key=B._priority),
               'alg': None, 'stuck': None, 'L': K, 'Pf': None}
        if free:
            out['Pf'] = PolynomialRing(K, [str(v) for v in out['free']])
            out['L'] = out['Pf'].fraction_field()
            point = {v: out['L'](out['Pf'](str(v))) for v in free}
        else:
            point = {}
        E = out['L']
        values = {}
        for v in reversed(Rl.gens()):              # smallest variables first
            if v in point:
                continue
            args = [E(point[w]) if w in point else E.zero()
                    for w in Rl.gens()]
            found = None
            for g in sorted(lead[v], key=lambda g: g.degree(v)):
                if g.degree(v) == 1:
                    uv = g.coefficient({v: 1})(*args)
                    if uv:
                        found = -g.subs({v: 0})(*args) / uv
                        break
            if found is None and out['alg'] is None:
                g = min(lead[v], key=lambda g: g.degree(v))
                T = PolynomialRing(E, str(v))
                q = g(*[T.gen() if w == v else T(a)
                        for w, a in zip(Rl.gens(), args)])
                if q.degree() > 1:
                    E = _extension(E, q.monic(), str(v))
                    found = E.gen()
                    out['alg'] = R(str(v))
            if found is None:
                values = None
                out['stuck'] = R(str(v))
                break
            point[v] = found
            values[R(str(v))] = found
        if values is not None:
            args = [E(point[w]) for w in Rl.gens()]
            if any(g(*args) for g in G):
                values = None
            else:
                values = {x: E(val) for x, val in values.items()}
        out['E'], out['values'] = E, values
        out['score'] = (values is None, out['alg'] is not None)
        return out

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: V = UniversalVertexAlgebra(QQ, ['J'], degrees={'J': 1})
            sage: B = OPEBootstrap(V); x0 = B.ring()('x0')
            sage: B.solve()
            [Bootstrap solution 1 of 1: free [x0]]
            sage: B._add_polys([x0^2 - 2]); B.solve()
            [Bootstrap solution 1 of 1: no free unknowns, x0^2 - 2 = 0]
            sage: B.fix({'x0': 1}); B.solve()
            []
        """
        s = 'Bootstrap solution {} of {}: '.format(self._k, self._n)
        parts = ['free {}'.format(self._free) if self._free
                 else 'no free unknowns']
        if self._values is None:
            parts.append('not parametrised (more than one algebraic '
                         'variable); prime ideal generated by {}'.format(
                             self._P.gens()))
        else:
            for v, x in sorted(self._values.items(),
                               key=lambda t: self._B._priority(t[0])):
                if v == self._alg:
                    parts.append('{} = 0'.format(self._minpoly()))
                else:
                    parts.append('{} = {}'.format(v, x))
            if not self._values and not self._free:
                parts.append('all determined')
        return s + ', '.join(parts)

    def _minpoly(self):
        """
        The defining polynomial of the algebraic extension (number field or
        quotient ring) of a solution with an algebraic variable.

        TESTS::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: K.<c> = Frac(QQ['c'])
            sage: for F, a in [(QQ, 2), (K, c)]:
            ....:     B = OPEBootstrap(UniversalVertexAlgebra(F, ['J'], degrees={'J': 1}))
            ....:     x0 = B.ring()('x0'); B._add_polys([x0^2 - a])
            ....:     [S] = B.solve(); print(S._minpoly())
            x0^2 - 2
            x0^2 - c
        """
        E = self._E
        try:
            return E.relative_polynomial()
        except AttributeError:
            return E.modulus()

    def prime_ideal(self):
        """
        The prime ideal of the component (in the bootstrap's ring).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: S, S0 = B.solve()
            sage: S.prime_ideal().gens(), S0.prime_ideal().gens()
            ([c^2 - 268*c - 44], [c])
        """
        return self._P

    def free_variables(self):
        """
        The free variables of the component (remaining unknowns and
        parameters): a transcendence basis.

        OUTPUT: a list of generators of the bootstrap's ring

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi(); [S] = B.solve(); S.free_variables()
            [c, x4]
        """
        return list(self._free)

    def values(self):
        """
        A dict: determined variable -> value in :meth:`extension`
        (``None`` if the component is not parametrised).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: S, S0 = B.solve()
            sage: S.values(), S0.values()
            ({c: c}, {c: 0})
            sage: S.values()[B.ring()('c')].parent() is S.extension()
            True
        """
        return None if self._values is None else dict(self._values)

    def extension(self):
        """
        The field of the values: the base field, the field of rational
        functions of the free variables, or an algebraic extension of it
        (generated by the algebraic variable, see :meth:`algebraic_variable`).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: S, S0 = B.solve()
            sage: S.extension(), S0.extension()
            (Number Field in c with defining polynomial c^2 - 268*c - 44, Rational Field)
        """
        return self._E

    def algebraic_variable(self):
        """
        The variable generating :meth:`extension` over the rational
        functions of the free ones (``None`` if there is none).

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, X> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'X'): [X.derivative(), 5*X]},
            ....:                            degrees={'L': 2, 'X': 5})
            sage: B = OPEBootstrap(V); B.fix({'x0': c/5}); B.impose_jacobi()
            sage: S, S0 = B.solve()
            sage: S.algebraic_variable(), S0.algebraic_variable()
            (c, None)
        """
        return self._alg

    def algebra(self):
        r"""
        The universal vertex algebra of this component, with the conformal
        weights (and grading) of the original algebra.  Its base ring is
        the base field, the polynomial ring of the free variables (when the
        values are polynomial in them), or :meth:`extension`.  The same
        algebra on each call (unique representation).  A vertex algebra if
        the Jacobi identities of all ordered generator triples were
        imposed, else a proto vertex algebra (``proto=True``); the
        bootstrap's own :meth:`OPEBootstrap.algebra` is always proto.

        OUTPUT: a :class:`universal_vertex_algebra.UniversalVertexAlgebra`

        EXAMPLES:

        The ``W_3`` family, polynomial in ``c`` and the ``:LL:`` coefficient
        ``x4``::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi(); [S] = B.solve()
            sage: A = S.algebra(); A.base_ring()
            Multivariate Polynomial Ring in c, x4 over Rational Field
            sage: A.satisfies_jacobi_identity(), A is S.algebra()
            (True, True)

        TESTS:

        After only some triples the solutions are proto vertex algebras,
        like the bootstrap's own algebra; after all triples they are vertex
        algebras and pass their category tests::

            sage: from vertex_algebras.vertex_algebra_categories import VertexAlgebras
            sage: Bp = OPEBootstrap(V); Bp.impose_jacobi([('L', 'W', 'W')])
            sage: [U in VertexAlgebras(U.base_ring()) for U in [Bp.algebra()] + [T.algebra() for T in Bp.solve()]]
            [False, False]
            sage: A in VertexAlgebras(A.base_ring())
            True
            sage: TestSuite(A).run()
        """
        from .universal_vertex_algebra import UniversalVertexAlgebra
        B = self._B
        if self._values is None:
            raise NotImplementedError(
                'this component needs more than one algebraic variable; '
                'substitute a solution by hand (see prime())')
        R, E = B._R, self._E
        S = E
        if self._free and self._alg is None and all(
                v.denominator().is_constant() for v in self._values.values()):
            S = self._Pf
        free = {str(v) for v in self._free}
        ims = []
        for g in R.gens():
            if str(g) in free:
                ims.append(E(self._Pf(str(g))))
            elif g in self._values:
                ims.append(self._values[g])
            else:
                ims.append(E.zero())          # eliminated: never occurs
        phi = R.hom(ims, E)

        def conv(x):
            x = phi(x)
            if S is E:
                return x
            return S(x.numerator()) / x.denominator().constant_coefficient()

        V, U = B._V, B._U
        F = NormallyOrderedExpressions(S, B._gens, V.parities())
        data = {}
        for (i, j), poles in U._opes.items():
            data[(B._gens[i], B._gens[j])] = [
                F._from_dict({m: conv(c) for m, c in
                              p._monomial_coefficients.items()},
                             remove_zeros=True)
                for p in poles]
        gd = V._grading
        horizontal_grading = group = None
        if gd is not None:
            horizontal_grading = {s: gd['group'].element(gd['g'][i])
                       for i, s in enumerate(B._gens)}
            group = gd['group'].group
        r = len(B._gens)
        done = set(B._done)
        complete = all((i, j, k) in done for i in range(r) for j in range(r)
                       for k in range(r))
        A = UniversalVertexAlgebra(S, B._gens, data, parities=V.parities(),
                                   degrees=dict(zip(B._gens, B._h)),
                                   horizontal_grading=horizontal_grading, horizontal_grading_group=group,
                                   implementation=_implementation_of(V),
                                   proto=not complete)
        return A

    def opes(self):
        """
        The OPEs of the ansatz pairs in :meth:`algebra` (in the
        orientation of the pairs).

        OUTPUT:

        A dictionary ``{(a, b): [a_(0) b, a_(1) b, ...]}`` keyed by pairs of
        generator names, the poles elements of :meth:`algebra`.

        EXAMPLES::

            sage: from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: from vertex_algebras.bootstrap import OPEBootstrap
            sage: P.<c> = QQ['c']
            sage: N.<L, W> = NormallyOrderedExpressions(P)
            sage: V = UniversalVertexAlgebra(P, N, {('L', 'L'): [L.derivative(), 2*L, 0, c/2],
            ....:                                   ('L', 'W'): [W.derivative(), 3*W]},
            ....:                            degrees={'L': 2, 'W': 3})
            sage: B = OPEBootstrap(V)
            sage: B.impose_jacobi(); [S] = B.solve()
            sage: S.opes()[('W', 'W')][1:]
            [x4*:L L: + (3/64*c*x4 - 3/32*x4)*d^2L,
             (5/32*c*x4 + 11/16*x4)*dL,
             (5/16*c*x4 + 11/8*x4)*L,
             0,
             5/96*c^2*x4 + 11/48*c*x4]
        """
        A = self.algebra()
        return {self._B._pair_names(p): A.gen(p[0]).ope(A.gen(p[1]))
                for p in self._B._pairs}
