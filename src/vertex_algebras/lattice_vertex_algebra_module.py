# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Modules over lattice vertex algebras.

For a lattice vertex algebra ``V_L`` (``L`` integral, non-degenerate, of
Gram matrix ``B``) and ``λ`` in the dual lattice ``L° = {μ : B(μ, L) ⊂ ZZ}``,
the module ``V_{λ+L}`` is the Fock space of the Heisenberg algebra
tensored with the span of the ``e^p``, ``p ∈ λ + L``, with

    Y(e^q, z) e^p = ε(q, p - λ) z^{B(q,p)} E^-(q, z) E^+(q, z) e^{q+p},

``h_i(0) e^p = B(x_i, p) e^p``, and the cocycle ``ε`` of ``V_L``
(``p - λ ∈ L``; any representative ``λ`` of the coset gives an isomorphic
module).  These are the simple modules of ``V_L``, one for each coset of
``L°/L`` (``V.simple_module(λ)``, ``V.simple_modules()``); ``V_{0+L}`` is
``V_L`` itself.  The conformal weights are those of ``V``: ``Δ(p) =
B(p,p)/2 + β·p`` for the weight data of ``V`` (a background charge
included).  For an odd lattice the module is super, ``e^{λ+q}`` of parity
``B(q, q)``.  Basis keys ``(letters, p)`` as for ``V_L``, ``p`` a tuple of
rationals in the coordinates of the basis of ``L``.

The action is computed by the Python kernel of
:class:`lattice_vertex_algebra.LatticeVertexAlgebra` (letter series and
``E^-``), which uses ``p`` only through ``B(q, p)`` and ``B(x_i, p)``
(integers for ``p ∈ L°``).

EXAMPLES:

The ``sl_2`` lattice ``A_1`` (``B = (2)``): two simple modules, ``V_L`` and
``V_{1/2 + L}`` of lowest weight ``1/4`` with the two vectors ``e^{±1/2}``
(the doublet of ``sl_2``), characters ``θ_{λ+L}/η``::

    sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
    sage: L0 = LatticeVertexAlgebra(QQ, matrix([[2]])); J = L0.J(0)
    sage: V = L0.with_conformal_vector(J.no(J) / 4)
    sage: M0, M1 = V.simple_modules(); M1
    Module V_(λ+L), λ = (1/2), over Lattice vertex algebra of rank 1 with Gram matrix [2] over Rational Field
    sage: M1.weight_space(1/4).basis()
    [e^(-1/2), e^(1/2)]
    sage: M1.character(5)
    2*q^(1/4) + 2*q^(5/4) + 6*q^(9/4) + 8*q^(13/4) + 14*q^(17/4) + O(q^5)
    sage: v = M1.monomial(((), (1/2,))); V.e((-1,)).ope(v)
    [e^(-1/2)]
    sage: J.ope(v), V.e((1,)).ope(v)
    ([e^(1/2)], [])
    sage: TestSuite(M1).run()

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.arith.functions import lcm
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.free_module_element import vector
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .lattice_vertex_algebra import _degree, _is_int_like, _letter_key
from .vertex_algebra_module import VertexAlgebraModule


def _canonical_coset(V, lam):
    """The representative of ``λ + L`` with coordinates in ``[0, 1)``, as a
    tuple of rationals; a ``ValueError`` if ``λ`` is not in the dual
    lattice.

    TESTS::

        sage: import vertex_algebras as va
        sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
        sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
        sage: from vertex_algebras.lattice_vertex_algebra_module import _canonical_coset
        sage: _canonical_coset(V, [-3/2]), _canonical_coset(V, (2,))
        ((1/2,), (0,))
        sage: _canonical_coset(V, [1/3])
        Traceback (most recent call last):
        ...
        ValueError: (1/3,) is not in the dual lattice (B λ must be integral)
    """
    n = V.rank()
    try:
        lam = vector(QQ, list(lam))
    except (TypeError, ValueError):
        raise ValueError('{!r} is not a vector of rationals'.format(lam))
    if len(lam) != n:
        raise ValueError('λ must have {} coordinates (in the basis of the lattice)'
                         .format(n))
    if any(x not in ZZ for x in V.gram_matrix().change_ring(QQ) * lam):
        raise ValueError('{} is not in the dual lattice (B λ must be integral)'
                         .format(tuple(lam)))
    return tuple(x - x.floor() for x in lam)


def _dual_lattice_cosets(V):
    r"""
    Representatives ``λ`` (coordinates in ``[0, 1)``) of the cosets ``L°/L``
    of the lattice of ``V``: ``0`` first, the others sorted.

    EXAMPLES::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: from vertex_algebras.lattice_vertex_algebra_module import _dual_lattice_cosets
        sage: _dual_lattice_cosets(LatticeVertexAlgebra(QQ, matrix([[2, -1], [-1, 2]])))
        [(0, 0), (1/3, 2/3), (2/3, 1/3)]
        sage: len(_dual_lattice_cosets(LatticeVertexAlgebra(QQ, matrix([[0, 3], [3, 0]]))))
        9
    """
    from itertools import product
    B = V.gram_matrix().change_ring(ZZ)
    n = B.nrows()
    D, U, _ = B.smith_form()                # D = U B W: Z^n / B Z^n ≅ ⊕ Z/D_ii
    Ui = U.inverse()
    Bi = B.change_ring(QQ).inverse()
    out = {_canonical_coset(V, Bi * (Ui * vector(ZZ, k)))
           for k in product(*[range(abs(D[i, i])) for i in range(n)])}
    zero = tuple(QQ.zero() for _ in range(n))
    return [zero] + sorted(out - {zero})


def _module_grading(V):
    r"""
    The grading of the modules ``V_{λ+L}`` by the grading of ``V``
    (``None`` if ``V`` is not graded): ``(B, ι, grade_of_p, q_grade)``.
    The grading of ``V`` is a homomorphism ``γ: L -> A`` (``J_i`` of grade 0);
    the grades of all modules lie in the pushout ``B = (A ⊕ L°) / {(γ(q),
    -q) : q ∈ L}``, with ``ι(a) = [(a, 0)]`` (injective, as ``L -> L°`` is)
    and ``horizontal_grade(e^p) = [(0, p)]``.  For the grading ``'lattice'`` (``γ`` the
    identity) this is ``L°`` itself, realised in ``QQ^n`` so that ``e^p`` has
    grade ``p``.  ``q_grade(G, λ)`` is the grade (in ``A``) of the ``q ∈ L``
    with ``horizontal_grade(e^{λ+q}) = G``, ``None`` if there are none.  Cached on
    ``V``: all modules share ``B``.

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
        sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
        sage: from vertex_algebras.lattice_vertex_algebra_module import _module_grading
        sage: _module_grading(V) is None
        True
        sage: Vg = V.with_horizontal_grading('lattice'); B, iota, grade_of_p, q_grade = _module_grading(Vg)
        sage: _module_grading(Vg)[0] is B
        True
        sage: grade_of_p((1/2,)), grade_of_p((3/2,)), q_grade(grade_of_p((3/2,)), (1/2,))
        ((1,), (3,), (1,))
    """
    gd = V._grading
    if gd is None:
        return None
    data = getattr(V, '_lattice_module_grading', None)
    if data is not None:
        return data
    from sage.matrix.constructor import matrix

    from ._grading_groups import GradingEmbedding, GradingGroup
    n = V.rank()
    Bg = V.gram_matrix().change_ring(QQ)
    Bi = Bg.inverse()
    A, Gamma = gd['group'], gd['Gamma']
    if A.kind in ('ZZ', 'free') and A.k == n and Gamma == 1:
        Lo = (QQ ** n).span(Bi.rows(), ZZ)
        BG = GradingGroup(Lo)
        cols = [BG.coords(vector(QQ, [int(i == j) for i in range(n)])) for j in range(n)]
        iota = GradingEmbedding(A, BG, matrix(ZZ, cols).transpose())

        def grade_of_p(p):
            return BG.coords(vector(QQ, p))

        def q_grade(G, lam):
            q = [x - y for x, y in zip(BG.element(G), lam)]
            if any(x not in ZZ for x in q):
                return None
            return A.reduce(tuple(ZZ(x) for x in q))
    else:
        k = A.k
        C = ZZ ** (k + n)
        rels = []
        for j, nj in enumerate(A.moduli):
            if nj:
                rels.append(C([nj * int(i == j) for i in range(k)] + [0] * n))
        for i in range(n):
            rels.append(C(list(Gamma.column(i)) + [-x for x in Bg.column(i)]))
        Bgrp = C / C.span(rels)
        BG = GradingGroup(Bgrp)
        cols = [BG.coords(Bgrp(C([int(i == j) for i in range(k)] + [0] * n)))
                for j in range(k)]
        iota = GradingEmbedding(A, BG, matrix(ZZ, k, BG.k, cols).transpose())

        def grade_of_p(p):
            y = Bg * vector(QQ, p)
            return BG.coords(Bgrp(C([0] * k + [ZZ(x) for x in y])))

        def q_grade(G, lam):
            d = (BG.element(G) - BG.element(grade_of_p(lam))).lift()
            x, y = vector(ZZ, list(d)[:k]), vector(QQ, list(d)[k:])
            s = Bi * y
            if any(c not in ZZ for c in s):
                return None
            return A.reduce(tuple(x + Gamma * s.change_ring(ZZ)))
    data = (BG, iota, grade_of_p, q_grade)
    V._lattice_module_grading = data
    return data


class _LatticeModuleKeys:
    """The basis keys ``(letters, p)`` of ``V_{λ+L}``: Heisenberg letters as
    for ``V_L`` and ``p ∈ λ + L``.

    TESTS::

        sage: import vertex_algebras as va
        sage: M = va.Lattice(QQ, matrix([[2]])).simple_module([1/2])
        sage: from vertex_algebras.lattice_vertex_algebra_module import _LatticeModuleKeys
        sage: K = _LatticeModuleKeys((1/2,)); K is not M._indices and ((), (1/2,)) in M._indices
        True
    """

    def __init__(self, lam):
        """
        TESTS::

            sage: from vertex_algebras.lattice_vertex_algebra_module import _LatticeModuleKeys
            sage: _LatticeModuleKeys((1/2,))._lam
            (1/2,)
        """
        self._lam = lam

    def __contains__(self, x):
        """
        Whether ``x`` is a key: letters in decreasing letter order, ``p``
        in ``λ + L``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: K = M._indices; ((), (5/2,)) in K, ((), (1,)) in K, 'e' in K
            (True, False, False)
            sage: (((0, 1), (0, 0)), (1/2,)) in K, (((0, 0), (0, 1)), (1/2,)) in K
            (True, False)
        """
        if not (isinstance(x, tuple) and len(x) == 2):
            return False
        letters, p = x
        n = len(self._lam)
        if not (isinstance(letters, tuple) and isinstance(p, tuple) and len(p) == n):
            return False
        try:
            if any(QQ(a) - b not in ZZ for a, b in zip(p, self._lam)):
                return False
        except (TypeError, ValueError):
            return False
        prev = None
        for t in letters:
            if not (isinstance(t, tuple) and len(t) == 2 and _is_int_like(t[0])
                    and _is_int_like(t[1]) and 0 <= t[0] < n and t[1] >= 0):
                return False
            if prev is not None and _letter_key(prev) < _letter_key(t):
                return False
            prev = t
        return True

    def an_element(self):
        """
        The key of ``e^λ``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._indices.an_element()
            ((), (1/2,))
        """
        return ((), self._lam)

    def cardinality(self):
        """
        Infinite.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._indices.cardinality()
            +Infinity
        """
        return Infinity


class LatticeVertexAlgebraModule(VertexAlgebraModule):
    r"""
    The module ``V_{λ+L}`` over a lattice vertex algebra ``V = V_L``, ``λ``
    in the dual lattice (``V.simple_module(λ)``); see the module
    documentation.

    INPUT:

    - ``V`` -- a :class:`lattice_vertex_algebra.LatticeVertexAlgebra`
    - ``lam`` -- ``λ``, coordinates in the basis of ``L`` (rationals with
      ``B λ`` integral); replaced by the representative of ``λ + L`` with
      coordinates in ``[0, 1)``

    EXAMPLES:

    An odd lattice (``B = (3)``: a superalgebra, the ``N = 2`` algebra at
    ``c = 1``) and its module ``V_{1/3+L}`` (weights ``Δ(p) = 3p^2/2``:
    ``e^{1/3}`` of weight ``1/6``, the odd ``e^{-2/3}`` of weight ``2/3``)::

        sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
        sage: L0 = LatticeVertexAlgebra(QQ, matrix([[3]])); J = L0.J(0)
        sage: V = L0.with_conformal_vector(J.no(J) / 6)
        sage: M = V.simple_module([1/3]); M.category()
        Category of conformally graded super modules with basis over Lattice vertex algebra of rank 1 with Gram matrix [3] over Rational Field
        sage: M.weight_space(1/6).basis(), M.weight_space(2/3).basis()
        ([e^(1/3)], [e^(-2/3)])
        sage: M.monomial(((), (-2/3,))).parity(), M.highest_weight_vector().parity()
        (1, 0)
        sage: V.simple_module([4/3]) is M, M.coset_representative()
        (True, (1/3,))
        sage: TestSuite(M).run()

    An indefinite lattice: the weight spaces are infinite, those of a grade
    finite.  With the lattice grading of ``V`` the grades of the modules lie
    in the dual lattice, ``e^p`` of grade ``p``::

        sage: H = LatticeVertexAlgebra(QQ, matrix([[0, 2], [2, 0]]))
        sage: J0, J1 = H.heisenberg_currents()
        sage: Hg = H.with_conformal_vector((J0.no(J1) + J1.no(J0)) / 4).with_horizontal_grading('lattice')
        sage: Mh = Hg.simple_module([1/2, 0]); Mh.weight_space(1).is_finite_dimensional()
        False
        sage: [Mh.weight_space(w, horizontal_grade=(1/2, 0)).dimension() for w in range(4)]
        [1, 2, 5, 10]
        sage: Mh.horizontal_grade(Mh.monomial(((), (1/2, 1)))), Mh.weight_space(1, horizontal_grade=(0, 0)).dimension()
        ((1/2, 1), 0)
        sage: len(Hg.simple_modules()), Mh.horizontal_grading_group() is Hg.simple_module([0, 1/2]).horizontal_grading_group()
        (4, True)

    Another grading of ``V`` (here by ``Z/2``, ``e^q`` of grade ``q_0 + q_1``):
    the grading group of the modules is the pushout of ``A ← L → L°``, of
    order ``|A| |L°/L| = 8`` (``2 e^{(1/2, 0)}`` has grade ``ι(1) ≠ 0``)::

        sage: H2 = Hg.with_horizontal_grading([1, 1], group=Zmod(2)); M2 = H2.simple_module([1/2, 0])
        sage: M2.horizontal_grading_group()
        Finitely generated module V/W over Integer Ring with invariants (2, 4)
        sage: [M2.horizontal_grade(M2.monomial(((), p))) for p in [(1/2, 0), (1/2, 1), (3/2, 1)]]
        [(0, 1), (0, 3), (0, 1)]
        sage: TestSuite(M2).run()

    TESTS::

        sage: V.simple_module([1/2])
        Traceback (most recent call last):
        ...
        ValueError: (1/2,) is not in the dual lattice (B λ must be integral)
        sage: loads(dumps(M)) is M
        True
        sage: V.simple_module([1/3, 0])
        Traceback (most recent call last):
        ...
        ValueError: λ must have 1 coordinates (in the basis of the lattice)
        sage: V.simple_module(['x'])
        Traceback (most recent call last):
        ...
        ValueError: ['x'] is not a vector of rationals

    ``|L°/L| = |det B|`` simple modules, each passing ``TestSuite`` (the
    module axioms), for ``A_1`` with a background charge, ``A_2``, the odd
    ``diag(1, 3)``, ``Z√6``; the indefinite ``[[1, 1], [1, -1]]`` has 2::

        sage: def sugawara(B, alpha=None):
        ....:     L = LatticeVertexAlgebra(QQ, matrix(B)); Js = L.heisenberg_currents()
        ....:     Bi = matrix(QQ, B).inverse(); n = len(Js)
        ....:     T = sum(Bi[i, j] * Js[i].no(Js[j]) for i in range(n) for j in range(n)) / 2
        ....:     if alpha:
        ....:         T += sum(a*J for a, J in zip(alpha, Js)).derivative()
        ....:     return L.with_conformal_vector(T)
        sage: lats = [sugawara([[2]], [1/2]), sugawara([[2, -1], [-1, 2]]),
        ....:         sugawara([[1, 0], [0, 3]]), sugawara([[6]])]
        sage: [len(W.simple_modules()) for W in lats]
        [2, 3, 3, 6]
        sage: for W in lats:
        ....:     for N in W.simple_modules()[:3]:
        ....:         TestSuite(N).run()
        sage: len(LatticeVertexAlgebra(QQ, matrix([[1, 1], [1, -1]])).simple_modules())
        2

    The characters are ``θ_{λ+L}/η^{rank}`` (brute force over the coset,
    also with a background charge)::

        sage: from itertools import product
        sage: q = PuiseuxSeriesRing(ZZ, 'q').gen()
        sage: def theta_over_eta(W, N, prec):
        ....:     lam, r, data = N.coset_representative(), W.rank(), W._weight_data()
        ....:     ws = [W._Delta(tuple(l + x for l, x in zip(lam, m)), data)
        ....:           for m in product(range(-6, 7), repeat=r)]
        ....:     return (sum(q^w for w in ws if w < prec)
        ....:             / prod((1 - q^n)^r for n in range(1, prec + 2))).add_bigoh(prec)
        sage: [N.character(4) == theta_over_eta(W, N, 4) for W in lats[:2] for N in W.simple_modules()]
        [True, True, True, True, True]

    ``V_{0+L}`` acts as ``V`` on itself, and the lowest weight spaces are the
    top levels of :class:`zhu_algebra.LatticeZhuAlgebra`::

        sage: def as_V(W):
        ....:     N = W.simple_modules()[0]
        ....:     keys = [key for w in (0, 1, 2) for key in W.weight_space(w).basis_keys()]
        ....:     return all({(m[0], tuple(ZZ(x) for x in m[1])): c for m, c in N._action_mono(
        ....:                     a, n, (b[0], tuple(QQ(x) for x in b[1]))).items()}
        ....:                == W._nprod_mono(a, n, b) for a in keys for b in keys for n in range(-3, 3))
        sage: as_V(lats[0]), as_V(lats[1])
        (True, True)
        sage: from vertex_algebras.zhu_algebra import LatticeZhuAlgebra
        sage: all(N.highest_weight_vector().degree() == h
        ....:     and sorted(m[1] for m in N.weight_space(h).basis_keys()) == sorted(mus)
        ....:     for W in lats[:2]
        ....:     for N, (h, mus) in zip(W.simple_modules(), LatticeZhuAlgebra(W).top_levels()))
        True

    Gradings: with the lattice grading of ``A_1`` the graded pieces of
    ``V_{λ+L}`` (``e^p`` of weight ``p^2``) have the dimensions of the
    partitions; direct sums share the grading group; without a grading of
    ``V`` the modules are not graded::

        sage: LA = LatticeVertexAlgebra(QQ, matrix([[2]])); J = LA.J(0); VA = LA.with_conformal_vector(J.no(J)/4)
        sage: Vg = VA.with_horizontal_grading('lattice'); N0, N1 = Vg.simple_modules()
        sage: all(N.weight_space(p^2 + d, horizontal_grade=(p,)).dimension() == Partitions(d).cardinality()
        ....:     for N in (N0, N1) for p in [N.coset_representative()[0] + j for j in range(-2, 3)]
        ....:     for d in range(5))
        True
        sage: D = N0.direct_sum(N1); D.horizontal_grading_group() is N0.horizontal_grading_group()
        True
        sage: D.weight_space(1/4, horizontal_grade=(1/2,)).dimension(), D.weight_space(1/4, horizontal_grade=(0,)).dimension()
        (1, 0)
        sage: TestSuite(D).run()
        sage: VA.simple_module([1/2]).horizontal_grading_group()
        Traceback (most recent call last):
        ...
        ValueError: Module V_(λ+L), λ = (1/2), ... is not graded by an abelian group

    A ``Z/3`` grading of ``A_2`` (``e^q`` of grade ``q_0 + 2 q_1``): the
    pushout has order ``|A| |L°/L| = 9``, shared by the three modules, and
    two vectors ``e^p``, ``e^{p'}`` of a module have the same grade iff ``p -
    p'`` lies in the kernel of the grading::

        sage: H3 = LatticeVertexAlgebra(QQ, matrix([[2, 1], [1, 2]])).with_horizontal_grading([1, 2], group=Zmod(3))
        sage: Ms = H3.simple_modules(); Bg = Ms[0].horizontal_grading_group(); Bg.cardinality()
        9
        sage: all(N.horizontal_grading_group() is Bg for N in Ms)
        True
        sage: lam = vector(QQ, Ms[1].coset_representative())
        sage: pts = [tuple(lam + vector(QQ, (a, b))) for a in range(-2, 3) for b in range(-2, 3)]
        sage: g = {p: Ms[1].horizontal_grade(Ms[1].monomial(((), p))) for p in pts}
        sage: all((g[p] == g[pp]) == ((p[0] - pp[0] + 2*(p[1] - pp[1])) % 3 == 0) for p in pts for pp in pts)
        True
        sage: TestSuite(Ms[1]).run()
    """

    @staticmethod
    def __classcall_private__(cls, V, lam):
        """
        Replace ``λ`` by its canonical representative (unique representation).

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: V.simple_module([-1/2]) is M, V.simple_module((5/2,)) is M
            (True, True)
        """
        return CombinatorialFreeModule.__classcall__(cls, V, _canonical_coset(V, lam))

    def __init__(self, V, lam):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([5/2])
            sage: M._lam, M.vertex_algebra() is V, M._ls_cache
            ((1/2,), True, {})
        """
        self._lam = lam
        self._ls_cache = {}
        g = _module_grading(V)
        if g is not None:
            self._mgrading = g[1]
            self._grade_of_p, self._q_grade = g[2], g[3]
        try:
            V._weight_data()
            graded = True
        except ValueError:
            graded = False
        VertexAlgebraModule.__init__(self, V, _LatticeModuleKeys(lam),
                                     super=V._has_odd(), graded=graded)

    def _repr_(self):
        """
        The string representation.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: repr(M)
            'Module V_(λ+L), λ = (1/2), over Lattice vertex algebra of rank 1 with Gram matrix [2] over Rational Field'
        """
        return 'Module V_(λ+L), λ = ({}), over {}'.format(
            ', '.join(str(x) for x in self._lam), self._V)

    def coset_representative(self):
        r"""
        The representative ``λ`` of the coset ``λ + L`` (coordinates in
        ``[0, 1)``, in the basis of ``L``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); L.simple_module([-1/2]).coset_representative()
            (1/2,)
            sage: A2 = va.Lattice(QQ, matrix([[2, -1], [-1, 2]]))
            sage: [M.coset_representative() for M in A2.simple_modules()]
            [(0, 0), (1/3, 2/3), (2/3, 1/3)]
        """
        return self._lam

    def _repr_term(self, m):
        """
        The string of the basis key ``m``: ``|0>``, ``e^(p)`` or ``:letters e^(p):``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._repr_term(((), (1/2,))), M._repr_term((((0, 0),), (1/2,)))
            ('e^(1/2)', ':J0 e^(1/2):')
            sage: V.simple_module([0])._repr_term(((), (0,)))
            '|0>'
        """
        letters, p = m
        if not letters and not any(p):
            return '|0>'
        V = self._V
        s = V._repr_term((letters, (0,) * len(p)))
        if not any(p):
            return s
        ep = 'e^({})'.format(','.join(str(v) for v in p))
        if not letters:
            return ep
        return ':{} {}:'.format(s.strip(':'), ep)

    def clear_caches(self):
        r"""
        Forget the computed actions and letter series (for memory; results
        do not change).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); M = L.simple_module([1/2])
            sage: v = M.highest_weight_vector(); x = L.e([1]).nproduct(v, -2); x
            e^(3/2)
            sage: M.clear_caches(); L.e([1]).nproduct(v, -2) == x
            True
        """
        VertexAlgebraModule.clear_caches(self)
        self._ls_cache.clear()

    # ------------------------------------------------------------------ #
    #  The action                                                          #
    # ------------------------------------------------------------------ #

    def _shift(self, p):
        """
        ``p - λ`` as an integer tuple.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._shift((5/2,)), M._shift((-1/2,))
            ((2,), (-1,))
        """
        return tuple(int(x - y) for x, y in zip(p, self._lam))

    def _nmax_key(self, a, m):
        """
        An upper bound of the ``n`` with ``a_(n) m != 0`` (``a`` a key of
        ``V``): the OPE starts at ``(z-w)^{B(q,p)}``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: a, m = ((), (1,)), ((), (-1/2,))
            sage: M._nmax_key(a, m), V.e((1,)).ope(M.monomial(m))
            (0, [e^(1/2)])
        """
        V = self._V
        return _degree(a[0]) + _degree(m[0]) - 1 - int(V._form(a[1], m[1]))

    def _action_key(self, a, n, m):
        """
        ``a_(n) m`` for a key ``a`` of ``V`` and a key ``m``, as a raw dict
        (letter series and ``E^-`` of ``V``'s Python kernel, cocycle
        ``ε(q, p - λ)``).

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._action_key(((), (1,)), -1, ((), (-1/2,)))
            {(((0, 0),), (1/2,)): 1}
            sage: M._action_key(((), (-1,)), 0, ((), (1/2,))), M._action_key(((), (1,)), 0, ((), (1/2,)))
            ({((), (-1/2,)): 1}, {})
        """
        V = self._V
        if n > self._nmax_key(a, m):
            return {}
        q, p = a[1], m[1]
        Bqp = int(V._form(q, p))
        W = V._letter_series_py(a, m, -n - 1 - Bqp, self._ls_cache)
        if not W:
            return {}
        return V._eminus_py(q, p, W, n, Bqp, V._eps(q, self._shift(p)))

    def _parity_key(self, m):
        """
        The parity of ``e^{λ+q}``: ``B(q, q)`` mod 2 (``e^λ`` even).

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._parity_key(((), (5/2,)))
            0
            sage: L3 = va.Lattice(QQ, matrix([[3]])).simple_module([1/3])
            sage: L3._parity_key(((), (1/3,))), L3._parity_key(((), (-2/3,)))
            (0, 1)
        """
        return int(self._V._parity_q(self._shift(m[1])))

    # ------------------------------------------------------------------ #
    #  Conformal weights and gradings                                      #
    # ------------------------------------------------------------------ #

    def _weight_key(self, m):
        """
        The conformal weight of a key: ``V``'s formula at rational ``p``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._weight_key(((), (3/2,))), M._weight_key((((0, 1),), (1/2,)))
            (9/4, 9/4)
        """
        V = self._V
        return V._weight_key(m, V._weight_data())

    def _lattice_part(self, G):
        """``V._lattice_part`` shifted by ``λ``: ``p = λ + q0 + K m`` and
        ``Δ(p) = m^T Q m + b·m + c``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: P = M._lattice_part(None); P['q0'], P['c'], P['dim']
            ((1/2), 1/4, 1)
            sage: Mg = V.with_horizontal_grading('lattice').simple_module([1/2])
            sage: P = Mg._lattice_part(Mg._grade_key(((), (3/2,)))); P['q0'], P['c'], P['dim']
            ((3/2), 9/4, 0)
        """
        V = self._V
        if G is not None:
            G = self._q_grade(G, self._lam)         # the grade of q = p - λ in V
            if G is None:
                return None
        P = V._lattice_part(G, V._weight_data())
        if P is None:
            return None
        A, beta, K = P['A'], P['beta'], P['K']
        q0 = P['q0'] + vector(QQ, self._lam)
        P = dict(P)
        P['q0'] = q0
        P['b'] = 2 * K.transpose() * A * q0 + K.transpose() * beta
        P['c'] = q0 * A * q0 + beta * q0
        return P

    def _weight_space_enumeration(self, W, G):
        """
        The keys of weight ``W`` and grade ``G`` (or all grades): lattice
        points ``p`` of the shifted lattice part times Heisenberg monomials
        of weight ``W - Δ(p)``; finite (ellipsoids) for positive definite
        parts, else staged.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: E = M._weight_space_enumeration(9/4, None); E.finite
            True
            sage: sorted(E.factory()) == sorted(M.weight_space(9/4).basis_keys()), len(list(E.factory()))
            (True, 6)
            sage: H = va.Lattice(QQ, matrix([[0, 2], [2, 0]])); J0, J1 = H.heisenberg_currents()
            sage: Mh = H.with_conformal_vector((J0.no(J1) + J1.no(J0)) / 4).simple_module([1/2, 0])
            sage: E = Mh._weight_space_enumeration(1, None); E.finite, E.first_within(3) is not None
            (False, True)
        """
        from .weight_spaces import (
            _box_shell,
            _ellipsoid_points,
            _KeyEnumeration,
            _LetterMultisets,
        )
        V = self._V
        data = V._weight_data()
        n = V.rank()
        P = self._lattice_part(G)
        if P is None:
            return _KeyEnumeration(True, factory=lambda: iter(()))
        hJ, q0, K, dim = P['hJ'], P['q0'], P['K'], P['dim']
        zeros = [()] * n
        evens = [False] * n

        def heis(p):
            return _LetterMultisets(hJ, zeros, evens, W - V._Delta(p, data), ())

        def p_of(m):
            return tuple(QQ(x) for x in (q0 + K * vector(ZZ, list(m))))

        if V._lattice_part_finite(P):
            def factory():
                for m in _ellipsoid_points(P['Q'], P['b'], P['c'], W):
                    p = p_of(m)
                    for letters in heis(p):
                        yield (letters, p)
            return _KeyEnumeration(True, factory=factory)

        def stage(s):
            for t in range(s + 1):
                for m in _box_shell(dim, t):
                    p = p_of(m)
                    H = heis(p)
                    for Kl in (range(s + 1) if t == s else [s]):
                        for letters in H.exactly(Kl):
                            yield (letters, p)
        return _KeyEnumeration(False, stage=stage)

    def _weight_bounds(self, G):
        """
        ``(lb, D)``: weights of grade ``G`` lie in ``Δ(λ) + (1/D) ZZ``, ``lb``
        a lower bound when finite (the real minimum of the quadratic form),
        ``None`` if not certified, ``+Infinity`` if the grade is empty.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._weight_bounds(None)
            (0, 4)
            sage: Mg = V.with_horizontal_grading('lattice').simple_module([1/2])
            sage: Mg._weight_bounds(Mg._grade_key(((), (3/2,))))
            (9/4, 4)
        """
        V = self._V
        d = V._weight_data()
        P = self._lattice_part(G)
        den = lcm([QQ(x).denominator() for row in d['A'] for x in row]
                  + [QQ(x).denominator() for x in d['beta']]
                  + [QQ(x).denominator() for x in d['hJ']] + [1])
        if P is None:
            return Infinity, int(den)
        den = lcm(den, QQ(P['c']).denominator())      # weights in Δ(λ) + (1/den) ZZ
        if not V._lattice_part_finite(P):
            return None, int(den)
        if P['dim'] == 0:
            return P['c'], int(den)
        Q, b, c = P['Q'], P['b'], P['c']
        mstar = -Q.inverse() * b / 2
        return mstar * Q * mstar + b * mstar + c, int(den)

    def _grade_key(self, m):
        """
        The grade (coordinates in the grading group of the modules) of a key:
        that of ``e^p``.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: Mg = V.with_horizontal_grading('lattice').simple_module([1/2])
            sage: Mg._grade_key(((), (3/2,))), Mg.horizontal_grade(Mg.monomial(((), (3/2,))))
            ((3,), (3/2))
        """
        return self._grade_of_p(m[1])

    def _is_simple(self):
        """
        ``V_{λ+L}`` is simple.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._is_simple(), M.maximal_submodule().weight_space(1/4).dimension()
            (True, 0)
        """
        return True

    def _key_order(self, m):
        """
        The key order of canonical rows: number of letters, then ``p``, then
        the letters.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._key_order((((0, 0),), (1/2,)))
            (1, (1/2,), ((0, 0),))
        """
        return (len(m[0]), m[1], m[0])

    def _l0_grading(self):
        """
        Whether the weights are ``L_0`` eigenvalues: iff ``V`` has a conformal
        vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M._l0_grading(), L0.simple_module([1/2])._l0_grading()
            (True, False)
        """
        return self._V.conformal_vector() is not None

    # ------------------------------------------------------------------ #
    #  Elements                                                            #
    # ------------------------------------------------------------------ #

    def highest_weight_vector(self):
        r"""
        A vector ``e^μ`` of lowest conformal weight, ``μ ∈ λ + L`` (the first
        one in sorted order) when the weights are bounded below (positive
        definite lattices); otherwise ``e^λ``.

        EXAMPLES::

            sage: from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
            sage: L0 = LatticeVertexAlgebra(QQ, matrix([[3]])); J = L0.J(0)
            sage: M = L0.with_conformal_vector(J.no(J) / 6).simple_module([2/3])
            sage: v = M.highest_weight_vector(); v, v.degree()
            (e^(-1/3), 1/6)
        """
        mu = getattr(self, '_lowest_mu', None)
        if mu is None:
            mu = self._lam
            try:
                lb, D = self._weight_bounds(None)
            except ValueError:
                lb = None
            if lb is not None and lb != Infinity:
                W = (QQ(lb) * D).ceil() / D
                for _ in range(10 * D + 10):
                    keys = [kk for kk in self.weight_space(W).basis_keys() if not kk[0]]
                    if keys:
                        mu = min(keys)[1]
                        break
                    W += QQ((1, D))
            self._lowest_mu = mu
        return self.monomial(((), mu))

    def some_elements(self):
        r"""
        Some vectors: the lowest weight vector, ``J0_(-1) e^λ``,
        ``e^{λ ± x_0}`` and an inhomogeneous sum.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: va.Lattice(QQ, matrix([[2]])).simple_module([1/2]).some_elements()
            [e^(1/2), :J0 e^(1/2):, e^(3/2), e^(-1/2), e^(1/2) + 2*:dJ0 e^(-1/2):]
        """
        V = self._V
        lam = self._lam
        x0 = tuple(QQ(int(i == 0)) for i in range(V.rank()))
        up = tuple(a + b for a, b in zip(lam, x0))
        down = tuple(a - b for a, b in zip(lam, x0))
        v = self.highest_weight_vector()
        return [v, self.monomial((((0, 0),), lam)), self.monomial(((), up)),
                self.monomial(((), down)), v + 2 * self.monomial((((0, 1),), down))]

    def _an_element_(self):
        """
        The lowest weight vector.

        TESTS::

            sage: import vertex_algebras as va
            sage: L0 = va.Lattice(QQ, matrix([[2]])); J = L0.J(0)
            sage: V = L0.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.an_element()
            e^(-1/2)
        """
        return self.highest_weight_vector()
