# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Pullbacks, twists and spectral flows of modules.

- :class:`PullbackModule` (``M.pullback(φ)``): for a morphism of vertex
  algebras ``φ: U -> V`` and a ``V``-module ``M``, the ``U``-module on the
  same space with ``a_(n) m = φ(a)_(n) m``; ``M.twist(σ)`` for an
  automorphism ``σ`` of ``V`` (``M^σ``).  Restrictions to subalgebras are
  pullbacks along their inclusions.

- :class:`SpectralFlowModule` (``M.spectral_flow(h)``): H. Li's
  ``Δ``-operator twist: [Li1997]_, (3.6), Lemma 3.3, Props. 2.1 and
  3.2 (stated there for regular vertex operator algebras and primary
  ``h`` of weight 1; the code checks only the conditions below).  For
  ``h ∈ V`` with ``h_(n) h = δ_{n,1} γ`` (``n >= 0``) and ``h_(0)``
  semisimple with integer eigenvalues on ``V``,

  ``Δ(h, z) = z^{h_(0)} exp(Σ_{k >= 1} h_(k) / (-k) (-z)^{-k})``

  and ``(M, Y_M(Δ(h, z) ·, z))`` is again a module.  For a basis vector
  ``a`` of ``V`` with ``h_(0) a = q a``, ``Δ(h, z) a = Σ_j a_j z^{q - j}``
  with ``a_0 = a``, ``j a_j = Σ_{k=1}^j k c_k h_(k) a_{j-k}``, ``c_k =
  (-1)^{k+1}/k`` (the ``h_(k)``, ``k >= 1``, commute), so the new action is
  ``a_(n) m = Σ_j (a_j)_(n + q - j) m``.  With a conformal vector ``T``,
  ``Δ(h, z) T = T + z^{-1} T_1 + z^{-2} T_2`` (``T_1 = h_(1) T``, ``= h``
  for primary ``h``; ``T_2 = γ/2`` then), so the new ``L_0`` is ``L_0 +
  (T_1)_(0) + T_2``: weights shift by the eigenvalue of ``(T_1)_(0)``, a
  linear function of the grade.  Spectral flow leaves positive energy;
  the graded weight spaces stay finite.

EXAMPLES:

Spectral flow of a lattice module by ``J_μ``, ``μ`` in the dual lattice, is
the module of the shifted coset (Li's simple currents; the flow keeps
the grades of ``M``, those of the coset are shifted by ``μ``)::

    sage: import vertex_algebras as va
    sage: J = va.Lattice(QQ, matrix([[2]])).J(0)
    sage: L = va.Lattice(QQ, matrix([[2]]), conformal_vector=J.no(J)/4,
    ....:                horizontal_grading='lattice')
    sage: M = L.simple_module([0]); F = M.spectral_flow(L.J(0)/2)
    sage: N = L.simple_module([1/2]); G = N.horizontal_grading_group()
    sage: all(F.character(5, G([p])) == N.character(5, G([p + 1/2])) for p in [0, 1, -1, 2])
    True
    sage: F.character(3, G([0]))
    q^(1/4) + q^(5/4) + 2*q^(9/4) + O(q^3)

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.arith.functions import lcm
from sage.misc.functional import denominator
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ

from .vertex_algebra_module import VertexAlgebraModule


def _key_dict(V, a):
    r"""
    The coefficients of a vertex algebra element (``a`` an element).

    TESTS::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.module_twists import _key_dict
        sage: V = va.Virasoro(QQ, 1/2); _key_dict(V, V.gen(0))
        {((0, 0),): 1}
    """
    return dict(V(a).monomial_coefficients())


def _some_key(M):
    r"""
    A basis key of ``M``: of its highest weight vector, else of lowest
    weight.

    TESTS::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.module_twists import _some_key
        sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
        sage: _some_key(H.adjoint_module())
        ()
    """
    try:
        return next(iter(M.highest_weight_vector().support()))
    except AttributeError:
        return next(iter(M.weight_space(M._weight_base()).iter_basis_keys()))


def _rewrap_weight_space(P, S, weight, horizontal_grade):
    r"""
    The weight space ``S`` of the underlying module (spanned by basis keys)
    as a weight space of ``P`` (same keys) of weight ``weight``.

    TESTS::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(0).simple_quotient()
        sage: P = M.pullback(End(V).identity()); P.weight_space(2).basis()
        [L_{-2} v]
    """
    from .weight_spaces import WeightSpace, _KeyEnumeration
    if not hasattr(S, 'iter_basis_keys'):
        raise NotImplementedError('weight spaces of {} are not spanned by basis keys'.format(P))
    finite = S.is_finite_dimensional()
    return WeightSpace(P, weight, horizontal_grade, _KeyEnumeration(finite, factory=S.iter_basis_keys))


def _inverse(phi):
    r"""
    The inverse of an endomorphism ``phi`` of a vertex algebra (also of an
    involution whose inverse is not computed by ``inverse()``), or ``None``.

    TESTS::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.module_twists import _inverse
        sage: L = va.Lattice(QQ, matrix([[2]])); J, ep, em = L.gens()
        sage: s = L.hom([em, ep], L); _inverse(s) == s
        True
        sage: H = va.FreeBosons(QQ); J = H.gen(0)
        sage: _inverse(va.Virasoro(QQ, 1).hom([J.no(J)/2], H)) is None
        True
    """
    if phi.domain() is not phi.codomain():
        return None
    for cand in (None, phi):
        try:
            return phi.inverse(cand)
        except (ValueError, NotImplementedError, TypeError):
            pass
    return None


class PullbackModule(VertexAlgebraModule):
    r"""
    The pullback ``φ^* M`` of a ``V``-module ``M`` along a morphism ``φ: U ->
    V``: the space of ``M`` with ``a_(n) m = φ(a)_(n) m`` for ``a ∈ U``.
    Conformally graded with ``M``'s weights iff ``φ`` maps the conformal
    vector of ``U`` to that of ``V``; graded by ``M``'s grading group when
    ``φ`` maps every generator to an element of the same grade.

    INPUT:

    - ``M`` -- a module over a vertex algebra ``V``
    - ``phi`` -- a morphism of vertex algebras ``φ: U -> V`` (its codomain
      must be ``V`` itself)
    - ``name`` -- (default: ``'Pullback'``) a string used in the repr;
      ``'Twist'`` (as ``M.twist(σ)`` passes) prints ``Twist of M by σ``

    EXAMPLES:

    The Chevalley involution of ``sl_2`` swaps the highest weights of the
    Weyl modules ``E_0 ↦ -F_0``: the doublet stays a doublet, with the
    roles of its vectors exchanged::

        sage: import vertex_algebras as va
        sage: V = va.Affine(QQ, 'A1', 1, conformal_vector='sugawara'); E, H, F = V.gen(0), V.gen(1), V.gen(2)
        sage: sigma = V.hom([-F, -H, -E], V)
        sage: U = V.zhu_algebra().module({'alpha[1]': matrix([[0, 1], [0, 0]]),
        ....:     'alphacheck[1]': matrix([[1, 0], [0, -1]]), '-alpha[1]': matrix([[0, 0], [1, 0]])})
        sage: M = V.induced_module(U); Mt = M.twist(sigma); Mt
        Twist of Induced module ... by Vertex algebra endomorphism of ...
        sage: u0, u1 = [Mt(M.monomial(((), j))) for j in (0, 1)]
        sage: H.mode(u0, 0) == -u0, H.mode(u1, 0) == u1
        (True, True)
        sage: Mt.character(3) == M.character(3)
        True
        sage: TestSuite(Mt).run()
    """

    def __init__(self, M, phi, name='Pullback'):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: P = M.pullback(End(V).identity()); P.vertex_algebra() is V
            True
            sage: M.pullback(va.FreeBosons(QQ).hom([va.FreeBosons(QQ).gen(0)]))
            Traceback (most recent call last):
            ...
            ValueError: the codomain of the morphism must be the vertex algebra of the module
        """
        if phi.codomain() is not M.vertex_algebra():
            raise ValueError('the codomain of the morphism must be the vertex algebra of the module')
        U = phi.domain()
        self._M = M
        self._phi = phi
        self._name = name
        self._phi_memo = {}
        graded = M._is_graded_module and phi.is_conformal()
        if M._mgrading is not None and self._grade_preserving(phi):
            self._mgrading = M._mgrading
        VertexAlgebraModule.__init__(self, U, M._indices, super=M._is_super_module, graded=graded)

    @staticmethod
    def _grade_preserving(phi):
        r"""
        Whether ``φ`` maps every generator to an element of the same grade
        (same grading group).

        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.module_twists import PullbackModule
            sage: V = va.Affine(QQ, 'A1', 1); E, H, F = V.gens()
            sage: PullbackModule._grade_preserving(End(V).identity()), PullbackModule._grade_preserving(V.hom([-F, -H, -E], V))
            (True, False)
        """
        U, V = phi.domain(), phi.codomain()
        if U._grading is None or V._grading is None or U._grading['group'] is not V._grading['group']:
            return False
        gens = dict(zip(U.variable_names(), U.gens()))
        for n, img in phi.im_gens_dict().items():
            if img.is_zero():
                continue
            try:
                if V.horizontal_grade(img) != U.horizontal_grade(gens[n]):
                    return False
            except ValueError:
                return False
        return True

    def _repr_(self):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())
            Pullback of Verma module ... along Vertex algebra endomorphism of ...
        """
        prep = 'by' if self._name == 'Twist' else 'along'
        return '{} of {} {} {}'.format(self._name, self._M, prep, repr(self._phi).split('\n')[0])

    def underlying_module(self):
        r"""
        The module ``M`` pulled back.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity()).underlying_module() is M
            True
        """
        return self._M

    def morphism(self):
        r"""
        The morphism ``φ`` pulled back along.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity()).morphism() == End(V).identity()
            True
        """
        return self._phi

    def _image(self, a):
        r"""
        ``φ`` of the basis key ``a`` of ``U``, a raw dict over ``V``'s keys.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.FermionicGhosts(QQ); b, c = V.gens()
            sage: M = V.adjoint_module().pullback(V.hom([c, b])); M._image(((0, 0),))
            {((1, 0),): 1}
        """
        try:
            return self._phi_memo[a]
        except KeyError:
            U, V = self._phi.domain(), self._phi.codomain()
            d = self._phi_memo[a] = _key_dict(V, self._phi(U.monomial(a)))
            return d

    def _action_key(self, a, n, m):
        r"""
        ``a_(n) m = φ(a)_(n) m``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.FermionicGhosts(QQ); b, c = V.gens()
            sage: A = V.adjoint_module(); M = A.pullback(V.hom([c, b]))
            sage: b.nproduct(M(A(c)), 0) == M(A(c.nproduct(c, 0))), b.nproduct(M(A(b)), 0)
            (True, |0>)
        """
        return self._M._action_raw(self._image(a), n, {m: self._one_R})

    def _nmax_key(self, a, m):
        r"""
        The largest ``n`` with ``φ(a)_(n) m`` possibly nonzero.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())._nmax_key(((0, 0),), ((), 0))
            1
        """
        d = self._image(a)
        return max((self._M._nmax_key(b, m) for b in d), default=-1)

    def _parity_key(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.FermionicGhosts(QQ); b, c = V.gens()
            sage: V.adjoint_module().pullback(V.hom([c, b]))._parity_key(((0, 0),))
            1
        """
        return self._M._parity_key(m)

    def _repr_term(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())._repr_term(((), 0))
            'v'
        """
        return self._M._repr_term(m)

    def _weight_key(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())._weight_key(((), 0))
            1/16
        """
        return self._M._weight_key(m)

    def _weight_space_enumeration(self, W, G):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity()).weight_space(1/16 + 2).dimension()
            2
        """
        return self._M._weight_space_enumeration(W, G)

    def weight_space(self, weight, horizontal_grade=None):
        r"""
        ``M``'s weight space, with the vectors of ``self``.

        INPUT:

        - ``weight`` -- a conformal weight
        - ``horizontal_grade`` -- (default: ``None``) a grade of ``M``'s
          grading group, or ``None`` for all grades

        OUTPUT:

        A weight space of ``self`` on the basis keys of ``M``'s weight
        space (``ValueError`` if ``self`` is not conformally graded,
        ``NotImplementedError`` if ``M``'s weight space is not spanned by
        basis keys).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity()).weight_space(1/16 + 1).basis()
            [L_{-1} v]
        """
        if not self._is_graded_module:
            raise ValueError('{} is not conformally graded'.format(self))
        return _rewrap_weight_space(self, self._M.weight_space(weight, horizontal_grade), weight, horizontal_grade)

    def _weight_bounds(self, G):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())._weight_bounds(None)
            (1/16, 16)
        """
        return self._M._weight_bounds(G)

    def _weight_base(self):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity())._weight_base()
            1/16
        """
        return self._M._weight_base()

    def _key_order(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: P = M.pullback(End(V).identity()); P._key_order(((), 0)) == M._key_order(((), 0))
            True
        """
        return self._M._key_order(m)

    def _grade_key(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]), horizontal_grading='lattice')
            sage: M = L.simple_module([1/2]); P = M.pullback(End(L).identity())
            sage: P._grade_key(((), (1/2,)))
            (1,)
        """
        if self._mgrading is None:
            return VertexAlgebraModule._grade_key(self, m)
        return self._M._grade_key(m)

    def _is_simple(self):
        r"""
        A twist of a simple module by an automorphism is simple; other
        pullbacks are not known to be.

        TESTS::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); J, ep, em = L.gens()
            sage: M = L.simple_module([0]); M.twist(L.hom([em, ep], L))._is_simple()
            True
        """
        return _inverse(self._phi) is not None and self._M._is_simple()

    def highest_weight_vector(self):
        r"""
        ``M``'s highest weight vector, in ``self``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: M.pullback(End(V).identity()).highest_weight_vector()
            v
        """
        return self(self._M.highest_weight_vector())

    def _element_constructor_(self, x):
        r"""
        Vectors of ``M`` convert (same space).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); M = V.verma_module(1/16)
            sage: P = M.pullback(End(V).identity()); P(M.highest_weight_vector()).parent() is P
            True
        """
        if getattr(x, 'parent', None) is not None and x.parent() is self._M:
            return self._from_dict(dict(x.monomial_coefficients()))
        return VertexAlgebraModule._element_constructor_(self, x)


class SpectralFlowModule(VertexAlgebraModule):
    r"""
    The spectral flow (H. Li's ``Δ``-twist) of a ``V``-module ``M`` by ``h
    ∈ V``: the space of ``M`` with ``Y(a, z) = Y_M(Δ(h, z) a, z)``; see the
    module documentation and [Li1997]_.

    INPUT:

    - ``M`` -- a module over a vertex algebra ``V``
    - ``h`` -- an element of ``V`` (converted into ``V``) with ``h_(n) h =
      δ_{n,1} γ`` for ``n >= 0`` (``γ`` a scalar), ``h_(0)`` acting with
      integer eigenvalues on the generators of ``V``, and of grade 0 if
      ``V`` is graded; ``ValueError`` otherwise

    The result is conformally graded when ``M`` is, ``V`` has a conformal
    vector ``T`` and ``Δ(h, z) T`` has the form described in the module
    documentation.

    EXAMPLES:

    Spectral flow of the vacuum of ``V^1(sl_2)`` by ``H/2`` (``h_(0)`` has
    the eigenvalues ``±1`` on ``E``, ``F``): the top of the result is the
    image of the vacuum, of weight ``k/4 = 1/4`` and charge ``H_0 = k = 1``,
    the highest weight vector of the doublet module ([Li1997]_, Prop. 3.5)::

        sage: import vertex_algebras as va
        sage: V = va.Affine(QQ, 'A1', 1, conformal_vector='sugawara'); E, H, F = V.gens()
        sage: M = V.induced_module(V.zhu_algebra().module({'alpha[1]': matrix([[0]]),
        ....:     'alphacheck[1]': matrix([[0]]), '-alpha[1]': matrix([[0]])}))
        sage: S = M.spectral_flow(H/2); v = S(M.highest_weight_vector())
        sage: S._weight_key(((), 0)), H.mode(v, 0) == v, E.mode(v, 0).is_zero(), F.mode(v, 0).is_zero()
        (1/4, True, True, False)
        sage: TestSuite(S).run()

    TESTS:

    The flow of ``V_{λ+L}`` by ``J_μ`` (``μ`` in the dual lattice, in the
    coordinates of the basis of ``L``) is ``V_{λ+μ+L}``: graded characters
    agree with the grades shifted by ``μ``, and ``J_i(0)`` acts on the image
    of ``e^p`` by ``B(x_i, p + μ)``. Checked for ``A_2`` and the odd
    lattice ``√3 Z``::

        sage: import vertex_algebras as va
        sage: for B, lam, mu in [(matrix([[2, -1], [-1, 2]]), [0, 0], [2/3, 1/3]),
        ....:                    (matrix([[2, -1], [-1, 2]]), [1/3, 2/3], [1/3, 2/3]),
        ....:                    (matrix([[3]]), [1/3], [1/3])]:
        ....:     L0 = va.Lattice(QQ, B); r = B.nrows(); Bi = B.inverse()
        ....:     T = sum(Bi[i, j] * L0.J(i).no(L0.J(j)) for i in range(r) for j in range(r)) / 2
        ....:     L = va.Lattice(QQ, B, conformal_vector=T, horizontal_grading='lattice')
        ....:     M = L.simple_module(lam); F = M.spectral_flow(sum(mu[i] * L.J(i) for i in range(r)))
        ....:     N = L.simple_module([lam[i] + mu[i] for i in range(r)]); G = N.horizontal_grading_group()
        ....:     l, m = vector(QQ, lam), vector(QQ, mu)
        ....:     grades = [l + vector(QQ, [s] + [0] * (r - 1)) for s in (0, 1, -1)]
        ....:     w = M.highest_weight_vector(); v = F(w); p = vector(QQ, w.leading_support()[1])
        ....:     print(F.character(2, G(grades[0])),
        ....:           all(F.character(4, G(g)) == N.character(4, G(g + m)) for g in grades),
        ....:           all(L.J(i).nproduct(v, 0) == (B * (p + m))[i] * v for i in range(r)))
        q^(1/3) + 2*q^(4/3) + O(q^2) True True
        q^(4/3) + O(q^2) True True
        q^(2/3) + q^(5/3) + O(q^2) True True

    Heisenberg Fock modules: the flow of ``F_0`` by ``3J`` acts as ``F_3``;
    flows compose (by ``J``, then ``2J``), the flow by ``-3J`` undoes the one
    by ``3J``, and the flow by 0 changes nothing::

        sage: Hb = va.FreeBosons(QQ); J = Hb.gen(0); Hb = Hb.with_conformal_vector(J.no(J)/2); J = Hb.gen(0)
        sage: fock = lambda a: Hb.induced_module(Hb.zhu_algebra().module({'alpha': matrix([[a]])}))
        sage: F0, F3 = fock(0), fock(3)
        sage: keys = [k for w in range(4) for k in F0.weight_space(w).basis_keys()]
        sage: def same(S1, S2):
        ....:     return all(J.nproduct(S1.monomial(k), n).monomial_coefficients()
        ....:                == J.nproduct(S2.monomial(k), n).monomial_coefficients()
        ....:                for k in keys for n in range(-2, 3))
        sage: S3 = F0.spectral_flow(3*J)
        sage: same(S3, F3), S3.character(9) == F3.character(9)
        (True, True)
        sage: same(F0.spectral_flow(J).spectral_flow(2*J), S3), same(F3.spectral_flow(-3*J), F0), same(F0.spectral_flow(0*J), F0)
        (True, True, True)

    At the admissible level ``k = -1/2`` the flow of ``L(-1/2)`` by ``H/2``
    is not positive energy: the lowest weight of a grade decreases without
    bound as the grade decreases. The graded pieces are finite, and the new
    ``L_0 = T_(1)`` acts on every basis vector by its weight::

        sage: V = va.Affine(QQ, 'A1', -1/2, conformal_vector='sugawara'); E, H, F = V.gens()
        sage: L = V.induced_module(V.zhu_algebra().module({'alpha[1]': matrix([[0]]),
        ....:     'alphacheck[1]': matrix([[0]]), '-alpha[1]': matrix([[0]])})).simple_quotient()
        sage: S = L.spectral_flow(H/2); G = L.horizontal_grading_group()
        sage: ws = {g: S._shift(S._grade_coords(G([g]))) for g in [-4, -2, 0, 2, 4]}; ws
        {-4: -17/8, -2: -9/8, 0: -1/8, 2: 7/8, 4: 15/8}
        sage: [S.weight_space(ws[g] + d, G([g])).dimension() for g in [-2, 0, 2] for d in range(3)]
        [0, 1, 2, 1, 1, 3, 0, 1, 2]
        sage: T = V.conformal_vector()
        sage: keys = [b for g in [-2, 0, 2] for d in range(3) for b in S.weight_space(ws[g] + d, G([g])).basis_keys()]
        sage: all(T.nproduct(S.monomial(b), 1) == S._weight_key(b) * S.monomial(b) for b in keys)
        True
    """

    def __init__(self, M, h):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, 'A1', 1, conformal_vector='sugawara'); E, H, F = V.gens()
            sage: M = V.adjoint_module()
            sage: M.spectral_flow(E)
            Traceback (most recent call last):
            ...
            ValueError: h_(0) does not act with integer eigenvalues on the generators of V (alphacheck[1]: None)
            sage: M.spectral_flow(H.no(H))
            Traceback (most recent call last):
            ...
            ValueError: h_(n) h must be zero for n >= 0 except n = 1 (a multiple of the vacuum)
            sage: M.spectral_flow(H/4)
            Traceback (most recent call last):
            ...
            ValueError: h_(0) does not act with integer eigenvalues on the generators of V (alpha[1]: 1/2)
        """
        V = M.vertex_algebra()
        h = V(h)
        self._M = M
        self._h = h
        poles = h.ope(h)
        if any(not p.is_zero() for i, p in enumerate(poles) if i != 1):
            raise ValueError('h_(n) h must be zero for n >= 0 except n = 1 (a multiple of the vacuum)')
        gam = poles[1] if len(poles) > 1 else V.zero()
        if gam.is_zero():
            self._gamma = QQ.zero()
        else:
            d = gam.monomial_coefficients()
            if set(d) != {V._vacuum_key()}:
                raise ValueError('h_(n) h must be zero for n >= 0 except n = 1 (a multiple of the vacuum)')
            self._gamma = d[V._vacuum_key()]
        self._coeffs = {}
        for name, g in zip(V.variable_names(), V.gens()):
            q = self._eigenvalue(g)
            if q is None or q not in ZZ:
                raise ValueError('h_(0) does not act with integer eigenvalues on the generators of V ({}: {})'.format(name, q))
        if V._grading is not None and any(any(x != 0 for x in V._grade_key(k)) for k in h.support()):
            raise ValueError('h must be homogeneous of grade 0')
        T = V.conformal_vector()
        graded = M._is_graded_module and T is not None
        self._mgrading = M._mgrading
        self._eig_memo = {}
        if graded:
            Tj = self._delta(T)
            if any(j >= 3 and not x.is_zero() for j, x in Tj.items()) or any(j < 0 for j in Tj):
                graded = False
            else:
                self._T1 = Tj.get(1, V.zero())
                T2 = Tj.get(2, V.zero())
                d = T2.monomial_coefficients()
                if set(d) - {V._vacuum_key()}:
                    graded = False
                self._t2 = d.get(V._vacuum_key(), QQ.zero())
        VertexAlgebraModule.__init__(self, V, M._indices, super=M._is_super_module, graded=graded)

    def _repr_(self):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); M = H.adjoint_module(); M.spectral_flow(H.gen(0))
            Spectral flow of Adjoint module of ... by alpha
        """
        return 'Spectral flow of {} by {}'.format(self._M, self._h)

    def underlying_module(self):
        r"""
        The module ``M`` flowed.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); M = H.adjoint_module(); M.spectral_flow(H.gen(0)).underlying_module() is M
            True
        """
        return self._M

    def _eigenvalue(self, a):
        r"""
        The eigenvalue of ``h_(0)`` on the element ``a`` of ``V``, ``None``
        if ``a`` is not an eigenvector.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, 'A1', 1); E, H, F = V.gens()
            sage: S = V.adjoint_module().spectral_flow(H/2)
            sage: S._eigenvalue(E), S._eigenvalue(E + F)
            (1, None)
        """
        b = self._h.nproduct(a, 0)
        if b.is_zero():
            return QQ.zero()
        da, db = a.monomial_coefficients(), b.monomial_coefficients()
        if set(da) != set(db):
            return None
        k = next(iter(da))
        q = db[k] / da[k]
        return q if b == q * a else None

    def _delta(self, a):
        r"""
        ``Δ(h, z) a`` for an ``h_(0)``-eigenvector ``a`` with eigenvalue
        ``q``: ``{j: a_j}`` with ``Δ(h, z) a = Σ_j a_j z^{q - j}``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, 'A1', 1, conformal_vector='sugawara'); E, H, F = V.gens()
            sage: S = V.adjoint_module().spectral_flow(H/2)
            sage: d = S._delta(V.conformal_vector()); sorted(d), d[1] == H/2, d[2]
            ([0, 1, 2], True, 1/4)
        """
        h = self._h
        out = {0: a}
        reach = len(h.ope(a)) - 1
        j = 1
        while j <= reach:
            acc = a.parent().zero()
            for k in range(1, j + 1):
                prev = out.get(j - k)
                if prev is None or prev.is_zero():
                    continue
                acc += k * (QQ(-1) ** (k + 1) / k) * h.nproduct(prev, k)
            acc = acc / j
            if not acc.is_zero():
                out[j] = acc
                reach = max(reach, j + len(h.ope(acc)) - 1)
            j += 1
        return out

    def _coefficients(self, a):
        r"""
        For a basis key ``a`` of ``V``: ``(q, [(j, raw dict of a_j)])``.

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); S = H.adjoint_module().spectral_flow(H.gen(0))
            sage: S._coefficients(((0, 0),))
            (0, [(0, {((0, 0),): 1}), (1, {(): 1})])
        """
        try:
            return self._coeffs[a]
        except KeyError:
            pass
        V = self._M.vertex_algebra()
        x = V.monomial(a)
        q = self._eigenvalue(x)
        if q is None or q not in ZZ:
            raise ValueError('h_(0) is not diagonal with integer eigenvalues on {}'.format(x))
        q = ZZ(q)
        out = (q, [(j, dict(y.monomial_coefficients())) for j, y in sorted(self._delta(x).items())])
        self._coeffs[a] = out
        return out

    def _action_key(self, a, n, m):
        r"""
        ``a_(n) m = Σ_j (a_j)_(n + q - j) m``.

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); A = H.adjoint_module()
            sage: S = A.spectral_flow(2*J); v = S(A(H.one()))
            sage: J.nproduct(v, 0) == 2*v
            True
        """
        q, coeffs = self._coefficients(a)
        out = {}
        one = self._one_R
        for j, d in coeffs:
            r = self._M._action_raw(d, n + q - j, {m: one})
            for k, c in r.items():
                out[k] = out.get(k, 0) + c
        return {k: c for k, c in out.items() if c}

    def _nmax_key(self, a, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); A = H.adjoint_module(); S = A.spectral_flow(H.gen(0))
            sage: S._nmax_key(((0, 0),), ())
            0
        """
        q, coeffs = self._coefficients(a)
        return max((self._M._nmax_key(b, m) - q + j for j, d in coeffs for b in d), default=-1)

    def _parity_key(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); A = H.adjoint_module(); A.spectral_flow(H.gen(0))._parity_key(())
            0
        """
        return self._M._parity_key(m)

    def _repr_term(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); A = H.adjoint_module(); A.spectral_flow(H.gen(0))._repr_term(())
            '|0>'
        """
        return self._M._repr_term(m)

    def _key_order(self, m):
        r"""
        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); A = H.adjoint_module(); A.spectral_flow(H.gen(0))._key_order(())
            (0, ())
        """
        return self._M._key_order(m)

    def _grade_key(self, m):
        r"""
        ``M``'s grades (``h`` has grade 0).

        TESTS::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]]), horizontal_grading='lattice')
            sage: M = L.simple_module([1/2]); M.spectral_flow(L.J(0))._grade_key(((), (1/2,)))
            (1,)
        """
        return self._M._grade_key(m)

    # -------------------------------------------------------------- #
    #  Weights                                                         #
    # -------------------------------------------------------------- #

    def _t1_eigenvalue(self, m):
        r"""
        The eigenvalue of ``(T_1)_(0)`` on the basis key ``m`` of ``M``
        (``ValueError`` if ``m`` is not an eigenvector).

        TESTS::

            sage: import vertex_algebras as va
            sage: J = va.Lattice(QQ, matrix([[2]])).J(0)
            sage: L = va.Lattice(QQ, matrix([[2]]), conformal_vector=J.no(J)/4)
            sage: S = L.simple_module([1/2]).spectral_flow(L.J(0))
            sage: S._t1_eigenvalue(((), (1/2,)))
            1
        """
        try:
            return self._eig_memo[m]
        except KeyError:
            pass
        d = self._M._action_raw(dict(self._T1.monomial_coefficients()), 0, {m: self._one_R})
        if set(d) - {m}:
            raise ValueError('(T_1)_(0) is not diagonal on the basis of {}'.format(self._M))
        e = self._eig_memo[m] = QQ(d.get(m, 0))
        return e

    def _weight_key(self, m):
        r"""
        ``wt_M(m) + (T_1)_(0)-eigenvalue + T_2``.

        TESTS::

            sage: import vertex_algebras as va
            sage: J = va.Lattice(QQ, matrix([[2]])).J(0)
            sage: L = va.Lattice(QQ, matrix([[2]]), conformal_vector=J.no(J)/4)
            sage: S = L.simple_module([0]).spectral_flow(L.J(0)/2)
            sage: S._weight_key(((), (0,))), S._weight_key(((), (-1,)))
            (1/4, 1/4)
        """
        return self._M._weight_key(m) + self._t1_eigenvalue(m) + self._t2

    def _functional(self):
        r"""
        ``(ν, c)``: the eigenvalue of ``(T_1)_(0)`` on a vector of grade
        ``G`` (coordinates of the module's grading group) is ``ν·G + c``;
        ``ν`` is ``None`` when it is zero on ``V`` (then ``c`` is the
        eigenvalue everywhere).

        TESTS::

            sage: import vertex_algebras as va
            sage: J = va.Lattice(QQ, matrix([[2]])).J(0)
            sage: L = va.Lattice(QQ, matrix([[2]]), conformal_vector=J.no(J)/4,
            ....:                horizontal_grading='lattice')
            sage: L.simple_module([1/2]).spectral_flow(L.J(0))._functional()
            ((1), 0)
        """
        try:
            return self._nu
        except AttributeError:
            pass
        from sage.matrix.constructor import matrix
        from sage.modules.free_module_element import vector
        M = self._M
        V = M.vertex_algebra()
        T1 = self._T1
        eigs = []
        for g in V.gens():
            b = T1.nproduct(g, 0)
            q = QQ.zero()
            if not b.is_zero():
                k = next(iter(g.monomial_coefficients()))
                q = b.monomial_coefficients().get(k, 0) / g.monomial_coefficients()[k]
            eigs.append((g, q))
        m0 = _some_key(M)
        if all(q == 0 for g, q in eigs):
            self._nu = (None, self._t1_eigenvalue(m0))
            return self._nu
        iota = M._require_module_grading()
        A, B = iota.A, iota.B
        if A.torsion:
            raise NotImplementedError('spectral flow on modules graded by groups with torsion')
        rows = [vector(QQ, [QQ(x) for x in A.coords(V.horizontal_grade(g))]) for g, q in eigs]
        nuA = matrix(QQ, rows).solve_right(vector(QQ, [q for g, q in eigs]))
        Mi = iota.M.change_ring(QQ) if B.kind == 'vector' else iota.M.change_ring(QQ).matrix_from_rows(list(B.free))
        if Mi.rank() < Mi.nrows():
            raise NotImplementedError('the grading group of the module is larger than that of V')
        nuB = Mi.transpose().solve_right(nuA)
        free = list(range(B.k)) if B.kind == 'vector' else list(B.free)

        def nu(G):
            return sum((nuB[i] * QQ(G[j]) for i, j in enumerate(free)), QQ.zero())
        c = self._t1_eigenvalue(m0) - nu(M._grade_key(m0))
        self._nu = (nuB, c)
        self._nu_fun = nu
        return self._nu

    def _shift(self, G):
        r"""
        The weight shift of the vectors of grade ``G`` (``None``: all
        grades, only if the shift does not depend on the grade).

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
            sage: S = H.adjoint_module().spectral_flow(2*H.gen(0)); S._shift(None)
            2
        """
        nuB, c = self._functional()
        if nuB is None:
            return c + self._t2
        if G is None:
            raise ValueError('the weights of a spectral flow depend on the grade: give a grade')
        return self._nu_fun(G) + c + self._t2

    def _weight_space_enumeration(self, W, G):
        r"""
        ``M``'s keys of weight ``W - shift(G)`` and grade ``G``.

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
            sage: S = H.adjoint_module().spectral_flow(2*H.gen(0)); S.weight_space(4).dimension()
            2
        """
        return self._M._weight_space_enumeration(W - self._shift(G), G)

    def weight_space(self, weight, horizontal_grade=None):
        r"""
        ``M``'s weight space of weight ``weight - shift(G)`` and grade
        ``G``, with the vectors of ``self``.

        INPUT:

        - ``weight`` -- a conformal weight of ``self``
        - ``horizontal_grade`` -- (default: ``None``) a grade ``G`` of
          ``M``'s grading group (the flow keeps ``M``'s grades), or
          ``None`` for all grades (then the shift must not depend on the
          grade)

        OUTPUT:

        A weight space of ``self`` on the basis keys of ``M``'s weight
        space of weight ``weight - shift(G)`` (``ValueError`` if ``self``
        is not conformally graded).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: J = va.Lattice(QQ, matrix([[2]])).J(0)
            sage: L = va.Lattice(QQ, matrix([[2]]), conformal_vector=J.no(J)/4,
            ....:                horizontal_grading='lattice')
            sage: S = L.simple_module([0]).spectral_flow(L.J(0)/2); G = S.horizontal_grading_group()
            sage: S.weight_space(1/4, G([-1])).basis()
            [e^(-1)]
        """
        if not self._is_graded_module:
            raise ValueError('{} is not conformally graded'.format(self))
        G = None if horizontal_grade is None else self._grade_coords(horizontal_grade)
        S = self._M.weight_space(weight - self._shift(G), horizontal_grade)
        return _rewrap_weight_space(self, S, weight, horizontal_grade)

    def _weight_bounds(self, G):
        r"""
        ``M``'s bounds shifted (``None`` for all grades when the shift
        depends on the grade).

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); J = H.gen(0); H = H.with_conformal_vector(J.no(J)/2)
            sage: S = H.adjoint_module().spectral_flow(2*H.gen(0)); S._weight_bounds(None)
            (2, 1)
        """
        lb, D = self._M._weight_bounds(G)
        try:
            s = self._shift(G)
        except ValueError:
            return None, D
        D = lcm(ZZ(D), denominator(QQ(s)))
        if lb is None:
            return None, D
        from sage.rings.infinity import Infinity
        if lb == Infinity:
            return lb, D
        return lb + s, D

    def _is_simple(self):
        r"""
        A spectral flow of a simple module is simple (the flow by ``-h``
        undoes it).

        TESTS::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); M = L.simple_module([0])
            sage: M.spectral_flow(L.J(0)/2)._is_simple()
            True
        """
        return self._M._is_simple()

    def highest_weight_vector(self):
        r"""
        ``M``'s highest weight vector, in ``self`` (not of lowest weight
        in general).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: L = va.Lattice(QQ, matrix([[2]])); M = L.simple_module([1/2])
            sage: M.spectral_flow(L.J(0)).highest_weight_vector()
            e^(1/2)
        """
        return self(self._M.highest_weight_vector())

    def _element_constructor_(self, x):
        r"""
        Vectors of ``M`` convert (same space).

        TESTS::

            sage: import vertex_algebras as va
            sage: H = va.FreeBosons(QQ); A = H.adjoint_module(); S = A.spectral_flow(H.gen(0))
            sage: S(A(H.one())).parent() is S
            True
        """
        if getattr(x, 'parent', None) is not None and x.parent() is self._M:
            return self._from_dict(dict(x.monomial_coefficients()))
        return VertexAlgebraModule._element_constructor_(self, x)
