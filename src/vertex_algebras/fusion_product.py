# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Fusion products of modules: the Nahm–Gaberdiel–Kausch algorithm.

``M.fusion_product(N)`` is the fusion (vertex tensor) product ``M ⊠ N``
of two modules over the same vertex algebra ``V``.  What is computed is
its depth-``d`` quotient ``(M ⊠ N)^(d)``: ``M ⊠ N`` modulo the span of
the words of modes of total degree ``> d`` (the degree of a mode is how
much it raises the conformal weight), a finite-dimensional space on which
the modes of degree ``0 .. d`` act, in particular ``L_0`` (Jordan blocks:
logarithmic fusion) and the zero modes.  At depth 0 it is a module over
the Zhu algebra ``A(V)``.

ALGORITHM:

Nahm [Nah1994]_; Gaberdiel–Kausch [GK1996]_; in the form of Kanade–Ridout
[KR2019]_, §5 and §8.  ``M`` sits at ``w = 1``, ``N`` at ``0``.  For a
strong generator ``v`` of weight ``h`` and a rational function ``f`` with
poles at ``0, 1, ∞`` only, ``v f`` acts on ``M ⊗ N`` by

.. MATH::

    \Delta(v f)(x \otimes y) = (v f)_1 x \otimes y
        + (-1)^{|v||x|} x \otimes (v f)_0 y,

``(v f)_1 = Σ_j c_j v_(j)`` from ``f = Σ_j c_j (t-1)^j`` at ``1`` and
``(v f)_0 = Σ_k c_k v_(k)`` from ``f = Σ_k c_k t^k`` at ``0``; both
expansions terminate on vectors.  In particular ``Δ(v_(k)) = Σ_{j >= 0}
C(k, j) v_(j) ⊗ 1 + 1 ⊗ v_(k)``.  The fusion product is ``M ⊗ N`` modulo
``Δ(v f) - Δ(v ι_∞ f)`` (``ι_∞ f``: the expansion at infinity, an
infinite sum of modes, finite modulo the modes of degree ``> d``).  By
partial fractions ``f`` runs over ``t^k`` and ``(t-1)^b``, ``b < 0``, and
the depth-``d`` quotient is ``M ⊗ N`` modulo

- ``D_d``: the words ``Δ(u_1) ... Δ(u_r) x`` of generator modes of total
  degree ``> d`` (``D_e = Σ_{deg u > e} Δ(u) (M ⊗ N) + Σ_{1 <= deg u <= e}
  Δ(u) D_{e - deg u}``);
- the pole relations ``Δ(v (t-1)^b) x - Σ_m C(b, m) (-1)^m Δ(v_(b-m)) x``,
  the sum over the modes of degree ``<= d``, closed under the generator
  modes of degree ``0 .. d`` (the exact relations form a submodule; these
  modes keep ``D_d``).  For ``d < h`` this is Nahm's rule moving a mode
  ``v_(b)``, ``b < 0``, from ``M`` to the non-negative modes on ``N``.

The pole relations of composite fields and the products of modes are
consequences (checked in the tests: ``Δ`` is a homomorphism of mode
algebras).

THE WINDOW.  Everything is computed on the tensors of total depth ``<=
max_depth`` (depth = weight minus the module's lowest weight; one cutoff
shared by the two factors), with the relations whose terms all lie in
it.  By Nahm [Nah1994]_ and Gaberdiel–Kausch [GK1996]_ the quotient is
spanned by the images of ``M^ss ⊗ N^(d)``, ``M^ss = M / Σ v_(b) M``
(``b < 0``, Nahm's special subspace) and ``N^(d)`` = ``N`` modulo the modes of degree ``> d``; the
result is the image of their lifts in the window, so it maps onto
``(M ⊠ N)^(d)`` as soon as ``M^ss`` is spanned in the window.  Its
dimension can go up as well as down while ``max_depth`` grows (in small
windows: Ising ``ε ⊠ ε`` at depth 0 gives 1, 2, 2, 1, 1, 1 for
``max_depth`` 0 to 5): stability is the user's check, as for induced
modules of quotients.  The relations that cut the product down
come from the null vectors of the factors, so the window must also reach
the null vectors generating their maximal submodules (Eberle–Flohr's
remark on their ``L̃_max`` [EF2006]_): below that, a spurious state can survive
several consecutive windows (Virasoro at ``c = 0``: ``V(1/3) ⊠ V(21/8)``
keeps a weight ``143/24`` for ``max_depth`` 3 to 7, ``V(21/8)`` has its
null vector at level 8).  :meth:`FusionQuotientUpTo.special_subspace_dimensions`
shows ``M^ss`` by depth.

LATTICE VERTEX ALGEBRAS.  The algorithm needs a finite set of strong
generators, which a lattice vertex algebra does not have; there the answer
is known, ``V_{λ+L} ⊠ V_{μ+L} = V_{λ+μ+L}`` for every non-degenerate
integral lattice (e.g. Dong–Lepowsky [DL1993]_), and ``M.fusion_product(N)``
gives it (:class:`LatticeFusionProduct`: the product as a module, and
exact depth quotients with the interface of :class:`FusionQuotientUpTo`),
for simple modules, the adjoint module and direct sums of these.

Not yet: ``M ⊠ N`` itself as a module (except for lattice algebras),
infinite weight spaces (with gradings), twisted sectors.

EXAMPLES:

The Lee–Yang model: ``φ ⊠ φ = 1 + φ``, seen in the conformal weights of the
depth-0 quotient::

    sage: import vertex_algebras as va
    sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
    sage: phi = V.verma_module(-1/5).simple_quotient()
    sage: P = phi.fusion_product(phi); P
    Fusion product of Simple quotient of ... and Simple quotient of ...
    sage: F = P.depth_quotient_up_to(0, max_depth=3); F
    Depth-0 quotient of the fusion product of ... (from the tensors of
    total depth <= 3): dimension 2
    sage: F.conformal_weights()
    [(-1/5, [1]), (0, [1])]

Gaberdiel–Kausch's logarithmic fusion at ``c = -2`` [GK1996]_ ([KR2019]_ §8):
``L_0`` has a Jordan block on the depth-0 quotient of ``M ⊠ M`` for the
module ``M`` of weight ``-1/8``::

    sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
    sage: M = V.verma_module(-1/8).simple_quotient()
    sage: F = M.fusion_product(M).depth_quotient_up_to(0, max_depth=3)
    sage: F.conformal_weights()
    [(0, [2])]
    sage: M.fusion_product(M).depth_quotient_up_to(1, max_depth=4).conformal_weights()
    [(0, [2]), (1, [1])]

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from sage.arith.functions import lcm
from sage.functions.other import binomial, ceil, floor
from sage.matrix.constructor import matrix
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.structure.sage_object import SageObject

from ._keyed_spans import KeyedSpan


def _add(res, d, c):
    """
    ``res += c d`` on raw dicts.

    TESTS::

        sage: from vertex_algebras.fusion_product import _add
        sage: res = {'a': 1, 'b': 2}; _add(res, {'a': -1, 'c': 3}, 1); res
        {'b': 2, 'c': 3}
    """
    for k, v in d.items():
        r = res.get(k, 0) + c * v
        if r:
            res[k] = r
        else:
            res.pop(k, None)


def _tensor_into(res, dA, dB, c):
    """
    ``res += c dA ⊗ dB`` (keys ``(a, b)``).

    TESTS::

        sage: from vertex_algebras.fusion_product import _tensor_into
        sage: res = {('x', 'u'): -2}; _tensor_into(res, {'x': 1, 'y': 2}, {'u': 1}, 2); res
        {('y', 'u'): 4}
    """
    for a, ca in dA.items():
        for b, cb in dB.items():
            key = (a, b)
            r = res.get(key, 0) + c * ca * cb
            if r:
                res[key] = r
            else:
                res.pop(key, None)


_BINOMIALS = {}


def _binomial(n, k):
    """
    ``binomial(n, k)`` (Sage's, any integer ``n``), cached.

    TESTS::

        sage: from vertex_algebras.fusion_product import _binomial
        sage: _binomial(-3, 2), _binomial(5, 2), _binomial(2, 5)
        (6, 10, 0)
    """
    key = (n, k)
    c = _BINOMIALS.get(key)
    if c is None:
        c = _BINOMIALS[key] = binomial(n, k)
    return c


class FusionProduct(SageObject):
    r"""
    The fusion product ``M ⊠ N`` of two modules over the same vertex
    algebra (``M`` at ``1``, ``N`` at ``0``); see the module
    documentation.  Nothing is computed until a depth quotient is asked
    for (:meth:`depth_quotient_up_to`).  Built by ``M.fusion_product(N)``.

    INPUT:

    - ``M`` -- a module over a vertex algebra ``V`` (the factor at ``1``)
    - ``N`` -- a module over the same ``V`` (the factor at ``0``)

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
        sage: s = V.verma_module(1/16).simple_quotient()
        sage: P = s.fusion_product(s); P.factors() == (s, s)
        True

    TESTS::

        sage: W = va.Virasoro(QQ, 1/3, conformal_vector='L')
        sage: s.fusion_product(W.verma_module(0))
        Traceback (most recent call last):
        ...
        ValueError: the modules are modules over different vertex algebras

    The Ising fusion rules, in both orders (rows and columns ``1, ε, σ``)::

        sage: I = [V.verma_module(h).simple_quotient() for h in (0, 1/2, 1/16)]
        sage: def weights(A, B):
        ....:     F = A.fusion_product(B).depth_quotient_up_to(0, 3)
        ....:     return [h for h, _ in F.conformal_weights()]
        sage: [[weights(A, B) for B in I] for A in I]
        [[[0], [1/2], [1/16]], [[1/2], [0], [1/16]], [[1/16], [1/16], [0, 1/2]]]

    The induced modules without the C++ engine give the same depth-1
    quotient as the default ones (``c = -2``, ``W(-1/8) ⊠ W(-1/8)``)::

        sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
        sage: U = V.zhu_algebra().module({'L': -1/8})
        sage: Mp = V.induced_module(U, implementation='python').simple_quotient()
        sage: Me = V.induced_module(U).simple_quotient()
        sage: Fp = Mp.fusion_product(Mp).depth_quotient_up_to(1, 4)
        sage: Fe = Me.fusion_product(Me).depth_quotient_up_to(1, 4)
        sage: Fp.l0_matrix() == Fe.l0_matrix(), Fp.conformal_weights()
        (True, [(0, [2]), (1, [1])])
    """

    def __init__(self, M, N):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: s = V.verma_module(1/16).simple_quotient()
            sage: from vertex_algebras.fusion_product import FusionProduct
            sage: FusionProduct(s, s)._quotients
            {}
        """
        if M._V is not N._V:
            raise ValueError("the modules are modules over different vertex algebras")
        self._M, self._N = M, N
        self._quotients = {}

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: v = V.verma_module(0)
            sage: v.fusion_product(v)
            Fusion product of Verma module ... and Verma module ...
        """
        return "Fusion product of {} and {}".format(self._M, self._N)

    def factors(self):
        """
        The two modules ``(M, N)``.

        OUTPUT:

        A pair of modules.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: v = V.verma_module(0); v.fusion_product(v).factors() == (v, v)
            True
        """
        return (self._M, self._N)

    def depth_quotient_up_to(self, depth=0, max_depth=None):
        r"""
        The depth-``depth`` quotient of ``M ⊠ N`` computed from the tensors
        of total depth ``<= max_depth`` (see the module documentation: it
        maps onto the true quotient once Nahm's special subspace of ``M``
        lies in the window; check that it is stable as ``max_depth`` grows).

        INPUT:

        - ``depth`` -- (default: 0) a non-negative rational, the depth
          ``d`` of the quotient
        - ``max_depth`` -- (default: ``depth + 2``) the cutoff of the total
          depth of the tensors, ``>= depth``

        OUTPUT:

        A :class:`FusionQuotientUpTo` (cached per ``(depth, max_depth)``).

        EXAMPLES:

        The Ising model, ``σ ⊠ σ = 1 + ε``::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: s = V.verma_module(1/16).simple_quotient()
            sage: F = s.fusion_product(s).depth_quotient_up_to(1, max_depth=4)
            sage: F.dimension(), F.conformal_weights()
            (3, [(0, [1]), (1/2, [1]), (3/2, [1])])
            sage: s.fusion_product(s).depth_quotient_up_to(1, max_depth=4) is F
            True

        TESTS::

            sage: s.fusion_product(s).depth_quotient_up_to(2, max_depth=1)
            Traceback (most recent call last):
            ...
            ValueError: need 0 <= depth <= max_depth

        The result does not grow with the window; here it is the same for
        ``max_depth`` 1 to 6 (``c = -2``, ``W(-1/8) ⊠ W(-1/8)`` at depth 1)::

            sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
            sage: M = V.verma_module(-1/8).simple_quotient()
            sage: [M.fusion_product(M).depth_quotient_up_to(1, K).dimension() for K in range(1, 7)]
            [3, 3, 3, 3, 3, 3]
        """
        d = QQ(depth)
        K = d + 2 if max_depth is None else QQ(max_depth)
        if not 0 <= d <= K:
            raise ValueError("need 0 <= depth <= max_depth")
        key = (d, K)
        F = self._quotients.get(key)
        if F is None:
            F = FusionQuotientUpTo(self, d, K)
            self._quotients[key] = F
        return F


class _Side:
    """One factor: its keys by depth, its raw generator actions (memoised)."""

    def __init__(self, M, depths):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: from vertex_algebras.fusion_product import _Side
            sage: S = _Side(V.verma_module(1/16).simple_quotient(), [0, 1, 2])
            sage: [len(S.keys[t]) for t in (0, 1, 2)]
            [1, 1, 1]
        """
        self.M = M
        self.base = M._weight_base()
        self.depth = {}
        self.keys = {}
        for t in depths:
            ks = []
            for b in M.weight_space(self.base + t).basis():
                d = b.monomial_coefficients()
                if len(d) != 1:
                    raise NotImplementedError("modules whose weight spaces have no basis of basis keys")
                (k, c), = d.items()
                ks.append(k)
                self.depth[k] = t
            self.keys[t] = ks
        self._memo = {}

    def act(self, g, gd, n, k):
        """``g_(n) k`` (raw), memoised by the label ``g``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L'); L = V.gen('L')
            sage: from vertex_algebras.fusion_product import _Side
            sage: S = _Side(V.verma_module(1/16), [0, 1])
            sage: S.act('L', L.monomial_coefficients(), 1, ((), 0))
            {((), 0): 1/16}
        """
        key = (g, n, k)
        r = self._memo.get(key)
        if r is None:
            r = self.M._action_raw(gd, n, {k: 1})
            self._memo[key] = r
        return r


class FusionQuotientUpTo(SageObject):
    r"""
    The depth-``d`` quotient of a fusion product computed in a window of
    total depth ``K`` (see the module documentation and
    :meth:`FusionProduct.depth_quotient_up_to`).

    Its basis is a set of representatives ``x ⊗ y`` (pairs of basis keys
    of ``M`` and ``N``); the modes of degree ``0 .. d`` act on it.

    Built by :meth:`FusionProduct.depth_quotient_up_to`.

    INPUT:

    - ``P`` -- a :class:`FusionProduct`
    - ``d`` -- a non-negative rational number, the depth of the quotient
    - ``K`` -- a rational number ``>= d``, the cutoff of the total depth
      of the tensors

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
        sage: M = V.verma_module(-1/8).simple_quotient()
        sage: F = M.fusion_product(M).depth_quotient_up_to(0, max_depth=3)
        sage: F.basis()
        [(v, v), (L_{-1} v, v)]
        sage: F.l0_matrix()                 # Kanade-Ridout (8.7), sign corrected
        [ -1/4 -1/16]
        [    1   1/4]

    TESTS:

    The vacuum module is the unit, on both sides (depths 0, 1, 2)::

        sage: one = V.verma_module(0).simple_quotient()
        sage: [one.fusion_product(M).depth_quotient_up_to(d, 5).conformal_weights() for d in (0, 1, 2)]
        [[(-1/8, [1])], [(-1/8, [1]), (7/8, [1])], [(-1/8, [1]), (7/8, [1]), (15/8, [1])]]
        sage: [M.fusion_product(one).depth_quotient_up_to(d, 5).dimension() for d in (0, 1, 2)]
        [1, 2, 3]

    Kanade–Ridout's spurious state ([KR2019]_ (8.12), with the term ``-1/8 γ ⊗
    γ`` that their (8.2) gives; the printed (8.7) and (8.10) carry sign
    slips) lies in the relations at depth 1::

        sage: F1 = M.fusion_product(M).depth_quotient_up_to(1, max_depth=5)
        sage: g0, g1 = ((), 0), (((0, -1),), 0)
        sage: F1._E.normal_form({(g1, g1): 2, (g1, g0): -1/2, (g0, g1): 1/2, (g0, g0): -1/8})
        {}
        sage: F1.dimension()
        3

    Lee–Yang at depth 2 is ``L(0) + L(-1/5)``, weight by weight::

        sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
        sage: phi = V.verma_module(-1/5).simple_quotient()
        sage: phi.fusion_product(phi).depth_quotient_up_to(2, 6).conformal_weights()
        [(-1/5, [1]), (0, [1]), (4/5, [1]), (9/5, [1]), (2, [1])]
    """

    def __init__(self, P, d, K):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 3); F.dimension()
            1
        """
        M, N = P.factors()
        V = M._V
        self._P, self._V, self._d, self._K = P, V, d, K
        gens = V._strong_generators_for_ideals()
        self._gens = [(('g', i), g.monomial_coefficients(), QQ(V.degree(g)), g.parity())
                      for i, g in enumerate(gens)]
        D = lcm([g[2].denominator() for g in self._gens] + [1])
        self._step = QQ(1) / D
        self._depths = [i * self._step for i in range(floor(K / self._step) + 1)]
        self._A = _Side(M, self._depths)
        self._B = _Side(N, self._depths)
        # the coproduct's window tests in integers: depths and degrees are
        # multiples of 1/D (the Sage rationals cost most of its time)
        self._D = int(D)
        self._Ks = int(floor(K * D))
        self._hs = {}                               # label -> h D (or None)
        for S in (self._A, self._B):
            S.sdepth = {k: int(t * D) for k, t in S.depth.items()}
        self._parity = {}
        self._R = V.base_ring()
        self._build()

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(0, 3)
            Depth-0 quotient of the fusion product of ... (from the tensors of
            total depth <= 3): dimension 1
        """
        M, N = self._P.factors()
        return ("Depth-{} quotient of the fusion product of {} and {} (from the tensors "
                "of total depth <= {}): dimension {}").format(
                    self._d, M, N, self._K, self.dimension())

    # ---- the coproduct --------------------------------------------------
    def _sign(self, p, a):
        """The Koszul sign of moving a mode of parity ``p`` past ``a``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(0, 2)._sign(1, ((), 0))
            1
        """
        if not p:
            return 1
        q = self._parity.get(a)
        if q is None:
            q = self._parity[a] = self._A.M._parity_key(a)
        return -1 if q else 1

    def _scaled_weight(self, g):
        """``h D`` for the field ``g`` of weight ``h`` (an integer), or
        ``None`` if ``h`` is not a multiple of ``1/D`` (then the coproduct
        computes with rationals).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: F._scaled_weight(F._gens[0]), F._scaled_weight(('x', {}, 1/3, 0))
            (2, None)
        """
        i = g[0]
        try:
            return self._hs[i]
        except KeyError:
            hs = g[2] * self._D
            hs = int(hs) if hs in ZZ else None
            self._hs[i] = hs
            return hs

    def _delta_mode(self, g, k, x):
        r"""
        ``Δ(v_(k)) x`` for a basis tensor ``x = (a, b)``, or ``None`` if a
        term would leave the window.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: x = (((), 0), ((), 0))
            sage: F._delta_mode(F._gens[0], 1, x)          # L_{-1} ⊗ 1 + L_0 ⊗ 1 + 1 ⊗ L_0
            {(((), 0), ((), 0)): 1, ((((0, -1),), 0), ((), 0)): 1}
            sage: F._delta_mode(F._gens[0], -5, x) is None
            True

        ``Δ`` is a homomorphism of mode algebras: the commutator formula
        holds for ``L``, ``∂L`` and ``:LL:`` on all tensors of the window
        where it can be evaluated (Lee-Yang, ``φ ⊠ φ``)::

            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: F = phi.fusion_product(phi).depth_quotient_up_to(1, 5)
            sage: def g(v):
            ....:     d = v.monomial_coefficients()
            ....:     return (('e', tuple(sorted(d.items(), key=repr))), d, QQ(V.degree(v)), v.parity())
            sage: def delta(v, k, vec):
            ....:     res = {}
            ....:     for x, c in vec.items():
            ....:         r = F._delta_mode(g(v), k, x)
            ....:         if r is None:
            ....:             return None
            ....:         for y, cc in r.items():
            ....:             res[y] = res.get(y, 0) + c * cc
            ....:     return {y: c for y, c in res.items() if c}
            sage: els = [L, L.derivative(), L.no(L)]
            sage: xs = [x for x, room in F._tensors() if room >= 2]
            sage: tried = bad = 0
            sage: for u, w, m, n, x in cartesian_product([els, els, range(-1, 4), range(-1, 4), xs]):
            ....:     y1, y2 = delta(w, n, {x: 1}), delta(u, m, {x: 1})
            ....:     if y1 is None or y2 is None:
            ....:         continue
            ....:     l1, l2 = delta(u, m, y1), delta(w, n, y2)
            ....:     rhs = [delta(u.nproduct(w, j), m + n - j, {x: binomial(m, j)})
            ....:            for j in range(6) if u.nproduct(w, j)]
            ....:     if l1 is None or l2 is None or any(r is None for r in rhs):
            ....:         continue
            ....:     tried += 1
            ....:     diff = dict(l1)
            ....:     for r in [l2] + rhs:
            ....:         for y, c in r.items():
            ....:             diff[y] = diff.get(y, 0) - c
            ....:     bad += any(diff.values())
            sage: tried, bad
            (728, 0)

        The pole relations of the composite fields ``∂L`` and ``:LL:``
        follow from those of the generator: they lie in the relations::

            sage: total = missing = 0
            sage: for v in [L.derivative(), L.no(L)]:
            ....:     G = g(v); h = G[2]
            ....:     for x, room in F._tensors():
            ....:         bb = -1
            ....:         while h - 1 - bb <= room:
            ....:             r = F._delta_pole(G, bb, x)
            ....:             for m in range(floor(F._d - (h - 1 - bb)) + 1):
            ....:                 rm = None if r is None else F._delta_mode(G, bb - m, x)
            ....:                 if rm is None:
            ....:                     r = None
            ....:                     break
            ....:                 for y, c in rm.items():
            ....:                     r[y] = r.get(y, 0) - binomial(bb, m) * (-1)**m * c
            ....:             if r is not None and any(r.values()):
            ....:                 total += 1
            ....:                 missing += bool(F._E.normal_form({y: c for y, c in r.items() if c}))
            ....:             bb -= 1
            sage: total, missing
            (14, 0)
        """
        hs = self._scaled_weight(g)
        if hs is None:
            return self._delta_mode_q(g, k, x)
        i, gd, h, p = g
        a, b = x
        A, B, D, Ks = self._A, self._B, self._D, self._Ks
        ta, tb = A.sdepth[a], B.sdepth[b]          # depths and h, times D
        top = ta + hs - D                           # (depth(a) + h - 1) D
        res = {}
        for j in range(top // D + 1):               # v_(j) a = 0 beyond
            c = _binomial(k, j)
            if c:
                if top - j * D + tb > Ks:
                    return None
                for a2, ca in A.act(i, gd, j, a).items():
                    key = (a2, b)
                    r = res.get(key, 0) + c * ca
                    if r:
                        res[key] = r
                    else:
                        res.pop(key, None)
        e = tb + top - ta - k * D                   # (depth(b) + h - 1 - k) D
        if e >= 0:
            if ta + e > Ks:
                return None
            s = self._sign(p, a)
            for b2, cb in B.act(i, gd, k, b).items():
                key = (a, b2)
                r = res.get(key, 0) + s * cb
                if r:
                    res[key] = r
                else:
                    res.pop(key, None)
        return res

    def _delta_pole(self, g, bb, x):
        r"""
        ``Δ(v (t-1)^bb) x`` for ``bb < 0`` (``v_(bb)`` on ``M`` plus the
        expansion at 0 on ``N``), or ``None`` if a term would leave the
        window.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 3)
            sage: F._delta_pole(F._gens[0], -1, (((), 0), ((), 0)))
            {(((), 0), ((), 0)): -1/2,
             (((), 0), (((0, -1),), 0)): -1,
             ((((0, -2),), 0), ((), 0)): 1}
        """
        hs = self._scaled_weight(g)
        if hs is None:
            return self._delta_pole_q(g, bb, x)
        i, gd, h, p = g
        a, b = x
        A, B, D, Ks = self._A, self._B, self._D, self._Ks
        ta, tb = A.sdepth[a], B.sdepth[b]          # depths and h, times D
        if ta + hs - D - bb * D + tb > Ks:
            return None
        res = {}
        for a2, ca in A.act(i, gd, bb, a).items():
            if ca:
                res[(a2, b)] = ca
        s = self._sign(p, a)
        top = tb + hs - D                           # (depth(b) + h - 1) D
        for k in range(top // D + 1):
            c = _binomial(bb, k) * (-1) ** ((bb - k) % 2)
            if c:
                if ta + top - k * D > Ks:
                    return None
                sc = s * c
                for b2, cb in B.act(i, gd, k, b).items():
                    key = (a, b2)
                    r = res.get(key, 0) + sc * cb
                    if r:
                        res[key] = r
                    else:
                        res.pop(key, None)
        return res

    def _delta_mode_q(self, g, k, x):
        """:meth:`_delta_mode` with rational depths (any weight).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: F = phi.fusion_product(phi).depth_quotient_up_to(1, 5)
            sage: all(F._delta_mode_q(g, k, x) == F._delta_mode(g, k, x)
            ....:     for g in F._gens for k in range(-4, 4) for x, _ in F._tensors())
            True

        With odd generators (Koszul signs): symplectic fermions, the module
        induced from the Grassmann top ``C^{2|2}``::

            sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
            sage: Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
            sage: SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp)
            sage: Ep = matrix(QQ, 4, 4, {(1, 0): 1, (3, 2): 1})
            sage: Em = matrix(QQ, 4, 4, {(2, 0): 1, (3, 1): -1})
            sage: G = SF.induced_module(SF.zhu_algebra().module({'cp': Ep, 'cm': Em},
            ....:                                               parities=[0, 1, 1, 0]))
            sage: F = G.fusion_product(G).depth_quotient_up_to(1, 3)
            sage: all(F._delta_mode_q(g, k, x) == F._delta_mode(g, k, x)
            ....:     for g in F._gens for k in range(-3, 3) for x, _ in F._tensors())
            True
            sage: all(F._delta_pole_q(g, bb, x) == F._delta_pole(g, bb, x)
            ....:     for g in F._gens for bb in range(-4, 0) for x, _ in F._tensors())
            True
        """
        i, gd, h, p = g
        a, b = x
        A, B, K = self._A, self._B, self._K
        ta, tb = A.depth[a], B.depth[b]
        res = {}
        for j in range(floor(ta + h - 1) + 1):      # v_(j) a = 0 beyond
            c = _binomial(k, j)
            if c:
                if ta + h - 1 - j + tb > K:
                    return None
                _tensor_into(res, A.act(i, gd, j, a), {b: 1}, c)
        if tb + h - 1 - k >= 0:
            if ta + tb + h - 1 - k > K:
                return None
            _tensor_into(res, {a: 1}, B.act(i, gd, k, b), self._sign(p, a))
        return res

    def _delta_pole_q(self, g, bb, x):
        """:meth:`_delta_pole` with rational depths (any weight).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: F = phi.fusion_product(phi).depth_quotient_up_to(1, 5)
            sage: all(F._delta_pole_q(g, bb, x) == F._delta_pole(g, bb, x)
            ....:     for g in F._gens for bb in range(-5, 0) for x, _ in F._tensors())
            True
        """
        i, gd, h, p = g
        a, b = x
        A, B, K = self._A, self._B, self._K
        ta, tb = A.depth[a], B.depth[b]
        if ta + h - 1 - bb + tb > K:
            return None
        res = {}
        _tensor_into(res, A.act(i, gd, bb, a), {b: 1}, 1)
        s = self._sign(p, a)
        for k in range(floor(tb + h - 1) + 1):
            c = _binomial(bb, k) * (-1) ** ((bb - k) % 2)
            if c:
                if ta + tb + h - 1 - k > K:
                    return None
                _tensor_into(res, {a: 1}, B.act(i, gd, k, b), s * c)
        return res

    def _delta_vec(self, g, k, vec):
        """``Δ(v_(k))`` on a vector of the window (``None`` if a term
        leaves it).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: F._delta_vec(F._gens[0], 2, {(((), 0), ((), 0)): 3})     # Δ(L_1) = L_{-1} + 2 L_0 + L_1 ⊗ 1 + 1 ⊗ L_1
            {(((), 0), ((), 0)): 3, ((((0, -1),), 0), ((), 0)): 3}
        """
        res = {}
        for x, c in vec.items():
            r = self._delta_mode(g, k, x)
            if r is None:
                return None
            _add(res, r, c)
        return res

    # ---- the window -----------------------------------------------------
    def _tensors(self):
        """The basis tensors of the window with the depth left above them.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: [room for x, room in F._tensors()]
            [2, 1, 0, 1, 0, 0]
        """
        A, B, K = self._A, self._B, self._K
        for ta in self._depths:
            for tb in self._depths:
                if ta + tb > K:
                    break
                for a in A.keys[ta]:
                    for b in B.keys[tb]:
                        yield (a, b), K - ta - tb

    def _degrees(self, h, lo, hi):
        """The degrees ``h - 1 - k`` (``k`` an integer) in ``(lo, hi]``,
        decreasing.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: F._degrees(3/2, 0, 2), F._degrees(2, -1, 1)
            ([3/2, 1/2], [1, 0])
        """
        kmin = ceil(h - 1 - hi)
        kmax = floor(h - 1 - lo)
        if h - 1 - kmax == lo:
            kmax -= 1
        return [h - 1 - k for k in range(kmin, kmax + 1)]

    def _key_order(self, x):
        """Leading tensors: larger total depth, then larger depth on ``N``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: F = e.fusion_product(e).depth_quotient_up_to(0, 2)
            sage: F._key_order((((), 0), ((), 0)))[:2]
            (0, 0)
        """
        a, b = x
        A, B = self._A, self._B
        return (A.depth[a] + B.depth[b], B.depth[b],
                A.M._key_order(a), B.M._key_order(b))

    def _depth_words(self):
        r"""
        The rows of ``D_d`` in the window: the words of generator modes of
        total degree ``> d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: len(e.fusion_product(e).depth_quotient_up_to(0, 2)._depth_words())
            4
        """
        D = {}
        for e in [t for t in self._depths if t <= self._d]:
            vecs = []
            for x, room in self._tensors():
                for g in self._gens:
                    for deg in self._degrees(g[2], e, room):
                        r = self._delta_vec(g, g[2] - 1 - deg, {x: 1})
                        if r:
                            vecs.append(r)
            for g in self._gens:
                for deg in self._degrees(g[2], 0, e):
                    for y in D[e - deg]:
                        r = self._delta_vec(g, g[2] - 1 - deg, y)
                        if r:
                            vecs.append(r)
            E = KeyedSpan(self._R, key_order=self._key_order)
            E.insert_many(vecs, greedy=False)       # only the span (its rows) matters
            D[e] = E.rows()
        return D[self._d]

    def _pole_relations(self):
        r"""
        The pole relations of the strong generators, modulo ``D_d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: len(e.fusion_product(e).depth_quotient_up_to(0, 2)._pole_relations())
            1
        """
        d = self._d
        rels = []
        for x, room in self._tensors():
            for g in self._gens:
                h = g[2]
                bb = -1
                while h - 1 - bb <= room:       # v_(bb) raises M by h - 1 - bb
                    r = self._delta_pole(g, bb, x)
                    for m in range(floor(d - (h - 1 - bb)) + 1):
                        if r is None:
                            break
                        rm = self._delta_mode(g, bb - m, x)
                        if rm is None:
                            r = None
                        else:
                            _add(r, rm, -_binomial(bb, m) * (-1) ** (m % 2))
                    if r:
                        rels.append(r)
                    bb -= 1
        return rels

    def _relations(self):
        r"""
        The relations of the window: ``D_d`` and the pole relations, the
        latter closed under the generator modes of degree ``0 .. d``.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(0, 2)._relations().rank()
            4
        """
        rels = self._pole_relations()
        E = KeyedSpan(self._R, key_order=self._key_order)
        E.insert_many(self._depth_words(), greedy=False)
        # the closure in rounds, one batch per round (single membership
        # tests were dense solves: 131 of 137 s at c = 0, depth 2, window 9)
        todo = rels
        while todo:
            flags = E.insert_many(todo, greedy=False)    # only the span matters
            new = [y for y, f in zip(todo, flags) if f]
            todo = []
            for y in new:
                for g in self._gens:
                    for deg in self._degrees(g[2], -1, self._d):
                        if deg < 0:
                            continue
                        r = self._delta_vec(g, g[2] - 1 - deg, y)
                        if r:
                            todo.append(r)
        return E

    def _complement(self, side, modes):
        r"""
        Basis keys of ``side`` (by depth) complementing the span of the
        images of the generator modes ``modes(h)`` (math indices).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
            sage: M = V.verma_module(-1/8).simple_quotient()
            sage: F = M.fusion_product(M).depth_quotient_up_to(0, 3)
            sage: {t: len(ks) for t, ks in F._ss.items()}        # γ, L_{-1} γ
            {0: 1, 1: 1, 2: 0, 3: 0}
        """
        M = side.M
        E = KeyedSpan(self._R, key_order=M._key_order)
        vecs = []
        for t in self._depths:
            for k in side.keys[t]:
                for g in self._gens:
                    i, gd, h, p = g
                    for n in modes(h):
                        if t + h - 1 - n <= self._K:
                            r = side.act(i, gd, n, k)
                            if r:
                                vecs.append(r)
        E.insert_many(vecs, greedy=False)           # only the span (its pivots) matters
        piv = set(E.pivots())
        return {t: [k for k in side.keys[t] if k not in piv] for t in self._depths}

    def _depth_complement(self, side, d):
        r"""
        Basis keys of ``side`` (by depth) complementing the words of
        generator modes of total degree ``> d`` (``N^(d)``; single modes
        are not enough: ``L_{-1}^3 v`` is no ``L_{-n} w``, ``n > 2``).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 0, conformal_vector='L')
            sage: M = V.verma_module(5/8).simple_quotient()
            sage: F = M.fusion_product(M).depth_quotient_up_to(2, 5)
            sage: {t: len(ks) for t, ks in F._nd.items()}
            {0: 1, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
        """
        K = self._K
        D = {}
        for e in [t for t in self._depths if t <= d]:
            vecs = []
            for t in self._depths:
                for k in side.keys[t]:
                    for g in self._gens:
                        i, gd, h, p = g
                        for deg in self._degrees(h, e, K - t):
                            r = side.act(i, gd, h - 1 - deg, k)
                            if r:
                                vecs.append(r)
            for g in self._gens:
                i, gd, h, p = g
                for deg in self._degrees(h, 0, e):
                    for y in D[e - deg]:
                        if max(side.depth[k] for k in y) + deg <= K:
                            r = side.M._action_raw(gd, h - 1 - deg, y)
                            if r:
                                vecs.append(r)
            E = KeyedSpan(self._R, key_order=side.M._key_order)
            E.insert_many(vecs, greedy=False)       # only the span (its rows) matters
            D[e] = E.rows()
        piv = set(E.pivots())
        return {t: [k for k in side.keys[t] if k not in piv] for t in self._depths}

    def _build(self):
        r"""
        Relations, Nahm's special subspace of ``M``, ``N^(d)``, and the
        independent representatives.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(0, 3)._basis
            [(((), 0), ((), 0))]

        A window too small to reduce ``L_{-1} ε ⊗ ε`` keeps it::

            sage: e.fusion_product(e).depth_quotient_up_to(0, 2)._basis
            [(((), 0), ((), 0)), ((((0, -1),), 0), ((), 0))]
        """
        K, d = self._K, self._d
        self._ss = self._complement(self._A, lambda h: range(-1, -floor(K) - 2, -1))
        self._nd = self._depth_complement(self._B, d)
        self._E = self._relations()
        reps = [(a, b) for ta in self._depths for a in self._ss[ta]
                for tb in self._depths if ta + tb <= K for b in self._nd[tb]]
        F = KeyedSpan(self._R)
        basis = []
        for x in reps:
            nf = self._E.normal_form({x: 1})
            if nf and F.insert(nf, label=x):
                basis.append(x)
        self._F = F
        self._basis = basis

    # ---- the result -----------------------------------------------------
    def dimension(self):
        """
        The dimension.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: s, e = [V.verma_module(h).simple_quotient() for h in (1/16, 1/2)]
            sage: s.fusion_product(e).depth_quotient_up_to(0, 3).dimension()
            1
        """
        return len(self._basis)

    def depth(self):
        """
        The depth ``d`` of the quotient.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(1, 3).depth()
            1
        """
        return self._d

    def fusion_product(self):
        """
        The fusion product this is a quotient of.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient(); P = e.fusion_product(e)
            sage: P.depth_quotient_up_to(0, 2).fusion_product() is P
            True
        """
        return self._P

    def basis(self):
        r"""
        The representatives ``(x, y)`` of the basis: ``x ⊗ y`` with ``x``
        in ``M`` and ``y`` in ``N`` (vectors of the two modules).

        OUTPUT:

        A list of pairs ``(x, y)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: phi.fusion_product(phi).depth_quotient_up_to(0, 3).basis()
            [(v, v), (L_{-1} v, v)]
        """
        M, N = self._P.factors()
        return [(M.monomial(a), N.monomial(b)) for a, b in self._basis]

    def _coordinates(self, vec):
        """The coordinates of a vector of the window in the basis.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: F = phi.fusion_product(phi).depth_quotient_up_to(0, 3)
            sage: F._coordinates({(((), 0), (((0, -1),), 0)): 1})       # v ⊗ L_{-1} v
            [0, -1]
            sage: F._coordinates(None)
            Traceback (most recent call last):
            ...
            ValueError: the action leaves the window: increase max_depth
        """
        if vec is None:
            raise ValueError("the action leaves the window: increase max_depth")
        nf = self._E.normal_form(vec)
        c = self._F.coordinates(nf) if nf else {}
        if c is None:
            raise ValueError("the result is not in the span of the representatives: increase max_depth")
        return [c.get(x, 0) for x in self._basis]

    def mode_matrix(self, a, n):
        r"""
        The matrix (columns: images of the basis) of ``Δ(a_(n))``, ``a`` a
        homogeneous element of the vertex algebra of weight ``h``, of degree
        ``k = h - 1 - n``.

        Modes of degree ``0 <= k <= d`` act on the depth-``d`` quotient
        (they keep the words of degree ``> d``). A lowering mode, ``-d <= k
        < 0``, maps the words of degree ``> d`` into those of degree ``> d +
        k``, so it is a map to the depth-``(d + k)`` quotient (with the same
        window): then the rows are coordinates there.

        INPUT:

        - ``a`` -- a homogeneous element of the vertex algebra
        - ``n`` -- an integer, the mode (``a_(n)``)

        OUTPUT:

        A matrix over the base ring, with ``dimension()`` columns.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -22/5, conformal_vector='L'); L = V.gen('L')
            sage: phi = V.verma_module(-1/5).simple_quotient()
            sage: F = phi.fusion_product(phi).depth_quotient_up_to(1, 4)
            sage: F.mode_matrix(L, 0).rank()                  # L_{-1}
            1
            sage: F.mode_matrix(L, 2).dimensions()            # L_1: to the depth-0 quotient
            (2, 3)

        The logarithmic coupling ``β = 5/6`` of the rank-2 module ``R(0, 2)_7
        = V(5/8) ⊠ V(5/8)`` at ``c = 0`` ([EF2006]_, Table 2): ``ξ`` with
        ``(L_0 - 2) ξ = L_{-2} Ω`` and ``L_2 ξ = β Ω``::

            sage: V = va.Virasoro(QQ, 0, conformal_vector='L'); L = V.gen('L')
            sage: M = V.verma_module(5/8).simple_quotient()
            sage: P = M.fusion_product(M)
            sage: F2, F0 = P.depth_quotient_up_to(2, 6), P.depth_quotient_up_to(0, 6)
            sage: L0 = F2.l0_matrix(); Omega = L0.right_kernel().basis()[0]
            sage: psi = F2.mode_matrix(L, -1) * Omega                       # L_{-2} Ω
            sage: xi = (L0 - 2).solve_right(psi)
            sage: b = F2.mode_matrix(L, 3) * xi; o = F2.projection(0) * Omega  # L_2 ξ, Ω
            sage: matrix([b, o]).rank(), b[0] / o[0]
            (1, 5/6)

        TESTS::

            sage: F2.mode_matrix(L, -2)
            Traceback (most recent call last):
            ...
            ValueError: the modes of degree 3 do not act on the depth-2 quotient (degrees -2 .. 2)
        """
        V = self._V
        a = V(a)
        h = QQ(V.degree(a))
        deg = h - 1 - n
        if not -self._d <= deg <= self._d:
            raise ValueError("the modes of degree {} do not act on the depth-{} quotient (degrees {} .. {})".format(
                deg, self._d, -self._d, self._d))
        target = self if deg >= 0 else self._P.depth_quotient_up_to(self._d + deg, self._K)
        da = a.monomial_coefficients()
        g = (('e', tuple(sorted(da.items(), key=repr))), da, h, a.parity())
        cols = [target._coordinates(self._delta_vec(g, n, {x: 1})) for x in self._basis]
        return matrix(self._R, self.dimension(), target.dimension(), cols).transpose()

    def projection(self, depth):
        r"""
        The matrix of the natural surjection onto the quotient of depth
        ``depth <= d`` (with the same window).

        INPUT:

        - ``depth`` -- a rational number with ``0 <= depth <= d``

        OUTPUT:

        A matrix over the base ring (columns: images of the basis).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
            sage: M = V.verma_module(-1/8).simple_quotient()
            sage: F1 = M.fusion_product(M).depth_quotient_up_to(1, 5)
            sage: F1.projection(0).dimensions(), F1.projection(0).rank()
            ((2, 3), 2)
        """
        depth = QQ(depth)
        if not 0 <= depth <= self._d:
            raise ValueError("need 0 <= depth <= {}".format(self._d))
        target = self if depth == self._d else self._P.depth_quotient_up_to(depth, self._K)
        cols = [target._coordinates({x: 1}) for x in self._basis]
        return matrix(self._R, self.dimension(), target.dimension(), cols).transpose()

    def zero_mode_matrix(self, a):
        r"""
        The matrix of the zero mode ``Δ(a_(h - 1))`` of a homogeneous
        element ``a`` of integer weight ``h``.

        INPUT:

        - ``a`` -- a homogeneous element of the vertex algebra of integer
          weight

        OUTPUT:

        A square matrix over the base ring (see :meth:`mode_matrix`).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: Z = V.zhu_algebra()
            sage: U = Z.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: D = V.induced_module(U).simple_quotient()
            sage: F = D.fusion_product(D).depth_quotient_up_to(0, 3)
            sage: F.dimension(), F.zero_mode_matrix(V.gen('H'))
            (1, [0])

        TESTS:

        The vacuum module is a unit on both sides of the doublet, the
        depth-0 quotients are ``U(sl_2)``-modules, and the doublet squared
        has depth-1 part the adjoint representation::

            sage: Z1 = Z.module({'E': matrix(QQ, 1), 'H': matrix(QQ, 1), 'F': matrix(QQ, 1)})
            sage: O = V.induced_module(Z1).simple_quotient()
            sage: Fod = O.fusion_product(D).depth_quotient_up_to(0, 3)
            sage: Fdo = D.fusion_product(O).depth_quotient_up_to(0, 3)
            sage: Fod.conformal_weights(), Fdo.conformal_weights()
            ([(1/4, [1, 1])], [(1/4, [1, 1])])
            sage: Fod.zero_mode_matrix(V.gen('H')), sorted(Fod.horizontal_grades())
            (
            [ 1  0]
            [ 0 -1], [(-1,), (1,)]
            )
            sage: Fod.zhu_algebra_module(Z).dimension(), Fdo.zhu_algebra_module(Z).dimension()
            (2, 2)
            sage: D.fusion_product(D).depth_quotient_up_to(1, 4).conformal_weights()
            [(0, [1]), (1, [1, 1, 1])]
        """
        V = self._V
        a = V(a)
        return self.mode_matrix(a, QQ(V.degree(a)) - 1)

    def l0_matrix(self):
        r"""
        The matrix of ``Δ(L_0) = L_{-1} ⊗ 1 + L_0 ⊗ 1 + 1 ⊗ L_0``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: s = V.verma_module(1/16).simple_quotient()
            sage: L0 = s.fusion_product(s).depth_quotient_up_to(0, 3).l0_matrix(); L0
            [ 1/8 3/64]
            [   1  3/8]
            sage: L0.eigenvalues()
            [1/2, 0]
        """
        T = self._V.conformal_vector()
        if T is None:
            raise ValueError("the vertex algebra has no conformal vector")
        return self.zero_mode_matrix(T)

    def conformal_weights(self):
        r"""
        The generalised eigenvalues of ``L_0`` with the sizes of their
        Jordan blocks, ``[(h, [sizes]), ...]`` (sorted).

        OUTPUT:

        A list of pairs ``(h, sizes)``, ``sizes`` a decreasing list of
        integers.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
            sage: M = V.verma_module(-1/8).simple_quotient()
            sage: M.fusion_product(M).depth_quotient_up_to(0, 3).conformal_weights()
            [(0, [2])]
        """
        L0 = self.l0_matrix()
        n = L0.nrows()
        if not n:
            return []
        p = L0.charpoly()
        roots = p.roots()
        if sum(m for _, m in roots) != n:
            raise NotImplementedError("L_0 has eigenvalues outside the base ring")
        out = []
        for lam, m in roots:
            X = L0 - lam
            ranks = [n]
            P = X.parent().identity_matrix()
            while ranks[-1] > n - m:
                P = P * X
                ranks.append(P.rank())
            # blocks of size >= k: ranks[k-1] - ranks[k]
            ge = [ranks[k - 1] - ranks[k] for k in range(1, len(ranks))] + [0]
            sizes = []
            for k in range(1, len(ge)):
                sizes += [k] * (ge[k - 1] - ge[k])
            out.append((lam, sorted(sizes, reverse=True)))
        try:
            out.sort()
        except TypeError:
            pass
        return out

    def parities(self):
        """
        The parities of the basis tensors.

        OUTPUT:

        A list of integers ``0`` or ``1``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.NeveuSchwarz(QQ, 7/10, conformal_vector='L')
            sage: tau = V.verma_module(1/10).simple_quotient()
            sage: F = tau.fusion_product(tau).depth_quotient_up_to(1/2, 3)
            sage: F.conformal_weights(), F.parities()
            ([(0, [1]), (1/10, [1]), (3/5, [1])], [0, 1, 0])

        TESTS:

        The vacuum is a unit on both sides of ``τ``::

            sage: vac = V.verma_module(0).simple_quotient()
            sage: all(vac.fusion_product(tau).depth_quotient_up_to(d, 3).conformal_weights()
            ....:     == tau.fusion_product(vac).depth_quotient_up_to(d, 3).conformal_weights()
            ....:     == [(1/10, [1]), (3/5, [1]), (11/10, [1])][:2*d + 1] for d in (0, 1/2, 1))
            True

        Symplectic fermions: the vacuum module and the module induced from
        the Grassmann top ``C^{2|2}`` fuse to the same space in both orders,
        with the same parities and Jordan blocks::

            sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
            sage: Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
            sage: SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp,
            ....:                             horizontal_grading={'cp': 1, 'cm': -1})
            sage: A = SF.zhu_algebra()
            sage: Ep = matrix(QQ, 4, 4, {(1, 0): 1, (3, 2): 1})
            sage: Em = matrix(QQ, 4, 4, {(2, 0): 1, (3, 1): -1})
            sage: G = SF.induced_module(A.module({'cp': Ep, 'cm': Em}, parities=[0, 1, 1, 0],
            ....:                                horizontal_grades=[0, 1, -1, 0]))
            sage: Vs = SF.induced_module(A.module({'cp': matrix(QQ, 1), 'cm': matrix(QQ, 1)}))
            sage: F1 = Vs.fusion_product(G).depth_quotient_up_to(1, 3)
            sage: F2 = G.fusion_product(Vs).depth_quotient_up_to(1, 3)
            sage: F1.dimension(), F2.dimension(), sorted(F1.parities()) == sorted(F2.parities())
            (12, 12, True)
            sage: sorted(F1.parities())
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            sage: F1.conformal_weights() == F2.conformal_weights()
            True
            sage: F1.conformal_weights()
            [(0, [2, 1, 1]), (1, [2, 2, 1, 1, 1, 1])]
        """
        A, B = self._A.M, self._B.M
        return [(A._parity_key(a) + B._parity_key(b)) % 2 for a, b in self._basis]

    def horizontal_grades(self):
        r"""
        The horizontal grades (coordinate tuples) of the basis tensors, the
        sums of the factors' grades, or ``None`` if a factor is ungraded or
        their grading groups differ.

        OUTPUT:

        A list of tuples, or ``None``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
            sage: Z = V.zhu_algebra()
            sage: U = Z.module({'E': matrix([[0, 1], [0, 0]]), 'H': matrix([[1, 0], [0, -1]]),
            ....:               'F': matrix([[0, 0], [1, 0]])})
            sage: D = V.induced_module(U).simple_quotient()
            sage: D.fusion_product(D).depth_quotient_up_to(0, 3).horizontal_grades()
            [(0,)]
        """
        A, B = self._A.M, self._B.M
        if A._mgrading is None or B._mgrading is None:
            return None
        if A.horizontal_grading_group() != B.horizontal_grading_group():
            return None
        out = []
        for a, b in self._basis:
            ga, gb = A._grade_key(a), B._grade_key(b)
            out.append(tuple(x + y for x, y in zip(ga, gb)))
        return out

    def special_subspace_dimensions(self):
        r"""
        The dimensions of Nahm's special subspace ``M / Σ v_(b) M``
        (``b < 0``) of the first factor by depth, as far as the window
        sees it (a diagnostic: the result is reliable once these vanish
        well below ``max_depth``).

        OUTPUT:

        A dictionary ``{depth: dimension}``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Virasoro(QQ, -2, conformal_vector='L')
            sage: M = V.verma_module(-1/8).simple_quotient()
            sage: M.fusion_product(M).depth_quotient_up_to(0, 3).special_subspace_dimensions()
            {0: 1, 1: 1, 2: 0, 3: 0}
        """
        return {t: len(ks) for t, ks in self._ss.items()}

    def zhu_algebra_module(self, A):
        r"""
        The depth-0 quotient as a module over the Zhu algebra ``A`` of the
        vertex algebra (or of a quotient), through the zero modes of the
        strong generators named by ``A``; the module checks the relations
        of ``A``.

        INPUT:

        - ``A`` -- a Zhu algebra of the vertex algebra (or of a quotient)
          whose generator names are names of strong generators

        OUTPUT:

        A module over ``A`` (``A.module(...)``), graded when
        :meth:`horizontal_grades` is not ``None``.

        EXAMPLES:

        The triplet algebra ``W(2)``: ``W(-1/8) ⊠ W(-1/8)`` has the Jordan
        top of the projective cover ``P(0)`` ([GK1996]_)::

            sage: import vertex_algebras as va
            sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
            sage: Y.<cp, cm> = NormallyOrderedExpressions(QQ, parities={'cp': 1, 'cm': 1})
            sage: SF = UniversalVertexAlgebra(QQ, Y, {('cp', 'cm'): [0, 1]}, conformal_vector=cm*cp,
            ....:                             horizontal_grading={'cp': 1, 'cm': -1})
            sage: a, b = SF.gens()
            sage: W2 = SF.subalgebra([SF.conformal_vector(), a.no(a.derivative()),
            ....:                     a.no(b.derivative()) + b.no(a.derivative()), b.no(b.derivative())],
            ....:                    names=['T', 'Wp', 'W0', 'Wm'], horizontal_grading='ambient',
            ....:                    conformal_vector='ambient')
            sage: P = W2.presentation(); Z = P.zhu_algebra_up_to(8)
            sage: U = Z.module({'T': -1/8, 'Wp': 0, 'W0': 0, 'Wm': 0}, horizontal_grades=[0])
            sage: M = P.induced_module_up_to(U, 8, 4)
            sage: F = M.fusion_product(M).depth_quotient_up_to(0, 3)
            sage: F.conformal_weights()
            [(0, [2])]
            sage: F.zhu_algebra_module(Z)
            Module of dimension 2 over Quotient of the free algebra ...

        TESTS:

        Gaberdiel-Kausch's table for the four simple ``W(2)``-modules
        ``W(0), W(1), W(-1/8), W(3/8)``: weights and Jordan blocks of the
        depth-0 quotients, each a graded ``A(W(2))``-module::

            sage: tops = {0: Z.module({'T': 0, 'Wp': 0, 'W0': 0, 'Wm': 0}, horizontal_grades=[0]), # long time
            ....:     -1/8: U,
            ....:     1: Z.module({'T': identity_matrix(QQ, 2), 'Wp': matrix([[0, 1], [0, 0]]),
            ....:                  'W0': diagonal_matrix([2, -2]), 'Wm': matrix([[0, 0], [-4, 0]])},
            ....:                 horizontal_grades=[1, -1]),
            ....:     3/8: Z.module({'T': 3/8*identity_matrix(QQ, 2), 'Wp': matrix([[0, 1], [0, 0]]),
            ....:                    'W0': diagonal_matrix([1/2, -1/2]), 'Wm': matrix([[0, 0], [-1/4, 0]])},
            ....:                   horizontal_grades=[1, -1])}
            sage: W = {h: P.induced_module_up_to(T, 8, 3) for h, T in tops.items()}  # long time
            sage: W[0] = W[0].simple_quotient()                     # long time
            sage: hs = [0, 1, -1/8, 3/8]
            sage: for i, h1 in enumerate(hs):                       # long time
            ....:     for h2 in hs[i:]:
            ....:         F = W[h1].fusion_product(W[h2]).depth_quotient_up_to(0, 3)
            ....:         assert F.zhu_algebra_module(Z).dimension() == F.dimension()
            ....:         assert F.horizontal_grades() is not None
            ....:         print(h1, h2, F.conformal_weights())
            0 0 [(0, [1])]
            0 1 [(1, [1, 1])]
            0 -1/8 [(-1/8, [1])]
            0 3/8 [(3/8, [1, 1])]
            1 1 [(0, [1])]
            1 -1/8 [(3/8, [1, 1])]
            1 3/8 [(-1/8, [1])]
            -1/8 -1/8 [(0, [2])]
            -1/8 3/8 [(0, [1, 1]), (1, [1, 1])]
            3/8 3/8 [(0, [2])]

        The fused top of ``W(-1/8) ⊠ W(-1/8)`` induces ``P(0)`` (weight
        space dimensions ``2, 4, 6, 12, 18``)::

            sage: F = M.fusion_product(M).depth_quotient_up_to(0, 3)
            sage: Mu = P.induced_module_up_to(F.zhu_algebra_module(Z), 8, 4)
            sage: [Mu.weight_space(k).dimension() for k in range(5)]
            [2, 4, 6, 12, 18]

        A Zhu algebra module needs the depth-0 quotient::

            sage: V = va.Virasoro(QQ, 1/2, conformal_vector='L')
            sage: e = V.verma_module(1/2).simple_quotient()
            sage: e.fusion_product(e).depth_quotient_up_to(1, 3).zhu_algebra_module(V.zhu_algebra())
            Traceback (most recent call last):
            ...
            ValueError: only the depth-0 quotient is a module over the Zhu algebra
        """
        if self._d != 0:
            raise ValueError("only the depth-0 quotient is a module over the Zhu algebra")
        V = self._V
        action = {name: self.zero_mode_matrix(V.gen(name)) for name in A.variable_names()}
        grades = self.horizontal_grades()
        if grades is not None:
            B = self._A.M._mgrading.B
            grades = [B.element(g) for g in grades]
        return A.module(action, parities=self.parities(), horizontal_grades=grades)


###########################################################################
#  Lattice vertex algebras: the fusion product is known                   #
###########################################################################

def _lattice_cosets(X):
    r"""
    The cosets ``λ`` (canonical representatives) of the simple modules
    ``V_{λ+L}`` of which the module ``X`` of a lattice vertex algebra is the
    direct sum: ``X`` a simple module ``V.simple_module(λ)``, the adjoint
    module (``λ = 0``), or a direct sum of these.

    TESTS::

        sage: import vertex_algebras as va
        sage: from vertex_algebras.fusion_product import _lattice_cosets
        sage: V = va.Lattice(QQ, matrix([[2]]))
        sage: M = V.simple_module([1/2])
        sage: _lattice_cosets(M), _lattice_cosets(V.adjoint_module())
        ([(1/2,)], [(0,)])
        sage: _lattice_cosets(M.direct_sum(M, V.adjoint_module()))
        [(1/2,), (1/2,), (0,)]
        sage: _lattice_cosets(M.submodule([M.highest_weight_vector()]))
        Traceback (most recent call last):
        ...
        NotImplementedError: fusion over a lattice vertex algebra is implemented for the simple modules V_(λ+L) (V.simple_module(λ)), the adjoint module and direct sums of these, not for Submodule ...
    """
    from .lattice_vertex_algebra_module import LatticeVertexAlgebraModule
    from .vertex_algebra_module import AdjointModule
    from .vertex_algebra_submodule import DirectSumOfModules
    if isinstance(X, LatticeVertexAlgebraModule):
        return [tuple(X.coset_representative())]
    if isinstance(X, AdjointModule):
        return [tuple(QQ.zero() for _ in range(X._V.rank()))]
    if isinstance(X, DirectSumOfModules):
        return [c for S in X.summands() for c in _lattice_cosets(S)]
    raise NotImplementedError(
        "fusion over a lattice vertex algebra is implemented for the simple "
        "modules V_(λ+L) (V.simple_module(λ)), the adjoint module and direct "
        "sums of these, not for {}".format(X))


class LatticeFusionProduct(SageObject):
    r"""
    The fusion product ``M ⊠ N`` of modules over a lattice vertex algebra
    ``V_L`` that are direct sums of simple modules: it is known,
    ``V_{λ+L} ⊠ V_{μ+L} = V_{λ+μ+L}`` for every non-degenerate integral
    lattice ``L`` (the intertwining operators among lattice modules, e.g.
    Dong–Lepowsky [DL1993]_), and fusion distributes over direct sums.
    Built by ``M.fusion_product(N)`` when the vertex algebra is a lattice
    vertex algebra; :meth:`module` is the product itself, and
    :meth:`depth_quotient_up_to` gives its depth quotients with the
    interface of :class:`FusionQuotientUpTo` (exact: there is no window).

    INPUT:

    - ``M``, ``N`` -- modules over the same lattice vertex algebra: simple
      modules ``V.simple_module(λ)``, the adjoint module, or direct sums of
      these

    EXAMPLES:

    The ``A_2`` lattice: the cosets of ``L°/L ≅ Z/3`` add::

        sage: import vertex_algebras as va
        sage: V = va.Lattice(QQ, matrix([[2, -1], [-1, 2]])); J0, J1 = V.J(0), V.J(1)
        sage: V = V.with_conformal_vector((2*J0.no(J0) + J0.no(J1) + J1.no(J0) + 2*J1.no(J1)) / 6)
        sage: V0, V1, V2 = V.simple_modules()
        sage: P = V1.fusion_product(V1); P
        Fusion product of Module V_(λ+L), λ = (1/3, 2/3), ... and Module V_(λ+L), λ = (1/3, 2/3), ...
        sage: P.module() is V2, P.is_known_exact()
        (True, True)
        sage: F = P.depth_quotient_up_to(1); F.conformal_weights()
        [(1/3, [1, 1, 1]), (4/3, [1, 1, 1, 1, 1, 1, 1, 1, 1])]

    Direct sums distribute; multiplicities are kept::

        sage: S = V1.fusion_product(V1.direct_sum(V2, V2)); S.module()
        Direct sum of [Module V_(λ+L), λ = (2/3, 1/3), ..., Module V_(λ+L), λ = (0, 0), ..., Module V_(λ+L), λ = (0, 0), ...]
        sage: S.depth_quotient_up_to(0).conformal_weights()
        [(0, [1, 1]), (1/3, [1, 1, 1])]

    The check against the Nahm–Gaberdiel–Kausch algorithm through the
    Frenkel–Kac isomorphism ``L_1(sl_2) ≅ V_{A_1}``: the doublet fuses to
    the vacuum on both sides, with the same conformal weights at depth 1::

        sage: A = va.Lattice(QQ, matrix([[2]])); J = A.J(0)
        sage: A = A.with_conformal_vector(J.no(J) / 4)
        sage: D = A.simple_module([1/2])
        sage: Fl = D.fusion_product(D).depth_quotient_up_to(1)
        sage: Va = va.Affine(QQ, ['A', 1], 1, names=['E', 'H', 'F'], conformal_vector='sugawara')
        sage: U = Va.zhu_algebra().module({'E': matrix([[0, 1], [0, 0]]),
        ....:     'H': matrix([[1, 0], [0, -1]]), 'F': matrix([[0, 0], [1, 0]])})
        sage: Da = Va.induced_module(U).simple_quotient()
        sage: Fa = Da.fusion_product(Da).depth_quotient_up_to(1, max_depth=4)  # long time
        sage: Fa.conformal_weights() == Fl.conformal_weights()  # long time
        True
        sage: Fl.conformal_weights()
        [(0, [1]), (1, [1, 1, 1])]

    TESTS::

        sage: V1.fusion_product(V2).module() is V0, V0.fusion_product(V2).module() is V2
        (True, True)
        sage: V1.fusion_product(V2) is V1.fusion_product(V2)
        True
        sage: W = va.Lattice(QQ, matrix([[4]]))
        sage: V1.fusion_product(W.simple_module([1/4]))
        Traceback (most recent call last):
        ...
        ValueError: the modules are modules over different vertex algebras

    Odd and indefinite lattices (the module is defined without a conformal
    vector)::

        sage: Z1 = va.Lattice(QQ, matrix([[1]])); Z1.simple_modules()
        [Module V_(λ+L), λ = (0), over ...]
        sage: H = va.Lattice(QQ, matrix([[0, 3], [3, 0]]))
        sage: M = H.simple_module([1/3, 0]); N = H.simple_module([2/3, 1/3])
        sage: M.fusion_product(N).module().coset_representative()
        (0, 1/3)
    """
    def __init__(self, M, N):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.fusion_product import LatticeFusionProduct
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: LatticeFusionProduct(M, M)._cosets
            [(0,)]
        """
        if M._V is not N._V:
            raise ValueError("the modules are modules over different vertex algebras")
        self._M, self._N = M, N
        self._V = V = M._V
        A, B = _lattice_cosets(M), _lattice_cosets(N)
        self._cosets = [tuple(V.simple_module([x + y for x, y in zip(a, b)]).coset_representative())
                        for a in A for b in B]
        self._module = None
        self._quotients = {}

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: M.fusion_product(M)
            Fusion product of Module V_(λ+L), λ = (1/2), ... and Module V_(λ+L), λ = (1/2), ...
        """
        return "Fusion product of {} and {}".format(self._M, self._N)

    def factors(self):
        """
        The two modules ``(M, N)``.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: M.fusion_product(M).factors() == (M, M)
            True
        """
        return (self._M, self._N)

    def summands(self):
        """
        The simple modules ``V_{λ+μ+L}`` of the product, one for each pair of
        summands of ``M`` and ``N`` (with multiplicities).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: M.fusion_product(M.direct_sum(V.adjoint_module())).summands()
            [Module V_(λ+L), λ = (0), ..., Module V_(λ+L), λ = (1/2), ...]
        """
        return [self._V.simple_module(list(c)) for c in self._cosets]

    def module(self):
        """
        The fusion product as a module: ``V_{λ+μ+L}``, or the direct sum of
        these for direct sums.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: M.fusion_product(M).module() is V.simple_module([0])
            True
        """
        if self._module is None:
            S = self.summands()
            if len(S) == 1:
                self._module = S[0]
            else:
                from .vertex_algebra_submodule import DirectSumOfModules
                self._module = DirectSumOfModules(S)
        return self._module

    def is_known_exact(self):
        """
        ``True``: the product is known, not computed up to a window.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); M = V.simple_module([1/2])
            sage: M.fusion_product(M).is_known_exact()
            True
        """
        return True

    def depth_quotient_up_to(self, depth=0, max_depth=None):
        r"""
        The depth-``depth`` quotient of ``M ⊠ N``: each summand ``V_{ν+L}``
        cut off at depth ``depth`` above its own lowest weight (a simple
        module is generated by any vector, so its vectors of depth ``> d``
        are the words of modes of degree ``> d`` applied to it).  Exact;
        ``max_depth`` is accepted for the interface of
        :meth:`FusionProduct.depth_quotient_up_to` and ignored.

        INPUT:

        - ``depth`` -- (default: 0) a non-negative rational
        - ``max_depth`` -- ignored

        OUTPUT:

        A :class:`LatticeFusionQuotient` (cached per ``depth``).

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: F = M.fusion_product(M).depth_quotient_up_to(2, max_depth=5); F
            Depth-2 quotient of the fusion product of ... (exact): dimension 8
            sage: F.conformal_weights()
            [(0, [1]), (1, [1, 1, 1]), (2, [1, 1, 1, 1])]
            sage: M.fusion_product(M).depth_quotient_up_to(-1)
            Traceback (most recent call last):
            ...
            ValueError: the depth must be >= 0
        """
        d = QQ(depth)
        if d < 0:
            raise ValueError("the depth must be >= 0")
        F = self._quotients.get(d)
        if F is None:
            F = LatticeFusionQuotient(self, d)
            self._quotients[d] = F
        return F


class LatticeFusionQuotient(FusionQuotientUpTo):
    r"""
    The depth-``d`` quotient of a :class:`LatticeFusionProduct`: the vectors
    of each summand ``V_{ν+L}`` of depth ``<= d`` above its lowest weight.
    It has the interface of :class:`FusionQuotientUpTo`; :meth:`basis` gives
    vectors of :meth:`LatticeFusionProduct.module` instead of tensors.

    INPUT:

    - ``P`` -- a :class:`LatticeFusionProduct`
    - ``d`` -- a non-negative rational number

    EXAMPLES::

        sage: import vertex_algebras as va
        sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
        sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
        sage: F = M.fusion_product(M).depth_quotient_up_to(1)
        sage: F.basis()
        [|0>, e^(-1), e^(1), J0]
        sage: F.zero_mode_matrix(J), F.horizontal_grades(), F.parities()
        (
        [ 0  0  0  0]
        [ 0 -2  0  0]
        [ 0  0  2  0]
        [ 0  0  0  0], None, [0, 0, 0, 0]
        )
        sage: F.mode_matrix(J, 1)                    # J_(1) J = 2: to the depth-0 quotient
        [0 0 0 2]
        sage: F.projection(0)
        [1 0 0 0]

    TESTS::

        sage: F.special_subspace_dimensions()
        Traceback (most recent call last):
        ...
        NotImplementedError: the fusion product of lattice modules is known: Nahm's special subspace is not computed
        sage: F.mode_matrix(J, -3)
        Traceback (most recent call last):
        ...
        ValueError: the modes of degree 3 do not act on the depth-1 quotient (degrees -1 .. 1)
    """
    def __init__(self, P, d):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: from vertex_algebras.fusion_product import LatticeFusionQuotient
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: LatticeFusionQuotient(M.fusion_product(M), 0)._basis
            [(0, ((), (0,)))]
        """
        from sage.arith.functions import lcm
        self._P, self._d, self._K = P, QQ(d), None
        self._V = V = P._V
        self._R = V.base_ring()
        self._S = S = P.summands()
        # the conformal weights of a summand differ by multiples of 1/D
        r = V.rank()
        units = [tuple(QQ(s) if j == i else QQ.zero() for j in range(r)) for i in range(r) for s in (1, -1)]
        D = 2 * lcm([QQ(V.degree(V.e(u))).denominator() for u in units] + [1])
        self._basis = []
        self._top = []
        for i, M in enumerate(S):
            h = QQ(M.highest_weight_vector().degree())
            self._top.append(h)
            for k in range(int((self._d * D).floor()) + 1):
                for m in sorted(M.weight_space(h + QQ((k, D))).basis_keys(), key=M._key_order):
                    self._basis.append((i, m))
        self._index = {b: j for j, b in enumerate(self._basis)}

    def _repr_(self):
        """
        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M).depth_quotient_up_to(0)
            Depth-0 quotient of the fusion product of ... (exact): dimension 1
        """
        return "Depth-{} quotient of the {} (exact): dimension {}".format(
            self._d, "f" + repr(self._P)[1:], self.dimension())

    def basis(self):
        """
        The basis: vectors of :meth:`LatticeFusionProduct.module`.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M.direct_sum(M)).depth_quotient_up_to(0).basis()
            [[0] |0>, [1] |0>]
        """
        R = self._P.module()
        if len(self._S) == 1:
            return [R.monomial(m) for _, m in self._basis]
        return [R.embed(i, self._S[i].monomial(m)) for i, m in self._basis]

    def _coords_of(self, i, x):
        """
        The coordinates of the vector ``x`` of summand ``i``, the components
        of depth ``> d`` dropped.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: F = M.fusion_product(M).depth_quotient_up_to(0)
            sage: F._coords_of(0, J.no(F._S[0].monomial(((), (0,)))))
            (0)
        """
        v = [self._R.zero()] * len(self._basis)
        for m, c in x.monomial_coefficients().items():
            j = self._index.get((i, m))
            if j is not None:
                v[j] = c
        from sage.modules.free_module_element import vector
        return vector(self._R, v)

    def mode_matrix(self, a, n):
        r"""
        The matrix of ``a_(n)``, ``a`` homogeneous of weight ``h``, of degree
        ``k = h - 1 - n``: on the depth-``d`` quotient for ``0 <= k <= d``, to
        the depth-``(d + k)`` quotient for ``-d <= k < 0`` (as for
        :meth:`FusionQuotientUpTo.mode_matrix`).

        INPUT:

        - ``a`` -- a homogeneous element of the vertex algebra
        - ``n`` -- an integer

        OUTPUT:

        A matrix over the base ring, with ``dimension()`` columns.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: F = M.fusion_product(M).depth_quotient_up_to(1)
            sage: F.mode_matrix(V.e((1,)), 0).rank()      # e^α_(0): degree 0
            2
        """
        V = self._V
        a = V(a)
        deg = QQ(V.degree(a)) - 1 - n
        if not -self._d <= deg <= self._d:
            raise ValueError("the modes of degree {} do not act on the depth-{} quotient (degrees {} .. {})".format(
                deg, self._d, -self._d, self._d))
        target = self if deg >= 0 else self._P.depth_quotient_up_to(self._d + deg)
        cols = [target._coords_of(i, a.nproduct(self._S[i].monomial(m), n)) for i, m in self._basis]
        return matrix(self._R, self.dimension(), target.dimension(), cols).transpose()

    def projection(self, depth):
        r"""
        The matrix of the surjection onto the quotient of depth ``depth <= d``.

        INPUT:

        - ``depth`` -- a rational number with ``0 <= depth <= d``

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M).depth_quotient_up_to(2).projection(1).dimensions()
            (4, 8)
        """
        depth = QQ(depth)
        if not 0 <= depth <= self._d:
            raise ValueError("need 0 <= depth <= {}".format(self._d))
        target = self._P.depth_quotient_up_to(depth)
        cols = [target._coords_of(i, self._S[i].monomial(m)) for i, m in self._basis]
        return matrix(self._R, self.dimension(), target.dimension(), cols).transpose()

    def parities(self):
        """
        The parities of the basis vectors.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[1]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 2)
            sage: P = V.adjoint_module().fusion_product(V.adjoint_module())
            sage: P.depth_quotient_up_to(1/2).parities()
            [0, 1, 1]
        """
        return [self._S[i]._parity_key(m) for i, m in self._basis]

    def horizontal_grades(self):
        """
        The horizontal grades of the basis vectors, or ``None`` when the
        modules are not graded.

        EXAMPLES::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])).with_horizontal_grading('lattice'); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M).depth_quotient_up_to(1).horizontal_grades()
            [(0,), (-2,), (2,), (0,)]
        """
        if any(M._mgrading is None for M in self._S):
            return None
        return [self._S[i]._grade_key(m) for i, m in self._basis]

    def special_subspace_dimensions(self):
        """
        Not computed: the product is known.

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M).depth_quotient_up_to(0).special_subspace_dimensions()
            Traceback (most recent call last):
            ...
            NotImplementedError: the fusion product of lattice modules is known: Nahm's special subspace is not computed
        """
        raise NotImplementedError("the fusion product of lattice modules is known: "
                                  "Nahm's special subspace is not computed")

    def zhu_algebra_module(self, A):
        """
        Not implemented for lattice vertex algebras: the product is
        :meth:`LatticeFusionProduct.module`, whose top levels are modules
        over the Zhu algebra (``V.zhu_algebra()``).

        TESTS::

            sage: import vertex_algebras as va
            sage: V = va.Lattice(QQ, matrix([[2]])); J = V.J(0)
            sage: V = V.with_conformal_vector(J.no(J) / 4); M = V.simple_module([1/2])
            sage: M.fusion_product(M).depth_quotient_up_to(0).zhu_algebra_module(V.zhu_algebra())
            Traceback (most recent call last):
            ...
            NotImplementedError: use the fusion product's module(): its top levels are modules over the Zhu algebra
        """
        raise NotImplementedError("use the fusion product's module(): its top levels "
                                  "are modules over the Zhu algebra")
