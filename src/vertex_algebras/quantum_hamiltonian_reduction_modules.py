# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Quantum Hamiltonian reduction of highest weight modules.

The reduction functor ``H`` of Kac–Roan–Wakimoto [KRW2003]_ takes a
module ``M`` of the universal affine vertex (super)algebra ``V^k(g)`` to a
module of the W-algebra ``W^k(g, f)`` (the cohomology of the complex ``M
⊗ F^ch ⊗ F^ne``).  The graded pieces of that complex are infinite
dimensional (Arakawa, [Ara2005]_ §3.9), so ``H(M)`` is not computed as
cohomology here: for highest weight modules the theorems below determine
it, and this module implements them (with their hypotheses checked; other
cases are refused).

- ``reduction='+'`` (the differential of [KRW2003]_, the one the
  W-algebras of :mod:`quantum_hamiltonian_reduction` are built with):

  * Verma modules, any good grading ``(x, f)`` ([KW2004]_ Thm 6.2, 6.3):
    ``H_i(M(λ)) = 0`` for ``i ≠ 0`` and ``H(M(λ))`` is the Verma module of
    ``W`` whose highest weight is the image of ``λ``, with ``λ_W|h♮ =
    λ|h♮`` and ``L_0 = (λ|λ+2ρ)/(2(k+h^∨)) - λ(x)``, for the positive
    system ``Δ_{0,+} ⊔ Δ_{>0}``;
  * simple modules, ``f`` minimal ([Ara2005]_ Thm 6.7.1, 6.7.4): ``H(L(λ))
    = 0`` if ``k - (λ|θ)`` is a non-negative integer, else the simple
    quotient of ``H(M(λ))``;
  * simple modules, ``f`` principal, ``g`` a Lie algebra, under the
    genericity condition (385) of [Ara2007]_ (Thm 9.1.4).

- ``reduction='-'`` (``f`` principal, ``g`` a Lie algebra, [Ara2007]_ Thm
  7.5.1, 7.6.1, 7.6.3; the "+" differential twisted by ``♯ ∘ t_{-x}``):
  ``H(M(λ))`` is the Verma module of the central character of ``-w_0(λ)``
  and ``H(L(λ))`` its simple quotient if ``λ`` is anti-dominant, else 0.

The highest weight of ``W`` (the eigenvalues of the zero modes of its
generators on the highest weight vector) is computed through the Miura
map ``π: W -> V^τ(g_0) ⊗ F(g_{1/2})`` (see
:mod:`quantum_hamiltonian_reduction`): the Harish-Chandra projection of
the image of ``[π(G)]`` in the Zhu algebra of the free field algebra,
evaluated at ``μ = λ`` for ``'+'`` and at ``μ = -λ - 2ρ + (k + h^∨) x``
for ``'-'``.  Checked against [KW2004]_ (6.14) (``h♮`` and ``L_0``, for
principal, minimal, rectangular and super cases) and [Ara2007]_ (285) and
the Weyl group invariance of the central character.

EXAMPLES:

The Virasoro Verma module of the reduction of an ``sl_2`` Verma module::

    sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
    sage: K.<k, a> = Frac(QQ['k, a'])
    sage: V = UniversalAffineVertexAlgebra(K, ['A', 1], level=k)
    sage: values, h = V.quantum_hamiltonian_reduction_highest_weight('principal', [a])
    sage: h == a*(a + 2)/(4*(k + 2)) - a/2
    True

The Lee–Yang modules (``c = -22/5``) from the admissible ``sl_2``
modules at ``k = -2 + 2/5``, by the "-" reduction ([Ara2007]_ Thm 7.6.3):
the weight ``0`` is not anti-dominant and gives 0, the others give the two
Lee–Yang modules, twice::

    sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-8/5)
    sage: A = V.zhu_algebra()
    sage: def L(l):
    ....:     return V.induced_module(A.verma_module({'alphacheck[1]': l})).simple_quotient()
    sage: H = [L(l).quantum_hamiltonian_reduction('principal', reduction='-')
    ....:      for l in (0, -2/5, -4/5, -6/5, -8/5)]
    sage: H[0]
    Zero module over Universal vertex algebra on generators ['J{E[-alpha[1]]}'] over Rational Field
    sage: H[1].vertex_algebra().central_charge(), [M.top_weight() for M in H[1:]]
    (-22/5, [0, -1/5, -1/5, 0])
    sage: H[1].character(6), H[2].character(5)                 # Rogers–Ramanujan
    (1 + q^2 + q^3 + q^4 + q^5 + O(q^6),
     q^(-1/5) + q^(4/5) + q^(9/5) + q^(14/5) + 2*q^(19/5) + 2*q^(24/5) + O(q^5))

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""
from sage.matrix.constructor import matrix
from sage.misc.misc_c import prod
from sage.modules.free_module_element import vector
from sage.rings.infinity import Infinity
from sage.rings.integer_ring import ZZ
from sage.rings.rational_field import QQ


def _is_integer(v):
    """
    Whether ``v`` is an integer (``False`` for an element that depends on
    an indeterminate).

    TESTS::

        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _is_integer
        sage: K.<k> = Frac(QQ['k'])
        sage: _is_integer(K(3)), _is_integer(K(1/2)), _is_integer(k), _is_integer(k - k + 2)
        (True, False, False, True)
    """
    try:
        return v in ZZ
    except (TypeError, ValueError, ArithmeticError):
        return False


class _ReductionData:
    r"""
    The data of the reduction of ``V^k(g)`` by ``(x, f)``: the Lie
    superalgebra, the W-algebra with and without its conformal vector, the
    Miura map, the invariant form on the Cartan subalgebra, the roots as
    functionals and the positive system ``Δ_{0,+} ⊔ Δ_{>0}``.

    Weights ``λ`` are vectors of the values ``λ(h_i)`` on the Cartan basis
    ``h_1, ..., h_r`` of ``g`` (for a Cartan type, ``h_i`` is the coroot
    ``alphacheck[i]``).

    TESTS::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2)
        sage: D = _ReductionData(V, 'minimal', None, None)
        sage: D.cartan_names, D.rho, D.is_minimal, D.is_principal
        (['alphacheck[1]', 'alphacheck[2]'], (1, 1), True, False)
    """
    def __init__(self, V, f, x, names):
        """
        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1/2)
            sage: D = _ReductionData(V, 'principal', None, None)
            sage: D.W
            Universal vertex algebra on generators ['J{E[-alpha[1]]}'] over Rational Field
            sage: D.is_minimal, D.is_principal, D.is_lie
            (True, True, True)
        """
        from .quantum_hamiltonian_reduction import _miura_data, _prepare
        g, k = V._reduction_data()
        R = V.base_ring()
        self.V, self.g = V, g
        f_, x_, R, k = _prepare(R, g, f, k, x)
        self.R, self.k, self.f, self.x = R, k, f_, x_
        impl = V._data['implementation']
        self.W, self.M, self.images = _miura_data(g, f_, x_, R, k, names, impl)
        self.WT = V.quantum_hamiltonian_reduction(f, x, names, conformal_vector='reduction')
        r = g.rank()
        self.r = r
        H = [g.h(i) for i in range(1, r + 1)]
        self.H = H
        a = V._affine
        if a['kind'] == 'cartan':
            from sage.combinat.root_system.cartan_matrix import CartanMatrix
            self.cartan_names = ['alphacheck[%s]' % i for i in CartanMatrix(a['obj']).index_set()]
        else:
            self.cartan_names = [str(h.leading_support()) for h in H]
        self.B = matrix(R, r, r, lambda i, j: R(g.invariant_form(H[i], H[j])))
        self.Binv = self.B.inverse()
        ev = [g.ad_eigenvalues(h) for h in H]
        xev = g.ad_eigenvalues(x_)
        basis = g.basis()
        self.root_values = {}           # root (simple-root coordinates) -> values on H
        self.x_value = {}               # root -> β(x)
        self.parity = {}
        for key in basis.keys():
            beta = tuple(g.root_on_basis(key))
            if not any(beta):
                continue
            self.root_values[beta] = vector(R, [e[key] for e in ev])
            self.x_value[beta] = xev[key]
            self.parity[beta] = basis[key].is_even_odd()
        self.positive = [b for b in self.root_values if self._is_positive(b)]
        rho = vector(R, [0] * r)
        for b in self.positive:
            rho += (-1) ** self.parity[b] * self.root_values[b] / 2
        self.rho = rho
        self.x_form = vector(R, [R(g.invariant_form(x_, h)) for h in H])
        self.kappa = k + g.dual_coxeter_number()
        self.is_lie = not any(self.parity.values())
        dynkin = x is None or x_ == g.dynkin_grading(f_)
        self.is_minimal = dynkin and f_ == _prepare(R, g, 'minimal', k, None)[0]
        self.is_principal = dynkin and f_ == _prepare(R, g, 'principal', k, None)[0]

    def _is_positive(self, beta):
        """
        Whether the root ``β`` is in ``Δ_{0,+} ⊔ Δ_{>0}`` (``Δ_{0,+}``: the
        roots of ``g_0`` with non-negative simple-root coordinates).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1), 'principal', None, None)
            sage: D._is_positive((1,)), D._is_positive((-1,))
            (True, False)
        """
        v = self.x_value[beta]
        return v > 0 or (v == 0 and all(c >= 0 for c in beta))

    def ip(self, u, v):
        """
        The form ``(u|v)`` of two weights (vectors of values on the ``h_i``).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1), 'principal', None, None)
            sage: D.ip(D.rho, D.rho)                    # (ρ|ρ) = 1/2 for sl_2
            1/2
        """
        return u * self.Binv * v

    def weight(self, weight):
        r"""
        The vector ``(λ(h_1), ..., λ(h_r))`` of a weight given as a list of
        values or a dict by the names of the Cartan currents.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1), 'principal', None, None)
            sage: D.weight([1, 2]), D.weight({'alphacheck[1]': 3, 'alphacheck[2]': 0})
            ((1, 2), (3, 0))
            sage: D.weight([1])
            Traceback (most recent call last):
            ...
            ValueError: give the values of the weight on the Cartan currents ['alphacheck[1]', 'alphacheck[2]']
        """
        if isinstance(weight, dict):
            if set(map(str, weight)) != set(self.cartan_names):
                raise ValueError('give the values of the weight on the Cartan currents '
                                 '{}'.format(self.cartan_names))
            w = {str(s): c for s, c in weight.items()}
            weight = [w[s] for s in self.cartan_names]
        weight = list(weight)
        if len(weight) != self.r:
            raise ValueError('give the values of the weight on the Cartan currents '
                             '{}'.format(self.cartan_names))
        return vector(self.R, weight)

    def check_reduction(self, reduction):
        """
        Refuse the "-" reduction outside the principal nilpotent of a Lie
        algebra with its Dynkin grading.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2), 'minimal', None, None)
            sage: D.check_reduction('-')
            Traceback (most recent call last):
            ...
            NotImplementedError: the '-' reduction is implemented for the principal nilpotent of a Lie algebra (Dynkin grading) only: for other f it gives Ramond twisted modules
            sage: D.check_reduction('x')
            Traceback (most recent call last):
            ...
            ValueError: reduction must be '+' or '-'
        """
        if reduction not in ('+', '-'):
            raise ValueError("reduction must be '+' or '-'")
        if reduction == '-' and not (self.is_principal and self.is_lie):
            raise NotImplementedError("the '-' reduction is implemented for the principal "
                                      "nilpotent of a Lie algebra (Dynkin grading) only: "
                                      "for other f it gives Ramond twisted modules")

    def miura_point(self, lam, reduction):
        r"""
        The weight ``μ`` at which the Miura images are evaluated: ``λ`` for
        ``'+'``, ``-λ - 2ρ + (k + h^∨) x`` for ``'-'``.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-8/5), 'principal', None, None)
            sage: D.miura_point(D.weight([-2/5]), '+'), D.miura_point(D.weight([-2/5]), '-')
            ((-2/5), (-6/5))
        """
        if reduction == '+':
            return lam
        return -lam - 2 * self.rho + self.kappa * self.x_form

    def _t_basis(self):
        r"""
        A basis of ``h ∩ g^f`` (coefficient vectors over the ``h_i``).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2), 'minimal', None, None)
            sage: D._t_basis()
            [(1, -1)]
        """
        g, R = self.g, self.R
        keys = sorted(self.f.monomial_coefficients())
        rows = []
        for h in self.H:
            br = h.bracket(self.f)
            rows.append([R(br[key]) for key in keys])
        return list(matrix(R, rows).left_kernel().basis())

    def t_weight(self, element):
        r"""
        The weight under ``h ∩ g^f`` of a homogeneous element of the free
        field algebra (its first monomial).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2), 'minimal', None, None)
            sage: [D.t_weight(D.images[s]) for s in sorted(D.images)]
            [(0,), (-3,), (3,), (0,)]
        """
        t = getattr(self, '_t', None)
        if t is None:
            t = self._t = self._t_basis()
        names = self.M.variable_names()
        key_of = {'J{%s}' % key: key for key in self.g.basis().keys()}
        key_of.update({'Phi{%s}' % key: key for key in self.g.basis().keys()})
        out = vector(self.R, [0] * len(t))
        mono = next(iter(element.monomial_coefficients()))
        for i, _ in mono:
            beta = tuple(self.g.root_on_basis(key_of[names[i]]))
            if any(beta):
                vals = self.root_values[beta]
                out += vector(self.R, [vals * c for c in t])
        return tuple(out)

    def top(self, mu):
        r"""
        The zero modes of the generators of ``W`` acting by scalars on the
        highest weight vector of the image (``Δ_{0,+}`` highest weight
        ``μ``) under the Miura map, a dict by name, and ``L_0``.

        The scalars: integer weight generators of weight 0 under ``h ∩ g^f``;
        the value is the Harish-Chandra projection of ``[π(G)]`` (the
        ordered monomials of the Zhu algebra of the free field algebra read
        lowering · Cartan · raising).

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=1/2), 'minimal', None, None)
            sage: D.top(D.weight([1, 2]))
            ({'J{E[-alpha[1] - alpha[2]]}': -2/3, 'J{h1 - h2}': -1}, 1/42)
        """
        M, W = self.M, self.W
        A = M.zhu_algebra()
        an = A.variable_names()
        cart = {'J{%s}' % h.leading_support(): mu[i] for i, h in enumerate(self.H)}

        def hc(v):
            out = self.R.zero()
            for mono, c in A(v).monomial_coefficients().items():
                if all(an[i] in cart for i in mono):
                    out += c * prod(cart[an[i]] for i in mono)
            return out
        values = {}
        for s, img in self.images.items():
            if W.degree(W.gen(s)) in ZZ and not any(self.t_weight(img)):
                values[s] = hc(img)
        phi = W.hom(self.images, M, check=False)
        T = self.WT.conformal_vector()
        return values, hc(phi(W(T)))

    def scalar_top(self):
        """
        Whether every integer weight generator of ``W`` acts by a scalar on
        the top of a Verma module (weight 0 under ``h ∩ g^f``): then the
        Verma modules of ``W`` are those of
        :meth:`universal_vertex_algebra.UniversalVertexAlgebra.verma_module`.

        TESTS::

            sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
            sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData
            sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 3], level=1/2)
            sage: _ReductionData(V, 'minimal', None, None).scalar_top()
            False
        """
        W = self.W
        return all(not any(self.t_weight(img)) for s, img in self.images.items()
                   if W.degree(W.gen(s)) in ZZ)


def reduced_highest_weight(V, f, weight, x=None, reduction='+', names=None):
    r"""
    The highest weight of the reduction of the Verma module ``M(λ)``:
    ``(values, h)``, the eigenvalues of the zero modes of the generators of
    the W-algebra that act by scalars on the highest weight vector (a dict
    by name) and its conformal weight.  See
    :meth:`affine_vertex_algebra.UniversalAffineVertexAlgebra.quantum_hamiltonian_reduction_highest_weight`.

    TESTS:

    [KW2004]_ (6.14) for the Bershadsky–Polyakov algebra: ``J♮`` takes
    ``λ(h_1 - h_2)``, ``L_0 = (λ|λ+2ρ)/(2(k+3)) - λ(x)``::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import reduced_highest_weight
        sage: K.<k, a, b> = Frac(QQ['k, a, b'])
        sage: V = UniversalAffineVertexAlgebra(K, ['A', 2], level=k)
        sage: values, h = reduced_highest_weight(V, 'minimal', [a, b])
        sage: values['J{h1 - h2}']
        a - b
        sage: h == (2*(a^2 + a*b + b^2)/3 + 2*a + 2*b)/(2*(k + 3)) - (a + b)/2
        True
    """
    D = _ReductionData(V, f, x, names)
    D.check_reduction(reduction)
    lam = D.weight(weight)
    return D.top(D.miura_point(lam, reduction))


def euler_poincare_character(V, f, weight, prec, x=None):
    r"""
    The Euler–Poincaré character of the reduction of a Verma module (or of
    an alternating sum of them), [KRW2003]_ Theorem 3.1 at ``h = 0``.  See
    :meth:`affine_vertex_algebra.UniversalAffineVertexAlgebra.quantum_hamiltonian_reduction_character`.

    TESTS:

    The Bershadsky–Polyakov Verma modules: the product over the creation
    modes of the generators (``J_{-n}``, ``G^±_{-n+1/2}``, ``L_{-n}``,
    ``n >= 1``), the character of the Verma module of ``W``::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import euler_poincare_character
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=-3/2)
        sage: euler_poincare_character(V, 'minimal', [0, 0], 3)
        1 + 2*q^(1/2) + 5*q + 10*q^(3/2) + 20*q^2 + 36*q^(5/2) + O(q^3)
        sage: W = V.quantum_hamiltonian_reduction('minimal', conformal_vector='reduction')
        sage: W.verma_module(V.quantum_hamiltonian_reduction_highest_weight(
        ....:     'minimal', [0, 0])[0]).character(3)
        1 + 2*q^(1/2) + 5*q + 10*q^(3/2) + 20*q^2 + 36*q^(5/2) + O(q^3)
        sage: euler_poincare_character(V, 'minimal', {(0, 0): 1, (-1, 0): -1}, 3/2)
        1 - q^(1/18) + 2*q^(1/2) - 2*q^(5/9) + 5*q - 5*q^(19/18) + O(q^(3/2))
        sage: euler_poincare_character(UniversalAffineVertexAlgebra(QQ, ['A', 3], level=1),
        ....:                          'minimal', [0, 0, 0], 2)
        Traceback (most recent call last):
        ...
        NotImplementedError: the even roots of g_0 make the specialisation to q singular: the character needs the variables of h ∩ g^f
    """
    from sage.rings.power_series_ring import PowerSeriesRing
    from sage.rings.puiseux_series_ring import PuiseuxSeriesRing
    D = _ReductionData(V, f, x, None)
    if isinstance(weight, dict) and all(isinstance(w, tuple) for w in weight):
        terms = [(D.weight(w), c) for w, c in weight.items()]
    else:
        terms = [(D.weight(weight), 1)]
    if any(v == 0 and not D.parity[b] for b, v in D.x_value.items()):
        raise NotImplementedError('the even roots of g_0 make the specialisation to q '
                                  'singular: the character needs the variables of h ∩ g^f')
    prec = QQ(prec)
    tops = []
    for lam, c in terms:
        h = (D.ip(lam, lam + 2 * D.rho) / (2 * D.kappa)
             - D.ip(lam, D.x_form))
        try:
            h = QQ(h)
        except (TypeError, ValueError):
            raise ValueError('the conformal weight {} is not rational'.format(h))
        tops.append((h, ZZ(c)))
    low = min(h for h, _ in tops)
    N = max(ZZ(0), (2 * (prec - low)).ceil())          # precision in t = q^(1/2)
    S = PowerSeriesRing(ZZ, 't', default_prec=N)
    t = S.gen()
    prod_ = S.one().add_bigoh(N)
    # 1 / prod_j (1 - q^j)^{dim h}
    for j in range(1, N // 2 + 1):
        prod_ *= (~(1 - t ** (2 * j)).add_bigoh(N)) ** D.r
    for b in D.positive:
        v, s = D.x_value[b], (-1) ** D.parity[b]
        exps = []
        if v == QQ((1, 2)):
            exps = [2 * n - 1 for n in range(1, N + 1)]
        elif v == 0:
            exps = [2 * (n - 1) for n in range(1, N + 1)] + [2 * n for n in range(1, N + 1)]
        for e in exps:
            if e >= N:
                continue
            factor = (1 - s * t ** e).add_bigoh(N)
            prod_ *= factor ** (-s) if s == 1 else factor
    P = PuiseuxSeriesRing(ZZ, 'q')
    q = P.gen()
    out = P.zero()
    for h, c in tops:
        for i, a in enumerate(prod_.list()):
            if a and h + QQ(i) / 2 < prec:
                out += c * a * q ** (h + QQ(i) / 2)
    return out.add_bigoh(prec)


def reduce_module(M, f, x=None, reduction='+', names=None):
    r"""
    The quantum Hamiltonian reduction ``H(M)`` of a Verma module ``M(λ)`` or
    of its simple quotient ``L(λ)`` over a universal affine vertex algebra,
    as given by the theorems in the module documentation.  See
    :meth:`vertex_algebra_module.VertexAlgebraModule.quantum_hamiltonian_reduction`.

    TESTS:

    The criterion of [Ara2005]_ for the minimal reduction of ``sl_3`` (the
    Bershadsky–Polyakov algebra) at ``k = -3/2``: ``k - (λ|θ) ∈ Z_{>=0}``
    gives 0::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import reduce_module
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 2], level=-3/2)
        sage: A = V.zhu_algebra()
        sage: def L(l):
        ....:     return V.induced_module(A.verma_module(
        ....:         {'alphacheck[1]': l[0], 'alphacheck[2]': l[1]})).simple_quotient()
        sage: reduce_module(L([-1/2, -1/2]), 'minimal')
        Simple quotient of Verma module ...
        sage: reduce_module(L([-1/2, -1]), 'minimal')
        Zero module over ...
    """
    from .affine_vertex_algebra import UniversalAffineVertexAlgebra
    from .higher_zhu_algebra import HigherInducedModule
    from .vertex_algebra_module import InducedModule
    from .vertex_algebra_submodule import (MaximalSubmodule,
                                           QuotientVertexAlgebraModule, ZeroModule)
    from .zhu_algebra_module import ZhuVermaModule
    V = M.vertex_algebra()
    if not isinstance(V, UniversalAffineVertexAlgebra):
        raise NotImplementedError('the reduction of modules is implemented for universal '
                                  'affine vertex algebras only')
    simple = (isinstance(M, QuotientVertexAlgebraModule)
              and isinstance(M._N, MaximalSubmodule))
    cover = M._cover_module if simple else M
    if not (isinstance(cover, InducedModule) and not isinstance(cover, HigherInducedModule)
            and isinstance(cover._U, ZhuVermaModule)):
        raise NotImplementedError('the reduction is implemented for Verma modules (induced '
                                  'from a Verma module of the Zhu algebra) and their simple '
                                  'quotients only')
    U = cover._U
    D = _ReductionData(V, f, x, names)
    D.check_reduction(reduction)
    _check_borel(D, U)
    if not D.scalar_top():
        raise NotImplementedError('the Verma modules of {} have infinite top levels (a '
                                  'generator of integer weight with nonzero weight under '
                                  'h ∩ g^f): not implemented yet'.format(D.W))
    lam = D.weight(U.highest_weight())
    if simple and not _reduction_is_nonzero(D, lam, reduction):
        return ZeroModule(D.WT)
    values, _ = D.top(D.miura_point(lam, reduction))
    N = D.WT.verma_module(values)
    return N.simple_quotient() if simple else N


def _check_borel(D, U):
    r"""
    Refuse a Verma module whose positive roots (the raising generators of
    the Zhu algebra) are not ``Δ_{0,+} ⊔ Δ_{>0}``.

    TESTS::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import _ReductionData, _check_borel
        sage: V = UniversalAffineVertexAlgebra(QQ, ['A', 1], level=1)
        sage: D = _ReductionData(V, 'principal', None, None)
        sage: _check_borel(D, V.zhu_algebra().verma_module({'alphacheck[1]': 0}))
        sage: _check_borel(D, V.zhu_algebra().verma_module({'alphacheck[1]': 0}, functional=[-1]))
        Traceback (most recent call last):
        ...
        NotImplementedError: the Verma module is not highest weight for the positive roots Δ_{0,+} ⊔ Δ_{>0} of the reduction
    """
    for z, grade in enumerate(U._g):
        beta = tuple(ZZ(c) for c in grade)
        if not any(beta):
            continue
        if beta not in D.x_value or (z in U._raising) != D._is_positive(beta):
            raise NotImplementedError('the Verma module is not highest weight for the '
                                      'positive roots Δ_{0,+} ⊔ Δ_{>0} of the reduction')


def _reduction_is_nonzero(D, lam, reduction):
    r"""
    Whether ``H(L(λ))`` is nonzero, by [Ara2005]_ (minimal, '+'),
    [Ara2007]_ Thm 9.1.4 (principal, '+', under its condition (385)) and
    Thm 7.6.3 (principal, '-').

    TESTS::

        sage: from vertex_algebras.affine_vertex_algebra import UniversalAffineVertexAlgebra
        sage: from vertex_algebras.quantum_hamiltonian_reduction_modules import (
        ....:     _ReductionData, _reduction_is_nonzero)
        sage: D = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 1], level=-8/5),
        ....:                    'principal', None, None)
        sage: [_reduction_is_nonzero(D, D.weight([l]), '-') for l in (0, -2/5, 1)]
        [False, True, False]
        sage: _reduction_is_nonzero(D, D.weight([0]), '+')        # minimal = principal for sl_2
        True
        sage: D3 = _ReductionData(UniversalAffineVertexAlgebra(QQ, ['A', 2], level=-3/2),
        ....:                     'principal', None, None)
        sage: _reduction_is_nonzero(D3, D3.weight([0, 0]), '+')
        Traceback (most recent call last):
        ...
        NotImplementedError: H(L(λ)) for the principal '+' reduction is known under the condition (385) of [Ara2007] only (e.g. an irrational level)
    """
    g = D.g
    if reduction == '-':
        for b in D.positive:
            vb = D.root_values[b]
            v = 2 * D.ip(lam + D.rho, vb) / D.ip(vb, vb)
            if _is_integer(v) and v >= 1:
                return False
        return True
    if D.is_minimal:
        theta = tuple(g.theta())
        c = D.k - D.ip(lam, D.root_values[theta])
        return not (_is_integer(c) and c >= 0)
    if D.is_principal and D.is_lie:
        for b in D.positive:
            vb = D.root_values[b]
            for n in range(1, sum(b) + 1):
                v = 2 * (-D.ip(lam + D.rho, vb) + n * D.kappa) / D.ip(vb, vb)
                if _is_integer(v):
                    raise NotImplementedError("H(L(λ)) for the principal '+' reduction is "
                                              "known under the condition (385) of [Ara2007] "
                                              "only (e.g. an irrational level)")
        return True
    raise NotImplementedError("H(L(λ)) is implemented for the minimal nilpotent ('+'), the "
                              "principal nilpotent of a Lie algebra ('+' under (385), '-') "
                              "with the Dynkin grading")
