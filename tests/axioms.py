# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Vertex algebra axiom checks for any :class:`vertex_algebra.VertexAlgebra`,
shared by the smoke tests.  Everything goes through ``nproduct`` (all
integer ``n``) and ``derivative``, so these also test the base class's
generic negative ``n``-th products.
"""
from sage.arith.misc import binomial, factorial


def nmax(V, a, b):
    """Upper bound for ``n`` with ``a_(n) b != 0`` (elements)."""
    return max((V._nmax_key(x, y) for x in a.support() for y in b.support()),
               default=-1)


def skew_ok(V, a, b, nlow=-3):
    r"""
    Skew-symmetry ``b_(n) a = (-1)^{|a||b|} Σ_j (-1)^{n+j+1} ∂^j (a_(n+j) b)/j!``.
    """
    s = (-1) ** (a.parity() * b.parity())
    top = nmax(V, a, b)
    for n in range(nlow, nmax(V, b, a) + 2):
        rhs = V.zero()
        for j in range(0, max(top - n, 0) + 1):
            rhs += (-1) ** (n + j + 1) / factorial(j) * \
                a.nproduct(b, n + j).derivative(j)
        if b.nproduct(a, n) != s * rhs:
            return False
    return True


def commutator_ok(V, a, b, c, rng=range(-2, 3)):
    r"""
    Borcherds' commutator formula
    ``[a_(m), b_(n)] c = Σ_j C(m,j) (a_(j) b)_(m+n-j) c``.
    """
    s = (-1) ** (a.parity() * b.parity())
    for m in rng:
        for n in rng:
            lhs = (a.nproduct(b.nproduct(c, n), m)
                   - s * b.nproduct(a.nproduct(c, m), n))
            rhs = V.zero()
            for j in range(0, max(nmax(V, a, b), -1) + 1):
                rhs += binomial(m, j) * a.nproduct(b, j).nproduct(c, m + n - j)
            if lhs != rhs:
                return False
    return True


def associativity_ok(V, a, b, c, ms=(-1, -2), ns=range(-1, 2)):
    r"""
    Borcherds' associativity formula for ``(a_(m) b)_(n) c`` with negative
    ``m`` (normal ordered products of composites).
    """
    s = (-1) ** (a.parity() * b.parity())
    for m in ms:
        for n in ns:
            lhs = a.nproduct(b, m).nproduct(c, n)
            rhs = V.zero()
            jmax = max(nmax(V, b, c) - n, nmax(V, a, c), 0) + 2
            for j in range(0, jmax + 1):
                rhs += (-1) ** j * binomial(m, j) * (
                    a.nproduct(b.nproduct(c, n + j), m - j)
                    - s * (-1) ** m * b.nproduct(a.nproduct(c, j), m + n - j))
            if lhs != rhs:
                return False
    return True


def raw_caches_ok(V):
    """Every key-level cache of ``V`` holds raw dicts (or tuples of them)."""
    for name in V._raw_caches():
        for v in getattr(V, name).values():
            if isinstance(v, tuple):
                if name == '_ls_cache':
                    v = v[1:]
                if not all(isinstance(x, dict) for x in v):
                    return False
            elif not isinstance(v, dict):
                return False
    return True


class Reporter:
    """Collects ``ok``/``FAIL`` lines and prints ``RESULT: PASS/FAIL``."""
    def __init__(self):
        self.ok = True

    def __call__(self, name, good, extra=''):
        self.ok = self.ok and bool(good)
        print('%-50s %s %s' % (name, 'ok' if good else 'FAIL', extra),
              flush=True)

    def result(self):
        print('RESULT:', 'PASS' if self.ok else 'FAIL')
