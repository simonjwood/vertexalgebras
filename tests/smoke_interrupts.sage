# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Ctrl-C (SIGALRM raising KeyboardInterrupt) in the middle of the long
# computations of the constructions over the algebras, then the computation
# finished: the results must equal those of an uninterrupted run (no state
# left half-built): subalgebra spans over QQ (Sage path), a number field and
# Q(c) (engine-form spans, C++ elimination), commutant weight spaces and
# strong generator search, ideal weight spaces with certificates, quotient
# weight spaces.  Each case is interrupted at three fractions of its
# uninterrupted time, each on a fresh ambient algebra (no warm caches).
# Run from the repository root:  sage tests/smoke_interrupts.sage  -> RESULT: PASS
import os
import signal
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from axioms import Reporter

report = Reporter()
t_start = time.time()


def _alarm(*args):
    raise KeyboardInterrupt


def interrupted(f, t):
    """Run f(), raising KeyboardInterrupt after t seconds: 'interrupted' or
    'finished'."""
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(max(t, 0.01r)))
    try:
        f()
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    finally:
        signal.setitimer(signal.ITIMER_REAL, float(0r))
        signal.signal(signal.SIGALRM, old)
    return got


def check_case(name, make, heavy, result, fracs=(0.15r, 0.4r, 0.7r)):
    """make(tag) -> a fresh object; heavy(X) its long computation;
    result(X) -> plain data to compare after heavy(X) has finished."""
    X = make('reference')
    t0 = time.perf_counter()
    heavy(X)
    tref = time.perf_counter() - t0
    ref = result(X)
    hits, ok = 0, True
    for i, f in enumerate(fracs):
        Y = make(i)
        hits += interrupted(lambda: heavy(Y), f * tref) == 'interrupted'
        heavy(Y)                                  # finished after the interrupt
        ok = ok and result(Y) == ref
    report('%s: interrupted %d of %d times (%.2f s), then the same results'
           % (name, hits, len(fracs), tref), ok and hits > 0)


def fresh(name, tag):
    """A generator name for this tag: algebras are cached by their data, so
    a fresh algebra (no warm caches) per tag gets its own names."""
    return name + '_' + ''.join(ch if ch.isalnum() else '_' for ch in str(tag))


def w3over(R, cv, tag):
    nL, nW = fresh('L', tag), fresh('W', tag)
    X = NormallyOrderedExpressions(R, [nL, nW])
    L, W = X.gens()
    Lam = L * L - 3 / 10 * L.derivative(2)
    b = 16 / (22 + 5 * R(cv))
    return UniversalVertexAlgebra(
        R, X, {(nL, nL): [L.derivative(), 2 * L, 0, R(cv) / 2],
               (nL, nW): [W.derivative(), 3 * W],
               (nW, nW): [L.derivative(3) / 15 + b * Lam.derivative(),
                          3 / 10 * L.derivative(2) + 2 * b * Lam, L.derivative(),
                          2 * L, 0, R(cv) / 3]},
        degrees={nL: 2, nW: 3})


# 1. subalgebra spans: QQ (Sage path), Q(sqrt 5) and Q(c) (engine form) ----
K5.<s5> = QuadraticField(5)
Fc.<cc> = Frac(QQ['cc'])
for R, cv, W, name in [(QQ, QQ(1) / 2, 17, 'QQ'), (K5, 134 + 60 * s5, 17, 'Q(sqrt 5)'),
                       (Fc, cc, 16, 'Q(c)')]:
    def make(tag, R=R, cv=cv, name=name):
        V = w3over(R, cv, (name, tag))
        L, Wg = V.gens()
        return V.subalgebra([L.no(L) + Wg.derivative(), Wg.no(Wg)])

    def heavy(S, W=W):
        for w in range(W + 1):
            S.weight_space(w)

    def result(S, W=W):
        V = S.ambient()
        L, Wg = V.gens()
        P = (L.no(L) + Wg.derivative()).ope(Wg.no(Wg))
        return ([S.weight_space(w).dimension() for w in range(W + 1)],
                [S._span(w).echelon.labels() for w in range(W + 1)],
                [S.express(P[k]).monomial_coefficients() for k in (2, 4)],
                [L.no(Wg) in S, Wg.no(Wg).derivative(3) in S])
    check_case('subalgebra spans over %s' % name, make, heavy, result)

# 2. commutant: weight spaces and the strong generator search (GKO coset) --


def make_gko(tag):
    # a fresh algebra (and commutant) per tag: algebras are cached by their
    # data, so each tag gets its own names
    LA = LatticeVertexAlgebra(QQ, matrix([[2, 0], [0, 2]]),
                              names='J{}_'.format(tag))
    J0, J1 = LA.J(0), LA.J(1)
    LA = LA.with_conformal_vector((J0.no(J0) + J1.no(J1)) / 4)
    C = LA.commutant([LA.e((1, 0)) + LA.e((0, 1)), LA.e((-1, 0)) + LA.e((0, -1))])
    C._spaces.clear()
    return C


def heavy_gko(C):
    C.strong_generators_up_to(9)


def result_gko(C):
    return ([C.weight_space(w).dimension() for w in range(10)],
            [(g['name'], g['h'], g['origin'], dict(g['el']._monomial_coefficients))
             for g in C._gens],
            [sorted(r.items()) for r in C._space(8, None).rows()])


check_case('commutant (GKO coset): strong generators up to 9', make_gko, heavy_gko, result_gko)

# 3. ideal and quotient: Lee-Yang (Virasoro, c = -22/5, Λ = :TT: - 3/10 ∂²T) --


def make_ly(tag):
    nT = fresh('T', tag)
    T = NormallyOrderedExpressions(QQ, [nT]).gen(0)
    Vir = UniversalVertexAlgebra(QQ, T.parent(),
                                 {(nT, nT): [T.derivative(), 2 * T, 0, QQ(-22) / 10]},
                                 degrees={nT: 2})
    T = Vir.gen(0)
    return Vir.ideal([T.no(T) - 3 / 10 * T.derivative(2)])


def heavy_ly(I):
    for w in range(23):
        I.weight_space(w)


def result_ly(I):
    V = I.ambient()
    T = V.gen(0)
    Lam = T.no(T) - 3 / 10 * T.derivative(2)
    x = T.no(T.no(T.no(T.no(T)))) - 3 / 10 * T.derivative(2).no(T.no(T.no(T)))
    y = T.derivative(2).no(T.no(Lam)) + 2 * Lam.derivative(5)       # in the ideal
    return ([I.weight_space(w).dimension() for w in range(23)],
            I.weight_space(20).leading_monomials(),
            sorted(I.normal_form(x.derivative(8)).monomial_coefficients().items()),
            y in I, I.express(y).terms())


check_case('ideal (Lee-Yang) weight spaces to 22', make_ly, heavy_ly, result_ly)


def make_q(tag):
    I = make_ly(('quotient', tag))
    return I.ambient().quotient(I)


def heavy_q(Q):
    for w in range(21):
        Q.weight_space(w).dimension()       # (weight spaces are lazy)


def result_q(Q):
    return [Q.weight_space(w).dimension() for w in range(21)]


check_case('quotient (Lee-Yang) weight spaces to 20', make_q, heavy_q, result_q)

print('total %.1fs' % (time.time() - t_start))
report.result()
