# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# save(S, file) / load(file) of vertex subalgebras (VertexSubalgebra.save):
# round trips (sparse and dense echelons, graded closures, lattice), no
# closure on load, spans growing after a load, loading in a new session
# without the compiled core, saved states loaded into other ambient algebras
# with the same data (as in a new session), refusals.
# Run from the repository root:  sage tests/smoke_subalgebra_state.sage  -> RESULT: PASS
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from sage.misc.persist import dumps, loads
from sage.misc.temporary_file import tmp_filename
from vertex_algebras.universal_vertex_algebra import UniversalVertexAlgebra
from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
from vertex_algebras.lattice_vertex_algebra import LatticeVertexAlgebra
from vertex_algebras import vertex_subalgebra
from vertex_algebras.vertex_subalgebra import VertexSubalgebra
from axioms import Reporter

report = Reporter()
t_start = time.time()


def raises(exc, f, *args, **kw):
    try:
        f(*args, **kw)
    except exc as err:
        return str(err)
    return False


def same_spans(S, S2):
    if sorted(S._spans) != sorted(S2._spans):
        return False
    for k, sp in S._spans.items():
        e, e2 = sp.echelon, S2._spans[k].echelon
        if (e.labels() != e2.labels() or e.pivots() != e2.pivots()
                or e.rows() != e2.rows() or sp.inserted != S2._spans[k].inserted
                or sp.ngens != S2._spans[k].ngens or e._dense != e2._dense):
            return False
    return True


def load_into(ambient, gens, st, names=None, horizontal_grading=None,
              conformal_vector=None):
    """Load the saved state ``st`` into the subalgebra of ``ambient`` (what
    ``load`` does in a new session, where the ambient is new)."""
    args = (ambient, tuple(ambient(g) for g in gens),
            None if names is None else tuple(names), 5, horizontal_grading, None,
            conformal_vector)
    return vertex_subalgebra._load_saved_subalgebra(args, st)


class NoClosure:
    """Make the closure fail while active (loading must not run it)."""
    def __enter__(self):
        self.old = VertexSubalgebra._close
        def boom(*a, **k):
            raise AssertionError('closure called')
        VertexSubalgebra._close = boom

    def __exit__(self, *a):
        VertexSubalgebra._close = self.old


# 1. Virasoro in Heisenberg: dense spans (rank >= DENSE_MIN) ----------------
# another implementation gives another algebra with the same data (a new
# session's algebra, without the subalgebras of this one)
def heisenberg(implementation):
    J = NormallyOrderedExpressions(QQ, ['J']).gen(0)
    H = UniversalVertexAlgebra(QQ, J.parent(), {('J', 'J'): [0, 1]},
                               implementation=implementation,
                               conformal_vector=J * J / 2 + J.derivative() / 3)
    return H, H.gen('J'), H.conformal_vector()


H, J, T = heisenberg(None)
S = H.subalgebra([T], names=['T'])
dims = [S.weight_space(n).dimension() for n in range(21)]
x = T.no(T.no(T.derivative(2))).derivative(3) + 5 * T.derivative(8)
ex = repr(S.express(x))
fn = tmp_filename(ext='.sobj')
save(S, fn)
stS = S._saved_state()
report('heisenberg: has dense spans', any(sp.echelon._dense
                                          for sp in S._spans.values()))
report('heisenberg: load in the same session gives S', load(fn) is S
       and dumps(S) != dumps(stS) and len(dumps(S)) < 2000)
H2, J2, T2 = heisenberg('python')
with NoClosure():
    S2 = load_into(H2, [T2], stS, names=['T'])
report('heisenberg: loaded without closure', S2 is not S
       and list(S2.variable_names()) == ['T'])
report('heisenberg: spans identical', same_spans(S, S2))
x2 = T2.no(T2.no(T2.derivative(2))).derivative(3) + 5 * T2.derivative(8)
report('heisenberg: express (dense coordinates) identical',
       repr(S2.express(x2)) == ex and S2(S2.express(x2)) == x2)
report('heisenberg: membership', x2 in S2 and J2.no(J2) not in S2)
report('heisenberg: new weights after load',
       S2.weight_space(22).dimension() == S.weight_space(22).dimension()
       == Partitions(22).cardinality() - Partitions(21).cardinality())
report('heisenberg: same subalgebra returned again (state merged)',
       load_into(H2, [T2], stS, names=['T']) is S2
       and H2.subalgebra([T2], names=['T']) is S2)

# 2. Graded closure in a lattice VA (sl2 level 1 from e^{±α}) ---------------
def a1(implementation):
    L = LatticeVertexAlgebra(QQ, matrix([[2]]), implementation=implementation)
    Jl = L.J(0)
    return L.with_conformal_vector(Jl.no(Jl) / 4).with_horizontal_grading('lattice')


L = a1(None)
A = L.subalgebra([L.e((1,)), L.e((-1,))], names=['E', 'F'], horizontal_grading='ambient')
dA = {(W, q): A.weight_space(W, horizontal_grade=q).dimension()
      for W in range(9) for q in range(-2, 3)}
y = L.e((1,)).no(L.e((-1,))).derivative(2)
exA = repr(A.express(y))
fnA = tmp_filename(ext='.sobj')
save(A, fnA)
stA = A._saved_state()
L2 = a1('python')
with NoClosure():
    A2 = load_into(L2, [L2.e((1,)), L2.e((-1,))], stA, names=['E', 'F'],
                   horizontal_grading='ambient')
report('lattice: generators and origins',
       A2.strong_generator_info() == A.strong_generator_info())
report('lattice: spans identical', same_spans(A, A2))
report('lattice: dims and express',
       all(A2.weight_space(W, horizontal_grade=q).dimension() == d
           for (W, q), d in dA.items())
       and repr(A2.express(L2._el_from_dict(dict(y._monomial_coefficients))))
       == exA)
report('lattice: grades of the strong generators',
       [g.horizontal_grade() for g in A2.gens()] == [1, -1, 0])
# with a conformal vector (the Sugawara vector of the subalgebra): its file
# is the same construction's, loaded with or without the vector
AT = A.with_conformal_vector('ambient')
stT = AT._saved_state()
L3 = a1('engine')
with NoClosure():
    A3T = load_into(L3, [L3.e((1,)), L3.e((-1,))], stT, names=['E', 'F'],
                    horizontal_grading='ambient', conformal_vector='ambient')
report('lattice with T: loaded without closure, spans identical, c = 1',
       same_spans(AT, A3T) and A3T.central_charge() == 1
       and A3T._core is L3.subalgebra([L3.e((1,)), L3.e((-1,))], names=['E', 'F'],
                                      horizontal_grading='ambient')._core)

# 3. load in a new session without the compiled core (another "machine") -----
code = '''
from vertex_algebras import lattice_vertex_algebra as lva
from vertex_algebras.vertex_subalgebra import VertexSubalgebra
assert lva._LVA_ENGINE is None
def boom(*a, **k):
    raise AssertionError('closure called')
VertexSubalgebra._close = boom
A = load(%r)
L = A.ambient()
print('DIMS', [A.weight_space(W, horizontal_grade=0).dimension() for W in range(9)])
print('EXPR', repr(A.express(L.e((1,)).no(L.e((-1,))).derivative(2))))
''' % fnA
env = dict(os.environ, VERTEX_ALGEBRAS_NO_ENGINE='1')
out = subprocess.run(['sage', '-c', code], env=env, capture_output=True,
                     text=True).stdout
report('no core: loads and agrees',
       ('DIMS %s' % [dA[(W, 0)] for W in range(9)]) in out
       and ('EXPR ' + exA) in out, out[-300:] if 'DIMS' not in out else '')

# 4. Refusals ----------------------------------------------------------------
L3 = LatticeVertexAlgebra(QQ, matrix([[2]]))
J3 = L3.J(0)
L3 = L3.with_conformal_vector(J3.no(J3) / 4 + J3.derivative() / 2)    # other weights
L3 = L3.with_horizontal_grading('lattice')
report('refused: other conformal weights',
       'conformal weights' in (raises(ValueError, load_into, L3,
                               [L3.e((1,)), L3.e((-1,))], stA, names=['E', 'F'],
                               horizontal_grading='ambient') or ''))
L4 = a1(None)
report('refused: grading option differs',
       'grading' in (raises(ValueError, load_into, L4,
                            [L4.e((1,)), L4.e((-1,))], stA, names=['E', 'F']) or ''))
report('refused: names differ',
       'other generators' in (raises(ValueError, load_into, L4,
                              [L4.e((1,)), L4.e((-1,))], stA, names=['X', 'Y'],
                              horizontal_grading='ambient') or ''))
# for rank 1 the 'trivial' and 'standard' cocycles are the same matrix
L5 = LatticeVertexAlgebra(QQ, matrix([[2]]), epsilon='trivial')
J5 = L5.J(0); L5 = L5.with_conformal_vector(J5.no(J5) / 4).with_horizontal_grading('lattice')
report('accepted: equal cocycle given differently',
       load_into(L5, [L5.e((1,)), L5.e((-1,))], stA, names=['E', 'F'],
                 horizontal_grading='ambient').variable_names()
       == ('E', 'F', 'S0'))
L6 = LatticeVertexAlgebra(QQ, matrix([[4]]))
J6 = L6.J(0); L6 = L6.with_conformal_vector(J6.no(J6) / 8).with_horizontal_grading('lattice')
report('refused: other Gram matrix',
       'different ambient' in (raises(ValueError, load_into, L6,
                               [L6.e((1,)), L6.e((-1,))], stA, names=['E', 'F'],
                               horizontal_grading='ambient') or ''))
bad = dict(stA)
bad['format'] = vertex_subalgebra.STATE_FORMAT + 1
report('refused: other state format',
       'state format' in (raises(ValueError, load_into, L4,
                          [L4.e((1,)), L4.e((-1,))], bad, names=['E', 'F'],
                          horizontal_grading='ambient') or ''))
report('refused: not a saved state',
       'not a saved' in (raises(ValueError, load_into, L4, [L4.e((1,))],
                                {'something': 'else'}) or ''))
report('a refused load leaves no subalgebra behind',
       L4.subalgebra([L4.e((1,)), L4.e((-1,))], names=['E', 'F'],
                     horizontal_grading='ambient').variable_names() == ('E', 'F', 'S0'))

print('total %.1fs' % (time.time() - t_start))
# engine-form spans (number field, Frac(QQ[c]) with a lifted denominator):
# saved, loaded into a new ambient (no closure), same spans and expressions
from vertex_algebras._keyed_spans import _EngineSpan, _frac_engine


def w3over(R, cv, implementation=None):
    X = NormallyOrderedExpressions(R, ['L', 'W'])
    L, W = X.gens()
    Lam = L * L - 3 / 10 * L.derivative(2)
    b = 16 / (22 + 5 * R(cv))
    return UniversalVertexAlgebra(
        R, X, {('L', 'L'): [L.derivative(), 2 * L, 0, R(cv) / 2],
               ('L', 'W'): [W.derivative(), 3 * W],
               ('W', 'W'): [L.derivative(3) / 15 + b * Lam.derivative(),
                            3 / 10 * L.derivative(2) + 2 * b * Lam, L.derivative(),
                            2 * L, 0, R(cv) / 3]},
        degrees={'L': 2, 'W': 3}, implementation=implementation)


K5.<s5> = QuadraticField(5)
Fc.<cc> = Frac(QQ['cc'])
for R, cv, name in [(K5, 134 + 60 * s5, 'Q(sqrt 5)'), (Fc, cc, 'Q(c)')]:
    V1 = w3over(R, cv)
    L1, W1 = V1.gens()
    S1 = V1.subalgebra([L1.no(L1) + W1.derivative(), W1.no(W1)])
    d1 = [S1.weight_space(w).dimension() for w in range(13)]
    x1 = (L1.no(L1) + W1.derivative()).ope(W1.no(W1))[2]
    e1 = S1.express(x1)
    engine = all(isinstance(sp.echelon.inner, _EngineSpan) for sp in S1._spans.values())
    st1 = S1._saved_state()
    V2 = w3over(R, cv, 'python')        # another algebra (and no engine): S2 is new
    L2, W2 = V2.gens()
    with NoClosure():
        S2 = load_into(V2, [L2.no(L2) + W2.derivative(), W2.no(W2)], st1)
    d2 = [S2.weight_space(w).dimension() for w in range(13)]
    x2 = (L2.no(L2) + W2.derivative()).ope(W2.no(W2))[2]
    report('engine-form spans over %s: saved, loaded (same dims, express)' % name,
           (engine or _frac_engine() is None or V1._engine() is None)
           and d1 == d2 and e1.monomial_coefficients() == S2.express(x2).monomial_coefficients())

report.result()
