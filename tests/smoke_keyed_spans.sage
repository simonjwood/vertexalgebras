# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# KeyedSpan: the dense backend (insert_many of >= DENSE_MIN vectors over
# QQ, forced with DENSE_FILL = 0) against the incremental sparse one: labels,
# pivots, rows, membership, coordinates (incl. keys outside the columns, empty
# spans), export/import; the automatic choice (fill-in): sparse batches stay
# sparse, dense ones switch, also in the middle of a batch and after earlier
# batches, with the same labels, pivots and rows.
# Run from the repository root:  sage tests/smoke_keyed_spans.sage  -> RESULT: PASS
import os
import random
import time
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))
from vertex_algebras import _keyed_spans
from vertex_algebras._keyed_spans import KeyedSpan
from axioms import Reporter

report = Reporter()
random.seed(1r)
FILL = _keyed_spans.DENSE_FILL


def rvec(keys, density):
    return {k: QQ(random.randint(-5r, 5r)) / random.randint(1r, 4r)
            for k in keys if random.random() < density}


def combo(vs):
    out = {}
    for v in vs:
        c = QQ(random.randint(-3r, 3r))
        for k, a in v.items():
            out[k] = out.get(k, 0) + c * a
    return {k: a for k, a in out.items() if a}


ok_struct = ok_member = ok_coord = ok_export = True
for trial in range(6):
    keys = list(range(40 + 10 * trial))
    base = [rvec(keys, 0.3) for _ in range(30 + 5 * trial)]
    vecs = base + [combo(random.sample(base, 3)) for _ in range(30)]
    random.shuffle(vecs)
    labels = list(range(len(vecs)))
    order = (lambda k: k) if trial % 2 else None
    D = KeyedSpan(QQ, key_order=order)
    S = KeyedSpan(QQ, key_order=order)
    _keyed_spans.DENSE_FILL = 0          # always dense
    D.insert_many(vecs, labels=labels)
    _keyed_spans.DENSE_FILL = FILL
    for v, l in zip(vecs, labels):
        S.insert(v, l)
    ok_struct = ok_struct and D._dense and not S._dense and \
        D.labels() == S.labels() and D.rank() == S.rank()
    if order is not None:
        ok_struct = ok_struct and D.pivots() == S.pivots() and D.rows() == S.rows()
    tests = ([combo(random.sample(vecs, 4)) for _ in range(10)]
             + [rvec(keys, 0.3) for _ in range(10)]
             + [dict(rvec(keys, 0.3), extra=1)]               # key outside
             + [{}])
    for t in tests:
        ok_member = ok_member and D.contains(t) == S.contains(t)
        cd, cs = D.coordinates(t), S.coordinates(t)
        ok_coord = ok_coord and cd == cs
        if cd is not None:                     # coordinates reproduce t
            r = {}
            for l, c in cd.items():
                for k, a in vecs[l].items():
                    r[k] = r.get(k, 0) + c * a
            ok_coord = ok_coord and {k: a for k, a in r.items() if a} == t
    E = KeyedSpan(QQ, key_order=order)
    E._import(D._export())
    ok_export = ok_export and E.labels() == D.labels() and all(
        E.coordinates(t) == D.coordinates(t) for t in tests)

report('dense = sparse: labels, rank, pivots, rows', ok_struct)
report('dense = sparse: membership', ok_member)
report('dense = sparse: coordinates (and they reproduce v)', ok_coord)
report('export/import keeps the dense echelon', ok_export)

# dense blocks compute their reduced echelon form only on demand (columns P
# with A[:, P] invertible from a mod-p rank profile); without a usable
# prime (PROFILE_PRIMES = ()) they compute it at once; both give the same
# rows, pivots, membership and coordinates, also after export/import
PRIMES = _keyed_spans.PROFILE_PRIMES
ok_lazy = ok_eager = ok_same = ok_lexport = True
for trial in range(4):
    keys = list(range(50))
    base = [rvec(keys, 0.4) for _ in range(35)]
    vecs = base + [combo(random.sample(base, 3)) for _ in range(20)]
    random.shuffle(vecs)
    order = (lambda k: k) if trial % 2 else None
    tests = ([combo(random.sample(vecs, 3)) for _ in range(5)]
             + [rvec(keys, 0.3) for _ in range(5)])
    _keyed_spans.DENSE_FILL = 0
    D = KeyedSpan(QQ, key_order=order)
    D.insert_many(vecs, labels=list(range(len(vecs))))
    _keyed_spans.PROFILE_PRIMES = ()
    X = KeyedSpan(QQ, key_order=order)
    X.insert_many(vecs, labels=list(range(len(vecs))))
    _keyed_spans.PROFILE_PRIMES = PRIMES
    _keyed_spans.DENSE_FILL = FILL
    ok_lazy = ok_lazy and D._dense and all(b._E is None for b in D._blocks if b._dense)
    ok_eager = ok_eager and X._dense and all(b._E is not None for b in X._blocks if b._dense)
    st = D._export()
    ok_lexport = ok_lexport and all(b['E'] is None for b in st['blocks'] if b['dense'])
    F = KeyedSpan(QQ, key_order=order)
    F._import(st)
    ok_same = ok_same and all(
        D.contains(t) == X.contains(t) == F.contains(t)
        and D.coordinates(t) == X.coordinates(t) == F.coordinates(t) for t in tests)
    ok_same = (ok_same and D.labels() == X.labels() == F.labels()
               and D.pivots() == X.pivots() == F.pivots()
               and D.rows() == X.rows() == F.rows()
               and all(D.reduce(t) == X.reduce(t) for t in tests))
report('lazy dense echelon: none computed by inserts', ok_lazy)
report('no usable profile prime: echelon form at once', ok_eager)
report('lazy = eager: membership, coordinates, labels, pivots, rows', ok_same)
report('export of a lazy block has no echelon form; import rebuilds it', ok_lexport)

# number fields (also a tower), QQbar, AA: exactly over QQ (_ExpandedEchelon)
# = plain elimination over the field (_Block): labels, membership,
# coordinates, with a key order pivots, rows, normal forms; large batches
# (dense over QQ), export / import, in-place relabelling, left_kernel; QQbar
# grows its field when a vector needs it
from vertex_algebras._keyed_spans import _ExpandedEchelon, _Block, _left_kernel_block, left_kernel
Kq.<sq> = QuadraticField(5)
K3.<r3> = NumberField(QQ['z']([-1, -1, 0, 3]))
Kt.<it> = Kq.extension(QQ['y']([1, 0, 1]))
fields = [(Kq, [sq, 1 + sq / 3]), (K3, [r3, r3^2 - 1]), (Kt, [it, sq * it + 1]),
          (QQbar, [QQbar(2).sqrt(), QQbar.zeta(3)]), (AA, [AA(3).sqrt(), AA(2)^(1/3)])]


def fvec(keys, density, gens):
    return {k: QQ(random.randint(-3r, 3r)) + random.choice(gens) * random.randint(-2r, 2r)
            for k in keys if random.random() < density}


def fcombo(vs, gens):
    out = {}
    for v in vs:
        c = QQ(random.randint(-2r, 2r)) + random.choice(gens)
        for k, a in v.items():
            out[k] = out.get(k, 0) + c * a
    return {k: a for k, a in out.items() if a}


ok_fields = ok_fdense = ok_fexport = ok_fkernel = ok_grow = True
for R, gens in fields:
    for trial in range(2):
        keys = list(range(12 + 4 * trial))
        base = [fvec(keys, 0.4, gens) for _ in range(8 + trial)]
        vecs = base + [fcombo(random.sample(base, 2), gens) for _ in range(4)]
        random.shuffle(vecs)
        order = (lambda k: k) if trial else None
        E = KeyedSpan(R, key_order=order)
        B = _Block(R, key_order=order)
        B._combos_ok = True
        flags = E.insert_many(vecs, labels=['v%d' % i for i in range(len(vecs))])
        ref = [B.insert(v, 'v%d' % i) for i, v in enumerate(vecs)]
        tests = [fcombo(random.sample(vecs, 3), gens) for _ in range(4)] + [fvec(keys, 0.3, gens)]
        ok_fields = (ok_fields and type(E) is _ExpandedEchelon and flags == ref
                     and E.labels() == B.labels() and E.rank() == B.rank()
                     and all(E.contains(t) == B.contains(t) and E.coordinates(t) == B.coordinates(t)
                             for t in tests))
        if order is not None:
            ok_fields = (ok_fields and E.pivots() == sorted(B.pivots(), reverse=True)
                         and E.row_dict() == B.row_dict()
                         and all(E.normal_form(t) == B.reduce(t) for t in tests))
        F = KeyedSpan(R, key_order=order)
        F._import(E._export())
        ok_fexport = (ok_fexport and F.labels() == E.labels()
                      and all(F.coordinates(t) == E.coordinates(t) for t in tests))
        E._labels[:] = ['w%d' % i for i in range(E.rank())]       # in-place relabelling
        ok_fexport = ok_fexport and all(
            set(E.coordinates(t) or {}) <= set(E._labels) for t in tests)
        ok_fkernel = ok_fkernel and left_kernel(vecs, R) == _left_kernel_block(vecs, R)
    # a large batch: the QQ span switches to the dense backend
    keys = list(range(30))
    base = [fvec(keys, 0.5, gens) for _ in range(20)]
    vecs = base + [fcombo(random.sample(base, 3), gens) for _ in range(10)]
    E = KeyedSpan(R, key_order=lambda k: k)
    B = _Block(R, key_order=lambda k: k)
    B._combos_ok = True
    flags = E.insert_many(vecs)
    ref = [B.insert(v) for v in vecs]
    ok_fdense = (ok_fdense and E._dense and flags == ref
                 and E.row_dict() == B.row_dict())
# Frac(K[c]), Frac(K[a, b]): expanded over Frac(QQ[x...]), = elimination
# over the field (small: Sage's gcds over number fields stall on larger
# ones); Frac(L[a, b]), L a tower (no usable Sage arithmetic to compare
# with): exactness checks, coordinates reproduce the vector
FKc.<kc> = Frac(Kq['kc'])
FKab.<ka, kb> = Frac(Kq['ka, kb'])
for R, gens in [(FKc, [sq, kc / (kc + sq), 1 / (kc^2 - 5)]), (FKab, [sq, ka / (kb - sq), kb])]:
    for trial in range(2):
        keys = list(range(5 + trial))
        base = [fvec(keys, 0.5, gens) for _ in range(3 + trial)]
        vecs = base + [fcombo(random.sample(base, 2), gens) for _ in range(2)]
        random.shuffle(vecs)
        order = (lambda k: k) if trial else None
        E = KeyedSpan(R, key_order=order)
        B = _Block(R, key_order=order)
        B._combos_ok = True
        labs = ['v%d' % i for i in range(len(vecs))]
        flags = E.insert_many(vecs, labels=labs)
        ref = [B.insert(v, l) for v, l in zip(vecs, labs)]
        tests = [fcombo(random.sample(vecs, 2), gens) for _ in range(2)] + [fvec(keys, 0.3, gens)]
        ok_fields = (ok_fields and type(E) is _ExpandedEchelon and flags == ref
                     and all(E.contains(t) == B.contains(t) and E.coordinates(t) == B.coordinates(t)
                             for t in tests))
        if order is not None:
            ok_fields = (ok_fields and E.pivots() == sorted(B.pivots(), reverse=True)
                         and E.row_dict() == B.row_dict()
                         and all(E.normal_form(t) == B.reduce(t) for t in tests))
        ok_fkernel = ok_fkernel and left_kernel(vecs, R) == _left_kernel_block(vecs, R)
FLab.<la, lb> = Frac(Kt['la, lb'])
gens = [it, la / (lb - it), sq * lb]
keys = list(range(6))
base = [fvec(keys, 0.5, gens) for _ in range(4)]
E = KeyedSpan(FLab, key_order=lambda k: k)
flags = E.insert_many(base + [fcombo(base[:2], gens)], labels=list('abcde'))
t = fcombo(base, gens)
co = E.coordinates(t)
back = {}
for l, c in co.items():
    for k, a in base['abcd'.index(l)].items():
        back[k] = back.get(k, 0) + c * a
ok_fields = (ok_fields and flags == [True] * 4 + [False] and E.contains(t)
             and {k: a for k, a in back.items() if a} == t
             and not E.contains({0: la, 1: lb, 2: it, 3: sq, 4: la * lb, 5: 1}) or E.rank() == 6)
report('number fields, tower, QQbar, AA over QQ; Frac(K[c]), Frac(L[a, b]) over QQ(x...): '
       'labels, membership, coordinates, rows', ok_fields)
report('... large batches go dense over QQ, same results', ok_fdense)
report('... export / import, in-place relabelling', ok_fexport)
report('... left_kernel = elimination over the field', ok_fkernel)
E = KeyedSpan(QQbar)
E.insert({1: QQbar(2).sqrt(), 2: 1})
d0 = E._F.d
E.insert({1: QQbar(3).sqrt(), 3: 1})                 # outside Q(sqrt 2): field grows
ok_grow = (d0 == 2 and E._F.d == 4 and E.rank() == 2
           and E.contains({1: QQbar(2).sqrt() + QQbar(3).sqrt(), 2: 1, 3: 1})
           and not E.contains({1: QQbar(5).sqrt()})
           and E.coordinates({1: 2 * QQbar(3).sqrt(), 3: 2}) == {None: 2})
G = KeyedSpan(QQbar)
G.insert({(): QQbar(1)}, 'one')
ok_grow = (ok_grow and G.contains({(): QQbar(7).sqrt()})       # in the span, outside QQ
           and G.coordinates({(): QQbar(7).sqrt()}) == {'one': QQbar(7).sqrt()}
           and G.reduce({(): QQbar(7).sqrt(), 1: 1}) == {1: 1})
report('QQbar: the field grows (degree 2 -> 4), the span is kept', ok_grow)

# rational function fields Frac(QQ[x...]): the C++ elimination (_FracBlock)
# = the Python one (_Block): labels, membership, coordinates (combos built
# on demand), with a key order pivots, rows, normal forms; blocks merging
# (rebuilt), export / import, left_kernel
from vertex_algebras import _keyed_spans as SLAm
from vertex_algebras import universal_vertex_algebra as UVAm
if SLAm._frac_engine() is None:
    # skipped only where the engines are switched off too (developer
    # switches): this module alone missing is an error
    report('C++ linear algebra switched off (%s)' % SLAm._LAE_OFF,
           UVAm._UVA_ENGINE is None)
else:
    from vertex_algebras._keyed_spans import _FracBlock
    Fc1.<x1> = Frac(QQ['x1'])
    Fc2.<y1, y2> = Frac(QQ['y1, y2'])
    ok_frac = ok_fmerge = ok_fexp2 = ok_fker = True
    for R, gens in [(Fc1, [x1, 1 / (x1 + 2), x1^2 / 3]), (Fc2, [y1, y2 / (y1 - 1), y1 * y2 + 1])]:
        for trial in range(3):
            keys = list(range(10 + 3 * trial))
            base = [fvec(keys, 0.4, gens) for _ in range(7 + trial)]
            vecs = base + [fcombo(random.sample(base, 2), gens) for _ in range(4)]
            random.shuffle(vecs)
            order = (lambda k: (k % 3, -k)) if trial else None
            E = KeyedSpan(R, key_order=order)
            B = _Block(R, key_order=order)
            B._combos_ok = True
            labs = ['v%d' % i for i in range(len(vecs))]
            flags = E.insert_many(vecs, labels=labs)
            ref = [B.insert(v, l) for v, l in zip(vecs, labs)]
            tests = [fcombo(random.sample(vecs, 3), gens) for _ in range(4)] + [fvec(keys, 0.3, gens)]
            ok_frac = (ok_frac and all(type(b) is _FracBlock for b in E._blocks) and flags == ref
                       and E.labels() == B.labels()
                       and all(E.contains(t) == B.contains(t) and E.coordinates(t) == B.coordinates(t)
                               for t in tests))
            if order is not None:
                ok_frac = (ok_frac and E.pivots() == sorted(B.pivots(), key=order, reverse=True)
                           and E.row_dict() == B.row_dict()
                           and all(E.normal_form(t) == B.reduce(t) for t in tests))
            F2 = KeyedSpan(R, key_order=order)
            F2._import(E._export())
            ok_fexp2 = ok_fexp2 and F2.labels() == E.labels() and all(
                F2.coordinates(t) == E.coordinates(t) for t in tests)
        # two blocks, then a vector joining them: merged by a rebuild
        M = KeyedSpan(R, key_order=lambda k: k)
        a = [fvec(range(0, 5), 0.8, gens) for _ in range(3)]
        b = [fvec(range(10, 15), 0.8, gens) for _ in range(3)]
        M.insert_many(a + b)
        nb = len(M._blocks)
        j = dict(a[0]); j.update(b[0]); j[20] = R(1)
        ok_fmerge = ok_fmerge and nb == 2 and M.insert(j) and len(M._blocks) == 1 and M.rank() == 7 \
            and M.contains(fcombo(a + b + [j], gens))
    # left_kernel: C++ = Python elimination
    for R, gens in [(Fc1, [x1, 1 / (x1 + 2)]), (Fc2, [y1, y2 / (y1 - 1)])]:
        keys = list(range(8))
        base = [fvec(keys, 0.5, gens) for _ in range(5)]
        vecs = base + [fcombo(random.sample(base, 2), gens) for _ in range(3)] + [{}]
        SLAm._USE_FRAC_ENGINE = False
        ref = left_kernel(vecs, R)
        SLAm._USE_FRAC_ENGINE = True
        ok_fker = ok_fker and left_kernel(vecs, R) == ref
    # Ctrl-C in a long C++ batch: the committed inserts are recorded (each
    # insert is atomic), the rest of the batch gives the uninterrupted span
    import signal

    def _alarm(*a):
        raise KeyboardInterrupt
    keys = list(range(32))
    gl = [x1, 1 / (x1 + 2), x1^2 / 3, (x1 - 1) / (x1^2 + 1)]
    big = [fvec(keys, 0.9, gl) for _ in range(36)]
    Eref = KeyedSpan(Fc1, key_order=lambda k: k)
    t0 = time.time()
    Eref.insert_many(big, labels=list(range(36)))
    tfull = time.time() - t0
    E = KeyedSpan(Fc1, key_order=lambda k: k)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(tfull / 3))
    try:
        E.insert_many(big, labels=list(range(36)))
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    signal.signal(signal.SIGALRM, old)
    n = E.rank()
    last = E.labels()[-1] if n else -1
    E.insert_many(big[last + 1:], labels=list(range(last + 1, 36)))
    report('Q(x) in C++: Ctrl-C in a batch (%s after %d of %d independent vectors, %.1f s), '
           'then the same span' % (got, n, Eref.rank(), tfull),
           got == 'interrupted' and 0 < n < Eref.rank() and E.labels() == Eref.labels()
           and E.row_dict() == Eref.row_dict())
    # the Python elimination (another ring: GF(p)) is interrupt-safe too
    Fp = GF(10007)
    vp = [{k: Fp.random_element() for k in range(220) if random.random() < 0.9} for _ in range(230)]
    Pref = KeyedSpan(Fp)
    t0 = time.time()
    Pref.insert_many(vp, labels=list(range(230)))
    tfull = time.time() - t0
    P = KeyedSpan(Fp)
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(tfull / 3))
    try:
        P.insert_many(vp, labels=list(range(230)))
        got = 'finished'
    except KeyboardInterrupt:
        got = 'interrupted'
    signal.setitimer(signal.ITIMER_REAL, float(0r))
    signal.signal(signal.SIGALRM, old)
    n = P.rank()
    last = P.labels()[-1] if n else -1
    P.insert_many(vp[last + 1:], labels=list(range(last + 1, 230)))
    test = [sum((Fp.random_element() * vp[j].get(k, 0) for j in range(3)), Fp(0)) for k in range(220)]
    test = {k: c for k, c in enumerate(test) if c}
    report('Python elimination: Ctrl-C in a batch (%s after %d of %d, %.1f s), then the same span'
           % (got, n, Pref.rank(), tfull),
           got == 'interrupted' and 0 < n < Pref.rank() and P.labels() == Pref.labels()
           and P.coordinates(test) == Pref.coordinates(test))
    report('Q(x), Q(y1, y2) in C++: labels, membership, coordinates, pivots, rows', ok_frac)
    report('... blocks joined by a vector: merged (rebuilt)', ok_fmerge)
    report('... export / import', ok_fexp2)
    report('... left_kernel = the Python elimination', ok_fker)

# a batch of zero vectors (forced dense): no block at all, rank 0
_keyed_spans.DENSE_FILL = 0
Z = KeyedSpan(QQ)
Z.insert_many([{}] * _keyed_spans.DENSE_MIN)
_keyed_spans.DENSE_FILL = FILL
report('rank 0 (zero batch, forced dense)', not Z._blocks and Z.rank() == 0
       and Z.contains({}) and not Z.contains({1: 1}) and Z.coordinates({}) == {}
       and Z.coordinates({1: 1}) is None)
Z = KeyedSpan(QQ)
Z.insert_many([{}] * _keyed_spans.DENSE_MIN)
report('zero batch stays sparse', not Z._dense and Z.rank() == 0)

# the automatic choice: fill-in of the incremental elimination
def incremental(vecs, order):
    S = KeyedSpan(QQ, key_order=order)
    for l, v in enumerate(vecs):
        S.insert(v, l)
    return S


def same(E, S, order):
    return (E.labels() == S.labels() and E.rank() == S.rank()
            and (order is None or (E.pivots() == S.pivots()
                                   and E.rows() == S.rows())))


ok_sparse = ok_switch = ok_late = ok_member = True
for trial in range(4):
    order = (lambda k: k) if trial % 2 else None
    keys = list(range(3000))
    # very sparse batch (a chain: no fill-in): stays sparse
    sp = [rvec(random.sample(keys, 3), 1) for _ in range(150)]
    sp += [combo(random.sample(sp, 2)) for _ in range(30)]
    E = KeyedSpan(QQ, key_order=order)
    E.insert_many(sp, labels=list(range(len(sp))))
    ok_sparse = ok_sparse and not E._dense and same(E, incremental(sp, order), order)
    # sparse vectors, then dense ones on few keys: switches in the middle
    few = keys[:60]
    dn = [rvec(few, 0.5) for _ in range(50)] + [combo(random.sample(sp[:50], 2)) for _ in range(5)]
    batch = sp[:50] + dn
    E = KeyedSpan(QQ, key_order=order)
    out = E.insert_many(batch, labels=list(range(len(batch))))
    S = incremental(batch, order)
    ok_switch = ok_switch and E._dense and same(E, S, order) and \
        out == [l in set(S.labels()) for l in range(len(batch))]
    # an earlier sparse batch, then a dense one (old vectors kept)
    E = KeyedSpan(QQ, key_order=order)
    E.insert_many(sp[:60], labels=list(range(60)))
    ok_late = ok_late and not E._dense
    E.insert_many(dn, labels=list(range(60, 60 + len(dn))))
    S = incremental(sp[:60] + dn, order)
    ok_late = ok_late and E._dense and same(E, S, order)
    tests = [combo(random.sample(sp[:60] + dn, 3)) for _ in range(5)] + [rvec(few, 0.5)]
    ok_member = ok_member and all(E.contains(t) == S.contains(t)
                                  and E.coordinates(t) == S.coordinates(t)
                                  for t in tests)

report('sparse batch stays sparse (same labels, pivots, rows)', ok_sparse)
report('switch in the middle of a batch: same labels, pivots, rows, flags', ok_switch)
report('switch after an earlier sparse batch: same labels, pivots, rows', ok_late)
report('after a switch: membership and coordinates', ok_member)

# dense blocks of small codimension: batches tested against an exact
# annihilator of the span (also the prefix elimination of a tall first
# batch); the same labels, pivots and rows as incremental elimination and as
# the plain dense path (ANNIHILATOR_MAX = 0)
ok_ann = used = True
AMAX = _keyed_spans.ANNIHILATOR_MAX
for trial in range(2):
    order = (lambda k: k) if trial % 2 else None
    keys = list(range(140))         # > ANNIHILATOR_MAX: the prefix path too
    base = [rvec(keys, 0.08) for _ in range(len(keys) - 6)]
    first = base[:len(base) // 2] + [combo(random.sample(base[:20], 3)) for _ in range(40)]
    first += base[len(base) // 2:]
    random.shuffle(first)
    later = [[combo(random.sample(base, 2)) for _ in range(50)]
             + [rvec(keys, 0.08) for _ in range(3)]
             + [rvec([1000 + 10 * trial + j], 1) for j in range(2)]
             + [combo([base[0], rvec([1000 + 10 * trial], 1)])] for _ in range(3)]
    batches = [first] + later
    allv = [v for b in batches for v in b]
    S = incremental(allv, order)
    res = []
    for amax in (AMAX, 0):
        _keyed_spans.ANNIHILATOR_MAX = amax
        E = KeyedSpan(QQ, key_order=order)
        n = 0
        flags = []
        for b in batches:
            flags += E.insert_many(b, labels=list(range(n, n + len(b))))
            n += len(b)
        res.append((E.labels(), flags))
        ok_ann = ok_ann and E._dense and same(E, S, order)
        if amax:
            used = used and any(getattr(B, '_K', None) is not None for B in E._blocks)
    _keyed_spans.ANNIHILATOR_MAX = AMAX
    ok_ann = ok_ann and res[0] == res[1]
report('annihilator of dense blocks: same labels, pivots, rows, flags', ok_ann and used)

# blocks: connected components of the keys, each sparse or dense by its own
# fill; against the single-block engine _Block (the former KeyedSpan)
from vertex_algebras._keyed_spans import _Block


def reference(vecs, order):
    B = _Block(QQ, key_order=order)
    for l, v in enumerate(vecs):
        B.insert(v, l)
    return B


def same_ref(E, B, order, tests):
    # without a key order the pivots are arbitrary keys (backend-dependent)
    return (E.labels() == B.labels() and E.rank() == B.rank()
            and (order is None or (E.pivots() == B.pivots() and E.rows() == B.rows()))
            and all(E.contains(t) == B.contains(t) and E.coordinates(t) == B.coordinates(t)
                    and (E.reduce(t) == B.reduce(t) if order is not None
                         else (not E.reduce(t)) == (not B.reduce(t)))
                    for t in tests))


ok_blocks = ok_merge = ok_relabel = ok_old = ok_bexport = True
for trial in range(4):
    order = (lambda k: k) if trial % 2 else None
    G, per = 12, 40
    groups = [list(range(g * per, (g + 1) * per)) for g in range(G)]
    # 55 vectors per block (>= DENSE_MIN, so each block may switch)
    base = [rvec(groups[g], 0.5) for g in range(G) for _ in range(50)]
    deps = [combo([base[50 * g + i] for i in random.sample(range(50), 3)])
            for g in range(G) for _ in range(5)]
    vecs = base + deps
    random.shuffle(vecs)
    tests = ([combo(random.sample(vecs, 2)) for _ in range(6)]
             + [combo(random.sample(base[50 * g: 50 * g + 50], 4)) for g in range(4)]
             + [rvec(groups[0], 0.5), dict(rvec(groups[1], 0.5), extra=1), {}])
    E = KeyedSpan(QQ, key_order=order)
    flags = E.insert_many(vecs, labels=list(range(len(vecs))))
    B = reference(vecs, order)
    # overall fill ~1/12 of a block's: the blocks switch, one span would not
    ok_blocks = (ok_blocks and len(E._blocks) == G and all(b._dense for b in E._blocks)
                 and flags == [l in set(B.labels()) for l in range(len(vecs))]
                 and same_ref(E, B, order, tests))
    # merges: a vector joining two sparse blocks, a batch joining a dense
    # block with sparse ones, a single vector joining two dense blocks
    E = KeyedSpan(QQ, key_order=order)
    a = [rvec(random.sample(groups[0], 4), 1) for _ in range(5)]
    b = [rvec(random.sample(groups[1], 4), 1) for _ in range(5)]
    bridge = dict(a[0]); bridge.update({groups[1][0]: QQ(3), groups[2][0]: QQ(1)})
    seq = a + b
    for i, v in enumerate(seq):
        E.insert(v, i)
    nb = len(E._blocks)
    E.insert(bridge, len(seq)); seq.append(bridge)
    dense_part = [rvec(groups[3], 0.6) for _ in range(60)]
    jv = rvec(groups[3], 0.6)
    jv[groups[0][1]] = QQ(1)                 # joins the dense block and a's
    E.insert_many(dense_part, labels=list(range(len(seq), len(seq) + 60))); seq += dense_part
    had_dense = any(bl._dense for bl in E._blocks)
    extra = [jv] + [rvec(groups[4], 0.5) for _ in range(3)]
    E.insert_many(extra, labels=list(range(len(seq), len(seq) + 4))); seq += extra
    more = [rvec(groups[5], 0.6) for _ in range(60)]
    E.insert_many(more, labels=list(range(len(seq), len(seq) + 60))); seq += more
    tie = {groups[3][0]: QQ(1), groups[5][0]: QQ(-2), groups[6][0]: QQ(1)}
    E.insert(tie, len(seq)); seq.append(tie)
    B = reference(seq, order)
    ok_merge = (ok_merge and nb >= 2 and had_dense
                and same_ref(E, B, order, [combo(random.sample(seq, 3)) for _ in range(6)]
                             + [rvec(groups[3] + groups[5], 0.3)]))
    # relabelling in place after a batch (vertex_ideal._add_many)
    E = KeyedSpan(QQ, key_order=order)
    E.insert_many(vecs[:100], labels=list(range(100)))
    n0 = E.rank()
    out = E.insert_many(vecs[100:], labels=list(range(100, len(vecs))))
    acc = [100 + i for i, ok in enumerate(out) if ok]
    E._labels[n0:] = ['L%d' % i for i in acc]
    t = combo([vecs[i] for i in acc[:3]])
    cd = E.coordinates(t)
    r = {}
    for l, c in cd.items():
        v = vecs[int(l[1:])] if isinstance(l, str) else vecs[l]
        for k, x in v.items():
            r[k] = r.get(k, 0) + c * x
    ok_relabel = ok_relabel and {k: x for k, x in r.items() if x} == t
    # export / import with sparse and dense blocks
    E = KeyedSpan(QQ, key_order=order)
    E.insert_many(vecs, labels=list(range(len(vecs))))
    E.insert_many([rvec(random.sample(range(2000, 3000), 3), 1) for _ in range(5)],
                  labels=list(range(len(vecs), len(vecs) + 5)))
    F = KeyedSpan(QQ, key_order=order)
    F._import(E._export())
    ok_bexport = (ok_bexport and F.labels() == E.labels() and len(F._blocks) == len(E._blocks)
                  and F._dense == E._dense and all(
                      F.coordinates(x) == E.coordinates(x) and F.reduce(x) == E.reduce(x)
                      for x in tests))
    # a state saved before the blocks (one _Block with user labels) loads
    Old = _Block(QQ, key_order=order)
    for l, v in enumerate(vecs[:120]):
        Old.insert(v, 'u%d' % l)
    F = KeyedSpan(QQ, key_order=order)
    F._import(Old._export())
    ok_old = (ok_old and F.labels() == Old.labels() and all(
        F.coordinates(x) == Old.coordinates(x) for x in tests)
        and (order is None or F.rows() == Old.rows()))

report('blocks: dense per block, same labels, flags, pivots, rows, coordinates', ok_blocks)
report('blocks merge (sparse+sparse, dense+sparse batch, dense+dense vector)', ok_merge)
report('labels relabelled in place after a batch (as vertex_ideal)', ok_relabel)
report('export/import with several blocks', ok_bexport)
report('import of a single-echelon state (files saved before blocks)', ok_old)

# left_kernel: relations of sparse and dense batches (automatic, forced dense,
# forced sparse) are a basis of Sage's left kernel
from vertex_algebras._keyed_spans import left_kernel
KFILL = _keyed_spans.KERNEL_DENSE_FILL
ok_kernel = True
for trial in range(8):
    keys = list(range(30 + 10 * trial))
    base = [rvec(keys, 0.1 if trial % 2 else 0.6) for _ in range(20 + 8 * trial)]
    vs = base + [combo(random.sample(base, 3)) for _ in range(trial + 2)] + [{}]
    random.shuffle(vs)
    M = matrix(QQ, len(vs), len(keys), {(i, k): c for i, v in enumerate(vs)
                                        for k, c in v.items()})
    K = M.left_kernel()
    for fill in (KFILL, 0, 2):
        _keyed_spans.KERNEL_DENSE_FILL = fill
        rel = left_kernel(vs, QQ)
        R = matrix(QQ, len(rel), len(vs), {(a, j): c for a, r in enumerate(rel)
                                          for j, c in r.items()})
        ok_kernel = (ok_kernel and R.rank() == len(rel) == K.dimension()
                     and (R * M).is_zero()
                     and all(r[max(r)] == 1 for r in rel))
    _keyed_spans.KERNEL_DENSE_FILL = KFILL
report('left_kernel = Sage left kernel (automatic, forced dense, forced sparse)',
       ok_kernel)

# left_kernel on block-diagonal images: per component, the same relations
# as one elimination of everything (the canonical basis)
from vertex_algebras._keyed_spans import _left_kernel_block
ok_kblocks = True
for trial in range(3):
    G = 10
    groups = [list(range(g * 30, (g + 1) * 30)) for g in range(G)]
    base = [rvec(groups[g], 0.6) for g in range(G) for _ in range(20)]
    vs = base + [combo(random.sample(base, 2)) for _ in range(8)] + [{}, {}]
    vs += [combo([base[20 * g + i] for i in range(3)]) for g in range(G)]
    random.shuffle(vs)
    for fill in (KFILL, 0, 2):
        _keyed_spans.KERNEL_DENSE_FILL = fill
        _keyed_spans.KERNEL_DENSE_FILL = 2
        ref = _left_kernel_block(vs, QQ)            # one block, all sparse
        _keyed_spans.KERNEL_DENSE_FILL = fill
        ok_kblocks = ok_kblocks and left_kernel(vs, QQ) == ref
    _keyed_spans.KERNEL_DENSE_FILL = KFILL
report('left_kernel by components = one elimination (exact relations)', ok_kblocks)

report.result()
