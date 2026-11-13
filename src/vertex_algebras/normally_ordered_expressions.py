# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Normally ordered expressions in named fields.

The OPEs of the strong generators of a universal vertex algebra have
coefficients that are normally ordered products of the generators and their
derivatives, i.e. elements of the algebra being defined.  This module
provides a language to write them down *before* that algebra exists: the
tensor algebra ``T(R)`` of the free ``C[∂]``-module ``R`` on the generators
([DSK2005]_).

An element is a linear combination of *words* of letters ``∂^n a``; the
word ``a b c`` stands for the right-nested normally ordered product
``:a :b c::`` (Thielemans' convention [Thi1991]_), in any order of the letters.  The
product ``*`` is the concatenation of words and ``∂`` acts by the Leibniz
rule.  Evaluating an expression in a vertex algebra (see
:class:`universal_vertex_algebra.UniversalVertexAlgebra`) gives the
right-nested normally ordered products there.

.. WARNING::

    The concatenation is associative, the normally ordered product of a
    vertex algebra is not: ``(a*b)*c`` is the word ``a b c``, i.e.
    ``:a :b c::``, not ``:(:a b:) c:``.  Products whose left factor is a
    single letter (``a*w``) agree with the vertex algebra product.

EXAMPLES::

    sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
    sage: E.<L, W> = NormallyOrderedExpressions(QQ)
    sage: E
    Normally ordered expressions in the fields ['L', 'W'] over Rational Field
    sage: Lam = L*L - 3/10*L.derivative(2)
    sage: Lam
    :L L: - 3/10*d^2L
    sage: Lam.derivative()
    :L dL: + :dL L: - 3/10*d^3L
    sage: W*L - L*W
    -:L W: + :W L:
    sage: (L*W)*L == L*(W*L)
    True
    sage: E.one(), L + 2
    (1, 2 + L)

Words are tuples of letters ``(generator index, derivative order)``::

    sage: list(Lam.derivative().monomial_coefficients().items())
    [(((0, 0), (0, 1)), 1), (((0, 1), (0, 0)), 1), (((0, 3),), -3/10)]
    sage: E((('L', 1), ('W', 0)))
    :dL W:

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

from math import comb as _comb

from sage.categories.algebras_with_basis import AlgebrasWithBasis
from sage.combinat.free_module import CombinatorialFreeModule
from sage.modules.with_basis.indexed_element import IndexedFreeModuleElement
from sage.rings.infinity import Infinity
from sage.rings.rational_field import QQ
from sage.structure.element import parent as _parent

from .vertex_algebra import _derivative_str, _format_terms, _is_int_like, _word_str


def _normalize_generators(generators, parities, names, what):
    """
    The generator tuple and the sorted parity items of the constructor
    arguments of a universal vertex algebra or of
    :class:`NormallyOrderedExpressions` (``what`` names the class in error
    messages).

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import _normalize_generators
        sage: _normalize_generators(['b', 'c'], {'c': 1}, None, 'X')
        (('b', 'c'), (('c', 1),))
        sage: _normalize_generators(None, None, ('x', 'y'), 'X')
        (('x', 'y'), ())
        sage: _normalize_generators(['a', 'a'], None, None, 'X')
        Traceback (most recent call last):
        ...
        ValueError: generators must be distinct: ('a', 'a')
    """
    if generators is None:
        if names is None:
            raise TypeError('{}(base_ring, generators, ...): a list of '
                            'generators must be given'.format(what))
        generators = names
    if isinstance(generators, str):
        generators = [g.strip() for g in generators.split(',')]
    generators = tuple(generators)
    if len(set(generators)) != len(generators):
        raise ValueError('generators must be distinct: {}'.format(generators))
    parities = dict(parities or {})
    for g in parities:
        if g not in generators:
            raise ValueError('parity given for unknown generator {!r}'.format(g))
        if parities[g] not in (0, 1):
            raise ValueError('parity must be 0 or 1, got {}'.format(parities[g]))
    return generators, tuple(sorted((g, int(p)) for g, p in parities.items()
                                    if p))


class _Words:
    r"""
    All words (tuples) of letters ``(generator index, derivative)`` on
    ``ngens`` generators: the key set of :class:`NormallyOrderedExpressions`
    (infinite, not iterable).

    TESTS::

        sage: from vertex_algebras.normally_ordered_expressions import _Words
        sage: W = _Words(2); ((0, 1), (1, 0)) in W, () in W
        (True, True)
    """
    def __init__(self, ngens):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: _Words(2)._ngens
            2
        """
        self._ngens = ngens

    def __contains__(self, x):
        """
        Whether ``x`` is a tuple of letters ``(i, d)``, ``0 <= i < ngens``,
        ``d >= 0``.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: W = _Words(2)
            sage: [w in W for w in [((1, 3),), ((2, 0),), ((0, -1),), [(0, 0)], ((0,),)]]
            [True, False, False, False, False]
        """
        return isinstance(x, tuple) and all(
            isinstance(t, tuple) and len(t) == 2
            and _is_int_like(t[0]) and _is_int_like(t[1])
            and 0 <= t[0] < self._ngens and t[1] >= 0 for t in x)

    def an_element(self):
        """
        A word: the first generator, or the empty word without generators.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: _Words(2).an_element(), _Words(0).an_element()
            (((0, 0),), ())
        """
        return ((0, 0),) if self._ngens else ()

    def cardinality(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: _Words(1).cardinality()
            +Infinity
        """
        return Infinity

    def __eq__(self, other):
        """
        Equal iff the numbers of generators agree.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: _Words(2) == _Words(2), _Words(2) == _Words(3), _Words(2) == 2
            (True, False, False)
        """
        return isinstance(other, _Words) and other._ngens == self._ngens

    def __hash__(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: hash(_Words(2)) == hash(_Words(2))
            True
        """
        return hash((_Words, self._ngens))

    def __reduce__(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import _Words
            sage: loads(dumps(_Words(2))) == _Words(2)
            True
        """
        return (_Words, (self._ngens,))


def _derivative_of_word(word, n):
    r"""
    ``∂^n`` of a word by the Leibniz rule, as a dict ``{word: integer}``
    (letters carry plain derivatives ``∂^d``, not divided by ``d!``).

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import _derivative_of_word
        sage: _derivative_of_word(((0, 0), (1, 0)), 2)
        {((0, 0), (1, 2)): 1, ((0, 1), (1, 1)): 2, ((0, 2), (1, 0)): 1}
        sage: _derivative_of_word((), 1), _derivative_of_word((), 0)
        ({}, {(): 1})
    """
    if n == 0:
        return {word: 1}
    if not word:
        return {}
    (i, d), rest = word[0], word[1:]
    out = {}
    for j in range(n + 1):
        b = _comb(n, j)
        for w, a in _derivative_of_word(rest, n - j).items():
            key = ((i, d + j),) + w
            out[key] = out.get(key, 0) + b * a
    return out


class NormallyOrderedExpression(IndexedFreeModuleElement):
    r"""
    An element of :class:`NormallyOrderedExpressions`: a linear combination
    of words, each the right-nested normally ordered product of its letters.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: E.<a, b> = NormallyOrderedExpressions(QQ)
        sage: x = 2*a*b - a.derivative(2); x
        2*:a b: - d^2a
        sage: (a*b)*a == a*(b*a), a*b == b*a
        (True, False)
    """
    def derivative(self, n=1):
        r"""
        The ``n``-th derivative (Leibniz rule on every word).

        INPUT:

        - ``n`` -- (default: 1) a non-negative integer

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<a, b> = NormallyOrderedExpressions(QQ)
            sage: (a*b).derivative()
            :a db: + :da b:
            sage: (a*b).derivative(0) == a*b
            True
            sage: a.derivative(-1)
            Traceback (most recent call last):
            ...
            ValueError: the derivative order must be non-negative
        """
        n = int(n)
        if n < 0:
            raise ValueError('the derivative order must be non-negative')
        P = self.parent()
        out = {}
        for w, c in self._monomial_coefficients.items():
            for u, k in _derivative_of_word(w, n).items():
                out[u] = out.get(u, 0) + c * k
        return P._from_dict({u: P.base_ring()(c) for u, c in out.items()},
                            remove_zeros=True)

    def no(self, right):
        r"""
        The word concatenation ``self * right``: the normally ordered
        product when ``self`` is a single letter (see the module warning).

        INPUT:

        - ``right`` -- an element of the same parent

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<a, b> = NormallyOrderedExpressions(QQ)
            sage: a.no(b.derivative())
            :a db:
        """
        return self * right

    def parity(self):
        r"""
        The parity (0 even, 1 odd) of a homogeneous expression.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<b, c> = NormallyOrderedExpressions(QQ, parities={'c': 1})
            sage: (b*c).parity(), (c*c).parity(), E.zero().parity()
            (1, 0, 0)
            sage: (b + c).parity()
            Traceback (most recent call last):
            ...
            ValueError: element is not homogeneous
        """
        P = self.parent()
        pars = {P._parity_key(w) for w in self._monomial_coefficients}
        if not pars:
            return 0
        if len(pars) > 1:
            raise ValueError('element is not homogeneous')
        return pars.pop()

    def _styled(self, style):
        r"""
        The element printed in ``style`` (``'ascii'``, ``'unicode'`` or
        ``'latex'``), terms sorted by word.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: x = 2*L.derivative() + 1; x._styled('ascii'), x._styled('unicode')
            ('1 + 2*dL', '1 + 2*∂ L')
        """
        P = self.parent()
        return _format_terms(P, [(P._term_str(w, style) if w else None, c)
                                 for w, c in sorted(
                                     self._monomial_coefficients.items())], style)

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: repr(L*L.derivative(2))
            ':L d^2L:'
        """
        return self._styled('ascii')

    def _ascii_art_(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: ascii_art(L*L.derivative(2))
            :L d^2L:
        """
        from sage.typeset.ascii_art import AsciiArt
        return AsciiArt([self._styled('ascii')])

    def _unicode_art_(self):
        r"""
        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: x = (L*L).derivative(); x, unicode_art(x), latex(x)
            (:L dL: + :dL L:, :L ∂ L: + :∂ L L:, :L\, \partial L: + :\partial L\, L:)
        """
        from sage.typeset.unicode_art import UnicodeArt
        return UnicodeArt([self._styled('unicode')])

    def _latex_(self):
        r"""
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E.<L> = NormallyOrderedExpressions(QQ)
            sage: latex(L.derivative(2))
            \partial^{2} L
        """
        return self._styled('latex')


class NormallyOrderedExpressions(CombinatorialFreeModule):
    r"""
    The normally ordered expressions (the tensor algebra ``T(R)``) in named
    fields with given parities, over a commutative ring.

    INPUT:

    - ``base_ring`` -- (default: ``None``) the coefficient ring (``None``:
      ``QQ``)
    - ``generators`` -- (default: ``None``) the field names (a list, or a
      comma separated string); with ``E.<a, b> = ...`` the names may be
      given that way only
    - ``parities`` -- (default: ``None``) dict ``name -> 0 or 1``; omitted
      names are even
    - ``names`` -- (default: ``None``) the names given by ``E.<a, b> =
      ...``: the field names when ``generators`` is ``None``; otherwise
      only the Python variables bound to the fields, which keep their own
      names (so a field can be named for typesetting and typed under a
      short name; Sage's ``LieAlgebra(QQ, cartan_type=...)`` binds its
      fixed generators to such variables in the same way)

    The order of the names is the generator order of the universal vertex
    algebras built from these expressions.

    EXAMPLES::

        sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
        sage: E = NormallyOrderedExpressions(QQ, ['b', 'c'], parities={'c': 1})
        sage: b, c = E.gens()
        sage: E.variable_names(), E.parities()
        (('b', 'c'), {'b': 0, 'c': 1})
        sage: E.gen('c') is c or E.gen('c') == c
        True
        sage: c*b.derivative()
        :c db:
        sage: E is NormallyOrderedExpressions(QQ, 'b, c', parities={'c': 1})
        True

    Fields named for typesetting, typed under short names (the names
    after ``E.<...>`` are only the Python variables)::

        sage: G.<b, g> = NormallyOrderedExpressions(QQ, ['β', 'γ'])
        sage: G.variable_names(), b, g.derivative()*b
        (('β', 'γ'), β, :dγ β:)

    Pickling::

        sage: from vertex_algebras import normally_ordered_expressions as noe
        sage: F = noe.NormallyOrderedExpressions(QQ, ['b', 'c'], parities={'c': 1})
        sage: loads(dumps(F)) is F
        True
        sage: x = F.gen(0) * F.gen(1).derivative(2)
        sage: loads(dumps(x)) == x
        True
        sage: TestSuite(F).run(skip=['_test_associativity',
        ....:     '_test_distributivity', '_test_prod', '_test_one',
        ....:     '_test_zero_divisors', '_test_some_elements'])
    """
    Element = NormallyOrderedExpression
    _is_normally_ordered_expressions = True    # duck typing (doctests)

    @staticmethod
    def __classcall_private__(cls, base_ring=None, generators=None,
                              parities=None, names=None):
        """
        Normalize the arguments (unique representation).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: NormallyOrderedExpressions(QQ, ['a']) is NormallyOrderedExpressions(QQ, ('a',), {'a': 0})
            True
        """
        if base_ring is None:
            base_ring = QQ
        generators, parity_items = _normalize_generators(
            generators, parities, names, 'NormallyOrderedExpressions')
        return CombinatorialFreeModule.__classcall__(
            cls, base_ring, generators, parity_items)

    def __init__(self, base_ring, generators, parity_items):
        """
        Initialize ``self``.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['b', 'c'], parities={'c': 1})
            sage: E._gens, E._idx, E._parities
            (('b', 'c'), {'b': 0, 'c': 1}, {1: 1})
        """
        self._gens = tuple(generators)
        self._idx = {name: k for k, name in enumerate(self._gens)}
        self._parities = {self._idx[g]: p for g, p in parity_items}
        CombinatorialFreeModule.__init__(
            self, base_ring, _Words(len(self._gens)), prefix='',
            category=AlgebrasWithBasis(base_ring))

    def _repr_(self):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: NormallyOrderedExpressions(QQ, ['L', 'W'])
            Normally ordered expressions in the fields ['L', 'W'] over Rational Field
        """
        return 'Normally ordered expressions in the fields {} over {}'.format(
            list(self._gens), self.base_ring())

    # the algebra structure: concatenation of words

    def one_basis(self):
        """
        The empty word (the vacuum).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a'])
            sage: E.one_basis(), E.one()
            ((), 1)
        """
        return ()

    def product_on_basis(self, u, v):
        """
        Concatenation of words.

        INPUT:

        - ``u``, ``v`` -- words: tuples of letters ``(field index, derivative)``

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
            sage: E.product_on_basis(((1, 0),), ((0, 2),))
            :b d^2a:
        """
        return self.monomial(u + v)

    # generators

    def variable_names(self):
        """
        The field names (the generator order), a tuple.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: NormallyOrderedExpressions(QQ, ['L', 'W']).variable_names()
            ('L', 'W')
        """
        return tuple(self._gens)

    def parities(self):
        """
        A dict mapping the field names to their parities (0 even, 1 odd).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: NormallyOrderedExpressions(QQ, ['L', 'G'], parities={'G': 1}).parities()
            {'G': 1, 'L': 0}
        """
        return {g: self._parities.get(k, 0) for k, g in enumerate(self._gens)}

    def gen(self, name_or_index):
        """
        The field with the given name or index.

        INPUT:

        - ``name_or_index`` -- a field name or its index (0-based)

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
            sage: E.gen('b'), E.gen(0), E.gen(1)
            (b, a, b)
        """
        if _is_int_like(name_or_index):
            k = int(name_or_index)
        else:
            k = self._idx[name_or_index]
        return self.monomial(((k, 0),))

    def gens(self):
        """
        The fields, as a tuple.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: NormallyOrderedExpressions(QQ, ['L', 'W']).gens()
            (L, W)
        """
        return tuple(self.gen(k) for k in range(len(self._gens)))

    def algebra_generators(self):
        """
        The fields, as a tuple (with their derivatives they generate).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['L', 'W'])
            sage: E.algebra_generators() == E.gens()
            True
        """
        return self.gens()

    def _first_ngens(self, n=0):
        """The first ``n`` fields, enabling ``E.<a, b> = ...``.

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
            sage: E._first_ngens(2), E._first_ngens(1)
            ((a, b), (a,))
            sage: E._first_ngens(3)
            Traceback (most recent call last):
            ...
            ValueError: requested 3 generators but only 2 are defined
        """
        if n > len(self._gens):
            raise ValueError('requested {} generators but only {} are defined'
                             .format(n, len(self._gens)))
        return self.gens()[:n]

    # helpers

    def _parity_key(self, w):
        """
        The parity of the word ``w`` (the sum of its letters' parities).

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'], {'b': 1})
            sage: E._parity_key(((0, 1), (1, 0))), E._parity_key(((1, 0), (1, 2))), E._parity_key(())
            (1, 0, 0)
        """
        return sum(self._parities.get(i, 0) for i, _ in w) % 2

    def _term_str(self, w, style='ascii'):
        r"""
        The word ``w`` printed in ``style`` (``'ascii'``, ``'unicode'``,
        ``'latex'``).

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'], {'b': 1})
            sage: w = ((0, 1), (1, 0))
            sage: E._term_str(w), E._term_str(w, 'unicode'), E._term_str(((1, 2),), 'latex')
            (':da b:', ':∂ a b:', '\\partial^{2} b')
        """
        return _word_str([_derivative_str(self._gens[i], n, style) for i, n in w], style)

    def _repr_term(self, w):
        """
        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'], {'b': 1})
            sage: E._repr_term(((0, 2), (1, 0)))
            ':d^2a b:'
        """
        return self._term_str(w, 'ascii')

    def _rc(self, c):
        """Coerce a scalar into the base ring (no conversion if it is
        already there).

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'], {'b': 1})
            sage: c = E._rc(1/2); c, c.parent()
            (1/2, Rational Field)
            sage: E._rc(3).parent()
            Rational Field
        """
        R = self.base_ring()
        return c if _parent(c) is R else R(c)

    def _word(self, x):
        """
        A word from a tuple of letters given by names or indices, e.g.
        ``(('a', 1), ('b', 0))`` or ``('a', 1)`` (one letter).

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
            sage: E._word((('a', 1), ('b', 0))), E._word(('b', 2)), E._word(())
            (((0, 1), (1, 0)), ((1, 2),), ())
            sage: E._word((('z', 0),))
            Traceback (most recent call last):
            ...
            TypeError: bad letter ('z', 0)
        """
        def letter(t):
            if (isinstance(t, tuple) and len(t) == 2 and _is_int_like(t[1])
                    and int(t[1]) >= 0):
                g = t[0]
                if _is_int_like(g) and 0 <= int(g) < len(self._gens):
                    return (int(g), int(t[1]))
                if isinstance(g, str) and g in self._idx:
                    return (self._idx[g], int(t[1]))
            raise TypeError('bad letter {!r}'.format(t))
        if not isinstance(x, tuple):
            raise TypeError('cannot convert {!r} to a word'.format(x))
        if (len(x) == 2 and _is_int_like(x[1])
                and (isinstance(x[0], str) or _is_int_like(x[0]))):
            return (letter(x),)
        return tuple(letter(t) for t in x)

    def _element_constructor_(self, x):
        """
        Convert an expression, a field name, a word (tuple of letters given
        by names or indices) or a scalar.

        EXAMPLES::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'])
            sage: E('b'), E((('a', 0), ('b', 1))), E(3), E(E.gen(0))
            (b, :a db:, 3, a)
        """
        if isinstance(x, NormallyOrderedExpression) and x.parent() is self:
            return x
        if isinstance(x, str) and x in self._idx:
            return self.gen(x)
        if isinstance(x, tuple):
            return self.monomial(self._word(x))
        R = self.base_ring()
        if _parent(x) is R or R.has_coerce_map_from(_parent(x)):
            return self.term((), R(x))
        return super()._element_constructor_(x)

    def _coerce_map_from_(self, S):
        """Scalars coerce in (as multiples of the empty word).

        TESTS::

            sage: from vertex_algebras.normally_ordered_expressions import NormallyOrderedExpressions
            sage: E = NormallyOrderedExpressions(QQ, ['a', 'b'], {'b': 1})
            sage: E._coerce_map_from_(ZZ), E._coerce_map_from_(QQ['x'])
            (True, False)
            sage: E.gen(0) + 1/2
            1/2 + a
        """
        return self.base_ring().has_coerce_map_from(S)
