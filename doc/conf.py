# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
"""
Sphinx configuration of the reference manual.

Build with ``sphinx-build -b html doc doc/_build/html`` in an environment
where the package is installed (``import vertex_algebras`` works).  The
docstrings follow Sage's conventions; the settings below reproduce the
parts of Sage's documentation setup they rely on (``:arxiv:`` and
``:doi:`` links, single backquotes as mathematics, MathJax).
"""
import importlib.util
import inspect

import vertex_algebras

project = 'vertex_algebras'
author = 'Christopher Raymond, Simon Wood'
copyright = '2026, Cardiff University and University of Hamburg'
release = vertex_algebras.__version__
version = release

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

# As in Sage: `x` is mathematics.
default_role = 'math'

extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:%s'),
    'doi': ('https://doi.org/%s', 'doi:%s'),
}

# Sage's classes and functions: the parts of Sage's reference manual the
# docstrings link to (each part has its own inventory).
_SAGE_DOC = 'https://doc.sagemath.org/html/en/reference/'
intersphinx_mapping = {
    'sage_' + part: (_SAGE_DOC + part + '/', None)
    for part in ['combinat', 'parallel']
}

# Every cross-reference must resolve (with ``_resolve_relative`` below).
nitpicky = True

# Doctests (``sage: ...``) are literal blocks: no Python lexer.
highlight_language = 'text'

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'none'
add_module_names = False

exclude_patterns = ['_build']

# Sage's theme when it is installed (optional: only for building these
# pages), else Sphinx's default.
if importlib.util.find_spec('furo') is not None:
    html_theme = 'furo'
else:
    html_theme = 'alabaster'
html_title = 'Vertex algebras for SageMath'


def _document_nested_classes():
    # Sage renames nested classes for pickling: ``__name__`` of the
    # category class ``ProtoVertexAlgebras.Graded`` is
    # ``'ProtoVertexAlgebras.Graded'``.  Autodoc compares it with the
    # attribute name, documents the class as "alias of Graded" and skips
    # its members (``ParentMethods``, ``ElementMethods``: the category
    # methods such as ``weight_space``).  Compare with the last part of
    # the dotted name instead, as Sage's own documentation builder does.
    try:
        from sphinx.ext.autodoc._property_types import _ClassDefProperties
    except ImportError:         # Sphinx < 9
        from sphinx.ext.autodoc import ClassDocumenter
        import_object = ClassDocumenter.import_object

        def patched(self, raiseerror=False):
            ret = import_object(self, raiseerror)
            name = getattr(self.object, '__name__', None)
            if ret and isinstance(name, str):
                self.doc_as_attr = self.objpath[-1] != name.rsplit('.', 1)[-1]
            return ret
        ClassDocumenter.import_object = patched
        return

    def doc_as_attr(self):
        name = self._obj___name__
        if name is None:
            return True
        return self.parts[-1] != name.rsplit('.', 1)[-1]
    _ClassDefProperties.doc_as_attr = property(doc_as_attr)


_document_nested_classes()


def _skip_pickling_aliases(app, what, name, obj, skip, options):
    # Sage's nested classes (categories' ParentMethods, WithBasis, ...) are
    # also registered in their module under dotted names, for pickling;
    # they are documented once, nested in their class.
    if '.' in name:
        return True
    return None


def _public_constructor_signature(app, what, name, obj, options,
                                  signature, return_annotation):
    # Classes with ``__classcall_private__`` (unique representation) take
    # their public arguments there; ``__init__`` gets normalised ones.
    if what != 'class' or '__classcall_private__' not in vars(obj):
        return None
    f = vars(obj)['__classcall_private__']
    f = getattr(f, '__func__', f)
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return None
    # drop ``cls`` and private arguments (saved states and the like)
    params = [q for q in list(sig.parameters.values())[1:]
              if not q.name.startswith('_')]
    return str(sig.replace(parameters=params)), return_annotation


def _resolve_relative(app, env, node, contnode):
    # The docstrings name targets as Sage's own manual does: relative to
    # the package (``universal_vertex_algebra.UniversalVertexAlgebra``),
    # or by a final part of the name (``VertexAlgebra.ideal``, ``gens``),
    # which Sage's builder resolves and plain Sphinx does not.  Resolve
    # them to the unique documented object (preferring the current class, the
    # classes it is nested in, then the current module); otherwise leave the reference unresolved
    # (``nitpicky`` then warns).
    if node.get('refdomain') != 'py':
        return None
    target = node['reftarget']
    domain = env.get_domain('py')
    objects = domain.objects
    full = None
    if 'vertex_algebras.' + target in objects:
        full = 'vertex_algebras.' + target
    else:
        matches = [name for name in objects
                   if name.endswith('.' + target)
                   and name.startswith('vertex_algebras.')]
        module = node.get('py:module')
        cls = node.get('py:class')
        scopes = []
        if module and cls:
            parts = cls.split('.')
            scopes = [module + '.' + '.'.join(parts[:k])
                      for k in range(len(parts), 0, -1)]
        if module:
            scopes.append(module)
        for scope in scopes + ['']:
            inner = [m for m in matches if m.startswith(scope)]
            if len(inner) == 1:
                full = inner[0]
                break
            if inner:
                break
    if full is None:
        return None
    return domain.resolve_xref(env, node['refdoc'], app.builder, 'obj',
                               full, node, contnode)


def setup(app):
    app.connect('missing-reference', _resolve_relative)
    app.connect('autodoc-skip-member', _skip_pickling_aliases)
    app.connect('autodoc-process-signature', _public_constructor_signature)
