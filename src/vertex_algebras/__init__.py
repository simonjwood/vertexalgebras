# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
r"""
Vertex algebras for SageMath.

The constructions users call directly are imported here; everything else
is reached through them (``V.subalgebra``, ``V.ideal``, ``V.quotient``,
``V.commutant``, ``V.tensor``, ``V.zhu_algebra``, ...).  The modules of the
package hold the classes and the internals (names with a leading
underscore, e.g. the modules ``_keyed_spans``, ``_grading_groups``
and the compiled engines, are not part of the public interface).

EXAMPLES::

    sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
    sage: E.<L> = NormallyOrderedExpressions(QQ)
    sage: V = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, 1/4]},
    ....:                            conformal_vector=L)
    sage: L = V.gen('L'); L.ope(L)
    [dL, 2*L, 0, 1/4]

The package needs SageMath (version 10.9 or later), checked when it is
imported: Sage is not a declared pip dependency, since no pip name covers
every Sage installation.

TESTS:

The version check, and the version of the package::

    sage: import vertex_algebras
    sage: vertex_algebras._sage_version_ok('10.9'), vertex_algebras._sage_version_ok('10.10.beta3')
    (True, True)
    sage: vertex_algebras._sage_version_ok('10.8')
    False
    sage: vertex_algebras.__version__
    '0.1.0a4'

AUTHORS:

- Christopher Raymond, Simon Wood (2026): initial version
"""

import re as _re

#: The oldest SageMath release supported.
_MIN_SAGE = (10, 9)


def _sage_version_ok(version):
    r"""
    Whether the SageMath version string ``version`` is supported: its first
    two numbers (major, minor) are at least ``_MIN_SAGE``.

    TESTS::

        sage: from vertex_algebras import _sage_version_ok
        sage: [_sage_version_ok(v) for v in ['10.9', '10.9.rc0', '11.0', '10.8.beta2', '9.9']]
        [True, True, True, False, False]
    """
    return tuple(int(x) for x in _re.findall(r'\d+', version)[:2]) >= _MIN_SAGE


try:
    import sage.all  # noqa: F401  (Sage first: the modules import parts of it)
    from sage.version import version as _sage_version
except ImportError as _err:
    raise ImportError(
        'vertex_algebras needs SageMath {}.{} or later: install this package '
        'into the Python of a Sage installation (conda, built from source, or '
        'a distribution package)'.format(*_MIN_SAGE)) from _err
if not _sage_version_ok(_sage_version):
    raise ImportError('vertex_algebras needs SageMath {}.{} or later, found {}'
                      .format(*_MIN_SAGE, _sage_version))

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version('sage-vertex-algebras')
except PackageNotFoundError:          # not installed (sources on sys.path)
    __version__ = 'unknown'

from .affine_vertex_algebra import UniversalAffineVertexAlgebra
from .bootstrap import OPEBootstrap

# The catalog (Sage's lie_conformal_algebras names), reached as
# vertex_algebras.Virasoro etc.; not in __all__: `import *` would bind
# short names such as W.
from .catalog import (  # noqa: F401  (re-exported, not in __all__)
    N2,
    W3,
    Abelian,
    Affine,
    BosonicGhosts,
    FermionicGhosts,
    FreeBosons,
    FreeFermions,
    Lattice,
    NeveuSchwarz,
    Virasoro,
    W,
    Weyl,
)
from .ghost_systems import FreeSuperFermions
from .lattice_vertex_algebra import LatticeVertexAlgebra
from .lie_conformal import universal_enveloping_vertex_algebra
from .lie_superalgebras import LieSuperalgebra, LieSuperalgebras
from .normally_ordered_expressions import NormallyOrderedExpressions
from .quantum_hamiltonian_reduction import WAlgebra
from .tensor_product_vertex_algebra import TensorProductVertexAlgebra
from .universal_vertex_algebra import UniversalVertexAlgebra
from .vertex_algebra import LogarithmicWarning, VertexAlgebraParallelism
from .vertex_algebra_categories import (
    ProtoVertexAlgebras,
    VertexAlgebraModules,
    VertexAlgebras,
    VertexOperatorAlgebras,
)
from .vertex_subalgebra import StrongGenerationError

__all__ = ['UniversalVertexAlgebra', 'NormallyOrderedExpressions',
           'UniversalAffineVertexAlgebra', 'LieSuperalgebra', 'LieSuperalgebras',
           'FreeSuperFermions',
           'LatticeVertexAlgebra',
           'TensorProductVertexAlgebra', 'OPEBootstrap', 'WAlgebra',
           'universal_enveloping_vertex_algebra', 'VertexAlgebraParallelism',
           'LogarithmicWarning', 'StrongGenerationError',
           'ProtoVertexAlgebras', 'VertexAlgebras', 'VertexOperatorAlgebras',
           'VertexAlgebraModules']
