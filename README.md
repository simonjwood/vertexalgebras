# vertex_algebras

Vertex algebras and vertex superalgebras for [SageMath](https://www.sagemath.org).

**Status:** preview release (version 0.1.0a1). The interface may still
change.

Authors: Christopher Raymond and Simon Wood. Copyright (C) Cardiff University 2026.
Licence: GNU General Public License, version 2 or (at your option) any later
version (GPL-2.0-or-later, see
[LICENSE](https://github.com/simonjwood/vertexalgebras/blob/main/LICENSE)).

## What it does

- **Universal vertex algebras** from strong generators and their OPEs
  (`UniversalVertexAlgebra`), with coefficients written as normally
  ordered expressions (`NormallyOrderedExpressions`); Jacobi identity
  checks; conformal vectors, conformal weights and gradings.
- **Basic classical Lie superalgebras** from a Cartan matrix with odd simple
  roots or by name (sl(m|n), psl(m|m), osp(m|n), D(2,1;a), F(4), G(3)), with
  the invariant form normalised as in Kac–Wakimoto (`LieSuperalgebra`).
- **Universal affine vertex (super)algebras** of a Cartan type, a Lie
  superalgebra, a Sage Lie algebra or structure constants, with the
  Sugawara vector on request (`UniversalAffineVertexAlgebra`).
- **Ghost systems and free fields:** βγ and bc systems of any rank and free
  superfermions of a Gram matrix, with their usual conformal vectors
  (`BosonicGhosts`, `FermionicGhosts`, `FreeSuperFermions`).
- **Lattice vertex algebras** of integral, possibly indefinite, forms
  (`LatticeVertexAlgebra`), and **tensor products** of any of these
  (`V.tensor(W)`).
- **Constructions:** subalgebras with their strong generators
  (`V.subalgebra`), ideals and quotients (`V.ideal`, `V.quotient`, `V / I`),
  commutants and cosets (`V.commutant`), weight spaces and characters.
- **W-algebras** by quantum Hamiltonian reduction (Kac–Roan–Wakimoto):
  `WAlgebra(R, g, f, k)` for a Lie superalgebra, a nilpotent (`'minimal'`,
  `'principal'`, a Jordan type for the classical types, e.g. `[3, 1]` for
  `sl_4` or `([2, 1], [1])` for `sl(3|1)`, roots or an element) and a
  level, as a universal vertex algebra (its conformal vector with
  `conformal_vector='reduction'`); the reduction of highest weight
  modules (`M.quantum_hamiltonian_reduction(f)`: Verma modules and their
  simple quotients, by the theorems of Kac–Wakimoto and Arakawa).
- **A catalog** with the names of Sage's Lie conformal algebras:
  `vertex_algebras.Virasoro(R, c)`, `NeveuSchwarz`, `N2`, `Affine`,
  `BosonicGhosts`, `FermionicGhosts`, `FreeBosons`, `FreeFermions`, `Weyl`,
  `Abelian`, and `Lattice`, `W3`, `W`.
- **OPE bootstrap:** OPE ansätze with unknown coefficients, solved through
  the Jacobi identities (`OPEBootstrap`).
- **Zhu algebras** of universal vertex algebras and of their quotients
  (`V.zhu_algebra()`, `Q.zhu_algebra_up_to(W)`), also H-twisted (Ramond
  sectors: `V.zhu_algebra(twist='ramond')`).

Products are computed by C++ engines with [FLINT](https://flintlib.org)
coefficients over the rationals, finite fields, number fields, polynomial
rings and their fraction fields, optionally in several threads; other
coefficient rings use pure Python. Every algebra is a Sage parent
in a category of vertex algebras, and `TestSuite` checks the vertex algebra
axioms.

## Requirements

- SageMath 10.9 or later (Python 3.12 to 3.14).
- To build: a C++17 compiler, pkg-config, and the FLINT (3.0 or later) and
  GMP headers. A conda environment with Sage has all of them; pip fetches
  the other build tools (Cython, meson, meson-python, ninja) itself.

Linux and macOS are supported. On Windows, use Sage in the Windows
Subsystem for Linux (WSL), as Sage itself recommends.

## Installation

The package is installed with pip into the Python of an existing Sage
installation. Sage is not a declared dependency, because no single pip
package name covers all the ways Sage is installed; importing
`vertex_algebras` checks the Sage version.

### Sage from conda (recommended)

A conda environment with Sage already has the compilers, FLINT and GMP. If
you do not have one yet, install
[Miniforge](https://github.com/conda-forge/miniforge) and create it with
`conda create -n sage sage`. Then:

```
conda activate sage
pip install sage-vertex-algebras
```

pip builds the C++ engines on your machine (about a minute). On macOS, see
the next section first.

### macOS

The same steps work on Apple Silicon and on Intel Macs, with a few
additions:

1. Install the Xcode Command Line Tools once: `xcode-select --install`.
   Conda's compilers use Apple's SDK from them.
2. Install the Miniforge build for your processor (`arm64` for Apple
   Silicon, `x86_64` for Intel), then create the environment with
   `conda create -n sage sage`. Sage 10.10 or later is faster here than
   10.9 for computations over number fields; the package works with
   both.
3. Install as above. If the build reports that it cannot find a C++
   compiler, run `conda install cxx-compiler` in the environment and
   install again.
4. Check that FLINT and GMP come from the conda environment, not from
   Homebrew or elsewhere:

   ```
   otool -L "$(python -c 'import vertex_algebras._universal_engine as m; print(m.__file__)')"
   ```

   The `libflint` and `libgmp` lines should point into the environment
   (`@rpath/...` or a path under `miniforge3/envs/sage`).

When reporting a problem, include the output of `sage --version`,
`uname -m`, and the last 40 lines of the failing build or test run.

### Other Sage installations

These routes have not been tested yet.

- **Sage built from source** has the build tools, FLINT and GMP:
  `sage -pip install sage-vertex-algebras`.
- **Sage from a Linux distribution** additionally needs the FLINT and GMP
  development headers (e.g. `flint-devel`) and a C++17 compiler from the
  distribution.

### Checking the installation

```
sage -c "import vertex_algebras; print(vertex_algebras.__version__)"
python -m sage.doctest --long "$(python -c 'import vertex_algebras, os; print(os.path.dirname(vertex_algebras.__file__))')"
```

The second command runs all the examples and tests of the documentation
(a minute or two; add `-p 4` to use four processes).

### From the repository (for development)

```
git clone https://github.com/simonjwood/vertexalgebras.git
cd vertexalgebras
conda activate sage
conda install --freeze-installed meson meson-python ninja pkg-config   # once per environment
pip install --no-build-isolation -e .
```

`-e` makes an editable install: after a `git pull`, changed Python files
take effect at once, and changed C++ or Cython files are recompiled at the
next import. `--no-build-isolation` builds with the environment's own
tools and libraries. `tests/run_all.sh` then runs the whole test suite
(about ten minutes).

## A first example

The Virasoro algebra at central charge −22/5, its singular vector of
weight 4, and the quotient by it, the Lee–Yang minimal model:

```
sage: from vertex_algebras import UniversalVertexAlgebra, NormallyOrderedExpressions
sage: E.<L> = NormallyOrderedExpressions(QQ)
sage: Vir = UniversalVertexAlgebra(QQ, E, {('L', 'L'): [L.derivative(), 2*L, 0, -11/5]},
....:                              conformal_vector=L)
sage: L = Vir.gen('L')
sage: L.ope(L)                      # entry k: the coefficient of (z-w)^(-k-1)
[dL, 2*L, 0, -11/5]
sage: v = L.no(L) - 3/10*L.derivative(2)
sage: [L.mode(v, n) for n in (1, 2)]          # L_1 v = L_2 v = 0
[0, 0]
sage: Q = Vir.quotient([v])
sage: Q.character(12)                         # Rogers–Ramanujan
1 + q^2 + q^3 + q^4 + q^5 + 2*q^6 + 2*q^7 + 3*q^8 + 3*q^9 + 4*q^10 + 4*q^11 + O(q^12)
sage: Vir.zhu_algebra()(v)                    # roots 0, -1/5: the two modules
1/5*[L] + [L]^2
sage: Q.zhu_algebra_up_to(6).dimension()
2
```

## Documentation

- The docstrings, with examples: e.g. `UniversalVertexAlgebra?` in Sage.
- The reference manual, built from them: `sphinx-build -b html doc
  doc/_build/html` (with the package installed; `doc/requirements.txt`
  lists what the build needs), then open `doc/_build/html/index.html`.
- The notebooks in
  [`examples/`](https://github.com/simonjwood/vertexalgebras/tree/main/examples):
  lattice vertex algebras,
  bootstraps, singular vectors in Virasoro and affine vertex algebras,
  commutants and cosets, the triplet algebra, Zhu algebras, Lie
  superalgebras, ghost systems.

Computations run serially by default. `VertexAlgebraParallelism().set(n)`
(from `vertex_algebras`) lets the C++ engines use `n` threads for the
whole session; results do not depend on it.

## Development

- `tests/run_all.sh` runs the doctests and the test scripts;
  `tests/run_all.sh --fallback` runs the doctests in pure Python, and
  `VERTEX_ALGEBRAS_NO_ENGINE=1 tests/run_all.sh` runs everything without
  the C++ engines.

## Acknowledgements

The library started as a port of K. Thielemans' Mathematica package
OPEdefs (K. Thielemans, *A Mathematica package for computing operator
product expansions*, Int. J. Mod. Phys. C 2 (1991) 787); K. Thielemans
kindly agreed to its publication under an open-source licence. Universal
vertex algebras and their Zhu algebras follow A. De Sole and V. G. Kac, *Freely
generated vertex algebras and non-linear Lie conformal algebras*
(arXiv:math-ph/0312042) and *Finite vs affine W-algebras*
(arXiv:math-ph/0511055).
