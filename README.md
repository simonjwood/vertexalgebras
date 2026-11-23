# vertex_algebras

Vertex algebras and vertex superalgebras for [SageMath](https://www.sagemath.org).

**Status:** preview release (version 0.1.0a4). The interface may still
change.

## What it does

- **Universal vertex algebras** from strong generators and their OPEs
  (`UniversalVertexAlgebra`), with coefficients written as normally
  ordered expressions (`NormallyOrderedExpressions`); Jacobi identity
  checks; conformal vectors, conformal weights and gradings.
- **A catalog** with the names of Sage's Lie conformal algebras:
  `vertex_algebras.Virasoro(R, c)`, `NeveuSchwarz`, `N2`, `Affine`,
  `BosonicGhosts`, `FermionicGhosts`, `FreeBosons`, `FreeFermions`, `Weyl`,
  `Abelian`, and `Lattice`, `W3`, `W`; both directions to Sage's Lie
  conformal algebras (`V.lie_conformal_algebra()`,
  `universal_enveloping_vertex_algebra`).
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
  presentations by generators and relations (`S.presentation()`),
  commutants and cosets (`V.commutant`); weight spaces, characters
  (`V.character`) and singular vectors.
- **Morphisms** given by images of generators (`V.hom(images, W)`), with
  composition, kernels and images.
- **W-algebras** by quantum Hamiltonian reduction (Kac–Roan–Wakimoto):
  `WAlgebra(R, g, f, k)` for a Lie superalgebra, a nilpotent (`'minimal'`,
  `'principal'`, a Jordan type for the classical types, e.g. `[3, 1]` for
  `sl_4` or `([2, 1], [1])` for `sl(3|1)`, roots or an element) and a
  level, as a universal vertex algebra (its conformal vector with
  `conformal_vector='reduction'`); the reduction of highest weight
  modules (`M.quantum_hamiltonian_reduction(f)`: Verma modules and their
  simple quotients, by the theorems of Kac–Wakimoto and Arakawa).
- **Zhu algebras** of universal vertex algebras and of their quotients
  (`V.zhu_algebra()`, `Q.zhu_algebra_up_to(W)`), also H-twisted (Ramond
  sectors: `V.zhu_algebra(twist='ramond')`), and the **higher Zhu
  algebras** of Dong–Li–Mason (`V.zhu_algebra(level=n)`); modules over
  them (`A.module(...)`, `A.verma_module(λ)`).
- **Modules:** the adjoint module, the simple modules of lattice vertex
  algebras (`V.simple_module(λ)`, `V.simple_modules()`), modules induced
  from modules over the Zhu algebra (`V.induced_module(U)`,
  `Q.induced_module_up_to(U, ...)` for quotients); submodules, the
  maximal submodule, quotients and simple quotients
  (`M.simple_quotient()`), direct sums, singular vectors
  (`M.find_singular`); twists, pullbacks and spectral flow; module
  morphisms (`M.hom(images, N)`, `Hom(M, N).basis()`).
- **Fusion products** by the Nahm–Gaberdiel–Kausch algorithm
  (`M.fusion_product(N)`), as quotients up to a depth; for lattice vertex
  algebras the known product `V_{λ+L} ⊠ V_{μ+L} = V_{λ+μ+L}`.
- **Extensions:** projective covers, radical and socle series (Loewy
  layers) and Ext^1 up to a depth (`L.projective_cover`,
  `M.radical_series_up_to`, `M.ext1_up_to(N, ...)`).
- **OPE bootstrap:** OPE ansätze with unknown coefficients, solved through
  the Jacobi identities (`OPEBootstrap`).

Products are computed by C++ engines with [FLINT](https://flintlib.org)
coefficients over the rationals, finite fields, number fields, polynomial
rings and their fraction fields, optionally in several threads; other
coefficient rings use pure Python. Every algebra is a Sage parent
in a category of vertex algebras, and `TestSuite` checks the vertex algebra
axioms.

## Requirements

- SageMath 10.9 or later (Python 3.12 to 3.14).
- To build: a C++17 compiler, pkg-config, and the FLINT (3.0 or later) and
  GMP headers. A conda environment with Sage and `cxx-compiler` has all
  of them (see below); pip fetches the other build tools (Cython, meson,
  meson-python, ninja) itself.

Linux and macOS are supported. On Windows, use Sage in the Windows
Subsystem for Linux (WSL), as Sage itself recommends.

## Installation

The instructions below start from a computer with nothing installed, and
go through five steps for each operating system:

1. install Python and Sage, with conda;
2. install the package;
3. copy the example notebooks to a folder of your choice;
4. run the notebooks in Jupyter;
5. optional: run the test suite (15–30 minutes).

All commands are typed in a terminal. If you already have some of this
(for example a conda environment with Sage 10.9 or later), skip the steps
you have done. Other ways of installing Sage, and installing from this
repository for development, are described after the three operating
systems.

### Linux

These commands are for any Linux distribution on x86_64 or arm64
(aarch64).

**1. Python and Sage.** Install
[Miniforge](https://github.com/conda-forge/miniforge), a minimal conda
that installs from conda-forge, where Sage is packaged:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the licence and the default location (`~/miniforge3`). When the
installer asks whether to update your shell profile to initialise conda,
answer `no` (the default): your shell stays as it is, and conda is
switched on only when you ask for it, with the command below. Then create
a `sage` environment containing Sage, Python and Jupyter (this downloads
about 2 GB and takes a few minutes), and switch it on:

```bash
~/miniforge3/bin/conda create -n sage "sage>=10.9"
source ~/miniforge3/bin/activate sage
sage --version
```

The last command should print the Sage version, 10.9 or later. More
details are in Sage's guide
[Install from conda-forge](https://doc.sagemath.org/html/en/installation/conda.html).

The prompt now begins with `(sage)`: the environment is switched on. Every
later step starts with `source ~/miniforge3/bin/activate sage`, in each
new terminal. To switch it off again (the prompt loses its `(sage)`), type

```bash
conda deactivate
```

or close the terminal. Typing `sage` on its own starts Sage's interactive
prompt, `sage:`; leave it with `quit` (or Ctrl+D).

**2. The package.** In the `sage` environment, install the C++ compiler
and pkg-config from conda-forge, then the package from PyPI:

```bash
source ~/miniforge3/bin/activate sage
conda install cxx-compiler pkg-config
pip install sage-vertex-algebras
```

pip builds the C++ engines on your machine (about a minute). Check that it
worked:

```bash
sage -c "import vertex_algebras; print(vertex_algebras.__version__)"
```

**3. The example notebooks.** The package includes the example notebooks
and its test suite. Copy them to a folder of your own, where you can run
and change them:

```bash
source ~/miniforge3/bin/activate sage
python -m vertex_algebras.examples ~/vertex-algebra-notebooks
```

The folder can be any path that does not exist yet;
`~/vertex-algebra-notebooks` is only an example, and the commands below
use it. The command creates the folder, with the notebooks of the
installed version, their `README.md` (a guide to the notebooks) and the
test suite in the subfolder `tests`. It never overwrites an existing
folder: after upgrading the package, copy the new notebooks into a new
folder.

**4. Jupyter.** The `sage` environment already has Jupyter. Start it in
the notebook folder:

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
sage -n jupyter
```

Jupyter opens in your web browser with the list of notebooks (if it does
not, copy into the browser the address starting with
`http://localhost:8888/` that the terminal prints, with its `token=`
part). Click the first one, `01_getting_started.ipynb`. The kernel in the
top right corner should read "SageMath 10.x" (if Jupyter asks, choose
SageMath). Run the cells one after another with Shift+Enter. The terminal
must stay open while you work; to stop Jupyter, press Ctrl+C in it twice.

Next time, the same three commands start it again.

**5. Optional: the test suite.** Run the examples and tests of the
documentation of the installed package (a minute or two; add `-p 4` after
`--long` to use four processes):

```bash
source ~/miniforge3/bin/activate sage
python -m sage.doctest --long "$(python -c 'import vertex_algebras, os; print(os.path.dirname(vertex_algebras.__file__))')"
```

The last lines should say `All tests passed!`.

The platform report runs the whole test suite, with the C++ engines and in
pure Python, and writes one file with the results and a description of
your system: the file to send when something does not work. Run it in the
folder of step 3 (it takes 15–30 minutes):

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
bash tests/platform_report.sh
```

The report is the file `platform_report_<date>_<time>.txt` in that folder;
its last lines say `ALL PASSED` or list what failed.

### macOS

For Apple Silicon and Intel Macs. Open the Terminal app (in Applications →
Utilities).

**1. Python and Sage.** First install Apple's Command Line Tools, which
provide the system headers that the compiler of step 2 uses:

```bash
xcode-select --install
```

A window asks for confirmation; wait until the installation finishes (if
it says the tools are already installed, go on). Then install
[Miniforge](https://github.com/conda-forge/miniforge), a minimal conda
that installs from conda-forge, where Sage is packaged; the command picks
the build for your processor:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the licence and the default location (`~/miniforge3`). When the
installer asks whether to update your shell profile to initialise conda,
answer `no` (the default): your shell stays as it is, and conda is
switched on only when you ask for it, with the command below. Then create
a `sage` environment containing Sage, Python and Jupyter (this downloads
about 2 GB and takes a few minutes), and switch it on:

```bash
~/miniforge3/bin/conda create -n sage "sage>=10.9"
source ~/miniforge3/bin/activate sage
sage --version
```

The last command should print the Sage version, 10.9 or later; Sage 10.10
or later is faster than 10.9 for computations over number fields. More
details are in Sage's guide
[Install from conda-forge](https://doc.sagemath.org/html/en/installation/conda.html).

The prompt now begins with `(sage)`: the environment is switched on. Every
later step starts with `source ~/miniforge3/bin/activate sage`, in each
new Terminal window. To switch it off again (the prompt loses its
`(sage)`), type

```bash
conda deactivate
```

or close the Terminal window. Typing `sage` on its own starts Sage's
interactive prompt, `sage:`; leave it with `quit` (or Ctrl+D).

If you already have Anaconda or Miniconda, you can use it instead of
Miniforge, but Sage only comes from the conda-forge channel: replace
`~/miniforge3` in the commands by the folder of your installation (for
example `~/anaconda3`), create the environment with
`conda create -n sage -c conda-forge --override-channels "sage>=10.9"`,
and add `-c conda-forge --override-channels` to the `conda install`
commands below as well.

**2. The package.** In the `sage` environment, install the C++ compiler
and pkg-config from conda-forge, then the package from PyPI:

```bash
source ~/miniforge3/bin/activate sage
conda install cxx-compiler pkg-config
pip install sage-vertex-algebras
```

pip builds the C++ engines on your machine (about a minute). Check that it
worked:

```bash
sage -c "import vertex_algebras; print(vertex_algebras.__version__)"
```

If you also use Homebrew, check that FLINT and GMP come from the conda
environment and not from Homebrew:

```bash
otool -L "$(python -c 'import vertex_algebras._universal_engine as m; print(m.__file__)')"
```

The `libflint` and `libgmp` lines should point into the environment
(`@rpath/...` or a path under `miniforge3/envs/sage`).

**3. The example notebooks.** The package includes the example notebooks
and its test suite. Copy them to a folder of your own, where you can run
and change them:

```bash
source ~/miniforge3/bin/activate sage
python -m vertex_algebras.examples ~/vertex-algebra-notebooks
```

The folder can be any path that does not exist yet;
`~/vertex-algebra-notebooks` is only an example, and the commands below
use it. The command creates the folder, with the notebooks of the
installed version, their `README.md` (a guide to the notebooks) and the
test suite in the subfolder `tests`. It never overwrites an existing
folder: after upgrading the package, copy the new notebooks into a new
folder.

**4. Jupyter.** The `sage` environment already has Jupyter. Start it in
the notebook folder:

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
sage -n jupyter
```

Jupyter opens in your web browser with the list of notebooks (if it does
not, copy into the browser the address starting with
`http://localhost:8888/` that the terminal prints, with its `token=`
part). Click the first one, `01_getting_started.ipynb`. The kernel in the
top right corner should read "SageMath 10.x" (if Jupyter asks, choose
SageMath). Run the cells one after another with Shift+Enter. The Terminal
window must stay open while you work; to stop Jupyter, press Ctrl+C in it
twice.

Next time, the same three commands start it again.

**5. Optional: the test suite.** Run the examples and tests of the
documentation of the installed package (a minute or two; add `-p 4` after
`--long` to use four processes):

```bash
source ~/miniforge3/bin/activate sage
python -m sage.doctest --long "$(python -c 'import vertex_algebras, os; print(os.path.dirname(vertex_algebras.__file__))')"
```

The last lines should say `All tests passed!`.

The platform report runs the whole test suite, with the C++ engines and in
pure Python, and writes one file with the results and a description of
your system: the file to send when something does not work. Run it in the
folder of step 3 (it takes 15–30 minutes):

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
bash tests/platform_report.sh
```

The report is the file `platform_report_<date>_<time>.txt` in that folder;
its last lines say `ALL PASSED` or list what failed.

### Windows

Sage does not run on Windows itself. As Sage recommends, it runs in the
Windows Subsystem for Linux (WSL), a Linux system inside Windows 10
(version 2004 or later) or Windows 11; the package is then installed
exactly as on Linux.

**1. Python and Sage.** Install WSL with Ubuntu: open PowerShell as
administrator (right-click the Start button → "Terminal (Admin)" or
"Windows PowerShell (Admin)") and run

```powershell
wsl --install
```

Restart the computer when asked. Then open **Ubuntu** from the Start menu;
the first time, it asks you to choose a Linux user name and password. All
the following commands are typed in this Ubuntu window, not in PowerShell.
Microsoft's guide
[Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) has
more details.

In the Ubuntu window, install
[Miniforge](https://github.com/conda-forge/miniforge), a minimal conda
that installs from conda-forge, where Sage is packaged:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Accept the licence and the default location (`~/miniforge3`). When the
installer asks whether to update your shell profile to initialise conda,
answer `no` (the default): your shell stays as it is, and conda is
switched on only when you ask for it, with the command below. Then create
a `sage` environment containing Sage, Python and Jupyter (this downloads
about 2 GB and takes a few minutes), and switch it on:

```bash
~/miniforge3/bin/conda create -n sage "sage>=10.9"
source ~/miniforge3/bin/activate sage
sage --version
```

The last command should print the Sage version, 10.9 or later. More
details are in Sage's guide
[Install from conda-forge](https://doc.sagemath.org/html/en/installation/conda.html).

The prompt now begins with `(sage)`: the environment is switched on. Every
later step starts with `source ~/miniforge3/bin/activate sage`, in each
new Ubuntu window. To switch it off again (the prompt loses its `(sage)`),
type

```bash
conda deactivate
```

or close the Ubuntu window. Typing `sage` on its own starts Sage's
interactive prompt, `sage:`; leave it with `quit` (or Ctrl+D).

**2. The package.** In the `sage` environment, install the C++ compiler
and pkg-config from conda-forge, then the package from PyPI:

```bash
source ~/miniforge3/bin/activate sage
conda install cxx-compiler pkg-config
pip install sage-vertex-algebras
```

pip builds the C++ engines on your machine (about a minute). Check that it
worked:

```bash
sage -c "import vertex_algebras; print(vertex_algebras.__version__)"
```

**3. The example notebooks.** The package includes the example notebooks
and its test suite. Copy them to a folder of your own, where you can run
and change them:

```bash
source ~/miniforge3/bin/activate sage
python -m vertex_algebras.examples ~/vertex-algebra-notebooks
```

The folder can be any path that does not exist yet;
`~/vertex-algebra-notebooks` is only an example, and the commands below
use it. The command creates the folder, with the notebooks of the
installed version, their `README.md` (a guide to the notebooks) and the
test suite in the subfolder `tests`. It never overwrites an existing
folder: after upgrading the package, copy the new notebooks into a new
folder. Keep the folder in the Linux home folder (`~`) rather than in the
Windows folders under `/mnt/c`, which are much slower from WSL. In Windows' File
Explorer, the Linux home folder is under "Linux" → "Ubuntu" → `home` →
your Linux user name.

**4. Jupyter.** The `sage` environment already has Jupyter. Start it in
the notebook folder:

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
sage -n jupyter
```

Jupyter may not be able to open a web browser in Windows by itself. The
Ubuntu window prints an address starting with `http://localhost:8888/`,
with a `token=` part; copy it (select it and press Ctrl+Shift+C) into a
web browser in Windows, which shows the list of notebooks. Click the first
one, `01_getting_started.ipynb`. The kernel in the top right corner should
read "SageMath 10.x" (if Jupyter asks, choose SageMath). Run the cells one
after another with Shift+Enter. The Ubuntu window must stay open while you
work; to stop Jupyter, press Ctrl+C in it twice.

Next time, the same three commands start it again.

**5. Optional: the test suite.** Run the examples and tests of the
documentation of the installed package (a minute or two; add `-p 4` after
`--long` to use four processes):

```bash
source ~/miniforge3/bin/activate sage
python -m sage.doctest --long "$(python -c 'import vertex_algebras, os; print(os.path.dirname(vertex_algebras.__file__))')"
```

The last lines should say `All tests passed!`.

The platform report runs the whole test suite, with the C++ engines and in
pure Python, and writes one file with the results and a description of
your system: the file to send when something does not work. Run it in the
folder of step 3 (it takes 15–30 minutes):

```bash
source ~/miniforge3/bin/activate sage
cd ~/vertex-algebra-notebooks
bash tests/platform_report.sh
```

The report is the file `platform_report_<date>_<time>.txt` in that folder;
its last lines say `ALL PASSED` or list what failed. To attach it to an
email in Windows, find it in File Explorer under "Linux" → "Ubuntu" →
`home` → your Linux user name → `vertex-algebra-notebooks`.

### Common problems

Two things cause most problems:

- **Every new terminal starts outside the Sage environment.** Run
  `source ~/miniforge3/bin/activate sage` first in every new terminal,
  before `pip`, `sage` or `jupyter`; the prompt then begins with `(sage)`.
- **Jupyter must be the one in the Sage environment**, and the notebooks
  must run with the **SageMath** kernel, not "Python 3". Jupyter from
  another installation (Anaconda Navigator, a system Jupyter, VS Code with
  another Python) does not see the package.

If something else goes wrong, please
[open an issue](https://github.com/simonjwood/vertexalgebras/issues) and
attach the platform report of step 5. If the installation in step 2
failed, so that there is nothing to test, include instead the output of
`sage --version` and `uname -sm` and the last 40 lines of the output of
`pip install`.

### Other Sage installations

The package is installed with pip into the Python of an existing Sage
installation. Sage is not a declared dependency, because no single pip
package name covers all the ways Sage is installed; importing
`vertex_algebras` checks the Sage version. These routes have not been
tested yet.

- **Sage built from source** has the build tools, FLINT and GMP:
  `sage -pip install sage-vertex-algebras`.
- **Sage from a Linux distribution** additionally needs the FLINT and GMP
  development headers (e.g. `flint-devel`) and a C++17 compiler from the
  distribution.

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

For more examples, see the [notebooks](https://github.com/simonjwood/vertexalgebras/tree/main/examples)
(their index:
[examples/README.md](https://github.com/simonjwood/vertexalgebras/blob/main/examples/README.md)).

## Documentation

- The docstrings, with examples: e.g. `UniversalVertexAlgebra?` in Sage.
- The reference manual, built from them: `sphinx-build -b html doc
  doc/_build/html` (with the package installed; `doc/requirements.txt`
  lists what the build needs), then open `doc/_build/html/index.html`.
- The example notebooks in
  [`examples/`](https://github.com/simonjwood/vertexalgebras/tree/main/examples),
  numbered in reading order from `01_getting_started.ipynb` (the index is
  [`examples/README.md`](https://github.com/simonjwood/vertexalgebras/blob/main/examples/README.md)):
  the main families of vertex algebras, subalgebras, ideals and quotients,
  cosets, W-algebras, Zhu algebras, modules and morphisms, fusion and
  extensions, the OPE bootstrap, and research examples (singular vectors in
  affine vertex algebras, the triplet algebras, Virasoro fusion at c = 0).

Computations run serially by default. `VertexAlgebraParallelism().set(n)`
(from `vertex_algebras`) lets the C++ engines use `n` threads for the
whole session; results do not depend on it.

## Development

- `tests/run_all.sh` runs the doctests and the test scripts;
  `tests/run_all.sh --fallback` runs the doctests in pure Python, and
  `VERTEX_ALGEBRAS_NO_ENGINE=1 tests/run_all.sh` runs everything without
  the C++ engines.
- `VERTEX_ALGEBRAS_MODE_SPLIT=0` makes the C++ engine of induced modules
  apply the modes of a normally ordered product in the old order, without
  the faster split by the commutator formula; it is a developer switch for
  comparing the two orders, and the guarantees of induced modules do not
  depend on it.

## Acknowledgements

The library started as a port of K. Thielemans' Mathematica package
OPEdefs [Thi1991]: its computation of the OPEs of composite fields follows
the algorithms of OPEdefs, described in [Thi1991] and in K. Thielemans'
thesis [Thi1995]. OPEdefs itself is not included. SW thanks K. Thielemans
for a kind and helpful exchange about OPEdefs.

CR and SW thank Sven Möller and David Ridout for helpful discussions.

CR’s research is supported by the Deutsche Forschungsgemeinschaft through
the Collaborative Research Centre 1624: Higher Structures, Moduli Spaces
and Integrability, project number 506632645.

### Use of AI

Almost all of the source code was written with Claude Code (Anthropic),
under the direction of the authors. The authors designed the API, chose
the constructions and algorithms to support and the examples to present,
and checked the results against the literature. Every function is tested
by doctests, and the computations are cross-checked between independent
implementations and against known results. A line-by-line human review of
the code is planned; it will progress slowly while features are still
being added.

### Software

The library is built on [SageMath](https://www.sagemath.org). Its C++
engines compute with [FLINT](https://flintlib.org) and
[GMP](https://gmplib.org); Gröbner bases in Zhu algebras come from
[Singular](https://www.singular.uni-kl.de) (Plural and Letterplace); lattice
vectors are enumerated with [PARI](https://pari.math.u-bordeaux.fr).

### Where the methods come from

- **Universal vertex algebras:** the OPE recursion of OPEdefs [Thi1991];
  freely generated vertex algebras, their PBW bases and Jacobi identities
  [DSK2005].
- **Lie superalgebras:** the contragredient construction and the
  classification [Kac1977]; root systems [FSS1996]; invariant forms
  normalised as in the tables of [KW2004]; sl₂-triples by Morozov's lemma
  [CM1993, §3.3].
- **Affine vertex algebras and ghost systems:** Sugawara and ghost
  conformal vectors [KRW2003].
- **Lattice vertex algebras:** construction and cocycles [Kac1998,
  5.4–5.5]; their simple modules [Don1993], [DLM1997]; weight spaces by
  Fincke–Pohst enumeration [FP1985].
- **Ideals and quotients:** normal forms [KK2023]; overlap checks by the
  diamond lemma [Ber1978].
- **Morphisms:** the universal property of lattice vertex algebras
  [LL2004, §6.5].
- **W-algebras:** the quantum Hamiltonian reduction [KRW2003], [KW2004],
  [DSK2006]; the reduction of highest weight modules [Ara2005], [Ara2007].
- **Zhu algebras:** [Zhu1996], [FZ1992]; half-integer weights [KW1994];
  H-twisted Zhu algebras [DSK2006]; higher Zhu algebras [DLM1998] in the
  model of [He2017], with the caveat of [BVY2019].
- **Modules:** induced modules [Zhu1996], [DLM1998]; module criteria
  [Li1996]; homomorphisms from the vacuum module [Li1994]; spectral flow
  [Li1997].
- **Fusion products:** the Nahm–Gaberdiel–Kausch algorithm [Nah1994],
  [GK1996], with the coproduct formulas of [KR2019]; the fusion rules of
  lattice vertex algebras [DL1993].
- **Extensions:** projective covers from [DLM1998], [He2017]; Ext¹ from a
  presentation [Wei1994, §3.4]; idempotents and Cartan invariants of
  finite-dimensional algebras [CR1962, §54], [Ben1991, §1.7].
- **OPE bootstrap:** Jacobi identities of OPEs [Thi1991], [KW1991],
  [BFKNRV1991]; the W₃ algebra [Zam1985].

### References

- [Ara2005] T. Arakawa, *Representation theory of superconformal algebras
  and the Kac–Roan–Wakimoto conjecture*, Duke Math. J. 130 (2005), 435–478,
  [arXiv:math-ph/0405015](https://arxiv.org/abs/math-ph/0405015),
  [doi:10.1215/S0012-7094-05-13032-0](https://doi.org/10.1215/S0012-7094-05-13032-0).
- [Ara2007] T. Arakawa, *Representation theory of W-algebras*, Invent.
  Math. 169 (2007), 219–320, [arXiv:math/0506056](https://arxiv.org/abs/math/0506056),
  [doi:10.1007/s00222-007-0046-1](https://doi.org/10.1007/s00222-007-0046-1).
- [Ben1991] D. J. Benson, *Representations and cohomology I*, Cambridge
  Stud. Adv. Math. 30, Cambridge University Press (1991),
  [doi:10.1017/CBO9780511623615](https://doi.org/10.1017/CBO9780511623615).
- [Ber1978] G. M. Bergman, *The diamond lemma for ring theory*, Adv. Math.
  29 (1978), 178–218,
  [doi:10.1016/0001-8708(78)90010-5](https://doi.org/10.1016/0001-8708(78)90010-5).
- [BFKNRV1991] R. Blumenhagen, M. Flohr, A. Kliem, W. Nahm, A. Recknagel,
  R. Varnhagen, *W-algebras with two and three generators*, Nuclear Phys. B
  361 (1991), 255–289,
  [doi:10.1016/0550-3213(91)90624-7](https://doi.org/10.1016/0550-3213(91)90624-7).
- [BVY2019] K. Barron, N. Vander Werf, J. Yang, *Higher level Zhu algebras
  and modules for vertex operator algebras*, J. Pure Appl. Algebra 223
  (2019), 3295–3317, [arXiv:1710.04767](https://arxiv.org/abs/1710.04767),
  [doi:10.1016/j.jpaa.2018.11.002](https://doi.org/10.1016/j.jpaa.2018.11.002).
- [CM1993] D. H. Collingwood, W. M. McGovern, *Nilpotent orbits in
  semisimple Lie algebras*, Van Nostrand Reinhold (1993); CRC Press
  (2017), [doi:10.1201/9780203745809](https://doi.org/10.1201/9780203745809).
- [CR1962] C. W. Curtis, I. Reiner, *Representation theory of finite
  groups and associative algebras*, Interscience (1962); AMS Chelsea
  (2006), [doi:10.1090/chel/356](https://doi.org/10.1090/chel/356).
- [Don1993] C. Dong, *Vertex algebras associated with even lattices*, J.
  Algebra 161 (1993), 245–265,
  [doi:10.1006/jabr.1993.1217](https://doi.org/10.1006/jabr.1993.1217).
- [DL1993] C. Dong, J. Lepowsky, *Generalized vertex algebras and relative
  vertex operators*, Birkhäuser, Boston (1993),
  [doi:10.1007/978-1-4612-0353-7](https://doi.org/10.1007/978-1-4612-0353-7).
- [DLM1997] C. Dong, H. Li, G. Mason, *Regularity of rational vertex
  operator algebras*, Adv. Math. 132 (1997), 148–166,
  [arXiv:q-alg/9508018](https://arxiv.org/abs/q-alg/9508018),
  [doi:10.1006/aima.1997.1681](https://doi.org/10.1006/aima.1997.1681).
- [DLM1998] C. Dong, H. Li, G. Mason, *Vertex operator algebras and
  associative algebras*, J. Algebra 206 (1998), 67–96,
  [arXiv:q-alg/9612010](https://arxiv.org/abs/q-alg/9612010),
  [doi:10.1006/jabr.1998.7425](https://doi.org/10.1006/jabr.1998.7425).
- [DSK2005] A. De Sole, V. G. Kac, *Freely generated vertex algebras and
  non-linear Lie conformal algebras*, Comm. Math. Phys. 254 (2005),
  659–694, [arXiv:math-ph/0312042](https://arxiv.org/abs/math-ph/0312042),
  [doi:10.1007/s00220-004-1245-x](https://doi.org/10.1007/s00220-004-1245-x).
- [DSK2006] A. De Sole, V. G. Kac, *Finite vs affine W-algebras*, Jpn. J.
  Math. 1 (2006), 137–261,
  [arXiv:math-ph/0511055](https://arxiv.org/abs/math-ph/0511055),
  [doi:10.1007/s11537-006-0505-2](https://doi.org/10.1007/s11537-006-0505-2).
- [FP1985] U. Fincke, M. Pohst, *Improved methods for calculating vectors
  of short length in a lattice, including a complexity analysis*, Math.
  Comp. 44 (1985), 463–471,
  [doi:10.1090/S0025-5718-1985-0777278-8](https://doi.org/10.1090/S0025-5718-1985-0777278-8).
- [FSS1996] L. Frappat, A. Sciarrino, P. Sorba, *Dictionary on Lie
  superalgebras* (1996), [arXiv:hep-th/9607161](https://arxiv.org/abs/hep-th/9607161).
- [FZ1992] I. B. Frenkel, Y. Zhu, *Vertex operator algebras associated to
  representations of affine and Virasoro algebras*, Duke Math. J. 66
  (1992), 123–168,
  [doi:10.1215/S0012-7094-92-06604-X](https://doi.org/10.1215/S0012-7094-92-06604-X).
- [GK1996] M. R. Gaberdiel, H. G. Kausch, *Indecomposable fusion
  products*, Nuclear Phys. B 477 (1996), 293–318,
  [arXiv:hep-th/9604026](https://arxiv.org/abs/hep-th/9604026),
  [doi:10.1016/0550-3213(96)00364-1](https://doi.org/10.1016/0550-3213(96)00364-1).
- [He2017] X. He, *Higher level Zhu algebras are subquotients of universal
  enveloping algebras*, J. Algebra 491 (2017), 265–279,
  [arXiv:1712.05671](https://arxiv.org/abs/1712.05671),
  [doi:10.1016/j.jalgebra.2017.08.014](https://doi.org/10.1016/j.jalgebra.2017.08.014).
- [Kac1977] V. G. Kac, *Lie superalgebras*, Adv. Math. 26 (1977), 8–96,
  [doi:10.1016/0001-8708(77)90017-2](https://doi.org/10.1016/0001-8708(77)90017-2).
- [Kac1998] V. G. Kac, *Vertex algebras for beginners*, 2nd ed., Univ.
  Lecture Ser. 10, AMS (1998),
  [doi:10.1090/ulect/010](https://doi.org/10.1090/ulect/010).
- [KK2023] R. A. Kozlov, P. S. Kolesnikov, *The Gröbner–Shirshov bases
  method for vertex algebras*, J. Math. Sci. 296 (2026), 217–230,
  [arXiv:2312.01670](https://arxiv.org/abs/2312.01670),
  [doi:10.1007/s10958-026-08246-0](https://doi.org/10.1007/s10958-026-08246-0).
- [KR2019] S. Kanade, D. Ridout, *NGK and HLZ: fusion for physicists and
  mathematicians*, in: Affine, vertex and W-algebras, Springer INdAM Ser.
  37 (2019), 135–181, [arXiv:1812.10713](https://arxiv.org/abs/1812.10713),
  [doi:10.1007/978-3-030-32906-8_7](https://doi.org/10.1007/978-3-030-32906-8_7).
- [KRW2003] V. G. Kac, S.-S. Roan, M. Wakimoto, *Quantum reduction for
  affine superalgebras*, Comm. Math. Phys. 241 (2003), 307–342,
  [arXiv:math-ph/0302015](https://arxiv.org/abs/math-ph/0302015),
  [doi:10.1007/s00220-003-0926-1](https://doi.org/10.1007/s00220-003-0926-1).
- [KW1991] H. G. Kausch, G. M. T. Watts, *A study of W-algebras using
  Jacobi identities*, Nuclear Phys. B 354 (1991), 740–768,
  [doi:10.1016/0550-3213(91)90375-8](https://doi.org/10.1016/0550-3213(91)90375-8).
- [KW1994] V. G. Kac, W. Wang, *Vertex operator superalgebras and their
  representations*, Contemp. Math. 175 (1994), 161–191,
  [arXiv:hep-th/9312065](https://arxiv.org/abs/hep-th/9312065),
  [doi:10.1090/conm/175/01843](https://doi.org/10.1090/conm/175/01843).
- [KW2004] V. G. Kac, M. Wakimoto, *Quantum reduction and representation
  theory of superconformal algebras*, Adv. Math. 185 (2004), 400–458,
  [arXiv:math-ph/0304011](https://arxiv.org/abs/math-ph/0304011),
  [doi:10.1016/j.aim.2003.12.005](https://doi.org/10.1016/j.aim.2003.12.005);
  corrigendum, Adv. Math. 193 (2005), 453–455,
  [doi:10.1016/j.aim.2005.01.001](https://doi.org/10.1016/j.aim.2005.01.001).
- [Li1994] H. Li, *Symmetric invariant bilinear forms on vertex operator
  algebras*, J. Pure Appl. Algebra 96 (1994), 279–297,
  [doi:10.1016/0022-4049(94)90104-X](https://doi.org/10.1016/0022-4049(94)90104-X).
- [Li1996] H. Li, *Local systems of vertex operators, vertex superalgebras
  and modules*, J. Pure Appl. Algebra 109 (1996), 143–195,
  [arXiv:hep-th/9406185](https://arxiv.org/abs/hep-th/9406185),
  [doi:10.1016/0022-4049(95)00079-8](https://doi.org/10.1016/0022-4049(95)00079-8).
- [Li1997] H. Li, *The physics superselection principle in vertex operator
  algebra theory*, J. Algebra 196 (1997), 436–457,
  [doi:10.1006/jabr.1997.7126](https://doi.org/10.1006/jabr.1997.7126);
  based on the first part of the preprint [arXiv:q-alg/9504026](https://arxiv.org/abs/q-alg/9504026),
  whose numbering differs.
- [LL2004] J. Lepowsky, H. Li, *Introduction to vertex operator algebras
  and their representations*, Progr. Math. 227, Birkhäuser (2004),
  [doi:10.1007/978-0-8176-8186-9](https://doi.org/10.1007/978-0-8176-8186-9).
- [Nah1994] W. Nahm, *Quasi-rational fusion products*, Internat. J.
  Modern Phys. B 8 (1994), 3693–3702,
  [arXiv:hep-th/9402039](https://arxiv.org/abs/hep-th/9402039),
  [doi:10.1142/S0217979294001597](https://doi.org/10.1142/S0217979294001597).
- [Thi1991] K. Thielemans, *A Mathematica package for computing operator
  product expansions*, Internat. J. Modern Phys. C 2 (1991), 787–798,
  [doi:10.1142/S0129183191001001](https://doi.org/10.1142/S0129183191001001).
- [Thi1995] K. Thielemans, *An algorithmic approach to operator product
  expansions, W-algebras and W-strings*, PhD thesis, KU Leuven (1995),
  [arXiv:hep-th/9506159](https://arxiv.org/abs/hep-th/9506159).
- [Wei1994] C. A. Weibel, *An introduction to homological algebra*,
  Cambridge Stud. Adv. Math. 38, Cambridge University Press (1994),
  [doi:10.1017/CBO9781139644136](https://doi.org/10.1017/CBO9781139644136).
- [Zam1985] A. B. Zamolodchikov, *Infinite additional symmetries in
  two-dimensional conformal quantum field theory*, Theoret. and Math. Phys.
  65 (1985), 1205–1213,
  [doi:10.1007/BF01036128](https://doi.org/10.1007/BF01036128).
- [Zhu1996] Y. Zhu, *Modular invariance of characters of vertex operator
  algebras*, J. Amer. Math. Soc. 9 (1996), 237–302,
  [doi:10.1090/S0894-0347-96-00182-8](https://doi.org/10.1090/S0894-0347-96-00182-8).

## Authors and licence

Christopher Raymond and Simon Wood. Copyright (C) 2026 Cardiff University and
University of Hamburg.

Licensed under the GNU General Public License, version 2 or later
(GPL-2.0-or-later); see
[LICENSE](https://github.com/simonjwood/vertexalgebras/blob/main/LICENSE).

To cite the package, see
[CITATION.cff](https://github.com/simonjwood/vertexalgebras/blob/main/CITATION.cff)
(GitHub's "Cite this repository").
