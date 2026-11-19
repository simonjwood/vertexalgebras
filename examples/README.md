# Example notebooks

Sixteen Jupyter notebooks that introduce the package through worked
examples, numbered in reading order. They are saved with their outputs, so
they can be read without running them. To run them, open them in a Jupyter
session with the SageMath kernel, with the package installed (see the
[main README](../README.md)).

This file first describes how the notebooks are organised, then gives a
quick reference to the contents of each one, and ends with an index of
topics that appear in several notebooks.

## How the notebooks are organised

### The sequence

| notebooks | part | contents |
|---|---|---|
| 01 | Start here | the basic objects and conventions that every other notebook uses |
| 02–05 | The main families of algebras | Lie superalgebras, affine vertex algebras, ghost systems, lattice vertex algebras |
| 06–07 | Constructions inside an algebra | subalgebras, ideals, quotients, commutants and cosets |
| 08–12 | Further structures | W-algebras, Zhu algebras, modules, fusion and extensions, the OPE bootstrap |
| 13–16 | Research examples | longer computations that reproduce results from the literature |

A notebook only assumes notebooks with smaller numbers, so reading in order
always works. To reach a particular notebook sooner, follow its
prerequisites, listed in the quick reference below: for example, the
shortest path to `15_triplet_modules` is 01, 06, 09, 10, 11, 15.

### The layout of a notebook

Every notebook follows the same pattern.

1. **The opening cell** gives the title and says what the notebook shows,
   with the definitions and conventions needed to read it. Some also
   tabulate the constructions they cover with the section of each (03, 10,
   11). The cell ends with labelled paragraphs:
   - **Prerequisites:** the notebooks it assumes;
   - **Elsewhere:** related material in other notebooks;
   - **Contents:** the numbered list of its sections.
2. **The first code cell** loads the package,
   ```python
   import vertex_algebras as va      # the catalog: va.Virasoro, va.Affine, va.Lattice, ...
   from vertex_algebras import *     # the classes: UniversalVertexAlgebra, LieSuperalgebra, ...
   ```
   together with anything else the notebook needs. Each notebook is
   self-contained: it uses nothing defined in another notebook.
3. **Numbered sections** (`## 1. …`) follow the contents list. Inside a
   section, markdown and code cells alternate: a markdown cell explains the
   next computation and says what the code cell prints, and the markdown
   cell after it usually interprets the output.
4. **Limitations** is the last numbered section: what is not implemented,
   which results hold only up to a weight or a depth, and what is
   expensive.
5. **References** is the last cell: the works cited in the text, by keys
   such as [Kac1998].

### Conventions shared by the notebooks

- **Products.** `a.ope(b)` is the list of poles of $a(z)b(w)$, starting
  with the first-order pole: entry $k$ is $a_{(k)}b$. `a.nproduct(b, n)`
  is $a_{(n)}b$ for any integer $n$, `a.no(b)` is the normally ordered
  product ${:}ab{:}$, and `a.mode(m, n)` is the mode $a_n m$ in the physics
  convention. These are introduced in `01_getting_started.ipynb`, sections
  1 and 4.
- **Exact answers and answers up to a weight.** A computation only sees
  finitely many weights, and each method says, by its name or its result,
  how far its answer goes. Subalgebras, ideals and quotients are exact
  weight by weight; methods whose answers hold only up to a weight or a
  depth carry this in their names (`strong_generators_up_to`,
  `zhu_algebra_up_to`, `induced_module_up_to`, `depth_quotient_up_to`,
  `ext1_up_to`, ...), and `is_known_exact()` says whether a Zhu algebra is
  known to be $A(V)$. `06_subalgebras_ideals_quotients.ipynb`, section 2,
  explains this, and the Limitations sections return to it where it
  matters.
- **Cross-references** between notebooks are written as
  `10_modules.ipynb`, section 4.
- **Cells that fail on purpose.** Some cells show the library refusing
  invalid input. Those that raise an error (in 01, 03 and 06) are tagged
  `raises-exception`, so that executing a whole notebook, for instance
  with `jupyter nbconvert --execute`, does not stop there; the others catch
  the error and print it.

### Running time and threads

Two research examples contain long computations. Each says how long its
steps take and how to shorten them:

- `14_triplet_algebra`: the settings are in section 1 (`pp`, `pm`,
  `NCPUS`, `DATA_DIR`, `RECOMPUTE`). For $(p_+, p_-) = (2, 3)$, the first
  run verifies strong generation, which takes about 2 minutes and 5 GB of
  RAM on a 28-core workstation, and saves the results in `DATA_DIR`
  (`triplet_data/` by default, ignored by git); later runs load them in
  about half a minute. Other pairs take considerably longer.
- `16_virasoro_c0_fusion`: the table of section 3 takes about ten minutes.
  The two couplings at level 4 in section 4 take 10–15 minutes each, and
  are only recomputed with `SLOW = True`.

The C++ engines run serially unless `VertexAlgebraParallelism().set(n)` is
called (`01_getting_started.ipynb`, section 8); the results do not depend
on the number of threads. `13_singular_vectors_in_affine_vertex_algebras`
uses 4 threads and `14_triplet_algebra` uses `NCPUS`.

## Quick reference

Each entry says what the notebook shows, lists its sections, and names the
main methods it demonstrates and the notebooks it assumes.

### Start here

#### [01 Getting started](01_getting_started.ipynb)

How to define a vertex algebra by generators and OPEs, compute with its
elements, give it a conformal vector, look at its weight spaces and
character, and print its elements in different styles; then a tour of the
ready-made algebras. Its **Elsewhere** paragraph maps out the other
notebooks.

- **Sections:** 1 A first algebra: the Virasoro vertex algebra; 2 Defining
  an algebra by its generators and OPEs; 3 Odd generators; 4 Conformal
  vectors and conformal weights; 5 Weight spaces and characters;
  6 Printing; 7 The catalog of named algebras; 8 Coefficient rings and
  threads; 9 Limitations.
- **Main methods:** `UniversalVertexAlgebra`, `NormallyOrderedExpressions`,
  `a.ope(b)`, `a.nproduct(b, n)`, `a.no(b)`, `a.derivative(n)`,
  `V.with_conformal_vector(T)`, `V.weight_space(n)`, `V.character(n)`,
  `V.print_options(...)`, the catalog (`va.Virasoro`, `va.W3`,
  `va.NeveuSchwarz`, `va.N2`, `va.Affine`, `va.Lattice`, `va.W`, ...),
  `VertexAlgebraParallelism`.
- **Prerequisites:** none.

### The main families of algebras

#### [02 Lie superalgebras](02_lie_superalgebras.ipynb)

The basic classical Lie superalgebras, the input of affine vertex
superalgebras (03) and of quantum Hamiltonian reduction (08). They are
built from a Cartan matrix and a set of odd simple roots, or by name
(`sl(m|n)`, `psl(m|m)`, `osp(m|n)`, `spo(n|m)`, `D(2,1;a)`, `F(4)`,
`G(3)`), with the invariant form normalised as in Kac–Wakimoto's tables;
the opening cell tabulates the named types with their dual Coxeter
numbers. Since Sage has no Lie superalgebras, the library also provides
their category.

- **Sections:** 1 A first example: $\mathfrak{sl}(2|1)$; 2 The named types
  and Kac–Wakimoto's normalisation; 3 Roots; 4 The invariant form and the
  Killing form; 5 Structure constants and the category; 6 $D(2,1;a)$ with
  a symbolic parameter; 7 Other Dynkin diagrams, Lie algebras, refusals;
  8 Affine vertex superalgebras; 9 Unique representation and pickling;
  10 Limitations.
- **Main methods:** `LieSuperalgebra(R, name)`,
  `LieSuperalgebra(R, cartan_matrix, odd=..., theta=...)`,
  `g.positive_roots()`, `g.odd_simple_roots()`, `g.invariant_form`,
  `g.killing_form`, `g.dual_coxeter_number()`, `g.superdimension()`,
  `LieSuperalgebras`, `UniversalAffineVertexAlgebra`.
- **Prerequisites:** 01.

#### [03 Affine vertex algebras](03_affine_vertex_algebras.ipynb)

The universal affine vertex algebra $V^k(\mathfrak g)$ of a Lie algebra or
Lie superalgebra with an invariant form: its OPEs, four ways of giving
$\mathfrak g$ (a Cartan type, a `LieSuperalgebra`, a Lie algebra of Sage,
or structure constants), the same algebra built by hand, the Sugawara
vector and the critical level, the grading by the root lattice, and
singular vectors with the simple quotients $L_k(\mathfrak{sl}_2)$ at
levels 1 and 2.

- **Sections:** 1 The OPEs, the invariant form and the level; 2 Four ways
  to give the Lie algebra; 3 Building it by hand; 4 The Sugawara vector
  and the critical level; 5 The root lattice grading; 6 Singular vectors
  and the simple quotient at integrable level; 7 Limitations.
- **Main methods:** `va.Affine(R, g, k)`, `conformal_vector='sugawara'`,
  `V.invariant_form_matrix()`, `V.killing_form_matrix()`,
  `x.horizontal_grade()`, `V.weight_space(n, horizontal_grade=a)`,
  `V.find_singular(n, ...)`, `V / I`.
- **Prerequisites:** 01, 02.

#### [04 Ghost systems](04_ghost_systems.ipynb)

The free field algebras that appear as ghosts: the βγ system
(`BosonicGhosts`), the bc system (`FermionicGhosts`), and Kac–Roan–Wakimoto's
neutral free superfermions (`FreeSuperFermions`), given by a Gram matrix or
by a basis and a form. Named and general conformal vectors, the charge
grading, several pairs, tensor products, a comparison with Sage's Lie
conformal algebras of the same names, and affine $\mathfrak{sl}_2$ at
level $-1/2$ inside the βγ system.

- **Sections:** 1 The βγ system and its conformal vectors; 2 Weight
  spaces, characters and the charge grading; 3 Several pairs; 4 The bc
  system; 5 The same algebras as Sage's Lie conformal algebras; 6 Free
  superfermions from a Gram matrix; 7 Prescribed weights and general
  conformal vectors; 8 From a basis and a form: the neutral fermions of a
  reduction; 9 Tensor products of ghost systems; 10 Unique representation
  and pickling; 11 Affine $\mathfrak{sl}_2$ at level $-1/2$ inside the βγ
  system; 12 Limitations.
- **Main methods:** `BosonicGhosts`, `FermionicGhosts`,
  `FreeSuperFermions`, `conformal_vector=` (`'symmetric'`, `'b1c0'`,
  `'b0c1'`, `'degrees'`), `beta_weight=`, `b_weight=`,
  `horizontal_grading='charge'`, `V.tensor(W)`, `V.hom(images, W)` with
  `kernel()`, `image()` and `is_conformal()`.
- **Prerequisites:** 01.

#### [05 Lattice vertex algebras](05_lattice_vertex_algebras.ipynb)

The lattice vertex algebra $V_L$ of an integral lattice of any signature,
built from Heisenberg currents $J_i$ and vertex operators $e^q$: conformal
vectors (with background charges), weights and characters, the cocycle,
odd lattices and free fermions, the indefinite lattice $I_{1,1}$ and the
βγ system inside it, tensor products, and Sage's `IntegralLattice` as
input.

- **Sections:** 1 A first example: the $A_1$ lattice; 2 Conformal vectors,
  weights and characters; 3 The cocycle: the $A_2$ lattice; 4 Odd
  lattices: free fermions; 5 Indefinite lattices: $I_{1,1}$; 6 The βγ
  system inside $V_{I_{1,1}}$; 7 Tensor products; 8 Unique
  representation, pickling and Sage lattices; 9 Limitations.
- **Main methods:** `LatticeVertexAlgebra(R, B)` (also `va.Lattice`),
  `V.J(i)`, `V.e(q)`, `lattice_conformal_vector(V)`,
  `V.with_conformal_vector(T)`, `V.cocycle`, `V.tensor(W)`,
  `V.hom(images, W)`.
- **Prerequisites:** 01; sections 4, 6 and 7 compare with 04.

### Constructions inside an algebra

#### [06 Subalgebras, ideals and quotients](06_subalgebras_ideals_quotients.ipynb)

The subalgebra generated by a set of elements, with strong generators found
by a closure; the ideal generated by a set of elements, with membership
tests, certificates and normal forms; and quotients. Applications: the
Virasoro minimal models as quotients, compared with their known characters,
and presentations of subalgebras as universal vertex algebras modulo
relations. Section 2 explains which answers are exact and which hold up to
a weight, a distinction used throughout the later notebooks.

- **Sections:** 1 Subalgebras; 2 Exact answers and answers up to a weight;
  3 Ideals; 4 Quotients; 5 Minimal models; 6 Presentations; 7 Limitations.
- **Main methods:** `V.subalgebra(A, names=...)`,
  `S.strong_generator_info()`, `S.express(x)`, `V.ideal(A)`,
  `I.express(x)`, `I.normal_form(x)`, `V.quotient(I)` or `V / I`,
  `Q.lift(x)`, `V.find_singular(n)`, `S.presentation()`.
- **Prerequisites:** 01.

#### [07 Commutants and cosets](07_commutants_and_cosets.ipynb)

The commutant $\mathrm{Com}(A, V)$ of a subalgebra, the coset of the coset
construction: exact weight spaces, strong generators up to a weight, the
coset conformal vector $T_V - T_A$, and double commutants. Two classical
examples: GKO, where $\mathrm{Com}(L_2(\mathfrak{sl}_2), L_1(\mathfrak{sl}_2)
\otimes L_1(\mathfrak{sl}_2))$ is the Ising model, and Kazama–Suzuki, where
the commutant of a Heisenberg field in $V^k(\mathfrak{sl}_2) \otimes bc$ is
the $N = 2$ superconformal algebra.

- **Sections:** 1 Exact weight spaces, strong generators up to a weight;
  2 GKO: the Ising model as a coset; 3 Kazama–Suzuki: the $N = 2$
  superconformal algebra; 4 Limitations.
- **Main methods:** `V.commutant(A)` (also `V.coset(A)`),
  `C.weight_space(W)`, `C.strong_generators_up_to(W)`,
  `conformal_vector='coset'`, `C.commutant(max_weight=k)`.
- **Prerequisites:** 01, 05 (for section 2), 06.

### Further structures

#### [08 W-algebras](08_w_algebras.ipynb)

The universal W-algebra $W^k(\mathfrak g, f)$ of a basic Lie superalgebra,
a nilpotent element and a good grading, by quantum Hamiltonian reduction:
Virasoro from $\mathfrak{sl}_2$, $W_3$ from $\mathfrak{sl}_3$, the choice
of $f$ and of the grading, the nilpotent orbits of $\mathfrak{sl}_4$,
$\mathfrak{so}_5 \cong \mathfrak{sp}_4$ and $G_2$, and the superconformal
algebras as minimal reductions of Lie superalgebras. Also how the reduction
is computed, the H-twisted Zhu algebras of W-algebras, and the reduction
of highest weight modules.

- **Sections:** 1 The Virasoro algebra from $\mathfrak{sl}_2$; 2 $W_3$ from
  $\mathfrak{sl}_3$; 3 Choosing $f$ and $x$; 4 The nilpotent orbits of
  $\mathfrak{sl}_4$; 5 Other Lie algebras; 6 Superconformal algebras;
  7 How it is computed; 8 Zhu algebras; 9 Reduction of modules; 10 The
  affine method, caching, pickling; 11 Limitations.
- **Main methods:** `WAlgebra(R, g, f, k)` with `f = 'principal'`,
  `'minimal'` or a Jordan type, `conformal_vector='reduction'`,
  `g.nilpotent_of_jordan_type(...)`, `g.sl2_triple(...)`,
  `V.quantum_hamiltonian_reduction(f)`, `W.zhu_algebra(twist='H')`,
  `M.quantum_hamiltonian_reduction(f, reduction=...)`.
- **Prerequisites:** 01, 02, 03.

#### [09 Zhu algebras](09_zhu_algebras.ipynb)

Zhu's algebra $A(V)$, whose simple modules are the top spaces of the
simple positive-energy modules of $V$. For universal vertex algebras it is
computed by rewriting, with the Zhu map and the diamond lemma (Virasoro,
affine $\mathfrak{sl}_2$, $W_3$, odd generators, half-integer weights);
for quotients it is computed up to a weight (minimal models,
$L_k(\mathfrak g)$, and the 3-state Potts model, where the images of the
generators of the ideal are not enough). Also lattice vertex algebras,
tensor products, the H-twisted (Ramond) Zhu algebra, with the Ramond
sector of the tricritical Ising model, and higher Zhu algebras.

- **Sections:** 1 Virasoro: $A(V) = \mathbb C[x]$ and the Zhu map;
  2 Affine $\mathfrak{sl}_2$: $U(\mathfrak{sl}_2)$ and the Casimir; 3 $W_3$
  and the diamond lemma; 4 Odd generators: exterior and Clifford algebras;
  5 Half-integer weights: $N=1$, $N=2$, $\beta\gamma$; 6 Quotients: minimal
  models and $L_k(\mathfrak g)$; 7 Images of the generators of an ideal are
  not enough: the 3-state Potts model; 8 Lattice vertex algebras; 9 Tensor
  products; 10 The H-twisted (Ramond) Zhu algebra; 11 Higher Zhu algebras;
  12 `TestSuite`; 13 Limitations.
- **Main methods:** `V.zhu_algebra()`, the Zhu map `A(v)`,
  `A.relations()`, `A.check_overlaps()`, `A.is_known_exact()`,
  `A.as_g_algebra()`, `Q.zhu_algebra_up_to(W)`,
  `V.zhu_algebra(twist='H')`, `V.zhu_algebra(level=n)`.
- **Prerequisites:** 01, 06.

#### [10 Modules](10_modules.ipynb)

Modules over vertex algebras and the maps between them: the adjoint
module, the simple modules $V_{\lambda+L}$ of lattice vertex algebras,
modules over Zhu algebras as top levels, induced (Verma and Weyl) modules,
submodules, quotients and simple modules, gradings by abelian groups,
affine Verma modules with infinite top levels at admissible levels,
modules over quotients, morphisms of vertex algebras and of modules, and
new modules by pullback, twist and spectral flow.

- **Sections:** 1 The category and the adjoint module; 2 Lattice modules
  $V_{\lambda+L}$; 3 Modules over Zhu algebras; 4 Induced modules;
  5 Submodules, quotients and simple modules; 6 Gradings by abelian groups;
  7 Infinite top levels: affine Verma modules and admissible levels;
  8 Modules over quotients; 9 Morphisms of vertex algebras; 10 Module
  morphisms; 11 Pullbacks, twists and spectral flow; 12 Limitations.
- **Main methods:** `V.adjoint_module()`, `V.simple_module(λ)`,
  `V.simple_modules()`, `A.module(...)`, `V.induced_module(U)`,
  `V.verma_module(h)`, `M.submodule(...)`, `M / N`,
  `M.maximal_submodule()`, `M.simple_quotient()`, `M.direct_sum(N)`,
  `A.verma_module(λ)`, `Q.induced_module_up_to(U, max_weight, max_depth)`,
  `V.hom(...)`, `Hom(M, N)`, `M.pullback(φ)`, `M.twist(σ)`,
  `M.spectral_flow(h)`.
- **Prerequisites:** 01, 09.

#### [11 Fusion and extensions](11_fusion_and_extensions.ipynb)

Fusion products $M \boxtimes N$ by the Nahm–Gaberdiel–Kausch algorithm,
computed as depth quotients in a finite window, and extensions between
modules, from projective covers, $\operatorname{Ext}^1$ and Loewy layers.
Every example has a known answer: the Lee–Yang and Ising models, affine
$\mathfrak{sl}_2$ at level 1, lattice vertex algebras, the first
logarithmic fusion product at $c = -2$, and a Jordan block of $L_0$ on a
Virasoro module.

- **Sections:** 1 Fusion by the Nahm–Gaberdiel–Kausch algorithm; 2 The
  window; 3 More rational examples: Ising and affine $\mathfrak{sl}_2$ at
  level 1; 4 A logarithmic fusion product at $c = -2$; 5 Projective covers
  and $\operatorname{Ext}^1$: the semisimple case; 6 A non-split extension:
  a Jordan block of $L_0$; 7 Limitations.
- **Main methods:** `M.fusion_product(N)`,
  `P.depth_quotient_up_to(d, max_depth)` with `l0_matrix()`,
  `conformal_weights()`, `mode_matrix(a, n)` and `zhu_algebra_module(A)`,
  `F.special_subspace_dimensions()`,
  `L.projective_cover(max_weight, max_depth)`,
  `M.ext1_up_to(N, depth, cover)` with `extension(c)`,
  `M.radical_series_up_to(...)`, `M.socle_series_up_to(...)`.
- **Prerequisites:** 01, 09, 10.

#### [12 OPE bootstrap](12_ope_bootstrap.ipynb)

Checking that declared OPEs satisfy the Jacobi identities, and solving for
unknown ones: `OPEBootstrap` gives every undeclared OPE the most general
ansatz allowed by the weights, parities and grading, and imposes the
Jacobi identities as polynomial equations in the unknown coefficients.
Examples: $W_3$, checked and bootstrapped; the central charge as a
parameter; $W(2,4)$, whose self-coupling is a square root; the $N = 2$
superconformal algebra; $W(2,3,4)$; and saving and resuming a bootstrap.

- **Sections:** 1 Checking declared OPEs: the $W_3$ algebra;
  2 Bootstrapping $W_3$; 3 The central charge as a parameter; 4 Square
  roots: $W(2,4)$; 5 The $N=2$ superconformal algebra: odd generators and a
  grading; 6 $W(2,3,4)$; 7 Saving and resuming a bootstrap; 8 Limitations.
- **Main methods:** `V.jacobi_identity_failures()`,
  `V.satisfies_jacobi_identity()`,
  `UniversalVertexAlgebra(..., verify_jacobi=True)`, `OPEBootstrap(V)`,
  `B.impose_jacobi()`, `B.solve()`, `S.algebra()`, `save` and `load` of a
  bootstrap.
- **Prerequisites:** 01.

### Research examples

#### [13 Singular vectors in affine vertex algebras](13_singular_vectors_in_affine_vertex_algebras.ipynb)

Singular vectors of a given conformal weight and root lattice grade in
$V^k(\mathfrak g)$, found by linear algebra over $\mathbb Q[k]$, and the
ideals they generate. The main example is $D_4^{(1)}$ at level $k = -2$,
where $V^{-2}(\mathfrak{so}_8)$ has three singular vectors of weight 2;
their ideals and the sums of these lead to the simple quotient
$L_{-2}(D_4)$, whose graded dimensions are compared with the literature.

- **Sections:** 1 The algebra and its data; 2 Singular vectors of given
  conformal weight and grade; 3 $D_4^{(1)}$ at level $k = -2$: three
  singular vectors and sums of ideals; 4 Limitations.
- **Main methods:** `UniversalAffineVertexAlgebra(R, cartan, level=k)`,
  `V.weight_space(n, horizontal_grade=a)`, `V.ideal(...)`, `V / I`,
  `VertexAlgebraParallelism`.
- **Prerequisites:** 03, 06.

#### [14 The triplet algebra](14_triplet_algebra.ipynb)

The $(p_+, p_-)$ triplet algebra $\mathcal M_{p_+,p_-}$, a logarithmic extension of the Virasoro
minimal model, following Tsuchiya–Wood. Inside the lattice vertex algebra
of $\mathbb{Z}\sqrt{2 p_+ p_-}$, its $W$ fields are built from screening
operators through Jack polynomials; the notebook constructs the subalgebra
that they and $T$ generate (which verifies that they strongly generate
it), and computes and checks the OPEs of the $W$ fields. The default is
$(p_+, p_-) = (2, 3)$, at $c = 0$; any coprime pair $\ge 2$ can be set in
section 1. See *Running time and threads* above.

- **Sections:** 1 Parameters; 2 The lattice vertex algebra and the
  dictionary to [TW2013]; 3 Jack polynomials in power sums; 4 Screening
  operators; 5 The soliton vector $W_{0,0}$ and the $W$ fields; 6 The
  vertex subalgebra generated by $T, W^-, W^0, W^+$; 7 OPEs; 8 Checks: no
  vacuum terms, nothing below weight $\Delta_0$, poles in the ideal of
  $W_{0,0}$; 9 Normalisation; 10 Limitations.
- **Main methods:** `LatticeVertexAlgebra` with a background charge,
  Sage's `SymmetricFunctions`, `L.subalgebra(gens,
  horizontal_grading='ambient')`, `S.express(x)`, `V.ideal(...)`, `save`
  and `load`.
- **Prerequisites:** 05, 06.

#### [15 Triplet modules](15_triplet_modules.ipynb)

The triplet algebra $\mathcal W(2)$, the even part of symplectic fermions
at $c = -2$, computed from its definition as a subalgebra: its
presentation, whose OPEs satisfy the Jacobi identities only modulo null
fields; its Zhu algebra and four simple modules; the projective covers
$\mathcal P(0)$ and $\mathcal P(1)$, the second through the level-1 Zhu
algebra; fusion products; and the same results by the library's general
methods. Each result is checked against the symplectic fermion Fock
spaces.

- **Sections:** 1 $\mathcal W(2)$ inside symplectic fermions; 2 The
  presentation; 3 The Zhu algebra and its modules; 4 The simple modules;
  5 The module $\mathcal W(0) \to 2\,\mathcal W(1)$; 6 The projective cover
  $\mathcal P(0)$; 7 Up to a weight and a depth; 8 The projective cover
  $\mathcal P(1)$ through the level-1 Zhu algebra; 9 Fusion products;
  10 The same, by the library: simple modules, projective covers, Loewy
  layers, extensions; 11 Limitations.
- **Main methods:** `V.subalgebra(...)`, `S.presentation()`,
  `P.zhu_algebra_up_to(W)` (also with `level=1`),
  `P.induced_module_up_to(U, max_weight, max_depth)`,
  `M.simple_quotient()`, `M.fusion_product(N)`, `L.projective_cover(...)`,
  `M.radical_series_up_to(...)`, `M.socle_series_up_to(...)`,
  `M.ext1_up_to(...)`.
- **Prerequisites:** 06, 09, 10, 11.

#### [16 Virasoro fusion at c = 0](16_virasoro_c0_fusion.ipynb)

Eberle and Flohr's fusion products of the universal Virasoro vertex
algebra at $c = 0$, recomputed with `fusion_product`: the correction by
Gaberdiel–Runkel–Wood ($\mathcal V(2) \boxtimes \mathcal V(2)$ is the
contragredient $\mathcal V^*$ of the vacuum module, not $\mathcal V$),
Eberle and Flohr's table of 38 products at depth 0, the logarithmic
couplings of the rank-2 modules, and modules of rank 3. See *Running time
and threads* above.

- **Sections:** 1 The window; 2 The correction of Gaberdiel–Runkel–Wood;
  3 Eberle–Flohr's table at depth 0; 4 Rank 2: the logarithmic couplings;
  5 Rank 3; 6 Limitations.
- **Main methods:** `va.Virasoro(QQ, 0, conformal_vector='L')`,
  `V.verma_module(h)`, `M.simple_quotient()`, `M.fusion_product(N)`,
  `P.depth_quotient_up_to(d, max_depth)`, `l0_matrix()`,
  `mode_matrix(a, n)`.
- **Prerequisites:** 10, 11.

## Index of topics

Topics that appear in several notebooks; "03 §6" means
`03_affine_vertex_algebras.ipynb`, section 6.

| topic | where |
|---|---|
| characters and weight spaces | 01 §5; 04 §2; 05 §2; 06 §§4–5 |
| singular vectors | 03 §6; 06 §5; 10 §4; 13 |
| minimal models | 06 §5; 09 §§6, 10; 11 §§1, 3 |
| superconformal algebras | 07 §3; 08 §6; 09 §§5, 10; 12 §5 |
| tensor products | 04 §9; 05 §7; 07 §3; 09 §9 |
| morphisms of vertex algebras | 04 §11; 05 §§4, 6, 7; 10 §9 |
| presentations and Jacobi identities | 06 §6; 12; 15 §2 |
| gradings by abelian groups | 03 §5; 04 §2; 10 §6; 13 |
| Zhu algebras | 08 §8; 09; 10 §3; 15 §§3, 8 |
| twisted and higher Zhu algebras | 08 §8; 09 §§10–11; 10 §3; 15 §8 |
| Verma and induced modules | 08 §9; 10 §§4, 7; 16 |
| fusion products | 11; 15 §9; 16 |
| projective covers, $\operatorname{Ext}^1$, Loewy layers | 11 §§5–6; 15 §§6, 8, 10 |
| logarithmic modules (Jordan blocks of $L_0$) | 11 §§4, 6; 15; 16 |
| unique representation, pickling, saving | 02 §9; 04 §10; 05 §8; 08 §10; 12 §7; 14 §§6–7 |
