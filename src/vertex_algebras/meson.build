# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
py.install_sources(
  '__init__.py',
  'affine_vertex_algebra.py',
  'bootstrap.py',
  'catalog.py',
  'commutant_vertex_algebra.py',
  'ghost_systems.py',
  '_grading_groups.py',
  'lattice_vertex_algebra.py',
  'lattice_vertex_algebra_module.py',
  'lie_conformal.py',
  'lie_superalgebras.py',
  'normally_ordered_expressions.py',
  'quantum_hamiltonian_reduction.py',
  'quantum_hamiltonian_reduction_modules.py',
  'presented_vertex_algebra.py',
  'quotient_induced_module.py',
  'quotient_vertex_algebra.py',
  '_keyed_spans.py',
  'tensor_product_vertex_algebra.py',
  'universal_vertex_algebra.py',
  'vertex_algebra.py',
  'vertex_algebra_categories.py',
  'vertex_algebra_module.py',
  'vertex_algebra_submodule.py',
  'vertex_ideal.py',
  'vertex_subalgebra.py',
  'weight_spaces.py',
  'zhu_algebra.py',
  'higher_zhu_algebra.py',
  '_finite_dimensional_structure.py',
  'fusion_product.py',
  'vertex_algebra_morphism.py',
  'vertex_algebra_module_morphism.py',
  'vertex_algebra_module_ext.py',
  'module_twists.py',
  'zhu_algebra_module.py',
  subdir: 'vertex_algebras',
)

# The C++ engines: Cython wrappers (sharing _engine_base.pxi) of headers in
# this directory (_engine_base.hpp and one per engine).
foreach name : ['_universal_engine', '_lattice_engine', '_tensor_engine',
                '_linear_algebra_engine', '_module_engine']
  py.extension_module(name, name + '.pyx',
    override_options: ['cython_language=cpp'],
    include_directories: include_directories('.'),
    dependencies: [flint, gmp, threads],
    install: true,
    subdir: 'vertex_algebras',
  )
endforeach
