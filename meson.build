# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
project('sage-vertex-algebras', 'c', 'cpp', 'cython',
  version: '0.1.0a5',
  meson_version: '>= 1.1',
  default_options: ['cpp_std=c++17'],
)

py = import('python').find_installation(pure: false)

# The C++ engines compute with FLINT (>= 3.0: fmpz_mpoly_q and nf_elem) and
# GMP, and run threads.
flint = dependency('flint', version: '>= 3.0')
gmp = dependency('gmp')
threads = dependency('threads')

subdir('src/vertex_algebras')
