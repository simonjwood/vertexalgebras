# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) Cardiff University 2026
# Shared part of the Cython wrappers of the C++ vertex algebra engines
# (_engine_base.hpp), textually included by each engine's .pyx
# (`include "_engine_base.pxi"`): boundary types, conversion of
# integers and coefficients (Python ints and tuples only, no Sage), the
# translation of C++ exceptions (`except +_raise_error`) and Ctrl-C polling.
# A coefficient is a list of terms (exponents, numerator, denominator).
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libc.stdint cimport uint32_t
from cpython.exc cimport PyErr_Occurred

cdef extern from "Python.h":
    # declared here without an exception value: the pending exception is
    # left set and reported by _raise_error after the engine unwinds
    int _check_signals "PyErr_CheckSignals" ()

cdef extern from "flint/flint.h":
    ctypedef long slong

cdef extern from "_engine_base.hpp" namespace "vae":
    ctypedef int (*PollFn)() noexcept
    cdef cppclass Num:
        slong small
        string hex
    cdef cppclass Term:
        vector[slong] exps
        Num num
        Num den
    ctypedef vector[Term] CoefData
    ctypedef vector[pair[uint32_t, CoefData]] ElemData
    ctypedef vector[ElemData] PolesData
    int current_exception_kind()
    const char *current_exception_message()


cdef int _raise_error() except -1:
    """Translate the C++ exception being handled into a Python one."""
    if PyErr_Occurred() != NULL:       # e.g. KeyboardInterrupt set by polling
        return -1
    cdef int k = current_exception_kind()
    msg = current_exception_message().decode('utf-8', 'replace')
    if k == 1:
        raise ZeroDivisionError(msg)
    if k == 2:
        raise KeyboardInterrupt
    if k == 3:
        raise MemoryError(msg)
    if k == 4:
        raise ValueError(msg)
    raise RuntimeError(msg)


cdef int _poll() noexcept with gil:
    return _check_signals() != 0


cdef inline int _num_in(Num &n, object x) except -1:
    try:
        n.small = x
        n.hex.clear()
    except OverflowError:
        n.small = 0
        n.hex = format(x, 'x').encode()
    return 0


cdef inline object _num_out(const Num &n):
    if n.hex.empty():
        return n.small
    return int(n.hex.decode(), 16)


cdef CoefData _coef_in(object terms) except *:
    cdef CoefData d
    cdef Term t
    for exps, num, den in terms:
        t.exps.clear()
        for x in exps:
            t.exps.push_back(x)
        _num_in(t.num, num)
        _num_in(t.den, den)
        d.push_back(t)
    return d


cdef list _coef_out(const CoefData &d):
    cdef list out = []
    cdef size_t i, j
    for i in range(d.size()):
        out.append((tuple([d[i].exps[j] for j in range(d[i].exps.size())]),
                    _num_out(d[i].num), _num_out(d[i].den)))
    return out
