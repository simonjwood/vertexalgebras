// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// The parts of the C++ engines of vertex algebras that do not depend on the
// algebra: coefficient rings (FLINT), boundary data, a memo shared by
// threads, interned monomials, interrupt / error / allocator control, and
// EngineCore (element arithmetic, the scheduler: work stealing and run
// tokens, top-level sums, conversions).  An algebra's engine derives from
// EngineCore and supplies the recursion: _universal_engine.hpp.
// No Python objects in here.
//
// Coefficient rings (one engine template per ring):
//   RingQ      QQ                    fmpq
//   RingQPoly  QQ[x]                 fmpq_poly
//   RingQMPoly QQ[x1, ..., xn]       fmpq_mpoly (own context per engine)
//   RingNmod   GF(p), p a word prime ulong + nmod_t
// Coefficients cross the boundary as terms (exponents, numerator,
// denominator) with integers as slong or signed hex strings.
//
// Threads: a call with nthreads > 1 first computes the distinct key-level
// jobs of the call ("roots") in worker threads, then sums them (large sums
// split among threads by output monomial); results do not depend on the
// number of threads.  The caches are sharded hash maps in which every entry
// is computed once; a thread needing an entry that another thread is
// computing waits for it (no deadlock: the dependencies are acyclic and a
// thread only waits from its own chain of dependencies).  Inside one job,
// work stealing: at the recursion's fan-outs a worker pushes the children it
// is about to compute (while some worker is idle), idle workers steal the
// oldest pushed job and run it if nobody has started it.  nthreads run
// tokens: a worker blocked on an entry lends its token to one of the spare
// workers.  While workers run, GMP's allocator is plain malloc (Sage
// installs cysignals' sig_malloc, whose block counter is not thread-safe)
// and the calling thread polls a callback (Ctrl-C).  With nthreads <= 1
// everything runs in the calling thread, which polls every POLL_EVERY cache
// misses.  A computation that throws (interrupt, division by zero mod p)
// leaves no half-built cache entry behind.
#pragma once
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpq.h>
#include <flint/fmpq_poly.h>
#include <flint/fmpq_mpoly.h>
#include <flint/nf.h>
#include <flint/nf_elem.h>
#include <flint/nmod.h>
#include <flint/ulong_extras.h>
#include <gmp.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace vae {

// ------------------------------------------------------------------------ //
//  Errors                                                                  //
// ------------------------------------------------------------------------ //

struct Interrupted : std::runtime_error {
    Interrupted() : std::runtime_error("interrupted") {}
};
struct DivisionByZero : std::runtime_error {
    explicit DivisionByZero(const std::string &s) : std::runtime_error(s) {}
};
// an entry this thread waited for failed in the thread computing it
struct EntryFailed : std::runtime_error {
    EntryFailed() : std::runtime_error("a sub-result failed in another thread") {}
};

// For the wrapper: classify the exception being handled (call inside a
// catch block).  0 other, 1 division by zero, 2 interrupted, 3 out of memory,
// 4 invalid argument.
inline thread_local std::string tl_error_message;
inline int current_exception_kind() {
    try {
        throw;
    } catch (const DivisionByZero &e) {
        tl_error_message = e.what();
        return 1;
    } catch (const Interrupted &e) {
        tl_error_message = e.what();
        return 2;
    } catch (const std::bad_alloc &e) {
        tl_error_message = e.what();
        return 3;
    } catch (const std::invalid_argument &e) {
        tl_error_message = e.what();
        return 4;
    } catch (const std::exception &e) {
        tl_error_message = e.what();
        return 0;
    } catch (...) {
        tl_error_message = "unknown C++ exception";
        return 0;
    }
}
inline const char *current_exception_message() { return tl_error_message.c_str(); }

// ------------------------------------------------------------------------ //
//  Integers and boundary data                                              //
// ------------------------------------------------------------------------ //

// exact integers (binomials, rising factorials, factorials exceed 64 bits)
struct Z {
    fmpz_t v;
    Z() { fmpz_init(v); }
    explicit Z(slong s) { fmpz_init_set_si(v, s); }
    ~Z() { fmpz_clear(v); }
    Z(const Z &) = delete;
    Z &operator=(const Z &) = delete;
};
inline void comb(fmpz_t r, ulong n, ulong k) { fmpz_bin_uiui(r, n, k); }
inline void rising(fmpz_t r, ulong k, ulong i) { fmpz_rfac_uiui(r, k, i); }  // k(k+1)...(k+i-1)
inline void fact(fmpz_t r, ulong l) { fmpz_fac_ui(r, l); }

// an integer at the boundary: small, or (if hex is not empty) signed hex
struct Num {
    slong small = 0;
    std::string hex;
};
inline void num_get(fmpz_t z, const Num &n) {
    if (n.hex.empty()) {
        fmpz_set_si(z, n.small);
    } else if (fmpz_set_str(z, n.hex.c_str(), 16) != 0) {
        throw std::invalid_argument("bad integer " + n.hex);
    }
}
inline Num num_of(const fmpz_t z) {
    Num n;
    if (fmpz_fits_si(z)) {
        n.small = fmpz_get_si(z);
    } else {
        char *s = fmpz_get_str(nullptr, 16, z);
        n.hex = s;
        flint_free(s);
    }
    return n;
}
// one term of a coefficient: exponents (none for QQ and GF(p)), num / den
struct Term {
    std::vector<slong> exps;
    Num num, den;
};
using CoefData = std::vector<Term>;
using ElemData = std::vector<std::pair<uint32_t, CoefData>>;   // (monomial id, coefficient)
using PolesData = std::vector<ElemData>;

// ------------------------------------------------------------------------ //
//  Coefficient rings                                                       //
// ------------------------------------------------------------------------ //
// Interface: element type E (constructible from the ring, copyable,
// movable), is_zero, one, add (a += b), mul (r = a b), mul_fmpz, div_fmpz,
// mul_si, div_si (exact division by a nonzero integer), mul_fmpq (r = a q),
// from_data, to_data.  Elements must be canonical (is_zero exact) and the
// ring must not change after construction (threads share it).

// a univariate coefficient (terms ((k,), num, den)) as an fmpq_poly
inline void qpoly_from_data(fmpq_poly_t a, const CoefData &d) {
    fmpq_poly_zero(a);
    Z n, m;
    fmpq_t t, u;
    fmpq_init(t);
    fmpq_init(u);
    for (const Term &x : d) {
        if (x.exps.size() != 1 || x.exps[0] < 0) {
            fmpq_clear(t);
            fmpq_clear(u);
            throw std::invalid_argument("bad exponent");
        }
        num_get(n.v, x.num);
        num_get(m.v, x.den);
        fmpq_set_fmpz_frac(t, n.v, m.v);
        fmpq_poly_get_coeff_fmpq(u, a, x.exps[0]);
        fmpq_add(t, t, u);
        fmpq_poly_set_coeff_fmpq(a, x.exps[0], t);
    }
    fmpq_clear(t);
    fmpq_clear(u);
}
inline CoefData qpoly_to_data(const fmpq_poly_t a) {
    CoefData d;
    fmpq_t t;
    fmpq_init(t);
    for (slong i = 0; i < fmpq_poly_length(a); i++) {
        fmpq_poly_get_coeff_fmpq(t, a, i);
        if (!fmpq_is_zero(t)) d.push_back(Term{{i}, num_of(fmpq_numref(t)), num_of(fmpq_denref(t))});
    }
    fmpq_clear(t);
    return d;
}

struct RingQ {
    struct E {
        fmpq_t p;
        explicit E(const RingQ &) { fmpq_init(p); }
        E(const E &o) { fmpq_init(p); fmpq_set(p, o.p); }
        E(E &&o) noexcept { fmpq_init(p); fmpq_swap(p, o.p); }
        E &operator=(const E &o) { if (this != &o) fmpq_set(p, o.p); return *this; }
        E &operator=(E &&o) noexcept { fmpq_swap(p, o.p); return *this; }
        ~E() { fmpq_clear(p); }
    };
    bool is_zero(const E &a) const { return fmpq_is_zero(a.p); }
    void one(E &a) const { fmpq_one(a.p); }
    void add(E &a, const E &b) const { fmpq_add(a.p, a.p, b.p); }
    void mul(E &r, const E &a, const E &b) const { fmpq_mul(r.p, a.p, b.p); }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { fmpq_mul_fmpz(r.p, a.p, z); }
    void div_fmpz(E &a, const fmpz_t z) const { fmpq_div_fmpz(a.p, a.p, z); }
    void mul_si(E &r, const E &a, slong s) const { fmpq_mul_si(r.p, a.p, s); }
    void div_si(E &a, slong s) const { Z z(s); fmpq_div_fmpz(a.p, a.p, z.v); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const { fmpq_mul(r.p, a.p, q); }
    void from_data(E &a, const CoefData &d) const {
        fmpq_zero(a.p);
        Z n, m;
        fmpq_t t;
        fmpq_init(t);
        for (const Term &x : d) {
            num_get(n.v, x.num);
            num_get(m.v, x.den);
            fmpq_set_fmpz_frac(t, n.v, m.v);
            fmpq_add(a.p, a.p, t);
        }
        fmpq_clear(t);
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        if (!fmpq_is_zero(a.p)) d.push_back(Term{{}, num_of(fmpq_numref(a.p)), num_of(fmpq_denref(a.p))});
        return d;
    }
};

struct RingQPoly {
    struct E {
        fmpq_poly_t p;
        explicit E(const RingQPoly &) { fmpq_poly_init(p); }
        E(const E &o) { fmpq_poly_init(p); fmpq_poly_set(p, o.p); }
        E(E &&o) noexcept { fmpq_poly_init(p); fmpq_poly_swap(p, o.p); }
        E &operator=(const E &o) { if (this != &o) fmpq_poly_set(p, o.p); return *this; }
        E &operator=(E &&o) noexcept { fmpq_poly_swap(p, o.p); return *this; }
        ~E() { fmpq_poly_clear(p); }
    };
    bool is_zero(const E &a) const { return fmpq_poly_is_zero(a.p); }
    void one(E &a) const { fmpq_poly_one(a.p); }
    void add(E &a, const E &b) const { fmpq_poly_add(a.p, a.p, b.p); }
    void mul(E &r, const E &a, const E &b) const { fmpq_poly_mul(r.p, a.p, b.p); }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { fmpq_poly_scalar_mul_fmpz(r.p, a.p, z); }
    void div_fmpz(E &a, const fmpz_t z) const { fmpq_poly_scalar_div_fmpz(a.p, a.p, z); }
    void mul_si(E &r, const E &a, slong s) const { fmpq_poly_scalar_mul_si(r.p, a.p, s); }
    void div_si(E &a, slong s) const { fmpq_poly_scalar_div_si(a.p, a.p, s); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const { fmpq_poly_scalar_mul_fmpq(r.p, a.p, q); }
    void from_data(E &a, const CoefData &d) const { qpoly_from_data(a.p, d); }
    CoefData to_data(const E &a) const { return qpoly_to_data(a.p); }
};

struct RingQMPoly {
    fmpq_mpoly_ctx_t ctx;
    slong nvars;
    explicit RingQMPoly(slong n) : nvars(n) { fmpq_mpoly_ctx_init(ctx, n, ORD_LEX); }
    ~RingQMPoly() { fmpq_mpoly_ctx_clear(ctx); }
    RingQMPoly(const RingQMPoly &) = delete;
    RingQMPoly &operator=(const RingQMPoly &) = delete;
    // elements keep their context (FLINT wants it for init and clear)
    struct E {
        fmpq_mpoly_t p;
        const fmpq_mpoly_ctx_struct *c;
        explicit E(const RingQMPoly &R) : c(R.ctx) { fmpq_mpoly_init(p, c); }
        E(const E &o) : c(o.c) { fmpq_mpoly_init(p, c); fmpq_mpoly_set(p, o.p, c); }
        E(E &&o) noexcept : c(o.c) { fmpq_mpoly_init(p, c); fmpq_mpoly_swap(p, o.p, c); }
        E &operator=(const E &o) { if (this != &o) fmpq_mpoly_set(p, o.p, c); return *this; }
        E &operator=(E &&o) noexcept { fmpq_mpoly_swap(p, o.p, c); return *this; }
        ~E() { fmpq_mpoly_clear(p, c); }
    };
    bool is_zero(const E &a) const { return fmpq_mpoly_is_zero(a.p, ctx); }
    void one(E &a) const { fmpq_mpoly_one(a.p, ctx); }
    void add(E &a, const E &b) const { fmpq_mpoly_add(a.p, a.p, b.p, ctx); }
    void mul(E &r, const E &a, const E &b) const { fmpq_mpoly_mul(r.p, a.p, b.p, ctx); }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { fmpq_mpoly_scalar_mul_fmpz(r.p, a.p, z, ctx); }
    void div_fmpz(E &a, const fmpz_t z) const { fmpq_mpoly_scalar_div_fmpz(a.p, a.p, z, ctx); }
    void mul_si(E &r, const E &a, slong s) const { fmpq_mpoly_scalar_mul_si(r.p, a.p, s, ctx); }
    void div_si(E &a, slong s) const { fmpq_mpoly_scalar_div_si(a.p, a.p, s, ctx); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const { fmpq_mpoly_scalar_mul_fmpq(r.p, a.p, q, ctx); }
    void from_data(E &a, const CoefData &d) const {
        fmpq_mpoly_zero(a.p, ctx);
        Z n, m;
        fmpq_t t;
        fmpq_init(t);
        E term(*this);
        std::vector<ulong> e((size_t)nvars);
        for (const Term &x : d) {
            if ((slong)x.exps.size() != nvars) throw std::invalid_argument("bad exponent vector");
            for (slong j = 0; j < nvars; j++) {
                if (x.exps[j] < 0) throw std::invalid_argument("negative exponent");
                e[j] = (ulong)x.exps[j];
            }
            num_get(n.v, x.num);
            num_get(m.v, x.den);
            fmpq_set_fmpz_frac(t, n.v, m.v);
            fmpq_mpoly_zero(term.p, ctx);
            fmpq_mpoly_set_coeff_fmpq_ui(term.p, t, e.data(), ctx);
            fmpq_mpoly_add(a.p, a.p, term.p, ctx);
        }
        fmpq_clear(t);
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        fmpq_t t;
        fmpq_init(t);
        std::vector<ulong> e((size_t)nvars);
        for (slong i = 0; i < fmpq_mpoly_length(a.p, ctx); i++) {
            fmpq_mpoly_get_term_coeff_fmpq(t, a.p, i, ctx);
            fmpq_mpoly_get_term_exp_ui(e.data(), a.p, i, ctx);
            Term x;
            x.exps.assign(e.begin(), e.end());
            x.num = num_of(fmpq_numref(t));
            x.den = num_of(fmpq_denref(t));
            d.push_back(std::move(x));
        }
        fmpq_clear(t);
        return d;
    }
};

struct RingNmod {
    nmod_t mod;
    explicit RingNmod(ulong p) { nmod_init(&mod, p); }
    struct E {
        ulong v = 0;
        explicit E(const RingNmod &) {}
    };
    ulong red(const fmpz_t z) const { return fmpz_fdiv_ui(z, mod.n); }
    ulong red_si(slong s) const {        // mod.n < 2^63
        slong m = (slong)mod.n, r = s % m;
        return (ulong)(r < 0 ? r + m : r);
    }
    ulong inv(ulong x) const {
        if (x == 0) throw DivisionByZero("division by a multiple of the characteristic " +
                                         std::to_string(mod.n));
        return n_invmod(x, mod.n);
    }
    bool is_zero(const E &a) const { return a.v == 0; }
    void one(E &a) const { a.v = 1 % mod.n; }
    void add(E &a, const E &b) const { a.v = nmod_add(a.v, b.v, mod); }
    void mul(E &r, const E &a, const E &b) const { r.v = nmod_mul(a.v, b.v, mod); }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { r.v = nmod_mul(a.v, red(z), mod); }
    void div_fmpz(E &a, const fmpz_t z) const { a.v = nmod_mul(a.v, inv(red(z)), mod); }
    void mul_si(E &r, const E &a, slong s) const { r.v = nmod_mul(a.v, red_si(s), mod); }
    void div_si(E &a, slong s) const { a.v = nmod_mul(a.v, inv(red_si(s)), mod); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const {
        r.v = nmod_mul(a.v, nmod_mul(red(fmpq_numref(q)), inv(red(fmpq_denref(q))), mod), mod);
    }
    void from_data(E &a, const CoefData &d) const {
        ulong acc = 0;
        Z n, m;
        for (const Term &x : d) {
            num_get(n.v, x.num);
            num_get(m.v, x.den);
            acc = nmod_add(acc, nmod_mul(red(n.v), inv(red(m.v)), mod), mod);
        }
        a.v = acc;
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        if (a.v) {
            Term x;
            if (a.v <= (ulong)WORD_MAX) {
                x.num.small = (slong)a.v;
            } else {
                Z z;
                fmpz_set_ui(z.v, a.v);
                x.num = num_of(z.v);
            }
            x.den.small = 1;
            d.push_back(std::move(x));
        }
        return d;
    }
};

// A number field QQ[t]/(m(t)) (FLINT nf_elem), m irreducible over QQ (the
// defining polynomial of the field, any leading coefficient), passed in
// qpoly format.  Elements cross the boundary as polynomials in t of degree
// < deg m: coordinates in the power basis of the generator t.
struct RingNF {
    nf_t nf;
    explicit RingNF(const CoefData &mp) {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        qpoly_from_data(p, mp);
        if (fmpq_poly_degree(p) < 1) {
            fmpq_poly_clear(p);
            throw std::invalid_argument("number field: defining polynomial of degree < 1");
        }
        nf_init(nf, p);
        fmpq_poly_clear(p);
    }
    ~RingNF() { nf_clear(nf); }
    RingNF(const RingNF &) = delete;
    RingNF &operator=(const RingNF &) = delete;
    // elements keep their field (FLINT wants it for init and clear)
    struct E {
        nf_elem_t p;
        const nf_struct *k;
        explicit E(const RingNF &R) : k(R.nf) { nf_elem_init(p, k); }
        E(const E &o) : k(o.k) { nf_elem_init(p, k); nf_elem_set(p, o.p, k); }
        E(E &&o) noexcept : k(o.k) { nf_elem_init(p, k); nf_elem_swap(p, o.p, k); }
        E &operator=(const E &o) { if (this != &o) nf_elem_set(p, o.p, k); return *this; }
        E &operator=(E &&o) noexcept { nf_elem_swap(p, o.p, k); return *this; }
        ~E() { nf_elem_clear(p, k); }
    };
    bool is_zero(const E &a) const { return nf_elem_is_zero(a.p, nf); }
    void one(E &a) const { nf_elem_one(a.p, nf); }
    void add(E &a, const E &b) const { nf_elem_add(a.p, a.p, b.p, nf); }
    void mul(E &r, const E &a, const E &b) const {
        if (&r == &a || &r == &b) {
            E t(*this);
            nf_elem_mul(t.p, a.p, b.p, nf);
            nf_elem_swap(r.p, t.p, nf);
        } else {
            nf_elem_mul(r.p, a.p, b.p, nf);
        }
    }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { nf_elem_scalar_mul_fmpz(r.p, a.p, z, nf); }
    void div_fmpz(E &a, const fmpz_t z) const { nf_elem_scalar_div_fmpz(a.p, a.p, z, nf); }
    void mul_si(E &r, const E &a, slong s) const { nf_elem_scalar_mul_si(r.p, a.p, s, nf); }
    void div_si(E &a, slong s) const { nf_elem_scalar_div_si(a.p, a.p, s, nf); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const { nf_elem_scalar_mul_fmpq(r.p, a.p, q, nf); }
    void from_data(E &a, const CoefData &d) const {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        try {
            qpoly_from_data(p, d);
        } catch (...) {
            fmpq_poly_clear(p);
            throw;
        }
        nf_elem_set_fmpq_poly(a.p, p, nf);
        fmpq_poly_clear(p);
    }
    CoefData to_data(const E &a) const {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        nf_elem_get_fmpq_poly(p, a.p, nf);
        CoefData d = qpoly_to_data(p);
        fmpq_poly_clear(p);
        return d;
    }
};

// Polynomials in nvars variables over a number field QQ[t]/(m(t)):
// fmpq_mpoly in (x_1, ..., x_nvars, t), every product reduced modulo the
// monic m(t) (division by m: no term of t-degree >= deg m remains), so
// elements are in normal form (canonical, exact zero test; the other
// operations keep it).  Exponent vectors at the boundary: the x's, then t.
struct RingNFMPoly {
    fmpq_mpoly_ctx_t ctx;
    slong nvars;
    slong deg;
    fmpq_mpoly_t m;
    RingNFMPoly(slong n, const CoefData &mp) : nvars(n) {
        if (n < 1) throw std::invalid_argument("nfmpoly needs at least one variable");
        fmpq_poly_t p;
        fmpq_poly_init(p);
        try {
            qpoly_from_data(p, mp);
        } catch (...) {
            fmpq_poly_clear(p);
            throw;
        }
        deg = fmpq_poly_degree(p);
        if (deg < 1) {
            fmpq_poly_clear(p);
            throw std::invalid_argument("number field: defining polynomial of degree < 1");
        }
        fmpq_poly_make_monic(p, p);
        fmpq_mpoly_ctx_init(ctx, n + 1, ORD_LEX);
        fmpq_mpoly_init(m, ctx);
        std::vector<ulong> e((size_t)n + 1, 0);
        fmpq_t c;
        fmpq_init(c);
        for (slong k = 0; k <= deg; k++) {
            fmpq_poly_get_coeff_fmpq(c, p, k);
            e[(size_t)n] = (ulong)k;
            if (!fmpq_is_zero(c)) fmpq_mpoly_set_coeff_fmpq_ui(m, c, e.data(), ctx);
        }
        fmpq_clear(c);
        fmpq_poly_clear(p);
    }
    ~RingNFMPoly() {
        fmpq_mpoly_clear(m, ctx);
        fmpq_mpoly_ctx_clear(ctx);
    }
    RingNFMPoly(const RingNFMPoly &) = delete;
    RingNFMPoly &operator=(const RingNFMPoly &) = delete;
    struct E {
        fmpq_mpoly_t p;
        const fmpq_mpoly_ctx_struct *c;
        explicit E(const RingNFMPoly &R) : c(R.ctx) { fmpq_mpoly_init(p, c); }
        E(const E &o) : c(o.c) { fmpq_mpoly_init(p, c); fmpq_mpoly_set(p, o.p, c); }
        E(E &&o) noexcept : c(o.c) { fmpq_mpoly_init(p, c); fmpq_mpoly_swap(p, o.p, c); }
        E &operator=(const E &o) { if (this != &o) fmpq_mpoly_set(p, o.p, c); return *this; }
        E &operator=(E &&o) noexcept { fmpq_mpoly_swap(p, o.p, c); return *this; }
        ~E() { fmpq_mpoly_clear(p, c); }
    };
    // normal form modulo m(t)
    void reduce(fmpq_mpoly_t a) const {
        if (fmpq_mpoly_degree_si(a, nvars, ctx) < deg) return;
        fmpq_mpoly_t q, r;
        fmpq_mpoly_init(q, ctx);
        fmpq_mpoly_init(r, ctx);
        fmpq_mpoly_divrem(q, r, a, m, ctx);
        fmpq_mpoly_swap(a, r, ctx);
        fmpq_mpoly_clear(q, ctx);
        fmpq_mpoly_clear(r, ctx);
    }
    bool is_zero(const E &a) const { return fmpq_mpoly_is_zero(a.p, ctx); }
    void one(E &a) const { fmpq_mpoly_one(a.p, ctx); }
    void add(E &a, const E &b) const { fmpq_mpoly_add(a.p, a.p, b.p, ctx); }
    void mul(E &r, const E &a, const E &b) const {
        fmpq_mpoly_mul(r.p, a.p, b.p, ctx);
        reduce(r.p);
    }
    void mul_fmpz(E &r, const E &a, const fmpz_t z) const { fmpq_mpoly_scalar_mul_fmpz(r.p, a.p, z, ctx); }
    void div_fmpz(E &a, const fmpz_t z) const { fmpq_mpoly_scalar_div_fmpz(a.p, a.p, z, ctx); }
    void mul_si(E &r, const E &a, slong s) const { fmpq_mpoly_scalar_mul_si(r.p, a.p, s, ctx); }
    void div_si(E &a, slong s) const { fmpq_mpoly_scalar_div_si(a.p, a.p, s, ctx); }
    void mul_fmpq(E &r, const E &a, const fmpq_t q) const { fmpq_mpoly_scalar_mul_fmpq(r.p, a.p, q, ctx); }
    void from_data(E &a, const CoefData &d) const {
        fmpq_mpoly_zero(a.p, ctx);
        Z n, mm;
        fmpq_t t;
        fmpq_init(t);
        E term(*this);
        std::vector<ulong> e((size_t)nvars + 1);
        for (const Term &x : d) {
            if ((slong)x.exps.size() != nvars + 1) {
                fmpq_clear(t);
                throw std::invalid_argument("bad exponent vector");
            }
            for (slong j = 0; j <= nvars; j++) {
                if (x.exps[(size_t)j] < 0) {
                    fmpq_clear(t);
                    throw std::invalid_argument("negative exponent");
                }
                e[(size_t)j] = (ulong)x.exps[(size_t)j];
            }
            num_get(n.v, x.num);
            num_get(mm.v, x.den);
            fmpq_set_fmpz_frac(t, n.v, mm.v);
            fmpq_mpoly_zero(term.p, ctx);
            fmpq_mpoly_set_coeff_fmpq_ui(term.p, t, e.data(), ctx);
            fmpq_mpoly_add(a.p, a.p, term.p, ctx);
        }
        fmpq_clear(t);
        reduce(a.p);
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        fmpq_t t;
        fmpq_init(t);
        std::vector<ulong> e((size_t)nvars + 1);
        for (slong i = 0; i < fmpq_mpoly_length(a.p, ctx); i++) {
            fmpq_mpoly_get_term_coeff_fmpq(t, a.p, i, ctx);
            fmpq_mpoly_get_term_exp_ui(e.data(), a.p, i, ctx);
            Term x;
            x.exps.assign(e.begin(), e.end());
            x.num = num_of(fmpq_numref(t));
            x.den = num_of(fmpq_denref(t));
            d.push_back(std::move(x));
        }
        fmpq_clear(t);
        return d;
    }
};

// ------------------------------------------------------------------------ //
//  Shared memo and interned monomials                                      //
// ------------------------------------------------------------------------ //

struct VecHash {
    size_t operator()(const std::vector<uint32_t> &v) const {
        uint64_t h = 1469598103934665603ULL;
        for (uint32_t x : v) { h ^= x; h *= 1099511628211ULL; }
        return (size_t)h;
    }
};

inline uint64_t pk(uint32_t a, uint32_t b) { return ((uint64_t)a << 32) | b; }

inline thread_local int tl_wid = -1;            // worker index in a threaded call, else -1

// A memo shared by threads: get(key, compute) returns the value, computing it
// once.  Ready values never move or disappear (entries are heap nodes kept
// by the map), so references stay valid.  If compute throws, the entry is
// removed (a later call recomputes it) and threads waiting for it throw
// EntryFailed; a thread asking for an entry it is computing itself throws
// (a cycle; cannot happen for consistent data).  get_if_absent computes only
// if no thread has started the entry (for stolen jobs: never wait).
template <class V>
struct ConcCache {
    enum : int { COMPUTING = 0, READY = 1, FAILED = 2 };
    struct Entry {
        V val;
        std::atomic<int> state{COMPUTING};
        std::thread::id owner;
    };
    struct Shard {
        std::mutex m;
        std::condition_variable cv;
        std::unordered_map<uint64_t, std::shared_ptr<Entry>> map;
    };
    static constexpr size_t NS = 64;
    std::unique_ptr<Shard[]> shards{new Shard[NS]};
    std::atomic<uint64_t> *wait_count = nullptr, *wait_ns = nullptr;   // the engine's statistics
    // called (shard unlocked) before a pool worker blocks (phase 0) and after
    // (phase 1): the engine lends the worker's run token meanwhile
    void (*block_hook)(void *, int) = nullptr;
    void *hook_ctx = nullptr;
    static size_t shard_of(uint64_t k) { return (size_t)((k * 0x9E3779B97F4A7C15ULL) >> 58); }

    template <class F>
    const V &get(uint64_t key, F &&compute) {
        Shard &s = shards[shard_of(key)];
        std::unique_lock<std::mutex> lk(s.m);
        auto it = s.map.find(key);
        if (it != s.map.end()) {
            std::shared_ptr<Entry> e = it->second;
            if (e->state.load(std::memory_order_acquire) == READY) return e->val;
            if (e->owner == std::this_thread::get_id())
                throw std::logic_error("cyclic dependency in the OPE recursion");
            auto t0 = std::chrono::steady_clock::now();
            bool hooked = block_hook && tl_wid >= 0;
            if (hooked) {
                lk.unlock();
                block_hook(hook_ctx, 0);
                lk.lock();
            }
            s.cv.wait(lk, [&e] { return e->state.load(std::memory_order_acquire) != COMPUTING; });
            lk.unlock();
            if (hooked) block_hook(hook_ctx, 1);
            if (wait_count) {
                wait_count->fetch_add(1, std::memory_order_relaxed);
                wait_ns->fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - t0).count(),
                                   std::memory_order_relaxed);
            }
            if (e->state.load(std::memory_order_acquire) == FAILED) throw EntryFailed();
            return e->val;
        }
        return compute_new(s, lk, key, compute);
    }
    template <class F>
    bool get_if_absent(uint64_t key, F &&compute) {
        Shard &s = shards[shard_of(key)];
        std::unique_lock<std::mutex> lk(s.m);
        if (s.map.find(key) != s.map.end()) return false;
        compute_new(s, lk, key, compute);
        return true;
    }
    // with s.m held by lk and key absent
    template <class F>
    const V &compute_new(Shard &s, std::unique_lock<std::mutex> &lk, uint64_t key, F &&compute) {
        std::shared_ptr<Entry> e = std::make_shared<Entry>();
        e->owner = std::this_thread::get_id();
        s.map.emplace(key, e);
        lk.unlock();
        try {
            V v = compute();
            e->val = std::move(v);
        } catch (...) {
            lk.lock();
            e->state.store(FAILED, std::memory_order_release);
            s.map.erase(key);
            lk.unlock();
            s.cv.notify_all();
            throw;
        }
        lk.lock();
        e->state.store(READY, std::memory_order_release);
        lk.unlock();
        s.cv.notify_all();
        return e->val;
    }
    size_t size() const {
        size_t n = 0;
        for (size_t i = 0; i < NS; i++) n += shards[i].map.size();
        return n;
    }
    bool ready(uint64_t key) {
        Shard &s = shards[shard_of(key)];
        std::lock_guard<std::mutex> lk(s.m);
        auto it = s.map.find(key);
        return it != s.map.end() && it->second->state.load(std::memory_order_acquire) == READY;
    }
};

// Interned monomials (any vectors of uint32 codes, e.g. letters): records in
// fixed chunks (readers need no lock: an id only reaches a thread through a
// synchronised path), interning under a lock; tails and parities are
// computed at intern time.
struct MonoRec {
    std::vector<uint32_t> letters;
    uint32_t tail = 0;         // id of letters[1:]
    int par = 0;               // parity
};
struct MonoTable {
    static constexpr uint32_t CB = 12, CS = 1u << CB, NCH = 1u << 18;   // up to 2^30 monomials
    std::function<int(const std::vector<uint32_t> &)> parity;   // set by the engine
    bool tails = true;         // intern every suffix too (MonoRec::tail)
    MonoRec **chunks;          // calloc'd: untouched pointer pages cost nothing
    uint32_t n = 0;
    std::mutex m;
    std::unordered_map<std::vector<uint32_t>, uint32_t, VecHash> ids;
    MonoTable() {
        chunks = static_cast<MonoRec **>(std::calloc(NCH, sizeof(MonoRec *)));
        if (!chunks) throw std::bad_alloc();
    }
    ~MonoTable() {
        for (uint32_t i = 0; i < NCH && chunks[i]; i++) delete[] chunks[i];
        std::free(chunks);
    }
    MonoTable(const MonoTable &) = delete;
    MonoTable &operator=(const MonoTable &) = delete;
    const MonoRec &operator[](uint32_t id) const { return chunks[id >> CB][id & (CS - 1)]; }
    // with m held
    uint32_t intern_locked(const std::vector<uint32_t> &v) {
        auto it = ids.find(v);
        if (it != ids.end()) return it->second;
        uint32_t tail = 0;
        if (tails && v.size() > 1) tail = intern_locked(std::vector<uint32_t>(v.begin() + 1, v.end()));
        uint32_t id = n;
        if ((id >> CB) >= NCH) throw std::length_error("monomial table full (2^30 monomials)");
        if (!chunks[id >> CB]) chunks[id >> CB] = new MonoRec[CS];
        MonoRec &r = chunks[id >> CB][id & (CS - 1)];
        r.letters = v;
        r.tail = tail;
        r.par = parity ? (parity(v) & 1) : 0;
        ids.emplace(v, id);
        n = id + 1;
        return id;
    }
};

// ------------------------------------------------------------------------ //
//  Control: interrupts and errors across threads                           //
// ------------------------------------------------------------------------ //

using PollFn = int (*)();                  // nonzero: stop (e.g. a pending Ctrl-C)
constexpr unsigned POLL_EVERY = 256;       // cache misses between polls (serial)
constexpr int POLL_MS = 20;                // polling interval of the waiting thread
constexpr size_t MIN_THREADED_JOBS = 8;    // fewer uncached jobs (and all of them small):
constexpr size_t MIN_THREADED_LETTERS = 8; //   computed serially; a job this large: threads
constexpr int IDLE_WAIT_US = 2000;         // idle workers re-check for work at least this often
constexpr size_t PAR_SUM_TERMS = 20000;    // top-level sums with fewer term products: serial
constexpr int SPARE_THREADS_PER_TOKEN = 1; // spare workers take the run token of a blocked one

inline thread_local PollFn tl_poll = nullptr;   // set in the thread that polls while computing
inline thread_local unsigned tl_misses = 0;


struct PollGuard {
    PollFn saved;
    explicit PollGuard(PollFn f) : saved(tl_poll) { tl_poll = f; }
    ~PollGuard() { tl_poll = saved; }
};

// GMP allocates with plain malloc while worker threads run.
//
// Why: Sage sets GMP's memory functions to cysignals' sig_malloc /
// sig_realloc / sig_free (sage/ext/memory.pyx), which wrap malloc in
// sig_block() / sig_unblock(), i.e. ++ / -- of cysigs.block_sigint.
// FLINT integers above 62 bits allocate through GMP, so worker threads
// would update that counter concurrently.  cysignals built with C11
// atomics (CYSIGNALS_C_ATOMIC, e.g. 1.12.6 from conda-forge) makes the
// counter atomic; older or non-atomic builds do not, and a lost update
// leaves Ctrl-C blocked or the counter negative.  Second, on failure
// Sage's functions take the GIL and sig_error() longjmps to the
// thread's sig_on(), which a worker thread does not have (undefined
// behaviour); GMP's defaults abort with "GNU MP: Cannot allocate
// memory" instead.  The swap makes the engine independent of how
// cysignals was built.
//
// Why it is safe: sig_malloc is malloc with the counter around it, so
// blocks allocated by either set of functions may be freed or
// reallocated by the other (GMP objects created before the call and
// freed after it, or the other way round, are fine); the old functions
// are restored when the call returns or throws (RAII); it is only done
// for calls with nthreads > 1, which release the GIL for their whole
// duration.  The one side effect: another Python thread running GMP code
// during such a call allocates without sig_block(), i.e. a Ctrl-C may
// interrupt it inside malloc, as it would without Sage's functions.
struct GmpMallocGuard {
    void *(*fa)(size_t);
    void *(*fr)(void *, size_t, size_t);
    void (*ff)(void *, size_t);
    GmpMallocGuard() {
        mp_get_memory_functions(&fa, &fr, &ff);
        mp_set_memory_functions(nullptr, nullptr, nullptr);
    }
    ~GmpMallocGuard() { mp_set_memory_functions(fa, fr, ff); }
};

// the wrapper releases the GIL for a call with nthreads > 1: then GMP must
// use plain malloc for the WHOLE call (conversions and accumulation too),
// not only while the workers run
struct MaybeGmpMalloc {
    std::unique_ptr<GmpMallocGuard> g;
    explicit MaybeGmpMalloc(int nthreads) { if (nthreads > 1) g.reset(new GmpMallocGuard()); }
};

// the first error of a parallel run, real errors before secondary ones
struct ErrorBox {
    std::mutex m;
    std::exception_ptr e;
    int prio = -1;
    void record(std::exception_ptr x) {
        int p;
        try {
            std::rethrow_exception(x);
        } catch (const EntryFailed &) {
            p = 0;
        } catch (const Interrupted &) {
            p = 1;
        } catch (...) {
            p = 2;
        }
        std::lock_guard<std::mutex> g(m);
        if (p > prio) { prio = p; e = x; }
    }
    void rethrow() { if (e) std::rethrow_exception(e); }
};

// ------------------------------------------------------------------------ //
//  EngineCore: what every algebra's engine shares                          //
// ------------------------------------------------------------------------ //

// The base of an algebra's engine (CRTP: Derived is the engine).  It owns
// the coefficient ring (constructed first, destroyed last), the monomial
// table, element arithmetic, the scheduler (jobs, work stealing, run tokens,
// workers, warm), top-level sums and the boundary conversions.  Derived
// supplies the caches (registered with wire), the recursion, and for jobs
// {kind, a, b} of its own kinds: run_job (compute, waiting if another
// thread computes it), run_job_if_absent (compute only if nobody started
// it), job_cached (ready) and job_size (a size estimate: larger first;
// MIN_THREADED_LETTERS).
template <class Derived, class Ring>
class EngineCore {
  public:
    using C = typename Ring::E;
    using Elem = std::unordered_map<uint32_t, C>;   // monomial id -> coefficient
    using Poles = std::vector<Elem>;                 // entry k = pole k+1
    struct Job {                                     // a key-level job, kinds per engine
        int kind;
        uint32_t a, b;
    };

    // the coefficient ring, possibly shared with other engines (a tensor
    // product's factor engines): first member, outlives every coefficient
    std::shared_ptr<Ring> ring_ptr;
    Ring &R;
    MonoTable monos;
    std::atomic<bool> stop_own{false};
    std::atomic<bool> *stop = &stop_own;   // shared by a tensor product's factor engines

    // work stealing during a threaded call (warm): per-worker deques of
    // pushed jobs, idle workers sleep on icv until epoch changes
    struct WDeque {
        std::mutex m;
        std::deque<Job> dq;
    };
    std::vector<std::unique_ptr<WDeque>> wdq;
    std::atomic<int> idle{0};
    std::mutex im;
    std::condition_variable icv;
    std::atomic<uint64_t> epoch{0};
    std::atomic<bool> finished{false};
    // run tokens: nthreads of them; a worker computes only while holding
    // one, lends it while blocked on an entry another thread computes (a
    // spare worker takes it) and gets one back with priority
    std::mutex tm;
    std::condition_variable tcv;
    std::atomic<int> tokens_free{0};
    int resumers = 0;
    // statistics (cumulative): jobs pushed, stolen, waits for entries
    // computed by another thread and their total time, workers' idle time
    std::atomic<uint64_t> st_push{0}, st_steal{0}, st_wait{0}, st_wait_ns{0}, st_idle_ns{0};


    template <class... A>
    explicit EngineCore(A &&...ring_args)
        : ring_ptr(std::make_shared<Ring>(std::forward<A>(ring_args)...)), R(*ring_ptr) {}
    explicit EngineCore(std::shared_ptr<Ring> ring) : ring_ptr(std::move(ring)), R(*ring_ptr) {}
    EngineCore(const EngineCore &) = delete;
    EngineCore &operator=(const EngineCore &) = delete;
    Derived &derived() { return static_cast<Derived &>(*this); }
    const Derived &derived_c() const { return static_cast<const Derived &>(*this); }
    // the order of output terms (a derived engine may override key_less)
    bool key_less(uint32_t a, uint32_t b) const { return letters_less(a, b); }

    // a cache of the engine: statistics and the run-token hook
    template <class V>
    void wire(ConcCache<V> &c) {
        c.wait_count = &st_wait;
        c.wait_ns = &st_wait_ns;
        c.block_hook = &EngineCore::block_hook;
        c.hook_ctx = this;
    }
    // pushed, stolen, waits, wait ms, idle ms (cumulative)
    std::vector<size_t> sched_stats() const {
        return {(size_t)st_push.load(), (size_t)st_steal.load(), (size_t)st_wait.load(),
                (size_t)(st_wait_ns.load() / 1000000), (size_t)(st_idle_ns.load() / 1000000)};
    }

    // -- run tokens ---------------------------------------------------------- //
    static void block_hook(void *ctx, int phase) {
        EngineCore *E = static_cast<EngineCore *>(ctx);
        if (phase == 0) E->token_lend();
        else E->token_reclaim();
    }
    void token_lend() {             // blocked: a spare worker may run meanwhile
        {
            std::lock_guard<std::mutex> g(tm);
            tokens_free.fetch_add(1);
        }
        tcv.notify_one();
        {
            std::lock_guard<std::mutex> g(im);
            epoch.fetch_add(1, std::memory_order_relaxed);
        }
        icv.notify_one();
    }
    void token_reclaim() {          // unblocked: before the idle workers
        std::unique_lock<std::mutex> lk(tm);
        resumers++;
        tcv.wait(lk, [this] { return tokens_free.load() > 0; });
        tokens_free.fetch_sub(1);
        resumers--;
    }
    bool token_try() {              // an idle worker, only if no blocked one resumes
        std::lock_guard<std::mutex> g(tm);
        if (tokens_free.load() > 0 && resumers == 0) {
            tokens_free.fetch_sub(1);
            return true;
        }
        return false;
    }
    void token_return() {           // found no work: back, quietly
        {
            std::lock_guard<std::mutex> g(tm);
            tokens_free.fetch_add(1);
        }
        tcv.notify_one();
    }

    // Jobs the current worker is about to compute itself (in this order),
    // offered to idle workers: pushed (uncached only) to the bottom of its
    // deque in REVERSE order, so thieves, taking the oldest job from the
    // top, start with the last ones while the owner works from the first.
    // Only while a run token is free and some worker sleeps idle (no
    // overhead when all are busy; inactive in serial runs, tl_wid < 0).  A
    // thief runs a job only if no thread has started it, so the owner never waits for a job nobody
    // runs.  The scope takes its leftovers back when it ends: thieves only
    // remove from the top, nested scopes are balanced, so they are the
    // bottom min(pushed, size) jobs.
    struct Spawn {
        EngineCore &E;
        WDeque *d = nullptr;
        size_t n = 0;
        std::vector<Job> jobs;
        explicit Spawn(EngineCore &e) : E(e) {
            if (tl_wid >= 0 && E.tokens_free.load(std::memory_order_relaxed) > 0 &&
                E.idle.load(std::memory_order_relaxed) > 0)
                d = E.wdq[tl_wid].get();
        }
        Spawn(const Spawn &) = delete;
        Spawn &operator=(const Spawn &) = delete;
        explicit operator bool() const { return d != nullptr; }
        void add(int kind, uint32_t a, uint32_t b) { if (d) jobs.push_back(Job{kind, a, b}); }
        void push() {
            if (!d || jobs.empty()) return;
            std::vector<Job> todo;
            for (auto it = jobs.rbegin(); it != jobs.rend(); ++it)
                if (!E.derived().job_cached(*it)) todo.push_back(*it);
            jobs.clear();
            if (todo.empty()) return;
            {
                std::lock_guard<std::mutex> g(d->m);
                for (const Job &j : todo) d->dq.push_back(j);
            }
            n += todo.size();
            E.st_push.fetch_add(todo.size(), std::memory_order_relaxed);
            E.wake(todo.size());
        }
        ~Spawn() {
            if (!d || !n) return;
            std::lock_guard<std::mutex> g(d->m);
            size_t k = std::min(n, d->dq.size());
            d->dq.erase(d->dq.end() - (std::ptrdiff_t)k, d->dq.end());
        }
    };
    // wake as many idle workers as can run the n new jobs (free tokens)
    void wake(size_t n) {
        int k = std::min((int)std::min(n, (size_t)1 << 20),
                         std::min(idle.load(std::memory_order_relaxed), tokens_free.load()));
        if (k <= 0) return;
        {
            std::lock_guard<std::mutex> g(im);
            epoch.fetch_add(1, std::memory_order_relaxed);
        }
        for (int i = 0; i < k; i++) icv.notify_one();
    }

    // called at every cache miss: stop flag, and polling in a serial run
    void check() {
        if (stop->load(std::memory_order_relaxed)) throw Interrupted();
        if (tl_poll && (++tl_misses % POLL_EVERY) == 0 && tl_poll()) {
            stop->store(true);
            throw Interrupted();
        }
    }

    // -- element arithmetic ------------------------------------------------ //
    C zero() const { return C(R); }
    void acc(Elem &dst, uint32_t k, C &t) const {
        auto it = dst.find(k);
        if (it == dst.end()) {
            if (!R.is_zero(t)) dst.emplace(k, std::move(t));
        } else {
            R.add(it->second, t);
            if (R.is_zero(it->second)) dst.erase(it);
        }
    }
    void add_scaled(Elem &dst, const Elem &src, const C &s) const {
        if (src.empty() || R.is_zero(s)) return;
        C t(R);
        for (const auto &kv : src) { R.mul(t, s, kv.second); acc(dst, kv.first, t); }
    }
    void add_scaled_z(Elem &dst, const Elem &src, const fmpz_t n, const fmpz_t d) const {
        if (src.empty() || fmpz_is_zero(n)) return;
        C t(R);
        for (const auto &kv : src) {
            R.mul_fmpz(t, kv.second, n);
            if (d && !fmpz_is_one(d)) R.div_fmpz(t, d);
            acc(dst, kv.first, t);
        }
    }
    void add_scaled_si(Elem &dst, const Elem &src, slong n, slong d = 1) const {
        if (src.empty() || n == 0) return;
        C t(R);
        for (const auto &kv : src) {
            R.mul_si(t, kv.second, n);
            if (d != 1) R.div_si(t, d);
            acc(dst, kv.first, t);
        }
    }
    Elem single(uint32_t m) const {
        Elem e;
        C one(R);
        R.one(one);
        if (!R.is_zero(one)) e.emplace(m, std::move(one));
        return e;
    }

    // -- monomials --------------------------------------------------------- //
    uint32_t intern_codes(const std::vector<uint32_t> &m) {
        std::lock_guard<std::mutex> g(monos.m);
        return monos.intern_locked(m);
    }

    // -- jobs: warm the caches, serially or in threads ---------------------- //
    // f() in the calling thread, polling every POLL_EVERY cache misses
    template <class F>
    void serial(F &&f, PollFn poll) {
        stop->store(false);
        PollGuard g(poll);
        f();
    }

    // the oldest pushed job of another worker
    bool steal(int self, Job &out) {
        size_t W = wdq.size();
        for (size_t i = 1; i < W; i++) {
            WDeque &d = *wdq[((size_t)self + i) % W];
            std::lock_guard<std::mutex> g(d.m);
            if (!d.dq.empty()) {
                out = d.dq.front();
                d.dq.pop_front();
                return true;
            }
        }
        return false;
    }

    // a worker: with a run token, root jobs first, then stolen jobs; without
    // work or token, sleep until work is pushed or a token lent; stops when
    // every root job is done (every pushed job is needed by its pusher, so
    // nothing useful is left) or on stop
    void worker(int wid, const std::vector<Job> &roots, std::atomic<size_t> &next,
                std::atomic<size_t> &ndone, ErrorBox &err) {
        tl_wid = wid;
        bool token = false;
        try {
            for (;;) {
                if (stop->load(std::memory_order_relaxed) || finished.load(std::memory_order_acquire))
                    break;
                uint64_t e0 = epoch.load(std::memory_order_acquire);
                if (!token) token = token_try();
                if (token) {
                    if (next.load(std::memory_order_relaxed) < roots.size()) {
                        size_t k = next.fetch_add(1);
                        if (k < roots.size()) {
                            derived().run_job_if_absent(roots[k]);
                            if (ndone.fetch_add(1) + 1 == roots.size()) {
                                {
                                    std::lock_guard<std::mutex> g(im);
                                    finished.store(true, std::memory_order_release);
                                }
                                icv.notify_all();
                            }
                            continue;
                        }
                    }
                    Job j;
                    if (steal(wid, j)) {
                        st_steal.fetch_add(1, std::memory_order_relaxed);
                        derived().run_job_if_absent(j);
                        continue;
                    }
                    token_return();
                    token = false;
                }
                auto t0 = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lk(im);
                idle.fetch_add(1);
                icv.wait_for(lk, std::chrono::microseconds(IDLE_WAIT_US), [&] {
                    return epoch.load(std::memory_order_relaxed) != e0 ||
                           finished.load(std::memory_order_relaxed) || stop->load(std::memory_order_relaxed);
                });
                idle.fetch_sub(1);
                st_idle_ns.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0).count(),
                                     std::memory_order_relaxed);
            }
        } catch (...) {
            err.record(std::current_exception());
            {
                std::lock_guard<std::mutex> g(im);
                stop->store(true);
            }
            icv.notify_all();
        }
        if (token) token_return();
        tl_wid = -1;
    }

    // compute the jobs (distinct, larger monomials first) in nthreads workers
    // while the calling thread polls.  Few uncached jobs, all small: serially
    // (starting threads for many small calls costs more than it gains); a
    // single large job is split up by work stealing.
    void warm(std::vector<Job> jobs, int nthreads, PollFn poll) {
        if (nthreads > 1)
            jobs.erase(std::remove_if(jobs.begin(), jobs.end(),
                                      [this](const Job &j) { return derived().job_cached(j); }),
                       jobs.end());
        size_t big = 0;
        for (const Job &j : jobs) big = std::max(big, derived().job_size(j));
        if (nthreads <= 1 || jobs.empty() ||
            (jobs.size() < MIN_THREADED_JOBS && big < MIN_THREADED_LETTERS)) {
            serial([&] { for (const Job &j : jobs) derived().run_job(j); }, poll);
            return;
        }
        std::stable_sort(jobs.begin(), jobs.end(),
                         [this](const Job &x, const Job &y) { return derived().job_size(x) > derived().job_size(y); });
        stop->store(false);
        finished.store(false);
        GmpMallocGuard gmp;
        ErrorBox err;
        std::atomic<size_t> next{0}, ndone{0};
        tokens_free.store(nthreads);
        resumers = 0;
        int nworkers = nthreads * (1 + SPARE_THREADS_PER_TOKEN);
        wdq.clear();
        for (int i = 0; i < nworkers; i++) wdq.emplace_back(new WDeque());
        std::mutex dm;
        std::condition_variable dcv;
        int done = 0;
        auto work = [&](int wid) {
            worker(wid, jobs, next, ndone, err);
            flint_cleanup();
            {
                std::lock_guard<std::mutex> g(dm);
                done++;
            }
            dcv.notify_one();
        };
        std::vector<std::thread> th;
        int started = 0;
        try {
            for (; started < nworkers; started++) th.emplace_back(work, started);
        } catch (...) {
            err.record(std::current_exception());
            {
                std::lock_guard<std::mutex> g(im);
                stop->store(true);
            }
            icv.notify_all();
        }
        {
            std::unique_lock<std::mutex> lk(dm);
            while (done < started) {
                dcv.wait_for(lk, std::chrono::milliseconds(POLL_MS));
                if (done < started && poll && !stop->load()) {
                    lk.unlock();
                    if (poll()) {
                        {
                            std::lock_guard<std::mutex> g(im);
                            stop->store(true);
                        }
                        icv.notify_all();
                    }
                    lk.lock();
                }
            }
        }
        for (auto &t : th) t.join();
        wdq.clear();
        idle.store(0);
        tokens_free.store(0);
        if (stop->load() && !err.e) err.record(std::make_exception_ptr(Interrupted()));
        err.rethrow();
    }

    // f(k) for every k of order (task indices, larger tasks first), each task
    // computed serially in one thread: nthreads threads take the tasks in
    // order while the calling thread polls (the caches are shared, so
    // results do not depend on nthreads).  Serially (polling as in serial)
    // for nthreads <= 1 or a single task.  For batches of independent
    // element-level products (nprod_el_many); the caller holds a
    // MaybeGmpMalloc for the whole call.
    template <class F>
    void run_tasks(const std::vector<size_t> &order, int nthreads, PollFn poll, F &&f) {
        if (nthreads <= 1 || order.size() <= 1) {
            serial([&] { for (size_t k : order) f(k); }, poll);
            return;
        }
        stop->store(false);
        ErrorBox err;
        std::atomic<size_t> next{0};
        std::mutex dm;
        std::condition_variable dcv;
        int done = 0;
        auto work = [&]() {
            try {
                for (;;) {
                    if (stop->load(std::memory_order_relaxed)) break;
                    size_t i = next.fetch_add(1);
                    if (i >= order.size()) break;
                    f(order[i]);
                }
            } catch (...) {
                err.record(std::current_exception());
                stop->store(true);
            }
            flint_cleanup();
            {
                std::lock_guard<std::mutex> g(dm);
                done++;
            }
            dcv.notify_one();
        };
        int T = (int)std::min((size_t)nthreads, order.size());
        std::vector<std::thread> th;
        int started = 0;
        try {
            for (; started < T; started++) th.emplace_back(work);
        } catch (...) {
            err.record(std::current_exception());
            stop->store(true);
        }
        {
            std::unique_lock<std::mutex> lk(dm);
            while (done < started) {
                dcv.wait_for(lk, std::chrono::milliseconds(POLL_MS));
                if (done < started && poll && !stop->load()) {
                    lk.unlock();
                    if (poll()) stop->store(true);
                    lk.lock();
                }
            }
        }
        for (auto &t : th) t.join();
        if (stop->load() && !err.e) err.record(std::make_exception_ptr(Interrupted()));
        err.rethrow();
    }
    // the tasks (els[ia[k]], ns[k], els[ib[k]]) of a batch, checked; their
    // order for run_tasks: decreasing est(k)
    void check_tasks(size_t nels, const std::vector<int> &ia, const std::vector<long> &ns,
                     const std::vector<int> &ib) const {
        if (ia.size() != ns.size() || ib.size() != ns.size())
            throw std::invalid_argument("task vectors of different lengths");
        for (size_t k = 0; k < ns.size(); k++)
            if (ia[k] < 0 || ib[k] < 0 || (size_t)ia[k] >= nels || (size_t)ib[k] >= nels)
                throw std::out_of_range("task refers to an unknown element");
    }
    template <class Est>
    static std::vector<size_t> task_order(size_t N, Est &&est) {
        std::vector<size_t> order(N);
        std::vector<decltype(est(0))> e;
        e.reserve(N);
        for (size_t k = 0; k < N; k++) {
            order[k] = k;
            e.push_back(est(k));
        }
        std::stable_sort(order.begin(), order.end(), [&e](size_t x, size_t y) { return e[y] < e[x]; });
        return order;
    }

    // -- boundary ------------------------------------------------------------ //
    Elem elem_in(const ElemData &d) {
        Elem e;
        for (const auto &t : d) {
            if (t.first >= monos.n) throw std::out_of_range("unknown monomial id");
            C c(R);
            R.from_data(c, t.second);
            acc(e, t.first, c);
        }
        return e;
    }
    bool letters_less(uint32_t a, uint32_t b) const {
        const auto &x = monos[a].letters, &y = monos[b].letters;
        if (x.size() != y.size()) return x.size() < y.size();
        return x < y;
    }
    ElemData elem_out(const Elem &e) const {
        std::vector<uint32_t> ks;
        ks.reserve(e.size());
        for (const auto &kv : e) ks.push_back(kv.first);
        std::sort(ks.begin(), ks.end(), [this](uint32_t a, uint32_t b) { return derived_c().key_less(a, b); });
        ElemData d;
        d.reserve(ks.size());
        for (uint32_t k : ks) d.emplace_back(k, R.to_data(e.at(k)));
        return d;
    }
    PolesData poles_out(const Poles &P) const {
        PolesData d;
        for (const Elem &e : P) d.push_back(elem_out(e));
        return d;
    }

    // -- top-level sums (after the jobs of a call are computed) --------------- //
    // out[o] = sum of s * src over parts[o]; a large sum is split among
    // threads by output monomial (id % T): each thread multiplies only its
    // share, the shares are disjoint and merged by moves (exact: the same
    // result for any T)
    struct Part {
        const C *s;
        const Elem *src;
    };
    std::vector<Elem> sum_parts(const std::vector<std::vector<Part>> &parts, int nthreads) {
        size_t work = 0;
        for (const auto &ps : parts)
            for (const Part &p : ps) work += p.src->size();
        std::vector<Elem> out(parts.size());
        if (nthreads <= 1 || work < PAR_SUM_TERMS) {
            for (size_t o = 0; o < parts.size(); o++)
                for (const Part &p : parts[o]) add_scaled(out[o], *p.src, *p.s);
            return out;
        }
        uint32_t T = (uint32_t)nthreads;
        std::vector<std::vector<Elem>> share(T, std::vector<Elem>(parts.size()));
        run_parallel((int)T, [&](int t) {
            C x(R);
            for (size_t o = 0; o < parts.size(); o++)
                for (const Part &p : parts[o]) {
                    if (R.is_zero(*p.s)) continue;
                    for (const auto &kv : *p.src) {
                        if (kv.first % T != (uint32_t)t) continue;
                        R.mul(x, *p.s, kv.second);
                        acc(share[t][o], kv.first, x);
                    }
                }
        });
        for (uint32_t t = 0; t < T; t++)
            for (size_t o = 0; o < parts.size(); o++)
                for (auto &kv : share[t][o]) out[o].emplace(kv.first, std::move(kv.second));
        return out;
    }
    // f(t) for t = 0, ..., T-1 (t = 0 in the calling thread); errors rethrown
    template <class F>
    void run_parallel(int T, F &&f) {
        ErrorBox err;
        std::vector<std::thread> th;
        auto g = [&](int t) {
            try {
                f(t);
            } catch (...) {
                err.record(std::current_exception());
            }
            if (t) flint_cleanup();
        };
        try {
            for (int t = 1; t < T; t++) th.emplace_back(g, t);
        } catch (...) {
            err.record(std::current_exception());   // a missing share: the call fails
        }
        g(0);
        for (auto &x : th) x.join();
        err.rethrow();
    }

    // sum of c_a c_b f(a, b) over A x B, f(a, b) a cached element
    template <class F>
    Elem pair_sum(const Elem &A, const Elem &B, int nthreads, F &&f) {
        std::vector<C> scal;
        scal.reserve(A.size() * B.size());
        std::vector<std::vector<Part>> parts(1);
        for (const auto &a : A)
            for (const auto &b : B) {
                const Elem *e = f(a.first, b.first);
                if (!e || e->empty()) continue;
                scal.emplace_back(R);
                R.mul(scal.back(), a.second, b.second);
                parts[0].push_back(Part{&scal.back(), e});
            }
        return std::move(sum_parts(parts, nthreads)[0]);
    }
};

}  // namespace vae
