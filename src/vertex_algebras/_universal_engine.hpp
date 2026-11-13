// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// C++ engine of universal vertex algebras: the OPEdefs recursion of
// _universal_core.pyx (OPEs of monomials, normal ordering,
// derivatives) on interned monomials with FLINT coefficients, threads and
// all the rest from EngineCore (_engine_base.hpp).  No Python
// objects in here: the Cython wrapper _universal_engine.pyx
// converts at the boundary.
//
// Letters (generator i, derivative n) are packed as code = n * G + (G-1-i),
// so that code order is the core's letter order (higher derivative first,
// then earlier generator first); a monomial is a weakly decreasing list of
// codes, interned to a 32-bit id (0 = the vacuum).
#pragma once
#include "_engine_base.hpp"

namespace vae {

using Letters = std::vector<std::pair<int, int>>;   // (generator, derivative)

// what the Cython wrapper sees
struct EngineBase {
    virtual ~EngineBase() {}
    virtual uint32_t intern(const Letters &m) = 0;
    virtual Letters letters(uint32_t id) const = 0;
    virtual void set_ope(int i, int j, const PolesData &poles) = 0;
    virtual PolesData ope_mono(uint32_t a, uint32_t b, PollFn poll) = 0;
    virtual ElemData no_mono(uint32_t a, uint32_t b, PollFn poll) = 0;
    virtual ElemData deriv_mono(uint32_t m, int n, PollFn poll) = 0;
    virtual PolesData ope_el(const ElemData &A, const ElemData &B, int nthreads, PollFn poll) = 0;
    virtual std::vector<PolesData> ope_el_many(const ElemData &A, const std::vector<ElemData> &Bs,
                                               int nthreads, PollFn poll) = 0;
    virtual ElemData nprod_el(const ElemData &A, long n, const ElemData &B, int nthreads,
                              PollFn poll) = 0;
    virtual ElemData deriv_el(const ElemData &A, int n, int nthreads, PollFn poll) = 0;
    virtual std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &els, const std::vector<int> &ia,
                                                const std::vector<long> &ns, const std::vector<int> &ib,
                                                int nthreads, PollFn poll) = 0;
    virtual std::vector<size_t> cache_sizes() const = 0;
};

template <class Ring>
class Engine : public EngineCore<Engine<Ring>, Ring>, public EngineBase {
  public:
    using Core = EngineCore<Engine<Ring>, Ring>;
    using C = typename Core::C;
    using Elem = typename Core::Elem;
    using Poles = typename Core::Poles;
    using Job = typename Core::Job;
    using Spawn = typename Core::Spawn;
    using Part = typename Core::Part;
    using Core::R;
    using Core::monos;
    using Core::check;
    using Core::acc;
    using Core::add_scaled;
    using Core::add_scaled_z;
    using Core::add_scaled_si;
    using Core::single;
    using Core::intern_codes;
    using Core::serial;
    using Core::warm;
    using Core::sum_parts;
    using Core::pair_sum;
    using Core::elem_in;
    using Core::elem_out;
    using Core::poles_out;
    using Core::wire;
    using Core::run_tasks;
    using Core::check_tasks;

    // key-level jobs (what workers compute, push and steal)
    enum : int { J_OPE = 0, J_NO = 1, J_DERIV = 2 };    // J_DERIV: a = monomial, b = order

    int G;
    std::vector<int> par;
    static constexpr uint32_t MAXL = 1u << 12;     // letter codes with a lock-free id
    std::unique_ptr<std::atomic<uint32_t>[]> letter_ids{new std::atomic<uint32_t>[MAXL]()};
    std::vector<Poles> decl;                        // declared generator OPEs (i, j)
    std::vector<char> declared;
    ConcCache<Poles> gen_cache, ope_cache;
    ConcCache<Elem> nol_cache, noi_cache, nom_cache, der_cache;
    template <class... A>
    Engine(int ngens, const std::vector<int> &parities, A &&...ring_args)
        : Core(std::forward<A>(ring_args)...), G(ngens), par(parities) {
        if (G < 1) throw std::invalid_argument("no generators");
        par.resize((size_t)G, 0);
        monos.parity = [this](const std::vector<uint32_t> &v) {
            int p = 0;
            for (uint32_t c : v) p += lpar(c);
            return p;
        };
        decl.resize((size_t)G * G);
        declared.assign((size_t)G * G, 0);
        intern_codes(std::vector<uint32_t>());   // id 0 = vacuum
        wire(gen_cache);
        wire(ope_cache);
        wire(nol_cache);
        wire(noi_cache);
        wire(nom_cache);
        wire(der_cache);
    }

    // -- letters and monomials ------------------------------------------- //
    uint32_t code(int gen, int der) const { return (uint32_t)(der * G + (G - 1 - gen)); }
    int lgen(uint32_t c) const { return G - 1 - (int)(c % G); }
    int lder(uint32_t c) const { return (int)(c / G); }
    int lpar(uint32_t c) const { return par[lgen(c)] & 1; }

    uint32_t letter(uint32_t c) {
        if (c < MAXL) {
            uint32_t v = letter_ids[c].load(std::memory_order_acquire);
            if (v) return v;
        }
        uint32_t id = intern_codes(std::vector<uint32_t>{c});
        if (c < MAXL) letter_ids[c].store(id, std::memory_order_release);
        return id;
    }
    size_t len(uint32_t id) const { return monos[id].letters.size(); }
    uint32_t head(uint32_t id) const { return monos[id].letters[0]; }
    uint32_t tail(uint32_t id) const { return monos[id].tail; }
    uint32_t cons(uint32_t x, uint32_t mB) {       // (x,) + mB
        const std::vector<uint32_t> &b = monos[mB].letters;
        std::vector<uint32_t> m;
        m.reserve(b.size() + 1);
        m.push_back(x);
        m.insert(m.end(), b.begin(), b.end());
        return intern_codes(m);
    }
    int swap_sign_ll(uint32_t x, uint32_t y) const { return (lpar(x) & lpar(y)) ? -1 : 1; }
    int swap_sign_lm(uint32_t x, uint32_t m) const { return (lpar(x) & monos[m].par) ? -1 : 1; }

    // -- generator OPEs ------------------------------------------------------ //
    // (i, j) as declared, or from the declared (j, i) by skew-symmetry:
    // pole q = sum_{l >= q} sign (-1)^l / (l-q)! d^(l-q) AB[l]
    const Poles &gen_ope(int i, int j) {
        size_t k = (size_t)i * G + j;
        if (declared[k]) return decl[k];
        return gen_cache.get(pk((uint32_t)i, (uint32_t)j), [&]() -> Poles {
            check();
            Poles r;
            size_t kr = (size_t)j * G + i;
            if (!declared[kr]) return r;
            const Poles &AB = decl[kr];
            int sign = (par[i] & par[j] & 1) ? -1 : 1;
            size_t L = AB.size();
            Z f, s;
            for (size_t q = 1; q <= L; q++) {
                Elem tot;
                for (size_t l = q; l <= L; l++) {
                    Elem d = deriv_el_rec(AB[l - 1], (slong)(l - q));
                    fact(f.v, l - q);
                    fmpz_set_si(s.v, ((l & 1) ? -sign : sign));
                    add_scaled_z(tot, d, s.v, f.v);
                }
                r.push_back(std::move(tot));
            }
            return r;
        });
    }

    // -- normal ordering --------------------------------------------------- //
    Elem nc(uint32_t x, uint32_t y) {
        Poles poles = ope_letters(x, y);
        Elem res;
        Z f;
        slong i = 1;
        for (const Elem &p : poles) {
            Elem d = deriv_el_rec(p, i);
            fact(f.v, (ulong)i);
            Z s((i & 1) == 0 ? -1 : 1);
            add_scaled_z(res, d, s.v, f.v);
            i++;
        }
        return res;
    }

    const Elem &no_letters(uint32_t x, uint32_t y) {
        return nol_cache.get(pk(x, y), [&]() -> Elem {
            check();
            Elem res;
            if (x > y) {
                res = single(intern_codes(std::vector<uint32_t>{x, y}));
            } else if (x == y) {
                if (lpar(x)) {
                    Elem n = nc(x, y);
                    add_scaled_si(res, n, 1, 2);
                } else {
                    res = single(intern_codes(std::vector<uint32_t>{x, y}));
                }
            } else {
                int sign = swap_sign_ll(x, y);
                Elem n = nc(y, x);
                Elem d = single(intern_codes(std::vector<uint32_t>{y, x}));
                add_scaled_si(d, n, -1);
                add_scaled_si(res, d, sign);
            }
            return res;
        });
    }

    const Elem &no_insert(uint32_t x, uint32_t mB) {
        return noi_cache.get(pk(x, mB), [&]() -> Elem {
            check();
            Elem res;
            if (len(mB) == 1) {
                res = no_letters(x, head(mB));
            } else {
                uint32_t y = head(mB);
                uint32_t restB = tail(mB);
                if (x > y) {
                    res = single(cons(x, mB));
                } else if (x == y) {
                    if (lpar(x)) {
                        Elem whole = no_el_rec(nc(x, x), single(restB));
                        add_scaled_si(res, whole, 1, 2);
                    } else {
                        res = single(cons(x, mB));
                    }
                } else {
                    int sign = swap_sign_ll(x, y);
                    const Elem &inner = no_insert(x, restB);
                    Elem t1 = no_el_rec(single(letter(y)), inner);
                    Elem t2 = no_el_rec(nc(x, y), single(restB));
                    add_scaled_si(res, t1, sign);
                    add_scaled_si(res, t2, 1);
                }
            }
            return res;
        });
    }

    const Elem &no_mono_c(uint32_t mA, uint32_t mB) {
        return nom_cache.get(pk(mA, mB), [&]() -> Elem { return no_mono_compute(mA, mB); });
    }
    Elem no_mono_compute(uint32_t mA, uint32_t mB) {
        check();
        Elem res;
        if (len(mA) == 0) {
            res = single(mB);
        } else if (len(mB) == 0) {
            res = single(mA);
        } else if (len(mA) == 1 && len(mB) == 1) {
            res = no_letters(head(mA), head(mB));
        } else if (len(mA) >= 2) {
            uint32_t Al = head(mA), rest = tail(mA), Cm = mB;
            uint32_t lAl = letter(Al);
            Spawn sp(*this);
            if (sp) {
                if (rest != lAl) sp.add(J_OPE, rest, Cm);
                sp.add(J_NO, rest, Cm);
                sp.push();
            }
            const Poles &AC = ope_mono_c(lAl, Cm);
            if (sp) {
                for (size_t l = 1; l <= AC.size(); l++) sp.add(J_DERIV, rest, (uint32_t)l);
                sp.push();
            }
            const Poles &BC = (rest == lAl) ? AC : ope_mono_c(rest, Cm);
            if (sp) {           // the NO pairs of the l-loop below, all at once
                for (size_t l = 1; l <= BC.size(); l++) {
                    uint32_t x = letter(code(lgen(Al), lder(Al) + (int)l));
                    for (const auto &kv : BC[l - 1]) sp.add(J_NO, x, kv.first);
                }
                sp.push();
            }
            int sign = swap_sign_lm(Al, rest);
            const Elem &nr = no_mono_c(rest, Cm);   // cache nodes are stable
            res = no_el_rec(single(lAl), nr);
            Z f, one(1), s(sign);
            for (size_t l = 1; l <= BC.size(); l++) {
                Elem t = no_el_rec(single(letter(code(lgen(Al), lder(Al) + (int)l))), BC[l - 1]);
                fact(f.v, l);
                add_scaled_z(res, t, one.v, f.v);
            }
            for (size_t l = 1; l <= AC.size(); l++) {
                Elem t = no_el_rec(deriv_mono_c(rest, (int)l), AC[l - 1]);
                fact(f.v, l);
                add_scaled_z(res, t, s.v, f.v);
            }
        } else {
            res = no_insert(head(mA), mB);
        }
        return res;
    }

    Elem no_el_rec(const Elem &A, const Elem &B) {
        Spawn sp(*this);
        if (sp && A.size() * B.size() > 1) {
            for (const auto &a : A)
                for (const auto &b : B) sp.add(J_NO, a.first, b.first);
            sp.push();
        }
        Elem res;
        C s(R);
        for (const auto &a : A)
            for (const auto &b : B) {
                const Elem &t = no_mono_c(a.first, b.first);
                if (t.empty()) continue;
                R.mul(s, a.second, b.second);
                add_scaled(res, t, s);
            }
        return res;
    }

    // -- derivatives ------------------------------------------------------- //
    Elem deriv_el_rec(const Elem &d, slong n) {
        if (n == 0) return d;
        Spawn sp(*this);
        if (sp && d.size() > 1) {
            for (const auto &kv : d) sp.add(J_DERIV, kv.first, (uint32_t)n);
            sp.push();
        }
        Elem res;
        for (const auto &kv : d) add_scaled(res, deriv_mono_c(kv.first, (int)n), kv.second);
        return res;
    }

    const Elem &deriv_mono_c(uint32_t m, int n) {
        return der_cache.get(pk(m, (uint32_t)n), [&]() -> Elem { return deriv_mono_compute(m, n); });
    }
    Elem deriv_mono_compute(uint32_t m, int n) {
        check();
        Elem res;
        if (n == 0) {
            res = single(m);
        } else if (len(m) == 0) {
        } else if (len(m) == 1) {
            uint32_t x = head(m);
            res = single(letter(code(lgen(x), lder(x) + n)));
        } else {
            uint32_t x = head(m), rest = tail(m);
            Spawn sp(*this);
            if (sp) {
                for (int j = 0; j < n; j++) sp.add(J_DERIV, rest, (uint32_t)(n - j));
                sp.push();
            }
            Z c;
            for (int j = 0; j <= n; j++) {
                const Elem &d = deriv_mono_c(rest, n - j);
                if (d.empty()) continue;
                Elem t = no_el_rec(single(letter(code(lgen(x), lder(x) + j))), d);
                comb(c.v, (ulong)n, (ulong)j);
                add_scaled_z(res, t, c.v, nullptr);
            }
        }
        return res;
    }

    // -- OPEs -------------------------------------------------------------- //
    Poles ope_el_rec(const Elem &A, const Elem &B) {
        Spawn sp(*this);
        if (sp && A.size() * B.size() > 1) {
            for (const auto &a : A)
                for (const auto &b : B) sp.add(J_OPE, a.first, b.first);
            sp.push();
        }
        std::vector<std::pair<C, const Poles *>> parts;
        size_t maxq = 0;
        for (const auto &a : A)
            for (const auto &b : B) {
                const Poles &P = ope_mono_c(a.first, b.first);
                if (P.empty()) continue;
                C s(R);
                R.mul(s, a.second, b.second);
                if (R.is_zero(s)) continue;
                maxq = std::max(maxq, P.size());
                parts.emplace_back(std::move(s), &P);
            }
        Poles out(maxq);
        for (size_t q = 0; q < maxq; q++)
            for (auto &sp : parts)
                if (q < sp.second->size()) add_scaled(out[q], (*sp.second)[q], sp.first);
        return out;
    }

    static void trim(Poles &P) { while (!P.empty() && P.back().empty()) P.pop_back(); }

    const Poles &ope_mono_c(uint32_t mA, uint32_t mB) {
        return ope_cache.get(pk(mA, mB), [&]() -> Poles { return ope_mono_compute(mA, mB); });
    }
    Poles ope_mono_compute(uint32_t mA, uint32_t mB) {
        check();
        Poles out;
        if (len(mA) == 0 || len(mB) == 0) {
        } else if (len(mA) >= 2) {
            out = lq(head(mA), tail(mA), mB);
        } else if (len(mB) >= 2) {
            out = rab(head(mA), head(mB), tail(mB));
        } else {
            out = ope_letters(head(mA), head(mB));
        }
        trim(out);
        return out;
    }

    Poles ope_letters(uint32_t la, uint32_t lb) {
        int i = lgen(la), n = lder(la), j = lgen(lb), m = lder(lb);
        if (n == 0 && m == 0) return gen_ope(i, j);
        if (m == 0) return ope_derL(ope_letters(code(i, 0), code(j, 0)), n);
        if (n == 0) return ope_derR(ope_letters(code(i, 0), code(j, 0)), m);
        return ope_derL(ope_letters(code(i, 0), code(j, m)), n);
    }

    Poles ope_derL(const Poles &base, int i) {
        size_t L = base.size();
        if (L == 0) return Poles();
        Poles out(L + i);
        Z s;
        for (size_t q = i + 1; q <= L + i; q++) {
            size_t k = q - i;
            rising(s.v, k, (ulong)i);
            if (i & 1) fmpz_neg(s.v, s.v);
            add_scaled_z(out[q - 1], base[k - 1], s.v, nullptr);
        }
        return out;
    }

    Poles ope_derR(const Poles &base, int i) {
        long L = (long)base.size();
        if (L == 0) return Poles();
        Poles out(L + i);
        Z c, r, s;
        for (long q = 1; q <= L + i; q++) {
            Elem &tot = out[q - 1];
            long kmin = q - L > 0 ? q - L : 0;
            long kmax = i < q - 1 ? i : q - 1;
            for (long k = kmin; k <= kmax; k++) {
                long m = q - k;
                if (m < 1) continue;
                Elem d = deriv_el_rec(base[m - 1], i - k);
                comb(c.v, (ulong)i, (ulong)k);
                rising(r.v, (ulong)m, (ulong)k);
                fmpz_mul(s.v, c.v, r.v);
                add_scaled_z(tot, d, s.v, nullptr);
            }
        }
        return out;
    }

    // OPE[A, NO[B, C]]
    Poles rab(uint32_t Al, uint32_t Bl, uint32_t Cm) {
        uint32_t lA = letter(Al), lB = letter(Bl);
        Spawn sp(*this);
        if (sp && Cm != lB) {
            sp.add(J_OPE, lA, Cm);
            sp.push();
        }
        const Poles &AB = ope_mono_c(lA, lB);
        if (sp) {
            for (const Elem &e : AB)
                for (const auto &kv : e) sp.add(J_OPE, kv.first, Cm);
            sp.push();
        }
        const Poles &AC = (Cm == lB) ? AB : ope_mono_c(lA, Cm);
        size_t maxAB = AB.size(), maxAC = AC.size();
        std::vector<Poles> ABC;
        ABC.reserve(maxAB);
        Elem sC = single(Cm);
        for (size_t r = 1; r <= maxAB; r++) ABC.push_back(ope_el_rec(AB[r - 1], sC));
        size_t maxABC = 0, maxq = maxAC;
        for (size_t r = 1; r <= maxAB; r++) {
            maxABC = std::max(maxABC, ABC[r - 1].size());
            maxq = std::max(maxq, ABC[r - 1].size() + r);
        }
        if (sp) {           // the NO pairs of the q-loop below, all at once
            for (const Elem &e : AC)
                for (const auto &kv : e) sp.add(J_NO, lB, kv.first);
            for (const Elem &e : AB)
                for (const auto &kv : e) sp.add(J_NO, kv.first, Cm);
            sp.push();
        }
        int sign = swap_sign_ll(Al, Bl);
        Poles out(maxq);
        Elem sB = single(lB);
        Z c;
        for (size_t q = 1; q <= maxq; q++) {
            Elem &tot = out[q - 1];
            if (q <= maxAC) add_scaled_si(tot, no_el_rec(sB, AC[q - 1]), sign);
            if (q <= maxAB) add_scaled_si(tot, no_el_rec(AB[q - 1], sC), 1);
            long lo = std::max(1L, (long)q - (long)maxAB);
            long hi = std::min((long)q - 1, (long)maxABC);
            for (long l = lo; l <= hi; l++) {
                size_t r = q - l;
                if ((size_t)l <= ABC[r - 1].size()) {
                    comb(c.v, q - 1, (ulong)l);
                    add_scaled_z(tot, ABC[r - 1][l - 1], c.v, nullptr);
                }
            }
        }
        return out;
    }

    // OPE[NO[A, B], C]
    Poles lq(uint32_t Al, uint32_t Bm, uint32_t Cm) {
        uint32_t lA = letter(Al);
        Spawn sp(*this);
        if (sp && Bm != lA) {
            sp.add(J_OPE, Bm, Cm);
            sp.push();
        }
        const Poles &AC = ope_mono_c(lA, Cm);
        if (sp) {
            for (const Elem &e : AC)
                for (const auto &kv : e) sp.add(J_OPE, Bm, kv.first);
            for (size_t l = 1; l < AC.size(); l++) sp.add(J_DERIV, Bm, (uint32_t)l);
            sp.push();
        }
        const Poles &BC = (Bm == lA) ? AC : ope_mono_c(Bm, Cm);
        size_t maxAC = AC.size(), maxBC = BC.size();
        if (sp) {           // the NO pairs (letter, BC term) of the q-loop below
            for (size_t k = 0; k < maxBC; k++)
                for (size_t l = 0; l <= k; l++) {
                    uint32_t x = letter(code(lgen(Al), lder(Al) + (int)l));
                    for (const auto &kv : BC[k]) sp.add(J_NO, x, kv.first);
                }
            sp.push();
        }
        std::vector<Poles> BAC;
        BAC.reserve(maxAC);
        Elem sBm = single(Bm);
        for (size_t r = 1; r <= maxAC; r++) BAC.push_back(ope_el_rec(sBm, AC[r - 1]));
        size_t maxBAC = 0, maxq = std::max(maxAC, maxBC);
        for (size_t r = 1; r <= maxAC; r++) {
            maxBAC = std::max(maxBAC, BAC[r - 1].size());
            maxq = std::max(maxq, BAC[r - 1].size() + r);
        }
        int sign = swap_sign_lm(Al, Bm);
        Poles out(maxq);
        Z f, one(1), s(sign);
        for (size_t q = 1; q <= maxq; q++) {
            Elem &tot = out[q - 1];
            for (long l = 0; l <= (long)maxBC - (long)q; l++) {
                Elem t = no_el_rec(single(letter(code(lgen(Al), lder(Al) + (int)l))), BC[q + l - 1]);
                fact(f.v, (ulong)l);
                add_scaled_z(tot, t, one.v, f.v);
            }
            for (long l = 0; l <= (long)maxAC - (long)q; l++) {
                Elem t = no_el_rec(deriv_mono_c(Bm, (int)l), AC[q + l - 1]);
                fact(f.v, (ulong)l);
                add_scaled_z(tot, t, s.v, f.v);
            }
            long lo = std::max(1L, (long)q - (long)maxAC);
            long hi = std::min((long)q - 1, (long)maxBAC);
            for (long l = lo; l <= hi; l++) {
                size_t r = q - l;
                if ((size_t)l <= BAC[r - 1].size()) add_scaled_si(tot, BAC[r - 1][l - 1], sign);
            }
        }
        return out;
    }

    // -- jobs: warm the caches, serially or in threads ---------------------- //
    void run_job(const Job &j) {
        if (j.kind == J_OPE) ope_mono_c(j.a, j.b);
        else if (j.kind == J_NO) no_mono_c(j.a, j.b);
        else deriv_mono_c(j.a, (int)j.b);
    }
    size_t job_size(const Job &j) const { return len(j.a) + (j.kind == J_DERIV ? 0 : len(j.b)); }
    bool job_cached(const Job &j) {
        if (j.kind == J_OPE) return ope_cache.ready(pk(j.a, j.b));
        if (j.kind == J_NO) return nom_cache.ready(pk(j.a, j.b));
        return der_cache.ready(pk(j.a, j.b));
    }

    // a job no thread has started (a stolen or root job: never wait on it)
    void run_job_if_absent(const Job &j) {
        if (j.kind == J_OPE)
            ope_cache.get_if_absent(pk(j.a, j.b), [&]() -> Poles { return ope_mono_compute(j.a, j.b); });
        else if (j.kind == J_NO)
            nom_cache.get_if_absent(pk(j.a, j.b), [&]() -> Elem { return no_mono_compute(j.a, j.b); });
        else
            der_cache.get_if_absent(pk(j.a, j.b), [&]() -> Elem { return deriv_mono_compute(j.a, (int)j.b); });
    }

    // -- interface ------------------------------------------------------------ //
    uint32_t intern(const Letters &letters) override {
        std::vector<uint32_t> m;
        for (const auto &l : letters) {
            if (l.first < 0 || l.first >= G || l.second < 0) throw std::invalid_argument("bad letter");
            m.push_back(code(l.first, l.second));
        }
        for (size_t k = 1; k < m.size(); k++)
            if (m[k - 1] < m[k]) throw std::invalid_argument("letters not in canonical order");
        return intern_codes(m);
    }
    Letters letters(uint32_t id) const override {
        if (id >= monos.n) throw std::out_of_range("unknown monomial id");
        Letters out;
        for (uint32_t c : monos[id].letters) out.emplace_back(lgen(c), lder(c));
        return out;
    }
    void set_ope(int i, int j, const PolesData &poles) override {
        if (i < 0 || j < 0 || i >= G || j >= G) throw std::invalid_argument("bad generator index");
        if (ope_cache.size() || gen_cache.size())
            throw std::logic_error("generator OPEs must be set before computing");
        size_t k = (size_t)i * G + j;
        Poles P;
        for (const ElemData &e : poles) P.push_back(elem_in(e));
        decl[k] = std::move(P);
        declared[k] = 1;
    }
    PolesData ope_mono(uint32_t a, uint32_t b, PollFn poll) override {
        const Poles *P = nullptr;
        serial([&] { P = &ope_mono_c(a, b); }, poll);
        return poles_out(*P);
    }
    ElemData no_mono(uint32_t a, uint32_t b, PollFn poll) override {
        const Elem *E = nullptr;
        serial([&] { E = &no_mono_c(a, b); }, poll);
        return elem_out(*E);
    }
    ElemData deriv_mono(uint32_t m, int n, PollFn poll) override {
        if (n < 0) throw std::invalid_argument("negative derivative order");
        const Elem *E = nullptr;
        serial([&] { E = &deriv_mono_c(m, n); }, poll);
        return elem_out(*E);
    }

    // OPE[A, B] summed over the (computed) pair OPEs
    Poles ope_sum(const Elem &A, const Elem &B, int nthreads) {
        std::vector<C> scal;
        scal.reserve(A.size() * B.size());
        std::vector<const Poles *> src;
        size_t maxq = 0;
        for (const auto &a : A)
            for (const auto &b : B) {
                const Poles &P = ope_mono_c(a.first, b.first);
                if (P.empty()) continue;
                scal.emplace_back(R);
                R.mul(scal.back(), a.second, b.second);
                src.push_back(&P);
                maxq = std::max(maxq, P.size());
            }
        std::vector<std::vector<Part>> parts(maxq);
        for (size_t i = 0; i < src.size(); i++)
            for (size_t q = 0; q < src[i]->size(); q++) parts[q].push_back(Part{&scal[i], &(*src[i])[q]});
        return sum_parts(parts, nthreads);
    }
    void ope_jobs(std::vector<Job> &jobs, std::unordered_set<uint64_t> &seen, const Elem &A,
                  const Elem &B) {
        for (const auto &a : A)
            for (const auto &b : B)
                if (seen.insert(pk(a.first, b.first)).second) jobs.push_back(Job{J_OPE, a.first, b.first});
    }

    PolesData ope_el(const ElemData &dA, const ElemData &dB, int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA), B = elem_in(dB);
        std::vector<Job> jobs;
        std::unordered_set<uint64_t> seen;
        ope_jobs(jobs, seen, A, B);
        warm(std::move(jobs), nthreads, poll);
        Poles out;
        serial([&] { out = ope_sum(A, B, nthreads); }, poll);
        return poles_out(out);
    }
    std::vector<PolesData> ope_el_many(const ElemData &dA, const std::vector<ElemData> &dBs,
                                       int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA);
        std::vector<Elem> Bs;
        for (const ElemData &d : dBs) Bs.push_back(elem_in(d));
        std::vector<Job> jobs;
        std::unordered_set<uint64_t> seen;
        for (const Elem &B : Bs) ope_jobs(jobs, seen, A, B);
        warm(std::move(jobs), nthreads, poll);
        std::vector<PolesData> out;
        for (const Elem &B : Bs) {
            Poles P;
            serial([&] { P = ope_sum(A, B, nthreads); }, poll);
            out.push_back(poles_out(P));
        }
        return out;
    }
    // a_(n) b: n >= 0 pole n of the OPE; n = -1 normal ordering;
    // n < -1: sum over the terms of d^k a (k = -1-n) of NO / k!
    ElemData nprod_el(const ElemData &dA, long n, const ElemData &dB, int nthreads,
                      PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA), B = elem_in(dB);
        return elem_out(nprod_elem(A, n, B, nthreads, poll));
    }
    Elem nprod_elem(const Elem &A, long n, const Elem &B, int nthreads, PollFn poll) {
        std::vector<Job> jobs;
        std::unordered_set<uint64_t> seen;
        Elem res;
        if (n >= 0) {
            ope_jobs(jobs, seen, A, B);
            warm(std::move(jobs), nthreads, poll);
            serial([&] {
                res = pair_sum(A, B, nthreads, [&](uint32_t a, uint32_t b) -> const Elem * {
                    const Poles &P = ope_mono_c(a, b);
                    return (size_t)n < P.size() ? &P[n] : nullptr;
                });
            }, poll);
        } else if (n == -1) {
            for (const auto &a : A)
                for (const auto &b : B)
                    if (seen.insert(pk(a.first, b.first)).second) jobs.push_back(Job{J_NO, a.first, b.first});
            warm(std::move(jobs), nthreads, poll);
            serial([&] {
                res = pair_sum(A, B, nthreads, [&](uint32_t a, uint32_t b) { return &no_mono_c(a, b); });
            }, poll);
        } else {
            if (n < -(long)INT32_MAX) throw std::invalid_argument("n too small");
            int k = (int)(-1 - n);
            for (const auto &a : A) jobs.push_back(Job{J_DERIV, a.first, (uint32_t)k});
            warm(jobs, nthreads, poll);
            Elem dk;
            serial([&] { dk = deriv_el_rec(A, k); }, poll);
            jobs.clear();
            for (const auto &m : dk)
                for (const auto &b : B)
                    if (seen.insert(pk(m.first, b.first)).second) jobs.push_back(Job{J_NO, m.first, b.first});
            warm(std::move(jobs), nthreads, poll);
            serial([&] {
                res = pair_sum(dk, B, nthreads, [&](uint32_t a, uint32_t b) { return &no_mono_c(a, b); });
                Z f;
                fact(f.v, (ulong)k);
                Elem out;
                Z one(1);
                add_scaled_z(out, res, one.v, f.v);
                res = std::move(out);
            }, poll);
        }
        return res;
    }
    // [els[ia[k]]_(ns[k]) els[ib[k]] for every k] in one call: the key-level
    // jobs of all products with n >= -1 first (one warm), then their sums,
    // each in one of nthreads threads (larger |A| |B| first); n < -1 one by
    // one (derivatives first, as nprod_el)
    std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &dels, const std::vector<int> &ia,
                                        const std::vector<long> &ns, const std::vector<int> &ib,
                                        int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        check_tasks(dels.size(), ia, ns, ib);
        std::vector<Elem> E;
        E.reserve(dels.size());
        for (const ElemData &d : dels) E.push_back(elem_in(d));
        std::vector<Job> jobs;
        std::unordered_set<uint64_t> seen_ope, seen_no;
        std::vector<size_t> batch, rest;
        for (size_t k = 0; k < ns.size(); k++) {
            const Elem &A = E[(size_t)ia[k]], &B = E[(size_t)ib[k]];
            if (ns[k] < -1) {
                rest.push_back(k);
                continue;
            }
            batch.push_back(k);
            if (ns[k] >= 0) {
                ope_jobs(jobs, seen_ope, A, B);
                continue;
            }
            for (const auto &a : A)
                for (const auto &b : B)
                    if (seen_no.insert(pk(a.first, b.first)).second) jobs.push_back(Job{J_NO, a.first, b.first});
        }
        warm(std::move(jobs), nthreads, poll);
        std::vector<size_t> est(ns.size(), 0);
        for (size_t k : batch) est[k] = E[(size_t)ia[k]].size() * E[(size_t)ib[k]].size();
        std::stable_sort(batch.begin(), batch.end(), [&est](size_t x, size_t y) { return est[y] < est[x]; });
        std::vector<Elem> res(ns.size());
        run_tasks(batch, nthreads, poll, [&](size_t k) {
            const Elem &A = E[(size_t)ia[k]], &B = E[(size_t)ib[k]];
            long n = ns[k];
            if (n >= 0)
                res[k] = pair_sum(A, B, 1, [&](uint32_t a, uint32_t b) -> const Elem * {
                    const Poles &P = ope_mono_c(a, b);
                    return (size_t)n < P.size() ? &P[n] : nullptr;
                });
            else
                res[k] = pair_sum(A, B, 1, [&](uint32_t a, uint32_t b) { return &no_mono_c(a, b); });
        });
        for (size_t k : rest) res[k] = nprod_elem(E[(size_t)ia[k]], ns[k], E[(size_t)ib[k]], nthreads, poll);
        std::vector<ElemData> out;
        out.reserve(res.size());
        for (const Elem &e : res) out.push_back(elem_out(e));
        return out;
    }
    ElemData deriv_el(const ElemData &dA, int n, int nthreads, PollFn poll) override {
        if (n < 0) throw std::invalid_argument("negative derivative order");
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA);
        if (n == 0) return elem_out(A);
        std::vector<Job> jobs;
        for (const auto &a : A) jobs.push_back(Job{J_DERIV, a.first, (uint32_t)n});
        warm(std::move(jobs), nthreads, poll);
        Elem res;
        serial([&] {
            Elem one = single(0);
            res = pair_sum(A, one, nthreads, [&](uint32_t m, uint32_t) { return &deriv_mono_c(m, n); });
        }, poll);
        return elem_out(res);
    }
    std::vector<size_t> cache_sizes() const override {
        std::vector<size_t> s = this->sched_stats();
        s.insert(s.begin(), {ope_cache.size(), nol_cache.size() + noi_cache.size() + nom_cache.size(),
                der_cache.size(), gen_cache.size(), (size_t)monos.n});
        return s;
    }
};

// kind: "q", "qpoly", "qmpoly" (nvars >= 1), "nmod" (modulus a prime < 2^63)
// minpoly: the defining polynomial of the number field ("nf", "nfmpoly")
inline EngineBase *make_engine(const std::string &kind, int ngens, const std::vector<int> &parities,
                               long nvars, unsigned long modulus, const CoefData &minpoly) {
    if (kind == "nf") return new Engine<RingNF>(ngens, parities, minpoly);
    if (kind == "nfmpoly") return new Engine<RingNFMPoly>(ngens, parities, (slong)nvars, minpoly);
    if (kind == "q") return new Engine<RingQ>(ngens, parities);
    if (kind == "qpoly") return new Engine<RingQPoly>(ngens, parities);
    if (kind == "qmpoly") {
        if (nvars < 1) throw std::invalid_argument("qmpoly needs at least one variable");
        return new Engine<RingQMPoly>(ngens, parities, (slong)nvars);
    }
    if (kind == "nmod") {
        if (modulus < 2 || modulus > (unsigned long)WORD_MAX || !n_is_prime(modulus))
            throw std::invalid_argument("nmod needs a prime modulus below 2^63");
        return new Engine<RingNmod>(ngens, parities, (ulong)modulus);
    }
    throw std::invalid_argument("unknown coefficient kind " + kind);
}

}  // namespace vae
