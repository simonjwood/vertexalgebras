// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// Sparse linear algebra over fields in C++, without Python or Sage
// arithmetic in the loops: the reduced row echelon form of a growing span
// (SparseRREF), the algorithm of the Python _Block of _keyed_spans.py
// (so the same labels, pivots and rows).  Keys are integer ids; their order
// (for the pivot choice) integer ranks set by the caller (default: the id).
// Fields:
//   QFrac  QQ(x_1, ..., x_n), FLINT fmpz_mpoly_q (canonical fractions of
//          integer polynomials); input as a fraction (numerator,
//          denominator: polynomials over QQ) or as a polynomial in
//          x_1..x_n, u_1..u_m with u_j standing for 1/p_j (the lifted ring of
//          a C++ engine, universal_vertex_algebra._LiftedKernel);
//   NFld   a number field QQ[t]/(m(t)), FLINT nf_elem; input as a polynomial
//          in t (the power-basis coordinates of the engines' 'nf' kind).
// Coefficient data are engine terms (exponents, num, den)
// (_engine_base.hpp); output fractions have integer coefficients.
#pragma once
#include "_engine_base.hpp"
#include <flint/fmpz_mpoly.h>
#include <flint/fmpz_mpoly_q.h>
#include <flint/nf.h>
#include <flint/nf_elem.h>
#include <flint/nmod_mat.h>
#include <map>

namespace vae {

// QQ(x_1, ..., x_n), n >= 1; optional substitution u_j = 1/p_j (lifted input)
struct QFrac {
    fmpz_mpoly_ctx_t ctx;
    slong n;
    // p_j = subst[j] / subst_den[j] (integer polynomial / positive integer)
    std::vector<fmpz_mpoly_struct> subst;
    std::deque<Z> subst_den;                  // a deque: Z cannot be moved
    explicit QFrac(slong nv) : n(nv) {
        if (nv < 1) throw std::invalid_argument("a rational function field needs a variable");
        fmpz_mpoly_ctx_init(ctx, nv, ORD_LEX);
    }
    ~QFrac() {
        for (auto &p : subst) fmpz_mpoly_clear(&p, ctx);
        fmpz_mpoly_ctx_clear(ctx);
    }
    QFrac(const QFrac &) = delete;
    QFrac &operator=(const QFrac &) = delete;
    struct E {
        fmpz_mpoly_q_t v;
        const fmpz_mpoly_ctx_struct *c;
        explicit E(const QFrac &F) : c(F.ctx) { fmpz_mpoly_q_init(v, c); }
        E(const E &o) : c(o.c) { fmpz_mpoly_q_init(v, c); fmpz_mpoly_q_set(v, o.v, c); }
        E(E &&o) noexcept : c(o.c) { fmpz_mpoly_q_init(v, c); fmpz_mpoly_q_swap(v, o.v, c); }
        E &operator=(const E &o) { if (this != &o) fmpz_mpoly_q_set(v, o.v, c); return *this; }
        E &operator=(E &&o) noexcept { fmpz_mpoly_q_swap(v, o.v, c); return *this; }
        ~E() { fmpz_mpoly_q_clear(v, c); }
    };
    bool is_zero(const E &a) const { return fmpz_mpoly_q_is_zero(a.v, ctx); }
    void add(E &a, const E &b) const { fmpz_mpoly_q_add(a.v, a.v, b.v, ctx); }
    void mul(E &r, const E &a, const E &b) const { fmpz_mpoly_q_mul(r.v, a.v, b.v, ctx); }
    void neg(E &r, const E &a) const { fmpz_mpoly_q_neg(r.v, a.v, ctx); }
    void inv(E &r, const E &a) const { fmpz_mpoly_q_inv(r.v, a.v, ctx); }
    void set_si(E &a, slong s) const { fmpz_mpoly_q_set_si(a.v, s, ctx); }

    // the terms of d over the first nv variables (others must be 0 unless
    // rest is given: then rest receives them) as an integer polynomial,
    // times the common denominator den
    void poly_in(fmpz_mpoly_t p, fmpz_t den, const CoefData &d, slong nv) const {
        fmpz_mpoly_zero(p, ctx);
        fmpz_one(den);
        Z zn, m, g;
        for (const Term &t : d) {
            num_get(m.v, t.den);
            fmpz_lcm(den, den, m.v);
        }
        std::vector<ulong> e((size_t)n);
        fmpz_mpoly_t term;
        fmpz_mpoly_init(term, ctx);
        for (const Term &t : d) {
            if ((slong)t.exps.size() != nv) {
                fmpz_mpoly_clear(term, ctx);
                throw std::invalid_argument("bad exponent vector");
            }
            for (slong j = 0; j < n; j++) {
                if (t.exps[(size_t)j] < 0) {
                    fmpz_mpoly_clear(term, ctx);
                    throw std::invalid_argument("negative exponent");
                }
                e[(size_t)j] = (ulong)t.exps[(size_t)j];
            }
            num_get(zn.v, t.num);
            num_get(m.v, t.den);
            fmpz_divexact(g.v, den, m.v);
            fmpz_mul(g.v, g.v, zn.v);
            fmpz_mpoly_zero(term, ctx);
            fmpz_mpoly_set_coeff_fmpz_ui(term, g.v, e.data(), ctx);
            fmpz_mpoly_add(p, p, term, ctx);
        }
        fmpz_mpoly_clear(term, ctx);
    }
    void set_subst(const std::vector<CoefData> &ps) {
        for (auto &p : subst) fmpz_mpoly_clear(&p, ctx);
        subst.clear();
        subst_den.clear();
        for (const CoefData &d : ps) {
            subst.emplace_back();
            subst_den.emplace_back();
            fmpz_mpoly_init(&subst.back(), ctx);
            poly_in(&subst.back(), subst_den.back().v, d, n);   // p = poly / den
            if (fmpz_mpoly_is_zero(&subst.back(), ctx)) throw std::invalid_argument("zero substitution");
        }
    }
    // num / den, both polynomials over QQ in x_1..x_n
    void from_frac(E &a, const CoefData &num, const CoefData &den) const {
        Z dn, dd;
        fmpz_mpoly_t p, q;
        fmpz_mpoly_init(p, ctx);
        fmpz_mpoly_init(q, ctx);
        try {
            poly_in(p, dn.v, num, n);
            poly_in(q, dd.v, den, n);
        } catch (...) {
            fmpz_mpoly_clear(p, ctx);
            fmpz_mpoly_clear(q, ctx);
            throw;
        }
        bool bad = fmpz_mpoly_is_zero(q, ctx);
        if (!bad) {
            fmpz_mpoly_scalar_mul_fmpz(p, p, dd.v, ctx);       // (p/dn)/(q/dd) = (p dd)/(q dn)
            fmpz_mpoly_scalar_mul_fmpz(q, q, dn.v, ctx);
            fmpz_mpoly_swap(fmpz_mpoly_q_numref(a.v), p, ctx);
            fmpz_mpoly_swap(fmpz_mpoly_q_denref(a.v), q, ctx);
            fmpz_mpoly_q_canonicalise(a.v, ctx);
        }
        fmpz_mpoly_clear(p, ctx);
        fmpz_mpoly_clear(q, ctx);
        if (bad) throw std::invalid_argument("zero denominator");
    }
    // a polynomial in x_1..x_n, u_1..u_m with u_j = 1/p_j: the terms grouped
    // by their u exponents k, each group P_k / prod p_j^k_j
    void from_lifted(E &a, const CoefData &d) const {
        slong m = (slong)subst.size();
        std::map<std::vector<long>, CoefData> groups;
        for (const Term &t : d) {
            if ((slong)t.exps.size() != n + m) throw std::invalid_argument("bad lifted exponent vector");
            std::vector<long> k(t.exps.begin() + n, t.exps.end());
            Term x;
            x.exps.assign(t.exps.begin(), t.exps.begin() + n);
            x.num = t.num;
            x.den = t.den;
            groups[k].push_back(std::move(x));
        }
        fmpz_mpoly_q_set_si(a.v, 0, ctx);
        E g(*this);
        Z den;
        fmpz_mpoly_t pw;
        fmpz_mpoly_init(pw, ctx);
        for (const auto &kv : groups) {
            poly_in(fmpz_mpoly_q_numref(g.v), den.v, kv.second, n);
            fmpz_mpoly_set_fmpz(fmpz_mpoly_q_denref(g.v), den.v, ctx);
            for (slong j = 0; j < m; j++) {
                long e = kv.first[(size_t)j];
                if (e < 0) {
                    fmpz_mpoly_clear(pw, ctx);
                    throw std::invalid_argument("negative exponent");
                }
                if (!e) continue;
                // u_j^e = (den_j / poly_j)^e
                fmpz_mpoly_pow_ui(pw, &subst[(size_t)j], (ulong)e, ctx);
                fmpz_mpoly_mul(fmpz_mpoly_q_denref(g.v), fmpz_mpoly_q_denref(g.v), pw, ctx);
                Z de;
                fmpz_pow_ui(de.v, subst_den[(size_t)j].v, (ulong)e);
                fmpz_mpoly_scalar_mul_fmpz(fmpz_mpoly_q_numref(g.v), fmpz_mpoly_q_numref(g.v), de.v, ctx);
            }
            fmpz_mpoly_q_canonicalise(g.v, ctx);
            fmpz_mpoly_q_add(a.v, a.v, g.v, ctx);
        }
        fmpz_mpoly_clear(pw, ctx);
    }
    CoefData poly_out(const fmpz_mpoly_t p) const {
        CoefData d;
        Z c;
        std::vector<ulong> e((size_t)n);
        for (slong i = 0; i < fmpz_mpoly_length(p, ctx); i++) {
            fmpz_mpoly_get_term_coeff_fmpz(c.v, p, i, ctx);
            fmpz_mpoly_get_term_exp_ui(e.data(), p, i, ctx);
            Term t;
            t.exps.assign(e.begin(), e.end());
            t.num = num_of(c.v);
            t.den.small = 1;
            d.push_back(std::move(t));
        }
        return d;
    }
    std::pair<CoefData, CoefData> out(const E &a) const {
        return {poly_out(fmpz_mpoly_q_numref(a.v)), poly_out(fmpz_mpoly_q_denref(a.v))};
    }
};

// the number field QQ[t]/(m(t)), m irreducible over QQ (any leading coefficient)
struct NFld {
    nf_t nf;
    explicit NFld(const CoefData &mp) {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        try {
            qpoly_from_data(p, mp);
        } catch (...) {
            fmpq_poly_clear(p);
            throw;
        }
        if (fmpq_poly_degree(p) < 1) {
            fmpq_poly_clear(p);
            throw std::invalid_argument("number field: defining polynomial of degree < 1");
        }
        nf_init(nf, p);
        fmpq_poly_clear(p);
    }
    ~NFld() { nf_clear(nf); }
    NFld(const NFld &) = delete;
    NFld &operator=(const NFld &) = delete;
    struct E {
        nf_elem_t v;
        const nf_struct *k;
        explicit E(const NFld &F) : k(F.nf) { nf_elem_init(v, k); }
        E(const E &o) : k(o.k) { nf_elem_init(v, k); nf_elem_set(v, o.v, k); }
        E(E &&o) noexcept : k(o.k) { nf_elem_init(v, k); nf_elem_swap(v, o.v, k); }
        E &operator=(const E &o) { if (this != &o) nf_elem_set(v, o.v, k); return *this; }
        E &operator=(E &&o) noexcept { nf_elem_swap(v, o.v, k); return *this; }
        ~E() { nf_elem_clear(v, k); }
    };
    bool is_zero(const E &a) const { return nf_elem_is_zero(a.v, nf); }
    void add(E &a, const E &b) const { nf_elem_add(a.v, a.v, b.v, nf); }
    void mul(E &r, const E &a, const E &b) const {
        if (&r == &a || &r == &b) {
            E t(*this);
            nf_elem_mul(t.v, a.v, b.v, nf);
            nf_elem_swap(r.v, t.v, nf);
        } else {
            nf_elem_mul(r.v, a.v, b.v, nf);
        }
    }
    void neg(E &r, const E &a) const { nf_elem_neg(r.v, a.v, nf); }
    void inv(E &r, const E &a) const { nf_elem_inv(r.v, a.v, nf); }
    void set_si(E &a, slong s) const { nf_elem_set_si(a.v, s, nf); }
    void from_poly(E &a, const CoefData &d) const {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        try {
            qpoly_from_data(p, d);
        } catch (...) {
            fmpq_poly_clear(p);
            throw;
        }
        nf_elem_set_fmpq_poly(a.v, p, nf);
        fmpq_poly_clear(p);
    }
    CoefData out(const E &a) const {
        fmpq_poly_t p;
        fmpq_poly_init(p);
        nf_elem_get_fmpq_poly(p, a.v, nf);
        CoefData d = qpoly_to_data(p);
        fmpq_poly_clear(p);
        return d;
    }
};

// The reduced row echelon form of a span over a field F: rows {pivot:
// (row, combo)}, combos (rows as combinations of the inserted vectors, by
// label index) kept only if track_combos; the pivot of a new row is its
// key of largest rank.
template <class F>
class SparseRREF {
  public:
    using E = typename F::E;
    using Vec = std::unordered_map<uint32_t, E>;
    F &R;
    bool combos;
    std::vector<int64_t> rank;              // by key id
    std::unordered_map<uint32_t, std::pair<Vec, Vec>> rows;
    std::vector<uint32_t> pivot_order;      // pivots in insertion order
    uint32_t nlabels = 0;
    PollFn poll = nullptr;
    unsigned long ops = 0;

    SparseRREF(F &field, bool track_combos) : R(field), combos(track_combos) {}

    void set_rank(uint32_t key, int64_t r) {
        if (key >= rank.size()) {
            size_t old = rank.size();
            rank.resize((size_t)key + 1);
            for (size_t i = old; i < rank.size(); i++) rank[i] = (int64_t)i;
        }
        rank[key] = r;
    }
    int64_t rank_of(uint32_t key) const { return key < rank.size() ? rank[key] : (int64_t)key; }
    void tick() {
        if (poll && (++ops % 4096) == 0 && poll()) throw Interrupted();
    }
    // dst += c src (zeros dropped)
    void addmul(Vec &dst, const Vec &src, const E &c) {
        E t(R);
        for (const auto &kv : src) {
            tick();
            R.mul(t, kv.second, c);
            auto it = dst.find(kv.first);
            if (it == dst.end()) dst.emplace(kv.first, t);
            else {
                R.add(it->second, t);
                if (R.is_zero(it->second)) dst.erase(it);
            }
        }
    }
    // r reduced by the rows (no pivot keys left), t = the multipliers
    void reduce_vec(Vec &r, Vec &t) {
        std::vector<uint32_t> ps;
        for (const auto &kv : r)
            if (rows.count(kv.first)) ps.push_back(kv.first);
        E m(R);
        for (uint32_t p : ps) {
            auto it = r.find(p);
            if (it == r.end()) continue;
            E c = it->second;
            R.neg(m, c);
            addmul(r, rows.at(p).first, m);
            t.emplace(p, std::move(c));
        }
    }
    Vec combo_of(const Vec &t) {
        Vec out;
        for (const auto &kv : t) addmul(out, rows.at(kv.first).second, kv.second);
        return out;
    }
    // atomic: the new rows are computed first (polling may interrupt), then
    // committed without interruption points
    bool insert(Vec r) {
        Vec t;
        reduce_vec(r, t);
        if (r.empty()) return false;
        uint32_t idx = nlabels;
        Vec combo;
        E one(R), m(R);
        R.set_si(one, 1);
        if (combos) {
            R.set_si(m, -1);
            addmul(combo, combo_of(t), m);
            auto it = combo.find(idx);
            if (it == combo.end()) combo.emplace(idx, one);
            else R.add(it->second, one);
        }
        uint32_t p = 0;
        bool first = true;
        for (const auto &kv : r)
            if (first || rank_of(kv.first) > rank_of(p)) {
                p = kv.first;
                first = false;
            }
        E inv(R);
        R.inv(inv, r.at(p));
        for (auto &kv : r) R.mul(kv.second, kv.second, inv);
        for (auto &kv : combo) R.mul(kv.second, kv.second, inv);
        std::vector<std::pair<uint32_t, std::pair<Vec, Vec>>> upd;
        for (const auto &row : rows) {
            auto it = row.second.first.find(p);
            if (it == row.second.first.end()) continue;
            R.neg(m, it->second);
            std::pair<Vec, Vec> nr = row.second;
            addmul(nr.first, r, m);
            if (combos) addmul(nr.second, combo, m);
            upd.emplace_back(row.first, std::move(nr));
        }
        for (auto &u : upd) rows[u.first] = std::move(u.second);   // commit
        rows.emplace(p, std::make_pair(std::move(r), std::move(combo)));
        pivot_order.push_back(p);
        nlabels++;
        return true;
    }
    // coefficients by label index of v (reduced in place), ok = v in the span
    Vec coordinates(Vec r, bool &ok) {
        if (!combos) throw std::logic_error("coordinates need the combos");
        Vec t;
        reduce_vec(r, t);
        ok = r.empty();
        return ok ? combo_of(t) : Vec();
    }
    bool contains(Vec r) {
        Vec t;
        reduce_vec(r, t);
        return r.empty();
    }
    Vec reduce(Vec r) {
        Vec t;
        reduce_vec(r, t);
        return r;
    }
};

// the boundary: one class per field and input form, data in, data out
using FracData = std::vector<std::pair<uint32_t, std::pair<CoefData, CoefData>>>;  // (key, (num, den))
using PolyData = std::vector<std::pair<uint32_t, CoefData>>;                       // (key, lifted / t-poly)

class FracEchelon {
  public:
    QFrac F;
    SparseRREF<QFrac> S;
    std::vector<bool> done;     // flags of the last batch (also when interrupted)
    FracEchelon(slong nvars, bool combos) : F(nvars), S(F, combos) {}
    void set_subst(const std::vector<CoefData> &ps) { F.set_subst(ps); }
    void set_rank(uint32_t k, int64_t r) { S.set_rank(k, r); }
    using Vec = SparseRREF<QFrac>::Vec;
    void put(Vec &v, uint32_t k, QFrac::E &c) const {
        if (F.is_zero(c)) return;
        auto it = v.find(k);
        if (it == v.end()) v.emplace(k, std::move(c));
        else {
            F.add(it->second, c);
            if (F.is_zero(it->second)) v.erase(it);
        }
    }
    Vec in_frac(const FracData &d) const {
        Vec v;
        for (const auto &t : d) {
            QFrac::E c(F);
            F.from_frac(c, t.second.first, t.second.second);
            put(v, t.first, c);
        }
        return v;
    }
    Vec in_lifted(const PolyData &d) const {
        Vec v;
        for (const auto &t : d) {
            QFrac::E c(F);
            F.from_lifted(c, t.second);
            put(v, t.first, c);
        }
        return v;
    }
    FracData out(const Vec &v) const {
        std::vector<uint32_t> ks;
        for (const auto &kv : v) ks.push_back(kv.first);
        std::sort(ks.begin(), ks.end());
        FracData d;
        for (uint32_t k : ks) d.emplace_back(k, F.out(v.at(k)));
        return d;
    }
    std::vector<bool> insert_frac(const std::vector<FracData> &vs, PollFn pl) {
        S.poll = pl;
        done.clear();
        try {
            for (const FracData &d : vs) done.push_back(S.insert(in_frac(d)));
        } catch (...) {
            S.poll = nullptr;
            throw;              // done: the inserts committed before
        }
        S.poll = nullptr;
        return done;
    }
    std::vector<bool> insert_lifted(const std::vector<PolyData> &vs, PollFn pl) {
        S.poll = pl;
        done.clear();
        try {
            for (const PolyData &d : vs) done.push_back(S.insert(in_lifted(d)));
        } catch (...) {
            S.poll = nullptr;
            throw;              // done: the inserts committed before
        }
        S.poll = nullptr;
        return done;
    }
    bool contains_frac(const FracData &d, PollFn pl) {
        S.poll = pl;
        bool r = S.contains(in_frac(d));
        S.poll = nullptr;
        return r;
    }
    bool contains_lifted(const PolyData &d, PollFn pl) {
        S.poll = pl;
        bool r = S.contains(in_lifted(d));
        S.poll = nullptr;
        return r;
    }
    FracData reduce_frac(const FracData &d, PollFn pl) {
        S.poll = pl;
        FracData r = out(S.reduce(in_frac(d)));
        S.poll = nullptr;
        return r;
    }
    FracData coordinates_frac(const FracData &d, bool &ok, PollFn pl) {
        S.poll = pl;
        FracData r = out(S.coordinates(in_frac(d), ok));
        S.poll = nullptr;
        return r;
    }
    FracData coordinates_lifted(const PolyData &d, bool &ok, PollFn pl) {
        S.poll = pl;
        FracData r = out(S.coordinates(in_lifted(d), ok));
        S.poll = nullptr;
        return r;
    }
    std::vector<uint32_t> pivots() const { return S.pivot_order; }
    FracData row(uint32_t p) const { return out(S.rows.at(p).first); }
};

class NFEchelon {
  public:
    NFld F;
    SparseRREF<NFld> S;
    std::vector<bool> done;     // flags of the last batch (also when interrupted)
    NFEchelon(const CoefData &minpoly, bool combos) : F(minpoly), S(F, combos) {}
    void set_rank(uint32_t k, int64_t r) { S.set_rank(k, r); }
    using Vec = SparseRREF<NFld>::Vec;
    Vec in(const PolyData &d) const {
        Vec v;
        for (const auto &t : d) {
            NFld::E c(F);
            F.from_poly(c, t.second);
            if (F.is_zero(c)) continue;
            auto it = v.find(t.first);
            if (it == v.end()) v.emplace(t.first, std::move(c));
            else {
                F.add(it->second, c);
                if (F.is_zero(it->second)) v.erase(it);
            }
        }
        return v;
    }
    PolyData out(const Vec &v) const {
        std::vector<uint32_t> ks;
        for (const auto &kv : v) ks.push_back(kv.first);
        std::sort(ks.begin(), ks.end());
        PolyData d;
        for (uint32_t k : ks) d.emplace_back(k, F.out(v.at(k)));
        return d;
    }
    std::vector<bool> insert_many(const std::vector<PolyData> &vs, PollFn pl) {
        S.poll = pl;
        done.clear();
        try {
            for (const PolyData &d : vs) done.push_back(S.insert(in(d)));
        } catch (...) {
            S.poll = nullptr;
            throw;              // done: the inserts committed before
        }
        S.poll = nullptr;
        return done;
    }
    bool contains(const PolyData &d, PollFn pl) {
        S.poll = pl;
        bool r = S.contains(in(d));
        S.poll = nullptr;
        return r;
    }
    PolyData reduce(const PolyData &d, PollFn pl) {
        S.poll = pl;
        PolyData r = out(S.reduce(in(d)));
        S.poll = nullptr;
        return r;
    }
    PolyData coordinates(const PolyData &d, bool &ok, PollFn pl) {
        S.poll = pl;
        PolyData r = out(S.coordinates(in(d), ok));
        S.poll = nullptr;
        return r;
    }
    std::vector<uint32_t> pivots() const { return S.pivot_order; }
    PolyData row(uint32_t p) const { return out(S.rows.at(p).first); }
};

// The right kernel modulo p (a word-size prime) of the nrows x ncols matrix
// whose entries are the sums of the triplets (rows[t], cols[t], vals[t])
// (vals reduced mod p): by one dense reduced echelon form (FLINT
// nmod_mat_rref, in nthreads threads).  Returns one kernel vector per
// non-pivot column f: (f, [(pivot column, value)]), the vector being 1 at f,
// the values at the pivot columns and 0 elsewhere (the reduced echelon
// basis of the kernel for the non-pivot columns: canonical, so kernels
// modulo different primes with the same pivots can be combined by CRT).
using NmodKernel = std::vector<std::pair<int64_t, std::vector<std::pair<int64_t, uint64_t>>>>;

inline NmodKernel nmod_kernel(int64_t nrows, int64_t ncols, uint64_t p,
                              const int64_t *rows, const int64_t *cols,
                              const uint64_t *vals, size_t n, int nthreads) {
    if (nrows < 0 || ncols < 0 || p < 2)
        throw std::invalid_argument("nmod_kernel: bad dimensions or modulus");
    nmod_mat_t A;
    nmod_mat_init(A, nrows, ncols, p);
    for (size_t t = 0; t < n; ++t) {
        if (rows[t] < 0 || rows[t] >= nrows || cols[t] < 0 || cols[t] >= ncols) {
            nmod_mat_clear(A);
            throw std::invalid_argument("nmod_kernel: index out of range");
        }
        ulong *e = nmod_mat_entry_ptr(A, rows[t], cols[t]);
        *e = nmod_add(*e, vals[t] % p, A->mod);
    }
    int old = flint_get_num_threads();
    flint_set_num_threads(nthreads > 0 ? nthreads : 1);
    slong rank = nmod_mat_rref(A);
    flint_set_num_threads(old);
    std::vector<int64_t> piv;                  // pivot column of each row
    std::vector<char> is_piv(ncols, 0);
    for (slong i = 0, j = 0; i < rank; ++i) {
        while (j < ncols && nmod_mat_entry(A, i, j) == 0) ++j;
        piv.push_back(j);
        is_piv[j] = 1;
        ++j;
    }
    NmodKernel out;
    for (int64_t f = 0; f < ncols; ++f) {
        if (is_piv[f]) continue;
        std::vector<std::pair<int64_t, uint64_t>> v;
        for (slong i = 0; i < rank; ++i) {
            if (piv[i] > f) break;
            ulong a = nmod_mat_entry(A, i, f);
            if (a) v.emplace_back(piv[i], nmod_neg(a, A->mod));
        }
        out.emplace_back(f, std::move(v));
    }
    nmod_mat_clear(A);
    return out;
}

}  // namespace vae
