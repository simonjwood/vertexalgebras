// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// C++ engine of lattice vertex algebras: the free-field kernel of
// _lattice_core.pyx (letter series, E^+, annihilation parts,
// creation series, complete symmetric functions, E^-; see
// LatticeVertexAlgebra._letter_series) on interned monomials, with threads
// and the rest from EngineCore (_engine_base.hpp).  No Python objects
// in here: the Cython wrapper _lattice_engine.pyx converts at
// the boundary; the cocycle values eps(q, p) of a call come from Python.
//
//   Y(a, z) b = eps(q, p) z^{B(q,p)} E^-(q, z) W_{a,b}(z),
//
// a = :(Heisenberg letters) e^q:, b = :(letters) e^p:.  Everything in the
// letter series W and in the complete symmetric functions of E^- is rational (the Gram
// matrix is integral), so they are computed over QQ (fmpq) whatever the base
// ring; the base ring (the engine's Ring) enters through the coefficients
// of elements (the per-group sums of letter series) and eps.
//
// Letters (i, n) = d^n J_i are packed as code = n * r + (r-1-i) (code order
// = letter order: higher derivative first, then lower index first).  Three
// interned tables: Heisenberg monomials (weakly decreasing codes, id 0 = 1),
// lattice vectors (zigzag codes), pairs of keys; the engine's keys (the
// algebra's basis keys) are the lattice vector's codes followed by the
// letter codes.
#pragma once
#include "_engine_base.hpp"
#include <climits>
#include <functional>
#include <iterator>
#include <map>

namespace vae {

using Letters = std::vector<std::pair<int, int>>;   // (generator, derivative)

// cocycle values of a call: eps(qs[k], ps[k]) = vals[k]
struct EpsData {
    std::vector<std::vector<long>> qs, ps;
    std::vector<CoefData> vals;
};

// a sharded concurrent map uint64 -> V (memoised monomial operations)
template <class V>
struct CMap {
    static constexpr size_t NS = 64;
    struct Shard {
        std::mutex m;
        std::unordered_map<uint64_t, V> map;
    };
    std::unique_ptr<Shard[]> shards{new Shard[NS]};
    static size_t shard_of(uint64_t k) { return (size_t)((k * 0x9E3779B97F4A7C15ULL) >> 58); }
    bool get(uint64_t k, V &v) {
        Shard &s = shards[shard_of(k)];
        std::lock_guard<std::mutex> g(s.m);
        auto it = s.map.find(k);
        if (it == s.map.end()) return false;
        v = it->second;
        return true;
    }
    void put(uint64_t k, V v) {
        Shard &s = shards[shard_of(k)];
        std::lock_guard<std::mutex> g(s.m);
        s.map.emplace(k, v);
    }
    size_t size() const {
        size_t n = 0;
        for (size_t i = 0; i < NS; i++) n += shards[i].map.size();
        return n;
    }
};

// a per-thread direct-mapped front cache for memoised monomial operations
// (no locks on hits); entries belong to one engine (uid) and one table (tag)
struct TLSlot {
    uint64_t key = 0;     // 0 = empty (memo keys are never 0)
    uint32_t val = 0;
};
template <unsigned BITS>
struct TLCache {
    static constexpr size_t SIZE = (size_t)1 << BITS;
    uint64_t owner = 0;
    std::vector<TLSlot> slots;
    TLSlot &slot(uint64_t uid, uint64_t key) {
        if (owner != uid) {
            slots.assign(SIZE, TLSlot());
            owner = uid;
        }
        return slots[(size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - BITS))];
    }
};
inline thread_local TLCache<18> tl_hprod;    // monomial products (4 MB per thread)
inline thread_local TLCache<16> tl_full;     // (h, q) -> key
inline std::atomic<uint64_t> engine_uids{1};

inline uint32_t zigzag(long x) { return (uint32_t)(((uint64_t)x << 1) ^ (uint64_t)(x >> 63)); }
inline long unzigzag(uint32_t z) { return (long)(z >> 1) ^ -(long)(z & 1); }

// what the Cython wrapper sees
struct LatticeEngineBase {
    virtual ~LatticeEngineBase() {}
    virtual uint32_t intern(const Letters &letters, const std::vector<long> &q) = 0;
    virtual Letters key_letters(uint32_t id) = 0;
    virtual std::vector<long> key_q(uint32_t id) = 0;
    virtual PolesData ope_mono(uint32_t a, uint32_t b, const EpsData &eps, PollFn poll) = 0;
    virtual ElemData no_mono(uint32_t a, uint32_t b, const EpsData &eps, PollFn poll) = 0;
    virtual ElemData deriv_mono(uint32_t m, int n, PollFn poll) = 0;
    virtual PolesData ope_el(const ElemData &A, const ElemData &B, const EpsData &eps, int nthreads,
                             PollFn poll) = 0;
    virtual std::vector<PolesData> ope_el_many(const ElemData &A, const std::vector<ElemData> &Bs,
                                               const EpsData &eps, int nthreads, PollFn poll) = 0;
    virtual ElemData nprod_el(const ElemData &A, long n, const ElemData &B, const EpsData &eps,
                              int nthreads, PollFn poll) = 0;
    virtual ElemData deriv_el(const ElemData &A, int n, int nthreads, PollFn poll) = 0;
    virtual std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &els, const std::vector<int> &ia,
                                                const std::vector<long> &ns, const std::vector<int> &ib,
                                                const EpsData &eps, int nthreads, PollFn poll) = 0;
    virtual std::vector<size_t> cache_sizes() const = 0;
    // Engines of a module V_{λ+L}: the second vector of every product is a
    // module vector e^{λ+s} given by s; off = (B(x_i, λ))_i is added wherever
    // its lattice part enters (B(q, p), the zero modes B(x_i, p)).  Set it
    // before the first product (the caches assume one offset).
    virtual void set_offset(const std::vector<long> &off) = 0;
};

template <class Ring>
class LatticeEngine : public EngineCore<LatticeEngine<Ring>, Ring>, public LatticeEngineBase {
  public:
    using Core = EngineCore<LatticeEngine<Ring>, Ring>;
    using C = typename Core::C;
    using Elem = typename Core::Elem;
    using Poles = typename Core::Poles;
    using Job = typename Core::Job;
    using Part = typename Core::Part;
    using Core::R;
    using Core::monos;
    using Core::check;
    using Core::acc;
    using Core::add_scaled;
    using Core::single;
    using Core::intern_codes;
    using Core::serial;
    using Core::warm;
    using Core::sum_parts;
    using Core::pair_sum;
    using Core::run_parallel;
    using Core::run_tasks;
    using Core::check_tasks;
    using Core::elem_in;
    using Core::elem_out;
    using Core::poles_out;
    using Core::wire;

    // jobs (derivatives only; products are summed per group, see sums):
    // J_DERIV a = key, b = order
    enum : int { J_DERIV = 1 };

    using QE = RingQ::E;
    using QElem = std::unordered_map<uint32_t, QE>;   // Heisenberg monomial id -> rational
    using Series = std::map<long, QElem>;             // power of z -> polynomial
    using EpsMap = std::unordered_map<uint64_t, C>;   // pk(q id, p id) -> eps(q, p)

    uint64_t uid = engine_uids.fetch_add(1);   // owner of the per-thread caches
    int r;                                  // rank
    std::vector<std::vector<long>> B;       // Gram matrix
    std::vector<long> off;                  // B(x_i, λ) for a module V_{λ+L} (set_offset)
    bool has_off = false;
    RingQ Q;
    MonoTable hmons, qtab;                  // Heisenberg monomials, lattice vectors
    CMap<uint32_t> hprod_map, hdrop_map, letter_map, full_map;
    CMap<uint64_t> split_map;               // key -> pk(h, q)
    ConcCache<QElem> complete_h_cache;           // pk(q, k)
    ConcCache<Series> eplus_cache;          // pk(q, h of b)
    ConcCache<std::vector<QElem>> cser_cache;   // pk(h, smax)
    ConcCache<Elem> der_cache;              // pk(key, order)

    template <class... A>
    LatticeEngine(const std::vector<std::vector<long>> &gram, A &&...ring_args)
        : Core(std::forward<A>(ring_args)...), r((int)gram.size()), B(gram) {
        if (r < 1) throw std::invalid_argument("rank 0");
        for (const auto &row : B)
            if ((int)row.size() != r) throw std::invalid_argument("Gram matrix not square");
        monos.tails = hmons.tails = qtab.tails = false;
        hintern(std::vector<uint32_t>());       // h id 0 = the monomial 1
        wire(complete_h_cache);
        wire(eplus_cache);
        wire(cser_cache);
        wire(der_cache);
    }

    // -- lattice vectors and the form ---------------------------------------- //
    uint32_t qintern(const std::vector<long> &q) {
        if ((int)q.size() != r) throw std::invalid_argument("lattice vector of the wrong length");
        std::vector<uint32_t> v(q.size());
        for (size_t i = 0; i < q.size(); i++) v[i] = zigzag(q[i]);
        std::lock_guard<std::mutex> g(qtab.m);
        return qtab.intern_locked(v);
    }
    std::vector<long> qvec(uint32_t qid) const {
        const auto &v = qtab[qid].letters;
        std::vector<long> q(v.size());
        for (size_t i = 0; i < v.size(); i++) q[i] = unzigzag(v[i]);
        return q;
    }
    long form(const std::vector<long> &q, const std::vector<long> &p) const {
        long s = 0;
        for (int i = 0; i < r; i++)
            if (q[i])
                for (int j = 0; j < r; j++) s += q[i] * B[i][j] * p[j];
        return s;
    }
    // B(q, p) for p the lattice part of the SECOND vector of a product (a
    // module vector e^{λ+p} when an offset is set: B(q, λ + p))
    long pform(const std::vector<long> &q, const std::vector<long> &p) const {
        long s = form(q, p);
        if (has_off)
            for (int i = 0; i < r; i++) s += q[i] * off[i];
        return s;
    }
    void set_offset(const std::vector<long> &o) override {
        if ((int)o.size() != r) throw std::invalid_argument("offset of the wrong length");
        off = o;
        has_off = false;
        for (long x : off) has_off = has_off || x != 0;
    }
    long bxq(int i, const std::vector<long> &q) const {     // B(x_i, q)
        long s = 0;
        for (int j = 0; j < r; j++) s += B[i][j] * q[j];
        return s;
    }

    // -- Heisenberg monomials ------------------------------------------------- //
    uint32_t code(int i, int n) const { return (uint32_t)(n * r + (r - 1 - i)); }
    int lgen(uint32_t c) const { return r - 1 - (int)(c % (uint32_t)r); }
    int lder(uint32_t c) const { return (int)(c / (uint32_t)r); }
    const std::vector<uint32_t> &hletters(uint32_t h) const { return hmons[h].letters; }
    uint32_t hintern(const std::vector<uint32_t> &v) {
        std::lock_guard<std::mutex> g(hmons.m);
        return hmons.intern_locked(v);
    }
    uint32_t hletter(uint32_t c) {
        uint32_t v;
        if (letter_map.get(c, v)) return v;
        v = hintern(std::vector<uint32_t>{c});
        letter_map.put(c, v);
        return v;
    }
    uint32_t hmul(uint32_t x, uint32_t y) {      // commutative product
        if (x == 0) return y;
        if (y == 0) return x;
        uint64_t k = x < y ? pk(x, y) : pk(y, x);
        TLSlot &sl = tl_hprod.slot(uid, k);
        if (sl.key == k) return sl.val;
        uint32_t v;
        if (hprod_map.get(k, v)) {
            sl.key = k;
            sl.val = v;
            return v;
        }
        const auto &a = hletters(x), &b = hletters(y);
        std::vector<uint32_t> m;
        m.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(m), std::greater<uint32_t>());
        v = hintern(m);
        hprod_map.put(k, v);
        sl.key = k;
        sl.val = v;
        return v;
    }
    uint32_t hdrop(uint32_t x, uint32_t c) {     // one letter c less
        uint64_t k = pk(x, c);
        uint32_t v;
        if (hdrop_map.get(k, v)) return v;
        std::vector<uint32_t> m = hletters(x);
        auto it = std::find(m.begin(), m.end(), c);
        if (it == m.end()) throw std::logic_error("hdrop: letter not present");
        m.erase(it);
        v = hintern(m);
        hdrop_map.put(k, v);
        return v;
    }
    long degree(uint32_t h) const {          // sum of (n + 1) over the letters
        long d = 0;
        for (uint32_t c : hletters(h)) d += lder(c) + 1;
        return d;
    }

    // -- keys (letters, q) --------------------------------------------------- //
    uint32_t full(uint32_t h, uint32_t qid) {
        uint64_t k = pk(h, qid) + 1;              // + 1: never 0 (the empty slot)
        TLSlot &sl = tl_full.slot(uid, k);
        if (sl.key == k) return sl.val;
        uint32_t v;
        if (full_map.get(k, v)) {
            sl.key = k;
            sl.val = v;
            return v;
        }
        std::vector<uint32_t> m = qtab[qid].letters;
        const auto &l = hletters(h);
        m.insert(m.end(), l.begin(), l.end());
        v = intern_codes(m);
        split_map.put(v, k - 1);
        full_map.put(k, v);
        sl.key = k;
        sl.val = v;
        return v;
    }
    std::pair<uint32_t, uint32_t> split(uint32_t key) {    // (h, q) of a key
        uint64_t k;
        if (!split_map.get(key, k)) throw std::logic_error("unknown key");
        return {(uint32_t)(k >> 32), (uint32_t)k};
    }
    long nmax(uint32_t a, uint32_t b) {      // largest n with a_(n) b possibly nonzero
        auto sa = split(a), sb = split(b);
        return degree(sa.first) + degree(sb.first) - 1 - pform(qvec(sa.second), qvec(sb.second));
    }

    // -- rational helpers ------------------------------------------------------ //
    void qacc(QElem &dst, uint32_t k, QE &t) const {
        auto it = dst.find(k);
        if (it == dst.end()) {
            if (!Q.is_zero(t)) dst.emplace(k, std::move(t));
        } else {
            Q.add(it->second, t);
            if (Q.is_zero(it->second)) dst.erase(it);
        }
    }
    static void clean(Series &s) {
        for (auto it = s.begin(); it != s.end();)
            if (it->second.empty()) it = s.erase(it);
            else ++it;
    }
    static void set_frac(QE &x, const fmpz_t num, const fmpz_t den) { fmpq_set_fmpz_frac(x.p, num, den); }

    // -- kernels (the .pyx's, line by line) ---------------------------------- //
    // S_k(q) = h_k at p_m = q(-m): S_0 = 1, S_k = sum_{m=1..k} sum_i q_i / (k (m-1)!) x_{i,m-1} S_{k-m}
    const QElem &complete_h(uint32_t qid, long k) {
        return complete_h_cache.get(pk(qid, (uint32_t)k), [&]() -> QElem {
            check();
            QElem res;
            if (k == 0) {
                QE one(Q);
                Q.one(one);
                res.emplace(0, std::move(one));
                return res;
            }
            std::vector<long> q = qvec(qid);
            Z f, one(1);
            QE c0(Q), c(Q), t(Q);
            for (long m = 1; m <= k; m++) {
                const QElem &prev = complete_h(qid, k - m);
                if (prev.empty()) continue;
                fact(f.v, (ulong)(m - 1));
                fmpz_mul_si(f.v, f.v, k);
                set_frac(c0, one.v, f.v);
                for (int i = 0; i < r; i++) {
                    if (!q[i]) continue;
                    fmpq_mul_si(c.p, c0.p, q[i]);
                    uint32_t x = hletter(code(i, (int)m - 1));
                    for (const auto &kv : prev) {
                        Q.mul(t, c, kv.second);
                        qacc(res, hmul(kv.first, x), t);
                    }
                }
            }
            return res;
        });
    }

    // E^+(q, z) b: the exponential of the derivation removing a letter
    // (j, m-1) with factor -(m-1)! B(q, x_j) z^{-m}
    const Series &eplus(uint32_t qid, uint32_t hb) {
        return eplus_cache.get(pk(qid, hb), [&]() -> Series {
            check();
            std::vector<long> q = qvec(qid);
            std::vector<long> bq(r);
            for (int j = 0; j < r; j++) bq[j] = bxq(j, q);
            Series cur, res;
            QE one(Q);
            Q.one(one);
            cur[0].emplace(hb, one);
            res[0].emplace(hb, one);
            Z f, st;
            QE s(Q), t(Q);
            for (long step = 1; !cur.empty(); step++) {
                Series nxt;
                fmpz_set_si(st.v, step);
                for (const auto &pd : cur)
                    for (const auto &kv : pd.second) {
                        const auto &L = hletters(kv.first);
                        for (size_t t0 = 0; t0 < L.size();) {
                            uint32_t c = L[t0];
                            size_t t1 = t0;
                            while (t1 < L.size() && L[t1] == c) t1++;
                            long mult = (long)(t1 - t0);
                            t0 = t1;
                            long b = bq[lgen(c)];
                            if (!b) continue;
                            long m = lder(c) + 1;
                            fact(f.v, (ulong)(m - 1));
                            fmpz_mul_si(f.v, f.v, -mult * b);
                            set_frac(s, f.v, st.v);
                            Q.mul(t, kv.second, s);
                            qacc(nxt[pd.first - m], hdrop(kv.first, c), t);
                        }
                    }
                clean(nxt);
                for (const auto &pd : nxt) {
                    QElem &tgt = res[pd.first];
                    for (const auto &kv : pd.second) {
                        QE x = kv.second;
                        qacc(tgt, kv.first, x);
                    }
                }
                cur = std::move(nxt);
            }
            clean(res);
            return res;
        });
    }

    // the annihilation part of d^k J_i(z) applied to ser (zero mode included)
    Series ann(int i, long k, const Series &ser, const std::vector<long> &p) {
        const std::vector<long> &row = B[i];
        long sgn = (k & 1) ? -1 : 1;
        long c0 = has_off ? off[i] : 0;            // B(x_i, p) (+ B(x_i, λ))
        for (int j = 0; j < r; j++) c0 += row[j] * p[j];
        Z f;
        QE f0(Q), s(Q), t(Q);
        fact(f.v, (ulong)k);
        fmpz_mul_si(f.v, f.v, sgn * c0);
        fmpq_set_fmpz_frac(f0.p, f.v, Z(1).v);
        Series out;
        for (const auto &pd : ser) {
            long pw = pd.first;
            if (c0) {
                QElem &tgt = out[pw - 1 - k];
                for (const auto &kv : pd.second) {
                    Q.mul(t, kv.second, f0);
                    qacc(tgt, kv.first, t);
                }
            }
            for (const auto &kv : pd.second) {
                const auto &L = hletters(kv.first);
                for (size_t t0 = 0; t0 < L.size();) {
                    uint32_t c = L[t0];
                    size_t t1 = t0;
                    while (t1 < L.size() && L[t1] == c) t1++;
                    long mult = (long)(t1 - t0);
                    t0 = t1;
                    long bij = row[lgen(c)];
                    if (!bij) continue;
                    long m = lder(c) + 1;
                    fact(f.v, (ulong)(m + k));
                    fmpz_mul_si(f.v, f.v, mult * sgn * bij);
                    fmpq_set_fmpz_frac(s.p, f.v, Z(1).v);
                    Q.mul(t, kv.second, s);
                    qacc(out[pw - m - 1 - k], hdrop(kv.first, c), t);
                }
            }
        }
        clean(out);
        return out;
    }

    // creation series: prod over the letters (i, k) of sum_s x_{i,s+k} z^s / s!,
    // truncated at z^smax (index = power)
    const std::vector<QElem> &cseries(uint32_t h, long smax) {
        return cser_cache.get(pk(h, (uint32_t)smax), [&]() -> std::vector<QElem> {
            check();
            std::vector<QElem> ser((size_t)smax + 1);
            const auto &L = hletters(h);
            if (L.empty()) {
                QE one(Q);
                Q.one(one);
                ser[0].emplace(0, std::move(one));
                return ser;
            }
            uint32_t c = L[0];
            uint32_t rest = hintern(std::vector<uint32_t>(L.begin() + 1, L.end()));
            const std::vector<QElem> &rs = cseries(rest, smax);
            int i = lgen(c), k = lder(c);
            Z f, one(1);
            QE c1(Q), t(Q);
            for (long s = 0; s <= smax; s++) {
                QElem &d = ser[(size_t)s];
                for (long s1 = 0; s1 <= s; s1++) {
                    const QElem &src = rs[(size_t)(s - s1)];
                    if (src.empty()) continue;
                    fact(f.v, (ulong)s1);
                    set_frac(c1, one.v, f.v);
                    uint32_t x = hletter(code(i, (int)s1 + k));
                    for (const auto &kv : src) {
                        Q.mul(t, c1, kv.second);
                        qacc(d, hmul(kv.first, x), t);
                    }
                }
            }
            return ser;
        });
    }

    // d of a key: each letter's derivative raised (code + r), plus
    // q_i :J_i e^q: (the derivative of e^q)
    Elem d1(uint32_t key) {
        auto s = split(key);
        const auto &L = hletters(s.first);
        std::vector<long> q = qvec(s.second);
        Elem out;
        C t(R);
        for (size_t t0 = 0; t0 < L.size(); t0++) {
            std::vector<uint32_t> m = L;
            m.erase(m.begin() + (long)t0);
            uint32_t c = L[t0] + (uint32_t)r;
            m.insert(std::upper_bound(m.begin(), m.end(), c, std::greater<uint32_t>()), c);
            R.one(t);
            acc(out, full(hintern(m), s.second), t);
        }
        for (int i = 0; i < r; i++) {
            if (!q[i]) continue;
            C u(R);
            R.one(u);
            R.mul_si(t, u, q[i]);
            acc(out, full(hmul(s.first, hletter(code(i, 0))), s.second), t);
        }
        return out;
    }
    const Elem &deriv_c(uint32_t key, int n) {
        return der_cache.get(pk(key, (uint32_t)n), [&]() -> Elem { return deriv_compute(key, n); });
    }
    Elem deriv_compute(uint32_t key, int n) {
        check();
        if (n == 0) return single(key);
        Elem res;
        const Elem &prev = deriv_c(key, n - 1);
        for (const auto &kv : prev) add_scaled(res, d1(kv.first), kv.second);
        return res;
    }

    // -- jobs ------------------------------------------------------------------ //
    void run_job(const Job &j) { deriv_c(j.a, (int)j.b); }
    void run_job_if_absent(const Job &j) {
        der_cache.get_if_absent(pk(j.a, j.b), [&]() -> Elem { return deriv_compute(j.a, (int)j.b); });
    }
    bool job_cached(const Job &j) { return der_cache.ready(pk(j.a, j.b)); }
    size_t job_size(const Job &j) const { return monos[j.a].letters.size(); }

    // -- products: res[n] = sum over the pairs of c * a_(n) b -------------------- //
    struct Item {
        uint32_t a, b;
        const C *c;
    };
    EpsMap eps_in(const EpsData &d) {
        EpsMap m;
        for (size_t k = 0; k < d.vals.size(); k++) {
            C c(R);
            R.from_data(c, d.vals[k]);
            m.emplace(pk(qintern(d.qs[k]), qintern(d.ps[k])), std::move(c));
        }
        return m;
    }
    // the pairs grouped by lattice parts (q, p), with the n needed
    struct Group {
        uint32_t q, p;
        std::vector<long> gns;
        std::vector<Item> items;
    };
    std::vector<Group> plan(const std::vector<Item> &items, const std::vector<long> &ns) {
        std::map<uint64_t, std::vector<Item>> by;
        for (const Item &it : items) by[pk(split(it.a).second, split(it.b).second)].push_back(it);
        std::vector<Group> out;
        for (auto &kv : by) {
            long top = LONG_MIN;
            for (const Item &it : kv.second) top = std::max(top, nmax(it.a, it.b));
            Group g;
            g.q = (uint32_t)(kv.first >> 32);
            g.p = (uint32_t)kv.first;
            for (long n : ns)
                if (n <= top) g.gns.push_back(n);
            if (g.gns.empty()) continue;
            g.items = std::move(kv.second);
            out.push_back(std::move(g));
        }
        return out;
    }
    using SerC = std::map<long, Elem>;       // a series over the engine's ring (h ids)

    // sum over the sub-multisets S of a's letters of
    //   f_S (sum_b c_b ann_S(E^+(q) b)) * C_{Lc(S)}(z)   up to z^wmax,
    // added to dst: the sum over b comes BEFORE the product with the
    // creation series C, which depends on a only (one convolution per (a, S)
    // instead of per (a, b, S)).  Identical letters of a: sub-multisets
    // weighted by binomials.
    void a_series(uint32_t a, const std::vector<std::pair<uint32_t, const C *>> &bs, uint32_t qid,
                  const std::vector<long> &p, long wmax, SerC &dst) {
        check();
        const auto &La = hletters(split(a).first);
        std::vector<std::pair<uint32_t, long>> groups;
        for (uint32_t c : La) {
            if (!groups.empty() && groups.back().first == c) groups.back().second++;
            else groups.emplace_back(c, 1);
        }
        using SP = std::shared_ptr<const Series>;
        struct Sub {
            std::vector<long> counts;
            std::vector<SP> sers;          // per b (null: zero)
            QE f;
        };
        QE one(Q);
        Q.one(one);
        Sub s0{{}, {}, one};
        for (const auto &bc : bs) {
            const Series &E0 = eplus(qid, split(bc.first).first);
            s0.sers.push_back(E0.empty() ? SP() : SP(&E0, [](const Series *) {}));
        }
        std::vector<Sub> subsets{s0};
        Z bin;
        for (const auto &g : groups) {
            std::vector<Sub> nxt;
            for (const Sub &sub : subsets) {
                Sub z = sub;
                z.counts.push_back(0);
                nxt.push_back(z);
                std::vector<SP> cur = sub.sers;
                for (long j = 1; j <= g.second; j++) {
                    bool any = false;
                    for (SP &sp : cur) {
                        if (!sp) continue;
                        auto an = std::make_shared<Series>(ann(lgen(g.first), lder(g.first), *sp, p));
                        sp = an->empty() ? SP() : SP(an);
                        any = any || (bool)sp;
                    }
                    if (!any) break;
                    Sub sj{sub.counts, cur, QE(Q)};
                    sj.counts.push_back(j);
                    comb(bin.v, (ulong)g.second, (ulong)j);
                    fmpq_mul_fmpz(sj.f.p, sub.f.p, bin.v);
                    nxt.push_back(std::move(sj));
                }
            }
            subsets = std::move(nxt);
        }
        C x(R), t1(R), t(R);
        for (const Sub &sub : subsets) {
            check();
            SerC cb;                          // sum over b of c_b ann_S(E^+ b)
            for (size_t k = 0; k < bs.size(); k++) {
                if (!sub.sers[k]) continue;
                for (const auto &pd : *sub.sers[k]) {
                    if (pd.first > wmax) break;
                    Elem &d = cb[pd.first];
                    for (const auto &kv : pd.second) {
                        R.mul_fmpq(x, *bs[k].second, kv.second.p);
                        acc(d, kv.first, x);
                    }
                }
            }
            for (auto it = cb.begin(); it != cb.end();)
                if (it->second.empty()) it = cb.erase(it);
                else ++it;
            if (cb.empty()) continue;
            std::vector<uint32_t> lc;
            for (size_t gi = 0; gi < groups.size(); gi++)
                for (long j = 0; j < groups[gi].second - sub.counts[gi]; j++) lc.push_back(groups[gi].first);
            long smax = wmax - cb.begin()->first;
            if (smax < 0) continue;
            const std::vector<QElem> &Cs = cseries(hintern(lc), smax);
            for (const auto &pd : cb) {
                long pw = pd.first;
                check();
                for (long s = 0; s <= wmax - pw; s++) {
                    const QElem &dC = Cs[(size_t)s];
                    if (dC.empty()) continue;
                    Elem &dW = dst[pw + s];
                    for (const auto &ka : pd.second) {
                        R.mul_fmpq(t1, ka.second, sub.f.p);
                        for (const auto &kc : dC) {
                            R.mul_fmpq(t, t1, kc.second.p);
                            acc(dW, hmul(ka.first, kc.first), t);
                        }
                    }
                }
            }
        }
    }

    // per (q, p) group: Wsum = sum of the a_series of its a's (threads: the
    // a's, larger first, dynamically), then eps E^- for each n (threads: the
    // Heisenberg monomials of Wsum, id % T).  Only sums are merged (exact:
    // the same result for any number of threads).
    // one (q, p) group: the coefficients of z^{-n-1} for its n, before the
    // cocycle value (threads inside the group: the a's, then the Heisenberg
    // monomials of Wsum)
    std::map<long, Elem> group_sum(const Group &g, int nthreads) {
        std::vector<long> q = qvec(g.q), p = qvec(g.p);
        long Bqp = pform(q, p);
        long nmin = *std::min_element(g.gns.begin(), g.gns.end());
        long wmax = -nmin - 1 - Bqp;
        std::vector<long> qp(r);
        for (int i = 0; i < r; i++) qp[i] = q[i] + p[i];
        uint32_t qpid = qintern(qp);
        std::map<uint32_t, std::vector<std::pair<uint32_t, const C *>>> by_a;
        size_t big = 0;
        for (const Item &it : g.items) {
            by_a[it.a].emplace_back(it.b, it.c);
            big = std::max(big, (size_t)(monos[it.a].letters.size() + monos[it.b].letters.size()));
        }
        std::vector<std::pair<uint32_t, const std::vector<std::pair<uint32_t, const C *>> *>> as;
        for (const auto &kv : by_a) as.emplace_back(kv.first, &kv.second);
        std::stable_sort(as.begin(), as.end(), [this](const auto &x, const auto &y) {
            return monos[x.first].letters.size() > monos[y.first].letters.size();
        });
        bool threaded = nthreads > 1 && as.size() > 1 &&
                        (g.items.size() >= MIN_THREADED_JOBS || big >= MIN_THREADED_LETTERS + (size_t)r);
        uint32_t T1 = threaded ? (uint32_t)std::min((size_t)nthreads, as.size()) : 1;
        std::vector<SerC> part(T1);
        std::atomic<size_t> next{0};
        this->run_parallel((int)T1, [&](int t) {
            for (;;) {
                size_t k = next.fetch_add(1);
                if (k >= as.size()) break;
                a_series(as[k].first, *as[k].second, g.q, p, wmax, part[(size_t)t]);
            }
        });
        SerC Wsum = std::move(part[0]);
        for (uint32_t t = 1; t < T1; t++)
            for (auto &pd : part[t]) {
                Elem &dst = Wsum[pd.first];
                for (auto &kv : pd.second) acc(dst, kv.first, kv.second);
            }
        size_t work = 0;
        for (const auto &pd : Wsum) work += pd.second.size();
        uint32_t T = (nthreads > 1 && work >= PAR_SUM_TERMS / 8) ? (uint32_t)nthreads : 1;
        std::vector<std::map<long, Elem>> outs(T);
        this->run_parallel((int)T, [&](int t) {
            C x(R);
            for (long n : g.gns) {
                Elem &out = outs[(size_t)t][n];
                for (const auto &pd : Wsum) {
                    long sig = -n - 1 - Bqp - pd.first;
                    if (sig < 0 || pd.second.empty()) continue;
                    const QElem &S = complete_h(g.q, sig);
                    for (const auto &kw : pd.second) {
                        if (kw.first % T != (uint32_t)t) continue;
                        for (const auto &ks : S) {
                            R.mul_fmpq(x, kw.second, ks.second.p);
                            acc(out, full(hmul(kw.first, ks.first), qpid), x);
                        }
                    }
                }
            }
        });
        std::map<long, Elem> res;
        for (long n : g.gns) {
            Elem part_n = std::move(outs[0][n]);
            for (uint32_t t = 1; t < T; t++)
                for (auto &kv : outs[t][n]) acc(part_n, kv.first, kv.second);
            res[n] = std::move(part_n);
        }
        return res;
    }

    // per (q, p) group the sum of its products, times eps(q, p).  Many groups
    // (at least 2 per thread): the threads take whole groups (largest first,
    // each computed serially), so that a sum of many small groups does not
    // start threads per group; few groups: threads inside each group.  The
    // groups are added in their order (exact: the same result for any number
    // of threads).
    std::map<long, Elem> sums(const std::vector<Group> &groups, const EpsMap &eps, int nthreads) {
        std::vector<const C *> scale(groups.size());
        for (size_t k = 0; k < groups.size(); k++) {
            auto ie = eps.find(pk(groups[k].q, groups[k].p));
            if (ie == eps.end()) throw std::invalid_argument("missing cocycle value");
            scale[k] = &ie->second;
        }
        std::map<long, Elem> res;
        if (nthreads > 1 && groups.size() >= 2 * (size_t)nthreads) {
            std::vector<size_t> order(groups.size());
            for (size_t k = 0; k < order.size(); k++) order[k] = k;
            std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
                return groups[x].items.size() > groups[y].items.size();
            });
            std::vector<std::map<long, Elem>> parts(groups.size());
            std::atomic<size_t> next{0};
            this->run_parallel(nthreads, [&](int) {
                for (;;) {
                    size_t k = next.fetch_add(1);
                    if (k >= order.size()) break;
                    parts[order[k]] = group_sum(groups[order[k]], 1);
                }
            });
            for (size_t k = 0; k < groups.size(); k++)
                for (auto &kv : parts[k]) add_scaled(res[kv.first], kv.second, *scale[k]);
            return res;
        }
        for (size_t k = 0; k < groups.size(); k++)
            for (auto &kv : group_sum(groups[k], nthreads)) add_scaled(res[kv.first], kv.second, *scale[k]);
        return res;
    }
    std::map<long, Elem> products(const std::vector<Item> &items, const std::vector<long> &ns, const EpsMap &eps,
                                  int nthreads, PollFn poll) {
        std::vector<Group> groups = plan(items, ns);
        std::map<long, Elem> res;
        serial([&] { res = sums(groups, eps, nthreads); }, poll);
        return res;
    }
    // the pairs of A x B with their coefficients (kept alive in scal)
    void pairs_of(const Elem &A, const Elem &B, std::vector<Item> &items, std::deque<C> &scal) {
        for (const auto &a : A)
            for (const auto &b : B) {
                scal.emplace_back(R);
                R.mul(scal.back(), a.second, b.second);
                items.push_back(Item{a.first, b.first, &scal.back()});
            }
    }
    long nmax_of(const std::vector<Item> &items) {
        long N = -1;
        for (const Item &it : items) N = std::max(N, nmax(it.a, it.b));
        return N;
    }
    static Poles to_poles(std::map<long, Elem> &res, long N) {
        Poles P(N >= 0 ? (size_t)N + 1 : 0);
        for (auto &kv : res)
            if (kv.first >= 0 && kv.first <= N) P[(size_t)kv.first] = std::move(kv.second);
        return P;
    }

    // -- interface ------------------------------------------------------------ //
    uint32_t intern(const Letters &letters, const std::vector<long> &q) override {
        std::vector<uint32_t> m;
        for (const auto &l : letters) {
            if (l.first < 0 || l.first >= r || l.second < 0) throw std::invalid_argument("bad letter");
            m.push_back(code(l.first, l.second));
        }
        for (size_t k = 1; k < m.size(); k++)
            if (m[k - 1] < m[k]) throw std::invalid_argument("letters not in canonical order");
        return full(hintern(m), qintern(q));
    }
    Letters key_letters(uint32_t id) override {
        Letters out;
        for (uint32_t c : hletters(split(id).first)) out.emplace_back(lgen(c), lder(c));
        return out;
    }
    std::vector<long> key_q(uint32_t id) override { return qvec(split(id).second); }

    PolesData ope_mono(uint32_t a, uint32_t b, const EpsData &eps, PollFn poll) override {
        EpsMap em = eps_in(eps);
        long N = nmax(a, b);
        if (N < 0) return PolesData();
        C one(R);
        R.one(one);
        std::vector<Item> items{Item{a, b, &one}};
        std::vector<long> ns;
        for (long n = 0; n <= N; n++) ns.push_back(n);
        std::map<long, Elem> res = products(items, ns, em, 1, poll);
        Poles P = to_poles(res, N);
        while (!P.empty() && P.back().empty()) P.pop_back();
        return poles_out(P);
    }
    ElemData no_mono(uint32_t a, uint32_t b, const EpsData &eps, PollFn poll) override {
        EpsMap em = eps_in(eps);
        C one(R);
        R.one(one);
        std::vector<Item> items{Item{a, b, &one}};
        std::map<long, Elem> res = products(items, {-1}, em, 1, poll);
        return elem_out(res[-1]);
    }
    ElemData deriv_mono(uint32_t m, int n, PollFn poll) override {
        if (n < 0) throw std::invalid_argument("negative derivative order");
        const Elem *E = nullptr;
        serial([&] { E = &deriv_c(m, n); }, poll);
        return elem_out(*E);
    }
    PolesData ope_el(const ElemData &dA, const ElemData &dB, const EpsData &eps, int nthreads,
                     PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA), B = elem_in(dB);
        EpsMap em = eps_in(eps);
        std::vector<Item> items;
        std::deque<C> scal;
        pairs_of(A, B, items, scal);
        long N = nmax_of(items);
        if (N < 0) return PolesData();
        std::vector<long> ns;
        for (long n = 0; n <= N; n++) ns.push_back(n);
        std::map<long, Elem> res = products(items, ns, em, nthreads, poll);
        return poles_out(to_poles(res, N));
    }
    std::vector<PolesData> ope_el_many(const ElemData &dA, const std::vector<ElemData> &dBs, const EpsData &eps,
                                       int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA);
        EpsMap em = eps_in(eps);
        std::vector<PolesData> out;
        for (const ElemData &dB : dBs) {
            Elem B = elem_in(dB);
            std::vector<Item> items;
            std::deque<C> scal;
            pairs_of(A, B, items, scal);
            long N = nmax_of(items);
            if (N < 0) {
                out.push_back(PolesData());
                continue;
            }
            std::vector<long> ns;
            for (long n = 0; n <= N; n++) ns.push_back(n);
            std::map<long, Elem> res = products(items, ns, em, nthreads, poll);
            out.push_back(poles_out(to_poles(res, N)));
        }
        return out;
    }
    ElemData nprod_el(const ElemData &dA, long n, const ElemData &dB, const EpsData &eps, int nthreads,
                      PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        Elem A = elem_in(dA), B = elem_in(dB);
        EpsMap em = eps_in(eps);
        std::vector<Item> items;
        std::deque<C> scal;
        pairs_of(A, B, items, scal);
        std::map<long, Elem> res = products(items, {n}, em, nthreads, poll);
        return elem_out(res[n]);
    }
    // [els[ia[k]]_(ns[k]) els[ib[k]] for every k] in one call: each product
    // computed serially in one of nthreads threads (the products of a
    // subalgebra span: many small ones, where splitting one product over
    // the a's of its groups gains little), larger first (by the longest
    // key of A, then |A| |B|)
    std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &dels, const std::vector<int> &ia,
                                        const std::vector<long> &ns, const std::vector<int> &ib,
                                        const EpsData &eps, int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        check_tasks(dels.size(), ia, ns, ib);
        std::vector<Elem> E;
        E.reserve(dels.size());
        for (const ElemData &d : dels) E.push_back(elem_in(d));
        EpsMap em = eps_in(eps);
        std::vector<size_t> len(E.size(), 0);
        for (size_t i = 0; i < E.size(); i++)
            for (const auto &a : E[i]) len[i] = std::max(len[i], monos[a.first].letters.size());
        std::vector<size_t> order = Core::task_order(ns.size(), [&](size_t k) {
            return std::make_pair(len[(size_t)ia[k]], E[(size_t)ia[k]].size() * E[(size_t)ib[k]].size());
        });
        std::vector<Elem> res(ns.size());
        run_tasks(order, nthreads, poll, [&](size_t k) {
            std::vector<Item> items;
            std::deque<C> scal;
            pairs_of(E[(size_t)ia[k]], E[(size_t)ib[k]], items, scal);
            std::map<long, Elem> r = sums(plan(items, {ns[k]}), em, 1);
            res[k] = std::move(r[ns[k]]);
        });
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
            for (const auto &a : A) add_scaled(res, deriv_c(a.first, n), a.second);
        }, poll);
        return elem_out(res);
    }
    std::vector<size_t> cache_sizes() const override {
        std::vector<size_t> s = this->sched_stats();
        s.insert(s.begin(), {eplus_cache.size(), cser_cache.size(), complete_h_cache.size(),
                             der_cache.size(), (size_t)hmons.n, (size_t)monos.n, hprod_map.size()});
        return s;
    }
};

// kind: "q", "qpoly", "qmpoly" (nvars >= 1), "nmod" (modulus a prime < 2^63)
// minpoly: the defining polynomial of the number field ("nf", "nfmpoly")
inline LatticeEngineBase *make_lattice_engine(const std::string &kind, const std::vector<std::vector<long>> &gram,
                                              long nvars, unsigned long modulus, const CoefData &minpoly) {
    if (kind == "nf") return new LatticeEngine<RingNF>(gram, minpoly);
    if (kind == "nfmpoly") return new LatticeEngine<RingNFMPoly>(gram, (slong)nvars, minpoly);
    if (kind == "q") return new LatticeEngine<RingQ>(gram);
    if (kind == "qpoly") return new LatticeEngine<RingQPoly>(gram);
    if (kind == "qmpoly") {
        if (nvars < 1) throw std::invalid_argument("qmpoly needs at least one variable");
        return new LatticeEngine<RingQMPoly>(gram, (slong)nvars);
    }
    if (kind == "nmod") {
        if (modulus < 2 || modulus > (unsigned long)WORD_MAX || !n_is_prime(modulus))
            throw std::invalid_argument("nmod needs a prime modulus below 2^63");
        return new LatticeEngine<RingNmod>(gram, (ulong)modulus);
    }
    throw std::invalid_argument("unknown coefficient kind " + kind);
}

}  // namespace vae
