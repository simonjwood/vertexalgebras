// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// C++ engine of induced modules: the module M(U) of a universal vertex
// algebra V induced from a finite-dimensional module U over its Zhu algebra
// (vertex_algebra_module.InducedModule), on interned keys with FLINT
// coefficients, threads and all the rest from EngineCore (_engine_base.hpp).
// No Python objects in here: the Cython wrapper _module_engine.pyx converts
// at the boundary.  The recursions are those of InducedModule (_gen_act,
// _commutator_on) and _GeneratorActions (_action_key), line by line, except
// that the raising modes of a word's first letter act last when the products
// x_(i) B are given (set_splits, word_act_compute).  For a proto vertex
// algebra the two orders may differ before the result is stable in the
// relations' weight; why the induced module's guarantee holds either way:
// vertex_algebra_module._mode_split.
//
// Rational quantities are scaled by D (the common denominator of the
// generators' conformal weights h_i): h2[i] = D h_i, a physics mode m is
// m2 = D m (m in ZZ - h_i), the depth of a key is depth2 / D.
//
// A key g_{i_1, m_1} ... g_{i_k, m_k} u (creation modes in the mode order:
// m ascending, then the generator; odd modes at most once) is the code
// vector [c_1, ..., c_k, u] with c = (-m2) G + i, interned in the core's
// table (its tail is the key without the first mode).  A word (a basis
// monomial of V, letters (generator, derivative), first letter outermost)
// is the code vector of the universal engine (der G + (G-1-gen)) in a
// second table.
#pragma once
#include "_engine_base.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace vae {

using WordLetters = std::vector<std::pair<int, int>>;     // (generator, derivative)
using ModeList = std::vector<std::pair<int, long>>;       // (generator, D * physics mode)

// what the Cython wrapper sees
struct ModuleEngineBase {
    virtual ~ModuleEngineBase() {}
    virtual uint32_t intern_word(const WordLetters &w) = 0;
    virtual WordLetters word_letters(uint32_t id) const = 0;
    virtual uint32_t intern_key(const ModeList &modes, int u) = 0;
    virtual std::pair<ModeList, int> key_of(uint32_t id) const = 0;
    virtual void set_top(int dim, const std::vector<int> &parities) = 0;
    virtual void set_zero_mode(int i, const std::vector<std::vector<std::pair<int, CoefData>>> &columns) = 0;
    virtual void set_top_degrees(const std::vector<long> &degrees2) = 0;
    virtual void set_top_action(int i, long m2, int u, const ElemData &value) = 0;
    // He's free module P = U / sum_{i > n} U U_{-i} of the higher Zhu
    // algebras (higher_zhu_algebra._FreeModeModule): one top vector v of
    // degree n2 / D, EVERY mode kept in the word (no top action), a word of
    // negative degree is zero, and an all-lowering word lowering by more than
    // n is zero
    virtual void set_free(long n2) = 0;
    virtual void set_ope(int i, int j, const PolesData &poles) = 0;
    // the commuted form of a word w = :x B: (word_act_compute): x_(i) B for
    // i = 0, ..., r - 1 (r = raising(x): the x_(j) with wt x - j - 1 > 0),
    // elements over word ids
    virtual void set_splits(uint32_t w, const std::vector<ElemData> &xb) = 0;
    virtual std::vector<ElemData> act_many(const std::vector<ElemData> &els, const std::vector<ElemData> &vecs,
                                           const std::vector<int> &ie, const std::vector<long> &ns,
                                           const std::vector<int> &iv, int nthreads, PollFn poll) = 0;
    virtual std::vector<size_t> cache_sizes() const = 0;
    virtual std::vector<size_t> word_stats() const = 0;
    virtual std::vector<size_t> cache_profile(int which) const = 0;
    virtual std::vector<size_t> coefficient_stats(int which) const = 0;
};

// floor(a / b) for b > 0
inline long floor_div(long a, long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }

// The distinct rational values of a CompactCache, each stored once, packed
// (W(2,3), P(0) at depth 5: values are ~28% of the terms, with numerators of
// median 54 and mean 96 bits; as fmpq each term took ~78 bytes, 20 + the
// GMP integers and their allocations).  intern(q) gives a 32-bit index (the
// shard in the top 6 bits), get(i, q) decodes.  A value is the bytes
//   varint(2 * numerator length + sign), numerator, varint(denominator length), denominator
// (lengths in bytes, little endian; denominator length 0 = 1), canonical as
// fmpq is, so equal values have equal bytes.  Values live in per-shard byte
// arenas, never moved; get reads without a lock: an index reaches a reader
// only through a cache record published under that cache's lock, after the
// value was written under the table's.
struct QValueTable {
    static constexpr size_t NS = 64, ARENA = 1u << 20, BLOCK = 1u << 16, NBLOCKS = 1u << 10;
    static constexpr uint32_t LOCAL = 1u << 26;    // values per shard (NBLOCKS * BLOCK)
    struct Shard {
        std::mutex m;
        std::vector<uint64_t> table;               // open addressing: hash32 << 32 | (local + 1), 0 empty
        uint32_t count = 0;
        std::vector<std::unique_ptr<uint8_t[]>> arenas;
        size_t used = ARENA, bytes = 0;            // in the last arena
        std::unique_ptr<std::unique_ptr<const uint8_t *[]>[]> blocks{new std::unique_ptr<const uint8_t *[]>[NBLOCKS]};
    };
    std::unique_ptr<Shard[]> shards{new Shard[NS]};

    static void put_varint(std::vector<uint8_t> &b, size_t x) {
        while (x >= 0x80) {
            b.push_back((uint8_t)(x | 0x80));
            x >>= 7;
        }
        b.push_back((uint8_t)x);
    }
    static size_t get_varint(const uint8_t *&p) {
        size_t x = 0;
        for (int sh = 0;; sh += 7) {
            uint8_t c = *p++;
            x |= (size_t)(c & 0x7F) << sh;
            if (!(c & 0x80)) return x;
        }
    }
    // |z| as little-endian bytes, without leading zeros
    static void abs_bytes(std::vector<uint8_t> &out, const fmpz_t z) {
        out.clear();
        if (!COEFF_IS_MPZ(*z)) {
            uint64_t u = (uint64_t)(*z < 0 ? -*z : *z);
            while (u) {
                out.push_back((uint8_t)u);
                u >>= 8;
            }
            return;
        }
        const __mpz_struct *m = COEFF_TO_PTR(*z);
        int n = std::abs(m->_mp_size);
        for (int i = 0; i < n; i++) {
            uint64_t l = (uint64_t)m->_mp_d[i];
            for (int b = 0; b < 8; b++) out.push_back((uint8_t)(l >> (8 * b)));
        }
        while (!out.empty() && !out.back()) out.pop_back();
    }
    static void encode(std::vector<uint8_t> &out, const fmpq_t q) {
        thread_local std::vector<uint8_t> nb, db;
        abs_bytes(nb, fmpq_numref(q));
        if (fmpz_is_one(fmpq_denref(q)))
            db.clear();
        else
            abs_bytes(db, fmpq_denref(q));
        out.clear();
        put_varint(out, 2 * nb.size() + (fmpz_sgn(fmpq_numref(q)) < 0));
        out.insert(out.end(), nb.begin(), nb.end());
        put_varint(out, db.size());
        out.insert(out.end(), db.begin(), db.end());
    }
    static void set_bytes(fmpz_t z, const uint8_t *p, size_t len) {
        if (len <= 8) {
            uint64_t u = 0;
            for (size_t b = 0; b < len; b++) u |= (uint64_t)p[b] << (8 * b);
            fmpz_set_ui(z, (ulong)u);
            return;
        }
        thread_local std::vector<ulong> limbs;
        limbs.assign((len + 7) / 8, 0);
        for (size_t b = 0; b < len; b++) limbs[b / 8] |= (ulong)p[b] << (8 * (b % 8));
        fmpz_set_ui_array(z, limbs.data(), (slong)limbs.size());
    }
    static size_t encoded_size(const uint8_t *p) {
        const uint8_t *q = p;
        size_t nl = get_varint(q) >> 1;
        q += nl;
        size_t dl = get_varint(q);
        return (size_t)(q - p) + dl;
    }
    static uint64_t hash_bytes(const std::vector<uint8_t> &b) {
        uint64_t h = 0xCBF29CE484222325ULL;
        for (uint8_t c : b) h = (h ^ c) * 0x100000001B3ULL;
        return h ^ (h >> 31);
    }
    const uint8_t *ptr(uint32_t i) const {
        const Shard &s = shards[i >> 26];
        uint32_t l = i & (LOCAL - 1);
        return s.blocks[l / BLOCK][l % BLOCK];
    }
    void get(uint32_t i, fmpq_t q) const {
        const uint8_t *p = ptr(i);
        size_t ns = get_varint(p);
        set_bytes(fmpq_numref(q), p, ns >> 1);
        if (ns & 1) fmpz_neg(fmpq_numref(q), fmpq_numref(q));
        p += ns >> 1;
        size_t dl = get_varint(p);
        if (dl)
            set_bytes(fmpq_denref(q), p, dl);
        else
            fmpz_one(fmpq_denref(q));
    }
    uint32_t intern(const fmpq_t q) {
        thread_local std::vector<uint8_t> b;
        encode(b, q);
        uint64_t h = hash_bytes(b);
        size_t si = (size_t)(h >> 58);
        uint32_t h32 = (uint32_t)h;
        Shard &s = shards[si];
        std::lock_guard<std::mutex> g(s.m);
        if ((size_t)(s.count + 1) * 10 > s.table.size() * 7) {   // load <= 0.7
            size_t n = std::max<size_t>(64, s.table.size() * 2);
            std::vector<uint64_t> old;
            old.swap(s.table);
            s.table.assign(n, 0);
            for (uint64_t e : old)
                if (e) {
                    size_t j = (size_t)(e >> 32) & (n - 1);
                    while (s.table[j]) j = (j + 1) & (n - 1);
                    s.table[j] = e;
                }
        }
        size_t mask = s.table.size() - 1, j = h32 & mask;
        for (;; j = (j + 1) & mask) {
            uint64_t e = s.table[j];
            if (!e) break;
            if ((uint32_t)(e >> 32) != h32) continue;
            uint32_t l = (uint32_t)e - 1;
            const uint8_t *p = s.blocks[l / BLOCK][l % BLOCK];
            if (encoded_size(p) == b.size() && !std::memcmp(p, b.data(), b.size()))
                return (uint32_t)(si << 26) | l;
        }
        uint32_t l = s.count;
        if (l >= LOCAL) throw std::length_error("value table shard full");
        if (s.used + b.size() > ARENA || s.arenas.empty()) {
            s.arenas.emplace_back(new uint8_t[std::max(ARENA, b.size())]);
            s.used = 0;
        }
        uint8_t *dst = s.arenas.back().get() + s.used;
        std::memcpy(dst, b.data(), b.size());
        s.used += b.size();
        if (b.size() > ARENA) s.used = ARENA;     // a value of its own: the next starts a new arena
        s.bytes += b.size();
        if (!s.blocks[l / BLOCK]) s.blocks[l / BLOCK].reset(new const uint8_t *[BLOCK]);
        s.blocks[l / BLOCK][l % BLOCK] = dst;
        s.table[j] = ((uint64_t)h32 << 32) | (uint64_t)(l + 1);
        s.count++;
        return (uint32_t)(si << 26) | l;
    }
    // (values, bytes of the packed values)
    std::pair<size_t, size_t> sizes() const {
        size_t n = 0, b = 0;
        for (size_t i = 0; i < NS; i++) {
            std::lock_guard<std::mutex> g(shards[i].m);
            n += shards[i].count;
            b += shards[i].bytes;
        }
        return {n, b};
    }
};

// A memo shared by threads, as ConcCache (get(key, compute) computes each
// value once; other threads wait; a failed computation is redone by a later
// call, its waiters throw EntryFailed), but compact: the values (lists of
// (key id, coefficient)) live in per-shard arenas of fixed chunks, keys and
// coefficients in parallel arrays, never moved (views stay valid), indexed
// by an open-addressing table and 16-byte records.  About 40 bytes per entry
// and 20 per term (sizeof(fmpq) + 4), against ~190 and ~32 with ConcCache
// of vectors (map node, shared_ptr, control block, vector header, malloc
// padding): the module engine is limited by memory (W(2,3), P(0) at depth 5).
// Over QQ the coefficients are interned instead (QValueTable; unless
// VERTEX_ALGEBRAS_INTERN=0 when the cache is made): a term is then the key
// and a 32-bit value index, 8 bytes, plus each distinct value once; read
// with View::at, which decodes into the caller's temporary.
template <class Ring>
struct CompactCache {
    using C = typename Ring::E;
    using Flat = std::vector<std::pair<uint32_t, C>>;
    static constexpr bool CAN_INTERN = std::is_same<Ring, RingQ>::value;
    struct View {
        const uint32_t *k = nullptr;
        const C *c = nullptr;                      // the coefficients, or
        const uint32_t *v = nullptr;               // their value indices in
        const QValueTable *vt = nullptr;           // this table
        uint32_t n = 0;
        bool empty() const { return n == 0; }
        // the j-th coefficient: stored, or decoded into tmp
        const C &at(uint32_t j, C &tmp) const {
            if constexpr (CAN_INTERN) {
                if (v) {
                    vt->get(v[j], tmp.p);
                    return tmp;
                }
            }
            return c[j];
        }
    };
    enum : uint8_t { COMPUTING = 0, READY = 1, FAILED = 2 };
    struct Rec {
        uint32_t chunk = 0, off = 0, len = 0;
        uint8_t state = COMPUTING;
    };
    struct Chunk {
        uint32_t *k;
        C *c;                                      // nullptr when interning
        uint32_t *v;                               // value indices (interning)
        uint32_t used, cap;
    };
    struct Shard {
        std::mutex m;
        std::condition_variable cv;
        std::vector<uint64_t> tkeys;               // open addressing: key -> record
        std::vector<uint32_t> tvals;
        size_t tcount = 0;
        std::vector<Rec> recs;
        std::vector<Chunk> chunks;
        std::unordered_map<uint32_t, std::thread::id> owners;   // records being computed
        size_t terms = 0;
        // profiling only (prof): the clock of each record's last lookup, and
        // the reuse intervals (clock ticks between two lookups of a record) by
        // bit length, counted per lookup and weighted by the record's terms
        std::vector<uint64_t> last;
        uint64_t reuse_n[65] = {}, reuse_t[65] = {};
    };
    static constexpr size_t NS = 64;
    static constexpr uint32_t CHUNK = 1u << 16, NONE = 0xFFFFFFFFu;
    std::unique_ptr<Shard[]> shards{new Shard[NS]};
    // cache profiling (VERTEX_ALGEBRAS_CACHE_PROFILE=1; diagnostics): a
    // clock ticking once per lookup, see Shard::last; off, it costs nothing
    bool prof = false;
    std::atomic<uint64_t> clock{0};
    std::unique_ptr<QValueTable> vt;               // interning (over QQ)

    CompactCache() {
        if constexpr (CAN_INTERN) {
            const char *e = std::getenv("VERTEX_ALGEBRAS_INTERN");
            if (!(e && std::string(e) == "0")) vt.reset(new QValueTable());
        }
    }
    CompactCache(const CompactCache &) = delete;
    CompactCache &operator=(const CompactCache &) = delete;
    ~CompactCache() {
        for (size_t i = 0; i < NS; i++)
            for (Chunk &ch : shards[i].chunks) {
                if (ch.c) {
                    for (uint32_t j = 0; j < ch.used; j++) ch.c[j].~C();
                    ::operator delete(static_cast<void *>(ch.c));
                }
                delete[] ch.v;
                delete[] ch.k;
            }
    }
    static size_t shard_of(uint64_t k) { return (size_t)((k * 0x9E3779B97F4A7C15ULL) >> 58); }
    static size_t slot_of(uint64_t k, size_t mask) {
        return (size_t)(((k ^ (k >> 29)) * 0xBF58476D1CE4E5B9ULL) >> 17) & mask;
    }
    static uint32_t find(const Shard &s, uint64_t key) {
        if (s.tvals.empty()) return NONE;
        size_t mask = s.tvals.size() - 1;
        for (size_t i = slot_of(key, mask);; i = (i + 1) & mask) {
            uint32_t v = s.tvals[i];
            if (v == NONE || s.tkeys[i] == key) return v;
        }
    }
    static void put(Shard &s, uint64_t key, uint32_t rec) {
        if ((s.tcount + 1) * 10 > s.tvals.size() * 7) {   // load <= 0.7
            size_t n = std::max<size_t>(16, s.tvals.size() * 2);
            std::vector<uint64_t> ok;
            std::vector<uint32_t> ov;
            ok.swap(s.tkeys);
            ov.swap(s.tvals);
            s.tkeys.assign(n, 0);
            s.tvals.assign(n, NONE);
            for (size_t i = 0; i < ov.size(); i++)
                if (ov[i] != NONE) {
                    size_t j = slot_of(ok[i], n - 1);
                    while (s.tvals[j] != NONE) j = (j + 1) & (n - 1);
                    s.tkeys[j] = ok[i];
                    s.tvals[j] = ov[i];
                }
        }
        size_t mask = s.tvals.size() - 1, i = slot_of(key, mask);
        while (s.tvals[i] != NONE) i = (i + 1) & mask;
        s.tkeys[i] = key;
        s.tvals[i] = rec;
        s.tcount++;
    }
    View view(const Shard &s, const Rec &r) const {
        if (!r.len) return View();
        const Chunk &ch = s.chunks[r.chunk];
        View w;
        w.k = ch.k + r.off;
        w.n = r.len;
        if (ch.c)
            w.c = ch.c + r.off;
        else {
            w.v = ch.v + r.off;
            w.vt = vt.get();
        }
        return w;
    }
    // with s.m held: move the terms of f into the arena (ids: f's value
    // indices when interning, found before the lock was taken)
    void place(Shard &s, uint32_t ri, Flat &&f, const std::vector<uint32_t> &ids) {
        uint32_t n = (uint32_t)f.size();
        Rec &r = s.recs[ri];
        r.len = n;
        if (n) {
            if (s.chunks.empty() || n > s.chunks.back().cap - s.chunks.back().used) {
                uint32_t cap = std::max(n, CHUNK);
                Chunk ch;
                ch.k = new uint32_t[cap];
                ch.c = vt ? nullptr : static_cast<C *>(::operator new(sizeof(C) * (size_t)cap));
                ch.v = vt ? new uint32_t[cap] : nullptr;
                ch.used = 0;
                ch.cap = cap;
                s.chunks.push_back(ch);
            }
            Chunk &ch = s.chunks.back();
            r.chunk = (uint32_t)(s.chunks.size() - 1);
            r.off = ch.used;
            for (uint32_t j = 0; j < n; j++) {
                ch.k[ch.used + j] = f[j].first;
                if (vt)
                    ch.v[ch.used + j] = ids[j];
                else
                    new (&ch.c[ch.used + j]) C(std::move(f[j].second));
            }
            ch.used += n;
            s.terms += n;
        }
        r.state = READY;
    }
    template <class F>
    View get(uint64_t key, F &&compute) {
        Shard &s = shards[shard_of(key)];
        std::unique_lock<std::mutex> lk(s.m);
        uint32_t ri = find(s, key);
        uint64_t now = prof ? clock.fetch_add(1, std::memory_order_relaxed) : 0;
        if (ri != NONE) {
            if (prof) {
                uint64_t d = now - s.last[ri];
                int b = d ? 64 - __builtin_clzll(d) : 0;
                s.reuse_n[b]++;
                s.reuse_t[b] += s.recs[ri].len;
                s.last[ri] = now;
            }
            if (s.recs[ri].state == READY) return view(s, s.recs[ri]);
            if (s.recs[ri].state == COMPUTING) {
                if (s.owners[ri] == std::this_thread::get_id())
                    throw std::logic_error("cyclic dependency in the module recursion");
                s.cv.wait(lk, [&] { return s.recs[ri].state != COMPUTING; });
                if (s.recs[ri].state == READY) return view(s, s.recs[ri]);
                throw EntryFailed();
            }
            s.recs[ri].state = COMPUTING;          // failed before: computed again here
        } else {
            ri = (uint32_t)s.recs.size();
            if (ri == NONE) throw std::length_error("cache shard full");
            s.recs.push_back(Rec());
            if (prof) s.last.push_back(now);
            put(s, key, ri);
        }
        s.owners[ri] = std::this_thread::get_id();
        lk.unlock();
        Flat f;
        std::vector<uint32_t> ids;
        try {
            f = compute();
            if constexpr (CAN_INTERN) {
                if (vt) {
                    ids.reserve(f.size());
                    for (auto &kv : f) ids.push_back(vt->intern(kv.second.p));
                }
            }
        } catch (...) {
            lk.lock();
            s.recs[ri].state = FAILED;
            s.owners.erase(ri);
            lk.unlock();
            s.cv.notify_all();
            throw;
        }
        lk.lock();
        place(s, ri, std::move(f), ids);
        s.owners.erase(ri);
        View v = view(s, s.recs[ri]);
        lk.unlock();
        s.cv.notify_all();
        return v;
    }
    size_t size() const {
        size_t n = 0;
        for (size_t i = 0; i < NS; i++) {
            std::lock_guard<std::mutex> g(shards[i].m);
            n += shards[i].recs.size();
        }
        return n;
    }
    size_t terms() const {
        size_t n = 0;
        for (size_t i = 0; i < NS; i++) {
            std::lock_guard<std::mutex> g(shards[i].m);
            n += shards[i].terms;
        }
        return n;
    }
    // profiling: [clock, reuse_n[0..64], reuse_t[0..64], age_n[0..64],
    // age_t[0..64]], the ages (clock ticks since the last lookup, at the time
    // of the call) of the stored records by bit length, counted and weighted
    // by terms; empty unless prof
    std::vector<size_t> profile() const {
        if (!prof) return {};
        std::vector<size_t> out(1 + 4 * 65, 0);
        uint64_t now = clock.load();
        out[0] = (size_t)now;
        for (size_t i = 0; i < NS; i++) {
            Shard &s = shards[i];
            std::lock_guard<std::mutex> g(s.m);
            for (int b = 0; b < 65; b++) {
                out[1 + b] += s.reuse_n[b];
                out[66 + b] += s.reuse_t[b];
            }
            for (size_t r = 0; r < s.recs.size(); r++) {
                if (s.recs[r].state != READY) continue;
                uint64_t d = now - s.last[r];
                int b = d ? 64 - __builtin_clzll(d) : 0;
                out[131 + b]++;
                out[196 + b] += s.recs[r].len;
            }
        }
        return out;
    }
    // calls f(View) for every stored list (no other thread may run)
    template <class F>
    void for_each(F &&f) const {
        for (size_t i = 0; i < NS; i++)
            for (const Rec &r : shards[i].recs)
                if (r.state == READY) f(view(shards[i], r));
    }
    // (distinct values, their packed bytes); (0, 0) unless interning
    std::pair<size_t, size_t> value_sizes() const {
        return vt ? vt->sizes() : std::pair<size_t, size_t>(0, 0);
    }
};

// Statistics of the rational coefficients of the lists of a CompactCache
// (diagnostics, ModuleEngine::coefficient_stats): appended to out, in this
// order: terms; lists; terms whose numerator / denominator is a GMP integer
// (beyond FLINT's inline 62 bits); the limbs those allocate; terms with
// denominator 1; distinct values (by a 64-bit hash: may undercount by
// collisions, negligibly); total bits of numerators and denominators; total
// bits with one denominator per list (the lcm, plus the scaled numerators);
// lists with one distinct denominator; then the histograms of the numerators'
// and denominators' bit lengths (0..1023, longer in 1023).
template <class Cache>
void rational_stats(const Cache &cache, std::vector<size_t> &out) {
    size_t terms = 0, lists = 0, big_n = 0, big_d = 0, limbs = 0, den1 = 0, bits = 0, bits_cd = 0, one_den = 0;
    std::vector<size_t> hn(1024, 0), hd(1024, 0);
    std::vector<uint64_t> hashes;
    fmpz_t l, t;
    fmpz_init(l);
    fmpz_init(t);
    auto limbs_of = [](const fmpz_t z) -> size_t {
        return COEFF_IS_MPZ(*z) ? (size_t)COEFF_TO_PTR(*z)->_mp_alloc : 0;
    };
    auto mix = [](uint64_t h, uint64_t x) { return (h ^ (x + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2))) * 0xBF58476D1CE4E5B9ULL; };
    auto hash_of = [&](const fmpz_t z) -> uint64_t {
        if (!COEFF_IS_MPZ(*z)) return mix(1, (uint64_t)*z);
        const __mpz_struct *m = COEFF_TO_PTR(*z);
        uint64_t h = mix(2, (uint64_t)(int64_t)m->_mp_size);
        for (int i = 0; i < std::abs(m->_mp_size); i++) h = mix(h, (uint64_t)m->_mp_d[i]);
        return h;
    };
    RingQ R;
    RingQ::E tmp(R), first(R);
    cache.for_each([&](const typename Cache::View &v) {
        lists++;
        fmpz_one(l);
        bool same = true;
        if (v.n) first = v.at(0, tmp);
        for (uint32_t j = 0; j < v.n; j++) {
            const fmpq *q = v.at(j, tmp).p;
            size_t bn = fmpz_bits(fmpq_numref(q)), bd = fmpz_bits(fmpq_denref(q));
            terms++;
            hn[std::min<size_t>(bn, 1023)]++;
            hd[std::min<size_t>(bd, 1023)]++;
            bits += bn + bd;
            if (COEFF_IS_MPZ(*fmpq_numref(q))) big_n++;
            if (COEFF_IS_MPZ(*fmpq_denref(q))) big_d++;
            limbs += limbs_of(fmpq_numref(q)) + limbs_of(fmpq_denref(q));
            if (fmpz_is_one(fmpq_denref(q))) den1++;
            hashes.push_back(mix(hash_of(fmpq_numref(q)), hash_of(fmpq_denref(q))));
            if (j && !fmpz_equal(fmpq_denref(q), fmpq_denref(first.p))) same = false;
            fmpz_lcm(l, l, fmpq_denref(q));
        }
        if (same) one_den++;
        bits_cd += fmpz_bits(l);
        for (uint32_t j = 0; j < v.n; j++) {
            const fmpq *q = v.at(j, tmp).p;
            fmpz_divexact(t, l, fmpq_denref(q));
            fmpz_mul(t, t, fmpq_numref(q));
            bits_cd += fmpz_bits(t);
        }
    });
    fmpz_clear(l);
    fmpz_clear(t);
    std::sort(hashes.begin(), hashes.end());
    size_t distinct = (size_t)(std::unique(hashes.begin(), hashes.end()) - hashes.begin());
    for (size_t x : {terms, lists, big_n, big_d, limbs, den1, distinct, bits, bits_cd, one_den}) out.push_back(x);
    out.insert(out.end(), hn.begin(), hn.end());
    out.insert(out.end(), hd.begin(), hd.end());
}

template <class Ring>
class ModuleEngine : public EngineCore<ModuleEngine<Ring>, Ring>, public ModuleEngineBase {
  public:
    using Core = EngineCore<ModuleEngine<Ring>, Ring>;
    using C = typename Core::C;
    using Elem = typename Core::Elem;
    using Poles = typename Core::Poles;
    // computed values: flat (key, coefficient) lists, cached in CompactCaches
    // and read through views; sums are accumulated in Elems
    using Flat = std::vector<std::pair<uint32_t, C>>;
    using Cache = CompactCache<Ring>;
    using View = typename Cache::View;
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
    using Core::elem_in;
    using Core::elem_out;
    using Core::run_tasks;

    static constexpr long MOFF = 1L << 22;        // |m2| bound of the cache keys
    int G;
    std::vector<int> par;                          // parities of the generators
    std::vector<long> h2;                          // D * weights
    long D;
    int dim = -1;                                  // the top (set_top)
    std::vector<int> upar;
    std::vector<uint32_t> top_ids;                 // key id of the top vector u
    std::vector<std::vector<Flat>> zero;           // zero[i][u]: the zero mode of g_i on u
    std::vector<char> has_zero;
    // a "lowered" top (induction from degree n, higher Zhu algebras): top
    // vectors of degrees tdeg2[u] / D, the modes m >= 0 on them given as a
    // table (lowered[(i, m2, u)]; absent = 0), creation modes prepended
    std::vector<long> tdeg2;
    bool is_lowered = false;
    std::unordered_map<uint64_t, Flat> lowered;
    bool is_free = false;                          // set_free: He's free module
    long free_n2 = 0;
    std::vector<Poles> opes;                       // poles of OPE(g_i, g_j) over word ids
    // splits[w][i] = x_(i) B (w = :x B:, over word ids; set_splits); read
    // only while act_many runs
    std::unordered_map<uint32_t, std::vector<Flat>> splits;
    MonoTable words;                               // the monomials of V
    static constexpr uint32_t MAXL = 1u << 12;     // letter codes with a lock-free word id
    std::unique_ptr<std::atomic<uint32_t>[]> letter_words{new std::atomic<uint32_t>[MAXL]()};
    MonoTable wns;                                 // (word, n) -> id when it does not pack
    std::mutex wns_m;
    Cache gen_cache, word_cache;
    // statistics by word length (diagnostics, word_stats): lookups,
    // misses, terms stored
    static constexpr size_t STATL = 64;
    std::unique_ptr<std::atomic<uint64_t>[]> st_calls{new std::atomic<uint64_t>[STATL]()};
    std::unique_ptr<std::atomic<uint64_t>[]> st_miss{new std::atomic<uint64_t>[STATL]()};
    std::unique_ptr<std::atomic<uint64_t>[]> st_terms{new std::atomic<uint64_t>[STATL]()};

    template <class... A>
    ModuleEngine(int ngens, const std::vector<int> &parities, const std::vector<long> &weights2, long den,
                 A &&...ring_args)
        : Core(std::forward<A>(ring_args)...), G(ngens), par(parities), h2(weights2), D(den) {
        if (G < 1 || G > 256) throw std::invalid_argument("1 to 256 generators");
        if (D < 1) throw std::invalid_argument("bad denominator");
        const char *prof = std::getenv("VERTEX_ALGEBRAS_CACHE_PROFILE");
        gen_cache.prof = word_cache.prof = prof && std::string(prof) == "1";
        par.resize((size_t)G, 0);
        if ((int)h2.size() != G) throw std::invalid_argument("one weight per generator");
        for (long h : h2)
            if (h <= 0) throw std::invalid_argument("weights must be positive");
        words.parity = [this](const std::vector<uint32_t> &v) {
            int p = 0;
            for (uint32_t c : v) p += par[wgen(c)];
            return p;
        };
        {
            std::lock_guard<std::mutex> g(words.m);
            words.intern_locked(std::vector<uint32_t>());   // word id 0 = the vacuum
        }
        wns.tails = false;
        monos.parity = [this](const std::vector<uint32_t> &v) {
            int p = upar.empty() ? 0 : upar[v.back()];
            for (size_t k = 0; k + 1 < v.size(); k++) p += par[mgen(v[k])];
            return p;
        };
        opes.resize((size_t)G * G);
        zero.resize((size_t)G);
        has_zero.assign((size_t)G, 0);
    }

    // -- words (monomials of V) ----------------------------------------------- //
    uint32_t wcode(int gen, int der) const { return (uint32_t)(der * G + (G - 1 - gen)); }
    int wgen(uint32_t c) const { return G - 1 - (int)(c % G); }
    int wder(uint32_t c) const { return (int)(c / G); }
    uint32_t intern_wcodes(const std::vector<uint32_t> &v) {
        std::lock_guard<std::mutex> g(words.m);
        return words.intern_locked(v);
    }
    long wt2(uint32_t w) const {                     // D * the weight of a word
        long s = 0;
        for (uint32_t c : words[w].letters) s += h2[wgen(c)] + (long)wder(c) * D;
        return s;
    }
    // (word, n) as 32 bits: packed without a lock (w < 2^19, |n| < 2048), else
    // interned (top bit set)
    uint32_t wn_id(uint32_t w, long n) {
        if (w < (1u << 19) && n > -2048 && n < 2048) return (w << 12) | (uint32_t)(n + 2048);
        if (n <= -(long)INT32_MAX || n >= (long)INT32_MAX) throw std::invalid_argument("mode out of range");
        std::lock_guard<std::mutex> g(wns_m);
        return 0x80000000u | wns.intern_locked(std::vector<uint32_t>{w, (uint32_t)(n + (long)INT32_MAX)});
    }
    // the word of one letter (code c)
    uint32_t letter_word(uint32_t c) {
        if (c < MAXL) {
            uint32_t v = letter_words[c].load(std::memory_order_acquire);
            if (v) return v;
        }
        uint32_t id = intern_wcodes(std::vector<uint32_t>{c});
        if (c < MAXL) letter_words[c].store(id, std::memory_order_release);
        return id;
    }

    // -- keys ------------------------------------------------------------------ //
    // a mode as a letter code: (MOFF - m2) G + i, any |m2| < MOFF (the free
    // module keeps zero and lowering modes in its words)
    uint32_t mcode(int i, long m2) const { return (uint32_t)((MOFF - m2) * G + i); }
    int mgen(uint32_t c) const { return (int)(c % G); }
    long mval(uint32_t c) const { return MOFF - (long)(c / G); }
    static bool mode_less(int i, long m2, int j, long n2) { return m2 < n2 || (m2 == n2 && i < j); }
    long depth2(uint32_t key) const {
        const std::vector<uint32_t> &kl = monos[key].letters;
        long d = is_lowered ? tdeg2[kl.back()] : 0;
        for (size_t k = 0; k + 1 < kl.size(); k++) d -= mval(kl[k]);
        return d;
    }
    // the largest n with a_(n) key possibly nonzero: wt a + depth - 1
    long nmax(uint32_t w, uint32_t key) const { return floor_div(wt2(w) + depth2(key) - D, D); }
    uint64_t lowered_key(int i, long m2, uint32_t u) const {
        return ((uint64_t)(uint32_t)(m2 * G + i) << 32) | u;
    }
    uint32_t cons_mode(uint32_t c, uint32_t key) {
        const std::vector<uint32_t> &b = monos[key].letters;
        std::vector<uint32_t> v;
        v.reserve(b.size() + 1);
        v.push_back(c);
        v.insert(v.end(), b.begin(), b.end());
        return intern_codes(v);
    }

    // -- coefficients ------------------------------------------------------------ //
    // n (n-1) ... (n-k+1), any integer n, k >= 0
    static void falling(fmpz_t r, long n, long k) {
        fmpz_one(r);
        for (long i = 0; i < k; i++) fmpz_mul_si(r, r, n - i);
    }
    // binomial(n, p) = falling(n, p) / p!, any integer n
    static void gbinom(fmpz_t r, long n, long p) {
        falling(r, n, p);
        Z f;
        fact(f.v, (ulong)p);
        fmpz_divexact(r, r, f.v);
    }

    // -- flat values ---------------------------------------------------------------- //
    static Flat flat(Elem &&e) {
        Flat f;
        f.reserve(e.size());
        for (auto &kv : e) f.emplace_back(kv.first, std::move(kv.second));
        return f;
    }
    Flat single_flat(uint32_t k) const {
        Flat f;
        C one(R);
        R.one(one);
        if (!R.is_zero(one)) f.emplace_back(k, std::move(one));
        return f;
    }
    void add_flat(Elem &dst, const View &src, const C &s) const {
        if (src.empty() || R.is_zero(s)) return;
        C t(R), u(R);
        for (uint32_t j = 0; j < src.n; j++) {
            R.mul(t, s, src.at(j, u));
            acc(dst, src.k[j], t);
        }
    }

    // -- the recursions ------------------------------------------------------------ //
    // the physics mode g_{i, m} (m2 = D m) on the key: InducedModule._gen_act
    View gen_act(int i, long m2, uint32_t key) {
        if (m2 <= -MOFF || m2 >= MOFF) throw std::invalid_argument("mode out of range");
        uint64_t ck = ((uint64_t)((uint32_t)((m2 + MOFF) * G + i)) << 32) | key;
        return gen_cache.get(ck, [&]() -> Flat {
            check();
            return gen_act_compute(i, m2, key);
        });
    }
    Flat gen_act_compute(int i, long m2, uint32_t key) {
        const std::vector<uint32_t> &kl = monos[key].letters;
        if (is_free && depth2(key) - m2 < 0) return Flat();   // negative degree
        if (is_free && kl.size() == 1)                // every mode kept: g_m v
            return m2 > free_n2 ? Flat() : single_flat(cons_mode(mcode(i, m2), key));
        if (kl.size() == 1) {                         // the top vector u
            if (is_lowered && m2 >= 0) {
                auto it = lowered.find(lowered_key(i, m2, kl[0]));
                return it == lowered.end() ? Flat() : it->second;
            }
            if (m2 > 0) return Flat();
            if (m2 == 0) return has_zero[i] ? zero[(size_t)i][kl[0]] : Flat();
            return single_flat(cons_mode(mcode(i, m2), key));
        }
        uint32_t x = kl[0];
        int ix = mgen(x);
        long mx = mval(x);
        uint32_t rest = monos[key].tail;
        if (mode_less(i, m2, ix, mx) || (i == ix && m2 == mx && !(par[i] & 1))) {
            if (is_free && m2 > 0) {                  // all lowering: by at most n
                long s2 = m2;
                for (size_t k = 0; k + 1 < kl.size(); k++) s2 += mval(kl[k]);
                if (s2 > free_n2) return Flat();
            }
            return single_flat(cons_mode(mcode(i, m2), key));     // already in place
        }
        if (i == ix && m2 == mx) {                    // an odd mode squared: x x = 1/2 [x, x]
            Elem r;
            Elem c = commutator(i, m2, ix, mx, rest);
            add_scaled_si(r, c, 1, 2);
            return flat(std::move(r));
        }
        // g_m x rest = s x (g_m rest) + [g_m, x] rest
        int s = (par[i] & par[ix] & 1) ? -1 : 1;
        Elem r = commutator(i, m2, ix, mx, rest);
        View inner = gen_act(i, m2, rest);
        C t(R), u(R);
        for (uint32_t j = 0; j < inner.n; j++) {
            View o = gen_act(ix, mx, inner.k[j]);
            R.mul_si(t, inner.at(j, u), s);
            add_flat(r, o, t);
        }
        return flat(std::move(r));
    }
    // [g_{i,m}, g_{j,m'}] key = sum_p C(m + h_i - 1, p) (g_i(p) g_j)_{m+m'} key:
    // InducedModule._commutator_on
    Elem commutator(int i, long m2, int j, long mx2, uint32_t key) {
        const Poles &P = opes[(size_t)i * G + j];
        Elem out;
        long ni = m2 + h2[i] - D;
        if (ni % D) throw std::invalid_argument("mode not in ZZ - h");
        ni /= D;
        Z bc;
        C t(R);
        for (size_t p = 0; p < P.size(); p++) {
            if (P[p].empty()) continue;
            gbinom(bc.v, ni, (long)p);
            if (fmpz_is_zero(bc.v)) continue;
            for (const auto &kv : P[p]) {
                long nn = m2 + mx2 + wt2(kv.first) - D;
                if (nn % D) throw std::invalid_argument("composite mode not integral");
                View e = word_act(kv.first, nn / D, key);
                R.mul_fmpz(t, kv.second, bc.v);
                add_flat(out, e, t);
            }
        }
        return out;
    }
    // the word w (a monomial of V) acting by w_(n) on the key:
    // _GeneratorActions._action_key
    View word_act(uint32_t w, long n, uint32_t key) {
        uint64_t ck = ((uint64_t)wn_id(w, n) << 32) | key;
        size_t L = std::min(words[w].letters.size(), STATL - 1);
        st_calls[L].fetch_add(1, std::memory_order_relaxed);
        return word_cache.get(ck, [&]() -> Flat {
            check();
            st_miss[L].fetch_add(1, std::memory_order_relaxed);
            Flat f = word_act_compute(w, n, key);
            st_terms[L].fetch_add(f.size(), std::memory_order_relaxed);
            return f;
        });
    }
    // (calls, misses, terms) by word length: the reuse of the cached word
    // modes (W(2,3), depth 4: 5-7x even for the longest words, 7-20x for
    // lengths 2-6, which hold most terms -- so everything is cached)
    std::vector<size_t> word_stats() const override {
        std::vector<size_t> out;
        for (size_t L = 0; L < STATL; L++) {
            out.push_back((size_t)st_calls[L].load());
            out.push_back((size_t)st_miss[L].load());
            out.push_back((size_t)st_terms[L].load());
        }
        return out;
    }
    Flat word_act_compute(uint32_t w, long n, uint32_t key) {
        const std::vector<uint32_t> &wl = words[w].letters;
        if (wl.empty()) return n == -1 ? single_flat(key) : Flat();
        if (wl.size() == 1) {                         // (d^k g)_(n) = (-1)^k n ... (n-k+1) g_(n-k)
            int g = wgen(wl[0]), k = wder(wl[0]);
            Z f;
            falling(f.v, n, k);
            if (fmpz_is_zero(f.v)) return Flat();
            if (k & 1) fmpz_neg(f.v, f.v);
            long m2 = (n - k + 1) * D - h2[g];       // the physics mode (n-k) - h_g + 1
            View e = gen_act(g, m2, key);
            Flat r;
            r.reserve(e.n);
            C u(R);
            for (uint32_t j = 0; j < e.n; j++) {     // distinct keys: no accumulation
                C t(R);
                R.mul_fmpz(t, e.at(j, u), f.v);
                if (!R.is_zero(t)) r.emplace_back(e.k[j], std::move(t));
            }
            return r;
        }
        // (:x B:)_(n) v = sum_j x_(-1-j) B_(n+j) v + s sum_j B_(n-1-j) x_(j) v
        uint32_t xw = letter_word(wl[0]);
        uint32_t B = words[w].tail;
        int s = (words[xw].par & words[B].par & 1) ? -1 : 1;
        Elem res;
        C u(R);
        long j1 = std::max(nmax(B, key) - n + 1, 0L);
        for (long j = 0; j < j1; j++) {
            View bv = word_act(B, n + j, key);
            for (uint32_t q = 0; q < bv.n; q++) {
                View xv = word_act(xw, -1 - j, bv.k[q]);
                add_flat(res, xv, bv.at(q, u));
            }
        }
        long j2 = nmax(xw, key);
        C t(R);
        // The raising x_(j) (wt x - j - 1 > 0) would act first and visit keys
        // deeper than the result; with the products x_(i) B known (set_splits)
        // they act last instead:
        //   s B_(n-1-j) x_(j) v = x_(j) B_(n-1-j) v - sum_{i <= j} C(j, i) (x_(i) B)_(n-1-i) v,
        // and sum_{j < J} C(j, i) = C(J, i + 1)
        long J = 0;
        const std::vector<Flat> *sp = nullptr;
        if (!splits.empty()) {
            auto it = splits.find(w);
            if (it != splits.end()) {
                sp = &it->second;
                J = std::max(0L, std::min(j2 + 1, (long)sp->size()));
            }
        }
        for (long j = 0; j < J; j++) {
            View bv = word_act(B, n - 1 - j, key);
            for (uint32_t q = 0; q < bv.n; q++) {
                View xv = word_act(xw, j, bv.k[q]);
                add_flat(res, xv, bv.at(q, u));
            }
        }
        if (J > 0) {
            Z bc;
            for (long i = 0; i < J; i++) {
                gbinom(bc.v, J, i + 1);
                fmpz_neg(bc.v, bc.v);
                for (const auto &kv : (*sp)[(size_t)i]) {
                    R.mul_fmpz(t, kv.second, bc.v);
                    add_flat(res, word_act(kv.first, n - 1 - i, key), t);
                }
            }
        }
        for (long j = J; j <= j2; j++) {
            View xv = word_act(xw, j, key);
            for (uint32_t q = 0; q < xv.n; q++) {
                R.mul_si(t, xv.at(q, u), s);
                add_flat(res, word_act(B, n - 1 - j, xv.k[q]), t);
            }
        }
        return flat(std::move(res));
    }

    // -- interface --------------------------------------------------------------- //
    uint32_t intern_word(const WordLetters &w) override {
        std::vector<uint32_t> v;
        for (const auto &l : w) {
            if (l.first < 0 || l.first >= G || l.second < 0) throw std::invalid_argument("bad letter");
            v.push_back(wcode(l.first, l.second));
        }
        return intern_wcodes(v);
    }
    WordLetters word_letters(uint32_t id) const override {
        if (id >= words.n) throw std::out_of_range("unknown word id");
        WordLetters out;
        for (uint32_t c : words[id].letters) out.emplace_back(wgen(c), wder(c));
        return out;
    }
    uint32_t intern_key(const ModeList &modes, int u) override {
        if (dim < 0) throw std::logic_error("set_top first");
        if (u < 0 || u >= dim) throw std::invalid_argument("bad top index");
        std::vector<uint32_t> v;
        for (size_t k = 0; k < modes.size(); k++) {
            int i = modes[k].first;
            long m2 = modes[k].second;
            if (i < 0 || i >= G || m2 <= -MOFF || m2 >= MOFF || (!is_free && m2 >= 0) || (m2 + h2[i]) % D)
                throw std::invalid_argument(is_free ? "bad mode" : "bad creation mode");
            if (k) {
                int j = modes[k - 1].first;
                long n2 = modes[k - 1].second;
                if (!(mode_less(j, n2, i, m2) || (i == j && m2 == n2 && !(par[i] & 1))))
                    throw std::invalid_argument("modes not in the mode order");
            }
            v.push_back(mcode(i, m2));
        }
        v.push_back((uint32_t)u);
        return intern_codes(v);
    }
    std::pair<ModeList, int> key_of(uint32_t id) const override {
        if (id >= monos.n) throw std::out_of_range("unknown key id");
        const std::vector<uint32_t> &kl = monos[id].letters;
        ModeList modes;
        for (size_t k = 0; k + 1 < kl.size(); k++) modes.emplace_back(mgen(kl[k]), mval(kl[k]));
        return {modes, (int)kl.back()};
    }
    void set_top(int d, const std::vector<int> &parities) override {
        if (dim >= 0) throw std::logic_error("the top is set already");
        if (d < 1 || (int)parities.size() != d) throw std::invalid_argument("bad top");
        upar = parities;
        dim = d;
        for (int u = 0; u < d; u++) top_ids.push_back(intern_codes(std::vector<uint32_t>{(uint32_t)u}));
    }
    void set_zero_mode(int i, const std::vector<std::vector<std::pair<int, CoefData>>> &cols) override {
        if (dim < 0) throw std::logic_error("set_top first");
        if (i < 0 || i >= G || (int)cols.size() != dim) throw std::invalid_argument("bad zero mode");
        std::vector<Flat> z((size_t)dim);
        for (int u = 0; u < dim; u++) {
            Elem e;
            for (const auto &t : cols[(size_t)u]) {
                if (t.first < 0 || t.first >= dim) throw std::invalid_argument("bad top index");
                C c(R);
                R.from_data(c, t.second);
                acc(e, top_ids[(size_t)t.first], c);
            }
            z[(size_t)u] = flat(std::move(e));
        }
        zero[(size_t)i] = std::move(z);
        has_zero[(size_t)i] = 1;
    }
    void set_top_degrees(const std::vector<long> &degrees2) override {
        if (dim < 0) throw std::logic_error("set_top first");
        if ((int)degrees2.size() != dim) throw std::invalid_argument("one degree per top vector");
        for (long d : degrees2)
            if (d < 0 || d >= MOFF) throw std::invalid_argument("bad top degree");
        if (gen_cache.size() || word_cache.size())
            throw std::logic_error("the top must be set before computing");
        tdeg2 = degrees2;
        is_lowered = true;
    }
    void set_top_action(int i, long m2, int u, const ElemData &value) override {
        if (!is_lowered) throw std::logic_error("set_top_degrees first");
        if (i < 0 || i >= G || u < 0 || u >= dim || m2 < 0 || m2 >= MOFF)
            throw std::invalid_argument("bad top action");
        if (gen_cache.size() || word_cache.size())
            throw std::logic_error("the top must be set before computing");
        lowered[lowered_key(i, m2, (uint32_t)u)] = flat(elem_in(value));
    }
    void set_free(long n2) override {
        if (dim != 1) throw std::logic_error("set_top(1, ...) first: the free module has one top vector");
        if (n2 < 0 || n2 >= MOFF) throw std::invalid_argument("bad level");
        if (gen_cache.size() || word_cache.size())
            throw std::logic_error("the top must be set before computing");
        tdeg2 = {n2};
        is_lowered = true;                         // depth2 counts from the top degree
        is_free = true;
        free_n2 = n2;
    }
    void set_ope(int i, int j, const PolesData &poles) override {
        if (i < 0 || j < 0 || i >= G || j >= G) throw std::invalid_argument("bad generator index");
        if (gen_cache.size() || word_cache.size())
            throw std::logic_error("generator OPEs must be set before computing");
        Poles P;
        for (const ElemData &e : poles) P.push_back(word_elem_in(e));
        opes[(size_t)i * G + j] = std::move(P);
    }
    // the number of j >= 0 with x_(j) raising: wt x - j - 1 > 0
    long raising(uint32_t xw) const { return std::max(0L, (wt2(xw) + D - 1) / D - 1); }
    void set_splits(uint32_t w, const std::vector<ElemData> &xb) override {
        if (w >= words.n) throw std::out_of_range("unknown word id");
        if (words[w].letters.size() < 2) throw std::invalid_argument("splits of a word of at least two letters");
        uint32_t xw = letter_word(words[w].letters[0]);
        if ((long)xb.size() != raising(xw)) throw std::invalid_argument("one product per raising mode");
        std::vector<Flat> v;
        for (const ElemData &d : xb) v.push_back(flat(word_elem_in(d)));
        splits[w] = std::move(v);
    }
    Elem word_elem_in(const ElemData &d) {
        Elem e;
        for (const auto &t : d) {
            if (t.first >= words.n) throw std::out_of_range("unknown word id");
            C c(R);
            R.from_data(c, t.second);
            acc(e, t.first, c);
        }
        return e;
    }
    // [els[ie[k]]_(ns[k]) vecs[iv[k]] for every k]: each task in one of
    // nthreads threads (larger first), the caches shared
    std::vector<ElemData> act_many(const std::vector<ElemData> &dels, const std::vector<ElemData> &dvecs,
                                   const std::vector<int> &ie, const std::vector<long> &ns,
                                   const std::vector<int> &iv, int nthreads, PollFn poll) override {
        if (dim < 0) throw std::logic_error("set_top first");
        MaybeGmpMalloc gmp(nthreads);
        if (ie.size() != ns.size() || iv.size() != ns.size())
            throw std::invalid_argument("task vectors of different lengths");
        std::vector<Elem> E, Vs;
        for (const ElemData &d : dels) E.push_back(word_elem_in(d));
        for (const ElemData &d : dvecs) Vs.push_back(elem_in(d));
        for (size_t k = 0; k < ns.size(); k++)
            if (ie[k] < 0 || iv[k] < 0 || (size_t)ie[k] >= E.size() || (size_t)iv[k] >= Vs.size())
                throw std::out_of_range("task refers to an unknown element");
        std::vector<size_t> order = Core::task_order(ns.size(), [&](size_t k) {
            return E[(size_t)ie[k]].size() * Vs[(size_t)iv[k]].size();
        });
        std::vector<Elem> res(ns.size());
        run_tasks(order, nthreads, poll, [&](size_t k) {
            const Elem &A = E[(size_t)ie[k]], &B = Vs[(size_t)iv[k]];
            Elem out;
            C t(R);
            for (const auto &a : A)
                for (const auto &b : B) {
                    View e = word_act(a.first, ns[k], b.first);
                    if (e.empty()) continue;
                    R.mul(t, a.second, b.second);
                    add_flat(out, e, t);
                }
            res[k] = std::move(out);
        });
        std::vector<ElemData> out;
        out.reserve(res.size());
        for (const Elem &e : res) out.push_back(elem_out(e));
        return out;
    }
    // diagnostics of the caches (which = 0: generator modes, 1: word modes):
    // CompactCache::profile (empty unless VERTEX_ALGEBRAS_CACHE_PROFILE=1
    // when the engine was made), and rational_stats (empty unless over QQ);
    // call while no act_many runs
    std::vector<size_t> cache_profile(int which) const override {
        return (which ? word_cache : gen_cache).profile();
    }
    std::vector<size_t> coefficient_stats(int which) const override {
        std::vector<size_t> out;
        if constexpr (std::is_same<Ring, RingQ>::value) rational_stats(which ? word_cache : gen_cache, out);
        return out;
    }
    std::vector<size_t> cache_sizes() const override {
        auto gv = gen_cache.value_sizes(), wv = word_cache.value_sizes();
        return {gen_cache.size(), word_cache.size(), (size_t)monos.n, (size_t)words.n, (size_t)wns.n,
                gen_cache.terms(), word_cache.terms(), gv.first + wv.first, gv.second + wv.second};
    }
};

// kind as the universal engine's make_engine; weights2 = D * the weights
inline ModuleEngineBase *make_module_engine(const std::string &kind, int ngens, const std::vector<int> &parities,
                                            const std::vector<long> &weights2, long D, long nvars,
                                            unsigned long modulus, const CoefData &minpoly) {
    if (kind == "nf") return new ModuleEngine<RingNF>(ngens, parities, weights2, D, minpoly);
    if (kind == "nfmpoly")
        return new ModuleEngine<RingNFMPoly>(ngens, parities, weights2, D, (slong)nvars, minpoly);
    if (kind == "q") return new ModuleEngine<RingQ>(ngens, parities, weights2, D);
    if (kind == "qpoly") return new ModuleEngine<RingQPoly>(ngens, parities, weights2, D);
    if (kind == "qmpoly") {
        if (nvars < 1) throw std::invalid_argument("qmpoly needs at least one variable");
        return new ModuleEngine<RingQMPoly>(ngens, parities, weights2, D, (slong)nvars);
    }
    if (kind == "nmod") {
        if (modulus < 2 || modulus > (unsigned long)WORD_MAX || !n_is_prime(modulus))
            throw std::invalid_argument("nmod needs a prime modulus below 2^63");
        return new ModuleEngine<RingNmod>(ngens, parities, weights2, D, (ulong)modulus);
    }
    throw std::invalid_argument("unknown coefficient kind " + kind);
}

}  // namespace vae
