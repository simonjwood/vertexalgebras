// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// The C++ engine of lowering modules: L(U) = Mbar(U) / J(U), Zhu's induced
// module of a finite-dimensional top U modulo the largest submodule meeting
// U trivially, built depth by depth from lowering data, for a universal
// vertex algebra with strong generators of positive integer weights (even
// parity).  No relations of the algebra and no free PBW monomials: a vector
// of depth N is zero iff every lowering generator mode kills it, so the
// depth-N space is spanned by the candidates g_{-n} b (b a basis vector of
// depth N - n) modulo the common kernel of the lowering modes; their
// lowering data come from the commutator formula
//     g'_(p) g_(q) b = g_(q) g'_(p) b + [g'_(p), g_(q)] b,
//     [g'_(p), g_(q)] = sum_j C(p, j) (g'_(j) g)_(p+q-j),
// with the modes of the composite fields g'_(j) g (words of the universal
// algebra) by the normally ordered product formula
//     (:x B:)_(n) = sum_j x_(-1-j) B_(n+j) + sum_j B_(n-1-j) x_(j),
// a non-lowering x_(j) moved to the left (the mode split):
//     B_(m) x_(j) = x_(j) B_(m) - sum_i C(j, i) (x_(i) B)_(m+j-i),
// so that no intermediate vector leaves the depths <= max(source, target).
// Zero modes at the new depth from their lowering data.  Modes are bracket
// modes: g_(n) maps depth e to depth e + wt g - n - 1.
//
// Every mode is a sparse matrix (rows: target basis, columns: source basis);
// the modes of words are memoised per (word, n, depth) in a cache shared by
// threads (each entry computed once).  Fields: QQ (fmpq) and GF(p) (nmod).
// The words reachable from the OPEs, their tails and the split products
// x_(i) B are given by the caller (set_ope, set_splits): the engine never
// computes products of the vertex algebra itself.
#pragma once
#include "_engine_base.hpp"
#include <flint/nmod.h>
#include <cstring>
#include <map>
#include <unordered_set>

namespace vae {

// ------------------------------------------------------------------------ //
//  Fields                                                                  //
// ------------------------------------------------------------------------ //

struct QE {
    fmpq_t v;
    QE() { fmpq_init(v); }
    QE(const QE &o) { fmpq_init(v); fmpq_set(v, o.v); }
    QE(QE &&o) noexcept { fmpq_init(v); fmpq_swap(v, o.v); }
    QE &operator=(const QE &o) { if (this != &o) fmpq_set(v, o.v); return *this; }
    QE &operator=(QE &&o) noexcept { fmpq_swap(v, o.v); return *this; }
    ~QE() { fmpq_clear(v); }
};

struct FieldQ {
    using E = QE;
    E make() const { return E(); }
    bool is_zero(const E &a) const { return fmpq_is_zero(a.v); }
    bool is_one(const E &a) const { return fmpq_is_one(a.v); }
    void zero(E &a) const { fmpq_zero(a.v); }
    void one(E &a) const { fmpq_one(a.v); }
    void set_fmpz(E &a, const fmpz_t z) const {
        fmpz_set(fmpq_numref(a.v), z);
        fmpz_one(fmpq_denref(a.v));
    }
    void add(E &a, const E &b) const { fmpq_add(a.v, a.v, b.v); }
    void sub(E &a, const E &b) const { fmpq_sub(a.v, a.v, b.v); }
    void addmul(E &a, const E &b, const E &c) const { fmpq_addmul(a.v, b.v, c.v); }
    void submul(E &a, const E &b, const E &c) const { fmpq_submul(a.v, b.v, c.v); }
    void mul(E &r, const E &a, const E &b) const { fmpq_mul(r.v, a.v, b.v); }
    void neg(E &a) const { fmpq_neg(a.v, a.v); }
    void inv(E &r, const E &a) const {
        if (fmpq_is_zero(a.v)) throw DivisionByZero("division by zero");
        fmpq_inv(r.v, a.v);
    }
    size_t heap_bytes(const E &a) const {      // limbs outside the fmpq (approximate)
        size_t b = 0;
        if (COEFF_IS_MPZ(*fmpq_numref(a.v))) b += 16 + 8 * fmpz_size(fmpq_numref(a.v));
        if (COEFF_IS_MPZ(*fmpq_denref(a.v))) b += 16 + 8 * fmpz_size(fmpq_denref(a.v));
        return b;
    }
    void from_data(E &a, const CoefData &d) const {
        fmpq_zero(a.v);
        Z n, m;
        fmpq_t t;
        fmpq_init(t);
        for (const Term &x : d) {
            if (!x.exps.empty()) {
                fmpq_clear(t);
                throw std::invalid_argument("a rational coefficient has no exponents");
            }
            num_get(n.v, x.num);
            num_get(m.v, x.den);
            if (fmpz_is_zero(m.v)) {
                fmpq_clear(t);
                throw DivisionByZero("zero denominator");
            }
            fmpq_set_fmpz_frac(t, n.v, m.v);
            fmpq_add(a.v, a.v, t);
        }
        fmpq_clear(t);
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        if (!fmpq_is_zero(a.v)) d.push_back(Term{{}, num_of(fmpq_numref(a.v)), num_of(fmpq_denref(a.v))});
        return d;
    }
};

struct FieldP {
    nmod_t mod;
    explicit FieldP(ulong p) {
        if (p < 2 || !n_is_prime(p)) throw std::invalid_argument("the modulus must be a prime");
        nmod_init(&mod, p);
    }
    struct E {
        ulong v = 0;
    };
    E make() const { return E(); }
    ulong red(const fmpz_t z) const { return fmpz_fdiv_ui(z, mod.n); }
    ulong inv_u(ulong x) const {
        if (x == 0) throw DivisionByZero("division by a multiple of the characteristic " +
                                         std::to_string(mod.n));
        return n_invmod(x, mod.n);
    }
    bool is_zero(const E &a) const { return a.v == 0; }
    bool is_one(const E &a) const { return a.v == 1; }
    void zero(E &a) const { a.v = 0; }
    void one(E &a) const { a.v = 1; }
    void set_fmpz(E &a, const fmpz_t z) const { a.v = red(z); }
    void add(E &a, const E &b) const { a.v = nmod_add(a.v, b.v, mod); }
    void sub(E &a, const E &b) const { a.v = nmod_sub(a.v, b.v, mod); }
    void addmul(E &a, const E &b, const E &c) const { a.v = nmod_add(a.v, nmod_mul(b.v, c.v, mod), mod); }
    void submul(E &a, const E &b, const E &c) const { a.v = nmod_sub(a.v, nmod_mul(b.v, c.v, mod), mod); }
    void mul(E &r, const E &a, const E &b) const { r.v = nmod_mul(a.v, b.v, mod); }
    void neg(E &a) const { a.v = nmod_neg(a.v, mod); }
    void inv(E &r, const E &a) const { r.v = inv_u(a.v); }
    size_t heap_bytes(const E &) const { return 0; }
    void from_data(E &a, const CoefData &d) const {
        ulong acc = 0;
        Z n, m;
        for (const Term &x : d) {
            if (!x.exps.empty()) throw std::invalid_argument("a coefficient mod p has no exponents");
            num_get(n.v, x.num);
            num_get(m.v, x.den);
            acc = nmod_add(acc, nmod_mul(red(n.v), inv_u(red(m.v)), mod), mod);
        }
        a.v = acc;
    }
    CoefData to_data(const E &a) const {
        CoefData d;
        if (a.v) {
            Z z;
            fmpz_set_ui(z.v, a.v);
            d.push_back(Term{{}, num_of(z.v), Num{1, ""}});
        }
        return d;
    }
};

// ------------------------------------------------------------------------ //
//  Sparse and dense matrices                                               //
// ------------------------------------------------------------------------ //

// compressed sparse rows, immutable once built; nnz() == 0 never stored (a
// zero mode is a null pointer)
template <class F>
struct SMat {
    using E = typename F::E;
    uint32_t r = 0, c = 0;
    std::vector<uint32_t> ptr;     // r + 1
    std::vector<uint32_t> col;
    std::vector<E> val;
    size_t nnz() const { return col.size(); }
};
template <class F>
using MatP = std::shared_ptr<const SMat<F>>;

// a dense accumulator
template <class F>
struct DMat {
    using E = typename F::E;
    uint32_t r, c;
    std::vector<E> a;
    DMat(const F &K, uint32_t r_, uint32_t c_) : r(r_), c(c_), a((size_t)r_ * c_, K.make()) {}
    E &at(uint32_t i, uint32_t j) { return a[(size_t)i * c + j]; }
};

// A += s X B (s null: 1); X is A.r x k, B is k x A.c
template <class F>
void acc_mul(const F &K, DMat<F> &A, const SMat<F> &X, const SMat<F> &B, const typename F::E *s) {
    if (X.r != A.r || B.c != A.c || X.c != B.r) throw std::logic_error("matrix sizes do not match");
    typename F::E t = K.make();
    for (uint32_t i = 0; i < X.r; i++) {
        typename F::E *row = &A.a[(size_t)i * A.c];
        for (uint32_t p = X.ptr[i]; p < X.ptr[i + 1]; p++) {
            const typename F::E *x = &X.val[p];
            if (s) {
                K.mul(t, X.val[p], *s);
                x = &t;
            }
            uint32_t k = X.col[p];
            for (uint32_t q = B.ptr[k]; q < B.ptr[k + 1]; q++) K.addmul(row[B.col[q]], *x, B.val[q]);
        }
    }
}
// A += s C
template <class F>
void acc_scaled(const F &K, DMat<F> &A, const SMat<F> &C, const typename F::E &s) {
    if (C.r != A.r || C.c != A.c) throw std::logic_error("matrix sizes do not match");
    for (uint32_t i = 0; i < C.r; i++)
        for (uint32_t p = C.ptr[i]; p < C.ptr[i + 1]; p++) K.addmul(A.at(i, C.col[p]), s, C.val[p]);
}
template <class F>
MatP<F> compress(const F &K, DMat<F> &A) {
    size_t n = 0;
    for (auto &x : A.a)
        if (!K.is_zero(x)) n++;
    if (n == 0) return nullptr;
    auto M = std::make_shared<SMat<F>>();
    M->r = A.r;
    M->c = A.c;
    M->ptr.reserve(A.r + 1);
    M->col.reserve(n);
    M->val.reserve(n);
    M->ptr.push_back(0);
    for (uint32_t i = 0; i < A.r; i++) {
        for (uint32_t j = 0; j < A.c; j++) {
            auto &x = A.at(i, j);
            if (!K.is_zero(x)) {
                M->col.push_back(j);
                M->val.push_back(std::move(x));
            }
        }
        M->ptr.push_back((uint32_t)M->col.size());
    }
    return M;
}
template <class F>
MatP<F> scaled(const F &K, const SMat<F> &C, const typename F::E &s) {
    if (K.is_zero(s)) return nullptr;
    auto M = std::make_shared<SMat<F>>(C);
    for (auto &x : M->val) {
        typename F::E t = K.make();
        K.mul(t, x, s);
        x = std::move(t);
    }
    return M;
}
template <class F>
MatP<F> identity(const F &K, uint32_t n) {
    if (n == 0) return nullptr;
    auto M = std::make_shared<SMat<F>>();
    M->r = M->c = n;
    for (uint32_t i = 0; i <= n; i++) M->ptr.push_back(i);
    for (uint32_t i = 0; i < n; i++) {
        M->col.push_back(i);
        typename F::E one = K.make();
        K.one(one);
        M->val.push_back(std::move(one));
    }
    return M;
}

// generalised binomial C(p, j) = p (p - 1) ... (p - j + 1) / j!, any integer p
inline void gbinom(fmpz_t r, slong p, ulong j) {
    if (p >= 0) {
        fmpz_bin_uiui(r, (ulong)p, j);
        return;
    }
    // C(-a, j) = (-1)^j C(a + j - 1, j)
    fmpz_bin_uiui(r, (ulong)(-p) + j - 1, j);
    if (j & 1) fmpz_neg(r, r);
}
// p (p - 1) ... (p - k + 1)
inline void falling_si(fmpz_t r, slong p, ulong k) {
    fmpz_one(r);
    for (ulong i = 0; i < k; i++) fmpz_mul_si(r, r, p - (slong)i);
}

// ------------------------------------------------------------------------ //
//  The engine                                                              //
// ------------------------------------------------------------------------ //

using Letter = std::pair<int, int>;                     // (generator, derivative)
using Word = std::vector<Letter>;
using ModeWord = std::vector<std::pair<int, long>>;     // (generator, physics mode), first outermost
using Entries = std::vector<std::pair<std::pair<int, int>, CoefData>>;   // ((row, column), coefficient)
   // ((row, column), coefficient)

struct LoweringEngineBase {
    virtual ~LoweringEngineBase() {}
    virtual uint32_t intern_word(const Word &w) = 0;
    virtual void set_ope(int g1, int g2, const PolesData &poles) = 0;
    virtual void set_splits(uint32_t w, const std::vector<ElemData> &products) = 0;
    virtual std::vector<uint32_t> missing_splits() = 0;
    virtual void set_top(int dim, const std::vector<std::vector<std::vector<CoefData>>> &zero_modes) = 0;
    virtual int extend(int nthreads, PollFn poll) = 0;
    virtual int depth() const = 0;
    virtual std::vector<int> dims() const = 0;
    virtual std::pair<ModeWord, int> basis_word(int d, int i) const = 0;
    virtual Entries gen_matrix(int g, long nn, int e) = 0;
    virtual Entries word_matrix(uint32_t w, long nn, int e, PollFn poll) = 0;
    virtual std::vector<double> stats() = 0;
    virtual void set_logging(int on) = 0;
};

template <class F>
class LoweringEngine : public LoweringEngineBase {
  public:
    using E = typename F::E;
    using M = MatP<F>;
    F K;
    int G;
    std::vector<int> wt;                    // generator weights
    // words: letters, tail id, head id (the one-letter word), weight
    struct WordRec {
        Word letters;
        uint32_t head = 0, tail = 0;
        int weight = 0;
    };
    std::vector<WordRec> words;
    std::map<Word, uint32_t> word_ids;
    // OPEs: ope[g1 * G + g2][j] = g1_(j) g2 as (word, coefficient)
    struct Lin {
        std::vector<std::pair<uint32_t, E>> terms;
    };
    std::vector<std::vector<Lin>> ope;
    // splits: per word, x_(i) B for i < wt x, scaled by -C(wt x, i + 1)
    std::vector<std::vector<Lin>> splits;
    std::vector<char> has_splits;
    // the module
    int D = -1;                             // known depths 0..D
    std::vector<int> dim;
    std::vector<std::vector<std::pair<ModeWord, int>>> basis;
    // generator modes: gen[g][e] maps nn -> matrix (null: zero); written only
    // between parallel phases
    std::vector<std::vector<std::map<long, M>>> gen;
    ConcCache<M> memo;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> misses{0};
    bool logging = false;
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();

    template <class... A>
    LoweringEngine(int ngens, const std::vector<int> &weights, A &&...field_args)
        : K(std::forward<A>(field_args)...), G(ngens), wt(weights) {
        if ((int)wt.size() != G || G < 1) throw std::invalid_argument("one weight per generator");
        for (int w : wt)
            if (w < 1) throw std::invalid_argument("generator weights must be positive integers");
        ope.assign((size_t)G * G, {});
        intern_word(Word());                // id 0: the vacuum
    }

    // -- words --------------------------------------------------------------- //
    uint32_t intern_word(const Word &w) override {
        auto it = word_ids.find(w);
        if (it != word_ids.end()) return it->second;
        WordRec r;
        r.letters = w;
        for (const Letter &l : w) {
            if (l.first < 0 || l.first >= G || l.second < 0) throw std::invalid_argument("bad letter");
            r.weight += wt[l.first] + l.second;
        }
        if (w.size() > 1) {
            r.head = intern_word(Word(w.begin(), w.begin() + 1));
            r.tail = intern_word(Word(w.begin() + 1, w.end()));
        }
        uint32_t id = (uint32_t)words.size();
        words.push_back(std::move(r));
        splits.emplace_back();
        has_splits.push_back(0);
        word_ids.emplace(w, id);
        return id;
    }
    Lin lin_in(const ElemData &d) {
        Lin l;
        for (const auto &t : d) {
            if (t.first >= words.size()) throw std::out_of_range("unknown word id");
            E c = K.make();
            K.from_data(c, t.second);
            if (!K.is_zero(c)) l.terms.emplace_back(t.first, std::move(c));
        }
        return l;
    }
    void set_ope(int g1, int g2, const PolesData &poles) override {
        if (g1 < 0 || g2 < 0 || g1 >= G || g2 >= G) throw std::out_of_range("generator index");
        std::vector<Lin> p;
        for (const auto &el : poles) p.push_back(lin_in(el));
        ope[(size_t)g1 * G + g2] = std::move(p);
    }
    // products[i] = x_(i) B for i = 0 .. wt x - 1 (w = :x B:)
    void set_splits(uint32_t w, const std::vector<ElemData> &products) override {
        if (w >= words.size() || words[w].letters.size() < 2) throw std::invalid_argument("not a composite word");
        int wx = words[words[w].head].weight;
        if ((int)products.size() != wx) throw std::invalid_argument("need x_(i) B for i < wt x");
        std::vector<Lin> s;
        Z b;
        E cb = K.make();
        for (int i = 0; i < wx; i++) {
            Lin l = lin_in(products[i]);
            fmpz_bin_uiui(b.v, (ulong)wx, (ulong)(i + 1));
            fmpz_neg(b.v, b.v);
            K.set_fmpz(cb, b.v);
            for (auto &t : l.terms) {
                E u = K.make();
                K.mul(u, t.second, cb);
                t.second = std::move(u);
            }
            s.push_back(std::move(l));
        }
        splits[w] = std::move(s);
        has_splits[w] = 1;
    }
    // composite words reachable from the OPEs (tails, split products) whose
    // splits are not yet given
    std::vector<uint32_t> missing_splits() override {
        std::vector<char> seen(words.size(), 0);
        std::vector<uint32_t> todo, out;
        for (auto &p : ope)
            for (auto &l : p)
                for (auto &t : l.terms) todo.push_back(t.first);
        while (!todo.empty()) {
            uint32_t w = todo.back();
            todo.pop_back();
            if (w >= seen.size()) seen.resize(words.size(), 0);
            if (seen[w]) continue;
            seen[w] = 1;
            if (words[w].letters.size() < 2) continue;
            todo.push_back(words[w].tail);
            if (!has_splits[w]) {
                out.push_back(w);
                continue;
            }
            for (auto &l : splits[w])
                for (auto &t : l.terms) todo.push_back(t.first);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    // -- the top ----------------------------------------------------------- //
    // zero_modes[g][i][j] = entry (i, j) of g's zero mode on the top
    void set_top(int k, const std::vector<std::vector<std::vector<CoefData>>> &zm) override {
        if (D >= 0) throw std::logic_error("the top is already set");
        if (k < 1) throw std::invalid_argument("the top must not be zero");
        if ((int)zm.size() != G) throw std::invalid_argument("one zero mode per generator");
        gen.assign(G, {});
        for (int g = 0; g < G; g++) {
            if ((int)zm[g].size() != k) throw std::invalid_argument("zero mode of the wrong size");
            DMat<F> A(K, k, k);
            for (int i = 0; i < k; i++) {
                if ((int)zm[g][i].size() != k) throw std::invalid_argument("zero mode of the wrong size");
                for (int j = 0; j < k; j++) K.from_data(A.at(i, j), zm[g][i][j]);
            }
            gen[g].emplace_back();
            gen[g][0][wt[g] - 1] = compress(K, A);
        }
        D = 0;
        dim = {k};
        basis = {{}};
        for (int j = 0; j < k; j++) basis[0].push_back({ModeWord(), j});
    }

    // -- modes --------------------------------------------------------------- //
    static uint64_t mkey(uint32_t w, long nn, int e) {
        if (nn < -32768 || nn > 32767 || e < 0 || e > 65535) throw std::out_of_range("mode or depth out of range");
        return ((uint64_t)w << 32) | ((uint64_t)(uint16_t)(nn + 32768) << 16) | (uint64_t)e;
    }
    int target(int w, long nn, int e) const { return (int)(e + w - nn - 1); }
    void check_stop() {
        if (stop.load(std::memory_order_relaxed)) throw Interrupted();
        if (tl_wid < 0 && tl_poll && (++tl_misses % POLL_EVERY) == 0 && tl_poll()) {
            stop.store(true);
            throw Interrupted();
        }
    }
    // generator g's mode (nn) on depth e; null: zero
    const SMat<F> *gmode(int g, long nn, int e) const {
        int t = target(wt[g], nn, e);
        if (t < 0 || e < 0) return nullptr;
        if (e > D || t > D) throw std::logic_error("a generator mode outside the known depths");
        auto &m = gen[g][e];
        auto it = m.find(nn);
        if (it == m.end()) throw std::logic_error("a generator mode that is not known yet");
        return it->second.get();
    }
    // the mode (nn) of the word w on depth e; null: zero
    const SMat<F> *mode(uint32_t w, long nn, int e) {
        const WordRec &W = words[w];
        int t = target(W.weight, nn, e);
        if (t < 0 || e < 0) return nullptr;
        if (W.letters.size() == 1) {
            int g = W.letters[0].first, k = W.letters[0].second;
            if (k == 0) return gmode(g, nn, e);
        }
        return memo.get(mkey(w, nn, e), [&]() { return compute_mode(w, nn, e); }).get();
    }
    M compute_mode(uint32_t w, long nn, int e) {
        check_stop();
        misses.fetch_add(1, std::memory_order_relaxed);
        const WordRec &W = words[w];
        int t = target(W.weight, nn, e);
        if (W.letters.empty()) return nn == -1 ? identity(K, (uint32_t)dim[e]) : nullptr;
        if (W.letters.size() == 1) {        // (d^k g)_(nn) = (-1)^k nn (nn-1) ... (nn-k+1) g_(nn-k)
            int g = W.letters[0].first, k = W.letters[0].second;
            Z c;
            falling_si(c.v, nn, (ulong)k);
            if (k & 1) fmpz_neg(c.v, c.v);
            if (fmpz_is_zero(c.v)) return nullptr;
            const SMat<F> *Gm = gmode(g, nn - k, e);
            if (!Gm) return nullptr;
            E s = K.make();
            K.set_fmpz(s, c.v);
            return scaled(K, *Gm, s);
        }
        if (!has_splits[w]) throw std::logic_error("a composite word without its split products");
        uint32_t x = W.head, B = W.tail;
        int wx = words[x].weight, wB = words[B].weight;
        DMat<F> A(K, (uint32_t)dim[t], (uint32_t)dim[e]);
        bool any = false;
        // sum_j x_(-1-j) B_(nn+j): B lowers further than the target, x raises
        for (long j = 0; j < std::max<long>(e + wB - nn, 0); j++) {
            const SMat<F> *Bm = mode(B, nn + j, e);
            if (!Bm) continue;
            const SMat<F> *Xm = mode(x, -1 - j, target(wB, nn + j, e));
            if (!Xm) continue;
            acc_mul(K, A, *Xm, *Bm, (const E *)nullptr);
            any = true;
        }
        // sum_j B_(nn-1-j) x_(j)
        for (long j = 0; j < e + wx; j++) {
            if (wx - j - 1 < 0) {               // x_(j) lowers: as it stands
                const SMat<F> *Xm = mode(x, j, e);
                if (!Xm) continue;
                const SMat<F> *Bm = mode(B, nn - 1 - j, target(wx, j, e));
                if (!Bm) continue;
                acc_mul(K, A, *Bm, *Xm, (const E *)nullptr);
                any = true;
            } else {                            // x_(j) B_(m) - [x_(j), B_(m)]
                long m = nn - 1 - j;
                const SMat<F> *Bm = mode(B, m, e);
                if (!Bm) continue;
                const SMat<F> *Xm = mode(x, j, target(wB, m, e));
                if (!Xm) continue;
                acc_mul(K, A, *Xm, *Bm, (const E *)nullptr);
                any = true;
            }
        }
        // the commutator terms, summed over j < wt x:
        // - sum_i C(wt x, i + 1) (x_(i) B)_(nn-1-i)  (coefficients in splits)
        for (int i = 0; i < wx; i++) {
            for (const auto &tc : splits[w][i].terms) {
                const SMat<F> *Cm = mode(tc.first, nn - 1 - i, e);
                if (!Cm) continue;
                acc_scaled(K, A, *Cm, tc.second);
                any = true;
            }
        }
        return any ? compress(K, A) : nullptr;
    }
    // [g1_(p), g2_(q)] on depth e, added to A (with the factor s if given)
    bool add_commutator(DMat<F> &A, int g1, long p, int g2, long q, int e) {
        int t = e + wt[g1] + wt[g2] - (int)p - (int)q - 2;
        if (t < 0) return false;
        bool any = false;
        const auto &poles = ope[(size_t)g1 * G + g2];
        Z b;
        E cb = K.make(), s = K.make();
        for (int j = 0; j < (int)poles.size(); j++) {
            gbinom(b.v, p, (ulong)j);
            if (fmpz_is_zero(b.v)) continue;
            K.set_fmpz(cb, b.v);
            for (const auto &tc : poles[j].terms) {
                const SMat<F> *Cm = mode(tc.first, p + q - j, e);
                if (!Cm) continue;
                K.mul(s, cb, tc.second);
                acc_scaled(K, A, *Cm, s);
                any = true;
            }
        }
        return any;
    }

    // -- prefetch ------------------------------------------------------------ //
    // The memo entries a phase will ask for, computed beforehand level by
    // level (word weight ascending) in parallel.  A composite word's mode
    // needs modes of words of smaller weight only (its head and tail, the
    // split products x_(i) B), so the entries of one level never wait for
    // each other; without this, threads asking for the same deep entries
    // wait for one thread computing their recursion.  The keys are found by
    // the recursion of compute_mode without arithmetic: a superset of the
    // entries asked for (a term skipped there because a factor is zero is
    // collected here), all inside the known depths.
    struct PKey {
        uint32_t w;
        long nn;
        int e;
    };
    struct Prefetch {
        std::unordered_set<uint64_t> seen;
        std::vector<std::vector<PKey>> level;       // by word weight
        int open = -1;      // a depth whose generator zero modes are not known yet
    };
    void collect(Prefetch &P, uint32_t w, long nn, int e) {
        const WordRec &W = words[w];
        int t = target(W.weight, nn, e);
        if (t < 0 || e < 0) return;
        if (W.letters.size() == 1 && W.letters[0].second == 0) return;     // a generator mode
        if (e > D || t > D || (e == P.open && t == P.open)) return;
        uint64_t key = mkey(w, nn, e);
        if (P.seen.count(key) || memo.ready(key)) return;
        P.seen.insert(key);
        if ((size_t)W.weight >= P.level.size()) P.level.resize((size_t)W.weight + 1);
        P.level[(size_t)W.weight].push_back({w, nn, e});
        if (W.letters.size() < 2 || !has_splits[w]) return;
        uint32_t x = W.head, B = W.tail;
        int wx = words[x].weight, wB = words[B].weight;
        for (long j = 0; j < std::max<long>(e + wB - nn, 0); j++) {
            collect(P, B, nn + j, e);
            int tB = target(wB, nn + j, e);
            if (tB >= 0) collect(P, x, -1 - j, tB);
        }
        for (long j = 0; j < e + wx; j++) {
            if (wx - j - 1 < 0) {
                collect(P, x, j, e);
                int tx = target(wx, j, e);
                if (tx >= 0) collect(P, B, nn - 1 - j, tx);
            } else {
                long m = nn - 1 - j;
                collect(P, B, m, e);
                int tB = target(wB, m, e);
                if (tB >= 0) collect(P, x, j, tB);
            }
        }
        for (int i = 0; i < wx; i++)
            for (const auto &tc : splits[w][i].terms) collect(P, tc.first, nn - 1 - i, e);
    }
    // the words' modes in [g1_(p), g2_(q)] on depth e (as add_commutator)
    void collect_commutator(Prefetch &P, int g1, long p, int g2, long q, int e) {
        if (e + wt[g1] + wt[g2] - (int)p - (int)q - 2 < 0) return;
        const auto &poles = ope[(size_t)g1 * G + g2];
        Z b;
        for (int j = 0; j < (int)poles.size(); j++) {
            gbinom(b.v, p, (ulong)j);
            if (fmpz_is_zero(b.v)) continue;
            for (const auto &tc : poles[j].terms) collect(P, tc.first, p + q - j, e);
        }
    }
    // compute the collected entries; returns their number
    size_t run_prefetch(Prefetch &P, int nthreads, PollFn poll) {
        size_t n = 0;
        P.seen.clear();
        for (auto &lv : P.level) {
            if (lv.empty()) continue;
            std::stable_sort(lv.begin(), lv.end(), [&](const PKey &a, const PKey &b) {
                int ta = target(words[a.w].weight, a.nn, a.e), tb = target(words[b.w].weight, b.nn, b.e);
                return (size_t)dim[a.e] * dim[ta] > (size_t)dim[b.e] * dim[tb];
            });
            parallel_for(lv.size(), nthreads, poll, [&](size_t k) {
                const PKey &q = lv[k];
                memo.get(mkey(q.w, q.nn, q.e), [&]() { return compute_mode(q.w, q.nn, q.e); });
            });
            n += lv.size();
            lv.clear();
            lv.shrink_to_fit();
        }
        return n;
    }

    // -- threads ------------------------------------------------------------- //
    // f(k) for k < n, in nthreads threads (dynamic), the caller polling
    template <class Fn>
    void parallel_for(size_t n, int nthreads, PollFn poll, Fn &&f) {
        stop.store(false);
        if (nthreads <= 1 || n <= 1) {
            PollGuard pg(poll);
            for (size_t k = 0; k < n; k++) f(k);
            return;
        }
        ErrorBox err;
        std::atomic<size_t> next{0};
        std::mutex dm;
        std::condition_variable dcv;
        int done = 0;
        auto work = [&](int wid) {
            tl_wid = wid;
            try {
                for (;;) {
                    if (stop.load(std::memory_order_relaxed)) break;
                    size_t i = next.fetch_add(1);
                    if (i >= n) break;
                    f(i);
                }
            } catch (...) {
                err.record(std::current_exception());
                stop.store(true);
            }
            flint_cleanup();
            {
                std::lock_guard<std::mutex> g(dm);
                done++;
            }
            dcv.notify_one();
        };
        int T = (int)std::min((size_t)nthreads, n);
        std::vector<std::thread> th;
        int started = 0;
        try {
            for (; started < T; started++) th.emplace_back(work, started);
        } catch (...) {
            err.record(std::current_exception());
            stop.store(true);
        }
        {
            std::unique_lock<std::mutex> lk(dm);
            while (done < started) {
                dcv.wait_for(lk, std::chrono::milliseconds(POLL_MS));
                if (done < started && poll && !stop.load()) {
                    lk.unlock();
                    if (poll()) stop.store(true);
                    lk.lock();
                }
            }
        }
        for (auto &t : th) t.join();
        if (stop.load() && !err.e) err.record(std::make_exception_ptr(Interrupted()));
        err.rethrow();
    }

    // -- linear algebra ------------------------------------------------------ //
    // reduced row echelon form of a dense matrix (rows x cols, row-major) in
    // place; returns the pivot columns (leftmost first)
    std::vector<uint32_t> rref(std::vector<E> &a, uint32_t rows, uint32_t cols) {
        std::vector<uint32_t> piv;
        uint32_t r = 0;
        E inv = K.make(), f = K.make();
        std::vector<uint32_t> nzc;
        for (uint32_t c = 0; c < cols && r < rows; c++) {
            uint32_t p = r;
            while (p < rows && K.is_zero(a[(size_t)p * cols + c])) p++;
            if (p == rows) continue;
            if (p != r)
                for (uint32_t j = c; j < cols; j++) std::swap(a[(size_t)p * cols + j], a[(size_t)r * cols + j]);
            E *pr = &a[(size_t)r * cols];
            K.inv(inv, pr[c]);
            nzc.clear();
            for (uint32_t j = c; j < cols; j++)
                if (!K.is_zero(pr[j])) {
                    E u = K.make();
                    K.mul(u, pr[j], inv);
                    pr[j] = std::move(u);
                    nzc.push_back(j);
                }
            for (uint32_t i = 0; i < rows; i++) {
                if (i == r) continue;
                E *ri = &a[(size_t)i * cols];
                if (K.is_zero(ri[c])) continue;
                f = ri[c];
                for (uint32_t j : nzc) K.submul(ri[j], f, pr[j]);
            }
            piv.push_back(c);
            r++;
            if ((r & 15) == 0) check_stop();
        }
        return piv;
    }
    // the dense block (rows of A) of a matrix into a row-major buffer
    void put(std::vector<E> &buf, uint32_t cols, uint32_t r0, uint32_t c0, const SMat<F> &A) {
        for (uint32_t i = 0; i < A.r; i++)
            for (uint32_t p = A.ptr[i]; p < A.ptr[i + 1]; p++)
                buf[(size_t)(r0 + i) * cols + c0 + A.col[p]] = A.val[p];
    }

    // -- one depth more ------------------------------------------------------ //
    int extend(int nthreads, PollFn poll) override {
        if (D < 0) throw std::logic_error("set the top first");
        if (!missing_splits().empty()) throw std::logic_error("split products missing (missing_splits)");
        MaybeGmpMalloc gm(nthreads);
        const int N = D + 1;
        try {
            return extend_body(nthreads, poll);
        } catch (...) {
            // undo depth N: its basis and every generator mode into or out of
            // it (memo entries stay: they are determined by the earlier
            // depths, or are recomputed identically)
            if (D == N) {
                D = N - 1;
                dim.resize(N);
                basis.resize(N);
                for (int g = 0; g < G; g++) {
                    gen[g].resize(N);
                    for (int n = 1; n <= N; n++) gen[g][N - n].erase(-n + wt[g] - 1);
                }
            }
            throw;
        }
    }
    int extend_body(int nthreads, PollFn poll) {
        auto t0 = std::chrono::steady_clock::now();
        const int N = D + 1;
        // candidates g_{-n} b, by (word length, word)
        struct Cand {
            ModeWord modes;
            int u, g, n, b;
        };
        std::vector<Cand> cands;
        for (int n = 1; n <= N; n++)
            for (int g = 0; g < G; g++)
                for (int b = 0; b < dim[N - n]; b++) {
                    Cand c;
                    c.modes.push_back({g, -(long)n});
                    const auto &bw = basis[N - n][b];
                    c.modes.insert(c.modes.end(), bw.first.begin(), bw.first.end());
                    c.u = bw.second;
                    c.g = g;
                    c.n = n;
                    c.b = b;
                    cands.push_back(std::move(c));
                }
        std::stable_sort(cands.begin(), cands.end(), [](const Cand &x, const Cand &y) {
            if (x.modes.size() != y.modes.size()) return x.modes.size() < y.modes.size();
            if (x.modes != y.modes) return x.modes < y.modes;
            return x.u < y.u;
        });
        // column of each (g, n, b)
        std::vector<std::vector<std::vector<uint32_t>>> colof(G, std::vector<std::vector<uint32_t>>(N + 1));
        for (int g = 0; g < G; g++)
            for (int n = 1; n <= N; n++) colof[g][n].assign(dim[N - n], 0);
        for (uint32_t i = 0; i < cands.size(); i++) colof[cands[i].g][cands[i].n][cands[i].b] = i;
        // phase A: the blocks (g2, m; g, n) of the lowering data, in parallel
        struct Task {
            int g2, m, g, n;
        };
        std::vector<Task> tasks;
        for (int g2 = 0; g2 < G; g2++)
            for (int m = 1; m <= N; m++)
                for (int g = 0; g < G; g++)
                    for (int n = 1; n <= N; n++)
                        if (dim[N - n] > 0 && dim[N - m] > 0) tasks.push_back({g2, m, g, n});
        std::vector<M> blocks(tasks.size());
        // larger targets first
        std::vector<size_t> order(tasks.size());
        for (size_t k = 0; k < order.size(); k++) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
            return (size_t)dim[N - tasks[x].m] * dim[N - tasks[x].n] > (size_t)dim[N - tasks[y].m] * dim[N - tasks[y].n];
        });
        {
            Prefetch P;
            for (const Task &T : tasks)
                collect_commutator(P, T.g2, T.m + wt[T.g2] - 1, T.g, -T.n + wt[T.g] - 1, N - T.n);
            last_prefetched = run_prefetch(P, nthreads, poll);
        }
        D = N;      // depth N exists from here (its modes are set below)
        dim.push_back(0);
        basis.emplace_back();
        for (int g = 0; g < G; g++) gen[g].emplace_back();
        {
            parallel_for(order.size(), nthreads, poll, [&](size_t k) {
                const Task &T = tasks[order[k]];
                long p = T.m + wt[T.g2] - 1, nn = -T.n + wt[T.g] - 1;
                DMat<F> A(K, (uint32_t)dim[N - T.m], (uint32_t)dim[N - T.n]);
                bool any = false;
                const SMat<F> *L1 = gmode(T.g2, p, N - T.n);       // g'_m b, then g_{-n}
                if (L1) {
                    const SMat<F> *R1 = gmode(T.g, nn, N - T.n - T.m);
                    if (R1) {
                        acc_mul(K, A, *R1, *L1, (const E *)nullptr);
                        any = true;
                    }
                }
                any = add_commutator(A, T.g2, p, T.g, nn, N - T.n) || any;
                blocks[order[k]] = any ? compress(K, A) : nullptr;
            });
        }
        auto t1 = std::chrono::steady_clock::now();
        // the lowering data: rows = blocks (g2, m), columns = candidates
        std::vector<uint32_t> roff((size_t)G * (N + 1) + 1, 0);
        uint32_t rows = 0;
        for (int g2 = 0; g2 < G; g2++)
            for (int m = 1; m <= N; m++) {
                roff[(size_t)g2 * (N + 1) + m] = rows;
                rows += (uint32_t)dim[N - m];
            }
        uint32_t ncand = (uint32_t)cands.size();
        std::vector<E> L((size_t)rows * ncand, K.make());
        for (size_t k = 0; k < tasks.size(); k++) {
            if (!blocks[k]) continue;
            const Task &T = tasks[k];
            const SMat<F> &A = *blocks[k];
            uint32_t r0 = roff[(size_t)T.g2 * (N + 1) + T.m];
            for (uint32_t i = 0; i < A.r; i++)
                for (uint32_t q = A.ptr[i]; q < A.ptr[i + 1]; q++)
                    L[(size_t)(r0 + i) * ncand + colof[T.g][T.n][A.col[q]]] = A.val[q];
        }
        blocks.clear();
        std::vector<E> R = L;           // keep L: the lowering modes from N are its pivot columns
        std::vector<uint32_t> piv;
        {
            PollGuard pg(poll);
            piv = rref(R, rows, ncand);
        }
        const uint32_t dN = (uint32_t)piv.size();
        dim[N] = (int)dN;
        for (uint32_t i : piv) basis[N].push_back({cands[i].modes, cands[i].u});
        // raising into N: column b of g_{-n} on depth N - n = the RREF column of (g, n, b)
        for (int n = 1; n <= N; n++)
            for (int g = 0; g < G; g++) {
                if (dim[N - n] == 0) continue;
                DMat<F> A(K, dN, (uint32_t)dim[N - n]);
                for (int b = 0; b < dim[N - n]; b++) {
                    uint32_t c = colof[g][n][b];
                    for (uint32_t i = 0; i < dN; i++) A.at(i, b) = R[(size_t)i * ncand + c];
                }
                gen[g][N - n][-n + wt[g] - 1] = compress(K, A);
            }
        R.clear();
        // lowering from N: the lowering data of the basis (pivot columns of L)
        for (int g2 = 0; g2 < G; g2++)
            for (int m = 1; m <= N; m++) {
                uint32_t h = (uint32_t)dim[N - m], r0 = roff[(size_t)g2 * (N + 1) + m];
                DMat<F> A(K, h, dN);
                for (uint32_t i = 0; i < h; i++)
                    for (uint32_t j = 0; j < dN; j++) A.at(i, j) = L[(size_t)(r0 + i) * ncand + piv[j]];
                gen[g2][N][m + wt[g2] - 1] = compress(K, A);
            }
        // modes lowering below depth 0 are zero (gmode); generator modes
        // lowering N to N - m for m > N: target < 0, zero
        std::vector<E> Lb((size_t)rows * dN, K.make());
        for (uint32_t i = 0; i < rows; i++)
            for (uint32_t j = 0; j < dN; j++) Lb[(size_t)i * dN + j] = L[(size_t)i * ncand + piv[j]];
        L.clear();
        auto t2 = std::chrono::steady_clock::now();
        // phase B: zero modes at N by their lowering data,
        //   g''_(q) g'_(p0) c = g'_(p0) g''_(q) c + [g''_(q), g'_(p0)] c
        struct ZTask {
            int g1, g2, k;
        };
        std::vector<ZTask> zt;
        for (int g1 = 0; g1 < G; g1++)
            for (int g2 = 0; g2 < G; g2++)
                for (int k = 1; k <= N; k++)
                    if (dim[N - k] > 0 && dN > 0) zt.push_back({g1, g2, k});
        {
            Prefetch P;
            P.open = N;
            for (const ZTask &T : zt) collect_commutator(P, T.g2, T.k + wt[T.g2] - 1, T.g1, wt[T.g1] - 1, N);
            last_prefetched += run_prefetch(P, nthreads, poll);
        }
        std::vector<M> zb(zt.size());
        {
            parallel_for(zt.size(), nthreads, poll, [&](size_t k) {
                const ZTask &T = zt[k];
                long p0 = wt[T.g1] - 1, q = T.k + wt[T.g2] - 1;
                DMat<F> A(K, (uint32_t)dim[N - T.k], dN);
                bool any = false;
                const SMat<F> *Lk = gmode(T.g2, q, N);
                if (Lk) {
                    const SMat<F> *Z0 = gmode(T.g1, p0, N - T.k);
                    if (Z0) {
                        acc_mul(K, A, *Z0, *Lk, (const E *)nullptr);
                        any = true;
                    }
                }
                any = add_commutator(A, T.g2, q, T.g1, p0, N) || any;
                zb[k] = any ? compress(K, A) : nullptr;
            });
        }
        // solve Lb X_g1 = data_g1 for all g1 at once: RREF of [Lb | data]
        uint32_t cols = dN + (uint32_t)G * dN;
        std::vector<E> S((size_t)rows * cols, K.make());
        for (uint32_t i = 0; i < rows; i++)
            for (uint32_t j = 0; j < dN; j++) S[(size_t)i * cols + j] = Lb[(size_t)i * dN + j];
        Lb.clear();
        for (size_t k = 0; k < zt.size(); k++) {
            if (!zb[k]) continue;
            const ZTask &T = zt[k];
            put(S, cols, roff[(size_t)T.g2 * (N + 1) + T.k], dN + (uint32_t)T.g1 * dN, *zb[k]);
        }
        zb.clear();
        std::vector<uint32_t> piv2;
        {
            PollGuard pg(poll);
            piv2 = rref(S, rows, cols);
        }
        if (piv2.size() != dN || (dN > 0 && piv2.back() >= dN))
            throw std::runtime_error("the zero modes at depth " + std::to_string(N) +
                                     " are not determined by their lowering data (not a module?)");
        for (int g1 = 0; g1 < G; g1++) {
            DMat<F> A(K, dN, dN);
            for (uint32_t i = 0; i < dN; i++)
                for (uint32_t j = 0; j < dN; j++) A.at(i, j) = S[(size_t)i * cols + dN + (uint32_t)g1 * dN + j];
            gen[g1][N][wt[g1] - 1] = compress(K, A);
        }
        auto t3 = std::chrono::steady_clock::now();
        last_times = {std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count(),
                      std::chrono::duration<double>(t3 - t2).count()};
        last_cands = ncand;
        return (int)dN;
    }
    std::vector<double> last_times{0, 0, 0};
    size_t last_cands = 0;
    size_t last_prefetched = 0;

    // -- access ------------------------------------------------------------ //
    int depth() const override { return D; }
    std::vector<int> dims() const override { return dim; }
    std::pair<ModeWord, int> basis_word(int d, int i) const override {
        if (d < 0 || d > D || i < 0 || i >= dim[d]) throw std::out_of_range("no such basis vector");
        return basis[d][i];
    }
    Entries entries(const SMat<F> *A) const {
        Entries out;
        if (!A) return out;
        for (uint32_t i = 0; i < A->r; i++)
            for (uint32_t p = A->ptr[i]; p < A->ptr[i + 1]; p++)
                out.push_back({{(int)i, (int)A->col[p]}, K.to_data(A->val[p])});
        return out;
    }
    Entries gen_matrix(int g, long nn, int e) override {
        if (g < 0 || g >= G) throw std::out_of_range("generator index");
        return entries(gmode(g, nn, e));
    }
    Entries word_matrix(uint32_t w, long nn, int e,
                                                                     PollFn poll) override {
        if (w >= words.size()) throw std::out_of_range("unknown word id");
        int t = target(words[w].weight, nn, e);
        if (e < 0 || e > D || t > D) throw std::out_of_range("a mode outside the known depths");
        stop.store(false);
        PollGuard pg(poll);
        return entries(mode(w, nn, e));
    }
    // memo entries, nonzero matrices, stored entries, approximate bytes,
    // misses, then the last extension's times (phase A, elimination, phase B)
    // and candidates, memo entries computed by prefetch
    std::vector<double> stats() override {
        double n = 0, nz = 0, ent = 0, bytes = 0;
        for (size_t s = 0; s < memo.NS; s++) {
            std::lock_guard<std::mutex> lk(memo.shards[s].m);
            for (auto &kv : memo.shards[s].map) {
                n++;
                bytes += 96;
                const M &A = kv.second->val;
                if (!A) continue;
                nz++;
                ent += (double)A->nnz();
                bytes += 64 + 4.0 * (A->r + 1) + (4.0 + sizeof(E)) * A->nnz();
                for (auto &x : A->val) bytes += (double)K.heap_bytes(x);
            }
        }
        return {n, nz, ent, bytes, (double)misses.load(), last_times[0], last_times[1], last_times[2],
                (double)last_cands, (double)last_prefetched};
    }
    void set_logging(int on) override { logging = on != 0; }
};

inline LoweringEngineBase *make_lowering_engine(const std::string &kind, int ngens, const std::vector<int> &weights,
                                                unsigned long modulus) {
    if (kind == "q") return new LoweringEngine<FieldQ>(ngens, weights);
    if (kind == "nmod") return new LoweringEngine<FieldP>(ngens, weights, (ulong)modulus);
    throw std::invalid_argument("unknown field kind " + kind);
}

}  // namespace vae
