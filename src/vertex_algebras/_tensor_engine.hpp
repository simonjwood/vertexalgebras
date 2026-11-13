// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) Cardiff University 2026
// C++ engine of tensor products of universal and lattice vertex algebras:
// TensorProductVertexAlgebra._tprod / _tderiv in C++, on factor engines
// (universal Engine<Ring>, LatticeEngine<Ring>) that this engine OWNS and
// that share its coefficient ring and stop flag, so factor results are used
// without any conversion.  No Python objects in here: the Cython wrapper
// _tensor_engine.pyx converts at the boundary; the
// cocycle values of lattice factors come from Python per call.
//
//   (a (x) b)_(n) (c (x) d) = (-1)^{|b||c|} sum_k a_(k) c (x) b_(n-1-k) d,
//   k in [n-1-nmax(b, d), nmax(a, c)], recursively over the factors.
//
// Keys: vectors of factor ids (one per factor), interned; suffixes (factors
// i, ..., r-1) are interned in the same table, their length gives i.
#pragma once
#include "_universal_engine.hpp"
#include "_lattice_engine.hpp"
#include <map>

namespace vae {

// what the Cython wrapper sees
struct TensorEngineBase {
    virtual ~TensorEngineBase() {}
    // factors, in order, before any computation
    virtual int add_universal(int ngens, const std::vector<int> &parities) = 0;
    virtual int add_lattice(const std::vector<std::vector<long>> &gram) = 0;
    virtual void set_ope(int f, int i, int j, const PolesData &poles) = 0;   // poles in factor ids
    virtual uint32_t intern_universal(int f, const Letters &letters) = 0;
    virtual uint32_t intern_lattice(int f, const Letters &letters, const std::vector<long> &q) = 0;
    virtual Letters universal_letters(int f, uint32_t id) = 0;
    virtual Letters lattice_letters(int f, uint32_t id) = 0;
    virtual std::vector<long> lattice_q(int f, uint32_t id) = 0;
    virtual uint32_t intern(const std::vector<uint32_t> &factor_ids) = 0;
    virtual std::vector<uint32_t> components(uint32_t id) = 0;
    // eps[f] for the lattice factors (empty for universal ones)
    virtual PolesData ope_mono(uint32_t a, uint32_t b, const std::vector<EpsData> &eps, PollFn poll) = 0;
    virtual ElemData no_mono(uint32_t a, uint32_t b, const std::vector<EpsData> &eps, PollFn poll) = 0;
    virtual ElemData nprod_mono(uint32_t a, long n, uint32_t b, const std::vector<EpsData> &eps,
                                PollFn poll) = 0;
    virtual ElemData deriv_mono(uint32_t m, int n, PollFn poll) = 0;
    virtual PolesData ope_el(const ElemData &A, const ElemData &B, const std::vector<EpsData> &eps,
                             int nthreads, PollFn poll) = 0;
    virtual std::vector<PolesData> ope_el_many(const ElemData &A, const std::vector<ElemData> &Bs,
                                               const std::vector<EpsData> &eps, int nthreads, PollFn poll) = 0;
    virtual ElemData nprod_el(const ElemData &A, long n, const ElemData &B, const std::vector<EpsData> &eps,
                              int nthreads, PollFn poll) = 0;
    virtual ElemData deriv_el(const ElemData &A, int n, int nthreads, PollFn poll) = 0;
    virtual std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &els, const std::vector<int> &ia,
                                                const std::vector<long> &ns, const std::vector<int> &ib,
                                                const std::vector<EpsData> &eps, int nthreads, PollFn poll) = 0;
    virtual std::vector<size_t> cache_sizes() const = 0;
};

template <class Ring>
class TensorEngine : public EngineCore<TensorEngine<Ring>, Ring>, public TensorEngineBase {
  public:
    using Core = EngineCore<TensorEngine<Ring>, Ring>;
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
    using Core::elem_in;
    using Core::elem_out;
    using Core::poles_out;
    using Core::wire;
    using Core::run_tasks;
    using Core::check_tasks;

    using UEng = Engine<Ring>;
    using LEng = LatticeEngine<Ring>;
    using EpsMap = typename LEng::EpsMap;

    // jobs: J_OPE a = pair id, b = N (all n = 0..N); J_NPROD a = pair id,
    // b = n (offset); J_DERIV a = key, b = order
    enum : int { J_OPE = 0, J_NPROD = 1, J_DERIV = 2 };
    static constexpr long NOFF = 1L << 30;

    struct Factor {
        std::unique_ptr<UEng> u;
        std::unique_ptr<LEng> l;
    };
    std::vector<Factor> fac;
    MonoTable ptab;                               // pairs of suffix ids
    CMap<uint32_t> cons_map;                      // pk(factor id, suffix id) -> suffix id
    ConcCache<Elem> tp_cache;                     // pk(pair, n + NOFF): tprod
    ConcCache<Elem> fnp_cache;                    // pk(factor pair, n + NOFF): universal n < -1
    ConcCache<std::vector<Elem>> lat_cache;       // pk(factor pair, nlo + NOFF): lattice n-products
    ConcCache<Elem> td_cache;                     // pk(suffix, order): derivatives
    MonoTable fptab;                              // factor pairs (f, a, c)
    std::vector<EpsMap> eps_now;                  // the call's cocycle values per factor

    template <class... A>
    explicit TensorEngine(A &&...ring_args) : Core(std::forward<A>(ring_args)...) {
        monos.tails = ptab.tails = fptab.tails = false;
        wire(tp_cache);
        wire(fnp_cache);
        wire(lat_cache);
        wire(td_cache);
    }

    // -- factors ---------------------------------------------------------- //
    void need_no_keys() const {
        if (monos.n) throw std::logic_error("factors must be added before keys are interned");
    }
    int add_universal(int ngens, const std::vector<int> &parities) override {
        need_no_keys();
        Factor f;
        f.u.reset(new UEng(ngens, parities, this->ring_ptr));
        f.u->stop = this->stop;
        // blocked waits in the factor's caches lend THIS engine's run token
        wire(f.u->gen_cache);
        wire(f.u->ope_cache);
        wire(f.u->nol_cache);
        wire(f.u->noi_cache);
        wire(f.u->nom_cache);
        wire(f.u->der_cache);
        fac.push_back(std::move(f));
        return (int)fac.size() - 1;
    }
    int add_lattice(const std::vector<std::vector<long>> &gram) override {
        need_no_keys();
        Factor f;
        f.l.reset(new LEng(gram, this->ring_ptr));
        f.l->stop = this->stop;
        wire(f.l->schur_cache);
        wire(f.l->eplus_cache);
        wire(f.l->cser_cache);
        wire(f.l->der_cache);
        fac.push_back(std::move(f));
        eps_now.resize(fac.size());
        return (int)fac.size() - 1;
    }
    UEng &U(int f) {
        if (f < 0 || f >= (int)fac.size() || !fac[f].u) throw std::invalid_argument("not a universal factor");
        return *fac[f].u;
    }
    LEng &L(int f) {
        if (f < 0 || f >= (int)fac.size() || !fac[f].l) throw std::invalid_argument("not a lattice factor");
        return *fac[f].l;
    }
    void set_ope(int f, int i, int j, const PolesData &poles) override { U(f).set_ope(i, j, poles); }
    uint32_t intern_universal(int f, const Letters &letters) override { return U(f).intern(letters); }
    uint32_t intern_lattice(int f, const Letters &letters, const std::vector<long> &q) override {
        return L(f).intern(letters, q);
    }
    Letters universal_letters(int f, uint32_t id) override { return U(f).letters(id); }
    Letters lattice_letters(int f, uint32_t id) override { return L(f).key_letters(id); }
    std::vector<long> lattice_q(int f, uint32_t id) override { return L(f).key_q(id); }

    // -- keys and suffixes ---------------------------------------------------- //
    int nf() const { return (int)fac.size(); }
    const std::vector<uint32_t> &comps(uint32_t s) const { return monos[s].letters; }
    int start(uint32_t s) const { return nf() - (int)comps(s).size(); }     // first factor of a suffix
    uint32_t suffix_tail(uint32_t s) {
        const auto &v = comps(s);
        return intern_codes(std::vector<uint32_t>(v.begin() + 1, v.end()));
    }
    uint32_t cons(uint32_t k1, uint32_t s2) {           // (k1,) + suffix s2
        uint64_t k = pk(k1, s2);
        uint32_t v;
        if (cons_map.get(k, v)) return v;
        std::vector<uint32_t> m{k1};
        const auto &t = comps(s2);
        m.insert(m.end(), t.begin(), t.end());
        v = intern_codes(m);
        cons_map.put(k, v);
        return v;
    }
    uint32_t pair_id(uint32_t a, uint32_t c) {
        std::lock_guard<std::mutex> g(ptab.m);
        return ptab.intern_locked(std::vector<uint32_t>{a, c});
    }
    uint32_t fpair_id(int f, uint32_t a, uint32_t c) {
        std::lock_guard<std::mutex> g(fptab.m);
        return fptab.intern_locked(std::vector<uint32_t>{(uint32_t)f, a, c});
    }

    // -- factor operations ----------------------------------------------------- //
    int fparity(int f, uint32_t k) {
        if (fac[f].u) return fac[f].u->monos[k].par;
        auto s = fac[f].l->split(k);
        std::vector<long> q = fac[f].l->qvec(s.second);
        long b = fac[f].l->form(q, q);
        return (int)(((b % 2) + 2) % 2);
    }
    long fnmax(int f, uint32_t a, uint32_t c) {
        if (fac[f].u) return (long)fac[f].u->ope_mono_c(a, c).size() - 1;
        return fac[f].l->nmax(a, c);
    }
    // the lattice factor's a_(n) c for n = nlo, ..., nmax (one call: the
    // regrouped series serve all n)
    const std::vector<Elem> &lat_range(int f, uint32_t a, uint32_t c, long nlo) {
        uint32_t fp = fpair_id(f, a, c);
        return lat_cache.get(pk(fp, (uint32_t)(nlo + NOFF)), [&]() -> std::vector<Elem> {
            check();
            LEng &l = *fac[f].l;
            long N = l.nmax(a, c);
            std::vector<Elem> out;
            if (N < nlo) return out;
            C one(R);
            R.one(one);
            std::vector<typename LEng::Item> items{typename LEng::Item{a, c, &one}};
            std::vector<long> ns;
            for (long n = nlo; n <= N; n++) ns.push_back(n);
            std::map<long, Elem> res = l.sums(l.plan(items, ns), eps_now[(size_t)f], 1);
            for (long n = nlo; n <= N; n++) out.push_back(std::move(res[n]));
            return out;
        });
    }
    // factor f's a_(n) c (nlo: the lowest n this recursion asks of the pair)
    const Elem *fnprod(int f, uint32_t a, long n, uint32_t c, long nlo, Elem &tmp) {
        if (fac[f].u) {
            UEng &u = *fac[f].u;
            if (n >= 0) {
                const Poles &P = u.ope_mono_c(a, c);
                return (size_t)n < P.size() ? &P[(size_t)n] : nullptr;
            }
            if (n == -1) return &u.no_mono_c(a, c);
            uint32_t fp = fpair_id(f, a, c);
            return &fnp_cache.get(pk(fp, (uint32_t)(n + NOFF)), [&]() -> Elem {
                check();
                int k = (int)(-1 - n);
                Elem res;
                const Elem &dk = u.deriv_mono_c(a, k);
                for (const auto &kv : dk) add_scaled(res, u.no_mono_c(kv.first, c), kv.second);
                Z f1, one(1);
                fact(f1.v, (ulong)k);
                Elem out;
                this->add_scaled_z(out, res, one.v, f1.v);
                return out;
            });
        }
        (void)tmp;
        const std::vector<Elem> &v = lat_range(f, a, c, nlo);
        if (n < nlo || n - nlo >= (long)v.size()) return nullptr;
        return &v[(size_t)(n - nlo)];
    }
    const Elem &fderiv(int f, uint32_t m, int n) {
        if (fac[f].u) return fac[f].u->deriv_mono_c(m, n);
        return fac[f].l->deriv_c(m, n);
    }

    // output order: by the factors' keys (their contents), not by ids
    // (ids depend on the order of interning, so on the threads)
    bool key_less(uint32_t a, uint32_t b) const {
        const auto &va = comps(a), &vb = comps(b);
        for (size_t t = 0; t < va.size() && t < vb.size(); t++) {
            if (va[t] == vb[t]) continue;
            const Factor &f = fac[(size_t)(nf() - (int)va.size()) + t];
            return f.u ? f.u->letters_less(va[t], vb[t]) : f.l->letters_less(va[t], vb[t]);
        }
        return va.size() < vb.size();
    }

    // -- the recursion ----------------------------------------------------------- //
    int suffix_parity(uint32_t s) {
        int i0 = start(s), p = 0;
        const auto &v = comps(s);
        for (size_t t = 0; t < v.size(); t++) p += fparity(i0 + (int)t, v[t]);
        return p & 1;
    }
    long suffix_nmax(uint32_t a, uint32_t c) {
        int i0 = start(a);
        const auto &va = comps(a), &vc = comps(c);
        long N = 0;
        for (size_t t = 0; t < va.size(); t++) N += fnmax(i0 + (int)t, va[t], vc[t]) + (t ? 1 : 0);
        return N;
    }
    const Elem &tprod(uint32_t a, long n, uint32_t c) {
        uint32_t pr = pair_id(a, c);
        return tp_cache.get(pk(pr, (uint32_t)(n + NOFF)), [&]() -> Elem { return tprod_compute(a, n, c); });
    }
    Elem tprod_compute(uint32_t a, long n, uint32_t c) {
        check();
        int i = start(a);
        const auto &va = comps(a), &vc = comps(c);
        Elem r;
        Elem tmp;
        if (va.size() == 1) {
            long N1 = fnmax(i, va[0], vc[0]);
            if (n > N1) return r;
            const Elem *d = fnprod(i, va[0], n, vc[0], std::min(n, -1L), tmp);
            if (!d) return r;
            for (const auto &kv : *d) {
                C x = kv.second;
                acc(r, intern_codes(std::vector<uint32_t>{kv.first}), x);
            }
            return r;
        }
        uint32_t a1 = va[0], c1 = vc[0], ar = suffix_tail(a), cr = suffix_tail(c);
        bool neg = (suffix_parity(ar) * fparity(i, c1)) & 1;
        long N1 = fnmax(i, a1, c1), N2 = suffix_nmax(ar, cr);
        long klo = n - 1 - N2;
        // the lowest n asked of this factor pair: the same for every outer
        // n >= -1 (one lattice computation serves all of them)
        long nlo = std::min(n, -1L) - 1 - N2;
        C t(R), v1(R);
        for (long k = klo; k <= N1; k++) {
            const Elem *d1 = fnprod(i, a1, k, c1, nlo, tmp);
            if (!d1 || d1->empty()) continue;
            const Elem &d2 = tprod(ar, n - 1 - k, cr);
            if (d2.empty()) continue;
            for (const auto &k1 : *d1) {
                v1 = k1.second;
                if (neg) R.mul_si(v1, k1.second, -1);
                for (const auto &k2 : d2) {
                    R.mul(t, v1, k2.second);
                    acc(r, cons(k1.first, k2.first), t);
                }
            }
        }
        return r;
    }
    // d^n of a suffix (Leibniz over the factors)
    const Elem &tderiv(uint32_t m, int n) {
        return td_cache.get(pk(m, (uint32_t)n), [&]() -> Elem { return tderiv_compute(m, n); });
    }
    Elem tderiv_compute(uint32_t m, int n) {
        check();
        int i = start(m);
        const auto &vm = comps(m);
        Elem r;
        if (vm.size() == 1) {
            for (const auto &kv : fderiv(i, vm[0], n)) {
                C x = kv.second;
                acc(r, intern_codes(std::vector<uint32_t>{kv.first}), x);
            }
            return r;
        }
        uint32_t mr = suffix_tail(m);
        Z b;
        C t(R), u(R);
        for (int j = 0; j <= n; j++) {
            const Elem &d1 = fderiv(i, vm[0], j);
            if (d1.empty()) continue;
            const Elem &d2 = tderiv(mr, n - j);
            if (d2.empty()) continue;
            comb(b.v, (ulong)n, (ulong)j);
            for (const auto &k1 : d1) {
                R.mul_fmpz(u, k1.second, b.v);
                for (const auto &k2 : d2) {
                    R.mul(t, u, k2.second);
                    acc(r, cons(k1.first, k2.first), t);
                }
            }
        }
        return r;
    }

    // -- jobs ---------------------------------------------------------------- //
    void run_job(const Job &j) {
        if (j.kind == J_DERIV) {
            tderiv(j.a, (int)j.b);
            return;
        }
        const auto &ac = ptab[j.a].letters;
        if (j.kind == J_OPE) {
            for (long n = 0; n <= (long)j.b; n++) tprod(ac[0], n, ac[1]);
        } else {
            tprod(ac[0], (long)j.b - NOFF, ac[1]);
        }
    }
    void run_job_if_absent(const Job &j) {
        if (j.kind == J_DERIV) {
            td_cache.get_if_absent(pk(j.a, j.b), [&]() -> Elem { return tderiv_compute(j.a, (int)j.b); });
            return;
        }
        const auto &ac = ptab[j.a].letters;
        if (j.kind == J_OPE) {
            for (long n = 0; n <= (long)j.b; n++)
                tp_cache.get_if_absent(pk(j.a, (uint32_t)(n + NOFF)),
                                       [&]() -> Elem { return tprod_compute(ac[0], n, ac[1]); });
        } else {
            long n = (long)j.b - NOFF;
            tp_cache.get_if_absent(pk(j.a, j.b), [&]() -> Elem { return tprod_compute(ac[0], n, ac[1]); });
        }
    }
    bool job_cached(const Job &j) {
        if (j.kind == J_DERIV) return td_cache.ready(pk(j.a, j.b));
        if (j.kind == J_NPROD) return tp_cache.ready(pk(j.a, j.b));
        return tp_cache.ready(pk(j.a, (uint32_t)((long)j.b + NOFF)));
    }
    size_t job_size(const Job &j) const {
        if (j.kind == J_DERIV) return 1;
        return (size_t)(j.kind == J_OPE ? j.b + 1 : 1);
    }

    // -- calls --------------------------------------------------------------- //
    void eps_in(const std::vector<EpsData> &eps) {
        eps_now.assign(fac.size(), EpsMap());
        for (size_t f = 0; f < fac.size() && f < eps.size(); f++)
            if (fac[f].l) eps_now[f] = fac[f].l->eps_in(eps[f]);
    }
    uint32_t intern(const std::vector<uint32_t> &ids) override {
        if (ids.size() != fac.size()) throw std::invalid_argument("a key needs one id per factor");
        return intern_codes(ids);
    }
    std::vector<uint32_t> components(uint32_t id) override {
        if (id >= monos.n) throw std::out_of_range("unknown key");
        return comps(id);
    }
    PolesData ope_mono(uint32_t a, uint32_t b, const std::vector<EpsData> &eps, PollFn poll) override {
        eps_in(eps);
        Poles P;
        serial([&] {
            long N = suffix_nmax(a, b);
            for (long n = 0; n <= N; n++) P.push_back(tprod(a, n, b));
            while (!P.empty() && P.back().empty()) P.pop_back();
        }, poll);
        return poles_out(P);
    }
    ElemData no_mono(uint32_t a, uint32_t b, const std::vector<EpsData> &eps, PollFn poll) override {
        return nprod_mono(a, -1, b, eps, poll);
    }
    ElemData nprod_mono(uint32_t a, long n, uint32_t b, const std::vector<EpsData> &eps, PollFn poll) override {
        eps_in(eps);
        const Elem *E = nullptr;
        serial([&] { E = &tprod(a, n, b); }, poll);
        return elem_out(*E);
    }
    ElemData deriv_mono(uint32_t m, int n, PollFn poll) override {
        if (n < 0) throw std::invalid_argument("negative derivative order");
        if (n == 0) return elem_out(single(m));
        const Elem *E = nullptr;
        serial([&] { E = &tderiv(m, n); }, poll);
        return elem_out(*E);
    }
    // the OPE jobs of A x B (one per pair: all its poles) and their bounds
    void ope_jobs(const Elem &A, const Elem &B, std::vector<Job> &jobs, std::unordered_set<uint32_t> &seen,
                  PollFn poll) {
        serial([&] {
            for (const auto &a : A)
                for (const auto &b : B) {
                    uint32_t pr = pair_id(a.first, b.first);
                    if (!seen.insert(pr).second) continue;
                    long N = suffix_nmax(a.first, b.first);
                    if (N >= 0) jobs.push_back(Job{J_OPE, pr, (uint32_t)N});
                }
        }, poll);
    }
    Poles ope_sum(const Elem &A, const Elem &B, int nthreads) {
        std::vector<C> scal;
        scal.reserve(A.size() * B.size());
        std::vector<std::vector<Part>> parts;
        for (const auto &a : A)
            for (const auto &b : B) {
                long N = suffix_nmax(a.first, b.first);
                if (N < 0) continue;
                scal.emplace_back(R);
                R.mul(scal.back(), a.second, b.second);
                if ((size_t)N + 1 > parts.size()) parts.resize((size_t)N + 1);
                for (long n = 0; n <= N; n++) {
                    const Elem &e = tprod(a.first, n, b.first);
                    if (!e.empty()) parts[(size_t)n].push_back(Part{&scal.back(), &e});
                }
            }
        return sum_parts(parts, nthreads);
    }
    PolesData ope_el(const ElemData &dA, const ElemData &dB, const std::vector<EpsData> &eps, int nthreads,
                     PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        eps_in(eps);
        Elem A = elem_in(dA), B = elem_in(dB);
        std::vector<Job> jobs;
        std::unordered_set<uint32_t> seen;
        ope_jobs(A, B, jobs, seen, poll);
        warm(std::move(jobs), nthreads, poll);
        Poles out;
        serial([&] { out = ope_sum(A, B, nthreads); }, poll);
        return poles_out(out);
    }
    std::vector<PolesData> ope_el_many(const ElemData &dA, const std::vector<ElemData> &dBs,
                                       const std::vector<EpsData> &eps, int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        eps_in(eps);
        Elem A = elem_in(dA);
        std::vector<Elem> Bs;
        for (const ElemData &d : dBs) Bs.push_back(elem_in(d));
        std::vector<Job> jobs;
        std::unordered_set<uint32_t> seen;
        for (const Elem &B : Bs) ope_jobs(A, B, jobs, seen, poll);
        warm(std::move(jobs), nthreads, poll);
        std::vector<PolesData> out;
        for (const Elem &B : Bs) {
            Poles P;
            serial([&] { P = ope_sum(A, B, nthreads); }, poll);
            out.push_back(poles_out(P));
        }
        return out;
    }
    ElemData nprod_el(const ElemData &dA, long n, const ElemData &dB, const std::vector<EpsData> &eps,
                      int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        eps_in(eps);
        Elem A = elem_in(dA), B = elem_in(dB);
        std::vector<Job> jobs;
        std::unordered_set<uint32_t> seen;
        for (const auto &a : A)
            for (const auto &b : B) {
                uint32_t pr = pair_id(a.first, b.first);
                if (seen.insert(pr).second) jobs.push_back(Job{J_NPROD, pr, (uint32_t)(n + NOFF)});
            }
        warm(std::move(jobs), nthreads, poll);
        Elem res;
        serial([&] {
            res = pair_sum(A, B, nthreads, [&](uint32_t a, uint32_t b) { return &tprod(a, n, b); });
        }, poll);
        return elem_out(res);
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
            for (const auto &a : A) add_scaled(res, tderiv(a.first, n), a.second);
        }, poll);
        return elem_out(res);
    }
    // [els[ia[k]]_(ns[k]) els[ib[k]] for every k] in one call: the tensor
    // products of all key pairs first (one warm), then the sums, each in one
    // of nthreads threads (larger |A| |B| first)
    std::vector<ElemData> nprod_el_many(const std::vector<ElemData> &dels, const std::vector<int> &ia,
                                        const std::vector<long> &ns, const std::vector<int> &ib,
                                        const std::vector<EpsData> &eps, int nthreads, PollFn poll) override {
        MaybeGmpMalloc gmp(nthreads);
        check_tasks(dels.size(), ia, ns, ib);
        eps_in(eps);
        std::vector<Elem> E;
        E.reserve(dels.size());
        for (const ElemData &d : dels) E.push_back(elem_in(d));
        std::vector<Job> jobs;
        std::unordered_set<uint64_t> seen;
        for (size_t k = 0; k < ns.size(); k++) {
            if (ns[k] + NOFF < 0 || ns[k] + NOFF > (long)UINT32_MAX) throw std::invalid_argument("n out of range");
            for (const auto &a : E[(size_t)ia[k]])
                for (const auto &b : E[(size_t)ib[k]]) {
                    uint32_t pr = pair_id(a.first, b.first);
                    if (seen.insert(pk(pr, (uint32_t)(ns[k] + NOFF))).second)
                        jobs.push_back(Job{J_NPROD, pr, (uint32_t)(ns[k] + NOFF)});
                }
        }
        warm(std::move(jobs), nthreads, poll);
        std::vector<size_t> order = Core::task_order(ns.size(), [&](size_t k) {
            return E[(size_t)ia[k]].size() * E[(size_t)ib[k]].size();
        });
        std::vector<Elem> res(ns.size());
        run_tasks(order, nthreads, poll, [&](size_t k) {
            long n = ns[k];
            res[k] = pair_sum(E[(size_t)ia[k]], E[(size_t)ib[k]], 1,
                              [&](uint32_t a, uint32_t b) { return &tprod(a, n, b); });
        });
        std::vector<ElemData> out;
        out.reserve(res.size());
        for (const Elem &e : res) out.push_back(elem_out(e));
        return out;
    }
    std::vector<size_t> cache_sizes() const override {
        std::vector<size_t> s = this->sched_stats();
        s.insert(s.begin(), {tp_cache.size(), td_cache.size(), lat_cache.size(), fnp_cache.size(),
                             (size_t)monos.n});
        return s;
    }
};

// kind: "q", "qpoly", "qmpoly" (nvars >= 1), "nmod" (modulus a prime < 2^63),
// "nf", "nfmpoly" (nvars >= 1; minpoly: the defining polynomial of the field)
inline TensorEngineBase *make_tensor_engine(const std::string &kind, long nvars, unsigned long modulus,
                                            const CoefData &minpoly) {
    if (kind == "nf") return new TensorEngine<RingNF>(minpoly);
    if (kind == "nfmpoly") return new TensorEngine<RingNFMPoly>((slong)nvars, minpoly);
    if (kind == "q") return new TensorEngine<RingQ>();
    if (kind == "qpoly") return new TensorEngine<RingQPoly>();
    if (kind == "qmpoly") {
        if (nvars < 1) throw std::invalid_argument("qmpoly needs at least one variable");
        return new TensorEngine<RingQMPoly>((slong)nvars);
    }
    if (kind == "nmod") {
        if (modulus < 2 || modulus > (unsigned long)WORD_MAX || !n_is_prime(modulus))
            throw std::invalid_argument("nmod needs a prime modulus below 2^63");
        return new TensorEngine<RingNmod>((ulong)modulus);
    }
    throw std::invalid_argument("unknown coefficient kind " + kind);
}

}  // namespace vae
