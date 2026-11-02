//! Matrix-free GIJS distinguisher: the Hasse jet matrix M is NEVER formed.
//!
//! The dense `distinguish` in gijs.rs materializes M_tau, an E x N_d matrix
//! over GF(2^m) (E = (n-1)*J rows, J = C(k+s-1,s-1) jets, N_d = C(k+d-1,d)
//! monomials). At paper-like parameters (n'=348, k'=28, s=3, d=5) that is
//! 150,945 x 201,376 entries = 61 GB as u16: unstorable. This module provides
//! the same decision with O(E + N_d + per-point power tables) working memory:
//!
//! 1. An incidence-based matvec: M[(j,a), m] = C(m,a) * y_j^(m-a) is nonzero
//!    only when `a` is a bitwise submask of `m` (char 2, Lucas -- exactly what
//!    gijs::hasse computes). Summing only over incident (a,m) pairs gives a
//!    matvec costing the submask-incidence count per point, which is bounded
//!    by (and typically a constant factor below) the componentwise count
//!    sum_{t<s} C(k+t-1,t)*C(k+d-t-1,d-t) -- versus J*N_d for a dense matvec.
//!
//! 2. A scalar Wiedemann black-box kernel sampler on A = M_tau^T M_tau
//!    (two-pass: Berlekamp-Massey on a projected Krylov sequence, then a
//!    second pass to materialize the kernel vector; O(N_d) memory).
//!
//! Decision procedure (same contract as gijs::distinguish):
//!   v_tau in rowspan(M_tau)  <=>  z . v_tau = 0 for all z in ker(M_tau).
//! We sample vectors z from ker(A), A = M_tau^T M_tau. NOTE ker(A) >= ker(M):
//! the difference is the isotropic radical R = M(ker A) = col(M) ∩ ker(M^T),
//! which is usually {0} but NOT always -- at (n=41,k=19,s=2,d=3) it is
//! 1-dimensional, and a random z in ker(A) then has M z != 0 with
//! probability 1 - 1/q, so naive per-sample verification almost always
//! rejects good samples. Instead we CANCEL the radical: collect samples
//! z_1..z_m, compute w_i = M z_i, and take coefficient vectors c in
//! ker[w_1 ... w_m] (a small E x m dense kernel); each z(c) = sum_i c_i z_i
//! lies in ker(M) by construction and is still verified exactly (M z(c) == 0)
//! before use. If a verified z has z . v_tau != 0, v_tau is provably outside
//! the rowspace and we return false ("random") -- that answer is EXACT, it
//! carries a checkable certificate. If `samples` linearly independent
//! verified kernel vectors all satisfy z . v_tau = 0 we return true ("GRS").
//! The true-answer is the only heuristic side: conditioned on the collected
//! samples spanning ker(A) (resp. its relevant quotient), each independent
//! certified vector misses an existing separator with probability ~1/q, so
//! the false-accept probability is ~q^-samples. (When ker(M_tau) = {0} --
//! e.g. E >= N_d with generic points -- the rowspace is everything, `true`
//! is the exact answer, and the sampler detects this via a full-degree
//! minimal polynomial with no x-factor.)

use crate::gf::Gf;
use crate::gijs::{enum_exp, power_table};
use crate::rng::Rng;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Jet operator: implicit M_full (all points), with a hold-out exclusion.
// ---------------------------------------------------------------------------

/// Precomputed combinatorial structure of the Hasse jet matrix for a fixed
/// (k, s, d). The point-dependent values are monomial evaluations y^b, which
/// are recomputed per point on the fly (streaming), so the only large stored
/// object is the incidence pair list: (jet_idx, mono_idx, eval_idx) triples,
/// 12 bytes each, nnz-per-point many.
pub struct JetOp {
    pub k: usize,
    pub s: usize,
    pub d: usize,
    /// Monomials of degree exactly d (columns of M), N_d of them.
    pub monos: Vec<Vec<u8>>,
    /// Jets: all exponent vectors a with |a| < s (row blocks), J of them.
    pub jets: Vec<Vec<u8>>,
    /// All exponent vectors b with |b| <= d; eval index space for y^b.
    evals: Vec<Vec<u8>>,
    /// eval_parent[i], eval_coord[i]: evals[i] = evals[parent] + e_coord
    /// (for i = 0, the zero exponent, parent = 0 and coord is unused).
    eval_parent: Vec<u32>,
    eval_coord: Vec<u8>,
    /// Incidence pairs: (jet_idx, mono_idx, eval_idx of m - a), one per
    /// (a, m) with a a bitwise submask of m. Shared by all points.
    pub pairs: Vec<(u32, u32, u32)>,
}

impl JetOp {
    pub fn new(k: usize, s: usize, d: usize) -> Self {
        let monos = enum_exp(k, d);
        let mut jets: Vec<Vec<u8>> = Vec::new();
        for sum in 0..s {
            jets.extend(enum_exp(k, sum));
        }
        // Eval exponents: all b with |b| <= d, ordered by degree so that a
        // parent (degree one less) always precedes its children.
        let mut evals: Vec<Vec<u8>> = Vec::new();
        for e in 0..=d {
            evals.extend(enum_exp(k, e));
        }
        let mut index: HashMap<Vec<u8>, u32> = HashMap::with_capacity(evals.len());
        for (i, b) in evals.iter().enumerate() {
            index.insert(b.clone(), i as u32);
        }
        let mut eval_parent = vec![0u32; evals.len()];
        let mut eval_coord = vec![0u8; evals.len()];
        for (i, b) in evals.iter().enumerate() {
            if i == 0 {
                continue; // zero exponent
            }
            // first coordinate with b_i > 0 gives the parent b - e_i
            let mut par = b.clone();
            for c in 0..k {
                if par[c] > 0 {
                    par[c] -= 1;
                    eval_parent[i] = index[&par];
                    eval_coord[i] = c as u8;
                    break;
                }
            }
        }
        // Incidence pairs (jet a, monomial m): nonzero Hasse entry iff a is a
        // bitwise submask of m; then the entry is y^(m-a) (coefficient 1 in
        // char 2). This mirrors gijs::hasse exactly.
        let mut pairs = Vec::new();
        for (ai, a) in jets.iter().enumerate() {
            for (mi, m) in monos.iter().enumerate() {
                let mut ok = true;
                let mut b = vec![0u8; k];
                for i in 0..k {
                    if (a[i] & m[i]) != a[i] {
                        ok = false;
                        break;
                    }
                    b[i] = m[i] - a[i];
                }
                if ok {
                    let bi = index[&b];
                    pairs.push((ai as u32, mi as u32, bi));
                }
            }
        }
        JetOp {
            k,
            s,
            d,
            monos,
            jets,
            evals,
            eval_parent,
            eval_coord,
            pairs,
        }
    }

    pub fn nd(&self) -> usize {
        self.monos.len()
    }
    pub fn njets(&self) -> usize {
        self.jets.len()
    }
    /// Number of eval exponents (monomials of degree <= d).
    pub fn n_evals(&self) -> usize {
        self.evals.len()
    }

    /// Componentwise incidence count per point:
    /// sum_{t=0}^{s-1} C(k+t-1,t) * C(k+d-t-1,d-t). This counts pairs
    /// (a, m) with a <= m componentwise and is an UPPER bound on the true
    /// char-2 submask incidence `pairs.len()` (Lucas kills e.g. a_i=1,
    /// m_i=2). Kept as the standard cost-model estimate.
    pub fn nnz_formula(k: usize, s: usize, d: usize) -> usize {
        fn binom(n: usize, k: usize) -> usize {
            if k > n {
                return 0;
            }
            let k = k.min(n - k);
            let mut r = 1usize;
            for i in 0..k {
                r = r * (n - i) / (i + 1);
            }
            r
        }
        (0..s)
            .map(|t| binom(k + t - 1, t) * binom(k + d - t - 1, d - t))
            .sum()
    }

    /// Monomial evaluations y^b for all |b| <= d at one point, computed with
    /// one field multiplication per entry via the parent plan.
    fn eval_table(&self, gf: &Gf, y: &[u16]) -> Vec<u16> {
        let mut ev = vec![0u16; self.evals.len()];
        ev[0] = 1;
        for i in 1..self.evals.len() {
            let p = self.eval_parent[i] as usize;
            let c = self.eval_coord[i] as usize;
            ev[i] = gf.mul(ev[p], y[c]);
        }
        ev
    }

    /// out = M x, where M is the full jet matrix with the `excl` point's row
    /// block forced to zero (this is exactly M_tau for tau = excl, embedded).
    /// out has length n * J and is overwritten.
    pub fn matvec_m(&self, gf: &Gf, cols: &[Vec<u16>], excl: usize, x: &[u16], out: &mut [u16]) {
        let nj = self.jets.len();
        out.fill(0);
        for (j, y) in cols.iter().enumerate() {
            if j == excl {
                continue;
            }
            let ev = self.eval_table(gf, y);
            let base = j * nj;
            for &(ai, mi, bi) in self.pairs.iter() {
                let xv = x[mi as usize];
                if xv != 0 {
                    let c = gf.mul(xv, ev[bi as usize]);
                    if c != 0 {
                        out[base + ai as usize] ^= c;
                    }
                }
            }
        }
    }

    /// out = M^T u (full M; the caller zeroes the excluded block of u, which
    /// matvec_m already guarantees). out has length N_d and is overwritten.
    pub fn matvec_mt(&self, gf: &Gf, cols: &[Vec<u16>], u: &[u16], out: &mut [u16]) {
        let nj = self.jets.len();
        out.fill(0);
        for (j, y) in cols.iter().enumerate() {
            let ev = self.eval_table(gf, y);
            let base = j * nj;
            for &(ai, mi, bi) in self.pairs.iter() {
                let uv = u[base + ai as usize];
                if uv != 0 {
                    let c = gf.mul(uv, ev[bi as usize]);
                    if c != 0 {
                        out[mi as usize] ^= c;
                    }
                }
            }
        }
    }

    /// out = A x with A = M_tau^T M_tau (tau = excl). `mid` is scratch of
    /// length n * J.
    pub fn apply_a(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        x: &[u16],
        out: &mut [u16],
        mid: &mut Vec<u16>,
    ) {
        let need = cols.len() * self.jets.len();
        if mid.len() != need {
            mid.resize(need, 0);
        }
        self.matvec_m(gf, cols, excl, x, mid);
        self.matvec_mt(gf, cols, mid, out);
    }

    /// Convenience allocating variant of matvec_m (tests/benchmarks).
    pub fn matvec_m_vec(&self, gf: &Gf, cols: &[Vec<u16>], excl: usize, x: &[u16]) -> Vec<u16> {
        let mut out = vec![0u16; cols.len() * self.jets.len()];
        self.matvec_m(gf, cols, excl, x, &mut out);
        out
    }

    /// The evaluation vector v_tau at point y: v[mi] = prod_i y_i^{m_i},
    /// computed exactly as gijs::distinguish does.
    pub fn eval_vec(&self, gf: &Gf, y: &[u16]) -> Vec<u16> {
        let yp = power_table(gf, y, self.d);
        let mut v = vec![0u16; self.monos.len()];
        for (mi, m) in self.monos.iter().enumerate() {
            let mut val: u16 = 1;
            for i in 0..self.k {
                val = gf.mul(val, yp[i][m[i] as usize]);
                if val == 0 {
                    break;
                }
            }
            v[mi] = val;
        }
        v
    }

    /// One Wiedemann kernel sample of A = M_tau^T M_tau (tau = excl).
    /// KernelOutcome::Vector gives a nonzero z with A z = 0 (verified);
    /// NOTE z is in ker(A), which may be larger than ker(M_tau) by the
    /// isotropic radical -- callers should cancel the radical across
    /// several samples (as distinguish_matfree_with does) rather than
    /// expecting M_tau z = 0 for an individual sample.
    /// KernelOutcome::Invertible means ker(A) = {0}, hence ker(M_tau) = {0}.
    pub fn kernel_sample(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        rng: &mut Rng,
    ) -> KernelOutcome {
        let nd = self.nd();
        let e_tau = (cols.len() - 1) * self.jets.len();
        let deg_bound = nd.min(e_tau);
        let mut mid: Vec<u16> = Vec::new();
        let mut apply = |x: &[u16], out: &mut [u16]| {
            self.apply_a(gf, cols, excl, x, out, &mut mid);
        };
        wiedemann_kernel_ex(gf, nd, deg_bound, &mut apply, rng)
    }
}

// ---------------------------------------------------------------------------
// Small GF helpers
// ---------------------------------------------------------------------------

fn dot(gf: &Gf, a: &[u16], b: &[u16]) -> u16 {
    let mut acc: u16 = 0;
    for i in 0..a.len() {
        if a[i] != 0 && b[i] != 0 {
            acc ^= gf.mul(a[i], b[i]);
        }
    }
    acc
}

fn rand_vec(gf: &Gf, rng: &mut Rng, n: usize) -> Vec<u16> {
    let q = gf.order as u64;
    let mut v: Vec<u16> = (0..n).map(|_| rng.below(q) as u16).collect();
    if v.iter().all(|&x| x == 0) {
        v[0] = 1;
    }
    v
}

// ---------------------------------------------------------------------------
// Berlekamp-Massey over GF(2^m) and the Wiedemann kernel sampler
// ---------------------------------------------------------------------------

/// Berlekamp-Massey: given a scalar sequence s[0..len], returns the
/// connection polynomial c (c[0] = 1, length L+1) of the shortest LFSR
/// generating it: s[n] + sum_{i=1}^{L} c[i] * s[n-i] = 0 for L <= n < len.
fn berlekamp_massey(gf: &Gf, s: &[u16]) -> Vec<u16> {
    let mut c: Vec<u16> = vec![1];
    let mut b: Vec<u16> = vec![1];
    let mut l: usize = 0;
    let mut m: usize = 1;
    let mut bb: u16 = 1;
    for n in 0..s.len() {
        if c.len() < l + 1 {
            c.resize(l + 1, 0);
        }
        // discrepancy
        let mut d = s[n];
        for i in 1..=l {
            d ^= gf.mul(c[i], s[n - i]);
        }
        if d == 0 {
            m += 1;
        } else if 2 * l <= n {
            let t = c.clone();
            let coef = gf.mul(d, gf.inv(bb));
            if c.len() < b.len() + m {
                c.resize(b.len() + m, 0);
            }
            for i in 0..b.len() {
                let v = gf.mul(coef, b[i]);
                c[i + m] ^= v;
            }
            l = n + 1 - l;
            b = t;
            bb = d;
            m = 1;
        } else {
            let coef = gf.mul(d, gf.inv(bb));
            if c.len() < b.len() + m {
                c.resize(b.len() + m, 0);
            }
            for i in 0..b.len() {
                let v = gf.mul(coef, b[i]);
                c[i + m] ^= v;
            }
            m += 1;
        }
    }
    c.truncate(l + 1);
    c.resize(l + 1, 0);
    c
}

/// Convert a BM connection polynomial c (degree L, c[0]=1) into the
/// annihilating polynomial mu of the underlying vector sequence:
/// mu(x) = sum_{i=0}^{L} c[L-i] x^i, so mu(A) b = 0 whenever the projected
/// scalar sequence satisfies the recurrence throughout.
fn conn_to_mu(c: &[u16]) -> Vec<u16> {
    let l = c.len() - 1;
    (0..=l).map(|p| c[l - p]).collect()
}

/// Outcome of one Wiedemann kernel-sampling attempt on an operator A.
pub enum KernelOutcome {
    /// A verified nonzero kernel vector (A z = 0 checked exactly).
    Vector(Vec<u16>),
    /// BM converged to a minimal polynomial of full degree n with no
    /// x-factor: then the minimal polynomial of A has no root 0, so A is
    /// invertible and ker(A) = {0}. (Sound given the recovered polynomial
    /// is the true minimal polynomial of b, the standard Wiedemann
    /// projection heuristic; degree n forces it to equal the characteristic
    /// polynomial.)
    Invertible,
    /// Extraction failed (bad projection, premature stabilization, or the
    /// sampled b had no kernel component).
    Failed,
}

/// Scalar Wiedemann kernel sampler for a black-box operator A on F_q^n,
/// given as a closure apply(x, out) computing A x.
///
/// Method: pick random b, u. Generate the projected Krylov sequence
/// s_i = u . A^i b (in chunks, re-running BM until the connection polynomial
/// is stable, which is the standard early-termination heuristic). From
/// mu = annihilating polynomial of b w.r.t. A, write mu(x) = x^t g(x) with
/// g(0) != 0; if t >= 1 then z = g(A) A^{t-1} b satisfies A z = mu(A) b = 0.
/// A second pass over the Krylov sequence materializes z with O(n) memory.
/// The result is verified (A z == 0, z != 0) before being returned; any
/// heuristic failure (bad projection, premature stabilization) yields
/// KernelOutcome::Failed.
///
/// `deg_bound` is an upper bound on the expected degree of the minimal
/// polynomial (used only to cap the sequence length).
pub fn wiedemann_kernel_ex<F: FnMut(&[u16], &mut [u16])>(
    gf: &Gf,
    n: usize,
    deg_bound: usize,
    apply: &mut F,
    rng: &mut Rng,
) -> KernelOutcome {
    if n == 0 {
        return KernelOutcome::Failed;
    }
    let b = rand_vec(gf, rng, n);
    let u = rand_vec(gf, rng, n);
    let cap_terms = 2 * (deg_bound + 2) + 32;

    // Pass 1: projected scalar sequence with chunked BM stabilization.
    let mut w = b.clone();
    let mut w2 = vec![0u16; n];
    let mut seq: Vec<u16> = Vec::new();
    let mut mu: Option<Vec<u16>> = None;
    'gen: loop {
        for _ in 0..32 {
            if seq.len() >= cap_terms {
                break;
            }
            seq.push(dot(gf, &u, &w));
            apply(&w, &mut w2);
            std::mem::swap(&mut w, &mut w2);
        }
        let c = berlekamp_massey(gf, &seq);
        let new_mu = conn_to_mu(&c);
        if let Some(prev) = &mu {
            if *prev == new_mu && seq.len() >= 2 * (new_mu.len() - 1) + 8 {
                mu = Some(new_mu);
                break 'gen;
            }
        }
        mu = Some(new_mu);
        if seq.len() >= cap_terms {
            break 'gen;
        }
    }
    let mu = match mu {
        Some(m) => m,
        None => return KernelOutcome::Failed,
    };
    let l = mu.len() - 1; // degree of mu
    if l == 0 {
        return KernelOutcome::Failed; // sequence was identically zero: bad projection
    }
    // t = multiplicity of the root 0: mu[0..t] all zero, mu[t] != 0.
    let mut t = 0usize;
    while t <= l && mu[t] == 0 {
        t += 1;
    }
    if t == 0 {
        // No kernel component along b. If moreover mu has full degree n,
        // the minimal polynomial of A is the characteristic polynomial and
        // has no root 0: A is invertible.
        if l == n {
            return KernelOutcome::Invertible;
        }
        return KernelOutcome::Failed;
    }
    if t > l {
        return KernelOutcome::Failed;
    }
    // Pass 2: z = sum_{i=t-1}^{l-1} mu[i+1] * A^i b  (= g(A) A^{t-1} b).
    let mut z = vec![0u16; n];
    let mut w = b.clone();
    let mut w2 = vec![0u16; n];
    for i in 0..l {
        if i >= t - 1 {
            let coef = mu[i + 1];
            if coef != 0 {
                for c in 0..n {
                    if w[c] != 0 {
                        z[c] ^= gf.mul(coef, w[c]);
                    }
                }
            }
        }
        if i + 1 < l {
            apply(&w, &mut w2);
            std::mem::swap(&mut w, &mut w2);
        }
    }
    if z.iter().all(|&x| x == 0) {
        return KernelOutcome::Failed;
    }
    // Exact verification: A z == 0.
    let mut az = vec![0u16; n];
    apply(&z, &mut az);
    if az.iter().all(|&x| x == 0) {
        KernelOutcome::Vector(z)
    } else {
        KernelOutcome::Failed
    }
}

/// Option-returning wrapper around wiedemann_kernel_ex (kernel vector or
/// nothing), used where the invertible/failed distinction doesn't matter.
pub fn wiedemann_kernel<F: FnMut(&[u16], &mut [u16])>(
    gf: &Gf,
    n: usize,
    deg_bound: usize,
    apply: &mut F,
    rng: &mut Rng,
) -> Option<Vec<u16>> {
    match wiedemann_kernel_ex(gf, n, deg_bound, apply, rng) {
        KernelOutcome::Vector(z) => Some(z),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The distinguisher
// ---------------------------------------------------------------------------

/// Tunables for distinguish_matfree.
pub struct MatfreeConfig {
    /// Independent certified ker(M_tau) vectors (after radical
    /// cancellation) that must all satisfy z . v = 0 before we accept
    /// ("GRS"). False-accept probability ~ q^-samples.
    pub samples: usize,
    /// Hard cap on Wiedemann samples collected per decision. Must exceed
    /// samples + (dimension of the isotropic radical, usually 0 or 1).
    pub max_attempts: usize,
    /// Give up sampling after this many consecutive attempts whose minimal
    /// polynomial had no x-factor (evidence ker(A) = {0}: answer is `true`
    /// exactly in that case, since the rowspace is then everything... more
    /// precisely ker(M) = {0} iff the rowspace is all of F^{N_d}).
    pub max_no_kernel_streak: usize,
    pub seed: u64,
}

impl Default for MatfreeConfig {
    fn default() -> Self {
        MatfreeConfig {
            samples: 4,
            max_attempts: 32,
            max_no_kernel_streak: 4,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Counters for diagnostics/tests.
#[derive(Default, Debug, Clone)]
pub struct MatfreeStats {
    pub attempts: usize,
    pub extract_fail: usize,
    /// Certified combinations that failed the exact M z = 0 check
    /// (expected: 0; the cancellation is exact by construction).
    pub verify_fail: usize,
    /// Independent certified ker(M_tau) vectors tested (final count).
    pub verified: usize,
    pub no_kernel: usize,
    /// Dimension of the spanned part of the isotropic radical
    /// (rank of [M z_1 ... M z_m] at loop exit).
    pub radical_rank: usize,
}

/// Matrix-free GIJS distinguisher with explicit config and stats.
/// Same contract as gijs::distinguish: true = "GRS" (v_tau in rowspan),
/// false = "random". See the module docs for the exact error model:
/// `false` is always certified exactly; `true` is heuristic with
/// false-accept probability ~ q^-cfg.samples.
pub fn distinguish_matfree_with(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    tau: usize,
    cfg: &MatfreeConfig,
    stats: &mut MatfreeStats,
) -> bool {
    let k = cols[0].len();
    let op = JetOp::new(k, s, d);
    let v = op.eval_vec(gf, &cols[tau]);
    let mut rng = Rng::new(
        cfg.seed
            ^ (tau as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ ((cols.len() as u64) << 32)
            ^ 0xA076_1D64_78BD_642F,
    );
    let e_len = cols.len() * op.njets();
    let mut zs: Vec<Vec<u16>> = Vec::new(); // samples from ker(A)
    let mut ws: Vec<Vec<u16>> = Vec::new(); // w_i = M_tau z_i (length e_len)
    let mut attempts = 0usize;
    let mut fail_streak = 0usize;
    while attempts < cfg.max_attempts {
        attempts += 1;
        stats.attempts += 1;
        match op.kernel_sample(gf, cols, tau, &mut rng) {
            KernelOutcome::Invertible => {
                // ker(A) = {0}, hence ker(M_tau) = {0}: the rowspace is all
                // of F^{N_d} and v_tau is certainly in it. Exact answer.
                stats.no_kernel += 1;
                return true;
            }
            KernelOutcome::Failed => {
                stats.extract_fail += 1;
                fail_streak += 1;
                if zs.is_empty() && fail_streak >= cfg.max_no_kernel_streak {
                    break;
                }
            }
            KernelOutcome::Vector(z) => {
                fail_streak = 0;
                let w = op.matvec_m_vec(gf, cols, tau, &z);
                zs.push(z);
                ws.push(w);
                // Radical cancellation: coefficient vectors c in the kernel
                // of W = [w_1 ... w_m] (E x m, small m) give combinations
                // z(c) = sum_i c_i z_i lying in ker(M_tau). A basis of ker(W)
                // maps to a basis of span(zs) ∩ ker(M_tau); testing the whole
                // basis is complete for that subspace because z -> z . v is
                // linear.
                let m = zs.len();
                let wrows: Vec<Vec<u16>> = (0..e_len)
                    .map(|r| (0..m).map(|i| ws[i][r]).collect())
                    .collect();
                let (_wrank, ker) = crate::linalg::kernel_basis(gf, &wrows);
                stats.radical_rank = m - ker.len();
                let mut certified = 0usize;
                for c in ker.iter() {
                    let mut zc = vec![0u16; op.nd()];
                    for (i, &ci) in c.iter().enumerate() {
                        if ci != 0 {
                            for t in 0..zc.len() {
                                if zs[i][t] != 0 {
                                    zc[t] ^= gf.mul(ci, zs[i][t]);
                                }
                            }
                        }
                    }
                    if zc.iter().all(|&x| x == 0) {
                        continue;
                    }
                    // Exact verification (by construction M z(c) = 0; the
                    // check guards the whole pipeline).
                    let mzc = op.matvec_m_vec(gf, cols, tau, &zc);
                    if mzc.iter().any(|&x| x != 0) {
                        stats.verify_fail += 1;
                        continue;
                    }
                    certified += 1;
                    if dot(gf, &zc, &v) != 0 {
                        stats.verified = certified;
                        return false; // certified: v_tau not in rowspan
                    }
                }
                stats.verified = certified;
                if certified >= cfg.samples {
                    return true;
                }
            }
        }
    }
    true
}

/// Matrix-free GIJS distinguisher; same signature/contract as
/// gijs::distinguish, default config.
pub fn distinguish_matfree(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize, tau: usize) -> bool {
    let cfg = MatfreeConfig::default();
    let mut stats = MatfreeStats::default();
    distinguish_matfree_with(gf, cols, s, d, tau, &cfg, &mut stats)
}

// ---------------------------------------------------------------------------
// Tests and benchmarks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gijs::{distinguish, hasse, power_table};
    use crate::linalg::shorten;
    use crate::rlce::gen_rlce;

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order as u64;
        (0..n)
            .map(|_| (0..k).map(|_| rng.below(q) as u16).collect())
            .collect()
    }

    /// Pure GRS generator columns: y_j * (1, x_j, ..., x_j^{k-1}), distinct
    /// nonzero x_j, random nonzero y_j (mirrors rlce::gen_grs_cols).
    fn grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.below(q as u64) as usize;
            if !used[x] {
                used[x] = true;
                xs.push(x as u16);
            }
        }
        xs.into_iter()
            .map(|x| {
                let y = (rng.below((q - 1) as u64) + 1) as u16;
                let mut col = vec![0u16; k];
                let mut xp = 1u16;
                for i in 0..k {
                    col[i] = gf.mul(y, xp);
                    xp = gf.mul(xp, x);
                }
                col
            })
            .collect()
    }

    fn peak_rss_mb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
        -1.0
    }

    #[test]
    fn pairs_count_matches_formula() {
        // pairs.len() is the exact char-2 submask incidence; the closed form
        // is the componentwise count and must upper-bound it. Exactness of
        // the pair list itself is verified against gijs::hasse in
        // matvec_matches_dense_hasse (a generic point makes hasse != 0
        // exactly on submask pairs).
        let gf = Gf::new(6);
        let mut rng = Rng::new(2024);
        for &(k, s, d) in &[(5, 2, 3), (6, 3, 4), (8, 2, 4), (12, 3, 4), (19, 2, 3), (14, 2, 4)] {
            let op = JetOp::new(k, s, d);
            assert!(
                op.pairs.len() <= JetOp::nnz_formula(k, s, d),
                "submask nnz exceeds componentwise bound at (k,s,d)=({},{},{})",
                k,
                s,
                d
            );
            // exact count via hasse on an all-nonzero point
            let y: Vec<u16> = (0..k).map(|_| (rng.below(63) + 1) as u16).collect();
            let yp = power_table(&gf, &y, d);
            let mut cnt = 0usize;
            for a in op.jets.iter() {
                for m in op.monos.iter() {
                    if hasse(&gf, m, a, &yp) != 0 {
                        cnt += 1;
                    }
                }
            }
            assert_eq!(op.pairs.len(), cnt, "pair list != hasse support at ({},{},{})", k, s, d);
        }
    }

    #[test]
    fn matvec_matches_dense_hasse() {
        let gf = Gf::new(6); // q=64
        let mut rng = Rng::new(12345);
        for &(k, s, d) in &[(5, 2, 3), (6, 3, 4), (7, 2, 4)] {
            let n = 11;
            let cols = random_cols(&gf, &mut rng, n, k);
            let op = JetOp::new(k, s, d);
            let nd = op.nd();
            let nj = op.njets();
            let x = rand_vec(&gf, &mut rng, nd);
            let excl = 3;
            // --- M x, direct via hasse ---
            let got = op.matvec_m_vec(&gf, &cols, excl, &x);
            let mut want = vec![0u16; n * nj];
            for j in 0..n {
                if j == excl {
                    continue;
                }
                let yp = power_table(&gf, &cols[j], d);
                for (ai, a) in op.jets.iter().enumerate() {
                    let mut acc = 0u16;
                    for (mi, m) in op.monos.iter().enumerate() {
                        let h = hasse(&gf, m, a, &yp);
                        if h != 0 && x[mi] != 0 {
                            acc ^= gf.mul(h, x[mi]);
                        }
                    }
                    want[j * nj + ai] = acc;
                }
            }
            assert_eq!(got, want, "M x mismatch at (k,s,d)=({},{},{})", k, s, d);
            // --- M^T u, direct + dot identity ---
            let u = rand_vec(&gf, &mut rng, n * nj);
            let mut u0 = u.clone();
            u0[excl * nj..(excl + 1) * nj].fill(0);
            let mut mt_u = vec![0u16; nd];
            op.matvec_mt(&gf, &cols, &u0, &mut mt_u);
            let mut want_t = vec![0u16; nd];
            for j in 0..n {
                if j == excl {
                    continue;
                }
                let yp = power_table(&gf, &cols[j], d);
                for (ai, a) in op.jets.iter().enumerate() {
                    let uv = u0[j * nj + ai];
                    if uv == 0 {
                        continue;
                    }
                    for (mi, m) in op.monos.iter().enumerate() {
                        let h = hasse(&gf, m, a, &yp);
                        if h != 0 {
                            want_t[mi] ^= gf.mul(uv, h);
                        }
                    }
                }
            }
            assert_eq!(mt_u, want_t, "M^T u mismatch at (k,s,d)=({},{},{})", k, s, d);
            // dot identity: u . (M x) == (M^T u) . x
            let lhs = dot(&gf, &u0, &got);
            let rhs = dot(&gf, &mt_u, &x);
            assert_eq!(lhs, rhs, "dot identity failed");
        }
    }

    #[test]
    fn bm_and_wiedemann_on_dense_operator() {
        let gf = Gf::new(6);
        // BM sanity: s_i = alpha^i + beta^i -> c = [1, alpha+beta, alpha*beta]
        let (alpha, beta) = (3u16, 5u16);
        let mut seq = Vec::new();
        let (mut pa, mut pb) = (1u16, 1u16);
        for _ in 0..16 {
            seq.push(pa ^ pb);
            pa = gf.mul(pa, alpha);
            pb = gf.mul(pb, beta);
        }
        let c = berlekamp_massey(&gf, &seq);
        assert_eq!(c.len(), 3);
        assert_eq!(c[0], 1);
        assert_eq!(c[1], alpha ^ beta);
        assert_eq!(c[2], gf.mul(alpha, beta));

        // Wiedemann on a planted-kernel dense operator A = X*Y (40x30 * 30x40)
        let mut rng = Rng::new(999);
        let (n, r) = (40usize, 30usize);
        let xm: Vec<Vec<u16>> = (0..n)
            .map(|_| (0..r).map(|_| rng.below(64) as u16).collect())
            .collect();
        let ym: Vec<Vec<u16>> = (0..r)
            .map(|_| (0..n).map(|_| rng.below(64) as u16).collect())
            .collect();
        let mut amat = vec![vec![0u16; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0u16;
                for t in 0..r {
                    acc ^= gf.mul(xm[i][t], ym[t][j]);
                }
                amat[i][j] = acc;
            }
        }
        let mut apply = |x: &[u16], out: &mut [u16]| {
            for i in 0..n {
                let mut acc = 0u16;
                for j in 0..n {
                    if amat[i][j] != 0 && x[j] != 0 {
                        acc ^= gf.mul(amat[i][j], x[j]);
                    }
                }
                out[i] = acc;
            }
        };
        let mut found = 0;
        for _ in 0..20 {
            if let Some(z) = wiedemann_kernel(&gf, n, n, &mut apply, &mut rng) {
                assert!(z.iter().any(|&v| v != 0), "zero kernel vector");
                let mut az = vec![0u16; n];
                apply(&z, &mut az);
                assert!(az.iter().all(|&v| v == 0), "A z != 0");
                found += 1;
            }
        }
        assert!(found >= 15, "only {}/20 kernel extractions succeeded", found);
    }

    /// Core differential check: for every tested tau, matfree must agree
    /// with the dense oracle. Returns (decisions, mismatches).
    fn differential(
        gf: &Gf,
        cols: &[Vec<u16>],
        s: usize,
        d: usize,
        taus: &[usize],
        seed: u64,
    ) -> (usize, usize, MatfreeStats) {
        let mut cfg = MatfreeConfig::default();
        cfg.seed = seed;
        let mut agg = MatfreeStats::default();
        let mut mism = 0usize;
        for &tau in taus {
            let want = distinguish(gf, cols, s, d, tau);
            let mut st = MatfreeStats::default();
            let got = distinguish_matfree_with(gf, cols, s, d, tau, &cfg, &mut st);
            agg.attempts += st.attempts;
            agg.extract_fail += st.extract_fail;
            agg.verify_fail += st.verify_fail;
            agg.verified += st.verified;
            if got != want {
                mism += 1;
                eprintln!(
                    "MISMATCH tau={} oracle={} matfree={} stats={:?}",
                    tau, want, got, st
                );
            }
        }
        (taus.len(), mism, agg)
    }

    fn pick_taus(n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = vec![0, 1, 2, 3, n / 2, n - 3, n - 2, n - 1];
        v.retain(|&t| t < n);
        v.sort_unstable();
        v.dedup();
        v
    }

    #[test]
    fn differential_random_codes() {
        // (q, n_pts, k, s, d) -- the task grid triples (k,s,d) appear here.
        let cases: [(u32, usize, usize, usize, usize); 4] = [
            (6, 36, 14, 2, 4),
            (6, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
            (7, 36, 14, 2, 4),
        ];
        let mut rng = Rng::new(4242);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = random_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential(&gf, &cols, s, d, &taus, 1000 + idx as u64);
            eprintln!(
                "random case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, stats={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "random case {} had mismatches", idx);
        }
    }

    #[test]
    fn differential_grs_codes() {
        let cases: [(u32, usize, usize, usize, usize); 3] = [
            (6, 40, 14, 2, 4),
            (7, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
        ];
        let mut rng = Rng::new(777);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = grs_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential(&gf, &cols, s, d, &taus, 2000 + idx as u64);
            eprintln!(
                "grs case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, stats={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "grs case {} had mismatches", idx);
        }
    }

    #[test]
    fn differential_rlce_shortened() {
        // (m, n, k, w, nprime, s, d): shortened k' = k - (N - n') hits the
        // grid values 19, 14, 12 respectively.
        let cases: [(u32, usize, usize, usize, usize, usize, usize); 3] = [
            (7, 44, 30, 8, 41, 2, 3),
            (7, 34, 20, 4, 32, 2, 4),
            (6, 26, 16, 3, 25, 3, 4),
        ];
        for (idx, &(m, n, k, w, nprime, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let mut rng = Rng::new(10_000 + idx as u64);
            let key = gen_rlce(&gf, &mut rng, n, k, w);
            let n_hit = key.n_pub - nprime;
            let hit = rng.subset(key.n_pub, n_hit);
            let (short_cols, _kept) = shorten(&gf, &key.cols, &hit);
            let kp = short_cols[0].len();
            let taus = pick_taus(nprime);
            let (dec, mism, agg) =
                differential(&gf, &short_cols, s, d, &taus, 3000 + idx as u64);
            eprintln!(
                "rlce case {} (q={},n={},k={},w={} -> n'={},k'={},s={},d={}): {} decisions, {} mismatches, stats={:?}",
                idx,
                1 << m,
                n,
                k,
                w,
                nprime,
                kp,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "rlce case {} had mismatches", idx);
        }
    }

    // ------------------------------------------------------------------
    // Benchmarks (ignored; run individually in release for clean VmHWM):
    //   cargo test --release --offline --bin rlce-attack bench_a_matfree_random \
    //       -- --ignored --nocapture --exact <full test path>
    // ------------------------------------------------------------------

    use std::time::Instant;

    /// Diagnostic: measure dim ker(M) vs dim ker(M^T M) and the isotropic
    /// radical (ker A vectors whose M-image is nonzero) on dense analogues.
    #[test]
    fn diag_kernel_gap() {
        use crate::linalg::kernel_basis;
        let gf = Gf::new(6);
        let mut rng = Rng::new(555);
        for &(n, k, s, d) in &[(16usize, 10usize, 2usize, 3usize), (20, 12, 2, 3), (16, 10, 2, 4), (14, 8, 3, 4)] {
            let cols = random_cols(&gf, &mut rng, n, k);
            let op = JetOp::new(k, s, d);
            let nd = op.nd();
            let tau = 0usize;
            // dense M_tau rows via hasse
            let mut rows: Vec<Vec<u16>> = Vec::new();
            for j in 0..n {
                if j == tau {
                    continue;
                }
                let yp = power_table(&gf, &cols[j], d);
                for a in op.jets.iter() {
                    let mut row = vec![0u16; nd];
                    for (mi, m) in op.monos.iter().enumerate() {
                        row[mi] = hasse(&gf, m, a, &yp);
                    }
                    rows.push(row);
                }
            }
            let e = rows.len();
            let (rank_m, ker_m) = kernel_basis(&gf, &rows);
            // A = M^T M (nd x nd)
            let mut arows = vec![vec![0u16; nd]; nd];
            for i in 0..nd {
                for j2 in 0..nd {
                    let mut acc = 0u16;
                    for r in 0..e {
                        if rows[r][i] != 0 && rows[r][j2] != 0 {
                            acc ^= gf.mul(rows[r][i], rows[r][j2]);
                        }
                    }
                    arows[i][j2] = acc;
                }
            }
            let (rank_a, ker_a) = kernel_basis(&gf, &arows);
            // radical: ker A basis vectors with M z != 0
            let mut radical = 0usize;
            for z in ker_a.iter() {
                let mz = op.matvec_m_vec(&gf, &cols, tau, z);
                if mz.iter().any(|&x| x != 0) {
                    radical += 1;
                }
            }
            eprintln!(
                "DIAG (n={},k={},s={},d={}): E={} N_d={} rankM={} dimKerM={} rankA={} dimKerA={} radical_basis_vecs={}/{}",
                n, k, s, d, e, nd, rank_m, ker_m.len(), rank_a, ker_a.len(), radical, ker_a.len()
            );
        }
    }

    /// Diagnostic 2: replicate the exact failing instance (random case 1:
    /// q=64, n=41, k=19, s=2, d=3, tau=0) and inspect Wiedemann samples.
    #[test]
    fn diag_failing_case() {
        use crate::linalg::kernel_basis;
        let gf = Gf::new(6);
        let mut rng = Rng::new(4242);
        let _case0 = random_cols(&gf, &mut rng, 36, 14); // consume case-0 draws
        let cols = random_cols(&gf, &mut rng, 41, 19);
        let (s, d, tau) = (2usize, 3usize, 0usize);
        let op = JetOp::new(19, s, d);
        let nd = op.nd();
        // dense ranks
        let mut rows: Vec<Vec<u16>> = Vec::new();
        for j in 0..41 {
            if j == tau {
                continue;
            }
            let yp = power_table(&gf, &cols[j], d);
            for a in op.jets.iter() {
                let mut row = vec![0u16; nd];
                for (mi, m) in op.monos.iter().enumerate() {
                    row[mi] = hasse(&gf, m, a, &yp);
                }
                rows.push(row);
            }
        }
        let (rank_m, _km) = kernel_basis(&gf, &rows);
        let e = rows.len();
        let mut arows = vec![vec![0u16; nd]; nd];
        for i in 0..nd {
            for j2 in 0..nd {
                let mut acc = 0u16;
                for r in 0..e {
                    if rows[r][i] != 0 && rows[r][j2] != 0 {
                        acc ^= gf.mul(rows[r][i], rows[r][j2]);
                    }
                }
                arows[i][j2] = acc;
            }
        }
        let (rank_a, _ka) = kernel_basis(&gf, &arows);
        eprintln!(
            "DIAG2: E={} N_d={} rankM={} rankA={} dimKerM={} dimKerA={}",
            e,
            nd,
            rank_m,
            rank_a,
            nd - rank_m,
            nd - rank_a
        );
        // Wiedemann samples: check Az, Mz separately
        let mut rng2 = Rng::new(1234567);
        for t in 0..4 {
            match op.kernel_sample(&gf, &cols, tau, &mut rng2) {
                KernelOutcome::Vector(z) => {
                    let mz = op.matvec_m_vec(&gf, &cols, tau, &z);
                    let mz_nz = mz.iter().filter(|&&x| x != 0).count();
                    // M^T (Mz): should be A z = 0
                    let mut mtmz = vec![0u16; nd];
                    op.matvec_mt(&gf, &cols, &mz, &mut mtmz);
                    let mtmz_nz = mtmz.iter().filter(|&&x| x != 0).count();
                    eprintln!(
                        "DIAG2 sample {}: |Mz|_nz={} |M^T M z|_nz={}",
                        t, mz_nz, mtmz_nz
                    );
                }
                KernelOutcome::Invertible => eprintln!("DIAG2 sample {}: Invertible", t),
                KernelOutcome::Failed => eprintln!("DIAG2 sample {}: Failed", t),
            }
        }
    }

    /// Benchmark (a), dense oracle, random points:
    /// (q=256, n'=32, k'=12, s=3, d=5); dense M_tau is 2,821 x 4,368 (~25 MB).
    /// (A larger both-runnable size was attempted at (40,13,3,5): dense took
    /// 44.6 s / 51.8 MB RSS there, but one scalar-Wiedemann sample at
    /// deg ~ 4096 did not finish within ~7 min CPU on this VM, so the
    /// completed pair is reported at this size; see MATFREE_REPORT.md.)
    #[test]
    #[ignore]
    fn bench_a_dense_random() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(31337);
        let cols = random_cols(&gf, &mut rng, 32, 12);
        let t0 = Instant::now();
        let r = distinguish(&gf, &cols, 3, 5, 0);
        eprintln!(
            "BENCH-A dense/random: decision={} time={:.2?} peakRSS={:.1} MB",
            r,
            t0.elapsed(),
            peak_rss_mb()
        );
    }

    /// Benchmark (a), matfree, random points (same instance parameters).
    #[test]
    #[ignore]
    fn bench_a_matfree_random() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(31337);
        let cols = random_cols(&gf, &mut rng, 32, 12);
        let cfg = MatfreeConfig::default();
        let mut st = MatfreeStats::default();
        let t0 = Instant::now();
        let r = distinguish_matfree_with(&gf, &cols, 3, 5, 0, &cfg, &mut st);
        eprintln!(
            "BENCH-A matfree/random: decision={} time={:.2?} peakRSS={:.1} MB stats={:?}",
            r,
            t0.elapsed(),
            peak_rss_mb(),
            st
        );
    }

    /// Benchmark (a), dense oracle, pure GRS points (accept path).
    #[test]
    #[ignore]
    fn bench_a_dense_grs() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(27182);
        let cols = grs_cols(&gf, &mut rng, 32, 12);
        let t0 = Instant::now();
        let r = distinguish(&gf, &cols, 3, 5, 0);
        eprintln!(
            "BENCH-A dense/grs: decision={} time={:.2?} peakRSS={:.1} MB",
            r,
            t0.elapsed(),
            peak_rss_mb()
        );
    }

    /// Benchmark (a), matfree, pure GRS points (accept path, all samples run).
    #[test]
    #[ignore]
    fn bench_a_matfree_grs() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(27182);
        let cols = grs_cols(&gf, &mut rng, 32, 12);
        let cfg = MatfreeConfig::default();
        let mut st = MatfreeStats::default();
        let t0 = Instant::now();
        let r = distinguish_matfree_with(&gf, &cols, 3, 5, 0, &cfg, &mut st);
        eprintln!(
            "BENCH-A matfree/grs: decision={} time={:.2?} peakRSS={:.1} MB stats={:?}",
            r,
            t0.elapsed(),
            peak_rss_mb(),
            st
        );
    }

    /// Benchmark (b): matfree-ONLY at paper scale (n'=348, k'=28, s=3, d=5,
    /// q=1024). Dense M_tau would be 150,945 x 201,376 = 61 GB: impossible.
    /// Reports operator build time, single matvec timings, a short Krylov
    /// run, peak RSS, and an extrapolated full-decision cost.
    #[test]
    #[ignore]
    fn bench_b_matfree_flagship() {
        let gf = Gf::new(10);
        let mut rng = Rng::new(60606);
        let (n, k, s, d) = (348usize, 28usize, 3usize, 5usize);
        let cols = random_cols(&gf, &mut rng, n, k);
        let t0 = Instant::now();
        let op = JetOp::new(k, s, d);
        let build_t = t0.elapsed();
        let nd = op.nd();
        let nj = op.njets();
        let e_tau = (n - 1) * nj;
        eprintln!(
            "BENCH-B: N_d={} J={} E_tau={} pairs/pt={} (nnz total {}) build={:.2?}",
            nd,
            nj,
            e_tau,
            op.pairs.len(),
            op.pairs.len() * (n - 1),
            build_t
        );
        let x = rand_vec(&gf, &mut rng, nd);
        let mut mid = vec![0u16; n * nj];
        let mut out = vec![0u16; nd];

        let t0 = Instant::now();
        op.matvec_m(&gf, &cols, 0, &x, &mut mid);
        let tm = t0.elapsed();
        let t0 = Instant::now();
        op.matvec_mt(&gf, &cols, &mid, &mut out);
        let tmt = t0.elapsed();
        let mut mid2: Vec<u16> = Vec::new();
        let t0 = Instant::now();
        op.apply_a(&gf, &cols, 0, &x, &mut out, &mut mid2);
        let ta = t0.elapsed();
        eprintln!(
            "BENCH-B: matvec_M={:.3?} matvec_MT={:.3?} apply_A={:.3?}",
            tm, tmt, ta
        );
        // Short Krylov run: 30 A-applications.
        let steps = 30;
        let mut w = x.clone();
        let mut w2 = vec![0u16; nd];
        let mut mid3: Vec<u16> = Vec::new();
        let t0 = Instant::now();
        for _ in 0..steps {
            op.apply_a(&gf, &cols, 0, &w, &mut w2, &mut mid3);
            std::mem::swap(&mut w, &mut w2);
        }
        let tk = t0.elapsed();
        let per_a = tk / steps;
        // Extrapolation: deg(mu) ~ E_tau + 1; one kernel sample costs
        // ~2*deg (pass 1) + ~deg (pass 2) A-applications.
        let deg = (e_tau + 1) as f64;
        let per_sample = per_a.as_secs_f64() * 3.0 * deg;
        eprintln!(
            "BENCH-B: {} Krylov steps in {:.2?} ({:.3?}/A-apply); extrapolated 1 kernel sample ~ {:.1} s, \
             full decision (<= {} samples + verifies) ~ {:.1} s; peakRSS={:.1} MB",
            steps,
            tk,
            per_a,
            per_sample,
            MatfreeConfig::default().samples,
            per_sample * MatfreeConfig::default().samples as f64,
            peak_rss_mb()
        );
    }
}

// ---------------------------------------------------------------------------
// Block Wiedemann (Coppersmith/Kaltofen style) on top of JetOp
// ---------------------------------------------------------------------------
//
// Upgrades the scalar engine above. The scalar sampler costs ~3 * deg(mu)
// A-applies PER KERNEL SAMPLE, and a decision needs up to `samples` of them.
// Block Wiedemann generates ONE block Krylov sequence for a block of b
// random start vectors and extracts up to ~b independent kernel candidates
// from it, so the operator work is shared by all samples:
//
//   Pass 1: W_k = A^k B0 (block applies), record only the b x b projections
//           S_k = U^T W_k for random U. Sequence length 2L with
//           L ~ ceil(kappa / b), kappa = Krylov dimension (<= rank(A) + 1).
//   Generator: any f = (f_0..f_{L-1}), f_j in F_q^b, annihilating the
//           projected sequence on the SHIFTED window,
//               sum_j S_{i+j} f_j = 0   for i = 1..L,
//           is a null vector of the L x L block-Toeplitz matrix built
//           from the S_k. We compute its nullspace by flat forward
//           elimination (suffix updates) + back-substitution. The window
//           starts at i = 1 (not 0) on purpose: the relations must
//           constrain A Z, not Z itself -- see block_wiedemann_kernel.
//   Pass 2: regenerate W_j (operator applies are deterministic) and form
//           Z = sum_j W_j f_j for every null vector simultaneously.
//           The relations give U^T A^i (A Z) = 0 for i < L, and the
//           functionals x |-> U^T A^i x (i < L) separate points of the
//           Krylov space when L * b > kappa, so A Z = 0 w.h.p. As
//           everywhere in this module, the heuristic step is NOT
//           trusted: every candidate is checked A z == 0 exactly, and
//           near-miss candidates (A z != 0 but tiny) are combined by
//           defect cancellation -- c in ker[A z_1 ... A z_m] gives
//           A (sum c_t z_t) = 0 exactly -- before the decision layer
//           re-verifies M z == 0 after radical cancellation.
//
// Cost accounting vs scalar: one block run = (2L pass-1 + (L-1) pass-2)
// block applies + one Toeplitz solve (~ (Lb)^3 / 3 field ops, dense) +
// O(L * n * b * nullity) extraction accumulation. A block apply streams the
// incidence pairs ONCE for b right-hand sides (eval tables are also built
// once per point per apply, not once per column), and is point-parallel
// via rayon. Field addition in char 2 is XOR, so the parallel reduction in
// matvec_mt_block is exact and order-independent.

use rayon::prelude::*;
use std::time::Instant;

fn binom_usize(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r = 1usize;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

/// Random flat block: n rows x b columns, row-major.
fn rand_block(gf: &Gf, rng: &mut Rng, n: usize, b: usize) -> Vec<u16> {
    let q = gf.order as u64;
    let mut v: Vec<u16> = (0..n * b).map(|_| rng.below(q) as u16).collect();
    if v.iter().all(|&x| x == 0) {
        v[0] = 1;
    }
    v
}

impl JetOp {
    /// out = M x for a block x of width b (flat: nd x b in, (n_pts*J) x b
    /// out). Row blocks of distinct points are disjoint, so this is
    /// point-parallel via rayon; the excluded point's block stays zero.
    pub fn matvec_m_block(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        x: &[u16],
        b: usize,
        out: &mut [u16],
    ) {
        let nj = self.jets.len();
        out.fill(0);
        out.par_chunks_mut(nj * b)
            .zip(cols.par_iter())
            .enumerate()
            .for_each(|(j, (blk, y))| {
                if j == excl {
                    return;
                }
                let ev = self.eval_table(gf, y);
                for &(ai, mi, bi) in self.pairs.iter() {
                    let evv = ev[bi as usize];
                    if evv == 0 {
                        continue;
                    }
                    let xrow = &x[(mi as usize) * b..(mi as usize + 1) * b];
                    if xrow.iter().all(|&v| v == 0) {
                        continue;
                    }
                    let orow = &mut blk[(ai as usize) * b..(ai as usize + 1) * b];
                    for c in 0..b {
                        let xv = xrow[c];
                        if xv != 0 {
                            orow[c] ^= gf.mul(evv, xv);
                        }
                    }
                }
            });
    }

    /// out = M^T u for a block u of width b ((n_pts*J) x b in, nd x b out).
    /// Point-parallel with per-thread partial sums, XOR-reduced at the end
    /// (exact in char 2). Falls back to a sequential loop for tiny inputs.
    pub fn matvec_mt_block(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        u: &[u16],
        b: usize,
        out: &mut [u16],
    ) {
        let nj = self.jets.len();
        let nd = self.nd();
        let npts = cols.len();
        let nthreads = rayon::current_num_threads().max(1);
        if npts < 2 * nthreads || nthreads == 1 {
            out.fill(0);
            for (j, y) in cols.iter().enumerate() {
                let ev = self.eval_table(gf, y);
                let base = j * nj;
                for &(ai, mi, bi) in self.pairs.iter() {
                    let evv = ev[bi as usize];
                    if evv == 0 {
                        continue;
                    }
                    let urow = &u[(base + ai as usize) * b..(base + ai as usize + 1) * b];
                    if urow.iter().all(|&v| v == 0) {
                        continue;
                    }
                    let orow = &mut out[(mi as usize) * b..(mi as usize + 1) * b];
                    for c in 0..b {
                        let uv = urow[c];
                        if uv != 0 {
                            orow[c] ^= gf.mul(evv, uv);
                        }
                    }
                }
            }
            return;
        }
        let chunk = (npts + nthreads - 1) / nthreads;
        let partials: Vec<Vec<u16>> = (0..nthreads)
            .into_par_iter()
            .map(|t| {
                let mut acc = vec![0u16; nd * b];
                let lo = t * chunk;
                let hi = ((t + 1) * chunk).min(npts);
                for j in lo..hi {
                    let ev = self.eval_table(gf, &cols[j]);
                    let base = j * nj;
                    for &(ai, mi, bi) in self.pairs.iter() {
                        let evv = ev[bi as usize];
                        if evv == 0 {
                            continue;
                        }
                        let urow = &u[(base + ai as usize) * b..(base + ai as usize + 1) * b];
                        if urow.iter().all(|&v| v == 0) {
                            continue;
                        }
                        let orow = &mut acc[(mi as usize) * b..(mi as usize + 1) * b];
                        for c in 0..b {
                            let uv = urow[c];
                            if uv != 0 {
                                orow[c] ^= gf.mul(evv, uv);
                            }
                        }
                    }
                }
                acc
            })
            .collect();
        out.fill(0);
        for p in partials.iter() {
            for i in 0..nd * b {
                out[i] ^= p[i];
            }
        }
    }

    /// out = A x with A = M_tau^T M_tau (tau = excl; excl == cols.len()
    /// means the full matrix, no exclusion), block width b.
    /// `mid` is scratch of length n_pts * J * b.
    pub fn apply_a_block(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        x: &[u16],
        b: usize,
        out: &mut [u16],
        mid: &mut Vec<u16>,
    ) {
        let need = cols.len() * self.jets.len() * b;
        if mid.len() != need {
            mid.resize(need, 0);
        }
        self.matvec_m_block(gf, cols, excl, x, b, mid);
        self.matvec_mt_block(gf, cols, mid, b, out);
    }

    /// Control-group block apply: each column is an independent SCALAR
    /// apply_a, parallelised over columns with rayon. No pair-traversal
    /// amortisation (pairs are re-streamed per column and eval tables are
    /// rebuilt per column); used in benchmarks to isolate what the shared
    /// traversal of apply_a_block actually buys.
    pub fn apply_a_block_colpar(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        x: &[u16],
        b: usize,
        out: &mut [u16],
    ) {
        let nd = self.nd();
        let cols_x: Vec<Vec<u16>> = (0..b)
            .map(|c| (0..nd).map(|i| x[i * b + c]).collect())
            .collect();
        let mut results: Vec<Vec<u16>> = vec![vec![0u16; nd]; b];
        results
            .par_iter_mut()
            .zip(cols_x.par_iter())
            .for_each(|(res, xc)| {
                let mut mid: Vec<u16> = Vec::new();
                self.apply_a(gf, cols, excl, xc, res, &mut mid);
            });
        for c in 0..b {
            for i in 0..nd {
                out[i * b + c] = results[c][i];
            }
        }
    }
}

/// Nullspace basis of a flat row-major matrix (rows x cols) over gf, by
/// forward elimination (suffix updates only -- valid for nullspace
/// extraction: at step c only columns >= c are touched, and
/// back-substitution for a pivot row only reads columns past its pivot)
/// followed by back-substitution. Returns one basis vector per free
/// column, each of length cols.
fn nullspace_flat(gf: &Gf, mat: &[u16], rows: usize, cols: usize) -> Vec<Vec<u16>> {
    if rows == 0 || cols == 0 {
        return (0..cols)
            .map(|f| {
                let mut v = vec![0u16; cols];
                v[f] = 1;
                v
            })
            .collect();
    }
    let mut a: Vec<u16> = mat.to_vec();
    let mut pivot_of_col: Vec<i64> = vec![-1; cols];
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let mut piv: Option<usize> = None;
        for rr in r..rows {
            if a[rr * cols + c] != 0 {
                piv = Some(rr);
                break;
            }
        }
        let p = match piv {
            Some(p) => p,
            None => continue,
        };
        if p != r {
            for cc in 0..cols {
                a.swap(r * cols + cc, p * cols + cc);
            }
        }
        let inv = gf.inv(a[r * cols + c]);
        if inv != 1 {
            for cc in c..cols {
                let v = a[r * cols + cc];
                if v != 0 {
                    a[r * cols + cc] = gf.mul(v, inv);
                }
            }
        }
        // Eliminate column c in rows below r (suffix updates only).
        for rr in (r + 1)..rows {
            let f = a[rr * cols + c];
            if f != 0 {
                for cc in c..cols {
                    let pv = a[r * cols + cc];
                    if pv != 0 {
                        let sub = gf.mul(f, pv);
                        a[rr * cols + cc] ^= sub;
                    }
                }
                a[rr * cols + c] = 0;
            }
        }
        pivot_of_col[c] = r as i64;
        pivot_cols.push(c);
        r += 1;
    }
    // Basis: for each free column f, x_f = 1, other free vars 0, pivot vars
    // by back-substitution: x_pc + sum_{j>pc} a[row][j] x_j = 0, char 2.
    let mut basis = Vec::new();
    for f in 0..cols {
        if pivot_of_col[f] >= 0 {
            continue;
        }
        let mut v = vec![0u16; cols];
        v[f] = 1;
        for &pc in pivot_cols.iter().rev() {
            let row = pivot_of_col[pc] as usize;
            let mut acc = 0u16;
            for j in (pc + 1)..cols {
                let av = a[row * cols + j];
                if av != 0 && v[j] != 0 {
                    acc ^= gf.mul(av, v[j]);
                }
            }
            v[pc] = acc;
        }
        basis.push(v);
    }
    basis
}

/// Per-run diagnostics for one block Wiedemann extraction.
#[derive(Default, Debug, Clone)]
pub struct BlockRunStats {
    pub block_applies: usize,
    pub l_blocks: usize,
    pub nullity: usize,
    pub candidates: usize,
    pub krylov_secs: f64,
    pub toeplitz_secs: f64,
    /// Sigma-basis extraction time (sigma path only; 0 on Toeplitz path).
    pub sigma_secs: f64,
    pub extract_secs: f64,
}

/// Outcome of one block Wiedemann extraction (mirrors KernelOutcome).
pub enum BlockOutcome {
    /// Verified nonzero kernel vectors of A (each checked A z == 0).
    Vectors(Vec<Vec<u16>>),
    /// Toeplitz nullspace was empty with L*b >= n: the Krylov space is the
    /// whole space and has no kernel relation, so A is invertible.
    Invertible,
    /// Extraction failed (bad blocks, or no candidate verified).
    Failed,
}

/// Cap on Toeplitz width L*b in one extraction (dense solve memory/time).
const BW_MAX_TOEPLITZ_WIDTH: usize = 6144;
/// Cap on null vectors carried into the extraction pool.
const BW_MAX_POOL: usize = 160;

/// One block Wiedemann kernel extraction on a black-box operator A on
/// F_q^n. `apply_block` computes A * X for a flat n x b block;
/// `apply1` computes A * x for a single vector (used for exact
/// verification). `l_blocks` is the Krylov block length L (pass 1 uses
/// 2L block applies, pass 2 uses L-1). Returns up to `max_vecs` verified
/// kernel vectors.
pub fn block_wiedemann_kernel(
    gf: &Gf,
    n: usize,
    l_blocks: usize,
    b: usize,
    max_vecs: usize,
    apply_block: &mut dyn FnMut(&[u16], &mut [u16]),
    apply1: &mut dyn FnMut(&[u16], &mut [u16]),
    rng: &mut Rng,
    stats: &mut BlockRunStats,
) -> BlockOutcome {
    if n == 0 || max_vecs == 0 {
        return BlockOutcome::Failed;
    }
    let b = b.max(1).min(n);
    let l = l_blocks.max(2);
    let seq_len = 2 * l;
    let b0 = rand_block(gf, rng, n, b);
    let u = rand_block(gf, rng, n, b);

    // Pass 1: block Krylov sequence, projections only.
    let t0 = Instant::now();
    let mut seq = vec![0u16; seq_len * b * b];
    let mut w = b0.clone();
    let mut w2 = vec![0u16; n * b];
    for k in 0..seq_len {
        {
            let sk = &mut seq[k * b * b..(k + 1) * b * b];
            for i in 0..n {
                let urow = &u[i * b..(i + 1) * b];
                let wrow = &w[i * b..(i + 1) * b];
                if wrow.iter().all(|&v| v == 0) {
                    continue;
                }
                for r in 0..b {
                    let uv = urow[r];
                    if uv == 0 {
                        continue;
                    }
                    for c in 0..b {
                        let wv = wrow[c];
                        if wv != 0 {
                            sk[r * b + c] ^= gf.mul(uv, wv);
                        }
                    }
                }
            }
        }
        if k + 1 < seq_len {
            apply_block(&w, &mut w2);
            stats.block_applies += 1;
            std::mem::swap(&mut w, &mut w2);
        }
    }
    stats.krylov_secs = t0.elapsed().as_secs_f64();
    stats.l_blocks = l;

    // Block-Toeplitz nullspace: rows i = 1..=l, unknowns f_0..f_{l-1}.
    // Row block i imposes U^T A^i (A Z) = 0 where Z = sum_j W_j f_j, i.e.
    // the relations constrain A Z, NOT Z itself. (Including row block
    // i = 0 would impose U^T Z = 0 as well; when the functionals
    // {A^i U} span the whole space -- e.g. rank(A) + b >= n -- that
    // forces Z = 0 identically and no kernel vector can ever be
    // extracted, however large ker(A) is. Observed in practice on the
    // reuse operator B = M M^T at (k,s,d) = (14,2,4): all 64 extracted
    // candidates were the zero vector.)
    let t0 = Instant::now();
    let rows = l * b;
    let cols = l * b;
    let mut tmat = vec![0u16; rows * cols];
    for i in 1..=l {
        for j in 0..l {
            let sij = &seq[(i + j) * b * b..(i + j + 1) * b * b];
            for r in 0..b {
                let dst = ((i - 1) * b + r) * cols + j * b;
                tmat[dst..dst + b].copy_from_slice(&sij[r * b..(r + 1) * b]);
            }
        }
    }
    let basis = nullspace_flat(gf, &tmat, rows, cols);
    stats.toeplitz_secs = t0.elapsed().as_secs_f64();
    stats.nullity = basis.len();
    if basis.is_empty() {
        return if l * b >= n {
            BlockOutcome::Invertible
        } else {
            BlockOutcome::Failed
        };
    }

    // Pass 2: regenerate W_j and accumulate Z_all = sum_j W_j V_j for the
    // nullspace basis matrix V (cols x nu), all null vectors at once.
    let t0 = Instant::now();
    let nu = basis.len().min(BW_MAX_POOL);
    let mut vmat = vec![0u16; cols * nu];
    for (t, bv) in basis.iter().take(nu).enumerate() {
        for (idx, &val) in bv.iter().enumerate() {
            vmat[idx * nu + t] = val;
        }
    }
    let mut zall = vec![0u16; n * nu];
    let mut w = b0.clone();
    let mut w2 = vec![0u16; n * b];
    for j in 0..l {
        zall.par_chunks_mut(nu).enumerate().for_each(|(i, zrow)| {
            let wrow = &w[i * b..(i + 1) * b];
            for jj in 0..b {
                let wv = wrow[jj];
                if wv == 0 {
                    continue;
                }
                let vrow = &vmat[(j * b + jj) * nu..(j * b + jj + 1) * nu];
                for t in 0..nu {
                    let vv = vrow[t];
                    if vv != 0 {
                        zrow[t] ^= gf.mul(wv, vv);
                    }
                }
            }
        });
        if j + 1 < l {
            apply_block(&w, &mut w2);
            stats.block_applies += 1;
            std::mem::swap(&mut w, &mut w2);
        }
    }
    stats.candidates = nu;
    // Exact verification of every candidate column. Candidates whose
    // defect d = A z is nonzero are kept for defect cancellation below.
    let mut out_vecs: Vec<Vec<u16>> = Vec::new();
    let mut defected: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
    for t in 0..nu {
        let z: Vec<u16> = (0..n).map(|i| zall[i * nu + t]).collect();
        if z.iter().all(|&v| v == 0) {
            continue;
        }
        let mut az = vec![0u16; n];
        apply1(&z, &mut az);
        if az.iter().all(|&v| v == 0) {
            if out_vecs.len() < max_vecs {
                out_vecs.push(z);
            }
        } else {
            defected.push((z, az));
        }
    }
    // Defect cancellation: the Toeplitz relations force A Z into the
    // orthogonal complement of the Krylov functionals, which need not be
    // zero (observed defects have weight as low as 1-3 at some scales).
    // Combinations c in ker[d_1 ... d_m] of the defect vectors give
    // A z(c) = sum_t c_t d_t = 0 EXACTLY; each is verified again anyway.
    if out_vecs.len() < max_vecs && defected.len() >= 2 {
        let m = defected.len();
        let drows: Vec<Vec<u16>> = (0..n)
            .map(|r| (0..m).map(|t| defected[t].1[r]).collect())
            .collect();
        let (_drk, dker) = crate::linalg::kernel_basis(gf, &drows);
        for c in dker.iter() {
            if out_vecs.len() >= max_vecs {
                break;
            }
            let mut zc = vec![0u16; n];
            for (t, &ct) in c.iter().enumerate() {
                if ct != 0 {
                    for r in 0..n {
                        if defected[t].0[r] != 0 {
                            zc[r] ^= gf.mul(ct, defected[t].0[r]);
                        }
                    }
                }
            }
            if zc.iter().all(|&v| v == 0) {
                continue;
            }
            let mut azc = vec![0u16; n];
            apply1(&zc, &mut azc);
            if azc.iter().all(|&v| v == 0) {
                out_vecs.push(zc);
            }
        }
    }
    stats.extract_secs = t0.elapsed().as_secs_f64();
    if out_vecs.is_empty() {
        BlockOutcome::Failed
    } else {
        BlockOutcome::Vectors(out_vecs)
    }
}

/// Block extraction with L escalation: start from an Euler-corrected rank
/// estimate for the Krylov dimension and grow L while extraction fails
/// and the Toeplitz width stays under the dense-solve cap. Generic over
/// the operator via the two apply closures.
#[allow(clippy::too_many_arguments)]
pub fn bw_kernel_escalating(
    gf: &Gf,
    n: usize,
    deg_est: usize,
    b: usize,
    max_vecs: usize,
    apply_block: &mut dyn FnMut(&[u16], &mut [u16]),
    apply1: &mut dyn FnMut(&[u16], &mut [u16]),
    rng: &mut Rng,
    agg: &mut BlockRunStats,
) -> BlockOutcome {
    let b = b.max(1).min(n.max(1));
    let mut l = (deg_est + b - 1) / b + 2;
    loop {
        if l * b > BW_MAX_TOEPLITZ_WIDTH {
            // Dense Toeplitz solve would exceed its budget. (When
            // n > BW_MAX_TOEPLITZ_WIDTH this also means we cannot
            // certify Invertible on this path; Failed is the honest
            // outcome -- the decision layer treats it heuristically.)
            return BlockOutcome::Failed;
        }
        let mut st = BlockRunStats::default();
        let out = block_wiedemann_kernel(gf, n, l, b, max_vecs, apply_block, apply1, rng, &mut st);
        agg.block_applies += st.block_applies;
        agg.krylov_secs += st.krylov_secs;
        agg.toeplitz_secs += st.toeplitz_secs;
        agg.extract_secs += st.extract_secs;
        agg.nullity = st.nullity;
        agg.l_blocks = st.l_blocks;
        agg.candidates = st.candidates;
        match out {
            BlockOutcome::Failed => {
                if l * b >= n {
                    return BlockOutcome::Failed;
                }
                l += (l / 2).max(4);
            }
            other => return other,
        }
    }
}

// ---------------------------------------------------------------------------
// Block-Wiedemann distinguisher (same decision semantics as scalar)
// ---------------------------------------------------------------------------

/// Tunables for distinguish_matfree_bw.
pub struct BWConfig {
    pub block_size: usize,
    /// Independent certified ker(M_tau) vectors (after radical
    /// cancellation) that must all satisfy z . v = 0 before accepting.
    pub samples: usize,
    /// Hard cap on block extractions per decision.
    pub max_runs: usize,
    /// Give up after this many consecutive failed runs with an empty
    /// sample pool (mirrors the scalar config's no-kernel streak).
    pub max_no_kernel_streak: usize,
    pub seed: u64,
}

impl Default for BWConfig {
    fn default() -> Self {
        BWConfig {
            block_size: 32,
            samples: 4,
            max_runs: 8,
            max_no_kernel_streak: 4,
            seed: 0x243F_6A88_85A3_08D3,
        }
    }
}

/// Counters for the block distinguisher.
#[derive(Default, Debug, Clone)]
pub struct BWStats {
    pub runs: usize,
    pub block_applies: usize,
    pub extract_fail: usize,
    /// Certified combinations that failed the exact M z = 0 check
    /// (expected: 0; cancellation is exact by construction).
    pub verify_fail: usize,
    /// Independent certified ker(M_tau) vectors at decision time.
    pub verified: usize,
    pub no_kernel: usize,
    pub radical_rank: usize,
    pub candidates: usize,
    pub nullity: usize,
    pub l_blocks: usize,
    pub krylov_secs: f64,
    pub toeplitz_secs: f64,
    /// Accumulated sigma-basis extraction time (sigma distinguisher only).
    pub sigma_secs: f64,
    pub extract_secs: f64,
}

impl JetOp {
    /// Euler-corrected estimate of rank(A) (hence of the Krylov
    /// dimension): rank(M) <= E - (points) * C(k+s-2, s-2) by the
    /// homogeneity syzygies, exact on random controls. Used only to size
    /// L; extraction failure escalates L, so an underestimate is safe.
    fn deg_estimate(&self, npts_in: usize) -> usize {
        let euler_pt = if self.s >= 2 {
            binom_usize(self.k + self.s - 2, self.s - 2)
        } else {
            0
        };
        let e = npts_in * self.njets();
        let rank_est = e.saturating_sub(npts_in * euler_pt);
        rank_est.min(self.nd()).max(1) + 1
    }

    /// One block-Wiedemann batch of ker(A) samples for A = M_tau^T M_tau
    /// (tau = excl; excl == cols.len() gives the full-matrix operator).
    pub fn kernel_batch_bw(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        block_size: usize,
        max_vecs: usize,
        rng: &mut Rng,
        stats: &mut BWStats,
    ) -> BlockOutcome {
        let nd = self.nd();
        let npts_in = if excl < cols.len() {
            cols.len() - 1
        } else {
            cols.len()
        };
        let deg_est = self.deg_estimate(npts_in);
        let mut mid: Vec<u16> = Vec::new();
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            self.apply_a_block(gf, cols, excl, x, block_size, out, &mut mid);
        };
        let mut mid1: Vec<u16> = Vec::new();
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            self.apply_a(gf, cols, excl, x, out, &mut mid1);
        };
        let mut agg = BlockRunStats::default();
        let out = bw_kernel_escalating(
            gf,
            nd,
            deg_est,
            block_size,
            max_vecs,
            &mut apply_block,
            &mut apply1,
            rng,
            &mut agg,
        );
        stats.block_applies += agg.block_applies;
        stats.krylov_secs += agg.krylov_secs;
        stats.toeplitz_secs += agg.toeplitz_secs;
        stats.extract_secs += agg.extract_secs;
        stats.nullity = agg.nullity;
        stats.l_blocks = agg.l_blocks;
        stats.candidates = agg.candidates;
        out
    }
}

/// Radical cancellation + decision test over an accumulated sample pool,
/// identical logic to the scalar distinguisher: coefficient vectors c in
/// ker[w_1 ... w_m] (w_i = M z_i) give combinations z(c) in ker(M_tau);
/// the whole certified basis is verified exactly (one block matvec) and
/// tested against v. Returns Some(false) on a certified separator,
/// Some(true) once `samples` independent certified vectors all satisfy
/// z . v = 0, None if more samples are needed.
#[allow(clippy::too_many_arguments)]
fn cancel_and_test_bw(
    gf: &Gf,
    op: &JetOp,
    cols: &[Vec<u16>],
    tau: usize,
    zs: &[Vec<u16>],
    ws: &[Vec<u16>],
    e_len: usize,
    v: &[u16],
    samples: usize,
    stats: &mut BWStats,
) -> Option<bool> {
    let m = zs.len();
    if m == 0 {
        return None;
    }
    let wrows: Vec<Vec<u16>> = (0..e_len)
        .map(|r| (0..m).map(|i| ws[i][r]).collect())
        .collect();
    let (_wrank, ker) = crate::linalg::kernel_basis(gf, &wrows);
    stats.radical_rank = m - ker.len();
    // Build all candidate combinations, then verify M z(c) == 0 in ONE
    // block matvec (columns = combinations).
    let nd = op.nd();
    let ncomb = ker.len();
    if ncomb == 0 {
        return None;
    }
    let mut zcall = vec![0u16; nd * ncomb];
    for (t, c) in ker.iter().enumerate() {
        for (i, &ci) in c.iter().enumerate() {
            if ci != 0 {
                for r in 0..nd {
                    if zs[i][r] != 0 {
                        zcall[r * ncomb + t] ^= gf.mul(ci, zs[i][r]);
                    }
                }
            }
        }
    }
    let mut mzcall = vec![0u16; e_len * ncomb];
    op.matvec_m_block(gf, cols, tau, &zcall, ncomb, &mut mzcall);
    let mut certified = 0usize;
    for t in 0..ncomb {
        let zc: Vec<u16> = (0..nd).map(|r| zcall[r * ncomb + t]).collect();
        if zc.iter().all(|&x| x == 0) {
            continue;
        }
        let col_ok = (0..e_len).all(|r| mzcall[r * ncomb + t] == 0);
        if !col_ok {
            stats.verify_fail += 1;
            continue;
        }
        certified += 1;
        if dot(gf, &zc, v) != 0 {
            stats.verified = certified;
            return Some(false); // certified: v_tau not in rowspan
        }
    }
    stats.verified = certified;
    if certified >= samples {
        return Some(true);
    }
    None
}

/// Block-Wiedemann matrix-free GIJS distinguisher with explicit config
/// and stats. Same contract and error model as distinguish_matfree_with:
/// `false` is always certified exactly; `true` is heuristic with
/// false-accept probability ~ q^-cfg.samples.
pub fn distinguish_matfree_bw_with(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    tau: usize,
    cfg: &BWConfig,
    stats: &mut BWStats,
) -> bool {
    let k = cols[0].len();
    let op = JetOp::new(k, s, d);
    let v = op.eval_vec(gf, &cols[tau]);
    let mut rng = Rng::new(
        cfg.seed
            ^ (tau as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((cols.len() as u64) << 32)
            ^ 0x27D4_EB2F_1656_67C5,
    );
    let e_len = cols.len() * op.njets();
    let mut zs: Vec<Vec<u16>> = Vec::new();
    let mut ws: Vec<Vec<u16>> = Vec::new();
    let mut fail_streak = 0usize;
    let max_vecs = cfg.block_size + cfg.samples + 8;
    for _run in 0..cfg.max_runs {
        stats.runs += 1;
        match op.kernel_batch_bw(gf, cols, tau, cfg.block_size, max_vecs, &mut rng, stats) {
            BlockOutcome::Invertible => {
                stats.no_kernel += 1;
                return true;
            }
            BlockOutcome::Failed => {
                stats.extract_fail += 1;
                fail_streak += 1;
                if zs.is_empty() && fail_streak >= cfg.max_no_kernel_streak {
                    break;
                }
            }
            BlockOutcome::Vectors(batch) => {
                fail_streak = 0;
                for z in batch.iter() {
                    let w = op.matvec_m_vec(gf, cols, tau, z);
                    ws.push(w);
                }
                zs.extend(batch);
                if let Some(dec) = cancel_and_test_bw(
                    gf, &op, cols, tau, &zs, &ws, e_len, &v, cfg.samples, stats,
                ) {
                    return dec;
                }
            }
        }
    }
    true
}

/// Block-Wiedemann matrix-free GIJS distinguisher; same contract as
/// gijs::distinguish / distinguish_matfree.
pub fn distinguish_matfree_bw(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    tau: usize,
    block_size: usize,
    samples: usize,
) -> bool {
    let mut cfg = BWConfig::default();
    cfg.block_size = block_size;
    cfg.samples = samples;
    let mut stats = BWStats::default();
    distinguish_matfree_bw_with(gf, cols, s, d, tau, &cfg, &mut stats)
}

// ---------------------------------------------------------------------------
// Block Wiedemann with sigma-basis extraction (low memory)
// ---------------------------------------------------------------------------
//
// Same block Krylov pass 1, same candidate verification / defect
// cancellation, and the same decision semantics as the Toeplitz engine
// above -- but the generator extraction is the sigma-basis of src/sigma.rs
// (minimal approximant basis of the augmented series [Scr^T; I]; the
// M-Basis iteration of the Giorgi--Jeannerod--Villard PM-Basis family).
// The extraction's memory is O(sigma * b^2) field elements -- linear in
// the sequence length sigma = 2L at fixed b -- versus (L*b)^2 entries for
// the dense block-Toeplitz nullspace, so this path is NOT subject to
// BW_MAX_TOEPLITZ_WIDTH; its cap is BW_MAX_SIGMA_TERMS on the sequence
// length (a memory budget, see below).
//
// Candidates are the basis rows with nonzero F-part and delta = deg F
// <= L. The remainder degree decides which combination of the Krylov
// blocks W_p = A^p B0 lies in ker(A):
//   * deg G <= delta:     Z  = sum_p W_p F_{delta-p}   is in ker(A),
//     since U^T A^i (A Z) = (Scr F)_{i+delta+1} = G_{i+delta+1} = 0.
//   * deg G = delta + 1:  the same identity leaves one block,
//     U^T (A Z) = G_{delta+1}, possibly nonzero -- but the A-shifted
//     combination AZ = sum_p W_p F_{delta+1-p} satisfies
//     U^T A^i (A (AZ)) = (Scr F)_{i+delta+2} = 0 for all i >= 0, so
//     AZ is in ker(A). Rows of this second kind DO occur: they are the
//     G-pivot family of the augmented basis (observed on GRS instances,
//     where all 32 generator rows have deg G = deg F + 1). Both
//     combinations are emitted per pool row; exact verification in the
//     shared tail discards whichever does not apply. Reversal
//     f_j = F_{delta-j} turns the first kind into a null vector of the
//     Toeplitz system of block_wiedemann_kernel (window i = 1..=L), and
//     the shifted reversal x*f into one for the second kind; the
//     head-to-head test in sigma_tests checks both exactly.

/// Cap on Krylov sequence length (2L terms) on the sigma path. Memory is
/// ~3 * terms * b^2 field elements (series + residual + basis) plus the
/// (2b)^2 * terms basis coefficients counted once more; at b = 32 and
/// 2^15 terms that is ~200 MB worst case. Realistic runs sit far below:
/// flagship (3,5) needs 2L = 8,810 terms (~55 MB of basis-side storage).
const BW_MAX_SIGMA_TERMS: usize = 1 << 15;

/// Transpose of a flat b x b row-major matrix.
fn transpose_bxb(mat: &[u16], b: usize) -> Vec<u16> {
    let mut out = vec![0u16; b * b];
    for i in 0..b {
        for j in 0..b {
            out[j * b + i] = mat[i * b + j];
        }
    }
    out
}

/// One block Wiedemann kernel extraction, sigma-basis version of
/// block_wiedemann_kernel (identical signature and outcome contract).
/// rng draws (b0, u) happen in the same order as the Toeplitz version, so
/// a fixed seed gives the two engines the same Krylov sequence.
pub fn block_wiedemann_kernel_sigma(
    gf: &Gf,
    n: usize,
    l_blocks: usize,
    b: usize,
    max_vecs: usize,
    apply_block: &mut dyn FnMut(&[u16], &mut [u16]),
    apply1: &mut dyn FnMut(&[u16], &mut [u16]),
    rng: &mut Rng,
    stats: &mut BlockRunStats,
) -> BlockOutcome {
    if n == 0 || max_vecs == 0 {
        return BlockOutcome::Failed;
    }
    let b = b.max(1).min(n);
    let l = l_blocks.max(2);
    let seq_len = 2 * l;
    let b0 = rand_block(gf, rng, n, b);
    let u = rand_block(gf, rng, n, b);

    // Pass 1: block Krylov sequence, projections only (identical to
    // block_wiedemann_kernel).
    let t0 = Instant::now();
    let mut seq = vec![0u16; seq_len * b * b];
    let mut w = b0.clone();
    let mut w2 = vec![0u16; n * b];
    for k in 0..seq_len {
        {
            let sk = &mut seq[k * b * b..(k + 1) * b * b];
            for i in 0..n {
                let urow = &u[i * b..(i + 1) * b];
                let wrow = &w[i * b..(i + 1) * b];
                if wrow.iter().all(|&v| v == 0) {
                    continue;
                }
                for r in 0..b {
                    let uv = urow[r];
                    if uv == 0 {
                        continue;
                    }
                    for c in 0..b {
                        let wv = wrow[c];
                        if wv != 0 {
                            sk[r * b + c] ^= gf.mul(uv, wv);
                        }
                    }
                }
            }
        }
        if k + 1 < seq_len {
            apply_block(&w, &mut w2);
            stats.block_applies += 1;
            std::mem::swap(&mut w, &mut w2);
        }
    }
    stats.krylov_secs = t0.elapsed().as_secs_f64();
    stats.l_blocks = l;

    // Sigma-basis extraction on T = Scr^T (coefficientwise transposes).
    let t0 = Instant::now();
    let series_t: Vec<Vec<u16>> = (0..seq_len)
        .map(|k| transpose_bxb(&seq[k * b * b..(k + 1) * b * b], b))
        .collect();
    let basis = crate::sigma::sigma_basis_aug(gf, &series_t, seq_len, b);
    stats.sigma_secs = t0.elapsed().as_secs_f64();
    // Candidate pool: nonzero F-part, delta = f_deg <= l, and remainder
    // degree g_deg <= delta + 1 (zero G-part is the best case). Both the
    // plain and the A-shifted combination are emitted per pool row in
    // pass 2 below.
    let mut pool: Vec<usize> = Vec::new();
    for i in 0..basis.rows() {
        if basis.f_zero[i] {
            continue;
        }
        let delta = basis.f_deg[i];
        if delta > l {
            continue;
        }
        if basis.g_deg[i] != usize::MAX && basis.g_deg[i] > delta + 1 {
            continue;
        }
        pool.push(i);
    }
    if pool.is_empty() {
        return if l * b >= n {
            BlockOutcome::Invertible
        } else {
            BlockOutcome::Failed
        };
    }

    // Pass 2: regenerate W_p = A^p B0 for p = 0..=delta_max+1 and
    // accumulate, for every pool row, both combinations
    //   shift 0:  Z  = sum_{p=0}^{delta}     W_p * F_{delta-p}
    //   shift 1:  AZ = sum_{p=1}^{delta+1}   W_p * F_{delta+1-p}
    // (coefficient of W_p for shift s is F_{delta+s-p}, indices in range
    // by construction).
    let t0 = Instant::now();
    let mut cand: Vec<(Vec<Vec<u16>>, usize, usize)> = Vec::new();
    for &i in pool.iter() {
        let cc = basis.candidate_coeffs(i);
        let delta = basis.f_deg[i];
        cand.push((cc.clone(), delta, 0));
        cand.push((cc, delta, 1));
    }
    stats.candidates = cand.len();
    let nu = cand.len().min(BW_MAX_POOL);
    let cand = &cand[..nu];
    let dmax = cand.iter().map(|&(_, delta, shift)| delta + shift).max().unwrap();
    let mut vmat = vec![0u16; (dmax + 1) * b * nu];
    for (t, (cc, delta, shift)) in cand.iter().enumerate() {
        for p in *shift..=(*delta + *shift) {
            let g = &cc[delta + shift - p];
            for jj in 0..b {
                vmat[(p * b + jj) * nu + t] = g[jj];
            }
        }
    }
    let mut zall = vec![0u16; n * nu];
    let mut w = b0.clone();
    let mut w2 = vec![0u16; n * b];
    for p in 0..=dmax {
        zall.par_chunks_mut(nu).enumerate().for_each(|(i, zrow)| {
            let wrow = &w[i * b..(i + 1) * b];
            for jj in 0..b {
                let wv = wrow[jj];
                if wv == 0 {
                    continue;
                }
                let vrow = &vmat[(p * b + jj) * nu..(p * b + jj + 1) * nu];
                for t in 0..nu {
                    let vv = vrow[t];
                    if vv != 0 {
                        zrow[t] ^= gf.mul(wv, vv);
                    }
                }
            }
        });
        if p < dmax {
            apply_block(&w, &mut w2);
            stats.block_applies += 1;
            std::mem::swap(&mut w, &mut w2);
        }
    }
    // Exact verification of every candidate column, then defect
    // cancellation over the near-misses -- identical logic to
    // block_wiedemann_kernel (duplicated so the Toeplitz path is
    // untouched).
    let mut out_vecs: Vec<Vec<u16>> = Vec::new();
    let mut defected: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
    for t in 0..nu {
        let z: Vec<u16> = (0..n).map(|i| zall[i * nu + t]).collect();
        if z.iter().all(|&v| v == 0) {
            continue;
        }
        let mut az = vec![0u16; n];
        apply1(&z, &mut az);
        if az.iter().all(|&v| v == 0) {
            if out_vecs.len() < max_vecs {
                out_vecs.push(z);
            }
        } else {
            defected.push((z, az));
        }
    }
    if out_vecs.len() < max_vecs && defected.len() >= 2 {
        let m = defected.len();
        let drows: Vec<Vec<u16>> = (0..n)
            .map(|r| (0..m).map(|t| defected[t].1[r]).collect())
            .collect();
        let (_drk, dker) = crate::linalg::kernel_basis(gf, &drows);
        for c in dker.iter() {
            if out_vecs.len() >= max_vecs {
                break;
            }
            let mut zc = vec![0u16; n];
            for (t, &ct) in c.iter().enumerate() {
                if ct != 0 {
                    for r in 0..n {
                        if defected[t].0[r] != 0 {
                            zc[r] ^= gf.mul(ct, defected[t].0[r]);
                        }
                    }
                }
            }
            if zc.iter().all(|&v| v == 0) {
                continue;
            }
            let mut azc = vec![0u16; n];
            apply1(&zc, &mut azc);
            if azc.iter().all(|&v| v == 0) {
                out_vecs.push(zc);
            }
        }
    }
    stats.extract_secs = t0.elapsed().as_secs_f64();
    if out_vecs.is_empty() {
        BlockOutcome::Failed
    } else {
        BlockOutcome::Vectors(out_vecs)
    }
}

/// Sigma-basis extraction with L escalation: identical policy to
/// bw_kernel_escalating, except the budget check is on the sequence
/// length (memory O(terms * b^2)) instead of the Toeplitz width.
#[allow(clippy::too_many_arguments)]
pub fn bw_kernel_escalating_sigma(
    gf: &Gf,
    n: usize,
    deg_est: usize,
    b: usize,
    max_vecs: usize,
    apply_block: &mut dyn FnMut(&[u16], &mut [u16]),
    apply1: &mut dyn FnMut(&[u16], &mut [u16]),
    rng: &mut Rng,
    agg: &mut BlockRunStats,
) -> BlockOutcome {
    let b = b.max(1).min(n.max(1));
    let mut l = (deg_est + b - 1) / b + 2;
    loop {
        if 2 * l > BW_MAX_SIGMA_TERMS {
            // Sigma-basis storage budget exceeded. (When n is huge this
            // also means we cannot certify Invertible on this path;
            // Failed is the honest outcome -- the decision layer treats
            // it heuristically.)
            return BlockOutcome::Failed;
        }
        let mut st = BlockRunStats::default();
        let out = block_wiedemann_kernel_sigma(gf, n, l, b, max_vecs, apply_block, apply1, rng, &mut st);
        agg.block_applies += st.block_applies;
        agg.krylov_secs += st.krylov_secs;
        agg.sigma_secs += st.sigma_secs;
        agg.extract_secs += st.extract_secs;
        agg.nullity = st.nullity;
        agg.l_blocks = st.l_blocks;
        agg.candidates = st.candidates;
        match out {
            BlockOutcome::Failed => {
                if l * b >= n {
                    return BlockOutcome::Failed;
                }
                l += (l / 2).max(4);
            }
            other => return other,
        }
    }
}

impl JetOp {
    /// One sigma-basis block-Wiedemann batch of ker(A) samples for
    /// A = M_tau^T M_tau (tau = excl; excl == cols.len() gives the
    /// full-matrix operator). Sigma version of kernel_batch_bw.
    pub fn kernel_batch_bw_sigma(
        &self,
        gf: &Gf,
        cols: &[Vec<u16>],
        excl: usize,
        block_size: usize,
        max_vecs: usize,
        rng: &mut Rng,
        stats: &mut BWStats,
    ) -> BlockOutcome {
        let nd = self.nd();
        let npts_in = if excl < cols.len() {
            cols.len() - 1
        } else {
            cols.len()
        };
        let deg_est = self.deg_estimate(npts_in);
        let mut mid: Vec<u16> = Vec::new();
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            self.apply_a_block(gf, cols, excl, x, block_size, out, &mut mid);
        };
        let mut mid1: Vec<u16> = Vec::new();
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            self.apply_a(gf, cols, excl, x, out, &mut mid1);
        };
        let mut agg = BlockRunStats::default();
        let out = bw_kernel_escalating_sigma(
            gf,
            nd,
            deg_est,
            block_size,
            max_vecs,
            &mut apply_block,
            &mut apply1,
            rng,
            &mut agg,
        );
        stats.block_applies += agg.block_applies;
        stats.krylov_secs += agg.krylov_secs;
        stats.sigma_secs += agg.sigma_secs;
        stats.extract_secs += agg.extract_secs;
        stats.nullity = agg.nullity;
        stats.l_blocks = agg.l_blocks;
        stats.candidates = agg.candidates;
        out
    }
}

/// Sigma-basis block-Wiedemann GIJS distinguisher with explicit config
/// and stats. Same contract and error model as distinguish_matfree_bw_with:
/// `false` is always certified exactly; `true` is heuristic with
/// false-accept probability ~ q^-cfg.samples.
pub fn distinguish_matfree_bw_sigma_with(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    tau: usize,
    cfg: &BWConfig,
    stats: &mut BWStats,
) -> bool {
    let k = cols[0].len();
    let op = JetOp::new(k, s, d);
    let v = op.eval_vec(gf, &cols[tau]);
    let mut rng = Rng::new(
        cfg.seed
            ^ (tau as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ ((cols.len() as u64) << 32)
            ^ 0x27D4_EB2F_1656_67C5
            ^ 0x5369_676D_615F_7631, // "Sigma_v1": decorrelate from Toeplitz path
    );
    let e_len = cols.len() * op.njets();
    let mut zs: Vec<Vec<u16>> = Vec::new();
    let mut ws: Vec<Vec<u16>> = Vec::new();
    let mut fail_streak = 0usize;
    let max_vecs = cfg.block_size + cfg.samples + 8;
    for _run in 0..cfg.max_runs {
        stats.runs += 1;
        match op.kernel_batch_bw_sigma(gf, cols, tau, cfg.block_size, max_vecs, &mut rng, stats) {
            BlockOutcome::Invertible => {
                stats.no_kernel += 1;
                return true;
            }
            BlockOutcome::Failed => {
                stats.extract_fail += 1;
                fail_streak += 1;
                if zs.is_empty() && fail_streak >= cfg.max_no_kernel_streak {
                    break;
                }
            }
            BlockOutcome::Vectors(batch) => {
                fail_streak = 0;
                for z in batch.iter() {
                    let w = op.matvec_m_vec(gf, cols, tau, z);
                    ws.push(w);
                }
                zs.extend(batch);
                if let Some(dec) = cancel_and_test_bw(
                    gf, &op, cols, tau, &zs, &ws, e_len, &v, cfg.samples, stats,
                ) {
                    return dec;
                }
            }
        }
    }
    true
}

/// Sigma-basis block-Wiedemann matrix-free GIJS distinguisher; same
/// contract as gijs::distinguish / distinguish_matfree_bw.
pub fn distinguish_matfree_bw_sigma(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    tau: usize,
    block_size: usize,
    samples: usize,
) -> bool {
    let mut cfg = BWConfig::default();
    cfg.block_size = block_size;
    cfg.samples = samples;
    let mut stats = BWStats::default();
    distinguish_matfree_bw_sigma_with(gf, cols, s, d, tau, &cfg, &mut stats)
}

// ---------------------------------------------------------------------------
// Stretch: trial-level hold-out reuse on the matrix-free engine
// ---------------------------------------------------------------------------
//
// Criterion (dual of the dense reuse scheme): with M_full the jet matrix
// of ALL n' points (no exclusion), z in ker(M_tau) iff M_full z is
// supported on block tau. Writing S_tau = { (M_full z)|_tau : M_full z
// supported on block tau } one has S_tau = (span{ u_tau : u in
// ker(M_full^T) })^perp, and z . v_tau = (M_full z)_{(tau,0)}. Hence
//     tau is FLAGGED  <=>  exists w in S_tau with w_0 != 0
//                     <=>  e_0 NOT in span{ u_tau : u in ker(M_full^T) }.
// So ONE left-kernel sampling pass over M_full (operator
// B = M_full M_full^T, with the same isotropic-radical cancellation,
// roles of M and M^T swapped) decides all n' hold-outs via n' tiny
// (J x J) span tests -- PROVIDED the samples span the per-block
// restrictions of ker(M_full^T). That proviso is a real limitation,
// measured at (k,s,d) = (12,3,4), n' = 25 (see MATFREE_BW_REPORT.md
// section 5): dim ker(M_full^T) = 910 there, but block-Wiedemann
// sampling on B reaches only a fixed 64-dimensional subspace of it
// (per-run batch rank = block size; the union over runs saturates at
// 64, at both b = 32 and b = 64), so restrictions cap at rank 64.
// When J <= 64 the composition is exact; when J = 91 > 64 every clean
// tau is over-flagged. We sample until the certified pool's rank
// reaches njets + 40 or max_runs is exhausted; callers must check the
// pool rank (reuse_bw_flags_pool) before trusting the flags.

/// Diagnostics for reuse_bw_flags.
#[derive(Default, Debug, Clone)]
pub struct ReuseBwStats {
    pub runs: usize,
    pub block_applies: usize,
    pub certified: usize,
    pub radical_rank: usize,
    pub pool: usize,
}

/// Rank of a set of row vectors (via linalg::kernel_basis).
fn rank_of_rows(gf: &Gf, rows: &[Vec<u16>]) -> usize {
    if rows.is_empty() {
        return 0;
    }
    crate::linalg::kernel_basis(gf, rows).0
}

/// Trial-level flags from one matrix-free left-kernel pass over M_full.
/// Returns flags[tau] == true iff tau is flagged (i.e. the per-tau oracle
/// `distinguish` would return false / "random" there). Sampling-based:
/// if the certified left-kernel pool does not span the true restrictions,
/// a clean tau can be over-flagged; the stats expose the pool size so
/// callers can judge.
pub fn reuse_bw_flags(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    block_size: usize,
    seed: u64,
) -> (Vec<bool>, ReuseBwStats) {
    let (flags, stats, _pool) = reuse_bw_flags_pool(gf, cols, s, d, block_size, seed);
    (flags, stats)
}

/// reuse_bw_flags plus the certified left-kernel pool itself (for
/// diagnostics: pool rank and per-block restriction ranks).
pub fn reuse_bw_flags_pool(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    block_size: usize,
    seed: u64,
) -> (Vec<bool>, ReuseBwStats, Vec<Vec<u16>>) {
    let k = cols[0].len();
    let n = cols.len();
    let op = JetOp::new(k, s, d);
    let nj = op.njets();
    let nd = op.nd();
    let e_full = n * nj;
    let mut stats = ReuseBwStats::default();
    let mut rng = Rng::new(seed ^ ((n as u64) << 32) ^ 0xB10C_6E55_1A2B_3C4D);
    // Operator B = M_full M_full^T on F^{e_full}: apply(u) = M (M^T u).
    let deg_est = {
        let euler_pt = if s >= 2 { binom_usize(k + s - 2, s - 2) } else { 0 };
        let rank_est = e_full.saturating_sub(n * euler_pt).min(nd);
        rank_est.max(1) + 1
    };
    let want_rank = nj + 40;
    let mut zs: Vec<Vec<u16>> = Vec::new(); // ker(B) samples (length e_full)
    let mut ys: Vec<Vec<u16>> = Vec::new(); // y_i = M^T z_i (length nd)
    let mut certified: Vec<Vec<u16>> = Vec::new(); // u in ker(M_full^T)
    let mut pool_rank = 0usize;
    let max_runs = 14usize;
    let max_vecs = block_size + 16;
    for _run in 0..max_runs {
        if pool_rank >= want_rank {
            break;
        }
        stats.runs += 1;
        // Fresh closures per run (scratch buffers are per-run).
        let mut mid_n: Vec<u16> = Vec::new();
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            // x: e_full x b -> M^T x: nd x b -> M (M^T x): e_full x b
            let need_n = nd * block_size;
            if mid_n.len() != need_n {
                mid_n.resize(need_n, 0);
            }
            op.matvec_mt_block(gf, cols, x, block_size, &mut mid_n);
            op.matvec_m_block(gf, cols, n, &mid_n, block_size, out);
        };
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            let mut t = vec![0u16; nd];
            op.matvec_mt(gf, cols, x, &mut t);
            op.matvec_m(gf, cols, n, &t, out);
        };
        let mut agg = BlockRunStats::default();
        let out = bw_kernel_escalating(
            gf,
            e_full,
            deg_est,
            block_size,
            max_vecs,
            &mut apply_block,
            &mut apply1,
            &mut rng,
            &mut agg,
        );
        stats.block_applies += agg.block_applies;
        match out {
            BlockOutcome::Vectors(batch) => {
                for z in batch.iter() {
                    let mut y = vec![0u16; nd];
                    op.matvec_mt(gf, cols, z, &mut y);
                    ys.push(y);
                }
                zs.extend(batch);
                // Cancellation: c in ker[y_1..y_m] (as an nd x m matrix)
                // gives u = sum c_i z_i with M^T u = 0 exactly.
                let m = zs.len();
                stats.pool = m;
                let yrows: Vec<Vec<u16>> = (0..nd)
                    .map(|r| (0..m).map(|i| ys[i][r]).collect())
                    .collect();
                let (_rk, ker) = crate::linalg::kernel_basis(gf, &yrows);
                stats.radical_rank = m - ker.len();
                certified.clear();
                for c in ker.iter() {
                    let mut u = vec![0u16; e_full];
                    for (i, &ci) in c.iter().enumerate() {
                        if ci != 0 {
                            for r in 0..e_full {
                                if zs[i][r] != 0 {
                                    u[r] ^= gf.mul(ci, zs[i][r]);
                                }
                            }
                        }
                    }
                    if u.iter().all(|&x| x == 0) {
                        continue;
                    }
                    // Exact verification: M^T u == 0.
                    let mut mtu = vec![0u16; nd];
                    op.matvec_mt(gf, cols, &u, &mut mtu);
                    if mtu.iter().all(|&x| x == 0) {
                        certified.push(u);
                    }
                }
                stats.certified = certified.len();
                pool_rank = crate::linalg::kernel_basis(gf, &certified).0;
            }
            BlockOutcome::Invertible => break,
            BlockOutcome::Failed => break,
        }
    }
    stats.certified = certified.len();
    // Per-tau span test: flagged <=> e_0 not in span{ u_tau }.
    let mut flags = vec![false; n];
    for tau in 0..n {
        let rows_tau: Vec<Vec<u16>> = certified
            .iter()
            .map(|u| u[tau * nj..(tau + 1) * nj].to_vec())
            .collect();
        let r0 = rank_of_rows(gf, &rows_tau);
        let mut e0 = vec![0u16; nj];
        e0[0] = 1;
        let mut rows_aug = rows_tau.clone();
        rows_aug.push(e0);
        let r1 = rank_of_rows(gf, &rows_aug);
        flags[tau] = r1 > r0; // e_0 outside the span <=> flagged
    }
    (flags, stats, certified)
}

// ---------------------------------------------------------------------------
// Block Wiedemann tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bw_tests {
    use super::*;
    use crate::gijs::distinguish;
    use crate::linalg::shorten;
    use crate::rlce::gen_rlce;

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order as u64;
        (0..n)
            .map(|_| (0..k).map(|_| rng.below(q) as u16).collect())
            .collect()
    }

    fn grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.below(q as u64) as usize;
            if !used[x] {
                used[x] = true;
                xs.push(x as u16);
            }
        }
        xs.into_iter()
            .map(|x| {
                let y = (rng.below((q - 1) as u64) + 1) as u16;
                let mut col = vec![0u16; k];
                let mut xp = 1u16;
                for i in 0..k {
                    col[i] = gf.mul(y, xp);
                    xp = gf.mul(xp, x);
                }
                col
            })
            .collect()
    }

    fn peak_rss_mb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
        -1.0
    }

    #[test]
    fn bw_block_matvec_matches_scalar() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(777001);
        for &(k, s, d) in &[(5usize, 2usize, 3usize), (7, 3, 4), (9, 2, 4)] {
            let n = 11;
            let cols = random_cols(&gf, &mut rng, n, k);
            let op = JetOp::new(k, s, d);
            let nd = op.nd();
            let nj = op.njets();
            let b = 5usize;
            let excl = 3usize;
            // M block vs per-column scalar M
            let x = rand_block(&gf, &mut rng, nd, b);
            let mut out_b = vec![0u16; n * nj * b];
            op.matvec_m_block(&gf, &cols, excl, &x, b, &mut out_b);
            for c in 0..b {
                let xc: Vec<u16> = (0..nd).map(|i| x[i * b + c]).collect();
                let oc = op.matvec_m_vec(&gf, &cols, excl, &xc);
                for r in 0..n * nj {
                    assert_eq!(out_b[r * b + c], oc[r], "M block mismatch ({},{},{})", k, s, d);
                }
            }
            // M^T block vs per-column scalar M^T
            let u = rand_block(&gf, &mut rng, n * nj, b);
            let mut outt_b = vec![0u16; nd * b];
            op.matvec_mt_block(&gf, &cols, &u, b, &mut outt_b);
            for c in 0..b {
                let uc: Vec<u16> = (0..n * nj).map(|i| u[i * b + c]).collect();
                let mut oc = vec![0u16; nd];
                op.matvec_mt(&gf, &cols, &uc, &mut oc);
                for r in 0..nd {
                    assert_eq!(outt_b[r * b + c], oc[r], "MT block mismatch ({},{},{})", k, s, d);
                }
            }
            // A block vs per-column scalar A
            let mut outa_b = vec![0u16; nd * b];
            let mut mid: Vec<u16> = Vec::new();
            op.apply_a_block(&gf, &cols, excl, &x, b, &mut outa_b, &mut mid);
            for c in 0..b {
                let xc: Vec<u16> = (0..nd).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; nd];
                let mut mid1: Vec<u16> = Vec::new();
                op.apply_a(&gf, &cols, excl, &xc, &mut oc, &mut mid1);
                for r in 0..nd {
                    assert_eq!(outa_b[r * b + c], oc[r], "A block mismatch ({},{},{})", k, s, d);
                }
            }
            // colpar control agrees too
            let mut out_cp = vec![0u16; nd * b];
            op.apply_a_block_colpar(&gf, &cols, excl, &x, b, &mut out_cp);
            assert_eq!(out_cp, outa_b, "colpar mismatch ({},{},{})", k, s, d);
        }
    }

    /// Dense planted operator helpers: A stored flat row-major.
    fn dense_apply(amat: &[u16], n: usize, gf: &Gf, x: &[u16], out: &mut [u16]) {
        for i in 0..n {
            let mut acc = 0u16;
            for j in 0..n {
                if amat[i * n + j] != 0 && x[j] != 0 {
                    acc ^= gf.mul(amat[i * n + j], x[j]);
                }
            }
            out[i] = acc;
        }
    }

    #[test]
    fn bw_kernel_on_planted_operators() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(313131);
        // (a) rank-deficient A = X*Y (48x20 * 20x48): ker dim >= 28.
        let (n, r) = (48usize, 20usize);
        let xm: Vec<u16> = (0..n * r).map(|_| rng.below(64) as u16).collect();
        let ym: Vec<u16> = (0..r * n).map(|_| rng.below(64) as u16).collect();
        let mut amat = vec![0u16; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0u16;
                for t in 0..r {
                    acc ^= gf.mul(xm[i * r + t], ym[t * n + j]);
                }
                amat[i * n + j] = acc;
            }
        }
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            // width from slice length
            let b = x.len() / n;
            for c in 0..b {
                let xc: Vec<u16> = (0..n).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; n];
                dense_apply(&amat, n, &gf, &xc, &mut oc);
                for i in 0..n {
                    out[i * b + c] = oc[i];
                }
            }
        };
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            dense_apply(&amat, n, &gf, x, out);
        };
        let mut agg = BlockRunStats::default();
        let out = bw_kernel_escalating(
            &gf,
            n,
            n + 1,
            8,
            24,
            &mut apply_block,
            &mut apply1,
            &mut rng,
            &mut agg,
        );
        match out {
            BlockOutcome::Vectors(vs) => {
                assert!(vs.len() >= 6, "only {} kernel vectors", vs.len());
                for z in vs.iter() {
                    assert!(z.iter().any(|&v| v != 0));
                    let mut az = vec![0u16; n];
                    dense_apply(&amat, n, &gf, z, &mut az);
                    assert!(az.iter().all(|&v| v == 0), "returned vector not in kernel");
                }
                // independence: rank of the returned set
                let (rk, _ker) = crate::linalg::kernel_basis(&gf, &vs);
                assert!(rk >= 6, "returned vectors have rank {} < 6", rk);
                eprintln!(
                    "planted rank-deficient: {} vectors, rank {}, nullity {}, L {}, applies {}",
                    vs.len(),
                    rk,
                    agg.nullity,
                    agg.l_blocks,
                    agg.block_applies
                );
            }
            _ => panic!("expected Vectors on rank-deficient planted operator"),
        }

        // (b) zero operator: every vector is a kernel vector.
        let zn = 24usize;
        let zmat = vec![0u16; zn * zn];
        let mut apply_block0 = |x: &[u16], out: &mut [u16]| {
            let b = x.len() / zn;
            for c in 0..b {
                let xc: Vec<u16> = (0..zn).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; zn];
                dense_apply(&zmat, zn, &gf, &xc, &mut oc);
                for i in 0..zn {
                    out[i * b + c] = oc[i];
                }
            }
        };
        let mut apply10 = |x: &[u16], out: &mut [u16]| {
            dense_apply(&zmat, zn, &gf, x, out);
        };
        let mut agg0 = BlockRunStats::default();
        let out0 = bw_kernel_escalating(
            &gf,
            zn,
            zn + 1,
            8,
            8,
            &mut apply_block0,
            &mut apply10,
            &mut rng,
            &mut agg0,
        );
        match out0 {
            BlockOutcome::Vectors(vs) => assert!(vs.len() >= 4),
            _ => panic!("expected Vectors on zero operator"),
        }

        // (c) invertible operator: never a (nonzero, verified) vector.
        let (m2, _) = (40usize, 0usize);
        let mut amat2 = vec![0u16; m2 * m2];
        let mut full = false;
        while !full {
            for i in 0..m2 * m2 {
                amat2[i] = rng.below(64) as u16;
            }
            let rows: Vec<Vec<u16>> = (0..m2)
                .map(|i| amat2[i * m2..(i + 1) * m2].to_vec())
                .collect();
            let (rk, _) = crate::linalg::kernel_basis(&gf, &rows);
            full = rk == m2;
        }
        let mut apply_block2 = |x: &[u16], out: &mut [u16]| {
            let b = x.len() / m2;
            for c in 0..b {
                let xc: Vec<u16> = (0..m2).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; m2];
                dense_apply(&amat2, m2, &gf, &xc, &mut oc);
                for i in 0..m2 {
                    out[i * b + c] = oc[i];
                }
            }
        };
        let mut apply12 = |x: &[u16], out: &mut [u16]| {
            dense_apply(&amat2, m2, &gf, x, out);
        };
        let mut agg2 = BlockRunStats::default();
        let out2 = bw_kernel_escalating(
            &gf,
            m2,
            m2 + 1,
            8,
            8,
            &mut apply_block2,
            &mut apply12,
            &mut rng,
            &mut agg2,
        );
        match out2 {
            BlockOutcome::Vectors(vs) => {
                for z in vs.iter() {
                    let mut az = vec![0u16; m2];
                    dense_apply(&amat2, m2, &gf, z, &mut az);
                    assert!(az.iter().all(|&v| v == 0), "vector from invertible op not in kernel (impossible unless zero)");
                    assert!(z.iter().all(|&v| v == 0), "nonzero kernel vector for invertible operator!");
                }
                panic!("invertible operator produced kernel vectors");
            }
            BlockOutcome::Invertible => eprintln!("planted invertible: Invertible (expected)"),
            BlockOutcome::Failed => eprintln!("planted invertible: Failed (acceptable)"),
        }
    }

    fn pick_taus(n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = vec![0, 1, 2, 3, n / 2, n - 3, n - 2, n - 1];
        v.retain(|&t| t < n);
        v.sort_unstable();
        v.dedup();
        v
    }

    fn differential_bw(
        gf: &Gf,
        cols: &[Vec<u16>],
        s: usize,
        d: usize,
        taus: &[usize],
        seed: u64,
        block_size: usize,
    ) -> (usize, usize, BWStats) {
        let mut cfg = BWConfig::default();
        cfg.seed = seed;
        cfg.block_size = block_size;
        let mut agg = BWStats::default();
        let mut mism = 0usize;
        for &tau in taus {
            let want = distinguish(gf, cols, s, d, tau);
            let mut st = BWStats::default();
            let got = distinguish_matfree_bw_with(gf, cols, s, d, tau, &cfg, &mut st);
            agg.runs += st.runs;
            agg.block_applies += st.block_applies;
            agg.extract_fail += st.extract_fail;
            agg.verify_fail += st.verify_fail;
            agg.verified += st.verified;
            if got != want {
                mism += 1;
                eprintln!(
                    "BW MISMATCH tau={} oracle={} bw={} stats={:?}",
                    tau, want, got, st
                );
            }
        }
        (taus.len(), mism, agg)
    }

    #[test]
    fn bw_differential_random_codes() {
        let cases: [(u32, usize, usize, usize, usize); 4] = [
            (6, 36, 14, 2, 4),
            (6, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
            (7, 36, 14, 2, 4),
        ];
        let mut rng = Rng::new(4242);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = random_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential_bw(&gf, &cols, s, d, &taus, 5000 + idx as u64, 32);
            eprintln!(
                "BW random case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "BW random case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    #[test]
    fn bw_differential_grs_codes() {
        let cases: [(u32, usize, usize, usize, usize); 3] = [
            (6, 40, 14, 2, 4),
            (7, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
        ];
        let mut rng = Rng::new(777);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = grs_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential_bw(&gf, &cols, s, d, &taus, 6000 + idx as u64, 32);
            eprintln!(
                "BW grs case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "BW grs case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    #[test]
    fn bw_differential_rlce_shortened() {
        let cases: [(u32, usize, usize, usize, usize, usize, usize); 3] = [
            (7, 44, 30, 8, 41, 2, 3),
            (7, 34, 20, 4, 32, 2, 4),
            (6, 26, 16, 3, 25, 3, 4),
        ];
        for (idx, &(m, n, k, w, nprime, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let mut rng = Rng::new(10_000 + idx as u64);
            let key = gen_rlce(&gf, &mut rng, n, k, w);
            let n_hit = key.n_pub - nprime;
            let hit = rng.subset(key.n_pub, n_hit);
            let (short_cols, _kept) = shorten(&gf, &key.cols, &hit);
            let kp = short_cols[0].len();
            let taus = pick_taus(nprime);
            let (dec, mism, agg) =
                differential_bw(&gf, &short_cols, s, d, &taus, 7000 + idx as u64, 32);
            eprintln!(
                "BW rlce case {} (q={},n={},k={},w={} -> n'={},k'={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                w,
                nprime,
                kp,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "BW rlce case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    /// Same grid, small block size (b=8) to exercise a different geometry.
    #[test]
    fn bw_differential_small_block() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(4242);
        let _c0 = random_cols(&gf, &mut rng, 36, 14);
        let cols = random_cols(&gf, &mut rng, 41, 19);
        let taus = pick_taus(41);
        let (dec, mism, agg) = differential_bw(&gf, &cols, 2, 3, &taus, 8000, 8);
        eprintln!(
            "BW small-block (q=64,n=41,k=19,s=2,d=3,b=8): {} decisions, {} mismatches, agg={:?}",
            dec, mism, agg
        );
        assert_eq!(mism, 0);
    }

    /// Smoke test for the public wrapper with the deliverable signature.
    #[test]
    fn bw_public_wrapper_smoke() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(4242);
        let cols = random_cols(&gf, &mut rng, 36, 14);
        for tau in [0usize, 5] {
            let want = distinguish(&gf, &cols, 2, 4, tau);
            let got = distinguish_matfree_bw(&gf, &cols, 2, 4, tau, 32, 4);
            assert_eq!(got, want, "wrapper mismatch at tau={}", tau);
        }
    }

    /// Stretch: trial-level reuse composition vs per-tau oracle flags.
    /// Case (14,2,4): J = 15 is below the Krylov-reachable kernel
    /// ceiling (64, see bw_reuse_case1_krylov_ceiling), and the
    /// composition is exact on all 32 taus.
    #[test]
    fn bw_reuse_composition_rlce() {
        let (m, n, k, w, nprime, s, d) = (7u32, 34usize, 20usize, 4usize, 32usize, 2usize, 4usize);
        let gf = Gf::new(m);
        let mut rng = Rng::new(10_000);
        let key = gen_rlce(&gf, &mut rng, n, k, w);
        let n_hit = key.n_pub - nprime;
        let hit = rng.subset(key.n_pub, n_hit);
        let (short_cols, _kept) = shorten(&gf, &key.cols, &hit);
        let (flags, st) = reuse_bw_flags(&gf, &short_cols, s, d, 32, 9000);
        let mut mism = 0usize;
        for tau in 0..nprime {
            let oracle_flag = !distinguish(&gf, &short_cols, s, d, tau);
            if flags[tau] != oracle_flag {
                mism += 1;
                eprintln!(
                    "REUSE MISMATCH tau={} reuse_flag={} oracle_flag={} stats={:?}",
                    tau, flags[tau], oracle_flag, st
                );
            }
        }
        eprintln!(
            "BW reuse (n'={},s={},d={}): {} taus, {} mismatches, stats={:?}",
            nprime, s, d, nprime, mism, st
        );
        assert_eq!(mism, 0, "reuse composition had mismatches");
    }

    /// Documented limitation (MATFREE_BW_REPORT.md section 5): at
    /// (12,3,4), n' = 25, dim ker(M_full^T) = 910 but block-Wiedemann
    /// sampling on B = M M^T reaches only a 64-dimensional subspace of
    /// it (batch rank = block size per run; union saturates at 64 for
    /// b = 32 and b = 64). Since J = 91 > 64, restrictions cap at 64,
    /// e_0 is never in the sampled span, and every (structured, oracle
    /// unflagged) tau is over-flagged. This test pins that signature so
    /// a future sampling improvement shows up here as a change.
    #[test]
    fn bw_reuse_case1_krylov_ceiling() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(10_001);
        let key = gen_rlce(&gf, &mut rng, 26, 16, 3);
        let hit = rng.subset(key.n_pub, key.n_pub - 25);
        let (short_cols, _kept) = shorten(&gf, &key.cols, &hit);
        let (flags, st, pool) = reuse_bw_flags_pool(&gf, &short_cols, 3, 4, 32, 9001);
        let pool_rank = crate::linalg::kernel_basis(&gf, &pool).0;
        eprintln!(
            "BW reuse ceiling: pool_rank={} (J=91) stats={:?}",
            pool_rank, st
        );
        assert!(
            pool_rank <= 64 && pool_rank < 91,
            "expected the measured Krylov ceiling (<=64 < J=91), got pool_rank={}",
            pool_rank
        );
        assert!(
            flags.iter().all(|&f| f),
            "expected the over-flagging signature (all taus flagged)"
        );
        // The instance is structured: the dense oracle unflags these.
        for tau in [0usize, 1, 12, 24] {
            assert!(
                distinguish(&gf, &short_cols, 3, 4, tau),
                "oracle should call tau={} structured",
                tau
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Sigma-basis block Wiedemann tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod sigma_tests {
    use super::*;
    use crate::gijs::distinguish;
    use crate::linalg::shorten;
    use crate::rlce::gen_rlce;

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order as u64;
        (0..n)
            .map(|_| (0..k).map(|_| rng.below(q) as u16).collect())
            .collect()
    }

    fn grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.below(q as u64) as usize;
            if !used[x] {
                used[x] = true;
                xs.push(x as u16);
            }
        }
        xs.into_iter()
            .map(|x| {
                let y = (rng.below((q - 1) as u64) + 1) as u16;
                let mut col = vec![0u16; k];
                let mut xp = 1u16;
                for i in 0..k {
                    col[i] = gf.mul(y, xp);
                    xp = gf.mul(xp, x);
                }
                col
            })
            .collect()
    }

    fn pick_taus(n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = vec![0, 1, 2, 3, n / 2, n - 3, n - 2, n - 1];
        v.retain(|&t| t < n);
        v.sort_unstable();
        v.dedup();
        v
    }

    fn differential_sigma(
        gf: &Gf,
        cols: &[Vec<u16>],
        s: usize,
        d: usize,
        taus: &[usize],
        seed: u64,
        block_size: usize,
    ) -> (usize, usize, BWStats) {
        let mut cfg = BWConfig::default();
        cfg.seed = seed;
        cfg.block_size = block_size;
        let mut agg = BWStats::default();
        let mut mism = 0usize;
        for &tau in taus {
            let want = distinguish(gf, cols, s, d, tau);
            let mut st = BWStats::default();
            let got = distinguish_matfree_bw_sigma_with(gf, cols, s, d, tau, &cfg, &mut st);
            agg.runs += st.runs;
            agg.block_applies += st.block_applies;
            agg.extract_fail += st.extract_fail;
            agg.verify_fail += st.verify_fail;
            agg.verified += st.verified;
            if got != want {
                mism += 1;
                eprintln!(
                    "SIGMA MISMATCH tau={} oracle={} sigma={} stats={:?}",
                    tau, want, got, st
                );
            }
        }
        (taus.len(), mism, agg)
    }

    #[test]
    fn sigma_differential_random_codes() {
        let cases: [(u32, usize, usize, usize, usize); 4] = [
            (6, 36, 14, 2, 4),
            (6, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
            (7, 36, 14, 2, 4),
        ];
        let mut rng = Rng::new(4242);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = random_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential_sigma(&gf, &cols, s, d, &taus, 5000 + idx as u64, 32);
            eprintln!(
                "SIGMA random case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "SIGMA random case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    #[test]
    fn sigma_differential_grs_codes() {
        let cases: [(u32, usize, usize, usize, usize); 3] = [
            (6, 40, 14, 2, 4),
            (7, 41, 19, 2, 3),
            (6, 30, 12, 3, 4),
        ];
        let mut rng = Rng::new(777);
        for (idx, &(m, n, k, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let cols = grs_cols(&gf, &mut rng, n, k);
            let taus = pick_taus(n);
            let (dec, mism, agg) = differential_sigma(&gf, &cols, s, d, &taus, 6000 + idx as u64, 32);
            eprintln!(
                "SIGMA grs case {} (q={},n={},k={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "SIGMA grs case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    #[test]
    fn sigma_differential_rlce_shortened() {
        let cases: [(u32, usize, usize, usize, usize, usize, usize); 3] = [
            (7, 44, 30, 8, 41, 2, 3),
            (7, 34, 20, 4, 32, 2, 4),
            (6, 26, 16, 3, 25, 3, 4),
        ];
        for (idx, &(m, n, k, w, nprime, s, d)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let mut rng = Rng::new(10_000 + idx as u64);
            let key = gen_rlce(&gf, &mut rng, n, k, w);
            let n_hit = key.n_pub - nprime;
            let hit = rng.subset(key.n_pub, n_hit);
            let (short_cols, _kept) = shorten(&gf, &key.cols, &hit);
            let kp = short_cols[0].len();
            let taus = pick_taus(nprime);
            let (dec, mism, agg) =
                differential_sigma(&gf, &short_cols, s, d, &taus, 7000 + idx as u64, 32);
            eprintln!(
                "SIGMA rlce case {} (q={},n={},k={},w={} -> n'={},k'={},s={},d={}): {} decisions, {} mismatches, agg={:?}",
                idx,
                1 << m,
                n,
                k,
                w,
                nprime,
                kp,
                s,
                d,
                dec,
                mism,
                agg
            );
            assert_eq!(mism, 0, "SIGMA rlce case {} had mismatches", idx);
            assert_eq!(agg.verify_fail, 0);
        }
    }

    /// Same grid, small block size (b=8), mirroring the Toeplitz suite.
    #[test]
    fn sigma_differential_small_block() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(4242);
        let _c0 = random_cols(&gf, &mut rng, 36, 14);
        let cols = random_cols(&gf, &mut rng, 41, 19);
        let taus = pick_taus(41);
        let (dec, mism, agg) = differential_sigma(&gf, &cols, 2, 3, &taus, 8000, 8);
        eprintln!(
            "SIGMA small-block (q=64,n=41,k=19,s=2,d=3,b=8): {} decisions, {} mismatches, agg={:?}",
            dec, mism, agg
        );
        assert_eq!(mism, 0);
    }

    /// Smoke test for the public wrapper with the deliverable signature.
    #[test]
    fn sigma_public_wrapper_smoke() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(4242);
        let cols = random_cols(&gf, &mut rng, 36, 14);
        for tau in [0usize, 5] {
            let want = distinguish(&gf, &cols, 2, 4, tau);
            let got = distinguish_matfree_bw_sigma(&gf, &cols, 2, 4, tau, 32, 4);
            assert_eq!(got, want, "sigma wrapper mismatch at tau={}", tau);
        }
    }

    /// Dense planted operators through bw_kernel_escalating_sigma,
    /// mirroring bw_kernel_on_planted_operators.
    #[test]
    fn sigma_kernel_on_planted_operators() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(313131);
        let (n, r) = (48usize, 20usize);
        let xm: Vec<u16> = (0..n * r).map(|_| rng.below(64) as u16).collect();
        let ym: Vec<u16> = (0..r * n).map(|_| rng.below(64) as u16).collect();
        let mut amat = vec![0u16; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0u16;
                for t in 0..r {
                    acc ^= gf.mul(xm[i * r + t], ym[t * n + j]);
                }
                amat[i * n + j] = acc;
            }
        }
        let dense_apply = |amat: &[u16], x: &[u16], out: &mut [u16]| {
            for i in 0..n {
                let mut acc = 0u16;
                for j in 0..n {
                    if amat[i * n + j] != 0 && x[j] != 0 {
                        acc ^= gf.mul(amat[i * n + j], x[j]);
                    }
                }
                out[i] = acc;
            }
        };
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            let b = x.len() / n;
            for c in 0..b {
                let xc: Vec<u16> = (0..n).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; n];
                dense_apply(&amat, &xc, &mut oc);
                for i in 0..n {
                    out[i * b + c] = oc[i];
                }
            }
        };
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            dense_apply(&amat, x, out);
        };
        let mut agg = BlockRunStats::default();
        let out = bw_kernel_escalating_sigma(
            &gf,
            n,
            n + 1,
            8,
            24,
            &mut apply_block,
            &mut apply1,
            &mut rng,
            &mut agg,
        );
        match out {
            BlockOutcome::Vectors(vs) => {
                assert!(vs.len() >= 3, "only {} kernel vectors", vs.len());
                for z in vs.iter() {
                    assert!(z.iter().any(|&v| v != 0));
                    let mut az = vec![0u16; n];
                    dense_apply(&amat, z, &mut az);
                    assert!(az.iter().all(|&v| v == 0), "returned vector not in kernel");
                }
                let (rk, _ker) = crate::linalg::kernel_basis(&gf, &vs);
                assert!(rk >= 3, "returned vectors have rank {} < 3", rk);
                eprintln!(
                    "sigma planted rank-deficient: {} vectors, rank {}, L {}, applies {}, sigma={:.3}s",
                    vs.len(),
                    rk,
                    agg.l_blocks,
                    agg.block_applies,
                    agg.sigma_secs
                );
            }
            _ => panic!("expected Vectors on rank-deficient planted operator"),
        }

        // Invertible operator: never a nonzero verified vector.
        let (m2, _) = (40usize, 0usize);
        let mut amat2 = vec![0u16; m2 * m2];
        let mut full = false;
        while !full {
            for i in 0..m2 * m2 {
                amat2[i] = rng.below(64) as u16;
            }
            let rows: Vec<Vec<u16>> = (0..m2)
                .map(|i| amat2[i * m2..(i + 1) * m2].to_vec())
                .collect();
            let (rk, _) = crate::linalg::kernel_basis(&gf, &rows);
            full = rk == m2;
        }
        let dense_apply2 = |x: &[u16], out: &mut [u16]| {
            for i in 0..m2 {
                let mut acc = 0u16;
                for j in 0..m2 {
                    if amat2[i * m2 + j] != 0 && x[j] != 0 {
                        acc ^= gf.mul(amat2[i * m2 + j], x[j]);
                    }
                }
                out[i] = acc;
            }
        };
        let mut apply_block2 = |x: &[u16], out: &mut [u16]| {
            let b = x.len() / m2;
            for c in 0..b {
                let xc: Vec<u16> = (0..m2).map(|i| x[i * b + c]).collect();
                let mut oc = vec![0u16; m2];
                dense_apply2(&xc, &mut oc);
                for i in 0..m2 {
                    out[i * b + c] = oc[i];
                }
            }
        };
        let mut apply12 = |x: &[u16], out: &mut [u16]| {
            dense_apply2(x, out);
        };
        let mut agg2 = BlockRunStats::default();
        let out2 = bw_kernel_escalating_sigma(
            &gf,
            m2,
            m2 + 1,
            8,
            8,
            &mut apply_block2,
            &mut apply12,
            &mut rng,
            &mut agg2,
        );
        match out2 {
            BlockOutcome::Vectors(vs) => {
                for z in vs.iter() {
                    assert!(
                        z.iter().all(|&v| v == 0),
                        "nonzero kernel vector for invertible operator!"
                    );
                }
                panic!("invertible operator produced kernel vectors");
            }
            BlockOutcome::Invertible => eprintln!("sigma planted invertible: Invertible"),
            BlockOutcome::Failed => eprintln!("sigma planted invertible: Failed (acceptable)"),
        }
    }

    /// Head-to-head against the Toeplitz extraction on the SAME Krylov
    /// sequence (same seed => same B0, U draws and applies): every sigma
    /// candidate with delta <= L-1 and the Pade property, reversed
    /// (f_j = F_{delta-j}), must lie in the block-Toeplitz nullspace that
    /// block_wiedemann_kernel computes. Exact linear algebra, no oracle.
    #[test]
    fn sigma_candidates_satisfy_toeplitz_window() {
        let cases: [(u32, usize, usize, usize, usize, bool); 2] = [
            (6, 41, 19, 2, 3, false),
            (6, 30, 12, 3, 4, true),
        ];
        let mut total_shifted = 0usize;
        for (idx, &(m, n, k, s, d, grs)) in cases.iter().enumerate() {
            let gf = Gf::new(m);
            let mut rng = Rng::new(4242);
            let cols = if grs {
                // match the grs suite's rng consumption pattern loosely;
                // any fixed instance works for this algebraic check
                let mut r2 = Rng::new(777 + idx as u64);
                grs_cols(&gf, &mut r2, n, k)
            } else {
                random_cols(&gf, &mut rng, n, k)
            };
            let tau = 0usize;
            let op = JetOp::new(k, s, d);
            let nd = op.nd();
            let b = 32usize;
            // L sizing identical to kernel_batch_bw (deg_estimate).
            let deg_est = op.deg_estimate(n - 1);
            let l = (deg_est + b - 1) / b + 2;
            // Krylov pass 1 replica.
            let mut rng2 = Rng::new(777_000 + idx as u64);
            let b0 = rand_block(&gf, &mut rng2, nd, b);
            let u = rand_block(&gf, &mut rng2, nd, b);
            let seq_len = 2 * l;
            let mut seq = vec![0u16; seq_len * b * b];
            let mut w = b0.clone();
            let mut w2 = vec![0u16; nd * b];
            let mut mid: Vec<u16> = Vec::new();
            for kk in 0..seq_len {
                {
                    let sk = &mut seq[kk * b * b..(kk + 1) * b * b];
                    for i in 0..nd {
                        let urow = &u[i * b..(i + 1) * b];
                        let wrow = &w[i * b..(i + 1) * b];
                        if wrow.iter().all(|&v| v == 0) {
                            continue;
                        }
                        for r in 0..b {
                            let uv = urow[r];
                            if uv == 0 {
                                continue;
                            }
                            for c in 0..b {
                                let wv = wrow[c];
                                if wv != 0 {
                                    sk[r * b + c] ^= gf.mul(uv, wv);
                                }
                            }
                        }
                    }
                }
                if kk + 1 < seq_len {
                    op.apply_a_block(&gf, &cols, tau, &w, b, &mut w2, &mut mid);
                    std::mem::swap(&mut w, &mut w2);
                }
            }
            // Toeplitz matrix exactly as in block_wiedemann_kernel.
            let rows = l * b;
            let colsn = l * b;
            let mut tmat = vec![0u16; rows * colsn];
            for i in 1..=l {
                for j in 0..l {
                    let sij = &seq[(i + j) * b * b..(i + j + 1) * b * b];
                    for r in 0..b {
                        let dst = ((i - 1) * b + r) * colsn + j * b;
                        tmat[dst..dst + b].copy_from_slice(&sij[r * b..(r + 1) * b]);
                    }
                }
            }
            let toep_basis = nullspace_flat(&gf, &tmat, rows, colsn);
            // Sigma basis on the transposed sequence.
            let series_t: Vec<Vec<u16>> = (0..seq_len)
                .map(|kk| transpose_bxb(&seq[kk * b * b..(kk + 1) * b * b], b))
                .collect();
            let sb = crate::sigma::sigma_basis_aug(&gf, &series_t, seq_len, b);
            eprintln!(
                "head-to-head case {}: sigma f_deg={:?} g_deg={:?}",
                idx, sb.f_deg, sb.g_deg
            );
            let mut checked_plain = 0usize;
            let mut checked_shifted = 0usize;
            for i in 0..sb.rows() {
                if sb.f_zero[i] {
                    continue;
                }
                let delta = sb.f_deg[i];
                let cc = sb.candidate_coeffs(i);
                // f = reversal: f_j = F_{delta-j}, zero for j > delta.
                let mut f = vec![0u16; l * b];
                if delta <= l - 1 {
                    for j in 0..=delta {
                        let g = &cc[delta - j];
                        f[j * b..(j + 1) * b].copy_from_slice(g);
                    }
                }
                let check_null = |f: &[u16], kind: &str| {
                    for r in 0..rows {
                        let mut acc = 0u16;
                        for c in 0..colsn {
                            let tv = tmat[r * colsn + c];
                            if tv != 0 && f[c] != 0 {
                                acc ^= gf.mul(tv, f[c]);
                            }
                        }
                        assert_eq!(
                            acc, 0,
                            "case {}: sigma candidate row {} (delta {}, {}) not in Toeplitz nullspace (row {})",
                            idx, i, delta, kind, r
                        );
                    }
                };
                // Plain reversal: valid when deg G <= delta (window
                // i+delta+1 <= sigma-1 needs delta <= l-1).
                if delta <= l - 1 && (sb.g_deg[i] == usize::MAX || sb.g_deg[i] <= delta) {
                    check_null(&f, "plain");
                    checked_plain += 1;
                }
                // Shifted reversal x*f: valid when deg G <= delta + 1
                // (needs delta + 1 <= l - 1 for the layout).
                if delta <= l - 2 && (sb.g_deg[i] == usize::MAX || sb.g_deg[i] <= delta + 1) {
                    let mut fs = vec![0u16; l * b];
                    for j in 0..=delta {
                        fs[(j + 1) * b..(j + 2) * b].copy_from_slice(&f[j * b..(j + 1) * b]);
                    }
                    check_null(&fs, "shifted");
                    checked_shifted += 1;
                }
            }
            eprintln!(
                "head-to-head case {} (grs={}): L={}, Lb={}, toeplitz nullity={}, sigma candidates checked: plain={} shifted={}",
                idx,
                grs,
                l,
                l * b,
                toep_basis.len(),
                checked_plain,
                checked_shifted
            );
            assert!(
                checked_plain + checked_shifted >= 1,
                "case {}: no sigma candidates to check",
                idx
            );
            total_shifted += checked_shifted;
        }
        // The shifted (deg G = deg F + 1) extraction path must actually
        // have been exercised (it is the GRS generator family).
        assert!(
            total_shifted >= 1,
            "no shifted sigma candidates checked in either case"
        );
    }
}

// ---------------------------------------------------------------------------
// Block Wiedemann benchmarks (ignored tests; run one per process)
// ---------------------------------------------------------------------------
//
//   cargo test --release --offline --bin rlce-attack -- --ignored <name> --nocapture
//
// Each benchmark prints time + peak RSS (VmHWM) for this process; run in a
// fresh process per measurement for clean numbers.

#[cfg(test)]
mod bw_benches {
    use super::*;
    use crate::gijs::distinguish;
    use std::time::Instant;

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order as u64;
        (0..n)
            .map(|_| (0..k).map(|_| rng.below(q) as u16).collect())
            .collect()
    }

    fn grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.below(q as u64) as usize;
            if !used[x] {
                used[x] = true;
                xs.push(x as u16);
            }
        }
        xs.into_iter()
            .map(|x| {
                let y = (rng.below((q - 1) as u64) + 1) as u16;
                let mut col = vec![0u16; k];
                let mut xp = 1u16;
                for i in 0..k {
                    col[i] = gf.mul(y, xp);
                    xp = gf.mul(xp, x);
                }
                col
            })
            .collect()
    }

    fn peak_rss_mb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
        -1.0
    }

    fn median(mut xs: Vec<f64>) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    /// Bench A, random instance: q=256, n'=32, k'=12, (s,d)=(3,5), tau=0.
    #[test]
    #[ignore]
    fn bench_bw_a_random() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(31337);
        let cols = random_cols(&gf, &mut rng, 32, 12);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 424242;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "BW A-random: decision={} oracle={} time={:.2}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Bench A, pure GRS instance: same parameters, Rng::new(27182).
    #[test]
    #[ignore]
    fn bench_bw_a_grs() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(27182);
        let cols = grs_cols(&gf, &mut rng, 32, 12);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 434343;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "BW A-GRS: decision={} oracle={} time={:.2}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Larger point: q=256, n'=40, k'=13, (3,5), tau=0. Scalar Wiedemann
    /// did not finish one sample in ~7 min CPU here (MATFREE_REPORT).
    #[test]
    #[ignore]
    fn bench_bw_larger_40() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(987654);
        let cols = random_cols(&gf, &mut rng, 40, 13);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 515151;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "BW larger40: decision={} oracle={} time={:.1}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Apply-variant micro-benchmark at bench A size: point-parallel block
    /// apply vs 1-thread block apply vs column-parallel scalar applies.
    #[test]
    #[ignore]
    fn bench_bw_apply_variants() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(31337);
        let cols = random_cols(&gf, &mut rng, 32, 12);
        let op = JetOp::new(12, 3, 5);
        let nd = op.nd();
        let b = 32usize;
        let x = rand_block(&gf, &mut rng, nd, b);
        let mut out = vec![0u16; nd * b];
        let mut mid: Vec<u16> = Vec::new();
        let reps = 10;
        // warm
        op.apply_a_block(&gf, &cols, 0, &x, b, &mut out, &mut mid);
        let t = Instant::now();
        for _ in 0..reps {
            op.apply_a_block(&gf, &cols, 0, &x, b, &mut out, &mut mid);
        }
        let block_par = t.elapsed().as_secs_f64() / reps as f64;
        let out_par = out.clone();
        let pool1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let t = Instant::now();
        pool1.install(|| {
            for _ in 0..reps {
                op.apply_a_block(&gf, &cols, 0, &x, b, &mut out, &mut mid);
            }
        });
        let block_1t = t.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(out, out_par);
        let t = Instant::now();
        for _ in 0..reps {
            op.apply_a_block_colpar(&gf, &cols, 0, &x, b, &mut out);
        }
        let colpar = t.elapsed().as_secs_f64() / reps as f64;
        assert_eq!(out, out_par);
        eprintln!(
            "BW apply variants (bench A size, b=32, A-apply): point-parallel={:.3}s 1-thread={:.3}s colpar={:.3}s rss={:.1}MB",
            block_par,
            block_1t,
            colpar,
            peak_rss_mb()
        );
    }

    /// Paper-scale single hold-out (3,5): JetOp build + timed b=32 block
    /// applies + RSS; extrapolation printed with its formula.
    #[test]
    #[ignore]
    fn bench_bw_flagship_apply() {
        let t0 = Instant::now();
        let gf = Gf::new(10);
        let mut rng = Rng::new(20261020);
        let n = 348usize;
        let cols: Vec<Vec<u16>> = (0..n)
            .map(|_| (0..28).map(|_| rng.below(1024) as u16).collect())
            .collect();
        let op = JetOp::new(28, 3, 5);
        let nd = op.nd();
        let b = 32usize;
        let x = rand_block(&gf, &mut rng, nd, b);
        let mut out = vec![0u16; nd * b];
        let mut mid: Vec<u16> = Vec::new();
        // warm-up + timing (2 timed applies)
        op.apply_a_block(&gf, &cols, 0, &x, b, &mut out, &mut mid);
        let mut times = Vec::new();
        for _ in 0..2 {
            let t = Instant::now();
            op.apply_a_block(&gf, &cols, 0, &x, b, &mut out, &mut mid);
            times.push(t.elapsed().as_secs_f64());
        }
        let t_apply = median(times);
        let build = t0.elapsed().as_secs_f64();
        // Extrapolation inputs: E_tau = 347 * C(30,2) = 150,945;
        // deg_est = E_tau - 347 * C(29,1) + 1 = 140,883; L = ceil(deg/b)+2.
        let e_tau: usize = 347 * 435;
        let deg_est: usize = e_tau - 347 * 29 + 1;
        let l = (deg_est + b - 1) / b + 2;
        let applies_per_decision = 3 * l;
        eprintln!(
            "BW flagship (3,5): JetOp build+warm={:.1}s, block A-apply={:.2}s, RSS={:.1}MB, pairs/pt={}, E_tau={}, deg_est={}, L={}, ~{} block applies/decision-block-run => {:.1} days of applies per run (Toeplitz EXTRA, infeasible at Lb={})",
            build,
            t_apply,
            peak_rss_mb(),
            op.pairs.len(),
            e_tau,
            deg_est,
            l,
            applies_per_decision,
            applies_per_decision as f64 * t_apply / 86400.0,
            l * b
        );
    }
}

// ---------------------------------------------------------------------------
// Sigma-basis benchmarks (ignored tests; run one per process)
// ---------------------------------------------------------------------------
//
//   cargo test --release --offline --bin rlce-attack -- --ignored bench_sigma --nocapture
//
// Each benchmark prints time + peak RSS (VmHWM) for this process; run in a
// fresh process per measurement for clean numbers.

#[cfg(test)]
mod sigma_benches {
    use super::*;
    use crate::gijs::distinguish;
    use std::time::Instant;

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order as u64;
        (0..n)
            .map(|_| (0..k).map(|_| rng.below(q) as u16).collect())
            .collect()
    }

    fn grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.below(q as u64) as usize;
            if !used[x] {
                used[x] = true;
                xs.push(x as u16);
            }
        }
        xs.into_iter()
            .map(|x| {
                let y = (rng.below((q - 1) as u64) + 1) as u16;
                let mut col = vec![0u16; k];
                let mut xp = 1u16;
                for i in 0..k {
                    col[i] = gf.mul(y, xp);
                    xp = gf.mul(xp, x);
                }
                col
            })
            .collect()
    }

    fn peak_rss_mb() -> f64 {
        let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: f64 = rest
                    .trim()
                    .trim_end_matches("kB")
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
        -1.0
    }

    /// Bench A, random instance: q=256, n'=32, k'=12, (s,d)=(3,5), tau=0.
    #[test]
    #[ignore]
    fn bench_sigma_a_random() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(31337);
        let cols = random_cols(&gf, &mut rng, 32, 12);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 424242;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_sigma_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "SIGMA A-random: decision={} oracle={} time={:.2}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Bench A, pure GRS instance: same parameters, Rng::new(27182).
    #[test]
    #[ignore]
    fn bench_sigma_a_grs() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(27182);
        let cols = grs_cols(&gf, &mut rng, 32, 12);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 434343;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_sigma_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "SIGMA A-GRS: decision={} oracle={} time={:.2}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Larger point: q=256, n'=40, k'=13, (3,5), tau=0 (same instance as
    /// bench_bw_larger_40: Rng::new(987654), cfg seed 515151).
    #[test]
    #[ignore]
    fn bench_sigma_larger_40() {
        let gf = Gf::new(8);
        let mut rng = Rng::new(987654);
        let cols = random_cols(&gf, &mut rng, 40, 13);
        let (s, d, tau) = (3usize, 5usize, 0usize);
        let mut cfg = BWConfig::default();
        cfg.seed = 515151;
        let mut st = BWStats::default();
        let t = Instant::now();
        let dec = distinguish_matfree_bw_sigma_with(&gf, &cols, s, d, tau, &cfg, &mut st);
        let secs = t.elapsed().as_secs_f64();
        let oracle = distinguish(&gf, &cols, s, d, tau);
        eprintln!(
            "SIGMA larger40: decision={} oracle={} time={:.1}s rss={:.1}MB stats={:?}",
            dec,
            oracle,
            secs,
            peak_rss_mb(),
            st
        );
        assert_eq!(dec, oracle);
    }

    /// Flagship dims (k'=28, (3,5), q=1024, n'=348): run ONE sigma-path
    /// block extraction with a small forced L (the full decision needs
    /// L = 4,405 and ~13k block applies -- days; see MATFREE_BW_REPORT).
    /// The point is the memory profile: JetOp + Krylov + sigma-basis at
    /// flagship dims must stay ~100-150 MB, and the sigma working set for
    /// the full L is O(terms * b^2), NOT (L*b)^2 = 39.7 GB.
    #[test]
    #[ignore]
    fn bench_sigma_flagship_prefix() {
        let t0 = Instant::now();
        let gf = Gf::new(10);
        let mut rng = Rng::new(20261020);
        let n = 348usize;
        let cols: Vec<Vec<u16>> = (0..n)
            .map(|_| (0..28).map(|_| rng.below(1024) as u16).collect())
            .collect();
        let op = JetOp::new(28, 3, 5);
        let nd = op.nd();
        let b = 32usize;
        let l_forced = 4usize;
        let mut mid: Vec<u16> = Vec::new();
        let mut apply_block = |x: &[u16], out: &mut [u16]| {
            op.apply_a_block(&gf, &cols, 0, x, b, out, &mut mid);
        };
        let mut mid1: Vec<u16> = Vec::new();
        let mut apply1 = |x: &[u16], out: &mut [u16]| {
            op.apply_a(&gf, &cols, 0, x, out, &mut mid1);
        };
        let mut st = BlockRunStats::default();
        let mut rng2 = Rng::new(1234567);
        let out = block_wiedemann_kernel_sigma(
            &gf,
            nd,
            l_forced,
            b,
            8,
            &mut apply_block,
            &mut apply1,
            &mut rng2,
            &mut st,
        );
        let secs = t0.elapsed().as_secs_f64();
        // Storage accounting at the FULL flagship L (formulas):
        let l_full = 4405usize;
        let terms = 2 * l_full;
        let series_mb = (terms * b * b * 2) as f64 / 1e6;
        let basis_mb = ((terms + 1) * (2 * b) * (2 * b) * 2) as f64 / 1e6;
        let resid_mb = ((2 * b) * terms * b * 2) as f64 / 1e6;
        let toep_gb = ((l_full * b) as f64).powi(2) * 2.0 / 1e9;
        eprintln!(
            "SIGMA flagship prefix: outcome={} time={:.1}s rss={:.1}MB stats={:?} | at full L={}: sigma storage series={:.0}MB + basis={:.0}MB + residual={:.0}MB = {:.0}MB vs Toeplitz matrix {:.1}GB",
            match out {
                BlockOutcome::Vectors(_) => "Vectors",
                BlockOutcome::Invertible => "Invertible",
                BlockOutcome::Failed => "Failed(expected at forced L=4)",
            },
            secs,
            peak_rss_mb(),
            st,
            l_full,
            series_mb,
            basis_mb,
            resid_mb,
            series_mb + basis_mb + resid_mb,
            toep_gb
        );
    }

    fn mat_mul_b(gf: &Gf, a: &[u16], bb: &[u16], b: usize) -> Vec<u16> {
        let mut out = vec![0u16; b * b];
        for i in 0..b {
            for kk in 0..b {
                let av = a[i * b + kk];
                if av != 0 {
                    for j in 0..b {
                        let bv = bb[kk * b + j];
                        if bv != 0 {
                            out[i * b + j] ^= gf.mul(av, bv);
                        }
                    }
                }
            }
        }
        out
    }

    /// Memory scaling of the sigma-basis ALONE on a structured series
    /// T_k = D^k (D a fixed random b x b matrix over GF(1024), b = 32):
    /// the series has a generator of degree <= b, so after ~2b steps the
    /// discrepancies vanish on the F-side and the run is fast; the storage
    /// (series + residual + basis coefficients) is allocated at full size
    /// sigma, which is what this measures. Run one sigma value per
    /// process for a clean VmHWM.
    fn mem_structured(sigma: usize) {
        let gf = Gf::new(10);
        let b = 32usize;
        let mut rng = Rng::new(424242);
        let q = gf.order as u64;
        let dmat: Vec<u16> = (0..b * b).map(|_| rng.below(q) as u16).collect();
        let mut series_t: Vec<Vec<u16>> = Vec::with_capacity(sigma);
        let mut cur = vec![0u16; b * b];
        for i in 0..b {
            cur[i * b + i] = 1;
        }
        for _ in 0..sigma {
            series_t.push(cur.clone());
            cur = mat_mul_b(&gf, &cur, &dmat, b);
        }
        let t = Instant::now();
        let basis = crate::sigma::sigma_basis_aug(&gf, &series_t, sigma, b);
        let secs = t.elapsed().as_secs_f64();
        let toep_gb = ((sigma / 2 * b) as f64).powi(2) * 2.0 / 1e9;
        eprintln!(
            "SIGMA mem structured: sigma={} (L={}) time={:.1}s rss={:.1}MB basis_coeffs={} max_f_deg={} | Toeplitz at same L: {:.1}GB",
            sigma,
            sigma / 2,
            secs,
            peak_rss_mb(),
            basis.coeffs.len(),
            basis.f_deg.iter().cloned().filter(|&x| x != usize::MAX).max().unwrap_or(0),
            toep_gb
        );
    }

    #[test]
    #[ignore]
    fn bench_sigma_mem_structured_2200() {
        mem_structured(2200);
    }

    #[test]
    #[ignore]
    fn bench_sigma_mem_structured_4400() {
        mem_structured(4400);
    }

    #[test]
    #[ignore]
    fn bench_sigma_mem_structured_8810() {
        mem_structured(8810);
    }

    /// Time scaling of the sigma-basis on a GENERIC (random) series at
    /// b = 32: the honest quadratic-time profile of the M-Basis
    /// iteration. Kept at moderate sigma (256/512) for wall-clock.
    fn mem_random(sigma: usize) {
        let gf = Gf::new(8);
        let b = 32usize;
        let mut rng = Rng::new(13579);
        let q = gf.order as u64;
        let series_t: Vec<Vec<u16>> = (0..sigma)
            .map(|_| (0..b * b).map(|_| rng.below(q) as u16).collect())
            .collect();
        let t = Instant::now();
        let basis = crate::sigma::sigma_basis_aug(&gf, &series_t, sigma, b);
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "SIGMA time random: sigma={} time={:.1}s rss={:.1}MB basis_coeffs={}",
            sigma,
            secs,
            peak_rss_mb(),
            basis.coeffs.len()
        );
    }

    #[test]
    #[ignore]
    fn bench_sigma_time_random_256() {
        mem_random(256);
    }

    #[test]
    #[ignore]
    fn bench_sigma_time_random_512() {
        mem_random(512);
    }
}
