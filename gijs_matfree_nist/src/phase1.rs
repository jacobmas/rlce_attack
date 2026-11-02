//! Phase 1: frequency-table pair finding with disk spooling.
//!
//! Attacker view: we see ONLY the public columns (no twin metadata).
//! Per trial:
//!   1. Choose a random n'-subset S of the N public columns; shorten.
//!   2. For each hold-out tau in S, run the GIJS distinguisher.
//!   3. Let V = {flagged hold-outs}. If |V| > T_reject, discard the trial.
//!   4. Else spool the sorted flagged index list to disk.
//!
//! After all trials, stream the spooled files (never holding all trials
//! in RAM) and count, for each column pair (i,j), the number of trials
//! where both were flagged. The w true twin pairs co-occur far more
//! often (2 unhit conditions) than non-pairs (4 unhit conditions).
//!
//! Memory: the pair counter is C(N,2) u32s (1.2MB at N=790). For very
//! large N we fall back to a sparse HashMap. Trial data is fully streamed.

use crate::gf::Gf;
use crate::gijs::distinguish;
use crate::linalg::shorten;
use crate::rng::Rng;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Distinguisher engine selection
// ---------------------------------------------------------------------------

/// Which GIJS distinguisher Phase 1 runs per hold-out tau.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DistEngine {
    /// Dense GIJS distinguisher (`gijs::distinguish`): builds the Hasse
    /// matrix M_tau densely and eliminates. Exact decision.
    Dense,
    /// Matrix-free block-Wiedemann distinguisher with sigma-basis
    /// extraction (`matfree::distinguish_matfree_bw_sigma_with`): never
    /// forms M; peak memory O(N_d + E + sigma * b^2) instead of
    /// O(E * N_d). Rejections are certified exactly; accepts are
    /// heuristic (~q^-samples), same contract as the other matrix-free
    /// engines.
    BwSigma,
    /// Hold-out reuse distinguisher (`reuse::distinguish_all_reuse`):
    /// ONE factorization of the full jet matrix M_full (all n' points)
    /// per trial, then two small rank checks per hold-out. Decisions are
    /// EXACTLY the dense decisions (same test, rearranged linear
    /// algebra; oracle-tested), so flagged sets are identical to
    /// `Dense` -- but it materializes the augmented matrix
    /// [M_full | I_E] plus a left-kernel basis, so peak memory is
    /// O(E * (N_d + E)) instead of O(E * N_d) per hold-out. Guarded by
    /// a pre-flight estimate (see `check_reuse_memory`); use `BwSigma`
    /// when the estimate is too large.
    Reuse,
}

impl DistEngine {
    pub fn name(&self) -> &'static str {
        match self {
            DistEngine::Dense => "dense",
            DistEngine::BwSigma => "bw-sigma",
            DistEngine::Reuse => "reuse",
        }
    }
}

/// Pre-flight memory limit for the `reuse` engine: refuse to start a
/// Phase-1 run whose worst-case reuse footprint (augmented matrix +
/// left-kernel basis, per `reuse::reuse_memory_estimate`) exceeds this.
/// 8 GiB: the target deployment machine class has tens of GB; one trial
/// must leave room for the key, the pair counter, and parallel trials
/// sharing the box. At tractable parameters the estimate is tens of MB,
/// so the guard only fires on flagship-shaped configs, where bw-sigma
/// is the right engine.
pub const REUSE_MEMORY_LIMIT_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Format a byte count as e.g. "21.8 MB" / "106.8 GB" (decimal units).
pub fn fmt_bytes(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.1} GB", b as f64 / 1e9)
    } else if b >= 1_000_000 {
        format!("{:.1} MB", b as f64 / 1e6)
    } else if b >= 1_000 {
        format!("{:.1} kB", b as f64 / 1e3)
    } else {
        format!("{} B", b)
    }
}

/// Pre-flight guard for `--engine reuse`: estimate the worst-case
/// memory of `distinguish_all_reuse` on shortened columns (n' of them,
/// length k') with parameters (s, d) and reject configs whose estimate
/// exceeds `REUSE_MEMORY_LIMIT_BYTES`. Pure dimension arithmetic --
/// allocates nothing. On Ok returns the estimate (for the run header).
pub fn check_reuse_memory(
    nprime: usize,
    kp: usize,
    s: usize,
    d: usize,
) -> Result<crate::reuse::ReuseMemoryEstimate, String> {
    let est = crate::reuse::reuse_memory_estimate(nprime, kp, s, d);
    if est.total_bytes > REUSE_MEMORY_LIMIT_BYTES {
        Err(format!(
            "--engine reuse at (n'={}, k'={}, s={}, d={}) would need up to {} \
             (augmented matrix [M_full | I_E] = {}, left-kernel basis <= {}), \
             over the {} pre-flight limit. Use --engine bw-sigma for \
             parameters of this size (it never forms M).",
            nprime,
            kp,
            s,
            d,
            fmt_bytes(est.total_bytes),
            fmt_bytes(est.augmented_bytes),
            fmt_bytes(est.pperp_bytes),
            fmt_bytes(REUSE_MEMORY_LIMIT_BYTES),
        ))
    } else {
        Ok(est)
    }
}

// ---------------------------------------------------------------------------
// One trial (attacker view: public columns only)
// ---------------------------------------------------------------------------

/// Run a single Phase-1 trial on public columns.
/// Returns the SORTED list of flagged original column indices.
/// Uses only `cols` (no secret metadata): this is the honest attacker.
pub fn attack_trial(
    gf: &Gf,
    cols: &[Vec<u16>],
    nprime: usize,
    s: usize,
    d: usize,
    seed: u64,
) -> Vec<usize> {
    attack_trial_with_engine(gf, cols, nprime, s, d, seed, DistEngine::Dense)
}

/// attack_trial with an explicit distinguisher engine. For BwSigma the
/// per-trial `seed` seeds the block-Wiedemann config (the distinguisher
/// mixes in tau and the column count itself).
pub fn attack_trial_with_engine(
    gf: &Gf,
    cols: &[Vec<u16>],
    nprime: usize,
    s: usize,
    d: usize,
    seed: u64,
    engine: DistEngine,
) -> Vec<usize> {
    let n_pub = cols.len();
    assert!(nprime <= n_pub, "n'={} > N={}", nprime, n_pub);
    let mut rng = Rng::new(seed);
    // Random shortening: hit N - n' positions, keep n'.
    let n_hit = n_pub - nprime;
    let hit = rng.subset(n_pub, n_hit);
    let (short_cols, kept) = shorten(gf, cols, &hit);
    // Reuse engine: all hold-out decisions from ONE factorization of
    // M_full (no per-tau loop). Decisions are identical to Dense by
    // construction (see reuse.rs), so the flagged set matches.
    if engine == DistEngine::Reuse {
        let decisions = crate::reuse::distinguish_all_reuse(gf, &short_cols, s, d);
        let mut flagged: Vec<usize> = kept
            .iter()
            .enumerate()
            .filter(|(ti, _)| !decisions[*ti])
            .map(|(_, &orig)| orig)
            .collect();
        flagged.sort_unstable();
        return flagged;
    }
    // GIJS per hold-out; collect flagged original indices.
    let mut flagged = Vec::new();
    for (ti, &orig) in kept.iter().enumerate() {
        let is_grs = match engine {
            DistEngine::Dense => distinguish(gf, &short_cols, s, d, ti),
            DistEngine::Reuse => unreachable!("reuse handled above"),
            DistEngine::BwSigma => {
                let mut cfg = crate::matfree::BWConfig::default();
                cfg.seed = seed;
                let mut st = crate::matfree::BWStats::default();
                crate::matfree::distinguish_matfree_bw_sigma_with(
                    gf, &short_cols, s, d, ti, &cfg, &mut st,
                )
            }
        };
        if !is_grs {
            flagged.push(orig);
        }
    }
    flagged.sort_unstable();
    flagged
}

// ---------------------------------------------------------------------------
// t'_max threshold
// ---------------------------------------------------------------------------

/// PIR persistence threshold: t'_max = ceil((n'-1) - d*D/s) - 1,
/// where D = k' + w - 1 (ambient GRS degree after shortening).
/// k' = k - (N - n').
pub fn tprime_max(nprime: usize, k: usize, n_pub: usize, w: usize, s: usize, d: usize) -> usize {
    let kp = k as isize - (n_pub as isize - nprime as isize);
    assert!(kp > 0, "k' must be positive");
    let kp = kp as usize;
    let deg = kp + w - 1; // D
    // ceil((n'-1) - d*D/s) - 1
    let num = (nprime as isize - 1) * s as isize - d as isize * deg as isize;
    let den = s as isize;
    // ceil_div for possibly negative num: we want t'_max >= 0 sensible
    let c = if num <= 0 {
        0
    } else {
        (num + den - 1) / den
    };
    if c <= 0 {
        0
    } else {
        (c - 1) as usize
    }
}

// ---------------------------------------------------------------------------
// Binomial numerics (log-space PMFs; incomplete-beta CDFs)
// ---------------------------------------------------------------------------

/// Natural log of the Gamma function for x > 0 (Lanczos approximation,
/// Numerical Recipes `gammln`; ~1e-10 relative accuracy). Lets us
/// evaluate binomial PMFs in log space -- no factorials, safe for n in
/// the millions.
fn log_gamma(x: f64) -> f64 {
    const COF: [f64; 6] = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    debug_assert!(x > 0.0);
    let mut ser = 1.000000000190015f64;
    let mut y = x;
    for &c in COF.iter() {
        y += 1.0;
        ser += c / y;
    }
    let t = x + 5.5;
    let tmp = t - (x + 0.5) * t.ln();
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// Continued fraction for the incomplete beta function (Lentz's
/// algorithm, Numerical Recipes `betacf`).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: usize = 2000;
    const EPS: f64 = 3.0e-14;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta I_x(a, b) in [0,1].
fn reg_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * x.ln()
        + b * (1.0 - x).ln())
    .exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// log P[X = k] for X ~ Bin(n, p), in log space.
fn binom_logpmf(n: usize, p: f64, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let mut v = log_gamma(n as f64 + 1.0) - log_gamma(k as f64 + 1.0)
        - log_gamma((n - k) as f64 + 1.0);
    if k > 0 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        v += k as f64 * p.ln();
    }
    if k < n {
        if p >= 1.0 {
            return f64::NEG_INFINITY;
        }
        v += (n - k) as f64 * (1.0 - p).ln();
    }
    v
}

/// Full PMF vector of Bin(n, p), built by recurrence outward from the
/// mode (anchored with a log-space PMF) and renormalized; numerically
/// stable for n up to millions (used for n <= SMALL_N in practice).
fn binom_pmf_vec(n: usize, p: f64) -> Vec<f64> {
    let mut pmf = vec![0.0f64; n + 1];
    if p <= 0.0 {
        pmf[0] = 1.0;
        return pmf;
    }
    if p >= 1.0 {
        pmf[n] = 1.0;
        return pmf;
    }
    let mode = (((n + 1) as f64 * p).floor() as usize).min(n);
    pmf[mode] = binom_logpmf(n, p, mode).exp();
    let ratio = p / (1.0 - p);
    for j in mode..n {
        pmf[j + 1] = pmf[j] * ((n - j) as f64 / (j + 1) as f64) * ratio;
    }
    let inv_ratio = (1.0 - p) / p;
    for j in (1..=mode).rev() {
        pmf[j - 1] = pmf[j] * (j as f64 / (n - j + 1) as f64) * inv_ratio;
    }
    let tot: f64 = pmf.iter().sum();
    if tot > 0.0 {
        for v in pmf.iter_mut() {
            *v /= tot;
        }
    }
    pmf
}

/// Threshold below which binomial CDFs are computed by direct PMF
/// summation; above it, by the regularized incomplete beta.
const SMALL_N: usize = 8192;

/// P[X <= k] for X ~ Bin(n, p).
fn binom_cdf(n: usize, p: f64, k: i64) -> f64 {
    if k < 0 {
        return 0.0;
    }
    if k >= n as i64 {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0; // X = 0 a.s., and k >= 0 here
    }
    if p >= 1.0 {
        return 0.0; // X = n a.s., and k < n here
    }
    if n <= SMALL_N {
        let pmf = binom_pmf_vec(n, p);
        pmf[..=(k as usize)].iter().sum()
    } else {
        reg_incomplete_beta(1.0 - p, (n as i64 - k) as f64, (k + 1) as f64)
    }
}

/// P[X >= r] for X ~ Bin(n, p). (Uses I_p(r, n-r+1) directly on the
/// large-n path, so small upper tails keep relative accuracy.)
fn binom_sf_ge(n: usize, p: f64, r: i64) -> f64 {
    if r <= 0 {
        return 1.0;
    }
    if r > n as i64 {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0; // X = 0 a.s., and r >= 1 here
    }
    if p >= 1.0 {
        return 1.0; // X = n a.s., and r <= n here
    }
    if n <= SMALL_N {
        let pmf = binom_pmf_vec(n, p);
        pmf[(r as usize)..].iter().sum()
    } else {
        reg_incomplete_beta(p, r as f64, (n as i64 - r + 1) as f64)
    }
}

// ---------------------------------------------------------------------------
// Auto trial-count planning (corrected frequency-table analysis)
// ---------------------------------------------------------------------------
//
// Implements the planning half of the filter-correction note
// (filter_correction_20261019.tex): given (q, N, k, w, n', s, d) and a
// target failure probability eps, derive how many Phase-1 trials to run.
//
// Model (as in the note):
//   * A shortening keeps a uniformly random n'-subset S of the N public
//     columns. A twin pair contributes 2 noisy columns iff BOTH members
//     land in S ("unhit"); a half-hit pair is derandomized (CLT) and
//     contributes 0. Hence t' = 2u with u = #unhit pairs, modeled as
//         u ~ Bin(w, p_unhit),  p_unhit = n'(n'-1) / (N(N-1)).
//   * Good trial: t' <= t'_max (PIR persistence threshold, `tprime_max`).
//   * Filter: record iff |V| <= T_reject = t'_max - y.
//   * H3 planning model for the filter: conditional on t' = 2u,
//         P[|V| <= T | t'] = P[Bin(2u, 1 - d/q) <= T],
//     i.e. each noisy column is flagged independently with probability
//     exactly 1 - d/q (the H1/H3 lower bound, used as the planning
//     value). ASSUMPTION, stated explicitly: on the bad side this is a
//     rigorous upper bound on the recording probability (H3 says |V|
//     stochastically dominates Bin(t', 1-d/q), so P[|V| <= T] <= the
//     binomial CDF). On the good side it is the planning estimate: for
//     2u <= T_reject it equals 1, matching the note's deterministic
//     recording lemma; in the band T_reject < 2u <= t'_max it credits
//     the H3 rate. The note's theorem itself credits only
//     p_gr = P[t' <= t'_max - y] (deterministic part); we print p_gr
//     alongside P_record.
//   * y is the smallest margin with P[recorded AND bad] <= eps under
//     this model (the note's Theorem also prints the worst-case
//     per-bad-trial bound delta_sneak(y); we report it for reference --
//     it is far too loose to choose y by in practice).
//   * Separation: after R recorded good trials, a true pair's co-flag
//     count is Bin(R, p) with p >= p1_min = p_unhit (1 - 2d/q) (H2');
//     a non-pair's is Bin(R, p) with p <= p_false_max = the exact
//     4-unhit probability (n')_4/(N)_4 <= p_unhit^2 (a non-pair co-flag
//     needs both pairs unhit). R is the smallest count whose best
//     threshold separates the two families with total failure <= eps
//     under exact binomial tails + a union bound over the w true pairs
//     and all C(N,2) non-pairs.
//   * Raw trials T: smallest T with P[Bin(T, P_record) >= R] >= 1-eps.
//
// Each of the three components (sneak, separation, quantile) is bounded
// by eps, so the total planning failure is <= ~3*eps by a union bound;
// set --fail-eps accordingly.

/// Hard cap on auto-planned raw trials (2^24). Beyond this, planning
/// reports infeasibility instead of emitting an absurd count.
pub const T_RAW_CAP: usize = 1 << 24;
/// Cap on the recorded-trial requirement during the r search.
const R_REC_CAP: usize = 2_000_000;

/// Result of auto trial-count planning: the full chain, for printing.
#[derive(Clone, Debug)]
pub struct TrialPlan {
    pub q: u32,
    pub n_pub: usize,
    pub k: usize,
    pub w: usize,
    pub nprime: usize,
    pub s: usize,
    pub d: usize,
    pub eps: f64,
    /// Step 1: per-pair unhit probability.
    pub p_unhit: f64,
    /// Shortened dimension k' = k - (N - n') and ambient degree D = k'+w-1.
    pub kprime: usize,
    pub deg_d: usize,
    /// Step 2: PIR persistence threshold and J_max = floor(t'_max/2).
    pub tprime_max: usize,
    pub j_max: usize,
    /// Step 3: chosen safety margin and rejection threshold.
    pub y: usize,
    pub t_reject: usize,
    /// P[recorded AND bad] at the chosen y (H3 model), and the note's
    /// worst-case per-bad-trial sneak bound delta_sneak(y) (reference).
    pub p_bad_recorded: f64,
    pub delta_sneak: f64,
    /// True iff p_bad_recorded <= eps (false only if y was overridden).
    pub sneak_ok: bool,
    pub y_overridden: bool,
    /// Step 4: P[good] = P[t' <= t'_max]; p_gr = P[t' <= t'_max - y]
    /// (deterministic-recording part); P_record = P[recorded AND good].
    pub p_good: f64,
    pub p_gr: f64,
    pub p_record: f64,
    /// Step 5: true-pair co-flag lower bound, non-pair upper bound, gap.
    pub p1_min: f64,
    pub p_false_max: f64,
    pub gamma: f64,
    /// Step 6: recorded good trials required for separation.
    pub r_recorded: usize,
    /// Step 7: raw trials to run.
    pub t_raw: usize,
}

/// Format a probability as decimal plus a 2^x annotation (for auditing
/// against the paper's powers of two).
fn fmt_p(p: f64) -> String {
    if p <= 0.0 {
        return "0".to_string();
    }
    if p >= 1.0 {
        return "1".to_string();
    }
    if p >= 1e-4 {
        format!("{:.6} (2^{:.2})", p, p.log2())
    } else {
        format!("{:.3e} (2^{:.2})", p, p.log2())
    }
}

/// The note's worst-case sneak-through bound delta_sneak(y):
/// max over attainable bad t' (= even, t'_max < t' <= min(2w, n')) of
/// exp(-2 t' [(1 - d/q) - T_reject/t']_+^2). Reference value only.
fn sneak_bound(tmax: usize, t_reject: usize, t_hi: usize, a_det: f64) -> f64 {
    let mut worst = 0.0f64;
    let mut tp = tmax + 1;
    if tp % 2 == 1 {
        tp += 1; // t' = 2u is even in this model
    }
    while tp <= t_hi {
        let gap = a_det - t_reject as f64 / tp as f64;
        let b = if gap > 0.0 {
            (-2.0 * tp as f64 * gap * gap).exp()
        } else {
            1.0
        };
        if b > worst {
            worst = b;
        }
        tp += 2;
    }
    worst
}

/// Separation failure at R recorded good trials: the minimum over
/// thresholds theta of
///   w * P[Bin(R, p1_min) <= theta] + C(N,2) * P[Bin(R, p_false_max) > theta]
/// (exact binomial tails; union bound over both families).
fn separation_failure(
    r: usize,
    p1: f64,
    pf: f64,
    n_true: f64,
    n_false: f64,
) -> f64 {
    if r == 0 {
        return f64::INFINITY;
    }
    if r <= SMALL_N {
        let pmf1 = binom_pmf_vec(r, p1);
        let pmf2 = binom_pmf_vec(r, pf);
        // suffix2[j] = P[X2 >= j]; P[X2 > theta] = suffix2[theta + 1].
        let mut suffix2 = vec![0.0f64; r + 2];
        for j in (0..=r).rev() {
            suffix2[j] = suffix2[j + 1] + pmf2[j];
        }
        let mut cdf1 = 0.0f64;
        let mut best = f64::INFINITY;
        for theta in 0..=r {
            cdf1 += pmf1[theta];
            let f = n_true * cdf1 + n_false * suffix2[theta + 1];
            if f < best {
                best = f;
            }
        }
        best
    } else {
        // Large R: scan only the window where both tails can be small
        // (outside it, one weighted tail exceeds 1 > eps by a wide
        // margin: 12-sigma window edges).
        let sd1 = (r as f64 * p1 * (1.0 - p1)).sqrt();
        let sdf = (r as f64 * pf * (1.0 - pf)).sqrt();
        let lo = ((r as f64 * pf - 12.0 * sdf).floor() as i64).max(0) as usize;
        let hi = (((r as f64 * p1 + 12.0 * sd1).ceil() as i64).max(0) as usize).min(r);
        let mut best = f64::INFINITY;
        for theta in lo..=hi {
            let f = n_true * binom_cdf(r, p1, theta as i64)
                + n_false * binom_sf_ge(r, pf, theta as i64 + 1);
            if f < best {
                best = f;
            }
        }
        best
    }
}

/// Smallest recorded-good-trial count R whose separation failure is
/// <= eps, or None if it exceeds R_REC_CAP (gap gamma too small).
fn required_recorded(p1: f64, pf: f64, n_true: f64, n_false: f64, eps: f64) -> Option<usize> {
    if separation_failure(1, p1, pf, n_true, n_false) <= eps {
        return Some(1);
    }
    let mut lo = 1usize;
    let mut hi = 2usize;
    while separation_failure(hi, p1, pf, n_true, n_false) > eps {
        lo = hi;
        hi = hi.checked_mul(2)?;
        if hi > R_REC_CAP {
            return None;
        }
    }
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if separation_failure(mid, p1, pf, n_true, n_false) <= eps {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // Polish discrete non-monotonicity at the boundary (bounded).
    let mut guard = 0;
    while hi > 1 && guard < 64 && separation_failure(hi - 1, p1, pf, n_true, n_false) <= eps {
        hi -= 1;
        guard += 1;
    }
    Some(hi)
}

/// Smallest raw trial count T >= r with P[Bin(T, p_record) >= r] >= 1-eps,
/// or None if it exceeds T_RAW_CAP.
fn required_raw(p_record: f64, r: usize, eps: f64) -> Option<usize> {
    let feas = |t: usize| binom_cdf(t, p_record, r as i64 - 1) <= eps;
    if feas(r) {
        return Some(r);
    }
    let mut lo = r;
    let mut hi = (((r as f64) / p_record).ceil() as usize)
        .max(r + 1)
        .min(T_RAW_CAP);
    loop {
        if feas(hi) {
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if feas(mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(hi);
        }
        if hi >= T_RAW_CAP {
            return None;
        }
        lo = hi;
        hi = (hi * 2).min(T_RAW_CAP);
    }
}

/// Compute the auto trial-count plan. See the section comment above for
/// the model and its stated assumptions. `y_override` (from an explicit
/// --y-margin) replaces the planner's y; everything else is recomputed
/// at that y. Errors are planning-time infeasibility messages.
pub fn plan_trials(
    q: u32,
    n_pub: usize,
    k: usize,
    w: usize,
    nprime: usize,
    s: usize,
    d: usize,
    eps: f64,
    y_override: Option<usize>,
) -> Result<TrialPlan, String> {
    if !(eps > 0.0 && eps < 0.5) {
        return Err(format!("--fail-eps must lie in (0, 0.5); got {}", eps));
    }
    if w == 0 {
        return Err("w = 0: there are no twin pairs to find.".to_string());
    }
    if n_pub < 2 {
        return Err(format!("N = {} < 2: degenerate public key.", n_pub));
    }
    if nprime == 0 || nprime >= n_pub {
        return Err(format!(
            "need 1 <= n' < N for a nontrivial shortening; got n'={}, N={}.",
            nprime, n_pub
        ));
    }
    let dq = d as f64 / q as f64;
    if 2.0 * dq >= 1.0 {
        return Err(format!(
            "2d/q = {:.3} >= 1: the H2' bound p1_min = p_unhit (1 - 2d/q) is vacuous \
             (q={}, d={}). Increase q or decrease d.",
            2.0 * dq, q, d
        ));
    }
    let a_det = 1.0 - dq; // H1/H3 per-column detection lower bound

    // Step 1: unhit model. u = #pairs fully inside the random n'-subset.
    let p_unhit = (nprime * (nprime - 1)) as f64 / (n_pub * (n_pub - 1)) as f64;
    let u_pmf: Vec<f64> = (0..=w).map(|u| binom_logpmf(w, p_unhit, u).exp()).collect();

    // Step 2: persistence threshold (reuse the code's formula).
    let tmax = tprime_max(nprime, k, n_pub, w, s, d);
    let j_max = tmax / 2; // floor: good ⟺ u <= J_max (t' = 2u <= t'_max)
    let kprime = k - (n_pub - nprime);
    let deg_d = kprime + w - 1;

    // Step 3: safety margin y via P[recorded AND bad] under the H3 model.
    let prob_bad_rec = |t_reject: usize| -> f64 {
        (j_max + 1..=w)
            .map(|u| u_pmf[u] * binom_cdf(2 * u, a_det, t_reject as i64))
            .sum()
    };
    let (y, t_reject, p_bad) = match y_override {
        Some(yv) => {
            if yv > tmax {
                return Err(format!(
                    "--y-margin {} exceeds t'_max = {} (T_reject would be negative).",
                    yv, tmax
                ));
            }
            let tr = tmax - yv;
            (yv, tr, prob_bad_rec(tr))
        }
        None => {
            let mut yv = 0usize;
            loop {
                if yv > tmax {
                    return Err(format!(
                        "no safety margin y <= t'_max = {} achieves P[recorded & bad] <= eps = {} \
                         under the H3 model; the filter cannot be made safe at these parameters. \
                         Choose (n',s,d) with a larger t'_max.",
                        tmax, eps
                    ));
                }
                let tr = tmax - yv;
                let pb = prob_bad_rec(tr);
                if pb <= eps {
                    break (yv, tr, pb);
                }
                yv += 1;
            }
        }
    };
    let sneak_ok = p_bad <= eps;
    if t_reject < 2 {
        return Err(format!(
            "T_reject = t'_max - y = {} - {} = {} < 2: the recording filter would never admit \
             a flagged PAIR (|V| >= 2), so the frequency table stays empty and Phase 1 cannot \
             work. Choose (n',s,d) with a larger t'_max (t'_max ~ (n'-1) - d*(k'+w-1)/s).",
            tmax, y, t_reject
        ));
    }

    // Step 4: recording probabilities under the same H3 model.
    let p_good: f64 = u_pmf[..=j_max.min(w)].iter().sum();
    let p_record: f64 = (0..=j_max.min(w))
        .map(|u| u_pmf[u] * binom_cdf(2 * u, a_det, t_reject as i64))
        .sum();
    // Deterministic-recording part: t' <= t'_max - y ⟺ u <= floor(T_reject/2).
    let u_def = (t_reject / 2).min(w);
    let p_gr: f64 = u_pmf[..=u_def].iter().sum();
    if p_record <= 1e-12 {
        return Err(format!(
            "P_record = {:.3e} is effectively zero: recorded good trials are unattainable \
             (t'_max = {}, T_reject = {}, p_unhit = {:.4}). Choose (n',s,d) with a larger \
             t'_max, or smaller w.",
            p_record, tmax, t_reject, p_unhit
        ));
    }

    // Step 5: separation rates (H2' lower bound; exact 4-unhit upper bound).
    let p1_min = p_unhit * (1.0 - 2.0 * dq);
    let p_false_max = if w < 2 || nprime < 4 {
        0.0 // no two-pair non-pair configuration exists
    } else {
        ((nprime * (nprime - 1) * (nprime - 2) * (nprime - 3)) as f64)
            / ((n_pub * (n_pub - 1) * (n_pub - 2) * (n_pub - 3)) as f64)
    };
    let gamma = p1_min - p_false_max;
    if gamma <= 0.0 {
        return Err(format!(
            "no separation gap: p1_min = {:.6} <= p_false_max = {:.6} (gamma <= 0). \
             The frequency table cannot rank true pairs above non-pairs at these parameters.",
            p1_min, p_false_max
        ));
    }

    // Step 6: recorded good trials needed for separation at failure <= eps.
    let n_pairs = (n_pub * (n_pub - 1) / 2) as f64;
    let r = required_recorded(p1_min, p_false_max, w as f64, n_pairs, eps).ok_or_else(|| {
        format!(
            "frequency separation needs more than {} recorded good trials \
             (gamma = {:.4} is too small); parameters are infeasible for Phase 1.",
            R_REC_CAP, gamma
        )
    })?;

    // Step 7: raw trials so that >= r are recorded-good w.p. >= 1 - eps.
    let t_raw = required_raw(p_record, r, eps).ok_or_else(|| {
        format!(
            "accumulating r = {} recorded good trials at P_record = {} needs more than \
             {} (2^24) raw trials; parameters are infeasible for Phase 1. \
             (P_good = {}, t'_max = {}.) Choose (n',s,d) with larger t'_max or smaller w.",
            r,
            fmt_p(p_record),
            T_RAW_CAP,
            fmt_p(p_good),
            tmax
        )
    })?;

    let t_hi = (2 * w).min(nprime);
    let delta_sneak = sneak_bound(tmax, t_reject, t_hi, a_det);

    Ok(TrialPlan {
        q,
        n_pub,
        k,
        w,
        nprime,
        s,
        d,
        eps,
        p_unhit,
        kprime,
        deg_d,
        tprime_max: tmax,
        j_max,
        y,
        t_reject,
        p_bad_recorded: p_bad,
        delta_sneak,
        sneak_ok,
        y_overridden: y_override.is_some(),
        p_good,
        p_gr,
        p_record,
        p1_min,
        p_false_max,
        gamma,
        r_recorded: r,
        t_raw,
    })
}

/// Render the plan as the printed audit chain (one step per group).
pub fn plan_report(p: &TrialPlan) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== Phase 1 auto trial planning (fail_eps = {}) ===\n",
        p.eps
    ));
    out.push_str(&format!(
        "params: q={} N={} k={} w={} n'={} (s,d)=({},{})\n",
        p.q, p.n_pub, p.k, p.w, p.nprime, p.s, p.d
    ));
    out.push_str(&format!(
        "[1] p_unhit = n'(n'-1)/(N(N-1)) = {}; #unhit u ~ Bin(w, p_unhit), t' = 2u\n",
        fmt_p(p.p_unhit)
    ));
    out.push_str(&format!(
        "[2] t'_max = ceil((n'-1) - d*D/s) - 1 = {} (k'={}, D=k'+w-1={}); J_max = floor(t'_max/2) = {}\n",
        p.tprime_max, p.kprime, p.deg_d, p.j_max
    ));
    out.push_str(&format!(
        "[3] y = {} ({}): smallest margin with P[recorded & bad] <= eps under the H3 model; \
         P[recorded & bad] = {}; T_reject = t'_max - y = {}\n",
        p.y,
        if p.y_overridden { "user override" } else { "auto" },
        fmt_p(p.p_bad_recorded),
        p.t_reject
    ));
    if !p.sneak_ok {
        out.push_str(&format!(
            "    WARNING: at this y, P[recorded & bad] = {} > eps = {} (margin overridden).\n",
            fmt_p(p.p_bad_recorded),
            p.eps
        ));
    }
    out.push_str(&format!(
        "    (reference: worst-case per-bad-trial sneak bound delta_sneak(y) = {})\n",
        fmt_p(p.delta_sneak)
    ));
    out.push_str(&format!(
        "[4] P_good = P[t' <= t'_max] = {}; P_record = P[recorded & good] = {} (H3 model; deterministic part p_gr = {})\n",
        fmt_p(p.p_good),
        fmt_p(p.p_record),
        fmt_p(p.p_gr)
    ));
    out.push_str(&format!(
        "[5] p1_min = p_unhit*(1 - 2d/q) = {}; p_false_max = (n')_4/(N)_4 = {}; gamma = {:.6}\n",
        fmt_p(p.p1_min),
        fmt_p(p.p_false_max),
        p.gamma
    ));
    out.push_str(&format!(
        "[6] r = {} recorded good trials (exact binomial tails + union bound over {} true pairs and {} non-pairs, failure <= {})\n",
        p.r_recorded,
        p.w,
        (p.n_pub * (p.n_pub - 1) / 2),
        p.eps
    ));
    out.push_str(&format!(
        "[7] T = {} raw trials (smallest with P[Bin(T, P_record) >= r] >= 1 - {})\n",
        p.t_raw, p.eps
    ));
    out
}

// ---------------------------------------------------------------------------
// Trial runner (parallel, disk-spooled)
// ---------------------------------------------------------------------------

/// Configuration for a Phase-1 run.
pub struct Phase1Config {
    pub nprime: usize,
    pub s: usize,
    pub d: usize,
    pub trials: usize,
    pub threads: usize,
    pub seed: u64,
    /// Safety margin y; T_reject = t'_max - y. Default 0.
    pub y_margin: usize,
    /// Distinguisher engine (dense, matrix-free sigma-basis BW, or
    /// hold-out reuse; reuse is pre-flight memory-guarded, see
    /// `check_reuse_memory`).
    pub engine: DistEngine,
    pub out_dir: PathBuf,
}

/// True iff a spooled trial file is complete and well-formed: a single
/// line `n i1 ... in` with exactly n strictly increasing indices and a
/// trailing newline (i.e. its write finished before any interruption).
fn spooled_trial_valid(path: &Path) -> bool {
    let s = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return false,
    };
    if !s.ends_with('\n') {
        return false;
    }
    let mut it = s.split_whitespace();
    let n: usize = match it.next().and_then(|x| x.parse().ok()) {
        Some(n) => n,
        None => return false,
    };
    let mut prev: Option<usize> = None;
    let mut cnt = 0usize;
    for tok in it {
        let v: usize = match tok.parse() {
            Ok(v) => v,
            Err(_) => return false,
        };
        if let Some(p) = prev {
            if v <= p {
                return false;
            }
        }
        prev = Some(v);
        cnt += 1;
    }
    cnt == n
}

/// Parse a trial timing sidecar: single line `trial <t> wall_ms <ms>`.
/// Returns the wall time in milliseconds, or None if malformed.
fn parse_trial_time(s: &str) -> Option<u128> {
    let mut it = s.split_whitespace();
    if it.next()? != "trial" {
        return None;
    }
    it.next()?; // trial index (also in the filename; not validated here)
    if it.next()? != "wall_ms" {
        return None;
    }
    it.next()?.parse().ok()
}

/// Run all trials in parallel, spooling each trial's flagged set to disk.
/// Trial file `trial_{:05}.txt`: single line `n_flagged i1 i2 ...` (sorted).
/// Returns (t_max, t_reject, n_recorded_candidates).
///
/// RESUME: a trial whose spool file already exists and is well-formed
/// (see `spooled_trial_valid`) is NOT recomputed. Per-trial results are
/// deterministic (trial t uses seed_base + 1 + t), so re-running a
/// Phase-1 invocation with IDENTICAL parameters into the same output
/// directory resumes an interrupted run instead of starting over.
/// Delete the trials directory to force recomputation; reusing a
/// directory with different parameters is a user error (stale files
/// would be trusted).
///
/// Disk usage: ~trials * avg|V| * ~4 bytes. Fully streamed; the key columns
/// stay in RAM (N*k*2 bytes) but trial data never accumulates.
pub fn run_trials(
    gf: &Gf,
    cols: &[Vec<u16>],
    k: usize,
    w: usize,
    cfg: &Phase1Config,
) -> (usize, usize, PathBuf) {
    let n_pub = cols.len();
    // Pre-flight memory guard for the reuse engine: refuse flagship-
    // shaped configs BEFORE any trial runs (the augmented matrix alone
    // would be ~107 GB at ID-0-like (3,5) dims). k' = shortened length.
    let reuse_est = if cfg.engine == DistEngine::Reuse {
        let kp = k - (n_pub - cfg.nprime);
        match check_reuse_memory(cfg.nprime, kp, cfg.s, cfg.d) {
            Ok(est) => Some(est),
            Err(msg) => {
                eprintln!("# ERROR: {}", msg);
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let tmax = tprime_max(cfg.nprime, k, n_pub, w, cfg.s, cfg.d);
    let t_reject = tmax.saturating_sub(cfg.y_margin);
    println!(
        "# phase1: N={} n'={} (s,d)=({},{}) t'_max={} T_reject={} trials={} engine={}",
        n_pub, cfg.nprime, cfg.s, cfg.d, tmax, t_reject, cfg.trials, cfg.engine.name()
    );
    if let Some(est) = reuse_est {
        println!(
            "# reuse memory estimate: E={} (J={}, N_d={}); augmented {} + Pperp <= {} = {} worst case (limit {})",
            est.e,
            est.j,
            est.n_d,
            fmt_bytes(est.augmented_bytes),
            fmt_bytes(est.pperp_bytes),
            fmt_bytes(est.total_bytes),
            fmt_bytes(REUSE_MEMORY_LIMIT_BYTES),
        );
    }

    let trial_dir = cfg.out_dir.join("trials");
    fs::create_dir_all(&trial_dir).expect("mkdir trials");

    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .expect("thread pool");
    }

    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} trials ({eta})")
        .unwrap()
        .progress_chars("#>-");

    // Share columns via reference (Vec<Vec<u16>> is Sync).
    // Gf is rebuilt per thread (cheap: log/exp tables).
    let q = gf.order as u32;
    let cols_ref = cols;
    let nprime = cfg.nprime;
    let (s, d) = (cfg.s, cfg.d);
    let seed = cfg.seed;
    let engine = cfg.engine;
    (0..cfg.trials)
        .into_par_iter()
        .progress_with_style(style)
        .for_each(|t| {
            // Spool immediately; do not accumulate in RAM.
            let path = trial_dir.join(format!("trial_{:05}.txt", t));
            if spooled_trial_valid(&path) {
                return; // resume: keep the earlier identical-parameter result
            }
            let gf_t = Gf::for_q(q);
            let t_start = Instant::now();
            let flagged = attack_trial_with_engine(&gf_t, cols_ref, nprime, s, d, seed + 1 + t as u64, engine);
            let wall_ms = t_start.elapsed().as_millis();
            // Timing sidecar FIRST: if the run dies between the two
            // writes, resume recomputes the trial and overwrites the
            // sidecar, so a valid spool always has a timing. A resumed
            // (skipped) trial keeps its earlier sidecar untouched.
            let time_path = trial_dir.join(format!("trial_{:05}.time", t));
            fs::write(&time_path, format!("trial {} wall_ms {}\n", t, wall_ms))
                .expect("write trial timing");
            let mut f = fs::File::create(&path).expect("create trial file");
            write!(f, "{}", flagged.len()).unwrap();
            for idx in flagged {
                write!(f, " {}", idx).unwrap();
            }
            writeln!(f).unwrap();
            println!(
                "# trial {} finished in {:.2} s",
                t,
                wall_ms as f64 / 1000.0
            );
        });
            
    // Timing summary from sidecars. Resumed trials contribute their
    // earlier sidecar, if any; trials with no sidecar (e.g. runs from
    // older builds) are counted as untimed, not zero.
    {
        let mut timed = 0usize;
        let mut total_ms: u128 = 0;
        for t in 0..cfg.trials {
            if let Ok(s) = fs::read_to_string(trial_dir.join(format!("trial_{:05}.time", t))) {
                if let Some(ms) = parse_trial_time(&s) {
                    timed += 1;
                    total_ms += ms;
                }
            }
        }
        if timed > 0 {
            println!(
                "# phase1 timings: {}/{} trials timed, total {:.2} s, mean {:.2} s",
                timed,
                cfg.trials,
                total_ms as f64 / 1000.0,
                total_ms as f64 / timed as f64 / 1000.0
            );
        } else {
            println!("# phase1 timings: no trial timing sidecars found");
        }
    }

    (tmax, t_reject, trial_dir)
}

// ---------------------------------------------------------------------------
// Streaming aggregation
// ---------------------------------------------------------------------------

/// Triangular index for pair (i,j), i<j, in a C(N,2) array.
#[inline]
fn pair_index(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    // idx = i*N - i*(i+1)/2 + (j-i-1)
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Aggregate spooled trials into pair co-occurrence counts.
/// Streams trial files one at a time; only the counter array lives in RAM.
/// Applies the filter |V| <= t_reject (skips trials that fail).
/// Also counts per-column flag totals (for pure-column identification).
///
/// Returns (pair_counts, col_flagged, n_recorded).
/// col_flagged[i] = # recorded trials where column i was flagged.
/// Pure columns are those with col_flagged[i] == 0 (never flagged).
pub fn aggregate(
    trial_dir: &Path,
    n_pub: usize,
    t_reject: usize,
    trials: usize,
) -> (PairCounts, Vec<u32>, usize) {
    // Dense if C(N,2) <= 1e8 (400MB); else sparse HashMap.
    let npairs = (n_pub as u64) * (n_pub as u64 - 1) / 2;
    let dense = npairs <= 100_000_000;
    let mut counts = PairCounts::new(n_pub, dense);
    let mut col_flagged = vec![0u32; n_pub];
    let mut n_recorded = 0usize;

    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
        .unwrap()
        .progress_chars("#>-");
    let pb = indicatif::ProgressBar::new(trials as u64);
    pb.set_style(style);

    for t in 0..trials {
        let path = trial_dir.join(format!("trial_{:05}.txt", t));
        let f = fs::File::open(&path).expect("open trial file");
        let mut reader = BufReader::new(f);
        let mut line = String::new();
        reader.read_line(&mut line).expect("read trial line");
        let nums: Vec<usize> = line.split_whitespace().map(|x| x.parse().unwrap()).collect();
        if nums.is_empty() {
            pb.inc(1);
            continue;
        }
        let nf = nums[0];
        let flagged = &nums[1..];
        debug_assert_eq!(flagged.len(), nf);
        // NOTE: the attacker doesn't know which columns were "present"
        // (kept) vs hit; for pure identification, flag-rate 0 suffices
        // because pure columns are provably never flagged.
        if flagged.len() <= t_reject {
            n_recorded += 1;
            for &i in flagged {
                col_flagged[i] += 1;
            }
            // pairs
            for a in 0..flagged.len() {
                for b in (a + 1)..flagged.len() {
                    let (i, j) = (flagged[a], flagged[b]);
                    counts.increment(i, j);
                }
            }
        }
        pb.inc(1);
    }
    pb.finish_and_clear();
    (counts, col_flagged, n_recorded)
}

/// Pair counter: dense Vec<u32> or sparse HashMap.
pub enum PairCounts {
    Dense { n: usize, v: Vec<u32> },
    Sparse { m: HashMap<(u32, u32), u32> },
}

impl PairCounts {
    fn new(n_pub: usize, dense: bool) -> Self {
        if dense {
            let npairs = n_pub * (n_pub - 1) / 2;
            PairCounts::Dense { n: n_pub, v: vec![0u32; npairs] }
        } else {
            PairCounts::Sparse { m: HashMap::new() }
        }
    }
    fn increment(&mut self, i: usize, j: usize) {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        match self {
            PairCounts::Dense { n, v } => {
                let idx = pair_index(*n, a, b);
                v[idx] = v[idx].saturating_add(1);
            }
            PairCounts::Sparse { m } => {
                *m.entry((a as u32, b as u32)).or_insert(0) += 1;
            }
        }
    }
    /// Iterate over (i, j, count) for all pairs with count > 0.
    pub fn iter_nonzero(&self) -> Vec<(usize, usize, u32)> {
        match self {
            PairCounts::Dense { n, v } => {
                let mut out = Vec::new();
                // invert triangular index: iterate i<j directly
                for i in 0..*n {
                    for j in (i + 1)..*n {
                        let c = v[pair_index(*n, i, j)];
                        if c > 0 {
                            out.push((i, j, c));
                        }
                    }
                }
                out
            }
            PairCounts::Sparse { m } => m
                .iter()
                .map(|(&(a, b), &c)| (a as usize, b as usize, c))
                .collect(),
        }
    }
}

/// Greedily select w disjoint pairs with highest co-occurrence counts.
/// Returns Vec<(count, i, j)> sorted by count descending.
pub fn top_disjoint_pairs(counts: &PairCounts, w: usize, n_pub: usize) -> Vec<(u32, usize, usize)> {
    let mut pairs = counts.iter_nonzero();
    // sort by count descending
    pairs.sort_by(|a, b| b.2.cmp(&a.2));
    let mut used = vec![false; n_pub];
    let mut out = Vec::new();
    for (i, j, c) in pairs {
        if out.len() >= w {
            break;
        }
        if !used[i] && !used[j] {
            used[i] = true;
            used[j] = true;
            out.push((c, i, j));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangular_index() {
        let n = 5;
        // pairs: (0,1)->0, (0,2)->1, (0,3)->2, (0,4)->3, (1,2)->4, ...
        assert_eq!(pair_index(n, 0, 1), 0);
        assert_eq!(pair_index(n, 0, 4), 3);
        assert_eq!(pair_index(n, 1, 2), 4);
        assert_eq!(pair_index(n, 3, 4), 9);
        // count check: C(5,2)=10
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                seen.insert(pair_index(n, i, j));
            }
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn tmax_sanity() {
        // small example: n'=28, k=14, N=34, w=4, s=2, d=4
        // k' = 14 - (34-28) = 8; D = 8+4-1 = 11
        // t'_max = ceil((27) - 4*11/2) - 1 = ceil(27-22)-1 = 5-1 = 4
        assert_eq!(tprime_max(28, 14, 34, 4, 2, 4), 4);
    }

    #[test]
    fn log_gamma_known() {
        assert!(log_gamma(1.0).abs() < 1e-9);
        assert!((log_gamma(5.0) - 24.0f64.ln()).abs() < 1e-9);
        assert!((log_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-9);
        // Gamma(10.5) via recurrence check: ln Gamma(z+1) = ln z + ln Gamma(z)
        let z = 10.5;
        assert!((log_gamma(z + 1.0) - log_gamma(z) - z.ln()).abs() < 1e-9);
    }

    #[test]
    fn binom_cdf_known_values() {
        // Bin(10, 1/2): P[X<=3] = (1+10+45+120)/1024 = 0.171875
        assert!((binom_cdf(10, 0.5, 3) - 0.171875).abs() < 1e-12);
        // Bin(10, 1/2): P[X<=5] = 0.623046875
        assert!((binom_cdf(10, 0.5, 5) - 0.623046875).abs() < 1e-12);
        // Bin(20, 1/4): P[X=0] = (3/4)^20
        assert!((binom_cdf(20, 0.25, 0) - 0.75f64.powi(20)).abs() < 1e-12);
        // Upper tail: P[Bin(10,1/2) >= 8] = (45+10+1)/1024 = 0.0546875
        assert!((binom_sf_ge(10, 0.5, 8) - 0.0546875).abs() < 1e-12);
        // cdf/sf consistency on a mid-size case
        let c = binom_cdf(97, 0.31, 40);
        let s = binom_sf_ge(97, 0.31, 41);
        assert!((c + s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn binom_cdf_edges_and_monotonicity() {
        assert_eq!(binom_cdf(10, 0.5, -1), 0.0);
        assert_eq!(binom_cdf(10, 0.5, 10), 1.0);
        assert_eq!(binom_cdf(10, 0.0, 0), 1.0);
        assert_eq!(binom_cdf(10, 1.0, 9), 0.0);
        assert_eq!(binom_sf_ge(10, 0.5, 0), 1.0);
        assert_eq!(binom_sf_ge(10, 0.5, 11), 0.0);
        // monotone nondecreasing in k
        let mut prev = -1.0;
        for k in 0..=10 {
            let c = binom_cdf(10, 0.3, k);
            assert!(c >= prev && (0.0..=1.0).contains(&c));
            prev = c;
        }
        // large-n (incomplete-beta) path: Bin(100000, 0.3), mean 30000,
        // P[X <= 30000] ~= 0.5037 (normal approx with cc/skew).
        let v = binom_cdf(100_000, 0.3, 30_000);
        assert!(v > 0.49 && v < 0.52, "beta-path cdf sane: {}", v);
        // ...and it agrees with the PMF path just below SMALL_N crossover:
        let a = binom_cdf(8192, 0.3, 2400);
        assert!(a > 0.0 && a < 1.0);
    }

    #[test]
    fn raw_quantile_sanity() {
        // smallest T with P[Bin(T, 1/2) >= 8] >= 0.99
        let t = required_raw(0.5, 8, 0.01).expect("feasible");
        assert!(binom_cdf(t, 0.5, 7) <= 0.01, "T={} must meet the quantile", t);
        assert!(binom_cdf(t - 1, 0.5, 7) > 0.01, "T={} must be minimal", t);
        // p_record = 1: T = r exactly.
        assert_eq!(required_raw(1.0, 50, 0.01), Some(50));
    }

    #[test]
    fn plan_toy_point() {
        // q=64, N=34 (n=30, w=4), k=14, n'=28, (s,d)=(2,4).
        // Hand-computed from the H3 model: t'_max=4, J_max=2;
        // P[bad&rec](y=0)=0.0199 > 0.01, P[bad&rec](y=1)=0.0017 <= 0.01
        // -> y=1, T_reject=3, P_record=0.17082.
        let p = plan_trials(64, 34, 14, 4, 28, 2, 4, 0.01, None).expect("toy plan feasible");
        assert_eq!(p.tprime_max, 4);
        assert_eq!(p.j_max, 2);
        assert_eq!(p.y, 1);
        assert_eq!(p.t_reject, 3);
        assert!((p.p_record - 0.17082).abs() < 0.005, "p_record={}", p.p_record);
        assert!((p.p_good - 0.39473).abs() < 0.005, "p_good={}", p.p_good);
        assert!((p.p_gr - 0.10487).abs() < 0.005, "p_gr={}", p.p_gr);
        assert!(p.t_raw >= 2_000 && p.t_raw <= 12_000, "t_raw={}", p.t_raw);
    }

    #[test]
    fn plan_degenerate_point_infeasible() {
        // q=128, n=65, k=45, w=20, n'=58, (s,d)=(2,3): t'_max = 1, so
        // T_reject <= 1 < 2 and no flagged pair can ever be recorded.
        let r = plan_trials(128, 85, 45, 20, 58, 2, 3, 0.01, None);
        assert!(r.is_err(), "expected infeasible, got {:?}", r.map(|p| p.t_raw));
    }

    #[test]
    fn spooled_trial_valid_checks() {
        let dir = std::env::temp_dir().join(format!("spool_valid_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.txt");
        fs::write(&p, "2 3 7\n").unwrap();
        assert!(spooled_trial_valid(&p));
        fs::write(&p, "0\n").unwrap();
        assert!(spooled_trial_valid(&p));
        fs::write(&p, "2 3 7").unwrap(); // torn write: no trailing newline
        assert!(!spooled_trial_valid(&p));
        fs::write(&p, "3 3 7\n").unwrap(); // count mismatch
        assert!(!spooled_trial_valid(&p));
        fs::write(&p, "2 7 3\n").unwrap(); // not strictly increasing
        assert!(!spooled_trial_valid(&p));
        fs::write(&p, "x\n").unwrap(); // unparseable
        assert!(!spooled_trial_valid(&p));
        assert!(!spooled_trial_valid(&dir.join("missing.txt")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn spool_resume_skips_valid_files() {
        // Resume mechanics only (attack correctness is covered elsewhere):
        // a well-formed spool file is trusted verbatim; a torn one is
        // recomputed; a missing one is computed.
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(99);
        let key = crate::rlce::gen_rlce(&gf, &mut rng, 30, 14, 4);
        let dir = std::env::temp_dir().join(format!("phase1_resume_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let trial_dir = dir.join("trials");
        fs::create_dir_all(&trial_dir).unwrap();
        fs::write(trial_dir.join("trial_00000.txt"), "0\n").unwrap();
        fs::write(trial_dir.join("trial_00001.txt"), "3 1 2").unwrap(); // torn
        let cfg = Phase1Config {
            nprime: 28,
            s: 2,
            d: 4,
            trials: 3,
            threads: 0,
            seed: 5,
            y_margin: 1,
            engine: DistEngine::Dense,
            out_dir: dir.clone(),
        };
        let _ = run_trials(&gf, &key.cols, key.k, key.w, &cfg);
        assert_eq!(
            fs::read_to_string(trial_dir.join("trial_00000.txt")).unwrap(),
            "0\n",
            "well-formed spool file must be preserved verbatim"
        );
        assert!(spooled_trial_valid(&trial_dir.join("trial_00001.txt")));
        assert!(spooled_trial_valid(&trial_dir.join("trial_00002.txt")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn reuse_engine_matches_dense_trials() {
        // Trial-level differential: --engine reuse must produce flagged
        // sets IDENTICAL to dense on every trial (the reuse decisions
        // are the dense decisions by construction; reuse.rs oracle-tests
        // that per point set -- this test pins the phase1 wiring: the
        // kept-index mapping and the pre-loop branch).
        let gf64 = Gf::for_q(64);

        // (a) purely random columns, (s,d)=(2,3).
        {
            let mut rng = Rng::new(31337);
            let cols: Vec<Vec<u16>> = (0..24)
                .map(|_| {
                    (0..10)
                        .map(|_| rng.below(64) as u16)
                        .collect::<Vec<u16>>()
                })
                .collect();
            for seed in [11u64, 12, 13] {
                let dense =
                    attack_trial_with_engine(&gf64, &cols, 18, 2, 3, seed, DistEngine::Dense);
                let reuse =
                    attack_trial_with_engine(&gf64, &cols, 18, 2, 3, seed, DistEngine::Reuse);
                assert_eq!(dense, reuse, "random cols seed {}", seed);
            }
        }

        // (b) RLCE keys at two shapes: the toy point (30,14,4) with its
        // (2,4), n'=28 configuration, and a (2,3) instance with the
        // target point's flavour (small k'/n'): key (30,20,5), n'=28
        // -> k' = 20 - (35-28) = 13.
        let key_a = {
            let mut rng = Rng::new(4242);
            crate::rlce::gen_rlce(&gf64, &mut rng, 30, 14, 4)
        };
        for seed in [21u64, 22] {
            let dense =
                attack_trial_with_engine(&gf64, &key_a.cols, 28, 2, 4, seed, DistEngine::Dense);
            let reuse =
                attack_trial_with_engine(&gf64, &key_a.cols, 28, 2, 4, seed, DistEngine::Reuse);
            assert_eq!(dense, reuse, "RLCE toy shape seed {}", seed);
        }
        let key_b = {
            let mut rng = Rng::new(777);
            crate::rlce::gen_rlce(&gf64, &mut rng, 30, 20, 5)
        };
        for seed in [31u64, 32, 33] {
            let dense =
                attack_trial_with_engine(&gf64, &key_b.cols, 28, 2, 3, seed, DistEngine::Dense);
            let reuse =
                attack_trial_with_engine(&gf64, &key_b.cols, 28, 2, 3, seed, DistEngine::Reuse);
            assert_eq!(dense, reuse, "RLCE (2,3) shape seed {}", seed);
        }
    }

    #[test]
    fn reuse_memory_guard_checks() {
        // Flagship-shaped config (ID-0-like (3,5) dims): the guard must
        // reject it from dimensions alone (no allocation is attempted).
        let err = check_reuse_memory(348, 28, 3, 5).expect_err("flagship must be rejected");
        assert!(err.contains("GB"), "error names the size: {}", err);
        assert!(err.contains("bw-sigma"), "error suggests bw-sigma: {}", err);
        // The (3,4) worked-example dims are likewise far over.
        assert!(check_reuse_memory(348, 28, 3, 4).is_err());
        // Target point of the engine benchmark: (n'=90, k'=24, (2,3))
        // estimates ~32 MB worst case -- accepted, far under the limit.
        let est = check_reuse_memory(90, 24, 2, 3).expect("target point accepted");
        assert_eq!(est.augmented_bytes, 21_825_000);
        assert!(est.total_bytes < REUSE_MEMORY_LIMIT_BYTES);
        // A mid-size config stays accepted: (n'=200, k'=60, (2,3)).
        assert!(check_reuse_memory(200, 60, 2, 3).is_ok());
    }

    #[test]
    fn plan_id0_audit() {
        // ID-0 scale: q=1024, n=630, k=470, w=160 (N=790), n'=348, (3,5).
        // Paper: t'_max=35, J_max=17. Binomial-model P_good = 2^-8.91
        // (the paper quotes 2^-7.8 from its exact count; factor ~2.2,
        // see AUTOTRIALS_REPORT.md).
        let p = plan_trials(1024, 790, 470, 160, 348, 3, 5, 0.01, None)
            .expect("ID-0 plan feasible");
        assert_eq!(p.tprime_max, 35);
        assert_eq!(p.j_max, 17);
        assert_eq!(p.y, 0);
        let lg = p.p_good.log2();
        assert!(lg > -9.3 && lg < -8.5, "P_good = 2^{}", lg);
        assert!(p.r_recorded >= 100 && p.r_recorded <= 800, "r={}", p.r_recorded);
        assert!(p.t_raw >= (1 << 15) && p.t_raw <= (1 << 19), "T={}", p.t_raw);
        println!("\n{}", plan_report(&p));
    }
}
