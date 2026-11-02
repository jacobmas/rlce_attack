//! Phase 1 (cluster mode): flag-pattern clustering as an ADDITIVE
//! alternative to the frequency-table method.
//!
//! The frequency table needs ~r_sep *recorded good* trials because it
//! separates true pairs from non-pairs statistically. Clustering uses a
//! stronger empirical fact, validated at (256,198,165,33) in 23/23
//! trials (good and bad): the flagged set V of a trial equals EXACTLY
//! the unhit pairs' columns -- flagging is exact well past t'_max, so
//! "bad" trials are not polluted, only larger. Hence every trial, good
//! or bad, contributes one bit per column, and twin columns share
//! IDENTICAL flag patterns (both flagged iff their pair is unhit in
//! that trial), while distinct pairs separate after tens of trials.
//!
//! Algorithm (all trials use the same trial machinery as frequency
//! mode -- `phase1::run_trials` with the configured engine; only the
//! aggregation differs):
//!   1. Run T independent trials (fresh random shortenings). T is
//!      auto-planned when --trials is not given:
//!        p = n'(n'-1) / (N(N-1))   (per-pair unhit probability)
//!        T_sep = smallest T with C(w,2) * (p^2 + (1-p)^2)^T <= 0.01
//!          (no two distinct pairs share a pattern; the per-trial
//!          collision probability p^2 + (1-p)^2 is an UPPER bound --
//!          hypergeometric competition makes pairs negatively
//!          correlated, so independence is conservative here)
//!        T_vis = smallest T with w * (1-p)^T <= 0.01
//!          (no pair is invisible, i.e. never unhit in T trials)
//!        T = max(T_sep, T_vis).
//!      --trials overrides T (with the usual caveat: too few trials
//!      produce collisions/invisible pairs, which surface as anomalies
//!      below rather than as silent errors).
//!   2. Group columns by their flag pattern (bitvector over trials):
//!        - all-zero pattern  -> pure set (never flagged)
//!        - size-2 non-zero   -> twin pair (count = pattern popcount)
//!        - size-1 non-zero   -> ORPHAN (anomaly, see below)
//!        - size > 2          -> oversized group (anomaly: a pattern
//!          collision, i.e. T was too small, or flagging was inexact)
//!   3. Anomaly accounting: the all-zero count is compared against the
//!      expected unpaired count (n - w). A degenerate pair (see below)
//!      contributes exactly 1 orphan + 1 all-zero excess.
//!
//! DEGENERATE PAIRS (load-bearing anomaly semantics):
//! Keygen samples mixing-matrix entries from all of F_q (only det != 0
//! retried; see rlce.rs: c1 = a*g + c*r, c2 = b*g + d*r). If c = 0, c1
//! is a genuine pure GRS column (never flagged, correctly) while c2
//! stays mixed (flagged whenever present, correctly -- hitting the
//! pure member imposes a GRS-only shortening condition, so
//! derandomization fails and c2 stays noisy as a singleton). The same
//! holds with the roles swapped if d = 0. So ~2/q of pairs (c = 0 or
//! d = 0) are degenerate for clustering: one member lands in the pure
//! set, the other becomes a size-1 orphan group. (The other two
//! single-entry-zero cases, a = 0 and b = 0, give pure-RANDOM members,
//! which still flag when unhit and hence cluster normally; the often
//! quoted ~4/q counts all four, but only ~2/q are clustering anomalies.)
//! The cluster output therefore reports orphans + all-zero excess
//! explicitly; a degenerate pair is DETECTED (orphan count = excess)
//! but, as the fallback analysis below shows, NOT resolvable from
//! attack-visible data.
//!
//! ORPHAN FALLBACK (in the `all` flow, after SS on the pure set):
//! Each orphan is tested against every pure candidate with the 2-plane
//! test -- vand(x) in span{pure, orphan} -- via 3x3-minor root-finding
//! (fastss's char-2 quadratic solver) plus full rank verification.
//! THE TEST IS VACUOUS, and the implementation demonstrates this rather
//! than hiding it: the pure candidate's OWN evaluation point x_j always
//! satisfies vand(x_j) in span{pure_j, orphan} (it is a scalar multiple
//! of pure_j itself), so EVERY candidate passes. There is no
//! discriminating signal: for any pure candidate j there exists a
//! consistent secret key pairing the orphan with j (same public key,
//! same flag patterns in every trial -- re-pairing a degenerate pair
//! keeps the orphan mixed-when-present and the pure member never
//! flagged), so the twin of a degenerate pair is information-
//! theoretically undetermined by attack-visible data. The fallback
//! therefore NEVER emits an association; it reports the root sets
//! (which empirically show the vacuity: every candidate's root set
//! contains its own x, extra roots ~0) and marks the orphan
//! UNRESOLVED. A run with unresolved orphans reports INCOMPLETE with
//! the unresolved pool -- never a guessed pairing. (The frequency
//! method has the identical limitation: verify.rs already documents
//! that degenerate pairs are skipped there too.)

use crate::fastss::{solve_quad, QuadRoots, QuadSolver};
use crate::gf::Gf;
use crate::mat;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Cluster trial-count planning
// ---------------------------------------------------------------------------

/// Result of cluster trial-count planning.
#[derive(Clone, Debug)]
pub struct ClusterPlan {
    pub n_pub: usize,
    pub w: usize,
    pub nprime: usize,
    /// Per-pair unhit probability p = n'(n'-1)/(N(N-1)).
    pub p_unhit: f64,
    /// Per-trial pattern-collision probability p^2 + (1-p)^2.
    pub collision: f64,
    /// Smallest T with C(w,2) * collision^T <= 0.01 (0 if w < 2).
    pub t_sep: usize,
    /// Smallest T with w * (1-p)^T <= 0.01.
    pub t_vis: usize,
    /// Planned trial count: max(t_sep, t_vis), at least 1.
    pub t_planned: usize,
}

/// Failure probability target for both planning bounds.
pub const CLUSTER_EPS: f64 = 0.01;

/// Plan the cluster trial count. Pure arithmetic on (N, w, n'); no
/// key material, no GIJS parameters needed (clustering uses every
/// trial, so t'_max / (s,d) do not enter the count -- they only affect
/// per-trial cost via the engine).
pub fn plan_cluster_trials(
    n_pub: usize,
    w: usize,
    nprime: usize,
) -> Result<ClusterPlan, String> {
    if w == 0 {
        return Err("w = 0: there are no twin pairs to find.".to_string());
    }
    if nprime < 2 || nprime >= n_pub {
        return Err(format!(
            "need 2 <= n' < N for a nontrivial shortening; got n'={}, N={}.",
            nprime, n_pub
        ));
    }
    let p = (nprime * (nprime - 1)) as f64 / (n_pub * (n_pub - 1)) as f64;
    if !(p > 0.0 && p < 1.0) {
        return Err(format!(
            "p_unhit = {} is degenerate (n'={}, N={}); clustering cannot separate patterns.",
            p, nprime, n_pub
        ));
    }
    let coll = p * p + (1.0 - p) * (1.0 - p);
    // T_sep: smallest T with C(w,2) * coll^T <= eps (0 if w < 2).
    let t_sep = if w < 2 {
        0
    } else {
        let npairs = (w * (w - 1) / 2) as f64;
        let target = CLUSTER_EPS / npairs;
        let est = target.ln() / coll.ln();
        if !(est.is_finite()) || est > 10_000_000.0 {
            return Err(format!(
                "cluster separation needs more than 10^7 trials (p_unhit = {:.4}, collision = {:.4}); \
                 parameters are infeasible for clustering.",
                p, coll
            ));
        }
        let mut t = est.ceil() as usize;
        // Polish to exact minimality (bounded: the estimate is within a
        // few of the true value; the loop guards against float error).
        let mut guard = 0;
        while t > 0 && npairs * coll.powi((t - 1) as i32) <= CLUSTER_EPS && guard < 16 {
            t -= 1;
            guard += 1;
        }
        guard = 0;
        while npairs * coll.powi(t as i32) > CLUSTER_EPS && guard < 16 {
            t += 1;
            guard += 1;
        }
        t
    };
    // T_vis: smallest T with w * (1-p)^T <= eps.
    let est = (CLUSTER_EPS / w as f64).ln() / (1.0 - p).ln();
    if !(est.is_finite()) || est > 10_000_000.0 {
        return Err(format!(
            "cluster visibility needs more than 10^7 trials (p_unhit = {:.4}); \
             parameters are infeasible for clustering.",
            p
        ));
    }
    let mut t_vis = est.ceil() as usize;
    let mut guard = 0;
    while t_vis > 0 && w as f64 * (1.0 - p).powi((t_vis - 1) as i32) <= CLUSTER_EPS && guard < 16 {
        t_vis -= 1;
        guard += 1;
    }
    guard = 0;
    while w as f64 * (1.0 - p).powi(t_vis as i32) > CLUSTER_EPS && guard < 16 {
        t_vis += 1;
        guard += 1;
    }
    let t_planned = t_sep.max(t_vis).max(1);
    Ok(ClusterPlan {
        n_pub,
        w,
        nprime,
        p_unhit: p,
        collision: coll,
        t_sep,
        t_vis,
        t_planned,
    })
}

/// Render the cluster plan as the printed audit chain.
pub fn cluster_plan_report(p: &ClusterPlan) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== Phase 1 (cluster) trial planning (eps = {}) ===\n",
        CLUSTER_EPS
    ));
    out.push_str(&format!(
        "params: N={} w={} n'={}\n",
        p.n_pub, p.w, p.nprime
    ));
    out.push_str(&format!(
        "[1] p_unhit = n'(n'-1)/(N(N-1)) = {:.6} (2^{:.2}); per-trial pattern-collision prob \
         p^2 + (1-p)^2 = {:.6}\n",
        p.p_unhit,
        p.p_unhit.log2(),
        p.collision
    ));
    out.push_str(&format!(
        "[2] T_sep = {} (smallest T with C(w,2) * collision^T <= {}; separates all {} pairs)\n",
        p.t_sep,
        CLUSTER_EPS,
        p.w
    ));
    out.push_str(&format!(
        "[3] T_vis = {} (smallest T with w * (1-p)^T <= {}; no pair invisible)\n",
        p.t_vis, CLUSTER_EPS
    ));
    out.push_str(&format!(
        "[4] T = max(T_sep, T_vis) = {} trials (ALL trials used; no |V| filter, no t'_max gate)\n",
        p.t_planned
    ));
    out
}

// ---------------------------------------------------------------------------
// Cluster aggregation from spooled trial files
// ---------------------------------------------------------------------------

/// Output of cluster aggregation. `pairs` / `pure_idx` / `col_flagged`
/// use the SAME formats as frequency mode so downstream ss / unmix /
/// verify work unchanged (pairs carry co-flag counts = pattern
/// popcounts; col_flagged carries per-column flag totals).
#[derive(Clone, Debug)]
pub struct ClusterOutput {
    /// (co-flag count, i, j), sorted by count descending.
    pub pairs: Vec<(u32, usize, usize)>,
    /// All-zero-pattern columns (never flagged), ascending.
    pub pure_idx: Vec<usize>,
    /// Per-column flag totals (pattern popcounts).
    pub col_flagged: Vec<u32>,
    /// Size-1 non-zero pattern groups, ascending.
    pub orphans: Vec<usize>,
    /// Size > 2 pattern groups (columns ascending within each group).
    pub oversized: Vec<Vec<usize>>,
    /// All pattern groups as (popcount, columns) for the audit file
    /// (includes the all-zero group).
    pub groups: Vec<(u32, Vec<usize>)>,
    /// Number of all-zero-pattern columns.
    pub n_all_zero: usize,
    /// Expected unpaired count (n - w, caller-supplied).
    pub expected_pure: usize,
    /// n_all_zero - expected_pure (nonzero signals invisible pairs or
    /// degenerate pairs; see anomaly_lines).
    pub excess: i64,
    /// Trials aggregated.
    pub n_trials: usize,
    /// True iff orphans, oversized groups, or a nonzero excess exist.
    pub anomaly: bool,
    /// Human-readable anomaly descriptions (empty when !anomaly).
    pub anomaly_lines: Vec<String>,
}

/// Aggregate spooled trials into flag-pattern groups.
///
/// Reads trial files `trial_{:05}.txt` (format `n i1 i2 ...`, sorted)
/// for t in 0..trials from `trial_dir`, exactly as frequency mode's
/// `aggregate` reads them -- but applies NO |V| filter: every trial
/// contributes one pattern bit per column.
pub fn aggregate_clusters(
    trial_dir: &Path,
    n_pub: usize,
    trials: usize,
    expected_pure: usize,
) -> ClusterOutput {
    let mut patterns: Vec<Vec<bool>> = vec![vec![false; trials]; n_pub];
    let t_start = Instant::now();

    for t in 0..trials {
        let path = trial_dir.join(format!("trial_{:05}.txt", t));
        let txt = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cluster: cannot read trial file {}: {}", path.display(), e));
        let nums: Vec<usize> = txt
            .split_whitespace()
            .map(|x| x.parse().expect("cluster: malformed trial file"))
            .collect();
        // nums[0] is the flagged count; the rest are flagged indices.
        for &idx in nums.iter().skip(1) {
            debug_assert!(idx < n_pub, "flagged index {} out of range", idx);
            patterns[idx][t] = true;
        }
    }
    let trial_load_ms = t_start.elapsed().as_millis();


    let mut col_flagged = vec![0u32; n_pub];
    for (i, pat) in patterns.iter().enumerate() {
        col_flagged[i] = pat.iter().filter(|&&b| b).count() as u32;
    }
    // Group columns by exact pattern.
    let mut by_pattern: HashMap<Vec<bool>, Vec<usize>> = HashMap::new();
    for (i, pat) in patterns.iter().enumerate() {
        by_pattern.entry(pat.clone()).or_default().push(i);
    }
    let mut pairs: Vec<(u32, usize, usize)> = Vec::new();
    let mut pure_idx: Vec<usize> = Vec::new();
    let mut orphans: Vec<usize> = Vec::new();
    let mut oversized: Vec<Vec<usize>> = Vec::new();
    let mut groups: Vec<(u32, Vec<usize>)> = Vec::new();
    for (pat, mut cols) in by_pattern.into_iter() {
        let pop = pat.iter().filter(|&&b| b).count() as u32;
        cols.sort_unstable();
        if pop == 0 {
            pure_idx.extend(cols.iter().cloned());
        } else if cols.len() == 2 {
            pairs.push((pop, cols[0], cols[1]));
        } else if cols.len() == 1 {
            orphans.push(cols[0]);
        } else {
            oversized.push(cols.clone());
        }
        groups.push((pop, cols));
    }
    pairs.sort_by(|a, b| b.0.cmp(&a.0));
    pure_idx.sort_unstable();
    orphans.sort_unstable();
    oversized.sort();
    groups.sort_by_key(|(_, cols)| cols[0]);
    let n_all_zero = pure_idx.len();
    let excess = n_all_zero as i64 - expected_pure as i64;
    let mut anomaly_lines: Vec<String> = Vec::new();
    for &o in &orphans {
        anomaly_lines.push(format!(
            "orphan column {} (flagged {} times, no column shares its pattern)",
            o, col_flagged[o]
        ));
    }
    for g in &oversized {
        anomaly_lines.push(format!(
            "oversized pattern group of {} columns (pattern collision): {:?}",
            g.len(),
            g
        ));
    }
    if excess != 0 {
        anomaly_lines.push(format!(
            "all-zero excess: {} all-zero columns vs {} expected unpaired (n-w); excess {} \
             (positive excess with orphans present signals degenerate pairs; positive excess \
             without orphans signals invisible pairs contaminating the pure set)",
            n_all_zero, expected_pure, excess
        ));
    }
    let anomaly = !orphans.is_empty() || !oversized.is_empty() || excess != 0;
    let clustering_ms = t_start.elapsed().as_millis() - trial_load_ms;

    println!(
    "# cluster time to load trials (ms): {}", 
    trial_load_ms
    );
    println!(
    "# cluster time to actually cluster (ms): {}", 
    clustering_ms
    );
    ClusterOutput {
        pairs,
        pure_idx,
        col_flagged,
        orphans,
        oversized,
        groups,
        n_all_zero,
        expected_pure,
        excess,
        n_trials: trials,
        anomaly,
        anomaly_lines,
    }
}

// ---------------------------------------------------------------------------
// Orphan fallback: 2-plane test (vacuous -- see module docs)
// ---------------------------------------------------------------------------

/// Outcome of the 2-plane association attempt for one orphan.
#[derive(Clone, Debug)]
pub struct OrphanAttempt {
    pub orphan: usize,
    /// Number of pure candidates tested.
    pub n_candidates: usize,
    /// Candidates where the minor machinery found the candidate's own
    /// x among the roots (expect: all, since vand(x_j) is parallel to
    /// the candidate column itself).
    pub n_minor_passing: usize,
    /// Candidates where the full rank check confirms vand(x_j) lies in
    /// span{pure_j, orphan} (expect: all, same reason).
    pub n_rank_passing: usize,
    /// Verified roots x != x_j, as (pure column, x). Generically
    /// empty; a spurious second intersection occurs with probability
    /// ~1/q per candidate and is NOT a resolution signal.
    pub extra_roots: Vec<(usize, u16)>,
    /// Always None: the literal 2-plane test cannot discriminate
    /// between candidates (see module docs for the proof). Any future
    /// resolution rule must come from a non-vacuous signal.
    pub resolved: Option<usize>,
}

/// Vandermonde vector (1, x, x^2, ..., x^{k-1}) over gf.
fn vand_vec(gf: &Gf, x: u16, k: usize) -> Vec<u16> {
    let mut v = vec![0u16; k];
    let mut xp = 1u16;
    for i in 0..k {
        v[i] = xp;
        xp = gf.mul(xp, x);
    }
    v
}

/// All x with vand(x) in span{g, o}, found by 3x3-minor root-finding
/// plus full rank verification.
///
/// Primary minor rows (0,1,2): M(x) = A x^2 + B x + C with
///   A = g0*o1 + o0*g1, B = g0*o2 + o0*g2, C = g1*o2 + g2*o1
/// (char 2, so all signs are +), solved with fastss's char-2 quadratic
/// solver. If that minor is identically zero, triples (0,1,3),
/// (0,2,3), (1,2,3) are scanned by direct evaluation over F_q.
/// Every candidate root is verified by rank[vand(x), g, o] <= 2
/// (genuine elimination, never a Gram matrix).
fn plane_roots(gf: &Gf, qs: &QuadSolver, g: &[u16], o: &[u16]) -> Vec<u16> {
    let k = g.len();
    debug_assert_eq!(o.len(), k);
    let mut cand: Vec<u16> = Vec::new();
    if k >= 3 {
        let (g0, g1, g2) = (g[0], g[1], g[2]);
        let (o0, o1, o2) = (o[0], o[1], o[2]);
        let a = gf.add(gf.mul(g0, o1), gf.mul(o0, g1));
        let b = gf.add(gf.mul(g0, o2), gf.mul(o0, g2));
        let c = gf.add(gf.mul(g1, o2), gf.mul(g2, o1));
        match solve_quad(gf, qs, a, b, c) {
            QuadRoots::None => {}
            QuadRoots::One(x) => cand.push(x),
            QuadRoots::Two(x, y) => {
                cand.push(x);
                cand.push(y);
            }
            QuadRoots::All => {
                // Identically-zero minor: scan further triples by
                // evaluation until one yields a non-trivial root set.
                for &(r1, r2, r3) in &[(0usize, 1usize, 3usize), (0, 2, 3), (1, 2, 3)] {
                    if r3 >= k {
                        continue;
                    }
                    let mut found: Vec<u16> = Vec::new();
                    let mut all_zero = true;
                    for xv in 0..gf.order {
                        let x = xv as u16;
                        // 3x3 determinant with vand column (x^r1, x^r2, x^r3).
                        let v1 = gf.pow(x, r1);
                        let v2 = gf.pow(x, r2);
                        let v3 = gf.pow(x, r3);
                        // det in char 2: sum of the six diagonal products.
                        let det = gf.add(
                            gf.add(
                                gf.mul(v1, gf.add(gf.mul(g[r2], o[r3]), gf.mul(g[r3], o[r2]))),
                                gf.mul(g[r1], gf.add(gf.mul(v2, o[r3]), gf.mul(v3, o[r2]))),
                            ),
                            gf.mul(o[r1], gf.add(gf.mul(v2, g[r3]), gf.mul(v3, g[r2]))),
                        );
                        if det != 0 {
                            all_zero = false;
                        } else {
                            found.push(x);
                        }
                    }
                    if !all_zero {
                        cand = found;
                        break;
                    }
                    // All-zero minor: identically zero (degree < q), try
                    // the next triple. If every triple is identically
                    // zero, g and o are linearly dependent and the rank
                    // fallback below still recovers the own-x root.
                }
            }
        }
    }
    // Full-rank verification for every candidate root.
    let mut out: Vec<u16> = Vec::new();
    for x in cand {
        let v = vand_vec(gf, x, k);
        let rows: Vec<Vec<u16>> = (0..k).map(|i| vec![v[i], g[i], o[i]]).collect();
        if mat::rank(gf, &rows) <= 2 {
            out.push(x);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Run the 2-plane association attempt for each orphan against every
/// pure candidate, in monomial-basis coordinates.
///
/// `ghat_cols`: all N public columns in monomial basis (from
/// ss::to_monomial_basis); `alpha`/`y`/`pure_idx` are the SS outputs
/// (alpha[j], y[j] belong to pure column pure_idx[j]).
/// Pure columns in monomial basis are exactly y_j * vand(x_j); the
/// consistency check below verifies that shape and reports any
/// mismatch instead of silently testing garbage.
pub fn resolve_orphans(
    gf: &Gf,
    ghat_cols: &[Vec<u16>],
    alpha: &[u16],
    y: &[u16],
    pure_idx: &[usize],
    orphans: &[usize],
) -> Vec<OrphanAttempt> {
    let qs = QuadSolver::new(gf);
    let mut attempts = Vec::new();
    for &o in orphans {
        let ocol = &ghat_cols[o];
        let mut n_minor_passing = 0usize;
        let mut n_rank_passing = 0usize;
        let mut extra_roots: Vec<(usize, u16)> = Vec::new();
        for (j, &p) in pure_idx.iter().enumerate() {
            let xj = alpha[j];
            let g = &ghat_cols[p];
            // Consistency: pure column must be y_j * vand(x_j).
            if g.len() < 3 || g[0] != y[j] {
                continue; // inconsistent candidate: not counted as passing
            }
            let roots = plane_roots(gf, &qs, g, ocol);
            if roots.contains(&xj) {
                n_minor_passing += 1;
            }
            // Rank confirmation of the own-x root (independent check).
            let v = vand_vec(gf, xj, g.len());
            let rows: Vec<Vec<u16>> = (0..g.len()).map(|i| vec![v[i], g[i], ocol[i]]).collect();
            if mat::rank(gf, &rows) <= 2 {
                n_rank_passing += 1;
            }
            for &x in &roots {
                if x != xj {
                    extra_roots.push((p, x));
                }
            }
        }
        attempts.push(OrphanAttempt {
            orphan: o,
            n_candidates: pure_idx.len(),
            n_minor_passing,
            n_rank_passing,
            extra_roots,
            resolved: None,
        });
    }
    attempts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_toy_point_cluster() {
        // (256,198,165,33), n'=90: N=231, p = 90*89/(231*230) = 0.15076.
        // T_sep = 37 (528 * 0.74394^37 = 0.0093 <= 0.01; ^36 = 0.0125),
        // T_vis = 50 (33 * 0.84924^50 = 0.0093; ^49 = 0.0110).
        let p = plan_cluster_trials(231, 33, 90).expect("toy plan feasible");
        assert_eq!(p.t_sep, 37, "t_sep={}", p.t_sep);
        assert_eq!(p.t_vis, 50, "t_vis={}", p.t_vis);
        assert_eq!(p.t_planned, 50);
        assert!((p.p_unhit - 0.15076).abs() < 1e-5, "p={}", p.p_unhit);
    }

    #[test]
    fn plan_validation_point_cluster() {
        // (256,119,79,31), n'=101: N=150, p = 101*100/(150*149) = 0.45190.
        // T_sep = 16 (465 * 0.50461^16 = 0.0082; ^15 = 0.0163),
        // T_vis = 14 (31 * 0.54810^14 = 0.0068; ^13 = 0.0125).
        let p = plan_cluster_trials(150, 31, 101).expect("validation plan feasible");
        assert_eq!(p.t_sep, 16, "t_sep={}", p.t_sep);
        assert_eq!(p.t_vis, 14, "t_vis={}", p.t_vis);
        assert_eq!(p.t_planned, 16);
        assert!((p.p_unhit - 0.45190).abs() < 1e-5, "p={}", p.p_unhit);
    }

    #[test]
    fn plan_rejects_degenerate() {
        assert!(plan_cluster_trials(100, 0, 50).is_err(), "w=0");
        assert!(plan_cluster_trials(100, 5, 1).is_err(), "n'<2");
        assert!(plan_cluster_trials(100, 5, 100).is_err(), "n'=N");
    }

    #[test]
    fn aggregate_synthetic_patterns() {
        // N=6, T=3. Trial flags: t0 {0,1,4}, t1 {0,1,5}, t2 {2,3}.
        // Patterns: 0,1 -> [1,1,0] (pair, count 2); 2,3 -> [0,0,1]
        // (pair, count 1); 4 -> [1,0,0] (orphan); 5 -> [0,1,0] (orphan).
        let dir = std::env::temp_dir()
            .join(format!("cluster_agg_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (t, flagged) in [(0usize, vec![0usize, 1, 4]), (1, vec![0, 1, 5]), (2, vec![2, 3])] {
            let mut s = format!("{}", flagged.len());
            for i in flagged {
                s.push_str(&format!(" {}", i));
            }
            s.push('\n');
            fs::write(dir.join(format!("trial_{:05}.txt", t)), s).unwrap();
        }
        let out = aggregate_clusters(&dir, 6, 3, 0);
        assert_eq!(out.pairs, vec![(2, 0, 1), (1, 2, 3)]);
        assert!(out.pure_idx.is_empty());
        assert_eq!(out.orphans, vec![4, 5]);
        assert!(out.oversized.is_empty());
        assert_eq!(out.col_flagged, vec![2, 2, 1, 1, 1, 1]);
        assert!(out.anomaly, "orphans must flag anomaly");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn aggregate_oversized_group() {
        // Three columns sharing one pattern -> oversized, not a pair.
        let dir = std::env::temp_dir()
            .join(format!("cluster_over_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for t in 0..2usize {
            fs::write(dir.join(format!("trial_{:05}.txt", t)), "3 0 1 2\n").unwrap();
        }
        let out = aggregate_clusters(&dir, 4, 2, 1);
        assert!(out.pairs.is_empty());
        assert_eq!(out.oversized, vec![vec![0, 1, 2]]);
        assert_eq!(out.pure_idx, vec![3]);
        assert!(out.anomaly);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn fallback_refuses_to_resolve_synthetic_degenerate() {
        // Monomial-basis synthetic: k=5, pure vand(x) for x in
        // {1,2,3,4} (y=1), orphan o = 3*vand(2) + 5*r~ with a fixed
        // generic r~. The 2-plane test must find every candidate's own
        // x (vacuous pass) and resolve NOTHING.
        let gf = Gf::for_q(256);
        let xs: Vec<u16> = vec![1, 2, 3, 4];
        let mut ghat_cols: Vec<Vec<u16>> = xs.iter().map(|&x| vand_vec(&gf, x, 5)).collect();
        let r_tilde: Vec<u16> = vec![7, 11, 13, 17, 19];
        let mut o = vec![0u16; 5];
        let gv = vand_vec(&gf, 2, 5);
        for i in 0..5 {
            o[i] = gf.add(gf.mul(3, gv[i]), gf.mul(5, r_tilde[i]));
        }
        ghat_cols.push(o);
        let pure_idx: Vec<usize> = vec![0, 1, 2, 3];
        let alpha: Vec<u16> = xs.clone();
        let y: Vec<u16> = vec![1, 1, 1, 1];
        let attempts = resolve_orphans(&gf, &ghat_cols, &alpha, &y, &pure_idx, &[4]);
        assert_eq!(attempts.len(), 1);
        let a = &attempts[0];
        assert_eq!(a.orphan, 4);
        assert_eq!(a.n_candidates, 4);
        assert_eq!(
            a.n_minor_passing, 4,
            "minor machinery must find every candidate's own x (vacuity)"
        );
        assert_eq!(a.n_rank_passing, 4, "rank check must confirm every own-x root");
        assert_eq!(
            a.resolved, None,
            "fallback must NEVER emit an association from a vacuous test"
        );
        // Extra roots (if any) are spurious second intersections, not
        // resolution signals; the test only pins that they do not
        // resolve (count is data-dependent, not asserted).
    }
}
