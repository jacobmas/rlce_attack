//! Hold-out / span tests for the flag-none investigation.
//!
//! Question these tests answer (see the 2026-10-25 thread): in the
//! flag-none regime, is the jet matrix's rowspan the *whole* space
//! (so nothing could ever be flagged), or is it a proper subspace
//! that happens to contain every RLCE column (dirty *and* clean) while
//! a truly independent vector stays outside?
//!
//! Tests provided, per Phase-1 trial (same shortening as `phase1`):
//!
//! 1. **rank / codimension.** Rank of the full jet matrix M_full
//!    (all n' points) and of M_tau for one dirty tau. Codimension
//!    N_d - rank is the log_q probability that a uniform random
//!    vector lands in the span: if codim is hundreds, a random
//!    vector is essentially never in the span, so "whole space
//!    spanned" is refuted even when flagging is all-zero.
//!
//! 2. **random vectors.** Uniform vectors in F_q^{k'} (the shortened
//!    ambient space), tested against M_full and against M_tau.
//!    Expected: ~0 in span in *every* regime, working or flag-none.
//!
//! 3. **pair hold-out (jet-span, NOT re-shortening).** For a dirty
//!    tau (identified from key.meta, testing only), test tau's
//!    monomial vector against the jets of all other kept points
//!    *except its mate*, and, as a control, except a non-mate
//!    kept point (the first such column, deterministic). If tau
//!    stays in the span without its mate,
//!    the spanning is collective (other points supply the missing
//!    Taylor orders). If it falls out without its mate but stays
//!    in with a random point removed, the mate was essential.
//!    IMPORTANT: this deletes the mate's *shortened column* from
//!    the point set; it does NOT re-shorten the code with the
//!    mate hit (that would change the shortening basis and drop
//!    k' by 1). It is a linear-algebra test of the jet span, which
//!    is exactly the mechanism question, not an attack trial.
//!
//! Method: one forward elimination of the jet matrix to row-echelon
//! form per matrix tested, then every test vector is reduced by
//! the echelon rows (pivot order). This is the same test as
//! `gijs::distinguish` (v in rowspan iff it reduces to zero),
//! rearranged so many vectors can be tested against one matrix.
//! Cost is one dense elimination per matrix, so this tool is for
//! tractable points only: it refuses (unless --allow-large) when
//! E * N_d exceeds 200M entries (ID-0 (2,3) is 296M).

use crate::gf::Gf;
use crate::gijs::{enum_exp, hasse, power_table};
use crate::linalg::shorten;
use crate::rng::Rng;

/// Jet matrix for a point set: E = n * J rows, N_d columns.
/// Rows are (point, jet) in the same order as `gijs::build_matrix`
/// (jet a = 0 first within each point's block).
pub fn jet_matrix(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize) -> (Vec<u16>, usize, usize, usize) {
    let n = cols.len();
    let k = if n == 0 { 0 } else { cols[0].len() };
    let monos = enum_exp(k, d);
    let mut jets: Vec<Vec<u8>> = Vec::new();
    for sum in 0..s {
        jets.extend(enum_exp(k, sum));
    }
    let nd = monos.len();
    let jn = jets.len();
    let e = n * jn;
    let mut mat = vec![0u16; e * nd];
    for (j, col) in cols.iter().enumerate() {
        let yp = power_table(gf, col, d);
        for (ai, a) in jets.iter().enumerate() {
            let r = j * jn + ai;
            let base = r * nd;
            for (mi, m) in monos.iter().enumerate() {
                mat[base + mi] = hasse(gf, m, a, &yp);
            }
        }
    }
    (mat, e, nd, jn)
}

/// Monomial evaluation vector of one ambient vector (length k).
pub fn eval_vector(gf: &Gf, vec: &[u16], d: usize) -> Vec<u16> {
    let k = vec.len();
    let monos = enum_exp(k, d);
    let yp = power_table(gf, vec, d);
    let mut v = vec![0u16; monos.len()];
    for (mi, m) in monos.iter().enumerate() {
        let mut val: u16 = 1;
        for i in 0..k {
            val = gf.mul(val, yp[i][m[i] as usize]);
            if val == 0 {
                break;
            }
        }
        v[mi] = val;
    }
    v
}

/// Row-echelon form of a matrix (forward elimination only, pivot
/// columns in increasing order). Returns (echelon rows, pivot columns,
/// rank). The echelon rows are a basis of the matrix's row space:
/// a vector is in the row space iff it reduces to zero by them
/// (see `in_span_echelon`).
pub fn row_echelon(gf: &Gf, mut mat: Vec<u16>, e: usize, nd: usize) -> (Vec<u16>, Vec<usize>, usize) {
    let mut pivots: Vec<usize> = Vec::new();
    let mut prow = 0usize;
    for col in 0..nd {
        if prow >= e {
            break;
        }
        let mut piv: Option<usize> = None;
        for r in prow..e {
            if mat[r * nd + col] != 0 {
                piv = Some(r);
                break;
            }
        }
        let p = match piv {
            Some(p) => p,
            None => continue,
        };
        if p != prow {
            for c in 0..nd {
                mat.swap(prow * nd + c, p * nd + c);
            }
        }
        let inv = gf.inv(mat[prow * nd + col]);
        if inv != 1 {
            for c in col..nd {
                let idx = prow * nd + c;
                mat[idx] = gf.mul(mat[idx], inv);
            }
        }
        for r in (prow + 1)..e {
            let f = mat[r * nd + col];
            if f != 0 {
                let base_r = r * nd;
                let base_p = prow * nd;
                for c in col..nd {
                    let sub = gf.mul(f, mat[base_p + c]);
                    mat[base_r + c] ^= sub;
                }
                mat[base_r + col] = 0;
            }
        }
        pivots.push(col);
        prow += 1;
    }
    let rank = pivots.len();
    // Keep only the pivot rows (the rest are zero).
    let mut ech = vec![0u16; rank * nd];
    for (i, _) in pivots.iter().enumerate() {
        ech[i * nd..(i + 1) * nd].copy_from_slice(&mat[i * nd..(i + 1) * nd]);
    }
    (ech, pivots, rank)
}

/// Test whether `v` lies in the row space given by echelon rows
/// (from `row_echelon`). Reduces v by pivot rows in pivot order.
pub fn in_span_echelon(gf: &Gf, ech: &[u16], pivots: &[usize], nd: usize, v: &[u16]) -> bool {
    let mut w = v.to_vec();
    for (i, &pc) in pivots.iter().enumerate() {
        let vc = w[pc];
        if vc != 0 {
            let base = i * nd;
            for c in pc..nd {
                let sub = gf.mul(vc, ech[base + c]);
                w[c] ^= sub;
            }
        }
    }
    w.iter().all(|&x| x == 0)
}

/// One matrix test: build the jet matrix for `cols`, echelon it,
/// and test each vector in `test_vecs` for membership in its row
/// space. Returns (rank, in_span per test vector).
pub fn test_against_matrix(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    test_vecs: &[Vec<u16>],
) -> (usize, usize, usize, Vec<bool>) {
    let (mat, e, nd, _jn) = jet_matrix(gf, cols, s, d);
    let (ech, pivots, rank) = row_echelon(gf, mat, e, nd);
    let mut out = Vec::with_capacity(test_vecs.len());
    for vec in test_vecs {
        let v = eval_vector(gf, vec, d);
        out.push(in_span_echelon(gf, &ech, &pivots, nd, &v));
    }
    (e, nd, rank, out)
}

/// One trial's hold-out results.
#[derive(Debug)]
pub struct HoldoutTrial {
    pub trial: usize,
    pub nprime: usize,
    pub kp: usize,
    /// Normal (reuse) flagged count for this trial, for comparison.
    pub flagged_normal: usize,
    pub e_full: usize,
    pub nd: usize,
    pub rank_full: usize,
    /// Random vectors in span of M_full / total.
    pub random_in_full: usize,
    pub random_total: usize,
    /// Random vectors in span of M_tau (tau = first dirty tau, or
    /// first kept column if no meta / no dirty tau) / total.
    pub random_in_tau: usize,
    pub pair_tests: Vec<PairTest>,
}

/// One dirty tau's pair-hold-out result.
#[derive(Debug)]
pub struct PairTest {
    pub tau_orig: usize,
    pub mate_orig: usize,
    /// Normal decision: tau in span of M_tau (mate present)?
    /// True = in span = NOT flagged (GRS).
    pub in_with_mate: bool,
    /// Tau in span of jets excluding tau AND mate?
    pub in_without_mate: bool,
    /// Control: tau in span of jets excluding tau AND a random
    /// non-mate kept column?
    pub in_without_random: bool,
    pub control_orig: usize,
}

/// Run hold-out tests on `trials` Phase-1 trials (same shortening
/// as `phase1::attack_trial`: trial t uses seed base_seed + 1 + t).
/// `meta` is testing-only (dirty/mate labels); if None, pair tests
/// are skipped and only rank + random-vector tests run.
/// `pair_tests` caps dirty taus tested per trial (0 = skip pair tests).
pub fn run_holdout_trials(
    gf: &Gf,
    cols: &[Vec<u16>],
    meta: Option<&crate::keyio::KeyMeta>,
    nprime: usize,
    s: usize,
    d: usize,
    trials: usize,
    base_seed: u64,
    random_vecs: usize,
    pair_tests: usize,
) -> Vec<HoldoutTrial> {
    let n_pub = cols.len();
    let mut results = Vec::new();
    for t in 0..trials {
        let trial_seed = base_seed + 1 + t as u64;
        let mut rng = Rng::new(trial_seed);
        let hit = rng.subset(n_pub, n_pub - nprime);
        let (short_cols, kept) = shorten(gf, cols, &hit);
        let kp = short_cols[0].len();

        // Normal decisions via reuse (exactly phase1's reuse engine).
        let decisions = crate::reuse::distinguish_all_reuse(gf, &short_cols, s, d);
        let flagged_normal = decisions.iter().filter(|&&b| !b).count();

        // Rank of M_full + random vectors against M_full.
        let (e_full, nd, rank_full, _) = test_against_matrix(gf, &short_cols, s, d, &[]);
        // Random vectors: independent stream (xorshift seed, not the
        // trial-selection stream, so vectors are independent of the
        // kept set given the trial).
        let mut rrng = Rng::new(trial_seed ^ 0x9e37_79b9_7f4a_7c15);
        let rand_vecs: Vec<Vec<u16>> = (0..random_vecs)
            .map(|_| {
                (0..kp)
                    .map(|_| rrng.below(gf.order as u64) as u16)
                    .collect()
            })
            .collect();
        let (_, _, _, random_in_full) = test_against_matrix(gf, &short_cols, s, d, &rand_vecs);
        let random_in_full = random_in_full.iter().filter(|&&b| b).count();

        // Choose tau for the M_tau random test: first dirty kept
        // column by meta, else first kept column.
        let tau_pos: usize = match meta {
            Some(m) => kept
                .iter()
                .position(|&orig| {
                    // dirty = twin member whose mate is also kept
                    if let Some(pid) = m.pair_id[orig] {
                        let (a, b) = m.pairs[pid];
                        let mate = if a == orig { b } else { a };
                        kept.contains(&mate)
                    } else {
                        false
                    }
                })
                .unwrap_or(0),
            None => 0,
        };
        let cols_tau: Vec<Vec<u16>> = short_cols
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != tau_pos)
            .map(|(_, c)| c.clone())
            .collect();
        let (_, _, _, random_in_tau) = test_against_matrix(gf, &cols_tau, s, d, &rand_vecs);
        let random_in_tau = random_in_tau.iter().filter(|&&b| b).count();

        // Pair hold-out tests (meta required).
        let mut pair_results = Vec::new();
        if pair_tests > 0 {
            if let Some(m) = meta {
                let mut tested = 0usize;
                // Deterministic control choice: first kept column that
                // is neither tau nor its mate (per tau).
                for (pos, &orig) in kept.iter().enumerate() {
                    if tested >= pair_tests {
                        break;
                    }
                    let pid = match m.pair_id[orig] {
                        Some(p) => p,
                        None => continue,
                    };
                    let (a, b) = m.pairs[pid];
                    let mate = if a == orig { b } else { a };
                    if !kept.contains(&mate) {
                        continue; // not dirty in this trial
                    }
                    let mate_pos = kept.iter().position(|&x| x == mate).unwrap();
                    let control_pos = kept
                        .iter()
                        .position(|&x| x != orig && x != mate)
                        .unwrap();
                    let control_orig = kept[control_pos];
                    let tau_vec = short_cols[pos].clone();
                    // Matrix excluding tau and mate.
                    let cols_nm: Vec<Vec<u16>> = short_cols
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != pos && *i != mate_pos)
                        .map(|(_, c)| c.clone())
                        .collect();
                    let (_, _, _, r_nm) =
                        test_against_matrix(gf, &cols_nm, s, d, std::slice::from_ref(&tau_vec));
                    // Matrix excluding tau and a random (control) point.
                    let cols_nc: Vec<Vec<u16>> = short_cols
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != pos && *i != control_pos)
                        .map(|(_, c)| c.clone())
                        .collect();
                    let (_, _, _, r_nc) =
                        test_against_matrix(gf, &cols_nc, s, d, std::slice::from_ref(&tau_vec));
                    pair_results.push(PairTest {
                        tau_orig: orig,
                        mate_orig: mate,
                        in_with_mate: decisions[pos],
                        in_without_mate: r_nm[0],
                        in_without_random: r_nc[0],
                        control_orig,
                    });
                    tested += 1;
                }
            }
        }

        results.push(HoldoutTrial {
            trial: t,
            nprime,
            kp,
            flagged_normal,
            e_full,
            nd,
            rank_full,
            random_in_full,
            random_total: random_vecs,
            random_in_tau,
            pair_tests: pair_results,
        });
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gijs::distinguish;
    use crate::rlce::gen_rlce;

    /// Oracle for "vec in span of jets of cols": append vec as a new
    /// point and run dense distinguish holding it out. Must equal
    /// the echelon batch test.
    fn oracle_in_span(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize, vec: &[u16]) -> bool {
        let mut plus = cols.to_vec();
        plus.push(vec.to_vec());
        let tau = plus.len() - 1;
        distinguish(gf, &plus, s, d, tau)
    }

    #[test]
    fn batch_matches_dense_oracle_small() {
        let gf = Gf::for_q(64);
        for seed in 0..4u64 {
            let mut rng = Rng::new(100 + seed);
            let cols: Vec<Vec<u16>> = (0..12)
                .map(|_| (0..8).map(|_| rng.below(64) as u16).collect())
                .collect();
            let mut vrng = Rng::new(900 + seed);
            let vecs: Vec<Vec<u16>> = (0..6)
                .map(|_| (0..8).map(|_| vrng.below(64) as u16).collect())
                .collect();
            let (_, _, _, got) = test_against_matrix(&gf, &cols, 2, 3, &vecs);
            for (v, g) in vecs.iter().zip(got.iter()) {
                let expect = oracle_in_span(&gf, &cols, 2, 3, v);
                assert_eq!(*g, expect, "batch vs oracle mismatch seed={}", seed);
            }
            // Rank sanity: rank <= min(E, N_d), and every point's own
            // vector is in span of the full matrix (its order-0 row).
            let (e, nd, rank, in_full) = test_against_matrix(&gf, &cols, 2, 3, &cols);
            assert!(rank <= e.min(nd));
            assert!(in_full.iter().all(|&b| b), "own points must be in full span");
        }
    }

    #[test]
    fn batch_matches_dense_oracle_rlce_shortened() {
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(42);
        let key = gen_rlce(&gf, &mut rng, 30, 20, 5);
        let mut srng = Rng::new(7);
        let hit = srng.subset(key.n_pub, 8);
        let (short, _kept) = shorten(&gf, &key.cols, &hit);
        assert_eq!(short[0].len(), 12);
        let mut vrng = Rng::new(555);
        let vecs: Vec<Vec<u16>> = (0..4)
            .map(|_| (0..12).map(|_| vrng.below(64) as u16).collect())
            .collect();
        let (_, _, _, got) = test_against_matrix(&gf, &short, 2, 3, &vecs);
        for (v, g) in vecs.iter().zip(got.iter()) {
            assert_eq!(*g, oracle_in_span(&gf, &short, 2, 3, v));
        }
        // Internal taus: batch against M_tau must equal dense
        // distinguish for every tau on this small instance.
        for tau in 0..short.len() {
            let cols_tau: Vec<Vec<u16>> = short
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != tau)
                .map(|(_, c)| c.clone())
                .collect();
            let (_, _, _, r) =
                test_against_matrix(&gf, &cols_tau, 2, 3, std::slice::from_ref(&short[tau]));
            assert_eq!(r[0], distinguish(&gf, &short, 2, 3, tau), "tau {}", tau);
        }
    }

    #[test]
    fn random_vectors_rarely_in_span_small() {
        // On a small random point set, a uniform vector is in the
        // jet span only with probability ~ q^{-(N_d - rank)}: with
        // N_d = C(10,3) = 120 and rank <= E = 11*9 = 99, codim >= 21,
        // so 200 random vectors should essentially never land in span.
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(1234);
        let cols: Vec<Vec<u16>> = (0..12)
            .map(|_| (0..8).map(|_| rng.below(64) as u16).collect())
            .collect();
        // Note: this point set has k=8, not 10; N_d = C(10,3)=120 is
        // for (k+d-1 choose d) with k=8, d=3: C(10,3)=120, E=12*9=108.
        let mut vrng = Rng::new(777);
        let vecs: Vec<Vec<u16>> = (0..200)
            .map(|_| (0..8).map(|_| vrng.below(64) as u16).collect())
            .collect();
        let (_, _, rank, got) = test_against_matrix(&gf, &cols, 2, 3, &vecs);
        assert!(rank <= 108);
        let n_in = got.iter().filter(|&&b| b).count();
        assert_eq!(n_in, 0, "random vectors in span: {} (rank={})", n_in, rank);
    }
}
