//! Hold-out reuse GIJS distinguisher.
//!
//! `gijs::distinguish(gf, cols, s, d, tau)` tests one hold-out tau by
//! building the Hasse jet matrix M_tau (all points except tau) and checking
//! whether the evaluation vector v_tau lies in rowspan(M_tau). A Phase-1
//! trial calls it once per point (n' calls), each rebuilding and
//! re-eliminating a matrix that differs from the others only by one point's
//! block of J rows. This module computes ALL n' decisions from ONE
//! factorization of the full matrix M_full (all points' blocks).
//!
//! Mathematics. Rows of M_full are indexed (j, a): point j, Hasse multi-index
//! a with |a| < s; columns are the homogeneous degree-d monomials. Write
//! P_z for the degree-d polynomial with coefficient vector z. Then
//!   (M_full z)_{(j,a)} = (Hasse_a P_z)(y_j),
//! and since v_tau is exactly the (tau, a=0) row of M_full,
//!   z . v_tau = (M_full z)_{(tau, 0)}.
//! Standard duality (v in rowspan(M) iff v.z = 0 for all z in ker M) gives:
//!   tau is FLAGGED ("random")  iff  exists z with M_tau z = 0 and
//!   (M_full z)_{(tau,0)} != 0, i.e. iff some syndrome s_full = M_full z is
//!   supported only on block tau's J rows and has s_{(tau,0)} != 0.
//!
//! Let Pperp be a basis of the LEFT kernel of M_full (Pperp * M_full = 0,
//! rows spanning all row relations). The achievable syndromes are exactly
//! ker(Pperp), so the achievable block-tau syndromes are
//!   S_tau = ker(Pperp restricted to block tau's J columns),
//! and tau is flagged iff some s in S_tau has s_0 != 0. For a matrix A,
//! a vector s in ker(A) with s_0 != 0 is exactly a column relation
//! col_0 = sum_{i>=1} (s_i/s_0) col_i, so the 0-coordinate is
//! non-vanishing on ker(A) iff column 0 of A IS in the span of the other
//! columns, i.e. iff
//!   rank(A_tau) == rank(A_tau without column 0).
//! So: one left-kernel computation, then two small rank computations per tau.
//!
//! Pperp is computed by forward elimination on the augmented matrix
//! [M_full | I_E]: if U * M_full = R in row-echelon form, the rows of U
//! corresponding to zero rows of R annihilate M_full and form a basis of
//! the left kernel (U is invertible, so they are independent, and their
//! count E - rank matches the left nullity).
//!
//! Cost (prototype, dense stored transforms): one elimination on
//! E x (N_d + E) plus n' rank computations on (E - rank) x J systems.
//! At flagship scale (E ~ 151k, N_d ~ 31k) the augmented matrix alone is
//! E*(N_d+E) entries (~106 GB at (s,d)=(3,5) ID-0-like sizes) and Pperp is
//! (E-rank) x E dense -- this prototype is NOT viable at that scale; it is
//! meant for tractable parameters and for validating the scheme.

use crate::gf::Gf;
use crate::gijs::{enum_exp, hasse, power_table};
use indicatif::ProgressBar;

/// Build the augmented matrix [M_full | I_E] directly (one allocation,
/// E = n * J rows, row stride nd + E): M_full is the full jet matrix (all
/// points); block j occupies rows j*J..(j+1)*J in the same jet order as
/// gijs::build_matrix (a = 0 first within each block).
fn build_augmented(
    gf: &Gf,
    cols: &[Vec<u16>],
    monos: &[Vec<u8>],
    jets: &[Vec<u8>],
    d: usize,
) -> Vec<u16> {
    let n = cols.len();
    let nd = monos.len();
    let js = jets.len();
    let e = n * js;
    let w = nd + e;
    let mut aug = vec![0u16; e * w];
    for (j, col) in cols.iter().enumerate() {
        let yp = power_table(gf, col, d);
        for (ai, a) in jets.iter().enumerate() {
            let r = j * js + ai;
            let base = r * w;
            for (mi, m) in monos.iter().enumerate() {
                aug[base + mi] = hasse(gf, m, a, &yp);
            }
            aug[base + nd + r] = 1;
        }
    }
    aug
}

/// Basis of the left kernel of M_full, from its augmented matrix
/// [M_full | I_E] (consumed): forward elimination; the rows of the
/// transformation U (with U * M_full in echelon form) that correspond to
/// zero rows of the echelon form annihilate M_full and are returned
/// (rows of length E, each u with u^T * M_full = 0).
fn left_kernel_basis(
    gf: &Gf,
    mut aug: Vec<u16>,
    e: usize,
    nd: usize,
    progress: Option<&ProgressBar>,
) -> Vec<Vec<u16>> {
    if e == 0 || nd == 0 {
        return Vec::new();
    }
    let w = nd + e;
    // Work-weighted progress: elimination below a pivot in column
    // `col` touches (e - prow) rows of length (w - col), so early
    // columns dominate the cost and a plain column-count bar would
    // crawl at first and then sprint. Total work below assumes a
    // pivot in every column until the rank limit, which is the
    // generic case; the bar is set to 100% at the end regardless,
    // so a rank-deficient matrix only makes the last step jump.
    let rank_limit = nd.min(e);
    let total_work: u64 = (0..rank_limit)
        .map(|col| (e - col) as u64 * (w - col) as u64)
        .sum();
    if let Some(pb) = progress {
        pb.set_length(total_work);
        pb.set_position(0);
    }
    let mut prow = 0usize;
    for col in 0..nd {
        // find pivot in rows prow..e
        let mut piv: Option<usize> = None;
        for r in prow..e {
            if aug[r * w + col] != 0 {
                piv = Some(r);
                break;
            }
        }
        let p = match piv {
            Some(p) => p,
            None => continue,
        };
        if p != prow {
            for c in 0..w {
                aug.swap(prow * w + c, p * w + c);
            }
        }
        // normalize pivot row
        let inv = gf.inv(aug[prow * w + col]);
        if inv != 1 {
            for c in col..w {
                let idx = prow * w + c;
                aug[idx] = gf.mul(aug[idx], inv);
            }
        }
        // eliminate below (pivot entry is 1, so factor is the entry itself)
        for r in (prow + 1)..e {
            let f = aug[r * w + col];
            if f != 0 {
                let base_r = r * w;
                let base_p = prow * w;
                for c in col..w {
                    let sub = gf.mul(f, aug[base_p + c]);
                    aug[base_r + c] ^= sub;
                }
            }
        }
        if let Some(pb) = progress {
            pb.inc((e - prow) as u64 * (w - col) as u64);
        }
        prow += 1;
        if prow == e {
            break;
        }
    }
    if let Some(pb) = progress {
        pb.set_position(total_work);
    }
    // rows prow..e of aug have zero M-part; their I-parts are the basis.
    (prow..e)
        .map(|r| aug[r * w + nd..r * w + w].to_vec())
        .collect()
}

/// Rank of the submatrix of `rows` (each of length >= max(cols)+1)
/// restricted to the given column indices.
fn col_rank(gf: &Gf, rows: &[Vec<u16>], cols: &[usize]) -> usize {
    let m = rows.len();
    let w = cols.len();
    if m == 0 || w == 0 {
        return 0;
    }
    let mut a = vec![0u16; m * w];
    for (i, row) in rows.iter().enumerate() {
        for (t, &c) in cols.iter().enumerate() {
            a[i * w + t] = row[c];
        }
    }
    let mut rank = 0usize;
    let mut prow = 0usize;
    for c in 0..w {
        let mut piv: Option<usize> = None;
        for r in prow..m {
            if a[r * w + c] != 0 {
                piv = Some(r);
                break;
            }
        }
        if let Some(p) = piv {
            if p != prow {
                for cc in 0..w {
                    a.swap(prow * w + cc, p * w + cc);
                }
            }
            let inv = gf.inv(a[prow * w + c]);
            if inv != 1 {
                for cc in c..w {
                    let idx = prow * w + cc;
                    a[idx] = gf.mul(a[idx], inv);
                }
            }
            for r in (prow + 1)..m {
                let f = a[r * w + c];
                if f != 0 {
                    let base_r = r * w;
                    let base_p = prow * w;
                    for cc in c..w {
                        let sub = gf.mul(f, a[base_p + cc]);
                        a[base_r + cc] ^= sub;
                    }
                }
            }
            prow += 1;
            rank += 1;
            if prow == m {
                break;
            }
        }
    }
    rank
}

/// All hold-out decisions for one point set, from ONE factorization.
/// Returns Vec<bool> indexed by tau: true = "GRS" (as in gijs::distinguish).
pub fn distinguish_all_reuse(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize) -> Vec<bool> {
    distinguish_all_reuse_with_progress(gf, cols, s, d, None)
}

/// Same as `distinguish_all_reuse`, but with per-factorization
/// progress: when `progress` is given, its message (set by the
/// caller, e.g. "shortening 3 puncturing 7") is kept, its length
/// is set to the elimination work estimate, and it is advanced
/// work-weighted as pivot columns are eliminated, so it moves
/// roughly linearly in time instead of stalling for the whole
/// factorization and then jumping. Building the augmented matrix
/// and the per-tau rank checks are not counted; both are small
/// next to the elimination at the sizes where this matters.
pub fn distinguish_all_reuse_with_progress(
    gf: &Gf,
    cols: &[Vec<u16>],
    s: usize,
    d: usize,
    progress: Option<&ProgressBar>,
) -> Vec<bool> {
    let n = cols.len();
    if n == 0 {
        return Vec::new();
    }
    let k = cols[0].len();
    let monos = enum_exp(k, d);
    let mut jets: Vec<Vec<u8>> = Vec::new();
    for sum in 0..s {
        jets.extend(enum_exp(k, sum));
    }
    let nd = monos.len();
    let jn = jets.len();
    let e = n * jn;

    let aug = build_augmented(gf, cols, &monos, &jets, d);
    let pperp = left_kernel_basis(gf, aug, e, nd, progress);

    if pperp.is_empty() {
        // M_full has full row rank: its column space is everything, so for
        // every tau the syndrome e_{(tau,0)} is achievable on block tau
        // alone -> every tau is flagged.
        return vec![false; n];
    }

    let mut out = Vec::with_capacity(n);
    for tau in 0..n {
        let base = tau * jn;
        let all: Vec<usize> = (base..base + jn).collect();
        let rest: Vec<usize> = (base + 1..base + jn).collect();
        // flagged iff col 0 of Pperp_tau lies in the span of the rest,
        // i.e. including it does NOT raise the rank.
        let flagged = col_rank(gf, &pperp, &all) == col_rank(gf, &pperp, &rest);
        out.push(!flagged);
    }
    out
}

// ---------------------------------------------------------------------------
// Reuse hold-out: mate present / mate absent / control absent, from ONE
// left-kernel factorization (see reuse-holdout-test in attack.rs).
//
// Generalization of the per-tau argument in the module doc. For a set S
// of blocks to delete (e.g. tau alone, tau+mate, tau+control), let
// A be Pperp restricted to the columns of S (first column = (tau,0)).
// A syndrome supported only on S is achievable iff it lies in ker(A),
// and such a syndrome with coordinate (tau,0) != 0 exists iff column 0
// of A lies in the span of the other columns of A, i.e. iff
// rank(A) == rank(A without column 0). So tau is IN the span of the
// point set with blocks S deleted iff that rank strictly increases
// when column 0 is added. Empty Pperp means every syndrome is
// achievable, so tau is OUT (flagged) for every S containing tau.
// This is exact, not a heuristic: it is the same duality as
// distinguish_all_reuse, with a larger deleted set S.
// ---------------------------------------------------------------------------

/// Left-kernel basis Pperp for one point set, plus J (jet rows per
/// point), E (total rows) and N_d. This is the first half of
/// distinguish_all_reuse, exposed so a caller can run many deleted-set
/// verdicts against the same factorization instead of re-eliminating
/// a matrix per deleted set.
pub fn pperp_for_cols(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize) -> (Vec<Vec<u16>>, usize, usize, usize) {
    let n = cols.len();
    if n == 0 {
        return (Vec::new(), 0, 0, 0);
    }
    let k = cols[0].len();
    let monos = enum_exp(k, d);
    let mut jets: Vec<Vec<u8>> = Vec::new();
    for sum in 0..s {
        jets.extend(enum_exp(k, sum));
    }
    let nd = monos.len();
    let jn = jets.len();
    let e = n * jn;
    let aug = build_augmented(gf, cols, &monos, &jets, d);
    let pperp = left_kernel_basis(gf, aug, e, nd, None);
    (pperp, jn, e, nd)
}

/// True iff tau's monomial vector lies in the row space of the point
/// set with blocks `block_cols` deleted, where `block_cols` are
/// Pperp column indices (i.e. M_full row indices), first element
/// (tau,0). `block_cols` must contain the tau block and any other
/// blocks to delete (mate or control). Empty Pperp => false.
pub fn in_span_from_pperp(gf: &Gf, pperp: &[Vec<u16>], block_cols: &[usize]) -> bool {
    if pperp.is_empty() || block_cols.is_empty() {
        return false;
    }
    let rest: Vec<usize> = block_cols[1..].to_vec();
    let flagged = col_rank(gf, pperp, block_cols) == col_rank(gf, pperp, &rest);
    !flagged
}

// ---------------------------------------------------------------------------
// Memory footprint estimate (used by the Phase-1 pre-flight guard)
// ---------------------------------------------------------------------------

/// Dimensions and worst-case memory footprint of one `distinguish_all_reuse`
/// call on `n` shortened columns of length `k` with parameters (s, d).
///
/// The two allocations that matter:
///   * the augmented matrix [M_full | I_E]: E * (N_d + E) u16 entries;
///   * the left-kernel basis Pperp: (E - rank) rows of length E. The row
///     count is bounded by E (rank >= 0); in practice rank = min(N_d, E)
///     for generic point sets, so the basis is usually far smaller (and
///     empty when E <= N_d and M_full has full row rank). This is a
///     worst-case bound, sized for a safety guard, not a prediction.
/// The per-tau `col_rank` scratch (at most E x J) is negligible next to
/// these and is not counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReuseMemoryEstimate {
    /// J = C(k+s-1, s-1): jet rows per point.
    pub j: usize,
    /// N_d = C(k+d-1, d): degree-d monomials.
    pub n_d: usize,
    /// E = n * J: rows of M_full.
    pub e: usize,
    /// Bytes of the augmented matrix [M_full | I_E] (u16 entries).
    pub augmented_bytes: u64,
    /// Worst-case bytes of the left-kernel basis (<= E rows of length E).
    pub pperp_bytes: u64,
    /// augmented_bytes + pperp_bytes.
    pub total_bytes: u64,
}

fn binom_u128(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut res: u128 = 1;
    for i in 0..k {
        res = res * (n - i) as u128 / (i + 1) as u128;
    }
    res
}

/// Worst-case memory estimate for `distinguish_all_reuse`, computed purely
/// from dimensions (allocates nothing). Saturates at u64::MAX on absurd
/// inputs rather than overflowing.
pub fn reuse_memory_estimate(n: usize, k: usize, s: usize, d: usize) -> ReuseMemoryEstimate {
    let j128 = binom_u128(k + s - 1, s - 1);
    let nd128 = binom_u128(k + d - 1, d);
    let e128 = n as u128 * j128;
    let aug = e128 * (nd128 + e128) * 2;
    let pperp = e128 * e128 * 2;
    let sat = |v: u128| -> u64 {
        if v > u64::MAX as u128 {
            u64::MAX
        } else {
            v as u64
        }
    };
    let sat_usize = |v: u128| -> usize {
        if v > usize::MAX as u128 {
            usize::MAX
        } else {
            v as usize
        }
    };
    ReuseMemoryEstimate {
        j: sat_usize(j128),
        n_d: sat_usize(nd128),
        e: sat_usize(e128),
        augmented_bytes: sat(aug),
        pperp_bytes: sat(pperp),
        total_bytes: sat(aug + pperp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gijs::distinguish;
    use crate::linalg::shorten;
    use crate::rlce::gen_rlce;
    use crate::rng::Rng;

    /// Differential check: reuse decisions must equal the oracle
    /// (per-tau `distinguish`) on EVERY tau of the instance.
    fn check_instance(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize, label: &str) {
        let expect: Vec<bool> = (0..cols.len())
            .map(|tau| distinguish(gf, cols, s, d, tau))
            .collect();
        let got = distinguish_all_reuse(gf, cols, s, d);
        assert_eq!(
            expect, got,
            "mismatch on instance {} (n={}, s={}, d={})",
            label,
            cols.len(),
            s,
            d
        );
    }

    fn random_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
        (0..n)
            .map(|_| {
                (0..k)
                    .map(|_| rng.below(gf.order as u64) as u16)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn random_codes_small_grid() {
        // (a) random codes across the required (k,s,d) grid + extras.
        let gf64 = Gf::for_q(64);
        let gf128 = Gf::for_q(128);
        for seed in 0..6u64 {
            let mut rng = Rng::new(1000 + seed);
            let cols = random_cols(&gf64, &mut rng, 14, 8);
            check_instance(&gf64, &cols, 2, 3, "rand64 k8 s2 d3");
        }
        for seed in 0..3u64 {
            let mut rng = Rng::new(2000 + seed);
            let cols = random_cols(&gf64, &mut rng, 12, 14);
            check_instance(&gf64, &cols, 2, 4, "rand64 k14 s2 d4");
        }
        for seed in 0..3u64 {
            let mut rng = Rng::new(3000 + seed);
            let cols = random_cols(&gf128, &mut rng, 14, 19);
            check_instance(&gf128, &cols, 2, 3, "rand128 k19 s2 d3");
        }
        for seed in 0..2u64 {
            let mut rng = Rng::new(4000 + seed);
            let cols = random_cols(&gf64, &mut rng, 6, 12);
            check_instance(&gf64, &cols, 3, 4, "rand64 k12 s3 d4");
        }
    }

    #[test]
    fn random_codes_degenerate_columns() {
        // Duplicated point and zero column: linear algebra must still agree.
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(42);
        let mut cols = random_cols(&gf, &mut rng, 10, 8);
        cols[3] = cols[1].clone(); // duplicate point
        cols[5] = vec![0u16; 8]; // zero column
        check_instance(&gf, &cols, 2, 3, "rand degenerate");
    }

    #[test]
    fn pure_grs_sets() {
        // (b) pure GRS column sets (gen_rlce with w = 0).
        let gf64 = Gf::for_q(64);
        let gf128 = Gf::for_q(128);
        for seed in 0..2u64 {
            let mut rng = Rng::new(5000 + seed);
            let key = gen_rlce(&gf64, &mut rng, 16, 10, 0);
            check_instance(&gf64, &key.cols, 2, 3, "grs64 k10");
        }
        {
            let mut rng = Rng::new(6000);
            let key = gen_rlce(&gf128, &mut rng, 16, 14, 0);
            check_instance(&gf128, &key.cols, 2, 4, "grs128 k14 s2 d4");
        }
        {
            let mut rng = Rng::new(7000);
            let key = gen_rlce(&gf64, &mut rng, 14, 10, 0);
            check_instance(&gf64, &key.cols, 3, 4, "grs64 k10 s3 d4");
        }
    }

    #[test]
    fn rlce_shortened_sets() {
        // (c) RLCE keys shortened the Phase-1 way.
        let gf = Gf::for_q(64);
        // k' = 12 after hitting 8 of N=35: (s,d) = (2,3) and (2,4).
        for seed in 0..3u64 {
            let mut rng = Rng::new(8000 + seed);
            let key = gen_rlce(&gf, &mut rng, 30, 20, 5);
            let mut srng = Rng::new(9000 + seed);
            let hit = srng.subset(key.n_pub, 8);
            let (short, _kept) = shorten(&gf, &key.cols, &hit);
            assert_eq!(short[0].len(), 12);
            check_instance(&gf, &short, 2, 3, "rlce-short k12 s2 d3");
            check_instance(&gf, &short, 2, 4, "rlce-short k12 s2 d4");
        }
        // k' = 8 after hitting 3 of N=18: (s,d) = (3,4).
        {
            let mut rng = Rng::new(9500);
            let key = gen_rlce(&gf, &mut rng, 16, 11, 2);
            let mut srng = Rng::new(9600);
            let hit = srng.subset(key.n_pub, 3);
            let (short, _kept) = shorten(&gf, &key.cols, &hit);
            assert_eq!(short[0].len(), 8);
            check_instance(&gf, &short, 3, 4, "rlce-short k8 s3 d4");
        }
    }

    // -----------------------------------------------------------------
    // Benchmarks: full trial (all taus), old way vs reuse way.
    // Run with: cargo test --release --bin rlce-attack bench_ -- --ignored --nocapture
    // -----------------------------------------------------------------

    use std::time::Instant;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Shorten an RLCE key exactly like attack_trial does (same Rng usage).
    fn shortened_instance(
        gf: &Gf,
        n: usize,
        k: usize,
        w: usize,
        key_seed: u64,
        n_hit: usize,
        trial_seed: u64,
    ) -> Vec<Vec<u16>> {
        let mut rng = Rng::new(key_seed);
        let key = gen_rlce(gf, &mut rng, n, k, w);
        let mut srng = Rng::new(trial_seed);
        let hit = srng.subset(key.n_pub, n_hit);
        let (short, _kept) = shorten(gf, &key.cols, &hit);
        short
    }

    #[test]
    #[ignore]
    fn bench_point1() {
        // q=128, n'=67, k'=19, (s,d)=(2,3): from key (n=80,k=56,w=24), hit 37.
        let gf = Gf::for_q(128);
        let cols = shortened_instance(&gf, 80, 56, 24, 42, 37, 100);
        assert_eq!(cols.len(), 67);
        assert_eq!(cols[0].len(), 19);
        let (s, d) = (2usize, 3usize);

        // sanity: decisions agree on this instance too
        let expect: Vec<bool> = (0..cols.len())
            .map(|tau| distinguish(&gf, &cols, s, d, tau))
            .collect();
        let got = distinguish_all_reuse(&gf, &cols, s, d);
        assert_eq!(expect, got, "bench point 1: decisions differ");

        let mut old_times = Vec::new();
        for _ in 0..3 {
            let t0 = Instant::now();
            let mut acc = 0usize;
            for tau in 0..cols.len() {
                if distinguish(&gf, &cols, s, d, tau) {
                    acc += 1;
                }
            }
            std::hint::black_box(acc);
            old_times.push(t0.elapsed().as_secs_f64());
        }
        let mut new_times = Vec::new();
        for _ in 0..5 {
            let t0 = Instant::now();
            let r = distinguish_all_reuse(&gf, &cols, s, d);
            std::hint::black_box(r);
            new_times.push(t0.elapsed().as_secs_f64());
        }
        let old_med = median(old_times);
        let new_med = median(new_times);
        println!(
            "BENCH point1 (q=128 n'=67 k'=19 s=2 d=3): old={:.3}s new={:.3}s speedup={:.1}x",
            old_med,
            new_med,
            old_med / new_med
        );
    }

    #[test]
    #[ignore]
    fn bench_point2_old_single_tau() {
        // q=256, n'=58, k'=28, (s,d)=(3,4): from key (n=60,k=40,w=10), hit 12.
        // Old way is far too slow for all 58 taus; time a few single taus.
        let gf = Gf::for_q(256);
        let cols = shortened_instance(&gf, 60, 40, 10, 42, 12, 100);
        assert_eq!(cols.len(), 58);
        assert_eq!(cols[0].len(), 28);
        for tau in [0usize, 1] {
            let t0 = Instant::now();
            let r = distinguish(&gf, &cols, 3, 4, tau);
            println!(
                "BENCH point2 old: tau={} -> {} in {:.1}s",
                tau,
                r,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    #[test]
    fn memory_estimate_dims() {
        // Target point of the Phase-1 engine benchmark: n'=90 shortened
        // columns of length k'=24, (s,d)=(2,3).
        // J = C(25,1) = 25, N_d = C(26,3) = 2600, E = 90*25 = 2250.
        let est = reuse_memory_estimate(90, 24, 2, 3);
        assert_eq!(est.j, 25);
        assert_eq!(est.n_d, 2600);
        assert_eq!(est.e, 2250);
        // augmented: 2250 * (2600 + 2250) * 2 bytes = 21,825,000 (~21 MB).
        assert_eq!(est.augmented_bytes, 21_825_000);
        // Pperp worst case: 2250^2 * 2.
        assert_eq!(est.pperp_bytes, 2250u64 * 2250 * 2);
        assert_eq!(est.total_bytes, est.augmented_bytes + est.pperp_bytes);
        // cross-check J / N_d against the actual enumerations used by
        // distinguish_all_reuse on a small case.
        let est2 = reuse_memory_estimate(7, 5, 3, 4);
        let mut jets = 0usize;
        for sum in 0..3 {
            jets += enum_exp(5, sum).len();
        }
        assert_eq!(est2.j, jets);
        assert_eq!(est2.n_d, enum_exp(5, 4).len());
        assert_eq!(est2.e, 7 * jets);
    }

    #[test]
    fn memory_estimate_flagship_is_huge() {
        // ID-0-like flagship dims: n'=348, k'=28, (s,d)=(3,5).
        // J = C(30,2) = 435, N_d = C(32,5) = 201376, E = 348*435 = 151380.
        let est = reuse_memory_estimate(348, 28, 3, 5);
        assert_eq!(est.j, 435);
        assert_eq!(est.n_d, 201_376);
        assert_eq!(est.e, 151_380);
        // The augmented matrix alone is ~1.07e11 bytes (~107 GB decimal):
        // this is the number the Phase-1 guard must reject.
        assert!(
            est.augmented_bytes > 100_000_000_000,
            "augmented_bytes = {}",
            est.augmented_bytes
        );
        assert!(est.total_bytes > est.augmented_bytes);
    }

    #[test]
    #[ignore]
    fn bench_point2_new() {
        // Same instance as bench_point2_old_single_tau, reuse way (all taus).
        let gf = Gf::for_q(256);
        let cols = shortened_instance(&gf, 60, 40, 10, 42, 12, 100);
        assert_eq!(cols.len(), 58);
        assert_eq!(cols[0].len(), 28);
        let t0 = Instant::now();
        let r = distinguish_all_reuse(&gf, &cols, 3, 4);
        println!(
            "BENCH point2 new: all {} taus in {:.1}s ({} flagged)",
            r.len(),
            t0.elapsed().as_secs_f64(),
            r.iter().filter(|&&b| !b).count()
        );
    }
}
