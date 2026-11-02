//! Fast Phase 2 bootstrap ("direction B" + direction-A confirm).
//!
//! Setting (paper Phase 2 at w = n-k): Phase 1 has identified exactly
//! k pure GRS columns and the twin pairs. SS needs k+3 columns, so we
//! must unmix `need = 3` pairs. The brute-force scan tries all of
//! F_q^3 lambda triples, running full Sidelnikov-Shestakov on each
//! (measured ~7.8 ms per candidate at (q,k) = (256,165), i.e. an
//! expected ~18 h single-threaded scan). This module finds the same
//! lambda triple in ~q^2 * O(1) + O(q * k) work instead.
//!
//! Method:
//!   Normalization (once): let P be the k x k matrix whose columns are
//!   the pure columns (invertible: scaled Vandermonde). Left-multiply
//!   by P^{-1} (a row operation; the code is unchanged). Pure columns
//!   become e_1..e_k and each bootstrap pair becomes two vectors
//!   a_i, b_i with candidate unmixing g_i(l_i) = a_i + l_i * b_i, so
//!   the candidate generator is [ I_k | g_1 g_2 g_3 ].
//!
//!   The dual of the candidate code is then explicit:
//!   H = [ -g_1^T ; -g_2^T ; -g_3^T | I_3 ]  (char 2: -x = x).
//!   Writing gamma_j = (g_1[j], g_2[j], g_3[j]) in F_q^3 for the k pure
//!   coordinates j, and m_j = (g1 g2, g1 g3, g2 g3) at j, the candidate
//!   is GRS iff the 3 x k matrix M* = [m_j] has rank <= 2, the (then
//!   unique) relation c with c . M* = 0 defines a SMOOTH conic
//!   (c12*c13*c23 != 0), and the gamma_j are nonzero and pairwise
//!   non-proportional (distinct support). This is the dual-square
//!   test: dim(D^2) = 3 + rank M*, and GRS_3 <=> dim(D^2) = 5.
//!   (Rank is computed by genuine elimination, never via a Gram
//!   matrix: in char 2 a Gram matrix can be singular on a rank-3 M*,
//!   which would create false positives.)
//!
//!   Stage 1 (root finding): fix three pure coordinate positions
//!   J = {j1,j2,j3}. Delta_J(l) = det of the 3x3 minor of M* on J has
//!   multidegree <= (2,2,2). For fixed (l1,l2) it is a quadratic
//!   A l3^2 + B l3 + C in l3 (A, B, C from four 3x3 determinants,
//!   O(1) in k), solved in closed form in char 2. The true l3* is
//!   always a root at the true (l1*,l2*) unless the quadratic is
//!   identically zero there (a "degenerate cell", handled by a direct
//!   minor scan of that cell).
//!   Stage 2: a second, disjoint position triple J' filters roots by
//!   Delta_J'(l) = 0 (~q survivors expected).
//!   Stage 3: the exact dual-rank test above per survivor.
//!   Stage 4: for each survivor, build the candidate columns in
//!   ORIGINAL coordinates and run the existing ss_attack (which has
//!   its own internal verification); first success wins.
//!
//!   lambda = infinity: the true unmixing can be the b_i direction
//!   itself (degenerate pair, GL(2) coefficient d = 0, rate ~1/q per
//!   pair); no finite lambda represents it in the orientation
//!   v = c1 + l*c2. If the finite pipeline finds nothing, we rerun
//!   with pair orientations flipped (swap the two columns of a pair:
//!   the infinite direction becomes l_new = 0). All 8 flip
//!   combinations are tried before giving up.
//!
//!   If this module returns None, the caller falls back to the
//!   existing q^3 SS scan unchanged, so behavior never regresses.
//!
//! Soundness discipline: stages 1-3 only *generate* candidates. The
//! only acceptance test is ss_attack on original-coordinate columns
//! (step 4), exactly as in the scan path.

use crate::gf::Gf;
use crate::linalg::kernel_basis;
use crate::mat;
use crate::ss;
use rayon::prelude::*;
use std::collections::HashSet;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Char-2 quadratic solver
// ---------------------------------------------------------------------------

/// Roots of A x^2 + B x + C = 0 over GF(2^m).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuadRoots {
    /// No roots.
    None,
    /// Exactly one root.
    One(u16),
    /// Two distinct roots.
    Two(u16, u16),
    /// Identically zero polynomial: every x is a root.
    All,
}

/// Solver for z^2 + z = c over GF(2^m), via a precomputed table.
///
/// z^2 + z is GF(2)-linear with image = ker(Tr) and kernel {0,1}, so
/// each c in the image has exactly two roots z, z+1; we store one.
pub struct QuadSolver {
    table: Vec<i32>, // table[c] = a root z, or -1 if unsolvable
}

impl QuadSolver {
    pub fn new(gf: &Gf) -> Self {
        let q = gf.order;
        let mut table = vec![-1i32; q];
        for z in 0..q {
            let z = z as u16;
            let c = gf.mul(z, z) ^ z;
            if table[c as usize] < 0 {
                table[c as usize] = z as i32;
            }
        }
        QuadSolver { table }
    }

    /// One root of z^2 + z = c, if one exists.
    pub fn solve_z2z(&self, c: u16) -> Option<u16> {
        let v = self.table[c as usize];
        if v < 0 {
            None
        } else {
            Some(v as u16)
        }
    }
}

/// Solve A x^2 + B x + C = 0 over GF(2^m), all branches.
pub fn solve_quad(gf: &Gf, qs: &QuadSolver, a: u16, b: u16, c: u16) -> QuadRoots {
    if a == 0 {
        // Linear (or constant) equation B x + C = 0.
        if b != 0 {
            return QuadRoots::One(gf.mul(c, gf.inv(b)));
        }
        return if c == 0 { QuadRoots::All } else { QuadRoots::None };
    }
    if b == 0 {
        // x^2 = C/A; unique root by Frobenius sqrt: x = (C/A)^{2^{m-1}}.
        let ca = gf.mul(c, gf.inv(a));
        let x = gf.pow(ca, 1usize << (gf.m - 1));
        return QuadRoots::One(x);
    }
    // Substitute x = b1 * z with b1 = B/A:  z^2 + z = A*C/B^2.
    // Solvable iff Tr(A*C/B^2) = 0; roots z and z+1.
    let b1 = gf.mul(b, gf.inv(a));
    let arg = gf.mul(gf.mul(a, c), gf.inv(gf.mul(b, b)));
    match qs.solve_z2z(arg) {
        None => QuadRoots::None,
        Some(z) => {
            let x1 = gf.mul(b1, z);
            let x2 = x1 ^ b1; // b1*(z+1)
            QuadRoots::Two(x1, x2)
        }
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// The three bootstrap pairs in systematic (P^{-1}) coordinates.
/// Candidate for pair i: g_i(l) = a[i] + l * b[i], a vector in F_q^k.
struct Normalized {
    a: [Vec<u16>; 3],
    b: [Vec<u16>; 3],
    k: usize,
}

fn matvec(gf: &Gf, m_rows: &[Vec<u16>], v: &[u16]) -> Vec<u16> {
    m_rows
        .iter()
        .map(|row| {
            let mut acc = 0u16;
            for (x, &y) in row.iter().zip(v.iter()) {
                acc ^= gf.mul(*x, y);
            }
            acc
        })
        .collect()
}

/// Build systematic coordinates. `pure_cols` are the k pure columns;
/// `pairs` are the 3 bootstrap pairs in the CURRENT orientation
/// (base, other): candidate = base + l * other.
/// Returns None if the pure columns do not form a basis.
fn normalize(
    gf: &Gf,
    pure_cols: &[Vec<u16>],
    pairs: &[(Vec<u16>, Vec<u16>)],
) -> Option<Normalized> {
    let k = pure_cols.len();
    debug_assert_eq!(pairs.len(), 3);
    // P in rows form: P[r][j] = pure_cols[j][r].
    let p_rows = mat::transpose(pure_cols);
    let pinv = mat::matinv(gf, &p_rows)?;
    let mut a: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut b: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..3 {
        a[i] = matvec(gf, &pinv, &pairs[i].0);
        b[i] = matvec(gf, &pinv, &pairs[i].1);
    }
    Some(Normalized { a, b, k })
}

/// g_i(l_i)[j]: coordinate j of the i-th candidate column.
#[inline]
fn gval(gf: &Gf, norm: &Normalized, i: usize, j: usize, lam: u16) -> u16 {
    norm.a[i][j] ^ gf.mul(lam, norm.b[i][j])
}

/// Determinant of the 3x3 matrix with rows r0, r1, r2 (char 2: the
/// signed expansion collapses to the all-+ sum).
fn det3_rows(gf: &Gf, r0: &[u16; 3], r1: &[u16; 3], r2: &[u16; 3]) -> u16 {
    let t = |x: u16, y: u16, z: u16| gf.mul(gf.mul(x, y), z);
    t(r0[0], r1[1], r2[2])
        ^ t(r0[0], r1[2], r2[1])
        ^ t(r0[1], r1[0], r2[2])
        ^ t(r0[1], r1[2], r2[0])
        ^ t(r0[2], r1[0], r2[1])
        ^ t(r0[2], r1[1], r2[0])
}

/// Delta_J(l1,l2,l3): determinant of the 3x3 minor of M* on the
/// position triple `pos` (rows = pairs (12,13,23), cols = positions).
fn delta_at(gf: &Gf, norm: &Normalized, pos: &[usize; 3], lams: [u16; 3]) -> u16 {
    let pair_rows = [(0usize, 1usize), (0, 2), (1, 2)];
    let mut rows = [[0u16; 3]; 3];
    for (ri, &(pa, pb)) in pair_rows.iter().enumerate() {
        for t in 0..3 {
            let j = pos[t];
            rows[ri][t] = gf.mul(
                gval(gf, norm, pa, j, lams[pa]),
                gval(gf, norm, pb, j, lams[pb]),
            );
        }
    }
    det3_rows(gf, &rows[0], &rows[1], &rows[2])
}

/// Coefficients (A, B, C) of Delta_J as a quadratic in l3 at fixed
/// (l1, l2):  Delta_J = A l3^2 + B l3 + C.
///
/// With e_t = R_12(j_t) (l3-free), and rows R_13 = p + l3 q,
/// R_23 = r + l3 s (entrywise, over t = 0..3):
///   A = det[e;q;s],  B = det[e;q;r] + det[e;p;s],  C = det[e;p;r].
fn quad_coeffs(
    gf: &Gf,
    norm: &Normalized,
    pos: &[usize; 3],
    l1: u16,
    l2: u16,
) -> (u16, u16, u16) {
    let mut e = [0u16; 3];
    let mut p = [0u16; 3];
    let mut qq = [0u16; 3];
    let mut r = [0u16; 3];
    let mut s = [0u16; 3];
    for t in 0..3 {
        let j = pos[t];
        let g1 = gval(gf, norm, 0, j, l1);
        let g2 = gval(gf, norm, 1, j, l2);
        e[t] = gf.mul(g1, g2);
        p[t] = gf.mul(g1, norm.a[2][j]);
        qq[t] = gf.mul(g1, norm.b[2][j]);
        r[t] = gf.mul(g2, norm.a[2][j]);
        s[t] = gf.mul(g2, norm.b[2][j]);
    }
    let a_coef = det3_rows(gf, &e, &qq, &s);
    let b_coef = det3_rows(gf, &e, &qq, &r) ^ det3_rows(gf, &e, &p, &s);
    let c_coef = det3_rows(gf, &e, &p, &r);
    (a_coef, b_coef, c_coef)
}

// ---------------------------------------------------------------------------
// Stage 3: exact dual-rank test (direction A)
// ---------------------------------------------------------------------------

/// Exact test: is the candidate code [pure | g_1(l1), g_2(l2), g_3(l3)]
/// (in systematic coordinates) a GRS code?
///
/// rank M* <= 2 + smooth conic + distinct support, as documented at
/// the top of this file. Rank == 2 uses the unique relation c; the
/// rank < 2 case is passed through (stage 4's SS decides), subject to
/// the distinctness requirements, which are necessary in all cases.
pub fn dual_test(gf: &Gf, norm: &Normalized, lams: [u16; 3]) -> bool {
    let k = norm.k;
    // M*: 3 rows x k cols; column j is m_j.
    let mut mstar: Vec<Vec<u16>> = vec![vec![0u16; k]; 3];
    let mut gammas: Vec<[u16; 3]> = Vec::with_capacity(k);
    for j in 0..k {
        let g = [
            gval(gf, norm, 0, j, lams[0]),
            gval(gf, norm, 1, j, lams[1]),
            gval(gf, norm, 2, j, lams[2]),
        ];
        gammas.push(g);
        mstar[0][j] = gf.mul(g[0], g[1]);
        mstar[1][j] = gf.mul(g[0], g[2]);
        mstar[2][j] = gf.mul(g[1], g[2]);
    }
    // Genuine Gaussian elimination (NOT a Gram matrix; char-2 trap).
    let rk = mat::rank(gf, &mstar);
    if rk > 2 {
        return false;
    }
    if rk == 2 {
        // Unique (up to scale) left relation c with c . M* = 0:
        // right kernel of M*^T (k rows x 3 cols).
        let mt = mat::transpose(&mstar);
        let (_r, basis) = kernel_basis(gf, &mt);
        if basis.len() != 1 {
            return false; // cannot happen at rank 2; fail safe
        }
        let c = &basis[0];
        // Smooth conic iff no coefficient vanishes (char 2 included).
        if c[0] == 0 || c[1] == 0 || c[2] == 0 {
            return false;
        }
    }
    // Distinctness: gamma_j nonzero, pairwise non-proportional, and
    // not proportional to any of e_1, e_2, e_3 (the candidate
    // positions' dual columns). Normalize by first nonzero coordinate.
    let mut seen: HashSet<[u16; 3]> = HashSet::with_capacity(k + 3);
    seen.insert([1, 0, 0]);
    seen.insert([0, 1, 0]);
    seen.insert([0, 0, 1]);
    for g in &gammas {
        if g.iter().all(|&x| x == 0) {
            return false;
        }
        let mut gn = *g;
        for &x in g.iter() {
            if x != 0 {
                let inv = gf.inv(x);
                gn = [
                    gf.mul(g[0], inv),
                    gf.mul(g[1], inv),
                    gf.mul(g[2], inv),
                ];
                break;
            }
        }
        if !seen.insert(gn) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct FastStats {
    pub prep_secs: f64,
    /// Stages 1+2 are fused in one sweep; timed together.
    pub stage12_secs: f64,
    pub stage3_secs: f64,
    pub stage4_secs: f64,
    pub orientations_tried: usize,
    pub cells: usize,
    pub stage1_roots: usize,
    pub degenerate_cells: usize,
    pub stage2_survivors: usize,
    pub stage3_survivors: usize,
    pub ss_attempts: usize,
    pub j_swapped: bool,
}

// ---------------------------------------------------------------------------
// One orientation: stages 1-4
// ---------------------------------------------------------------------------

/// Run the full fast pipeline for one orientation (bit i of `mask`
/// set = pair i's two columns swapped). Returns the SS result
/// (alpha, y) over the k+3 candidate columns (pure first), if a
/// verified candidate was found, and accumulates stats.
#[allow(clippy::too_many_arguments)]
fn try_orientation(
    gf: &Gf,
    qs: &QuadSolver,
    pinv: &[Vec<u16>],
    pure_cols: &[Vec<u16>],
    pair_cols: &[(Vec<u16>, Vec<u16>)],
    mask: usize,
    pos_j: &[usize; 3],
    pos_j2: &[usize; 3],
    stats: &mut FastStats,
) -> Option<(Vec<u16>, Vec<u16>)> {
    let k = pure_cols.len();
    // Orient the pairs and normalize.
    let mut oriented: Vec<(Vec<u16>, Vec<u16>)> = Vec::with_capacity(3);
    for (i, (c1, c2)) in pair_cols.iter().enumerate() {
        if (mask >> i) & 1 == 1 {
            oriented.push((c2.clone(), c1.clone()));
        } else {
            oriented.push((c1.clone(), c2.clone()));
        }
    }
    let mut a: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut b: [Vec<u16>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..3 {
        a[i] = matvec(gf, pinv, &oriented[i].0);
        b[i] = matvec(gf, pinv, &oriented[i].1);
    }
    let norm = Normalized { a, b, k };

    // ---- Stages 1+2: scan (l1, l2) in F_q^2, root-find l3, filter.
    let t12 = Instant::now();
    let q = gf.order;
    let per_l1: Vec<(Vec<[u16; 3]>, usize, usize)> = (0..q)
        .into_par_iter()
        .map(|l1| {
            let l1 = l1 as u16;
            let mut surv: Vec<[u16; 3]> = Vec::new();
            let mut nroots = 0usize;
            let mut ndegen = 0usize;
            for l2i in 0..q {
                let l2 = l2i as u16;
                let (ca, cb, cc) = quad_coeffs(gf, &norm, pos_j, l1, l2);
                match solve_quad(gf, qs, ca, cb, cc) {
                    QuadRoots::None => {}
                    QuadRoots::All => {
                        // Degenerate cell: Delta_J vanishes for every
                        // l3. Filter by the second minor directly.
                        ndegen += 1;
                        for l3i in 0..q {
                            let l3 = l3i as u16;
                            nroots += 1;
                            if delta_at(gf, &norm, pos_j2, [l1, l2, l3]) == 0 {
                                surv.push([l1, l2, l3]);
                            }
                        }
                    }
                    QuadRoots::One(l3) => {
                        nroots += 1;
                        if delta_at(gf, &norm, pos_j2, [l1, l2, l3]) == 0 {
                            surv.push([l1, l2, l3]);
                        }
                    }
                    QuadRoots::Two(l3a, l3b) => {
                        nroots += 2;
                        for l3 in [l3a, l3b] {
                            if delta_at(gf, &norm, pos_j2, [l1, l2, l3]) == 0 {
                                surv.push([l1, l2, l3]);
                            }
                        }
                    }
                }
            }
            (surv, nroots, ndegen)
        })
        .collect();
    let mut stage2: Vec<[u16; 3]> = Vec::new();
    for (surv, nroots, ndegen) in per_l1 {
        stats.stage1_roots += nroots;
        stats.degenerate_cells += ndegen;
        stage2.extend(surv);
    }
    stats.cells += q * q;
    stats.stage2_survivors += stage2.len();
    stats.stage12_secs += t12.elapsed().as_secs_f64();

    // ---- Stage 3: exact dual-rank test per survivor.
    let t3 = Instant::now();
    let stage3: Vec<[u16; 3]> = stage2
        .into_iter()
        .filter(|&lams| dual_test(gf, &norm, lams))
        .collect();
    stats.stage3_survivors += stage3.len();
    stats.stage3_secs += t3.elapsed().as_secs_f64();

    // ---- Stage 4: SS on original-coordinate candidates.
    let t4 = Instant::now();
    for lams in &stage3 {
        stats.ss_attempts += 1;
        let mut cols: Vec<Vec<u16>> = pure_cols.to_vec();
        let mut ok = true;
        for (i, (base, other)) in oriented.iter().enumerate() {
            let lam = lams[i];
            let mut v = vec![0u16; k];
            for l in 0..k {
                v[l] = base[l] ^ gf.mul(lam, other[l]);
            }
            if v.iter().all(|&x| x == 0) {
                ok = false;
                break;
            }
            cols.push(v);
        }
        if !ok {
            continue;
        }
        let m_rows = mat::transpose(&cols);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ss::ss_attack(gf, &m_rows)
        }));
        if let Ok(Ok((alpha, y))) = res {
            stats.stage4_secs += t4.elapsed().as_secs_f64();
            return Some((alpha, y));
        }
    }
    stats.stage4_secs += t4.elapsed().as_secs_f64();
    None
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Fast bootstrap for need = 3: find (alpha, y) of the GRS code on
/// [pure columns | 3 unmixed pair columns], via direction B.
///
/// * `pure_cols`: exactly k pure columns (each length k).
/// * `pair_cols`: exactly 3 pairs, each the pair's two public columns.
///
/// Returns (Some((alpha, y)), stats) on success -- alpha, y have
/// length k+3, aligned with [pure_cols..., cand_1, cand_2, cand_3] --
/// or (None, stats) if no orientation produced a verified candidate
/// (the caller then falls back to the q^3 scan).
pub fn fast_bootstrap(
    gf: &Gf,
    pure_cols: &[Vec<u16>],
    pair_cols: &[(Vec<u16>, Vec<u16>)],
) -> (Option<(Vec<u16>, Vec<u16>)>, FastStats) {
    let mut stats = FastStats::default();
    let k = pure_cols.len();
    if pair_cols.len() != 3 || k < 6 || pure_cols[0].len() != k {
        return (None, stats);
    }
    let t0 = Instant::now();
    // P^{-1} once (shared across orientations).
    let p_rows = mat::transpose(pure_cols);
    let pinv = match mat::matinv(gf, &p_rows) {
        Some(m) => m,
        None => return (None, stats),
    };
    let qs = QuadSolver::new(gf);
    stats.prep_secs = t0.elapsed().as_secs_f64();

    // Position triples: primary J, disjoint filter triple J'.
    let mut pos_j = [0usize, 1, 2];
    let mut pos_j2 = [3usize, 4, 5];

    for mask in 0..8usize {
        stats.orientations_tried += 1;
        let degen_before = stats.degenerate_cells;
        let cells_before = stats.cells;
        let res = try_orientation(
            gf, &qs, &pinv, pure_cols, pair_cols, mask, &pos_j, &pos_j2, &mut stats,
        );
        if res.is_some() {
            return (res, stats);
        }
        // If the primary triple J was degenerate on more than half of
        // the cells of the first orientation (Delta_J essentially
        // uninformative), swap the roles of J and J' for the rest.
        if mask == 0 && stats.cells > cells_before {
            let cells_here = stats.cells - cells_before;
            let degen_here = stats.degenerate_cells - degen_before;
            if degen_here * 2 > cells_here {
                std::mem::swap(&mut pos_j, &mut pos_j2);
                stats.j_swapped = true;
            }
        }
    }
    (None, stats)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::rlce;

    /// z^2+z solver + full quadratic solver: brute-force cross-check
    /// over GF(64) on random (A,B,C), covering all branches.
    #[test]
    fn quad_solver_cross_check() {
        let gf = Gf::new(6);
        let qs = QuadSolver::new(&gf);
        let mut rng = Rng::new(4242);
        let q = gf.order as u16;
        // Branch-targeted cases first.
        // A=0, B!=0: unique root C/B.
        for _ in 0..50 {
            let b = (rng.below(63) + 1) as u16;
            let c = rng.below(64) as u16;
            match solve_quad(&gf, &qs, 0, b, c) {
                QuadRoots::One(x) => assert_eq!(gf.mul(b, x), c),
                other => panic!("A=0,B!=0: got {:?}", other),
            }
        }
        // A=0,B=0: All iff C=0 else None.
        assert_eq!(solve_quad(&gf, &qs, 0, 0, 0), QuadRoots::All);
        assert_eq!(solve_quad(&gf, &qs, 0, 0, 5), QuadRoots::None);
        // A!=0, B=0: unique Frobenius root.
        for _ in 0..50 {
            let a = (rng.below(63) + 1) as u16;
            let c = rng.below(64) as u16;
            match solve_quad(&gf, &qs, a, 0, c) {
                QuadRoots::One(x) => {
                    assert_eq!(gf.mul(a, gf.mul(x, x)), c, "A x^2 = C failed")
                }
                other => panic!("B=0: got {:?}", other),
            }
        }
        // Trace-0 (two roots) and trace-1 (none) with A=B=1:
        // z^2+z=c solvable iff Tr(c)=0; pick c from the map itself.
        {
            let z0 = 37u16;
            let c = gf.mul(z0, z0) ^ z0;
            match solve_quad(&gf, &qs, 1, 1, c) {
                QuadRoots::Two(x1, x2) => {
                    assert!(x1 != x2);
                    for x in [x1, x2] {
                        assert_eq!(gf.mul(x, x) ^ x, c);
                    }
                }
                other => panic!("trace-0: got {:?}", other),
            }
            // find a c with Tr(c)=1 by scanning
            let mut bad: Option<u16> = None;
            for cand in 0..64u16 {
                if qs.solve_z2z(cand).is_none() {
                    bad = Some(cand);
                    break;
                }
            }
            let bad = bad.expect("some c must have Tr(c)=1");
            assert_eq!(solve_quad(&gf, &qs, 1, 1, bad), QuadRoots::None);
        }
        // Random brute-force agreement (skip the All case: A=B=C=0).
        for _ in 0..400 {
            let a = rng.below(64) as u16;
            let b = rng.below(64) as u16;
            let c = rng.below(64) as u16;
            if a == 0 && b == 0 && c == 0 {
                continue;
            }
            let mut brute: Vec<u16> = Vec::new();
            for x in 0..q {
                if gf.mul(a, gf.mul(x, x)) ^ gf.mul(b, x) ^ c == 0 {
                    brute.push(x);
                }
            }
            let got: Vec<u16> = match solve_quad(&gf, &qs, a, b, c) {
                QuadRoots::None => Vec::new(),
                QuadRoots::One(x) => vec![x],
                QuadRoots::Two(x, y) => vec![x, y],
                QuadRoots::All => panic!("unexpected All"),
            };
            assert_eq!(got.len(), brute.len(), "count mismatch A={} B={} C={}", a, b, c);
            for x in &got {
                assert!(brute.contains(x), "root {} not in brute set", x);
            }
        }
    }

    // ---- helpers for GRS-instance tests -----------------------------

    fn grs_col(gf: &Gf, x: u16, y: u16, k: usize) -> Vec<u16> {
        let mut col = vec![0u16; k];
        let mut xp = 1u16;
        for i in 0..k {
            col[i] = gf.mul(y, xp);
            xp = gf.mul(xp, x);
        }
        col
    }

    fn rand_invertible(gf: &Gf, rng: &mut Rng, k: usize) -> Vec<Vec<u16>> {
        loop {
            let m: Vec<Vec<u16>> = (0..k)
                .map(|_| (0..k).map(|_| rng.below(gf.order as u64) as u16).collect())
                .collect();
            if mat::matinv(gf, &m).is_some() {
                return m;
            }
        }
    }

    fn apply_mat(gf: &Gf, m: &[Vec<u16>], v: &[u16]) -> Vec<u16> {
        matvec(gf, m, v)
    }

    fn lin_comb(gf: &Gf, s: u16, u: &[u16], t: u16, v: &[u16]) -> Vec<u16> {
        u.iter()
            .zip(v.iter())
            .map(|(&x, &y)| gf.mul(s, x) ^ gf.mul(t, y))
            .collect()
    }

    /// A synthetic instance: k pure columns (scrambled GRS) and 3 pairs
    /// with known mixing. Returns (pure_cols, pair_cols, lambda_stars)
    /// with candidate convention v = c1 + l*c2, l* = c/d.
    fn build_instance(
        gf: &Gf,
        rng: &mut Rng,
        k: usize,
    ) -> (Vec<Vec<u16>>, Vec<(Vec<u16>, Vec<u16>)>, Vec<u16>) {
        let q = gf.order;
        // distinct nonzero support points for pure + pair columns
        let mut used = vec![false; q];
        used[0] = true;
        let mut next_x = move |rng: &mut Rng| -> u16 {
            loop {
                let x = rng.below(q as u64) as usize;
                if !used[x] {
                    used[x] = true;
                    return x as u16;
                }
            }
        };
        let s_mat = rand_invertible(gf, rng, k);
        let mut pure = Vec::with_capacity(k);
        for _ in 0..k {
            let x = next_x(rng);
            let y = (rng.below((q - 1) as u64) + 1) as u16;
            pure.push(apply_mat(gf, &s_mat, &grs_col(gf, x, y, k)));
        }
        let mut pairs = Vec::with_capacity(3);
        let mut stars = Vec::with_capacity(3);
        for _ in 0..3 {
            let x = next_x(rng);
            let y = (rng.below((q - 1) as u64) + 1) as u16;
            let g = apply_mat(gf, &s_mat, &grs_col(gf, x, y, k));
            let r: Vec<u16> = (0..k).map(|_| rng.below(q as u64) as u16).collect();
            // random GL(2) [[a,b],[c,d]] with d != 0 (finite lambda*)
            let (a, b, c, d) = loop {
                let a = rng.below(q as u64) as u16;
                let b = rng.below(q as u64) as u16;
                let c = rng.below(q as u64) as u16;
                let d = rng.below(q as u64) as u16;
                if d != 0 && (gf.mul(a, d) ^ gf.mul(b, c)) != 0 {
                    break (a, b, c, d);
                }
            };
            let c1 = lin_comb(gf, a, &g, c, &r);
            let c2 = lin_comb(gf, b, &g, d, &r);
            pairs.push((c1, c2));
            stars.push(gf.mul(c, gf.inv(d))); // c + l d = 0 -> l = c/d
        }
        (pure, pairs, stars)
    }

    /// Delta_J coefficients vs brute-force evaluation at random
    /// (l1, l2, l3).
    #[test]
    fn delta_coeffs_vs_brute_force() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(777777);
        let k = 9;
        let (pure, pairs, _stars) = build_instance(&gf, &mut rng, k);
        let norm = normalize(&gf, &pure, &pairs).expect("pure basis");
        let pos = [0usize, 1, 2];
        for _ in 0..300 {
            let l1 = rng.below(64) as u16;
            let l2 = rng.below(64) as u16;
            let (ca, cb, cc) = quad_coeffs(&gf, &norm, &pos, l1, l2);
            for _ in 0..6 {
                let l3 = rng.below(64) as u16;
                let poly = gf.mul(ca, gf.mul(l3, l3)) ^ gf.mul(cb, l3) ^ cc;
                let direct = delta_at(&gf, &norm, &pos, [l1, l2, l3]);
                assert_eq!(poly, direct, "coeff mismatch at ({},{},{})", l1, l2, l3);
            }
        }
    }

    /// Dual-rank test vs ss_attack agreement on true and wrong
    /// candidates at small k.
    #[test]
    fn dual_test_vs_ss() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(31337);
        let k = 8;
        let (pure, pairs, stars) = build_instance(&gf, &mut rng, k);
        let norm = normalize(&gf, &pure, &pairs).expect("pure basis");
        let lstar = [stars[0], stars[1], stars[2]];
        // True candidate: dual test accepts, SS succeeds.
        assert!(dual_test(&gf, &norm, lstar), "true triple rejected by dual test");
        {
            let mut cols = pure.clone();
            for (i, (c1, c2)) in pairs.iter().enumerate() {
                cols.push(lin_comb(&gf, 1, c1, lstar[i], c2));
            }
            let rows = mat::transpose(&cols);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ss::ss_attack(&gf, &rows)
            }));
            assert!(matches!(res, Ok(Ok(_))), "SS must succeed on true candidate");
        }
        // Wrong candidates: perturb one coordinate; agreement required.
        let mut n_wrong = 0;
        let mut attempts = 0;
        while n_wrong < 40 && attempts < 400 {
            attempts += 1;
            let mut lams = lstar;
            let idx = (rng.below(3)) as usize;
            let delta = (rng.below(63) + 1) as u16;
            lams[idx] ^= delta;
            if lams == lstar {
                continue;
            }
            n_wrong += 1;
            let dt = dual_test(&gf, &norm, lams);
            let mut cols = pure.clone();
            for (i, (c1, c2)) in pairs.iter().enumerate() {
                cols.push(lin_comb(&gf, 1, c1, lams[i], c2));
            }
            let rows = mat::transpose(&cols);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ss::ss_attack(&gf, &rows)
            }));
            let ss_ok = matches!(res, Ok(Ok(_)));
            assert_eq!(dt, ss_ok, "dual test / SS disagree at {:?}", lams);
            assert!(!ss_ok, "wrong candidate unexpectedly GRS");
        }
        assert!(n_wrong >= 40, "not enough wrong samples: {}", n_wrong);
    }

    /// lambda = infinity: pair 1 has d = 0, so its true unmixing is
    /// the second column alone. Orientation 0 must fail; the flip path
    /// inside fast_bootstrap must succeed.
    #[test]
    fn degenerate_pair_flip() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(90210);
        let k = 10;
        let q = gf.order;
        let mut used = vec![false; q];
        used[0] = true;
        let mut next_x = move |rng: &mut Rng| -> u16 {
            loop {
                let x = rng.below(q as u64) as usize;
                if !used[x] {
                    used[x] = true;
                    return x as u16;
                }
            }
        };
        let s_mat = rand_invertible(&gf, &mut rng, k);
        let mut pure = Vec::with_capacity(k);
        for _ in 0..k {
            let x = next_x(&mut rng);
            let y = (rng.below((q - 1) as u64) + 1) as u16;
            pure.push(apply_mat(&gf, &s_mat, &grs_col(&gf, x, y, k)));
        }
        let mut pairs: Vec<(Vec<u16>, Vec<u16>)> = Vec::with_capacity(3);
        for pi in 0..3 {
            let x = next_x(&mut rng);
            let y = (rng.below((q - 1) as u64) + 1) as u16;
            let g = apply_mat(&gf, &s_mat, &grs_col(&gf, x, y, k));
            let r: Vec<u16> = (0..k).map(|_| rng.below(q as u64) as u16).collect();
            let (c1, c2);
            if pi == 1 {
                // Degenerate: d = 0, b,c != 0 -> det = b*c != 0.
                // c2 = b*g is the pure column itself (lambda = infinity).
                let a = (rng.below(63) + 1) as u16;
                let b = (rng.below(63) + 1) as u16;
                let c = (rng.below(63) + 1) as u16;
                c1 = lin_comb(&gf, a, &g, c, &r);
                c2 = lin_comb(&gf, b, &g, 0, &r);
            } else {
                let (a, b, c, d) = loop {
                    let a = rng.below(q as u64) as u16;
                    let b = rng.below(q as u64) as u16;
                    let c = rng.below(q as u64) as u16;
                    let d = rng.below(q as u64) as u16;
                    if d != 0 && (gf.mul(a, d) ^ gf.mul(b, c)) != 0 {
                        break (a, b, c, d);
                    }
                };
                c1 = lin_comb(&gf, a, &g, c, &r);
                c2 = lin_comb(&gf, b, &g, d, &r);
            }
            pairs.push((c1, c2));
        }
        // Orientation 0 alone must fail (pair 1 unmixable finitely).
        let p_rows = mat::transpose(&pure);
        let pinv = mat::matinv(&gf, &p_rows).expect("pinv");
        let qs = QuadSolver::new(&gf);
        let mut st = FastStats::default();
        let r0 = try_orientation(
            &gf, &qs, &pinv, &pure, &pairs, 0, &[0, 1, 2], &[3, 4, 5], &mut st,
        );
        assert!(r0.is_none(), "orientation 0 must fail on the degenerate pair");
        // Full fast_bootstrap (with flips) must succeed, and the
        // recovered GRS must match the pure columns (stacked rank).
        let (res, stats) = fast_bootstrap(&gf, &pure, &pairs);
        let (alpha, y) = res.expect("fast_bootstrap must succeed via a flip");
        assert!(stats.orientations_tried >= 2, "expected a flip to be used");
        assert_eq!(alpha.len(), k + 3);
        // Stacked-rank check on the pure positions (as in attack.rs).
        let cand_rows: Vec<Vec<u16>> = (0..k)
            .map(|r| {
                (0..k)
                    .map(|cc| {
                        let mut pw = 1u16;
                        for _ in 0..r {
                            pw = gf.mul(pw, alpha[cc]);
                        }
                        gf.mul(y[cc], pw)
                    })
                    .collect()
            })
            .collect();
        let mut stacked = cand_rows;
        stacked.extend(mat::transpose(&pure));
        assert_eq!(mat::rank(&gf, &stacked), k);
    }

    /// fast_bootstrap at the existing mocked-Phase-1 point
    /// (n=20, k=14, w=6, q=64, w = n-k): recovers the GRS.
    #[test]
    fn fast_bootstrap_w_eq_n_minus_k() {
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(777);
        let key = rlce::gen_rlce(&gf, &mut rng, 20, 14, 6);
        let k = key.k;
        let pure_idx: Vec<usize> = (0..key.n_pub).filter(|&i| key.is_pure[i]).collect();
        assert_eq!(pure_idx.len(), k);
        let pure_cols: Vec<Vec<u16>> = pure_idx.iter().map(|&i| key.cols[i].clone()).collect();
        let pair_cols: Vec<(Vec<u16>, Vec<u16>)> = key.pairs[..3]
            .iter()
            .map(|&(i, j)| (key.cols[i].clone(), key.cols[j].clone()))
            .collect();
        let (res, stats) = fast_bootstrap(&gf, &pure_cols, &pair_cols);
        let (alpha, y) = res.expect("fast_bootstrap must find the GRS");
        assert_eq!(alpha.len(), k + 3);
        println!(
            "fast stats: cells={} roots={} degen={} s2={} s3={} ss={} t12={:.3}s t3={:.3}s t4={:.3}s",
            stats.cells, stats.stage1_roots, stats.degenerate_cells,
            stats.stage2_survivors, stats.stage3_survivors, stats.ss_attempts,
            stats.stage12_secs, stats.stage3_secs, stats.stage4_secs
        );
        // Verify on pure positions.
        let cand_rows: Vec<Vec<u16>> = (0..k)
            .map(|r| {
                (0..k)
                    .map(|cc| {
                        let mut pw = 1u16;
                        for _ in 0..r {
                            pw = gf.mul(pw, alpha[cc]);
                        }
                        gf.mul(y[cc], pw)
                    })
                    .collect()
            })
            .collect();
        let mut stacked = cand_rows;
        stacked.extend(mat::transpose(&pure_cols));
        assert_eq!(mat::rank(&gf, &stacked), k);
    }

    /// Phase-2-only integration at the real target point
    /// (q=256, n=198, k=165, w=33): keygen + ground-truth Phase 1
    /// (pure set and pairing from the key, as Phase 1 would output),
    /// then ONLY the fast Phase 2. Expected: seconds on this VM,
    /// vs ~36 h for the q^3 scan.
    #[test]
    fn fast_bootstrap_target_256_165_33() {
        let gf = Gf::for_q(256);
        let mut rng = Rng::new(20261021);
        let key = rlce::gen_rlce(&gf, &mut rng, 198, 165, 33);
        let k = key.k;
        assert_eq!(key.n_pub, 231);
        let pure_idx: Vec<usize> = (0..key.n_pub).filter(|&i| key.is_pure[i]).collect();
        assert_eq!(pure_idx.len(), k);
        let pure_cols: Vec<Vec<u16>> = pure_idx.iter().map(|&i| key.cols[i].clone()).collect();
        let pair_cols: Vec<(Vec<u16>, Vec<u16>)> = key.pairs[..3]
            .iter()
            .map(|&(i, j)| (key.cols[i].clone(), key.cols[j].clone()))
            .collect();
        let t0 = Instant::now();
        let (res, stats) = fast_bootstrap(&gf, &pure_cols, &pair_cols);
        let elapsed = t0.elapsed().as_secs_f64();
        println!(
            "target (256,198,165,33): elapsed={:.2}s prep={:.3}s stage12={:.3}s stage3={:.3}s stage4={:.3}s",
            elapsed, stats.prep_secs, stats.stage12_secs, stats.stage3_secs, stats.stage4_secs
        );
        println!(
            "  cells={} roots={} degen={} stage2_surv={} stage3_surv={} ss_attempts={} orientations={}",
            stats.cells, stats.stage1_roots, stats.degenerate_cells,
            stats.stage2_survivors, stats.stage3_survivors, stats.ss_attempts,
            stats.orientations_tried
        );
        let (alpha, y) = res.expect("fast_bootstrap must succeed at the target point");
        assert_eq!(alpha.len(), k + 3);
        // Verify the recovered GRS on the pure positions.
        let cand_rows: Vec<Vec<u16>> = (0..k)
            .map(|r| {
                (0..k)
                    .map(|cc| {
                        let mut pw = 1u16;
                        for _ in 0..r {
                            pw = gf.mul(pw, alpha[cc]);
                        }
                        gf.mul(y[cc], pw)
                    })
                    .collect()
            })
            .collect();
        let mut stacked = cand_rows;
        stacked.extend(mat::transpose(&pure_cols));
        assert_eq!(mat::rank(&gf, &stacked), k);
        assert!(elapsed < 120.0, "fast Phase 2 too slow: {:.1}s", elapsed);
    }
}
