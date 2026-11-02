//! Sigma-basis for block Krylov sequences via a minimal approximant basis
//! (M-Basis, Beckermann--Labahn) of an AUGMENTED series. This replaces the
//! dense block-Toeplitz nullspace extraction of `matfree::block_wiedemann_kernel`
//! (an (L*b)^2 dense matrix, ~(L*b)^3/3 elimination) with an extraction whose
//! memory is O(b^2 * sigma): at fixed block size b it scales LINEARLY in the
//! sequence length. The (L*b) x (L*b) Toeplitz matrix is never formed.
//!
//! WHAT IS COMPUTED, PRECISELY.
//! Let S_k (k = 0..sigma-1) be the b x b block Krylov projections
//! S_k = U^T A^k B0 of the (symmetric) operator A, and
//!     Scr(x) = sum_k S_k x^k,   T(x) = Scr(x)^T  (coefficientwise transpose).
//! The block-Wiedemann extraction needs, for each candidate, a polynomial
//! vector F(x) = sum_j F_j x^j (F_j in F_q^b) and delta = deg F such that
//! Z = sum_j A^{delta-j} B0 F_j is (nearly) in ker A. Projecting,
//!     U^T A^i (A Z) = sum_j S_{i+1+delta-j} F_j = (Scr F)_{i+1+delta},
//! so it suffices that Scr * F have vanishing coefficients on the window
//! [delta+1, sigma): i.e. (F, G) is a right Hermite--Pade pair for Scr,
//!     Scr * F == G (mod x^sigma),   deg G < deg F = delta <= L.
//! Such pairs are produced by a minimal approximant basis of the augmented
//! series M(x) = [ T(x) ; I ]  (2b x b), as follows. A basis row is a pair
//! (f, g) of 1 x b polynomial vectors with
//!     f * T + g == 0 (mod x^sigma).
//! Transposing, Scr * f^T == -g^T (mod x^sigma): a right Pade pair for Scr
//! with F = f^T, G = -g^T (char 2: -g^T = g^T). We compute the approximant
//! basis w.r.t. the shift s = (0,...,0, 1,...,1) (0 on the T-block columns
//! of the basis matrix, 1 on the I-block columns). The shifted degree of a
//! row (f, g) is then max(deg f, deg g + 1), so rows whose shifted leading
//! position lies in the T-block ("F-pivot rows") satisfy deg g < deg f:
//! exactly the Pade property above. This is the classical identification of
//! Coppersmith's sigma-basis / Hermite--Pade bases with minimal approximant
//! bases (Beckermann--Labahn; Giorgi--Jeannerod--Villard's PM-Basis is the
//! divide-and-conquer version of the iteration implemented here; we use the
//! linear M-Basis iteration itself: O(sigma^2 * b^3)-ish field ops with
//! small constants, O(sigma * b^2) memory).
//!
//! ALGORITHM (M-Basis with shift, rectangular series M in F[x]^{m x n},
//! here m = 2b, n = b). Maintain a basis candidate P in F[x]^{m x m} and the
//! residual R = P * M (mod x^sigma). Invariant after step k:
//! R == 0 (mod x^{k+1}). Initially P = I, R = M. At step k, let
//! Delta = [x^k] R (an m x n constant matrix). Process rows in increasing
//! shifted degree rho_i = max_j (deg P_{ij} + s_j): greedily select pivot
//! rows -- a row whose Delta-row is not in the span of previously selected
//! pivots' Delta-rows becomes a pivot; the others are reduced to zero by
//! elementary row operations using only pivots of shifted degree <= their
//! own (recorded as ops; the accumulated transform Theta is unitriangular
//! in processing order, hence invertible, and the degree-greedy choice keeps
//! the shifted leading matrix of P invertible -- the standard M-Basis
//! argument). Then P <- D * Theta * P where D multiplies pivot rows by x,
//! and R is updated by the same ops; the x-multiplication of residual rows
//! is tracked by a per-row offset (row i of R is stored pre-shifted), so R
//! is never physically shifted. After sigma steps, P * M == 0 (mod x^sigma)
//! and P is an s-shifted weak-Popov approximant basis of M.
//!
//! Storage: P has <= sigma+1 coefficient matrices of size (2b)^2; the
//! residual is 2b rows x sigma positions x b entries; the input series is
//! sigma * b^2. Total O(sigma * b^2) field elements -- versus
//! (L*b)^2 = (sigma*b/2)^2 for the dense Toeplitz extraction.

use crate::gf::Gf;

/// Output of the augmented sigma-basis computation: a minimal approximant
/// basis P of M = [T; I] (2b x 2b matrix polynomial), together with the
/// per-row degrees of its T-block ("F-part", columns 0..b) and I-block
/// ("G-part", columns b..2b). Degree usize::MAX encodes an identically
/// zero part.
pub struct SigmaBasisAug {
    /// coeffs[j] = coefficient of x^j in P: flat (2b) x (2b), row-major.
    pub coeffs: Vec<Vec<u16>>,
    /// f_deg[i] = degree of row i's F-part (max degree over columns 0..b).
    pub f_deg: Vec<usize>,
    /// g_deg[i] = degree of row i's G-part (max degree over columns b..2b).
    pub g_deg: Vec<usize>,
    /// True if row i's F-part is identically zero.
    pub f_zero: Vec<bool>,
    pub b: usize,
}

impl SigmaBasisAug {
    pub fn rows(&self) -> usize {
        2 * self.b
    }

    /// Candidate polynomial for basis row `i`: the coefficient vectors
    /// g_j = (row i of coeffs[j], F-part columns 0..b), j = 0..=f_deg[i],
    /// i.e. F(x) = sum_j g_j x^j (read as a column-vector polynomial)
    /// satisfies Scr * F == G (mod x^sigma) with G from the G-part.
    pub fn candidate_coeffs(&self, i: usize) -> Vec<Vec<u16>> {
        let b = self.b;
        let m = 2 * b;
        let delta = self.f_deg[i];
        (0..=delta)
            .map(|j| {
                let cj = &self.coeffs[j];
                (0..b).map(|c| cj[i * m + c]).collect::<Vec<u16>>()
            })
            .collect()
    }

    /// The G-part coefficient vectors of row i (the Pade remainder):
    /// h_j = (row i of coeffs[j], G-part columns b..2b), j = 0..=g_deg[i].
    pub fn remainder_coeffs(&self, i: usize) -> Vec<Vec<u16>> {
        let b = self.b;
        let m = 2 * b;
        if self.g_deg[i] == usize::MAX {
            return Vec::new();
        }
        (0..=self.g_deg[i])
            .map(|j| {
                let cj = &self.coeffs[j];
                (0..b).map(|c| cj[i * m + b + c]).collect::<Vec<u16>>()
            })
            .collect()
    }
}

/// Minimal approximant basis (M-Basis) of the augmented series
/// M(x) = [ T(x) ; I ], where `series_t[k]` is the coefficient T_k of T,
/// flat b x b row-major. Order `sigma` (uses T_0..T_{sigma-1}; missing
/// coefficients are read as zero). Shift s = (0_b, 1_b).
pub fn sigma_basis_aug(gf: &Gf, series_t: &[Vec<u16>], sigma: usize, b: usize) -> SigmaBasisAug {
    let m = 2 * b;
    let n = b;
    if b == 0 || sigma == 0 {
        // Degenerate: identity basis.
        let mut id = vec![0u16; m * m];
        for i in 0..m {
            id[i * m + i] = 1;
        }
        return SigmaBasisAug {
            coeffs: vec![id],
            f_deg: (0..m).map(|i| if i < b { 0 } else { usize::MAX }).collect(),
            g_deg: (0..m).map(|i| if i < b { usize::MAX } else { 0 }).collect(),
            f_zero: (0..m).map(|i| i >= b).collect(),
            b,
        };
    }
    // Shift: s_j = 0 for j < b (T-block), 1 for j >= b (I-block).
    let shift = |j: usize| -> i64 {
        if j < b {
            0
        } else {
            1
        }
    };

    // P: coefficient list; coeffs[j] flat m x m row-major. P = I initially.
    let mut p: Vec<Vec<u16>> = Vec::with_capacity(sigma + 1);
    {
        let mut id = vec![0u16; m * m];
        for i in 0..m {
            id[i * m + i] = 1;
        }
        p.push(id);
    }
    // rho[i] = shifted degree of row i of P. Initially row i has its single
    // 1 at column i: rho_i = s_i. pdeg[i] = an upper bound on the (plain)
    // degree of row i of P, maintained incrementally (used only to size the
    // x-shift loop; exact degrees are recomputed at the end).
    let mut rho: Vec<i64> = (0..m).map(|i| shift(i)).collect();
    let mut pdeg = vec![0usize; m];

    // Residual R = P * M (mod x^sigma), stored per row i as a flat array
    // over stored positions u in 0..sigma of row-vectors of length n:
    // the true coefficient of x^t in row i is res[i][(t - off[i]) * n + c]
    // when 0 <= t - off[i] < sigma (for t < sigma the invariant of the
    // algorithm makes the out-of-range cases read as the correct zeros).
    // Rows of M: for i < b, M's row i is T's row i (series); for i >= b,
    // M's row i is e_{i-b} at t = 0.
    let mut res: Vec<Vec<u16>> = vec![vec![0u16; sigma * n]; m];
    for i in 0..b {
        for t in 0..sigma {
            if t < series_t.len() {
                let tk = &series_t[t];
                for c in 0..n {
                    res[i][t * n + c] = tk[i * b + c];
                }
            }
        }
    }
    for i in b..m {
        res[i][i - b] = 1; // t = 0 position, column i-b
    }
    let mut off = vec![0usize; m];

    // Scratch: current discrepancy rows.
    let mut dcur: Vec<Vec<u16>> = vec![vec![0u16; n]; m];

    for k in 0..sigma {
        // Delta = [x^k] R (m x n).
        let mut any = false;
        for i in 0..m {
            if k >= off[i] && k - off[i] < sigma {
                let u = k - off[i];
                dcur[i].copy_from_slice(&res[i][u * n..u * n + n]);
                if dcur[i].iter().any(|&v| v != 0) {
                    any = true;
                }
            } else {
                dcur[i].fill(0);
            }
        }
        if !any {
            continue;
        }
        // Greedy pivot selection in increasing (rho_i, i).
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by_key(|&i| (rho[i], i));
        let mut pivots: Vec<(usize, usize, u16)> = Vec::new(); // (row, lead col, lead val)
        let mut is_pivot = vec![false; m];
        // Elementary ops (target, source, coef): row_target += coef *
        // row_source, in application order; replayed on res and P.
        let mut ops: Vec<(usize, usize, u16)> = Vec::new();
        for &i in order.iter() {
            for &(prow, lead, lval) in pivots.iter() {
                let cur = dcur[i][lead];
                if cur != 0 {
                    let coef = gf.mul(cur, gf.inv(lval));
                    // dcur[i] += coef * dcur[prow] (char 2: kills entry `lead`)
                    for c in 0..n {
                        let pv = dcur[prow][c];
                        if pv != 0 {
                            let add = gf.mul(coef, pv);
                            dcur[i][c] ^= add;
                        }
                    }
                    ops.push((i, prow, coef));
                }
            }
            if dcur[i].iter().any(|&v| v != 0) {
                let lead = dcur[i].iter().position(|&v| v != 0).unwrap();
                pivots.push((i, lead, dcur[i][lead]));
                is_pivot[i] = true;
            }
        }
        let _ = is_pivot;
        if pivots.is_empty() {
            continue; // cannot happen when `any` (first nonzero row pivots)
        }
        // Apply ops to the residual at absolute positions k..sigma.
        // Index arithmetic: pos >= k >= off[*] (offsets grow by at most 1
        // per completed step), and pos < sigma, so both stored indices
        // lie in [0, sigma).
        for &(tgt, src, coef) in ops.iter() {
            let (ot, os) = (off[tgt], off[src]);
            for pos in k..sigma {
                let ut = pos - ot;
                let us = pos - os;
                for c in 0..n {
                    let sv = res[src][us * n + c];
                    if sv != 0 {
                        let add = gf.mul(coef, sv);
                        res[tgt][ut * n + c] ^= add;
                    }
                }
            }
        }
        // Apply ops to P (every coefficient matrix) and track degree bound.
        for &(tgt, src, coef) in ops.iter() {
            for pj in p.iter_mut() {
                for c in 0..m {
                    let sv = pj[src * m + c];
                    if sv != 0 {
                        let add = gf.mul(coef, sv);
                        pj[tgt * m + c] ^= add;
                    }
                }
            }
            if pdeg[src] > pdeg[tgt] {
                pdeg[tgt] = pdeg[src];
            }
        }
        // D: multiply pivot rows of P by x (coefficient shift), and track
        // the residual offset / shifted degree.
        for &(prow, _, _) in pivots.iter() {
            let rdeg = pdeg[prow];
            if rdeg + 1 >= p.len() {
                p.push(vec![0u16; m * m]);
            }
            for j in (1..=rdeg + 1).rev() {
                let src_row: Vec<u16> = p[j - 1][prow * m..(prow + 1) * m].to_vec();
                p[j][prow * m..(prow + 1) * m].copy_from_slice(&src_row);
            }
            p[0][prow * m..(prow + 1) * m].fill(0);
            pdeg[prow] = rdeg + 1;
            off[prow] += 1;
            rho[prow] += 1;
        }
    }

    // Trim trailing zero coefficient matrices (keep at least one).
    while p.len() > 1 {
        let last = p.len() - 1;
        if p[last].iter().all(|&v| v == 0) {
            p.pop();
        } else {
            break;
        }
    }
    // Per-row F/G degrees (exact, by scanning).
    let mut f_deg = vec![usize::MAX; m];
    let mut g_deg = vec![usize::MAX; m];
    let mut f_zero = vec![true; m];
    for i in 0..m {
        for (j, pj) in p.iter().enumerate() {
            let row = &pj[i * m..(i + 1) * m];
            if row[..b].iter().any(|&v| v != 0) {
                f_deg[i] = j;
                f_zero[i] = false;
            }
            if row[b..].iter().any(|&v| v != 0) {
                g_deg[i] = j;
            }
        }
    }
    SigmaBasisAug {
        coeffs: p,
        f_deg,
        g_deg,
        f_zero,
        b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rand_mat(gf: &Gf, rng: &mut Rng, b: usize) -> Vec<u16> {
        let q = gf.order as u64;
        (0..b * b).map(|_| rng.below(q) as u16).collect()
    }

    fn mat_mul(gf: &Gf, a: &[u16], bb: &[u16], b: usize) -> Vec<u16> {
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

    fn mat_inv(gf: &Gf, a: &[u16], b: usize) -> Option<Vec<u16>> {
        // Gauss-Jordan on [a | I].
        let mut aug: Vec<Vec<u16>> = (0..b)
            .map(|i| {
                let mut r: Vec<u16> = a[i * b..(i + 1) * b].to_vec();
                r.extend((0..b).map(|j| if j == i { 1 } else { 0 }));
                r
            })
            .collect();
        let mut row = 0usize;
        for col in 0..b {
            let mut piv = None;
            for r in row..b {
                if aug[r][col] != 0 {
                    piv = Some(r);
                    break;
                }
            }
            let pr = piv?;
            aug.swap(row, pr);
            let inv = gf.inv(aug[row][col]);
            for c in 0..2 * b {
                let v = aug[row][c];
                if v != 0 {
                    aug[row][c] = gf.mul(v, inv);
                }
            }
            for r in 0..b {
                if r != row && aug[r][col] != 0 {
                    let f = aug[r][col];
                    for c in 0..2 * b {
                        let sub = gf.mul(f, aug[row][c]);
                        aug[r][c] ^= sub;
                    }
                }
            }
            row += 1;
        }
        let mut out = vec![0u16; b * b];
        for i in 0..b {
            out[i * b..(i + 1) * b].copy_from_slice(&aug[i][b..2 * b]);
        }
        Some(out)
    }

    /// Direct check: every basis row (f, g) satisfies f * T + g == 0
    /// (mod x^sigma), computed by naive convolution from the stored
    /// coefficients and the input series.
    fn assert_approximant(gf: &Gf, basis: &SigmaBasisAug, series_t: &[Vec<u16>], sigma: usize) {
        let b = basis.b;
        let m = 2 * b;
        for i in 0..m {
            for t in 0..sigma {
                let mut acc = vec![0u16; b];
                for (j, pj) in basis.coeffs.iter().enumerate() {
                    if j > t {
                        break;
                    }
                    let tt = t - j;
                    if tt >= series_t.len() {
                        continue;
                    }
                    // (F-part row vector of P_j row i) times matrix T_tt
                    for c1 in 0..b {
                        let fv = pj[i * m + c1];
                        if fv == 0 {
                            continue;
                        }
                        for c in 0..b {
                            let tv = series_t[tt][c1 * b + c];
                            if tv != 0 {
                                acc[c] ^= gf.mul(fv, tv);
                            }
                        }
                    }
                }
                // + g coefficient at position t
                if t < basis.coeffs.len() {
                    let pj = &basis.coeffs[t];
                    for c in 0..b {
                        acc[c] ^= pj[i * m + b + c];
                    }
                }
                assert!(
                    acc.iter().all(|&v| v == 0),
                    "row {} is not an approximant at position {}: {:?}",
                    i,
                    t,
                    acc
                );
            }
        }
    }

    /// Plant a Pade pair: random Lambda (b x b matrix polynomial, degree
    /// delta, Lambda_delta = I, Lambda_0 invertible) and random G of
    /// degree < delta; define the series T by Lambda * T == G (mod x^sigma)
    /// coefficient by coefficient. The sigma-basis of [T; I] must
    /// (a) consist of exact approximants, and
    /// (b) contain the planted pair in its row module: each row of
    ///     (Lambda | G) must be an F[x]-combination of basis rows, checked
    ///     by a rank-consistency solve. By the predictable-degree property
    ///     of weak-Popov bases, combination degrees <= delta (+ slack)
    ///     suffice.
    fn planted_case(b: usize, delta: usize, sigma: usize, seed: u64) {
        let gf = Gf::new(6);
        let mut rng = Rng::new(seed);
        let mut lam: Vec<Vec<u16>> = Vec::new();
        let lam0 = loop {
            let cand = rand_mat(&gf, &mut rng, b);
            if mat_inv(&gf, &cand, b).is_some() {
                break cand;
            }
        };
        lam.push(lam0);
        for _ in 1..delta {
            lam.push(rand_mat(&gf, &mut rng, b));
        }
        let mut top = vec![0u16; b * b];
        for i in 0..b {
            top[i * b + i] = 1;
        }
        lam.push(top);
        let mut gpoly: Vec<Vec<u16>> = Vec::new();
        for _ in 0..delta {
            gpoly.push(rand_mat(&gf, &mut rng, b));
        }
        // Solve for T: sum_{j=0}^{min(delta,k)} Lam_j T_{k-j} = G_k
        // (G_k = 0 for k >= delta).
        let lam0_inv = mat_inv(&gf, &lam[0], b).unwrap();
        let mut series_t: Vec<Vec<u16>> = Vec::new();
        for k in 0..sigma {
            let mut rhs = if k < gpoly.len() {
                gpoly[k].clone()
            } else {
                vec![0u16; b * b]
            };
            for j in 1..=delta.min(k) {
                let prod = mat_mul(&gf, &lam[j], &series_t[k - j], b);
                for idx in 0..b * b {
                    rhs[idx] ^= prod[idx];
                }
            }
            series_t.push(mat_mul(&gf, &lam0_inv, &rhs, b));
        }
        let basis = sigma_basis_aug(&gf, &series_t, sigma, b);
        // (a) exact approximants.
        assert_approximant(&gf, &basis, &series_t, sigma);
        // (b) module membership of each planted row.
        let m = 2 * b;
        let nq = delta + 3; // combination coefficients Q_t[0..=delta+2]
        let nun = m * nq;
        let npos = sigma + delta + 3;
        // Target pair as coefficient matrices over the m columns:
        // row r (0..b): (Lambda_r | G_r); rows b..m: zero.
        let target: Vec<Vec<u16>> = (0..npos)
            .map(|j| {
                let mut mat = vec![0u16; m * m];
                if j < lam.len() {
                    for r in 0..b {
                        for c in 0..b {
                            mat[r * m + c] = lam[j][r * b + c];
                        }
                    }
                }
                if j < gpoly.len() {
                    for r in 0..b {
                        for c in 0..b {
                            mat[r * m + b + c] = gpoly[j][r * b + c];
                        }
                    }
                }
                mat
            })
            .collect();
        for r in 0..b {
            let mut rows_sys: Vec<Vec<u16>> = Vec::new();
            let mut rhs_sys: Vec<u16> = Vec::new();
            for p in 0..npos {
                for c in 0..m {
                    let mut eq = vec![0u16; nun];
                    for t in 0..m {
                        for e in 0..nq {
                            let pj = p as isize - e as isize;
                            if pj < 0 || pj as usize >= basis.coeffs.len() {
                                continue;
                            }
                            eq[t * nq + e] = basis.coeffs[pj as usize][t * m + c];
                        }
                    }
                    rows_sys.push(eq);
                    rhs_sys.push(target[p][r * m + c]);
                }
            }
            let (rk, _) = crate::linalg::kernel_basis(&gf, &rows_sys);
            let mut rows_aug = rows_sys.clone();
            for (i2, row) in rows_aug.iter_mut().enumerate() {
                row.push(rhs_sys[i2]);
            }
            let (rk_aug, _) = crate::linalg::kernel_basis(&gf, &rows_aug);
            assert_eq!(
                rk, rk_aug,
                "planted pair row {} not in the basis row module (b={}, delta={}, sigma={})",
                r, b, delta, sigma
            );
        }
        // (c) diagnostic + weak existence: at least one F-pivot-style row
        // with f_deg <= delta + 1 and g_deg < f_deg.
        let good = (0..m)
            .filter(|&i| {
                !basis.f_zero[i] && basis.f_deg[i] <= delta + 1 && basis.g_deg[i] < basis.f_deg[i]
            })
            .count();
        eprintln!(
            "planted (b={}, delta={}, sigma={}): f_deg={:?} g_deg={:?} pade-good rows={}",
            b, delta, sigma, basis.f_deg, basis.g_deg, good
        );
        assert!(good >= 1, "no Pade row surfaced for planted case");
    }

    #[test]
    fn sigma_basis_planted_pade() {
        planted_case(2, 2, 16, 111);
        planted_case(2, 3, 24, 222);
        planted_case(3, 3, 28, 333);
        planted_case(3, 2, 20, 444);
        planted_case(1, 3, 16, 555);
    }

    #[test]
    fn sigma_basis_zero_series() {
        let gf = Gf::new(6);
        let b = 3usize;
        let sigma = 10usize;
        let series_t: Vec<Vec<u16>> = (0..sigma).map(|_| vec![0u16; b * b]).collect();
        let basis = sigma_basis_aug(&gf, &series_t, sigma, b);
        assert_approximant(&gf, &basis, &series_t, sigma);
        // For the zero series, rows (e_i | 0) are approximants of F-degree 0.
        let zero_f_rows = (0..2 * b)
            .filter(|&i| !basis.f_zero[i] && basis.f_deg[i] == 0)
            .count();
        eprintln!(
            "zero series: f_deg={:?} g_deg={:?} zero-f rows={}",
            basis.f_deg, basis.g_deg, zero_f_rows
        );
        assert!(zero_f_rows >= b, "expected >= b rows with F-degree 0");
    }

    #[test]
    fn sigma_basis_random_series_approximant() {
        // A generic series: the basis must still consist of exact
        // approximants (this has teeth: a wrong M-Basis update breaks it).
        let gf = Gf::new(6);
        let mut rng = Rng::new(999);
        let b = 4usize;
        let sigma = 22usize;
        let series_t: Vec<Vec<u16>> = (0..sigma).map(|_| rand_mat(&gf, &mut rng, b)).collect();
        let basis = sigma_basis_aug(&gf, &series_t, sigma, b);
        assert_approximant(&gf, &basis, &series_t, sigma);
    }
}
