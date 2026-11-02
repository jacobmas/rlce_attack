//! Sidelnikov-Shestakov GRS recovery + Welch-Berlekamp decoder.
//!
//! Ported from `ss.py` (which was validated end-to-end on the CLT break).
//! Given a k x Np generator matrix `Gp` (as ROWS: k rows, each length Np)
//! of an unknown GRS_k(alpha, y) code, recovers the support `alpha`
//! and multipliers `y` (up to the inherent affine reparametrisation).
//!
//! Also provides `wb_decode_k`: Welch-Berlekamp decoding of a received
//! word with <= tau errors, used by the `verify` step to prove a full
//! encrypt/decrypt round-trip with the recovered key.
//!
//! Matrix convention: `Vec<Vec<u16>>` = list of rows.

use crate::gf::Gf;
use crate::linalg::kernel_basis;
use crate::mat::{matinv, matmul, rank, vandermonde};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Unique (up to scale) right-kernel vector of the r x (r+1) submatrix
/// B[:, cols]. Panics if the kernel is not 1-dimensional (should not
/// happen for generic GRS parity checks).
fn right_kernel_1d(gf: &Gf, b: &[Vec<u16>], cols: &[usize]) -> Vec<u16> {
    // Restrict each row of B to the chosen columns -> r rows of length r+1.
    let mrows: Vec<Vec<u16>> = b
        .iter()
        .map(|row| cols.iter().map(|&c| row[c]).collect())
        .collect();
    let (_rank, basis) = kernel_basis(gf, &mrows);
    assert_eq!(
        basis.len(),
        1,
        "right kernel dim {} != 1 for cols {:?}",
        basis.len(),
        cols
    );
    basis.into_iter().next().unwrap()
}

/// All-distinct check for a slice of field elements.
fn all_distinct(v: &[u16]) -> bool {
    let mut s = std::collections::HashSet::new();
    v.iter().all(|x| s.insert(*x))
}

// ---------------------------------------------------------------------------
// Sidelnikov-Shestakov
// ---------------------------------------------------------------------------

/// Recover (alpha, y) with Gp spanning the same code as GRS_k(alpha, y).
///
/// `gp_rows`: k rows x Np cols generator matrix of the (unknown) GRS code.
/// Returns (alpha, y), each length Np.
///
/// Algorithm (Sidelnikov-Shestakov 1992, as in ss.py):
///  1. B = right kernel of Gp (parity check, r x Np, r = Np - k).
///     For GRS, B = M * V(alpha) * diag(yhat) for unknown M, yhat.
///  2. Take overlapping (r+1)-windows J0 = 0..r, J1 = 1..r+1 of columns.
///     Their 1-dim right kernels c0, c1 encode cross-ratios
///     gamma_j = c0[j] / c1[j-1] of the support.
///  3. Guess the affine reparametrisation via rho in F_q^*: set
///     alpha_0 = 0, alpha_{r+1} = 1, alpha_j = 1/(1 - rho*gamma_j).
///     (Char 2: subtraction is XOR.)
///  4. Extend to all positions via the cross-ratio formula, recover the
///     dual multipliers yhat from Vandermonde structure, solve for the
///     scrambler M, and verify B == M * V * diag(yhat).
///  5. Convert dual multipliers to primal: y_p = 1/(yhat_p * u_p).
///  6. Verify the candidate GRS_k(alpha, y) spans the same code as Gp.
pub fn ss_attack(gf: &Gf, gp_rows: &[Vec<u16>]) -> Result<(Vec<u16>, Vec<u16>), String> {
    let k = gp_rows.len();
    if k == 0 {
        return Err("empty generator".to_string());
    }
    let np = gp_rows[0].len();
    if np <= k {
        return Err(format!("need Np > k, got Np={} k={}", np, k));
    }
    let r = np - k; // parity-check rows

    // B: r x Np parity check (right kernel of Gp).
    let (_brank, b_basis) = kernel_basis(gf, gp_rows);
    if b_basis.len() != r {
        return Err(format!(
            "parity-check rank {} != Np-k = {}",
            b_basis.len(),
            r
        ));
    }
    let b = b_basis; // r rows x Np cols

    // Two overlapping windows of r+1 columns.
    let j0: Vec<usize> = (0..r + 1).collect();
    let j1: Vec<usize> = (1..r + 2).collect();
    let c0 = right_kernel_1d(gf, &b, &j0);
    let c1 = right_kernel_1d(gf, &b, &j1);

    // gamma[j] = c0[j] / c1[j-1] for j = 1..=r.
    let mut gamma = vec![0u16; r + 1]; // 1-indexed; ignore [0]
    for j in 1..=r {
        if c1[j - 1] == 0 {
            return Err("zero divisor in gamma".to_string());
        }
        gamma[j] = gf.mul(c0[j], gf.inv(c1[j - 1]));
    }

    let q = gf.order as u16;
    // Try every rho in F_q^* (affine reparametrisation guess).
    for rho_int in 1..q {
        let rho = rho_int;
        // alpha_0 = 0, alpha_{r+1} = 1, alpha_j = 1/(1 - rho*gamma_j).
        let mut alpha = vec![0u16; np];
        alpha[r + 1] = 1;
        let mut ok = true;
        for j in 1..=r {
            // den = 1 - rho*gamma_j = 1 + rho*gamma_j (char 2)
            let den = 1 ^ gf.mul(rho, gamma[j]);
            if den == 0 {
                ok = false;
                break;
            }
            alpha[j] = gf.inv(den);
        }
        if !ok {
            continue;
        }
        if !all_distinct(&alpha[..r + 2]) {
            continue;
        }

        // Extend alpha to positions p = r+2 .. Np-1 via cross-ratios.
        let mut good = true;
        for p in (r + 2)..np {
            // Jp = {1..=r} union {p}; its kernel cp.
            let mut jp: Vec<usize> = (1..=r).collect();
            jp.push(p);
            let cp = right_kernel_1d(gf, &b, &jp);
            // ratio(jj) = c0[jj] / cp[jj-1]  (c0 over J0, cp over Jp)
            let ratio = |jj: usize| gf.mul(c0[jj], gf.inv(cp[jj - 1]));
            let r_ = gf.mul(ratio(2), gf.inv(ratio(1)));
            let a1 = alpha[1];
            let a2 = alpha[2];
            // alpha_p = a1*a2*(1-R) / (a1 - R*a2); char 2: 1-R = 1^R
            let num = gf.mul(gf.mul(a1, a2), 1 ^ r_);
            let den = a1 ^ gf.mul(r_, a2);
            if den == 0 {
                good = false;
                break;
            }
            alpha[p] = gf.mul(num, gf.inv(den));
        }
        if !good || !all_distinct(&alpha) {
            continue;
        }

        // Dual multipliers yhat on J0 from the kernel:
        // d_j = c0[j]*yhat[j] = 1 / prod_{l != j}(a_j - a_l).
        let mut yhat = vec![0u16; np];
        let mut yhat_ok = true;
        for &j in j0.iter() {
            let mut prod = 1u16;
            for &l in j0.iter() {
                if l != j {
                    prod = gf.mul(prod, alpha[j] ^ alpha[l]);
                }
            }
            if prod == 0 || c0[j] == 0 {
                yhat_ok = false;
                break;
            }
            yhat[j] = gf.mul(gf.inv(prod), gf.inv(c0[j]));
        }
        if !yhat_ok {
            continue;
        }

        // Solve for the scrambler M from r columns of J0:
        // B[:, Jc] = M * V[:, Jc] * diag(yhat[Jc])  =>  M = B[:,Jc] * inv(...)
        let jc: Vec<usize> = (0..r).collect();
        let v = vandermonde(gf, &alpha, r); // r x Np
        // V[:, Jc]: r x r
        let v_jc: Vec<Vec<u16>> = v.iter().map(|row| jc.iter().map(|&c| row[c]).collect()).collect();
        // diag(yhat[Jc]) as r x r
        let mut dmat = vec![vec![0u16; r]; r];
        for (i, &c) in jc.iter().enumerate() {
            dmat[i][i] = yhat[c];
        }
        let mv = matmul(gf, &v_jc, &dmat); // r x r
        let b_jc: Vec<Vec<u16>> = b.iter().map(|row| jc.iter().map(|&c| row[c]).collect()).collect();
        let mv_inv = match matinv(gf, &mv) {
            Some(m) => m,
            None => continue,
        };
        let m = matmul(gf, &b_jc, &mv_inv); // r x r scrambler
        let mall = matmul(gf, &m, &v); // r x Np

        // yhat[p] = B[nz, p] / Mall[nz, p] for a nonzero row nz.
        let mut mall_ok = true;
        for p in 0..np {
            let nz = (0..r).find(|&i| mall[i][p] != 0);
            match nz {
                Some(i) => {
                    yhat[p] = gf.mul(b[i][p], gf.inv(mall[i][p]));
                }
                None => {
                    mall_ok = false;
                    break;
                }
            }
        }
        if !mall_ok {
            continue;
        }
        // Verify B == M * V * diag(yhat).
        let mut yhat_diag = vec![vec![0u16; np]; np];
        for p in 0..np {
            yhat_diag[p][p] = yhat[p];
        }
        let check = matmul(gf, &mall, &yhat_diag);
        if check != b {
            continue;
        }

        // Primal multipliers: y_p = 1 / (yhat_p * u_p),
        // u_p = prod_{l != p}(a_p - a_l).
        let mut y = vec![0u16; np];
        let mut y_ok = true;
        for p in 0..np {
            let mut u = 1u16;
            for l in 0..np {
                if l != p {
                    let d = alpha[p] ^ alpha[l];
                    if d == 0 {
                        y_ok = false;
                        break;
                    }
                    u = gf.mul(u, d);
                }
            }
            if !y_ok || yhat[p] == 0 {
                y_ok = false;
                break;
            }
            y[p] = gf.inv(gf.mul(yhat[p], u));
        }
        if !y_ok {
            continue;
        }

        // Final verify: GRS_k(alpha, y) spans the same code as Gp.
        let vk = vandermonde(gf, &alpha, k); // k x Np
        let gcand: Vec<Vec<u16>> = vk
            .iter()
            .map(|row| {
                row.iter()
                    .zip(y.iter())
                    .map(|(&a, &yy)| gf.mul(a, yy))
                    .collect()
            })
            .collect();
        // stack Gp rows and Gcand rows; rank must stay k
        let mut both = gp_rows.to_vec();
        both.extend(gcand.iter().cloned());
        if rank(gf, &both) == k {
            return Ok((alpha, y));
        }
    }
    Err("SS attack failed for all rho".to_string())
}

// ---------------------------------------------------------------------------
// Welch-Berlekamp decoder for GRS_k(alpha, y)
// ---------------------------------------------------------------------------

/// Decode `recv` (length Np) with at most `tau` errors.
/// Returns the message polynomial coefficients (length k, low-to-high),
/// or None on failure.
///
/// Sets up Q (deg < k+tau) and E (deg <= tau) with
///   Q(a_p) = rho_p * E(a_p),  rho_p = recv_p / y_p,
/// solves the linear system, then divides Q/E.
pub fn wb_decode_k(
    gf: &Gf,
    alpha: &[u16],
    y: &[u16],
    recv: &[u16],
    k: usize,
    tau: usize,
) -> Option<Vec<u16>> {
    let np = alpha.len();
    assert_eq!(y.len(), np);
    assert_eq!(recv.len(), np);

    // rho_p = recv_p / y_p
    let mut rho = vec![0u16; np];
    for p in 0..np {
        if y[p] == 0 {
            return None;
        }
        rho[p] = gf.mul(recv[p], gf.inv(y[p]));
    }

    let nq = k + tau; // Q coefficients
    let ne = tau + 1; // E coefficients
    // A is Np x (nQ + nE): [Vandermonde(nQ) | diag(rho)*Vandermonde(nE)]
    let mut a_mat = vec![vec![0u16; nq + ne]; np];
    // powers[e][p] = alpha_p^e
    let maxe = nq.max(ne);
    let mut powers: Vec<Vec<u16>> = Vec::with_capacity(maxe);
    powers.push(vec![1u16; np]);
    for _ in 1..maxe {
        let prev = powers.last().unwrap().clone();
        powers.push(prev.iter().zip(alpha.iter()).map(|(&pw, &al)| gf.mul(pw, al)).collect());
    }
    for p in 0..np {
        for j in 0..nq {
            a_mat[p][j] = powers[j][p];
        }
        for j in 0..ne {
            a_mat[p][nq + j] = gf.mul(rho[p], powers[j][p]);
        }
    }
    let (_rk, basis) = kernel_basis(gf, &a_mat);
    if basis.is_empty() {
        return None;
    }
    let sol = &basis[0];
    let q_c = &sol[..nq];
    let e_c = &sol[nq..];

    // Polynomial long division Q / E (coeffs low -> high), char 2.
    fn deg(v: &[u16]) -> isize {
        for i in (0..v.len()).rev() {
            if v[i] != 0 {
                return i as isize;
            }
        }
        -1
    }
    let de = deg(e_c);
    if de < 0 {
        return None;
    }
    let mut rem = q_c.to_vec();
    let mut quo = vec![0u16; nq]; // over-allocated; trimmed below
    let lead_inv = gf.inv(e_c[de as usize]);
    loop {
        let dr = deg(&rem);
        if dr < de || dr < 0 {
            break;
        }
        let coef = gf.mul(rem[dr as usize], lead_inv);
        let shift = (dr - de) as usize;
        quo[shift] ^= coef; // char 2: accumulate (should be single write)
        for i in 0..=de as usize {
            let idx = shift + i;
            if idx < rem.len() {
                rem[idx] ^= gf.mul(coef, e_c[i]);
            }
        }
    }
    if rem.iter().any(|&x| x != 0) {
        return None; // not divisible
    }
    let mut f = vec![0u16; k];
    let qlen = quo.len().min(k);
    f[..qlen].copy_from_slice(&quo[..qlen]);
    Some(f)
}

// ---------------------------------------------------------------------------
// Monomial basis transform (for unmixing in the clean basis)
// ---------------------------------------------------------------------------

/// Rewrite the public generator in the monomial basis {1, X, ..., X^{k-1}}
/// using the recovered GRS structure on `grs_pos`.
///
/// * `gpub_rows`: k rows x N cols public generator.
/// * `grs_pos`: pure positions (length Np), in the order matching `alpha`/`y`.
/// * `alpha`, `y`: SS output on `grs_pos`.
///
/// Returns `(Ghat, Sinv)`:
///   - `Ghat` (k x N rows): public key in monomial basis.
///   - `Sinv` (k x k): the transform with Ghat = Sinv * Gpub.
///     If WB decodes a codeword to message m' (in monomial basis),
///     the original message is m = m' * inv(Sinv).
///     (Since Gpub = S*Gmono, Ghat = inv(S)*Gpub, so Sinv = inv(S).)
pub fn to_monomial_basis(
    gf: &Gf,
    gpub_rows: &[Vec<u16>],
    grs_pos: &[usize],
    alpha: &[u16],
    y: &[u16],
) -> Result<(Vec<Vec<u16>>, Vec<Vec<u16>>), String> {
    let k = gpub_rows.len();
    let n = if k > 0 { gpub_rows[0].len() } else { 0 };
    let np = grs_pos.len();
    if np < k {
        return Err(format!("need |grs_pos| >= k, got {} < {}", np, k));
    }
    // I = first k pure positions.
    let i_pos: Vec<usize> = grs_pos[..k].to_vec();
    // W = k x Np Vandermonde on alpha.
    let w = vandermonde(gf, alpha, k);
    // pos_of: public position -> index in grs_pos (= column in W).
    // Wsel = W[:, [pos_of[p] for p in I]]  (k x k).
    let mut wsel = vec![vec![0u16; k]; k];
    for (a, &p) in i_pos.iter().enumerate() {
        // find p in grs_pos
        let idx = grs_pos
            .iter()
            .position(|&q| q == p)
            .ok_or("position not in grs_pos")?;
        for r in 0..k {
            wsel[r][a] = w[r][idx];
        }
    }
    // F = Gpub[:, I] ./ y  (divide each selected column by its multiplier).
    // Gpub[:, I]: k x k (rows x selected cols).
    let mut f = vec![vec![0u16; k]; k];
    for (a, &p) in i_pos.iter().enumerate() {
        let idx = grs_pos.iter().position(|&q| q == p).unwrap();
        let yy = y[idx];
        if yy == 0 {
            return Err("zero multiplier".to_string());
        }
        let yinv = gf.inv(yy);
        for r in 0..k {
            f[r][a] = gf.mul(gpub_rows[r][p], yinv);
        }
    }
    // C = F * inv(Wsel)  (the scrambler S).
    let wsel_inv = matinv(gf, &wsel).ok_or("Wsel singular")?;
    let c = matmul(gf, &f, &wsel_inv);
    // Ghat = inv(C) * Gpub. Return both Ghat and Sinv = inv(C).
    let c_inv = matinv(gf, &c).ok_or("scrambler singular")?;
    let ghat = matmul(gf, &c_inv, gpub_rows);
    Ok((ghat, c_inv))
}

// ---------------------------------------------------------------------------
// GRS encode (for the verify round-trip)
// ---------------------------------------------------------------------------

/// Encode message polynomial `msg` (k coeffs, low-to-high) on support alpha
/// with multipliers y: codeword_p = y_p * msg(alpha_p).
pub fn grs_encode(gf: &Gf, alpha: &[u16], y: &[u16], msg: &[u16]) -> Vec<u16> {
    let np = alpha.len();
    let mut cw = vec![0u16; np];
    for p in 0..np {
        // Horner: msg(alpha_p)
        let mut acc = 0u16;
        for &c in msg.iter().rev() {
            acc = gf.mul(acc, alpha[p]) ^ c;
        }
        cw[p] = gf.mul(y[p], acc);
    }
    cw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Build a random GRS_k generator (k rows x np cols) with known (alpha,y),
    /// then check SS recovers an equivalent code.
    #[test]
    fn ss_roundtrip() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(1234);
        let k = 5;
        let np = 14;
        // distinct nonzero support
        let mut alpha = Vec::new();
        let mut used = vec![false; 64];
        while alpha.len() < np {
            let x = rng.below(64) as u16;
            if x != 0 && !used[x as usize] {
                used[x as usize] = true;
                alpha.push(x);
            }
        }
        let y: Vec<u16> = (0..np).map(|_| (rng.below(63) + 1) as u16).collect();
        // G = S * V * diag(y) for random invertible S (scrambled basis)
        let v = vandermonde(&gf, &alpha, k);
        let mut s_mat = vec![vec![0u16; k]; k];
        loop {
            for i in 0..k {
                for j in 0..k {
                    s_mat[i][j] = rng.below(64) as u16;
                }
            }
            if matinv(&gf, &s_mat).is_some() {
                break;
            }
        }
        // Gp rows = S * (V .* y)
        let vy: Vec<Vec<u16>> = v
            .iter()
            .map(|row| row.iter().zip(y.iter()).map(|(&a, &yy)| gf.mul(a, yy)).collect())
            .collect();
        let gp = matmul(&gf, &s_mat, &vy);
        let (ra, ry) = ss_attack(&gf, &gp).expect("SS failed");
        assert_eq!(ra.len(), np);
        assert_eq!(ry.len(), np);
        // verify same code: stack and check rank == k
        let vk = vandermonde(&gf, &ra, k);
        let gcand: Vec<Vec<u16>> = vk
            .iter()
            .map(|row| row.iter().zip(ry.iter()).map(|(&a, &yy)| gf.mul(a, yy)).collect())
            .collect();
        let mut both = gp.clone();
        both.extend(gcand.iter().cloned());
        assert_eq!(rank(&gf, &both), k);
    }

    #[test]
    fn wb_roundtrip() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(99);
        let k = 5;
        let np = 14;
        let tau = 2;
        let mut alpha = Vec::new();
        let mut used = vec![false; 64];
        while alpha.len() < np {
            let x = rng.below(64) as u16;
            if x != 0 && !used[x as usize] {
                used[x as usize] = true;
                alpha.push(x);
            }
        }
        let y: Vec<u16> = (0..np).map(|_| (rng.below(63) + 1) as u16).collect();
        let msg: Vec<u16> = (0..k).map(|_| rng.below(64) as u16).collect();
        let mut cw = grs_encode(&gf, &alpha, &y, &msg);
        // add tau errors at random positions
        let mut epos = Vec::new();
        while epos.len() < tau {
            let p = rng.below(np as u64) as usize;
            if !epos.contains(&p) {
                epos.push(p);
            }
        }
        for &p in &epos {
            cw[p] ^= (rng.below(63) + 1) as u16;
        }
        let dec = wb_decode_k(&gf, &alpha, &y, &cw, k, tau).expect("WB failed");
        assert_eq!(&dec[..k], &msg[..]);
    }
}
