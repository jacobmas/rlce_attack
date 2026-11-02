//! GIJS GRS distinguisher core (from gijs_rust v2 lib.rs).
//!
//! Given columns in F_q^k (n columns):
//! 1. Hold out column tau.
//! 2. For each other column, compute the Hasse jet of every homogeneous
//!    degree-d monomial at that column (derivative orders |a| < s).
//! 3. Build E x N_d matrix M, E=(n-1)*C(k+s-1,s-1), N_d=C(k+d-1,d).
//! 4. Check if ev_{y_tau} lies in rowspace(M) via interleaved forward
//!    elimination. Returns true = "GRS", false = "random".

use crate::gf::Gf;

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut res: usize = 1;
    for i in 0..k {
        res = res * (n - i) / (i + 1);
    }
    res
}

/// All exponent vectors e in N^k with sum(e) == sum.
pub(crate) fn enum_exp(k: usize, sum: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    if k == 0 {
        if sum == 0 {
            out.push(Vec::new());
        }
        return out;
    }
    let mut cur = vec![0u8; k];
    fn rec(k: usize, rem: usize, pos: usize, cur: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if pos == k - 1 {
            cur[pos] = rem as u8;
            out.push(cur.clone());
            return;
        }
        for e in 0..=rem {
            cur[pos] = e as u8;
            rec(k, rem - e, pos + 1, cur, out);
        }
    }
    rec(k, sum, 0, &mut cur, &mut out);
    out
}

/// Hasse derivative (d^{[a]} m)(y) for monomial m (exponents e), order a.
/// In characteristic 2: C(e_i,a_i) is odd iff (a_i & e_i) == a_i (Lucas).
#[inline(always)]
pub(crate) fn hasse(gf: &Gf, e: &[u8], a: &[u8], ypows: &[Vec<u16>]) -> u16 {
    let mut val: u16 = 1;
    for i in 0..e.len() {
        let ei = e[i] as u32;
        let ai = a[i] as u32;
        if ai > ei || (ai & ei) != ai {
            return 0;
        }
        let p = (ei - ai) as usize;
        val = gf.mul(val, ypows[i][p]);
        if val == 0 {
            return 0;
        }
    }
    val
}

/// Precompute y[i]^p for p=0..dmax.
pub(crate) fn power_table(gf: &Gf, y: &[u16], dmax: usize) -> Vec<Vec<u16>> {
    let k = y.len();
    let mut t = vec![vec![0u16; dmax + 1]; k];
    for i in 0..k {
        t[i][0] = 1;
        for p in 1..=dmax {
            t[i][p] = gf.mul(t[i][p - 1], y[i]);
        }
    }
    t
}

/// Build the E x N_d jet matrix (row-major flat). Rows: (j,a) for j != tau.
fn build_matrix(
    gf: &Gf,
    cols: &[Vec<u16>],
    tau: usize,
    monos: &[Vec<u8>],
    jets: &[Vec<u8>],
    d: usize,
) -> Vec<u16> {
    let n = cols.len();
    let nd = monos.len();
    let js = jets.len();
    let e = (n - 1) * js;
    let mut mat = vec![0u16; e * nd];
    let mut r = 0usize;
    for j in 0..n {
        if j == tau {
            continue;
        }
        let yp = power_table(gf, &cols[j], d);
        for a in jets.iter() {
            let base = r * nd;
            for (mi, m) in monos.iter().enumerate() {
                mat[base + mi] = hasse(gf, m, a, &yp);
            }
            r += 1;
        }
    }
    mat
}

/// Distinguisher. `cols[j]` is column j (length k). Returns true = "GRS".
///
/// Interleaved forward elimination: for each pivot column, eliminate from
/// the evaluation vector v using the pivot row, and eliminate below in M.
/// If v has a nonzero in a column where M has no pivot -> "random".
/// Otherwise v reduces to zero -> "GRS".
pub fn distinguish(gf: &Gf, cols: &[Vec<u16>], s: usize, d: usize, tau: usize) -> bool {
    let k = cols[0].len();
    let monos = enum_exp(k, d);
    let mut jets: Vec<Vec<u8>> = Vec::new();
    for sum in 0..s {
        jets.extend(enum_exp(k, sum));
    }
    let nd = monos.len();
    let nrows = (cols.len() - 1) * jets.len();
    let mut mat = build_matrix(gf, cols, tau, &monos, &jets, d);

    // v[c] = m_c(y_tau)
    let yp = power_table(gf, &cols[tau], d);
    let mut v = vec![0u16; nd];
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

    // Interleaved forward elimination on M, tracking v.
    let mut row = 0usize;
    for col in 0..nd {
        // Find pivot in M among rows row..nrows
        let mut piv: Option<usize> = None;
        for r in row..nrows {
            if mat[r * nd + col] != 0 {
                piv = Some(r);
                break;
            }
        }
        match piv {
            Some(p) => {
                if p != row {
                    for c in 0..nd {
                        mat.swap(row * nd + c, p * nd + c);
                    }
                }
                // Normalize pivot row
                let inv = gf.inv(mat[row * nd + col]);
                if inv != 1 {
                    for c in col..nd {
                        let idx = row * nd + c;
                        mat[idx] = gf.mul(mat[idx], inv);
                    }
                }
                // Eliminate v[col] using this pivot row.
                let vc = v[col];
                if vc != 0 {
                    let base = row * nd;
                    for c in col..nd {
                        let sub = gf.mul(vc, mat[base + c]);
                        v[c] ^= sub;
                    }
                }
                // Eliminate below in M (forward only).
                for r in (row + 1)..nrows {
                    let f = mat[r * nd + col];
                    if f != 0 {
                        let base_r = r * nd;
                        let base_p = row * nd;
                        for c in (col + 1)..nd {
                            let sub = gf.mul(f, mat[base_p + c]);
                            mat[base_r + c] ^= sub;
                        }
                        mat[base_r + col] = 0;
                    }
                }
                row += 1;
                if row == nrows {
                    break;
                }
            }
            None => {
                // No pivot in M for this column. If v[col] != 0,
                // v has a component outside rowspace(M).
                if v[col] != 0 {
                    return false; // "random"
                }
            }
        }
    }
    // v reduced to zero -> in rowspace -> "GRS".
    for &x in v.iter() {
        if x != 0 {
            return false;
        }
    }
    true
}

/// Dimensions of the jet matrix: (E, N_d).
pub fn dims(k: usize, n: usize, s: usize, d: usize) -> (usize, usize) {
    (
        (n - 1) * binom(k + s - 1, s - 1),
        binom(k + d - 1, d),
    )
}
