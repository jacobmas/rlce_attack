//! Dense linear algebra over GF(2^m) for the end-to-end RLCE attack.
//!
//! Conventions (matching `ss.py`):
//!   - A "matrix" is `Vec<Vec<u16>>`, a list of ROWS. `a[i][j]` is row i,
//!     column j. All rows have equal length.
//!   - A "column vector" of length k (e.g. a code column) is `Vec<u16>`.
//!   - A list of N column vectors, each of length k, is `Vec<Vec<u16>>`
//!     of length N (this is how `rlce.rs` stores the public key).
//!   - When we need the k x N generator matrix as rows, we transpose.
//!
//! All arithmetic goes through `Gf::{add,mul,inv}` (characteristic 2,
//! so subtraction = addition = XOR on field elements represented as u16).

use crate::gf::Gf;

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

/// Transpose: rows <-> columns.
/// Input `a` is m rows x n cols; output is n rows x m cols.
pub fn transpose(a: &[Vec<u16>]) -> Vec<Vec<u16>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0u16; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Matrix multiplication: (m x p) * (p x n) -> (m x n).
/// In characteristic 2, "addition" of products is XOR.
pub fn matmul(gf: &Gf, a: &[Vec<u16>], b: &[Vec<u16>]) -> Vec<Vec<u16>> {
    let m = a.len();
    let p = if m > 0 { a[0].len() } else { 0 };
    let n = if !b.is_empty() { b[0].len() } else { 0 };
    assert!(b.len() == p, "matmul inner dims {} vs {}", b.len(), p);
    let mut c = vec![vec![0u16; n]; m];
    for i in 0..m {
        for k in 0..p {
            let aik = a[i][k];
            if aik == 0 {
                continue; // skip zeros (common in sparse steps)
            }
            for j in 0..n {
                // c[i][j] ^= a[i][k] * b[k][j]
                c[i][j] ^= gf.mul(aik, b[k][j]);
            }
        }
    }
    c
}

/// Rank of a matrix given as rows, via forward Gaussian elimination.
/// Does not modify the input.
pub fn rank(gf: &Gf, rows: &[Vec<u16>]) -> usize {
    let m = rows.len();
    if m == 0 {
        return 0;
    }
    let n = rows[0].len();
    if n == 0 {
        return 0;
    }
    let mut a: Vec<Vec<u16>> = rows.to_vec();
    let mut r = 0usize; // current pivot row
    for col in 0..n {
        // find a pivot row at or below r with nonzero in column col
        let mut piv: Option<usize> = None;
        for i in r..m {
            if a[i][col] != 0 {
                piv = Some(i);
                break;
            }
        }
        let p = match piv {
            Some(p) => p,
            None => continue, // no pivot in this column
        };
        a.swap(r, p);
        // normalize pivot row (optional for rank, but keeps numbers small)
        let inv = gf.inv(a[r][col]);
        if inv != 1 {
            for c in col..n {
                a[r][c] = gf.mul(a[r][c], inv);
            }
        }
        // eliminate below only (forward elimination suffices for rank)
        for i in (r + 1)..m {
            let f = a[i][col];
            if f != 0 {
                for c in col..n {
                    let sub = gf.mul(f, a[r][c]);
                    a[i][c] ^= sub;
                }
            }
        }
        r += 1;
        if r == m {
            break;
        }
    }
    r
}

/// Inverse of a square n x n matrix, or None if singular.
/// Gauss-Jordan on the augmented [A | I] matrix.
pub fn matinv(gf: &Gf, a: &[Vec<u16>]) -> Option<Vec<Vec<u16>>> {
    let n = a.len();
    if n == 0 || a[0].len() != n {
        return None;
    }
    // augmented: n rows x 2n cols
    let mut aug: Vec<Vec<u16>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = vec![0u16; 2 * n];
        row[..n].copy_from_slice(&a[i]);
        row[n + i] = 1;
        aug.push(row);
    }
    for col in 0..n {
        // pivot search
        let mut piv: Option<usize> = None;
        for r in col..n {
            if aug[r][col] != 0 {
                piv = Some(r);
                break;
            }
        }
        let p = piv?; // singular -> None
        aug.swap(col, p);
        let inv = gf.inv(aug[col][col]);
        if inv != 1 {
            for c in col..2 * n {
                aug[col][c] = gf.mul(aug[col][c], inv);
            }
        }
        // eliminate in ALL other rows (Gauss-Jordan)
        for r in 0..n {
            if r != col && aug[r][col] != 0 {
                let f = aug[r][col];
                for c in col..2 * n {
                    let sub = gf.mul(f, aug[col][c]);
                    aug[r][c] ^= sub;
                }
            }
        }
    }
    // right half is the inverse
    let mut inv = vec![vec![0u16; n]; n];
    for i in 0..n {
        inv[i].copy_from_slice(&aug[i][n..2 * n]);
    }
    Some(inv)
}

/// Solve A x = b for square invertible A (n x n), b length n.
/// Returns None if A is singular.
pub fn solve(gf: &Gf, a: &[Vec<u16>], b: &[u16]) -> Option<Vec<u16>> {
    let inv = matinv(gf, a)?;
    let n = b.len();
    let mut x = vec![0u16; n];
    for i in 0..n {
        let mut acc = 0u16;
        for j in 0..n {
            acc ^= gf.mul(inv[i][j], b[j]);
        }
        x[i] = acc;
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// GRS helpers
// ---------------------------------------------------------------------------

/// Vandermonde matrix: `dim` rows x `alpha.len()` cols.
/// Row i is alpha^i (coordinate-wise), so column j = (1, a_j, ..., a_j^{dim-1}).
/// Matches `ss.py::vandermonde`.
pub fn vandermonde(gf: &Gf, alpha: &[u16], dim: usize) -> Vec<Vec<u16>> {
    let n = alpha.len();
    let mut v = vec![vec![0u16; n]; dim];
    let mut row = vec![1u16; n]; // alpha^0
    for i in 0..dim {
        v[i] = row.clone();
        for j in 0..n {
            row[j] = gf.mul(row[j], alpha[j]);
        }
    }
    v
}

/// Dimension of the Schur (coordinate-wise) square of a set of column vectors.
/// `cols`: m vectors in F_q^k. The square is spanned by the C(m+1,2)
/// pairwise coordinate-wise products (including i==j).
/// For a GRS code with k_p = dim(span(cols)) and enough columns,
/// this equals 2*k_p - 1. A larger value means the columns are NOT on a
/// rational normal curve (e.g. a dirty unmixing candidate).
pub fn schur_square_dim(gf: &Gf, cols: &[Vec<u16>]) -> usize {
    let m = cols.len();
    if m == 0 {
        return 0;
    }
    let k = cols[0].len();
    // Build all pairwise products as rows of a C(m+1,2) x k matrix.
    // To save memory we could stream, but at tractable params this is fine.
    let mut prods: Vec<Vec<u16>> = Vec::with_capacity(m * (m + 1) / 2);
    for i in 0..m {
        for j in i..m {
            let mut p = vec![0u16; k];
            for l in 0..k {
                p[l] = gf.mul(cols[i][l], cols[j][l]);
            }
            prods.push(p);
        }
    }
    rank(gf, &prods)
}

/// Span dimension of a set of column vectors.
pub fn span_dim(gf: &Gf, cols: &[Vec<u16>]) -> usize {
    rank(gf, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn mat_identities() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(7);
        let n = 5;
        // random invertible matrix via product of elementary matrices is
        // overkill; just retry until invertible
        let a: Vec<Vec<u16>>;
        loop {
            let cand: Vec<Vec<u16>> = (0..n)
                .map(|_| (0..n).map(|_| rng.below(64) as u16).collect())
                .collect();
            if matinv(&gf, &cand).is_some() {
                a = cand;
                break;
            }
        }
        let ai = matinv(&gf, &a).unwrap();
        let prod = matmul(&gf, &a, &ai);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(prod[i][j], if i == j { 1 } else { 0 });
            }
        }
        assert_eq!(rank(&gf, &a), n);
    }

    #[test]
    fn square_of_grs_codewords() {
        // Square of CODEWORDS (rows, length n), not columns!
        // GRS_k code: k basis codewords length n; square dim = 2k-1.
        let gf = Gf::new(6);
        let k = 4;
        let n = 12;
        // k rows x n cols: row i = (x_1^i, ..., x_n^i), x_j = j
        let rows: Vec<Vec<u16>> = (0..k)
            .map(|i| {
                (1..=n as u16)
                    .map(|x| gf.pow(x, i))
                    .collect()
            })
            .collect();
        assert_eq!(span_dim(&gf, &rows), k);
        assert_eq!(schur_square_dim(&gf, &rows), 2 * k - 1);
    }
}
