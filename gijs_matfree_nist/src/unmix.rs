//! Twin-pair unmixing in the MONOMIAL basis.
//!
//! After Sidelnikov-Shestakov recovers the GRS structure on the pure
//! columns and `to_monomial_basis` rewrites the public key, every pure
//! column is exactly y * (1, x, x^2, ..., x^{k-1}) for some x, y.
//!
//! A twin pair is c1 = a*g + c*r, c2 = b*g + d*r (g pure, r random).
//! For alpha in F_q^*: v_alpha = c1 + alpha*c2.
//! v_alpha is pure iff it is a geometric progression:
//!   v[0] != 0, x = v[1]/v[0], and v[i] == v[0] * x^i for all i.
//! Exactly one alpha* (if d != 0) makes this hold; we find it by brute
//! force over F_q^*. This is O(q*k) per pair -- trivial at tractable q.
//!
//! This mirrors ss.py::recover_twin_blocks but is simpler: instead of
//! searching for a fresh beta outside the known support, we directly
//! test the geometric-progression property which characterizes RNC points
//! in monomial basis.

use crate::gf::Gf;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::prelude::*;

/// Result of unmixing one pair (in monomial basis).
pub struct UnmixedPair {
    /// Public column indices of the pair (original numbering).
    pub pair: (usize, usize),
    /// The unmixing coefficient alpha* (None if not found / degenerate).
    pub alpha_star: Option<u16>,
    /// Support x of the recovered pure column (v = y*(1,x,...,x^{k-1})).
    pub x: Option<u16>,
    /// Multiplier y of the recovered pure column.
    pub y_mult: Option<u16>,
}

/// Check if `v` (length k, monomial basis) is a pure GRS column, i.e.
/// a nonzero scalar multiple of (1, x, ..., x^{k-1}) for some x.
/// Returns Some((x, y)) with v = y*(1,x,...), or None.
pub fn is_geometric(gf: &Gf, v: &[u16]) -> Option<(u16, u16)> {
    let k = v.len();
    if k < 2 {
        return None;
    }
    let y = v[0];
    if y == 0 {
        // v(0) = 0 would mean x = infinity (not in our keygen) or y = 0.
        // Our keygen uses nonzero x and y, so v[0] != 0 for pure columns.
        return None;
    }
    let yinv = gf.inv(y);
    // x = v[1] / v[0]
    let x = gf.mul(v[1], yinv);
    // check v[i] == y * x^i
    let mut xp = 1u16; // x^0
    for i in 0..k {
        if v[i] != gf.mul(y, xp) {
            return None;
        }
        xp = gf.mul(xp, x);
    }
    Some((x, y))
}

/// Unmix all twin pairs in monomial basis.
///
/// * `gf`: field.
/// * `ghat_cols`: public key columns in MONOMIAL basis (N vectors length k).
///   Obtain via `ss::to_monomial_basis`.
/// * `pairs`: twin pair (column index, column index) list from Phase 1.
///
/// Returns one `UnmixedPair` per input pair, in order.
/// Parallelized over pairs with rayon; progress via indicatif.
pub fn unmix_pairs_monomial(
    gf: &Gf,
    ghat_cols: &[Vec<u16>],
    pairs: &[(usize, usize)],
) -> Vec<UnmixedPair> {
    let q = gf.order as u16;

    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} pairs ({eta})")
        .unwrap()
        .progress_chars("#>-");

    pairs
        .par_iter()
        .progress_with_style(style)
        .map(|&(i, j)| {
            let c1 = &ghat_cols[i];
            let c2 = &ghat_cols[j];
            let k = c1.len();

            // Brute force alpha in F_q^*: v = c1 + alpha*c2.
            // Exactly one alpha* (for non-degenerate pairs) gives geometric v.
            let mut found: Option<(u16, u16, u16)> = None; // (alpha, x, y)
            for alpha_int in 1..q {
                let alpha = alpha_int;
                let mut v = vec![0u16; k];
                for l in 0..k {
                    v[l] = c1[l] ^ gf.mul(alpha, c2[l]);
                }
                if v.iter().all(|&e| e == 0) {
                    continue; // zero vector; degenerate
                }
                if let Some((x, y)) = is_geometric(gf, &v) {
                    found = Some((alpha, x, y));
                    break; // first (should be unique); break for speed
                }
            }
            match found {
                Some((a, x, y)) => UnmixedPair {
                    pair: (i, j),
                    alpha_star: Some(a),
                    x: Some(x),
                    y_mult: Some(y),
                },
                None => UnmixedPair {
                    pair: (i, j),
                    alpha_star: None,
                    x: None,
                    y_mult: None,
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Build a twin pair in monomial basis and check unmixing.
    #[test]
    fn unmix_monomial_roundtrip() {
        let gf = Gf::new(6);
        let mut rng = Rng::new(555);
        let k = 6;
        // g = v(7) in monomial basis (y=1).
        let x0 = 7u16;
        let g: Vec<u16> = {
            let mut c = vec![0u16; k];
            let mut xp = 1u16;
            for i in 0..k {
                c[i] = xp;
                xp = gf.mul(xp, x0);
            }
            c
        };
        // random r (NOT geometric).
        let r: Vec<u16> = (0..k).map(|_| rng.below(64) as u16).collect();
        // GL(2) mixing with d != 0.
        let (a, b, c, d) = (3u16, 5u16, 2u16, 9u16);
        assert_ne!(gf.mul(a, d) ^ gf.mul(b, c), 0);
        let c1: Vec<u16> = g
            .iter()
            .zip(r.iter())
            .map(|(&gi, &ri)| gf.mul(a, gi) ^ gf.mul(c, ri))
            .collect();
        let c2: Vec<u16> = g
            .iter()
            .zip(r.iter())
            .map(|(&gi, &ri)| gf.mul(b, gi) ^ gf.mul(d, ri))
            .collect();
        let cols = vec![c1, c2];
        let res = unmix_pairs_monomial(&gf, &cols, &[(0, 1)]);
        assert_eq!(res.len(), 1);
        let up = &res[0];
        let astar = up.alpha_star.expect("expected alpha*");
        // alpha* satisfies c + alpha*d = 0  (r-coefficient vanishes)
        assert_eq!(c ^ gf.mul(astar, d), 0);
        // recovered x should be x0 = 7
        assert_eq!(up.x, Some(x0));
    }

    #[test]
    fn geometric_check() {
        let gf = Gf::new(6);
        // v = 3 * (1, 5, 25, ...) in GF(64)
        let (x, y) = (5u16, 3u16);
        let v: Vec<u16> = {
            let mut c = vec![0u16; 6];
            let mut xp = 1u16;
            for i in 0..6 {
                c[i] = gf.mul(y, xp);
                xp = gf.mul(xp, x);
            }
            c
        };
        assert_eq!(is_geometric(&gf, &v), Some((x, y)));
        // corrupt one coordinate -> not geometric
        let mut w = v.clone();
        w[3] ^= 1;
        assert_eq!(is_geometric(&gf, &w), None);
    }
}
