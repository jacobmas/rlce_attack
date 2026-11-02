//! RLCE key generation with twin-pair structure.
//!
//! - GRS_{n,k} over F_q: n distinct nonzero support points x_j, nonzero
//!   multipliers y_j. Column j = y_j * (1, x_j, ..., x_j^{k-1}).
//! - Choose w positions to twin. For position i with GRS column g_i:
//!     pick random r_i in F_q^k, random GL(2) matrix [[a,b],[c,d]],
//!     replace g_i with the pair (a*g_i + c*r_i, b*g_i + d*r_i).
//! - Public key: N = n + w columns ((n-w) pure + 2w mixed), permuted.
//! - Tracks purity and twin pairing by original (pre-permutation) index.

use crate::gf::Gf;
use crate::rng::Rng;

pub struct RlceKey {
    pub n: usize,                    // GRS length
    pub k: usize,                    // dimension
    pub w: usize,                    // number of twin pairs
    pub n_pub: usize,                // N = n + w
    pub cols: Vec<Vec<u16>>,         // N columns of length k (permuted)
    pub is_pure: Vec<bool>,          // per public column
    pub pair_id: Vec<Option<usize>>, // per public column: twin pair index
    pub pairs: Vec<(usize, usize)>,  // w pairs, public column indices
    pub r_vecs: Vec<Vec<u16>>,       // per twin pair, in pair_id order (DEBUG, NIST build only)
    pub gls: Vec<(u16, u16, u16, u16)>, // per twin pair (a,b,c,d)
}

/// Random canonical GRS_{n,k}: n columns of length k.
fn gen_grs_cols(gf: &Gf, rng: &mut Rng, n: usize, k: usize) -> Vec<Vec<u16>> {
    let q = gf.order;
    assert!(
        n <= q - 1,
        "n={} needs {} distinct nonzero x_j, F_{} has {}",
        n, n, q, q - 1
    );
    let mut used = vec![false; q];
    used[0] = true;
    let mut xs: Vec<u16> = Vec::with_capacity(n);
    while xs.len() < n {
        let x = rng.below(q as u64) as usize;
        if !used[x] {
            used[x] = true;
            xs.push(x as u16);
        }
    }
    xs.into_iter()
        .map(|x| {
            let y = (rng.below((q - 1) as u64) + 1) as u16;
            let mut col = vec![0u16; k];
            let mut xp = 1u16;
            for i in 0..k {
                col[i] = gf.mul(y, xp);
                xp = gf.mul(xp, x);
            }
            col
        })
        .collect()
}

/// Generate an RLCE public key. Returns the key with twin metadata.
pub fn gen_rlce(gf: &Gf, rng: &mut Rng, n: usize, k: usize, w: usize) -> RlceKey {
    assert!(w <= n, "w={} twin pairs needs w <= n={}", w, n);
    let g_cols = gen_grs_cols(gf, rng, n, k);

    // choose w indices to twin
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..w {
        let j = i + rng.below((n - i) as u64) as usize;
        idx.swap(i, j);
    }
    let twin_set: Vec<usize> = idx[..w].to_vec();

    // build public columns: (n - w) pure + 2w mixed
    // each entry: (column, is_pure, pair_id)
    let mut entries: Vec<(Vec<u16>, bool, Option<usize>)> = Vec::with_capacity(n + w);
    let mut pair_counter = 0usize;
    let mut r_vecs: Vec<Vec<u16>> = Vec::new();
    let mut gls: Vec<(u16, u16, u16, u16)> = Vec::new();
    for (gi, g) in g_cols.into_iter().enumerate() {
        if twin_set.contains(&gi) {
            // twin this column
            let r: Vec<u16> = (0..k).map(|_| rng.below(gf.order as u64) as u16).collect();
            // random GL(2): [[a,b],[c,d]], ad - bc != 0
            let (a, b, c, d) = loop {
                let a = rng.below(gf.order as u64) as u16;
                let b = rng.below(gf.order as u64) as u16;
                let cc = rng.below(gf.order as u64) as u16;
                let dd = rng.below(gf.order as u64) as u16;
                let det = gf.add(gf.mul(a, dd), gf.mul(b, cc)); // char 2: ad+bc
                if det != 0 {
                    break (a, b, cc, dd);
                }
            };
            let mut c1 = vec![0u16; k];
            let mut c2 = vec![0u16; k];
            for i in 0..k {
                // c1 = a*g + c*r ; c2 = b*g + d*r
                c1[i] = gf.add(gf.mul(a, g[i]), gf.mul(c, r[i]));
                c2[i] = gf.add(gf.mul(b, g[i]), gf.mul(d, r[i]));
            }
            r_vecs.push(r.clone());
            gls.push((a, b, c, d));
            let pid = pair_counter;
            pair_counter += 1;
            entries.push((c1, false, Some(pid)));
            entries.push((c2, false, Some(pid)));
        } else {
            entries.push((g, true, None));
        }
    }
    assert_eq!(entries.len(), n + w);

    // random permutation
    for i in 0..entries.len() {
        let j = i + rng.below((entries.len() - i) as u64) as usize;
        entries.swap(i, j);
    }

    let n_pub = n + w;
    let mut cols = Vec::with_capacity(n_pub);
    let mut is_pure = Vec::with_capacity(n_pub);
    let mut pair_id = Vec::with_capacity(n_pub);
    for (c, p, pid) in entries {
        cols.push(c);
        is_pure.push(p);
        pair_id.push(pid);
    }
    // rebuild pair member list
    let mut pair_members: Vec<Vec<usize>> = vec![Vec::new(); w];
    for (j, pid) in pair_id.iter().enumerate() {
        if let Some(p) = pid {
            pair_members[*p].push(j);
        }
    }
    let mut pairs = Vec::with_capacity(w);
    for pm in pair_members {
        assert_eq!(pm.len(), 2, "twin pair must have exactly 2 members");
        pairs.push((pm[0], pm[1]));
    }

    RlceKey { n, k, w, n_pub, cols, is_pure, pair_id, pairs, r_vecs, gls }
}
