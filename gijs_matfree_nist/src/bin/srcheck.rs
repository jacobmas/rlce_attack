#[path = "../rng.rs"] mod rng;
#[path = "../gf.rs"] mod gf;
#[path = "../rlce.rs"] mod rlce;
#[path = "../linalg.rs"] mod linalg;
use gf::Gf;
use rng::Rng;
use rlce::gen_rlce;
use linalg::{kernel_basis, shorten};
fn rank(gf: &Gf, rows: Vec<Vec<u16>>) -> usize {
    if rows.is_empty() { return 0; }
    let mut a = rows; let nc = a[0].len(); let mut r = 0usize;
    for c in 0..nc {
        let mut piv = None;
        for i in r..a.len() { if a[i][c] != 0 { piv = Some(i); break; } }
        let Some(p) = piv else { continue };
        a.swap(r, p);
        let iv = gf.inv(a[r][c]);
        for j in c..nc { a[r][j] = gf.mul(a[r][j], iv); }
        for i in 0..a.len() { if i != r && a[i][c] != 0 { let f = a[i][c]; for j in c..nc { a[i][j] = gf.add(a[i][j], gf.mul(f, a[r][j])); } } }
        r += 1; if r == a.len() { break; }
    }
    r
}
fn matvec(gf: &Gf, m: &[Vec<u16>], v: &[u16]) -> Vec<u16> {
    m.iter().map(|row| row.iter().zip(v.iter()).fold(0u16, |acc, (a,b)| gf.add(acc, gf.mul(*a, *b)))).collect()
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q: u32 = a[1].parse().unwrap(); let n: usize = a[2].parse().unwrap(); let k: usize = a[3].parse().unwrap(); let w: usize = a[4].parse().unwrap(); let seed: u64 = a[5].parse().unwrap(); let nprime: usize = a[6].parse().unwrap(); let shseed: u64 = a[7].parse().unwrap();
    let gf = Gf::for_q(q);
    let key = gen_rlce(&gf, &mut Rng::new(seed), n, k, w);
    let n_pub = key.n_pub;
    let hit = Rng::new(shseed).subset(n_pub, n_pub - nprime);
    let hit_rows: Vec<Vec<u16>> = hit.iter().map(|&j| key.cols[j].clone()).collect();
    let (hrank, s) = kernel_basis(&gf, &hit_rows);
    println!("hit_rank {} S_rows {} kprime {}", hrank, s.len(), s.len());
    let sr: Vec<Vec<u16>> = key.r_vecs.iter().map(|rv| matvec(&gf, &s, rv)).collect();
    println!("R_rank {} SR_rank_all {}", rank(&gf, key.r_vecs.clone()), rank(&gf, sr.clone()));
    // kept pairs
    let (_sc, kept) = shorten(&gf, &key.cols, &hit);
    let kept_set: std::collections::HashSet<usize> = kept.iter().cloned().collect();
    let kept_sr: Vec<Vec<u16>> = key.pairs.iter().enumerate().filter(|(_, pr)| kept_set.contains(&pr.0) && kept_set.contains(&pr.1)).map(|(i, _)| sr[i].clone()).collect();
    println!("both_kept_pairs {} SR_rank_kept {}", kept_sr.len(), rank(&gf, kept_sr));
    // random control
    let mut rr = Rng::new(987654321);
    let rnd: Vec<Vec<u16>> = (0..160).map(|_| (0..k).map(|_| rr.below(q as u64) as u16).collect()).collect();
    let srnd: Vec<Vec<u16>> = rnd.iter().map(|v| matvec(&gf, &s, v)).collect();
    println!("random_rank {} Srandom_rank {}", rank(&gf, rnd), rank(&gf, srnd));
}
