#[path = "../rng.rs"] mod rng;
#[path = "../gf.rs"] mod gf;
use gf::Gf;
use rng::Rng;
fn rank(gf: &Gf, mut rows: Vec<Vec<u16>>) -> usize {
    let mut r = 0usize;
    let nc = if rows.is_empty() { 0 } else { rows[0].len() };
    for c in 0..nc {
        let mut piv = None;
        for i in r..rows.len() { if rows[i][c] != 0 { piv = Some(i); break; } }
        let Some(p) = piv else { continue };
        rows.swap(r, p);
        let iv = gf.inv(rows[r][c]);
        for j in c..nc { rows[r][j] = gf.mul(rows[r][j], iv); }
        for i in 0..rows.len() {
            if i != r && rows[i][c] != 0 {
                let f = rows[i][c];
                for j in c..nc { rows[i][j] = gf.add(rows[i][j], gf.mul(f, rows[r][j])); }
            }
        }
        r += 1;
        if r == rows.len() { break; }
    }
    r
}
fn main() {
    let gf = Gf::for_q(1024);
    for (seed, w, k) in [(42u64, 160usize, 470usize), (42u64, 30usize, 78usize), (43u64, 160usize, 470usize)] {
        let mut rng = Rng::new(seed);
        let mut vecs = Vec::new();
        for _ in 0..w {
            let v: Vec<u16> = (0..k).map(|_| rng.below(1024) as u16).collect();
            vecs.push(v);
            // mimic GL draws interleaved in keygen
            for _ in 0..4 { let _ = rng.below(1024); }
        }
        println!("seed {} w {} k {} R-rank {}", seed, w, k, rank(&gf, vecs));
    }
}
