#[path = "../rng.rs"] mod rng;
#[path = "../gf.rs"] mod gf;
#[path = "../rlce.rs"] mod rlce;
use gf::Gf;
use rng::Rng;
use rlce::gen_rlce;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let q: u32 = args[1].parse().unwrap();
    let n: usize = args[2].parse().unwrap();
    let k: usize = args[3].parse().unwrap();
    let w: usize = args[4].parse().unwrap();
    let seed: u64 = args[5].parse().unwrap();
    let gf = Gf::for_q(q);
    let mut r = Rng::new(seed);
    let key = gen_rlce(&gf, &mut r, n, k, w);
    // Verify public columns reconstruct from G? Just output R vectors, GL, pairs, and public cols hash check via first/last col sums
    println!("N {} k {} w {}", key.n_pub, key.k, key.w);
    for (i, rv) in key.r_vecs.iter().enumerate() {
        let gl = key.gls[i];
        print!("R {} gl {} {} {} {} :", i, gl.0, gl.1, gl.2, gl.3);
        for v in rv { print!(" {}", v); }
        println!();
    }
    for (i, pr) in key.pairs.iter().enumerate() {
        println!("PAIR {} {} {}", i, pr.0, pr.1);
    }
    // public columns, for cross-check against key.pub
    for (j, col) in key.cols.iter().enumerate() {
        print!("PUB {} :", j);
        for v in col { print!(" {}", v); }
        println!();
    }
}
