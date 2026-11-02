#[path = "../rng.rs"] mod rng;
#[path = "../gf.rs"] mod gf;
#[path = "../rlce.rs"] mod rlce;
use gf::Gf;
use rng::Rng;
use rlce::gen_rlce;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q: u32 = a[1].parse().unwrap(); let n: usize = a[2].parse().unwrap(); let k: usize = a[3].parse().unwrap(); let w: usize = a[4].parse().unwrap(); let seed: u64 = a[5].parse().unwrap(); let nprime: usize = a[6].parse().unwrap(); let shseed: u64 = a[7].parse().unwrap();
    let gf = Gf::for_q(q);
    let key = gen_rlce(&gf, &mut Rng::new(seed), n, k, w);
    let n_pub = key.n_pub;
    let hit = Rng::new(shseed).subset(n_pub, n_pub - nprime);
    let hit_set: std::collections::HashSet<usize> = hit.iter().cloned().collect();
    let kept: Vec<usize> = (0..n_pub).filter(|j| !hit_set.contains(j)).collect();
    let kept_set: std::collections::HashSet<usize> = kept.iter().cloned().collect();
    let mut pure_kept=0; let mut singleton_hit=0; let mut both_pairs=0; let mut both_cols=0;
    for &j in &kept {
        if key.is_pure[j] { pure_kept+=1; } else {
            let pid = key.pair_id[j].unwrap();
            let (a2,b2) = key.pairs[pid];
            let mate = if a2==j { b2 } else { a2 };
            if kept_set.contains(&mate) { both_cols+=1; } else { singleton_hit+=1; }
        }
    }
    both_pairs = both_cols/2;
    println!("n_pub {} kept {} pure_kept {} singleton_hit {} both_kept_pairs {} both_kept_cols {}", n_pub, kept.len(), pure_kept, singleton_hit, both_pairs, both_cols);
    // first 10 both-kept pairs in kept order, like holdout pair tests
    let mut shown=0;
    for &j in &kept {
        if shown>=10 { break; }
        if key.is_pure[j] { continue; }
        let pid = key.pair_id[j].unwrap();
        let (a2,b2) = key.pairs[pid];
        let mate = if a2==j { b2 } else { a2 };
        if kept_set.contains(&mate) { println!("both-kept tau={} mate={}", j, mate); shown+=1; }
    }
}
