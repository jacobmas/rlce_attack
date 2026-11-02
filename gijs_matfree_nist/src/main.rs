//! rlce-test: heuristic attack tester for RLCE via direct GIJS.
//!
//! Tests two heuristics (v4):
//!   H1: a dirty column behaves like a random column in the GIJS
//!       distinguisher (no twin rerandomization; direct per-hold-out run).
//!   H2: trials (random punctures of one fixed key) behave independently;
//!       co-occurrence counts fit Bin(T, p1).
//!
//! Flow:
//!   1. Generate ONE RLCE key (N=n+w columns, k rows, w twin pairs).
//!   2. For T trials (rayon-parallel): random n'-subset (puncture),
//!      run direct GIJS on each hold-out, record suspicious set.
//!      Each trial's result is spooled to disk ({out}/trial_*.txt).
//!   3. Aggregate from disk: frequency table F, co-occurrence per pair.
//!   4. Report H1 (dirty flag rate vs 1-d/q) and H2 (chi-squared vs Bin).
//!
//! Usage:
//!   rlce-test --q 64 --n 37 --k 15 --w 8 --nprime 30 --s 2 --d 3 \
//!             --trials 100 --threads 2 --seed 42 --out results/
//!   rlce-test --self-test

mod gf;
mod gijs;
mod linalg;
mod rlce;
mod rng;

use gf::Gf;
use gijs::{distinguish, dims};
use linalg::shorten;
use rlce::gen_rlce;
use rng::Rng;

use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn print_usage() {
    eprintln!("usage: rlce-test [options]");
    eprintln!("  --q Q         field size (32,64,128,256,512,1024,2048)");
    eprintln!("  --n N         GRS length");
    eprintln!("  --k K         dimension");
    eprintln!("  --w W         number of twin pairs (N = n + w public columns)");
    eprintln!("  --nprime NP   columns kept per trial (puncture)");
    eprintln!("  --s S --d D   GIJS Hasse parameters");
    eprintln!("  --trials T    number of trials");
    eprintln!("  --threads TH  rayon threads (0 = auto)");
    eprintln!("  --seed S      base RNG seed");
    eprintln!("  --out DIR     output directory (trial files + CSVs)");
    eprintln!("  --search      find valid (s,d,n') for given (q,n,k,w), then exit");
    eprintln!("  --self-test   tiny run to validate the pipeline");
}

struct Cfg {
    q: u32,
    n: usize,
    k: usize,
    w: usize,
    nprime: usize,
    s: usize,
    d: usize,
    trials: usize,
    threads: usize,
    seed: u64,
    out: PathBuf,
    search: bool,
}

fn parse_args(args: &[String]) -> Option<Cfg> {
    let mut c = Cfg {
        q: 64, n: 28, k: 20, w: 12, nprime: 32,
        s: 2, d: 4, trials: 100, threads: 0,
        seed: 42, out: PathBuf::from("results"), search: false,
    };
    let mut self_test = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--self-test" => { self_test = true; i += 1; }
            "--search" => { c.search = true; i += 1; }
            "--help" | "-h" => { print_usage(); std::process::exit(0); }
            flag => {
                if i + 1 >= args.len() { print_usage(); return None; }
                let val = &args[i + 1];
                match flag {
                    "--q" => c.q = val.parse().ok()?,
                    "--n" => c.n = val.parse().ok()?,
                    "--k" => c.k = val.parse().ok()?,
                    "--w" => c.w = val.parse().ok()?,
                    "--nprime" => c.nprime = val.parse().ok()?,
                    "--s" => c.s = val.parse().ok()?,
                    "--d" => c.d = val.parse().ok()?,
                    "--trials" => c.trials = val.parse().ok()?,
                    "--threads" => c.threads = val.parse().ok()?,
                    "--seed" => c.seed = val.parse().ok()?,
                    "--out" => c.out = PathBuf::from(val),
                    _ => { eprintln!("unknown flag {}", flag); print_usage(); return None; }
                }
                i += 2;
            }
        }
    }
    if self_test {
        c.q = 64; c.n = 20; c.k = 12; c.w = 5; c.nprime = 20;
        c.s = 2; c.d = 4; c.trials = 4; c.seed = 7;
        c.out = PathBuf::from("selftest_out");
    }
    Some(c)
}

/// One trial: random shortening (hit N-nprime positions), direct GIJS per hold-out.
/// Returns vec of (orig_idx, suspicious, pair_intact) for kept columns.
/// pair_intact: for twin members, whether the twin was also kept.
fn one_trial(
    gf: &Gf,
    key: &rlce::RlceKey,
    nprime: usize,
    s: usize,
    d: usize,
    seed: u64,
) -> Vec<(usize, bool, bool)> {
    let mut rng = Rng::new(seed);
    let n_hit = key.n_pub - nprime;
    let hit = rng.subset(key.n_pub, n_hit);
    let hit_set: std::collections::HashSet<usize> = hit.iter().cloned().collect();
    let (short_cols, kept) = shorten(gf, &key.cols, &hit);
    // short_cols: nprime columns of length k'
    let mut out = Vec::with_capacity(nprime);
    for (ti, &orig) in kept.iter().enumerate() {
        let is_grs = distinguish(gf, &short_cols, s, d, ti);
        // pair intact?
        let intact = match key.pair_id[orig] {
            Some(pid) => {
                let (a, b) = key.pairs[pid];
                let twin = if a == orig { b } else { a };
                !hit_set.contains(&twin)
            }
            None => false,
        };
        out.push((orig, !is_grs, intact)); // suspicious = not GRS
    }
    out
}

/// Binomial coefficient as f64 (for large values in search).
fn binom_f64(n: usize, k: usize) -> f64 {
    if k > n { return 0.0; }
    let k = k.min(n - k);
    let mut res = 1.0f64;
    for i in 0..k {
        res = res * (n - i) as f64 / (i + 1) as f64;
        if res > 1e300 { return f64::INFINITY; }
    }
    res
}

/// Search for valid GIJS parameters given (q,n,k,w).
/// Prints candidates with kernel margin, reconstruction margin, and cost.
fn run_search(q: u32, n: usize, k: usize, w: usize) {
    let n_pub = n + w;
    let qf = q as f64;
    println!("# search: q={} n={} k={} w={} N={}", q, n, k, w, n_pub);
    println!("# n'   k'   (s,d)  kernel_margin  recon_margin  E        N_d      cost_log2");
    let mut found = 0;
    // n' from large to small (prefer larger n' for better p_unhit, but cost grows)
    for np in (20..=n_pub.min(q as usize)).rev() {
        let kp = k as isize - (n_pub as isize - np as isize);
        if kp < 8 { continue; } // need reasonable k'
        let kp = kp as usize;
        for s in 2..=6usize {
            for d in s..=8usize {
                if d as f64 >= qf { continue; }
                let e = (np - 1) as f64 * binom_f64(kp + s - 1, s - 1);
                let nd = binom_f64(kp + d - 1, d);
                if e >= nd { continue; } // kernel trivial
                let kernel_margin = 1.0 - e / nd;
                if kernel_margin < 0.01 { continue; } // need healthy margin
                // reconstruction: s*(n'-1) > d*(k'+w-1)
                let lhs = s as f64 * (np - 1) as f64;
                let rhs = d as f64 * (kp + w - 1) as f64;
                if lhs <= rhs { continue; }
                let recon_margin = lhs / rhs - 1.0;
                // cost: E^2 * N_d (field ops per GIJS run)
                let cost_log2 = 2.0 * e.log2() + nd.log2();
                // memory: E*N_d*2 bytes (u16)
                let mem_mb = e * nd * 2.0 / 1e6;
                if mem_mb > 2000.0 { continue; } // >2GB, skip
                println!(
                    "# {:>4} {:>4}  ({},{})  {:>6.1}%       {:>6.1}%       {:.0e}  {:.0e}  {:>6.1}  ({:.0}MB)",
                    np, kp, s, d,
                    kernel_margin * 100.0, recon_margin * 100.0,
                    e, nd, cost_log2, mem_mb
                );
                found += 1;
                if found >= 20 { return; }
            }
        }
    }
    if found == 0 {
        println!("# no valid parameters found");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = match parse_args(&args) {
        Some(c) => c,
        None => std::process::exit(1),
    };

    // rayon thread pool
    if cfg.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .build_global()
            .expect("thread pool");
    }

    let gf = Gf::for_q(cfg.q);
    // --search mode: find valid params and exit
    if cfg.search {
        run_search(cfg.q, cfg.n, cfg.k, cfg.w);
        std::process::exit(0);
    }
    let (e, nd) = dims(cfg.k, cfg.nprime, cfg.s, cfg.d);
    let bytes = (e as u64) * (nd as u64) * 2;
    println!(
        "# q={} n={} k={} w={} N={} nprime={} s={} d={} trials={} seed={}",
        cfg.q, cfg.n, cfg.k, cfg.w, cfg.n + cfg.w,
        cfg.nprime, cfg.s, cfg.d, cfg.trials, cfg.seed
    );
    println!("# GIJS dims: E={} N_d={} (~{:.1} MB per run)", e, nd, bytes as f64 / 1e6);
    if e >= nd {
        eprintln!("# WARNING: E >= N_d, kernel always trivial; distinguisher always says 'random'");
    }
    if bytes > 2_000_000_000 {
        eprintln!("# ERROR: jet matrix > 2GB, refusing");
        std::process::exit(2);
    }

    // 1. generate ONE key
    let t0 = std::time::Instant::now();
    let mut rng = Rng::new(cfg.seed);
    let key = gen_rlce(&gf, &mut rng, cfg.n, cfg.k, cfg.w);
    println!(
        "# keygen: N={} ({} pure, {} in {} pairs) in {:.1}s",
        key.n_pub,
        key.is_pure.iter().filter(|&&b| b).count(),
        key.is_pure.iter().filter(|&&b| !b).count(),
        key.w,
        t0.elapsed().as_secs_f64()
    );

    // 2. trials (parallel), spool each to disk
    fs::create_dir_all(&cfg.out).expect("mkdir out");
    let t1 = std::time::Instant::now();
    let out_dir = cfg.out.clone();
    // share key via reference; Gf is not Sync (has Vec), so rebuild per thread
    // via thread-local. Simpler: Gf::for_q is cheap; construct inside closure.
    let q = cfg.q;
    let nprime = cfg.nprime;
    let (s, d) = (cfg.s, cfg.d);
    let seed = cfg.seed;
    // key.cols borrowed; need Sync. Vec<Vec<u16>> is Sync. RlceKey fields are all Sync.
    let key_ref = &key;
    // indicatif progress bar (rayon-compatible)
    use indicatif::{ParallelProgressIterator, ProgressStyle};
    let pb_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} trials ({eta})")
        .unwrap()
        .progress_chars("#>-");
    (0..cfg.trials).into_par_iter().progress_with_style(pb_style).for_each(|t| {
        let gf_t = Gf::for_q(q);
        let res = one_trial(&gf_t, key_ref, nprime, s, d, seed + 1 + t as u64);
        let path = out_dir.join(format!("trial_{:05}.txt", t));
        let mut f = fs::File::create(&path).expect("create trial file");
        for (orig, susp, intact) in res {
            writeln!(
                f, "{} {} {}",
                orig,
                if susp { 1 } else { 0 },
                if intact { 1 } else { 0 }
            )
            .unwrap();
        }
    });
    let dt_trials = t1.elapsed();
    println!(
        "# {} trials in {:.1}s ({:.2}s/trial, {} threads)",
        cfg.trials,
        dt_trials.as_secs_f64(),
        dt_trials.as_secs_f64() / cfg.trials as f64,
        rayon::current_num_threads()
    );

    // 3. aggregate from disk
    let n_pub = key.n_pub;
    let mut present = vec![0u64; n_pub];
    let mut flagged = vec![0u64; n_pub];
    // for H1: split by (pure vs intact-dirty)
    let mut present_pure = vec![0u64; n_pub];
    let mut flagged_pure = vec![0u64; n_pub];
    let mut present_dirty = vec![0u64; n_pub];
    let mut flagged_dirty = vec![0u64; n_pub];
    for t in 0..cfg.trials {
        let path = cfg.out.join(format!("trial_{:05}.txt", t));
        let txt = fs::read_to_string(&path).expect("read trial file");
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            let orig: usize = it.next().unwrap().parse().unwrap();
            let susp: u64 = it.next().unwrap().parse().unwrap();
            let intact: u64 = it.next().unwrap().parse().unwrap();
            present[orig] += 1;
            flagged[orig] += susp;
            if key.is_pure[orig] {
                present_pure[orig] += 1;
                flagged_pure[orig] += susp;
            } else if intact == 1 {
                // twin member with intact pair: reliably dirty
                present_dirty[orig] += 1;
                flagged_dirty[orig] += susp;
            }
        }
    }

    // pair co-occurrence: need per-trial suspicious sets; re-read
    // T_p = trials where both present, C_p = trials where both flagged
    let mut pair_tp = vec![0u64; key.w];
    let mut pair_cp = vec![0u64; key.w];
    for t in 0..cfg.trials {
        let path = cfg.out.join(format!("trial_{:05}.txt", t));
        let txt = fs::read_to_string(&path).expect("read trial file");
        let mut susp_map = vec![false; n_pub];
        let mut pres_map = vec![false; n_pub];
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            let orig: usize = it.next().unwrap().parse().unwrap();
            let susp: u64 = it.next().unwrap().parse().unwrap();
            pres_map[orig] = true;
            susp_map[orig] = susp == 1;
        }
        for (pi, &(a, b)) in key.pairs.iter().enumerate() {
            if pres_map[a] && pres_map[b] {
                pair_tp[pi] += 1;
                if susp_map[a] && susp_map[b] {
                    pair_cp[pi] += 1;
                }
            }
        }
    }

    // 4a. H1: dirty (intact pair) flag rate vs 1 - d/q; pure should be ~0
    let dq = cfg.d as f64 / cfg.q as f64;
    let theory = 1.0 - dq;
    let mut dirty_rates = Vec::new();
    let mut pure_rates = Vec::new();
    for j in 0..n_pub {
        if present_dirty[j] > 0 {
            dirty_rates.push(flagged_dirty[j] as f64 / present_dirty[j] as f64);
        }
        if present_pure[j] > 0 {
            pure_rates.push(flagged_pure[j] as f64 / present_pure[j] as f64);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!("# H1: dirty flag rate (theory 1-d/q = {:.4})", theory);
    println!(
        "#   dirty: n={} mean={:.4} min={:.4} max={:.4}",
        dirty_rates.len(),
        mean(&dirty_rates),
        dirty_rates.iter().cloned().fold(f64::INFINITY, f64::min),
        dirty_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    println!(
        "#   pure : n={} mean={:.4} min={:.4} max={:.4}",
        pure_rates.len(),
        mean(&pure_rates),
        pure_rates.iter().cloned().fold(f64::INFINITY, f64::min),
        pure_rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // 4b. H2: co-occurrence vs Bin(T_p, p1); chi-squared across pairs
    let tot_c: u64 = pair_cp.iter().sum();
    let tot_t: u64 = pair_tp.iter().sum();
    let p1_hat = if tot_t > 0 { tot_c as f64 / tot_t as f64 } else { 0.0 };
    let mut chi2 = 0.0;
    let mut dof = 0u64;
    for pi in 0..key.w {
        let tp = pair_tp[pi] as f64;
        let cp = pair_cp[pi] as f64;
        if tp > 0.0 && p1_hat > 0.0 && p1_hat < 1.0 {
            let exp = tp * p1_hat;
            let var = tp * p1_hat * (1.0 - p1_hat);
            if var > 0.0 {
                chi2 += (cp - exp).powi(2) / var;
                dof += 1;
            }
        }
    }
    println!("# H2: pair co-occurrence");
    println!("#   p1_hat (empirical joint flag rate) = {:.4}", p1_hat);
    println!("#   chi2 = {:.2} on {} dof (pairs)", chi2, dof);
    println!("#   (chi2/dof ~ 1 supports independence; >>1 rejects)");

    // 4c. H2': per-pair serial independence (lag-1 transition chi-square).
    // For each pair, sequence X_t = 1 if both flagged (over trials where both present).
    // Under independence, P[X_{t+1}|X_t] = P[X_{t+1}]; test via 2x2 contingency.
    // Collect per-pair sequences first.
    let mut pair_seqs: Vec<Vec<u8>> = vec![Vec::new(); key.w];
    for t in 0..cfg.trials {
        let path = cfg.out.join(format!("trial_{:05}.txt", t));
        let txt = fs::read_to_string(&path).expect("read trial file");
        let mut susp_map = vec![false; n_pub];
        let mut pres_map = vec![false; n_pub];
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            let orig: usize = it.next().unwrap().parse().unwrap();
            let susp: u64 = it.next().unwrap().parse().unwrap();
            pres_map[orig] = true;
            susp_map[orig] = susp == 1;
        }
        for (pi, &(a, b)) in key.pairs.iter().enumerate() {
            if pres_map[a] && pres_map[b] {
                pair_seqs[pi].push(if susp_map[a] && susp_map[b] { 1 } else { 0 });
            }
        }
    }
    println!("# H2': per-pair serial independence (lag-1)");
    println!("#   pair_id,n_seq,p_hat,chi2_serial");
    let mut n_tested = 0u64;
    let mut n_reject05 = 0u64; // chi2 > 3.84 (p<0.05, 1 dof)
    for (pi, seq) in pair_seqs.iter().enumerate() {
        let m = seq.len();
        if m < 10 {
            continue; // too few to test
        }
        // 2x2: rows = X_t, cols = X_{t+1}
        let mut n00 = 0u64; let mut n01 = 0u64;
        let mut n10 = 0u64; let mut n11 = 0u64;
        for w in seq.windows(2) {
            match (w[0], w[1]) {
                (0, 0) => n00 += 1,
                (0, 1) => n01 += 1,
                (1, 0) => n10 += 1,
                (1, 1) => n11 += 1,
                _ => {}
            }
        }
        let n = (n00 + n01 + n10 + n11) as f64;
        // chi-square for independence in 2x2 (1 dof), with Yates guard
        let r0 = (n00 + n01) as f64; let r1 = (n10 + n11) as f64;
        let c0 = (n00 + n10) as f64; let c1 = (n01 + n11) as f64;
        let mut c2 = 0.0;
        let mut valid = true;
        for (o, e) in [
            (n00 as f64, r0 * c0 / n),
            (n01 as f64, r0 * c1 / n),
            (n10 as f64, r1 * c0 / n),
            (n11 as f64, r1 * c1 / n),
        ] {
            if e < 1.0 { valid = false; break; }
            c2 += (o - e).powi(2) / e;
        }
        let p_hat = seq.iter().map(|&x| x as f64).sum::<f64>() / m as f64;
        n_tested += 1;
        if valid {
            if c2 > 3.841 {
                n_reject05 += 1;
            }
            println!("#     {:>3},{:>4},{:.3},{:.2}", pi, m, p_hat, c2);
        } else {
            println!("#     {:>3},{:>4},{:.3},n/a (sparse)", pi, m, p_hat);
        }
    }
    println!(
        "#   serial: {}/{} pairs reject independence at p<0.05 (expect ~5% under H2')",
        n_reject05, n_tested
    );

    // 5. CSVs
    {
        let mut f = fs::File::create(cfg.out.join("columns.csv")).unwrap();
        writeln!(f, "orig_idx,is_pure,pair_id,present,flagged,rate").unwrap();
        for j in 0..n_pub {
            let rate = if present[j] > 0 {
                flagged[j] as f64 / present[j] as f64
            } else {
                -1.0
            };
            let pid = match key.pair_id[j] {
                Some(p) => p.to_string(),
                None => String::new(),
            };
            writeln!(
                f, "{},{},{},{},{},{:.4}",
                j, if key.is_pure[j] { 1 } else { 0 }, pid,
                present[j], flagged[j], rate
            )
            .unwrap();
        }
    }
    {
        let mut f = fs::File::create(cfg.out.join("pairs.csv")).unwrap();
        writeln!(f, "pair_id,member_a,member_b,trials_both_present,coflagged").unwrap();
        for (pi, &(a, b)) in key.pairs.iter().enumerate() {
            writeln!(f, "{},{},{},{},{}", pi, a, b, pair_tp[pi], pair_cp[pi]).unwrap();
        }
    }
    println!("# wrote {}/columns.csv {}/pairs.csv", cfg.out.display(), cfg.out.display());
    println!("# DONE");
}
