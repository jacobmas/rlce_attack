//! End-to-end RLCE key-recovery attack.
//!
//! This binary implements the complete attack chain from the paper:
//!
//!   Phase 1 (pair finding):
//!     Random shortenings + GIJS distinguisher -> frequency table of
//!     co-flagged column pairs -> top-w pairs are the twin pairs.
//!     Trials are multiprocessed (rayon) and spooled to disk; aggregation
//!     streams the spooled files so RAM stays bounded.
//!
//!   Phase 2a (Sidelnikov-Shestakov):
//!     From the (n-w) pure columns (never flagged in Phase 1), recover the
//!     GRS support `alpha` and multipliers `y`, and rewrite the public key
//!     in the monomial basis.
//!
//!   Phase 2b (unmixing):
//!     In monomial basis, each twin pair (c1,c2) is unmixed by trying all
//!     alpha in F_q^*: v = c1+alpha*c2. The alpha* giving a geometric
//!     progression (pure GRS column) is the unmixing coefficient.
//!
//!   Verify:
//!     - Found pairs match the true twin pairs (needs key.meta).
//!     - Recovered GRS code spans the same code as the pure columns.
//!     - Full round-trip: encrypt a random message with the public key,
//!       add errors, decode with Welch-Berlekamp on the recovered GRS code.
//!
//! Subcommands:
//!   keygen  --q Q --n N --k K --w W --seed S --out DIR
//!   phase1  --key DIR/key.pub --nprime NP --s S --d D [--trials T]
//!           --threads TH --seed S [--y-margin Y] [--fail-eps E]
//!           [--plan-only] --out DIR/phase1/
//!   unmix   --key DIR/key.pub --phase1 DIR/phase1/ --out DIR/unmixed/
//!   ss      --unmixed DIR/unmixed/ --key DIR/key.pub --out DIR/ss/
//!   verify  --key DIR/ --phase1 DIR/phase1/ --unmixed DIR/unmixed/
//!           --ss DIR/ss/
//!   all     (all of the above in one go; needs keygen params + phase1 params)
//!
//! If --trials is omitted, the Phase-1 trial count (and the filter
//! margin y, unless --y-margin is given) is AUTO-CALCULATED from the
//! attack parameters by the corrected frequency-table analysis
//! (phase1::plan_trials); the full computation chain is printed and
//! stored in phase1/config.txt. --trials overrides the count.
//!
//! The `all` command is the recommended entry point for testing on
//! tractable parameters, e.g.:
//!   rlce-attack all --q 64 --n 30 --k 14 --w 4 --nprime 28 --s 2 --d 4 \
//!       --threads 0 --seed 42 --out /tmp/atk/
//!
//! Design notes:
//!   - The ATTACK never reads key.meta (secret twin metadata). Only `verify`
//!     reads it, to score the attack. This keeps the security experiment honest.
//!   - Trial data is spooled to disk per-trial; aggregation streams files.
//!     The pair counter is C(N,2) u32s in RAM (1.2MB at N=790) or a sparse
//!     HashMap for very large N.
//!   - Key columns live in RAM (N*k*2 bytes). For multi-GB keys this is the
//!     bottleneck; trial matrices are allocated per hold-out and freed.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

mod cluster;
mod fastss;
mod gf;
mod gijs;
mod holdout;
mod keyio;
mod puncture;
mod linalg;
mod mat;
mod matfree;
mod phase1;
mod reuse;
mod rlce;
mod rng;
mod sigma;
mod ss;
mod unmix;

use gf::Gf;
use mat::rank;
use rng::Rng;

// ---------------------------------------------------------------------------
// CLI parsing (manual; avoids extra dependencies)
// ---------------------------------------------------------------------------

fn get_flag(args: &[String], name: &str) -> Option<String> {
    let mut it = args.iter().peekable();
    while let Some(a) = it.next() {
        if a == name {
            return it.next().cloned();
        }
        // also support --name=value
        if let Some(v) = a.strip_prefix(&format!("{}=", name)) {
            return Some(v.to_string());
        }
    }
    None
}
fn get_usize(args: &[String], name: &str, default: usize) -> usize {
    get_flag(args, name).map(|v| v.parse().expect("bad int")).unwrap_or(default)
}
fn get_u32(args: &[String], name: &str, default: u32) -> u32 {
    get_flag(args, name).map(|v| v.parse().expect("bad int")).unwrap_or(default)
}
fn get_u64(args: &[String], name: &str, default: u64) -> u64 {
    get_flag(args, name).map(|v| v.parse().expect("bad int")).unwrap_or(default)
}
fn get_path(args: &[String], name: &str, default: &str) -> PathBuf {
    PathBuf::from(get_flag(args, name).unwrap_or_else(|| default.to_string()))
}
fn get_f64(args: &[String], name: &str, default: f64) -> f64 {
    get_flag(args, name).map(|v| v.parse().expect("bad float")).unwrap_or(default)
}
/// True if a value-less flag (e.g. --plan-only) is present.
fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn usage() {
    eprintln!("rlce-attack: end-to-end RLCE key recovery");
    eprintln!();
    eprintln!("  rlce-attack keygen --q Q --n N --k K --w W --seed S --out DIR");
    eprintln!("  rlce-attack phase1 --key DIR/key.pub --nprime NP --s S --d D \\");
    eprintln!("      [--trials T] --threads TH --seed S [--y-margin Y] [--fail-eps E] \\");
    eprintln!("      --engine E [--phase1-mode frequency|cluster|puncture] [--plan-only] --out DIR/phase1/");
    eprintln!("  rlce-attack unmix --key DIR/key.pub --phase1 DIR/phase1/ --ss DIR/ss/ --out DIR/unmixed/");
    eprintln!("  rlce-attack ss --key DIR/key.pub --phase1 DIR/phase1/ [--phase2 fast|scan] --out DIR/ss/");
    eprintln!("  rlce-attack verify --key DIR/ --phase1 DIR/phase1/ --unmixed DIR/unmixed/ --ss DIR/ss/");
    eprintln!("  rlce-attack holdout-test --key DIR/key.pub [--meta DIR/key.meta] --nprime NP \\");
    eprintln!("      --s S --d D [--trials T] [--seed S] [--random-vecs R] [--pair-tests P] \\");
    eprintln!("      [--allow-large] [--out FILE]");
    eprintln!("  rlce-attack reuse-holdout-test --key DIR/key.pub [--meta DIR/key.meta] --nprime NP \\");
    eprintln!("      --s S --d D [--trials T] [--seed S] [--pair-tests P] \\");
    eprintln!("      [--allow-large] [--out FILE]");
    eprintln!("  rlce-attack puncture-test --key DIR/key.pub [--meta DIR/key.meta] --nprime NP \\");
    eprintln!("      --s S --d D [--shortenings SH] [--puncturings PU] [--puncture-size P] \\");
    eprintln!("      [--puncture-frac F] [--seed S] [--puncture-seed PS] [--allow-large] [--out FILE]");
    eprintln!("  rlce-attack all --q Q --n N --k K --w W --nprime NP --s S --d D \\");
    eprintln!("      [--trials T] --threads TH --seed S [--y-margin Y] [--fail-eps E] --engine E \\");
    eprintln!("      [--phase1-mode frequency|cluster|puncture] [--phase2 fast|scan] --out DIR/");
    eprintln!("  (puncture mode also takes [--puncture-size P] [--puncturings PU] [--allow-large])");
    eprintln!("--phase1-mode M: Phase-1 pair-finding method: 'frequency' (default; the");
    eprintln!("  historical frequency table, unchanged), 'cluster' (flag-pattern");
    eprintln!("  clustering: every trial contributes one pattern bit per column,");
    eprintln!("  twins share identical patterns; see cluster.rs. Cluster mode");
    eprintln!("  defaults to --engine reuse when --engine is absent, records");
    eprintln!("  phase1_mode=cluster in phase1/config.txt, and reports orphans /");
    eprintln!("  pattern collisions / all-zero excess as anomalies (degenerate");
    eprintln!("  pairs are detected, not guessed: unresolved orphans make the run");
    eprintln!("  INCOMPLETE). --y-margin/--fail-eps are ignored in cluster mode.");
    eprintln!("  'puncture' (see puncture.rs and README_PUNCTURE.md): per shortening,");
    eprintln!("  run many small puncturings (default size 4) on the shortened code;");
    eprintln!("  in the flag-none regime only a column whose mate was punctured");
    eprintln!("  flags, so the intersection of the punctured sets that flagged a");
    eprintln!("  column should shrink to its true mate. Pairs in pairs.txt are");
    eprintln!("  ONLY exact intersections (greedy disjoint, ranked by the number");
    eprintln!("  of shortenings where the pair was exact); columns that flag but");
    eprintln!("  never reach an exact intersection are listed, not guessed, in");
    eprintln!("  puncture_candidates.txt. --trials = shortenings (auto-planned),");
    eprintln!("  --puncturings = puncturings per shortening (auto-planned for a");
    eprintln!("  mean of ~6 flags per member). Puncture mode always uses the");
    eprintln!("  reuse distinguisher (one factorization per puncturing) and");
    eprintln!("  ignores --y-margin/--fail-eps. It is the mode for flag-none");
    eprintln!("  points where frequency/cluster flag nothing; it does NOT rescue");
    eprintln!("  a point whose punctured code leaves the flag-none regime (see");
    eprintln!("  README_PUNCTURE.md for the p=20/28 working-regime case).");
    eprintln!();
    eprintln!("`all` runs keygen -> phase1 -> unmix -> ss -> verify and prints a verdict.");
    eprintln!("--engine E: Phase-1 distinguisher: 'dense' (default; exact, forms M_tau),");
    eprintln!("  'bw-sigma' (matrix-free block Wiedemann + sigma-basis extraction;");
    eprintln!("  low memory, certified rejections / heuristic accepts), or 'reuse'");
    eprintln!("  (hold-out reuse: one factorization of M_full per trial, exact --");
    eprintln!("  decisions identical to 'dense' -- but O(E*(N_d+E)) memory, refused");
    eprintln!("  by a pre-flight guard above 8 GiB estimated; use bw-sigma there).");
    eprintln!("--trials T: Phase-1 raw trial count. DEFAULT: auto-calculated from");
    eprintln!("  (q, N, k, w, n', s, d) by the corrected frequency-table analysis");
    eprintln!("  (see README_Attack.md 'Auto trial planning'); the full chain is");
    eprintln!("  printed and stored in phase1/config.txt. --trials overrides it.");
    eprintln!("--fail-eps E: per-component failure target for auto-planning");
    eprintln!("  (default 0.01; total planning failure <= ~3E by union bound).");
    eprintln!("--y-margin Y: filter safety margin (default: chosen by auto-planning).");
    eprintln!("--plan-only (phase1 only): print the auto-planning chain and exit.");
    eprintln!("--phase2 P (ss/all): Phase-2 bootstrap when Phase 1 yields < k+3 pure");
    eprintln!("  columns: 'fast' (default; direction B dual-square root-finding,");
    eprintln!("  ~q^2 work instead of a q^need SS scan; falls back to the scan");
    eprintln!("  automatically) or 'scan' (the original brute-force scan).");
}

// ---------------------------------------------------------------------------
// Shared: write/read small text artifacts
// ---------------------------------------------------------------------------

fn write_pairs_txt(path: &Path, pairs: &[(u32, usize, usize)]) {
    let mut f = fs::File::create(path).expect("create pairs.txt");
    for (c, i, j) in pairs {
        writeln!(f, "{} {} {}", c, i, j).unwrap();
    }
}
fn read_pairs_txt(path: &Path) -> Vec<(u32, usize, usize)> {
    let s = fs::read_to_string(path).expect("read pairs.txt");
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<&str> = l.split_whitespace().collect();
            (v[0].parse().unwrap(), v[1].parse().unwrap(), v[2].parse().unwrap())
        })
        .collect()
}
fn write_pure_txt(path: &Path, pure_idx: &[usize]) {
    let mut f = fs::File::create(path).expect("create pure.txt");
    for i in pure_idx {
        writeln!(f, "{}", i).unwrap();
    }
}
fn read_pure_txt(path: &Path) -> Vec<usize> {
    let s = fs::read_to_string(path).expect("read pure.txt");
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

// ---------------------------------------------------------------------------
// keygen
// ---------------------------------------------------------------------------

fn cmd_keygen(args: &[String]) {
    let q = get_u32(args, "--q", 64);
    let n = get_usize(args, "--n", 30);
    let k = get_usize(args, "--k", 14);
    let w = get_usize(args, "--w", 4);
    let seed = get_u64(args, "--seed", 42);
    let out = get_path(args, "--out", "keydir");
    fs::create_dir_all(&out).expect("mkdir out");

    println!("# keygen: q={} n={} k={} w={} seed={}", q, n, k, w, seed);
    let gf = Gf::for_q(q);
    let mut rng = Rng::new(seed);
    // NOTE: gen_rlce returns secret twin metadata; we split it into
    // the public file (attacker view) and the meta file (testing only).
    let key = rlce::gen_rlce(&gf, &mut rng, n, k, w);
    println!(
        "# generated N={} ({} pure, {} in {} pairs)",
        key.n_pub,
        key.is_pure.iter().filter(|&&b| b).count(),
        key.is_pure.iter().filter(|&&b| !b).count(),
        key.w
    );
    keyio::write_pub(&out.join("key.pub"), q, n, k, w, &key.cols).expect("write key.pub");
    keyio::write_meta(
        &out.join("key.meta"),
        q, n, k, w,
        &key.is_pure, &key.pair_id, &key.pairs,
    )
    .expect("write key.meta");
    println!("# wrote {}/key.pub and {}/key.meta", out.display(), out.display());
}

// ---------------------------------------------------------------------------
// phase1
// ---------------------------------------------------------------------------

/// Run Phase 1 and write artifacts. Returns (pairs, pure_idx, n_recorded).
///
/// The auto trial-count plan (phase1::plan_trials) is ALWAYS computed and
/// printed. If `trials_override` is None, the plan's raw count T is used;
/// if Some, it is used verbatim (the plan is still printed for comparison).
/// Likewise the plan's y drives the filter unless `y_margin_override` is
/// Some. If planning fails and no --trials override was given, we exit at
/// planning time with the infeasibility message (no trials are run).
#[allow(clippy::too_many_arguments)]
fn do_phase1(
    pubkey: &keyio::PublicKey,
    nprime: usize,
    s: usize,
    d: usize,
    trials_override: Option<usize>,
    threads: usize,
    seed: u64,
    y_margin_override: Option<usize>,
    fail_eps: f64,
    engine: phase1::DistEngine,
    out_dir: &Path,
) -> (Vec<(u32, usize, usize)>, Vec<usize>, usize) {
    fs::create_dir_all(out_dir).expect("mkdir phase1 out");
    // --- auto trial-count planning -------------------------------------
    let plan = phase1::plan_trials(
        pubkey.q, pubkey.n_pub, pubkey.k, pubkey.w, nprime, s, d, fail_eps,
        y_margin_override,
    );
    let (trials, y_margin, plan_text, trials_source): (usize, usize, String, &str) =
        match &plan {
            Ok(p) => {
                let text = phase1::plan_report(p);
                for line in text.lines() {
                    println!("# {}", line);
                }
                let t = trials_override.unwrap_or(p.t_raw);
                let y = y_margin_override.unwrap_or(p.y);
                if trials_override.is_some() {
                    println!(
                        "# trials: OVERRIDE --trials={} (auto plan suggested {})",
                        t, p.t_raw
                    );
                } else {
                    println!("# trials: AUTO -> {} raw trials (override with --trials)", t);
                }
                (t, y, text, if trials_override.is_some() { "override" } else { "auto" })
            }
            Err(msg) => {
                eprintln!("# ERROR: Phase 1 auto-planning failed:");
                for line in msg.lines() {
                    eprintln!("#   {}", line);
                }
                match trials_override {
                    Some(t) => {
                        eprintln!(
                            "#   Proceeding anyway with manual --trials={} (NOT recommended).",
                            t
                        );
                        (
                            t,
                            y_margin_override.unwrap_or(0),
                            format!("planning failed: {}", msg.replace('\n', " ")),
                            "override",
                        )
                    }
                    None => {
                        eprintln!("#   No --trials override given, so there is no safe trial count.");
                        eprintln!("#   Fix: choose (n',s,d) with a larger t'_max (see rlce-test --search),");
                        eprintln!("#   smaller w, or pass --trials explicitly if you accept the risk.");
                        std::process::exit(1);
                    }
                }
            }
        };
    // ------------------------------------------------------------------
    let gf = Gf::for_q(pubkey.q);
    let cfg = phase1::Phase1Config {
        nprime, s, d, trials, threads, seed,
        y_margin,
        engine,
        out_dir: out_dir.to_path_buf(),
    };
    // 1. trials (parallel, disk-spooled)
    let (_tmax, t_reject, trial_dir) =
        phase1::run_trials(&gf, &pubkey.cols, pubkey.k, pubkey.w, &cfg);
    // 2. streaming aggregation
    println!("# aggregating {} trial files (streaming)...", trials);
    let (counts, col_flagged, n_recorded) =
        phase1::aggregate(&trial_dir, pubkey.n_pub, t_reject, trials);
    println!("# recorded {}/{} trials (|V| <= {})", n_recorded, trials, t_reject);
    if n_recorded == 0 {
        eprintln!("# ERROR: Phase 1 recorded 0/{} trials.", trials);
        eprintln!("#   t'_max={} (T_reject={}) is too small for w={} dirty pairs.", _tmax, t_reject, pubkey.w);
        eprintln!("#   The filter |V| <= {} rejected every trial: with {} twin pairs,", t_reject, pubkey.w);
        eprintln!("#   the expected number of noisy columns far exceeds the threshold.");
        eprintln!("#   Fix: use smaller w, or (n',s,d) giving larger t'_max.");
        eprintln!("#   t'_max ~= (n'-1) - d*(k'+w-1)/s; try rlce-test --search to tune (s,d).");
        std::process::exit(1);
    }
    // 3. rank pairs (greedy disjoint top-w)
    let pairs = phase1::top_disjoint_pairs(&counts, pubkey.w, pubkey.n_pub);
    println!("# top-{} disjoint pairs:", pubkey.w.min(pairs.len()));
    for (r, (c, i, j)) in pairs.iter().enumerate() {
        println!("#   rank {}: count={} pair=({},{})", r, c, i, j);
    }
    if pairs.len() < pubkey.w {
        eprintln!(
            "# WARNING: only {} disjoint pairs found (w={}); attack may fail",
            pairs.len(),
            pubkey.w
        );
    }
    // 4. pure columns: never flagged in any recorded trial
    let pure_idx: Vec<usize> = (0..pubkey.n_pub).filter(|&i| col_flagged[i] == 0).collect();
    println!(
        "# pure candidates (never flagged): {} columns",
        pure_idx.len()
    );
    // 5. write artifacts
    write_pairs_txt(&out_dir.join("pairs.txt"), &pairs);
    write_pure_txt(&out_dir.join("pure.txt"), &pure_idx);
    {
        let mut f = fs::File::create(out_dir.join("col_flagged.txt")).expect("col_flagged");
        for (i, c) in col_flagged.iter().enumerate() {
            writeln!(f, "{} {}", i, c).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("config.txt")).expect("config");
        writeln!(f, "nprime={} s={} d={} trials={} seed={} y_margin={} engine={}", nprime, s, d, trials, seed, y_margin, engine.name()).unwrap();
        writeln!(f, "t_reject={} n_recorded={}", t_reject, n_recorded).unwrap();
        writeln!(f, "trials_source={} fail_eps={}", trials_source, fail_eps).unwrap();
        writeln!(f, "# auto-planning chain (see phase1::plan_trials):").unwrap();
        for line in plan_text.lines() {
            writeln!(f, "{}", line).unwrap();
        }
    }
    (pairs, pure_idx, n_recorded)
}

/// Phase 1 mode: the frequency-table method (default, unchanged),
/// flag-pattern clustering (additive option; see cluster.rs), or
/// puncture-intersection (additive option; see puncture.rs).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase1Mode {
    Frequency,
    Cluster,
    Puncture,
}

impl Phase1Mode {
    fn name(&self) -> &'static str {
        match self {
            Phase1Mode::Frequency => "frequency",
            Phase1Mode::Cluster => "cluster",
            Phase1Mode::Puncture => "puncture",
        }
    }
}

/// Parse the --phase1-mode flag: "frequency" (default), "cluster",
/// or "puncture".
fn get_phase1_mode(args: &[String]) -> Phase1Mode {
    match get_flag(args, "--phase1-mode").as_deref() {
        None | Some("frequency") => Phase1Mode::Frequency,
        Some("cluster") => Phase1Mode::Cluster,
        Some("puncture") => Phase1Mode::Puncture,
        Some(other) => {
            eprintln!(
                "unknown --phase1-mode '{}': expected 'frequency', 'cluster', or 'puncture'",
                other
            );
            std::process::exit(1);
        }
    }
}

/// Parse the --engine flag: "dense" (default), "bw-sigma"
/// (matrix-free block Wiedemann with sigma-basis extraction), or
/// "reuse" (hold-out reuse: one factorization per trial, exact dense
/// decisions; pre-flight memory-guarded in phase1::run_trials).
fn get_engine(args: &[String]) -> phase1::DistEngine {
    match get_flag(args, "--engine").as_deref() {
        None | Some("dense") => phase1::DistEngine::Dense,
        Some("bw-sigma") => phase1::DistEngine::BwSigma,
        Some("reuse") => phase1::DistEngine::Reuse,
        Some(other) => {
            eprintln!(
                "unknown --engine '{}': expected 'dense', 'bw-sigma' or 'reuse'",
                other
            );
            std::process::exit(1);
        }
    }
}

/// Engine selection honoring the Phase 1 mode: cluster mode defaults
/// to the reuse engine (exact flagging, one factorization per trial)
/// when --engine is absent; an explicit --engine always wins, and
/// frequency mode keeps its historical default (dense). Puncture
/// mode always uses reuse: its trial runner is
/// `puncture::run_shortening`, which is reuse by construction, so
/// an explicit non-reuse --engine is ignored with a warning.
fn get_engine_for_mode(args: &[String], mode: Phase1Mode) -> phase1::DistEngine {
    if mode == Phase1Mode::Cluster && get_flag(args, "--engine").is_none() {
        return phase1::DistEngine::Reuse;
    }
    if mode == Phase1Mode::Puncture {
        if let Some(e) = get_flag(args, "--engine") {
            if e != "reuse" {
                eprintln!(
                    "# note: puncture mode always uses the reuse distinguisher; ignoring --engine {}",
                    e
                );
            }
        }
        return phase1::DistEngine::Reuse;
    }
    get_engine(args)
}

/// Unified Phase 1 output for both modes. `cluster` is Some only in
/// cluster mode (carrying orphans / oversized groups / anomaly status
/// for the `all`-flow fallback); the frequency path is untouched and
/// returns None here exactly as before.
struct Phase1Run {
    pairs: Vec<(u32, usize, usize)>,
    pure_idx: Vec<usize>,
    n_recorded: usize,
    cluster: Option<cluster::ClusterOutput>,
}

/// Dispatch Phase 1 by mode. Frequency mode calls the historical
/// `do_phase1` verbatim; cluster mode calls `do_phase1_cluster`;
/// puncture mode calls `do_phase1_puncture` (the last three
/// arguments are puncture-only and ignored by the other modes).
#[allow(clippy::too_many_arguments)]
fn run_phase1(
    mode: Phase1Mode,
    pubkey: &keyio::PublicKey,
    nprime: usize,
    s: usize,
    d: usize,
    trials_override: Option<usize>,
    threads: usize,
    seed: u64,
    y_margin_override: Option<usize>,
    fail_eps: f64,
    engine: phase1::DistEngine,
    out_dir: &Path,
    puncture_size_override: Option<usize>,
    puncturings_override: Option<usize>,
    allow_large: bool,
) -> Phase1Run {
    match mode {
        Phase1Mode::Frequency => {
            let (pairs, pure_idx, n_recorded) = do_phase1(
                pubkey,
                nprime,
                s,
                d,
                trials_override,
                threads,
                seed,
                y_margin_override,
                fail_eps,
                engine,
                out_dir,
            );
            Phase1Run {
                pairs,
                pure_idx,
                n_recorded,
                cluster: None,
            }
        }
        Phase1Mode::Cluster => {
            let (pairs, pure_idx, n_trials, co) = do_phase1_cluster(
                pubkey,
                nprime,
                s,
                d,
                trials_override,
                threads,
                seed,
                engine,
                out_dir,
            );
            Phase1Run {
                pairs,
                pure_idx,
                n_recorded: n_trials,
                cluster: Some(co),
            }
        }
        Phase1Mode::Puncture => {
            let (pairs, pure_idx, n_shortenings) = do_phase1_puncture(
                pubkey,
                nprime,
                s,
                d,
                trials_override,
                threads,
                seed,
                puncture_size_override,
                puncturings_override,
                allow_large,
                out_dir,
            );
            Phase1Run {
                pairs,
                pure_idx,
                n_recorded: n_shortenings,
                cluster: None,
            }
        }
    }
}

/// Cluster-mode Phase 1: same trial machinery as frequency mode
/// (`phase1::run_trials`, same spool files, same resume semantics),
/// but the trial count is cluster-planned and aggregation is flag-
/// pattern grouping (`cluster::aggregate_clusters`) with NO |V| filter:
/// every trial contributes one pattern bit per column.
///
/// Artifacts use the SAME formats as frequency mode (pairs.txt,
/// pure.txt, col_flagged.txt) so ss / unmix / verify work unchanged,
/// plus cluster-only files: clusters.txt (all pattern groups),
/// orphans.txt, anomaly.txt, and a `phase1_mode=cluster` line plus the
/// cluster planning chain in config.txt (frequency mode's config.txt
/// is byte-identical to before -- no mode line is added there).
#[allow(clippy::too_many_arguments)]
fn do_phase1_cluster(
    pubkey: &keyio::PublicKey,
    nprime: usize,
    s: usize,
    d: usize,
    trials_override: Option<usize>,
    threads: usize,
    seed: u64,
    engine: phase1::DistEngine,
    out_dir: &Path,
) -> (Vec<(u32, usize, usize)>, Vec<usize>, usize, cluster::ClusterOutput) {
    fs::create_dir_all(out_dir).expect("mkdir phase1 out");
    // --- cluster trial-count planning --------------------------------
    let plan = cluster::plan_cluster_trials(pubkey.n_pub, pubkey.w, nprime);
    let (trials, plan_text, trials_source): (usize, String, &str) = match &plan {
        Ok(p) => {
            let text = cluster::cluster_plan_report(p);
            for line in text.lines() {
                println!("# {}", line);
            }
            let t = trials_override.unwrap_or(p.t_planned);
            if trials_override.is_some() {
                println!(
                    "# trials: OVERRIDE --trials={} (cluster plan suggested {})",
                    t, p.t_planned
                );
            } else {
                println!("# trials: AUTO (cluster) -> {} trials (override with --trials)", t);
            }
            (
                t,
                text,
                if trials_override.is_some() {
                    "override"
                } else {
                    "auto"
                },
            )
        }
        Err(msg) => {
            eprintln!("# ERROR: cluster trial planning failed: {}", msg);
            match trials_override {
                Some(t) => {
                    eprintln!(
                        "#   Proceeding anyway with manual --trials={} (NOT recommended).",
                        t
                    );
                    (
                        t,
                        format!("planning failed: {}", msg),
                        "override",
                    )
                }
                None => {
                    eprintln!("#   No --trials override given, so there is no safe trial count.");
                    std::process::exit(1);
                }
            }
        }
    };
    // ------------------------------------------------------------------
    let gf = Gf::for_q(pubkey.q);
    let cfg = phase1::Phase1Config {
        nprime,
        s,
        d,
        trials,
        threads,
        seed,
        y_margin: 0,
        engine,
        out_dir: out_dir.to_path_buf(),
    };
    // Same trial runner as frequency mode (spool + resume identical).
    let (_tmax, _t_reject, trial_dir) =
        phase1::run_trials(&gf, &pubkey.cols, pubkey.k, pubkey.w, &cfg);
    // Cluster aggregation: ALL trials used, no |V| filter.
    println!(
        "# cluster: aggregating {} trial files (all trials used; no |V| filter)...",
        trials
    );
    let co = cluster::aggregate_clusters(
        &trial_dir,
        pubkey.n_pub,
        trials,
        pubkey.n - pubkey.w,
    );
    println!(
        "# cluster: {} pairs, {} pure columns, {} orphans, {} oversized groups ({} trials)",
        co.pairs.len(),
        co.pure_idx.len(),
        co.orphans.len(),
        co.oversized.len(),
        trials
    );
    for line in &co.anomaly_lines {
        println!("# cluster ANOMALY: {}", line);
    }
    if co.pairs.len() < pubkey.w {
        eprintln!(
            "# WARNING: cluster found {} pairs (w={}); unmatched columns live in \
             orphans / oversized groups / the pure set (invisible pairs)",
            co.pairs.len(),
            pubkey.w
        );
    }
    // Artifacts (same formats as frequency mode).
    write_pairs_txt(&out_dir.join("pairs.txt"), &co.pairs);
    write_pure_txt(&out_dir.join("pure.txt"), &co.pure_idx);
    {
        let mut f = fs::File::create(out_dir.join("col_flagged.txt")).expect("col_flagged");
        for (i, c) in co.col_flagged.iter().enumerate() {
            writeln!(f, "{} {}", i, c).unwrap();
        }
    }
    // Cluster-only audit files.
    {
        let mut f = fs::File::create(out_dir.join("clusters.txt")).expect("clusters");
        for (pop, cols) in &co.groups {
            write!(f, "{} {}", pop, cols.len()).unwrap();
            for c in cols {
                write!(f, " {}", c).unwrap();
            }
            writeln!(f).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("orphans.txt")).expect("orphans");
        for o in &co.orphans {
            writeln!(f, "{}", o).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("anomaly.txt")).expect("anomaly");
        writeln!(
            f,
            "status={}",
            if co.anomaly { "ANOMALY" } else { "OK" }
        )
        .unwrap();
        writeln!(f, "n_pairs={} n_pure={} n_orphans={} n_oversized={} n_all_zero={} expected_pure={} excess={}",
            co.pairs.len(), co.pure_idx.len(), co.orphans.len(), co.oversized.len(),
            co.n_all_zero, co.expected_pure, co.excess).unwrap();
        for line in &co.anomaly_lines {
            writeln!(f, "{}", line).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("config.txt")).expect("config");
        writeln!(
            f,
            "nprime={} s={} d={} trials={} seed={} y_margin=0 engine={}",
            nprime, s, d, trials, seed, engine.name()
        )
        .unwrap();
        writeln!(f, "phase1_mode=cluster").unwrap();
        writeln!(f, "n_trials_used={}", trials).unwrap();
        writeln!(f, "trials_source={}", trials_source).unwrap();
        writeln!(f, "# cluster planning chain (see cluster::plan_cluster_trials):").unwrap();
        for line in plan_text.lines() {
            writeln!(f, "{}", line).unwrap();
        }
    }
    (
        co.pairs.clone(),
        co.pure_idx.clone(),
        trials,
        co,
    )
}

/// Progress bar for puncture work, in the same style as
/// `phase1::run_trials` (spinner, elapsed time, bar, count, ETA).
/// One step = one reuse factorization. When output is not a
/// terminal, indicatif hides the bar, so callers also print a
/// plain "# ... done" line per shortening (see below).
fn puncture_progress_bar(total_factorizations: usize) -> indicatif::ProgressBar {
    let style = indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} factorizations ({eta})")
        .unwrap()
        .progress_chars("#>-");
    let pb = indicatif::ProgressBar::new(total_factorizations as u64);
    pb.set_style(style);
    pb
}

/// Per-factorization bar: one bar per shortening in flight, showing
/// the current factorization's elimination progress. Its message
/// is set to "shortening S puncturing Q" by `puncture::run_shortening`
/// before each factorization, and its length/position are set
/// work-weighted by the reuse elimination, so it moves roughly
/// linearly in time. It is created through a `MultiProgress` by
/// the caller (see below) so concurrent shortenings each get
/// their own line instead of fighting over one bar.
fn factorization_progress_bar() -> indicatif::ProgressBar {
    let style = indicatif::ProgressStyle::default_bar()
        .template("{msg} [{bar:30.cyan/blue}] {percent}% ({eta})")
        .unwrap()
        .progress_chars("#>-");
    let pb = indicatif::ProgressBar::new(0);
    pb.set_style(style);
    pb
}

/// Puncture-mode planning (attacker-only, no meta).
///
/// Shortenings: a twin pair is usable in a shortening only if both
/// members survive shortening, with probability
///   p_both = (n'/N) * ((n'-1)/(N-1)).
/// Choose S so the expected number of pairs never both-kept is
/// below 0.5: S = ceil(ln(0.5/w) / ln(1 - p_both)), at least 5.
/// Puncturings per shortening: a both-kept member flags in a
/// puncturing only if its mate is punctured and it is not, with
/// probability p_single = p*(n'-p) / (n'*(n'-1)). Choose P for a
/// mean of ~6 flags per member (so a member is never flagged with
/// probability e^-6 ~ 0.2%, and an exact intersection, which needs
/// at least two flags with different punctured sets, is typical),
/// at least 50. Both counts are overridable: --trials = shortenings,
/// --puncturings = puncturings per shortening, --puncture-size = p
/// (default 4, the validated kp28 size: p=2 works but needs twice
/// the puncturings for the same mean, p=20/28 leaves the flag-none
/// regime and must NOT be used here).
struct PuncturePlan {
    shortenings: usize,
    puncturings: usize,
    puncture_size: usize,
    p_both: f64,
    p_single: f64,
    mean_flags: f64,
    total_factorizations: usize,
    text: String,
}

fn puncture_plan(
    n_pub: usize,
    w: usize,
    nprime: usize,
    puncture_size_override: Option<usize>,
    puncturings_override: Option<usize>,
    trials_override: Option<usize>,
) -> PuncturePlan {
    let puncture_size = puncture_size_override.unwrap_or(4);
    let p_both = if n_pub > 1 {
        (nprime as f64 / n_pub as f64) * ((nprime - 1) as f64 / (n_pub - 1) as f64)
    } else {
        0.0
    };
    let shortenings_auto = if p_both > 0.0 && p_both < 1.0 && w > 0 {
        let s = ((0.5 / w as f64).ln() / (1.0 - p_both).ln()).ceil() as usize;
        s.max(5)
    } else if p_both >= 1.0 {
        5
    } else {
        0
    };
    let shortenings = trials_override.unwrap_or(shortenings_auto);
    let p_single = if nprime > 1 {
        (puncture_size as f64 * (nprime - puncture_size) as f64)
            / (nprime as f64 * (nprime - 1) as f64)
    } else {
        0.0
    };
    let puncturings_auto = if p_single > 0.0 {
        (6.0 / p_single).ceil() as usize
    } else {
        0
    }
    .max(50);
    let puncturings = puncturings_override.unwrap_or(puncturings_auto);
    let mean_flags = p_single * puncturings as f64;
    let text = format!(
        "puncture plan: N={} w={} n'={} p={} p_both={:.4} shortenings={} (auto {}) \
         p_single={:.4} puncturings={} (auto {}) mean_flags/member={:.2} \
         total_reuse_factorizations={}",
        n_pub,
        w,
        nprime,
        puncture_size,
        p_both,
        shortenings,
        shortenings_auto,
        p_single,
        puncturings,
        puncturings_auto,
        mean_flags,
        shortenings.saturating_mul(puncturings)
    );
    PuncturePlan {
        shortenings,
        puncturings,
        puncture_size,
        p_both,
        p_single,
        mean_flags,
        total_factorizations: shortenings.saturating_mul(puncturings),
        text,
    }
}

/// Spool file for one puncture shortening, so a long run can resume
/// after interruption: completed shortenings are reloaded, not
/// recomputed, and a shortening interrupted mid-way resumes from
/// its last checkpointed puncturing, not from scratch. Format
/// (all original indices):
///   line 1: header (parameters; must match exactly to reload)
///   line 2: kept <i1> <i2> ...
///   line 3: done <k> <total>  (puncturings completed so far)
///   then one line per kept column: <orig> <flag_count> [<inter> ...]
/// Spools written before the per-puncturing checkpointing have
/// no `done` line; those are treated as complete, since they
/// were only ever written at the end of a shortening.
/// Writes go to a `.tmp` file first and are renamed into
/// place, so a kill mid-write leaves the previous checkpoint
/// intact instead of a truncated spool.
fn puncture_spool_path(out_dir: &Path, sh: usize) -> PathBuf {
    out_dir
        .join("trials")
        .join(format!("puncture_shortening_{:05}.txt", sh))
}

fn puncture_spool_header(
    nprime: usize,
    s: usize,
    d: usize,
    puncture_size: usize,
    puncturings: usize,
    seed: u64,
    sh: usize,
) -> String {
    format!(
        "puncture nprime={} s={} d={} puncture_size={} puncturings={} seed={} shortening={}",
        nprime, s, d, puncture_size, puncturings, seed, sh
    )
}

fn write_puncture_spool(
    path: &Path,
    header: &str,
    kept: &[usize],
    flag_counts: &[usize],
    intersections: &[(usize, std::collections::HashSet<usize>)],
    done: usize,
    total: usize,
) {
    let tmp = path.with_extension("tmp");
    {
        let mut f = fs::File::create(&tmp).expect("create puncture spool");
        writeln!(f, "{}", header).unwrap();
        write!(f, "kept").unwrap();
        for k in kept {
            write!(f, " {}", k).unwrap();
        }
        writeln!(f).unwrap();
        writeln!(f, "done {} {}", done, total).unwrap();
        for (idx, (orig, inter)) in intersections.iter().enumerate() {
            write!(f, "{} {}", orig, flag_counts[idx]).unwrap();
            let mut members: Vec<usize> = inter.iter().cloned().collect();
            members.sort_unstable();
            for m in members {
                write!(f, " {}", m).unwrap();
            }
            writeln!(f).unwrap();
        }
    }
    fs::rename(&tmp, path).expect("rename puncture spool");
}

fn read_puncture_spool(
    path: &Path,
    expected_header: &str,
) -> Option<(
    Vec<usize>,
    Vec<usize>,
    Vec<(usize, std::collections::HashSet<usize>)>,
    usize,
)> {
    let s = fs::read_to_string(path).ok()?;
    let mut lines = s.lines();
    let header = lines.next()?;
    if header != expected_header {
        return None;
    }
    // Total puncturings, from the header, for old-format spools
    // that have no `done` line (those are complete by construction).
    let total_from_header: usize = header
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix("puncturings=")?.parse().ok())
        .unwrap_or(0);
    let kept_line = lines.next()?;
    let mut kept = Vec::new();
    for tok in kept_line.split_whitespace().skip(1) {
        kept.push(tok.parse().ok()?);
    }
    let rest: Vec<&str> = lines.collect();
    let (done, col_lines): (usize, &[&str]) = match rest.first() {
        Some(line) if line.starts_with("done ") => {
            let toks: Vec<&str> = line.split_whitespace().collect();
            let done: usize = toks.get(1)?.parse().ok()?;
            (done, &rest[1..])
        }
        _ => (total_from_header, &rest[..]),
    };
    let mut flag_counts = Vec::new();
    let mut intersections = Vec::new();
    for line in col_lines {
        let toks: Vec<&str> = line.split_whitespace().collect();
        if toks.len() < 2 {
            return None;
        }
        let orig: usize = toks[0].parse().ok()?;
        let fc: usize = toks[1].parse().ok()?;
        let inter: std::collections::HashSet<usize> = toks[2..]
            .iter()
            .map(|t| t.parse())
            .collect::<Result<_, _>>()
            .ok()?;
        flag_counts.push(fc);
        intersections.push((orig, inter));
    }
    if kept.len() != intersections.len() {
        return None;
    }
    Some((kept, flag_counts, intersections, done))
}

/// Puncture-mode Phase 1: same artifacts as frequency/cluster mode
/// (pairs.txt, pure.txt, col_flagged.txt, config.txt) so ss / unmix /
/// verify work unchanged, plus puncture_candidates.txt and
/// puncture_intersections.txt.
///
/// Aggregation, attacker-only, no meta anywhere:
/// - col_flagged[i] = total puncturings (over all shortenings) where
///   column i was kept' and flagged. Pure = never flagged.
/// - Per shortening, a kept column's attacker intersection (over the
///   puncturings where it flagged) is computed by
///   `puncture::run_shortening`. If it is exactly {j}, that shortening
///   votes once for the unordered pair {i,j}. Pairs are ranked by
///   votes (number of shortenings voting exact) and taken greedily
///   disjoint, up to w. A pair is in pairs.txt ONLY on exact votes.
/// - Columns that flag but never reach an exact pair are NOT guessed:
///   they go to puncture_candidates.txt with their smallest seen
///   intersection, so the user can see what remains unresolved.
/// - If fewer than w pairs are exact, a warning is printed and the
///   run continues with the pairs found (same posture as frequency
///   mode's warning); downstream stages may fail for lack of pairs,
///   and that failure is honest, not a guessed pairing.
#[allow(clippy::too_many_arguments)]
fn do_phase1_puncture(
    pubkey: &keyio::PublicKey,
    nprime: usize,
    s: usize,
    d: usize,
    trials_override: Option<usize>,
    threads: usize,
    seed: u64,
    puncture_size_override: Option<usize>,
    puncturings_override: Option<usize>,
    allow_large: bool,
    out_dir: &Path,
) -> (Vec<(u32, usize, usize)>, Vec<usize>, usize) {
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};
    fs::create_dir_all(out_dir).expect("mkdir phase1 out");
    fs::create_dir_all(out_dir.join("trials")).expect("mkdir trials");
    let plan = puncture_plan(
        pubkey.n_pub,
        pubkey.w,
        nprime,
        puncture_size_override,
        puncturings_override,
        trials_override,
    );
    println!("# {}", plan.text);
    if plan.shortenings == 0 {
        eprintln!("# ERROR: puncture planning failed: p_both = {} (no pair can survive shortening).", plan.p_both);
        std::process::exit(1);
    }
    if trials_override.is_some() {
        println!("# shortenings: OVERRIDE --trials={}", plan.shortenings);
    }
    if puncturings_override.is_some() {
        println!("# puncturings: OVERRIDE --puncturings={}", plan.puncturings);
    }
    let kp = pubkey.k as isize - (pubkey.n_pub as isize - nprime as isize);
    if kp <= 0 {
        eprintln!("# ERROR: k' = {} <= 0 for n'={} (N={}, k={})", kp, nprime, pubkey.n_pub, pubkey.k);
        std::process::exit(1);
    }
    let kp = kp as usize;
    if plan.puncture_size == 0 || plan.puncture_size >= nprime {
        eprintln!("# ERROR: puncture-size {} must satisfy 1 <= p < n'={}", plan.puncture_size, nprime);
        std::process::exit(1);
    }
    let npp = nprime - plan.puncture_size;
    if !allow_large {
        if let Err(msg) = phase1::check_reuse_memory(npp, kp, s, d) {
            eprintln!("# ERROR: punctured reuse would exceed the guard: {}", msg);
            eprintln!("#   Re-run with --allow-large if you really mean it (and have the RAM).");
            std::process::exit(1);
        }
    } else if let Err(msg) = phase1::check_reuse_memory(npp, kp, s, d) {
        eprintln!("# WARNING: --allow-large: proceeding despite guard: {}", msg);
    }
    println!(
        "# puncture phase1: N={} n'={} k'={} n''={} (s,d)=({},{}) shortenings={} puncturings/shortening={} total_factorizations={} threads={}",
        pubkey.n_pub, nprime, kp, npp, s, d, plan.shortenings, plan.puncturings, plan.total_factorizations, threads
    );
    if threads > 0 {
        // Same pattern as phase1::run_trials; only called in puncture
        // mode, which never calls run_trials in the same process.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }
    let puncture_seed = seed + 1_000_000;
    let q = pubkey.q;
    let cols = &pubkey.cols;
    // One shared bar over all factorizations (spool-reloaded
    // shortenings jump it by their full puncturing count, so the
    // bar still ends at 100% on a resumed run), plus one
    // per-factorization bar per shortening in flight, all drawn
    // through one MultiProgress so concurrent shortenings each
    // get their own line. When output is not a terminal both
    // kinds of bar hide and only the per-shortening lines below
    // appear.
    let mp = std::sync::Arc::new(indicatif::MultiProgress::new());
    let pb = mp.add(puncture_progress_bar(plan.total_factorizations));
    let results: Vec<(
        usize,
        Vec<usize>,
        Vec<usize>,
        Vec<(usize, HashSet<usize>)>,
    )> = (0..plan.shortenings)
        .into_par_iter()
        .map(|sh| {
            let pb = pb.clone();
            let factor = mp.add(factorization_progress_bar());
            let header = puncture_spool_header(
                nprime, s, d, plan.puncture_size, plan.puncturings, seed, sh,
            );
            let spool = puncture_spool_path(out_dir, sh);
            // Checkpoint after every puncturing, so a kill loses
            // at most the one factorization in flight. The
            // closure below is called by run_shortening after
            // each puncturing with the running totals.
            let checkpoint = |done: usize,
                              kept: &[usize],
                              flag_counts: &[usize],
                              intersections: &[(usize, HashSet<usize>)]| {
                write_puncture_spool(
                    &spool,
                    &header,
                    kept,
                    flag_counts,
                    intersections,
                    done,
                    plan.puncturings,
                );
            };
            if let Some((kept, flag_counts, intersections, done)) =
                read_puncture_spool(&spool, &header)
            {
                if done >= plan.puncturings {
                    pb.inc(plan.puncturings as u64);
                    println!("# puncture: shortening {} done (reloaded from spool)", sh);
                    return (sh, kept, flag_counts, intersections);
                }
                // Partial checkpoint: resume this shortening
                // from puncturing `done`, counting the already
                // completed puncturings toward the overall bar
                // so it still ends at 100%.
                pb.inc(done as u64);
                println!(
                    "# puncture: shortening {} resuming from puncturing {}/{} (spool)",
                    sh, done, plan.puncturings
                );
                let resume = puncture::PunctureResume {
                    done,
                    flag_counts,
                    intersections,
                };
                let gf_t = Gf::for_q(q);
                let shortening_seed = seed + 1 + sh as u64;
                let rep = puncture::run_shortening(
                    &gf_t,
                    cols,
                    None,
                    nprime,
                    s,
                    d,
                    sh,
                    shortening_seed,
                    plan.puncturings,
                    plan.puncture_size,
                    puncture_seed,
                    Some(&pb),
                    Some(&factor),
                    Some(&resume),
                    Some(&checkpoint),
                );
                factor.finish_and_clear();
                println!("# puncture: shortening {} done", sh);
                let flag_counts = rep.flag_counts;
                write_puncture_spool(
                    &spool,
                    &header,
                    &rep.kept,
                    &flag_counts,
                    &rep.intersections,
                    plan.puncturings,
                    plan.puncturings,
                );
                return (sh, rep.kept, flag_counts, rep.intersections);
            }
            let gf_t = Gf::for_q(q);
            let shortening_seed = seed + 1 + sh as u64;
            let rep = puncture::run_shortening(
                &gf_t,
                cols,
                None,
                nprime,
                s,
                d,
                sh,
                shortening_seed,
                plan.puncturings,
                plan.puncture_size,
                puncture_seed,
                Some(&pb),
                Some(&factor),
                None,
                Some(&checkpoint),
            );
            factor.finish_and_clear();
            println!("# puncture: shortening {} done", sh);
            let flag_counts = rep.flag_counts;
            write_puncture_spool(
                &spool,
                &header,
                &rep.kept,
                &flag_counts,
                &rep.intersections,
                plan.puncturings,
                plan.puncturings,
            );
            (sh, rep.kept, flag_counts, rep.intersections)
        })
        .collect();
    pb.finish_and_clear();
    // Aggregation (sequential; results are small).
    let mut col_flagged = vec![0usize; pubkey.n_pub];
    let mut exact_votes: HashMap<(usize, usize), usize> = HashMap::new();
    let mut shortenings_flagged: HashMap<usize, usize> = HashMap::new();
    let mut total_flags: HashMap<usize, usize> = HashMap::new();
    let mut smallest_inter: HashMap<usize, HashSet<usize>> = HashMap::new();
    for (sh, kept, flag_counts, intersections) in &results {
        let _ = sh;
        let _ = kept;
        let mut exact_here: HashSet<(usize, usize)> = HashSet::new();
        for (idx, (orig, inter)) in intersections.iter().enumerate() {
            let fc = flag_counts[idx];
            if fc > 0 {
                col_flagged[*orig] += fc;
                *shortenings_flagged.entry(*orig).or_insert(0) += 1;
                *total_flags.entry(*orig).or_insert(0) += fc;
                let cur = smallest_inter.get(orig);
                if cur.map(|c| inter.len() < c.len()).unwrap_or(true) {
                    smallest_inter.insert(*orig, inter.clone());
                }
                if inter.len() == 1 {
                    let j = *inter.iter().next().unwrap();
                    let a = *orig;
                    let pair = (a.min(j), a.max(j));
                    exact_here.insert(pair);
                }
            }
        }
        for pair in exact_here {
            *exact_votes.entry(pair).or_insert(0) += 1;
        }
    }
    // Greedy disjoint pairs by votes, then by indices (deterministic).
    let mut ranked: Vec<((usize, usize), usize)> =
        exact_votes.iter().map(|(p, v)| (*p, *v)).collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut used: HashSet<usize> = HashSet::new();
    let mut pairs: Vec<(u32, usize, usize)> = Vec::new();
    for ((i, j), votes) in ranked {
        if pairs.len() >= pubkey.w {
            break;
        }
        if used.contains(&i) || used.contains(&j) {
            continue;
        }
        used.insert(i);
        used.insert(j);
        pairs.push((votes as u32, i, j));
    }
    println!("# puncture: {} exact pairs (w={})", pairs.len(), pubkey.w);
    for (r, (c, i, j)) in pairs.iter().enumerate() {
        println!("#   rank {}: exact_shortenings={} pair=({},{})", r, c, i, j);
    }
    if pairs.len() < pubkey.w {
        eprintln!(
            "# WARNING: puncture found only {} exact pairs (w={}); unmatched flagged columns live in puncture_candidates.txt (not guessed)",
            pairs.len(),
            pubkey.w
        );
    }
    let pure_idx: Vec<usize> = (0..pubkey.n_pub)
        .filter(|&i| col_flagged[i] == 0)
        .collect();
    println!("# pure candidates (never flagged): {} columns", pure_idx.len());
    // Artifacts (same formats as frequency/cluster mode).
    write_pairs_txt(&out_dir.join("pairs.txt"), &pairs);
    write_pure_txt(&out_dir.join("pure.txt"), &pure_idx);
    {
        let mut f = fs::File::create(out_dir.join("col_flagged.txt")).expect("col_flagged");
        for (i, c) in col_flagged.iter().enumerate() {
            writeln!(f, "{} {}", i, c).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("puncture_candidates.txt"))
            .expect("puncture_candidates");
        let mut cols: Vec<usize> = smallest_inter.keys().cloned().collect();
        cols.sort_unstable();
        for orig in cols {
            let inter = &smallest_inter[&orig];
            let mut members: Vec<usize> = inter.iter().cloned().collect();
            members.sort_unstable();
            let in_pair = pairs.iter().any(|&(_, i, j)| i == orig || j == orig);
            writeln!(
                f,
                "{} shortenings_flagged={} total_flags={} smallest_inter_size={} smallest_inter={:?} in_exact_pair={}",
                orig,
                shortenings_flagged.get(&orig).cloned().unwrap_or(0),
                total_flags.get(&orig).cloned().unwrap_or(0),
                members.len(),
                members,
                in_pair
            )
            .unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("puncture_intersections.txt"))
            .expect("puncture_intersections");
        for (sh, _kept, flag_counts, intersections) in &results {
            for (idx, (orig, inter)) in intersections.iter().enumerate() {
                if flag_counts[idx] == 0 {
                    continue;
                }
                let mut members: Vec<usize> = inter.iter().cloned().collect();
                members.sort_unstable();
                writeln!(f, "shortening {} col {} flag_count {} inter {:?}", sh, orig, flag_counts[idx], members).unwrap();
            }
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("config.txt")).expect("config");
        writeln!(
            f,
            "nprime={} s={} d={} shortenings={} puncturings={} puncture_size={} seed={} puncture_seed={} engine=reuse",
            nprime, s, d, plan.shortenings, plan.puncturings, plan.puncture_size, seed, puncture_seed
        )
        .unwrap();
        writeln!(f, "phase1_mode=puncture").unwrap();
        writeln!(f, "total_factorizations={} n_pairs={} n_pure={}", plan.total_factorizations, pairs.len(), pure_idx.len()).unwrap();
        writeln!(f, "{}", plan.text).unwrap();
    }
    (pairs, pure_idx, plan.shortenings)
}

/// Which Phase 2 bootstrap to use when SS needs unmixed pairs
/// (need = (k+3) - n_pure > 0; the w = n-k case has need = 3).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase2Mode {
    /// Direction B (src/fastss.rs): q^2 root-finding via the dual
    /// square minor + exact dual-rank confirm + SS; falls back to the
    /// scan automatically if it finds no verified candidate.
    Fast,
    /// The original brute-force q^need scan over lambda tuples.
    Scan,
}

impl Phase2Mode {
    fn name(&self) -> &'static str {
        match self {
            Phase2Mode::Fast => "fast",
            Phase2Mode::Scan => "scan",
        }
    }
}

/// Parse the --phase2 flag: "fast" (default) or "scan".
fn get_phase2(args: &[String]) -> Phase2Mode {
    match get_flag(args, "--phase2").as_deref() {
        None | Some("fast") => Phase2Mode::Fast,
        Some("scan") => Phase2Mode::Scan,
        Some(other) => {
            eprintln!("unknown --phase2 '{}': expected 'fast' or 'scan'", other);
            std::process::exit(1);
        }
    }
}

fn cmd_phase1(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let nprime = get_usize(args, "--nprime", 28);
    let s = get_usize(args, "--s", 2);
    let d = get_usize(args, "--d", 4);
    // --trials / --y-margin: None means "not given" (auto-planning).
    let trials: Option<usize> = get_flag(args, "--trials").map(|v| v.parse().expect("bad int"));
    let threads = get_usize(args, "--threads", 0);
    let seed = get_u64(args, "--seed", 100);
    let y_margin: Option<usize> = get_flag(args, "--y-margin").map(|v| v.parse().expect("bad int"));
    let fail_eps = get_f64(args, "--fail-eps", 0.01);
    let mode = get_phase1_mode(args);
    let engine = get_engine_for_mode(args, mode);
    let out = get_path(args, "--out", "phase1");
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    println!("# phase1: key N={} k={} w={} engine={} mode={}", pubkey.n_pub, pubkey.k, pubkey.w, engine.name(), mode.name());
    if mode == Phase1Mode::Cluster
        && (get_flag(args, "--y-margin").is_some() || get_flag(args, "--fail-eps").is_some())
    {
        println!("# note: cluster mode ignores --y-margin/--fail-eps (no filter, no separation)");
    }
    if mode == Phase1Mode::Puncture
        && (get_flag(args, "--y-margin").is_some() || get_flag(args, "--fail-eps").is_some())
    {
        println!("# note: puncture mode ignores --y-margin/--fail-eps (no filter, no separation)");
    }
    let puncture_size: Option<usize> = get_flag(args, "--puncture-size").map(|v| v.parse().expect("bad int"));
    let puncturings: Option<usize> = get_flag(args, "--puncturings").map(|v| v.parse().expect("bad int"));
    let allow_large = has_flag(args, "--allow-large");
    if has_flag(args, "--plan-only") {
        match mode {
            Phase1Mode::Frequency => match phase1::plan_trials(
                pubkey.q, pubkey.n_pub, pubkey.k, pubkey.w, nprime, s, d, fail_eps, y_margin,
            ) {
                Ok(p) => {
                    for line in phase1::plan_report(&p).lines() {
                        println!("# {}", line);
                    }
                }
                Err(msg) => {
                    eprintln!("# ERROR: Phase 1 auto-planning failed:");
                    for line in msg.lines() {
                        eprintln!("#   {}", line);
                    }
                    std::process::exit(1);
                }
            },
            Phase1Mode::Cluster => match cluster::plan_cluster_trials(pubkey.n_pub, pubkey.w, nprime)
            {
                Ok(p) => {
                    for line in cluster::cluster_plan_report(&p).lines() {
                        println!("# {}", line);
                    }
                }
                Err(msg) => {
                    eprintln!("# ERROR: cluster trial planning failed: {}", msg);
                    std::process::exit(1);
                }
            },
            Phase1Mode::Puncture => {
                let plan = puncture_plan(
                    pubkey.n_pub,
                    pubkey.w,
                    nprime,
                    puncture_size,
                    puncturings,
                    trials,
                );
                println!("# {}", plan.text);
            }
        }
        return;
    }
    run_phase1(
        mode,
        &pubkey,
        nprime,
        s,
        d,
        trials,
        threads,
        seed,
        y_margin,
        fail_eps,
        engine,
        &out,
        puncture_size,
        puncturings,
        allow_large,
    );
    println!("# phase1 done -> {}", out.display());
}

// ---------------------------------------------------------------------------
// unmix (in monomial basis; needs SS output first)
// ---------------------------------------------------------------------------

/// Unmix pairs in monomial basis; returns the UnmixedPair list.
/// Writes report.txt and unmixed_xy.txt (pair, alpha*, x, y per line).
fn do_unmix(
    pubkey: &keyio::PublicKey,
    pairs: &[(u32, usize, usize)],
    ss_dir: &Path,
    out_dir: &Path,
) -> Vec<unmix::UnmixedPair> {
    fs::create_dir_all(out_dir).expect("mkdir unmix out");
    let gf = Gf::for_q(pubkey.q);
    // Load SS output and transform the public key to monomial basis.
    let (alpha, y, positions) = read_ss(ss_dir);
    println!("# unmix: transforming public key to monomial basis...");
    let gpub_rows = mat::transpose(&pubkey.cols); // k x N
    let (ghat_rows, _sinv) = ss::to_monomial_basis(&gf, &gpub_rows, &positions, &alpha, &y)
        .expect("monomial basis transform");
    let ghat_cols = mat::transpose(&ghat_rows); // N columns, monomial basis
    println!("# unmix: {} pairs in monomial basis", pairs.len());
    let pair_list: Vec<(usize, usize)> = pairs.iter().map(|&(_, i, j)| (i, j)).collect();
    let res = unmix::unmix_pairs_monomial(&gf, &ghat_cols, &pair_list);
    let mut ok = 0;
    {
        let mut f = fs::File::create(out_dir.join("report.txt")).expect("report");
        for (r, u) in res.iter().enumerate() {
            writeln!(
                f, "pair{} ({},{}) alpha*={:?} x={:?} y={:?}",
                r, u.pair.0, u.pair.1, u.alpha_star, u.x, u.y_mult
            )
            .unwrap();
            if u.alpha_star.is_some() {
                ok += 1;
            }
        }
    }
    println!("# unmix: {}/{} pairs unmixed", ok, res.len());
    // Machine-readable: i j alpha x y (-1 for None).
    {
        let mut f = fs::File::create(out_dir.join("xy.txt")).expect("xy");
        for u in res.iter() {
            let a = u.alpha_star.map(|v| v as i32).unwrap_or(-1);
            let x = u.x.map(|v| v as i32).unwrap_or(-1);
            let y = u.y_mult.map(|v| v as i32).unwrap_or(-1);
            writeln!(f, "{} {} {} {} {}", u.pair.0, u.pair.1, a, x, y).unwrap();
        }
    }
    res
}

fn cmd_unmix(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let ph1 = get_path(args, "--phase1", "keydir/phase1");
    let ssdir = get_path(args, "--ss", "keydir/ss");
    let out = get_path(args, "--out", "keydir/unmixed");
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    let pairs = read_pairs_txt(&ph1.join("pairs.txt"));
    do_unmix(&pubkey, &pairs, &ssdir, &out);
    println!("# unmix done -> {}", out.display());
}

// ---------------------------------------------------------------------------
// ss (on pure columns from Phase 1)
// ---------------------------------------------------------------------------

/// Run Sidelnikov-Shestakov on the pure columns.
/// Returns (alpha, y, positions) with positions = pure_idx.
///
/// The original Phase 2 bootstrap scan: enumerate all F_q^need lambda
/// tuples, candidate v = c1 + l*c2 per pair, run SS on each k x (k+3)
/// candidate; first success wins. Returns (alpha, y) restricted to the
/// pure positions, or None. This is the path taken with
/// `--phase2 scan`, and the automatic fallback for `--phase2 fast`.
fn bootstrap_scan(
    gf: &Gf,
    pure_cols: &[Vec<u16>],
    pair_cols: &[(Vec<u16>, Vec<u16>)],
    need: usize,
    q: usize,
) -> Option<(Vec<u16>, Vec<u16>)> {
    use rayon::prelude::*;
    let k = pure_cols[0].len();
    let n_pure = pure_cols.len();
    let total = q.pow(need as u32);
    println!("# ss: searching {} lambda tuples (q={})...", total, q);
    // Parallel search; first success wins.
    // (For need=3, q=64: 262k SS runs on k x (k+3); feasible.)
    (0..total).into_par_iter().find_map_any(|t| {
        // Decode t into lambdas (base-q).
        let mut lambdas = vec![0u16; need];
        let mut tt = t;
        for i in 0..need {
            lambdas[i] = (tt % q) as u16;
            tt /= q;
        }
        // Candidate pure columns.
        let mut cols = pure_cols.to_vec();
        for (i, (c1, c2)) in pair_cols.iter().enumerate() {
            let lam = lambdas[i];
            let mut v = vec![0u16; k];
            for l in 0..k {
                v[l] = c1[l] ^ gf.mul(lam, c2[l]);
            }
            // Skip zero vector (degenerate lambda).
            if v.iter().all(|&x| x == 0) {
                return None;
            }
            cols.push(v);
        }
        // SS on k x (k+3). Internal verification rejects non-GRS.
        // (Catch panics: ss_attack may panic on non-GRS input
        // instead of returning Err; treat as rejection.)
        let m_rows = mat::transpose(&cols);
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ss::ss_attack(gf, &m_rows)
        }));
        match res {
            Ok(Ok((a, y))) => {
                // a,y aligned with cols: first n_pure are pure_idx.
                Some((a[..n_pure].to_vec(), y[..n_pure].to_vec()))
            }
            _ => None,
        }
    })
}

/// Paper Phase 2: SS needs k+3 columns. If we have >= k+3 pure columns,
/// run SS directly. Otherwise (e.g. w = n-k gives exactly k pure),
/// unmix `need = (k+3) - n_pure` pairs to obtain `need` extra pure
/// columns, then SS on the k+3. The unmixing search is either the
/// direction-B fast path (--phase2 fast, default; src/fastss.rs) with
/// automatic fallback to the brute-force scan, or the scan itself
/// (--phase2 scan).
/// (Uses Phase 1's pairing; the paper tries 15 pairings for provability.)
fn do_ss(
    pubkey: &keyio::PublicKey,
    pure_idx: &[usize],
    pairs: &[(u32, usize, usize)],
    phase2: Phase2Mode,
    out_dir: &Path,
) -> (Vec<u16>, Vec<u16>, Vec<usize>) {
    fs::create_dir_all(out_dir).expect("mkdir ss out");
    let gf = Gf::for_q(pubkey.q);
    let k = pubkey.k;
    let pure_cols: Vec<Vec<u16>> = pure_idx.iter().map(|&i| pubkey.cols[i].clone()).collect();
    let n_pure = pure_cols.len();
    println!(
        "# ss: {} pure columns, k={} (need k+3={})",
        n_pure, k, k + 3
    );

    // (alpha, y) aligned with pure_idx.
    let (alpha_pure, y_pure): (Vec<u16>, Vec<u16>);

    if n_pure >= k + 3 {
        // Enough pure columns: SS directly on all of them.
        let gp_rows = mat::transpose(&pure_cols);
        println!("# ss: running Sidelnikov-Shestakov on {} pure columns...", n_pure);
        let t0 = std::time::Instant::now();
        let (a, y) = match ss::ss_attack(&gf, &gp_rows) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("# ERROR: Sidelnikov-Shestakov failed: {}", e);
                std::process::exit(1);
            }
        };
        println!("# ss: done in {:.2}s", t0.elapsed().as_secs_f64());
        alpha_pure = a;
        y_pure = y;
    } else {
        // Bootstrap (paper Phase 2, Sec. "brute-force unmixing and SS"):
        // unmix `need` pairs by brute force over F_q^need, SS on k+3.
        let need = (k + 3) - n_pure;
        println!(
            "# ss: only {} pure (< k+3); unmixing {} pairs (paper Phase 2, --phase2 {})...",
            n_pure,
            need,
            phase2.name()
        );
        if pairs.len() < need {
            eprintln!(
                "# ERROR: need {} pairs for bootstrap but Phase 1 found {}",
                need,
                pairs.len()
            );
            std::process::exit(1);
        }
        // Top-`need` pairs by count (Phase 1's pairing).
        let mut sorted = pairs.to_vec();
        sorted.sort_by_key(|&(c, _, _)| std::cmp::Reverse(c));
        let use_pairs: Vec<(usize, usize)> =
            sorted[..need].iter().map(|&(_, i, j)| (i, j)).collect();
        let pair_cols: Vec<(Vec<u16>, Vec<u16>)> = use_pairs
            .iter()
            .map(|&(i, j)| (pubkey.cols[i].clone(), pubkey.cols[j].clone()))
            .collect();

        let q = pubkey.q as usize;
        let t0 = std::time::Instant::now();

        // Search dispatch. `fast` (need=3 only) runs direction B and
        // falls back to the scan if it finds no verified candidate;
        // `scan` (and need != 3) goes straight to the scan.
        let found: Option<(Vec<u16>, Vec<u16>)> = match phase2 {
            Phase2Mode::Fast if need == 3 => {
                println!(
                    "# ss: fast Phase 2 (direction B: dual-square root-finding + SS confirm)..."
                );
                let pc3: Vec<(Vec<u16>, Vec<u16>)> = pair_cols.clone();
                let (res, st) = fastss::fast_bootstrap(&gf, &pure_cols, &pc3);
                println!(
                    "# ss fast: prep={:.3}s stages1+2={:.3}s stage3={:.3}s stage4={:.3}s \
                     orientations={} cells={} roots={} degenerate_cells={} \
                     stage2_survivors={} stage3_survivors={} ss_attempts={} j_swapped={}",
                    st.prep_secs,
                    st.stage12_secs,
                    st.stage3_secs,
                    st.stage4_secs,
                    st.orientations_tried,
                    st.cells,
                    st.stage1_roots,
                    st.degenerate_cells,
                    st.stage2_survivors,
                    st.stage3_survivors,
                    st.ss_attempts,
                    st.j_swapped
                );
                match res {
                    Some((a, y)) => Some((a[..n_pure].to_vec(), y[..n_pure].to_vec())),
                    None => {
                        println!(
                            "# ss: fast Phase 2 found no verified candidate; \
                             falling back to the q^need scan"
                        );
                        bootstrap_scan(&gf, &pure_cols, &pair_cols, need, q)
                    }
                }
            }
            Phase2Mode::Fast => {
                println!(
                    "# ss: --phase2 fast implements need=3 (here need={}); using the scan",
                    need
                );
                bootstrap_scan(&gf, &pure_cols, &pair_cols, need, q)
            }
            Phase2Mode::Scan => bootstrap_scan(&gf, &pure_cols, &pair_cols, need, q),
        };

        match found {
            Some((a, y)) => {
                println!("# ss: bootstrap succeeded in {:.2}s", t0.elapsed().as_secs_f64());
                alpha_pure = a;
                y_pure = y;
            }
            None => {
                eprintln!("# ERROR: bootstrap failed: no lambda tuple gave GRS k+3 columns.");
                eprintln!("#   (Pairs may be wrong, or q^need search incomplete.)");
                std::process::exit(1);
            }
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("config.txt")).expect("ss config");
        writeln!(f, "phase2={}", phase2.name()).unwrap();
    }
    {
        let mut f = fs::File::create(out_dir.join("alpha.txt")).expect("alpha");
        for a in &alpha_pure {
            writeln!(f, "{}", a).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("y.txt")).expect("y");
        for yy in &y_pure {
            writeln!(f, "{}", yy).unwrap();
        }
    }
    {
        let mut f = fs::File::create(out_dir.join("positions.txt")).expect("positions");
        for p in pure_idx {
            writeln!(f, "{}", p).unwrap();
        }
    }
    (alpha_pure, y_pure, pure_idx.to_vec())
}

fn read_ss(out: &Path) -> (Vec<u16>, Vec<u16>, Vec<usize>) {
    let rd = |p: &Path| {
        fs::read_to_string(p)
            .expect("read ss file")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<u16>().unwrap())
            .collect::<Vec<_>>()
    };
    let alpha = rd(&out.join("alpha.txt"));
    let y = rd(&out.join("y.txt"));
    let positions: Vec<usize> = fs::read_to_string(out.join("positions.txt"))
        .expect("positions")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect();
    (alpha, y, positions)
}

fn cmd_ss(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let ph1 = get_path(args, "--phase1", "keydir/phase1");
    let out = get_path(args, "--out", "keydir/ss");
    let phase2 = get_phase2(args);
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    let pure_idx = read_pure_txt(&ph1.join("pure.txt"));
    let pairs = read_pairs_txt(&ph1.join("pairs.txt"));
    do_ss(&pubkey, &pure_idx, &pairs, phase2, &out);
    println!("# ss done -> {}", out.display());
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: &[String]) {
    let key_dir = get_path(args, "--key", "keydir");
    let ph1 = get_path(args, "--phase1", "keydir/phase1");
    let _unm = get_path(args, "--unmixed", "keydir/unmixed");
    let ssdir = get_path(args, "--ss", "keydir/ss");

    let pubkey = keyio::read_pub(&key_dir.join("key.pub")).expect("key.pub");
    let meta = keyio::read_meta(&key_dir.join("key.meta")).expect("key.meta");
    let gf = Gf::for_q(pubkey.q);

    println!("=== VERIFY ===");
    let mut all_ok = true;

    // 1. Pair accuracy (needs key.meta -- testing only).
    // Note: degenerate pairs (c=0 or b=0 in GL(2), ~1/q each) have one
    // member that's actually pure, hence never flagged, hence never
    // co-flagged. The frequency table can't find them; that's expected.
    // We detect "effectively degenerate" via col_flagged == 0.
    let found_pairs = read_pairs_txt(&ph1.join("pairs.txt"));
    let norm = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let found_set: HashSet<(usize, usize)> =
        found_pairs.iter().map(|&(_, i, j)| norm(i, j)).collect();
    let true_set: HashSet<(usize, usize)> =
        meta.pairs.iter().map(|&(a, b)| norm(a, b)).collect();
    // col_flagged for degenerate detection
    let col_flagged: Vec<u32> = {
        let s = fs::read_to_string(ph1.join("col_flagged.txt")).expect("col_flagged");
        let mut v = vec![0u32; pubkey.n_pub];
        for line in s.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let i: usize = parts[0].parse().unwrap();
                v[i] = parts[1].parse().unwrap();
            }
        }
        v
    };
    // non-degenerate: both members flagged at least once (observable proxy)
    let nondeg_true: HashSet<(usize, usize)> = true_set
        .iter()
        .filter(|&&(a, b)| col_flagged[a] > 0 && col_flagged[b] > 0)
        .cloned()
        .collect();
    let n_degen = true_set.len() - nondeg_true.len();
    let correct = found_set.intersection(&nondeg_true).count();
    let false_pos = found_set.difference(&true_set).count();
    println!(
        "# pairs: {}/{} non-degenerate correct, {} false positives (w={}, {} degenerate)",
        correct,
        nondeg_true.len(),
        false_pos,
        meta.w,
        n_degen
    );
    if false_pos > 0 || correct != nondeg_true.len() {
        println!("# FAIL: pair recovery incorrect");
        all_ok = false;
    } else {
        println!(
            "# PASS: all non-degenerate twin pairs found ({} degenerate skipped)",
            n_degen
        );
    }

    // 2. SS code check: recovered GRS(alpha,y) spans the same code as the
    //    pure columns SS ran on.
    let (alpha, y, positions) = read_ss(&ssdir);
    let pure_idx = read_pure_txt(&ph1.join("pure.txt"));
    let mut pure_cols: Vec<Vec<u16>> = Vec::new();
    for &i in &pure_idx {
        pure_cols.push(pubkey.cols[i].clone());
    }
    // candidate generator from (alpha,y)
    let vk = mat::vandermonde(&gf, &alpha, pubkey.k);
    let gcand: Vec<Vec<u16>> = vk
        .iter()
        .map(|row| {
            row.iter()
                .zip(y.iter())
                .map(|(&a, &yy)| gf.mul(a, yy))
                .collect()
        })
        .collect();
    // stack pure_cols (as rows) and gcand rows; rank must equal k
    let mut both = mat::transpose(&pure_cols);
    for r in gcand {
        both.push(r);
    }
    let rk = rank(&gf, &both);
    println!("# SS code check: stacked rank = {} (k={})", rk, pubkey.k);
    if rk != pubkey.k {
        println!("# FAIL: recovered GRS does not match pure columns");
        all_ok = false;
    } else {
        println!("# PASS: SS recovered the correct GRS code");
    }

    // 3. End-to-end round-trip: encrypt with public key, decode with
    //    recovered GRS code via Welch-Berlekamp.
    //
    //    IMPORTANT: use a CONSISTENT (alpha,y) parameterization. SS alpha
    //    and unmixing x may differ by an affine transform, so we read ALL
    //    (x,y) directly from the monomial-basis columns:
    //      pure positions: x = Ghat[1][p]/Ghat[0][p], y = Ghat[0][p]
    //      unmixed: (x, y) from xy.txt (already monomial basis).
    let (alpha_ss, y_ss, positions_ss) = (&alpha, &y, &positions);
    let _ = (alpha_ss, y_ss); // (used above for the code check)
    // Recompute Ghat (monomial basis) for consistent (x,y) read-off.
    let gpub_rows = mat::transpose(&pubkey.cols);
    let (ghat_rows, sinv) = ss::to_monomial_basis(&gf, &gpub_rows, positions_ss, &alpha, &y)
        .expect("monomial basis for verify");
    let ghat_cols = mat::transpose(&ghat_rows);
    // Build (pa, py) for the true pure positions from monomial-basis columns.
    // (Unmixed vectors don't align with ciphertext positions, so the
    // round-trip uses only original pure positions.)
    let mut pa: Vec<u16> = Vec::new();
    let mut py: Vec<u16> = Vec::new();
    for &p in positions_ss {
        let col = &ghat_cols[p];
        if col[0] == 0 {
            println!("# FAIL: zero leading coordinate in monomial basis");
            all_ok = false;
        } else {
            py.push(col[0]);
            pa.push(gf.mul(col[1], gf.inv(col[0])));
        }
    }
    // Build (pa, py) for the true pure positions from monomial-basis columns.
    let n_pure = positions_ss.len();
    let full_alpha = pa;
    let full_y = py;
    let full_pos: Vec<usize> = positions_ss.to_vec();
    // Conservative t for WB: need 2t <= n-k-2 (strictly below the bound
    // for a uniquely determined system).
    let t = n_pure.saturating_sub(pubkey.k + 2) / 2; // error capacity
    println!(
        "# round-trip pure set: {} positions",
        n_pure
    );
    if n_pure <= pubkey.k || t == 0 {
        // At w=n-k, n_pure=k, so no round-trip possible. The structural
        // checks (pairs + SS code) are the verification; round-trip is
        // skipped (not a failure).
        println!("# SKIP: round-trip (n_pure={} <= k={}); structural checks suffice", n_pure, pubkey.k);
    } else {
    println!("# round-trip: n_pure={} k={} t={}", n_pure, pubkey.k, t);
    let mut rng = Rng::new(20261019);
    let msg: Vec<u16> = (0..pubkey.k).map(|_| rng.below(pubkey.q as u64) as u16).collect();
    // c = m * G_pub  (1 x N)
    let mut c = vec![0u16; pubkey.n_pub];
    for j in 0..pubkey.n_pub {
        let mut acc = 0u16;
        for i in 0..pubkey.k {
            acc ^= gf.mul(msg[i], pubkey.cols[j][i]);
        }
        c[j] = acc;
    }
    // add t random errors at random positions
    let mut epos = Vec::new();
    while epos.len() < t {
        let p = rng.below(pubkey.n_pub as u64) as usize;
        if !epos.contains(&p) {
            epos.push(p);
        }
    }
    for &p in &epos {
        c[p] ^= (rng.below((pubkey.q - 1) as u64) + 1) as u16;
    }
    // restrict to full pure positions; decode with (full_alpha, full_y)
    let recv: Vec<u16> = full_pos.iter().map(|&p| c[p]).collect();
    let e_on_pure = epos.iter().filter(|p| full_pos.contains(p)).count();
    println!("# errors on pure positions: {}/{}", e_on_pure, t);
    match ss::wb_decode_k(&gf, &full_alpha, &full_y, &recv, pubkey.k, t) {
        Some(mprime) => {
            // m' is the message in monomial basis (m' = m*S, where
            // Gpub = S*Gmono and Sinv = inv(S)).
            // Recover original: m = m' * inv(S) = m' * Sinv.
            let mut m_rec = vec![0u16; pubkey.k];
            for j in 0..pubkey.k {
                let mut acc = 0u16;
                for i in 0..pubkey.k {
                    acc ^= gf.mul(mprime[i], sinv[i][j]);
                }
                m_rec[j] = acc;
            }
            if m_rec[..] == msg[..] {
                println!("# PASS: round-trip decrypt recovered the message");
            } else {
                println!("# FAIL: WB decoded to wrong message");
                all_ok = false;
            }
        }
        None => {
            println!("# FAIL: WB decoding failed");
            all_ok = false;
        }
    }
    } // end n_pure ok

    println!("=== VERDICT: {} ===", if all_ok { "ATTACK SUCCEEDED" } else { "ATTACK FAILED" });
    if !all_ok {
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// all
// ---------------------------------------------------------------------------

fn cmd_all(args: &[String]) {
    let q = get_u32(args, "--q", 64);
    let n = get_usize(args, "--n", 30);
    let k = get_usize(args, "--k", 14);
    let w = get_usize(args, "--w", 4);
    let nprime = get_usize(args, "--nprime", 28);
    let s = get_usize(args, "--s", 2);
    let d = get_usize(args, "--d", 4);
    // --trials / --y-margin: None means "not given" (auto-planning).
    let trials: Option<usize> = get_flag(args, "--trials").map(|v| v.parse().expect("bad int"));
    let threads = get_usize(args, "--threads", 0);
    let seed = get_u64(args, "--seed", 42);
    let y_margin: Option<usize> = get_flag(args, "--y-margin").map(|v| v.parse().expect("bad int"));
    let fail_eps = get_f64(args, "--fail-eps", 0.01);
    let mode = get_phase1_mode(args);
    let engine = get_engine_for_mode(args, mode);
    let phase2 = get_phase2(args);
    let puncture_size: Option<usize> = get_flag(args, "--puncture-size").map(|v| v.parse().expect("bad int"));
    let puncturings: Option<usize> = get_flag(args, "--puncturings").map(|v| v.parse().expect("bad int"));
    let allow_large = has_flag(args, "--allow-large");
    let out = get_path(args, "--out", "attack_out");
    fs::create_dir_all(&out).expect("mkdir out");

    println!("=== RLCE ATTACK (end-to-end) ===");
    println!("# params: q={} n={} k={} w={}", q, n, k, w);
    println!("# phase1: n'={} (s,d)=({},{}) trials={} threads={} seed={} y={} engine={} mode={}",
        nprime, s, d,
        trials.map(|t| t.to_string()).unwrap_or_else(|| "auto".to_string()),
        threads, seed,
        y_margin.map(|t| t.to_string()).unwrap_or_else(|| "auto".to_string()),
        engine.name(), mode.name());
    println!("# phase2: bootstrap={} (need=3 fast path; falls back to scan)", phase2.name());

    // keygen (in-memory; also write files for the subcommands)
    let gf = Gf::for_q(q);
    let mut rng = Rng::new(seed);
    let key = rlce::gen_rlce(&gf, &mut rng, n, k, w);
    keyio::write_pub(&out.join("key.pub"), q, n, k, w, &key.cols).expect("key.pub");
    keyio::write_meta(&out.join("key.meta"), q, n, k, w, &key.is_pure, &key.pair_id, &key.pairs)
        .expect("key.meta");
    let pubkey = keyio::PublicKey {
        q, n, k, w,
        n_pub: key.n_pub,
        cols: key.cols,
    };

    // phase1
    let ph1_dir = out.join("phase1");
    let run = run_phase1(
        mode,
        &pubkey,
        nprime,
        s,
        d,
        trials,
        threads,
        seed + 1000,
        y_margin,
        fail_eps,
        engine,
        &ph1_dir,
        puncture_size,
        puncturings,
        allow_large,
    );
    let mut pairs = run.pairs;
    let pure_idx = run.pure_idx;

    // ss (on pure columns)
    let ss_dir = out.join("ss");
    let (alpha, y, positions) = do_ss(&pubkey, &pure_idx, &pairs, phase2, &ss_dir);

    // Cluster-mode orphan fallback (needs the SS parametrization, so it
    // runs here, between ss and unmix). Attempts the 2-plane test for
    // each orphan against every pure candidate; see cluster.rs for why
    // the test cannot discriminate (it passes for every candidate) and
    // therefore never resolves. Unresolved orphans are reported as
    // INCOMPLETE with the unresolved pool -- never a guessed pairing.
    // Resolved pairs (if any ever occur) are appended to pairs.txt so
    // unmix and verify see the complete pairing.
    if mode == Phase1Mode::Cluster {
        if let Some(co) = &run.cluster {
            if co.orphans.is_empty() {
                println!("# cluster fallback: no orphans, nothing to resolve");
            } else {
                println!(
                    "# cluster fallback: {} orphan(s), {} pure candidates each; running 2-plane tests...",
                    co.orphans.len(),
                    pure_idx.len()
                );
                let gpub_rows = mat::transpose(&pubkey.cols);
                let (ghat_rows, _sinv) =
                    ss::to_monomial_basis(&gf, &gpub_rows, &positions, &alpha, &y)
                        .expect("monomial basis transform for orphan fallback");
                let ghat_cols = mat::transpose(&ghat_rows);
                let attempts = cluster::resolve_orphans(
                    &gf,
                    &ghat_cols,
                    &alpha,
                    &y,
                    &positions,
                    &co.orphans,
                );
                let mut n_resolved = 0usize;
                let mut unresolved: Vec<usize> = Vec::new();
                {
                    let mut f = fs::File::create(ph1_dir.join("orphan_resolution.txt"))
                        .expect("orphan_resolution");
                    for a in &attempts {
                        writeln!(
                            f,
                            "orphan {} candidates={} minor_passing={} rank_passing={} extra_roots={:?} resolved={:?}",
                            a.orphan,
                            a.n_candidates,
                            a.n_minor_passing,
                            a.n_rank_passing,
                            a.extra_roots,
                            a.resolved
                        )
                        .unwrap();
                        println!(
                            "#   orphan {}: {}/{} candidates pass the 2-plane test \
                             (minor), {}/{} by rank; extra roots: {:?}; resolved: {:?}",
                            a.orphan,
                            a.n_minor_passing,
                            a.n_candidates,
                            a.n_rank_passing,
                            a.n_candidates,
                            a.extra_roots,
                            a.resolved
                        );
                        match a.resolved {
                            Some(p) => {
                                n_resolved += 1;
                                pairs.push((0, a.orphan.min(p), a.orphan.max(p)));
                            }
                            None => unresolved.push(a.orphan),
                        }
                    }
                }
                if n_resolved > 0 {
                    write_pairs_txt(&ph1_dir.join("pairs.txt"), &pairs);
                    println!(
                        "# cluster fallback: {} orphan(s) resolved and appended to pairs.txt",
                        n_resolved
                    );
                }
                if !unresolved.is_empty() {
                    println!("=== PHASE 1 INCOMPLETE ===");
                    println!(
                        "# {} orphan(s) could NOT be associated to a pure twin: {:?}",
                        unresolved.len(),
                        unresolved
                    );
                    println!(
                        "# (The 2-plane test passes for every pure candidate -- the pure \
                         member's own direction always lies in the span -- so no \
                         attack-visible data discriminates between candidates. See \
                         cluster.rs. The run continues with the {} normal pairs; \
                         the degenerate pair(s) are documented, not guessed.)",
                        pairs.len()
                    );
                }
            }
        }
    }

    // unmix (in monomial basis, needs SS output)
    let unm_dir = out.join("unmixed");
    do_unmix(&pubkey, &pairs, &ss_dir, &unm_dir);

    // verify (uses key.meta for scoring; the attack itself never did)
    println!();
    let vargs = vec![
        "verify".to_string(),
        "--key".to_string(), out.to_string_lossy().to_string(),
        "--phase1".to_string(), ph1_dir.to_string_lossy().to_string(),
        "--unmixed".to_string(), unm_dir.to_string_lossy().to_string(),
        "--ss".to_string(), ss_dir.to_string_lossy().to_string(),
    ];
    cmd_verify(&vargs);
}

// ---------------------------------------------------------------------------
// holdout-test (flag-none investigation; see holdout.rs)
// ---------------------------------------------------------------------------

fn cmd_holdout_test(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let meta_path: Option<PathBuf> = get_flag(args, "--meta").map(PathBuf::from);
    let nprime = get_usize(args, "--nprime", 28);
    let s = get_usize(args, "--s", 2);
    let d = get_usize(args, "--d", 3);
    let trials = get_usize(args, "--trials", 2);
    let seed = get_u64(args, "--seed", 100);
    let random_vecs = get_usize(args, "--random-vecs", 20);
    let pair_tests = get_usize(args, "--pair-tests", 3);
    let out_path: Option<PathBuf> = get_flag(args, "--out").map(PathBuf::from);
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    let meta = match &meta_path {
        Some(p) => Some(keyio::read_meta(p).expect("read key.meta")),
        None => {
            // Default location next to key.pub, if it exists; otherwise
            // pair tests are skipped (see below). This mirrors verify's
            // testing-only use of meta: the attack itself never needs it.
            let cand = key_path.with_file_name("key.meta");
            if cand.exists() {
                Some(keyio::read_meta(&cand).expect("read key.meta"))
            } else {
                None
            }
        }
    };
    if meta.is_none() {
        println!("# holdout-test: no key.meta found; pair-hold-out tests skipped (rank + random-vector tests still run)");
    }
    let kp = pubkey.k as isize - (pubkey.n_pub as isize - nprime as isize);
    if kp <= 0 {
        eprintln!("# ERROR: k' = {} <= 0 for n'={} (N={}, k={})", kp, nprime, pubkey.n_pub, pubkey.k);
        std::process::exit(1);
    }
    let kp = kp as usize;
    // Dense-echelon size guard: E = n' * J, N_d = C(k'+d-1, d).
    // J and N_d via the same enumerations the jet matrix uses.
    let jn: usize = (0..s).map(|sum| gijs::enum_exp(kp, sum).len()).sum();
    let nd = gijs::enum_exp(kp, d).len();
    let entries = (nprime * jn) as u128 * nd as u128;
    println!(
        "# holdout-test: key N={} k={} w={} n'={} k'={} (s,d)=({},{}) trials={} seed={} random_vecs={} pair_tests={}",
        pubkey.n_pub, pubkey.k, pubkey.w, nprime, kp, s, d, trials, seed, random_vecs, pair_tests
    );
    println!(
        "# holdout-test: jet matrix E={} (J={}) N_d={} entries={} per matrix; ~{} dense eliminations per trial (1 random + 2 per pair test)",
        nprime * jn, jn, nd, entries, 1 + 2 * pair_tests
    );
    if entries > 200_000_000 && !has_flag(args, "--allow-large") {
        eprintln!(
            "# ERROR: jet matrix has {} entries (>200M); a dense elimination here is not a diagnostic, it is a flagship run. Refusing. Re-run with --allow-large if you really mean it.",
            entries
        );
        std::process::exit(1);
    }
    let gf = Gf::for_q(pubkey.q);
    let results = holdout::run_holdout_trials(
        &gf,
        &pubkey.cols,
        meta.as_ref(),
        nprime,
        s,
        d,
        trials,
        seed,
        random_vecs,
        pair_tests,
    );
    let mut report = String::new();
    let mut tot_random_in_full = 0usize;
    let mut tot_random = 0usize;
    let mut tot_random_in_tau = 0usize;
    let mut tot_pair = 0usize;
    let mut tot_pair_without_mate_in = 0usize;
    let mut tot_pair_without_random_in = 0usize;
    for r in &results {
        report.push_str(&format!(
            "trial {}: n'={} k'={} flagged_normal={} E_full={} N_d={} rank_full={} codim_full={} random_in_full={}/{} random_in_tau={}/{}\n",
            r.trial,
            r.nprime,
            r.kp,
            r.flagged_normal,
            r.e_full,
            r.nd,
            r.rank_full,
            r.nd.saturating_sub(r.rank_full),
            r.random_in_full,
            r.random_total,
            r.random_in_tau,
            r.random_total
        ));
        tot_random_in_full += r.random_in_full;
        tot_random += r.random_total;
        tot_random_in_tau += r.random_in_tau;
        for p in &r.pair_tests {
            report.push_str(&format!(
                "  pair tau={} mate={} control={}: in_with_mate={} in_without_mate={} in_without_random={}\n",
                p.tau_orig, p.mate_orig, p.control_orig, p.in_with_mate, p.in_without_mate, p.in_without_random
            ));
            tot_pair += 1;
            if p.in_without_mate {
                tot_pair_without_mate_in += 1;
            }
            if p.in_without_random {
                tot_pair_without_random_in += 1;
            }
        }
    }
    report.push_str(&format!(
        "SUMMARY: trials={} random_in_full={}/{} random_in_tau={}/{} pair_tests={} in_without_mate={} in_without_random={}\n",
        results.len(),
        tot_random_in_full,
        tot_random,
        tot_random_in_tau,
        tot_random,
        tot_pair,
        tot_pair_without_mate_in,
        tot_pair_without_random_in
    ));
    report.push_str("# Reading: random_in_* should be ~0 in every regime (span is proper, not whole space).\n");
    report.push_str("# in_without_mate == in_with_mate (both true) for dirty taus => collective spanning.\n");
    report.push_str("# in_without_mate == false while in_without_random == true => the mate was essential.\n");
    report.push_str("# NOTE: pair tests delete the mate's shortened column from the point set; they do NOT re-shorten (see holdout.rs).\n");
    print!("{}", report);
    if let Some(p) = out_path {
        fs::write(&p, &report).expect("write holdout report");
        println!("# holdout-test report -> {}", p.display());
    }
    println!("# holdout-test done");
}

// ---------------------------------------------------------------------------
// reuse-holdout-test: mate present / mate absent / control absent,
// from ONE left-kernel factorization per trial (see reuse.rs).
// Exact, not heuristic: for a deleted block set S, tau is in the span
// of the remaining points iff the (tau,0) column of Pperp restricted
// to S raises the rank. This replaces holdout-test's 2+2P dense
// eliminations per trial with 1 reuse factorization per trial, so
// the cost is independent of the number of pair tests. It does NOT
// include random-vector tests (those need a row-space basis, not a
// left kernel); the control-absent verdict is the random-deletion
// control. Same shortening, same pair order, same control choice as
// holdout-test, so outputs are directly comparable.
// ---------------------------------------------------------------------------

fn cmd_reuse_holdout_test(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let meta_path: Option<PathBuf> = get_flag(args, "--meta").map(PathBuf::from);
    let nprime = get_usize(args, "--nprime", 28);
    let s = get_usize(args, "--s", 2);
    let d = get_usize(args, "--d", 3);
    let trials = get_usize(args, "--trials", 2);
    let seed = get_u64(args, "--seed", 100);
    let pair_tests = get_usize(args, "--pair-tests", 3);
    let out_path: Option<PathBuf> = get_flag(args, "--out").map(PathBuf::from);
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    let meta = match &meta_path {
        Some(p) => Some(keyio::read_meta(p).expect("read key.meta")),
        None => {
            let cand = key_path.with_file_name("key.meta");
            if cand.exists() {
                Some(keyio::read_meta(&cand).expect("read key.meta"))
            } else {
                None
            }
        }
    };
    if meta.is_none() {
        println!("# reuse-holdout-test: no key.meta found; pair-hold-out tests skipped (rank + flagged_normal still run)");
    }
    let kp = pubkey.k as isize - (pubkey.n_pub as isize - nprime as isize);
    if kp <= 0 {
        eprintln!("# ERROR: k' = {} <= 0 for n'={} (N={}, k={})", kp, nprime, pubkey.n_pub, pubkey.k);
        std::process::exit(1);
    }
    let kp = kp as usize;
    if !has_flag(args, "--allow-large") {
        if let Err(msg) = phase1::check_reuse_memory(nprime, kp, s, d) {
            eprintln!("# ERROR: reuse would exceed the guard: {}", msg);
            std::process::exit(1);
        }
    }
    let gf = Gf::for_q(pubkey.q);
    let n_pub = pubkey.cols.len();
    println!(
        "# reuse-holdout-test: key N={} k={} w={} n'={} k'={} (s,d)=({},{}) trials={} seed={} pair_tests={} (1 reuse factorization per trial, no dense eliminations)",
        n_pub, pubkey.k, pubkey.w, nprime, kp, s, d, trials, seed, pair_tests
    );
    let mut report = String::new();
    let mut tot_pair = 0usize;
    let mut tot_without_mate_in = 0usize;
    let mut tot_without_random_in = 0usize;
    for tr in 0..trials {
        let trial_seed = seed + 1 + tr as u64;
        let mut rng = Rng::new(trial_seed);
        let hit = rng.subset(n_pub, n_pub - nprime);
        let (short_cols, kept) = linalg::shorten(&gf, &pubkey.cols, &hit);
        let (pperp, jn, e_full, nd) = reuse::pperp_for_cols(&gf, &short_cols, s, d);
        let rank_full = e_full.saturating_sub(pperp.len());
        let mut flagged_normal = 0usize;
        if pperp.is_empty() {
            flagged_normal = short_cols.len();
        } else {
            for pos in 0..short_cols.len() {
                let base = pos * jn;
                let block: Vec<usize> = (base..base + jn).collect();
                if !reuse::in_span_from_pperp(&gf, &pperp, &block) {
                    flagged_normal += 1;
                }
            }
        }
        report.push_str(&format!(
            "trial {}: n'={} k'={} flagged_normal={} E_full={} N_d={} rank_full={} codim_full={}\n",
            tr, nprime, kp, flagged_normal, e_full, nd, rank_full, nd.saturating_sub(rank_full)
        ));
        if pair_tests == 0 {
            continue;
        }
        let Some(m) = meta.as_ref() else { continue };
        let mut tested = 0usize;
        for (pos, &orig) in kept.iter().enumerate() {
            if tested >= pair_tests {
                break;
            }
            let pid = match m.pair_id[orig] {
                Some(p) => p,
                None => continue,
            };
            let (a, b) = m.pairs[pid];
            let mate = if a == orig { b } else { a };
            if !kept.contains(&mate) {
                continue;
            }
            let mate_pos = kept.iter().position(|&x| x == mate).unwrap();
            let control_pos = kept.iter().position(|&x| x != orig && x != mate).unwrap();
            let control_orig = kept[control_pos];
            let tau_base = pos * jn;
            let mate_base = mate_pos * jn;
            let control_base = control_pos * jn;
            let tau_block: Vec<usize> = (tau_base..tau_base + jn).collect();
            let mut tau_mate: Vec<usize> = tau_block.clone();
            tau_mate.extend(mate_base..mate_base + jn);
            let mut tau_control: Vec<usize> = tau_block.clone();
            tau_control.extend(control_base..control_base + jn);
            let in_with_mate = reuse::in_span_from_pperp(&gf, &pperp, &tau_block);
            let in_without_mate = reuse::in_span_from_pperp(&gf, &pperp, &tau_mate);
            let in_without_random = reuse::in_span_from_pperp(&gf, &pperp, &tau_control);
            report.push_str(&format!(
                "  pair tau={} mate={} control={}: in_with_mate={} in_without_mate={} in_without_random={}\n",
                orig, mate, control_orig, in_with_mate, in_without_mate, in_without_random
            ));
            tot_pair += 1;
            if in_without_mate {
                tot_without_mate_in += 1;
            }
            if in_without_random {
                tot_without_random_in += 1;
            }
            tested += 1;
        }
    }
    report.push_str(&format!(
        "SUMMARY: trials={} pair_tests={} in_without_mate={} in_without_random={}\n",
        trials, tot_pair, tot_without_mate_in, tot_without_random_in
    ));
    report.push_str("# Reading: in_without_mate == in_with_mate (both true) => collective spanning.\n");
    report.push_str("# in_without_mate == false while in_without_random == true => the mate was essential.\n");
    report.push_str("# NOTE: pair tests delete the mate's / control's shortened column from the point set; they do NOT re-shorten.\n");
    report.push_str("# NOTE: no random-vector test in this command; use holdout-test for random_in_full / random_in_tau.\n");
    print!("{}", report);
    if let Some(p) = out_path {
        fs::write(&p, &report).expect("write reuse-holdout report");
        println!("# reuse-holdout-test report -> {}", p.display());
    }
    println!("# reuse-holdout-test done");
}

// ---------------------------------------------------------------------------
// puncture-test (shortening + puncturing; see puncture.rs)
// ---------------------------------------------------------------------------

fn cmd_puncture_test(args: &[String]) {
    let key_path = get_path(args, "--key", "keydir/key.pub");
    let meta_path: Option<PathBuf> = get_flag(args, "--meta").map(PathBuf::from);
    let nprime = get_usize(args, "--nprime", 28);
    let s = get_usize(args, "--s", 2);
    let d = get_usize(args, "--d", 3);
    let shortenings = get_usize(args, "--shortenings", 1);
    let puncturings = get_usize(args, "--puncturings", 20);
    let seed = get_u64(args, "--seed", 100);
    let puncture_seed = get_u64(args, "--puncture-seed", seed + 1_000_000);
    let out_path: Option<PathBuf> = get_flag(args, "--out").map(PathBuf::from);
    let pubkey = keyio::read_pub(&key_path).expect("read key.pub");
    let meta = match &meta_path {
        Some(p) => Some(keyio::read_meta(p).expect("read key.meta")),
        None => {
            let cand = key_path.with_file_name("key.meta");
            if cand.exists() {
                Some(keyio::read_meta(&cand).expect("read key.meta"))
            } else {
                None
            }
        }
    };
    if meta.is_none() {
        println!("# puncture-test: no key.meta found; category counts and pair intersections skipped (flagged counts still run)");
    }
    let kp = pubkey.k as isize - (pubkey.n_pub as isize - nprime as isize);
    if kp <= 0 {
        eprintln!("# ERROR: k' = {} <= 0 for n'={} (N={}, k={})", kp, nprime, pubkey.n_pub, pubkey.k);
        std::process::exit(1);
    }
    let kp = kp as usize;
    let puncture_size: usize = match get_flag(args, "--puncture-size") {
        Some(v) => v.parse().expect("bad int"),
        None => {
            let frac = get_f64(args, "--puncture-frac", 0.5);
            ((nprime as f64 * frac) as usize).clamp(1, nprime.saturating_sub(1))
        }
    };
    if puncture_size == 0 || puncture_size >= nprime {
        eprintln!("# ERROR: puncture-size {} must satisfy 1 <= p < n'={}", puncture_size, nprime);
        std::process::exit(1);
    }
    let npp = nprime - puncture_size;
    // Reuse guard on the *punctured* set (the set actually factored):
    // refuse flagship-shaped punctured configs unless --allow-large.
    // phase1::check_reuse_memory uses the same 8 GiB limit.
    if !has_flag(args, "--allow-large") {
        if let Err(msg) = phase1::check_reuse_memory(npp, kp, s, d) {
            eprintln!("# ERROR: punctured reuse would exceed the guard: {}", msg);
            std::process::exit(1);
        }
    }
    println!(
        "# puncture-test: key N={} k={} w={} n'={} k'={} (s,d)=({},{}) shortenings={} puncturings={} puncture_size={} n''={} seed={} puncture_seed={}",
        pubkey.n_pub, pubkey.k, pubkey.w, nprime, kp, s, d, shortenings, puncturings, puncture_size, npp, seed, puncture_seed
    );
    let gf = Gf::for_q(pubkey.q);
    let mp = indicatif::MultiProgress::new();
    let pb = mp.add(puncture_progress_bar(shortenings.saturating_mul(puncturings)));
    let factor = mp.add(factorization_progress_bar());
    let mut report = String::new();
    for sh in 0..shortenings {
        let shortening_seed = seed + 1 + sh as u64;
        let rep = puncture::run_shortening(
            &gf,
            &pubkey.cols,
            meta.as_ref(),
            nprime,
            s,
            d,
            sh,
            shortening_seed,
            puncturings,
            puncture_size,
            puncture_seed,
            Some(&pb),
            Some(&factor),
            None,
            None,
        );
        println!("# puncture-test: shortening {} done", sh);
        report.push_str(&format!(
            "shortening {}: kept={} ({} columns)\n",
            sh,
            rep.kept
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            rep.kept.len()
        ));
        for t in &rep.trials {
            report.push_str(&format!(
                "  puncturing {}: flagged={} [pure={} singleton_hit={} singleton_punctured={} both_kept={}] totals [pure={} singleton_hit={} singleton_punctured={} both_kept={}]\n",
                t.puncturing,
                t.flagged.len(),
                t.flagged_by_cat[0],
                t.flagged_by_cat[1],
                t.flagged_by_cat[2],
                t.flagged_by_cat[3],
                t.total_by_cat[0],
                t.total_by_cat[1],
                t.total_by_cat[2],
                t.total_by_cat[3]
            ));
        }
        // Attacker intersections, per twin pair both kept, only for
        // pairs where at least one member was flagged (otherwise the
        // intersection is trivially empty and uninformative).
        let mut exact = 0usize;
        let mut contains = 0usize;
        let mut missing = 0usize;
        let mut listed = 0usize;
        for (a, b, ia, ib) in &rep.pair_intersections {
            let fa = rep.trials.iter().any(|t| t.flagged.contains(a));
            let fb = rep.trials.iter().any(|t| t.flagged.contains(b));
            if !fa && !fb {
                continue;
            }
            listed += 1;
            let mut line = format!("  pair ({},{})", a, b);
            for (member, mate, inter) in [(a, b, ia), (b, a, ib)] {
                let flagged_m = rep.trials.iter().any(|t| t.flagged.contains(member));
                if !flagged_m {
                    line.push_str(&format!(" {}:never_flagged", member));
                    continue;
                }
                let mut sorted: Vec<usize> = inter.iter().cloned().collect();
                sorted.sort_unstable();
                let inter_str = sorted
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                if inter.len() == 1 && inter.contains(mate) {
                    exact += 1;
                    line.push_str(&format!(" {}:exact{{{}}}", member, inter_str));
                } else if inter.contains(mate) {
                    contains += 1;
                    line.push_str(&format!(" {}:contains_mate{{{}}}", member, inter_str));
                } else {
                    missing += 1;
                    line.push_str(&format!(" {}:mate_missing{{{}}}", member, inter_str));
                }
            }
            report.push_str(&line);
            report.push('\n');
        }
        report.push_str(&format!(
            "  shortening {} summary: pairs_listed={} member_exact={} member_contains_mate={} member_mate_missing={}\n",
            sh, listed, exact, contains, missing
        ));
    }
    report.push_str("# Reading: in the flag-none regime, flagged should be mostly singleton_punctured, not both_kept or pure.\n");
    report.push_str("# A member is 'exact' iff its attacker intersection (over all puncturings where it was kept and flagged) is exactly {true mate}.\n");
    report.push_str("# NOTE: puncturing deletes columns from the shortened point set; it does NOT re-shorten (see puncture.rs).\n");
    factor.finish_and_clear();
    pb.finish_and_clear();
    print!("{}", report);
    if let Some(p) = out_path {
        fs::write(&p, &report).expect("write puncture report");
        println!("# puncture-test report -> {}", p.display());
    }
    println!("# puncture-test done");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        std::process::exit(1);
    }
    match args[1].as_str() {
        "keygen" => cmd_keygen(&args[2..]),
        "phase1" => cmd_phase1(&args[2..]),
        "unmix" => cmd_unmix(&args[2..]),
        "ss" => cmd_ss(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "holdout-test" => cmd_holdout_test(&args[2..]),
        "reuse-holdout-test" => cmd_reuse_holdout_test(&args[2..]),
        "puncture-test" => cmd_puncture_test(&args[2..]),
        "all" => cmd_all(&args[2..]),
        "-h" | "--help" | "help" => usage(),
        other => {
            eprintln!("unknown subcommand: {}", other);
            usage();
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bootstrap_tests {
    use super::*;

    /// Paper Phase 2 bootstrap, tested in isolation (Phase 1 mocked with
    /// ground truth): at w=n-k we have exactly k pure columns; brute-force
    /// unmixing 3 pairs over F_q^3 + SS on k x (k+3) must recover the GRS.
    #[test]
    fn bootstrap_w_eq_n_minus_k() {
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(777);
        // w = n - k boundary: n=20, k=14, w=6.
        let key = rlce::gen_rlce(&gf, &mut rng, 20, 14, 6);
        let k = key.k;
        // Ground-truth pure columns (as Phase 1 would output).
        let pure_idx: Vec<usize> = (0..key.n_pub).filter(|&i| key.is_pure[i]).collect();
        assert_eq!(pure_idx.len(), k, "at w=n-k there are exactly k pure columns");
        let pure_cols: Vec<Vec<u16>> = pure_idx.iter().map(|&i| key.cols[i].clone()).collect();
        let need = (k + 3) - pure_cols.len();
        assert_eq!(need, 3);
        // Take 3 true pairs.
        let use_pairs: Vec<(usize, usize)> = key.pairs[..need].to_vec();
        let q = 64usize;
        let total = q.pow(need as u32);
        let mut found: Option<(Vec<u16>, Vec<u16>)> = None;
        for t in 0..total {
            let mut lambdas = vec![0u16; need];
            let mut tt = t;
            for i in 0..need {
                lambdas[i] = (tt % q) as u16;
                tt /= q;
            }
            let mut cols = pure_cols.clone();
            let mut ok = true;
            for (idx, &(pi, pj)) in use_pairs.iter().enumerate() {
                let (c1, c2) = (&key.cols[pi], &key.cols[pj]);
                let lam = lambdas[idx];
                let mut v = vec![0u16; k];
                for l in 0..k {
                    v[l] = c1[l] ^ gf.mul(lam, c2[l]);
                }
                if v.iter().all(|&x| x == 0) {
                    ok = false;
                    break;
                }
                cols.push(v);
            }
            if !ok {
                continue;
            }
            let m_rows = mat::transpose(&cols);
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ss::ss_attack(&gf, &m_rows)
            }));
            if let Ok(Ok((a, y))) = res {
                found = Some((a, y));
                break;
            }
        }
        let (alpha, y) = found.expect("bootstrap must find GRS k+3 columns");
        assert_eq!(alpha.len(), k + 3);
        // Verify: the (alpha,y) GRS code on the k pure positions matches
        // the pure columns' code (stacked rank == k).
        let cand_rows = {
            let g: Vec<Vec<u16>> = (0..k)
                .map(|r| alpha[..k].iter().map(|&a| {
                    let mut pw = 1u16;
                    for _ in 0..r { pw = gf.mul(pw, a); }
                    pw
                }).collect::<Vec<_>>())
                .collect();
            let mut rows = Vec::new();
            for r in 0..k {
                let mut row = Vec::new();
                for c in 0..k {
                    row.push(gf.mul(y[c], g[r][c]));
                }
                rows.push(row);
            }
            rows
        };
        let mut stacked = cand_rows.clone();
        stacked.extend(mat::transpose(&pure_cols));
        assert_eq!(mat::rank(&gf, &stacked), k, "SS code must match pure columns");
    }

    /// Check that (alpha, y) restricted to the pure positions spans
    /// exactly the pure columns' code.
    fn check_grs_on_pure(
        gf: &Gf,
        alpha: &[u16],
        y: &[u16],
        pure_cols: &[Vec<u16>],
    ) {
        let k = pure_cols.len();
        let cand_rows: Vec<Vec<u16>> = (0..k)
            .map(|r| {
                (0..k)
                    .map(|cc| {
                        let mut pw = 1u16;
                        for _ in 0..r {
                            pw = gf.mul(pw, alpha[cc]);
                        }
                        gf.mul(y[cc], pw)
                    })
                    .collect()
            })
            .collect();
        let mut stacked = cand_rows;
        stacked.extend(mat::transpose(pure_cols));
        assert_eq!(mat::rank(gf, &stacked), k);
    }

    /// Differential test: direction-B fast Phase 2 vs the original
    /// q^3 scan (`bootstrap_scan`, the exact helper the CLI uses) on
    /// several random w = n-k instances. Both must succeed and both
    /// results must verify against the pure columns.
    #[test]
    fn fast_vs_scan_differential() {
        for &(q, n, k, w, seed) in &[
            (32u32, 14usize, 10usize, 4usize, 11u64),
            (32, 14, 10, 4, 222),
            (32, 14, 10, 4, 3333),
        ] {
            let gf = Gf::for_q(q);
            let mut rng = Rng::new(seed);
            let key = rlce::gen_rlce(&gf, &mut rng, n, k, w);
            let pure_idx: Vec<usize> =
                (0..key.n_pub).filter(|&i| key.is_pure[i]).collect();
            assert_eq!(pure_idx.len(), k);
            let pure_cols: Vec<Vec<u16>> =
                pure_idx.iter().map(|&i| key.cols[i].clone()).collect();
            let pair_cols: Vec<(Vec<u16>, Vec<u16>)> = key.pairs[..3]
                .iter()
                .map(|&(i, j)| (key.cols[i].clone(), key.cols[j].clone()))
                .collect();
            // Scan path.
            let scan = bootstrap_scan(&gf, &pure_cols, &pair_cols, 3, q as usize)
                .expect("scan must succeed");
            check_grs_on_pure(&gf, &scan.0, &scan.1, &pure_cols);
            // Fast path.
            let (fast, _stats) = fastss::fast_bootstrap(&gf, &pure_cols, &pair_cols);
            let (alpha, y) = fast.expect("fast path must succeed");
            assert_eq!(alpha.len(), k + 3);
            // fast returns the full k+3 frame; check its pure part.
            check_grs_on_pure(&gf, &alpha[..k], &y[..k], &pure_cols);
        }
    }
}
