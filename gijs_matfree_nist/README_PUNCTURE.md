
# puncture-test — new in this version (2026-10-25)

Shortening + puncturing measurement for the flag-none regime.
See `src/puncture.rs` for the full rationale, including the
singleton-hit vs singleton-punctured distinction (shortening
at the mate derandomizes it to clean; puncturing does not).

## Build

```bash
cargo build --release --bin rlce-attack
BIN=./target/release/rlce-attack
```

## Run

```bash
$BIN puncture-test --key DIR/key.pub [--meta DIR/key.meta] \
  --nprime NP --s S --d D \
  [--shortenings SH] [--puncturings PU] \
  [--puncture-size P | --puncture-frac F] \
  [--seed S] [--puncture-seed PS] [--allow-large] [--out report.txt]
```

Defaults: `--shortenings 1 --puncturings 20 --puncture-frac 0.5
--seed 100 --puncture-seed 1000100` (seed + 1,000,000).
Meta defaults to `key.meta` next to `key.pub` when present;
without meta, category counts and pair intersections are
skipped (flagged counts still run).

Each puncturing is ONE reuse factorization on the punctured
point set (n'' = n' - p columns, dimension k' unchanged) —
no dense elimination, no per-pair factorization. The reuse
guard (8 GiB) is applied to the punctured set unless
`--allow-large` is given.

## Output, per puncturing

`flagged=T [pure=A singleton_hit=B singleton_punctured=C
both_kept=D]` with totals in the same order. Categories
(meta, testing only):

- pure: is_pure column.
- singleton_hit: twin member whose mate was hit by shortening.
  Derandomized, behaves clean.
- singleton_punctured: twin member whose mate was in the
  shortened kept set but was punctured this trial. Mixed,
  expected flagged in the flag-none regime.
- both_kept: twin member whose mate is also in the punctured
  kept set. Expected GRS (not flagged) in the flag-none
  regime, flagged in the working regime.

Per shortening, per twin pair both kept, the attacker
intersection over puncturings where that member was kept
and flagged is reported as exact / contains_mate /
mate_missing, with a per-shortening summary.

## Measured on the VM (2026-10-25) — read before choosing p

- Tiny point, q=64, n=30, k=20, w=5, n'=28, k'=13, (2,3),
  puncture to n''=14: flagged = 14/14 in every puncturing
  (flag-all). Punctured code is far below its PIR floor
  ((n''-1)=13 vs d*D/s=25.5), so this point cannot test
  puncturing at all.
- kp28 flag-none point, q=1024, n=135, k=78, w=30, n'=115,
  k'=28, (2,3), 1 shortening, 10 puncturings, p=57
  (n''=58), 30 seconds total on the 2-CPU VM:
  flagged = 58/58 in every puncturing, in every category
  (pure, singleton_hit, singleton_punctured, both_kept all
  100%). Intersection: 0 exact, 0 contains_mate, 28
  mate_missing. Half-puncturing does NOT give a singleton
  signal at this point — it pushes the punctured code into
  flag-all, so every column flags and intersection is
  meaningless.
- Consequence: p must keep n'' above the PIR floor, not
  just maximize singleton probability. At kp28, (2,3),
  D=57, floor is (n''-1) > 85.5, i.e. n'' >= 87, i.e.
  p <= 28 for n'=115. Try p=10, 20, 28 next, not p=57.
  The p=57 result does not refute puncturing in general,
  it refutes half-puncturing at this point.

## Tests

```bash
cargo test --release --bin rlce-attack puncture -- --nocapture
```

Three tests, passing: classification of all four categories
on a hand-built meta, intersection shrinking, and a tiny
puncture run whose counts sum correctly (flagged subset of
kept', flagged counts sum to flagged total, totals sum to n'').

---

# Phase-1 puncture mode (full end-to-end, added 2026-10-25)

`--phase1-mode puncture` turns the puncture test into a real
Phase-1 option. It works on `phase1` and on `all`, with the
same output files as the other modes (`pairs.txt`, `pure.txt`,
`col_flagged.txt`, `config.txt`), so `ss` / `unmix` / `verify`
run unchanged. No meta is used anywhere in this mode.

```bash
# Plan only (prints shortenings, puncturings, total factorizations):
$BIN phase1 --key DIR/key.pub --nprime NP --s S --d D \
  --phase1-mode puncture --plan-only

# Run it (defaults below are auto-planned; --trials and
# --puncturings override them):
$BIN phase1 --key DIR/key.pub --nprime NP --s S --d D \
  --phase1-mode puncture [--trials SH] [--puncturings PU] \
  [--puncture-size P] --threads TH --seed S --out DIR/phase1/

# End to end:
$BIN all --q Q --n N --k K --w W --nprime NP --s S --d D \
  --phase1-mode puncture [--trials SH] [--puncturings PU] \
  [--puncture-size P] --threads TH --seed S --out DIR/
```

## How it decides pairs

Per shortening, it runs P puncturings of size p on the
shortened code (one reuse factorization each). For each kept
column it intersects the punctured sets of the puncturings
where that column flagged. In the flag-none regime only a
column whose mate was punctured flags, so that intersection
should shrink to the true mate. A shortening votes once for
a pair when either member's intersection is exactly the
other member. Pairs in `pairs.txt` are ONLY exact pairs,
ranked by the number of shortenings voting exact and taken
greedily disjoint, up to w. Columns that flag but never
reach an exact pair are NOT guessed: they are listed in
`puncture_candidates.txt` with their smallest intersection,
and per-shortening intersections are in
`puncture_intersections.txt`. If fewer than w pairs are
exact, a warning is printed and the run continues with the
pairs found — downstream stages may then fail honestly.

Pure columns are columns never flagged in any puncturing
of any shortening (same definition as the other modes).

## Auto-planning (see `puncture_plan` in attack.rs)

- `--trials` = shortenings S. Auto: S = ceil(ln(0.5/w) /
  ln(1 - p_both)), at least 5, where p_both =
  (n'/N) * ((n'-1)/(N-1)) is the chance a pair survives
  shortening both-kept. This makes the expected number of
  pairs never both-kept below 0.5.
- `--puncturings` = puncturings per shortening P. Auto:
  P = ceil(6 / p_single), at least 50, where p_single =
  p*(n'-p) / (n'*(n'-1)) is the chance a both-kept member
  flags in one puncturing. This gives a mean of ~6 flags
  per member per shortening.
- `--puncture-size` default 4 (the validated kp28 size).
  p=2 also works but needs twice the puncturings for the
  same mean. p=20/28 at kp28 left the flag-none regime
  (both-kept flagged too) and must NOT be used here.
- `--threads` parallelizes over shortenings only (one
  worker per shortening; puncturings inside a shortening
  run in order). So wall time is roughly P * (one reuse
  factorization) when threads >= shortenings, not
  (S*P / threads) * factorization. Spool files in
  `phase1/trials/puncture_shortening_*.txt` are
  checkpointed after EVERY puncturing (atomically, via
  a temp file plus rename), with a `done K Q` line, so
  a run killed mid-shortening resumes from puncturing
  K, not from the start of the shortening: the loss on
  a crash is at most the one factorization in flight
  (about 11 s at kp28, about 28 min at ID-0), not the
  whole shortening (about 33 min at kp28, about 11 days
  at ID-0). Completed shortenings are reloaded, not
  recomputed (a second run with identical parameters is
  near-instant). Spools written by older versions, with
  no `done` line, are still read as complete.
- The reuse memory guard is applied to the punctured set
  (n'', k'); `--allow-large` overrides it.
- Progress: both `puncture-test` and Phase-1 puncture mode
  show two levels of bar, drawn through one MultiProgress
  so concurrent shortenings each get their own line:
  an overall bar (same style as frequency mode) that
  counts completed reuse factorizations, with elapsed time
  and ETA, plus one per-factorization bar per shortening
  in flight, labelled "shortening S puncturing Q", that
  tracks that factorization's elimination. The
  per-factorization bar is work-weighted — elimination
  below a pivot touches a shrinking submatrix, so a plain
  column-count bar would crawl then sprint; the work
  estimate makes it move roughly linearly in time.
  Building the augmented matrix and the per-tau rank
  checks are not counted, both being small next to the
  elimination at the sizes where this matters. Both bars
  are hidden when output is piped to a file; the plain
  `# puncture: shortening N done` line per shortening
  (or `done (reloaded from spool)` on a resumed run)
  still appears then. Run in a terminal, not piped, to
  see the bars.

## Cost examples (factorizations = shortenings x puncturings)

- kp28 (N=165, n'=115, w=30, p=4): auto S=7, P=177,
  1,239 factorizations. At ~11 s per factorization (the
  rate measured on Jacob's machine for puncture-test at
  this point), that is ~32 min wall with 7 workers, or
  ~3.8 h if run one shortening at a time.
- ID-0 (N=790, n'=366, w=160, p=4): auto S=24, P=554,
  13,296 factorizations. At ~60 min per factorization
  (the ID-0 reuse rate measured at n'=375), that is
  ~13,300 CPU-hours, ~28 days wall on a 20-thread
  machine, less on many VMs. And this is only worth
  attempting if the flag-none regime holds at n''=362
  (n''-k'=316, still far above the ~130-140 ceiling
  extrapolated from kp28/kp40) — untested at ID-0 scale.
  Do not start an ID-0 puncture run on the kp28 evidence
  alone; a single ID-0 shortening with a few dozen
  puncturings is the cheap regime check first.

## Validation (VM, 2026-10-25)

Small flag-none point, q=64, n=40, k=15, w=5, N=45,
n'=42, k'=12, (2,3): frequency mode flags 0 columns in
3/3 trials (0 pairs, pure set wrongly = all 45 columns).
Puncture mode, 5 shortenings x 50 puncturings, p=4,
13.5 s on the 2-CPU VM: 5/5 exact pairs, pure set exactly
the 35 true pure columns, and `all` with the same settings
ends `ATTACK SUCCEEDED` (5/5 pairs, 5/5 unmixed, SS
correct, round-trip decrypt passes). A second run with
identical parameters reloads the spool files and is
near-instant. This validates the code path and the
aggregation; it does NOT validate ID-0, where the
regime and the cost are both untested.
