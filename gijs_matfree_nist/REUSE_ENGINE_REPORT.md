# Hold-out reuse engine (`--engine reuse`) — wiring report

Phase 1 gains a third distinguisher engine: **reuse**
(`src/reuse.rs::distinguish_all_reuse`, previously built and
oracle-tested but not reachable from the CLI). It computes all n'
hold-out decisions of a trial from ONE factorization of the full jet
matrix M_full, instead of n' separate per-hold-out eliminations
(dense) or n' matrix-free Krylov runs (bw-sigma).

## Wiring

- `phase1::DistEngine` gains a `Reuse` variant (name `"reuse"`),
  parsed from `--engine reuse` in `attack.rs::get_engine`, so it works
  on both the `phase1` and `all` paths. The choice is printed in the
  run headers and recorded in `phase1/config.txt` via the existing
  `engine=` field.
- In `phase1::attack_trial_with_engine`, the Reuse engine branches
  BEFORE the per-hold-out loop: one `distinguish_all_reuse` call on
  the shortened columns, then flagged = `kept[ti]` for each `ti` whose
  decision is false (not-GRS), sorted. The dense and bw-sigma code
  paths are untouched (the Reuse arm inside the per-tau match is
  `unreachable!`).
- `src/reuse.rs` itself: the ONLY changes are additive — a public
  `reuse_memory_estimate` function (+ `ReuseMemoryEstimate` struct)
  and two estimator unit tests. `distinguish_all_reuse` and all its
  internals are byte-identical to before. The estimator must live next
  to `build_augmented` because it mirrors that function's exact
  allocation layout; duplicating the dimension arithmetic in phase1.rs
  would risk silent drift from the allocator it guards.

## Memory guard (pre-flight)

The reuse engine materializes the augmented matrix [M_full | I_E]
(E = n' * J rows, N_d + E columns of u16) plus the left-kernel basis
Pperp ((E - rank) rows of length E). At ID-0-like flagship dims
(n'=348, k'=28, (s,d)=(3,5)) the augmented matrix alone is ~107 GB,
so an unguarded `--engine reuse` there would OOM.

`phase1::check_reuse_memory(n', k', s, d)` computes, from dimensions
alone (allocating nothing):

    J = C(k'+s-1, s-1),  N_d = C(k'+d-1, d),  E = n' * J
    augmented bytes = E * (N_d + E) * 2
    Pperp bytes    <= E * E * 2        (worst case, rank >= 0;
                                        in practice rank ~= min(N_d, E),
                                        so the basis is usually far
                                        smaller -- often empty)

`run_trials` calls it before creating the trial directory or the
thread pool; if the total estimate exceeds **8 GiB**
(`REUSE_MEMORY_LIMIT_BYTES`), the run exits with an error naming the
estimated sizes and suggesting bw-sigma. The estimate is printed in
the run header for accepted configs.

Verified at the CLI (ID-0 key, `phase1 --nprime 348 --s 3 --d 5
--engine reuse`): exits 1 before any trial with

    # ERROR: --engine reuse at (n'=348, k'=28, s=3, d=5) would need up
    to 152.6 GB (augmented matrix [M_full | I_E] = 106.8 GB,
    left-kernel basis <= 45.8 GB), over the 8.6 GB pre-flight limit.
    Use --engine bw-sigma for parameters of this size (it never
    forms M).

At the benchmark target point below the estimate is 31.9 MB worst
case, so the guard never fires in normal tractable use.

## Correctness

Decisions of `distinguish_all_reuse` are the dense decisions by
construction (same GIJS test, rearranged linear algebra; see the
module docs in reuse.rs). Evidence, all green:

1. **Pre-existing oracle tests** (reuse.rs, unchanged): reuse ==
   per-tau `distinguish` on EVERY hold-out across random codes, pure
   GRS sets, degenerate columns, and RLCE shortenings.
2. **NEW trial-level differential**
   (`phase1::tests::reuse_engine_matches_dense_trials`):
   `attack_trial_with_engine` Dense vs Reuse flagged sets identical
   on 8 trials: random columns (24 cols of length 10, n'=18, (2,3),
   3 seeds); RLCE toy shape (30,14,4) at (2,4), n'=28 (2 seeds);
   RLCE (30,20,5) at (2,3), n'=28 -> k'=13 (3 seeds). This pins the
   wiring (kept-index mapping, pre-loop branch) specifically.
3. **NEW guard/estimator tests**
   (`phase1::tests::reuse_memory_guard_checks`,
   `reuse::tests::memory_estimate_dims`,
   `reuse::tests::memory_estimate_flagship_is_huge`): flagship dims
   rejected from dimensions alone; target-point dims exact
   (J=25, N_d=2600, E=2250, augmented = 21,825,000 B).
4. **Binary end-to-end**: `rlce-attack all --engine reuse` at the
   established toy point (q=64, n=30, k=14, w=4, n'=28, (2,4),
   auto-trials, Phase 2 fast): **ATTACK SUCCEEDED** in 39.9 s wall on
   this VM (the same pipeline with the dense engine took 626 s).
   Phase 1 recorded **319/3836** trials — identical to the known
   dense run — and pairs.txt is identical:
   `92 (21,26)`, `80 (23,30)`, `72 (3,27)`, `61 (11,12)`.
5. **Binary spool diff**: a standalone dense Phase 1 (same key,
   seed 1042, auto-trials) was run into a separate directory and
   compared against the reuse `all` run's Phase 1 output: **all
   3,836 trial spool files byte-identical**, and `pairs.txt`,
   `col_flagged.txt`, `pure.txt` all identical.

## Benchmark — target point (measured on this VM)

Parameters: q=256, n=198, k=165, w=33 (N=231), key seed 42;
Phase 1 n'=90, (s,d)=(2,3) -> k'=24, J=25, N_d=2600, E=2250.
Auto-plan: T=3,106 raw trials, r=293, T_reject=4.

| engine | per-trial time (threads=1) | source |
|---|---|---|
| reuse | **7.8 s** | this run: 1 trial 7.87 s, 10 trials 77.79 s (fixed overhead ~0.1 s) |
| dense | 415 s | reference measurement, same VM (not re-run) |
| bw-sigma | > 1,500 s (one trial unfinished at the 25-min cap) | reference measurement, same VM |

- Speedup of reuse over dense at this point: **~53x**.
- Peak RSS of the reuse Phase-1 process (threads=1, 10 sequential
  trials): **24.9 MiB** — one trial's footprint plus the key; the
  21.8 MB augmented matrix dominates, as estimated (the Pperp basis
  is empty at these dims since E < N_d and M_full has full row rank;
  the 10.1 MB figure is the guard's worst-case bound, not observed).
- Sanity: 1 of the 10 benchmark trials was recorded; the auto-plan's
  model prediction is P_record = 0.107 per trial.
- Extrapolation (ESTIMATE): 3,106 trials x 7.8 s = ~24,200 s
  = **~6.7 thread-hours**. On 20 threads at the same per-core speed
  that is **~20 min wall**; trials are independent and each needs
  ~25 MB, so 20-way parallelism costs ~500 MB and should scale
  near-linearly. The user's machine's per-core speed is expected to
  be at least this VM's.
- Contention labeling: the reuse timings above were measured on an
  otherwise idle 2-core VM. The dense and bw-sigma reference numbers
  were measured earlier on the same VM during a head-to-head race in
  which the two engines shared the box (each pinned to one core);
  treat all three as same-VM, order-of-magnitude-solid figures, not
  cross-machine absolutes.

## Spool / resume compatibility

Spool files (`trials/trial_*.txt`) record only the flagged index
list, not the engine. Because reuse decisions are identical to dense
decisions (proven by the oracle tests and verified byte-identical
over 3,836 spooled trials above), **mixing dense and reuse across a
resume is sound**: resuming a dense-spooled run with `--engine reuse`
(or vice versa) yields exactly the files both engines would have
produced. (config.txt does record which engine the invoking run
used.)

## Regression

Full suite (`cargo test --release --offline`), this VM:

- rlce-attack: **64 passed / 0 failed / 22 ignored** (baseline 60 +
  the 4 new tests listed in "Correctness" items 2-3), finished in
  1,429 s (dominated by pre-existing Phase-1 / matrix-free e2e
  tests; the new tests together take < 3 s).
- rlce-test: **4 passed / 0 failed** (unchanged).

## When to use which engine

- **reuse**: tractable points passing the 8 GiB guard — fastest by
  ~50x at moderate dims, exact, output identical to dense.
- **dense**: reference implementation; small points; anywhere reuse's
  estimate is unwanted.
- **bw-sigma**: flagship-shaped params where the reuse estimate
  exceeds the guard — the only engine whose memory survives there,
  at a large per-decision time cost.
