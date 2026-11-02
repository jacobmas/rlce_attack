# rlce-attack: End-to-End RLCE Key Recovery

This binary implements the **complete** key-recovery attack from the paper
"Heuristic Key Recovery Against RLCE in 2^75 Time", running end-to-end on
tractable parameters as an empirical demonstration.

## Attack chain

```
Phase 1 (pair finding):
  Random shortenings + GIJS distinguisher -> per-trial flagged sets
  -> frequency table of co-flagged pairs -> top-w = twin pairs.
  Multiprocessed (rayon), trials spooled to disk, aggregation streams
  files (RAM stays bounded).

Phase 2a (Sidelnikov-Shestakov):
  From the (n-w) pure columns (never flagged), recover GRS support
  `alpha` and multipliers `y`; rewrite public key in monomial basis.

Phase 2b (unmixing):
  In monomial basis, for each pair (c1,c2), try all alpha in F_q^*:
  v = c1 + alpha*c2. The alpha* giving a geometric progression
  (pure GRS column) unmixes the pair.

Verify:
  - Found pairs match true pairs (needs key.meta; attack never sees it).
  - SS code check.
  - Round-trip: encrypt random message with public key, add errors,
    decode with Welch-Berlekamp on recovered GRS code.
```

## Quick start

```bash
# Build
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release

# Run the full attack (recommended; trial count auto-calculated)
./target/release/rlce-attack all \
  --q 64 --n 30 --k 14 --w 4 \
  --nprime 28 --s 2 --d 4 \
  --threads 0 --seed 42 \
  --out /tmp/attack_demo/

# Expected output ends with:
#   === VERDICT: ATTACK SUCCEEDED ===
```

Parameters:
- `--q 64 --n 30 --k 14 --w 4`: RLCE key (GF(64), 30 GRS cols, dim 14, 4 pairs).
- `--nprime 28 --s 2 --d 4`: GIJS params (use `rlce-test --search` to tune).
- `--trials T`: Phase-1 raw trial count. **Optional** — when omitted it is
  auto-calculated from the attack parameters (see "Auto trial planning"
  below); when given it is used verbatim (the auto chain is still printed
  for comparison).
- `--threads 0`: 0 = all cores; or set explicitly.
- `--seed 42`: keygen seed (attack uses seed+1000 for trials).
- `--y-margin Y`: filter safety margin (T_reject = t'_max - y).
  **Optional** — when omitted the planner chooses y; when given it
  overrides the planner's y.
- `--fail-eps E`: per-component failure target for auto-planning
  (default 0.01; the three components are each bounded by E, so the
  total planning failure is <= ~3E by a union bound).

## Subcommands (for step-by-step)

```bash
# 1. Key generation (writes key.pub + key.meta; meta is TESTING ONLY)
rlce-attack keygen --q 64 --n 30 --k 14 --w 4 --seed 42 --out keydir/

# 2. Phase 1: pair finding (reads key.pub ONLY - honest attacker)
#    (--trials omitted -> auto-calculated; add --plan-only to just print
#     the planning chain without running any trials)
rlce-attack phase1 --key keydir/key.pub --nprime 28 --s 2 --d 4 \
  --threads 0 --seed 100 --out keydir/phase1/

# 3. SS: recover GRS structure from pure columns
rlce-attack ss --key keydir/key.pub --phase1 keydir/phase1/ --out keydir/ss/

# 4. Unmix pairs in monomial basis
rlce-attack unmix --key keydir/key.pub --phase1 keydir/phase1/ \
  --ss keydir/ss/ --out keydir/unmixed/

# 5. Verify (reads key.meta to SCORE; attack never saw it)
rlce-attack verify --key keydir/ --phase1 keydir/phase1/ \
  --unmixed keydir/unmixed/ --ss keydir/ss/
```

## Choosing (n', s, d)

Use the `--search` mode of `rlce-test` to find valid GIJS parameters:

```bash
./target/release/rlce-test --search --q 64 --n 30 --k 14 --w 4
```

Pick a row with good kernel margin (>10%) and reconstruction margin (>5%).
Invalid (s,d) either flag nothing or everything -- that's a parameter
error, not an attack failure.

Tested working:
- q=64, n=30, k=14, w=4, n'=28, (s,d)=(2,4), trials=200-300
- q=128, n=40, k=28, w=6, n'=31, (s,d)=(2,3)

## Auto trial planning

When `--trials` is not given, Phase 1 derives its raw trial count `T`
(and the filter margin `y`, unless `--y-margin` is given) from the
attack parameters, implementing the corrected frequency-table analysis
(the filter-correction note for the paper's Theorem 3). The full chain
is printed at startup and stored in `phase1/config.txt`. Notation:
`N = n + w`, `k' = k - (N - n')`, `D = k' + w - 1`, eps = `--fail-eps`
(default 0.01). Model: a shortening keeps a uniform n'-subset S; a pair
contributes 2 noisy columns iff both members are in S ("unhit"), a
half-hit pair is derandomized, so `t' = 2u` with `u ~ Bin(w, p_unhit)`.

1. `p_unhit = n'(n'-1) / (N(N-1))`.
2. `t'_max = ceil((n'-1) - d*D/s) - 1` (PIR persistence threshold,
   the same formula the code already used); `J_max = floor(t'_max/2)`;
   a trial is *good* iff `t' <= t'_max`, i.e. `u <= J_max`.
3. Filter margin: `T_reject = t'_max - y` with `y` the smallest margin
   such that `P[recorded AND bad] <= eps`, computed under heuristic H3
   as `sum_{u>J_max} P[u] * P[Bin(2u, 1-d/q) <= T_reject]` (given t',
   |V| stochastically dominates Bin(t', 1-d/q), so this is an upper
   bound on the bad side). The note's worst-case per-bad-trial bound
   `delta_sneak(y)` is printed for reference; it is too loose to choose
   y by. If `T_reject < 2` the filter could never record a flagged
   pair, and planning fails with an infeasibility message.
4. `P_record = P[recorded AND good] = sum_{u<=J_max} P[u] *
   P[Bin(2u, 1-d/q) <= T_reject]` (H3 planning model: detection
   probability set to its lower bound 1 - d/q; for `2u <= T_reject`
   this is the deterministic-recording case of the note). The note's
   `p_gr = P[t' <= t'_max - y]` (deterministic part only) is printed
   alongside, as is `P_good = P[t' <= t'_max]`.
5. Separation rates: true pairs co-flag at rate
   `p1_min = p_unhit * (1 - 2d/q)` per recorded good trial (H2');
   non-pairs need both pairs unhit, rate at most the exact 4-unhit
   probability `p_false_max = (n')_4 / (N)_4 <= p_unhit^2`.
   Gap `gamma = p1_min - p_false_max` (planning fails if <= 0).
6. `r` = smallest recorded-good-trial count whose best flag-count
   threshold separates the families with failure <= eps: exact binomial
   tails, union bound over the `w` true pairs (must stay above the
   threshold) and all `C(N,2)` non-pairs (must stay below).
7. `T` = smallest raw count with `P[Bin(T, P_record) >= r] >= 1 - eps`.

Each of steps 3, 6, 7 is bounded by eps, so the total planning failure
is <= ~3*eps by a union bound. If any step is unattainable (e.g.
`P_record ~ 0`, or `T` would exceed the 2^24 cap), planning fails
*before any trial runs* with an explicit infeasibility message —
mirroring the old "0 trials recorded" failure, but at planning time.
With `--trials` given, the run proceeds with the manual count even if
planning fails (a warning is printed; not recommended).

Worked example (the demo point q=64, n=30, k=14, w=4, n'=28, (2,4),
eps=0.01): p_unhit=0.6738, t'_max=4, J_max=2, y=1, T_reject=3,
P_good=0.3947, P_record=0.1708 (p_gr=0.1049), p1_min=0.5896,
p_false_max=0.4415, gamma=0.1481, r=602, **T=3836** raw trials
(the old manual default was 300, which happened to suffice at this
toy point because the true detection rates beat the H1/H2' bounds).

`phase1 --plan-only` prints the chain without running trials — handy
for auditing large parameter sets (e.g. ID-0: t'_max=35, J_max=17,
y=0, P_good = 2^-8.91 in this binomial model — the paper quotes
2^-7.8 from its exact count, a factor ~2.2 — r=317, T=173135 ~ 2^17.4;
see AUTOTRIALS_REPORT.md for the reconciliation).

## Phase 2: Sidelnikov-Shestakov (with w=n-k bootstrap)

SS needs k+3 columns. If Phase 1 yields >= k+3 pure columns (w < n-k-3),
SS runs directly. Otherwise (e.g. w=n-k gives exactly k pure),
`need=(k+3)-n_pure` pairs must be unmixed first. Two searches are
available via `--phase2` (for `ss` and `all`; recorded in
`ss/config.txt`):

- `--phase2 fast` (default; need=3): "direction B" (src/fastss.rs).
  Normalize by the pure columns (P^{-1}, once), so candidates are
  g_i(l_i) = a_i + l_i*b_i in systematic form. The candidate's dual is
  then explicit, and GRS-ness is equivalent to a rank condition on the
  3×k matrix M* of pairwise products of the g_i coordinates
  (dim D^2 = 3 + rank M*; GRS iff rank <= 2 with a smooth relation
  conic and distinct support). Stage 1 fixes (l1,l2) in F_q^2 and
  root-finds l3 from a 3×3 minor of M* (a char-2 quadratic, O(1) in
  k); stage 2 filters by a second minor; stage 3 is the exact
  dual-rank test per survivor; stage 4 runs SS on the survivors in
  original coordinates. lambda=infinity (degenerate pairs) is covered
  by retrying with pair orientations flipped (8 combinations). If the
  fast path finds no verified candidate it falls back to the scan
  automatically. Measured at (q,k)=(256,165): 0.03 s total for the
  fast path vs ~7.8 ms *per candidate* (and ~2^23 candidates
  expected) for the scan. See FASTPHASE2_REPORT.md.
- `--phase2 scan`: the original brute-force search over F_q^need,
  running SS on each k×(k+3) candidate; the first success (SS has
  internal verification) gives the GRS structure. (Follows the paper
  Sec. "Phase 2: brute-force unmixing and Sidelnikov--Shestakov";
  we use Phase 1's pairing, not all 15.)

## Memory / disk

- Trial flagged sets are spooled to `phase1/trials/trial_*.txt` (one line
  each) and streamed during aggregation. RAM stays O(C(N,2)) for the
  pair counter (1.2MB at N=790) or sparse HashMap for very large N.
- **Resume:** re-running Phase 1 with identical parameters into the same
  output directory does not recompute trials whose spool file already
  exists and is well-formed (per-trial seeds are deterministic, so the
  files are interchangeable). This makes long Phase-1 runs
  interruptible. Delete the `trials/` directory to force recomputation;
  reusing a directory with *different* parameters is a user error.
- Key columns live in RAM (N*k*2 bytes). GIJS matrices are allocated
  per hold-out and freed.
- For multi-GB keys, the trial spooling already streams; the pair
  counter switches to sparse automatically when C(N,2) > 1e8.
- Low-memory Phase 1: `phase1`/`all` accept `--engine bw-sigma`, which
  replaces the dense distinguisher with the matrix-free block-Wiedemann
  distinguisher whose extraction is a sigma-basis (minimal approximant
  basis, `src/sigma.rs` + `src/matfree.rs`). It never forms the Hasse
  matrix M or the (L*b)^2 block-Toeplitz matrix: peak memory is
  O(N_d + E + sigma * b^2) instead of O(E * N_d) for the dense engine.
  Rejections are certified exactly; accepts are heuristic (same contract
  as the other matrix-free engines). See MATFREE_SIGMA_REPORT.md.
  Example: `rlce-attack all --q 64 --n 30 --k 14 --w 4 --nprime 28
  --s 2 --d 4 --trials 300 --seed 42 --engine bw-sigma --out DIR/`.

## Phase-1 engines (dense / bw-sigma / reuse)

`phase1`/`all` accept `--engine dense|bw-sigma|reuse` (default dense).
All three answer the same per-hold-out GIJS question; they differ in
cost shape:

- **dense**: one Hasse matrix M_tau built and eliminated per hold-out
  (n' eliminations per trial). Exact. Peak memory O(E * N_d) for one
  hold-out at a time. The safe default; the slowest per trial at
  moderate-to-large (n', k').
- **reuse**: ONE factorization of the full jet matrix M_full per trial
  (all n' points at once), then two small rank checks per hold-out
  (`src/reuse.rs`). Decisions are *identical* to dense -- it is the
  same test with the linear algebra rearranged, oracle-tested against
  `dense` on every hold-out -- so flagged sets, spool files, and pair
  tables are interchangeable with dense runs (spool files don't record
  the engine, and mixing engines across a resume is sound for the same
  reason). Measured ~53x faster than dense per trial at
  (q=256, n'=90, k'=24, (2,3)): 7.8 s vs 415 s single-thread.
  The catch is memory: it materializes the augmented matrix
  [M_full | I_E] plus a left-kernel basis, worst case
  O(E * (N_d + E)) -- ~32 MB at the point above, but ~153 GB at
  ID-0-like (n'=348, k'=28, (3,5)) dims. A pre-flight guard estimates
  the footprint from (n', k', s, d) before any trial runs and refuses
  configs over **8 GiB** with an error naming the estimate; use
  bw-sigma there.
- **bw-sigma**: matrix-free; never forms M at all, so it is the only
  engine whose memory survives flagship dims (peak
  O(N_d + E + sigma * b^2)). Per-decision time is Krylov-bound and at
  small/moderate dims it is *slower* than dense (at the (2,3) point
  above it did not finish one trial in 25 min where dense took
  415 s/trial). Rejections certified exact; accepts heuristic.

Rule of thumb: **reuse** for tractable points that pass the guard
(fastest, exact, dense-identical output); **dense** as the reference /
for small points; **bw-sigma** when the reuse estimate exceeds the
guard (flagship memory regime). See REUSE_ENGINE_REPORT.md.

## Honesty

The attack subcommands (`phase1`, `ss`, `unmix`) read ONLY `key.pub`.
The secret twin metadata (`key.meta`) is read ONLY by `verify` to score
the attack. This keeps the security experiment honest.

## Degenerate pairs

With probability ~1/q per pair, the GL(2) mixing has c=0, making one
member actually pure. Such pairs are never co-flagged (the pure member
is never flagged), so the frequency table can't find them. This is
expected behavior (H2' excludes them). `verify` reports them as
"degenerate skipped" and checks the non-degenerate pairs.

## Files

- `src/attack.rs` - CLI + orchestration (keygen, phase1, unmix, ss, verify, all)
- `src/phase1.rs` - trial loop, disk spooling, streaming aggregation
- `src/fastss.rs` - direction-B fast Phase 2 bootstrap (dual-square root-finding)
- `src/ss.rs` - Sidelnikov-Shestakov + Welch-Berlekamp + monomial transform
- `src/unmix.rs` - twin unmixing via geometric-progression check
- `src/mat.rs` - dense linear algebra (rank, matmul, matinv, etc.)
- `src/keyio.rs` - binary key serialization
- `src/gf.rs`, `src/gijs.rs`, `src/linalg.rs`, `src/rlce.rs`, `src/rng.rs` - shared
