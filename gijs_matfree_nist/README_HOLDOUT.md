# holdout-test — new in this version (2026-10-25)

Added for the flag-none investigation: does a truly independent
hold-out get flagged in the flag-none regime, or is the jet span
the whole space?

## Build

```bash
cargo build --release --bin rlce-attack
BIN=./target/release/rlce-attack
```

## Run

```bash
$BIN holdout-test --key DIR/key.pub [--meta DIR/key.meta] \
  --nprime NP --s S --d D \
  [--trials T] [--seed S] [--random-vecs R] [--pair-tests P] \
  [--allow-large] [--out report.txt]
```

Defaults: `--trials 2 --seed 100 --random-vecs 20 --pair-tests 3`.
If `--meta` is omitted, it defaults to `key.meta` next to `key.pub`
when that file exists; without meta, pair tests are skipped and
only rank + random-vector tests run. Meta is testing-only, exactly
like `verify`: the attack itself never reads it.

Trial generation is identical to Phase 1: trial `t` uses seed
`base_seed + 1 + t`, a random `(N - n')`-subset hit, and
`linalg::shorten`.

## What it prints, per trial

- `flagged_normal`: reuse-engine flagged count for the trial
  (the Phase-1 spool count, for comparison).
- `E_full`, `N_d`, `rank_full`, `codim_full = N_d - rank_full`:
  codimension is the log_q probability a uniform random vector
  lands in the span. Codim in the hundreds refutes "whole space
  spanned" even when `flagged_normal = 0`.
- `random_in_full` / `random_in_tau`: uniform vectors in
  F_q^{k'} tested against M_full and against M_tau (tau = first
  dirty kept column by meta, else first kept column).
  Expected ~0 in every regime.
- Per pair (dirty taus only, meta required):
  - `in_with_mate`: tau in span of M_tau (mate present).
    True = NOT flagged (GRS); false = flagged.
  - `in_without_mate`: tau in span of jets excluding tau AND mate.
  - `in_without_random`: control, tau in span excluding tau AND
    the first non-mate kept column.

Reading: if `in_with_mate` is true (flag-none) and
`in_without_mate` is also true, spanning is collective.
If `in_with_mate` is true but `in_without_mate` is false while
`in_without_random` is true, the mate was essential.
If `in_with_mate` is false (working regime), the pair test is
uninformative about the mate: tau was already out of the span.

## Important scope note

Pair tests delete the mate's *shortened column* from the point
set. They do NOT re-shorten the code with the mate hit (that
would change the shortening basis and drop k' by 1). This is a
linear-algebra test of the jet span, which is the mechanism
question, not an attack trial.

## Cost and guard

One dense echelon elimination per matrix tested: 1 for random
vectors + 2 per pair test, per trial. This is tractable-point
only. The command refuses when `E * N_d > 200M` entries unless
`--allow-large` is given (ID-0 (2,3) is 296M). At kp28 (2,3)
the matrix is ~13M entries; at kp28 (2,4) it is ~129M, so
pair tests there will be slow — start with
`--trials 1 --pair-tests 2`.

## Tests

```bash
cargo test --release --bin rlce-attack holdout -- --nocapture
```

Three tests, all passing at time of writing: batch in-span
matches the dense oracle (`gijs::distinguish` on the point set
plus the test vector, held out) on random small instances and
on a shortened RLCE instance, for every tau; and 200 uniform
random vectors are all outside the span on a small random
point set with codim >= 21.

## Verified on the VM (2026-10-25)

- kp28 flag-none point, q=1024, n=131, k=78, w=30, n'=111,
  k'=28, (2,3), 1 trial, seed 100 (2m39s on the 2-CPU VM,
  7 dense eliminations): flagged_normal=0, rank_full=3050,
  codim_full=1010, random vectors 0/20 against M_full and
  0/20 against M_tau. Pair tests (3 dirty taus):
  in_with_mate=true in all 3, in_without_mate=false in all 3,
  in_without_random=false in all 3. So: the span is NOT the
  whole space (codim 1010, random vectors stay outside), and
  the mate is NOT uniquely essential — removing a random
  non-mate point flips tau out of the span just as removing
  the mate does. Tau's in-span status at n'=111 is fragile
  to removing any one point, not mate-specific. That is a
  first-trial result (n=1), not a multi-trial statistic.

## Verified on the VM (2026-10-25) — tiny points

- Tiny working point, q=64, n=30, k=20, w=5, n'=28, k'=13,
  (2,3), 2 trials: flagged_normal=6, rank_full=360,
  codim_full=95, random 0/40, pair tests all
  in_with_mate=false (flagged, as expected in working regime).
- Tiny degenerate point, q=64, n=30, k=14, w=4, n'=28, k'=8,
  (2,3): rank_full=120=N_d, codim 0, random 20/20 — the span
  really is the whole space there, so that point cannot
  discriminate the hypotheses. Pick points with E < N_d
  (codim > 0) for the random-vector test.
