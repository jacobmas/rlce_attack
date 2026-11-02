# Auto trial-count planning for Phase 1 — report (2026-10-21)

Phase 1's raw trial count is now **auto-calculated** from the attack
parameters when `--trials` is not given (`src/phase1.rs::plan_trials`,
wired into `phase1` and `all` in `src/attack.rs`). `--trials` remains an
explicit override (used verbatim; the auto chain is still printed for
comparison). `--fail-eps E` sets the per-component failure target
(default 0.01). `--y-margin Y` still overrides the planner's filter
margin. `phase1 --plan-only` prints the chain without running trials.
The chain is printed on stdout and stored in `phase1/config.txt`.
Engine-independent (identical for `--engine dense` and `bw-sigma`).

## The calculation (filter-correction analysis)

For (q, N = n+w, k, w, n′, s, d), eps:

1. `p_unhit = n'(n'-1)/(N(N-1))`; #unhit `u ~ Bin(w, p_unhit)`,
   `t' = 2u` (a half-hit pair derandomizes, CLT).
2. `t'_max = ceil((n'-1) - d*D/s) - 1`, `D = k' + w - 1` (the code's
   existing `tprime_max`); `J_max = floor(t'_max/2)`; good ⟺ `u <= J_max`.
3. `y` = smallest margin with **P[recorded AND bad] <= eps**, computed
   under the note's H3 model:
   `sum_{u>J_max} P[u] * P[Bin(2u, 1-d/q) <= T_reject]`,
   `T_reject = t'_max - y`. (H3: given t', |V| stochastically dominates
   Bin(t', 1-d/q), so the binomial CDF is a rigorous upper bound on the
   bad side.) The note's worst-case per-bad-trial bound
   `delta_sneak(y) = max_{t'>t'_max} exp(-2 t' [(1-d/q) - T_reject/t']_+^2)`
   is printed for reference only — it is far too loose to choose y by
   (at ID-0 it equals 0.963 at y=0 and would force y=9, cutting
   P_record by ~2^4; at the toy point it forces T_reject < 2, where no
   flagged pair can ever be recorded).
   If `T_reject < 2`, planning fails: the filter could never admit a
   pair, so the frequency table would stay empty by construction.
4. `P_record = P[recorded AND good] = sum_{u<=J_max} P[u] *
   P[Bin(2u, 1-d/q) <= T_reject]`. **Stated assumption**: on the good
   side the same H3 model is used as the planning value, with detection
   probability set to its lower bound 1 - d/q. For `2u <= T_reject` it
   reduces to the note's deterministic recording; in the band
   `T_reject < 2u <= t'_max` it credits the H3 rate. The note's theorem
   credits only `p_gr = P[t' <= t'_max - y]`; p_gr is printed alongside.
5. `p1_min = p_unhit * (1 - 2d/q)` (H2', per recorded good trial);
   `p_false_max = (n')_4/(N)_4` exactly (≤ p_unhit²; a non-pair
   co-flag needs both pairs unhit); `gamma = p1_min - p_false_max`
   (fail if ≤ 0).
6. `r` = smallest recorded-good count whose best threshold separates
   the families at failure ≤ eps, via exact binomial tails + union
   bound: `min_theta [ w * P[Bin(r,p1_min) <= theta] +
   C(N,2) * P[Bin(r,p_false_max) > theta] ] <= eps`.
7. `T` = smallest raw count with `P[Bin(T, P_record) >= r] >= 1 - eps`.

Each of steps 3, 6, 7 is ≤ eps, so total planning failure ≲ 3·eps.
Numerics: log-space PMFs (Lanczos log-gamma), PMF-recurrence CDFs for
n ≤ 8192, regularized incomplete beta (continued fraction) above —
accurate for w in the hundreds and T up to the 2^24 cap. If P_record
underflows (~≤1e-12) or T would exceed 2^24, planning fails *before any
trial runs* with an explicit infeasibility message.

## Printed chains

### Toy point (q=64, n=30, k=14, w=4, n'=28, (s,d)=(2,4), eps=0.01)

```
[1] p_unhit = 0.673797 (2^-0.57)
[2] t'_max = 4 (k'=8, D=11); J_max = 2
[3] y = 1; P[recorded & bad] = 0.001698; T_reject = 3
    (delta_sneak(1) = 0.100573, reference only)
[4] P_good = 0.394733; P_record = 0.170825 (p_gr = 0.104875)
[5] p1_min = 0.589572; p_false_max = 0.441500; gamma = 0.148072
[6] r = 602 recorded good trials
[7] T = 3836 raw trials        (old manual count: 300)
```

### ID-0 audit (q=1024, n=630, k=470, w=160, N=790, n'=348, (3,5), eps=0.01)

```
[1] p_unhit = 0.193733 (2^-2.37)
[2] t'_max = 35 (k'=28, D=187); J_max = 17        <- matches the paper
[3] y = 0; P[recorded & bad] = 0.000346 (2^-11.50); T_reject = 35
    (delta_sneak(0) = 0.962962, reference only)
[4] P_good = P_record = p_gr = 0.002078 (2^-8.91)
[5] p1_min = 0.191842; p_false_max = 0.037291; gamma = 0.154551
[6] r = 317 recorded good trials
[7] T = 173135 raw trials = 2^17.40              (paper: 2^14.8)
```

**Reconciliation with the paper.** The chains agree on the structural
quantities the paper prints (t'_max = 35, J_max = 17, p_unhit ≈ 0.194,
y = 0). Two factors explain T = 2^17.40 vs the paper's 2^14.8
(total ratio 6.05×):

- **P_good (factor 2.16×).** The binomial model mandated here gives
  P_good = P[Bin(160, 0.1937) ≤ 17] = 2^-8.91. The paper quotes
  2^-7.8. (An earlier independent model check in this project also
  obtained 2.1e-3 = 2^-8.9 from the binomial model, so the gap is in
  the paper's number/model, not in this code: presumably their exact
  count or a slightly different conditioning.) With P_record =
  2^-7.8, step 7 alone would give T ≈ 2^16.3 at the same r.
- **r (factor 2.48×).** The paper's 2^14.8 trials at its P_good imply
  ~128 recorded good trials (2^7.0). Our step 6 requires r = 317: the
  exact-tail union bound at total failure 0.01 over 160 true pairs and
  311,655 non-pairs, using only the H2' worst-case rates
  (p1_min = 0.192, p_false_max = 0.0373), is stricter than the paper's
  threshold condition. (Empirically the true rates are far better
  separated — detection ≈ 1, not 1 - d/q — which is why the paper's
  count works in practice.)
- The 0.99 quantile in step 7 contributes the remaining ~1.13×.

Net: at the paper's own standards of proof, ID-0 Phase 1 costs
~2^2.6 ≈ 6× more trials than v5's headline 2^14.8 — the difference is
accounted for item-by-item above, not hidden.

## End-to-end runs (no `--trials` anywhere)

### Dense — ATTACK SUCCEEDED

```
rlce-attack all --q 64 --n 30 --k 14 --w 4 --nprime 28 --s 2 --d 4 \
    --seed 42 --engine dense --out /tmp/atk_auto_dense
```
Auto plan drove the run: `trials=3836, y_margin=1, T_reject=3,
trials_source=auto` (from `phase1/config.txt`). Recorded 319/3836
trials; top-4 pair counts 92/80/72/61 (all true pairs, next-best
non-pair below 61); verify: 4/4 pairs, 0 false positives, SS code
check pass, round-trip pass → **ATTACK SUCCEEDED** (11m44s wall on
this contended 2-core VM).

Note on the recorded count: 319 is below the model's expected
T * P_record ≈ 655. Two reasons, both in the conservative direction
for the *old* runs and now visible: (a) the model credits u=2 trials
(t'=4) a recording probability P[Bin(4, 1-d/q) <= 3] = 0.2275, but the
true detection rate is ≈ 1 (not the H1 bound 1 - d/q), so those trials
actually record with probability ≈ 0; (b) the Bin(w, p_unhit) lower
tail is heavier than the exact without-replacement pair-occupancy
tail (p_gr = 0.1049 already overstates the exact P[u <= 1]). The
attack still succeeded with 319 recorded trials — fewer than the
r = 602 the worst-case bounds demand — because the empirical
separation (detection ≈ 1) is much stronger than H1/H2' guarantee.

### bw-sigma — ATTACK SUCCEEDED

```
rlce-attack all --q 64 --n 30 --k 14 --w 4 --nprime 28 --s 2 --d 4 \
    --seed 42 --engine bw-sigma --out ~/workspace/rlce_break/bw_auto_out
```
Same auto plan (T=3836, engine-independent), `trials_source=auto`.
Recorded 320/3836 trials; top-4 pair counts 93/80/72/61 — the *same*
four pairs as the dense run ((21,26), (23,30), (3,27), (11,12));
4/4 pairs, 0 false positives, SS pass, round-trip pass →
**ATTACK SUCCEEDED**.

Infrastructure note: this run was killed twice by VM/service
restarts (at ~300 and at 2,024 of 3,836 spooled trials) before
completing on the third launch. That motivated a small addition,
**spool resume**: `run_trials` now skips a trial whose spool file
already exists and is well-formed (per-trial seeds are deterministic,
so reuse is sound for an identical-parameter rerun; torn files from a
kill are detected by format validation and recomputed). The final
launch reused the 2,024 surviving files verbatim (all 2,024 passed
validation) and computed only the remaining 1,812. New unit tests
cover the validator and the resume mechanics; see the Tests section.

## Degenerate point — planning-time infeasibility

q=128, n=65, k=45, w=20, n'=58, (s,d)=(2,3): t'_max = 1. Both
`phase1 --plan-only` and a full `phase1` without `--trials` exit 1
in ~4 ms (no trials run) with:

```
# ERROR: Phase 1 auto-planning failed:
#   T_reject = t'_max - y = 1 - 0 = 1 < 2: the recording filter would
#   never admit a flagged PAIR (|V| >= 2), so the frequency table stays
#   empty and Phase 1 cannot work. Choose (n',s,d) with a larger
#   t'_max (t'_max ~ (n'-1) - d*(k'+w-1)/s).
```

(Previously this parameter set failed only *after* running trials,
with the "0 recorded" error.)

## Tests

- Unit (in `src/phase1.rs`): log-gamma known values; binomial CDF/SF
  known values (Bin(10,½), Bin(20,¼)), edge cases (p ∈ {0,1}, k<0,
  k≥n, large-n beta path at n=100,000), monotonicity; raw-quantile
  minimality; plan at the toy point (y=1, T_reject=3, P_record=0.17082,
  T ∈ [2000, 12000]); plan at the degenerate point (Err); ID-0 audit
  (t'_max=35, J_max=17, y=0, P_good ∈ 2^(-9.3..-8.5), r, T ranges).
  All 9 planning/numerics tests pass, plus 2 spool-resume tests
  (`spooled_trial_valid_checks`, `spool_resume_skips_valid_files`)
  for 11 phase1 tests total.
- Full regression (`cargo test --release --offline`, both binaries),
  run on the final code including spool resume: **rlce-attack 53
  passed / 0 failed / 22 ignored; rlce-test 4 passed / 0 failed.**
  (Baseline before all of this work: 44 passed. +7 planning/numerics
  tests, +2 spool-resume tests; nothing regressed.)
