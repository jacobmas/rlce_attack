# Fast Phase 2 (direction B) — implementation report

Date: 2026-10-21. Crate: `gijs_matfree` (the `gijs_rust_v4` crate is
untouched). New code: `src/fastss.rs` (~700 lines incl. tests), plus
wiring in `src/attack.rs` (`--phase2 fast|scan`, default `fast`).

All timings below are from the shared 2-vCPU build VM under
contention (the regression suite and other jobs were running
concurrently); treat absolute times as upper bounds. The user-facing
comparison numbers (per-candidate scan cost 7.8 ms at (256,165)) are
from the 2026-10-21 measurement pass.

## Design as implemented

Setting: Phase 1 produced exactly k pure columns and the pairing;
`need = (k+3) - n_pure = 3` pairs must be unmixed. Old path: scan
F_q^3, full SS per candidate.

1. **Normalization (once).** P = k×k matrix of pure columns;
   P^{-1} by Gauss-Jordan (`mat::matinv`). Each pair (c_i, d_i) maps
   to a_i = P^{-1} c_i, b_i = P^{-1} d_i; candidates are
   g_i(l_i) = a_i + l_i b_i and the candidate generator is
   [ I_k | g_1 g_2 g_3 ].
2. **Dual test (direction A, stage 3 / standalone `dual_test`).**
   The dual is H = [ -g_1^T; -g_2^T; -g_3^T | I_3 ] (char 2: sign-free).
   With gamma_j = (g_1[j], g_2[j], g_3[j]) and
   m_j = (g1g2, g1g3, g2g3) at pure position j, M* = [m_j] (3×k):
   candidate is GRS iff rank M* <= 2 AND (at rank 2) the unique left
   relation c (c·M* = 0) has c12·c13·c23 != 0 (smooth conic) AND all
   gamma_j are nonzero and pairwise non-proportional and not
   proportional to e_1, e_2, e_3 (distinct support). Rank is by
   genuine Gaussian elimination (`mat::rank`) — never a Gram matrix
   (char-2 isotropy would create false positives). Rank < 2 passes
   through to stage 4 (SS decides); distinctness is enforced in all
   cases since it is necessary for any GRS code.
3. **Stage 1 (root-finding).** J = {0,1,2}, J' = {3,4,5} (pure
   coordinate positions). Delta_J(l) = det of the 3×3 minor of M* on
   J has multidegree <= (2,2,2). At fixed (l1,l2) its coefficients
   (A,B,C) in l3 come from four 3×3 determinants (~150 field ops,
   O(1) in k): with e = R_12 row (l3-free), R_13 = p + l3 q,
   R_23 = r + l3 s: A = det[e;q;s], B = det[e;q;r] + det[e;p;s],
   C = det[e;p;r]. The char-2 quadratic is solved in closed form
   (`solve_quad`): A=0 -> linear; A=B=0 -> All/None by C; B=0 ->
   unique root by Frobenius sqrt x -> x^{2^{m-1}}; else substitute
   x = (B/A) z, giving z^2 + z = AC/B^2, solved via a precomputed
   table (`QuadSolver`, O(q) build) — solvable iff the trace
   condition holds, roots z and z+1. Parallel over l1 (rayon).
4. **Stage 2 (filter).** Each root is checked against Delta_J'
   (second, disjoint triple). ~q survivors expected.
5. **Stage 4 (confirm).** For each stage-3 survivor, candidate
   columns are rebuilt in ORIGINAL coordinates and the existing
   `ss_attack` runs under `catch_unwind`; first success wins. Stages
   1-3 only generate candidates; acceptance is exactly the old
   path's test.
6. **lambda = infinity / degenerate pairs.** If an orientation
   yields no verified candidate, the pipeline retries with pair
   columns swapped per pair — all 8 flip masks. Flipping pair i maps
   the infinite direction (true column = second-listed column alone,
   i.e. GL(2) coefficient d = 0) to l_new = 0. Only after all 8
   orientations fail does `do_ss` fall back to the scan.
7. **Degenerate cells** (A = B = C = 0 at some (l1,l2)): Delta_J is
   identically zero in l3 there, so no root-finding is possible; the
   cell's l3 values are scanned directly against Delta_J' and pass
   to the exact stage 3. Cells are counted in the stats. If more
   than half of orientation 0's cells are degenerate, J and J' swap
   roles for the remaining orientations (observed count at real
   points: 4 cells out of q^2 — the swap never triggered).

CLI: `ss` and `all` accept `--phase2 fast|scan` (default fast). The
choice is printed in the `all` header, recorded in `ss/config.txt`,
and the fast path prints per-stage survivor counts and timings.
`--phase2 fast` with need != 3 prints a note and uses the scan. The
scan itself was extracted verbatim into `bootstrap_scan` (same
enumeration, same messages) and remains reachable byte-for-byte via
`--phase2 scan`; it is also the automatic fallback.

## Measurements at the real target (q=256, n=198, k=165, w=33)

Phase-2-only integration test (`fast_bootstrap_target_256_165_33`):
keygen + ground-truth Phase 1 output (pure set + pairing from
key.meta, the existing mocked-Phase-1 pattern), then only the fast
Phase 2. VM, contended:

| stage | time | counts |
|---|---|---|
| prep (P^{-1} + pair transforms + solver table) | 0.011 s | |
| stages 1+2 (65,536 cells) | 0.008 s | 68,406 roots; 4 degenerate cells; 270 stage-2 survivors |
| stage 3 (exact dual-rank) | 0.001 s | 1 survivor |
| stage 4 (SS confirm) | 0.012 s | 1 SS attempt |
| **total** | **0.03 s** | orientations tried: 1 |

Recovered (alpha, y) verified against the pure columns (stacked
rank = k). For comparison, the scan at this point: ~7.8 ms per
candidate measured on the real enumeration path, ~2^23 candidates
expected to the true tuple (~55 min on 20 threads; full q^3 scan
~36 h single-threaded). Concrete operation counts for the fast path
match the design estimate (~2×10^7 field ops): 65,536 cells × ~150
ops ≈ 1.0×10^7 for stage 1, ~5×10^5 for stage 3, ~5×10^6 for the
one-time inversion.

At the small w=n-k point (q=64, n=20, k=14, w=6), binary-level run
(`keygen` -> synthetic phase1 -> `ss` -> `unmix` -> `verify`):
fast mode 0.008 s wall (cells 4,096; roots 4,758; degenerate cells 4;
stage-2 survivors 79; stage-3 survivors 1; SS attempts 1), then
unmix 6/6 and verify **ATTACK SUCCEEDED**. The same point with
`--phase2 scan`: 120.6 s on this VM.

## Correctness evidence

Unit tests (all in `src/fastss.rs`, release mode):

- `quad_solver_cross_check`: all `solve_quad` branches — A=0/B!=0
  (50 random), A=B=0 (All and None), B=0 Frobenius root (50 random),
  trace-0 two-root and trace-1 no-root cases, plus 400 random
  (A,B,C) triples cross-checked against brute-force root sets over
  GF(64): exact agreement.
- `delta_coeffs_vs_brute_force`: (A,B,C) from `quad_coeffs` vs
  direct `delta_at` evaluation at 300 random (l1,l2) × 6 random l3
  on a random scrambled-GRS instance (k=9): exact agreement.
- `dual_test_vs_ss`: at k=8, q=64 — true triple accepted by
  `dual_test` and by SS; 40 wrong-lambda triples (one coordinate
  perturbed): `dual_test` and `ss_attack` agreed on every one (all
  rejected). Exact agreement on this sample; note the sample is
  necessarily small (SS is the expensive side).
- `degenerate_pair_flip`: synthetic instance (k=10) whose pair 1 has
  GL(2) coefficient d = 0 (true unmixing is the second column alone,
  l = infinity). Orientation 0 alone provably fails (asserted);
  `fast_bootstrap` succeeds via a flip (orientations_tried >= 2
  asserted) and the result verifies (stacked rank = k).
- `fast_bootstrap_w_eq_n_minus_k`: the existing mocked point
  (20,14,6) q=64 seed 777 — succeeds, stage-3 survivors 1, SS
  attempts 1, verifies.
- `fast_bootstrap_target_256_165_33`: as above.

Differential test (`bootstrap_tests::fast_vs_scan_differential` in
attack.rs): three random w=n-k instances (q=32, n=14, k=10, w=4,
seeds 11/222/3333) — the extracted `bootstrap_scan` (the exact CLI
scan helper) and `fast_bootstrap` both succeed and both results
verify against the pure columns. The pre-existing serial-scan test
`bootstrap_w_eq_n_minus_k` still passes unchanged.

End-to-end: binary path at (20,14,6) ATTACK SUCCEEDED (above).
Note on Phase 1 at w=n-k points: no (n', s, d) satisfies the GIJS
inequalities at (20,14,6) at all (the documented three-way squeeze),
so a genuine-Phase-1 end-to-end run at a w=n-k point is not
possible at small scale on any hardware — the mocked-Phase-1 pattern
above is the same one the pre-existing test suite uses. A full
`all` run with genuine Phase 1 at the known valid toy point
(q=64, n=30, k=14, w=4, auto trials, --phase2 fast) completed with
**ATTACK SUCCEEDED** in 10m26s on this VM (Phase 1 recorded
319/3836 trials, exactly reproducing the deterministic counts from
the auto-trials delivery; 26 pure columns, so SS took the direct
branch — this run exercises the `all` -> `do_ss` wiring and the
`--phase2` plumbing, not the bootstrap itself).

Full regression: **rlce-attack 60 passed / 0 failed / 22 ignored;
rlce-test 4 passed / 0 failed** (baseline before this change:
53 / 0 / 22 and 4 / 0; the +7 are the 6 fastss tests and the
differential test). Suite wall time 1,495 s on this VM — the bulk is
pre-existing Phase-1 end-to-end tests, not the new code (all fastss
tests together: < 1 s).

## Honest gaps

- Stage 1/2 completeness is generic-position: if both J and J'
  minors were identically zero as polynomials (they are not, at any
  tested point — degenerate cells are per-(l1,l2), counted, and
  handled exactly), candidate generation could in principle miss
  the true triple. The automatic scan fallback bounds the damage to
  wasted time, never to a wrong answer: acceptance is always SS on
  original coordinates.
- A wrong-but-GRS stage-3 survivor (a different GRS code through the
  same pure columns + 3 wrong unmixings) would be accepted by stage 4
  exactly as the scan accepts it; this is inherent to the attack
  (any such frame completes the key recovery), not a fast-path
  artifact.
- `dual_test`'s rank<2 pass-through is deliberate (deferred to SS);
  at rank exactly 2 the smoothness + distinctness conditions are
  exact by the dual-square characterization.
- The degenerate-cell count of 4 appeared at both tested points
  (q=64 and q=256); the cause was not isolated (plausibly cells
  where a candidate coordinate row of M* vanishes on J). The
  handling does not depend on the cause.
- No Phase-2-only benchmark of the scan at (256,165) was run (it is
  the ~36 h figure from the earlier measurement pass, not re-run).
