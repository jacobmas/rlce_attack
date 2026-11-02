# Matrix-free block Wiedemann with sigma-basis extraction

Implements `distinguish_matfree_bw_sigma`, a drop-in replacement for the
block-Wiedemann GIJS distinguisher `distinguish_matfree_bw` in which the
dense block-Toeplitz nullspace extraction is replaced by a sigma-basis
(minimal approximant basis) computation. The (L*b)^2 block-Toeplitz
matrix -- 39.7 GB at flagship (k'=28, s=3, d=5) dims, L*b = 140,960 --
is never formed. The full attack binary (`rlce-attack`) can run Phase 1
with this engine end to end (`--engine bw-sigma`).

All timings in this report are from a contended 2-core VM (the same
conditions flagged in MATFREE_REPORT.md / MATFREE_BW_REPORT.md, ~2x
noise); peak RSS is VmHWM of a fresh process per measurement unless
stated otherwise. Absolute times will be much better on the user's
20-thread machine; the memory numbers are the point.

## What changed

- New module `src/sigma.rs`: minimal approximant basis for the augmented
  block series (below), O(sigma * b^2) storage.
- New engine in `src/matfree.rs` (Toeplitz and scalar paths untouched):
  `block_wiedemann_kernel_sigma`, `bw_kernel_escalating_sigma`,
  `JetOp::kernel_batch_bw_sigma`, `distinguish_matfree_bw_sigma_with`,
  `distinguish_matfree_bw_sigma`. Same contract as the BW engine:
  `false` (flag) is always certified by an exactly verified kernel
  vector; `true` (accept) is heuristic with false-accept probability
  ~q^-samples. The char-2 isotropic-radical defect cancellation
  (`cancel_and_test_bw`) is shared unchanged.
- `BWConfig` is reused as-is; per-decision seeding differs from the
  Toeplitz engine by one XOR constant ("Sigma_v1") so the two engines
  are decorrelated.
- Attack wiring: `phase1::DistEngine { Dense, BwSigma }`,
  `--engine dense|bw-sigma` on the `phase1` and `all` subcommands
  (default `dense`), recorded in `phase1/config.txt`.

## The algorithm, as implemented

Pass 1 is byte-for-byte the Toeplitz engine's: with random blocks
B0, U (n x b), record the Krylov sequence S_k = U^T A^k B0, k = 0..2L-1,
for A = M_tau^T M_tau (matrix-free JetOp applies). Only the extraction
differs.

Let Scr(x) = sum_k S_k x^k. A pair of polynomial vectors (F, G) with
Scr * F = G (coefficientwise, modulo x^{2L}) gives kernel vectors from
the Krylov blocks W_p = A^p B0:

- if deg G <= deg F =: delta, then Z = sum_p W_p F_{delta-p} is in
  ker(A), because U^T A^i (A Z) = (Scr F)_{i+delta+1} = G_{i+delta+1} = 0
  for all i >= 0;
- if deg G = delta + 1, the shifted combination
  AZ = sum_p W_p F_{delta+1-p} is in ker(A), by the same identity read
  one index later. This case is not exotic: on GRS instances ALL 32
  generator rows of the basis have deg G = deg F + 1 (measured, see the
  head-to-head test). Both combinations are emitted per basis row and
  exact verification (one extra block apply) discards whichever does
  not apply.

The pairs (F, G) come from a minimal approximant basis of the augmented
series M(x) = [Scr^T(x); I] (2b x b), computed by `sigma_basis_aug`:
the M-Basis iteration of Beckermann-Labahn -- the base case of the
Giorgi-Jeannerod-Villard PM-Basis divide-and-conquer -- run to order
sigma = 2L with shift s = (0,...,0, 1,...,1) (0 on the Scr^T-block
columns of the basis, 1 on the I-block columns). The basis P (2b x 2b
polynomial matrix) satisfies P * M == 0 (mod x^sigma) row by row; a row
(f, g) transposes to f * Scr^T + g == 0, i.e. (F, G) = (f^T, g^T) is a
pair for Scr. The iteration maintains the residual R = P * M with
per-row offsets (only coefficients >= the current order are stored),
selects pivots greedily in shifted-degree order, and applies the
degree step to pivot rows. Storage: the series (sigma * b^2), the
residual (2b * sigma * b), and the basis coefficients
((sigma+1) * (2b)^2 worst case) -- all linear in sigma at fixed b.
Unlike the Toeplitz path there is no BW_MAX_TOEPLITZ_WIDTH cap; the
escalation budget is BW_MAX_SIGMA_TERMS = 2^15 sequence terms, a
storage budget (~200 MB worst case at b=32; flagship needs 8,810).

Honest complexity note: the M-Basis iteration is O(sigma^2 * b^3)
field operations worst case (vs near-O(sigma log sigma) for full
PM-Basis divide-and-conquer); what this deliverable guarantees and
measures is the MEMORY scaling. At the sizes where a decision is
feasible at all here (L <= ~120), extraction time was 1.6-4.1 s
against 14-38 s of Krylov applies -- the extraction is not the
bottleneck. At flagship sigma = 8,810 on a structured series the
basis alone took 28.0 s (see memory table); a generic flagship
sequence would take far longer, and a full flagship decision remains
infeasible for the Krylov reason already documented (13,215 applies),
not for an extraction reason.

## Correctness

### Sigma-basis unit tests (src/sigma.rs)

- Planted generators at b in {1,2,3}, degrees {2,3}: series built from
  a known (Lambda, G) pair; every basis row verified to be an exact
  approximant by direct convolution; the planted rows verified to lie
  in the returned basis's row module (rank-consistency over GF(2^m));
  Padé rows (deg G < deg F <= planted degree + 1) present.
- Zero series: the trivial basis is returned (rows with F = I).
- Random series: all rows exact approximants.

### Head-to-head vs the Toeplitz extraction (exact, same sequence)

On operator-generated Krylov sequences (same seed => same B0, U,
applies), every sigma candidate combination, reversed appropriately
(f_j = F_{delta-j}, resp. the shifted reversal x*f), was multiplied
against the explicitly built (L*b) x (L*b) Toeplitz matrix:

- random (41,19,2,3), L=26: Toeplitz nullity 74; sigma candidates
  checked: 32 plain + 32 shifted, ALL exactly in the nullspace.
- GRS (30,12,3,4), L=45: Toeplitz nullity 203; sigma candidates
  checked: 0 plain + 32 shifted, ALL exactly in the nullspace (the
  generator family here is entirely of the deg G = deg F + 1 kind).

### Differential decisions vs the dense oracle (same suites as BW)

`distinguish_matfree_bw_sigma_with` vs `gijs::distinguish`, b=32,
8 taus per case (the (12,3,4) cases are degenerate -- A invertible on
random instances -- and both engines decide them heuristically; the
Toeplitz engine's behavior there is documented in MATFREE_BW_REPORT):

| suite | cases | decisions | mismatches | certified (agg. verified) |
|---|---|---|---|---|
| random codes | (36,14,2,4), (41,19,2,3), (30,12,3,4), (36,14,2,4)@q128 | 32 | 0 | 9 / 8 / 0* / 8 |
| pure GRS | (40,14,2,4), (41,19,2,3)@q128, (30,12,3,4) | 24 | 0 | 256 / 256 / 256 |
| RLCE shortened | (44,30,8)->41, (34,20,4)->32, (26,16,3)->25 | 24 | 0 | 225 / 8 / 0* |
| small block b=8 | (41,19,2,3) | 8 | 0 | 8 |

Total: 88 decisions, 0 mismatches, verify_fail = 0 everywhere.
(* = the degenerate (12,3,4) cases: extraction correctly finds no
kernel because A is invertible; decisions fall through heuristically,
matching the oracle and the Toeplitz engine's decisions.)

Planted-operator tests (dense A = X*Y of rank 20 in 48 dims, and a
random invertible 40x40) pass through `bw_kernel_escalating_sigma`:
>= 3 independent exactly-verified kernel vectors for the
rank-deficient operator; no nonzero vector for the invertible one.

### End-to-end attack (deliverable 5)

```
./target/release/rlce-attack all --q 64 --n 30 --k 14 --w 4 \
  --nprime 28 --s 2 --d 4 --trials 300 --threads 0 --seed 42 \
  --engine bw-sigma --out /tmp/attack_sigma/
```
Result: **ATTACK SUCCEEDED** -- 107/300 trials recorded, 4/4
non-degenerate twin pairs found, 0 false positives, SS recovered the
correct GRS code, encrypt/decrypt round-trip passed. Wall 726 s on
this contended 2-core VM. A standalone 30-trial Phase 1 with the same
seed measured **peak RSS 11.5 MB** (the dense engine measures the
same 11.5 MB at this toy scale -- both are dominated by the binary and
key; the engines' memory only separates at scale, see below), and its
outputs (all 30 trial spools, pairs.txt, pure.txt, col_flagged.txt)
are **byte-identical** to the dense engine's on the same key/seed.
Dense Phase 1 took 4.3 s for the same 30 trials vs 73.4 s for
bw-sigma: at toy scale the matrix-free engine is ~17x slower; it is
the memory-feasibility engine, not the speed engine.

## Memory and time benchmarks

### Sigma-basis storage scaling (b=32, GF(1024), structured series T_k = D^k)

| sigma (terms) | L | basis time | peak RSS | Toeplitz matrix at same L |
|---|---|---|---|---|
| 2,200 | 1,100 | 1.8 s | 33.8 MB | 2.5 GB |
| 4,400 | 2,200 | 7.1 s | 63.9 MB | 9.9 GB |
| 8,810 | 4,405 | 28.0 s | 124.7 MB | 39.7 GB |

RSS grows ~linearly in sigma (33.8 -> 63.9 -> 124.7 MB), as designed:
series + residual + basis coefficients, all O(sigma * b^2) field
elements. (Time grows ~quadratically here: 1.8 -> 7.1 -> 28.0 s --
the M-Basis iteration's honest cost on this series; see the
complexity note.) On generic random series the basis took 0.1 s at
sigma=256 and 0.3 s at sigma=512 (RSS 7.0 / 10.5 MB).

### Decisions (fresh process each; both engines measured in the same session)

| point | engine | decision | time | peak RSS |
|---|---|---|---|---|
| bench A random (32,12,3,5) q=256 | sigma | false = oracle | 25.2 s | 29.4 MB |
| bench A random | Toeplitz BW | false = oracle | 31.8 s | 30.9 MB |
| bench A GRS (32,12,3,5) q=256 | sigma | true = oracle | 22.6 s | 32.4 MB |
| bench A GRS | Toeplitz BW | true = oracle | 34.7 s | 33.4 MB |
| larger (40,13,3,5) q=256 | sigma | false = oracle | 62.2 s | 56.9 MB |
| larger (40,13,3,5) | Toeplitz BW | false = oracle | 86.7 s | 58.2 MB |

At these mid sizes the two engines' RSS is close (dominated by JetOp
tables and Krylov blocks; the Toeplitz matrix at L*b = 2,496 is only
12.5 MB) -- the memory separation is in L, see the scaling table. The
sigma engine is modestly faster in this session because its
extraction (1.9 s bench A random, 1.6 s GRS, 4.1 s larger-40) beats
the dense Toeplitz nullspace (9.2 s / 10.4 s / 28.1 s); Krylov applies
(14-38 s) and candidate assembly (7-20 s) are shared costs. Phase
breakdown, sigma bench A random: Krylov 15.8 s, sigma-basis 1.9 s,
assembly/verification 7.3 s. (MATFREE_BW_REPORT's session measured
156 s / 138 s / 290 s for the Toeplitz rows under heavier contention;
the same-session reruns above supersede those for comparison.)

### Flagship dims (k'=28, (3,5), q=1024, n'=348)

One `block_wiedemann_kernel_sigma` call at the flagship operator
size with forced L=4 (bench_sigma_flagship_prefix, run alone, stats
read from the process output file): **time 1091.2 s, peak RSS
306.1 MB**, outcome Failed (expected: the true generator degree is
~4,405, so no L=4 candidate can verify). Breakdown: 12 block applies
(7 pass-1 + 5 pass-2, dmax = 5 as designed), Krylov 479.5 s
(~68 s/apply this run; the same-session Toeplitz flagship bench
measured 145 s/apply under heavier contention), sigma extraction at
sigma=8 terms 0.006 s, pass-2 assembly + verification +
cancellation 603.6 s. The 306 MB peak is the whole pipeline resident
at flagship dims -- JetOp tables (~2.35M incidence pairs), Krylov
blocks, the 237,336 x 64 candidate matrix, and, at this forced tiny
L, a 237,336 x 64 defect-cancellation system (at L=4 all 64
candidates defect; at a realistic L the verified/defected split is
concentrated and this term shrinks). Compare: 61 GB for the dense
Hasse matrix alone, 39.7 GB for the full-L Toeplitz matrix. A full
flagship decision needs ~13,215 applies (~10-22 days on this 2-core
VM at the measured per-apply costs) plus the sigma basis at
sigma=8,810 (28.0 s on a structured series; more on a generic one).
So a flagship decision remains infeasible here -- but now for Krylov
TIME only, not memory: the extraction state at full L is ~125 MB
(measured scaling table), not ~40 GB.

## Regression

Baseline before this change (measured): rlce-attack 34 passed /
0 failed / 13 ignored; rlce-test 4 passed. After: **rlce-attack
44 passed / 0 failed / 22 ignored** (the +10 are the new sigma tests,
the +9 ignored are the new sigma benches); **rlce-test 4 passed /
0 failed**. Scalar and Toeplitz BW paths unmodified in behavior.

## Limits

- Accepts remain heuristic (false-accept ~q^-samples), exactly as for
  the scalar and Toeplitz matrix-free engines; rejections are exact.
- M-Basis is quadratic-time in sigma in the worst case; a
  divide-and-conquer PM-Basis would be needed for asymptotic time at
  flagship sigma. Memory -- the point of this change -- is linear.
- The reuse composition (shared left-kernel across taus) is
  unaffected and remains subject to the Krylov-ceiling caveat of
  MATFREE_BW_REPORT; it is not wired to the sigma engine.
- BW_MAX_SIGMA_TERMS caps the sequence at 32,768 terms; beyond that
  the escalating wrapper returns Failed (the decision layer then
  treats the tau heuristically). Flagship (3,5) needs 8,810 terms.
