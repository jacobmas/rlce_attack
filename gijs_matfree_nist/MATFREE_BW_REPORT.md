# Matrix-free GIJS distinguisher, part 2: block Wiedemann

Companion to `MATFREE_REPORT.md` (scalar Wiedemann engine). Everything
below is implemented in `src/matfree.rs` (new code appended; no existing
function modified), in the crate copy `gijs_matfree/`. The original crate
`gijs_rust_v4/` is untouched. Build/test offline, release mode:

```
cargo test  --release --offline --bin rlce-attack           # full suite
cargo test  --release --offline --bin rlce-attack -- --ignored bench_bw_a_random --nocapture
```

Public API added:

- `JetOp::matvec_m_block / matvec_mt_block / apply_a_block` (+
  `apply_a_block_colpar`, a control group), `JetOp::kernel_batch_bw`
- `block_wiedemann_kernel`, `bw_kernel_escalating` (operator-generic core)
- `distinguish_matfree_bw(gf, cols, s, d, tau, block_size, samples)` and
  `distinguish_matfree_bw_with(..., &BWConfig, &mut BWStats)`
- `reuse_bw_flags` / `reuse_bw_flags_pool` (stretch, §5)

Same decision semantics as the scalar engine: `false` is always backed
by an exactly verified separating vector; `true` is heuristic with
false-accept probability ~ q^-samples (samples = 4 by default);
certified combinations pass the same isotropic-radical cancellation and
an exact M z = 0 check (`verify_fail = 0` in every run below).

## 1. Design

**Block Krylov.** Random blocks B0, U (n x b, b = 32 default). Pass 1
computes W_k = A^k B0 for k = 0..2L-1 by block applies and records only
the b x b projections S_k = U^T W_k (the full Krylov sequence is never
stored). L is sized from the Euler-deflated rank estimate
deg_est = min(N_d, E - (points)·C(k+s-2, s-2)) + 1, L = ceil(deg_est/b) + 2,
escalating L <- L + max(L/2, 4) on extraction failure.

**Generator extraction: shifted block-Toeplitz nullspace** (not a full
sigma-basis; see §6 for why that matters at flagship scale). A vector
f = (f_0..f_{L-1}), f_j in F_q^b, with

    sum_j S_{i+j} f_j = 0    for i = 1..L

is a null vector of the L x L block-Toeplitz matrix built from the S_k;
its nullspace is computed by flat forward elimination (suffix updates)
+ back-substitution (~(Lb)^3/3 field ops). Pass 2 regenerates W_j and
forms Z = sum_j W_j f_j for the whole nullspace basis at once
(Z_all = sum_j W_j V_j, V = basis matrix, pool capped at 160 columns).

Two details of the extraction are load-bearing; both were found by
differential testing against the B = M M^T reuse operator (§5), where
the first version failed systematically:

1. **The window starts at i = 1, not i = 0.** The relations must
   constrain A Z, not Z: row block i = 0 imposes U^T Z = 0, and when
   rank(A) + b >= n the functionals {A^i U} span the whole space, so
   Z = 0 is forced identically — however large ker(A) is. Measured on
   B at (k,s,d) = (14,2,4) (rank-bound 448 + b = 480 = n): all 64
   extracted candidates were exactly the zero vector. With the shifted
   window the same instance yields 48 verified kernel vectors on the
   first attempt. (The scalar algorithm has the same shape: mu(A) b = 0
   constrains A z via z = (mu(A)/x) b, never z itself.)
2. **Defect cancellation.** Toeplitz candidates can be near-misses:
   A Z != 0 but tiny (weight 1-3 observed at (12,3,4) on B, out of
   n = 2275). Instead of discarding them, we keep (z_t, d_t = A z_t)
   and take combinations c in ker[d_1 ... d_m]; then
   A (sum c_t z_t) = sum c_t d_t = 0 exactly. Every output is still
   verified A z == 0 with a fresh apply before use.

Soundness sketch for the candidates: the relations give
U^T A^i (A Z) = sum_j S_{i+j+1} f_j = 0 for i < L; when L·b exceeds the
Krylov dimension kappa the functionals {U^T A^i} separate points of the
Krylov space w.h.p., so A Z = 0 w.h.p. — and in any case verification
is exact, so heuristic failure degrades to retries, never wrong output.

**Block applies.** `apply_a_block` = M-block then M^T-block. The
M-block is point-parallel via rayon (per-point row blocks are disjoint);
the M^T-block uses per-thread partial sums XOR-reduced at the end
(exact and order-independent in char 2). The incidence pair list is
streamed once per apply for all b columns, and eval tables are built
once per point per apply (the scalar apply rebuilds them per column).
Measured at bench-A size (b = 32, per A-apply, 10 reps): point-parallel
block 0.435 s, same code on a 1-thread pool 0.537 s, column-parallel
control (32 independent scalar applies, rayon over columns) 0.594 s.
So the shared traversal buys 1.37x over per-column applies, and rayon
(point-parallel) only a further 1.23x over single-thread on this
2-core VM — the M^T reduction is bandwidth-bound.

## 2. Correctness

All differential tests use the dense `gijs::distinguish` as oracle,
same parameter sets as the scalar report:

| suite | cases | decisions | mismatches |
|---|---|---|---|
| random codes | (64,36,14,2,4), (64,41,19,2,3), (64,30,12,3,4), (128,36,14,2,4) | 32 | 0 |
| pure GRS | (64,40,14,2,4), (128,41,19,2,3), (64,30,12,3,4) | 24 | 0 |
| RLCE-shortened | (128: 44/30/8 -> 41,19,(2,3)), (128: 34/20/4 -> 32,14,(2,4)), (64: 26/16/3 -> 25,12,(3,4)) | 24 | 0 |
| small block b=8 | (64,41,19,2,3) | 8 | 0 |
| **total** | | **88** | **0** |

`verify_fail = 0` in every aggregate. Runs per decision: 136 block
runs / 88 decisions ≈ 1.5 overall; most decisions take exactly 1 run.
Two cases are degenerate and worth flagging: at (k,s,d) = (12,3,4)
(random n=30 and RLCE n'=25), E_tau = 2,639 > N_d = 1,365, M_tau has
full column rank and A = M_tau^T M_tau is invertible, so there is no
kernel to extract (32/32 extraction runs fail, correctly) and both
engines fall back to heuristic accept — matching the dense oracle,
which is trivially "structured" there (ker(M_tau) = {0} makes its
test vacuous). Those parameter sets sit outside the GIJS regime
(condition (2) fails); they are inherited from the scalar suite.
Unit tests: block matvecs
match the scalar matvecs entry-for-entry (M, M^T, A, and the colpar
control) at three (k,s,d); planted dense operators — rank-deficient
A = X·Y (48x48, rank <= 20) returns >= 6 independent exactly-verified
kernel vectors, the zero operator returns vectors, a full-rank random
operator never returns a nonzero "kernel" vector. Public-wrapper smoke
test (`distinguish_matfree_bw` with the exact deliverable signature)
passes. Full crate suite: **34 passed, 0 failed, 13 ignored** on
rlce-attack (all pre-existing tests plus 9 new block-Wiedemann tests
and 5 new ignored benchmarks) and **4 passed, 0 failed** on rlce-test.

## 3. Benchmarks

Machine note: same 2-core VM as the scalar report, but contended ~2x
during this session (dense bench A re-measured at 21.9 s vs 8.7 s in
the scalar session; scalar random 331 s vs 169 s). Ratios below are
same-session wherever possible. Single runs except where noted;
peak RSS = VmHWM of a fresh process per measurement.

**Bench A** — q=256, n'=32, k'=12, (s,d)=(3,5), tau=0; dense M_tau
2,821 x 4,368 = 24.6 MB:

| engine | random (reject) | GRS (accept) | peak RSS |
|---|---|---|---|
| dense (this session) | 21.9 s | 14.5 s | 27 MB |
| scalar matfree (this session) | 331.4 s | 505.3 s | ~5 MB |
| scalar matfree (scalar-report session) | 168.7 s | 334.8 s | ~5 MB |
| **block matfree (this session)** | **156.2 s** | **138.3 s** | **30.6 / 33.3 MB** |

Same-session speedups of block over scalar: **2.1x (reject path),
3.7x (accept path)**. Block is still ~7-10x slower than dense where
dense fits. Phase split (block, random): Krylov pass 1 68.9 s
(155 applies @ 0.44 s), Toeplitz solve 46.5 s, extraction pass 39.8 s
(77 applies + 79 candidate verifications + accumulation); one run,
nullity 79. GRS: 58.5 / 43.7 / 35.5 s, nullity 408, **44 certified
samples from the single batch** (the decision needs 4) — this
amortization is exactly where the accept-path win comes from; the
scalar engine pays a full ~2·deg(mu) extraction per sample (4 attempts
for 4 samples in its GRS run above).

**Larger point** — q=256, n'=40, k'=13, (3,5), tau=0, random instance
(seed 987654, chosen for this report; the scalar report recorded no
seed): **block finishes a decision in 290.4 s** (correct, 1 run;
Krylov 130.9 s, Toeplitz 99.6 s, extract 58.6 s; RSS 58.2 MB). The
scalar engine did not finish even one sample here in ~7 min of CPU
(scalar report). Dense is still faster at this size (44.6 s, 51.8 MB,
scalar-report session).

**Paper scale** — q=1024, n'=348, k'=28, (3,5), single hold-out
(dense M_tau would be 150,945 x 201,376 = 61 GB):

- JetOp build + first (warm) block apply: 444 s total setup.
- Block A-apply at b=32: **140.2 s** (median of 2), i.e. 4.4 s per
  column vs ~4.6 s for one scalar A-apply in the scalar-report session
  (~2x better per-column throughput at matched contention, plus 32
  samples per run instead of 1).
- Peak RSS **120.4 MB** (~500x under the dense matrix alone; scalar
  engine: 94 MB).
- Extrapolation (formula shown): deg_est = E_tau - 347·C(29,1) + 1
  = 150,945 - 10,063 + 1 = 140,883; L = ceil(deg_est/32) + 2 = 4,405;
  applies per block run ~= 3L = 13,215; apply cost = 13,215 x 140.2 s
  ~= **21.4 days per block run** at this session's contention
  (~11 days at the scalar session's). A scalar decision was
  extrapolated at ~96 days (scalar session). So the apply side improves
  ~4-9x — **but the decision is still infeasible with this
  implementation**: the dense Toeplitz solve at Lb = 140,960 would need
  a ~40 GB matrix and ~(141k)^3/3 ~= 9.3e14 field ops. The code caps
  Toeplitz width at 6,144 and declines. A sigma-basis / minimal matrix
  generator (O(L^2 b^3) time, O(L b^2) memory) is the missing piece for
  flagship scale; it is not implemented here.

## 4. Where block Wiedemann did and didn't pay off

Paid off:
- **Accept path (3.7x measured).** One block run yields up to ~b
  certified kernel samples (44 observed) after one radical
  cancellation; the scalar engine re-runs its whole extraction per
  sample. Any use needing several samples per hold-out (the attack
  needs samples=4) inherits this directly.
- **Mid-scale reach.** The (40,13,3,5) point completes (290 s) where
  scalar stalled; per-column apply throughput is ~1.4-2x better
  (shared pair traversal + shared eval tables + rayon).
- **The reuse composition works when J is small** (exact at
  (14,2,4), J = 15) but is **blocked at (12,3,4)** by a Krylov
  reachability ceiling — see §5 for the measured blocker analysis.

Didn't pay off:
- **Reject path (only 2.1x).** A random hold-out is rejected by the
  FIRST certified separator; the scalar engine exits after 1-2 samples
  (~2-4·deg(mu) applies), while the block engine pays the full
  3L block applies + Toeplitz + extraction regardless.
- **The dense Toeplitz extraction is the new bottleneck.** It is
  30-45% of decision time already at bench A (Lb = 2,496) and grows as
  (Lb)^3 while Krylov work grows as L·b·(apply cost per column): past
  kappa ~ 3,000 the solve dominates, and at flagship (Lb = 140,960) it
  is simply impossible. Block Wiedemann's asymptotic win is real only
  when paired with a quasi-linear generator computation
  (sigma-basis), which this build does not have.
- **Memory:** block RSS (31-58 MB at bench sizes) is worse than scalar
  (~5 MB) — Toeplitz matrix + Z_all pool — though still at/below dense
  RSS and irrelevant at flagship (120 MB vs 61 GB dense).

## 5. Stretch: trial-level reuse composition — works at J = 15, blocked at J = 91 (measured)

`reuse_bw_flags(gf, cols, s, d, block_size, seed)` decides all n'
hold-outs from one left-kernel sampling over M_full. Criterion (dual
of the dense reuse scheme): tau is flagged iff
e_0 not in span{ u_tau : u in ker(M_full^T) }, since the achievable
syndrome space S_tau equals (span{u_tau})^perp. Left-kernel vectors
are sampled by block Wiedemann on B = M_full M_full^T with the same
radical cancellation (roles of M, M^T swapped; samples z in ker(B),
y = M^T z, combinations over ker[y_1..y_m], each certified u verified
M^T u = 0 exactly).

**The criterion is correct** — verified against a dense ground-truth
ker(M^T) at (12,3,4), n' = 25: dim ker(M^T) = 910 (rank(M) = 1365 =
N_d, e_full = 2,275), and with the FULL kernel the restriction rank
is 91 = J at every tau tested, with e_0 in span exactly at the taus
the dense oracle calls structured.

**Where it works:** at (14,2,4), n' = 32 (J = 15), the composition
agrees with the per-tau oracle on **all 32 taus, 0 mismatches**
(test `bw_reuse_composition_rlce`).

**Where it is blocked:** at (12,3,4), J = 91 > 64, it fails
completely — all 25 taus over-flagged (test
`bw_reuse_case1_krylov_ceiling` pins the signature). The obstruction
is a hard reachability ceiling of Krylov sampling on B, measured
directly:

- Per block-Wiedemann run, the extracted kernel batch has rank
  exactly b (= 32): one start block reaches only ker(B) intersected
  with its own block-Krylov hull.
- The union over runs **saturates at rank 64**: run 1 adds 32, run 2
  reaches 64, runs 3-14 add nothing (672 certified samples at b = 32,
  1,120 at b = 64 — the ceiling is 64 at both block sizes). All
  sampled kernel vectors lie in a fixed 64-dimensional subspace
  R of the 910-dimensional ker(M^T).
- Restrictions of R have rank 64 at every tau (R is "generic" — the
  restrictions preserve its rank), so the span test caps at
  64 < 91 = J and e_0 is never in the sampled span.

So the reachable kernel R satisfies dim R = 64 << dim ker(M^T) = 910.
For scale: B's block-Krylov hull (Toeplitz rank ~1,397 of 2,275)
captures almost all of the 1,365-dimensional im(B) but only 64
dimensions of ker(B) — iteration generates rich structure on the
invertible part, while on ker(B) (where B acts as 0) the hull only
ever sees the start blocks' shadows, and the extraction relations
tie the reachable part to a fixed thin subspace. This is not the
isotropic radical: dim ker(M^T M) = 0 at this instance (measured
densely), i.e. B has index 1 and ker(B) ∩ im(B) = {0}. (The exact
invariant-theoretic identity of the 64 was not pinned down; the
ceiling itself is measured at two block sizes and 14 restarts.)

Consequences: the composition is sound exactly when J <= dim R
(here: J <= 64) — more sampling, larger blocks, and restarts do not
help past the ceiling. The per-tau engine is unaffected: it needs
only a handful of kernel vectors of A = M_tau^T M_tau, and those
live in the reachable part (that is precisely the part both
Wiedemann engines exploit). A trial-level matrix-free reuse at
J ~ 91+ would need a left-kernel sampler that reaches outside the
Krylov hull (e.g. dense or Lanczos-style orthogonalization), which
defeats the matrix-free purpose at these sizes; the dense reuse
scheme in `gijs_rust_v4` remains the right tool there.

## 6. Limitations / what is not here

- No sigma-basis: generator extraction is the dense Toeplitz solve,
  capped at Lb <= 6,144. Flagship-scale decisions remain infeasible
  (§3); the apply engine and memory profile are flagship-ready
  (120 MB), the extraction step is not.
- Block size fixed at b = 32 by default (parameter). Larger b shortens
  L but fattens the Toeplitz (Lb ~ kappa is invariant) and the block
  applies; b in 16-64 all pass the differential suites (b=8 tested).
- Benchmarks are single runs on a contended 2-core VM (±2x); ratios
  quoted same-session. Bench timings include the dense oracle check
  inside the test process only where stated (decision times exclude
  it).
