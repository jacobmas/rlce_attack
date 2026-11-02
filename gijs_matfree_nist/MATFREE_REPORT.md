# Matrix-free GIJS distinguisher — report

Implementation: `src/matfree.rs` in this crate copy (`gijs_matfree`), wired
into the `rlce-attack` binary as `mod matfree`. Public entry point:
`pub fn distinguish_matfree(gf, cols, s, d, tau) -> bool`, same contract as
`gijs::distinguish` (true = "GRS" / v_tau in rowspan(M_tau)).

All measurements below: release build, this VM (shared/contended, load
avg ~5-7 during the runs; treat times as ±2x), single-threaded code paths.

## Design

**Implicit operator (`JetOp`).** For fixed (k, s, d) we precompute once:
the N_d degree-d monomials, the J jets (|a| < s), an evaluation plan for
all monomials of degree <= d (each entry = parent entry x one coordinate),
and the incidence pair list (jet a, monomial m, shift m-a) over all pairs
where `a` is a bitwise submask of `m` — exactly the support of
`gijs::hasse` in char 2 (coefficient 1 there; verified against `hasse` on
generic points). Per point and matvec, monomial evaluations y^b are
recomputed by streaming (one field mul each), so no E x N_d object and no
per-point tables are ever stored. Memory: pairs table (12 B/pair) + O(N_d)
vectors + O(E) scratch.

Measured incidence: pairs/pt = 2,350,208 at (k,s,d)=(28,3,5) vs the
componentwise bound sum_t C(k+t-1,t)C(k+d-t-1,d-t) = 2,420,769 (97%);
at (13,3,5): 54,730 vs bound 71,253 (77%). The submask count is the true
matvec cost driver.

**Decision.** v in rowspan(M) iff z.v = 0 for all z in ker(M). We sample
kernel vectors of A = M_tau^T M_tau by scalar Wiedemann (random b, u;
projected Krylov sequence; Berlekamp-Massey with chunked stabilization;
two-pass extraction, O(N_d) memory; every returned z is checked A z = 0
exactly).

**The isotropic-radical catch (found by testing, not theory).** ker(A) can
strictly contain ker(M): at (n=41,k=19,s=2,d=3) rank(M) = 760 but
rank(A) = 759, and naive per-sample verification (M z = 0) failed 32/32
samples — a random z in ker(A) lies in ker(M) with probability only ~1/q
when the radical R = M(ker A) = col(M) ∩ ker(M^T) is 1-dimensional.
Fix (implemented): collect samples z_1..z_m, compute w_i = M z_i, and take
combinations z(c) = sum c_i z_i with c in ker[w_1 ... w_m] (tiny E x m
dense kernel via linalg::kernel_basis). Those z(c) are in ker(M) by
construction, span span(zs) ∩ ker(M), and are still verified exactly.
A basis of them is tested against v: any separator certifies `false`
exactly; `samples` (=4) independent non-separating certified vectors give
`true`. Remaining heuristic content of `true`: the sampled subspace of
ker(M) behaves like a random >= samples-dimensional one, so a missed
separator has probability ~ q^-samples. Additional exact early exit: a
full-degree minimal polynomial without x-factor certifies ker(A) = {0}
("Invertible"), hence ker(M) = {0} and the answer is exactly `true`.

## Correctness (differential vs `gijs::distinguish` oracle)

80 decisions across the task grid, **0 mismatches**:

| suite | instances | decisions | notes |
|---|---|---|---|
| random codes | (q,n,k,s,d) = (64,36,14,2,4), (64,41,19,2,3), (64,30,12,3,4), (128,36,14,2,4) | 32 | (19,2,3) case is the radical case: needs 2-4 samples/decision |
| pure GRS | (64,40,14,2,4), (128,41,19,2,3), (64,30,12,3,4) | 24 | all accept path, 4 certified vectors each |
| RLCE shortened | (128, 44/30/8 -> n'=41,k'=19, 2,3), (128, 34/20/4 -> 32,14, 2,4), (64, 26/16/3 -> 25,12, 3,4) | 24 | mixed accept / certified-reject, as the attack expects |

Unit tests: matvec M and M^T match a direct `hasse` computation entry-for-
entry (plus the dot identity u.(Mx) = (M^T u).x); BM recovers a planted
recurrence; Wiedemann extracts verified kernel vectors of a planted-rank
dense operator (test requires >= 15 of 20 attempts to succeed; it
passed); pair list equals the `hasse` support exactly and is
bounded by the closed-form componentwise count. Diagnostic tests
(`diag_kernel_gap`, `diag_failing_case`) document rank(M) vs rank(A):
Euler relations make rank(M) = E - (n-1) in the tested s=2 cases, and the
(41,19,2,3) radical gap rank(M) - rank(A) = 1.

Extraction failures do occur (BM/projection heuristic): 5 failed attempts
in ~200 across the suites; all absorbed by retries/cancellation,
verify_fail of certified combinations was 0 everywhere.

## Benchmarks

Bench A — largest completed both-ways single hold-out, (q=256, n'=32,
k'=12, s=3, d=5), tau=0, dense M_tau = 2,821 x 4,368 (24.6 MB):

| instance | dense time | dense RSS | matfree time | matfree RSS | decision agree |
|---|---|---|---|---|---|
| random points | 8.73 s | 26.9 MB | 168.7 s | 4.9 MB | false = false |
| pure GRS points | 7.97 s | 27.0 MB | 334.8 s | 5.0 MB | true = true |

Larger partial at (q=256, n'=40, k'=13, s=3, d=5), random, tau=0: dense
44.6 s / 51.8 MB RSS. Matfree: one Wiedemann sample at deg(mu) ~ 4096 did
not complete within ~7 min CPU on this VM (killed), so no completed
matfree decision at that size; RSS while running was 2.5 MB.

Bench B — matfree-only at paper scale (q=1024, n'=348, k'=28, s=3, d=5),
single hold-out. Dense would need 150,945 x 201,376 x 2 B = 61 GB:
- JetOp build 16.5 s; pairs/pt 2,350,208; nnz total ~ 8.16e8.
- matvec_M 4.73 s, matvec_MT 4.16 s, steady-state A-apply 4.60 s.
- **Peak RSS 94.0 MB** (pairs table ~28 MB + eval structures + vectors;
  ~650x below the dense matrix alone).
- Extrapolation: one kernel sample ~ 3*(E_tau+1) ~ 4.5e5 A-applies
  ~ 2.1e6 s ~ 24 days; a full decision (<= 4 samples) ~ 96 days on this
  VM. NOT feasible end-to-end with scalar Wiedemann; the deliverable at
  this scale is the memory feasibility + matvec costs.

**Honest summary of the time trade-off.** Scalar Wiedemann costs
Theta(deg mu) A-applies with deg mu ~ rank(A) ~ E: at bench-A size that is
~19-42x slower than dense elimination. The matfree version wins on
memory at every size (5.5x at bench A, ~650x at flagship) and is the only
option once M does not fit in RAM. Closing the time gap needs block
Wiedemann (block size 32-64 amortizes the pair traversal over many
right-hand sides and cuts sequence length by the block factor in wall
clock) — not implemented in v1.

## Composition with hold-out reuse (design note)

Per-tau flag criterion in reuse form: tau is flagged iff
e_0 in span{ u_tau : u in ker(M_full^T) } (restrictions of left-kernel
vectors of the FULL matrix to tau's jet block; e_0 = the a=0 unit vector).
Assessment for a Wiedemann engine:
- The natural black-box operator for left kernels is B = M_full M_full^T
  (E x E) or A^T = (M_full^T M_full) transposed — same matvec machinery as
  here, run on M_full instead of M_tau. Sampling a basis of ker(M_full^T)
  costs ~ dim ker(M_full^T) Wiedemann extractions in the worst case; but
  block Wiedemann returns many kernel vectors per Krylov run, and we only
  need the J-dimensional restrictions u_tau, so ~J + O(samples) independent
  left-kernel samples suffice to determine every tau's span w.h.p.
- Total: ONE matrix-free pass over M_full (~ (J + samples) kernel samples
  worth of Krylov work, shared by all n' taus) + n' tiny (J x J) span
  tests. Against the current per-tau design (one M_tau factorization per
  tau in `distinguish`, or one M_tau kernel sampling per tau here), that
  removes the factor n' of redundant operator work — the same n' the
  dense reuse scheme in `reuse.rs` removes — while keeping memory at the
  matfree level (no 61 GB matrix, no N_d^2 factor).
- Caveats: (i) the isotropic-radical issue has a left-sided analogue:
  ker(B) vs ker(M_full^T) needs the same cancellation care (here it is
  benign: we need the restrictions themselves, and any u with
  M_full^T u = 0 exactly is usable; Wiedemann on B = M M^T produces
  candidates whose M^T-image must be cancelled exactly as in this
  module). (ii) Scalar Wiedemann makes the shared pass cost
  ~(J+samples) x (one extraction) which at flagship numbers (J=435) is
  the same order as the ~n' extractions it replaces; the real win needs
  block Wiedemann so the J restrictions come from O(1) block runs.
  Verdict: composition is sound and is the right v2, but its payoff is
  gated on block Wiedemann, not on the reuse algebra.
