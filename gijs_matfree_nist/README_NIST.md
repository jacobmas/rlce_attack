# gijs_matfree_nist — NIST DRBG version

This is a copy of gijs_matfree with ONLY the RNG replaced, plus a mechanical
API change. The original crate is untouched elsewhere. No audit fixes have
been applied here.

## What changed
- src/rng.rs: xorshift64 replaced by NIST SP 800-90A Rev. 1 CTR_DRBG with
  AES-256, via nist-drbg-rs 0.1.3 (feature "aes-ctr"). No ChaCha, no
  Bernstein construction. Seed expansion: entropy = seed bytes repeated to
  32 bytes, nonce = seed bytes + zeros to 16 bytes, personalization
  "rlce-test-v1". A u64 test seed is NOT claimed to be 256 bits of entropy;
  the point is AES-based derivation diffuses seed differences nonlinearly,
  unlike the old F2-linear stream.
- below(n): rejection sampling, no modulo bias. n==1 returns 0.
- Seed 0: old code used 0x2545_F491_4F6C_DD1D as a guard because Rng(0) was a
  stuck state. The DRBG has no stuck state; seed 0 maps to the same guard
  value inside rng.rs, and the four seed-0 guard blocks in matfree.rs were
  removed.
- All Rng(x) call sites changed mechanically to Rng::new(x). No other
  logic changed (verified by diff against the original).
- Cargo.toml: added nist-drbg-rs = { version = "0.1", features = ["aes-ctr"] }.
- src/rlce.rs (debug only): RlceKey also carries r_vecs and gls, the true
  keygen R vectors and GL matrices per pair, so randomness can be checked
  from a generated key. This is test-harness data, not part of the attack.

## Audit tools (src/bin/, local testing only)
- dumpr: prints R vectors, GL, pairs, public columns for a keygen.
  Usage: dumpr <q> <n> <k> <w> <seed>
- srcheck: R rank, SR rank after a shortening, random control.
  Usage: srcheck <q> <n> <k> <w> <seed> <nprime> <shortening_seed>
- rankcheck: R-rank of RNG draws mimicking keygen (earlier, weaker check).
These tools print private key material. Use only on local test keys.

## Build
  cargo build --release --bin rlce-attack
  cargo build --release --bin rlce-test

## Verified results with this build (2026-10-26)
- R rank from actual keygen: ID-0 (1024,630,470,160) 160/160, seeds 42 and 43;
  kp28 (1024,135,78,30) 30/30. Combined seed42+43 ID-0 rank 320/320.
  Coordinate chi2 at ID-0: 974 (seed 42), 1076 (seed 43), df 1023.
  SR rank at ID-0 n'=366, k'=46: 46/46 all, 36/36 both-kept, vs 10/10 old.
- Non-puncture frequency Phase 1, reuse engine, same kp28 key (seed 42),
  seed 100, 20 trials each: n'=111, 112, 113, 114, 115 all flagged 0/20,
  top-0 pairs, 165/165 pure candidates. The RNG fix does NOT eliminate
  flag-none at this point; the mechanism is elsewhere.
  (At these points k' = n' - 87, so 111..115 is k' = 24..28 on the tied curve.)

## Example commands (kp28)
  rlce-attack keygen --q 1024 --n 135 --k 78 --w 30 --seed 42 --out kp28
  rlce-attack phase1 --key kp28/key.pub --nprime 111 --s 2 --d 3 \
    --phase1-mode frequency --engine reuse --trials 20 --threads 2 \
    --seed 100 --out kp28_p111


## New in this build: reuse-holdout-test (2026-10-26)
  rlce-attack reuse-holdout-test --key DIR/key.pub --meta DIR/key.meta \
    --nprime NP --s 2 --d 3 [--trials T] [--seed S] [--pair-tests P] \
    [--allow-large] [--out FILE]

Same shortening, same pair order, same control choice as holdout-test,
but exact and fast: one left-kernel (Pperp) factorization per trial,
then mate present / mate absent / control absent for every pair from
small rank checks on Pperp restricted to the deleted blocks (tau alone,
tau+mate, tau+control). This is the same duality as reuse's per-tau
test, generalized from a J-column deleted set to a 2J-column deleted
set; it is not a heuristic. Cost per trial is independent of the
number of pair tests, versus 2+2P dense eliminations per trial for
holdout-test. It does NOT include random-vector tests; use
holdout-test for random_in_full / random_in_tau.

Validated 2026-10-26 at kp28, n'=112, 5 trials, 10 pair tests: output
matches the dense holdout line for line, all 50 pairs, rank 2564,
codim 361, flagged_normal 0, in_without_mate 0/50,
in_without_random 50/50. VM time for those 5 trials: 1m59s.

ID-0 estimate (n'=366, k'=46, (2,3)): one reuse factorization per
trial, roughly 30-50 minutes on an i7-13800H single core, memory about
1.8 GB, regardless of pair-tests. Compare dense holdout: 8 eliminations
for 3 pair tests (2-3.5 hours) and 22 eliminations for 10 pair tests
(5.5-9 hours) per trial. So for 10 pair tests the speedup is about
11x in elimination count, and for 5 trials about 2.5-4 hours total
instead of 27-46 hours. The reuse factorization itself is not
parallelized; run different trials in parallel processes if you want
wall-time below that.

## Audit findings (2026-10-26)
See AUDIT_20261026.md in this directory for the full six-batch
line-by-line audit of the original crate. No audit finding was fixed
in the original crate, and, except for the RNG (F0) in this NIST
build, no audit finding was fixed here either. In particular verify
never checks unmixing, verify's pair check is vacuous on flag-none
runs, puncture spools carry no key identity, frequency/cluster spools
carry no parameter binding, puncture positions repeat across
shortenings, and unmix misses the infinity, alpha=0, and alpha=infinity
cases. Read a verify PASS, a spool file, or an unmix count with those
caveats.
