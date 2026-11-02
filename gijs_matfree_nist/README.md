# rlce-test (gijs_rust_v4)

High-performance multiprocessor test harness for the **v4 heuristic RLCE attack**.

## Purpose

Tests two heuristics empirically:
- **H1**: A dirty (twin-mixed) column behaves like a random column in the *direct*
  GIJS distinguisher — **without** rerandomizing its twin. (v4 drops the costly
  `(n')²` replacement sweep; runs direct GIJS per hold-out.)
- **H2**: Trials (random shortenings of one fixed key) behave independently;
  pair co-occurrence counts fit `Bin(T, p1)`.

## Layout

- `src/gf.rs` — GF(2^m) log/exp tables, m = 5..=11 (q = 32..2048). (from v2)
- `src/gijs.rs` — GIJS distinguisher core: Hasse jets, interleaved forward
  elimination. Returns `true` = "GRS". (from v2)
- `src/linalg.rs` — `kernel_basis`, `shorten` (proper shortening via Gaussian
  elimination on the hit positions).
- `src/rlce.rs` — RLCE keygen: GRS(n,k), w twin pairs via random GL(2) mixing
  with random vectors, random permutation. Tracks purity and pairing.
- `src/rng.rs` — deterministic xorshift64.
- `src/main.rs` — CLI, rayon-parallel trial loop, disk-spooled results, H1/H2 stats.

## Build

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release
```

## Usage

```bash
./target/release/rlce-test --q 64 --n 30 --k 14 --w 4 --nprime 28 \
    --s 2 --d 4 --trials 60 --threads 2 --seed 11 --out test_w4
```

Flags:
- `--q`: field size (32, 64, 128, 256, 512, 1024, 2048)
- `--n`: GRS length; `--k`: dimension; `--w`: twin pairs (N = n+w public cols)
- `--nprime`: columns kept per trial (shorten `N - nprime` positions)
- `--s`, `--d`: GIJS Hasse jet parameters
- `--trials`: number of trials; `--threads`: rayon threads (0 = auto)
- `--seed`: base RNG seed; `--out`: output directory
- `--self-test`: tiny 4-trial validation run

## How it works

1. **One key**: generates a single RLCE public key (N columns).
2. **Trials** (rayon parallel): each trial picks a random hit set, computes the
   *properly shortened* code (dimension `k' = k - |hit|`), runs direct GIJS on
   each of the `n'` hold-outs. Suspicious = GIJS says "random".
   Each trial spools to `{out}/trial_*.txt` (disk, not RAM).
3. **Aggregate**: reads trial files, builds frequency table.
4. **H1**: for twin members with intact pairs (reliably dirty), reports
   mean/min/max flag rate vs theory `1 - d/q`. Pure columns should be ~0.
5. **H2**: for each true pair, `T_p` = trials both present, `C_p` = both flagged.
   Estimates `p1_hat`, chi-squared vs `Bin(T_p, p1)`.
6. Writes `{out}/columns.csv`, `{out}/pairs.csv`.

## Results (2026-10-17, 2 threads, 7GB VM)

| config | trials | time | H1 dirty | H1 pure | H2 |
|---|---|---|---|---|---|
| q=64,n=28,k=12,w=2,n'=25 | 20 | 1.2s | 1.000 | 0.000 | p1=1.0 |
| q=64,n=30,k=14,w=4,n'=28 | 60 | 11.5s | 1.000 | 0.000 | p1=1.0 |
| q=64,n=30,k=14,w=4,n'=28 | 100 | 17.0s | 0.875 | 0.000 | p1=0.73, chi2=272/4dof |

**H1**: Supported with a caveat. Dirty columns flagged at high rate (0.875–1.0,
theory lower bound 0.9375); pure at 0.0. **Caveat**: degenerate twin pairs
(where the GL(2) has c=0, making one member *actually pure*) appear at the
expected rate (~1/q per pair). Such a member is never flagged — correctly, as
it is pure. The test labels it "dirty" by pair membership, causing min=0.0.

**H2**: chi-squared **rejects homogeneity** (272 on 4 dof). Investigation shows
this is driven by *pair heterogeneity* (including the degenerate pair above,
whose co-occurrence is 0), not necessarily trial dependence. A per-pair
independence test (serial correlation within each pair's trial sequence) is
not implemented; the current chi2 tests a common `p1` across pairs, which
heterogeneity violates. **H2 as stated (Bin fit with common p1) is not
supported**; a pair-specific model may be needed.

**Scaling**: per-trial cost is `n'` GIJS runs. At `(E,N_d) = (405,2380)`,
~0.17s/trial (28 hold-outs) on 2 threads. 100 trials ≈ 17s. Projected
< 1 min on 16 threads. Disk: 408KB for 100 trials (trial files are tiny).

## Notes

- v4 uses **direct GIJS** (no replacement sweep). The `(n')²` replacement from
  v3 is dropped per the heuristic H1 (dirty ≈ random without rerandomization).
- Shortening is *proper* (Gaussian elimination), not puncturing.
- The `distinguish` early-exits "random" when the evaluation vector has a
  component outside the row space; "GRS" requires full elimination.
- If `E >= N_d`, the kernel is always trivial → always "random" (warns).
