RLCE full package, 2026-10-26 (updated with ID-0 cluster success, exact numbers).

paper_v8/ : v8 paper source, including cluster.tex
  (A Third Phase 1: Flag-Pattern Clustering, H5).
  ID-0 end-to-end cluster run: 160/160 pairs, 470 pure columns,
  0 orphans, 0 oversized groups, unmix 160/160, verify PASS,
  a little over 3 hours wall time across 12 threads.
  Build with: pdflatex subcode_paper_v8.tex (run twice)
  No prebuilt PDF is included, because PDFs in the working
  directory predate cluster.tex and would be stale.

gijs_matfree_nist/ : NIST DRBG crate source (same as v3 zip:
  includes AUDIT_20261026.md, reuse-holdout-test, cluster mode,
  dumpr/srcheck/trialcount). Build with: cargo build --release
  Excludes target/ (build artifacts).
