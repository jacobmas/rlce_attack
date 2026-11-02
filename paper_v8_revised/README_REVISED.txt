Revised paper_v8 -- changes in this version

Files changed:
  phase2.tex         -- rewritten in the informal voice of the
                        earlier sections. Same mathematics, same
                        algorithm, same dual-square lemma and
                        soundness theorem, with the same labels:
                        sec:fastp2, lem:fastp2-dual, alg:fastp2,
                        thm:fastp2, rem:fastp2-codim,
                        lem:verify-sound, sec:id0.
                        The old total-cost subsection and the
                        detailed ID-0 cost subsection are removed
                        from Phase 2; their replacement is
                        totalcosts.tex. The formal pure-direct
                        lemma (lem:pure-direct, referenced nowhere
                        else) is replaced by a paragraph, since
                        Phase 1 already supplies the pure columns.
  stretchedparams.tex -- w=n section updated. ISD table
                        unchanged. Pairing is now clustering
                        (30-31 trials, not ~500) followed by
                        Phase 2 root-finding, not pair-frequency
                        counting plus per-pair GIJS disentangling.
                        The old full-cost table is kept, labelled
                        as superseded, for comparison only; new
                        totals are NOT claimed here.
  totalcosts.tex     -- NEW. Template total-costs section,
                        labelled sec:total-costs and sec:key-cost
                        (so the old \ref{sec:key-cost} in the w=n
                        section still resolves). Contains:
                        a run-report checklist, model estimates
                        for comparison only, a timing table and
                        an outcome table for ID-0/ID-1/ID-2 with
                        [TBD] placeholders, and a w=n template
                        table (30-31 planned trials). The ID-0
                        old-machine row (41 trials, 12 threads,
                        a little over 3 h, SS 0.25 s, 160/160,
                        0 orphans) is filled in only as an
                        example of the format.
  subcode_paper.tex, subcode_paper_v8.tex
                     -- \input{totalcosts} added after phase2,
                        before stretchedparams. No other changes.

Compile check (2026-10-29, pdflatex + bibtex, v8 main):
  compiles to 35 pages, no errors. No new undefined references:
  all labels introduced or preserved above resolve. Pre-existing
  issues in the uploaded version remain untouched: undefined
  sec:phase1-cost, sec:reuse-cost, tab:reuse-cost, thm:persist,
  sec:heuristics, sec:empirical-h1h2, and multiply-defined
  cluster labels (phase1.tex also \input{cluster} at its end).

To fill in: replace [TBD] in totalcosts.tex from new-machine
runs, following the instructions in the comments at the top
of that file. Do not copy model estimates into the timing
table as measurements.
