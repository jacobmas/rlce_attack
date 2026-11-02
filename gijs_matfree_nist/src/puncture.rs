//! Shortening + puncturing test for the flag-none regime.
//!
//! Idea (Alt 2 in the 2026-10-25 synthesis; no attack code here,
//! only a measurement tool):
//!
//!   1. Shorten as in Phase 1 to n' columns (dimension k').
//!   2. Puncture: delete p kept columns from the *point set*,
//!      without re-shortening. Dimension stays k', length drops
//!      to n'' = n' - p. The punctured columns' shortened vectors
//!      are unchanged; only the point set shrinks.
//!   3. Run the normal reuse distinguisher on the punctured set
//!      (ONE reuse factorization per puncturing, exactly like a
//!      Phase-1 trial, and cheaper, since E'' < E).
//!
//! Why this matters: a twin member whose mate is *punctured* is
//! a singleton-punctured column. By the holdout-test results at
//! kp28 n'=115 (50/50 pair tests), such a column outputs random
//! (flagged), whereas a both-kept dirty column outputs GRS in
//! the flag-none regime. A twin member whose mate was *hit by
//! shortening* is different and must not be confused with it:
//! shortening at the mate ties the clean and random projections
//! together (S * mate = 0 forces S*g proportional to S*r), so
//! the surviving member becomes clean (derandomized) in the
//! shortened code. Puncturing does not change S, so the surviving
//! member stays mixed and is flagged. Categories below keep
//! these two singleton cases separate.
//!
//! Identification by intersection (attacker view, no meta):
//! for each kept column, collect the punctured sets of the
//! puncturings where that column was kept *and* flagged, and
//! intersect them. In the flag-none regime, flagged trials for
//! a twin member occur (ideally) only when its true mate was
//! punctured, so the true mate lies in every such punctured
//! set, while a non-mate lies in them only with probability
//! (p/n') per trial. The intersection should therefore shrink
//! to {true mate} as puncturings accumulate. In the working
//! regime this fails by design: both-kept dirty columns are
//! also flagged, and their mate is *kept*, not punctured, in
//! those trials, so the intersection over all flagged trials
//! does not contain the mate. The report therefore gives both
//! the attacker intersection *and* the meta breakdown of flagged
//! trials by category, so a failed intersection can be explained
//! instead of misread.
//!
//! Monotonicity note: removing points can only shrink the jet
//! span. Hence a singleton-punctured column that was outside
//! the span of all other points stays outside after further
//! puncturing (its flagging is robust). A clean column inside
//! the span can fall out when its representing points are
//! punctured, so clean false positives after puncturing are
//! the quantity to measure, not assume away.

use crate::gf::Gf;
use crate::linalg::shorten;
use crate::rng::Rng;
use std::collections::HashSet;

/// Category of a column in a punctured trial, by meta (testing only).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PunctureCategory {
    /// Pure column (is_pure in meta).
    Pure,
    /// Twin member whose mate was hit by shortening (mate not in
    /// the shortened kept set). Derandomized: behaves clean.
    SingletonHit,
    /// Twin member whose mate was in the shortened kept set but
    /// was punctured in this trial. Mixed: expected flagged.
    SingletonPunctured,
    /// Twin member whose mate is also in the punctured kept set.
    BothKept,
}

/// Classify one kept' column. `kept` is the shortened kept set
/// (original indices), `punctured_orig` is the set of original
/// indices punctured in this trial, `kept_prime` is the punctured
/// kept set. `meta` is testing-only.
pub fn classify(
    meta: &crate::keyio::KeyMeta,
    kept: &[usize],
    punctured_orig: &HashSet<usize>,
    orig: usize,
) -> PunctureCategory {
    if meta.is_pure[orig] {
        return PunctureCategory::Pure;
    }
    let pid = match meta.pair_id[orig] {
        Some(p) => p,
        None => return PunctureCategory::Pure, // unreachable for twin meta
    };
    let (a, b) = meta.pairs[pid];
    let mate = if a == orig { b } else { a };
    if !kept.contains(&mate) {
        PunctureCategory::SingletonHit
    } else if punctured_orig.contains(&mate) {
        PunctureCategory::SingletonPunctured
    } else {
        PunctureCategory::BothKept
    }
}

/// Intersection of a list of sets (empty list -> empty set, so a
/// column flagged in zero trials has no candidate, not "everyone").
pub fn intersect_sets(sets: &[HashSet<usize>]) -> HashSet<usize> {
    let mut it = sets.iter();
    let first = match it.next() {
        Some(s) => s.clone(),
        None => return HashSet::new(),
    };
    it.fold(first, |acc, s| acc.intersection(s).cloned().collect())
}

/// One puncturing's results (meta breakdown + flagged list).
#[derive(Debug)]
pub struct PunctureTrial {
    pub shortening: usize,
    pub puncturing: usize,
    /// Original indices punctured in this trial.
    pub punctured_orig: Vec<usize>,
    /// Flagged original indices (reuse decisions, !is_grs).
    pub flagged: Vec<usize>,
    /// Flagged counts by category, in order
    /// [Pure, SingletonHit, SingletonPunctured, BothKept].
    pub flagged_by_cat: [usize; 4],
    /// Total counts by category among kept' columns, same order.
    pub total_by_cat: [usize; 4],
}

/// Partial state of a shortening, as checkpointed after each
/// puncturing (see `run_shortening`): the number of puncturings
/// completed, the total flag count per kept column so far, and
/// the attacker intersection per kept column so far. The
/// intersection is incremental and order-independent, so
/// resuming from it and intersecting with the remaining
/// puncturings gives exactly the intersection the uninterrupted
/// run would have computed. A column with flag count 0 has no
/// intersection yet; its stored set is empty and the first
/// flagging puncturing sets it, rather than intersecting with
/// empty.
#[derive(Debug, Clone)]
pub struct PunctureResume {
    pub done: usize,
    pub flag_counts: Vec<usize>,
    pub intersections: Vec<(usize, std::collections::HashSet<usize>)>,
}

/// One shortening's full puncturing run, including attacker
/// intersections per kept column.
#[derive(Debug)]
pub struct ShorteningReport {
    pub shortening: usize,
    pub kept: Vec<usize>,
    pub trials: Vec<PunctureTrial>,
    /// Total flag count per kept column, over ALL puncturings
    /// of this shortening, including any resumed from a
    /// checkpoint. (`trials` alone only covers puncturings run
    /// in this process when resuming, so aggregation must use
    /// this, not a recount over `trials`.)
    pub flag_counts: Vec<usize>,
    /// Per kept column (original index): attacker intersection
    /// over puncturings where the column was kept' and flagged.
    /// Empty if the column was never flagged.
    pub intersections: Vec<(usize, HashSet<usize>)>,
    /// Per twin pair both kept in this shortening: (pair members,
    /// intersection for member A, intersection for member B).
    /// Only pairs where at least one member was flagged are listed
    /// in the attack.rs report; this struct keeps all pairs.
    pub pair_intersections: Vec<(usize, usize, HashSet<usize>, HashSet<usize>)>,
}

fn cat_index(c: PunctureCategory) -> usize {
    match c {
        PunctureCategory::Pure => 0,
        PunctureCategory::SingletonHit => 1,
        PunctureCategory::SingletonPunctured => 2,
        PunctureCategory::BothKept => 3,
    }
}

/// Run puncturings for one shortening. `shortening_seed` is the
/// Phase-1 trial seed for this shortening (hit = Rng::new(seed).subset).
/// Puncturing j uses Rng::new(puncture_seed + j).subset(n', p).
/// Returns the report; the caller prints it.
///
/// `overall`, when given, is advanced once per puncturing (i.e.
/// once per reuse factorization). It is shared across shortenings
/// in Phase-1 puncture mode, where shortenings run in parallel:
/// `indicatif::ProgressBar` is thread-safe, so one bar tracks the
/// total factorization count.
/// `factor`, when given, is the per-factorization bar for this
/// shortening: its message is set to the current puncturing and
/// it is advanced work-weighted through that factorization's
/// elimination (see `reuse::distinguish_all_reuse_with_progress`),
/// then reused for the next puncturing of the same shortening.
/// Without it, a factorization is a black box for its whole
/// duration — at flagship scale on the order of an hour — so
/// this is the bar that tells you a long factorization is
/// actually moving rather than stuck.
pub fn run_shortening(
    gf: &Gf,
    cols: &[Vec<u16>],
    meta: Option<&crate::keyio::KeyMeta>,
    nprime: usize,
    s: usize,
    d: usize,
    shortening: usize,
    shortening_seed: u64,
    puncturings: usize,
    puncture_size: usize,
    puncture_seed: u64,
    overall: Option<&indicatif::ProgressBar>,
    factor: Option<&indicatif::ProgressBar>,
    resume: Option<&PunctureResume>,
    checkpoint: Option<
        &dyn Fn(usize, &[usize], &[usize], &[(usize, std::collections::HashSet<usize>)]),
    >,
) -> ShorteningReport {
    let n_pub = cols.len();
    let mut rng = Rng::new(shortening_seed);
    let hit = rng.subset(n_pub, n_pub - nprime);
    let (short_cols, kept) = shorten(gf, cols, &hit);

    // Incremental attacker intersections, maintained across
    // puncturings (and across a resume). `flagged_before` marks
    // columns that have flagged at least once, so the first
    // flagging puncturing sets the intersection instead of
    // intersecting with the empty set.
    let mut flag_counts = vec![0usize; kept.len()];
    let mut cur_inter: Vec<HashSet<usize>> = vec![HashSet::new(); kept.len()];
    let mut flagged_before = vec![false; kept.len()];
    let mut start_j = 0usize;
    if let Some(res) = resume {
        if res.flag_counts.len() == kept.len() && res.intersections.len() == kept.len() {
            let mut ok = true;
            for (i, (orig, inter)) in res.intersections.iter().enumerate() {
                if *orig != kept[i] {
                    ok = false;
                    break;
                }
                flag_counts[i] = res.flag_counts[i];
                cur_inter[i] = inter.clone();
                flagged_before[i] = res.flag_counts[i] > 0;
            }
            if ok && res.done <= puncturings {
                start_j = res.done;
            } else {
                // Mismatched checkpoint: safest is to restart
                // this shortening from scratch.
                flag_counts = vec![0usize; kept.len()];
                cur_inter = vec![HashSet::new(); kept.len()];
                flagged_before = vec![false; kept.len()];
                start_j = 0;
            }
        }
    }

    let mut trials = Vec::new();

    for j in start_j..puncturings {
        let mut prng = Rng::new(puncture_seed + j as u64);
        let punct_pos = prng.subset(nprime, puncture_size);
        let punct_pos_set: HashSet<usize> = punct_pos.iter().cloned().collect();
        let punctured_orig: Vec<usize> = punct_pos.iter().map(|&p| kept[p]).collect();
        let punctured_orig_set: HashSet<usize> = punctured_orig.iter().cloned().collect();
        let kept_prime_pos: Vec<usize> = (0..nprime)
            .filter(|p| !punct_pos_set.contains(p))
            .collect();
        let short_prime: Vec<Vec<u16>> = kept_prime_pos
            .iter()
            .map(|&p| short_cols[p].clone())
            .collect();
        if let Some(fb) = factor {
            fb.set_message(format!("shortening {} puncturing {}", shortening, j));
        }
        let decisions =
            crate::reuse::distinguish_all_reuse_with_progress(gf, &short_prime, s, d, factor);
        if let Some(pb) = overall {
            pb.inc(1);
        }
        let mut flagged: Vec<usize> = Vec::new();
        let mut flagged_by_cat = [0usize; 4];
        let mut total_by_cat = [0usize; 4];
        for (idx, &pos) in kept_prime_pos.iter().enumerate() {
            let orig = kept[pos];
            // Attacker intersections must work without meta (Phase-1
            // puncture mode is attacker-only): update the running
            // intersection for every flagged column, whether or
            // not meta is present to break the flagged count down
            // by category.
            if !decisions[idx] {
                flagged.push(orig);
                flag_counts[pos] += 1;
                if !flagged_before[pos] {
                    cur_inter[pos] = punctured_orig_set.clone();
                    flagged_before[pos] = true;
                } else {
                    cur_inter[pos] = cur_inter[pos]
                        .intersection(&punctured_orig_set)
                        .cloned()
                        .collect();
                }
            }
            if let Some(m) = meta {
                let cat = classify(m, &kept, &punctured_orig_set, orig);
                total_by_cat[cat_index(cat)] += 1;
                if !decisions[idx] {
                    flagged_by_cat[cat_index(cat)] += 1;
                }
            }
        }
        flagged.sort_unstable();
        trials.push(PunctureTrial {
            shortening,
            puncturing: j,
            punctured_orig,
            flagged,
            flagged_by_cat,
            total_by_cat,
        });
        // Checkpoint after every puncturing, not only at the end
        // of the shortening: a kill mid-shortening then loses at
        // most the one factorization in flight, not the whole
        // shortening (at ID-0, 554 puncturings, that is the
        // difference between ~28 minutes and ~11 days).
        if let Some(checkpoint) = checkpoint {
            let intersections_now: Vec<(usize, HashSet<usize>)> = kept
                .iter()
                .enumerate()
                .map(|(i, &orig)| (orig, cur_inter[i].clone()))
                .collect();
            checkpoint(j + 1, &kept, &flag_counts, &intersections_now);
        }
    }

    let intersections: Vec<(usize, HashSet<usize>)> = kept
        .iter()
        .enumerate()
        .map(|(i, &orig)| (orig, cur_inter[i].clone()))
        .collect();

    let mut pair_intersections = Vec::new();
    if let Some(m) = meta {
        for &(a, b) in &m.pairs {
            if !kept.contains(&a) || !kept.contains(&b) {
                continue;
            }
            let ia = intersections
                .iter()
                .find(|(o, _)| *o == a)
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            let ib = intersections
                .iter()
                .find(|(o, _)| *o == b)
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            pair_intersections.push((a, b, ia, ib));
        }
    }

    ShorteningReport {
        shortening,
        kept,
        trials,
        flag_counts,
        intersections,
        pair_intersections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlce::gen_rlce;

    #[test]
    fn classify_all_four_categories() {
        // Hand-built meta: N=6, pure 0,1; pair (2,3); pair (4,5).
        // Shortened kept = [0,2,3,4] (1 hit, 5 hit).
        // Punctured = {3} (original index).
        // 0 -> Pure; 2 -> SingletonPunctured (mate 3 punctured);
        // 4 -> SingletonHit (mate 5 hit, not in kept); 3 is not kept'.
        let meta = crate::keyio::KeyMeta {
            q: 64,
            n: 4,
            k: 4,
            w: 2,
            is_pure: vec![true, true, false, false, false, false],
            pair_id: vec![None, None, Some(0), Some(0), Some(1), Some(1)],
            pairs: vec![(2, 3), (4, 5)],
        };
        let kept = vec![0usize, 2, 3, 4];
        let punctured: HashSet<usize> = [3usize].iter().cloned().collect();
        assert_eq!(classify(&meta, &kept, &punctured, 0), PunctureCategory::Pure);
        assert_eq!(
            classify(&meta, &kept, &punctured, 2),
            PunctureCategory::SingletonPunctured
        );
        assert_eq!(
            classify(&meta, &kept, &punctured, 4),
            PunctureCategory::SingletonHit
        );
        // Same kept, punctured = {0} instead: 2's mate 3 is kept' -> BothKept.
        let punctured2: HashSet<usize> = [0usize].iter().cloned().collect();
        assert_eq!(
            classify(&meta, &kept, &punctured2, 2),
            PunctureCategory::BothKept
        );
        assert_eq!(
            classify(&meta, &kept, &punctured2, 3),
            PunctureCategory::BothKept
        );
    }

    #[test]
    fn intersect_sets_shrinks_and_empty_on_no_trials() {
        let a: HashSet<usize> = [1, 2, 3].iter().cloned().collect();
        let b: HashSet<usize> = [2, 3, 4].iter().cloned().collect();
        let c: HashSet<usize> = [2, 5].iter().cloned().collect();
        let inter = intersect_sets(&[a, b, c]);
        assert_eq!(inter, [2usize].iter().cloned().collect());
        assert!(intersect_sets(&[]).is_empty());
    }

    #[test]
    fn intersections_work_without_meta() {
        // Attacker-only path (Phase-1 puncture mode): with meta None,
        // flagged columns must still get non-empty intersections, and
        // every intersection must be a subset of the punctured sets
        // that flagged the column. Tiny point, flag-all regime, so
        // flags certainly occur.
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(42);
        let key = gen_rlce(&gf, &mut rng, 30, 20, 5);
        let rep = run_shortening(&gf, &key.cols, None, 28, 2, 3, 0, 101, 5, 14, 1001, None, None, None, None);
        assert_eq!(rep.kept.len(), 28);
        let mut any_flagged = false;
        for (orig, inter) in &rep.intersections {
            let flagged_here = rep.trials.iter().any(|t| t.flagged.contains(orig));
            if flagged_here {
                any_flagged = true;
                assert!(!inter.is_empty());
                // Intersection is a subset of every punctured set
                // that flagged this column.
                for t in rep.trials.iter().filter(|t| t.flagged.contains(orig)) {
                    let pset: HashSet<usize> =
                        t.punctured_orig.iter().cloned().collect();
                    assert!(inter.is_subset(&pset));
                }
            } else {
                assert!(inter.is_empty());
            }
        }
        assert!(any_flagged);
    }

    #[test]
    fn puncture_run_counts_sum_correctly() {
        // Tiny point, reuse only: counts must sum, flagged subset
        // of kept', and intersections are subsets of punctured sets.
        let gf = Gf::for_q(64);
        let mut rng = Rng::new(42);
        let key = gen_rlce(&gf, &mut rng, 30, 20, 5);
        let meta = crate::keyio::KeyMeta {
            q: 64,
            n: key.n,
            k: key.k,
            w: key.w,
            is_pure: key.is_pure.clone(),
            pair_id: key.pair_id.clone(),
            pairs: key.pairs.clone(),
        };
        let rep = run_shortening(&gf, &key.cols, Some(&meta), 28, 2, 3, 0, 101, 5, 14, 1001, None, None, None, None);
        assert_eq!(rep.kept.len(), 28);
        assert_eq!(rep.trials.len(), 5);
        for t in &rep.trials {
            assert_eq!(t.punctured_orig.len(), 14);
            assert_eq!(t.total_by_cat.iter().sum::<usize>(), 14); // n'' = 28-14
            assert_eq!(t.flagged_by_cat.iter().sum::<usize>(), t.flagged.len());
            assert!(t.flagged.len() <= 14);
            // Every flagged column is in kept' (kept minus punctured).
            for &f in &t.flagged {
                assert!(rep.kept.contains(&f));
                assert!(!t.punctured_orig.contains(&f));
            }
        }
        // Intersections: for a flagged column, intersection is a
        // subset of every punctured set that flagged it (by construction).
        for (orig, inter) in &rep.intersections {
            if !inter.is_empty() {
                // orig must have been flagged at least once.
                let any_flagged = rep.trials.iter().any(|t| t.flagged.contains(orig));
                assert!(any_flagged);
            }
        }
    }
}
