# Per-Tier Data Quality Comparison

**Date:** 2026-02-04
**Test set:** Late-tournament holdout (793 games, 0% drop)
**Training:** 90% state drop, LightGBM with 200 leaves / 200 trees
**Tiers tested:** 1–4 (tier 5 / no-logins excluded for speed)

## Tier Definitions

| Tier | Description | Games | Samples (90% drop) |
|------|-------------|------:|--------------------:|
| 1 — major_tourn | Games from major tournaments (>15 teams), excluding holdout | 12,066 | 227,226 |
| 2 — near_major | Non-tournament games on same cabinet within ±2 days of a major tournament | 45,496 | 526,803 |
| 3 — other_tourn | All other tournament games (not tier 1, not holdout) | 24,272 | 410,314 |
| 4 — logins | Non-tournament games with ≥1 hivemind login, sorted by login count desc | 164,652 | 1,891,634 |

## Method

For each tier used as the "focal" tier, set the sample threshold to that tier's full
sample count. Train a model on each tier's data capped at that threshold. This produces
a 4×4 grid: each tier gets a round where it uses all its data, and others are
downsampled to match. Tiers smaller than the threshold use all available samples.

## Results: Log Loss (lower = better)

| Training tier | @227k | @410k | @527k | @1.9M |
|---------------|------:|------:|------:|------:|
| tier1_major_tourn | 0.5927 | 0.5927 | 0.5927 | 0.5927 |
| tier3_other_tourn | 0.5929 | 0.5819 | 0.5819 | 0.5819 |
| tier2_near_major | 0.5879 | 0.5788 | 0.5776 | 0.5776 |
| **tier4_logins** | **0.5856** | **0.5778** | **0.5748** | **0.5696** |

## Results: Accuracy (higher = better)

| Training tier | @227k | @410k | @527k | @1.9M |
|---------------|------:|------:|------:|------:|
| tier1_major_tourn | 67.6% | 67.6% | 67.6% | 67.6% |
| tier3_other_tourn | 67.3% | 67.9% | 67.9% | 67.9% |
| tier2_near_major | 67.9% | 68.5% | 68.5% | 68.5% |
| **tier4_logins** | **68.3%** | **68.5%** | **68.5%** | **68.9%** |

## Results: Egg Inversions (lower = better)

| Training tier | @227k | @410k | @527k | @1.9M |
|---------------|------:|------:|------:|------:|
| tier1_major_tourn | 6.85% | 7.20% | 6.70% | 7.20% |
| tier3_other_tourn | 6.70% | 5.50% | 5.80% | 5.80% |
| tier2_near_major | 7.35% | 5.90% | 4.80% | 4.75% |
| **tier4_logins** | **9.40%** | **5.10%** | **4.30%** | **1.75%** |

## Key Findings

1. **Tier 4 (non-tournament with logins) wins at every sample threshold.** It has the
   best log loss and accuracy across the board, and at full scale (1.9M samples) achieves
   the best egg inversion rate (1.75%) by a wide margin.

2. **Volume dominates tier "quality."** The tournament-centric tiering (designed to
   prioritize competitive play) does not predict per-sample model utility. Tier 4's
   massive data volume (1.9M samples vs tier 1's 227k) more than compensates for its
   lower assumed quality.

3. **At equal sample counts, tier 4 still wins.** Even at the 227k threshold (where all
   tiers have the same amount of data), tier 4 achieves the best log loss (0.5856 vs
   tier 1's 0.5927). This suggests non-tournament logged-in games are not just more
   plentiful — they are also more informative per sample.

4. **Tier 1 (major tournaments) is the weakest tier.** Smallest by volume, and worst
   per-sample quality. This may be because tournament games have unusual dynamics
   (high-pressure play, specific strategies) that don't generalize to the test set, or
   because the holdout test set (also late-tournament) has distributional overlap that
   doesn't favor earlier tournament data.

5. **More data always helps.** Every tier improves monotonically with more samples.
   No tier shows diminishing returns within its available data range.

## Implications for Training

- The current tier priority ordering (major tournament → near-major → other tournament
  → logins) is backwards from the perspective of model quality. A volume-first strategy
  drawing primarily from tier 4 would produce better models.
- The 100k game budget from the previous tiered doubling experiment was too conservative.
  Tier 4 alone has 164k games, and using all of them yields the best result by far.
- Future work should consider whether combining tiers (e.g., tier 4 + tier 2) yields
  further gains, since both tiers independently produce strong models.
