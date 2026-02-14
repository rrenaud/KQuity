# Game Quality Classifier — Experiment Report

## Goal

Build a binary classifier to distinguish competitive Killer Queen games from
junk/practice/button-check games, using only in-game event features (no login
metadata at inference time).

**Labels**: games with lots of hivemind logins = 1 (proxy for quality), unfiltered games = 0 (mixed).
**Validation anchor**: late tournament games should almost all pass the quality filter.

## Architecture

LightGBM binary classifier. 69 hand-crafted features from game event streams.
Training: 16K games per class (32K total), 80/20 train/val split. Early stopping
on validation AUC.

## Feature Engineering Evolution

| Commit | Features | Change |
|--------|----------|--------|
| ceecec5 | 41 | Initial: duration, counts, rates, engagement, gate triples |
| 3d90bb3 | 51 | +10 per-cabinet-position first-event features |
| bb8ffca | 61 | +per-team and per-PID first maiden use |
| 6be5314 | 77 | +time_to_6_carry, per-maiden and per-maiden-per-team counts |
| 5b0f7b2 | 65 | Replace per-maiden counts with time-to-Nth-bless features |
| b98ebb7 | 63 | Remove frac_to_first_kill and frac_to_first_carry |
| 166bbdd | 66 | +snail escape count, rate, and time-to-first |
| 17746f9 | 69 | +get-off-snail count, rate, and time-to-first |

Feature categories (69 total):
- Basic game info (10): duration, event count, bot count, victory condition, map
- Event counts (12): kills, carries, deposits, blesses, snail actions
- Event rates (10): per-second rates for the above
- Temporal (7): time-to-first for key game actions
- Engagement (4): active player count, worker objective engagement
- Gate triples (2): fastest 3-gate-touch window
- Per-cabinet first event (10): first action timing per position
- Per-team/PID maiden use (10): maiden engagement by team and player
- Time-to-Nth milestones (4): carry and bless progression markers

## Data Pipeline Improvements

**Tournament leakage removal** (d77ea21): Tournament games appeared in both
logged_in and unfiltered datasets. Removed from training to prevent inflated metrics.

**Data size sweep** (869f36b): Swept 1K–32K training size. 16K per class chosen
as default (diminishing returns beyond this).

**Separate eval data** (7dc349e): Previously, unfiltered pass-rate metrics were
computed on data overlapping with training, artificially deflating Unf@95% (7-15%
vs true ~21%). Now uses separate eval shards (X02, X03) distinct from training
shards (X00, X01).

**Temporal stride** (7dc349e): Shards are time-ordered. Changed from contiguous
(000-019) to strided (X00, X01 for X=0-9) selection for better temporal coverage
across the full dataset.

## Hyperparameter Sweep Results

Swept 6 LightGBM params one-at-a-time, 16K training per class, 69 features,
evaluated on held-out unfiltered data.

### Baseline (old defaults): num_leaves=63, min_child_samples=20

```
Param                     Value     AUC  LogLoss  Unf@99%  Unf@95%
num_leaves                   63  0.9050   0.3453   27.9%   22.3%
min_child_samples            20  0.9050   0.3453   27.9%   22.3%
```

### Key findings (sweep with old defaults)

**num_leaves**: AUC peaked at 23 leaves (0.9074) but Unf filtering was flat
(20-22% across all values). Higher leaf counts slightly worse AUC.

**min_child_samples**: Clear improvement from 20→75-100 (AUC 0.905→0.908,
LogLoss 0.345→0.342). Higher values regularize against noisy proxy labels.

**learning_rate**: 0.01-0.03 slightly better than 0.05 (LogLoss improvement),
with auto-scaled num_boost_round=2000.

**feature_fraction**: Mild improvement at 0.5-0.7 (decorrelates trees with
partially-redundant features).

**bagging_fraction=0.9**: Small AUC/LogLoss improvement.

**reg_lambda=10**: Best AUC (0.9088) in initial sweep.

### After retuning defaults to num_leaves=127, min_child_samples=75

```
Param                     Value     AUC  LogLoss  Unf@99%  Unf@95%
num_leaves                  127  0.9078   0.3432   29.3%   22.2%  <- current
num_leaves                   95  0.9089   0.3413   28.3%   21.4%  <- slightly better
min_child_samples            75  0.9078   0.3432   29.3%   22.2%  <- current
min_child_samples           100  0.9075   0.3435   29.4%   21.8%
```

Resweep showed all params landing in a narrow band (AUC 0.904-0.909, Unf@95%
20.5-22.5%). The model is near its ceiling with this feature set and label noise.

## Final Configuration

```python
DEFAULT_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 127,
    'min_child_samples': 75,
    'learning_rate': 0.05,
    'verbose': -1,
}
# num_boost_round=500 (2000 if learning_rate < 0.05), early_stopping=20
```

## Current Performance

| Metric | Value |
|--------|-------|
| Validation AUC | ~0.908 |
| Validation LogLoss | ~0.343 |
| Unfiltered pass @ 99% tournament recall | ~28% |
| Unfiltered pass @ 95% tournament recall | ~22% |

Interpretation: at a threshold that catches 99% of tournament games, ~28% of
unfiltered games also pass (the rest are filtered as low-quality). At a stricter
95% tournament recall threshold, ~22% of unfiltered pass.

## Downstream Validation: Win-Probability Training

To test whether the classifier identifies games that are actually more useful
for training, we compared win-probability models trained on quality-filtered
(QF) vs logged-in (LI) game data at multiple scales (5K-40K games, 10 variance
runs each). See [data_quality_report.md](../model_experiments/data_quality_report.md)
for full results and plots.

**Exclusive comparison (equal-states)**: QF-exclusive games (high-quality
anonymous games the classifier found) consistently outperform LI-exclusive
games (logged-in games the classifier rejected) on log loss, AUC-ROC, accuracy,
and symmetry deviation. This validates that the classifier captures a real
quality signal beyond login status — it can find good training data among
anonymous games that the login heuristic misses, and correctly rejects
logged-in games that happen to be low quality.

**Caveat**: The exclusive comparison doesn't directly measure the marginal
value of quality filtering on top of login filtering (i.e., would adding
QF-exclusive games to the logged-in pool improve the model?). It establishes
that the classifier's quality signal is genuine, but the additive benefit
remains untested.

**Non-exclusive comparison**: QF outperforms LI across all scales tested
(5K-20K). Since both datasets are similar in size (~182-183K games), this
comparison is less confounded and provides additional evidence that
quality-filtered data produces better models.

## Downstream Validation: Win-Probability Training

To test whether quality filtering actually produces better training data, we
compared win-probability models trained on quality_filtered (QF) vs
logged_in_games (LI) datasets. Both datasets are ~182-183K games with ~108K
overlap; each has ~75K exclusive games.

**Exclusive experiment (equal-states)**: Trained only on games unique to each
dataset, subsampling to equalize state counts. QF-exclusive outperforms
LI-exclusive on log loss, AUC-ROC, accuracy, and symmetry deviation at every
scale tested (5K-40K games, 10 variance runs each).

**Interpretation**: QF-exclusive games are anonymous games the classifier
identified as high-quality from event features alone. LI-exclusive games are
logged-in games the classifier rejected. The experiment shows the classifier
finds good training data that the login heuristic misses — its quality signal
is genuine and not just a proxy for login status. Prior results established
that logged-in games outperform random unfiltered games, so login is already a
quality proxy; the classifier captures something beyond that.

**Limitation**: The exclusive comparison doesn't measure the marginal value of
quality filtering on top of login filtering (e.g., would adding QF-exclusive
games to the logged-in pool help?). That remains a future experiment.

Full results: [model_experiments/data_quality_report.md](../model_experiments/data_quality_report.md)

## Key Methodological Lesson

The original eval setup (scoring unfiltered data that overlapped with training)
produced artificially optimistic filtering numbers. With num_leaves=255, the old
setup showed Unf@95%=7.3% — but with held-out eval data the true number was
~22%. This was the single biggest correction in the experiment series: always
evaluate on data the model has never seen.
