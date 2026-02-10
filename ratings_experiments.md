# Ratings Experiment Log

## Experiment 1: Per-Player Ratings (10 shards)

**Date**: 2026-02-09
**Data**: `logged_in_games/gameevents_00[0-9].csv.gz` (10 shards, ~9.5K games, 1.8M samples)
**Split**: 50/50 chronological by game_id
**Model**: LightGBM, 200 leaves, 200 trees

Rating features: queen mu + 4 per-worker mu per team (10 total, 62 features).

| Metric   | Baseline (52) | Ratings (62) | Diff    |
|----------|---------------|--------------|---------|
| Log Loss | 0.5831        | 0.6035       | +0.0204 |
| Accuracy | 69.10%        | 69.81%       | +0.71%  |

Ratings improve accuracy slightly (+0.7%) but hurt log loss (+0.02), suggesting overconfidence on some predictions. Per-worker ratings may be too noisy — workers swap positions and the per-worker mu is keyed to seat position, not identity.

## Experiment 2: Queen + Avg Worker Ratings (10 shards)

**Date**: 2026-02-09
**Data**: same as Experiment 1
**Setup**: Collapse 4 per-worker mus into a single team average before materializing. Still uses the 62-feature pipeline (worker rating slots all get the same avg value).

| Metric   | Baseline (52) | Per-Player (62) | Condensed (62) | PP Diff  | C Diff   |
|----------|---------------|-----------------|----------------|----------|----------|
| Log Loss | 0.5831        | 0.6035          | 0.6617         | +0.0204  | +0.0787  |
| Accuracy | 69.10%        | 69.81%          | 68.04%         | +0.71%   | -1.06%   |

Condensed ratings are **worse than baseline** on both metrics. Per-player ratings at least improve accuracy. This suggests:
1. Individual worker identity matters more than team average skill.
2. Averaging away per-worker signal destroys useful information.
3. The log loss degradation across both rating variants may indicate the model is overfitting to rating features in the training half.

## ~~Experiment 3: Proper 56-Feature Condensed Ratings (10 shards)~~

Removed — condensed ratings consistently worse than baseline across experiments 2 and 3.

## Experiment 4: Per-Cabinet Anonymous Ratings (10 shards)

**Date**: 2026-02-10
**Data**: same as Experiment 1
**Ratings**: `ratings_queen_drone.pkl` — computed with per-cabinet anonymous ratings that fix rating deflation. Previously, anonymous players were excluded from `model.rate()`, so mu leaked out of the system (mean mu dropped 19.0 → 4.4 over 84K rated games). Now each anonymous position uses a shared per-cabinet rating, ensuring teams are always 5v5. Deflation reduced from -14.6 to -2.7 mu, and all 183K games contribute ratings (was 84K).

| Metric   | Baseline (52) | Per-Player (62) | Diff    |
|----------|---------------|-----------------|---------|
| Log Loss | 0.5831        | 0.5885          | +0.0055 |
| Accuracy | 69.10%        | 70.62%          | +1.53%  |

Substantial improvement over Experiment 1's ratings (+1.5% accuracy vs +0.7%). Log loss penalty is also much smaller (+0.005 vs +0.020). Fixing the deflation leak makes per-player mu values more meaningful — a player's mu now reflects actual skill rather than being dragged down by untracked losses to anonymous opponents.

## Experiment 5: Anonymous Sigma Floor Sweep

**Date**: 2026-02-10
**Data**: same as Experiment 1
**Motivation**: Cabinet anonymous ratings let sigma decay naturally (8.33 → ~1-2), but anonymous positions represent *different people each game* — uncertainty should stay high. With decayed sigma, anonymous positions barely absorb mu changes (`mu += (sigma²/team_sigma²) * omega`), forcing disproportionate updates onto logged-in players. A `min_anon_sigma` parameter floors anonymous sigma after each game's update.

### Sigma floor sweep (correlation metric)

| min_sigma | correlation |
|-----------|-------------|
| 0         | 0.1874      |
| 1         | 0.1874      |
| 2         | 0.1905      |
| 3         | 0.1941      |
| 4         | 0.1958      |
| **5**     | **0.1961**  |
| 6         | 0.1957      |
| 7         | 0.1949      |
| 8         | 0.1939      |

Peak at sigma=5. Correlation improves +0.009 over no floor.

### LightGBM A/B with min_anon_sigma=5

| Metric   | Baseline (52) | Per-Player (62) | Diff    |
|----------|---------------|-----------------|---------|
| Log Loss | 0.5765        | 0.6001          | +0.0236 |
| Accuracy | 69.10%        | 70.12%          | +1.02%  |

Sigma floor at 5 improves raw rating correlation but the downstream LightGBM accuracy dips slightly vs Experiment 4 (70.1% vs 70.6%). The model may have been exploiting decayed sigma as a proxy for cabinet activity level. Deflation remains similar (-2.6 mu).

### LightGBM A/B with min_anon_sigma=2

| Metric   | Baseline (52) | Per-Player (62) | Diff    |
|----------|---------------|-----------------|---------|
| Log Loss | 0.5765        | 0.5951          | +0.0186 |
| Accuracy | 69.10%        | 70.29%          | +1.19%  |

### Summary across sigma floor values

| min_sigma | Correlation | Accuracy | Log Loss |
|-----------|-------------|----------|----------|
| 0 (Exp 4) | 0.1874     | 70.62%   | 0.5885   |
| 2         | 0.1905      | 70.29%   | 0.5951   |
| 5         | 0.1961      | 70.12%   | 0.6001   |

Correlation and LightGBM accuracy pull in opposite directions. Higher sigma floors improve raw rating signal but the model benefits from sharper (lower sigma) anonymous ratings — likely using decayed sigma patterns as a proxy for cabinet activity. No floor (Experiment 4) remains the best downstream result.
