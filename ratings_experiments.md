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

TBD — collapse 4 per-worker mus into a single team average.
