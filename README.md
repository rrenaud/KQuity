# KQuity

KQuity powers live win-probability predictions in [Hivemind](http://kqhivemind.com/) and provides game analysis for Killer Queen — a 10-player arcade strategy game where teams of five race to win by military dominance, economic victory (berries), or snail ride.

## Models

### Win-probability model

A LightGBM classifier that predicts P(gold wins) from 52 in-game state features (berry counts, snail position, kills, warrior upgrades, etc.) extracted at each game event. A typical game produces 100–300 events, giving a real-time probability curve from start to finish. Trained on quality-filtered game data with symmetry augmentation.

### Game quality classifier

Not all recorded games are competitive — many are casual warm-ups, kids mashing buttons, or half-empty cabinets. The quality classifier separates real games from junk using 69 hand-crafted features computed over the full event stream. It achieves an AUC of ~0.908 using logged-in games as positive examples and the unfiltered population as negatives, with tournament games anchoring the decision threshold. Its primary role is curating clean training data for the win-probability model.

## Analysis

- [Worker State Values](https://rrenaud.github.io/KQuity/worker_state_values/worker_state_values.html) — How much does worker composition matter? Bradley-Terry linearization to isolate the effect of upgrades (warrior, speed drone, speed warrior) on win probability.
- [Lockout Analysis](lockout_analysis/writeup.md) — When do teams lockout, how long does it last, and does it matter?

## Documentation

- [Training Guide](docs/training_lightgbm_win_predictor.md) — How the win-probability model is trained end-to-end
- [Quality Classifier Report](game_quality_classifier/quality_classifier_report.md) — Design and evaluation of the game quality classifier
- [Experiment Log](experiment_log.md) — Chronological log of modeling experiments
- [Data Quality Report](model_experiments/data_quality_report.md) — Analysis of data filtering strategies
- [Combined Scaling Experiment](model_experiments/combined_li_qf/combined_scaling_report.md) — Scaling behavior across data sources
- [Symmetry Augmentation](model_experiments/symmetry_augmentation/report.md) — Gold/blue swap augmentation results
- [Data: Logged-In Games](logged_in_games/README.md) | [Data: Quality Filtered](quality_filtered/README.md) — Dataset documentation
