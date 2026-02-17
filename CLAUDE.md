# KQuity

Win-probability model for Killer Queen (arcade), trained on game event streams.

Two classifiers:
- **Win-probability model**: Partial game state → P(gold wins). Uses 52 in-game features (berry counts, snail position, kills, etc.) extracted at each event during a game.
- **Game quality classifier**: Full game event stream → is this a competitive game? Uses 69 hand-crafted heuristic features over the entire game. Trained on logged-in games as positives, unfiltered games as negatives. Tournament games anchor the threshold. Used to filter training data for the win-probability model.

## Data

Prefer the compact binary `.bin` files over raw `.csv.gz` — they load faster and use `event_codec.py` for encoding/decoding.

- `quality_filtered/encoded/all_games.bin` — Best dataset for experiments. Games passing the quality classifier (score >= 0.3643), sorted by quality score descending.
- `logged_in_games/encoded/all_games.bin` — Games with at least one logged-in player, sorted by login count.
- `late_tournament_games/encoded/all_games.bin` — Holdout tournament games for evaluation.
- `unfiltered_partitioned/` — Full unfiltered CSV dataset (all games, including anonymous). Use only when you specifically need the complete population or raw CSV access.
- `tests/` — Benchmark game events for unit tests.

To re-encode from CSV: `python encode_datasets.py`

## Key modules

- `event_codec.py` — Compact binary codec for game events. Use `materialize_entries()` or `fast_materialize_from_codec()` to go from binary → numpy features.
- `fast_materialize.py` — CSV events → numpy feature matrix (52 features). Use for raw CSV files only.
- `preprocess.py` — Slow OO path with GameEvent classes. Used for verification and event-level operations.
- `train_model.py` — Training pipeline: partition data, materialize features, train LightGBM.
- `symmetry.py` — Blue/Gold team swap for data augmentation.

## Quality classifier lessons (from `model_experiments/rating_based_quality_experiments.md`)

- **Login count is the dominant positive signal.** >=9 logins is the sweet spot. Player skill ratings add nothing beyond what login count already captures.
- **Self-distillation works.** Use the existing classifier to prune bottom 10% of positives and cherry-pick high-scoring 8-login games. Both improve training signal.
- **More data only helps if it's clean.** Adding 2K cherry-picked 8-login games helps; adding 4K (dipping into lower quality) hurts. Adding quality-filtered games (circular) doesn't help.
- **Bagged ensembles are free.** Train multiple models with 80% bootstrap bagging + different seeds, average predictions. Strictly better than any single model at no extra cost.
- **Always validate with multiple seeds.** Seed variance on unf@95% is +/- 0.4-0.6%. Any claimed improvement under ~1pp needs multi-seed confirmation.

## Plot preferences

Use `fill_between` for variance shading (mean +/- std band) on scaling plots, not just error bars. The translucent background bands make it easier to see overlap and separation between curves.

## Running tests

```
pytest tests/
pytest symmetry_test.py
```

## Training

```
python train_model.py                    # standard training
python train_model.py --symmetry-augment # with 2x symmetry augmentation
python train_model.py --slow-and-verify  # cross-check fast vs slow path
```
