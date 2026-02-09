# KQuity

Win-probability model for Killer Queen (arcade), trained on game event streams.

## Data

- `logged_in_games/` — Preferred dataset for experiments. Games filtered to those with at least one logged-in user, sorted by login count. Higher quality signal.
- `unfiltered_partitioned/` — Full unfiltered dataset (all games, including anonymous). Use only when you specifically need the complete population.
- `tests/` — Benchmark game events for unit tests.

## Key modules

- `fast_materialize.py` — Fast CSV events → numpy feature matrix (52 features). Production path.
- `preprocess.py` — Slow OO path with GameEvent classes. Used for verification and event-level operations.
- `train_model.py` — Training pipeline: partition data, materialize features, train LightGBM.
- `symmetry.py` — Blue/Gold team swap for data augmentation.

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
