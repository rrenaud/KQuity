# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Win-probability model for Killer Queen (arcade), trained on game event streams. [Design doc](https://docs.google.com/document/d/1JLwlZsr0hzYZl1MpvdBXiRgem5UdG-aGHa4v3Rmo2dY).

## Workspace layout

This repo lives at `/workspace/KQuity` on a RunPod instance. Multiple git worktrees coexist:

| Path | Branch | Purpose |
|------|--------|---------|
| `/workspace/KQuity` | `seq_model` | Main checkout, sequence model work |
| `/workspace/KQuity-lgb-scaling` | `lgb_scaling` | LGB scaling experiments |
| `/workspace/KQuity-scaling-login` | `main` | Login-based scaling |
| `/workspace/KQuity-tabnet` | `tabnet` | TabNet experiments |

Remote: `https://github.com/rrenaud/KQuity.git`

## Hardware

3x NVIDIA GeForce RTX 3090 (24GB each). Use `--device cuda:0`, `cuda:1`, `cuda:2` to target specific GPUs. Run parallel experiments on separate GPUs.

256 CPUs. **LightGBM deadlocks on this machine** when using the default thread count with small datasets (<100K samples). Always cap threads: set `OMP_NUM_THREADS=16` env var or pass `'num_threads': 16` in LightGBM params.

## Two classifiers

- **Win-probability model**: Partial game state → P(gold wins). 52 in-game features (berry counts, snail position, kills, etc.) extracted at each event. Trained with LightGBM.
- **Game quality classifier**: Full game → competitive or junk? 69 hand-crafted heuristic features. Trained on logged-in (positive) vs unfiltered (negative). Tournament games anchor the threshold (≥ 0.3643).

## Sequence model (transformer)

GPT-2-style model (nanoGPT variant) as an alternative to hand-engineered LGB features.

- **Architecture**: 4 layers, 4 heads, 128 embed dim, ~0.8M params. Config in `sequence_model/config.py`.
- **Tokenization**: 185 tokens — time-gap buckets, event tokens, map/side/player tokens. Defined in `sequence_model/vocab.py`.
- **Dual loss**: next-token CE + win-probability BCE (λ_wp=0.1).
- **Game-aligned batching**: complete games per batch element, no boundary spanning.
- Key files: `sequence_model/model.py` (KQModel), `sequence_model/train.py`, `sequence_model/tokenize_games.py`, `sequence_model/evaluate.py`, `sequence_model/compare_models.py`.

## Data

Prefer compact binary `.bin` files over raw `.csv.gz` — they load faster via `event_codec.py`.

- `quality_filtered/encoded/all_games.bin` — Best dataset. Games passing quality classifier, sorted by quality score descending.
- `logged_in_games/encoded/all_games.bin` — Games with ≥1 logged-in player.
- `late_tournament_games/encoded/all_games.bin` — Holdout tournament games for evaluation.
- `unfiltered_partitioned/` — Full unfiltered CSV. Use only when you need the complete population.

Re-encode from CSV: `python encode_datasets.py`

## Key modules

- `event_codec.py` — Binary codec for game events. Use `materialize_entries()` or `fast_materialize_from_codec()` for binary → numpy features.
- `fast_materialize.py` — CSV → 52-feature numpy matrix. Fast path using bulk list assignment.
- `preprocess.py` — Slow OO path (GameEvent/GameState classes). For verification and event-level operations.
- `train_model.py` — LightGBM training pipeline: partition data, materialize features, train.
- `symmetry.py` — Blue/Gold team swap for 2x data augmentation.
- `constants.py` — Enums: Team, Map, VictoryCondition, PlayerCategory, MaidenType.
- `map_structure.py` / `map_structure_info.json` — Map geometry (berry/maiden coords, snail track width).
- `compute_ratings.py` — Player ELO/skill ratings.

## Commands

### Tests
```bash
pytest tests/
pytest symmetry_test.py
```

### LGB training
```bash
python train_model.py                    # standard training
python train_model.py --symmetry-augment # with 2x symmetry augmentation
python train_model.py --slow-and-verify  # cross-check fast vs slow path
```

### Quality classifier
```bash
python -m game_quality_classifier.train_quality_classifier             # train
python -m game_quality_classifier.train_quality_classifier --recompute # recompute features
python -m game_quality_classifier.train_quality_classifier --sweep     # data size sweep
```

### Sequence model
```bash
python -m sequence_model.tokenize_games \
    --train-dir logged_in_games/ \
    --val-csv late_tournament_games/late_tournament_game_events.csv.gz

python -m sequence_model.train --device cuda --compile \
    --batch-size 64 --block-size 2560 --max-iters 2000

python -m sequence_model.evaluate --checkpoint sequence_model/out/ckpt.pt

python -m sequence_model.compare_models \
    --test-csv late_tournament_games/late_tournament_game_events.csv.gz
```

## Quality classifier lessons

- **Login count is the dominant positive signal.** ≥9 logins is the sweet spot. Player skill ratings add nothing beyond login count.
- **Self-distillation works.** Use the existing classifier to prune bottom 10% of positives and cherry-pick high-scoring 8-login games.
- **More data only helps if it's clean.** Adding 2K cherry-picked 8-login games helps; 4K (dipping into lower quality) hurts.
- **Bagged ensembles are free.** Train multiple models with 80% bootstrap bagging + different seeds, average predictions.
- **Always validate with multiple seeds.** Seed variance on unf@95% is ±0.4–0.6%. Improvements under ~1pp need multi-seed confirmation.

## Plot preferences

Use `fill_between` for variance shading (mean ± std band) on scaling plots, not error bars.
