# Sequence Model for Killer Queen Win Prediction

## Overview

A GPT-2 style transformer trained on game event sequences with dual objectives:
1. **Next-token prediction** — learn game dynamics by predicting the next event
2. **Win probability** — predict P(blue wins) at every position via an auxiliary head

This replaces/complements the LightGBM pipeline (events -> manual 52-feature vector -> GBDT) with a model that learns directly from raw event sequences.

## Architecture

**Model**: `KQModel` in `model.py`, vendored from [nanoGPT](https://github.com/karpathy/nanoGPT) with modifications.

Default config (adjustable via CLI):

| Parameter | Default |
|-----------|---------|
| Vocab size | 69 |
| Context length | 2560 tokens |
| Embedding dim | 256 |
| Layers | 6 |
| Heads | 4 |
| Parameters | ~4.7M |
| Dropout | 0.1 |

**Added over nanoGPT**: A linear win probability head (`n_embd -> 1`) applied at every position. Training loss is:

```
loss = CE_next_token + lambda_wp * BCE_win_prob
```

Default `lambda_wp = 0.1`.

## Tokenization

Each game event becomes 1-4 tokens, preceded by a time-gap token. The vocabulary is small (69 tokens) to let the transformer compose event semantics from parts.

### Vocabulary (69 tokens)

| Range | Count | Description |
|-------|-------|-------------|
| 0-2 | 3 | Special: `<PAD>`, `<BOS>`, `<EOS>` |
| 3-6 | 4 | Maps: `map_day`, `map_night`, `map_dusk`, `map_twilight` |
| 7-8 | 2 | Side: `gold_left`, `gold_right` |
| 9-19 | 11 | Event types: `spawn`, `carryFood`, `berryDeposit`, `berryKickIn`, `playerKill`, `blessMaiden`, `useMaiden`, `getOnSnail`, `getOffSnail`, `snailEat`, `snailEscape` |
| 20-25 | 6 | Victory: `victory_{blue,gold}_{military,economic,snail}` |
| 26-35 | 10 | Players: `player_1` .. `player_10` |
| 36-40 | 5 | Maidens: `maiden_0` .. `maiden_4` |
| 41-42 | 2 | Teams: `team_blue`, `team_gold` |
| 43-44 | 2 | Bot flag: `is_bot`, `is_human` |
| 45-46 | 2 | Maiden type: `maiden_speed`, `maiden_wings` |
| 47-48 | 2 | Kick direction: `own_team_goal`, `opp_team_goal` |
| 49-51 | 3 | Kill type: `killed_queen`, `killed_soldier`, `killed_worker` |
| 52-60 | 9 | Snail position: `snail_far_L` .. `snail_center` .. `snail_far_R` |
| 61-68 | 8 | Time gaps: `time_gap_0` .. `time_gap_7` |

### Time-Gap Tokens

A time-gap token is inserted before every event, encoding the elapsed time since the previous event. Boundaries were empirically fit from ~598K inter-event gaps across 3 shards (median 0.46s, p99 3.8s):

| Token | Range | Frequency |
|-------|-------|-----------|
| time_gap_0 | [0, 0.05s) | ~12% |
| time_gap_1 | [0.05s, 0.15s) | ~12% |
| time_gap_2 | [0.15s, 0.35s) | ~18% |
| time_gap_3 | [0.35s, 0.65s) | ~19% |
| time_gap_4 | [0.65s, 1.0s) | ~14% |
| time_gap_5 | [1.0s, 1.5s) | ~12% |
| time_gap_6 | [1.5s, 2.5s) | ~9% |
| time_gap_7 | [2.5s+) | ~4% |

### Snail Position Discretization

Snail position uses 9 buckets: 4 left + center + 4 right, with per-map empirical quantile boundaries symmetric around 0.5. The center bucket [0.49, 0.51) captures the untouched-snail spike (~5-37% of events depending on map). Boundaries were computed from ~100K+ snail events per map across 10 shards.

| Map | Far L | Mid L | Near L | Close L | Center | Close R | Near R | Mid R |
|-----|-------|-------|--------|---------|--------|---------|--------|-------|
| Day | .195 | .295 | .384 | .49 | .49-.51 | .51 | .616 | .705 | .805 |
| Dusk | .200 | .312 | .411 | .49 | .49-.51 | .51 | .589 | .688 | .800 |
| Night | .267 | .358 | .434 | .49 | .49-.51 | .51 | .566 | .642 | .733 |
| Twilight | .308 | .402 | .457 | .49 | .49-.51 | .51 | .543 | .598 | .692 |

### Event -> Token Mapping

| Event | Tokens | Example |
|-------|--------|---------|
| Game start | `<BOS> map_X gold_left\|right` | `<BOS> map_day gold_left` |
| spawn | `time_gap spawn player_N is_bot\|human` | `time_gap_0 spawn player_7 is_human` |
| carryFood | `time_gap carryFood player_N` | `time_gap_2 carryFood player_4` |
| berryDeposit | `time_gap berryDeposit player_N` | `time_gap_1 berryDeposit player_3` |
| berryKickIn | `time_gap berryKickIn player_N own\|opp` | `time_gap_3 berryKickIn player_7 opp_team_goal` |
| playerKill | `time_gap playerKill killer killed type` | `time_gap_0 playerKill player_2 player_3 killed_worker` |
| blessMaiden | `time_gap blessMaiden maiden_K team_X` | `time_gap_4 blessMaiden maiden_2 team_gold` |
| useMaiden | `time_gap useMaiden player_N type` | `time_gap_5 useMaiden player_6 maiden_wings` |
| getOnSnail | `time_gap getOnSnail player_N snail_pos` | `time_gap_2 getOnSnail player_6 snail_close_R` |
| getOffSnail | `time_gap getOffSnail player_N snail_pos` | `time_gap_1 getOffSnail player_5 snail_near_L` |
| snailEat | `time_gap snailEat rider eaten snail_pos` | `time_gap_0 snailEat player_5 player_6 snail_near_L` |
| snailEscape | `time_gap snailEscape player_N snail_pos` | `time_gap_0 snailEscape player_6 snail_near_L` |
| victory | `time_gap victory_team_condition` | `time_gap_3 victory_blue_military` |
| End | `<EOS>` | `<EOS>` |

### What's Not Tokenized

- **x,y coordinates** for kills, berries, maidens — maiden identity is captured via `maiden_0..4`; berry deposits just track which player deposited
- **Berry slot identity** — we track who deposits, not which of the 12 slots

### Token Statistics

With time-gap tokens, games are ~1.36x longer than without (one time-gap per event, events average ~2.8 tokens).

From 100K games (107 shards):
- ~650 tokens/game average
- p99 ~2500 tokens (fits in block_size=2560)

## Data Pipeline

### `tokenize_games.py`

Three modes:

```bash
# Directory mode: tokenize all shards, split 90/10
python -m sequence_model.tokenize_games \
    --train-dir logged_in_games/ \
    --val-csv late_tournament_games/late_tournament_game_events.csv.gz

# Directory mode with game limit
python -m sequence_model.tokenize_games \
    --train-dir logged_in_games/ --max-games 100000 \
    --val-csv late_tournament_games/late_tournament_game_events.csv.gz

# Single-CSV mode
python -m sequence_model.tokenize_games \
    --train-csv logged_in_games/gameevents_000.csv.gz \
    --val-csv late_tournament_games/late_tournament_game_events.csv.gz
```

Outputs:
- `{split}.bin` — uint16 token IDs, all games concatenated
- `{split}_labels.bin` — uint8 win labels (0/1), parallel array same length as tokens

### Train/Val Split

In directory mode (`--train-dir`), games from all shards are collected, shuffled, and split 90/10. With `--val-csv`, the val set comes from a separate CSV instead.

## Training

### `train.py`

Vendored from nanoGPT with modifications for dual loss and game-aligned batching.

```bash
python -m sequence_model.train \
    --device cuda --compile \
    --batch-size 64 --block-size 2560 \
    --n-embd 256 --n-layer 6 --n-head 4 \
    --max-iters 2000 --warmup-iters 200 --lr-decay-iters 2000
```

Key features:
- **Game-aligned batching**: scans `.bin` for BOS tokens, samples whole games per batch element. Pads short games, truncates long ones.
- AdamW optimizer with cosine LR schedule and warmup (all configurable via CLI)
- Mixed precision (bfloat16/float16)
- `torch.compile` support (~300ms/iter after compilation)
- Tracks LM loss, WP loss, and WP accuracy separately
- Checkpoint resume via `--init-from resume`
- wandb logging via `--wandb`

### Training Config Defaults

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Min LR | 3e-5 |
| Weight decay | 0.1 |
| Warmup iters | 200 |
| LR decay iters | 2000 |
| Max iters | 2000 |
| Grad clip | 1.0 |
| lambda_wp | 0.1 |
| Block size | 2560 |
| Batch size | 64 |

## Evaluation

### `compare_models.py`

Head-to-head comparison of seq model vs LightGBM on the same test games, with per-game-stage breakdown.

```bash
python -m sequence_model.compare_models \
    --test-csv late_tournament_games/late_tournament_game_events.csv.gz \
    --lgb-train-csv logged_in_games/gameevents_000.csv.gz \
    --block-size 2560
```

### `evaluate.py`

Standalone evaluation on val set:

```bash
python -m sequence_model.evaluate \
    --checkpoint sequence_model/out/ckpt.pt \
    --device cuda
```

## File Structure

```
sequence_model/
├── __init__.py              # Package marker
├── vocab.py                 # Token vocabulary (69 tokens) and event->token functions
├── config.py                # Model and training hyperparameter dataclasses
├── tokenize_games.py        # CSV events -> .bin token + label files
├── model.py                 # KQModel: GPT-2 transformer + win probability head
├── train.py                 # Training loop with dual loss, game-aligned batching
├── evaluate.py              # Evaluation metrics and baseline comparison
├── compare_models.py        # Head-to-head seq model vs LightGBM comparison
├── sequence_model_design_plan.md  # This file
├── TRAINING_REPORT.md       # Training experiments and results
├── SWEEP_REPORT.md          # Hyperparameter sweep analysis
├── data/                    # [gitignored] .bin token and label files
└── out/                     # [gitignored] Model checkpoints
```

## Future Work

- Train on full dataset (~180K games from all 199 shards)
- Add discretized spatial tokens for kill locations (map control)
- Add berry-slot tokens to berryDeposit (track proximity to economic victory)
- Team-swap data augmentation (double effective dataset)
- WP-head-specific regularization (dropout before wp_head)
- Per-stage accuracy breakdown (early/mid/late game)
