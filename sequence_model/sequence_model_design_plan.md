# Sequence Model for Killer Queen Win Prediction

## Overview

A GPT-2 style transformer trained on game event sequences with dual objectives:
1. **Next-token prediction** — learn game dynamics by predicting the next event
2. **Win probability** — predict P(blue wins) at every position via an auxiliary head

This replaces/complements the LightGBM pipeline (events → manual 52-feature vector → GBDT) with a model that learns directly from raw event sequences.

## Architecture

**Model**: `KQModel` in `model.py`, vendored from [nanoGPT](https://github.com/karpathy/nanoGPT) with modifications.

| Parameter | Value |
|-----------|-------|
| Vocab size | 62 |
| Context length | 1024 tokens |
| Embedding dim | 128 |
| Layers | 4 |
| Heads | 4 |
| Parameters | ~0.8M |
| Dropout | 0.1 |

**Added over nanoGPT**: A linear win probability head (`n_embd → 1`) applied at every position. Training loss is:

```
loss = CE_next_token + lambda_wp * BCE_win_prob
```

Default `lambda_wp = 0.1`.

## Tokenization

Each game event becomes 1-4 tokens from a 62-token vocabulary. This keeps the vocabulary small while preserving all game-relevant information.

### Vocabulary (62 tokens)

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
| 52-61 | 10 | Snail position deciles: `snail_p0` .. `snail_p9` |

### Event → Token Mapping

| Event | Tokens | Example |
|-------|--------|---------|
| Game start | `<BOS> map_X gold_left\|right` | `<BOS> map_day gold_left` |
| spawn | `spawn player_N is_bot\|is_human` | `spawn player_7 is_human` |
| carryFood | `carryFood player_N` | `carryFood player_4` |
| berryDeposit | `berryDeposit player_N` | `berryDeposit player_3` |
| berryKickIn | `berryKickIn player_N own\|opp` | `berryKickIn player_7 opp_team_goal` |
| playerKill | `playerKill killer killed type` | `playerKill player_2 player_3 killed_worker` |
| blessMaiden | `blessMaiden maiden_K team_X` | `blessMaiden maiden_2 team_gold` |
| useMaiden | `useMaiden player_N type` | `useMaiden player_6 maiden_wings` |
| getOnSnail | `getOnSnail player_N snail_pN` | `getOnSnail player_6 snail_p5` |
| getOffSnail | `getOffSnail player_N snail_pN` | `getOffSnail player_5 snail_p3` |
| snailEat | `snailEat rider eaten snail_pN` | `snailEat player_5 player_6 snail_p3` |
| snailEscape | `snailEscape player_N snail_pN` | `snailEscape player_6 snail_p3` |
| victory | `victory_team_condition` | `victory_blue_military` |
| End | `<EOS>` | `<EOS>` |

### Snail Position Discretization

The snail track is centered at x=960 (SCREEN_WIDTH/2). Track width varies by map (700-900px). The raw `snail_x` pixel coordinate is normalized to [0, 1] within the track range and bucketed into 10 deciles (`snail_p0` = leftmost, `snail_p9` = rightmost). The model knows `gold_left`/`gold_right` from the `<BOS>` header, so it can interpret which direction is which team's goal.

### What's Not Tokenized

- **x,y coordinates** for kills, berries, maidens — maiden identity is captured via `maiden_0..4`; berry deposits just track which player deposited
- **Timestamps** — ordering is implicit in sequence position
- **Berry slot identity** — we track who deposits, not which of the 12 slots

### Statistics

From 1000 games (partition 000):
- Mean: 317 tokens/game
- Median: 277
- p95: 647, p99: 905, max: 1918

Tournament games are longer (mean 755 tokens).

## Data Pipeline

### `tokenize_games.py`

Reads partitioned CSV files via `preprocess.iterate_events_from_csv()`, validates games, tokenizes, and writes:
- `{split}.bin` — uint16 token IDs, all games concatenated
- `{split}_labels.bin` — uint8 win labels (0/1), parallel array same length as tokens

Every position in the label array holds the blue_wins outcome for whichever game that token belongs to. This allows the training loop to sample random chunks without game boundary alignment.

```bash
# Tokenize 1000 games for train, tournament games for val
python -m sequence_model.tokenize_games \
  --max-games 1000 \
  --val-csv late_tournament_games/late_tournament_game_events.csv.gz

# Full dataset
python -m sequence_model.tokenize_games
```

### Train/Val Split

Default: partitions 0-739 train, 740-924 val (same as LightGBM pipeline). Can override with `--val-csv` to use tournament data for evaluation.

## Training

### `train.py`

Vendored from nanoGPT with modifications for dual loss and our data format.

```bash
python -m sequence_model.train \
  --device cuda --dtype bfloat16 \
  --batch-size 64 --block-size 256 \
  --max-seconds 60
```

Key features:
- AdamW optimizer with cosine LR schedule and warmup
- Mixed precision (bfloat16/float16)
- `--max-seconds` flag for time-limited runs
- Tracks LM loss, WP loss, and WP accuracy separately
- Checkpoint resume via `--init-from resume`

### Training Config Defaults

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Weight decay | 0.1 |
| Warmup iters | 1000 |
| Max iters | 50,000 |
| Grad clip | 1.0 |
| lambda_wp | 0.1 |

## Evaluation

### `evaluate.py`

```bash
python -m sequence_model.evaluate \
  --checkpoint sequence_model/out/ckpt.pt \
  --device cuda
```

Metrics:
1. **Next-token perplexity** — how well the model predicts game dynamics
2. **Win probability accuracy and log loss** — compared to LightGBM baseline (70.4% acc, 0.556 log loss)
3. **Calibration** — reliability diagram (predicted prob vs actual win rate by decile)
4. **Egg inversion test** — does a queen kill shift P(blue wins) in the correct direction?

### Preliminary Results (60s training, 1000 games, eval on tournament data)

| Metric | Sequence Model | LightGBM Baseline |
|--------|---------------|-------------------|
| Win prob accuracy (val) | 53.4% | 70.4% |
| Win prob log loss (val) | 0.691 | 0.556 |
| Next-token perplexity (val) | 4.68 | N/A |
| Egg inversion | 71.9% | — |

The model is undertrained (1000 games, 60 seconds). Full training on 740K games for 50K iterations is needed to close the gap.

## File Structure

```
sequence_model/
├── __init__.py              # Package marker
├── vocab.py                 # Token vocabulary (62 tokens) and event→token functions
├── config.py                # Model and training hyperparameter dataclasses
├── tokenize_games.py        # CSV events → .bin token + label files
├── model.py                 # KQModel: GPT-2 transformer + win probability head
├── train.py                 # Training loop with dual loss
├── evaluate.py              # Evaluation metrics and baseline comparison
├── sequence_model_design_plan.md  # This file
├── data/                    # [gitignored] .bin token and label files
└── out/                     # [gitignored] Model checkpoints
```

## Future Work

- Train on full dataset (740K games) for 50K+ iterations
- Add time-gap bucket tokens between events (game pace/urgency)
- Add discretized spatial tokens for kill locations (map control)
- Add berry-slot tokens to berryDeposit (track proximity to economic victory)
- Scale up model if accuracy plateaus (more layers/heads/embedding dim)
- Compare sequence model win prob vs LightGBM at different game stages (early/mid/late)
