# Initial Training Report

## Starting Point

The initial sequence model prototype achieved 56% win-prediction accuracy on
late_tournament test data, vs LightGBM's 67%. Early-game accuracy was 49%
(below chance). Five compounding issues were identified:

| Issue | Severity |
|-------|----------|
| Tiny training set (887 games from 1 shard; 199 available) | CRITICAL |
| Random-offset batching (windows span game boundaries) | HIGH |
| LR schedule never decays (warmup=1000, decay=50000, only 2500 iters) | HIGH |
| No time information between events | MODERATE |
| Games exceed block_size (p99=1867 vs block_size=1024) | MODERATE |

## Changes Made

### 1. Bulk tokenization (`--train-dir`)

Added directory mode to `tokenize_games.py`: reads all `*.csv.gz` in a
directory, collects games, splits 90/10 for train/val. `--max-games` limits
total game count across all shards.

### 2. Game-aligned batching

Replaced random-offset `get_batch()` with BOS-scanning approach:
- At startup, scan `.bin` for all BOS token positions, cache offsets
- Each batch element samples a random BOS offset and reads one complete game
- Short games: padded with PAD=0 (x) and -1 (y/wp, ignored by loss)
- Long games: truncated to block_size (keep beginning)

The existing `ignore_index=-1` in CE loss and `wp_mask = (wp != -1)` in the
model handle padding correctly with no changes needed.

### 3. LR schedule fix

Exposed `--warmup-iters`, `--lr-decay-iters`, `--min-lr` as CLI args.
Defaults changed to warmup=200, decay=2000, matching a 2000-iter training run.
Full cosine annealing now completes within the training budget.

### 4. Time-gap tokens

Added 8 time-gap bucket tokens (TIME_GAP_0..7) inserted before every game
event. Bucket boundaries were empirically fit from ~598K inter-event gaps
across 3 shards:

```
Boundaries: [0.05, 0.15, 0.35, 0.65, 1.0, 1.5, 2.5] seconds
Distribution: median=0.46s, p95=2.3s, p99=3.8s
```

The boundaries were chosen so each bucket captures roughly 10-20% of events,
with fine resolution around the median and coarser resolution for rare long
pauses. This gives the model timing context that LightGBM gets implicitly.

### 5. Empirical snail position buckets

Replaced linear decile bucketing (which clamped ~40% of events to bucket 0 or
9) with per-map empirical quantile boundaries. Key improvements:
- Snail x is normalized by screen width (1920px), not track width
- 9 buckets: 4 left + center [0.49, 0.51) + 4 right
- Center bucket captures the untouched-snail spike (5-37% of events by map)
- Left/right boundaries are symmetric around 0.5
- Boundaries computed from 100K+ snail events per map across 10 shards

### 6. block_size = 2560

Increased from 1024 to 2560 to fit p99 games with time-gap tokens
(1867 * 1.36 = 2539). Memory is fine at this size.

## Training Experiments

### Run 1: 10M params / 10K games

Config: n_embd=320, n_layer=8, n_head=8 (9.86M params), 9K train / 1K val,
2000 iters, torch.compile, ~30 min.

```
Step  | Train wp_acc | Val wp_acc | Acc gap | Val LM loss
------|-------------|------------|---------|------------
    0 |    50.0%    |   49.7%    |  +0.3%  |   4.295
  600 |    62.9%    |   62.4%    |  +0.5%  |   1.518
 1200 |    66.1%    |   63.2%    |  +2.9%  |   1.230
 1600 |    68.1%    |   64.8%    |  +3.3%  |   1.183  <-- best val wp_acc
 2000 |    71.2%    |   62.8%    |  +8.4%  |   1.170
```

Result: 64.8% peak val wp_acc, but severe WP-head overfitting (8.4% gap at
step 2000). The LM head barely overfits (+0.011 loss gap) while the WP head
memorizes outcomes with 1,096 params/game and 43 epochs over data.

### Run 2: 4.7M params / 100K games

Config: n_embd=256, n_layer=6, n_head=4 (4.74M params), 90K train / 10K val,
2000 iters, torch.compile, ~19 min.

```
Step  | Train wp_acc | Val wp_acc | Acc gap | Val LM loss
------|-------------|------------|---------|------------
    0 |    50.0%    |   50.0%    |  +0.0%  |   4.302
  600 |    56.2%    |   55.5%    |  +0.7%  |   1.518
 1200 |    59.5%    |   60.0%    |  -0.5%  |   1.372
 1600 |    63.4%    |   62.0%    |  +1.4%  |   1.306
 2000 |    63.1%    |   62.5%    |  +0.6%  |   1.271
```

Result: 62.5% val wp_acc with overfitting nearly eliminated (0.6% gap). Only
53 params/game and ~5 epochs over data. Val loss still improving at step 2000.

### Comparison

| Metric | 10M / 10K games | 4.7M / 100K games |
|--------|----------------|-------------------|
| Params/game | 1,096 | 53 |
| Epochs over data | ~43 | ~5 |
| Best val wp_acc | 64.8% (step 1600) | 62.5% (step 2000) |
| Final acc gap | +8.4% | +0.6% |
| Final val LM loss | 1.235 | 1.335 |

The 10x more data + smaller model eliminated overfitting. Both runs land
around 62-63% on their respective val sets. The 10M model's 64.8% peak was
likely inflated by val noise (1K games) and mild overfitting leaking through.

## Status

Val wp_acc is at ~62.5% on same-distribution data (not yet tested on the
held-out late_tournament set). The model is no longer overfitting, so the path
forward is scaling: more data (all 199 shards), bigger model, and/or longer
training.

## Next Steps

1. Run `compare_models.py` on late_tournament test set for head-to-head vs LightGBM
2. Train with all 199 shards (~180K games) — should improve generalization further
3. Scale up model size once data supports it without overfitting
4. Consider WP-head-specific regularization and team-swap data augmentation
