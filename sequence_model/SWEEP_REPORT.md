# Training Sweep Report: 10M Param Model on 10K Games

## Run Configuration

| Setting | Value |
|---------|-------|
| Model | n_embd=320, n_layer=8, n_head=8 (9.86M params) |
| Training data | 9,000 games / 7.5M tokens (11 shards) |
| Val data | 1,000 games / 846K tokens (same distribution) |
| Test data | 693 games / 699K tokens (late tournament, OOD) |
| Batch | 64 x 2560 = 164K tokens/iter |
| LR | 3e-4 peak, cosine -> 3e-5, warmup 200 iters |
| Total iters | 2,000 (~30 min with torch.compile) |

## Results

```
Step  | LM_gap | WP_gap | Train_acc | Val_acc | Acc_gap
------|--------|--------|-----------|---------|--------
    0 | +.0000 | +.0014 |    50.0%  |  49.7%  |  +0.3%
  600 | +.0051 | +.0140 |    62.9%  |  62.4%  |  +0.5%
 1200 | +.0106 | +.0335 |    66.1%  |  63.2%  |  +2.9%
 1600 | +.0127 | +.0475 |    68.1%  |  64.8%  |  +3.3%  <-- best val
 2000 | +.0111 | +.1180 |    71.2%  |  62.8%  |  +8.4%
```

Best val wp_acc: **64.8%** at step 1600 (up from 56% baseline).

## Overfitting Analysis

The model cycled through the training data **~43 times** (328M tokens seen / 7.5M
tokens in training set). The overfitting is almost entirely in the WP head:

- **LM loss gap stays small**: +0.011 at step 2000. The language model learns
  general event-sequence patterns that generalize well.
- **WP loss gap blows up**: +0.118 at step 2000. The WP head memorizes which
  team wins each training game. With 9K games and a 9.86M param model (~1,100
  params per game), there's enormous capacity to memorize outcomes.
- **Val acc is noisy**: It jumps around between 60-65% across evals, suggesting
  the 1,000-game val set has high variance.

## Ideas to Try

### HIGH IMPACT — More data (addresses root cause)

1. **Use all 199 shards (~180K games)**. This is the single biggest lever.
   The model has 1,096 params/game right now; with 180K games that drops to ~55
   params/game. The 43-epoch cycling would drop to <3 epochs. This alone should
   close most of the overfitting gap.

2. **Shuffle games across shards before splitting**. Currently the 90/10 split
   is by game order within the shard files. Games from the same session/day are
   adjacent, so the val split may be from different time periods than training.
   A random shuffle before splitting would give a more representative val set.

### HIGH IMPACT — Regularization

3. **Increase dropout** (currently 0.1). Try 0.2 or 0.3. The WP head has no
   dropout of its own — it reads directly from the final hidden state. Dropout
   in the transformer backbone is the only regularization on the WP prediction
   path.

4. **WP-head-specific dropout**. Add a dropout layer before the `wp_head`
   linear layer. The LM head doesn't overfit (small gap), but the WP head does,
   so targeted regularization makes sense.

5. **Increase weight decay** (currently 0.1). Try 0.3. This penalizes large
   weights which helps prevent memorization.

### MEDIUM IMPACT — Architecture / training

6. **Reduce model size for small-data regime**. Until all shards are used, a
   smaller model (2-4M params) may generalize better. The 0.80M model from the
   quick test didn't overfit in 50 iters. Run the same 2K iters with n_embd=192,
   n_layer=6, n_head=6 (~3M params) and compare val curves.

7. **Early stopping**. The best val loss was at step 2000 but best val wp_acc
   was at step 1600. Could train longer with early stopping on val wp_acc
   specifically (save checkpoint whenever val wp_acc improves, stop after N
   evals without improvement).

8. **Separate LR or schedule for WP head**. The WP head overfits faster than
   the LM head. A lower learning rate or earlier decay for the WP parameters
   could help. Alternatively, freeze the WP head for the first N iters to let
   the backbone learn good representations first.

### MEDIUM IMPACT — Data augmentation

9. **Team-swap augmentation**. For each game, create a mirror where blue/gold
   are swapped (swap player tokens 1-5 <-> 6-10, swap team tokens, flip labels).
   This doubles the effective dataset and enforces the symmetry prior that the
   model should be team-invariant.

10. **Random game truncation during training**. Instead of always starting from
    BOS, randomly truncate the first N% of tokens (while still starting at a
    time-gap boundary). This creates more diverse training contexts and prevents
    the model from relying on early-game patterns to memorize outcomes.

### LOWER IMPACT — WP head design

11. **Reduce lambda_wp** (currently 0.1). The WP signal is noisy (predicting
    the winner from partial game state is inherently uncertain). A smaller
    lambda_wp (0.01-0.05) would make the WP head a lighter auxiliary objective,
    letting the backbone focus on learning good sequence representations via the
    LM loss, which generalizes better.

12. **Progressive lambda_wp**. Start with lambda_wp=0 (pure LM pretraining),
    then linearly ramp it up over the last half of training. This gives the
    backbone time to learn event dynamics before the WP head starts pulling
    representations toward outcome prediction.

### INVESTIGATION — Understanding the gap

13. **Per-stage accuracy breakdown on val**. Is the overfitting concentrated in
    early-game predictions (where the WP signal is weakest) or late-game? If
    early-game, the model may be memorizing game-start patterns → outcomes. If
    late-game, it may be memorizing specific endgame sequences.

14. **Compare val vs test performance**. The test set is late_tournament (OOD).
    If val accuracy >> test accuracy, there's a distribution shift issue on top
    of overfitting. Run `compare_models.py` with the step-1600 checkpoint.

## Recommended Next Steps

1. **Immediate**: Retrain with all 199 shards (the data is there, just needs
   `--train-dir logged_in_games/` without `--max-games`). This should be the
   biggest single improvement.
2. **Quick experiment**: Same 10K games but dropout=0.2 + weight_decay=0.3 to
   see how much regularization helps in the small-data regime.
3. **After full-data training**: Run `compare_models.py` to get the head-to-head
   vs LightGBM with per-stage breakdown.
