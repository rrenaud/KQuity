# TabNet Next Steps

## What we've established

- **Scaling works up to a point.** Accuracy goes from 67.3% (1k games) to 69.3% (50k games), then plateaus. More data beyond 50k doesn't help accuracy.
- **Egg inversions keep improving with data.** 9.3% at 1k down to 0.65% at 205k — the model learns monotonicity with scale even when accuracy stops improving.
- **Log loss plateaus around 0.566.** Reached at 50k games, unchanged through 205k.
- **Architecture doesn't matter much.** 16-16-5 matches 8-8-3 at the same data size. The ~69% accuracy ceiling appears to be a feature/representation limit, not a model capacity limit.
- **TabNet peaks in 0-4 epochs.** Long training just overfits.
- **Bugs found and fixed:** early stopping was using accuracy (last eval_metric) instead of logloss; train/test overlap now excluded.

## Completed from original plan

- [x] Scale up training data (1k through 205k)
- [x] Use CUDA (RTX 3090)
- [x] Hyperparameter tuning (n_d/n_a 8-32, n_steps 3-5; sweep running)
- [x] Resolve train/test overlap

## What to try next

### 1. Feature engineering (highest potential)

The 69% accuracy ceiling with both TabNet and LightGBM (67.8% at 1k) suggests the 52-feature representation is the bottleneck, not the model. Ideas:

- **Temporal features:** Time since last kill, time since last berry deposit, game clock. The model currently sees a snapshot with no sense of momentum.
- **Derived ratios:** Berry differential / total berries remaining, kill differential, food count differential. These relative features may be easier to learn from than absolute counts.
- **History features:** Rolling averages of state changes over the last N events. The current per-event vectorization throws away trajectory information.
- **Queen-specific features:** Queen lives are the most decisive game state. Explicit queen-alive flags per team, queen kill differential, lives remaining ratio.

### 2. Learning rate schedule

The current StepLR (step_size=50, gamma=0.9) barely activates since models peak in 0-4 epochs. Try:

- **Cosine annealing** with a warm restart
- **Higher initial LR** (0.05-0.1) with aggressive decay — the model may need to move faster in early epochs
- **One-cycle policy** which peaks then decays within a single run

### 3. Regularization instead of early stopping

Models overfit very quickly (best at epoch 0-4). Instead of relying on early stopping, try training longer with stronger regularization:

- Increase `lambda_sparse` (currently 1e-3) — try 1e-2, 1e-1
- Add dropout via `momentum` parameter
- Increase `virtual_batch_size` closer to `batch_size` to reduce ghost batch norm noise

### 4. Ensemble with LightGBM

TabNet and LightGBM may capture different patterns. A simple average or stacking ensemble could beat both individual models. TabNet has better calibration (lower log loss), LightGBM has slightly better accuracy at small data — they're likely complementary.

### 5. Compare against the sequence model

The `seq_model` branch has a transformer-based sequence model. This is the natural comparison: does modeling the full event sequence beat the per-event snapshot approach? The 69% accuracy ceiling may exist precisely because per-event features lose temporal context.

### 6. State dropping as regularization

`fast_materialize` supports `drop_state_probability` — randomly dropping training states. This could act as data augmentation, especially at smaller data sizes where the model overfits early. Try drop_prob 0.5-0.9 at 50k games.
