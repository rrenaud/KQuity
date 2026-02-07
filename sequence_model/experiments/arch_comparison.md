# Architecture Comparison Experiment

Compare transformer, linear attention, and Mamba on KQ win-prediction.

## Data

- 65M train tokens, 6.2M val tokens (90K games, avg 723 tokens/game)
- Vocab size 69, block size 2560

## Design Principles

**Param-matched**: all runs in a tier have ~equal total parameter count.
Since Mamba's `expand=2` inflates params relative to transformer/linear-attn
at the same `n_embd × n_layer`, we shrink Mamba's width or depth to
compensate.

**Compute-matched**: at equal param count the 6×params×tokens FLOP
approximation is also matched (~32M FLOPs/tok for the small tier).
Wall-clock will differ — Mamba's pure-PyTorch scan is slower per step,
linear-attn's cumsum outer products use more memory — but total FLOPs
trained on are comparable.

**Two tiers** to check whether conclusions hold across scale:

| Tier  | Params | Train tokens seen |
|-------|--------|-------------------|
| Tiny  | ~1.1M  | ~470M (7200 iters × 64 batch × 2560 seq × 0.4 fill) |
| Small | ~5.4M  | ~470M (same budget) |

`0.4 fill` accounts for PAD in variable-length games (avg 723 / 2560 ≈ 0.28,
but some games are longer; empirical fill is ~0.4 with the batch sampler).

## Configurations

### Tiny tier (~1.1M params, 3000 iters)

| Run name          | model_type   | n_embd | n_layer | n_head | Params |
|-------------------|-------------|--------|---------|--------|--------|
| tiny-transformer  | transformer  | 128    | 4       | 4      | 1.12M  |
| tiny-linear-attn  | linear-attn  | 128    | 4       | 4      | 1.12M  |
| tiny-mamba        | mamba        | 112    | 6       | 4      | 1.14M  |

### Small tier (~5.4M params, 5000 iters)

| Run name           | model_type   | n_embd | n_layer | n_head | Params |
|--------------------|-------------|--------|---------|--------|--------|
| small-transformer  | transformer  | 256    | 6       | 4      | 5.40M  |
| small-linear-attn  | linear-attn  | 256    | 6       | 4      | 5.40M  |
| small-mamba        | mamba        | 224    | 7       | 4      | 5.12M  |

## Hyperparameters (shared across all runs)

| Param                  | Value  | Notes                              |
|------------------------|--------|------------------------------------|
| batch_size             | 64     |                                    |
| block_size             | 2560   |                                    |
| learning_rate          | 3e-4   |                                    |
| min_lr                 | 3e-5   |                                    |
| warmup_iters           | 200    |                                    |
| weight_decay           | 0.1    |                                    |
| dropout                | 0.1    |                                    |
| lambda_wp              | 0.1    |                                    |
| grad_clip              | 1.0    |                                    |
| dtype                  | bfloat16 |                                  |
| compile                | true   | false for mamba (graph breaks)     |
| lr_decay_iters         | = max_iters |                              |
| eval_interval          | 250    |                                    |

## Commands

```bash
# --- Tiny tier (3000 iters) ---

python -m sequence_model.train --model-type transformer \
    --n-embd 128 --n-layer 4 --n-head 4 \
    --max-iters 3000 --lr-decay-iters 3000 --eval-interval 250 \
    --out-dir sequence_model/out/tiny-transformer \
    --wandb --wandb-run-name tiny-transformer

python -m sequence_model.train --model-type linear-attn \
    --n-embd 128 --n-layer 4 --n-head 4 \
    --max-iters 3000 --lr-decay-iters 3000 --eval-interval 250 \
    --out-dir sequence_model/out/tiny-linear-attn \
    --wandb --wandb-run-name tiny-linear-attn

python -m sequence_model.train --model-type mamba \
    --n-embd 112 --n-layer 6 --n-head 4 \
    --max-iters 3000 --lr-decay-iters 3000 --eval-interval 250 \
    --compile=false \
    --out-dir sequence_model/out/tiny-mamba \
    --wandb --wandb-run-name tiny-mamba

# --- Small tier (5000 iters) ---

python -m sequence_model.train --model-type transformer \
    --n-embd 256 --n-layer 6 --n-head 4 \
    --max-iters 5000 --lr-decay-iters 5000 --eval-interval 250 \
    --out-dir sequence_model/out/small-transformer \
    --wandb --wandb-run-name small-transformer

python -m sequence_model.train --model-type linear-attn \
    --n-embd 256 --n-layer 6 --n-head 4 \
    --max-iters 5000 --lr-decay-iters 5000 --eval-interval 250 \
    --out-dir sequence_model/out/small-linear-attn \
    --wandb --wandb-run-name small-linear-attn

python -m sequence_model.train --model-type mamba \
    --n-embd 224 --n-layer 7 --n-head 4 \
    --max-iters 5000 --lr-decay-iters 5000 --eval-interval 250 \
    --compile=false \
    --out-dir sequence_model/out/small-mamba \
    --wandb --wandb-run-name small-mamba
```

## Metrics to Compare

1. **Val win-prediction accuracy** (`wp_acc`) — primary metric
2. **Val WP loss** (`wp_loss`) — calibration signal
3. **Val LM loss** (`lm_loss`) — how well it models game dynamics
4. **Val total loss** — combined objective
5. **Wall-clock time per iteration** — practical cost
6. **Learning curves** — convergence speed (loss vs. iteration)

## What to Look For

- Does linear-attn or mamba learn better WP representations than softmax
  attention at matched params? (Check `wp_acc` gap at convergence.)
- Does any architecture converge faster in iterations? (Check loss at
  iter 1000 vs. final.)
- Does the ranking hold across tiny → small? If not, which architecture
  benefits more from scale?
- Is the mamba sequential scan wall-clock cost prohibitive, or tolerable
  at this model size?
