# TabNet Hyperparameter Sweep Results

## Setup
- **Training data:** 50,000 games (~7.9M feature vectors)
- **Test set:** late_tournament_games (693 games, 161K vectors)
- **Train/test overlap:** Excluded from training
- **Early stopping:** logloss-based, patience=3, max_epochs=20
- **Batch size:** 16384, virtual batch size: 256
- **Device:** CUDA (RTX 3090)

## Log Loss (lower is better)

| n_d/n_a | steps=3 | steps=5 | steps=7 |
|---------|----------|----------|----------|
| **8** | 0.5656 | 0.5698 | 0.5698 |
| **16** | 0.5658 | 0.5691 | 0.5932 |
| **32** | 0.5687 | 0.5658 | 0.5694 |
| **64** | 0.5682 | 0.5695 | — |

## Accuracy (higher is better)

| n_d/n_a | steps=3 | steps=5 | steps=7 |
|---------|----------|----------|----------|
| **8** | 69.3% | 69.2% | 69.0% |
| **16** | 69.3% | 68.9% | 67.6% |
| **32** | 68.8% | 69.2% | 69.2% |
| **64** | 69.0% | 68.8% | — |

## Egg Inversions (lower is better)

| n_d/n_a | steps=3 | steps=5 | steps=7 |
|---------|----------|----------|----------|
| **8** | 1.9% | 0.9% | 1.6% |
| **16** | 0.9% | 0.7% | 2.6% |
| **32** | 0.3% | 0.9% | 2.5% |
| **64** | 1.8% | 0.8% | — |

## Training Time

| n_d/n_a | steps=3 | steps=5 | steps=7 |
|---------|----------|----------|----------|
| **8** | 2361s (ep 0) | 3939s (ep 1) | 6478s (ep 7) |
| **16** | 2312s (ep 0) | 4419s (ep 5) | 5244s (ep 0) |
| **32** | 2305s (ep 0) | 4040s (ep 3) | 6054s (ep 5) |
| **64** | 2285s (ep 0) | 4133s (ep 3) | — |

## Full-data runs (205k games, ~27M vectors)

| Config | Log Loss | Accuracy | Egg Inv. | Best Epoch | Train Time |
|--------|----------|----------|----------|------------|------------|
| **6-6-2** | **0.5660** | **69.2%** | 1.2% | 4 | 11396s |
| **8-8-3** | 0.5668 | 69.1% | **0.65%** | 1 | 17574s |

6-6-2 slightly edges out 8-8-3 on log loss and accuracy, trains 35% faster, and peaked later (epoch 4 vs 1) suggesting better regularization. 8-8-3 has better egg inversions. Both confirm the ~69% accuracy ceiling regardless of architecture or data scale.

## Best Configurations

- **Best log loss:** 8-8-3 at 50k (0.5656), 6-6-2 at 205k (0.5660)
- **Best accuracy:** 8-8-3 at 50k (69.3%)
- **Fewest egg inversions:** 32-32-3 at 50k (0.3%), 8-8-3 at 205k (0.65%)
