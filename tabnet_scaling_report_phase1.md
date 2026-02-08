# TabNet Scaling Experiment Results (Phase 1)

## Setup
- **Test set:** late_tournament_games (693 games, 161K vectors)
- **Train/test overlap:** Excluded from training
- **Device:** NVIDIA GeForce RTX 3090 (CUDA)
- **Early stopping:** accuracy-based (pytorch-tabnet default: last eval_metric)
- **Note:** Early stopping used accuracy, not logloss — fixed in Phase 2

## Results

| Config | Train Games | Train Vectors | Log Loss | Accuracy | Egg Inv. | Best Epoch | Train Time |
|--------|------------|---------------|----------|----------|----------|------------|------------|
| 1k/8-8-3 | 1,000 | 196,476 | 0.5981 | 67.3% | 9.3% | 1 | 1162s |
| 5k/8-8-3 | 5,000 | 909,467 | 0.6293 | 67.6% | 5.3% | 4 | 1197s |
| 10k/8-8-3 | 10,000 | 1,771,487 | 0.5839 | 68.3% | 1.9% | 1 | 775s |
| 50k/8-8-3 | 50,000 | 7,876,625 | 0.5656 | 69.3% | 1.9% | 0 | 2428s |
| 100k/8-8-3 | 100,000 | 14,681,844 | 0.5669 | 69.1% | 1.6% | 0 | 4666s |
| 205k/8-8-3 | 205,000 | 26,880,446 | 0.5668 | 69.1% | 0.7% | 1 | 14465s |
| 10k/16-16-5 | 10,000 | 1,771,487 | 0.5798 | 69.0% | 2.2% | 4 | 1278s |
| 50k/16-16-5 | 50,000 | 7,876,625 | 0.5719 | 69.1% | 0.7% | 1 | 4027s |

## Key Findings

### Data scaling (8-8-3 architecture)

- **1k games:** acc=67.3%, ll=0.5981, inv=9.3%
- **5k games:** acc=67.6%, ll=0.6293, inv=5.3%
- **10k games:** acc=68.3%, ll=0.5839, inv=1.9%
- **50k games:** acc=69.3%, ll=0.5656, inv=1.9%
- **100k games:** acc=69.1%, ll=0.5669, inv=1.6%
- **205k games:** acc=69.1%, ll=0.5668, inv=0.7%

### Architecture comparison at matched data sizes

- **10k games:** 8-8-3 acc=68.3% vs 16-16-5 acc=69.0%
- **50k games:** 8-8-3 acc=69.3% vs 16-16-5 acc=69.1%

### Observations

1. **Accuracy saturates around 69%** — going from 50k to 205k games barely helps with 8-8-3
2. **Egg inversions scale beautifully** — from 9.4% (1k) to 0.65% (205k)
3. **Larger architecture (16-16-5) doesn't improve** over 8-8-3 at matched data sizes
4. **Models peak at epoch 0-4** — patience wastes many epochs
5. **Early stopping was using accuracy** not logloss (pytorch-tabnet uses last eval_metric)
