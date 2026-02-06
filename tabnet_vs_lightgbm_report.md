# TabNet vs LightGBM Comparison

## Experiment Setup

- **Train data:** `logged_in_games/gameevents_000.csv.gz` — 986 games (highest login count, highest quality), no state dropping, 201,568 feature vectors
- **Test data:** `late_tournament_games/late_tournament_game_events.csv.gz` — 693 late tournament games, 161,233 feature vectors
- **Features:** 52-dimensional game state vectors (from `fast_materialize`)
- **Note:** 21 games overlap between train and test sets (2.1% of train, 3.0% of test)

## Model Configurations

- **TabNet:** n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=0.02, batch_size=1024, patience=15. Early stopped at epoch 18, best epoch 3.
- **LightGBM (100L/100T):** 100 leaves, 100 trees, default gbdt boosting
- **LightGBM (200L/200T):** 200 leaves, 200 trees

## Results

| Metric         | TabNet (8/8/3) | LightGBM (100L/100T) | LightGBM (200L/200T) |
|----------------|----------------|-----------------------|-----------------------|
| Log Loss       | **0.6239**     | 0.6278                | 0.7394                |
| Accuracy       | 66.9%          | **67.8%**             | 66.7%                 |
| Egg Inversions | **14.5%**      | 16.9%                 | 20.0%                 |
| Train Time     | 257s           | **2.2s**              | 8.1s                  |

## Analysis

LightGBM (100L/100T) has the best accuracy at 67.8%, roughly 1 percentage point ahead of TabNet. However, TabNet achieves the best log loss (0.624 vs 0.628), suggesting better-calibrated probabilities. TabNet also has noticeably fewer egg inversions (14.5% vs 16.9%), meaning it better respects the monotonic relationship between egg count and win probability.

The larger LightGBM (200L/200T) is clearly overfitting with only ~1000 training games — its log loss degrades to 0.739 while accuracy drops below TabNet.

TabNet trains ~100x slower than LightGBM on CPU (257s vs 2.2s).

## Next Steps

- **Scale up training data.** This experiment used only 1 of 199 available `logged_in_games` partitions (~1000 of ~200k games). TabNet's relative performance may improve with more data since it has more capacity than a shallow gradient-boosted ensemble. Try `--num-train-games 5000`, `10000`, `50000`, etc.
- **Use CUDA for TabNet.** Training ran on CPU. TabNet's batch-based training should benefit significantly from GPU acceleration, especially at larger data scales. pytorch-tabnet auto-detects CUDA when available — just ensure `torch.cuda.is_available()` returns True.
- **Hyperparameter tuning.** TabNet has many knobs (n_d, n_a, n_steps, lr, lambda_sparse). The defaults here are conservative. Larger n_d/n_a (16 or 32) with more training data could improve results.
- **Resolve train/test overlap.** 21 games appear in both sets. Consider filtering these from one side for cleaner evaluation.
