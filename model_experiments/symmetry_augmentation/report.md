# Symmetry Augmentation Experiment

## Question

Does Blue/Gold symmetry augmentation provide structural value beyond
simply having more training data?

## Background

Killer Queen has perfect Blue/Gold symmetry: any game state with Blue
holding resources X and Gold holding resources Y is strategically
equivalent (from the opposite perspective) to Blue having Y and Gold
having X. The `symmetry.swap_teams(X, y)` function exploits this by
permuting the 52-feature vector (swapping blue[0:20] with gold[20:40],
negating maiden control and snail direction) and flipping the label.

This creates "free" training data, but the question is whether it
helps the model learn the invariance, or just acts like more rows.

## Design

Three conditions, all trained with LightGBM (200 leaves, 200 trees):

| Condition  | Construction                          | Rows    |
|------------|---------------------------------------|---------|
| X          | 100k random training rows             | 100,000 |
| X + sym(X) | same 100k + their symmetry-swapped copies | 200,000 |
| 2X         | 200k random training rows (X + 100k fresh) | 200,000 |

X + sym(X) and 2X have the same row count, so any difference isolates
the effect of symmetry structure vs. raw data volume.

Test set: 1,666,521 holdout rows (unchanged across conditions).

## Results

| Condition  | Train Size | Log Loss | Accuracy | Sym Consistency | Egg Inv |
|------------|------------|----------|----------|-----------------|---------|
| X (100k)   | 100,000    | 0.5780   | 68.7%    | 0.0829          | 7.15%   |
| X + sym(X) | 200,000    | 0.5708   | 69.0%    | 0.0396          | 5.75%   |
| 2X (200k)  | 200,000    | 0.5709   | 69.1%    | 0.0639          | 5.85%   |

**Symmetry consistency** = mean |P(state) + P(swap(state)) - 1|; 0 is perfect.

## Key Comparisons

| Comparison                    | Log Loss Delta | Sym Consistency |
|-------------------------------|----------------|-----------------|
| X+sym(X) vs X (augmentation helps?) | -0.0073       | 0.0396 vs 0.0829 (2.1x better) |
| X+sym(X) vs 2X (symmetry vs data?) | -0.0002       | 0.0396 vs 0.0639 (1.6x better) |
| 2X vs X (more data helps?)         | -0.0071       | 0.0639 vs 0.0829 (1.3x better) |

## Conclusions

1. **Log loss**: X+sym(X) and 2X are essentially tied (0.5708 vs 0.5709).
   Symmetry augmentation matches the predictive benefit of 2x real data.

2. **Symmetry consistency**: X+sym(X) is substantially better than 2X
   (0.0396 vs 0.0639). The augmented model's predictions are 1.6x more
   self-consistent under Blue/Gold swap. This is structural value that
   more raw data alone does not provide.

3. **Recommendation**: Use symmetry augmentation in production training.
   It's free (just a numpy permutation + sign flip), adds no real compute
   cost, and produces a model that better respects the game's inherent
   symmetry without sacrificing predictive accuracy.
