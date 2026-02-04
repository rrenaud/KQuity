# Tournament Weight Experiment Report

## Overview

This report summarizes experiments investigating how to optimally combine tournament and non-tournament game data for training a win prediction model. Tournament games represent expert-level play and are the target domain for evaluation, while non-tournament games provide significantly more training data (~30x more).

## Dataset Summary

- **Tournament training games**: ~35,000 games → 616,476 state samples
- **Non-tournament training games**: ~115,000 games → 7,909,105 state samples
- **Test set**: 2,000 most recent tournament games (held out by time)
- **State sampling**: 90% drop rate during training materialization

## Experiments Conducted

### Experiment 1: Doubling Non-Tournament Data

Progressively added non-tournament data starting from tournament-only baseline, doubling each iteration.

| Non-Tourn States | Total Train | Log Loss | Accuracy | Egg Inversions |
|------------------|-------------|----------|----------|----------------|
| 0 (baseline)     | 616,476     | 0.5664   | 69.46%   | 5.40%          |
| 616,476          | 1,232,952   | 0.5598   | 69.98%   | 2.90%          |
| 1,232,952        | 1,849,428   | 0.5584   | 69.91%   | 3.05%          |
| 2,465,904        | 3,082,380   | 0.5566   | 70.03%   | 1.95%          |
| 4,931,808        | 5,548,284   | 0.5561   | 70.12%   | 1.20%          |
| 7,909,105 (all)  | 8,525,581   | **0.5557** | 70.09% | 1.30%          |

**Key Finding**: Adding non-tournament data consistently improves performance. Using all available data yields the best log loss (0.5557), with diminishing returns after ~5M non-tournament states.

### Experiment 2: Non-Tournament Weight Sweep

Used all non-tournament data but applied sample weights to control their influence. Tournament samples always weighted at 1.0.

| Non-Tourn Weight | Eff. Tourn Fraction | Log Loss | Accuracy | Egg Inversions |
|------------------|---------------------|----------|----------|----------------|
| 0.01             | 88.6%               | 0.5631   | 69.64%   | 5.50%          |
| 0.02             | 79.6%               | 0.5620   | 69.71%   | 4.55%          |
| 0.05             | 60.9%               | 0.5605   | 69.75%   | 3.90%          |
| 0.10             | 43.8%               | 0.5588   | 69.87%   | 3.15%          |
| 0.15             | 34.2%               | 0.5578   | 69.99%   | 2.30%          |
| 0.20             | 28.0%               | 0.5573   | 70.10%   | 2.25%          |
| 0.30             | 20.6%               | 0.5568   | 70.08%   | 1.50%          |
| 0.50             | 13.5%               | 0.5563   | 70.04%   | 1.45%          |
| 0.75             | 9.4%                | 0.5561   | 70.13%   | 1.30%          |
| 1.00             | 7.2%                | **0.5557** | 70.09% | 1.05%          |

**Key Finding**: Equal weighting (1.0) yields the best log loss. There is no benefit to down-weighting non-tournament data; even with domain shift, more data helps.

## Conclusions

1. **More data is better**: Despite tournament being the target domain, adding non-tournament data improves tournament test performance. The model generalizes from casual games to expert play.

2. **No down-weighting needed**: The optimal weight for non-tournament samples is 1.0 (equal to tournament). Sample weighting provides no benefit over simply using all available data.

3. **Egg inversions correlate with log loss**: As log loss improves, the problematic "egg inversion" phenomenon decreases from 5.4% to ~1%, indicating better model calibration.

4. **Diminishing returns**: Most of the benefit comes from the first few million non-tournament samples. Going from 5M to 8M provides only marginal improvement (0.5561 → 0.5557).

## Recommendations

- **Production model**: Train on all available data with equal weights
- **Future experiments**: Consider exploring:
  - Time-based weighting (more recent games weighted higher)
  - Player skill-level weighting
  - Curriculum learning (train on tournament first, then expand)
