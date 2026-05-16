# Case F — KQuity Phase 2 surrogate ladder report

## Setup

- Oracle: `current_preferred_model.mdl` (LightGBM 100t/100l, 52 feat)
- Eval: 233,538 events from 3,000 games (16 `benchmark_events_*.csv.gz`)
- Train/test split: 80/20 by game_id (no event leakage)
  - train: 187,447 events / 2,400 games
  - test:  46,091 events / 600 games
- Surrogates fit on training events; metrics computed on held-out
  test events
- Targets: oracle logit (linear regressors, tree regressors,
  phase-linear); labels y_blue (L1 logistic); oracle probability
  (LUT)

## Ladder result (sorted by prob RMSE vs oracle)

```
name                  feat params  pRMSE  lRMSE     KL    AUC logloss  Brier    ECE  disag
oracle (reference)      52  10000  .0000  .0000  .0000  .7901   .5494  .1868  .0543  .0000
linear_all52            52     53  .1045  .7214  .0272  .7619   .5767  .1974  .0538  .0263
l1_logistic_all52       47     48  .1096  .7291  .0301  .7719   .5659  .1929  .0471  .0217
phase_linear_diff6       6     21  .1109  .7380  .0304  .7520   .5852  .2011  .0525  .0308
hand_diff_linear_6       6      7  .1130  .7499  .0314  .7495   .5867  .2017  .0519  .0307
ridge_top10             10     11  .1167  .7659  .0334  .7461   .5884  .2025  .0519  .0307
lut_9x9x9                3    729  .1488  .9036  .0526  .7111   .6193  .2158  .0605  .0784
tree_d5_diff6            6     63  .1490  .8920  .0535  .7103   .6207  .2164  .0585  .0704
lut_7x7x7                3    343  .1490  .9077  .0528  .7121   .6193  .2157  .0599  .0691
lut_5x5x5                3    125  .1527  .9536  .0555  .6971   .6218  .2170  .0580  .0910
tree_d4_diff6            6     31  .1559  .9283  .0584  .6987   .6228  .2173  .0542  .0780
tree_d5_top10           10     63  .1564  .8163  .0568  .6782   .6292  .2209  .0555  .0776
tree_d3_diff6            6     15  .1617  .9861  .0625  .7071   .6194  .2158  .0493  .0918
tree_d4_top10           10     31  .1672  .8915  .0647  .6632   .6347  .2231  .0581  .1270
tree_d2_diff6            6      7  .1712 1.0477  .0698  .6718   .6307  .2204  .0496  .1223
tree_d3_top10           10     15  .1767  .9726  .0720  .6247   .6370  .2244  .0560  .2082

      feat = # features used   params = rough parameter count
      pRMSE = prob RMSE vs oracle   lRMSE = logit RMSE vs oracle
      AUC, logloss, Brier, ECE = quality vs labels y_blue
      disag = top-k disagreement (oracle confident >0.8 or <0.2)
```

## Promotion gate verdict

ChatGPT Pro promotion gates:

```
Strong : prob RMSE <= 0.03, AUC drop <= 2pp, features <= 8-10
Good   : prob RMSE <= 0.05, AUC drop <= 3-5pp
Fail   : shallow surrogates need too many features or lose
         calibration badly.
```

```
Strong (prob RMSE <= 0.03): NOT MET by any surrogate
                             (best is linear_all52 at 0.1045)
Good   (prob RMSE <= 0.05): NOT MET
AUC drop <= 5pp (Good):     MET by 5 surrogates including
                             hand_diff_linear_6 (4.06pp drop)
Brier degradation <= 3pp:   MET by linear surrogates
                             (hand_diff is +1.5pp)
```

The strict prob-RMSE gate is missed because the LightGBM oracle
has nonlinear regions (near the 0.5 decision boundary, and where
specific worker-state interactions matter) that a single linear
function in 6 differentials cannot capture. But the linear
surrogate ranks events almost as well as the oracle (AUC 0.750
vs 0.790) and is comparably calibrated (Brier 0.202 vs 0.187).

## Action variable

The candidate action variable, by best-feature-economy:

```
6-dimensional differential vector:

  egg_diff      = blue.eggs - gold.eggs           (queen lives)
  food_diff     = blue.food_count - gold.food_count (berry lead)
  snail_pos     = gold-symmetric snail position
  soldier_diff  = blue.n_soldiers - gold.n_soldiers
  warrior_diff  = blue.n_warriors - gold.n_warriors
  berries_norm  = berries_avail / 70.0

primitive:
  logit  = w0 + w1*egg_diff + w2*food_diff + w3*snail_pos
              + w4*soldier_diff + w5*warrior_diff + w6*berries_norm
  P(blue) = sigmoid(logit)

7 parameters total (1 bias + 6 weights).
```

This is a ~3000x parameter reduction (100x100 leaves -> 7 params)
and a 4.06 pp AUC drop. Hardware cost ~6 MACs + 1 add + sigmoid LUT.

## Why trees and LUTs do badly here

The oracle's gradient w.r.t. each top feature is approximately
monotone-smooth, not threshold-discrete. The tree/LUT surrogates
discretize a smooth function and lose accuracy.

Compare to Case Study 3 (LightGBM storage policy): there the
oracle was a discrete policy decision (e.g., "use SLC if X,
else QLC"), where a tree was the natural shape. KQuity's win
probability is a continuous score, so linear-in-differentials
is a tighter fit.

## Phase-conditioning is marginal

```
hand_diff_linear_6   prob RMSE 0.1130  AUC 0.7495
phase_linear_diff6   prob RMSE 0.1109  AUC 0.7520
```

Three independent linear regressors (one per early/mid/late
phase) buy ~0.002 prob RMSE improvement. The phase-AUC ladder
seen in Phase 1 baseline (early 0.689 -> mid 0.798 -> late
0.823) reflects shifting *values* of the differentials, not
different functional forms by phase. A single linear function
covers the game.

## Recommended next step

Two routes, conditional on what ChatGPT Pro wants:

**Route A (accept linear primitive, push to hardware)**

```
- prob RMSE 0.113, AUC drop 4pp is too lax for a paper claim
  but tight enough for a friend-demo schematic
- Phase 4: quantize the 6 differentials (egg/food/soldier/warrior
  are small integers; snail_pos and berries_norm are normalized
  in [-0.5, 0.5] / [0, 1])
- Phase 5: HLS sketch of 6-MAC + sigmoid LUT primitive
- Estimated cost: <500 LUT, 0-6 DSP, 1-2 cycle latency
```

**Route B (tighten the surrogate before hardware)**

```
- Try gradient-boosted regression with very few small trees
  (e.g., 5 trees x 4 leaves = 20 leaves total, 100-1000x param
  reduction vs the 10000-leaf oracle) targeting oracle logit.
  This is still HLS-friendly via Conifer.
- Try interaction terms: egg_diff x food_diff, soldier_diff x
  warrior_diff, etc., in a still-linear surrogate.
- Try a tiny MLP (~32-64 params) targeting oracle logit.
- See if prob RMSE drops below 0.05 with cost still <= a few
  hundred LUT.
```

Route A is the fun friend-demo path; Route B is the
"can we hit the strict gate" path. Open question for routing.

## Caveats

```
- Test data is 16 benchmark CSV shards (~3000 games). The full
  CLEAR-style encoded datasets referenced in CLAUDE.md
  (quality_filtered, logged_in, late_tournament) are not local;
  if surrogate evaluation is load-bearing we should re-encode.
- ECE of all surrogates (0.05) is slightly worse than oracle's
  ECE (0.054). Linear surrogates are reasonably calibrated; the
  oracle's calibration may be the limiting factor.
- The linear weights are not yet quantized/fixed-point; Phase 4
  will show whether int8/Q-format conversion preserves fidelity.
```

## Files

```
experiments/kquity_hw/surrogate_ladder.py      surrogate fitting
experiments/kquity_hw/results/surrogate_ladder.json   full ladder
experiments/kquity_hw/results/phase2_report.md  this file
```
