# Case F — KQuity Phase 0 / Phase 1 report

## 1. Model inventory

```
oracle: current_preferred_model.mdl
        -> model_experiments/combined_li_qf/qf_200k_symaug_100l_100t.mdl
        LightGBM Booster, 100 trees x 100 leaves, 52 features
        output: P(blue wins) -- label convention corrected vs README
```

## 2. Data inventory

```
local: 16 shards in tests/benchmark_events_*.csv.gz (~233k events
       after fast_materialize, span 5s..2925s game time)
not local: encoded .bin files referenced in CLAUDE.md
```

The 16 test shards are sufficient for Phase 0/1/2 fidelity work.
If sample-size becomes load-bearing, re-encode from CSV with
`encode_datasets.py`.

## 3. Baseline oracle metrics (event-level, full eval pool)

```
n = 233,538
P(blue=win) = 0.5186  (label rate)
mean P(blue) = 0.5060  (mean model output)
log loss = 0.558
Brier = 0.190
acc @ 0.5 = 0.706
AUC = 0.784
ECE (10 bin) = 0.030
```

Symmetry probe (blue<->gold feature swap should sum to 1):

```
mean |P(blue|X) + P(blue|swap(X)) - 1| = 0.019
RMS = 0.026
```

Near-symmetric (symaug training), small residual drift. Not
deployment-critical.

## 4. Phase split

```
phase   n        pos    logloss   Brier   acc@.5    AUC
early   58,383   0.518  0.641     0.225   0.637     0.689
mid     81,738   0.519  0.550     0.186   0.718     0.798
late    93,417   0.519  0.513     0.172   0.740     0.823
```

AUC climbs monotonically: more state -> more signal. The model
is much more confident in mid/late game. This is consistent with
the ChatGPT Pro hypothesis that early-game P(blue) depends mostly
on resource differential while late-game depends on imminent
win-condition completion.

## 5. Top feature importances (gain) on the preferred model

```
rank  feature                  gain%    cum%
[ 0]  gold.eggs               19.84    19.84
[ 1]  blue.eggs               18.10    37.95
[ 2]  blue.food_count         13.37    51.32
[ 3]  gold.food_count         13.17    64.49
[ 4]  snail_pos               10.37    74.87
[ 5]  gold.n_soldiers          3.82    78.69
[ 6]  blue.n_soldiers          3.63    82.32
[ 7]  blue.w2_has_wings        3.04    85.35
[ 8]  gold.w2_has_wings        2.92    88.27
[ 9]  blue.w3_has_wings        1.48    89.75
[10]  gold.w3_has_wings        1.38    91.13
```

The three Killer Queen win conditions
(eggs / berries / snail) account for **74.87% of total gain**.
Top 11 features cover 91.1%. The wing-bit features in worker
positions 2 and 3 are soldier-readiness proxies (workers are
power-sorted before vectorization, so w3 having wings = highest-
tier worker is a soldier).

## 6. Candidate action variables

The top-5 picture suggests:

```
P(blue wins) approx f( blue.eggs, gold.eggs,
                       blue.food_count, gold.food_count,
                       snail_pos )
```

Reformulated symmetrically (gold-relative differentials):

```
egg_diff   = blue.eggs - gold.eggs           (queen-life lead)
food_diff  = blue.food_count - gold.food_count (berry lead)
snail      = snail_pos                       (negative = toward gold)
soldier_diff = blue.n_soldiers - gold.n_soldiers (military lead)
phase      = event_index normalized
```

This is the candidate action variable for Phase 2 surrogate
fitting.

## 7. Next: surrogate ladder (Phase 2)

```
1. sparse linear  / L1 logistic   targeting oracle logit
2. ridge logistic over top-K features
3. hand-hypothesis linear (5-6 differentials)
4. depth-2/3/4 decision tree
5. phase-conditioned linear (early/mid/late)
6. binned LUT over (egg_diff, food_diff, snail)
```

Fidelity gates per ChatGPT Pro:

```
Strong:  prob RMSE <= 0.03, AUC drop <= 2pp, features <= 8-10
Good:    prob RMSE <= 0.05, metric drop <= 3-5pp, human-readable
Fail:    shallow surrogates need too many features or lose
         calibration badly
```

## 8. Standing by

Phase 0/1 complete. Ready to start Phase 2 surrogate ladder.

GPU: not used. CPU only. The W7900 remains tropical and at peace.
