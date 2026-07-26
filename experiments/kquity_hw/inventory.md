# KQuity model + 52-feature inventory

## Oracle

```
file:      current_preferred_model.mdl
target:    model_experiments/combined_li_qf/qf_200k_symaug_100l_100t.mdl
type:      LightGBM Booster
num_trees: 100
num_leaves/tree: 100
num_features: 52
output:    P(blue wins) (binary classifier, sigmoid of leaf-sum)
```

**Label convention** (verified by monotonic probes in
`baseline_eval.py`). The repo `README.md` describes the model as
"P(gold wins)", but `fast_materialize._process_game` sets
`label = 1 if last_vals[0] == 'Blue' else 0`, and all six
monotonic probes are consistent with the model output being
P(blue wins):

```
gold.eggs +1   -> dp = -0.054  (gold healthier -> P(blue) down)
blue.eggs +1   -> dp = +0.056  (blue healthier -> P(blue) up)
gold.food +1   -> dp = -0.041  (gold closer to win -> P(blue) down)
blue.food +1   -> dp = +0.041  (blue closer to win -> P(blue) up)
snail_pos +0.1 -> dp = +0.048  (snail away from gold -> P(blue) up)
snail_pos -0.1 -> dp = -0.050  (snail toward gold -> P(blue) down)
```

Use `y_blue` (1 if blue won) when scoring fidelity to the oracle.

## 52-feature decode

The feature vector concatenates 6 groups (from
`fast_materialize._vectorize_state`). Worker positions within each
team are sorted ascending by "power" = wings + speed*0.5 + food*0.25
so feature semantics are stable across worker permutations.

Per-team block (20 features, blue at 0–19, gold at 20–39):

```
[0]  eggs              queen lives remaining (starts at 2, can hit 5)
[1]  food_count        berries deposited (3 = potential win)
[2]  n_warriors        warrior count (has_wings & ~has_speed)
[3]  n_soldiers        soldier count (has_wings & has_speed)
[4]  w0_is_bot         worker 0 (lowest power)
[5]  w0_has_food
[6]  w0_has_speed
[7]  w0_has_wings
[8]  w1_is_bot
[9]  w1_has_food
[10] w1_has_speed
[11] w1_has_wings
[12] w2_is_bot
[13] w2_has_food
[14] w2_has_speed
[15] w2_has_wings
[16] w3_is_bot         worker 3 (highest power)
[17] w3_has_food
[18] w3_has_speed
[19] w3_has_wings
```

Full layout (columns are LightGBM `Column_i`):

```
[0  ..19]  blue team   (20 features)
[20 ..39]  gold team   (20 features)
[40 ..44]  maiden_states[0..4]   0=neutral, 1=blue, -1=gold
[45 ..48]  map one-hot (day, night, dusk, twilight)
[49]       snail_pos   (snail_x normalized to [-0.5,0.5] then * gold_sym)
[50]       snail_spd   (snail_vel / SPEED_SNAIL_PPS) * gold_sym
[51]       berries_avail / 70.0
```

`gold_sym = +1` if gold is on left, `-1` otherwise. This makes
`snail_pos > 0` mean "snail favors gold" regardless of geometry.

## Top features by gain (from preferred model)

```
rank  column  feature                gain%   cum%
[ 0]  Col 20  gold.eggs              19.84   19.84
[ 1]  Col  0  blue.eggs              18.10   37.95
[ 2]  Col  1  blue.food_count        13.37   51.32
[ 3]  Col 21  gold.food_count        13.17   64.49
[ 4]  Col 49  snail_pos              10.37   74.87
[ 5]  Col 23  gold.n_soldiers         3.82   78.69
[ 6]  Col  3  blue.n_soldiers         3.63   82.32
[ 7]  Col 15  blue.w2_has_wings       3.04   85.35
[ 8]  Col 35  gold.w2_has_wings       2.92   88.27
[ 9]  Col 19  blue.w3_has_wings       1.48   89.75
[10]  Col 39  gold.w3_has_wings       1.38   91.13
```

**The three Killer Queen win conditions (eggs / berries / snail)
account for ~75% of total gain.** Top 11 features cover ~91%.
Wing-bit features (w2/w3 has_wings) appear as soldier-population
proxies (since power-sorted, w3 has_wings = highest-tier worker
is wielding wings = soldier ready). The top-5 picture is exactly:

```
P(gold wins) ≈ f(gold_eggs, blue_eggs, gold_berries, blue_berries,
                  snail_pos)
```

This is the candidate action variable.

## Data

Local artifacts:

```
tests/benchmark_events_000.csv.gz .. _015.csv.gz   16 game shards
tests/benchmark_expected.npz                       expected materialization
current_preferred_model.mdl                        oracle symlink
```

Not present locally (referenced in repo CLAUDE.md):

```
quality_filtered/encoded/all_games.bin
logged_in_games/encoded/all_games.bin
late_tournament_games/encoded/all_games.bin
```

Phase 0/1/2 work uses the 16 test shards; if the surrogate ladder
needs bigger samples we can re-encode from CSV via
`encode_datasets.py`.

## Symmetry

`symmetry.py` provides blue↔gold swap; the preferred model was
trained with `symaug` (symmetry-augmented). Per-symmetry sanity
check: swapping team color of every feature should give
`P(gold)_swapped ≈ 1 - P(gold)`. Phase 1 baseline_eval verifies
this.
