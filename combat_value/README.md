# Break-Even Combat Value

When is a fight worth taking? If a vanilla warrior charges the enemy queen,
how likely does the strike have to be to land for the engagement to pay off?
This module answers that with a single number per situation: the **break-even
strike-success probability `p*`**.

[Model](#model) · [The math](#the-math) · [State edits](#state-edits) ·
[Fine vs coarse](#fine-grained-vs-coarse-grained) · [Usage](#usage) ·
[Findings](#validation-findings) · [Caveats](#caveats)

## Model

A combat between attacker piece **A** (attacker's team) and defender piece
**B** (opposing team) resolves as **exactly one death** — B dies with
probability `p`, or A dies with probability `1 − p`. No whiffs, no trades:
`p` is the share of engagements the attacker survives.

We value each outcome with the win-probability model
(`current_preferred_model.mdl`), oriented to the attacker's team
(`model.predict == P(blue wins)`, so `V = pred` if the attacker is blue else
`1 − pred`):

- `V(S)` — status-quo value (decline the fight)
- `V_kill` — value after defender **B** dies (`S_kill`)
- `V_death` — value after attacker **A** dies (`S_death`)

## The math

Taking the fight beats the status quo iff
`p·V_kill + (1−p)·V_death ≥ V(S)`, so the break-even success probability is

```
p* = (V(S) − V_death) / (V_kill − V_death)
```

`p*` is where `V(S)` sits on the segment `[V_death, V_kill]`. **Fight iff your
true survival probability exceeds `p*`.** Special cases fall out naturally:

- `p* < 0` — even certain death of your piece beats the status quo → always fight
- `p* > 1` — even a guaranteed kill doesn't help → never fight
- `V_kill ≈ V_death` — the combat barely matters; `p*` is unstable and returned
  as `NaN` (filtered, never plotted).

Terminal kills are set exactly rather than predicted (the model never sees the
terminal `eggs == −1`): killing a queen at `eggs == 0` wins the game
(`V_kill = 1`); losing your own last queen loses it (`V_death = 0`).

## State edits

Each outcome is the *exact persistent consequence* the win-prob model tracks
(`preprocess.PlayerKillEvent.modify_game_state`), applied to the 52-feature
vector:

| Outcome | Feature edit |
|---|---|
| queen kill | that team's `eggs −= 1` (2→1→0→−1 = military loss) |
| worker/warrior kill | that worker loses wings/speed/food (→ bare drone); team's `n_vanilla`/`n_speed_warrior` decrements; worker blocks re-sorted by power to stay on-distribution |

Pieces: `drone`, `speed_drone`, `vanilla_warrior`, `speed_warrior`, `queen`.
Any attacker × any defender is supported.

## Fine-grained vs coarse-grained

- **Fine-grained** (`analysis.summarize`, `analysis.game_curve`): per-state
  `p*` over a dataset (distribution, percentiles) and a per-event `p*` curve
  through a single game.
- **Coarse-grained** (`analysis.bucket_table`): bucket states by
  decision-relevant variables (`def_eggs`, `net_warriors`, `atk_eggs`,
  `berries`, …) and report `p*` computed **from bucket-mean V's** (stable),
  alongside the median and IQR of per-state `p*` (spread). Averaging the V's
  before dividing avoids letting a few tiny-denominator states dominate.

## Usage

```bash
# Vanilla warrior attacking a queen, coarse table by queen lives × warrior edge
python -m combat_value --attacker vanilla_warrior --defender queen \
    --coarse def_eggs net_warriors

# Warrior-vs-warrior, dump everything to JSON
python -m combat_value --attacker vanilla_warrior --defender vanilla_warrior \
    --json out.json

# Per-event p* curve for one game (single attacking side)
python -m combat_value --attacker vanilla_warrior --defender queen \
    --game-id 12345 --side blue
```

Library:

```python
from combat_value import core, analysis
res = core.evaluate_matchup_both_sides(X, model.predict,
                                       'vanilla_warrior', 'queen',
                                       game_ids=gids, timestamps=ts)
print(analysis.summarize(res))
print(analysis.format_bucket_table(
    analysis.bucket_table(res, ['def_eggs', 'net_warriors']),
    ['def_eggs', 'net_warriors']))
```

## Validation / findings

On 1,500 logged-in games (both attacking sides), the model reproduces every
sanity check:

- **Mirror matchup breaks even at 0.5.** Vanilla-vs-vanilla at equal warrior
  count gives `p* = 0.500` exactly — a strong correctness signal for the
  orientation + edit machinery.
- **Warrior vs queen is cheap.** Median `p* ≈ 0.23`: killing a queen has large
  upside, de-winging your warrior modest downside.
- **Last-life queen → almost always attack.** With `def_eggs == 0` the kill is
  game-winning (`V_kill = 1`), so `p* ≈ 0.11–0.14`.
- **Warrior edge lowers the bar.** More spare warriors → `V_death ≈ V(S)` →
  smaller `p*` (a trade costs you little).
- **Don't throw a warrior at a drone.** Speed-warrior-vs-drone pins to
  `p* ≈ 1.0` (killing a drone barely moves `V`, but you risk your best piece).

## Caveats

- `p*` is a threshold on the **conditional** survival probability (given the
  engagement resolves in a death), not "P(strike lands)". Compare it against
  empirical *conditional* bump-win rates. For warrior-vs-warrior the two are
  nearly the same; for warrior-vs-queen most real pokes resolve to *nothing*,
  so read `p*` as "given someone dies, how often must it be them."
- Edits are pure ceteris-paribus counterfactuals; a real queen death correlates
  with tempo/positioning the 52-feature state cannot see.
- The state has no positional information beyond the snail, so a warrior death
  that opens a lane scores identically to one that doesn't. `p*` values
  composition/eggs/snail/berries, nothing spatial.
- Baseline is the status quo `V(S)`, which ignores opportunity cost (you could
  be doing berries/snail instead) — it therefore tends to *understate* `p*`.
