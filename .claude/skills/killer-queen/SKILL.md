---
name: killer-queen
description: Domain knowledge for Killer Queen arcade. Use when reasoning about game mechanics, event semantics, feature engineering, or win condition logic. Covers the gameplay loop, player roles, map structure, and how game concepts map to code.
---

# Killer Queen Arcade — Game Design & Code Mapping

## Game Overview

Killer Queen is a 5v5 arcade strategy game played on a shared cabinet. Two teams (blue and gold) compete simultaneously across three independent win conditions. The first team to complete ANY win condition wins.

The map is horizontally contiguous — entities that exit one edge appear on the opposite side with preserved momentum.

## Win Conditions

| Win condition | Mechanism | Model state |
|---|---|---|
| **Military** | Kill the enemy queen 3 times | `eggs[team]` starts at 2 (not 3 — the queen is already alive), reaching 0 = loss |
| **Economic** | Fill all 12 berry slots on your side | `food_count[team]` reaches 12; `food_dep[team]` tracks which slots are filled |
| **Snail** | Ride the snail into the enemy goal | `snail_x` reaches 0 (gold goal when gold_on_left) or 1920 (blue goal when gold_on_left) |

All three run concurrently. Games typically end in 60-180 seconds. Military is most common, economic second, snail rare.

## Players & Roles

Each team has 5 players: 1 queen + 4 workers.

**Position IDs** (from HiveMind API):
- 1 = Gold Queen, 2 = Blue Queen
- 3/5/7/9 = Gold workers (Stripes/Abs/Skull/Checkers)
- 4/6/8/10 = Blue workers (Stripes/Abs/Skull/Checkers)
- Team = `pid % 2` (0 = blue, 1 = gold). Worker index = `(pid - 3) // 2`.

**Queen:** Combat only. Cannot use maiden gates but tags them by contact (`blessMaiden` event). Has 3 lives (eggs). Kills drones/warriors in one hit. Queen-vs-queen or warrior-vs-warrior combat resolves by Y-coordinate: higher unit wins. On ties, queen beats warrior; same-class ties cause knockback (no kill). Queens can dive (fast downward) and are the only unit that can tag (re-bless) gates.

**Workers** start as drones (unarmed, slow). Each worker has 4 boolean state flags:

| Flag | Code | Meaning |
|---|---|---|
| `is_bot` | `w[team][widx][0]` | AI-controlled (rare in competitive play) |
| `has_food` | `w[team][widx][1]` | Carrying a berry |
| `has_speed` | `w[team][widx][2]` | Speed upgrade from maiden gate |
| `has_wings` | `w[team][widx][3]` | Wings upgrade — becomes a warrior |

**Worker upgrade paths** (via maiden gates — consumes the carried berry):
- Drone → **Warrior** (wings gate): can fight, fly (hover, no dive). Most common upgrade.
- Drone → **Speed drone** (speed gate): 1.4x ground speed, rides snail faster. Still unarmed.
- Speed drone → **Speed warrior** (wings gate): fastest + can fight. Strongest form.

Warriors cannot carry berries, so they cannot use gates — the only path to speed warrior is drone → speed → wings.

Warriors cannot ride the snail. Workers die in one hit from queens or warriors. On death, all upgrades are lost (revert to drone). Carried berry is dropped as a loose berry.

## Maiden Gates

5 gates per map, each with a fixed type (speed or wings) and a current blessing color (blue/gold/neutral).

- Workers use a gate to receive its upgrade type (must be blessed to their team color)
- Queens tag (re-bless) gates by physical contact — this is how gate color changes
- Gate blessing persists until re-tagged by the other team's queen
- `maiden_states[0..4]`: +1 = blue, -1 = gold, 0 = neutral

Gate positions and types vary by map. The model tracks all 5 gate states as features. Gate control is strategically important — controlling gates denies enemy upgrades.

## Berries (Economic)

Each map has berry bushes on the left and right sides (totals vary: Day/Dusk=66, Night=54, Twilight=60). Workers pick up berries and deposit them in holes near their team's hive.

- `food_count[team]` = number of filled berry slots (0-12)
- `food_dep[team][0..11]` = which specific slots are filled
- `berries_avail` = remaining berries on the map (decrements on deposit or kick-in)
- `berryKickIn` = queen kicks a loose berry into a hole (can go to either team's slot)

Workers drop carried food on death (berry becomes loose) or consume it at maiden gates (berry destroyed).

**Famine cycle:** When all berries are consumed or deposited (none loose or carried), a 90-second countdown begins. When it expires, all original berries respawn at their initial positions.

## Snail

A large snail sits on a horizontal track. Workers ride it toward the enemy goal.

- `snail_x` = current position (0-1920 pixels). Starts at center (960).
- `snail_vel` = current velocity. 0 when not being ridden.
- Speed riders move snail at ~28.2 px/s, vanilla at ~20.9 px/s.
- `gold_sym` = +1 when gold_on_left, -1 otherwise. Used to normalize snail direction.
- Model feature: `snail_pos = (inferred_x / 1920 - 0.5) * gold_sym` (centered, direction-normalized).
- If both teams ride simultaneously, snail velocity = 0 (contested).
- A moving snail is lethal to ground-based enemies in front of it.
- `snailEat` = snail eats an enemy worker riding from the other side (rider continues).

## Maps

4 maps: Day, Dusk, Night, Twilight. Map determines:
- Berry hole positions and total count
- Maiden gate positions and types (always 2 speed + 3 wings)
- Snail track width
- Overall layout geometry

Map structure defined in `map_structure_info.json`. Model uses one-hot encoding: `map_idx` (0-3).

## gold_on_left

Each game has a `gold_on_left` boolean from `mapstart`. When True, gold hive is on the left side of the screen, blue on right. When False, the layout is mirrored (all x-coordinates flip: `x → 1920 - x`).

The model normalizes for this via `gold_sym` and team-relative features so predictions are orientation-invariant. The `_MAP_LOOKUPS` dict is keyed by `(map_name, gold_on_left)` to handle coordinate mapping.

## Key Game Events

| Event | What happens | State mutation |
|---|---|---|
| `gamestart` | Game begins | Sets reference timestamp |
| `mapstart` | Map loaded | Sets map name + gold_on_left |
| `spawn` | Player spawns | Sets `is_bot` flag |
| `carryFood` | Worker picks up berry | `has_food = True` |
| `berryDeposit` | Worker deposits berry | `food_count++`, `has_food = False` |
| `berryKickIn` | Queen kicks berry in | `food_count++` for target team |
| `blessMaiden` | Gate changes color | `maiden_states[idx] = +1/-1` |
| `useMaiden` | Worker uses gate | Gets speed or wings, `has_food = False` |
| `playerKill` | Player killed | Queen: `eggs--`. Worker: all flags reset to False |
| `getOnSnail` | Worker mounts snail | Sets `snail_vel` based on rider speed |
| `getOffSnail` | Worker dismounts | `snail_vel = 0` |
| `snailEat` | Snail eats enemy rider | Position updates, rider keeps going |
| `snailEscape` | Eaten rider escapes | `snail_vel = 0` |
| `victory` | Game ends | Blue or Gold wins |

## Models

**Win-probability model** (LightGBM): Takes the 52-feature state vector at each event and predicts P(gold wins). This is the model described below and used by the counterfactual system.

**Game quality classifier** (LightGBM): Takes 69 hand-crafted heuristic features computed over an entire game and predicts whether the game is competitive (vs. casual/incomplete). Trained on logged-in games as positives, unfiltered games as negatives. Used to filter training data for the win-probability model. See `game_quality_classifier/` and CLAUDE.md for details.

## Win-Probability Feature Vector (52 features)

Per team (20 features x 2 teams = 40):
- `eggs` (queen lives remaining)
- `food_count` (berry slots filled)
- `num_vanilla_warriors`, `num_speed_warriors`
- 4 workers sorted by power score, each with 4 bools: `[is_bot, has_food, has_speed, has_wings]`

Shared features (12):
- `maiden_states[0..4]` (5 gate colors)
- `map_one_hot[0..3]` (4 map indicators)
- `snail_pos` (normalized position)
- `snail_speed` (normalized velocity)
- `berries_avail` (fraction remaining)

## Counterfactual System

`export_predictions.py` computes "what-if" deltas: for each event, what would the win probability be if a hypothetical event happened instead?

Keys: `bqk`/`gqk` (queen kill), `bb`/`gb` (berry), `bswd`/`bvwd`/`gswd`/`gvwd` (warrior deaths), `bsdw`/`bdw`/`bws`/`gsdw`/`gdw`/`gws` (upgrades), `sb`/`sg` (snail), `mb0`-`mb4`/`mg0`-`mg4` (per-gate maiden flips).
