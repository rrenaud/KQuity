#!/usr/bin/env python3
"""Export model predictions for specified games as standalone JSON files.

Each model/experiment exports its predictions independently, so experiments
can make breaking changes to featurization — they just need to produce
{game_id: [{t, p}]} output.

Usage:
    # Export predictions for games referenced in a chapter file
    python export_predictions.py \
        --model model_experiments/new_data_model/model.mdl \
        --chapters ~/kq_stream_highlights/chapters/league_nights/sf-2026-01-06.json \
        --output predictions/baseline.json

    # Export predictions for specific game IDs
    python export_predictions.py \
        --model model_experiments/new_data_model/model.mdl \
        --game-ids 1714334 1714340 1714355 \
        --output predictions/baseline.json
"""

import argparse
import copy
import csv
import datetime
import glob
import gzip
import json
import os
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt

from fast_materialize import (
    NUM_FEATURES, SKIP_EVENTS, _NO_VALS_EVENTS,
    COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
    _MAP_LOOKUPS, _parse_ts, _snail_mult, _vectorize_state,
    SCREEN_WIDTH, VANILLA_SNAIL_PPS, SPEED_SNAIL_PPS,
)


def _vectorize_counterfactual(
    w: list[list[list[bool]]], eggs: list[int], food_count: list[int],
    maiden_states: list[int], map_idx: int, snail_x: float,
    snail_vel: float, snail_last_ts: float, rel_ts: float,
    berries_avail: int, gold_sym: float,
) -> list[float] | None:
    """Vectorize a counterfactual state, return feature list."""
    buf: list[list[float] | None] = [None]
    _vectorize_state(buf, 0, w, eggs, food_count, maiden_states,
                     map_idx, snail_x, snail_vel, snail_last_ts,
                     rel_ts, berries_avail, gold_sym)
    return buf[0]


def _compute_counterfactuals(
    w: list[list[list[bool]]], eggs: list[int], food_count: list[int],
    maiden_states: list[int], map_idx: int, snail_x: float,
    snail_vel: float, snail_last_ts: float, rel_ts: float,
    berries_avail: int, gold_sym: float,
) -> list[tuple[str, list[float] | None]]:
    """Compute counterfactual feature vectors for all possible events.

    Returns list of (event_name, feature_vector) tuples.
    Operates on deep copies of mutable state to avoid side effects.
    """
    results: list[tuple[str, list[float] | None]] = []

    def _vz(cw: list[list[list[bool]]], ce: list[int], cfc: list[int],
            cms: list[int], csx: float) -> list[float] | None:
        return _vectorize_counterfactual(
            cw, ce, cfc, cms, map_idx, csx, snail_vel, snail_last_ts,
            rel_ts, berries_avail, gold_sym)

    # --- Queen kills (2) ---
    for team, name in [(0, 'bqk'), (1, 'gqk')]:
        if eggs[team] > 0:
            ce = list(eggs)
            ce[team] -= 1
            results.append((name, _vz(w, ce, food_count, maiden_states, snail_x)))

    # --- Berry deposits (2) ---
    for team, name in [(0, 'bb'), (1, 'gb')]:
        cfc = list(food_count)
        cfc[team] += 1
        cba = berries_avail - 1
        buf: list[list[float] | None] = [None]
        _vectorize_state(buf, 0, w, eggs, cfc, maiden_states,
                         map_idx, snail_x, snail_vel, snail_last_ts,
                         rel_ts, cba, gold_sym)
        results.append((name, buf[0]))

    # --- Warrior deaths (4) ---
    # Worker state: [is_bot, has_food, has_speed, has_wings]
    # Speed warrior: has_wings=True, has_speed=True
    # Vanilla warrior: has_wings=True, has_speed=False
    for team, team_prefix in [(0, 'b'), (1, 'g')]:
        for has_speed, type_suffix in [(True, 'swd'),
                                       (False, 'vwd')]:
            # Find any matching worker
            for widx in range(4):
                worker = w[team][widx]
                if worker[3] and worker[2] == has_speed:  # has_wings and speed match
                    cw = copy.deepcopy(w)
                    cw[team][widx][1] = False  # has_food
                    cw[team][widx][2] = False  # has_speed
                    cw[team][widx][3] = False  # has_wings
                    results.append((f'{team_prefix}{type_suffix}',
                                    _vz(cw, eggs, food_count, maiden_states, snail_x)))
                    break

    # --- Warrior formations / maiden use (6) ---
    for team, team_prefix in [(0, 'b'), (1, 'g')]:
        # Speed drone gets wings: has_speed=True, has_wings=False
        for widx in range(4):
            worker = w[team][widx]
            if worker[2] and not worker[3]:  # has_speed, no wings
                cw = copy.deepcopy(w)
                cw[team][widx][3] = True   # gets wings
                cw[team][widx][1] = False  # drops food at maiden
                results.append((f'{team_prefix}sdw',
                                _vz(cw, eggs, food_count, maiden_states, snail_x)))
                break

        # Non-speed drone gets wings: no speed, no wings
        for widx in range(4):
            worker = w[team][widx]
            if not worker[2] and not worker[3]:  # no speed, no wings
                cw = copy.deepcopy(w)
                cw[team][widx][3] = True   # gets wings
                cw[team][widx][1] = False  # drops food at maiden
                results.append((f'{team_prefix}dw',
                                _vz(cw, eggs, food_count, maiden_states, snail_x)))
                break

        # Warrior gets speed: has_wings=True, has_speed=False
        for widx in range(4):
            worker = w[team][widx]
            if worker[3] and not worker[2]:  # has wings, no speed
                cw = copy.deepcopy(w)
                cw[team][widx][2] = True   # gets speed
                cw[team][widx][1] = False  # drops food at maiden
                results.append((f'{team_prefix}ws',
                                _vz(cw, eggs, food_count, maiden_states, snail_x)))
                break

    # --- Snail (2) ---
    # gold_sym: +1 when gold_on_left (right = toward blue), -1 otherwise
    # toward blue = snail_x + 50*gold_sym, toward gold = snail_x - 50*gold_sym
    csx_blue = snail_x + 50.0 * gold_sym
    csx_blue = max(0.0, min(float(SCREEN_WIDTH), csx_blue))
    results.append(('sb',
                    _vz(w, eggs, food_count, maiden_states, csx_blue)))

    csx_gold = snail_x - 50.0 * gold_sym
    csx_gold = max(0.0, min(float(SCREEN_WIDTH), csx_gold))
    results.append(('sg',
                    _vz(w, eggs, food_count, maiden_states, csx_gold)))

    # --- Maiden flips (per-gate) ---
    for target_color, prefix in [(1, 'mb'), (-1, 'mg')]:
        for midx in range(len(maiden_states)):
            if maiden_states[midx] != target_color:
                cms = list(maiden_states)
                cms[midx] = target_color
                results.append((f'{prefix}{midx}',
                                _vz(w, eggs, food_count, cms, snail_x)))

    return results


def _process_game_predictions(
    raw_events: list[tuple[datetime.datetime, str, str]],
    model: lgb.Booster,
    counterfactuals: bool = False,
) -> tuple[list[dict[str, Any]], bool] | None:
    """Process one game's events, return (predictions, gold_on_left) tuple.

    Returns None if the game is invalid (no gamestart/mapstart).
    """
    raw_events.sort(key=lambda x: x[0])

    gamestart_dt: datetime.datetime | None = None
    map_name: str | None = None
    gold_on_left: bool | None = None

    for dt, event_type, values_str in raw_events:
        if event_type == 'gamestart' and gamestart_dt is None:
            gamestart_dt = dt
        if event_type == 'mapstart' and map_name is None:
            vals = values_str[1:-1].split(',')
            map_name = vals[0]
            gold_on_left = (vals[1] == 'True')

    if gamestart_dt is None or map_name is None or gold_on_left is None:
        return None

    map_lookup = _MAP_LOOKUPS.get((map_name, gold_on_left))
    if map_lookup is None:
        return None

    berry_lookup: dict[tuple[int, int], int] = map_lookup['berry_lookup']  # type: ignore[assignment]
    maiden_lookup: dict[tuple[int, int], tuple[str, int]] = map_lookup['maiden_lookup']  # type: ignore[assignment]
    map_idx: int = map_lookup['map_index']  # type: ignore[assignment]
    total_berries: int = map_lookup['total_berries']  # type: ignore[assignment]

    # Initialize game state
    w: list[list[list[bool]]] = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
    eggs = [2, 2]
    food_dep = [[False] * 12, [False] * 12]
    food_count = [0, 0]
    maiden_states = [0, 0, 0, 0, 0]
    berries_avail = total_berries
    snail_x = float(SCREEN_WIDTH) / 2.0
    snail_vel = 0.0
    snail_last_ts = 0.0
    gold_sym = 1.0 if gold_on_left else -1.0

    # Collect feature vectors and timestamps
    states: list[list[float] | None] = []
    timestamps: list[float] = []
    # For counterfactuals: list of (event_idx, event_name, feature_vec_or_list)
    cf_entries: list[tuple[int, str, list[float] | None]] = [] if counterfactuals else []

    for dt, event_type, values_str in raw_events:
        rel_ts = (dt - gamestart_dt).total_seconds()
        vals = values_str[1:-1].split(',') if event_type not in _NO_VALS_EVENTS else []

        # Vectorize BEFORE applying this event (same as fast_materialize)
        if rel_ts > 5.0:
            buf: list[list[float] | None] = [None]
            _vectorize_state(buf, 0, w, eggs, food_count, maiden_states,
                             map_idx, snail_x, snail_vel, snail_last_ts,
                             rel_ts, berries_avail, gold_sym)
            states.append(buf[0])
            timestamps.append(rel_ts)

            if counterfactuals:
                event_idx = len(states) - 1
                cfs = _compute_counterfactuals(
                    w, eggs, food_count, maiden_states,
                    map_idx, snail_x, snail_vel, snail_last_ts,
                    rel_ts, berries_avail, gold_sym)
                for name, vec_or_list in cfs:
                    cf_entries.append((event_idx, name, vec_or_list))

        # Apply state mutation (copied from fast_materialize._process_game)
        if event_type == 'spawn':
            pid = int(vals[0])
            is_bot = vals[1] == 'True'
            team = pid % 2
            widx = (pid - 3) // 2
            w[team][widx][0] = is_bot
        elif event_type == 'carryFood':
            pid = int(vals[0])
            team = pid % 2
            widx = (pid - 3) // 2
            w[team][widx][1] = True
        elif event_type == 'berryDeposit':
            hole_x, hole_y = int(vals[0]), int(vals[1])
            pid = int(vals[2])
            team = pid % 2
            widx = (pid - 3) // 2
            w[team][widx][1] = False
            bi = berry_lookup[(hole_x, hole_y)]
            if not food_dep[team][bi]:
                food_dep[team][bi] = True
                food_count[team] += 1
            berries_avail -= 1
        elif event_type == 'berryKickIn':
            hole_x, hole_y = int(vals[0]), int(vals[1])
            pid = int(vals[2])
            own_team = vals[3] == 'True'
            team = pid % 2
            if not own_team:
                team = 1 - team
            bi = berry_lookup[(hole_x, hole_y)]
            if not food_dep[team][bi]:
                food_dep[team][bi] = True
                food_count[team] += 1
            berries_avail -= 1
        elif event_type == 'blessMaiden':
            mx, my = int(vals[0]), int(vals[1])
            color = 1 if vals[2] == 'Blue' else -1
            _, midx = maiden_lookup[(mx, my)]
            maiden_states[midx] = color
        elif event_type == 'useMaiden':
            mtype = vals[2]
            pid = int(vals[3])
            team = pid % 2
            widx = (pid - 3) // 2
            if mtype == 'maiden_speed':
                w[team][widx][2] = True
            else:
                w[team][widx][3] = True
            w[team][widx][1] = False
        elif event_type == 'playerKill':
            killed_pid = int(vals[3])
            killed_cat = vals[4]
            team = killed_pid % 2
            if killed_cat == 'Queen':
                eggs[team] -= 1
            else:
                widx = (killed_pid - 3) // 2
                w[team][widx][1] = False
                w[team][widx][2] = False
                w[team][widx][3] = False
        elif event_type == 'getOnSnail':
            sx = int(vals[0])
            rider_pid = int(vals[2])
            snail_x = float(sx)
            snail_last_ts = rel_ts
            rider_team = rider_pid % 2
            rider_widx = (rider_pid - 3) // 2
            has_speed = w[rider_team][rider_widx][2]
            base_speed = SPEED_SNAIL_PPS if has_speed else VANILLA_SNAIL_PPS
            snail_vel = base_speed * _snail_mult(gold_on_left, rider_team)
        elif event_type == 'snailEat':
            sx = int(vals[0])
            rider_pid = int(vals[2])
            snail_x = float(sx)
            snail_last_ts = rel_ts
            rider_team = rider_pid % 2
            rider_widx = (rider_pid - 3) // 2
            has_speed = w[rider_team][rider_widx][2]
            base_speed = SPEED_SNAIL_PPS if has_speed else VANILLA_SNAIL_PPS
            snail_vel = base_speed * _snail_mult(gold_on_left, rider_team)
        elif event_type == 'getOffSnail':
            snail_x = float(int(vals[0]))
            snail_last_ts = rel_ts
            snail_vel = 0.0
        elif event_type == 'snailEscape':
            snail_x = float(int(vals[0]))
            snail_last_ts = rel_ts
            snail_vel = 0.0

    if not states:
        return None

    if not counterfactuals or not cf_entries:
        # Simple path: just baseline predictions
        X = np.array(states, dtype=np.float32)
        preds = model.predict(X)
        return ([{'t': round(t, 2), 'p': round(float(p), 4)}
                 for t, p in zip(timestamps, preds)], gold_on_left)

    # Counterfactual path: batch all vectors into one predict call
    # Build combined matrix: baselines first, then all counterfactual vectors
    n_baselines = len(states)
    all_vectors = list(states)

    # Track where each counterfactual's prediction lands in the combined array
    cf_map: list[tuple[int, str, int]] = []
    for event_idx, name, vec in cf_entries:
        start = len(all_vectors)
        all_vectors.append(vec)
        cf_map.append((event_idx, name, start))

    X = np.array(all_vectors, dtype=np.float32)
    all_preds = model.predict(X)
    baseline_preds = all_preds[:n_baselines]

    # Build per-event counterfactual dicts
    cf_dicts: list[dict[str, float]] = [{} for _ in range(n_baselines)]
    for event_idx, name, start in cf_map:
        delta = float(all_preds[start] - baseline_preds[event_idx])
        if abs(delta) >= 0.001:  # skip negligible
            cf_dicts[event_idx][name] = round(delta, 4)

    results: list[dict[str, Any]] = []
    for i, (t, p) in enumerate(zip(timestamps, baseline_preds)):
        entry: dict[str, Any] = {'t': round(t, 2), 'p': round(float(p), 4)}
        if cf_dicts[i]:
            entry['c'] = cf_dicts[i]
        results.append(entry)
    return results, gold_on_left


def load_games_from_partitions(
    data_dir: str,
    target_game_ids: set[int],
) -> dict[int, list[tuple[datetime.datetime, str, str]]]:
    """Scan partition files and load events for target game IDs.

    Returns dict {game_id: [(datetime, event_type, values_str), ...]}.
    """
    games: dict[int, list[tuple[datetime.datetime, str, str]]] = {}
    found_ids: set[int] = set()
    target_set = set(target_game_ids)

    files = sorted(glob.glob(os.path.join(data_dir, 'gameevents_*.csv.gz')))
    if not files:
        print(f"No partition files found in {data_dir}", file=sys.stderr)
        return games

    for filename in files:
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                game_id = int(row[COL_GAME_ID])
                if game_id not in target_set:
                    continue
                event_type = row[COL_TYPE]
                if event_type in SKIP_EVENTS:
                    continue
                if game_id not in games:
                    games[game_id] = []
                    found_ids.add(game_id)
                games[game_id].append(
                    (_parse_ts(row[COL_TS]), event_type, row[COL_VALUES])
                )

        # Early exit if we found all target games
        if found_ids == target_set:
            break

    missing = target_set - found_ids
    if missing:
        print(f"Warning: {len(missing)} games not found: {sorted(missing)[:10]}",
              file=sys.stderr)

    return games


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export model predictions for specified games')
    parser.add_argument('--model', required=True,
                        help='Path to LightGBM model file')
    parser.add_argument('--chapters',
                        help='Path to chapter JSON file (extracts game IDs)')
    parser.add_argument('--game-ids', type=int, nargs='+',
                        help='Specific game IDs to export')
    parser.add_argument('--output', '-o', required=True,
                        help='Output JSON file path')
    parser.add_argument('--name',
                        help='Model name (default: derived from model filename)')
    parser.add_argument('--data-dir', default='unfiltered_partitioned',
                        help='Directory containing gameevents partitions '
                             '(default: unfiltered_partitioned)')
    parser.add_argument('--counterfactuals', action='store_true',
                        help='Compute counterfactual event deltas for each '
                             'prediction point')
    args = parser.parse_args()

    # Collect target game IDs
    target_game_ids: set[int] = set()
    if args.chapters:
        with open(args.chapters) as f:
            chapter_data = json.load(f)
        for ch in chapter_data['chapters']:
            target_game_ids.add(ch['game_id'])
        print(f"Loaded {len(target_game_ids)} game IDs from {args.chapters}")
    if args.game_ids:
        target_game_ids.update(args.game_ids)

    if not target_game_ids:
        parser.error('Must specify --chapters or --game-ids')

    # Load model
    print(f"Loading model from {args.model}...")
    model = lgb.Booster(model_file=args.model)

    # Load game events from partitions
    print(f"Scanning {args.data_dir}/ for {len(target_game_ids)} games...")
    games = load_games_from_partitions(args.data_dir, target_game_ids)
    print(f"Found {len(games)} games")

    # Generate predictions
    results: dict[str, list[dict[str, Any]]] = {}
    gold_on_left_map: dict[str, bool] = {}
    for game_id, events in sorted(games.items()):
        result = _process_game_predictions(events, model,
                                           counterfactuals=args.counterfactuals)
        if result is not None:
            preds, gol = result
            results[str(game_id)] = preds
            gold_on_left_map[str(game_id)] = gol

    print(f"Generated predictions for {len(results)} games")

    # Build output
    model_name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    output: dict[str, object] = {
        'name': model_name,
        'model_path': args.model,
        'exported_at': datetime.datetime.now().isoformat(),
        'games': results,
        'gold_on_left': gold_on_left_map,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
