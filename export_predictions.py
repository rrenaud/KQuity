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
import csv
import datetime
import glob
import gzip
import json
import os
import sys

import lightgbm as lgb
import numpy as np

from fast_materialize import (
    NUM_FEATURES, SKIP_EVENTS, _NO_VALS_EVENTS,
    COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
    _MAP_LOOKUPS, _parse_ts, _snail_mult, _vectorize_state,
    SCREEN_WIDTH, VANILLA_SNAIL_PPS, SPEED_SNAIL_PPS,
)


def _process_game_predictions(raw_events, model):
    """Process one game's events, return list of {t, p} predictions.

    Returns None if the game is invalid (no gamestart/mapstart).
    """
    raw_events.sort(key=lambda x: x[0])

    gamestart_dt = None
    map_name = None
    gold_on_left = None

    for dt, event_type, values_str in raw_events:
        if event_type == 'gamestart' and gamestart_dt is None:
            gamestart_dt = dt
        if event_type == 'mapstart' and map_name is None:
            vals = values_str[1:-1].split(',')
            map_name = vals[0]
            gold_on_left = (vals[1] == 'True')

    if gamestart_dt is None or map_name is None:
        return None

    map_lookup = _MAP_LOOKUPS.get((map_name, gold_on_left))
    if map_lookup is None:
        return None

    berry_lookup = map_lookup['berry_lookup']
    maiden_lookup = map_lookup['maiden_lookup']
    map_idx = map_lookup['map_index']
    total_berries = map_lookup['total_berries']

    # Initialize game state
    w = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
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
    states = []
    timestamps = []

    for dt, event_type, values_str in raw_events:
        rel_ts = (dt - gamestart_dt).total_seconds()
        vals = values_str[1:-1].split(',') if event_type not in _NO_VALS_EVENTS else None

        # Vectorize BEFORE applying this event (same as fast_materialize)
        if rel_ts > 5.0:
            buf = [None]
            _vectorize_state(buf, 0, w, eggs, food_count, maiden_states,
                             map_idx, snail_x, snail_vel, snail_last_ts,
                             rel_ts, berries_avail, gold_sym)
            states.append(buf[0])
            timestamps.append(rel_ts)

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

    # Run model predictions
    X = np.array(states, dtype=np.float32)
    preds = model.predict(X)

    return [{'t': round(t, 2), 'p': round(float(p), 4)}
            for t, p in zip(timestamps, preds)]


def load_games_from_partitions(data_dir, target_game_ids):
    """Scan partition files and load events for target game IDs.

    Returns dict {game_id: [(datetime, event_type, values_str), ...]}.
    """
    games = {}
    found_ids = set()
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


def main():
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
    args = parser.parse_args()

    # Collect target game IDs
    target_game_ids = set()
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
    results = {}
    for game_id, events in sorted(games.items()):
        preds = _process_game_predictions(events, model)
        if preds is not None:
            results[str(game_id)] = preds

    print(f"Generated predictions for {len(results)} games")

    # Build output
    model_name = args.name or os.path.splitext(os.path.basename(args.model))[0]
    output = {
        'name': model_name,
        'model_path': args.model,
        'exported_at': datetime.datetime.now().isoformat(),
        'games': results,
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
