"""Fast path for game state materialization: CSV events -> numpy feature matrix.

Produces numerically identical output to preprocess.create_game_states_matrix
but avoids the OO overhead: no GameEvent/GameState classes, no copy.deepcopy,
no per-event numpy array creation. Writes directly into pre-allocated buffers.
"""

import csv
import datetime
import glob
import gzip
import json
import os
import random

import numpy as np

# --- Constants (inlined from constants.py) ---

SCREEN_WIDTH = 1920
VANILLA_SNAIL_PPS = 20.896215463
SPEED_SNAIL_PPS = 28.209890875

MAP_NAMES = ['map_day', 'map_night', 'map_dusk', 'map_twilight']
MAP_INDEX = {name: i for i, name in enumerate(MAP_NAMES)}

NUM_FEATURES = 52

SKIP_EVENTS = frozenset({
    'gameend', 'playernames',
    'reserveMaiden', 'unreserveMaiden',
    'cabinetOnline', 'cabinetOffline',
    'bracket', 'tstart', 'tournamentValidation', 'checkIfTournamentRunning',
    'glance',
    'enteredGameScreen', 'signInPlayer', 'signOutPlayer',
})

# Events that don't need values parsing
_NO_VALS_EVENTS = frozenset({'gamestart', 'victory', 'mapstart'})

# CSV column indices (id, timestamp, event_type, values, game_id)
COL_TS = 1
COL_TYPE = 2
COL_VALUES = 3
COL_GAME_ID = 4


# --- Map lookup tables (built once at module load) ---

def _build_map_lookups(json_path=None):
    """Build flat lookup dicts for all 8 map configurations (4 maps x 2 orientations).

    Returns dict keyed by (map_name_str, gold_on_left_bool) -> {
        'berry_lookup': {(x,y) -> berry_index},
        'maiden_lookup': {(x,y) -> (type_str, maiden_index)},
        'total_berries': int,
        'snail_track_width': float,
        'map_index': int,
    }
    """
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'map_structure_info.json')

    with open(json_path, 'rb') as f:
        raw = json.load(f)

    lookups = {}
    for map_name, info in raw.items():
        map_idx = MAP_INDEX[map_name]

        # gold_on_left=True: left_berries=gold, right_berries=blue
        berry_gol = {}
        for i, (x, y) in enumerate(info['left_berries']):
            berry_gol[(x, y)] = i
        for i, (x, y) in enumerate(info['right_berries']):
            berry_gol[(x, y)] = i

        maiden_gol = {}
        for idx, (mtype, x, y) in enumerate(info['maiden_info']):
            maiden_gol[(x, y)] = (mtype, idx)

        lookups[(map_name, True)] = {
            'berry_lookup': berry_gol,
            'maiden_lookup': maiden_gol,
            'total_berries': info['total_berries'],
            'snail_track_width': info['snail_track_width'],
            'map_index': map_idx,
        }

        # gold_on_left=False: swap berry ownership, flip maiden x coords
        berry_gor = {}
        for i, (x, y) in enumerate(info['right_berries']):
            berry_gor[(x, y)] = i
        for i, (x, y) in enumerate(info['left_berries']):
            berry_gor[(x, y)] = i

        maiden_gor = {}
        for idx, (mtype, x, y) in enumerate(info['maiden_info']):
            maiden_gor[(SCREEN_WIDTH - x, y)] = (mtype, idx)

        lookups[(map_name, False)] = {
            'berry_lookup': berry_gor,
            'maiden_lookup': maiden_gor,
            'total_berries': info['total_berries'],
            'snail_track_width': info['snail_track_width'],
            'map_index': map_idx,
        }

    return lookups


_MAP_LOOKUPS = _build_map_lookups()


# --- Timestamp parsing ---

def _parse_ts(ts_str):
    """Parse ISO timestamp to datetime. Uses C-implemented fromisoformat."""
    return datetime.datetime.fromisoformat(ts_str)


# --- Snail direction helper ---

def _snail_mult(gold_on_left, team_int):
    """Snail movement multiplier. team_int: 0=blue, 1=gold."""
    if gold_on_left:
        return -1 if team_int == 1 else 1
    else:
        return 1 if team_int == 1 else -1


# --- Direct buffer vectorization ---

# Pre-built map one-hot vectors
_MAP_ONE_HOT = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]


def _vectorize_team(tw, eggs, fc):
    """Vectorize one team into a 20-element list. Returns list of floats."""
    # Compute powers and sort indices
    w0, w1, w2, w3 = tw[0], tw[1], tw[2], tw[3]
    pw = [
        (w0[3] + w0[2] * 0.5 + w0[1] * 0.25, 0, w0),
        (w1[3] + w1[2] * 0.5 + w1[1] * 0.25, 1, w1),
        (w2[3] + w2[2] * 0.5 + w2[1] * 0.25, 2, w2),
        (w3[3] + w3[2] * 0.5 + w3[1] * 0.25, 3, w3),
    ]
    pw.sort()

    # Count warrior types
    nv = 0
    ns = 0
    if w0[3]:
        ns += w0[2]; nv += (not w0[2])
    if w1[3]:
        ns += w1[2]; nv += (not w1[2])
    if w2[3]:
        ns += w2[2]; nv += (not w2[2])
    if w3[3]:
        ns += w3[2]; nv += (not w3[2])

    a, b, c, d = pw[0][2], pw[1][2], pw[2][2], pw[3][2]
    return [float(eggs), float(fc), float(nv), float(ns),
            float(a[0]), float(a[1]), float(a[2]), float(a[3]),
            float(b[0]), float(b[1]), float(b[2]), float(b[3]),
            float(c[0]), float(c[1]), float(c[2]), float(c[3]),
            float(d[0]), float(d[1]), float(d[2]), float(d[3])]


def _vectorize_state(buf, idx, w, eggs, food_count, maiden_states,
                     map_idx, snail_x, snail_vel, snail_last_ts,
                     event_ts, berries_avail, gold_sym):
    """Write 52 features directly into buf[idx] via list assignment."""
    inferred_pos = snail_x + (event_ts - snail_last_ts) * snail_vel
    snail_pos = (inferred_pos / SCREEN_WIDTH - 0.5) * gold_sym
    snail_spd = (snail_vel / SPEED_SNAIL_PPS) * gold_sym

    buf[idx] = (
        _vectorize_team(w[0], eggs[0], food_count[0])
        + _vectorize_team(w[1], eggs[1], food_count[1])
        + [float(maiden_states[0]), float(maiden_states[1]),
           float(maiden_states[2]), float(maiden_states[3]),
           float(maiden_states[4])]
        + _MAP_ONE_HOT[map_idx]
        + [snail_pos, snail_spd, berries_avail / 70.0]
    )


# --- Per-game processing ---

def _process_game(raw_events, output_buf, label_buf, write_idx, drop_prob, rng):
    """Process one game's events, write feature vectors into output_buf.

    raw_events: list of (datetime, event_type, values_str)
    Returns: new write_idx, or -1 if buffer needs growth.
    """
    # Sort by timestamp
    raw_events.sort(key=lambda x: x[0])

    # Find gamestart for time normalization, mapstart for map config
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
        return write_idx

    # Check last event is victory
    last_type = raw_events[-1][1]
    if last_type != 'victory':
        return write_idx

    last_vals = raw_events[-1][2][1:-1].split(',')
    label = 1 if last_vals[0] == 'Blue' else 0

    # Get map lookup
    map_lookup = _MAP_LOOKUPS.get((map_name, gold_on_left))
    if map_lookup is None:
        return write_idx

    berry_lookup = map_lookup['berry_lookup']
    maiden_lookup = map_lookup['maiden_lookup']
    map_idx = map_lookup['map_index']
    total_berries = map_lookup['total_berries']

    # Initialize game state as local variables
    # w[team][worker_idx] = [is_bot, has_food, has_speed, has_wings]
    w = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
    eggs = [2, 2]
    food_dep = [[False] * 12, [False] * 12]
    food_count = [0, 0]
    maiden_states = [0, 0, 0, 0, 0]  # 0=neutral, 1=blue, -1=gold
    berries_avail = total_berries
    snail_x = float(SCREEN_WIDTH) / 2.0
    snail_vel = 0.0
    snail_last_ts = 0.0

    gold_sym = 1.0 if gold_on_left else -1.0

    # Pre-compute drop_prob check
    no_drop = (drop_prob == 0.0)

    # Event replay loop
    for dt, event_type, values_str in raw_events:
        rel_ts = (dt - gamestart_dt).total_seconds()

        # Pre-split values once per event
        vals = values_str[1:-1].split(',') if event_type not in _NO_VALS_EVENTS else None

        # Vectorize BEFORE applying this event
        if rel_ts > 5.0 and (no_drop or rng.random() > drop_prob):
            _vectorize_state(output_buf, write_idx,
                             w, eggs, food_count, maiden_states,
                             map_idx, snail_x, snail_vel, snail_last_ts,
                             rel_ts, berries_avail, gold_sym)
            label_buf[write_idx] = label
            write_idx += 1

        # Apply state mutation
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

        # gamestart, mapstart, victory: no state mutation

    return write_idx


# --- Main entry point ---

def fast_materialize(csv_path, drop_state_probability=0.0):
    """Fast path: CSV events -> (feature_matrix, labels).

    Args:
        csv_path: Glob pattern for CSV/gzip files (e.g. 'data/gameevents_*.csv.gz')
        drop_state_probability: Probability of dropping each eligible state (0.0 = keep all)

    Returns:
        (states, labels): numpy arrays of shape (N, 52) and (N,)
    """
    # Phase 1: Read all CSV rows, group by game_id
    games = {}
    game_order = []

    for filename in glob.glob(csv_path):
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                event_type = row[COL_TYPE]
                if event_type in SKIP_EVENTS:
                    continue
                game_id = int(row[COL_GAME_ID])
                if game_id not in games:
                    games[game_id] = []
                    game_order.append(game_id)
                games[game_id].append((_parse_ts(row[COL_TS]), event_type, row[COL_VALUES]))

    # Phase 2: Pre-allocate output buffers (total_events is the absolute max rows)
    total_events = sum(len(evts) for evts in games.values())
    output_buf = np.empty((total_events, NUM_FEATURES), dtype=np.float32)
    label_buf = np.empty(total_events, dtype=np.int8)
    write_idx = 0

    # Phase 3: Process each game
    rng = random.Random(42)

    for game_id in game_order:
        raw_events = games[game_id]
        try:
            write_idx = _process_game(raw_events, output_buf, label_buf, write_idx,
                                      drop_state_probability, rng)
        except Exception:
            continue

    # Phase 4: Trim to actual size
    return output_buf[:write_idx], label_buf[:write_idx]
