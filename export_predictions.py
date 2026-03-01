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
import multiprocessing
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


def _compute_egg_grid(
    w: list[list[list[bool]]], food_count: list[int],
    maiden_states: list[int], map_idx: int, snail_x: float,
    snail_vel: float, snail_last_ts: float, rel_ts: float,
    berries_avail: int, gold_sym: float,
) -> list[list[float] | None]:
    """Compute feature vectors for all 9 (blue_eggs, gold_eggs) combinations.

    Returns list of 9 feature vectors indexed as [blue_eggs * 3 + gold_eggs].
    """
    vectors: list[list[float] | None] = []
    for be in range(3):
        for ge in range(3):
            ce = [be, ge]
            vec = _vectorize_counterfactual(
                w, ce, food_count, maiden_states, map_idx, snail_x,
                snail_vel, snail_last_ts, rel_ts, berries_avail, gold_sym)
            vectors.append(vec)
    return vectors


BERRY_DELTAS = [0, 1, 2, 3, 4]
N_BERRY_DELTAS = len(BERRY_DELTAS)


def _compute_berry_grid(
    w: list[list[list[bool]]], eggs: list[int], food_count: list[int],
    maiden_states: list[int], map_idx: int, snail_x: float,
    snail_vel: float, snail_last_ts: float, rel_ts: float,
    berries_avail: int, gold_sym: float,
) -> list[list[float] | None]:
    """Compute feature vectors for berry delta combinations [0..4].

    Each delta adds berries to the current food_count (clamped to 12).
    Returns list of N_BERRY_DELTAS^2 feature vectors indexed as
    [blue_delta_idx * N_BERRY_DELTAS + gold_delta_idx].
    """
    vectors: list[list[float] | None] = []
    for bd in BERRY_DELTAS:
        for gd in BERRY_DELTAS:
            cfc = [min(12, food_count[0] + bd),
                   min(12, food_count[1] + gd)]
            if cfc[0] >= 12 or cfc[1] >= 12:
                vectors.append(None)  # game-over state, skip prediction
                continue
            total_delta = (cfc[0] - food_count[0]) + (cfc[1] - food_count[1])
            cba = max(0, berries_avail - total_delta)
            vec = _vectorize_counterfactual(
                w, eggs, cfc, maiden_states, map_idx, snail_x,
                snail_vel, snail_last_ts, rel_ts, cba, gold_sym)
            vectors.append(vec)
    return vectors


SNAIL_CHUNKS = 12
# Snail track endpoints in pixel space (hive x-coordinates, same for all maps)
SNAIL_LEFT_PX = 60
SNAIL_RIGHT_PX = 1860
SNAIL_TRACK_PX = SNAIL_RIGHT_PX - SNAIL_LEFT_PX
# Grid positions: 12 evenly-spaced pixel positions along the track
SNAIL_POSITIONS_PX = [SNAIL_LEFT_PX + (i + 0.5) / SNAIL_CHUNKS * SNAIL_TRACK_PX
                      for i in range(SNAIL_CHUNKS)]


def _snail_px_to_index(px: float) -> int:
    """Quantize a pixel x-position to a snail bar cell index 0-11."""
    frac = (px - SNAIL_LEFT_PX) / SNAIL_TRACK_PX
    idx = int(frac * SNAIL_CHUNKS)
    return max(0, min(SNAIL_CHUNKS - 1, idx))


def _compute_snail_grid(
    w: list[list[list[bool]]], eggs: list[int], food_count: list[int],
    maiden_states: list[int], map_idx: int, snail_x: float,
    snail_vel: float, snail_last_ts: float, rel_ts: float,
    berries_avail: int, gold_sym: float,
) -> tuple[list[list[float] | None], list[float] | None]:
    """Compute feature vectors for 12 discrete snail positions + 1 takeover.

    Positions are evenly spaced along the track (pixel 60-1860).
    gold_on_left: pixel 60=gold hive (left), pixel 1860=blue hive (right).

    Returns (position_vectors, takeover_vector) where:
    - position_vectors: 12 feature vectors with snail at each discrete position
      (keeping current velocity/owner)
    - takeover_vector: 1 feature vector with current position but flipped velocity
    """
    vectors: list[list[float] | None] = []
    for px in SNAIL_POSITIONS_PX:
        # gold_on_left: px is the actual pixel position
        # !gold_on_left: the game mirror flips, so pixel 60 is blue hive
        # _vectorize_counterfactual handles the gold_sym normalization internally
        actual_px = px if gold_sym > 0 else (SCREEN_WIDTH - px)
        # rel_ts passed as both snail_last_ts and event_ts so that
        # inferred_pos = actual_px + (event_ts - snail_last_ts)*vel = actual_px
        vec = _vectorize_counterfactual(
            w, eggs, food_count, maiden_states, map_idx, actual_px,
            snail_vel, rel_ts, rel_ts, berries_avail, gold_sym)
        vectors.append(vec)

    # Takeover vector: same position, flipped velocity
    takeover_vec = _vectorize_counterfactual(
        w, eggs, food_count, maiden_states, map_idx, snail_x,
        -snail_vel, snail_last_ts, rel_ts, berries_avail, gold_sym)
    return vectors, takeover_vec


def _vectorize_game(
    raw_events: list[tuple[datetime.datetime, str, str]],
    counterfactuals: bool = False,
) -> dict[str, Any] | None:
    """Replay one game's events and produce all feature vectors.

    Returns a dict with vectorized data (no model needed), or None if invalid.
    This is the CPU-heavy step that benefits from multiprocessing.
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
    # For egg grid: list of (event_idx, 9 feature vectors, [current_blue_eggs, current_gold_eggs])
    eg_entries: list[tuple[int, list[list[float] | None], list[int]]] = [] if counterfactuals else []
    # For berry grid: list of (event_idx, 25 feature vectors, [current_blue_food, current_gold_food])
    bg_entries: list[tuple[int, list[list[float] | None], list[int]]] = [] if counterfactuals else []
    # For snail grid: list of (event_idx, 12 position vectors, takeover vector, snail_x, snail_vel, gold_sym)
    sg_entries: list[tuple[int, list[list[float] | None], list[float] | None, float, float, float]] = [] if counterfactuals else []

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
                eg_vecs = _compute_egg_grid(
                    w, food_count, maiden_states,
                    map_idx, snail_x, snail_vel, snail_last_ts,
                    rel_ts, berries_avail, gold_sym)
                eg_entries.append((event_idx, eg_vecs, list(eggs)))
                bg_vecs = _compute_berry_grid(
                    w, eggs, food_count, maiden_states,
                    map_idx, snail_x, snail_vel, snail_last_ts,
                    rel_ts, berries_avail, gold_sym)
                bg_entries.append((event_idx, bg_vecs, list(food_count)))
                sg_vecs, sg_takeover = _compute_snail_grid(
                    w, eggs, food_count, maiden_states,
                    map_idx, snail_x, snail_vel, snail_last_ts,
                    rel_ts, berries_avail, gold_sym)
                sg_entries.append((event_idx, sg_vecs, sg_takeover, snail_x, snail_vel, gold_sym))

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

    return {
        'states': states,
        'timestamps': timestamps,
        'gold_on_left': gold_on_left,
        'counterfactuals': counterfactuals,
        'cf_entries': cf_entries,
        'eg_entries': eg_entries,
        'bg_entries': bg_entries,
        'sg_entries': sg_entries,
    }


def _vectorize_game_wrapper(
    args: tuple[int, list[tuple[datetime.datetime, str, str]], bool],
) -> tuple[int, dict[str, Any] | None]:
    """Wrapper for multiprocessing Pool — unpacks args tuple."""
    game_id, events, counterfactuals = args
    return game_id, _vectorize_game(events, counterfactuals)


def _predict_and_assemble(
    vec_data: dict[str, Any],
    model: lgb.Booster,
) -> tuple[list[dict[str, Any]], bool]:
    """Run model prediction on vectorized game data and assemble results."""
    states = vec_data['states']
    timestamps = vec_data['timestamps']
    gold_on_left = vec_data['gold_on_left']
    counterfactuals = vec_data['counterfactuals']
    cf_entries = vec_data['cf_entries']
    eg_entries = vec_data['eg_entries']
    bg_entries = vec_data['bg_entries']
    sg_entries = vec_data['sg_entries']

    if not counterfactuals or not cf_entries:
        # Simple path: just baseline predictions
        X = np.array(states, dtype=np.float32)
        preds = model.predict(X)
        return ([{'t': round(t, 2), 'p': round(float(p), 4)}
                 for t, p in zip(timestamps, preds)], gold_on_left)

    # Counterfactual path: batch all vectors into one predict call
    n_baselines = len(states)
    all_vectors = list(states)

    cf_map: list[tuple[int, str, int]] = []
    for event_idx, name, vec in cf_entries:
        start = len(all_vectors)
        all_vectors.append(vec)
        cf_map.append((event_idx, name, start))

    eg_map: list[tuple[int, int, list[int]]] = []
    for event_idx, eg_vecs, current_eggs in eg_entries:
        start = len(all_vectors)
        all_vectors.extend(eg_vecs)
        eg_map.append((event_idx, start, current_eggs))

    bg_map: list[tuple[int, int, list[int], list[bool]]] = []
    for event_idx, bg_vecs, current_food in bg_entries:
        start = len(all_vectors)
        # Track which positions are None (game-over states) vs valid
        valid_mask = [v is not None for v in bg_vecs]
        all_vectors.extend(v for v in bg_vecs if v is not None)
        bg_map.append((event_idx, start, current_food, valid_mask))

    sg_map: list[tuple[int, int, int | None, float, float, float]] = []
    for event_idx, sg_vecs, sg_takeover, sx, sv, gsym in sg_entries:
        pos_start = len(all_vectors)
        all_vectors.extend(v for v in sg_vecs if v is not None)
        takeover_idx: int | None = None
        if sg_takeover is not None:
            takeover_idx = len(all_vectors)
            all_vectors.append(sg_takeover)
        sg_map.append((event_idx, pos_start, takeover_idx, sx, sv, gsym))

    X = np.array(all_vectors, dtype=np.float32)
    all_preds = model.predict(X)
    baseline_preds = all_preds[:n_baselines]

    # Build per-event counterfactual dicts
    cf_dicts: list[dict[str, float]] = [{} for _ in range(n_baselines)]
    for event_idx, name, start in cf_map:
        delta = float(all_preds[start] - baseline_preds[event_idx])
        if abs(delta) >= 0.001:  # skip negligible
            cf_dicts[event_idx][name] = round(delta, 4)

    # Build per-event egg grid arrays
    eg_arrays: dict[int, list[float]] = {}
    ee_arrays: dict[int, list[int]] = {}
    for event_idx, start, current_eggs in eg_map:
        eg_arrays[event_idx] = [round(float(all_preds[start + i]), 4) for i in range(9)]
        ee_arrays[event_idx] = current_eggs

    # Build per-event berry grid arrays (re-inserting None for game-over states)
    bg_arrays: dict[int, list[float | None]] = {}
    bc_arrays: dict[int, list[int]] = {}
    for event_idx, start, current_food, valid_mask in bg_map:
        arr: list[float | None] = []
        pred_idx = start
        for is_valid in valid_mask:
            if is_valid:
                arr.append(round(float(all_preds[pred_idx]), 4))
                pred_idx += 1
            else:
                arr.append(None)
        bg_arrays[event_idx] = arr
        bc_arrays[event_idx] = current_food

    # Build per-event snail grid arrays
    sp_arrays: dict[int, list[float]] = {}
    sc_arrays: dict[int, int] = {}
    st_arrays: dict[int, float] = {}
    for event_idx, pos_start, takeover_idx, sx, sv, gsym in sg_map:
        sp_arrays[event_idx] = [round(float(all_preds[pos_start + i]), 4) for i in range(SNAIL_CHUNKS)]
        # Quantize current snail position to track-relative index 0-11
        baseline_vec = states[event_idx]
        if baseline_vec is not None:
            norm_pos = baseline_vec[49]  # snail_pos feature
            # Recover pixel position: norm_pos = (px / 1920 - 0.5) * gold_sym
            inferred_px = (norm_pos / gsym + 0.5) * SCREEN_WIDTH
            # For !gold_on_left (gsym=-1), pixel space is mirrored
            if gsym < 0:
                inferred_px = SCREEN_WIDTH - inferred_px
            sc_arrays[event_idx] = _snail_px_to_index(inferred_px)
        if takeover_idx is not None:
            st_arrays[event_idx] = round(float(all_preds[takeover_idx]), 4)

    results: list[dict[str, Any]] = []
    for i, (t, p) in enumerate(zip(timestamps, baseline_preds)):
        entry: dict[str, Any] = {'t': round(t, 2), 'p': round(float(p), 4)}
        if cf_dicts[i]:
            entry['c'] = cf_dicts[i]
        if i in eg_arrays:
            entry['eg'] = eg_arrays[i]
            entry['ee'] = ee_arrays[i]
        if i in bg_arrays:
            entry['bg'] = bg_arrays[i]
            entry['bc'] = bc_arrays[i]
        if i in sp_arrays:
            entry['sp'] = sp_arrays[i]
            if i in sc_arrays:
                entry['sc'] = sc_arrays[i]
            if i in st_arrays:
                entry['st'] = st_arrays[i]
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

    # Phase 1: Vectorize all games in parallel (CPU-bound Python)
    work_items = [(game_id, events, args.counterfactuals)
                  for game_id, events in sorted(games.items())]
    n_workers = min(len(work_items), multiprocessing.cpu_count() or 1)
    print(f"Vectorizing {len(work_items)} games with {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        vec_results = pool.map(_vectorize_game_wrapper, work_items)

    # Phase 2: Predict sequentially (LightGBM uses threads internally)
    results: dict[str, list[dict[str, Any]]] = {}
    gold_on_left_map: dict[str, bool] = {}
    for game_id, vec_data in vec_results:
        if vec_data is None:
            continue
        preds, gol = _predict_and_assemble(vec_data, model)
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
