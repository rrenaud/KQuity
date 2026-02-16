"""Binary codec materialization: encoded games -> numpy feature matrices.

Provides single-game, per-shard, sequential, and parallel materialization
from the binary-encoded game format produced by event_codec.
"""

import glob as glob_mod
import multiprocessing
import random

import numpy as np

from constants import ContestableState, Team
from event_codec import read_packed_games, walk_game_states
from fast_materialize import (
    MAP_INDEX, NUM_FEATURES, SCREEN_WIDTH,
    _vectorize_state,
)


def _game_state_to_vectorize_args(game_state):
    """Extract the raw variables from GameState for _vectorize_state."""
    w = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
    eggs = [0, 0]
    food_count = [0, 0]

    for t in range(2):
        ts = game_state.teams[t]
        eggs[t] = ts.eggs
        food_count[t] = sum(ts.food_deposited)
        for wi in range(4):
            ws = ts.workers[wi]
            w[t][wi] = [ws.is_bot, ws.has_food, ws.has_speed, ws.has_wings]

    maiden_map = {
        ContestableState.NEUTRAL: 0,
        ContestableState.BLUE: 1,
        ContestableState.GOLD: -1,
    }
    maiden_states = [maiden_map[m] for m in game_state.maiden_states]
    map_idx = MAP_INDEX[game_state.map_info.map_id.value]
    gold_sym = 1.0 if game_state.map_info.gold_on_left else -1.0

    return (w, eggs, food_count, maiden_states, map_idx,
            game_state.snail_state.snail_x,
            game_state.snail_state.snail_velocity,
            game_state.snail_state.last_touch_timestamp,
            game_state.berries_available, gold_sym)


def _materialize_one_game_from_binary(game_id, encoded_bytes, drop_prob, rng_seed):
    """Materialize feature vectors from a single binary-encoded game.

    Returns (features, label) where features is a list of numpy rows and
    label is 0 or 1 (1 = blue wins), or ([], None) if game is invalid.
    """
    rng = random.Random(rng_seed)
    no_drop = (drop_prob == 0.0)

    rows = []
    game_state = None
    for rel_ts, game_state in walk_game_states(encoded_bytes):
        if rel_ts > 5.0 and (no_drop or rng.random() > drop_prob):
            args = _game_state_to_vectorize_args(game_state)
            (w, eggs, food_count, maiden_states, map_idx,
             snail_x, snail_vel, snail_last_ts,
             berries_avail, gold_sym) = args
            row_buf = np.empty((1, NUM_FEATURES), dtype=np.float32)
            _vectorize_state(row_buf, 0, w, eggs, food_count,
                             maiden_states, map_idx, snail_x,
                             snail_vel, snail_last_ts,
                             rel_ts, berries_avail, gold_sym)
            rows.append(row_buf[0].copy())

    if game_state is None or game_state.winning_team is None:
        return [], None

    label = 1 if game_state.winning_team == Team.BLUE else 0
    return rows, label


def _materialize_shard(shard_path, drop_prob, base_seed, max_games=None):
    """Materialize all games in one binary shard.

    Returns (features, labels, game_ids, timestamps) as numpy arrays,
    or empty arrays if no valid states.
    """
    all_rows = []
    all_labels = []
    all_game_ids = []
    games_processed = 0

    for game_id, encoded_bytes in read_packed_games(shard_path):
        if max_games is not None and games_processed >= max_games:
            break
        seed = base_seed + game_id
        rows, label = _materialize_one_game_from_binary(
            game_id, encoded_bytes, drop_prob, seed)
        games_processed += 1
        if not rows or label is None:
            continue
        for row in rows:
            all_rows.append(row)
            all_labels.append(label)
            all_game_ids.append(game_id)

    if not all_rows:
        return (np.empty((0, NUM_FEATURES), dtype=np.float32),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int64))

    return (np.array(all_rows, dtype=np.float32),
            np.array(all_labels, dtype=np.int8),
            np.array(all_game_ids, dtype=np.int64))


def parallel_materialize_bins(shard_glob, drop_prob=0.0, num_workers=4, base_seed=0,
                              max_games=None):
    """Materialize binary shards in parallel using multiprocessing.

    Args:
        max_games: Optional total game limit, distributed evenly across shards.
            With striped (round-robin) sharding, this gives each worker an equal
            slice and each shard stops early via per-shard limit.

    Returns (features, labels, game_ids) as numpy arrays.
    """
    shard_paths = sorted(glob_mod.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f'No shards found matching {shard_glob}')

    per_shard = None
    if max_games is not None:
        per_shard = max(1, max_games // len(shard_paths))

    args = [(path, drop_prob, base_seed, per_shard) for path in shard_paths]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(_materialize_shard, args)

    features = np.concatenate([r[0] for r in results if r[0].shape[0] > 0])
    labels = np.concatenate([r[1] for r in results if r[1].shape[0] > 0])
    game_ids = np.concatenate([r[2] for r in results if r[2].shape[0] > 0])

    return features, labels, game_ids


def sequential_materialize_bins(shard_glob, drop_prob=0.0, base_seed=0,
                                max_games=None):
    """Materialize binary shards sequentially (single-process baseline).

    Args:
        max_games: Optional total game limit, distributed evenly across shards.

    Returns (features, labels, game_ids) as numpy arrays.
    """
    shard_paths = sorted(glob_mod.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f'No shards found matching {shard_glob}')

    per_shard = None
    if max_games is not None:
        per_shard = max(1, max_games // len(shard_paths))

    results = []
    for path in shard_paths:
        results.append(_materialize_shard(path, drop_prob, base_seed, per_shard))

    features = np.concatenate([r[0] for r in results if r[0].shape[0] > 0])
    labels = np.concatenate([r[1] for r in results if r[1].shape[0] > 0])
    game_ids = np.concatenate([r[2] for r in results if r[2].shape[0] > 0])

    return features, labels, game_ids
