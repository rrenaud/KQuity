"""Binary codec materialization: encoded games -> numpy feature matrices.

Provides per-shard, sequential, and parallel materialization from the
binary-encoded game format produced by event_codec.
"""

import glob as glob_mod
import multiprocessing

import numpy as np

from event_codec import materialize_entries, read_packed_games
from fast_materialize import NUM_FEATURES


def _materialize_shard(shard_path, drop_prob, max_games=None):
    """Materialize all games in one binary shard.

    Returns (features, labels, game_ids, timestamps) as numpy arrays,
    or empty arrays if no valid states.
    """
    def _iter_entries():
        games_processed = 0
        for game_id, encoded_bytes in read_packed_games(shard_path):
            if max_games is not None and games_processed >= max_games:
                break
            games_processed += 1
            yield game_id, encoded_bytes

    return materialize_entries(_iter_entries(), drop_prob)


def _collect_results(results):
    """Concatenate per-shard results, handling the all-empty case."""
    non_empty = [r for r in results if r[0].shape[0] > 0]
    if not non_empty:
        return (np.empty((0, NUM_FEATURES), dtype=np.float32),
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32))
    features = np.concatenate([r[0] for r in non_empty])
    labels = np.concatenate([r[1] for r in non_empty])
    game_ids = np.concatenate([r[2] for r in non_empty])
    timestamps = np.concatenate([r[3] for r in non_empty])
    return features, labels, game_ids, timestamps


def parallel_materialize_bins(shard_glob, drop_prob=0.0, num_workers=4,
                              max_games=None):
    """Materialize binary shards in parallel using multiprocessing.

    Args:
        max_games: Optional total game limit, distributed evenly across shards.
            With striped (round-robin) sharding, this gives each worker an equal
            slice and each shard stops early via per-shard limit.
            Note: may slightly overshoot if max_games < num_shards (clamped to
            1 per shard minimum).

    Returns (features, labels, game_ids, timestamps) as numpy arrays.
    """
    shard_paths = sorted(glob_mod.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f'No shards found matching {shard_glob}')

    per_shard = None
    if max_games is not None:
        per_shard = max(1, max_games // len(shard_paths))

    args = [(path, drop_prob, per_shard) for path in shard_paths]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(_materialize_shard, args)

    return _collect_results(results)


def sequential_materialize_bins(shard_glob, drop_prob=0.0, max_games=None):
    """Materialize binary shards sequentially (single-process baseline).

    Args:
        max_games: Optional total game limit, distributed evenly across shards.
            Note: may slightly overshoot if max_games < num_shards (clamped to
            1 per shard minimum).

    Returns (features, labels, game_ids, timestamps) as numpy arrays.
    """
    shard_paths = sorted(glob_mod.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f'No shards found matching {shard_glob}')

    per_shard = None
    if max_games is not None:
        per_shard = max(1, max_games // len(shard_paths))

    results = []
    for path in shard_paths:
        results.append(_materialize_shard(path, drop_prob, per_shard))

    return _collect_results(results)
