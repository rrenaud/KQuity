"""Binary codec materialization: encoded games -> numpy feature matrices.

Provides per-shard, sequential, and parallel materialization from the
binary-encoded game format produced by event_codec.
"""

import collections.abc
import glob
import math
import multiprocessing

import numpy as np
import numpy.typing as npt

from event_codec import materialize_entries, read_packed_games
from fast_materialize import NUM_FEATURES
from _types import MaterializeResult


def _materialize_shard(shard_path: str, drop_prob: float,
                       max_games: int | None = None) -> MaterializeResult:
    """Materialize all games in one binary shard.

    Returns (features, labels, game_ids, timestamps) as numpy arrays,
    or empty arrays if no valid states.
    """
    def _iter_entries() -> collections.abc.Iterator[tuple[int, bytes]]:
        games_processed = 0
        for game_id, encoded_bytes in read_packed_games(shard_path):
            if max_games is not None and games_processed >= max_games:
                break
            games_processed += 1
            yield game_id, encoded_bytes

    return materialize_entries(_iter_entries(), drop_prob)


def _resolve_shards(shard_glob: str,
                    max_games: int | None) -> tuple[list[str], int | None]:
    """Resolve shard paths and compute per-shard game limit.

    Returns (shard_paths, per_shard) where per_shard is None if no limit.
    Uses ceiling division so total across shards is at most
    max_games + num_shards - 1 (each shard gets ceil(max_games / num_shards)).
    """
    shard_paths = sorted(glob.glob(shard_glob))
    if not shard_paths:
        raise ValueError(f'No shards found matching {shard_glob}')
    per_shard: int | None = None
    if max_games is not None:
        per_shard = max(1, math.ceil(max_games / len(shard_paths)))
    return shard_paths, per_shard


def _collect_results(results: list[MaterializeResult]) -> MaterializeResult:
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


def parallel_materialize_bins(shard_glob: str, drop_prob: float = 0.0,
                              num_workers: int = 4,
                              max_games: int | None = None) -> MaterializeResult:
    """Materialize binary shards in parallel using multiprocessing.

    Args:
        max_games: Optional total game limit, distributed across shards via
            ceiling division. Each shard processes at most
            ceil(max_games / num_shards) games, so the total may exceed
            max_games by up to num_shards - 1.

    Returns (features, labels, game_ids, timestamps) as numpy arrays.
    """
    shard_paths, per_shard = _resolve_shards(shard_glob, max_games)
    args = [(path, drop_prob, per_shard) for path in shard_paths]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(_materialize_shard, args)

    return _collect_results(results)


def sequential_materialize_bins(shard_glob: str, drop_prob: float = 0.0,
                                max_games: int | None = None) -> MaterializeResult:
    """Materialize binary shards sequentially (single-process baseline).

    Args:
        max_games: Optional total game limit, distributed across shards via
            ceiling division. Each shard processes at most
            ceil(max_games / num_shards) games, so the total may exceed
            max_games by up to num_shards - 1.

    Returns (features, labels, game_ids, timestamps) as numpy arrays.
    """
    shard_paths, per_shard = _resolve_shards(shard_glob, max_games)

    results: list[MaterializeResult] = []
    for path in shard_paths:
        results.append(_materialize_shard(path, drop_prob, per_shard))

    return _collect_results(results)
