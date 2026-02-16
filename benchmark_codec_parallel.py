"""Benchmark: CSV vs sequential binary vs parallel binary materialization.

Operates on logged_in_games/ dataset.
"""

import os
import time

import numpy as np

from event_codec import encode_csv_to_sharded_bins
from fast_materialize import fast_materialize
from codec_materialize import parallel_materialize_bins, sequential_materialize_bins

DATA_DIR = os.path.join(os.path.dirname(__file__), 'logged_in_games')
CSV_GLOB = os.path.join(DATA_DIR, 'gameevents_*.csv.gz')
SHARD_DIR = os.path.join(DATA_DIR, 'shards')
SHARD_GLOB = os.path.join(SHARD_DIR, 'shard_*.bin')
NUM_SHARDS = 20
MAX_GAMES = 10_000


def _total_csv_size():
    import glob
    total = 0
    for f in glob.glob(CSV_GLOB):
        total += os.path.getsize(f)
    return total


def _total_shard_size():
    import glob
    total = 0
    for f in glob.glob(SHARD_GLOB):
        total += os.path.getsize(f)
    return total


def main():
    print('=' * 60)
    print(f'Parallel Codec Materialization Benchmark (max {MAX_GAMES:,} games)')
    print('=' * 60)

    # Step 1: Encode CSV -> shards (if not already done)
    import glob
    existing_shards = glob.glob(SHARD_GLOB)
    if not existing_shards:
        print(f'\nEncoding CSV -> {NUM_SHARDS} binary shards...')
        t0 = time.perf_counter()
        encoded, rejected, shard_paths = encode_csv_to_sharded_bins(
            CSV_GLOB, SHARD_DIR, NUM_SHARDS)
        encode_time = time.perf_counter() - t0

        csv_size = _total_csv_size()
        shard_size = _total_shard_size()
        print(f'  Encoded: {encoded} games ({rejected} rejected)')
        print(f'  Time: {encode_time:.1f}s')
        print(f'  CSV size: {csv_size / 1e6:.1f} MB (gzipped)')
        print(f'  Shard size: {shard_size / 1e6:.1f} MB (binary)')
        print(f'  Compression vs gzipped CSV: {csv_size / shard_size:.2f}x')
    else:
        print(f'\nUsing existing {len(existing_shards)} shards in {SHARD_DIR}')

    # Step 2: CSV sequential baseline
    print('\n--- CSV sequential (fast_materialize) ---')
    t0 = time.perf_counter()
    csv_states, csv_labels, csv_game_ids, csv_ts = fast_materialize(
        CSV_GLOB, drop_state_probability=0.0, max_games=MAX_GAMES)
    csv_time = time.perf_counter() - t0
    csv_n_games = np.unique(csv_game_ids).shape[0]
    print(f'  Games: {csv_n_games:,}')
    print(f'  States: {csv_states.shape[0]:,}')
    print(f'  Time: {csv_time:.2f}s')

    # Step 3: Binary sequential
    print('\n--- Binary sequential ---')
    t0 = time.perf_counter()
    seq_states, seq_labels, seq_game_ids = sequential_materialize_bins(
        SHARD_GLOB, drop_prob=0.0, base_seed=0, max_games=MAX_GAMES)
    seq_time = time.perf_counter() - t0
    seq_n_games = np.unique(seq_game_ids).shape[0]
    print(f'  Games: {seq_n_games:,}')
    print(f'  States: {seq_states.shape[0]:,}')
    print(f'  Time: {seq_time:.2f}s')
    print(f'  Speedup vs CSV: {csv_time / seq_time:.2f}x')

    # Step 4: Binary parallel with various worker counts
    worker_counts = [2, 4, 8]
    for nw in worker_counts:
        print(f'\n--- Binary parallel ({nw} workers) ---')
        t0 = time.perf_counter()
        par_states, par_labels, par_game_ids = parallel_materialize_bins(
            SHARD_GLOB, drop_prob=0.0, num_workers=nw, base_seed=0,
            max_games=MAX_GAMES)
        par_time = time.perf_counter() - t0
        par_n_games = np.unique(par_game_ids).shape[0]
        print(f'  Games: {par_n_games:,}')
        print(f'  States: {par_states.shape[0]:,}')
        print(f'  Time: {par_time:.2f}s')
        print(f'  Speedup vs CSV: {csv_time / par_time:.2f}x')
        print(f'  Speedup vs seq binary: {seq_time / par_time:.2f}x')

    # Summary
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    print(f'  CSV sequential:    {csv_time:.2f}s ({csv_states.shape[0]:,} states)')
    print(f'  Binary sequential: {seq_time:.2f}s ({seq_states.shape[0]:,} states)')


if __name__ == '__main__':
    main()
