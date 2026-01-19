#!/usr/bin/env python3
"""Benchmark comparing memory usage between different data storage approaches."""

import glob
import gc
import os
import sys
import numpy as np
import concurrent.futures
from memory_benchmark import MemoryBenchmark, get_peak_memory_mb, print_memory_stats
from preprocess import process_single_file, iterate_events_from_csv, iterate_game_events_with_state, create_game_states_matrix
import map_structure


def process_single_file_with_dtype(args):
    """Process a single CSV file with specified dtype."""
    csv_file, drop_state_probability, dtype = args
    map_structure_infos = map_structure.MapStructureInfos()
    game_states_iterable = iterate_game_events_with_state(
        iterate_events_from_csv(csv_file), map_structure_infos)
    game_state_matrix, labels = create_game_states_matrix(
        game_states_iterable, drop_state_probability, noisy=False, dtype=dtype)
    return csv_file, game_state_matrix, labels


def materialize_shards_with_dtype(files, drop_prob, dtype, max_workers=8):
    """Materialize game states from files with specified dtype."""
    all_states, all_labels = [], []
    args_list = [(f, drop_prob, dtype) for f in files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file_with_dtype, args): args[0] for args in args_list}
        for future in concurrent.futures.as_completed(futures):
            _, states, labels = future.result()
            all_states.append(states)
            all_labels.append(labels)

    return np.vstack(all_states), np.concatenate(all_labels)


def run_benchmark(num_shards=10, drop_prob=0.9):
    """Run benchmark comparing float32 vs float64."""
    print("=" * 60)
    print("MEMORY BENCHMARK: float32 vs float64")
    print("=" * 60)
    print(f"Shards: {num_shards}, drop_prob: {drop_prob}")
    print()

    # Get test files
    all_files = sorted(glob.glob('new_data_partitioned/gameevents_*.csv.gz'))
    test_files = all_files[:num_shards]
    print(f"Using {len(test_files)} files")
    print()

    results = []

    # Benchmark float64
    print("-" * 40)
    print("Testing float64...")
    gc.collect()
    with MemoryBenchmark("float64 materialization") as bench64:
        states64, labels64 = materialize_shards_with_dtype(test_files, drop_prob, np.float64)

    result64 = {
        'dtype': 'float64',
        'elapsed': bench64.result.elapsed_seconds,
        'peak_mb': bench64.result.peak_memory_mb,
        'samples': len(labels64),
        'features': states64.shape[1],
        'array_mb': states64.nbytes / 1024 / 1024,
    }
    results.append(result64)
    print(f"  Samples: {result64['samples']:,}")
    print(f"  Features: {result64['features']}")
    print(f"  Array size: {result64['array_mb']:.1f} MB")
    print(f"  Dtype: {states64.dtype}")

    # Free memory
    del states64, labels64
    gc.collect()
    print()

    # Benchmark float32
    print("-" * 40)
    print("Testing float32...")
    gc.collect()
    with MemoryBenchmark("float32 materialization") as bench32:
        states32, labels32 = materialize_shards_with_dtype(test_files, drop_prob, np.float32)

    result32 = {
        'dtype': 'float32',
        'elapsed': bench32.result.elapsed_seconds,
        'peak_mb': bench32.result.peak_memory_mb,
        'samples': len(labels32),
        'features': states32.shape[1],
        'array_mb': states32.nbytes / 1024 / 1024,
    }
    results.append(result32)
    print(f"  Samples: {result32['samples']:,}")
    print(f"  Features: {result32['features']}")
    print(f"  Array size: {result32['array_mb']:.1f} MB")
    print(f"  Dtype: {states32.dtype}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dtype':<10} {'Time (s)':<12} {'Peak (MB)':<12} {'Array (MB)':<12} {'Samples':<12}")
    print("-" * 58)
    for r in results:
        print(f"{r['dtype']:<10} {r['elapsed']:<12.1f} {r['peak_mb']:<12.0f} {r['array_mb']:<12.1f} {r['samples']:<12,}")

    print()
    print(f"Memory savings (float32 vs float64):")
    print(f"  Array size: {result64['array_mb'] - result32['array_mb']:.1f} MB ({100 * (1 - result32['array_mb']/result64['array_mb']):.1f}% reduction)")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=int, default=10, help='Number of shards to process')
    parser.add_argument('--drop', type=float, default=0.9, help='Drop probability')
    args = parser.parse_args()

    run_benchmark(num_shards=args.shards, drop_prob=args.drop)
