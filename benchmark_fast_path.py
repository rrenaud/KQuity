#!/usr/bin/env python3
"""Benchmark comparing existing pipeline vs fast path."""

import cProfile
import gc
import io
import os
import pstats
import time
import tracemalloc

import numpy as np


def run_existing(csv_path):
    from preprocess import iterate_events_from_csv, iterate_game_events_with_state, create_game_states_matrix
    from map_structure import MapStructureInfos

    map_infos = MapStructureInfos()
    events = iterate_events_from_csv(csv_path)
    game_states = iterate_game_events_with_state(events, map_infos)
    return create_game_states_matrix(game_states, drop_state_probability=0.0)


def run_fast(csv_path):
    from fast_materialize import fast_materialize
    return fast_materialize(csv_path)


def measure_peak_memory(func, csv_path):
    """Run func and return (result, peak_memory_MB) using tracemalloc."""
    gc.collect()
    tracemalloc.start()
    result = func(csv_path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)


def profile_and_time(func, csv_path, name, num_runs=3):
    # Wall-clock timing
    times = []
    result = None
    for _ in range(num_runs):
        start = time.time()
        result = func(csv_path)
        elapsed = time.time() - start
        times.append(elapsed)
    median_time = sorted(times)[len(times) // 2]

    states, labels = result

    # Memory measurement (separate run to avoid interference)
    _, peak_mb = measure_peak_memory(func, csv_path)

    # cProfile run
    pr = cProfile.Profile()
    pr.enable()
    func(csv_path)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)

    print(f'\n{"=" * 60}')
    print(f'{name}')
    print(f'{"=" * 60}')
    print(f'Wall-clock times: {[f"{t:.3f}s" for t in times]}')
    print(f'Median: {median_time:.3f}s')
    print(f'Peak memory: {peak_mb:.1f} MB')
    print(f'States shape: {states.shape}, Labels shape: {labels.shape}')
    print(f'\nTop 20 functions by cumulative time:')
    print(s.getvalue())

    return states, labels, median_time, peak_mb


def main():
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests')
    csv_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')
    expected_path = os.path.join(test_dir, 'benchmark_expected.npz')

    # Run both
    existing_states, existing_labels, existing_time, existing_mem = profile_and_time(
        run_existing, csv_path, 'EXISTING PIPELINE')
    fast_states, fast_labels, fast_time, fast_mem = profile_and_time(
        run_fast, csv_path, 'FAST PATH')

    # Verify numerical identity
    print(f'\n{"=" * 60}')
    print('VERIFICATION')
    print(f'{"=" * 60}')

    shape_match = existing_states.shape == fast_states.shape
    print(f'Shape match: {shape_match}')
    if shape_match:
        max_diff = np.max(np.abs(existing_states - fast_states))
        labels_match = np.array_equal(existing_labels, fast_labels)
        print(f'Max absolute difference: {max_diff}')
        print(f'Labels match: {labels_match}')
        print(f'Numerically identical: {max_diff < 1e-10 and labels_match}')

        if max_diff >= 1e-10:
            # Find first differing row for debugging
            diffs = np.abs(existing_states - fast_states)
            row_max = diffs.max(axis=1)
            first_bad = np.argmax(row_max > 1e-10)
            print(f'\nFirst differing row: {first_bad}')
            print(f'Existing: {existing_states[first_bad]}')
            print(f'Fast:     {fast_states[first_bad]}')
            print(f'Diff:     {diffs[first_bad]}')
    else:
        print(f'Existing shape: {existing_states.shape}, Fast shape: {fast_states.shape}')

    # Also verify against expected benchmark
    if os.path.exists(expected_path):
        expected = np.load(expected_path)
        if fast_states.shape == expected['states'].shape:
            fast_vs_expected = np.max(np.abs(fast_states - expected['states']))
            print(f'\nFast vs benchmark expected max diff: {fast_vs_expected}')
            print(f'Fast vs benchmark labels match: {np.array_equal(fast_labels, expected["labels"])}')

    # Summary
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'{"":20s} {"Existing":>12s} {"Fast":>12s} {"Ratio":>8s}')
    print(f'{"Time (median)":20s} {existing_time:11.3f}s {fast_time:11.3f}s {existing_time / fast_time:7.1f}x')
    print(f'{"Peak memory":20s} {existing_mem:10.1f}MB {fast_mem:10.1f}MB {existing_mem / fast_mem:7.1f}x')


if __name__ == '__main__':
    main()
