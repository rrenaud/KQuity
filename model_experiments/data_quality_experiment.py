#!/usr/bin/env python3
"""Compare win-probability models trained on quality_filtered vs logged_in_games.

Uses tournament games as held-out evaluation set.
Supports single-point runs or a full scaling grid (--grid).
"""

import argparse
import gc
import json
import os
import sys
import time

# Add repo root to path so imports work from model_experiments/ subdir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import lightgbm as lgb
import sklearn.metrics

import random as random_mod

from event_codec import fast_materialize_from_codec, read_packed_games
from eval_metrics import (
    compute_standard_metrics, compute_kq_metrics,
)
from symmetry import swap_teams

SOURCES = {
    'quality_filtered': 'quality_filtered/encoded/all_games.bin',
    'logged_in_games': 'logged_in_games/encoded/all_games.bin',
}

# Grid informed by prior KQuity LGB scaling experiment (commit bf28e14).
# Loss plateaus ~5K games; egg inversions improve to ~10K.
# Large models overfit badly at small data sizes.
GRID = [
    # (max_games, num_leaves, num_trees)
    (500,   31,  50),
    (2000,  31,  50),
    (2000, 100, 100),
    (5000,  31,  50),
    (5000, 100, 100),
    (5000, 200, 200),
    (20000, 100, 100),
    (20000, 200, 200),
    (50000, 100, 100),
    (50000, 200, 200),
]


def load_and_materialize(bin_path, drop_state_probability=0.0, max_games=None):
    """Load binary-encoded games and materialize features."""
    print(f"  Materializing {bin_path} (max_games={max_games})...")
    start = time.time()
    states, labels, game_ids, timestamps = fast_materialize_from_codec(
        bin_path, drop_state_probability=drop_state_probability,
        max_games=max_games)
    elapsed = time.time() - start
    n_games = len(np.unique(game_ids))
    print(f"  {len(labels):,} states from {n_games:,} games in {elapsed:.1f}s")
    return states, labels, game_ids, timestamps


def load_all_entries(bin_path):
    """Read all (game_id, encoded_bytes) from a packed binary file into memory."""
    return list(read_packed_games(bin_path))


def materialize_from_entries(entries):
    """Materialize features from a list of (game_id, encoded_bytes) in memory.

    Same logic as fast_materialize_from_codec but operates on pre-loaded entries
    so we can shuffle/subsample without re-reading the file.
    """
    from event_codec import (
        walk_game_states, _game_state_to_vectorize_args, _vectorize_state,
        NUM_FEATURES, Team,
    )

    capacity = len(entries) * 300
    output_buf = np.empty((capacity, NUM_FEATURES), dtype=np.float32)
    label_buf = np.empty(capacity, dtype=np.int8)
    game_id_buf = np.empty(capacity, dtype=np.int64)
    timestamp_buf = np.empty(capacity, dtype=np.float32)
    write_idx = 0

    def _grow(needed):
        nonlocal capacity, output_buf, label_buf, game_id_buf, timestamp_buf
        new_cap = max(capacity + capacity // 2, needed)
        for old, dtype, cols in [
            (output_buf, np.float32, NUM_FEATURES),
            (label_buf, np.int8, None),
            (game_id_buf, np.int64, None),
            (timestamp_buf, np.float32, None),
        ]:
            shape = (new_cap, cols) if cols else (new_cap,)
            new = np.empty(shape, dtype=dtype)
            new[:write_idx] = old[:write_idx]
            if cols:
                output_buf = new
            elif dtype == np.int8:
                label_buf = new
            elif dtype == np.int64:
                game_id_buf = new
            else:
                timestamp_buf = new
        capacity = new_cap

    for game_id, encoded in entries:
        start_idx = write_idx
        game_state = None
        try:
            for rel_ts, game_state in walk_game_states(encoded):
                if rel_ts > 5.0:
                    if write_idx >= capacity:
                        _grow(write_idx + 1)
                    args = _game_state_to_vectorize_args(game_state)
                    (w, eggs, food_count, maiden_states, map_idx,
                     snail_x, snail_vel, snail_last_ts,
                     berries_avail, gold_sym) = args
                    _vectorize_state(output_buf, write_idx, w, eggs, food_count,
                                     maiden_states, map_idx, snail_x,
                                     snail_vel, snail_last_ts,
                                     rel_ts, berries_avail, gold_sym)
                    timestamp_buf[write_idx] = rel_ts
                    write_idx += 1
        except Exception:
            write_idx = start_idx
            continue

        if game_state is None or game_state.winning_team is None:
            write_idx = start_idx
            continue

        label = 1 if game_state.winning_team == Team.BLUE else 0
        label_buf[start_idx:write_idx] = label
        game_id_buf[start_idx:write_idx] = game_id

    return (output_buf[:write_idx], label_buf[:write_idx],
            game_id_buf[:write_idx], timestamp_buf[:write_idx])


def remove_leakage(train_states, train_labels, train_game_ids, train_timestamps,
                   holdout_game_ids):
    """Remove holdout game_ids from training data."""
    holdout_set = set(holdout_game_ids)
    mask = np.array([gid not in holdout_set for gid in train_game_ids])
    removed = (~mask).sum()
    if removed > 0:
        print(f"  Removed {removed:,} states from {len(np.unique(train_game_ids[~mask])):,} "
              f"overlapping games")
    return (train_states[mask], train_labels[mask],
            train_game_ids[mask], train_timestamps[mask])


def train_model(train_X, train_y, num_leaves, num_trees):
    """Train a LightGBM binary classifier."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': -1,
    }
    train_data = lgb.Dataset(train_X, train_y)
    start = time.time()
    model = lgb.train(param, train_data, num_boost_round=num_trees)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s ({num_leaves} leaves, {num_trees} trees)")
    return model


def evaluate_model(model, test_X, test_y, test_timestamps):
    """Evaluate model and return metrics dict."""
    predictions = model.predict(test_X)
    labels_float = test_y.astype(np.float64)

    std = compute_standard_metrics(predictions, labels_float)
    kq = compute_kq_metrics(model, test_X, labels_float)

    return {**std, **kq}


def run_single(holdout_X, holdout_y, holdout_gids, holdout_ts,
               max_games, num_leaves, num_trees):
    """Run one grid point: train both sources, return metrics dicts."""
    row = {}
    for name, bin_path in SOURCES.items():
        print(f"\n--- {name} | {max_games} games, {num_leaves}L/{num_trees}T ---")

        train_X, train_y, train_gids, train_ts = load_and_materialize(
            bin_path, max_games=max_games)

        train_X, train_y, train_gids, train_ts = remove_leakage(
            train_X, train_y, train_gids, train_ts, holdout_gids)

        n_states = len(train_y)
        n_games = len(np.unique(train_gids))
        print(f"  Training: {n_states:,} states from {n_games:,} games")

        model = train_model(train_X, train_y, num_leaves, num_trees)

        print("  Evaluating on tournament holdout...")
        metrics = evaluate_model(model, holdout_X, holdout_y, holdout_ts)
        metrics['n_states'] = n_states
        metrics['n_games'] = n_games
        row[name] = metrics

        del train_X, train_y, train_gids, train_ts, model
        gc.collect()

    return row


def print_grid_results(all_results):
    """Print a compact results table across the full grid."""
    metric_keys = ['log_loss', 'auc_roc', 'accuracy', 'egg_inversion_rate',
                   'symmetry_deviation']
    header = ('{:>7} {:>6} {:>5} {:>10}  '
              '{:>8} {:>8} {:>7} {:>7} {:>7}  '
              '{:>8} {:>8} {:>7} {:>7} {:>7}')
    print(header.format(
        'Games', 'Leaves', 'Trees', 'States',
        'QF_Loss', 'QF_AUC', 'QF_Acc', 'QF_Egg', 'QF_Sym',
        'LI_Loss', 'LI_AUC', 'LI_Acc', 'LI_Egg', 'LI_Sym',
    ))
    print('-' * 130)

    for r in all_results:
        qf = r.get('quality_filtered', {})
        li = r.get('logged_in_games', {})

        def fmt(d, k):
            v = d.get(k)
            if v is None:
                return 'N/A'
            return '{:.4f}'.format(v)

        print(header.format(
            r['max_games'], r['num_leaves'], r['num_trees'],
            '{:,}'.format(qf.get('n_states', 0)),
            fmt(qf, 'log_loss'), fmt(qf, 'auc_roc'), fmt(qf, 'accuracy'),
            fmt(qf, 'egg_inversion_rate'), fmt(qf, 'symmetry_deviation'),
            fmt(li, 'log_loss'), fmt(li, 'auc_roc'), fmt(li, 'accuracy'),
            fmt(li, 'egg_inversion_rate'), fmt(li, 'symmetry_deviation'),
        ))


def main():
    parser = argparse.ArgumentParser(
        description='Compare models trained on different data sources')
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--num-trees', type=int, default=50)
    parser.add_argument('--max-games', type=int, default=5000,
                        help='Max games per training source (default 5000). '
                             'Use 0 for all games.')
    parser.add_argument('--grid', action='store_true',
                        help='Run full scaling grid instead of single point')
    parser.add_argument('--variance', type=int, default=0, metavar='N',
                        help='Run N times with different random game samples')
    parser.add_argument('--exclusive', action='store_true',
                        help='Train only on games exclusive to each dataset '
                             '(not in the other). Combines with --variance.')
    parser.add_argument('--equal-states', action='store_true',
                        help='With --exclusive, subsample the larger source '
                             'to match the smaller source\'s state count.')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    # Load tournament holdout (shared across all grid points)
    print("--- Loading tournament holdout ---")
    holdout_X, holdout_y, holdout_gids, holdout_ts = load_and_materialize(
        'late_tournament_games/encoded/all_games.bin')

    # Double holdout via symmetry augmentation (team-swapped copies)
    swapped_X, swapped_y = swap_teams(holdout_X, holdout_y)
    holdout_X = np.concatenate([holdout_X, swapped_X])
    holdout_y = np.concatenate([holdout_y, swapped_y])
    holdout_gids = np.concatenate([holdout_gids, holdout_gids])
    holdout_ts = np.concatenate([holdout_ts, holdout_ts])
    print(f"  Symmetry-augmented holdout: {len(holdout_y):,} states")
    del swapped_X, swapped_y

    if args.exclusive:
        max_games = args.max_games if args.max_games > 0 else 5000
        num_leaves = args.num_leaves
        num_trees = args.num_trees
        n_runs = max(args.variance, 1)

        # Load all entries from both sources
        all_entries = {}
        for name, bin_path in SOURCES.items():
            print(f"  Loading all entries from {bin_path}...")
            all_entries[name] = load_all_entries(bin_path)
            print(f"  {len(all_entries[name]):,} games loaded")

        # Compute exclusive sets
        qf_ids = {gid for gid, _ in all_entries['quality_filtered']}
        li_ids = {gid for gid, _ in all_entries['logged_in_games']}
        qf_only_ids = qf_ids - li_ids
        li_only_ids = li_ids - qf_ids
        print(f"\n  Overlap: {len(qf_ids & li_ids):,} games")
        print(f"  QF-exclusive: {len(qf_only_ids):,} games")
        print(f"  LI-exclusive: {len(li_only_ids):,} games")

        exclusive_entries = {
            'qf_exclusive': [(gid, data) for gid, data in all_entries['quality_filtered']
                             if gid in qf_only_ids],
            'li_exclusive': [(gid, data) for gid, data in all_entries['logged_in_games']
                             if gid in li_only_ids],
        }
        del all_entries
        for name, entries in exclusive_entries.items():
            print(f"  {name}: {len(entries):,} entries retained")

        print(f"\n{'=' * 90}")
        mode = f"EXCLUSIVE VARIANCE: {n_runs} runs" if n_runs > 1 else "EXCLUSIVE"
        if args.equal_states:
            mode += " (equal-states)"
        print(f"{mode}, {max_games} games, {num_leaves}L/{num_trees}T")
        print(f"{'=' * 90}")

        holdout_set = set(holdout_gids)
        metric_keys = ['log_loss', 'auc_roc', 'accuracy',
                        'egg_inversion_rate', 'symmetry_deviation']
        all_runs = []

        for run_idx in range(n_runs):
            seed = run_idx
            print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")
            row = {'seed': seed}

            # Phase 1: materialize and remove leakage for all sources
            materialized = {}
            for name, entries in exclusive_entries.items():
                rng = random_mod.Random(seed)
                sampled = rng.sample(entries, min(max_games, len(entries)))

                start = time.time()
                train_X, train_y, train_gids, train_ts = materialize_from_entries(
                    sampled)
                elapsed = time.time() - start

                # Remove holdout leakage
                mask = np.array([gid not in holdout_set for gid in train_gids])
                train_X = train_X[mask]
                train_y = train_y[mask]
                train_gids = train_gids[mask]
                train_ts = train_ts[mask]

                print(f"  {name}: {len(train_y):,} states in {elapsed:.1f}s")
                materialized[name] = (train_X, train_y, train_gids, train_ts)

            # Phase 2: equalize state counts if requested
            if args.equal_states:
                counts = {n: len(d[1]) for n, d in materialized.items()}
                min_states = min(counts.values())
                for name in materialized:
                    if counts[name] > min_states:
                        train_X, train_y, train_gids, train_ts = materialized[name]
                        np_rng = np.random.RandomState(seed)
                        idx = np_rng.choice(counts[name], min_states, replace=False)
                        idx.sort()
                        materialized[name] = (
                            train_X[idx], train_y[idx],
                            train_gids[idx], train_ts[idx])
                        print(f"  {name}: subsampled {counts[name]:,} -> {min_states:,} states")

            # Phase 3: train and evaluate
            for name in materialized:
                train_X, train_y, train_gids, train_ts = materialized[name]
                n_states = len(train_y)
                print(f"  {name}: training on {n_states:,} states")

                model = train_model(train_X, train_y, num_leaves, num_trees)
                metrics = evaluate_model(model, holdout_X, holdout_y, holdout_ts)
                metrics['n_states'] = n_states
                row[name] = metrics

                del model
                gc.collect()

            del materialized
            gc.collect()

            all_runs.append(row)

        # Print per-run table
        print(f"\n{'=' * 90}")
        header = '{:>4}  {:>8} {:>8} {:>7} {:>7} {:>7}  {:>8} {:>8} {:>7} {:>7} {:>7}'
        print(header.format(
            'Seed', 'QFx_Los', 'QFx_AUC', 'QFx_Acc', 'QFx_Egg', 'QFx_Sym',
            'LIx_Los', 'LIx_AUC', 'LIx_Acc', 'LIx_Egg', 'LIx_Sym'))
        print('-' * 90)
        for r in all_runs:
            qf, li = r['qf_exclusive'], r['li_exclusive']
            print(header.format(
                r['seed'],
                '{:.4f}'.format(qf['log_loss']),
                '{:.4f}'.format(qf['auc_roc']),
                '{:.4f}'.format(qf['accuracy']),
                '{:.4f}'.format(qf['egg_inversion_rate']),
                '{:.4f}'.format(qf['symmetry_deviation']),
                '{:.4f}'.format(li['log_loss']),
                '{:.4f}'.format(li['auc_roc']),
                '{:.4f}'.format(li['accuracy']),
                '{:.4f}'.format(li['egg_inversion_rate']),
                '{:.4f}'.format(li['symmetry_deviation']),
            ))

        if n_runs > 1:
            # Print summary stats
            print(f"\n{'=' * 90}")
            print("SUMMARY (mean +/- std)")
            print('-' * 90)
            summary_header = '{:<25} {:>18} {:>18}'
            print(summary_header.format('Metric', 'qf_exclusive', 'li_exclusive'))
            print('-' * 65)
            for key in metric_keys:
                qf_vals = [r['qf_exclusive'][key] for r in all_runs]
                li_vals = [r['li_exclusive'][key] for r in all_runs]
                print(summary_header.format(
                    key,
                    '{:.4f} +/- {:.4f}'.format(np.mean(qf_vals), np.std(qf_vals)),
                    '{:.4f} +/- {:.4f}'.format(np.mean(li_vals), np.std(li_vals)),
                ))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_runs, f, indent=2)
            print(f"\nSaved to {args.output}")

    elif args.variance > 0:
        max_games = args.max_games if args.max_games > 0 else 5000
        num_leaves = args.num_leaves
        num_trees = args.num_trees
        n_runs = args.variance

        print(f"\n{'=' * 90}")
        print(f"VARIANCE TEST: {n_runs} runs, {max_games} games, "
              f"{num_leaves}L/{num_trees}T")
        print(f"{'=' * 90}")

        # Load all entries once per source
        all_entries = {}
        for name, bin_path in SOURCES.items():
            print(f"  Loading all entries from {bin_path}...")
            all_entries[name] = load_all_entries(bin_path)
            print(f"  {len(all_entries[name]):,} games loaded")

        holdout_set = set(holdout_gids)
        metric_keys = ['log_loss', 'auc_roc', 'accuracy',
                        'egg_inversion_rate', 'symmetry_deviation']
        all_runs = []

        for run_idx in range(n_runs):
            seed = run_idx
            print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={seed}) ---")
            row = {'seed': seed}

            for name in SOURCES:
                entries = all_entries[name]
                rng = random_mod.Random(seed)
                sampled = rng.sample(entries, min(max_games, len(entries)))

                start = time.time()
                train_X, train_y, train_gids, train_ts = materialize_from_entries(
                    sampled)
                elapsed = time.time() - start

                # Remove holdout leakage
                mask = np.array([gid not in holdout_set for gid in train_gids])
                train_X = train_X[mask]
                train_y = train_y[mask]

                n_states = len(train_y)
                print(f"  {name}: {n_states:,} states in {elapsed:.1f}s")

                model = train_model(train_X, train_y, num_leaves, num_trees)
                metrics = evaluate_model(model, holdout_X, holdout_y, holdout_ts)
                metrics['n_states'] = n_states
                row[name] = metrics

                del train_X, train_y, train_gids, train_ts, model
                gc.collect()

            all_runs.append(row)

        # Print per-run table
        print(f"\n{'=' * 90}")
        header = '{:>4}  {:>8} {:>8} {:>7} {:>7} {:>7}  {:>8} {:>8} {:>7} {:>7} {:>7}'
        print(header.format(
            'Seed', 'QF_Loss', 'QF_AUC', 'QF_Acc', 'QF_Egg', 'QF_Sym',
            'LI_Loss', 'LI_AUC', 'LI_Acc', 'LI_Egg', 'LI_Sym'))
        print('-' * 90)
        for r in all_runs:
            qf, li = r['quality_filtered'], r['logged_in_games']
            print(header.format(
                r['seed'],
                '{:.4f}'.format(qf['log_loss']),
                '{:.4f}'.format(qf['auc_roc']),
                '{:.4f}'.format(qf['accuracy']),
                '{:.4f}'.format(qf['egg_inversion_rate']),
                '{:.4f}'.format(qf['symmetry_deviation']),
                '{:.4f}'.format(li['log_loss']),
                '{:.4f}'.format(li['auc_roc']),
                '{:.4f}'.format(li['accuracy']),
                '{:.4f}'.format(li['egg_inversion_rate']),
                '{:.4f}'.format(li['symmetry_deviation']),
            ))

        # Print summary stats
        print(f"\n{'=' * 90}")
        print("SUMMARY (mean +/- std)")
        print('-' * 90)
        summary_header = '{:<25} {:>18} {:>18}'
        print(summary_header.format('Metric', 'quality_filtered', 'logged_in_games'))
        print('-' * 65)
        for key in metric_keys:
            for name in ['quality_filtered', 'logged_in_games']:
                vals = [r[name][key] for r in all_runs]
            qf_vals = [r['quality_filtered'][key] for r in all_runs]
            li_vals = [r['logged_in_games'][key] for r in all_runs]
            print(summary_header.format(
                key,
                '{:.4f} +/- {:.4f}'.format(np.mean(qf_vals), np.std(qf_vals)),
                '{:.4f} +/- {:.4f}'.format(np.mean(li_vals), np.std(li_vals)),
            ))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_runs, f, indent=2)
            print(f"\nSaved to {args.output}")

    elif args.grid:
        print("\n" + "=" * 130)
        print("SCALING GRID: quality_filtered vs logged_in_games")
        print("=" * 130)

        all_results = []
        for max_games, num_leaves, num_trees in GRID:
            row = run_single(holdout_X, holdout_y, holdout_gids, holdout_ts,
                             max_games, num_leaves, num_trees)
            row['max_games'] = max_games
            row['num_leaves'] = num_leaves
            row['num_trees'] = num_trees
            all_results.append(row)

            # Print running table after each point
            print("\n" + "=" * 130)
            print_grid_results(all_results)
            print("=" * 130)

            # Save incrementally
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"  (saved to {args.output})")

    else:
        max_games = args.max_games if args.max_games > 0 else None

        print("\n" + "=" * 60)
        print("Data Quality Experiment: quality_filtered vs logged_in_games")
        print(f"Model: {args.num_leaves} leaves, {args.num_trees} trees")
        print(f"Max games per source: {max_games or 'all'}")
        print("=" * 60)

        row = run_single(holdout_X, holdout_y, holdout_gids, holdout_ts,
                         max_games, args.num_leaves, args.num_trees)

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON (evaluated on tournament holdout)")
        print("=" * 60)
        print(f"Holdout: {len(holdout_y):,} states from "
              f"{len(np.unique(holdout_gids)):,} tournament games\n")

        metric_keys = [
            'log_loss', 'brier_score', 'auc_roc', 'accuracy',
            'egg_inversion_rate', 'symmetry_deviation',
        ]
        sources = list(row.keys())
        header = f"{'Metric':<25}"
        for src in sources:
            header += f"  {src:<18}"
        print(header)
        print("-" * len(header))
        for key in metric_keys:
            line = f"{key:<25}"
            for src in sources:
                val = row[src].get(key)
                if val is None:
                    line += f"  {'N/A':<18}"
                elif isinstance(val, float):
                    line += f"  {val:<18.6f}"
                else:
                    line += f"  {val!s:<18}"
            print(line)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(row, f, indent=2)
            print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
