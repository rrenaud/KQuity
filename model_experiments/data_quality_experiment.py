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

import random

from event_codec import (
    fast_materialize_from_codec, read_packed_games, materialize_entries,
)
from eval_metrics import (
    compute_standard_metrics, compute_kq_metrics,
)
from symmetry import swap_teams

SOURCES = {
    'quality_filtered': 'quality_filtered/encoded/all_games.bin',
    'logged_in_games': 'logged_in_games/encoded/all_games.bin',
}

METRIC_KEYS = ['log_loss', 'auc_roc', 'accuracy',
               'egg_inversion_rate', 'symmetry_deviation']

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


def load_and_materialize(bin_path, drop_state_probability=0.0, max_games=None,
                         exclude_game_ids=None):
    """Load binary-encoded games and materialize features."""
    print(f"  Materializing {bin_path} (max_games={max_games})...")
    start = time.time()
    states, labels, game_ids, timestamps = fast_materialize_from_codec(
        bin_path, drop_state_probability=drop_state_probability,
        max_games=max_games, exclude_game_ids=exclude_game_ids)
    elapsed = time.time() - start
    n_games = len(np.unique(game_ids))
    print(f"  {len(labels):,} states from {n_games:,} games in {elapsed:.1f}s")
    return states, labels, game_ids, timestamps


def load_all_entries(bin_path):
    """Read all (game_id, encoded_bytes) from a packed binary file into memory."""
    return list(read_packed_games(bin_path))


def train_model(train_X, train_y, num_leaves, num_trees):
    """Train a LightGBM binary classifier."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': -1,
    }
    train_data = lgb.Dataset(train_X, train_y, free_raw_data=True)
    start = time.time()
    model = lgb.train(param, train_data, num_boost_round=num_trees)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s ({num_leaves} leaves, {num_trees} trees)")
    return model


def evaluate_model(model, test_X, test_y):
    """Evaluate model and return metrics dict."""
    predictions = model.predict(test_X)
    labels_float = test_y.astype(np.float64)

    std = compute_standard_metrics(predictions, labels_float)
    kq = compute_kq_metrics(model, test_X, labels_float)

    return {**std, **kq}


def run_single(holdout_X, holdout_y, holdout_gids,
               max_games, num_leaves, num_trees, exclusive_gids=()):
    """Run one grid point: train both sources, return metrics dicts."""
    row = {}
    holdout_set = set(holdout_gids)
    exclude = holdout_set | set(exclusive_gids) if exclusive_gids else holdout_set
    for name, bin_path in SOURCES.items():
        print(f"\n--- {name} | {max_games} games, {num_leaves}L/{num_trees}T ---")

        train_X, train_y, train_gids, train_ts = load_and_materialize(
            bin_path, max_games=max_games, exclude_game_ids=exclude)

        n_states = len(train_y)
        n_games = len(np.unique(train_gids))
        print(f"  Training: {n_states:,} states from {n_games:,} games")

        model = train_model(train_X, train_y, num_leaves, num_trees)

        print("  Evaluating on tournament holdout...")
        metrics = evaluate_model(model, holdout_X, holdout_y)
        metrics['n_states'] = n_states
        metrics['n_games'] = n_games
        row[name] = metrics

        del train_X, train_y, train_gids, train_ts, model
        gc.collect()

    return row


def print_grid_results(all_results):
    """Print a compact results table across the full grid."""
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


def truncate_to_target(train_X, train_y, train_gids, train_ts,
                       target_states, seed):
    """Keep random whole games until the next would exceed target_states."""
    np_rng = np.random.RandomState(seed)
    unique_gids, counts = np.unique(train_gids, return_counts=True)
    order = np_rng.permutation(len(unique_gids))
    keep_gids = []
    kept = 0
    for idx in order:
        game_count = counts[idx]
        if kept + game_count > target_states:
            break
        keep_gids.append(unique_gids[idx])
        kept += game_count
    mask = np.isin(train_gids, keep_gids)
    return train_X[mask], train_y[mask], train_gids[mask], train_ts[mask]


def run_ranking(holdout_X, holdout_y, holdout_gids,
                k_sweep, num_leaves, num_trees, n_random, pool_factor,
                output_path=None, equal_states=False, exclusive=False,
                drop_prob=0.0):
    """Run ranked jitter experiment: sample K from top K*P games."""
    holdout_set = set(holdout_gids)

    # Load all entries from both sources
    all_entries = {}
    for name, bin_path in SOURCES.items():
        print(f"  Loading all entries from {bin_path}...")
        all_entries[name] = load_all_entries(bin_path)
        print(f"  {len(all_entries[name]):,} games loaded")

    if exclusive:
        qf_ids = {gid for gid, _ in all_entries['quality_filtered']}
        li_ids = {gid for gid, _ in all_entries['logged_in_games']}
        overlap = qf_ids & li_ids
        print(f"  Overlap: {len(overlap):,} games")
        for name in all_entries:
            before = len(all_entries[name])
            all_entries[name] = [(gid, data) for gid, data in all_entries[name]
                                 if gid not in overlap]
            print(f"  {name}: {before:,} -> {len(all_entries[name]):,} "
                  f"(removed {before - len(all_entries[name]):,} overlapping)")

    # Determine processing order by states-per-game (smaller SPG first)
    if equal_states:
        spg = {}
        cal_rng = random.Random(42)
        for name in SOURCES:
            entries = all_entries[name]
            cal_sample = cal_rng.sample(entries, min(200, len(entries)))
            _, cal_y, _, _ = materialize_entries(cal_sample)
            spg[name] = len(cal_y) / len(cal_sample)
            print(f"  {name}: ~{spg[name]:.1f} states/game "
                  f"(calibrated on {len(cal_sample)} games)")
            del cal_y
        ordered_names = sorted(SOURCES.keys(), key=lambda n: spg[n])
        print(f"  Processing order: {ordered_names[0]} (smaller) first")

    all_runs = []

    for k in k_sweep:
        pools = {}
        for name in SOURCES:
            entries = all_entries[name]
            pool_size = min(int(k * pool_factor), len(entries))
            pools[name] = (entries[:pool_size], pool_size)

        print(f"\n{'=' * 70}")
        print(f"K={k}, pool_factor={pool_factor}, "
              f"{n_random} seeds, {num_leaves}L/{num_trees}T")
        print(f"{'=' * 70}")

        for seed_idx in range(n_random):
            seed = seed_idx

            if equal_states:
                # Sequential path: process smaller-SPG dataset first to get
                # target state count, then truncate larger-SPG to match.
                # Peak memory: one materialized dataset at a time.
                target_states = None
                for name in ordered_names:
                    pool, pool_size = pools[name]
                    rng = random.Random(seed)
                    sampled = rng.sample(pool, min(k, len(pool)))
                    sampled = [(gid, data) for gid, data in sampled
                               if gid not in holdout_set]

                    start = time.time()
                    train_X, train_y, train_gids, train_ts = (
                        materialize_entries(sampled,
                                           drop_state_probability=drop_prob))
                    elapsed = time.time() - start

                    n_states = len(train_y)
                    n_games = len(np.unique(train_gids))
                    print(f"  {name} seed={seed}: {n_states:,} states from "
                          f"{n_games:,} games in {elapsed:.1f}s")

                    if target_states is not None:
                        # Truncate larger-SPG dataset to match
                        orig_states = n_states
                        train_X, train_y, train_gids, train_ts = (
                            truncate_to_target(
                                train_X, train_y, train_gids, train_ts,
                                target_states, seed))
                        n_states = len(train_y)
                        n_games = len(np.unique(train_gids))
                        print(f"    {name}: truncated {orig_states:,} -> "
                              f"{n_states:,} states ({n_games:,} games)")
                    else:
                        target_states = n_states

                    model = train_model(train_X, train_y, num_leaves, num_trees)
                    metrics = evaluate_model(model, holdout_X, holdout_y)
                    metrics['n_states'] = n_states
                    metrics['n_games'] = n_games

                    all_runs.append({
                        'dataset': name,
                        'k': k,
                        'pool_size': pool_size,
                        'strategy': f'jitter_seed{seed}',
                        'metrics': metrics,
                    })

                    del train_X, train_y, train_gids, train_ts, model
                    gc.collect()
            else:
                # Original path: materialize both, then train both
                materialized = {}

                for name in SOURCES:
                    pool, pool_size = pools[name]
                    rng = random.Random(seed)
                    sampled = rng.sample(pool, min(k, len(pool)))
                    sampled = [(gid, data) for gid, data in sampled
                               if gid not in holdout_set]

                    start = time.time()
                    train_X, train_y, train_gids, train_ts = (
                        materialize_entries(sampled,
                                           drop_state_probability=drop_prob))
                    elapsed = time.time() - start

                    materialized[name] = (
                        train_X, train_y, train_gids, train_ts, pool_size)
                    n_states = len(train_y)
                    n_games = len(np.unique(train_gids))
                    print(f"  {name} seed={seed}: {n_states:,} states from "
                          f"{n_games:,} games in {elapsed:.1f}s")

                # Train and evaluate both datasets
                for name in SOURCES:
                    train_X, train_y, train_gids, train_ts, pool_size = (
                        materialized[name])
                    n_states = len(train_y)
                    n_games = len(np.unique(train_gids))

                    model = train_model(train_X, train_y, num_leaves, num_trees)
                    metrics = evaluate_model(model, holdout_X, holdout_y)
                    metrics['n_states'] = n_states
                    metrics['n_games'] = n_games

                    all_runs.append({
                        'dataset': name,
                        'k': k,
                        'pool_size': pool_size,
                        'strategy': f'jitter_seed{seed}',
                        'metrics': metrics,
                    })

                    del model
                    gc.collect()

                del materialized
                gc.collect()

        # Save incrementally after each K
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'mode': 'ranking',
                    'pool_factor': pool_factor,
                    'equal_states': equal_states,
                    'exclusive': exclusive,
                    'drop_prob': drop_prob,
                    'num_leaves': num_leaves,
                    'num_trees': num_trees,
                    'k_sweep': k_sweep,
                    'n_random': n_random,
                    'runs': all_runs,
                }, f, indent=2)
                f.write('\n')
            print(f"  (saved to {output_path})")

    return all_runs


def print_ranking_results(all_runs, n_random):
    """Print summary table grouped by (dataset, K) with mean +/- std."""
    print(f"\n{'=' * 110}")
    print("RANKING RESULTS (mean +/- std across seeds)")
    print('=' * 110)

    header = '{:<20} {:>5} {:>6} {:>5}  {:>10}  {:>15} {:>15} {:>15} {:>15}'
    print(header.format(
        'Dataset', 'K', 'Pool', 'Seeds', 'States',
        'Loss', 'AUC', 'Egg', 'Sym'))
    print('-' * 110)

    # Group runs by (dataset, k)
    groups = {}
    for run in all_runs:
        key = (run['dataset'], run['k'])
        groups.setdefault(key, []).append(run)

    for (dataset, k), runs in groups.items():
        pool_size = runs[0]['pool_size']
        n_seeds = len(runs)

        states_vals = [r['metrics']['n_states'] for r in runs]
        loss_vals = [r['metrics']['log_loss'] for r in runs]
        auc_vals = [r['metrics']['auc_roc'] for r in runs]
        egg_vals = [r['metrics']['egg_inversion_rate'] for r in runs]
        sym_vals = [r['metrics']['symmetry_deviation'] for r in runs]

        def fmt_mean_std(vals):
            return '{:.4f}+/-{:.4f}'.format(np.mean(vals), np.std(vals))

        print(header.format(
            dataset, k, pool_size, n_seeds,
            '{:.0f}+/-{:.0f}'.format(np.mean(states_vals), np.std(states_vals)),
            fmt_mean_std(loss_vals),
            fmt_mean_std(auc_vals),
            fmt_mean_std(egg_vals),
            fmt_mean_std(sym_vals),
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
    parser.add_argument('--ranking', action='store_true',
                        help='Ranked jitter mode: sample K from top K*P games')
    parser.add_argument('--k-sweep', type=int, nargs='+',
                        default=[2000, 5000, 10000, 20000],
                        help='K values to test in --ranking mode')
    parser.add_argument('--n-random', type=int, default=5,
                        help='Number of random seeds in --ranking mode')
    parser.add_argument('--pool-factor', type=float, default=1.5,
                        help='Pool size multiplier P in --ranking mode (pool=K*P)')
    parser.add_argument('--exclusive', action='store_true',
                        help='Remove overlapping games between quality_filtered '
                             'and logged_in_games before training.')
    parser.add_argument('--equal-states', action='store_true',
                        help='With --ranking, equalize state counts between '
                             'datasets by truncating the larger one.')
    parser.add_argument('--drop-prob', type=float, default=0.0,
                        help='Probability of dropping each state during '
                             'materialization (0.0 = keep all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    if args.equal_states and not args.ranking:
        parser.error('--equal-states requires --ranking')

    if args.grid and args.ranking:
        parser.error('--grid cannot be combined with --ranking')

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

    overlap_gids = set()
    if args.exclusive and not args.ranking:
        # For ranking mode, run_ranking computes overlap from its in-memory entries.
        qf_ids = {gid for gid, _ in read_packed_games(SOURCES['quality_filtered'])}
        li_ids = {gid for gid, _ in read_packed_games(SOURCES['logged_in_games'])}
        overlap_gids = qf_ids & li_ids
        print(f"  Exclusive: {len(overlap_gids):,} overlapping games to remove "
              f"(QF: {len(qf_ids):,}, LI: {len(li_ids):,})")

    if args.ranking:
        num_leaves = args.num_leaves
        num_trees = args.num_trees

        mode_label = "RANKED JITTER"
        if args.exclusive:
            mode_label += " (exclusive)"
        if args.equal_states:
            mode_label += " (equal-states)"
        print(f"\n{'=' * 70}")
        drop_str = f", drop_prob={args.drop_prob}" if args.drop_prob > 0 else ""
        print(f"{mode_label}: K={args.k_sweep}, pool_factor={args.pool_factor}, "
              f"{args.n_random} seeds, {num_leaves}L/{num_trees}T{drop_str}")
        print(f"{'=' * 70}")

        all_runs = run_ranking(
            holdout_X, holdout_y, holdout_gids,
            k_sweep=args.k_sweep,
            num_leaves=num_leaves,
            num_trees=num_trees,
            n_random=args.n_random,
            pool_factor=args.pool_factor,
            output_path=args.output,
            equal_states=args.equal_states,
            exclusive=args.exclusive,
            drop_prob=args.drop_prob,
        )

        print_ranking_results(all_runs, args.n_random)

    elif args.grid:
        print("\n" + "=" * 130)
        print("SCALING GRID: quality_filtered vs logged_in_games")
        print("=" * 130)

        all_results = []
        for max_games, num_leaves, num_trees in GRID:
            row = run_single(holdout_X, holdout_y, holdout_gids,
                             max_games, num_leaves, num_trees,
                             exclusive_gids=overlap_gids)
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
                    f.write('\n')
                print(f"  (saved to {args.output})")

    else:
        max_games = args.max_games if args.max_games > 0 else None

        print("\n" + "=" * 60)
        print("Data Quality Experiment: quality_filtered vs logged_in_games")
        print(f"Model: {args.num_leaves} leaves, {args.num_trees} trees")
        print(f"Max games per source: {max_games or 'all'}")
        print("=" * 60)

        row = run_single(holdout_X, holdout_y, holdout_gids,
                         max_games, args.num_leaves, args.num_trees,
                         exclusive_gids=overlap_gids)

        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON (evaluated on tournament holdout)")
        print("=" * 60)
        print(f"Holdout: {len(holdout_y):,} states from "
              f"{len(np.unique(holdout_gids)):,} tournament games\n")

        single_metric_keys = ['brier_score'] + METRIC_KEYS
        sources = list(row.keys())
        header = f"{'Metric':<25}"
        for src in sources:
            header += f"  {src:<18}"
        print(header)
        print("-" * len(header))
        for key in single_metric_keys:
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
                f.write('\n')
            print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
