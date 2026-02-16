#!/usr/bin/env python3
"""Combined data scaling experiment: union and mix-ratio modes.

Tests what happens when QF and LI data sources are combined into a single
training pool, and what mix ratio is optimal at a given data budget.

Modes:
  --mode union      Deduplicated union of all games, scale up sample size.
  --mode mix-ratio  Fixed total budget, vary QF fraction via --ratios.
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# Add repo root to path so imports work from model_experiments/combined_li_qf/ subdir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import lightgbm as lgb

from event_codec import read_packed_games, materialize_entries
from eval_metrics import (
    compute_standard_metrics, compute_calibration_metrics,
    compute_temporal_metrics, compute_kq_metrics,
)
from symmetry import swap_teams

SOURCES = {
    'quality_filtered': 'quality_filtered/encoded/all_games.bin',
    'logged_in_games': 'logged_in_games/encoded/all_games.bin',
    'unfiltered': 'unfiltered_partitioned/encoded/all_games.bin',
}
HOLDOUT_PATH = 'late_tournament_games/encoded/all_games.bin'

SCALING_SCHEDULE = [
    # (max_games, num_leaves, num_trees)
    (2000,    15,   25),
    (5000,    35,   35),
    (10000,   50,   50),
    (20000,   50,   50),
    (50000,   75,   75),
    (100000, 100,  100),
    (200000, 150,  150),
]


def load_all_entries(bin_path):
    """Read all (game_id, encoded_bytes) from a packed binary file."""
    return list(read_packed_games(bin_path))


def build_deduplicated_pool(qf_entries, li_entries):
    """Build deduplicated union by interleaving QF and LI entries.

    Both input lists are pre-sorted by quality (QF by quality score desc,
    LI by login count desc). Interleaving ensures top-N sampling gets a
    balanced mix of both sources at every scale point.

    Walks both lists with two pointers, alternating QF/LI, skipping
    duplicates. When one source exhausts, drains the other.

    Returns:
        (pool, qf_ids, li_ids, stats)
    """
    seen = set()
    pool = []
    qf_ids = set()
    li_ids = set()

    qi, li = 0, 0
    while qi < len(qf_entries) or li < len(li_entries):
        # QF turn
        if qi < len(qf_entries):
            gid, data = qf_entries[qi]
            qf_ids.add(gid)
            qi += 1
            if gid not in seen:
                seen.add(gid)
                pool.append((gid, data))
        # LI turn
        if li < len(li_entries):
            gid, data = li_entries[li]
            li_ids.add(gid)
            li += 1
            if gid not in seen:
                seen.add(gid)
                pool.append((gid, data))

    overlap = qf_ids & li_ids
    stats = {
        'qf_total': len(qf_entries),
        'li_total': len(li_entries),
        'overlap': len(overlap),
        'union_total': len(pool),
    }
    return pool, qf_ids, li_ids, stats


def load_holdout():
    """Load and symmetry-augment tournament holdout data."""
    from event_codec import fast_materialize_from_codec

    print("--- Loading tournament holdout ---")
    start = time.time()
    holdout_X, holdout_y, holdout_gids, holdout_ts = fast_materialize_from_codec(
        HOLDOUT_PATH)
    elapsed = time.time() - start
    n_games = len(np.unique(holdout_gids))
    print(f"  {len(holdout_y):,} states from {n_games:,} games in {elapsed:.1f}s")

    # Symmetry augmentation
    swapped_X, swapped_y = swap_teams(holdout_X, holdout_y)
    holdout_X = np.concatenate([holdout_X, swapped_X])
    holdout_y = np.concatenate([holdout_y, swapped_y])
    holdout_gids = np.concatenate([holdout_gids, holdout_gids])
    holdout_ts = np.concatenate([holdout_ts, holdout_ts])
    print(f"  Symmetry-augmented holdout: {len(holdout_y):,} states")
    del swapped_X, swapped_y

    return holdout_X, holdout_y, holdout_gids, holdout_ts


def train_model(train_X, train_y, num_leaves, num_trees):
    """Train a LightGBM binary classifier."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': -1,
        'num_threads': min(os.cpu_count() or 1, 16),
    }
    train_data = lgb.Dataset(train_X, train_y, free_raw_data=True)
    start = time.time()
    model = lgb.train(param, train_data, num_boost_round=num_trees)
    elapsed = time.time() - start
    print(f"  Trained in {elapsed:.1f}s ({num_leaves} leaves, {num_trees} trees)")
    return model


def evaluate_model(model, holdout_X, holdout_y, holdout_ts):
    """Evaluate model with all four metric suites."""
    predictions = model.predict(holdout_X)
    labels_float = holdout_y.astype(np.float64)

    return {
        'standard': compute_standard_metrics(predictions, labels_float),
        'calibration': compute_calibration_metrics(predictions, labels_float),
        'temporal': compute_temporal_metrics(predictions, labels_float, holdout_ts),
        'kq': compute_kq_metrics(model, holdout_X, labels_float),
    }


def sample_entries(pool, n_games, seed, exclude_gids):
    """Take top n_games entries from pool, excluding holdout game IDs.

    Pool is pre-sorted by interleaved quality ranking, so top-N gives
    the best available data at each scale point. The seed parameter is
    unused here (variance comes from drop_state_probability RNG) but
    kept for API compatibility.
    """
    eligible = [(gid, data) for gid, data in pool if gid not in exclude_gids]
    return eligible[:n_games]


def sample_mix_entries(qf_entries, li_entries, n_games, qf_ratio, seed,
                       exclude_gids):
    """Sample a mix of QF and LI entries at a given ratio.

    Draws qf_ratio * n_games from QF and (1-qf_ratio) * n_games from LI,
    then deduplicates (QF preferred for overlaps).
    """
    rng = random.Random(seed)
    n_qf = int(round(qf_ratio * n_games))
    n_li = n_games - n_qf

    qf_eligible = [(gid, data) for gid, data in qf_entries
                   if gid not in exclude_gids]
    li_eligible = [(gid, data) for gid, data in li_entries
                   if gid not in exclude_gids]

    qf_sample = rng.sample(qf_eligible, min(n_qf, len(qf_eligible)))
    li_sample = rng.sample(li_eligible, min(n_li, len(li_eligible)))

    # Deduplicate: LI first, QF overwrites
    pool_dict = {}
    for gid, data in li_sample:
        pool_dict[gid] = (gid, data)
    for gid, data in qf_sample:
        pool_dict[gid] = (gid, data)

    return list(pool_dict.values())


def run_single_point(sampled_entries, num_leaves, num_trees,
                     holdout_X, holdout_y, holdout_ts, drop_prob=0.0,
                     seed=42, sym_aug=False):
    """Materialize, train, evaluate a single (scale_point, seed) run."""
    run_start = time.time()
    train_X, train_y, train_gids, train_ts = materialize_entries(
        sampled_entries, drop_state_probability=drop_prob, seed=seed)
    mat_elapsed = time.time() - run_start

    n_states = len(train_y)
    n_games = len(np.unique(train_gids))
    print(f"  Materialized {n_states:,} states from {n_games:,} games "
          f"in {mat_elapsed:.1f}s")

    if sym_aug:
        sw_X, sw_y = swap_teams(train_X, train_y)
        train_X = np.concatenate([train_X, sw_X])
        train_y = np.concatenate([train_y, sw_y])
        del sw_X, sw_y
        print(f"  Sym-augmented: {len(train_y):,} training states")

    model = train_model(train_X, train_y, num_leaves, num_trees)

    print("  Evaluating on tournament holdout...")
    metrics = evaluate_model(model, holdout_X, holdout_y, holdout_ts)

    elapsed = time.time() - run_start
    print(f"  Run completed in {elapsed:.1f}s")

    del train_X, train_y, train_gids, train_ts, model
    gc.collect()

    return n_games, n_states, metrics, elapsed


def _worker_run(args):
    """Worker function for parallel execution.

    Re-loads data from disk, samples, materializes, trains, evaluates.
    Returns the result dict for this (scale_point, seed) run.
    """
    (mode, max_games, num_leaves, num_trees, seed,
     qf_ratio, holdout_path, drop_prob, source, sym_aug) = args

    # Re-load data in worker
    qf_entries = load_all_entries(SOURCES['quality_filtered'])
    li_entries = load_all_entries(SOURCES['logged_in_games'])

    holdout_X, holdout_y, holdout_gids, holdout_ts = load_holdout_for_worker(
        holdout_path)
    exclude_gids = set(holdout_gids)

    if mode == 'union':
        if source == 'qf':
            pool = qf_entries
        elif source == 'li':
            pool = li_entries
        elif source == 'unfiltered':
            pool = load_all_entries(SOURCES['unfiltered'])
            random.Random(0).shuffle(pool)
        elif source == 'unfiltered-oldest':
            pool = load_all_entries(SOURCES['unfiltered'])
        elif source == 'unfiltered-newest':
            pool = load_all_entries(SOURCES['unfiltered'])
            pool.reverse()
        else:
            pool, _, _, _ = build_deduplicated_pool(qf_entries, li_entries)
        sampled = sample_entries(pool, max_games, seed, exclude_gids)
    else:
        sampled = sample_mix_entries(
            qf_entries, li_entries, max_games, qf_ratio, seed, exclude_gids)

    n_games, n_states, metrics, elapsed = run_single_point(
        sampled, num_leaves, num_trees,
        holdout_X, holdout_y, holdout_ts, drop_prob=drop_prob,
        seed=seed, sym_aug=sym_aug)

    result = {
        'max_games': max_games,
        'num_leaves': num_leaves,
        'num_trees': num_trees,
        'seed': seed,
        'n_games_actual': n_games,
        'n_states': n_states,
        'elapsed_s': round(elapsed, 1),
        'metrics': metrics,
    }
    if qf_ratio is not None:
        result['qf_ratio'] = qf_ratio

    return result


def load_holdout_for_worker(holdout_path):
    """Load holdout for a worker process (no printing)."""
    from event_codec import fast_materialize_from_codec

    holdout_X, holdout_y, holdout_gids, holdout_ts = fast_materialize_from_codec(
        holdout_path)
    swapped_X, swapped_y = swap_teams(holdout_X, holdout_y)
    holdout_X = np.concatenate([holdout_X, swapped_X])
    holdout_y = np.concatenate([holdout_y, swapped_y])
    holdout_gids = np.concatenate([holdout_gids, holdout_gids])
    holdout_ts = np.concatenate([holdout_ts, holdout_ts])
    del swapped_X, swapped_y
    return holdout_X, holdout_y, holdout_gids, holdout_ts


def run_union(pool, holdout_X, holdout_y, holdout_gids, holdout_ts,
              schedule, n_seeds, output_path, workers, drop_prob=0.0,
              source='union', sym_aug=False):
    """Run union-mode scaling experiment."""
    exclude_gids = set(holdout_gids)
    all_runs = []

    if workers > 1:
        tasks = []
        for max_games, num_leaves, num_trees in schedule:
            for seed in range(n_seeds):
                tasks.append((
                    'union', max_games, num_leaves, num_trees, seed,
                    None, HOLDOUT_PATH, drop_prob, source, sym_aug,
                ))

        print(f"\nRunning {len(tasks)} tasks with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_worker_run, tasks):
                all_runs.append(result)
                mg = result['max_games']
                s = result['seed']
                ll = result['metrics']['standard']['log_loss']
                el = result['elapsed_s']
                print(f"  Completed: {mg} games, seed={s}, "
                      f"log_loss={ll:.4f}, {el:.1f}s")

                if output_path:
                    _save_results(output_path, 'union', schedule, None,
                                  all_runs, drop_prob=drop_prob)
    else:
        for max_games, num_leaves, num_trees in schedule:
            for seed in range(n_seeds):
                print(f"\n{'=' * 60}")
                print(f"Union | {max_games} games, {num_leaves}L/{num_trees}T, "
                      f"seed={seed}")
                print(f"{'=' * 60}")

                sampled = sample_entries(pool, max_games, seed, exclude_gids)
                n_games, n_states, metrics, elapsed = run_single_point(
                    sampled, num_leaves, num_trees,
                    holdout_X, holdout_y, holdout_ts,
                    drop_prob=drop_prob, seed=seed, sym_aug=sym_aug)

                run_result = {
                    'max_games': max_games,
                    'num_leaves': num_leaves,
                    'num_trees': num_trees,
                    'seed': seed,
                    'n_games_actual': n_games,
                    'n_states': n_states,
                    'elapsed_s': round(elapsed, 1),
                    'metrics': metrics,
                }
                all_runs.append(run_result)
                _print_run_summary(run_result)

                if output_path:
                    _save_results(output_path, 'union', schedule, None,
                                  all_runs, drop_prob=drop_prob)

    return all_runs


def run_mix_ratio(qf_entries, li_entries, holdout_X, holdout_y, holdout_gids,
                  holdout_ts, max_games, num_leaves, num_trees, ratios,
                  n_seeds, output_path, workers, drop_prob=0.0, sym_aug=False):
    """Run mix-ratio experiment at a fixed game budget."""
    exclude_gids = set(holdout_gids)
    schedule = [(max_games, num_leaves, num_trees)]
    all_runs = []

    if workers > 1:
        tasks = []
        for qf_ratio in ratios:
            for seed in range(n_seeds):
                tasks.append((
                    'mix-ratio', max_games, num_leaves, num_trees, seed,
                    qf_ratio, HOLDOUT_PATH, drop_prob, 'union', sym_aug,
                ))

        print(f"\nRunning {len(tasks)} tasks with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(_worker_run, tasks):
                all_runs.append(result)
                r = result['qf_ratio']
                s = result['seed']
                ll = result['metrics']['standard']['log_loss']
                el = result['elapsed_s']
                print(f"  Completed: ratio={r}, seed={s}, "
                      f"log_loss={ll:.4f}, {el:.1f}s")

                if output_path:
                    _save_results(output_path, 'mix-ratio', schedule,
                                  ratios, all_runs, drop_prob=drop_prob)
    else:
        for qf_ratio in ratios:
            for seed in range(n_seeds):
                print(f"\n{'=' * 60}")
                print(f"Mix-ratio | QF={qf_ratio:.2f}, {max_games} games, "
                      f"{num_leaves}L/{num_trees}T, seed={seed}")
                print(f"{'=' * 60}")

                sampled = sample_mix_entries(
                    qf_entries, li_entries, max_games, qf_ratio, seed,
                    exclude_gids)
                n_games, n_states, metrics, elapsed = run_single_point(
                    sampled, num_leaves, num_trees,
                    holdout_X, holdout_y, holdout_ts,
                    drop_prob=drop_prob, seed=seed, sym_aug=sym_aug)

                run_result = {
                    'max_games': max_games,
                    'num_leaves': num_leaves,
                    'num_trees': num_trees,
                    'seed': seed,
                    'qf_ratio': qf_ratio,
                    'n_games_actual': n_games,
                    'n_states': n_states,
                    'elapsed_s': round(elapsed, 1),
                    'metrics': metrics,
                }
                all_runs.append(run_result)
                _print_run_summary(run_result)

                if output_path:
                    _save_results(output_path, 'mix-ratio', schedule,
                                  ratios, all_runs, drop_prob=drop_prob)

    return all_runs


def _save_results(output_path, mode, schedule, ratios, runs,
                  data_stats=None, drop_prob=None, sym_aug=None):
    """Save results JSON incrementally."""
    result = {
        'mode': mode,
        'schedule': [list(s) for s in schedule],
        'runs': runs,
    }
    if drop_prob is not None:
        result['drop_prob'] = drop_prob
    if sym_aug is not None:
        result['sym_aug'] = sym_aug
    if ratios is not None:
        result['ratios'] = ratios
    if data_stats is not None:
        result['data_stats'] = data_stats
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f"  (saved to {output_path})")


def _print_run_summary(run):
    """Print a compact summary of a single run."""
    std = run['metrics']['standard']
    kq = run['metrics']['kq']
    cal = run['metrics']['calibration']
    print(f"  Results: log_loss={std['log_loss']:.4f}, "
          f"auc_roc={std['auc_roc']:.4f}, "
          f"egg_inv={kq['egg_inversion_rate']:.4f}, "
          f"sym_dev={kq['symmetry_deviation']:.4f}, "
          f"ece={cal['ece']:.4f}")


def print_summary_table(runs, mode):
    """Print summary table grouped by scale point with mean +/- std."""
    print(f"\n{'=' * 100}")
    print(f"SUMMARY ({mode} mode, mean +/- std across seeds)")
    print(f"{'=' * 100}")

    if mode == 'union':
        header = ('{:>7} {:>6} {:>5} {:>5}  {:>12}  '
                  '{:>15} {:>15} {:>15} {:>15} {:>10}')
        print(header.format(
            'Games', 'Leaves', 'Trees', 'Seeds', 'States',
            'Loss', 'AUC', 'Egg', 'Sym', 'ECE'))
        print('-' * 100)

        groups = {}
        for run in runs:
            key = (run['max_games'], run['num_leaves'], run['num_trees'])
            groups.setdefault(key, []).append(run)

        for key in sorted(groups.keys()):
            group = groups[key]
            max_games, num_leaves, num_trees = key
            n_seeds = len(group)

            states = [r['n_states'] for r in group]
            loss = [r['metrics']['standard']['log_loss'] for r in group]
            auc = [r['metrics']['standard']['auc_roc'] for r in group]
            egg = [r['metrics']['kq']['egg_inversion_rate'] for r in group]
            sym = [r['metrics']['kq']['symmetry_deviation'] for r in group]
            ece = [r['metrics']['calibration']['ece'] for r in group]

            def fmt(vals):
                return f'{np.mean(vals):.4f}+/-{np.std(vals):.4f}'

            print(header.format(
                max_games, num_leaves, num_trees, n_seeds,
                f'{np.mean(states):.0f}+/-{np.std(states):.0f}',
                fmt(loss), fmt(auc), fmt(egg), fmt(sym), fmt(ece)))

    elif mode == 'mix-ratio':
        header = ('{:>7} {:>7} {:>5}  {:>12}  '
                  '{:>15} {:>15} {:>15} {:>15} {:>10}')
        print(header.format(
            'Ratio', 'Games', 'Seeds', 'States',
            'Loss', 'AUC', 'Egg', 'Sym', 'ECE'))
        print('-' * 100)

        groups = {}
        for run in runs:
            key = run['qf_ratio']
            groups.setdefault(key, []).append(run)

        for ratio in sorted(groups.keys()):
            group = groups[ratio]
            max_games = group[0]['max_games']
            n_seeds = len(group)

            states = [r['n_states'] for r in group]
            loss = [r['metrics']['standard']['log_loss'] for r in group]
            auc = [r['metrics']['standard']['auc_roc'] for r in group]
            egg = [r['metrics']['kq']['egg_inversion_rate'] for r in group]
            sym = [r['metrics']['kq']['symmetry_deviation'] for r in group]
            ece = [r['metrics']['calibration']['ece'] for r in group]

            def fmt(vals):
                return f'{np.mean(vals):.4f}+/-{np.std(vals):.4f}'

            print(header.format(
                f'{ratio:.2f}', max_games, n_seeds,
                f'{np.mean(states):.0f}+/-{np.std(states):.0f}',
                fmt(loss), fmt(auc), fmt(egg), fmt(sym), fmt(ece)))


def main():
    parser = argparse.ArgumentParser(
        description='Combined data scaling experiment: union and mix-ratio')
    parser.add_argument('--mode', choices=['union', 'mix-ratio'],
                        default='union',
                        help='Experiment mode (default: union)')
    parser.add_argument('--source', choices=['union', 'qf', 'li', 'unfiltered',
                                             'unfiltered-oldest', 'unfiltered-newest'],
                        default='union',
                        help='Data source for union mode: union (interleaved '
                             'QF+LI), qf (QF only), li (LI only), '
                             'unfiltered (shuffled), unfiltered-oldest '
                             '(chronological), unfiltered-newest (reverse chrono)')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Override max games (single scale point). '
                             'For union mode, overrides the full schedule.')
    parser.add_argument('--num-leaves', type=int, default=None,
                        help='Override num_leaves (used with --max-games)')
    parser.add_argument('--num-trees', type=int, default=None,
                        help='Override num_trees (used with --max-games)')
    parser.add_argument('--ratios', type=float, nargs='+',
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help='QF ratios for mix-ratio mode')
    parser.add_argument('--n-seeds', type=int, default=5,
                        help='Number of random seeds for variance estimation')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--drop-prob', type=float, default=0.9,
                        help='Probability of dropping each state during '
                             'materialization (default: 0.9)')
    parser.add_argument('--sym-aug', action='store_true',
                        help='Symmetry-augment training data (2x states)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    args = parser.parse_args()

    # Ensure output streams in real time (Python buffers when not on a TTY)
    sys.stdout.reconfigure(line_buffering=True)

    # Load data
    print("Loading QF entries...")
    qf_entries = load_all_entries(SOURCES['quality_filtered'])
    print(f"  {len(qf_entries):,} QF games loaded")

    print("Loading LI entries...")
    li_entries = load_all_entries(SOURCES['logged_in_games'])
    print(f"  {len(li_entries):,} LI games loaded")

    unfiltered_entries = None
    if args.source in ('unfiltered', 'unfiltered-oldest', 'unfiltered-newest'):
        print("Loading unfiltered entries...")
        unfiltered_entries = load_all_entries(SOURCES['unfiltered'])
        if args.source == 'unfiltered':
            random.Random(0).shuffle(unfiltered_entries)
            print(f"  {len(unfiltered_entries):,} unfiltered games loaded (shuffled)")
        elif args.source == 'unfiltered-newest':
            unfiltered_entries.reverse()
            print(f"  {len(unfiltered_entries):,} unfiltered games loaded (newest first)")
        else:
            print(f"  {len(unfiltered_entries):,} unfiltered games loaded (oldest first)")

    pool, qf_ids, li_ids, data_stats = build_deduplicated_pool(
        qf_entries, li_entries)
    print(f"\nData stats: QF={data_stats['qf_total']:,}, "
          f"LI={data_stats['li_total']:,}, "
          f"overlap={data_stats['overlap']:,}, "
          f"union={data_stats['union_total']:,}")
    if unfiltered_entries is not None:
        data_stats['unfiltered_total'] = len(unfiltered_entries)

    holdout_X, holdout_y, holdout_gids, holdout_ts = load_holdout()

    if args.mode == 'union':
        # Select pool based on --source
        if args.source == 'qf':
            active_pool = qf_entries
            source_label = 'QF-only'
        elif args.source == 'li':
            active_pool = li_entries
            source_label = 'LI-only'
        elif args.source in ('unfiltered', 'unfiltered-oldest', 'unfiltered-newest'):
            active_pool = unfiltered_entries
            source_label = {'unfiltered': 'Unfiltered',
                           'unfiltered-oldest': 'Unfiltered-Oldest',
                           'unfiltered-newest': 'Unfiltered-Newest'}[args.source]
        else:
            active_pool = pool
            source_label = 'Union'

        # Determine schedule
        if args.max_games is not None:
            num_leaves = args.num_leaves or 100
            num_trees = args.num_trees or 100
            schedule = [(args.max_games, num_leaves, num_trees)]
        else:
            schedule = SCALING_SCHEDULE

        print(f"\n{'=' * 60}")
        print(f"UNION MODE ({source_label}): {len(schedule)} scale points, "
              f"{args.n_seeds} seeds each")
        for mg, nl, nt in schedule:
            print(f"  {mg:>7} games, {nl}L/{nt}T")
        print(f"{'=' * 60}")

        all_runs = run_union(
            active_pool, holdout_X, holdout_y, holdout_gids, holdout_ts,
            schedule, args.n_seeds, args.output, args.workers,
            drop_prob=args.drop_prob, source=args.source,
            sym_aug=args.sym_aug)

        # Final save with data_stats
        if args.output:
            data_stats['source'] = args.source
            _save_results(args.output, 'union', schedule, None,
                          all_runs, data_stats=data_stats,
                          drop_prob=args.drop_prob,
                          sym_aug=args.sym_aug)

        print_summary_table(all_runs, 'union')

    elif args.mode == 'mix-ratio':
        if args.max_games is None:
            parser.error('--max-games is required for mix-ratio mode')
        num_leaves = args.num_leaves or 100
        num_trees = args.num_trees or 100

        print(f"\n{'=' * 60}")
        print(f"MIX-RATIO MODE: {args.max_games} games, "
              f"{num_leaves}L/{num_trees}T")
        print(f"  Ratios: {args.ratios}")
        print(f"  Seeds: {args.n_seeds}")
        print(f"{'=' * 60}")

        all_runs = run_mix_ratio(
            qf_entries, li_entries,
            holdout_X, holdout_y, holdout_gids, holdout_ts,
            args.max_games, num_leaves, num_trees, args.ratios,
            args.n_seeds, args.output, args.workers,
            drop_prob=args.drop_prob, sym_aug=args.sym_aug)

        schedule = [(args.max_games, num_leaves, num_trees)]
        if args.output:
            _save_results(args.output, 'mix-ratio', schedule,
                          args.ratios, all_runs, data_stats=data_stats,
                          drop_prob=args.drop_prob,
                          sym_aug=args.sym_aug)

        print_summary_table(all_runs, 'mix-ratio')


if __name__ == '__main__':
    main()
