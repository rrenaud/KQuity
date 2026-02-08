#!/usr/bin/env python3
"""LightGBM data scaling experiment.

Tests whether training on 10x more data (925 partitions vs ~92) improves
win prediction, and whether model capacity needs to scale with data volume.

Data splits:
  - Training: partitions 0-904 (excluding late tournament game IDs)
  - Validation: partitions 905-924 (excluding late tournament game IDs)
  - Holdout: late tournament games (separate CSV)
"""

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

import lightgbm as lgb
import sklearn.metrics

from fast_materialize import fast_materialize
from train_model import evaluate_model

# 256-CPU machine: LightGBM thread overhead kills perf on <1M samples.
LGB_NUM_THREADS = min(os.cpu_count(), 16)


def train_lgb_model(train_X, train_y, num_leaves=100, num_trees=100):
    """Train LightGBM with capped thread count."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_threads': LGB_NUM_THREADS,
        'verbose': 0,
    }
    train_data = lgb.Dataset(train_X, train_y)
    return lgb.train(param, train_data, num_boost_round=num_trees)

try:
    import wandb
except ImportError:
    wandb = None

DATA_DIR = 'new_data_partitioned'
LATE_TOURNAMENT_IDS_FILE = 'late_tournament_games/late_tournament_game_events_game_ids.txt'
LATE_TOURNAMENT_EVENTS = 'late_tournament_games/late_tournament_game_events.csv.gz'

TRAIN_PARTITIONS = (0, 905)      # [0, 905)
VAL_PARTITIONS = (905, 925)      # [905, 925)

DATA_FRACTIONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
CAPACITY_GRID = [
    (31, 50),    # small
    (100, 100),  # medium
    (200, 200),  # large
    (500, 300),  # XL
]

QUICK_FRACTIONS = [0.01, 0.05]
QUICK_CAPACITIES = [(31, 10), (100, 20)]


def load_exclude_ids():
    """Load late tournament game IDs as exclusion set."""
    ids = set()
    with open(LATE_TOURNAMENT_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(int(line))
    print(f"Loaded {len(ids)} late tournament game IDs to exclude", flush=True)
    return ids


def partition_path(partition_idx):
    return f'{DATA_DIR}/gameevents_{partition_idx:03d}.csv.gz'


def materialize_partition_range(start, end, drop_prob, exclude_ids, cache=None):
    """Materialize features from a range of partitions.

    If cache dict is provided, caches per-partition results keyed by
    (partition_idx, drop_prob). Since fast_materialize uses Random(42)
    per call, same partition + drop_prob always gives identical output.
    """
    all_states = []
    all_labels = []
    total = end - start
    cache_hits = 0
    for idx, i in enumerate(range(start, end)):
        cache_key = (i, drop_prob)
        if cache is not None and cache_key in cache:
            states, labels = cache[cache_key]
            cache_hits += 1
        else:
            path = partition_path(i)
            if not os.path.exists(path):
                continue
            try:
                states, labels = fast_materialize(path, drop_prob, exclude_game_ids=exclude_ids)
            except Exception as e:
                print(f"  Error processing partition {i}: {e}", flush=True)
                continue
            if cache is not None:
                cache[cache_key] = (states, labels)
        if len(labels) > 0:
            all_states.append(states)
            all_labels.append(labels)
        if (idx + 1) % 10 == 0 or idx + 1 == total:
            n = sum(len(l) for l in all_labels)
            hit_str = f", {cache_hits} cached" if cache_hits else ""
            print(f"  [{idx+1}/{total}] {n:,} samples so far{hit_str}", flush=True)
    if all_states:
        return np.vstack(all_states), np.concatenate(all_labels)
    return np.empty((0, 52), dtype=np.float32), np.empty(0, dtype=np.int8)


def num_partitions_for_fraction(fraction):
    """Return number of training partitions for a given data fraction."""
    total = TRAIN_PARTITIONS[1] - TRAIN_PARTITIONS[0]  # 905
    n = max(1, int(total * fraction))
    return n


def save_incremental(path, results, best_result=None, holdout_metrics=None):
    """Save results to JSON after each grid point so nothing is lost on interrupt."""
    output = {'grid_results': results}
    if best_result:
        output['best_config'] = best_result
    if holdout_metrics:
        output['holdout_metrics'] = holdout_metrics
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)


def print_results_table(results):
    """Print a formatted results table."""
    w = 130
    print("\n" + "=" * w)
    print("RESULTS")
    print("=" * w)
    header = (
        f"{'Fraction':>8} {'Parts':>5} {'Leaves':>6} {'Trees':>5} {'TrainN':>9} "
        f"{'ValLoss':>8} {'ValAcc':>7} {'VInv':>6} "
        f"{'HoLoss':>8} {'HoAcc':>7} {'HoInv':>6} "
        f"{'Time':>6}"
    )
    print(header)
    print("-" * w)
    for r in results:
        print(
            f"{r['fraction']:>7.0%} {r['num_partitions']:>5d} "
            f"{r['num_leaves']:>6d} {r['num_trees']:>5d} "
            f"{r['num_train_samples']:>9,d} "
            f"{r['val_log_loss']:>8.4f} {r['val_accuracy']:>6.1%} "
            f"{r['val_inversions']:>5.2%} "
            f"{r['holdout_log_loss']:>8.4f} {r['holdout_accuracy']:>6.1%} "
            f"{r['holdout_inversions']:>5.2%} "
            f"{r['train_time_s']:>5.1f}s"
        )
    print("=" * w)


def main():
    parser = argparse.ArgumentParser(description='LGB data scaling experiment')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 2 fractions x 2 capacities')
    parser.add_argument('--output', type=str, default='lgb_scaling_results.json',
                        help='Output JSON file')
    parser.add_argument('--model-dir', type=str, default='lgb_scaling_models',
                        help='Directory to save trained models')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='kquity-lgb-scaling',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name')
    args = parser.parse_args()

    fractions = QUICK_FRACTIONS if args.quick else DATA_FRACTIONS
    capacities = QUICK_CAPACITIES if args.quick else CAPACITY_GRID

    # W&B setup
    use_wandb = (not args.no_wandb) and wandb is not None
    if not args.no_wandb and wandb is None:
        print("WARNING: wandb not installed, skipping (use --no-wandb to silence)")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'fractions': fractions,
                'capacities': capacities,
                'quick': args.quick,
            }
        )

    # Step 1: Load exclusion set
    exclude_ids = load_exclude_ids()

    # Partition cache: since fast_materialize uses Random(42) per call,
    # same partition + drop_prob always gives identical output.
    partition_cache = {}

    # Step 2: Materialize validation set (once)
    # drop_prob=0.9 keeps ~10% of events — still tens of thousands of samples,
    # plenty for stable metrics, and 10x faster than keeping everything.
    print("\nMaterializing validation set (partitions 905-924, drop_prob=0.9)...", flush=True)
    t0 = time.time()
    val_X, val_y = materialize_partition_range(
        VAL_PARTITIONS[0], VAL_PARTITIONS[1], drop_prob=0.9, exclude_ids=exclude_ids
    )
    print(f"  Validation: {len(val_y):,} samples in {time.time() - t0:.1f}s", flush=True)

    # Step 3: Materialize holdout set (once)
    print("\nMaterializing holdout set (late tournament games, drop_prob=0.9)...", flush=True)
    t0 = time.time()
    holdout_X, holdout_y = fast_materialize(
        LATE_TOURNAMENT_EVENTS, drop_state_probability=0.9
    )
    print(f"  Holdout: {len(holdout_y):,} samples in {time.time() - t0:.1f}s", flush=True)

    # Step 4: Grid search
    model_dir = args.model_dir
    pathlib.Path(model_dir).mkdir(exist_ok=True, parents=True)

    results = []
    best_result = None
    best_model = None
    best_val_loss = float('inf')

    for fraction in fractions:
        n_parts = num_partitions_for_fraction(fraction)
        start = TRAIN_PARTITIONS[0]
        end = start + n_parts

        print(f"\n{'='*60}", flush=True)
        print(f"Data fraction: {fraction:.0%} ({n_parts} partitions, {start}-{end-1})", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        train_X, train_y = materialize_partition_range(
            start, end, drop_prob=0.9, exclude_ids=exclude_ids,
            cache=partition_cache
        )
        mat_time = time.time() - t0
        print(f"  Materialized {len(train_y):,} training samples in {mat_time:.1f}s", flush=True)

        for num_leaves, num_trees in capacities:
            print(f"\n  Training: leaves={num_leaves}, trees={num_trees}...", flush=True)
            t0 = time.time()
            model = train_lgb_model(train_X, train_y,
                                    num_leaves=num_leaves, num_trees=num_trees)
            train_time = time.time() - t0
            print(f"  Trained in {train_time:.1f}s", flush=True)

            # Save model
            model_name = f"frac{fraction:.2f}_l{num_leaves}_t{num_trees}.mdl"
            model_path = os.path.join(model_dir, model_name)
            model.save_model(model_path)

            val_metrics = evaluate_model(
                model, val_X, val_y,
                f"Val [frac={fraction:.0%}, leaves={num_leaves}, trees={num_trees}]"
            )
            holdout_metrics = evaluate_model(
                model, holdout_X, holdout_y,
                f"Holdout [frac={fraction:.0%}, leaves={num_leaves}, trees={num_trees}]"
            )

            capacity_label = f"{num_leaves}L/{num_trees}T"
            result = {
                'fraction': fraction,
                'num_partitions': n_parts,
                'capacity': capacity_label,
                'num_leaves': num_leaves,
                'num_trees': num_trees,
                'num_train_samples': int(len(train_y)),
                'val_log_loss': val_metrics['log_loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_inversions': val_metrics['inversions'],
                'holdout_log_loss': holdout_metrics['log_loss'],
                'holdout_accuracy': holdout_metrics['accuracy'],
                'holdout_inversions': holdout_metrics['inversions'],
                'train_time_s': round(train_time, 1),
                'model_path': model_path,
            }
            results.append(result)
            save_incremental(args.output, results, best_result)

            if use_wandb:
                wandb.log({
                    f"val_loss/{capacity_label}": val_metrics['log_loss'],
                    f"val_acc/{capacity_label}": val_metrics['accuracy'],
                    f"val_inv/{capacity_label}": val_metrics['inversions'],
                    f"holdout_loss/{capacity_label}": holdout_metrics['log_loss'],
                    f"holdout_acc/{capacity_label}": holdout_metrics['accuracy'],
                    f"holdout_inv/{capacity_label}": holdout_metrics['inversions'],
                    "fraction": fraction,
                    "num_train_samples": int(len(train_y)),
                })

            if val_metrics['log_loss'] < best_val_loss:
                best_val_loss = val_metrics['log_loss']
                best_result = result

        # Free combined training arrays (per-partition data stays in cache)
        del train_X, train_y

    print_results_table(results)
    save_incremental(args.output, results, best_result)

    print(f"\nBest config (by val loss): fraction={best_result['fraction']:.0%}, "
          f"leaves={best_result['num_leaves']}, trees={best_result['num_trees']} "
          f"(val_loss={best_result['val_log_loss']:.4f}, "
          f"holdout_loss={best_result['holdout_log_loss']:.4f})", flush=True)

    if use_wandb:
        # Log structured table for clean grid visualization
        table = wandb.Table(columns=[
            "fraction", "capacity", "num_train_samples",
            "val_log_loss", "val_accuracy", "val_inversions",
            "holdout_log_loss", "holdout_accuracy", "holdout_inversions",
            "train_time_s",
        ])
        for r in results:
            table.add_data(
                r['fraction'], r['capacity'], r['num_train_samples'],
                r['val_log_loss'], r['val_accuracy'], r['val_inversions'],
                r['holdout_log_loss'], r['holdout_accuracy'], r['holdout_inversions'],
                r['train_time_s'],
            )
        wandb.log({"results_table": table})
        wandb.log({
            'best/fraction': best_result['fraction'],
            'best/num_leaves': best_result['num_leaves'],
            'best/num_trees': best_result['num_trees'],
            'best/val_log_loss': best_result['val_log_loss'],
            'best/holdout_log_loss': best_result['holdout_log_loss'],
        })
        wandb.finish()

    print(f"\nModels saved to {model_dir}/")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
