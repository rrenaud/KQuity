#!/usr/bin/env python3
"""Retrain with drop_prob=0.9 + symmetry augmentation.

Two configs:
  A) 200K games from QF+LI union (interleaved top-N)
  B) 300K games from QF+LI+Unfiltered (3-way interleaved top-N)

Both use drop_prob=0.9 for materialization, then symmetry-augment.
100L/100T based on capacity sweep results.
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import lightgbm as lgb

from event_codec import read_packed_games, materialize_entries
from eval_metrics import (
    compute_standard_metrics, compute_calibration_metrics,
    compute_temporal_metrics, compute_kq_metrics,
)
from symmetry import swap_teams
from data_pool import interleave_dedup

SOURCES = {
    'qf': 'quality_filtered/encoded/all_games.bin',
    'li': 'logged_in_games/encoded/all_games.bin',
    'unfiltered': 'unfiltered_partitioned/encoded/all_games.bin',
}
HOLDOUT_PATH = 'late_tournament_games/encoded/all_games.bin'


def load_entries(name):
    path = SOURCES[name]
    entries = list(read_packed_games(path))
    print(f"  {name}: {len(entries):,} games")
    return entries


def load_holdout():
    from event_codec import fast_materialize_from_codec
    ho_X, ho_y, ho_gids, ho_ts = fast_materialize_from_codec(HOLDOUT_PATH)
    sw_X, sw_y = swap_teams(ho_X, ho_y)
    ho_X = np.concatenate([ho_X, sw_X])
    ho_y = np.concatenate([ho_y, sw_y])
    ho_ts = np.concatenate([ho_ts, ho_ts])
    ho_gids = np.concatenate([ho_gids, ho_gids])
    del sw_X, sw_y
    return ho_X, ho_y, ho_gids, ho_ts


def train_and_eval(name, pool, max_games, exclude_gids, ho_X, ho_y, ho_ts,
                   drop_prob, num_leaves, num_trees, out_dir):
    print(f"\n{'='*60}")
    print(f"{name}: {max_games:,} games, {num_leaves}L/{num_trees}T, "
          f"drop_prob={drop_prob}")
    print(f"{'='*60}")

    eligible = [(gid, d) for gid, d in pool if gid not in exclude_gids]
    sampled = eligible[:max_games]
    print(f"  Sampled {len(sampled):,} games from pool of {len(eligible):,}")

    t0 = time.time()
    train_X, train_y, train_gids, _ = materialize_entries(
        sampled, drop_state_probability=drop_prob, seed=42)
    n_games = len(np.unique(train_gids))
    mat_s = time.time() - t0
    print(f"  Materialized {len(train_y):,} states from {n_games:,} games "
          f"in {mat_s:.1f}s")

    # Symmetry augment
    sw_X, sw_y = swap_teams(train_X, train_y)
    train_X = np.concatenate([train_X, sw_X])
    train_y = np.concatenate([train_y, sw_y])
    del sw_X, sw_y
    print(f"  Sym-augmented: {len(train_y):,} training states")

    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': -1,
        'num_threads': min(os.cpu_count() or 1, 16),
    }
    ds = lgb.Dataset(train_X, train_y, free_raw_data=True)
    t0 = time.time()
    model = lgb.train(param, ds, num_boost_round=num_trees)
    train_s = time.time() - t0
    print(f"  Trained in {train_s:.1f}s")

    model_path = os.path.join(out_dir, f'{name}.mdl')
    model.save_model(model_path)
    print(f"  Saved {model_path}")

    # Evaluate
    preds = model.predict(ho_X)
    labels = ho_y.astype(np.float64)
    metrics = {
        'standard': compute_standard_metrics(preds, labels),
        'calibration': compute_calibration_metrics(preds, labels),
        'temporal': compute_temporal_metrics(preds, labels, ho_ts),
        'kq': compute_kq_metrics(model, ho_X, labels),
    }
    std = metrics['standard']
    kq = metrics['kq']
    cal = metrics['calibration']
    print(f"  log_loss={std['log_loss']:.4f}  acc={std['accuracy']:.4f}  "
          f"auc={std['auc_roc']:.4f}")
    print(f"  egg_inv={kq['egg_inversion_rate']:.4f}  "
          f"sym_dev={kq['symmetry_deviation']:.4f}  ece={cal['ece']:.4f}")

    return {
        'name': name,
        'max_games': max_games,
        'n_games_actual': n_games,
        'n_train_states': len(train_y),
        'drop_prob': drop_prob,
        'num_leaves': num_leaves,
        'num_trees': num_trees,
        'train_seconds': round(train_s, 1),
        'model_path': model_path,
        'metrics': metrics,
    }


def main():
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading data sources...")
    qf = load_entries('qf')
    li = load_entries('li')
    unf = load_entries('unfiltered')
    # Shuffle unfiltered (chronological order, not quality-ranked)
    random.Random(0).shuffle(unf)

    print("\nBuilding pools...")
    pool_2way = interleave_dedup(qf, li)
    print(f"  QF+LI union: {len(pool_2way):,} games")
    pool_3way = interleave_dedup(qf, li, unf)
    print(f"  QF+LI+Unfiltered union: {len(pool_3way):,} games")

    ho_X, ho_y, ho_gids, ho_ts = load_holdout()
    exclude = set(ho_gids)
    print(f"  Holdout: {len(ho_y):,} states (sym-augmented)")

    results = []

    # Config A: 200K from QF+LI, drop_prob=0.9, sym-aug
    r = train_and_eval('union_200k_drop09', pool_2way, 200_000, exclude,
                       ho_X, ho_y, ho_ts, drop_prob=0.9,
                       num_leaves=100, num_trees=100, out_dir=out_dir)
    results.append(r)

    # Config B: 300K from QF+LI+Unfiltered, drop_prob=0.9, sym-aug
    r = train_and_eval('union3_300k_drop09', pool_3way, 300_000, exclude,
                       ho_X, ho_y, ho_ts, drop_prob=0.9,
                       num_leaves=100, num_trees=100, out_dir=out_dir)
    results.append(r)

    # Save results
    out_path = os.path.join(out_dir, 'retrain_drop09_results.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Name':<25} {'States':>10} {'LogLoss':>9} {'Acc':>9} "
          f"{'AUC':>9} {'EggInv':>9} {'SymDev':>9} {'ECE':>9}")
    print('-' * 100)
    for r in results:
        s = r['metrics']['standard']
        k = r['metrics']['kq']
        c = r['metrics']['calibration']
        print(f"{r['name']:<25} {r['n_train_states']:>10,} {s['log_loss']:>9.4f} "
              f"{s['accuracy']:>9.4f} {s['auc_roc']:>9.4f} "
              f"{k['egg_inversion_rate']:>9.4f} {k['symmetry_deviation']:>9.4f} "
              f"{c['ece']:>9.4f}")


if __name__ == '__main__':
    main()
