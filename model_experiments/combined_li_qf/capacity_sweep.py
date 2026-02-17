#!/usr/bin/env python3
"""Sweep tree/leaf capacity on 200K union games with symmetry augmentation.

Trains models at different capacities, evaluates on tournament holdout,
and saves each model.
"""

import json
import os
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
    'quality_filtered': 'quality_filtered/encoded/all_games.bin',
    'logged_in_games': 'logged_in_games/encoded/all_games.bin',
}
HOLDOUT_PATH = 'late_tournament_games/encoded/all_games.bin'
MAX_GAMES = 200_000

SWEEP = [
    # (num_leaves, num_trees)
    (100, 100),
    (150, 150),
    (200, 200),
]


def build_interleaved_pool():
    """Load QF + LI and interleave by quality rank."""
    print("Loading QF entries...")
    qf = list(read_packed_games(SOURCES['quality_filtered']))
    print(f"  {len(qf):,} QF games")

    print("Loading LI entries...")
    li = list(read_packed_games(SOURCES['logged_in_games']))
    print(f"  {len(li):,} LI games")

    pool = interleave_dedup(qf, li)
    print(f"  Union pool: {len(pool):,} games")
    return pool


def main():
    sys.stdout.reconfigure(line_buffering=True)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Build pool and select top 200K
    pool = build_interleaved_pool()

    print(f"\nLoading holdout...")
    from event_codec import fast_materialize_from_codec
    ho_X, ho_y, ho_gids, ho_ts = fast_materialize_from_codec(HOLDOUT_PATH)
    exclude = set(ho_gids)
    sw_X, sw_y = swap_teams(ho_X, ho_y)
    ho_X = np.concatenate([ho_X, sw_X]); ho_y = np.concatenate([ho_y, sw_y])
    ho_ts = np.concatenate([ho_ts, ho_ts])
    del sw_X, sw_y
    print(f"  Holdout: {len(ho_y):,} states (sym-augmented)")

    # Sample top-N excluding holdout
    eligible = [(gid, d) for gid, d in pool if gid not in exclude]
    sampled = eligible[:MAX_GAMES]
    print(f"\nMaterializing {len(sampled):,} games (no state dropping)...")
    t0 = time.time()
    train_X, train_y, train_gids, _ = materialize_entries(sampled)
    n_games = len(np.unique(train_gids))
    print(f"  {len(train_y):,} states from {n_games:,} games in {time.time()-t0:.1f}s")

    # Symmetry augment training data
    print("Symmetry augmenting training data...")
    sw_X, sw_y = swap_teams(train_X, train_y)
    train_X = np.concatenate([train_X, sw_X])
    train_y = np.concatenate([train_y, sw_y])
    del sw_X, sw_y
    print(f"  {len(train_y):,} states after augmentation")

    results = []
    for num_leaves, num_trees in SWEEP:
        print(f"\n{'='*60}")
        print(f"Training {num_leaves}L/{num_trees}T")
        print(f"{'='*60}")

        param = {
            'num_leaves': num_leaves,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'verbose': -1,
            'num_threads': min(os.cpu_count() or 1, 16),
        }
        ds = lgb.Dataset(train_X, train_y, free_raw_data=False)
        t0 = time.time()
        model = lgb.train(param, ds, num_boost_round=num_trees)
        train_s = time.time() - t0
        print(f"  Trained in {train_s:.1f}s")

        # Save model
        model_path = os.path.join(out_dir, f'model_{num_leaves}l_{num_trees}t.mdl')
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

        results.append({
            'num_leaves': num_leaves,
            'num_trees': num_trees,
            'train_seconds': round(train_s, 1),
            'n_train_states': len(train_y),
            'n_games': n_games,
            'metrics': metrics,
        })

    # Save results
    out_path = os.path.join(out_dir, 'capacity_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'max_games': MAX_GAMES,
            'symmetry_augmented': True,
            'drop_prob': 0.0,
            'sweep': [list(s) for s in SWEEP],
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':>10}  {'Log Loss':>10}  {'Accuracy':>10}  "
          f"{'AUC':>10}  {'EggInv':>10}  {'SymDev':>10}  {'ECE':>10}")
    print('-' * 80)
    for r in results:
        s = r['metrics']['standard']
        k = r['metrics']['kq']
        c = r['metrics']['calibration']
        print(f"{r['num_leaves']}L/{r['num_trees']}T  "
              f"{s['log_loss']:>10.4f}  {s['accuracy']:>10.4f}  "
              f"{s['auc_roc']:>10.4f}  {k['egg_inversion_rate']:>10.4f}  "
              f"{k['symmetry_deviation']:>10.4f}  {c['ece']:>10.4f}")


if __name__ == '__main__':
    main()
