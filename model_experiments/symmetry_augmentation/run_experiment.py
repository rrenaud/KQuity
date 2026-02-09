#!/usr/bin/env python3
"""Symmetry augmentation experiment for KQuity.

Compares three training conditions to disentangle "more data" from
"structural invariance" provided by Blue/Gold symmetry augmentation:

  X         — random half of training rows (baseline)
  X + sym(X) — same half + symmetry-swapped copy (augmentation)
  2X        — all training rows (data-size control)

If X+sym(X) beats 2X at matched training set size, symmetry provides
structural value beyond just more data.
"""

import os
import sys
import time
import numpy as np

# Add project root to path so we can import from top-level modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from symmetry import swap_teams
from train_model import load_vectors, train_lgb_model, evaluate_model


def symmetry_consistency(model, test_X, test_y):
    """Measure how well a model respects Blue/Gold symmetry.

    Perfect symmetry means P(blue_wins | state) + P(blue_wins | swap(state)) = 1.
    Returns mean |P(X) + P(swap(X)) - 1|; lower is better (0 = perfect).
    """
    swap_X, swap_y = swap_teams(test_X, test_y)
    p_orig = model.predict(test_X)
    p_swap = model.predict(swap_X)
    return np.mean(np.abs(p_orig + p_swap - 1.0))


def main():
    project_root = os.path.join(os.path.dirname(__file__), '../..')
    expt_dir = os.path.join(project_root, 'model_experiments/new_data_model')

    # Load existing train/test data
    print("Loading data...")
    train_X, train_y = load_vectors(f'{expt_dir}/train')
    test_X, test_y = load_vectors(f'{expt_dir}/test')
    print(f"  Train: {train_X.shape[0]:,} rows x {train_X.shape[1]} features")
    print(f"  Test:  {test_X.shape[0]:,} rows x {test_X.shape[1]} features")

    # Create X subset: 100k random training rows
    X_SIZE = 100_000
    rng = np.random.RandomState(42)
    n = len(train_y)
    x_indices = rng.choice(n, X_SIZE, replace=False)

    half_X = train_X[x_indices]
    half_y = train_y[x_indices]

    # Build symmetry-augmented version: X + sym(X) = 200k rows
    swap_X, swap_y = swap_teams(half_X, half_y)
    aug_X = np.vstack([half_X, swap_X])
    aug_y = np.concatenate([half_y, swap_y])

    # 2X control: 200k different random rows (no overlap with X)
    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[x_indices] = False
    remaining_indices = np.where(remaining_mask)[0]
    twox_indices = np.concatenate([
        x_indices,
        rng.choice(remaining_indices, X_SIZE, replace=False)
    ])
    full_X = train_X[twox_indices]
    full_y = train_y[twox_indices]

    num_leaves = 200
    num_trees = 200

    conditions = [
        ("X (100k)", half_X, half_y),
        ("X + sym(X)", aug_X, aug_y),
        ("2X (200k)", full_X, full_y),
    ]

    results = []
    for name, tr_X, tr_y in conditions:
        print(f"\nTraining: {name} ({tr_X.shape[0]:,} rows)...")
        start = time.time()
        model = train_lgb_model(tr_X, tr_y, num_leaves=num_leaves, num_trees=num_trees)
        elapsed = time.time() - start
        print(f"  Trained in {elapsed:.1f}s")

        metrics = evaluate_model(model, test_X, test_y, name)
        metrics['sym_consistency'] = symmetry_consistency(model, test_X, test_y)
        metrics['train_size'] = len(tr_y)
        print(f"  Symmetry Consistency: {metrics['sym_consistency']:.4f}")
        results.append((name, metrics))

    # Print comparison table
    print("\n")
    print("SYMMETRY EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Condition':<15}| {'Train Size':>12} | {'Log Loss':>10} | {'Accuracy':>10} | {'Sym Consist':>12} | {'Egg Inv':>8}")
    print("-" * 80)
    for name, m in results:
        print(
            f"{name:<15}| {m['train_size']:>12,} | {m['log_loss']:>10.4f} | "
            f"{100*m['accuracy']:>9.1f}% | {m['sym_consistency']:>12.4f} | "
            f"{100*m['inversions']:>6.2f}%"
        )
    print("=" * 80)

    # Key comparisons
    x_metrics = results[0][1]
    aug_metrics = results[1][1]
    full_metrics = results[2][1]

    print("\nKey comparisons (log loss, lower is better):")
    print(f"  Augmentation helps vs baseline?  X+sym(X) vs X:   {aug_metrics['log_loss']:.4f} vs {x_metrics['log_loss']:.4f} (delta {aug_metrics['log_loss'] - x_metrics['log_loss']:+.4f})")
    print(f"  Symmetry vs more data?           X+sym(X) vs 2X:  {aug_metrics['log_loss']:.4f} vs {full_metrics['log_loss']:.4f} (delta {aug_metrics['log_loss'] - full_metrics['log_loss']:+.4f})")
    print(f"  More data helps vs baseline?     2X vs X:         {full_metrics['log_loss']:.4f} vs {x_metrics['log_loss']:.4f} (delta {full_metrics['log_loss'] - x_metrics['log_loss']:+.4f})")

    print(f"\nSymmetry consistency (lower is better):")
    print(f"  X: {x_metrics['sym_consistency']:.4f}  |  X+sym(X): {aug_metrics['sym_consistency']:.4f}  |  2X: {full_metrics['sym_consistency']:.4f}")


if __name__ == '__main__':
    main()
