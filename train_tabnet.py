#!/usr/bin/env python3
"""Train a TabNet model for Killer Queen game win prediction.

Uses logged-in games (sorted by login count, highest quality) for training
and late tournament games for test.

Requires: pip install pytorch-tabnet

Usage:
    python train_tabnet.py
    python train_tabnet.py --num-train-games 5000 --n-d 16 --n-a 16
"""

import argparse
import csv
import gzip
import os
import time

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from evaluate import evaluate_model
from fast_materialize import fast_materialize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train TabNet for KQ game win prediction')
    # Data
    parser.add_argument('--train-dir', type=str, default='logged_in_games',
                        help='Directory with partition files for training')
    parser.add_argument('--test-events', type=str,
                        default='late_tournament_games/late_tournament_game_events.csv.gz',
                        help='CSV file with test game events')
    parser.add_argument('--num-train-games', type=int, default=1000,
                        help='Number of training games (controls how many partitions to load)')
    # TabNet architecture
    parser.add_argument('--n-d', type=int, default=8)
    parser.add_argument('--n-a', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=1.3)
    parser.add_argument('--lambda-sparse', type=float, default=1e-3)
    # Training
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--virtual-batch-size', type=int, default=128)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def get_game_ids(csv_path):
    """Extract unique game IDs from a CSV file."""
    opener = gzip.open if csv_path.endswith('.gz') else open
    game_ids = set()
    with opener(csv_path, 'rt') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            game_ids.add(int(row[4]))  # game_id column
    return game_ids


def get_partition_files(train_dir, num_games, games_per_partition=1000):
    """Get sorted list of partition files needed to cover num_games."""
    files = sorted(
        f for f in os.listdir(train_dir) if f.endswith('.csv.gz')
    )
    num_partitions = (num_games + games_per_partition - 1) // games_per_partition
    selected = files[:num_partitions]
    return [os.path.join(train_dir, f) for f in selected]


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine which partition files to use for training
    partition_files = get_partition_files(args.train_dir, args.num_train_games)
    print(f"Training: {len(partition_files)} partition file(s) from {args.train_dir}")
    print(f"Test: {args.test_events}")

    # Check for game ID overlap
    print("\nChecking for game ID overlap...")
    train_game_ids = set()
    for pf in partition_files:
        train_game_ids |= get_game_ids(pf)
    test_game_ids = get_game_ids(args.test_events)
    overlap = train_game_ids & test_game_ids
    print(f"  Train games: {len(train_game_ids)}")
    print(f"  Test games: {len(test_game_ids)}")
    print(f"  Overlap: {len(overlap)}")
    if overlap:
        print(f"  WARNING: {len(overlap)} games appear in both train and test!")
        print(f"  Overlapping IDs (first 10): {sorted(overlap)[:10]}")

    # Materialize training data (no drop — use all states from selected games)
    print("\nMaterializing training data...")
    start = time.time()
    train_parts = []
    train_label_parts = []
    for pf in partition_files:
        print(f"  {pf}...")
        states, labels = fast_materialize(pf, drop_state_probability=0.0)
        if len(labels) > 0:
            train_parts.append(states)
            train_label_parts.append(labels)
    train_X = np.vstack(train_parts)
    train_y = np.concatenate(train_label_parts)
    print(f"  Train: {train_X.shape} in {time.time() - start:.1f}s")

    # Materialize test data (no drop)
    print("\nMaterializing test data...")
    start = time.time()
    test_X, test_y = fast_materialize(args.test_events, drop_state_probability=0.0)
    print(f"  Test: {test_X.shape} in {time.time() - start:.1f}s")

    # Train TabNet
    clf = TabNetClassifier(
        n_d=args.n_d,
        n_a=args.n_a,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lambda_sparse=args.lambda_sparse,
        optimizer_params=dict(lr=args.lr),
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1,
        seed=args.seed,
    )

    print(f"\nTraining TabNet (n_d={args.n_d}, n_a={args.n_a}, "
          f"n_steps={args.n_steps})...")
    start = time.time()
    clf.fit(
        train_X, train_y,
        eval_set=[(test_X, test_y)],
        eval_name=['test'],
        eval_metric=['logloss', 'accuracy'],
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        virtual_batch_size=args.virtual_batch_size,
    )
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s ({clf.best_epoch} best epoch)")

    def predict_fn(X):
        return clf.predict_proba(X)[:, 1]

    evaluate_model(predict_fn, test_X, test_y, "TabNet")


if __name__ == '__main__':
    main()
