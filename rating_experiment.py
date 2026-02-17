#!/usr/bin/env python3
"""A/B experiment: compare LightGBM with and without player rating features.

Loads ratings pickle and materializes logged_in_games data with two variants:
  - 52-feature baseline (no ratings)
  - 62-feature variant (with per-player mu)

Splits by game chronological order, trains both models, compares on test set.
"""

import pickle
import resource
import time

import lightgbm as lgb
import numpy as np
import sklearn.metrics

from fast_materialize import fast_materialize


def load_ratings(path='ratings.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def chronological_split(game_ids, fraction=0.5):
    """Split game_ids into train/test by chronological order (game_id proxy)."""
    unique_ids = np.unique(game_ids)
    unique_ids.sort()
    split_idx = int(len(unique_ids) * fraction)
    train_ids = set(unique_ids[:split_idx])
    test_ids = set(unique_ids[split_idx:])
    return train_ids, test_ids


def filter_by_game_ids(states, labels, game_ids, keep_ids):
    mask = np.array([gid in keep_ids for gid in game_ids])
    return states[mask], labels[mask]


def train_and_evaluate(train_X, train_y, test_X, test_y, name, num_leaves=200, num_trees=200):
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
    train_time = time.time() - start

    predictions = model.predict(test_X)
    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    print(f'\n{name}:')
    print(f'  Train time: {train_time:.1f}s')
    print(f'  Log Loss:   {log_loss:.4f}')
    print(f'  Accuracy:   {accuracy:.4f} ({100*accuracy:.1f}%)')

    return {'log_loss': log_loss, 'accuracy': accuracy, 'train_time': train_time}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='A/B experiment: ratings features')
    parser.add_argument('--ratings', type=str, default='ratings.pkl',
                        help='Path to ratings pickle file (default: ratings.pkl)')
    args = parser.parse_args()

    limit_bytes = 12 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    csv_pattern = 'logged_in_games/gameevents_00[0-9].csv.gz'

    print(f'Loading {args.ratings}...')
    ratings_by_game = load_ratings(args.ratings)
    print(f'  {len(ratings_by_game)} games with ratings')

    # Materialize with per-player ratings (62 features)
    print('\nMaterializing with per-player ratings (62 features)...')
    start = time.time()
    states_62, labels_62, game_ids_62, _ = fast_materialize(csv_pattern, ratings_by_game=ratings_by_game)
    print(f'  {states_62.shape[0]} samples, shape={states_62.shape}, {time.time()-start:.1f}s')

    # Chronological split
    train_ids, test_ids = chronological_split(game_ids_62, fraction=0.5)
    print(f'\nSplit: {len(train_ids)} train games, {len(test_ids)} test games')

    train_X_62, train_y_62 = filter_by_game_ids(states_62, labels_62, game_ids_62, train_ids)
    test_X_62, test_y_62 = filter_by_game_ids(states_62, labels_62, game_ids_62, test_ids)

    print(f'  62-feat: train={len(train_y_62)}, test={len(test_y_62)}')

    # Train and evaluate
    results = train_and_evaluate(
        train_X_62, train_y_62, test_X_62, test_y_62, 'Per-Player Ratings (62 features)')

    print('\n' + '=' * 60)
    print('RESULTS (baseline: 69.1% accuracy, 0.5765 log_loss)')
    print('=' * 60)
    print(f'{"Metric":<12} {"Baseline":<12} {"Per-Player":<12} {"Diff":<12}')
    print('-' * 60)
    baseline = {'log_loss': 0.5765, 'accuracy': 0.691}
    for metric in ['log_loss', 'accuracy']:
        vb = baseline[metric]
        vp = results[metric]
        dp = vp - vb
        sp = '+' if dp > 0 else ''
        print(f'{metric:<12} {vb:<12.4f} {vp:<12.4f} {sp}{dp:<12.4f}')


if __name__ == '__main__':
    main()
