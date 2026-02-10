#!/usr/bin/env python3
"""A/B experiment: compare LightGBM with and without player rating features.

Loads ratings.pkl and materializes logged_in_games data with three variants:
  - 52-feature baseline (no ratings)
  - 62-feature variant (with per-player mu)
  - 56-feature variant (queen mu + avg worker mu per team)

Splits by game chronological order, trains all models, compares on test set.
"""

import os
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


def condense_ratings(ratings_by_game):
    """Collapse per-worker ratings into queen_mu + avg_worker_mu per team.

    Input: {game_id: array(10)} where positions are [gold_queen, blue_queen,
           gold_w0, blue_w0, gold_w1, blue_w1, gold_w2, blue_w2, gold_w3, blue_w3]
    Output: {game_id: array(10)} where each worker slot gets the team average.
    """
    condensed = {}
    for gid, ratings in ratings_by_game.items():
        r = np.array(ratings, dtype=np.float32)
        # Queens stay the same (indices 0, 1)
        # Gold workers: indices 2,4,6,8 -> average
        gold_avg = np.mean(r[[2, 4, 6, 8]])
        # Blue workers: indices 3,5,7,9 -> average
        blue_avg = np.mean(r[[3, 5, 7, 9]])
        condensed_r = r.copy()
        condensed_r[[2, 4, 6, 8]] = gold_avg
        condensed_r[[3, 5, 7, 9]] = blue_avg
        condensed[gid] = condensed_r
    return condensed


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
    limit_bytes = 12 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    csv_pattern = 'logged_in_games/gameevents_00[0-9].csv.gz'

    print('Loading ratings.pkl...')
    ratings_by_game = load_ratings()
    print(f'  {len(ratings_by_game)} games with ratings')

    condensed_ratings = condense_ratings(ratings_by_game)

    # Materialize without ratings (52 features)
    print('\nMaterializing without ratings (52 features)...')
    start = time.time()
    states_52, labels_52, game_ids_52 = fast_materialize(csv_pattern)
    print(f'  {states_52.shape[0]} samples, shape={states_52.shape}, {time.time()-start:.1f}s')

    # Materialize with per-player ratings (62 features)
    print('\nMaterializing with per-player ratings (62 features)...')
    start = time.time()
    states_62, labels_62, game_ids_62 = fast_materialize(csv_pattern, ratings_by_game=ratings_by_game)
    print(f'  {states_62.shape[0]} samples, shape={states_62.shape}, {time.time()-start:.1f}s')

    # Materialize with condensed ratings (62 features, but worker ratings are team averages)
    print('\nMaterializing with queen + avg worker ratings (62 features, condensed)...')
    start = time.time()
    states_cond, labels_cond, game_ids_cond = fast_materialize(csv_pattern, ratings_by_game=condensed_ratings)
    print(f'  {states_cond.shape[0]} samples, shape={states_cond.shape}, {time.time()-start:.1f}s')

    # Chronological split (use 52-feature game_ids as reference)
    train_ids, test_ids = chronological_split(game_ids_52, fraction=0.5)
    print(f'\nSplit: {len(train_ids)} train games, {len(test_ids)} test games')

    train_X_52, train_y_52 = filter_by_game_ids(states_52, labels_52, game_ids_52, train_ids)
    test_X_52, test_y_52 = filter_by_game_ids(states_52, labels_52, game_ids_52, test_ids)

    train_X_62, train_y_62 = filter_by_game_ids(states_62, labels_62, game_ids_62, train_ids)
    test_X_62, test_y_62 = filter_by_game_ids(states_62, labels_62, game_ids_62, test_ids)

    train_X_cond, train_y_cond = filter_by_game_ids(states_cond, labels_cond, game_ids_cond, train_ids)
    test_X_cond, test_y_cond = filter_by_game_ids(states_cond, labels_cond, game_ids_cond, test_ids)

    print(f'  52-feat: train={len(train_y_52)}, test={len(test_y_52)}')
    print(f'  62-feat: train={len(train_y_62)}, test={len(test_y_62)}')
    print(f'  condensed: train={len(train_y_cond)}, test={len(test_y_cond)}')

    # Train and evaluate
    all_metrics = {}
    all_metrics['baseline'] = train_and_evaluate(
        train_X_52, train_y_52, test_X_52, test_y_52, 'Baseline (52 features)')
    all_metrics['per_player'] = train_and_evaluate(
        train_X_62, train_y_62, test_X_62, test_y_62, 'Per-Player Ratings (62 features)')
    all_metrics['condensed'] = train_and_evaluate(
        train_X_cond, train_y_cond, test_X_cond, test_y_cond, 'Queen + Avg Worker (62 features, condensed)')

    # Comparison table
    print('\n' + '=' * 75)
    print('COMPARISON')
    print('=' * 75)
    print(f'{"Metric":<12} {"Baseline":<12} {"Per-Player":<12} {"Condensed":<12} {"PP Diff":<12} {"C Diff":<12}')
    print('-' * 75)
    for metric in ['log_loss', 'accuracy']:
        vb = all_metrics['baseline'][metric]
        vp = all_metrics['per_player'][metric]
        vc = all_metrics['condensed'][metric]
        dp = vp - vb
        dc = vc - vb
        sp = '+' if dp > 0 else ''
        sc = '+' if dc > 0 else ''
        print(f'{metric:<12} {vb:<12.4f} {vp:<12.4f} {vc:<12.4f} {sp}{dp:<12.4f} {sc}{dc:<12.4f}')


if __name__ == '__main__':
    main()
