#!/usr/bin/env python3
"""Train a LightGBM classifier to distinguish high-quality competitive games
from junk/practice/button-check games.

Labels: logged_in=1 (high quality), unfiltered=0 (mixed quality).
Validation: tournament games should almost all score as high quality.

Usage:
    python train_quality_classifier.py              # uses cached features
    python train_quality_classifier.py --recompute  # recomputes features from CSVs
"""

import argparse
import os
import pathlib
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

from game_quality_features import compute_quality_features, FEATURE_COLUMNS

CACHE_DIR = 'quality_cache'

UNFILTERED_PATH = 'unfiltered_partitioned/gameevents_00[0-9].csv.gz'
LOGGED_IN_PATH = 'logged_in_games/gameevents_00[0-9].csv.gz'
TOURNAMENT_PATH = 'late_tournament_games/late_tournament_game_events.csv.gz'


def load_or_compute(name, csv_path, recompute=False):
    """Load cached features or compute from CSVs."""
    cache_path = f'{CACHE_DIR}/{name}.parquet'
    if not recompute and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f'Loaded {name}: {len(df)} games from cache')
        return df

    print(f'Computing features for {name}...')
    df = compute_quality_features(csv_path)
    pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f'Cached {name}: {len(df)} games')
    return df


def main():
    parser = argparse.ArgumentParser(description='Train game quality classifier')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute features from CSVs (delete cache)')
    args = parser.parse_args()

    if args.recompute and os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f'Deleted cache directory: {CACHE_DIR}')

    # === Data ===
    df_unfiltered = load_or_compute('unfiltered', UNFILTERED_PATH, args.recompute)
    df_logged_in = load_or_compute('logged_in', LOGGED_IN_PATH, args.recompute)
    df_tournament = load_or_compute('tournament', TOURNAMENT_PATH, args.recompute)

    print(f'\n=== Data ===')
    print(f'Unfiltered: {len(df_unfiltered)} games, Logged-in: {len(df_logged_in)} games, Tournament: {len(df_tournament)} games')

    # Remove tournament games from training sets to avoid data leakage
    tournament_ids = set(df_tournament['game_id'])
    unf_before = len(df_unfiltered)
    log_before = len(df_logged_in)
    df_unfiltered_train = df_unfiltered[~df_unfiltered['game_id'].isin(tournament_ids)]
    df_logged_in_train = df_logged_in[~df_logged_in['game_id'].isin(tournament_ids)]
    unf_removed = unf_before - len(df_unfiltered_train)
    log_removed = log_before - len(df_logged_in_train)
    print(f'\n=== Tournament Leakage Removal ===')
    print(f'Removed from unfiltered: {unf_removed}/{unf_before}')
    print(f'Removed from logged-in:  {log_removed}/{log_before}')
    print(f'Training sets: Unfiltered={len(df_unfiltered_train)}, Logged-in={len(df_logged_in_train)}')

    features = FEATURE_COLUMNS
    print(f'\n=== Features ({len(features)} total) ===')
    print(features)

    # Build training data: unfiltered=0, logged_in=1 (tournament games excluded)
    X_unf_train = df_unfiltered_train[features].values
    X_log_train = df_logged_in_train[features].values
    X = np.vstack([X_unf_train, X_log_train])
    y = np.concatenate([np.zeros(len(X_unf_train)), np.ones(len(X_log_train))])

    # 80/20 stratified split
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # === Train LightGBM ===
    train_data = lgb.Dataset(X_train, y_train, feature_name=features)
    val_data = lgb.Dataset(X_val, y_val, feature_name=features, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'verbose': -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True)],
    )

    # === Validation ===
    val_preds = model.predict(X_val)
    accuracy = sklearn.metrics.accuracy_score(y_val, val_preds > 0.5)
    auc = sklearn.metrics.roc_auc_score(y_val, val_preds)
    log_loss = sklearn.metrics.log_loss(y_val, val_preds)

    print(f'\n=== Validation ===')
    print(f'Accuracy: {accuracy:.4f}  AUC: {auc:.4f}  Log Loss: {log_loss:.4f}')

    # === Feature Importance ===
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(features, importance), key=lambda x: -x[1])

    print(f'\n=== Feature Importance (top 15) ===')
    for name, imp in feat_imp[:15]:
        print(f'  {name:<40} {imp:.1f}')

    # === Tournament-Anchored Quality Threshold ===
    # Score all games (including any that were removed from training)
    X_tournament = df_tournament[features].values
    X_unf_all = df_unfiltered[features].values
    X_log_all = df_logged_in[features].values
    tournament_scores = model.predict(X_tournament)
    unfiltered_scores = model.predict(X_unf_all)
    logged_in_scores = model.predict(X_log_all)

    print(f'\n=== Tournament-Anchored Quality Threshold ===')
    for recall_target in [0.99, 0.95]:
        # Find threshold that retains recall_target fraction of tournament games
        threshold = np.percentile(tournament_scores, 100 * (1 - recall_target))

        tourn_passing = (tournament_scores >= threshold).sum()
        unf_passing = (unfiltered_scores >= threshold).sum()
        log_passing = (logged_in_scores >= threshold).sum()

        pct_label = f'{int(recall_target * 100)}%'
        print(f'At {pct_label} tournament recall: threshold={threshold:.4f}')
        print(f'  Tournament passing:  {tourn_passing}/{len(tournament_scores)} ({100 * tourn_passing / len(tournament_scores):.1f}%)')
        print(f'  Unfiltered passing:  {unf_passing}/{len(unfiltered_scores)} ({100 * unf_passing / len(unfiltered_scores):.1f}%)  <- lower is better')
        print(f'  Logged-in passing:   {log_passing}/{len(logged_in_scores)} ({100 * log_passing / len(logged_in_scores):.1f}%)')


if __name__ == '__main__':
    main()
