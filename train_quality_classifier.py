#!/usr/bin/env python3
"""Train a LightGBM classifier to distinguish high-quality competitive games
from junk/practice/button-check games.

Labels: logged_in=1 (high quality), unfiltered=0 (mixed quality).
Validation: tournament games should almost all score as high quality.

Usage:
    python train_quality_classifier.py              # uses cached features
    python train_quality_classifier.py --recompute  # recomputes features from CSVs
    python train_quality_classifier.py --sweep      # data size sweep 1K-32K
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

# 20 shards (~20K games), sliced to DEFAULT_SIZE in training
UNFILTERED_PATH = 'unfiltered_partitioned/gameevents_0[0-1][0-9].csv.gz'
LOGGED_IN_PATH = 'logged_in_games/gameevents_0[0-1][0-9].csv.gz'
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


def train_and_evaluate(df_unfiltered_train, df_logged_in_train,
                       df_unfiltered_all, df_logged_in_all, df_tournament,
                       features, verbose=True):
    """Train model and return metrics dict."""
    X_unf_train = df_unfiltered_train[features].values
    X_log_train = df_logged_in_train[features].values
    X = np.vstack([X_unf_train, X_log_train])
    y = np.concatenate([np.zeros(len(X_unf_train)), np.ones(len(X_log_train))])

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, y_train, feature_name=features)
    val_data = lgb.Dataset(X_val, y_val, feature_name=features, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'verbose': -1,
    }

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=verbose)]
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )

    val_preds = model.predict(X_val)
    auc = sklearn.metrics.roc_auc_score(y_val, val_preds)
    log_loss = sklearn.metrics.log_loss(y_val, val_preds)

    # Tournament-anchored thresholds
    X_tournament = df_tournament[features].values
    X_unf_all = df_unfiltered_all[features].values
    X_log_all = df_logged_in_all[features].values
    tournament_scores = model.predict(X_tournament)
    unfiltered_scores = model.predict(X_unf_all)
    logged_in_scores = model.predict(X_log_all)

    results = {
        'auc': auc, 'log_loss': log_loss,
        'model': model, 'importance': model.feature_importance(importance_type='gain'),
    }
    for recall_target in [0.99, 0.95]:
        threshold = np.percentile(tournament_scores, 100 * (1 - recall_target))
        pct = int(recall_target * 100)
        results[f'unf_pass_{pct}'] = (unfiltered_scores >= threshold).mean()
        results[f'log_pass_{pct}'] = (logged_in_scores >= threshold).mean()
        results[f'threshold_{pct}'] = threshold

    return results


def main():
    parser = argparse.ArgumentParser(description='Train game quality classifier')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute features from CSVs (delete cache)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run data size sweep from 1K to 32K')
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
    df_unfiltered_clean = df_unfiltered[~df_unfiltered['game_id'].isin(tournament_ids)]
    df_logged_in_clean = df_logged_in[~df_logged_in['game_id'].isin(tournament_ids)]
    print(f'\n=== Tournament Leakage Removal ===')
    print(f'Removed from unfiltered: {len(df_unfiltered) - len(df_unfiltered_clean)}/{len(df_unfiltered)}')
    print(f'Removed from logged-in:  {len(df_logged_in) - len(df_logged_in_clean)}/{len(df_logged_in)}')

    features = FEATURE_COLUMNS

    if args.sweep:
        # === Data Size Sweep ===
        sizes = [1000, 2000, 4000, 8000, 16000, 32000]
        print(f'\n=== Data Size Sweep ({len(features)} features) ===')
        print(f'{"Size":>7} {"Unf":>7} {"Log":>7} {"AUC":>7} {"LogLoss":>8} {"Unf@99%":>8} {"Unf@95%":>8} {"Log@99%":>8} {"Log@95%":>8}')
        print('-' * 80)

        for n in sizes:
            df_unf_n = df_unfiltered_clean.iloc[:n]
            df_log_n = df_logged_in_clean.iloc[:n]
            actual_unf = len(df_unf_n)
            actual_log = len(df_log_n)

            r = train_and_evaluate(
                df_unf_n, df_log_n,
                df_unfiltered, df_logged_in, df_tournament,
                features, verbose=False,
            )
            print(f'{n:>7} {actual_unf:>7} {actual_log:>7} '
                  f'{r["auc"]:>7.4f} {r["log_loss"]:>8.4f} '
                  f'{r["unf_pass_99"]:>7.1%} {r["unf_pass_95"]:>7.1%} '
                  f'{r["log_pass_99"]:>7.1%} {r["log_pass_95"]:>7.1%}')
        return

    # === Single Run ===
    DEFAULT_SIZE = 16000
    df_unfiltered_train = df_unfiltered_clean.iloc[:DEFAULT_SIZE]
    df_logged_in_train = df_logged_in_clean.iloc[:DEFAULT_SIZE]
    print(f'Training sets: Unfiltered={len(df_unfiltered_train)}, Logged-in={len(df_logged_in_train)}')
    print(f'\n=== Features ({len(features)} total) ===')
    print(features)

    r = train_and_evaluate(
        df_unfiltered_train, df_logged_in_train,
        df_unfiltered, df_logged_in, df_tournament,
        features,
    )

    print(f'\n=== Validation ===')
    print(f'Accuracy: N/A  AUC: {r["auc"]:.4f}  Log Loss: {r["log_loss"]:.4f}')

    feat_imp = sorted(zip(features, r['importance']), key=lambda x: -x[1])
    print(f'\n=== Feature Importance (top 15) ===')
    for name, imp in feat_imp[:15]:
        print(f'  {name:<40} {imp:.1f}')

    print(f'\n=== Tournament-Anchored Quality Threshold ===')
    for pct in [99, 95]:
        print(f'At {pct}% tournament recall: threshold={r[f"threshold_{pct}"]:.4f}')
        print(f'  Unfiltered passing:  {r[f"unf_pass_{pct}"]:.1%}  <- lower is better')
        print(f'  Logged-in passing:   {r[f"log_pass_{pct}"]:.1%}')


if __name__ == '__main__':
    main()
