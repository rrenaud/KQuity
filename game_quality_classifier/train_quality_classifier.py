#!/usr/bin/env python3
"""Train a LightGBM classifier to distinguish high-quality competitive games
from junk/practice/button-check games.

Labels: logged_in=1 (high quality), unfiltered=0 (mixed quality).
Validation: tournament games should almost all score as high quality.

Usage (from repo root):
    python -m game_quality_classifier.train_quality_classifier              # uses cached features
    python -m game_quality_classifier.train_quality_classifier --recompute  # recomputes features from CSVs
    python -m game_quality_classifier.train_quality_classifier --sweep      # data size sweep 1K-32K
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

from .game_quality_features import compute_quality_features, FEATURE_COLUMNS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, 'quality_cache')

# Unfiltered shards are time-ordered. Stride over the hundreds digit for
# temporal coverage rather than taking contiguous shards from one time period.
# 20 train shards: X00, X01 for X=0-9. 20 eval shards: X02, X03 for X=0-9.
UNFILTERED_PATH = os.path.join(REPO_ROOT, 'unfiltered_partitioned/gameevents_[0-9]0[01].csv.gz')
UNFILTERED_EVAL_PATH = os.path.join(REPO_ROOT, 'unfiltered_partitioned/gameevents_[0-9]0[23].csv.gz')
# Logged-in shards are sorted by login count (highest quality first).
LOGGED_IN_PATH = os.path.join(REPO_ROOT, 'logged_in_games/gameevents_0[0-1][0-9].csv.gz')
TOURNAMENT_PATH = os.path.join(REPO_ROOT, 'late_tournament_games/late_tournament_game_events.csv.gz')


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


DEFAULT_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 127,
    'min_child_samples': 75,
    'learning_rate': 0.05,
    'verbose': -1,
}


def train_and_evaluate(df_unfiltered_train, df_logged_in_train,
                       df_unfiltered_all, df_logged_in_all, df_tournament,
                       features, verbose=True, params_override=None):
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

    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)

    num_boost_round = 500
    if params.get('learning_rate', 0.05) < 0.05:
        num_boost_round = 2000

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=verbose)]
    model = lgb.train(
        params, train_data, num_boost_round=num_boost_round,
        valid_sets=[val_data], callbacks=callbacks,
    )

    val_preds = model.predict(X_val)
    auc = sklearn.metrics.roc_auc_score(y_val, val_preds)
    log_loss = sklearn.metrics.log_loss(y_val, val_preds)

    # Tournament-anchored thresholds.
    # Note: logged-in pass rates are computed on training data (intentionally —
    # we want to see what fraction of logged-in games pass the quality bar,
    # not generalization error). Unfiltered uses separate eval shards.
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
    parser.add_argument('--param-sweep', action='store_true',
                        help='Sweep LightGBM hyperparameters one-at-a-time')
    args = parser.parse_args()

    if args.recompute and os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f'Deleted cache directory: {CACHE_DIR}')

    # === Data ===
    df_unfiltered = load_or_compute('unfiltered', UNFILTERED_PATH, args.recompute)
    df_unfiltered_eval = load_or_compute('unfiltered_eval', UNFILTERED_EVAL_PATH, args.recompute)
    df_logged_in = load_or_compute('logged_in', LOGGED_IN_PATH, args.recompute)
    df_tournament = load_or_compute('tournament', TOURNAMENT_PATH, args.recompute)

    print(f'\n=== Data ===')
    print(f'Unfiltered train: {len(df_unfiltered)}, Unfiltered eval: {len(df_unfiltered_eval)}, '
          f'Logged-in: {len(df_logged_in)}, Tournament: {len(df_tournament)}')

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

        # iloc[:n] gives deterministic subsets since shards are time-ordered
        # and strided across the hundreds digit for temporal coverage.
        for n in sizes:
            df_unf_n = df_unfiltered_clean.iloc[:n]
            df_log_n = df_logged_in_clean.iloc[:n]
            actual_unf = len(df_unf_n)
            actual_log = len(df_log_n)

            r = train_and_evaluate(
                df_unf_n, df_log_n,
                df_unfiltered_eval, df_logged_in, df_tournament,
                features, verbose=False,
            )
            print(f'{n:>7} {actual_unf:>7} {actual_log:>7} '
                  f'{r["auc"]:>7.4f} {r["log_loss"]:>8.4f} '
                  f'{r["unf_pass_99"]:>7.1%} {r["unf_pass_95"]:>7.1%} '
                  f'{r["log_pass_99"]:>7.1%} {r["log_pass_95"]:>7.1%}')
        return

    if args.param_sweep:
        # === Hyperparameter Sweep (one-at-a-time) ===
        DEFAULT_SIZE = 16000
        df_unf_train = df_unfiltered_clean.iloc[:DEFAULT_SIZE]
        df_log_train = df_logged_in_clean.iloc[:DEFAULT_SIZE]

        # LightGBM defaults for params not in DEFAULT_PARAMS
        lgb_defaults = {
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'reg_lambda': 0,
        }
        current_defaults = {**lgb_defaults, **DEFAULT_PARAMS}

        sweep_params = [
            # Tier 1
            ('num_leaves', [7, 15, 23, 31, 47, 63, 95, 127, 191, 255]),
            ('min_child_samples', [5, 10, 20, 35, 50, 75, 100, 150, 200, 300]),
            ('learning_rate', [0.01, 0.03, 0.05, 0.1]),
            # Tier 2
            ('feature_fraction', [0.5, 0.7, 0.9, 1.0]),
            ('bagging_fraction', [0.7, 0.8, 0.9, 1.0]),
            ('reg_lambda', [0, 1.0, 5.0, 10.0]),
        ]

        print(f'\n=== Param Sweep ({len(features)} features, {DEFAULT_SIZE} training per class) ===')
        print(f'{"Param":<22} {"Value":>8} {"AUC":>7} {"LogLoss":>8} {"Unf@99%":>8} {"Unf@95%":>8}')
        print('-' * 60)

        for param_name, values in sweep_params:
            for val in values:
                override = {param_name: val}
                if param_name == 'bagging_fraction' and val < 1.0:
                    override['bagging_freq'] = 1
                is_default = (val == current_defaults.get(param_name))
                r = train_and_evaluate(
                    df_unf_train, df_log_train,
                    df_unfiltered_eval, df_logged_in, df_tournament,
                    features, verbose=False, params_override=override,
                )
                marker = ' <- current' if is_default else ''
                print(f'{param_name:<22} {val:>8g} {r["auc"]:>7.4f} {r["log_loss"]:>8.4f} '
                      f'{r["unf_pass_99"]:>7.1%} {r["unf_pass_95"]:>7.1%}{marker}')
            print('---')
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
        df_unfiltered_eval, df_logged_in, df_tournament,
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
