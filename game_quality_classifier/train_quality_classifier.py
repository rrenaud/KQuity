#!/usr/bin/env python3
"""Train a LightGBM classifier to distinguish high-quality competitive games
from junk/practice/button-check games.

Labels: logged_in=1 (high quality), unfiltered=0 (mixed quality).
Validation: tournament games should almost all score as high quality.

Usage (from repo root):
    python -m game_quality_classifier.train_quality_classifier                  # single run, saves model
    python -m game_quality_classifier.train_quality_classifier --recompute      # recompute features from CSVs
    python -m game_quality_classifier.train_quality_classifier --sweep          # data size sweep 1K-32K
    python -m game_quality_classifier.train_quality_classifier --param-sweep    # hyperparameter sweep
    python -m game_quality_classifier.train_quality_classifier --self-distill   # self-distillation + ensemble
"""

import argparse
import json
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
MODEL_PATH = os.path.join(CACHE_DIR, 'quality_model.mdl')
THRESHOLD_PATH = os.path.join(CACHE_DIR, 'threshold.json')

# Unfiltered shards are time-ordered. Stride over the hundreds digit for
# temporal coverage rather than taking contiguous shards from one time period.
# 20 train shards: X00, X01 for X=0-9. 20 eval shards: X02, X03 for X=0-9.
UNFILTERED_PATH = os.path.join(REPO_ROOT, 'unfiltered_partitioned/gameevents_[0-9]0[01].csv.gz')
UNFILTERED_EVAL_PATH = os.path.join(REPO_ROOT, 'unfiltered_partitioned/gameevents_[0-9]0[23].csv.gz')
# Logged-in shards are sorted by login count (highest quality first).
LOGGED_IN_PATH = os.path.join(REPO_ROOT, 'logged_in_games/gameevents_0[0-3][0-9].csv.gz')
TOURNAMENT_PATH = os.path.join(REPO_ROOT, 'late_tournament_games/late_tournament_game_events.csv.gz')
DEFAULT_SIZE = 16000  # Max training examples per class


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
                       features, verbose=True, params_override=None,
                       random_state=42, bag_fraction=1.0):
    """Train model and return metrics dict.

    Args:
        bag_fraction: Fraction of training data to bootstrap sample
            (with replacement) before training. 1.0 = no bagging.
    """
    X_unf_train = df_unfiltered_train[features].values
    X_log_train = df_logged_in_train[features].values
    X = np.vstack([X_unf_train, X_log_train])
    y = np.concatenate([np.zeros(len(X_unf_train)), np.ones(len(X_log_train))])

    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # External bagging: bootstrap sample from training data.
    # Use a derived seed so bagging is independent of the train/val split.
    if bag_fraction < 1.0:
        rng = np.random.RandomState(random_state + 1_000_000)
        n_sample = int(len(X_train) * bag_fraction)
        bag_idx = rng.choice(len(X_train), size=n_sample, replace=True)
        X_train = X_train[bag_idx]
        y_train = y_train[bag_idx]

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
    # Logged-in pass rates use the full logged-in set (intentionally — we want
    # to see what fraction passes the quality bar, not generalization error).
    # Unfiltered pass rates use separate eval shards to avoid train leakage.
    X_tournament = df_tournament[features].values
    X_unf_all = df_unfiltered_all[features].values
    X_log_all = df_logged_in_all[features].values
    tournament_scores = model.predict(X_tournament)
    unfiltered_scores = model.predict(X_unf_all)
    logged_in_scores = model.predict(X_log_all)

    results = {
        'auc': auc, 'log_loss': log_loss,
        'model': model, 'importance': model.feature_importance(importance_type='gain'),
        'tournament_scores': tournament_scores,
        'unfiltered_scores': unfiltered_scores,
        'logged_in_scores': logged_in_scores,
    }
    for recall_target in [0.99, 0.95]:
        threshold = np.percentile(tournament_scores, 100 * (1 - recall_target))
        pct = int(recall_target * 100)
        results[f'unf_pass_{pct}'] = (unfiltered_scores >= threshold).mean()
        results[f'log_pass_{pct}'] = (logged_in_scores >= threshold).mean()
        results[f'threshold_{pct}'] = threshold

    return results


def run_self_distill(df_unfiltered_clean, df_unfiltered_eval, df_logged_in,
                     df_logged_in_clean, df_tournament, features):
    """Self-distillation pipeline: curate positives, train bagged ensemble.

    Uses the existing saved model to prune low-quality >=9-login games and
    cherry-pick high-scoring 8-login games, then trains an ensemble of bagged
    models. See Experiments 8-10 in rating_based_quality_experiments.md.
    """
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Self-distillation requires an existing model at {MODEL_PATH}')
        print('Run a single training first: python -m game_quality_classifier.train_quality_classifier')
        return

    from compute_ratings import load_usergame
    usergame = load_usergame(
        os.path.join(REPO_ROOT, 'unfiltered_partitioned/usergame.csv'))

    # Base: logged-in games with >=9 logins (Experiment 5: sweet spot threshold)
    game_ids = df_logged_in_clean['game_id'].values
    keep_9 = np.array([len(usergame.get(gid, {})) >= 9 for gid in game_ids])
    df_base = df_logged_in_clean[keep_9].reset_index(drop=True)
    print(f'\n=== Base: {len(df_base)} games with >=9 logins ===')

    # Score base games with existing quality model, remove bottom 10%
    # (Experiment 8: pruning mislabeled positives)
    existing_model = lgb.Booster(model_file=MODEL_PATH)
    base_scores = existing_model.predict(df_base[features].values)
    p10_threshold = np.percentile(base_scores, 10)
    keep_top90 = base_scores >= p10_threshold
    df_base_pruned = df_base[keep_top90].reset_index(drop=True)
    print(f'Quality scores on >=9 base: mean={base_scores.mean():.4f}, '
          f'p10={p10_threshold:.4f}, p50={np.percentile(base_scores, 50):.4f}')
    print(f'After removing bottom 10%: {len(df_base_pruned)} games '
          f'(removed {(~keep_top90).sum()})')

    # Get 8-login games, score them, sort by score desc
    # (Experiment 9: +2000 is the sweet spot before quality degrades)
    keep_8 = np.array([len(usergame.get(gid, {})) == 8 for gid in game_ids])
    df_login8 = df_logged_in_clean[keep_8].reset_index(drop=True)
    login8_scores = existing_model.predict(df_login8[features].values)
    sort_idx = np.argsort(-login8_scores)
    df_login8 = df_login8.iloc[sort_idx].reset_index(drop=True)
    login8_scores = login8_scores[sort_idx]
    print(f'\n=== 8-login games: {len(df_login8)} total ===')
    print(f'Quality scores: mean={login8_scores.mean():.4f}, '
          f'p10={np.percentile(login8_scores, 10):.4f}, '
          f'p50={np.percentile(login8_scores, 50):.4f}, '
          f'p90={np.percentile(login8_scores, 90):.4f}')

    # Best positive set: pruned >=9 + top 2000 login8
    n_login8_add = 2000
    df_best = pd.concat([df_base_pruned, df_login8.iloc[:n_login8_add]], ignore_index=True)
    print(f'\nPositive set: {len(df_best)} games (pruned >=9 + top {n_login8_add} login8)')

    # Train N models with bagging, collect per-seed and ensemble results
    # (Experiment 10: bagged ensemble beats every individual model)
    n_seeds = 10
    all_results = {m: [] for m in ['auc', 'log_loss', 'unf_pass_99', 'unf_pass_95']}
    all_tournament_scores = []
    all_unfiltered_scores = []
    all_logged_in_scores = []

    print(f'\n=== Training {n_seeds} bagged models ===')
    print(f'{"Seed":>5} {"AUC":>7} {"LogLoss":>8} {"Unf@99%":>8} {"Unf@95%":>8}')
    print('-' * 42)
    for seed in range(n_seeds):
        r = train_and_evaluate(
            df_unfiltered_clean.iloc[:DEFAULT_SIZE], df_best,
            df_unfiltered_eval, df_logged_in, df_tournament,
            features, verbose=False, random_state=seed,
            bag_fraction=0.8,
        )
        for m in ['auc', 'log_loss', 'unf_pass_99', 'unf_pass_95']:
            all_results[m].append(r[m])
        all_tournament_scores.append(r['tournament_scores'])
        all_unfiltered_scores.append(r['unfiltered_scores'])
        all_logged_in_scores.append(r['logged_in_scores'])
        print(f'{seed:>5} {r["auc"]:>7.4f} {r["log_loss"]:>8.4f} '
              f'{r["unf_pass_99"]:>7.1%} {r["unf_pass_95"]:>7.1%}')

    # Individual model summary
    print(f'\n=== Individual models (mean +/- std over {n_seeds} seeds) ===')
    for m in ['auc', 'log_loss', 'unf_pass_99', 'unf_pass_95']:
        vals = np.array(all_results[m])
        print(f'  {m:<15} {vals.mean():.4f} +/- {vals.std():.4f}')

    # Ensemble: average all N models' predictions
    ens_tournament = np.mean(all_tournament_scores, axis=0)
    ens_unfiltered = np.mean(all_unfiltered_scores, axis=0)
    ens_logged_in = np.mean(all_logged_in_scores, axis=0)

    print(f'\n=== Ensemble of {n_seeds} bagged models ===')
    for recall_target in [0.99, 0.95]:
        threshold = np.percentile(ens_tournament, 100 * (1 - recall_target))
        pct = int(recall_target * 100)
        unf_pass = (ens_unfiltered >= threshold).mean()
        log_pass = (ens_logged_in >= threshold).mean()
        print(f'  @{pct}% recall: threshold={threshold:.4f}, '
              f'unf_pass={unf_pass:.1%}, log_pass={log_pass:.1%}')

    # Plot: individual models + ensemble point
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ens_metrics = {}
    for recall_target in [0.99, 0.95]:
        threshold = np.percentile(ens_tournament, 100 * (1 - recall_target))
        pct = int(recall_target * 100)
        ens_metrics[f'unf_pass_{pct}'] = (ens_unfiltered >= threshold).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pct, label in zip(axes, [99, 95], ['Unf pass @99%', 'Unf pass @95%']):
        metric = f'unf_pass_{pct}'
        vals = np.array(all_results[metric]) * 100
        ens_val = ens_metrics[metric] * 100
        ax.scatter(range(n_seeds), vals, color='#3498db', s=60, zorder=3,
                   label=f'Individual models (mean={vals.mean():.1f}%)')
        ax.axhline(vals.mean(), color='#3498db', linestyle='--', alpha=0.5)
        ax.axhline(ens_val, color='#e74c3c', linewidth=2,
                   label=f'Ensemble of {n_seeds} ({ens_val:.1f}%)')
        ax.set_xlabel('Seed')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'Bagged Ensemble: {n_seeds} models, bag=0.8 ({len(df_best)} positives)', fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(REPO_ROOT, 'model_experiments', 'quality_positive_sweep.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'\nPlot saved to {plot_path}')


def main():
    parser = argparse.ArgumentParser(description='Train game quality classifier')
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute features from CSVs (delete cache)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run data size sweep from 1K to 32K')
    parser.add_argument('--param-sweep', action='store_true',
                        help='Sweep LightGBM hyperparameters one-at-a-time')
    parser.add_argument('--self-distill', action='store_true',
                        help='Self-distillation: prune >=9-login positives with existing model, '
                             'add top-scoring 8-login games, train bagged ensemble')
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

    if args.self_distill:
        run_self_distill(
            df_unfiltered_clean, df_unfiltered_eval, df_logged_in,
            df_logged_in_clean, df_tournament, features,
        )
        return

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

    # Save model and threshold for downstream use
    r['model'].save_model(MODEL_PATH)
    print(f'\nSaved model to {MODEL_PATH}')

    with open(THRESHOLD_PATH, 'w') as f:
        json.dump({'threshold_99': r['threshold_99'], 'threshold_95': r['threshold_95']}, f)
    print(f'Saved thresholds to {THRESHOLD_PATH}')


if __name__ == '__main__':
    main()
