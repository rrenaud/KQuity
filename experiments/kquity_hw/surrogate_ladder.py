#!/usr/bin/env python3
"""KQuity Phase 2 surrogate ladder: distill the LightGBM win-probability
oracle into progressively simpler primitives.

For each surrogate we report:
  fidelity vs oracle: prob RMSE, logit RMSE, KL, top-k disagreement
  fidelity vs labels: log loss, Brier, AUC, ECE10, acc@0.5
  size: feature count, parameter count (rough hardware cost)

Surrogate ladder (per ChatGPT Pro):
  1. linear regression on all 52 features (distillation upper bound)
  2. L1 logistic regression on all 52 (sparse linear, drops features)
  3. ridge regression on top-K features by gain
  4. hand-hypothesis linear on 5-6 differentials
     (egg_diff, food_diff, snail_pos, soldier_diff, warrior_diff)
  5. depth-2/3/4/5 decision tree regressor on top-10 features
  6. phase-conditioned linear (early/mid/late split, separate weights)
  7. small binned LUT over (egg_diff, food_diff, snail_pos)

Promotion gates (per ChatGPT Pro):
  Strong : prob RMSE <= 0.03, AUC drop <= 2pp, features <= 8-10
  Good   : prob RMSE <= 0.05, AUC drop <= 3-5pp
  Fail   : shallow surrogates need too many features or lose
           calibration badly.

Train/test split is by game_id (80/20). Targets the oracle's logit.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
import fast_materialize as fm  # noqa: E402


# --- Feature index map (matches inventory.md) -------------------------------

FEATURE_NAMES = (
    [
        f"blue.{n}"
        for n in (
            "eggs",
            "food_count",
            "n_warriors",
            "n_soldiers",
            "w0_is_bot",
            "w0_has_food",
            "w0_has_speed",
            "w0_has_wings",
            "w1_is_bot",
            "w1_has_food",
            "w1_has_speed",
            "w1_has_wings",
            "w2_is_bot",
            "w2_has_food",
            "w2_has_speed",
            "w2_has_wings",
            "w3_is_bot",
            "w3_has_food",
            "w3_has_speed",
            "w3_has_wings",
        )
    ]
    + [
        f"gold.{n}"
        for n in (
            "eggs",
            "food_count",
            "n_warriors",
            "n_soldiers",
            "w0_is_bot",
            "w0_has_food",
            "w0_has_speed",
            "w0_has_wings",
            "w1_is_bot",
            "w1_has_food",
            "w1_has_speed",
            "w1_has_wings",
            "w2_is_bot",
            "w2_has_food",
            "w2_has_speed",
            "w2_has_wings",
            "w3_is_bot",
            "w3_has_food",
            "w3_has_speed",
            "w3_has_wings",
        )
    ]
    + [f"maiden_{i}" for i in range(5)]
    + [f"map_{n}" for n in ("day", "night", "dusk", "twilight")]
    + ["snail_pos", "snail_spd", "berries_avail_norm"]
)
assert len(FEATURE_NAMES) == 52


# Indices for quick reference (column numbers in X).
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def differentials(X):
    """Return a (n, 6) matrix of hand-engineered differentials and the
    canonical phase feature."""
    egg_diff = X[:, IDX["blue.eggs"]] - X[:, IDX["gold.eggs"]]
    food_diff = X[:, IDX["blue.food_count"]] - X[:, IDX["gold.food_count"]]
    snail = X[:, IDX["snail_pos"]]
    soldier_diff = X[:, IDX["blue.n_soldiers"]] - X[:, IDX["gold.n_soldiers"]]
    warrior_diff = X[:, IDX["blue.n_warriors"]] - X[:, IDX["gold.n_warriors"]]
    berries_norm = X[:, IDX["berries_avail_norm"]]
    return np.column_stack(
        [egg_diff, food_diff, snail, soldier_diff, warrior_diff, berries_norm]
    )


DIFF_NAMES = [
    "egg_diff",
    "food_diff",
    "snail_pos",
    "soldier_diff",
    "warrior_diff",
    "berries_avail_norm",
]


# --- Data loading -----------------------------------------------------------


def materialize_csv_glob(csv_glob):
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"no shards match {csv_glob}")
    Xs, ys, gids, ts = [], [], [], []
    for p in paths:
        res = fm.fast_materialize(p, drop_state_probability=0.0)
        Xs.append(res[0])
        ys.append(res[1])
        gids.append(res[2])
        ts.append(res[3])
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(gids),
        np.concatenate(ts),
    )


def game_split(game_ids, frac_train=0.8, seed=0):
    """Returns (train_mask, test_mask) by held-out game_id."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(game_ids)
    rng.shuffle(uniq)
    n_train = int(len(uniq) * frac_train)
    train_games = set(uniq[:n_train].tolist())
    is_train = np.fromiter((g in train_games for g in game_ids), dtype=bool)
    return is_train, ~is_train


# --- Metrics ----------------------------------------------------------------


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def metrics(p_pred, oracle_p, y, name):
    """Fidelity vs oracle and labels for a single surrogate."""
    eps = 1e-12
    pp = np.clip(p_pred, eps, 1 - eps)
    # vs oracle
    prob_rmse = float(np.sqrt(((p_pred - oracle_p) ** 2).mean()))
    logit_rmse = float(np.sqrt(((logit(p_pred) - logit(oracle_p)) ** 2).mean()))
    # KL(oracle || surrogate) = sum oracle * log(oracle/surrogate)
    o = np.clip(oracle_p, eps, 1 - eps)
    kl = float(
        (
            o * (np.log(o) - np.log(pp)) + (1 - o) * (np.log(1 - o) - np.log(1 - pp))
        ).mean()
    )
    # top-k disagreement: oracle confident (>0.8 or <0.2), surrogate not on same side
    conf_mask = (oracle_p > 0.8) | (oracle_p < 0.2)
    if conf_mask.any():
        oracle_class = (oracle_p[conf_mask] > 0.5).astype(int)
        surrogate_class = (p_pred[conf_mask] > 0.5).astype(int)
        topk_disagree = float((oracle_class != surrogate_class).mean())
    else:
        topk_disagree = float("nan")

    # vs labels
    logloss = float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
    brier = float(((pp - y) ** 2).mean())
    acc = float(((pp >= 0.5).astype(int) == y).mean())
    # AUC
    order = np.argsort(pp)
    y_s = y[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        ranks = np.arange(1, len(y) + 1)
        auc = float(((ranks * y_s).sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    # ECE 10
    edges = np.linspace(0, 1, 11)
    n = len(pp)
    ece = 0.0
    for i in range(10):
        if i == 9:
            mask = (pp >= edges[i]) & (pp <= edges[i + 1])
        else:
            mask = (pp >= edges[i]) & (pp < edges[i + 1])
        m = int(mask.sum())
        if m == 0:
            continue
        ece += (m / n) * abs(pp[mask].mean() - y[mask].mean())

    return {
        "name": name,
        "prob_rmse": prob_rmse,
        "logit_rmse": logit_rmse,
        "kl_oracle_surrogate": kl,
        "top_k_disagreement": topk_disagree,
        "logloss": logloss,
        "brier": brier,
        "auc": auc,
        "ece10": float(ece),
        "acc@0.5": acc,
    }


# --- Surrogates -------------------------------------------------------------


def fit_linear_logit(X_tr, t_tr, X_te):
    """Linear regression on oracle logit. Returns surrogate probabilities."""
    m = LinearRegression()
    m.fit(X_tr, t_tr)
    return sigmoid(m.predict(X_te)), m


def fit_ridge_logit(X_tr, t_tr, X_te, alpha=1.0):
    """Ridge regression on oracle logit."""
    from sklearn.linear_model import Ridge

    m = Ridge(alpha=alpha)
    m.fit(X_tr, t_tr)
    return sigmoid(m.predict(X_te)), m


def fit_l1_logistic(X_tr, y_tr, X_te, C=0.1):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=200)
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    n_nonzero = int(np.sum(m.coef_.ravel() != 0))
    return p, m, n_nonzero


def fit_tree_logit(X_tr, t_tr, X_te, max_depth):
    m = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    m.fit(X_tr, t_tr)
    return sigmoid(m.predict(X_te)), m


def fit_phase_linear(diff_tr, t_tr, ts_tr, diff_te, ts_te, phase_edges=(0.25, 0.6)):
    """Three independent linear regressors over event-time phase."""
    # phase labels via global quantiles of training timestamps
    q1 = float(np.quantile(ts_tr, phase_edges[0]))
    q2 = float(np.quantile(ts_tr, phase_edges[1]))

    def phase_of(ts):
        ph = np.zeros(len(ts), dtype=np.int8)
        ph[(ts >= q1) & (ts < q2)] = 1
        ph[ts >= q2] = 2
        return ph

    ph_tr = phase_of(ts_tr)
    ph_te = phase_of(ts_te)
    models = []
    pred_te = np.zeros(len(ts_te))
    for ph in (0, 1, 2):
        m_tr = ph_tr == ph
        m_te = ph_te == ph
        if not m_tr.any():
            continue
        lr = LinearRegression()
        lr.fit(diff_tr[m_tr], t_tr[m_tr])
        models.append(lr)
        if m_te.any():
            pred_te[m_te] = lr.predict(diff_te[m_te])
    return sigmoid(pred_te), models, (q1, q2)


def fit_binned_lut(diff_tr, oracle_p_tr, diff_te, n_bins=(5, 7, 5)):
    """Bin (egg_diff, food_diff, snail_pos) into nb x nb x nb cells; mean
    oracle prob per cell. Smooth empty cells with a global prior.
    """
    # Use first three differentials: egg_diff, food_diff, snail_pos
    keys = diff_tr[:, :3]
    egg_edges = np.linspace(keys[:, 0].min(), keys[:, 0].max(), n_bins[0] + 1)
    food_edges = np.linspace(keys[:, 1].min(), keys[:, 1].max(), n_bins[1] + 1)
    snail_edges = np.linspace(keys[:, 2].min(), keys[:, 2].max(), n_bins[2] + 1)

    def bin_idx(x, edges):
        idx = np.searchsorted(edges, x, side="right") - 1
        return np.clip(idx, 0, len(edges) - 2)

    tr_b = (
        bin_idx(keys[:, 0], egg_edges),
        bin_idx(keys[:, 1], food_edges),
        bin_idx(keys[:, 2], snail_edges),
    )
    lut = np.full(n_bins, oracle_p_tr.mean(), dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(len(oracle_p_tr)):
        lut[tr_b[0][i], tr_b[1][i], tr_b[2][i]] = 0.0  # reset to accumulate
        counts[tr_b[0][i], tr_b[1][i], tr_b[2][i]] = 0  # reset to accumulate
    # Vectorize the accumulation
    flat_idx = tr_b[0] * (n_bins[1] * n_bins[2]) + tr_b[1] * n_bins[2] + tr_b[2]
    sums = np.bincount(flat_idx, weights=oracle_p_tr, minlength=np.prod(n_bins))
    cnts = np.bincount(flat_idx, minlength=np.prod(n_bins))
    prior = oracle_p_tr.mean()
    lut_flat = np.where(cnts > 0, sums / np.maximum(cnts, 1), prior)
    lut = lut_flat.reshape(n_bins).astype(np.float32)

    # Apply
    te_keys = diff_te[:, :3]
    te_b0 = bin_idx(te_keys[:, 0], egg_edges)
    te_b1 = bin_idx(te_keys[:, 1], food_edges)
    te_b2 = bin_idx(te_keys[:, 2], snail_edges)
    p_te = lut[te_b0, te_b1, te_b2]
    return p_te, {
        "lut": lut,
        "egg_edges": egg_edges,
        "food_edges": food_edges,
        "snail_edges": snail_edges,
        "n_bins": n_bins,
    }


# --- Main ladder ------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=str(REPO / "current_preferred_model.mdl"),
    )
    ap.add_argument(
        "--csv-glob",
        default=str(REPO / "tests/benchmark_events_*.csv.gz"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(HERE / "results"),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-k-features", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    booster = lgb.Booster(model_file=args.model)

    print(f"materializing eval frame ...")
    X, y, gids, ts = materialize_csv_glob(args.csv_glob)
    print(f"  X.shape={X.shape}, n_games={len(np.unique(gids))}")

    # game-id split
    tr_mask, te_mask = game_split(gids, frac_train=0.8, seed=args.seed)
    n_tr = int(tr_mask.sum())
    n_te = int(te_mask.sum())
    n_g_tr = len(np.unique(gids[tr_mask]))
    n_g_te = len(np.unique(gids[te_mask]))
    print(
        f"  train: {n_tr} events from {n_g_tr} games  |  "
        f"test: {n_te} events from {n_g_te} games"
    )

    X_tr, X_te = X[tr_mask], X[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]
    ts_tr, ts_te = ts[tr_mask], ts[te_mask]

    # Oracle predictions
    p_or_tr = booster.predict(X_tr)
    p_or_te = booster.predict(X_te)
    t_or_tr = logit(p_or_tr)
    t_or_te = logit(p_or_te)

    # Differentials
    diff_tr = differentials(X_tr)
    diff_te = differentials(X_te)

    # Top-K features by gain
    gains = booster.feature_importance(importance_type="gain")
    top_idx = np.argsort(-gains)[: args.top_k_features].tolist()
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    print(f"  top-{args.top_k_features} feature idx: {top_idx}")
    print(f"  top-{args.top_k_features} names: {top_names}")
    X_tr_top = X_tr[:, top_idx]
    X_te_top = X_te[:, top_idx]

    results = []

    # Surrogate 0: oracle itself (as fidelity reference)
    results.append(
        {
            **metrics(p_or_te, p_or_te, y_te, "oracle (reference)"),
            "n_features": 52,
            "n_params_rough": 100 * 100,  # 100 trees x 100 leaves
        }
    )

    # 1. Linear regression on all 52
    p_te, m = fit_linear_logit(X_tr, t_or_tr, X_te)
    results.append(
        {
            **metrics(p_te, p_or_te, y_te, "linear_all52"),
            "n_features": 52,
            "n_params_rough": 52 + 1,
        }
    )

    # 2. L1 logistic regression on all 52 (targets labels, so a different
    #    objective than the others; still useful as a feature-selection probe)
    p_te, m, nnz = fit_l1_logistic(X_tr, y_tr, X_te, C=0.05)
    results.append(
        {
            **metrics(p_te, p_or_te, y_te, "l1_logistic_all52"),
            "n_features": int(nnz),
            "n_params_rough": int(nnz) + 1,
            "C": 0.05,
        }
    )

    # 3. Ridge on top-K features
    p_te, m = fit_ridge_logit(X_tr_top, t_or_tr, X_te_top, alpha=1.0)
    results.append(
        {
            **metrics(p_te, p_or_te, y_te, f"ridge_top{args.top_k_features}"),
            "n_features": args.top_k_features,
            "n_params_rough": args.top_k_features + 1,
        }
    )

    # 4. Hand-hypothesis linear on 6 differentials
    p_te, m = fit_linear_logit(diff_tr, t_or_tr, diff_te)
    results.append(
        {
            **metrics(p_te, p_or_te, y_te, "hand_diff_linear_6"),
            "n_features": 6,
            "n_params_rough": 6 + 1,
            "weights": {n: float(w) for n, w in zip(DIFF_NAMES, m.coef_.tolist())},
            "intercept": float(m.intercept_),
        }
    )

    # 5. Decision trees over the 6 differentials
    for depth in (2, 3, 4, 5):
        p_te, m = fit_tree_logit(diff_tr, t_or_tr, diff_te, max_depth=depth)
        n_leaves = m.get_n_leaves()
        results.append(
            {
                **metrics(p_te, p_or_te, y_te, f"tree_d{depth}_diff6"),
                "n_features": 6,
                "n_params_rough": int(2 * n_leaves - 1),  # internal nodes + leaves
                "n_leaves": int(n_leaves),
                "max_depth": depth,
            }
        )

    # 5b. Decision trees over top-K raw features
    for depth in (3, 4, 5):
        p_te, m = fit_tree_logit(X_tr_top, t_or_tr, X_te_top, max_depth=depth)
        n_leaves = m.get_n_leaves()
        results.append(
            {
                **metrics(
                    p_te, p_or_te, y_te, f"tree_d{depth}_top{args.top_k_features}"
                ),
                "n_features": args.top_k_features,
                "n_params_rough": int(2 * n_leaves - 1),
                "n_leaves": int(n_leaves),
                "max_depth": depth,
            }
        )

    # 6. Phase-conditioned linear (early/mid/late) over differentials
    p_te, models, (q1, q2) = fit_phase_linear(diff_tr, t_or_tr, ts_tr, diff_te, ts_te)
    results.append(
        {
            **metrics(p_te, p_or_te, y_te, "phase_linear_diff6"),
            "n_features": 6,
            "n_params_rough": 3 * (6 + 1),
            "phase_edges": [q1, q2],
            "per_phase_weights": [
                {n: float(w) for n, w in zip(DIFF_NAMES, mdl.coef_.tolist())}
                for mdl in models
            ],
            "per_phase_intercept": [float(mdl.intercept_) for mdl in models],
        }
    )

    # 7. Binned LUT over (egg_diff, food_diff, snail_pos)
    for n_bins in [(5, 5, 5), (7, 7, 7), (9, 9, 9)]:
        p_te, info = fit_binned_lut(diff_tr, p_or_tr, diff_te, n_bins=n_bins)
        results.append(
            {
                **metrics(
                    p_te, p_or_te, y_te, f"lut_{n_bins[0]}x{n_bins[1]}x{n_bins[2]}"
                ),
                "n_features": 3,
                "n_params_rough": int(np.prod(n_bins)),
                "n_bins": list(n_bins),
            }
        )

    # Print summary table
    print("\n=== surrogate ladder (sorted by prob RMSE) ===\n")
    sorted_results = sorted(
        results,
        key=lambda r: (
            r.get("prob_rmse", 0.0) if r["name"] != "oracle (reference)" else -1
        ),
    )
    print(
        f"{'name':30s} {'feat':>5s} {'params':>7s} "
        f"{'pRMSE':>8s} {'lRMSE':>8s} {'KL':>8s} "
        f"{'AUC':>7s} {'logloss':>8s} {'Brier':>7s} "
        f"{'ECE':>7s} {'disag':>7s}"
    )
    for r in sorted_results:
        print(
            f"{r['name']:30s} {r['n_features']:>5d} "
            f"{r['n_params_rough']:>7d} "
            f"{r['prob_rmse']:>8.4f} {r['logit_rmse']:>8.4f} "
            f"{r['kl_oracle_surrogate']:>8.4f} {r['auc']:>7.4f} "
            f"{r['logloss']:>8.4f} {r['brier']:>7.4f} "
            f"{r['ece10']:>7.4f} {r['top_k_disagreement']:>7.4f}"
        )

    (out_dir / "surrogate_ladder.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "top_k_features": args.top_k_features,
                "top_k_names": top_names,
                "n_train_events": n_tr,
                "n_test_events": n_te,
                "n_train_games": n_g_tr,
                "n_test_games": n_g_te,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nsaved {out_dir / 'surrogate_ladder.json'}")


if __name__ == "__main__":
    main()
