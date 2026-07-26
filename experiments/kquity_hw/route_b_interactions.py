#!/usr/bin/env python3
"""KQuity Phase 2.5 / Route B: bounded interaction-term sweep.

Per ChatGPT Pro routing after Phase 2 found the hand-diff linear
primitive (6 features, 7 params, pRMSE 0.113) was the best linear
extraction.

Tighten with interaction terms over the same 6 differentials plus
a phase indicator, fit ridge / elastic-net targeting the oracle
logit. Hard constraints:

  CPU only
  <= 20 parameters total
  <= 12 selected terms/features
  one report, then freeze

Acceptance gate:

  pRMSE <= 0.085
  OR  pRMSE improves by >= 25% over 0.113 (i.e. <= 0.085)
  AND AUC, Brier, logloss do not regress vs hand_diff_linear_6
  AND params <= 20

If accept: use this primitive in Phase 4 (fixed-point).
Otherwise: freeze Route A (hand_diff_linear_6).

Candidate terms (per ChatGPT Pro):
  base (6):     egg_diff, food_diff, snail_pos,
                soldier_diff, warrior_diff, berries_norm
  interactions: egg_diff*food_diff
                egg_diff*snail_pos
                food_diff*snail_pos
                soldier_diff*warrior_diff
                soldier_diff*snail_pos
                warrior_diff*snail_pos
                berries_norm*food_diff
                berries_norm*egg_diff
                phase_norm*egg_diff
                phase_norm*food_diff
                phase_norm*snail_pos

phase_norm = per-game-normalized event timestamp in [0,1].

Selection: ridge first (all 17 terms) for the "ceiling" point,
then L1/elastic-net to drop terms down to <=12 selected.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
import fast_materialize as fm  # noqa: E402


# Feature indices in the 52-feature vector (matches inventory.md).
IDX_BLUE_EGGS = 0
IDX_BLUE_FOOD = 1
IDX_BLUE_NWAR = 2
IDX_BLUE_NSOL = 3
IDX_GOLD_EGGS = 20
IDX_GOLD_FOOD = 21
IDX_GOLD_NWAR = 22
IDX_GOLD_NSOL = 23
IDX_SNAIL = 49
IDX_BERRIES_NORM = 51


def materialize_csv_glob(csv_glob):
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(csv_glob)
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


def per_game_phase_norm(ts, gids):
    """Per-game normalized event timestamp in [0,1]. Vectorized
    via pandas groupby. For each game the min timestamp maps to 0
    and the max to 1; single-event games map to 0.
    """
    df = pd.DataFrame({"gid": gids, "ts": ts.astype(np.float64)})
    g = df.groupby("gid")["ts"]
    lo = g.transform("min").values
    hi = g.transform("max").values
    span = hi - lo
    out = np.where(span > 0, (df["ts"].values - lo) / np.maximum(span, 1e-9), 0.0)
    return out.astype(np.float32)


def build_features(X, ts, gids):
    """Return (n, 17) feature matrix + name list. Order:
    6 base differentials + 11 interactions."""
    egg = X[:, IDX_BLUE_EGGS] - X[:, IDX_GOLD_EGGS]
    food = X[:, IDX_BLUE_FOOD] - X[:, IDX_GOLD_FOOD]
    snail = X[:, IDX_SNAIL]
    sol = X[:, IDX_BLUE_NSOL] - X[:, IDX_GOLD_NSOL]
    war = X[:, IDX_BLUE_NWAR] - X[:, IDX_GOLD_NWAR]
    ber = X[:, IDX_BERRIES_NORM]
    phase = per_game_phase_norm(ts, gids)

    feats = [
        ("egg_diff", egg),
        ("food_diff", food),
        ("snail_pos", snail),
        ("soldier_diff", sol),
        ("warrior_diff", war),
        ("berries_avail_norm", ber),
        ("egg_x_food", egg * food),
        ("egg_x_snail", egg * snail),
        ("food_x_snail", food * snail),
        ("sol_x_war", sol * war),
        ("sol_x_snail", sol * snail),
        ("war_x_snail", war * snail),
        ("ber_x_food", ber * food),
        ("ber_x_egg", ber * egg),
        ("phase_x_egg", phase * egg),
        ("phase_x_food", phase * food),
        ("phase_x_snail", phase * snail),
    ]
    names = [n for n, _ in feats]
    F = np.column_stack([v for _, v in feats]).astype(np.float64)
    return F, names, phase


def game_split(game_ids, frac_train=0.8, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(game_ids)
    rng.shuffle(uniq)
    n_train = int(len(uniq) * frac_train)
    tr_games = set(uniq[:n_train].tolist())
    is_train = np.fromiter((g in tr_games for g in game_ids), dtype=bool)
    return is_train, ~is_train


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def metrics(p_pred, p_oracle, y):
    eps = 1e-12
    pp = np.clip(p_pred, eps, 1 - eps)
    prob_rmse = float(np.sqrt(((p_pred - p_oracle) ** 2).mean()))
    logit_rmse = float(np.sqrt(((logit(p_pred) - logit(p_oracle)) ** 2).mean()))
    logloss = float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
    brier = float(((pp - y) ** 2).mean())
    acc = float(((pp >= 0.5).astype(int) == y).mean())
    order = np.argsort(pp)
    ys = y[order]
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        auc = float("nan")
    else:
        ranks = np.arange(1, len(y) + 1)
        auc = float(((ranks * ys).sum() - npos * (npos + 1) / 2) / (npos * nneg))
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
        "prob_rmse": prob_rmse,
        "logit_rmse": logit_rmse,
        "logloss": logloss,
        "brier": brier,
        "auc": auc,
        "ece10": float(ece),
        "acc@0.5": acc,
    }


def fit_eval(model, F_tr, t_tr, F_te, p_or_te, y_te, name, names):
    model.fit(F_tr, t_tr)
    p_te = sigmoid(model.predict(F_te))
    m = metrics(p_te, p_or_te, y_te)
    coef = model.coef_.tolist()
    intercept = float(model.intercept_)
    nonzero = [(n, c) for n, c in zip(names, coef) if abs(c) > 1e-8]
    return {
        "name": name,
        "n_features_selected": len(nonzero),
        "n_params": len(nonzero) + 1,  # +1 for bias
        "intercept": intercept,
        "weights": {n: float(c) for n, c in zip(names, coef)},
        **m,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(REPO / "current_preferred_model.mdl"))
    ap.add_argument(
        "--csv-glob",
        default=str(REPO / "tests/benchmark_events_*.csv.gz"),
    )
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    booster = lgb.Booster(model_file=args.model)

    print("materializing eval frame ...")
    X, y, gids, ts = materialize_csv_glob(args.csv_glob)
    print(f"  X.shape={X.shape}, n_games={len(np.unique(gids))}")

    tr_mask, te_mask = game_split(gids, 0.8, seed=args.seed)
    n_tr = int(tr_mask.sum())
    n_te = int(te_mask.sum())
    n_g_tr = len(np.unique(gids[tr_mask]))
    n_g_te = len(np.unique(gids[te_mask]))
    print(f"  train: {n_tr} events / {n_g_tr} games  test: {n_te} / {n_g_te}")

    F_all, names, phase = build_features(X, ts, gids)
    F_tr, F_te = F_all[tr_mask], F_all[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]
    p_or_tr = booster.predict(X[tr_mask])
    p_or_te = booster.predict(X[te_mask])
    t_or_tr = logit(p_or_tr)

    # Reference: hand_diff_linear_6 (just the first 6 columns) ---------------
    ref = fit_eval(
        LinearRegression(),
        F_tr[:, :6],
        t_or_tr,
        F_te[:, :6],
        p_or_te,
        y_te,
        "hand_diff_linear_6 (reference)",
        names[:6],
    )

    runs = [ref]

    # All-17 ridge ceiling ----------------------------------------------------
    runs.append(
        fit_eval(
            Ridge(alpha=1.0),
            F_tr,
            t_or_tr,
            F_te,
            p_or_te,
            y_te,
            "ridge_all17",
            names,
        )
    )

    # All-17 linear (no regularization) — should be very close to ridge ------
    runs.append(
        fit_eval(
            LinearRegression(),
            F_tr,
            t_or_tr,
            F_te,
            p_or_te,
            y_te,
            "linear_all17",
            names,
        )
    )

    # L1 selection sweep — find smallest alpha that keeps <=12 nonzero -------
    print("\nL1 selection sweep (target <=12 selected terms):")
    best_l1 = None
    for alpha in [0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
        r = fit_eval(
            Lasso(alpha=alpha, max_iter=20000),
            F_tr,
            t_or_tr,
            F_te,
            p_or_te,
            y_te,
            f"lasso_a{alpha}",
            names,
        )
        sel = r["n_features_selected"]
        print(
            f"  alpha={alpha:<6} nnz={sel:>2d} params={r['n_params']:>2d} "
            f"pRMSE={r['prob_rmse']:.4f} AUC={r['auc']:.4f}"
        )
        runs.append(r)
        if sel <= 12 and (best_l1 is None or r["prob_rmse"] < best_l1["prob_rmse"]):
            best_l1 = r

    # ElasticNet at a couple of operating points -----------------------------
    for alpha, l1r in [(0.01, 0.5), (0.005, 0.5), (0.02, 0.5)]:
        r = fit_eval(
            ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=20000),
            F_tr,
            t_or_tr,
            F_te,
            p_or_te,
            y_te,
            f"elasticnet_a{alpha}_l1r{l1r}",
            names,
        )
        sel = r["n_features_selected"]
        print(
            f"  ElasticNet alpha={alpha} l1r={l1r} nnz={sel} "
            f"pRMSE={r['prob_rmse']:.4f}"
        )
        runs.append(r)

    # Summary -----------------------------------------------------------------
    print("\n=== results (sorted by prob_rmse) ===")
    runs_sorted = sorted(runs, key=lambda r: r["prob_rmse"])
    print(
        f"{'name':36s} {'sel':>4s} {'par':>4s} "
        f"{'pRMSE':>7s} {'lRMSE':>7s} {'AUC':>7s} "
        f"{'logloss':>8s} {'Brier':>7s} {'ECE':>7s}"
    )
    for r in runs_sorted:
        print(
            f"{r['name']:36s} {r['n_features_selected']:>4d} {r['n_params']:>4d} "
            f"{r['prob_rmse']:>7.4f} {r['logit_rmse']:>7.4f} {r['auc']:>7.4f} "
            f"{r['logloss']:>8.4f} {r['brier']:>7.4f} {r['ece10']:>7.4f}"
        )

    # Gate verdict ------------------------------------------------------------
    ref_pRMSE = ref["prob_rmse"]
    ref_AUC = ref["auc"]
    ref_brier = ref["brier"]
    ref_logloss = ref["logloss"]
    gate_abs = 0.085
    gate_rel = ref_pRMSE * 0.75  # 25% improvement

    accept_candidates = [
        r
        for r in runs
        if r["name"] != ref["name"]
        and r["n_params"] <= 20
        and r["n_features_selected"] <= 12
        and r["prob_rmse"] <= min(gate_abs, gate_rel)
        and r["auc"] >= ref_AUC
        and r["brier"] <= ref_brier
        and r["logloss"] <= ref_logloss
    ]
    print(f"\nreference: {ref['name']}  pRMSE={ref_pRMSE:.4f} AUC={ref_AUC:.4f}")
    print(
        f"acceptance gate: pRMSE <= {min(gate_abs, gate_rel):.4f} "
        f"AND AUC>={ref_AUC:.4f} AND Brier<={ref_brier:.4f} "
        f"AND logloss<={ref_logloss:.4f} AND params<=20 AND nfeat<=12"
    )
    if accept_candidates:
        chosen = min(accept_candidates, key=lambda r: r["prob_rmse"])
        print(f"\nROUTE B ACCEPTED: {chosen['name']}")
        print(f"  selected features ({chosen['n_features_selected']}):")
        for n, w in chosen["weights"].items():
            if abs(w) > 1e-8:
                print(f"    {n:24s}  w = {w:+.4f}")
        print(f"  intercept: {chosen['intercept']:+.4f}")
        verdict = "accept"
    else:
        chosen = ref
        print("\nROUTE B NOT ACCEPTED -- freezing Route A (hand_diff_linear_6).")
        verdict = "reject"

    out = {
        "seed": args.seed,
        "n_train_events": n_tr,
        "n_test_events": n_te,
        "reference": ref,
        "all_runs": runs,
        "gate": {
            "absolute": gate_abs,
            "relative_target": gate_rel,
            "verdict": verdict,
        },
        "chosen": chosen,
        "feature_names": names,
    }
    (out_dir / "route_b_interactions.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_dir / 'route_b_interactions.json'}")


if __name__ == "__main__":
    main()
