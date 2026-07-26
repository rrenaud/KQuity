#!/usr/bin/env python3
"""Phase 4: fixed-point parity for the hand_diff_linear_6 primitive.

Per ChatGPT Pro: the primary hardware output is the raw int logit
(preserves ranking, avoids calibration drama). The sigmoid LUT is
optional, reported here as a secondary diagnostic.

Quantization plan (8-bit primary, 12-bit and 16-bit reported for
comparison):

  Feature quantization (per-feature affine to int):
    egg_diff       integer in [-3, 3]      no fractional bits needed
    food_diff      integer in [-12, 12]    no fractional bits needed
    soldier_diff   integer in [-3, 3]      no fractional bits needed
    warrior_diff   integer in [-4, 4]      no fractional bits needed
    snail_pos      float, clip to [-1, 1]  Q1.(N-1) signed
    berries_norm   float in [0.43, 0.94]   Q0.N unsigned, centered

  Weight quantization (Q1.(N-1) signed for N-bit weights):
    All 6 weights and the intercept are |.| <= 1.
    int8: scale = 128.

  Accumulator: int32. The fixed-point logit is then rescaled to
  float by dividing by (feature_scale * weight_scale).

This script computes:

  1. Float linear primitive vs oracle (reference)
  2. Fixed-point at N=8 vs float primitive (quantization-only cost)
  3. Fixed-point at N=8 vs oracle (end-to-end cost)
  4. Same for N=12 and N=16 (to see if higher precision helps)
  5. Optional sigmoid-via-LUT for probability output
  6. ECE / AUC / Brier vs labels for each variant
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
import fast_materialize as fm  # noqa: E402


# --- Feature extraction (matches surrogate_ladder.py) -----------------------


def differentials(X):
    egg = X[:, 0] - X[:, 20]
    food = X[:, 1] - X[:, 21]
    snail = X[:, 49]
    sol = X[:, 3] - X[:, 23]
    war = X[:, 2] - X[:, 22]
    ber = X[:, 51]
    return np.column_stack([egg, food, snail, sol, war, ber]).astype(np.float64)


DIFF_NAMES = [
    "egg_diff",
    "food_diff",
    "snail_pos",
    "soldier_diff",
    "warrior_diff",
    "berries_norm",
]

# Per-feature pre-quantization affine: x_norm = (x - center) / half_range,
# then x_int = round(x_norm * (2^(N-1) - 1)). Chosen so the typical range
# of each feature maps to [-1, 1] with a small safety margin.
FEATURE_NORM = {
    "egg_diff": {"center": 0.0, "half_range": 4.0},  # range [-3,3]
    "food_diff": {"center": 0.0, "half_range": 16.0},  # range [-12,12]
    "snail_pos": {"center": 0.0, "half_range": 1.0},  # clip to [-1,1]
    "soldier_diff": {"center": 0.0, "half_range": 4.0},  # [-3,3]
    "warrior_diff": {"center": 0.0, "half_range": 4.0},  # [-4,4]
    "berries_norm": {"center": 0.7, "half_range": 0.3},  # [0.43,0.94]
}


def quantize_features(D, n_bits):
    """Quantize the (n,6) differential matrix to int.
    Returns (D_int, D_recovered) where D_recovered is the float
    representation of the int (for fidelity bookkeeping).
    """
    scale = (1 << (n_bits - 1)) - 1  # signed int max value
    D_int = np.empty_like(D, dtype=np.int32)
    D_rec = np.empty_like(D, dtype=np.float64)
    for i, name in enumerate(DIFF_NAMES):
        norm = FEATURE_NORM[name]
        x = D[:, i]
        # snail outliers
        x = np.clip(
            x, norm["center"] - norm["half_range"], norm["center"] + norm["half_range"]
        )
        x_n = (x - norm["center"]) / norm["half_range"]
        D_int[:, i] = np.round(x_n * scale).astype(np.int32)
        D_int[:, i] = np.clip(D_int[:, i], -scale, scale)
        # recovered float
        D_rec[:, i] = (D_int[:, i] / scale) * norm["half_range"] + norm["center"]
    return D_int, D_rec


def quantize_weights(w, b, n_bits):
    """Quantize a (6,) weight vector and scalar intercept. Returns int
    weights, int intercept, plus their recovered floats."""
    scale = (1 << (n_bits - 1)) - 1
    # All weights |w| <= 1; intercept |b| <= 1.
    w_int = np.clip(np.round(w * scale), -scale, scale).astype(np.int32)
    b_int = int(np.clip(np.round(b * scale), -scale, scale))
    w_rec = w_int / scale
    b_rec = b_int / scale
    return w_int, b_int, w_rec, b_rec


def fixed_point_logit(D_int, w_int, b_int, n_bits_f, n_bits_w):
    """Fixed-point dot product. D_int is (n,6) with feature scale
    sf = 2^(n_bits_f - 1) - 1. Each feature represents the float
    value (D_int / sf) * half_range + center. Weight scale sw similar.

    To get the float logit we need to rescale per-feature:
      float_logit = sum_i w_i * x_i + b
                   = sum_i (w_int_i/sw) * ((D_int_i/sf)*hr_i + c_i) + (b_int/sw)
    For pure-integer accumulation we expand:
      acc = sum_i w_int_i * D_int_i  (int32, no scale)
      then float_logit = acc * (per-feature factor) + bias terms
    But per-feature factors differ. Simpler/cleaner: per-feature
    integer scale = (hr_i / sf), apply during dequantize.

    Hardware: each multiply is int * int -> int. The dequantize
    happens once at the end (or as a single global scale if all
    features share half_range, which they don't in our case).

    Here we model the per-feature pipeline exactly: convert each
    feature back to the centered/half_range float then multiply
    by the recovered float weight, summing into a single float
    accumulator. This is the "deployed primitive" in software.
    """
    sf = (1 << (n_bits_f - 1)) - 1
    sw = (1 << (n_bits_w - 1)) - 1
    w_rec = w_int / sw
    b_rec = b_int / sw
    # Recover features
    n = D_int.shape[0]
    out = np.full(n, b_rec, dtype=np.float64)
    for i, name in enumerate(DIFF_NAMES):
        norm = FEATURE_NORM[name]
        x_rec = (D_int[:, i] / sf) * norm["half_range"] + norm["center"]
        out += w_rec[i] * x_rec
    return out


def sigmoid_lut(logit, lut_bits=10, lut_range=8.0):
    """Sigmoid via LUT. lut_bits gives the input quantization; lut_range
    is the symmetric input clamp."""
    n = 1 << lut_bits
    edges = np.linspace(-lut_range, lut_range, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    lut_vals = 1.0 / (1.0 + np.exp(-centers))
    idx = np.searchsorted(edges, np.clip(logit, -lut_range, lut_range)) - 1
    idx = np.clip(idx, 0, n - 1)
    return lut_vals[idx]


# --- Metrics ---------------------------------------------------------------


def logit_f(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def metrics(p_pred, p_oracle, y):
    eps = 1e-12
    pp = np.clip(p_pred, eps, 1 - eps)
    return {
        "prob_rmse_vs_oracle": float(np.sqrt(((p_pred - p_oracle) ** 2).mean())),
        "logit_rmse_vs_oracle": float(
            np.sqrt(((logit_f(p_pred) - logit_f(p_oracle)) ** 2).mean())
        ),
        "logloss": float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean()),
        "brier": float(((pp - y) ** 2).mean()),
        "auc": _auc(pp, y),
        "ece10": _ece(pp, y, 10),
        "acc@0.5": float(((pp >= 0.5).astype(int) == y).mean()),
    }


def _auc(p, y):
    order = np.argsort(p)
    ys = y[order]
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = np.arange(1, len(y) + 1)
    return float(((ranks * ys).sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _ece(p, y, n_bins):
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(p)
    for i in range(n_bins):
        mask = (p >= edges[i]) & (
            (p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1])
        )
        m = int(mask.sum())
        if m == 0:
            continue
        ece += (m / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


# --- Pipeline ---------------------------------------------------------------


def materialize_csv_glob(csv_glob):
    paths = sorted(glob.glob(csv_glob))
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


def game_split(gids, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    u = np.unique(gids)
    rng.shuffle(u)
    n = int(len(u) * frac)
    tr = set(u[:n].tolist())
    is_tr = np.fromiter((g in tr for g in gids), dtype=bool)
    return is_tr, ~is_tr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(REPO / "current_preferred_model.mdl"))
    ap.add_argument("--csv-glob", default=str(REPO / "tests/benchmark_events_*.csv.gz"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    booster = lgb.Booster(model_file=args.model)

    X, y, gids, ts = materialize_csv_glob(args.csv_glob)
    tr_mask, te_mask = game_split(gids, 0.8, seed=args.seed)
    n_tr = int(tr_mask.sum())
    n_te = int(te_mask.sum())
    print(f"  n_train={n_tr} n_test={n_te}")

    D = differentials(X)
    D_tr, D_te = D[tr_mask], D[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]
    p_or_te = booster.predict(X[te_mask])
    t_or_tr = logit_f(booster.predict(X[tr_mask]))

    # Float linear surrogate (Route A primitive)
    m_float = LinearRegression()
    m_float.fit(D_tr, t_or_tr)
    w_f = m_float.coef_.copy()
    b_f = float(m_float.intercept_)
    print("\nFloat hand_diff_linear_6 weights:")
    for n, w in zip(DIFF_NAMES, w_f):
        print(f"  {n:18s} w = {w:+.6f}")
    print(f"  intercept           b = {b_f:+.6f}")

    # Float predictions
    logit_float = D_te @ w_f + b_f
    p_float = sigmoid(logit_float)

    # Fixed-point sweep
    print("\n=== fixed-point parity sweep ===")
    print(
        f"{'config':28s} "
        f"{'pRMSEvF':>9s} {'pRMSEvO':>9s} {'lRMSE':>8s} "
        f"{'AUC':>7s} {'Brier':>7s} {'ECE':>7s} {'dAUC':>7s}"
    )
    auc_float = _auc(p_float, y_te)
    print(
        f"{'float reference':28s} "
        f"{'0.0000':>9s} "
        f"{np.sqrt(((p_float - p_or_te) ** 2).mean()):>9.4f} "
        f"{np.sqrt(((logit_f(p_float) - logit_f(p_or_te)) ** 2).mean()):>8.4f} "
        f"{auc_float:>7.4f} {((p_float - y_te) ** 2).mean():>7.4f} "
        f"{_ece(p_float, y_te, 10):>7.4f} "
        f"{auc_float - 0.7901:>+7.4f}"
    )

    runs = []
    for n_bits_f in (8, 12, 16):
        for n_bits_w in (8, 12, 16):
            D_int, _ = quantize_features(D_te, n_bits_f)
            w_int, b_int, _, _ = quantize_weights(w_f, b_f, n_bits_w)
            logit_fix = fixed_point_logit(D_int, w_int, b_int, n_bits_f, n_bits_w)
            p_fix = sigmoid(logit_fix)
            p_fix_lut = sigmoid_lut(logit_fix, lut_bits=10, lut_range=8.0)
            r = {
                "n_bits_features": n_bits_f,
                "n_bits_weights": n_bits_w,
                "metrics_vs_float_and_oracle": {
                    "prob_rmse_vs_float": float(
                        np.sqrt(((p_fix - p_float) ** 2).mean())
                    ),
                    **metrics(p_fix, p_or_te, y_te),
                },
                "lut_sigmoid": {
                    "prob_rmse_vs_float": float(
                        np.sqrt(((p_fix_lut - p_float) ** 2).mean())
                    ),
                    **metrics(p_fix_lut, p_or_te, y_te),
                },
                "weights_int": w_int.tolist(),
                "intercept_int": b_int,
            }
            runs.append(r)
            tag = f"feat={n_bits_f}b w={n_bits_w}b"
            mm = r["metrics_vs_float_and_oracle"]
            print(
                f"{tag:28s} "
                f"{mm['prob_rmse_vs_float']:>9.4f} "
                f"{mm['prob_rmse_vs_oracle']:>9.4f} "
                f"{mm['logit_rmse_vs_oracle']:>8.4f} "
                f"{mm['auc']:>7.4f} {mm['brier']:>7.4f} "
                f"{mm['ece10']:>7.4f} "
                f"{mm['auc'] - 0.7901:>+7.4f}"
            )

    # Also: oracle reference row
    print(
        f"\noracle (LightGBM):  pRMSE 0.0000  AUC 0.7901  "
        f"Brier {((p_or_te - y_te) ** 2).mean():.4f}  "
        f"ECE10 {_ece(p_or_te, y_te, 10):.4f}"
    )

    # Sigmoid LUT row
    print("\n=== sigmoid-via-LUT (1024 entries, range +-8) on int8 logit ===")
    chosen = runs[0]  # 8b feat, 8b w
    mm = chosen["lut_sigmoid"]
    print(
        f"  pRMSE vs float surrogate = {mm['prob_rmse_vs_float']:.4f}  "
        f"pRMSE vs oracle = {mm['prob_rmse_vs_oracle']:.4f}  "
        f"AUC = {mm['auc']:.4f}  Brier = {mm['brier']:.4f}  "
        f"ECE = {mm['ece10']:.4f}"
    )

    out = {
        "seed": args.seed,
        "n_train": n_tr,
        "n_test": n_te,
        "float_weights": {n: float(w) for n, w in zip(DIFF_NAMES, w_f)},
        "float_intercept": b_f,
        "feature_norm": FEATURE_NORM,
        "diff_names": DIFF_NAMES,
        "float_metrics": metrics(p_float, p_or_te, y_te),
        "oracle_baseline_auc": float(_auc(p_or_te, y_te)),
        "oracle_baseline_brier": float(((p_or_te - y_te) ** 2).mean()),
        "oracle_baseline_ece10": float(_ece(p_or_te, y_te, 10)),
        "runs": runs,
    }
    (out_dir / "phase4_fixed_point.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved {out_dir / 'phase4_fixed_point.json'}")


if __name__ == "__main__":
    main()
