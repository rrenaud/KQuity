#!/usr/bin/env python3
"""KQuity baseline metrics on the LightGBM win-probability oracle.

Phase 1 of Case F. Loads current_preferred_model.mdl, materializes
a held-out eval frame from tests/benchmark_events_*.csv.gz, computes
log loss / Brier / AUC / accuracy / calibration / ECE, then runs
two sanity probes:

  - symmetry: swap blue<->gold features and assert P(gold)_swap ~ 1-P(gold)
  - monotonic: +1 gold egg should usually increase P(gold); +1 blue
    egg should usually decrease P(gold); snail toward gold goal should
    increase P(gold).

Phase split (early/mid/late) reported by event timestamp percentile
within each game.

Writes results/baseline_metrics.json, results/calibration.csv,
results/phase_metrics.csv.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path so we can import fast_materialize
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
import fast_materialize as fm  # noqa: E402

# Feature index map for nice reporting
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


def materialize_csv_glob(
    csv_glob: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize a 52-feature matrix + binary labels + game_ids + per-event
    timestamps from the gzipped CSV shards. Labels are 1 if blue won (matching
    fast_materialize's convention -- we flip below to P(gold) for the oracle).
    """
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"no shards match {csv_glob}")
    Xs, ys, gids, ts = [], [], [], []
    for p in paths:
        res = fm.fast_materialize(p, drop_state_probability=0.0)
        # MaterializeResult is a NamedTuple-like: (states, labels, game_ids, timestamps)
        X = res[0]
        y = res[1]
        g = res[2]
        t = res[3]
        Xs.append(X)
        ys.append(y)
        gids.append(g)
        ts.append(t)
    return (
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(gids),
        np.concatenate(ts),
    )


def expected_calibration_error(probs, y, n_bins=10):
    """ECE: mean absolute gap between bin-mean prob and bin empirical rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    rows = []
    for i in range(n_bins):
        mask = (probs >= edges[i]) & (probs < edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= edges[i]) & (probs <= edges[i + 1])
        m = int(mask.sum())
        if m == 0:
            rows.append(
                {
                    "bin_lo": float(edges[i]),
                    "bin_hi": float(edges[i + 1]),
                    "n": 0,
                    "mean_pred": float("nan"),
                    "emp_rate": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        mean_pred = float(probs[mask].mean())
        emp_rate = float(y[mask].mean())
        gap = abs(mean_pred - emp_rate)
        ece += (m / n) * gap
        rows.append(
            {
                "bin_lo": float(edges[i]),
                "bin_hi": float(edges[i + 1]),
                "n": m,
                "mean_pred": mean_pred,
                "emp_rate": emp_rate,
                "gap": gap,
            }
        )
    return float(ece), rows


def symmetric_swap(X: np.ndarray) -> np.ndarray:
    """Swap blue<->gold features: columns 0-19 <-> 20-39, and negate
    snail_pos/snail_spd (which are gold_sym-multiplied so the swap
    requires flipping sign). Maiden states (40-44) and map one-hot
    (45-48) are color-neutral. berries_avail_norm (51) is symmetric.

    Note: maiden_state values use {-1, 0, +1} where -1=gold, 1=blue;
    symmetric swap should also negate them.
    """
    Xs = X.copy()
    blue = X[:, 0:20].copy()
    gold = X[:, 20:40].copy()
    Xs[:, 0:20] = gold
    Xs[:, 20:40] = blue
    Xs[:, 40:45] = -X[:, 40:45]
    Xs[:, 49] = -X[:, 49]
    Xs[:, 50] = -X[:, 50]
    return Xs


def monotonic_probe(model, X: np.ndarray, feature_idx: int, delta: float):
    """Add `delta` to one feature column and report mean dP(gold)/d(feature)."""
    Xm = X.copy()
    Xm[:, feature_idx] = Xm[:, feature_idx] + delta
    p0 = model.predict(X)
    pm = model.predict(Xm)
    dp = pm - p0
    return {
        "delta": float(delta),
        "mean_dp": float(dp.mean()),
        "median_dp": float(np.median(dp)),
        "frac_positive_delta": float((dp > 0).mean()),
        "frac_zero_delta": float((dp == 0).mean()),
    }


def phase_partition(timestamps: np.ndarray, edges=(0.25, 0.6)) -> np.ndarray:
    """Per-event phase label 0=early, 1=mid, 2=late, computed from
    the event's relative timestamp within its game. Since the test
    shards intermix games we approximate by global quantiles, which
    is fine for a sanity report (game length varies but distribution
    of relative timestamps is consistent)."""
    q1 = float(np.quantile(timestamps, edges[0]))
    q2 = float(np.quantile(timestamps, edges[1]))
    phase = np.zeros(len(timestamps), dtype=np.int8)
    phase[(timestamps >= q1) & (timestamps < q2)] = 1
    phase[timestamps >= q2] = 2
    return phase


def metric_block(probs, y):
    eps = 1e-12
    p = np.clip(probs, eps, 1 - eps)
    logloss = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    brier = float(((p - y) ** 2).mean())
    acc = float(((p >= 0.5).astype(int) == y).mean())
    # AUC via rank
    order = np.argsort(p)
    y_sorted = y[order]
    ranks = np.arange(1, len(y) + 1)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
    else:
        sum_rank_pos = float((ranks * y_sorted).sum())
        auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "logloss": logloss,
        "brier": brier,
        "acc@0.5": acc,
        "auc": auc,
    }


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
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading model: {args.model}")
    booster = lgb.Booster(model_file=args.model)
    assert booster.num_feature() == 52, "expected 52-feature win-probability model"

    print(f"materializing eval frame from: {args.csv_glob}")
    X, y_blue, game_ids, ts = materialize_csv_glob(args.csv_glob)
    print(
        f"  X.shape={X.shape}, y.shape={y_blue.shape}, ts range={ts.min():.1f}..{ts.max():.1f}s"
    )
    print(f"  pos rate (blue wins) = {y_blue.mean():.4f}")

    # NOTE: The README says "P(gold wins)" but the labels in
    # fast_materialize set label=1 if Blue wins, and the monotonic
    # probes (gold.eggs+1 -> dp<0, blue.food+1 -> dp>0, etc.) all
    # confirm the model output is actually P(blue wins). We use
    # y_blue as the matching label.
    p_blue = booster.predict(X)
    print(f"  mean P(blue) over events = {p_blue.mean():.4f}")
    print(f"  mean y_blue (label) = {y_blue.mean():.4f}")

    # Overall metrics
    overall = metric_block(p_blue, y_blue)
    print("\noverall (event-level):")
    for k, v in overall.items():
        print(f"  {k:>10s} = {v}")

    # Calibration
    ece, cal_rows = expected_calibration_error(p_blue, y_blue, n_bins=10)
    print(f"\nECE (10 bins) = {ece:.4f}")
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_dir / "calibration.csv", index=False)

    # Phase metrics
    phase = phase_partition(ts)
    phase_metrics = {}
    for ph_idx, ph_name in enumerate(["early", "mid", "late"]):
        mask = phase == ph_idx
        phase_metrics[ph_name] = metric_block(p_blue[mask], y_blue[mask])
    print("\nphase metrics (event-level):")
    pdf = pd.DataFrame(phase_metrics).T
    pdf.to_csv(out_dir / "phase_metrics.csv")
    print(pdf.to_string())

    # Symmetry probe: P(blue|X) + P(blue|swap(X)) should sum to 1
    # because swap(X) is the "what if I were the other team" view.
    Xs = symmetric_swap(X)
    p_swap = booster.predict(Xs)
    sym_err = float(np.abs((p_blue + p_swap) - 1.0).mean())
    sym_rms = float(np.sqrt(((p_blue + p_swap - 1.0) ** 2).mean()))
    print(
        f"\nsymmetry probe: |P(blue) + P(blue|swap) - 1| mean={sym_err:.4f} rms={sym_rms:.4f}"
    )

    # Monotonic probes — expectations against P(BLUE wins).
    # blue.eggs = blue queen lives remaining; higher = blue healthier.
    # blue.food_count = blue berries deposited; higher = blue closer
    # to economic win. snail_pos > 0 means snail away from gold goal,
    # equivalently toward blue's side (snail_pos = (frac-0.5)*gold_sym).
    monot = {}
    print("\nmonotonic probes (mean dP(blue) when adding delta):")
    for name, idx, delta, expected in [
        ("gold.eggs +1", 20, 1.0, "neg: gold healthier -> blue less likely"),
        ("blue.eggs +1", 0, 1.0, "pos: blue healthier -> blue more likely"),
        ("gold.food_count +1", 21, 1.0, "neg: gold closer to win"),
        ("blue.food_count +1", 1, 1.0, "pos: blue closer to win"),
        ("snail_pos +0.1", 49, 0.1, "pos: snail away from gold -> blue more likely"),
        ("snail_pos -0.1", 49, -0.1, "neg: snail toward gold -> blue less likely"),
    ]:
        r = monotonic_probe(booster, X, idx, delta)
        monot[name] = r
        sign = "+" if r["mean_dp"] >= 0 else "-"
        print(
            f"  {name:>22s}: dp={sign}{abs(r['mean_dp']):.4f} "
            f"(frac>0={r['frac_positive_delta']:.3f}, "
            f"frac=0={r['frac_zero_delta']:.3f})  [expected: {expected}]"
        )

    # Per-feature top importances (gain) for the record
    gains = list(
        zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain").tolist())
    )
    gains.sort(key=lambda kv: -kv[1])
    total_gain = sum(g for _, g in gains)
    top_gain = [
        {
            "feature": n,
            "gain": g,
            "gain_frac": g / total_gain if total_gain > 0 else 0.0,
        }
        for n, g in gains[:15]
    ]
    splits = list(
        zip(FEATURE_NAMES, booster.feature_importance(importance_type="split").tolist())
    )
    splits.sort(key=lambda kv: -kv[1])
    total_split = sum(s for _, s in splits)
    top_split = [
        {
            "feature": n,
            "splits": s,
            "split_frac": s / total_split if total_split > 0 else 0.0,
        }
        for n, s in splits[:15]
    ]

    summary = {
        "model": args.model,
        "label_convention": "y=1 means BLUE wins; model output is P(blue wins)",
        "n_events": int(len(y_blue)),
        "pos_rate_blue_win": float(y_blue.mean()),
        "overall": overall,
        "ece10": ece,
        "symmetry": {
            "mean_abs_err": sym_err,
            "rms_err": sym_rms,
        },
        "phase_metrics": phase_metrics,
        "monotonic_probes": monot,
        "top15_gain": top_gain,
        "top15_split": top_split,
    }
    (out_dir / "baseline_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
