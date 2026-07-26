#!/usr/bin/env python3
"""Conifer model-to-fabric baseline for the KQuity comparison.

Per ChatGPT Pro directive (Case F Conifer baseline):

  Run a direct model-to-fabric baseline using Conifer. Compare
  against the 7-parameter extracted objective-pressure primitive
  (Phase 5: 269 LUT / 2 DSP / 4 cyc / 288.9 MHz). The point is
  not to replace the extraction story but to show what direct
  model compilation costs for the same task.

Ladder (per directive):
  1. small 50t × 32l × 52 feat LightGBM ("Good fidelity")
       - pRMSE ~0.047 vs oracle, AUC ~0.786
       - this is the main Conifer comparison
  2. full preferred LightGBM (100t × 100l × 52)
       - only if conversion is easy

Conversion path:
  LightGBM -> ONNX (onnxmltools) -> Conifer (convert_from_onnx)

Targets:
  XilinxPart  xck26-sfvc784-2LV-c (KV260)
  ClockPeriod 5 ns
  Precision   ap_fixed<16,6>  (first pass)
              ap_fixed<12,6>  (optional, time permitting)

Stop condition: 2 hours of toolchain pain. Document blocker and
move on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import lightgbm as lgb  # noqa: E402
import fast_materialize as fm  # noqa: E402


# --- Differential features (same as Phase 4) ---------------------------------


def differentials(X):
    egg = X[:, 0] - X[:, 20]
    food = X[:, 1] - X[:, 21]
    snail = X[:, 49]
    sol = X[:, 3] - X[:, 23]
    war = X[:, 2] - X[:, 22]
    ber = X[:, 51]
    return np.column_stack([egg, food, snail, sol, war, ber]).astype(np.float32)


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
        np.concatenate(Xs).astype(np.float32),
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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def metrics_block(p_pred, p_oracle, y):
    eps = 1e-12
    pp = np.clip(p_pred, eps, 1 - eps)
    prob_rmse = float(np.sqrt(((p_pred - p_oracle) ** 2).mean()))
    logloss = float(-(y * np.log(pp) + (1 - y) * np.log(1 - pp)).mean())
    brier = float(((pp - y) ** 2).mean())
    order = np.argsort(pp)
    ys = y[order]
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    if npos == 0 or nneg == 0:
        auc = float("nan")
    else:
        ranks = np.arange(1, len(y) + 1)
        auc = float(((ranks * ys).sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return {
        "prob_rmse_vs_oracle": prob_rmse,
        "logloss": logloss,
        "brier": brier,
        "auc": auc,
    }


def train_small_lgbm(X_tr, y_tr, n_estimators=50, num_leaves=32, seed=0):
    m = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=0.1,
        verbose=-1,
        random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m


def train_small_xgboost(X_tr, y_tr, n_estimators=50, max_depth=5, seed=0):
    import xgboost as xgb

    m = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method="hist",
        random_state=seed,
        verbosity=0,
        eval_metric="logloss",
    )
    m.fit(X_tr, y_tr)
    return m


# --- ONNX export + Conifer convert -----------------------------------------


def export_lgbm_to_onnx(clf, n_features, out_path):
    """Export a LightGBM classifier to ONNX. Returns the in-memory
    ONNX ModelProto and writes it to disk.
    """
    from onnxmltools.convert.common.data_types import FloatTensorType
    from onnxmltools.convert.lightgbm.convert import convert as convert_lightgbm

    initial_types = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(
        clf,
        initial_types=initial_types,
        target_opset=12,
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return onnx_model


def conifer_convert_and_build(
    source_model,
    source_kind,
    project_name,
    output_dir,
    xilinx_part="xck26-sfvc784-2LV-c",
    clock_period_ns=5,
    precision="ap_fixed<16,6>",
):
    """Convert a tree model with Conifer and run synthesis.

    source_kind: 'onnx' (uses convert_from_onnx) or 'xgboost'
                 (uses convert_from_xgboost on an XGBClassifier).
    """
    import conifer  # late import: conifer logs noisy warnings on import

    cfg = conifer.backends.xilinxhls.auto_config()
    # NOTE: do NOT override ProjectName. Conifer's bridge.cpp template
    # hardcodes 'firmware/my_prj.h', so changing ProjectName breaks
    # compile/build. We keep the default 'my_prj' and use OutputDir
    # to namespace the projects instead.
    cfg["OutputDir"] = str(output_dir)
    cfg["XilinxPart"] = xilinx_part
    cfg["ClockPeriod"] = str(clock_period_ns)
    cfg["Precision"] = precision

    if source_kind == "onnx":
        model = conifer.converters.convert_from_onnx(source_model, cfg)
    elif source_kind == "xgboost":
        # XGBClassifier -> use get_booster()
        booster = (
            source_model.get_booster()
            if hasattr(source_model, "get_booster")
            else source_model
        )
        model = conifer.converters.convert_from_xgboost(booster, cfg)
    else:
        raise ValueError(f"unknown source_kind {source_kind}")
    model.compile()
    return model, cfg


def parse_csynth_report(report_path):
    """Pull LUT / FF / DSP / BRAM / latency / II / Fmax from a Vitis
    HLS csynth report. Returns dict or {} if not found."""
    if not Path(report_path).exists():
        return {}
    text = Path(report_path).read_text()
    out = {}
    # Timing
    import re

    m = re.search(r"\|ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns", text)
    if m:
        out["clock_target_ns"] = float(m.group(1))
        out["clock_estimated_ns"] = float(m.group(2))
        if out["clock_estimated_ns"] > 0:
            out["fmax_mhz"] = 1000.0 / out["clock_estimated_ns"]
    # Latency
    m = re.search(
        r"Latency \(cycles\) \|\s+Latency \(absolute\)\s*\|\s*Interval\s*\|\s*Pipeline\|\s*\n"
        r"\s*\|\s*min\s*\|\s*max\s*\|\s*min\s*\|\s*max\s*\|\s*min\s*\|\s*max\s*\|\s*Type\s*\|\s*\n"
        r"\s*\+[-+]+\+\s*\n"
        r"\s*\|\s*(\d+)\|\s*(\d+)\|.*?\|.*?\|\s*(\d+)\|\s*(\d+)\|",
        text,
        re.DOTALL,
    )
    if m:
        out["latency_min"] = int(m.group(1))
        out["latency_max"] = int(m.group(2))
        out["interval_min"] = int(m.group(3))
        out["interval_max"] = int(m.group(4))
    # Utilization "Total" row (skip after Available)
    m = re.search(
        r"\|Total\s*\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|\s*(\d+)\|",
        text,
    )
    if m:
        out["bram_18k"] = int(m.group(1))
        out["dsp"] = int(m.group(2))
        out["ff"] = int(m.group(3))
        out["lut"] = int(m.group(4))
        out["uram"] = int(m.group(5))
    return out


# --- Main --------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-glob", default=str(REPO / "tests/benchmark_events_*.csv.gz"))
    ap.add_argument(
        "--oracle",
        default=str(REPO / "current_preferred_model.mdl"),
    )
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--num-leaves", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="ap_fixed<16,6>")
    ap.add_argument("--xilinx-part", default="xck26-sfvc784-2LV-c")
    ap.add_argument("--clock-period", type=int, default=5)
    ap.add_argument(
        "--also-precision-12",
        action="store_true",
        help="Run a second pass at ap_fixed<12,6>",
    )
    ap.add_argument(
        "--source",
        choices=["lightgbm-onnx", "xgboost"],
        default="xgboost",
        help="Tree source for Conifer. LightGBM->ONNX failed on "
        "Conifer's onnx parser (KeyError base_values); xgboost is "
        "the documented fallback.",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="XGBoost max_depth (only used when --source=xgboost). "
        "depth 5 gives up to 32 leaves -- approximate match to "
        "the 32-leaf LightGBM.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"materializing eval pool from {args.csv_glob}")
    X, y, gids, ts = materialize_csv_glob(args.csv_glob)
    print(f"  {X.shape=}, n_games={len(np.unique(gids))}")
    tr_mask, te_mask = game_split(gids, 0.8, seed=args.seed)
    X_tr, X_te = X[tr_mask], X[te_mask]
    y_tr, y_te = y[tr_mask], y[te_mask]
    n_te = int(te_mask.sum())
    print(f"  train events {int(tr_mask.sum())}  test events {n_te}")

    # Oracle predictions for fidelity comparison
    booster_oracle = lgb.Booster(model_file=args.oracle)
    p_or_te = booster_oracle.predict(X_te)

    if args.source == "lightgbm-onnx":
        print(
            f"\ntraining small LightGBM: n_estimators={args.n_estimators}, "
            f"num_leaves={args.num_leaves}, n_features=52 ..."
        )
        clf = train_small_lgbm(
            X_tr,
            y_tr,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            seed=args.seed,
        )
        p_small_te = clf.predict_proba(X_te)[:, 1]
        onnx_path = out_dir / f"kquity_{args.n_estimators}t_{args.num_leaves}l.onnx"
        print(f"exporting LightGBM -> ONNX: {onnx_path}")
        source_model = export_lgbm_to_onnx(clf, n_features=52, out_path=onnx_path)
        source_kind = "onnx"
        cfg_tag = f"lgbm_{args.n_estimators}t_{args.num_leaves}l"
    else:
        print(
            f"\ntraining small XGBoost: n_estimators={args.n_estimators}, "
            f"max_depth={args.max_depth}, n_features=52 ..."
        )
        clf = train_small_xgboost(
            X_tr,
            y_tr,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            seed=args.seed,
        )
        p_small_te = clf.predict_proba(X_te)[:, 1]
        source_model = clf
        source_kind = "xgboost"
        cfg_tag = f"xgb_{args.n_estimators}t_d{args.max_depth}"

    small_metrics = metrics_block(p_small_te, p_or_te, y_te)
    print(f"  small model fidelity vs oracle:")
    for k, v in small_metrics.items():
        print(f"    {k}: {v}")

    tag = f"{cfg_tag}_p16_6"
    project_dir = out_dir / f"conifer_{tag}"
    print(f"\nConifer convert + build: project_dir={project_dir}")
    print(f"  source = {source_kind}")
    print(f"  precision = {args.precision}")
    print(f"  xilinx_part = {args.xilinx_part}")
    print(f"  clock = {args.clock_period} ns")
    try:
        model, cfg = conifer_convert_and_build(
            source_model,
            source_kind,
            project_name=f"kquity_{tag}",
            output_dir=project_dir,
            xilinx_part=args.xilinx_part,
            clock_period_ns=args.clock_period,
            precision=args.precision,
        )
        print("  Conifer compile OK")
        # Run synthesis
        print("  invoking model.build() (runs Vitis HLS csynth, several min) ...")
        t_build = time.time()
        try:
            model.build(synth=True, vsynth=False)  # vsynth=False -> csynth only
        except TypeError:
            model.build()
        print(f"  build wall: {time.time()-t_build:.0f}s")
    except Exception as e:
        print(f"  Conifer flow failed: {e!r}")
        out = {
            "stage": "conifer_convert_or_build",
            "error": repr(e),
            "wall_sec": time.time() - t0,
            "small_lgbm_metrics": small_metrics,
        }
        (out_dir / "conifer_baseline_blocked.json").write_text(
            json.dumps(out, indent=2)
        )
        print(f"\nblocker recorded at {out_dir / 'conifer_baseline_blocked.json'}")
        return

    # Parse report
    report_paths = sorted(Path(project_dir).rglob("*csynth.rpt"))
    print(f"\nfound {len(report_paths)} csynth reports:")
    for p in report_paths:
        print(f"  {p}")
    parsed = {}
    for p in report_paths:
        parsed[str(p.relative_to(project_dir))] = parse_csynth_report(p)

    summary = {
        "config": {
            "source": args.source,
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "n_features": 52,
            "xilinx_part": args.xilinx_part,
            "clock_period_ns": args.clock_period,
            "precision": args.precision,
        },
        "small_model_metrics": small_metrics,
        "csynth_reports": parsed,
        "wall_sec": time.time() - t0,
    }
    out_file = out_dir / f"conifer_baseline_{tag}.json"
    out_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved {out_file}")

    print("\n=== summary ===")
    print(f"small LightGBM ({args.n_estimators}t x {args.num_leaves}l, 52 feat):")
    print(f"  AUC = {small_metrics['auc']:.4f}")
    print(f"  Brier = {small_metrics['brier']:.4f}")
    print(f"  pRMSE vs oracle = {small_metrics['prob_rmse_vs_oracle']:.4f}")
    print()
    if parsed:
        top_rpt = next(iter(parsed.values()))
        if top_rpt:
            print(f"Conifer csynth ({args.precision} on {args.xilinx_part}):")
            for k in (
                "lut",
                "ff",
                "dsp",
                "bram_18k",
                "uram",
                "latency_min",
                "latency_max",
                "interval_min",
                "interval_max",
                "clock_estimated_ns",
                "fmax_mhz",
            ):
                if k in top_rpt:
                    print(f"  {k:>18s} = {top_rpt[k]}")
    print()
    print("Extracted primitive (Phase 5 reference):")
    print("  AUC = 0.7496, Brier = 0.2017")
    print("  LUT = 269, FF = 180, DSP = 2, BRAM = 0")
    print("  latency = 4 cycles, II = 1, Fmax = 288.9 MHz")


if __name__ == "__main__":
    main()
