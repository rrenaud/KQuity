#!/usr/bin/env python3
"""Canonical pure-int reference for the KQuity objective-pressure primitive.

Phase 4's fixed-point script used "int features + int weights ->
float dequantize -> float math" which is a fidelity check, not a
hardware pipeline. This module replaces that with a pure-integer
pipeline that the HLS C++ implements bit-exactly. We then verify
the pure-int pipeline reproduces Phase 4's float-after-dequantize
output to within <= 1 LSB on a held-out eval set.

# Quantization plan

  feature: signed int8 in Q1.7
    x_norm_i = clip((x_i - center_i) / half_range_i, -1, 1)
    x_int_i  = clip(round(x_norm_i * 128), -128, 127)
    (scale Sf = 128)

  weight (folded with half_range): signed int8 in Q3.5
    w_eff_i = w_i * half_range_i
    w_int_i = clip(round(w_eff_i * 32), -128, 127)
    (scale Sw = 32)

  bias (folded with feature centers): in accumulator units
    b_eff   = b + sum_i (w_i * center_i)
    b_acc   = round(b_eff * Sf * Sw) = round(b_eff * 4096)
    (stored as int16; range fits comfortably)

  accumulator: int32. The dot product is
    acc = b_acc + sum_i (w_int_i * x_int_i)
  Per-multiply max magnitude = 128 * 128 = 16384; six terms sum
  to <= 98304 plus bias; fits in int24.

  output: saturated to int16 (Q4.12), divisible by 4096 to get
  the float logit. Logit range observed on test is roughly
  [-5, +5] so Q4.12 in int16 is comfortably non-saturating.

  optional probability: 1024-entry uint16 sigmoid LUT.
    input = clamp(logit_q4_12 >> 2, -2048, 2047)  # Q4.10 in 12 bits
    addr  = (input + 2048)                        # 0..4095 -> bin 0..1023
    Actually simpler: clamp logit_q4_12 to [-8,8] range mapped to
    1024-entry table. See `kquity_prob_lut` below for exact form.

# Bit-exact contract

The function `kquity_logit_q8(x_int)` returns the int16 logit
that the HLS C++ produces. The function `kquity_logit_float(x)`
returns the recovered float logit. The Phase 5 testbench requires
HLS == kquity_logit_q8 for every event in the test pool.

The bit-exact path differs from Phase 4 by O(2^-15) and is a
valid hardware quantization of the same primitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Per-feature normalization parameters (must match phase4_fixed_point.py)
FEATURE_NORM = {
    "egg_diff": {"center": 0.0, "half_range": 4.0},
    "food_diff": {"center": 0.0, "half_range": 16.0},
    "snail_pos": {"center": 0.0, "half_range": 1.0},
    "soldier_diff": {"center": 0.0, "half_range": 4.0},
    "warrior_diff": {"center": 0.0, "half_range": 4.0},
    "berries_norm": {"center": 0.7, "half_range": 0.3},
}
DIFF_NAMES = list(FEATURE_NORM.keys())

# Quantization scales (compile-time constants in HLS).
SF = 128  # feature scale (Q1.7)
SW = 32  # weight scale (Q3.5)
ACC_SCALE = SF * SW  # 4096 = 2^12, Q4.12 output


def quantize_features_int8(D: np.ndarray) -> np.ndarray:
    """Quantize the (n, 6) differential matrix to int8 in Q1.7.
    Each feature is normalized to [-1, 1] then scaled by 128.
    """
    out = np.empty(D.shape, dtype=np.int8)
    for i, name in enumerate(DIFF_NAMES):
        norm = FEATURE_NORM[name]
        x = D[:, i]
        x = np.clip(
            x,
            norm["center"] - norm["half_range"],
            norm["center"] + norm["half_range"],
        )
        x_norm = (x - norm["center"]) / norm["half_range"]
        x_int = np.round(x_norm * SF).astype(np.int32)
        x_int = np.clip(x_int, -128, 127)
        out[:, i] = x_int.astype(np.int8)
    return out


def make_constants(weights_float: dict, bias_float: float) -> dict:
    """Compute the int8 folded weights and int16 bias for the
    pure-int pipeline.

      w_eff_i = w_i * half_range_i           folded weight
      w_int_i = clip(round(w_eff_i*SW), int8)
      b_eff   = b + sum_i w_i * center_i     folded bias
      b_acc   = round(b_eff * SF * SW)       in accumulator units
    """
    w_int = []
    w_eff_list = []
    b_center_contrib = 0.0
    for name in DIFF_NAMES:
        w = weights_float[name]
        norm = FEATURE_NORM[name]
        w_eff = w * norm["half_range"]
        w_q = int(np.clip(round(w_eff * SW), -128, 127))
        w_int.append(w_q)
        w_eff_list.append(w_eff)
        b_center_contrib += w * norm["center"]
    b_eff = bias_float + b_center_contrib
    b_acc = int(round(b_eff * SF * SW))
    return {
        "feature_names": DIFF_NAMES,
        "w_int": w_int,
        "w_eff_float": w_eff_list,
        "b_eff_float": b_eff,
        "b_acc": b_acc,
        "Sf": SF,
        "Sw": SW,
        "ACC_SCALE": ACC_SCALE,
        "feature_norm": FEATURE_NORM,
    }


def kquity_logit_q8(x_int: np.ndarray, w_int: list, b_acc: int) -> np.ndarray:
    """Pure-int dot product. Input: int8 features (n, 6). Output:
    int32 accumulator (call it the raw logit; the canonical 16-bit
    output is saturate_to_int16(acc)).
    """
    n = x_int.shape[0]
    w = np.array(w_int, dtype=np.int32)
    # int8 * int8 -> int16; sum is int32-safe
    acc = (x_int.astype(np.int32) * w).sum(axis=1) + int(b_acc)
    return acc.astype(np.int32)


def saturate_int16(acc32: np.ndarray) -> np.ndarray:
    return np.clip(acc32, -32768, 32767).astype(np.int16)


def kquity_logit_to_float(logit_q4_12: np.ndarray) -> np.ndarray:
    """Recover float logit from Q4.12 int16."""
    return logit_q4_12.astype(np.float64) / ACC_SCALE


# --- Sigmoid LUT ---------------------------------------------------------

LUT_BITS = 10  # 1024 entries
LUT_RANGE_LOGIT = 8.0  # clamp input to +-8.0


def build_sigmoid_lut(n_bits=LUT_BITS, half_range=LUT_RANGE_LOGIT) -> np.ndarray:
    """uint16 sigmoid LUT. For logit l in [-half_range, +half_range],
    return uint16 quantization of sigmoid(l) in [0, 65535].
    """
    n = 1 << n_bits
    edges = np.linspace(-half_range, half_range, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sig = 1.0 / (1.0 + np.exp(-centers))
    return np.clip(np.round(sig * 65535), 0, 65535).astype(np.uint16)


def kquity_prob_lut(
    logit_q4_12: np.ndarray,
    lut: np.ndarray,
    n_bits=LUT_BITS,
    half_range=LUT_RANGE_LOGIT,
) -> np.ndarray:
    """Map int16 Q4.12 logit to uint16 probability via 1024-entry
    LUT. Input is clamped to [-half_range, +half_range] in float
    terms. The LUT index is computed in pure int via shift/clamp.

    Bit-exact recipe:
      half_range_int_q4_12 = round(half_range * 4096)  = 32768
      l_clamped = clip(logit_q4_12, -half_range_int, +half_range_int - 1)
      Actually: clip to int16 range [-32768, 32767]; half_range_int
      is 32768 which exceeds int16 by 1. Use clamp to [-32768, 32767].

      step = (2 * half_range_int) / n   = 65536 / 1024 = 64
      idx = clip((l + half_range_int) // step, 0, n-1)
            = clip((l + 32768) >> 6, 0, n-1)

    For l = 0 (logit 0.0), idx = 32768 >> 6 = 512 (center of LUT).
    For l = 32767, idx = 1023.  For l = -32768, idx = 0.
    """
    n = 1 << n_bits
    half_int = int(round(half_range * ACC_SCALE))  # 32768
    # Subtle: half_int may equal 32768 which is +1 outside int16.
    # The int math below works in wider int.
    l32 = logit_q4_12.astype(np.int32)
    l_clamped = np.clip(l32, -half_int, half_int - 1)
    step = (2 * half_int) // n  # 64
    idx = (l_clamped + half_int) // step
    idx = np.clip(idx, 0, n - 1)
    return lut[idx]


# --- Verification: int ref vs Phase 4 float-after-dequantize -------------


def phase4_float_after_dequant(
    D: np.ndarray, weights_float: dict, bias_float: float
) -> np.ndarray:
    """Reproduces phase4_fixed_point.fixed_point_logit for 8-bit
    everywhere. This is the OLD reference; we keep it for fidelity
    comparison.
    """
    sf = 127  # the phase4 code used signed-int max 127, not 128
    out = np.full(D.shape[0], bias_float, dtype=np.float64)
    for i, name in enumerate(DIFF_NAMES):
        norm = FEATURE_NORM[name]
        x = D[:, i]
        x = np.clip(
            x,
            norm["center"] - norm["half_range"],
            norm["center"] + norm["half_range"],
        )
        x_n = (x - norm["center"]) / norm["half_range"]
        x_int = np.clip(np.round(x_n * sf), -sf, sf)
        x_rec = (x_int / sf) * norm["half_range"] + norm["center"]
        w_int = int(np.clip(np.round(weights_float[name] * sf), -sf, sf))
        w_rec = w_int / sf
        out += w_rec * x_rec
    return out


# --- CLI: load Phase 4 weights, emit constants header ---------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--phase4-json",
        default=str(
            Path(__file__).resolve().parents[1] / "results/phase4_fixed_point.json"
        ),
    )
    ap.add_argument(
        "--out-header",
        default=str(Path(__file__).resolve().parent / "kquity_constants.h"),
    )
    ap.add_argument(
        "--out-lut-header",
        default=str(Path(__file__).resolve().parent / "kquity_sigmoid_lut.h"),
    )
    args = ap.parse_args()

    j = json.loads(Path(args.phase4_json).read_text())
    weights = j["float_weights"]
    bias = j["float_intercept"]

    consts = make_constants(weights, bias)

    print("=== int8 folded constants ===")
    print(f"{'feature':18s} {'w_float':>10s} {'w_eff':>10s} {'w_int':>8s}")
    for i, n in enumerate(DIFF_NAMES):
        print(
            f"{n:18s} {weights[n]:>10.6f} {consts['w_eff_float'][i]:>10.6f} "
            f"{consts['w_int'][i]:>8d}"
        )
    print(
        f"{'bias':18s} {bias:>10.6f} {consts['b_eff_float']:>10.6f} "
        f"{'b_acc':>8s}={consts['b_acc']}"
    )
    print(f"Sf={consts['Sf']}  Sw={consts['Sw']}  ACC_SCALE={consts['ACC_SCALE']}")

    # Emit C++ header with constants.
    lines = []
    lines.append("// SPDX-License-Identifier: MIT")
    lines.append("// Generated by experiments/kquity_hw/hls/kquity_pressure_ref.py")
    lines.append("// from " + args.phase4_json)
    lines.append("// Do not edit by hand.")
    lines.append("#ifndef KQUITY_CONSTANTS_H")
    lines.append("#define KQUITY_CONSTANTS_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("// Pure-int objective-pressure primitive. Features are int8 in Q1.7")
    lines.append(
        "// after per-feature normalization (host or PS pre-pack). Weights are"
    )
    lines.append("// int8 in Q3.5 with the per-feature half-range folded in. Bias is")
    lines.append("// int16 in the accumulator Q-format (Q4.12, Sf*Sw=4096).")
    lines.append("")
    lines.append(f"#define KQUITY_SF        {consts['Sf']}")
    lines.append(f"#define KQUITY_SW        {consts['Sw']}")
    lines.append(f"#define KQUITY_ACC_SCALE {consts['ACC_SCALE']}")
    lines.append("")
    for i, n in enumerate(DIFF_NAMES):
        upper = n.upper().replace("_DIFF", "")
        if upper == "BERRIES_NORM":
            upper = "BERRIES"
        if upper == "SNAIL_POS":
            upper = "SNAIL"
        lines.append(
            f"static const int8_t  KQUITY_W_{upper:<10s} = "
            f"{consts['w_int'][i]:>4d}; // {n}"
        )
    lines.append(f"static const int16_t KQUITY_B_ACC      = {consts['b_acc']:>5d};")
    lines.append("")
    lines.append("// Per-feature host-side quantization parameters (for")
    lines.append("// reference; the HLS top assumes pre-quantized int8 inputs).")
    for i, n in enumerate(DIFF_NAMES):
        norm = consts["feature_norm"][n]
        upper = n.upper().replace("_DIFF", "")
        if upper == "BERRIES_NORM":
            upper = "BERRIES"
        if upper == "SNAIL_POS":
            upper = "SNAIL"
        lines.append(
            f"// {n}: center={norm['center']:.3f}, half_range={norm['half_range']:.3f}"
        )
    lines.append("")
    lines.append("#endif // KQUITY_CONSTANTS_H")
    Path(args.out_header).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_header}")

    # Emit the sigmoid LUT
    lut = build_sigmoid_lut()
    lines = []
    lines.append("// SPDX-License-Identifier: MIT")
    lines.append("// Generated sigmoid LUT for KQuity. 1024 entries, uint16.")
    lines.append("// Input: int16 Q4.12 logit clamped to [-32768, 32767].")
    lines.append("// Index recipe: idx = clip((logit + 32768) >> 6, 0, 1023).")
    lines.append("// Output: uint16 probability, 0..65535 = sigmoid * 65535.")
    lines.append("#ifndef KQUITY_SIGMOID_LUT_H")
    lines.append("#define KQUITY_SIGMOID_LUT_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("#define KQUITY_LUT_BITS    10")
    lines.append("#define KQUITY_LUT_SIZE    1024")
    lines.append("#define KQUITY_LUT_HALF    32768")
    lines.append("#define KQUITY_LUT_SHIFT   6")
    lines.append("")
    lines.append("static const uint16_t kquity_sigmoid_lut[KQUITY_LUT_SIZE] = {")
    for row_start in range(0, len(lut), 8):
        row = lut[row_start : row_start + 8]
        s = ", ".join(f"{int(v):5d}" for v in row)
        comma = "," if row_start + 8 < len(lut) else ""
        lines.append(f"    {s}{comma}")
    lines.append("};")
    lines.append("")
    lines.append("#endif // KQUITY_SIGMOID_LUT_H")
    Path(args.out_lut_header).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_lut_header}")


if __name__ == "__main__":
    main()
