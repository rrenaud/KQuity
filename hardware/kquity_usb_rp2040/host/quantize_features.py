#!/usr/bin/env python3
"""Quantize raw game-state differentials into the six int8 features
the firmware expects. This is the same per-feature affine the HLS
firmware uses (see firmware/kquity_constants.h comments).

Usage:
  echo "egg_diff food_diff snail_pos soldier_diff warrior_diff berries_avail" \\
    | quantize_features.py

For example:
  python quantize_features.py 2 1 0.34 1 -1 0.83
  -> 64 8 43 32 -32 84
"""
from __future__ import annotations

import argparse
import sys

# Per-feature affine (mirrors firmware/kquity_constants.h and
# experiments/kquity_hw/hls/kquity_pressure_ref.py).
FEATURE_NORM = [
    ("egg_diff", 0.0, 4.0),
    ("food_diff", 0.0, 16.0),
    ("snail_pos", 0.0, 1.0),
    ("soldier_diff", 0.0, 4.0),
    ("warrior_diff", 0.0, 4.0),
    ("berries_avail", 0.7, 0.3),
]
SF = 128


def clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def quantize(values):
    if len(values) != 6:
        raise ValueError(f"expected 6 values, got {len(values)}")
    out = []
    for v, (_, c, hr) in zip(values, FEATURE_NORM):
        x = clip(v, c - hr, c + hr)
        x_n = (x - c) / hr
        q = int(round(x_n * SF))
        q = clip(q, -128, 127)
        out.append(q)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("values", nargs="*", type=float)
    ap.add_argument(
        "--stdin",
        action="store_true",
        help="Read one whitespace-separated row per line from stdin "
        "and emit one int8-row per line.",
    )
    args = ap.parse_args()

    if args.stdin:
        for line in sys.stdin:
            row = line.strip().split()
            if not row:
                continue
            q = quantize([float(x) for x in row])
            print(" ".join(str(v) for v in q))
        return

    if not args.values:
        ap.print_help()
        sys.exit(2)
    q = quantize(args.values)
    print(" ".join(str(v) for v in q))


if __name__ == "__main__":
    main()
