#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact device test: sends every golden vector to the
RP2040 over USB CDC and checks logit and probability against the
python reference. Pass criterion: 0/N mismatches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kquity_usb_client import KQuityDevice


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=1.0)
    ap.add_argument(
        "--golden",
        default=str(
            Path(__file__).resolve().parents[1] / "tests" / "golden_vectors.json"
        ),
    )
    ap.add_argument("--max-print", type=int, default=8)
    ap.add_argument(
        "--ping-first",
        action="store_true",
        help="ping/info handshake before sending vectors",
    )
    args = ap.parse_args()

    j = json.loads(Path(args.golden).read_text())
    records = j["records"]
    n = len(records)

    dev = KQuityDevice(args.port, baudrate=args.baud, timeout=args.timeout)
    try:
        if args.ping_first:
            print(dev.ping())
            print(dev.info())

        t0 = time.time()
        logit_miss = 0
        prob_miss = 0
        max_logit_err = 0
        max_prob_err = 0
        for i, rec in enumerate(records):
            feats = rec["features"]
            exp_logit = int(rec["logit"])
            exp_prob = int(rec["prob"])
            logit, prob = dev.score(feats)
            dl = logit - exp_logit
            dp = prob - exp_prob
            if dl:
                logit_miss += 1
                max_logit_err = max(max_logit_err, abs(dl))
                if logit_miss <= args.max_print:
                    print(
                        f"  [{i:3d}] logit mismatch: "
                        f"got={logit} expected={exp_logit} diff={dl:+d}"
                    )
            if dp:
                prob_miss += 1
                max_prob_err = max(max_prob_err, abs(dp))
                if prob_miss <= args.max_print:
                    print(
                        f"  [{i:3d}] prob mismatch: "
                        f"got={prob} expected={exp_prob} diff={dp:+d}"
                    )
        dt = time.time() - t0

        print()
        print(f"=== RP2040 device test ===")
        print(f"N = {n}  ({dt:.2f}s, {n / dt:.1f} vec/s)")
        print(
            f"logit mismatches: {logit_miss}  "
            f"(max abs err = {max_logit_err} LSB Q4.12)"
        )
        print(
            f"prob  mismatches: {prob_miss}  "
            f"(max abs err = {max_prob_err} uint16 LSB)"
        )
        ok = logit_miss == 0 and prob_miss == 0
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    finally:
        dev.close()


if __name__ == "__main__":
    main()
