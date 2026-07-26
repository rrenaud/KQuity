#!/usr/bin/env python3
"""Host-side USB-serial client for the KQuity RP2040 firmware.

Subcommands:
  ping
  info
  score e f s sol war ber       inputs are int8 already-quantized

For an unquantized input (raw float game-state differentials),
quantize first with `quantize_features.py` then pass the int8 row
to `score`. The device never sees floats.
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    print(
        "pyserial is required: pip install pyserial",
        file=sys.stderr,
    )
    raise SystemExit(2)


class KQuityDevice:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        # baudrate is ignored on RP2040 USB CDC but pyserial requires it.
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.05)
        # Drain any banner residue.
        self.ser.reset_input_buffer()

    def close(self):
        self.ser.close()

    def cmd(self, line: str) -> str:
        if not line.endswith("\n"):
            line += "\n"
        self.ser.write(line.encode("ascii"))
        self.ser.flush()
        resp = self.ser.readline().decode("ascii", errors="replace").strip()
        return resp

    def ping(self) -> str:
        r = self.cmd("PING")
        if not r.startswith("KQ1 PONG"):
            raise RuntimeError(f"unexpected PING response: {r!r}")
        return r

    def info(self) -> str:
        r = self.cmd("INFO")
        if not r.startswith("KQ1 INFO"):
            raise RuntimeError(f"unexpected INFO response: {r!r}")
        return r

    def score(self, feats: list[int]) -> tuple[int, int]:
        if len(feats) != 6:
            raise ValueError("expected 6 int8 features")
        for v in feats:
            if not (-128 <= int(v) <= 127):
                raise ValueError(f"feature out of int8 range: {v}")
        line = "SCORE " + " ".join(str(int(v)) for v in feats)
        r = self.cmd(line)
        parts = r.split()
        if len(parts) != 4 or parts[0] != "KQ1" or parts[1] != "OK":
            raise RuntimeError(f"unexpected SCORE response: {r!r}")
        return int(parts[2]), int(parts[3])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=1.0)

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping")
    sub.add_parser("info")
    p_score = sub.add_parser("score")
    p_score.add_argument("e", type=int)
    p_score.add_argument("f", type=int)
    p_score.add_argument("s", type=int)
    p_score.add_argument("sol", type=int)
    p_score.add_argument("war", type=int)
    p_score.add_argument("ber", type=int)

    args = ap.parse_args()
    dev = KQuityDevice(args.port, baudrate=args.baud, timeout=args.timeout)
    try:
        if args.cmd == "ping":
            print(dev.ping())
        elif args.cmd == "info":
            print(dev.info())
        elif args.cmd == "score":
            logit, prob = dev.score(
                [args.e, args.f, args.s, args.sol, args.war, args.ber]
            )
            print(
                f"logit_q4_12 = {logit}    "
                f"float_logit = {logit / 4096.0:+.4f}    "
                f"prob_u16 = {prob}    "
                f"prob = {prob / 65535.0:.4f}"
            )
    finally:
        dev.close()


if __name__ == "__main__":
    main()
