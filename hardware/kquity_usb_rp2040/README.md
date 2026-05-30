# KQuity win-probability on a $4 USB stick (RP2040)

Run the KQuity win predictor on a Raspberry Pi Pico (RP2040) over USB —
no Python, no LightGBM, no ML stack on the host. You send six int8
features over a USB serial line and the device returns `P(blue wins)`.

The firmware runs the **objective-pressure primitive**: a 6-feature
int8 linear score (6 weights + 1 bias) plus a 1024-entry sigmoid LUT,
extracted from the full LightGBM win predictor. It is **bit-exact** to
the reference implementation (`tests/kquity_pressure_ref.py`) across
all 152 golden vectors. No floats on device, no malloc, no model file
to load.

Why bother: you get the win-probability signal on any machine with a
USB port and `pip install pyserial` — a coin-sized backup that needs
zero ML dependencies. Fidelity vs the full model and a hardware-cost
comparison against an off-the-shelf tree compiler (Conifer) are in
[`COMPARISON.md`](COMPARISON.md).

## Layout

```
firmware/
  main.c                  USB CDC line-protocol loop
  kquity_score.{h,c}      pure-int objective-pressure score
  kquity_constants.h      int8 weights + int16 bias
  kquity_sigmoid_lut.h    1024-entry uint16 sigmoid LUT
  CMakeLists.txt          Pico SDK build
host/
  kquity_usb_client.py    pyserial client (ping / info / score)
  test_device.py          golden-vector bit-exact device test
  quantize_features.py    raw differentials -> int8 features
tests/
  test_native.c           native (no-Pico) bit-exact gate
  golden_vectors.{json,c} 152 vectors (128 random + 24 edge)
  kquity_pressure_ref.py  reference implementation (the exact math)
  run_native_tests.sh     compile-and-test driver
```

## Protocol

```
PING                                  -> KQ1 PONG
INFO                                  -> KQ1 INFO kquity-rp2040 ...
SCORE <e> <f> <s> <sol> <war> <ber>   -> KQ1 OK <int16_logit> <uint16_prob>

unknown line                          -> KQ1 ERR unknown_command
bad SCORE args                        -> KQ1 ERR bad_args
```

The six SCORE arguments are signed int8 decimals in [-128, 127]. The
device returns the logit in Q4.12 (divide by 4096 for a float in
roughly [-8, +8]) and the probability in [0, 65535] (divide by 65535
for `P(blue wins)`).

## Native test (no Pico required)

Proves the int math is bit-exact before you touch hardware:

```
cd hardware/kquity_usb_rp2040/tests
./run_native_tests.sh
```

This compiles `kquity_score.c` with the host gcc and runs it against
the committed 152-vector gate (128 random pool draws + 24 endpoint
clamp probes). Pass criterion: 0/152 logit and 0/152 prob mismatches.

## Build the firmware

One-time Pico SDK setup on the build host:

```
sudo apt install -y cmake gcc-arm-none-eabi build-essential \
                    git python3 python3-pip
git clone https://github.com/raspberrypi/pico-sdk.git ~/pico-sdk
( cd ~/pico-sdk && git submodule update --init )
export PICO_SDK_PATH=~/pico-sdk
```

Build (produces `kquity_usb_rp2040.uf2`):

```
cd hardware/kquity_usb_rp2040/firmware
cmake -S . -B build -DPICO_BOARD=pico
cmake --build build -j
# -> build/kquity_usb_rp2040.uf2  (plus .elf/.bin/.hex siblings)
```

Tested with the Pico SDK 2.x and the ARM GNU toolchain 13.3.

## Flash the Pico

1. Hold the **BOOTSEL** button.
2. Plug into USB while still holding BOOTSEL.
3. The board mounts as a USB drive named `RPI-RP2`.
4. Copy `build/kquity_usb_rp2040.uf2` onto it.
5. It reboots and re-enumerates as `/dev/ttyACM0` (or `/dev/ttyACM1`).

The flash is non-destructive; re-flash as many times as you like.

On Linux you may need serial access:

```
sudo usermod -aG dialout $USER     # then log out and back in
```

## Talk to the device

```
pip install pyserial

python3 host/kquity_usb_client.py --port /dev/ttyACM0 ping     # -> KQ1 PONG
python3 host/kquity_usb_client.py --port /dev/ttyACM0 info
python3 host/kquity_usb_client.py --port /dev/ttyACM0 score 64 8 43 32 -32 84

# full 152-vector bit-exact device test
python3 host/test_device.py --port /dev/ttyACM0 --ping-first
# -> logit mismatches: 0 / prob mismatches: 0 / PASS
```

## Drop-in host class

A minimal wrapper (only dependency: `pyserial`) you can paste into a
project to replace a per-frame `lgbm.predict(...)` call:

```python
import serial


class KQuityStick:
    # affine quantization; must match firmware/kquity_constants.h
    QPARAMS = [
        ("egg_diff",      0.0, 4.0),
        ("food_diff",     0.0, 16.0),
        ("snail_pos",     0.0, 1.0),
        ("soldier_diff",  0.0, 4.0),
        ("warrior_diff",  0.0, 4.0),
        ("berries_avail", 0.7, 0.3),
    ]

    def __init__(self, port="/dev/ttyACM0", timeout=2.0):
        self.ser = serial.Serial(port, timeout=timeout)
        self.ser.write(b"PING\n")
        if "PONG" not in self.ser.readline().decode():
            raise RuntimeError("no PONG from device")

    @staticmethod
    def _q(x, c, hr):
        x = max(c - hr, min(c + hr, x))
        return int(round((x - c) / hr * 128))

    def score(self, egg_diff, food_diff, snail_pos,
              soldier_diff, warrior_diff, berries_avail):
        vals = [egg_diff, food_diff, snail_pos,
                soldier_diff, warrior_diff, berries_avail]
        qs = [self._q(v, c, h) for v, (_, c, h) in zip(vals, self.QPARAMS)]
        self.ser.write(b"SCORE " + b" ".join(str(q).encode() for q in qs) + b"\n")
        parts = self.ser.readline().decode().split()
        if parts[:2] != ["KQ1", "OK"]:
            raise RuntimeError(f"device error: {' '.join(parts)}")
        return int(parts[2]) / 4096.0, int(parts[3]) / 65535.0   # logit, P(blue)
```

`host/quantize_features.py` does the same quantization from the
command line if you want to inspect the intermediate ints.

## Fidelity vs the full LightGBM

The device runs the extracted primitive, not the full ensemble:

- **AUC:** ~4 pp below the full LightGBM oracle on the held-out pool.
- **Brier:** ~1.5 pp worse.
- **Ordering:** the score monotonically tracks the full LightGBM
  probability — wins are ranked the same; magnitudes are slightly
  less calibrated.
- **Output is `P(blue)`**, not `P(gold)`.

If you need the exact full-model probability, keep LightGBM on the
host and use the stick as a cheap backup or a ranking/threshold
signal. See [`COMPARISON.md`](COMPARISON.md) for the hardware-cost
tradeoff vs compiling the trees directly with Conifer.

## Performance

- ~2938 scores/sec over USB CDC; ~340 µs deterministic round trip.
- No GC pause, no Python startup, no model load.
- USB CDC is built into every modern OS — no driver install.

## Caveats

- The quantization constants are folded with each feature's
  half-range and center (`w_eff = w * hr`, `b_eff = b + sum(w*c)`).
  If you refit the primitive, regenerate `kquity_constants.h` and
  `kquity_sigmoid_lut.h` from `tests/kquity_pressure_ref.py` and rerun
  `tests/run_native_tests.sh` before reflashing.
- USB CDC line-rate is unbounded by `pyserial`'s `baud` argument on
  RP2040; the parameter is accepted for compatibility but ignored.

## Troubleshooting

- **Device doesn't enumerate.** Usually a charge-only USB cable — try
  a known data cable / port.
- **Permission denied on `/dev/ttyACM0`.** Linux: add yourself to the
  `dialout` group (above), or `sudo` the test once.
- **Wrong numbers.** Check your host quantization matches
  `firmware/kquity_constants.h`.

## License

Apache 2.0 (`SPDX-License-Identifier: Apache-2.0`), matching the repository.
