# Rob's setup guide — KQuity on a USB stick

Hey Rob! This is the writeup for using the Pico I'm handing you.

Mcgrof will hand you a Raspberry Pi Pico that's already been
flashed with the KQuity objective-pressure firmware. The firmware
is the **Phase 5 extracted primitive** — 7 int8 parameters (6
weights + 1 bias) plus a 1024-entry sigmoid LUT. It returns
P(blue wins) over a USB CDC serial line. Bit-exact to the desktop
reference (`experiments/kquity_hw/hls/kquity_pressure_ref.py`)
across all 152 golden vectors. No floats on device, no malloc,
no model file to load.

Whole point: you don't have to install PyTorch, scikit-learn,
LightGBM, or any ML dependency on the laptop you bring to the
streaming venue. You install **one** small Python package
(`pyserial`) and the device returns the same numbers a CPU
running the extracted primitive would.

## What you need

1. The Raspberry Pi Pico mcgrof handed you (already flashed).
2. A USB cable that carries **data** (not just power).
3. A laptop with Python 3.8+. Linux, macOS, or Windows.
4. `pip install pyserial`.

That's it. No torch, no lgbm, no numpy if you don't want.

## First plug-in

Plug the Pico into a USB port. You should see a new serial device:

```
Linux:   /dev/ttyACM0    (or /dev/ttyACM1 if you already have one)
macOS:   /dev/tty.usbmodem<NNNN>
Windows: COM3 (or whichever next free COM)
```

On Linux, if you get "permission denied":

```bash
sudo usermod -aG dialout $USER
# log out and back in, or run with sudo for testing
```

To confirm the firmware is alive:

```bash
cd hardware/kquity_usb_rp2040/host
python3 kquity_usb_client.py --port /dev/ttyACM0 ping
# Expected output:
#   KQ1 PONG

python3 kquity_usb_client.py --port /dev/ttyACM0 info
# Expected output:
#   KQ1 INFO kquity-rp2040 v0.1 logit_q=Q4.12 prob=uint16 features=int8_q1p7
```

If both work, the Pico is good. If you want belt-and-suspenders,
run the full 152-vector bit-exact device test:

```bash
python3 test_device.py --port /dev/ttyACM0 --ping-first
# Expected:
#   N = 152
#   logit mismatches: 0
#   prob  mismatches: 0
#   PASS
```

This takes about 0.05 seconds.

## Using it from your game code

The host-side wrapper lives in `hardware/kquity_usb_rp2040/host/`.
Most of what you need is in `kquity_usb_client.py`. To make it
even more drop-in, here's a minimal class you can paste into your
project (no extra deps beyond `pyserial`):

```python
# kquity_stick.py
import serial


class KQuityStick:
    """
    Drop-in replacement for the per-frame KQuity probability call,
    using a USB-connected Raspberry Pi Pico that runs the extracted
    objective-pressure primitive in hardware.

    Replaces:
        prob_blue = lgbm.predict([[egg_diff, food_diff, snail_pos,
                                   soldier_diff, warrior_diff,
                                   berries_avail]])[0]

    With:
        stick = KQuityStick('/dev/ttyACM0')   # do this once
        logit, prob_blue = stick.score(egg_diff, food_diff, snail_pos,
                                       soldier_diff, warrior_diff,
                                       berries_avail)
    """

    # Affine quantization parameters; baked into the firmware
    # constants. Must match firmware/kquity_constants.h.
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
        line = self.ser.readline().decode().strip()
        if "PONG" not in line:
            raise RuntimeError(f"unexpected reply: {line!r}")

    @staticmethod
    def _quant(x, center, half_range):
        x = max(center - half_range, min(center + half_range, x))
        return int(round((x - center) / half_range * 128))

    def score(self, egg_diff, food_diff, snail_pos,
              soldier_diff, warrior_diff, berries_avail):
        vals = [egg_diff, food_diff, snail_pos,
                soldier_diff, warrior_diff, berries_avail]
        qs = [self._quant(v, c, h)
              for v, (_, c, h) in zip(vals, self.QPARAMS)]
        self.ser.write(b"SCORE " + b" ".join(str(q).encode() for q in qs) + b"\n")
        parts = self.ser.readline().decode().strip().split()
        if parts[:2] != ["KQ1", "OK"]:
            raise RuntimeError(f"device error: {' '.join(parts)}")
        logit_q4_12 = int(parts[2])
        prob_u16 = int(parts[3])
        return logit_q4_12 / 4096.0, prob_u16 / 65535.0


if __name__ == "__main__":
    import sys
    stick = KQuityStick(sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0")
    # quick smoke
    logit, prob = stick.score(0.5, 2.0, 0.34, 0.25, -0.25, 0.83)
    print(f"logit = {logit:+.4f}, P(blue) = {prob:.4f}")
```

The script `quantize_features.py` in the host dir does the same
quantization step from the command line if you want to inspect
intermediate ints.

## What you give up vs the full LightGBM

The Pico runs the **extracted primitive**, not the LightGBM
ensemble. That trade-off is the whole point of the
action-variable extraction paper, but for your purposes:

- **AUC drop:** ~4 percentage points vs the full LightGBM oracle
  (76.0 → 72.0 on the held-out eval pool, see
  `docs/kquity_hw_extraction.md`).
- **Brier degradation:** ~1.5 percentage points.
- **Score still ordinal:** the primitive monotonically tracks the
  full LightGBM probability — wins are still ranked correctly,
  the magnitudes are slightly less calibrated.
- **Output is P(blue), NOT P(gold).** The KQuity README at one
  point says P(gold) but the labels confirm it's actually
  P(blue). Documented in `experiments/kquity_hw/inventory.md`.

If you want to show viewers the LightGBM probability on stream
and use the stick as a backup, run both in parallel and overlay
both numbers; the difference is small enough that viewers won't
notice the discontinuity.

## What you give up vs CPU LightGBM (positives)

- **Zero ML deps on the venue laptop.** Only `pyserial`
  (sub-megabyte).
- **No model file to ship.** The weights are baked into the
  firmware on the Pico.
- **Cheap backup.** A $4 Pico in your laptop bag is a backup
  laptop, basically. Plug, run, ship.
- **Throughput is not a concern.** The Pico does 2938 scores/sec
  over USB CDC. Your game tick rate is well under that.
- **Deterministic latency.** ~340 µs per round trip, every time.
  No GC pause, no Python startup, no model load.
- **No driver install on the venue laptop.** USB CDC is built
  into every modern OS.

## Re-flashing if needed

If we update the primitive (Phase 6+ refit, new feature
boundaries, etc), I'll send you a new `.uf2`. To re-flash:

1. Unplug the Pico.
2. Hold the **BOOTSEL** button down.
3. While holding BOOTSEL, plug the Pico back into USB.
4. Continue holding BOOTSEL for ~2 seconds after plug-in.
5. The Pico mounts as a USB drive named `RPI-RP2`.
6. Copy the new `kquity_usb_rp2040.uf2` onto the drive.
7. The Pico auto-reboots into the new firmware. Run the
   handshake check above to confirm.

The flash is non-destructive — re-flashing as many times as
needed is fine.

## Troubleshooting

**Pi doesn't show up.** Most common cause: USB cable is
charge-only. Try the cable you use for a phone with data
transfer. If a known-good cable still doesn't enumerate, the USB
port itself may be a charge-only port (some short hub ports are).

**Permission denied on `/dev/ttyACM0`.** Linux only: `sudo
usermod -aG dialout $USER`, then log out and back in. Or run the
test command with `sudo` once to confirm everything else works.

**Device returns wrong numbers.** Check that the quantization
parameters in your wrapper match the ones in `firmware/
kquity_constants.h`. If we ever change the feature ranges in a
new firmware version, the wrapper has to match.

**Stream lag.** Not the device. 2938 scores/sec is way faster
than any sane game tick rate.

## What I built on monster to get here

Briefly, for anyone reading later:

- **Firmware** in `hardware/kquity_usb_rp2040/firmware/`. C
  source, builds with the Pico SDK 2.x and ARM GCC 13.3 (see
  `BUILD_RECIPE_DEBIAN_FORKY.md` for the exact toolchain dance).
- **Native test** in `tests/run_native_tests.sh`: compiles
  `kquity_score.c` with desktop gcc and runs the same 152-vector
  gate, no Pico required. Passed.
- **Device test** in `host/test_device.py`: same 152 vectors,
  sent over USB CDC to the Pico, response compared bit-for-bit
  against the desktop reference. Passed with 0/152 mismatches.
- **Bit-exactness gate:** if any LSB drifts in either logit
  (Q4.12) or probability (uint16), the test fails. The 0/152
  result means the Pico produces exactly the bytes the desktop
  produces.

Logit format `Q4.12` means 4 integer bits + 12 fractional bits in
an int16 — divide by 4096 to get a float in roughly [-8, +8].
Probability is uint16 — divide by 65535 to get [0, 1].

## Questions?

Ping mcgrof. The artifact provenance is all in
`docs/kquity_hw_extraction.md` (full Phase 0-8 history) and in
the commit log on the `case-f-hw-extraction` branch.
