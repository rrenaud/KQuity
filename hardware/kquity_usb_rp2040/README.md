# KQuity USB firmware for Raspberry Pi Pico / RP2040

Bit-exact port of the Phase 5 objective-pressure primitive to
RP2040. The board accepts six int8 features over USB CDC serial
and returns an int16 Q4.12 logit plus a uint16 sigmoid
probability. No floats on device. No malloc. No vibes.

The point: hand Rob a coin-sized stick that returns the same
P(blue wins) signal the FPGA design would, so the demo works on
any computer with a USB port.

## Layout

```
firmware/
  main.c                  USB CDC line-protocol loop
  kquity_score.{h,c}      pure-int objective-pressure score
  kquity_constants.h      int8 weights + int16 bias (Phase 5)
  kquity_sigmoid_lut.h    1024-entry uint16 sigmoid LUT (Phase 5)
  CMakeLists.txt          Pico SDK build
host/
  kquity_usb_client.py    pyserial client (ping / info / score)
  test_device.py          golden-vector bit-exact device test
  quantize_features.py    raw differentials -> int8 features
tests/
  gen_golden_json.py      regenerates golden_vectors.{json,c}
  golden_vectors.json     152 vectors (128 random + 24 edge)
  golden_vectors.c        same data as C arrays for test_native.c
  test_native.c           native (no-Pico) bit-exact gate
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

All six SCORE arguments are signed int8 decimals in [-128, 127].
The device returns logit in Q4.12 (divide by 4096 for float) and
probability in [0, 65535] (divide by 65535 for P(blue wins)).

## Native test (no Pico required)

```
cd hardware/kquity_usb_rp2040/tests
./run_native_tests.sh
```

This compiles `kquity_score.c` with desktop gcc, regenerates the
golden vectors from `kquity_pressure_ref.py`, and runs
`test_native` against 152 inputs (128 random pool draws + 24
endpoint clamp probes covering each feature's min/max + over-cap
clamp). Pass criterion: 0/152 logit and 0/152 prob mismatches.

This is what we run before the board lands.

## Pico SDK setup

One-time, on the host that will build the UF2:

```
sudo apt install -y cmake gcc-arm-none-eabi build-essential \
                    git python3 python3-pip

mkdir -p ~/devel/rp2040
cd ~/devel/rp2040
git clone https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init

export PICO_SDK_PATH=~/devel/rp2040/pico-sdk
```

(Adjust path to taste.)

## Build the firmware

```
cd hardware/kquity_usb_rp2040/firmware
cmake -S . -B build -DPICO_BOARD=pico
cmake --build build -j
```

Output:

```
firmware/build/kquity_usb_rp2040.uf2
```

(plus `.elf`, `.bin`, `.hex` siblings)

## Flashing the Pico

1. Hold the **BOOTSEL** button on the Pico.
2. Plug it into USB while still holding BOOTSEL.
3. The board appears as a USB mass-storage drive named `RPI-RP2`.
4. Copy `build/kquity_usb_rp2040.uf2` onto the drive.
5. The board reboots and re-enumerates as `/dev/ttyACM0`
   (or `/dev/ttyACM1`).

On Linux you may need dialout group access:

```
sudo usermod -aG dialout $USER     # log out and back in
```

(or just `sudo` the first test command).

## Run the device test

```
pip install pyserial

# quick handshake
python3 hardware/kquity_usb_rp2040/host/kquity_usb_client.py \
        --port /dev/ttyACM0 ping
python3 hardware/kquity_usb_rp2040/host/kquity_usb_client.py \
        --port /dev/ttyACM0 info

# one-shot
python3 hardware/kquity_usb_rp2040/host/kquity_usb_client.py \
        --port /dev/ttyACM0 score 64 8 43 32 -32 84

# full 152-vector bit-exact test
python3 hardware/kquity_usb_rp2040/host/test_device.py \
        --port /dev/ttyACM0 --ping-first
```

Expected:

```
KQ1 PONG
KQ1 INFO kquity-rp2040 v0.1 logit_q=Q4.12 prob=uint16 features=int8_q1p7

=== RP2040 device test ===
N = 152  (...s, ... vec/s)
logit mismatches: 0  (max abs err = 0 LSB Q4.12)
prob  mismatches: 0  (max abs err = 0 uint16 LSB)
PASS
```

## Going from raw game state to int8 features

The device expects pre-quantized features. The host quantizer:

```
python3 hardware/kquity_usb_rp2040/host/quantize_features.py \
        2 1 0.34 1 -1 0.83
# -> 64 8 43 32 -32 84
```

Per-feature affine (matches `firmware/kquity_constants.h`):

```
egg_diff      center=0.0   half_range=4.0     int8 = round((x-0)/4   * 128)
food_diff     center=0.0   half_range=16.0    int8 = round((x-0)/16  * 128)
snail_pos     center=0.0   half_range=1.0     int8 = round((x-0)/1   * 128)
soldier_diff  center=0.0   half_range=4.0     int8 = round((x-0)/4   * 128)
warrior_diff  center=0.0   half_range=4.0     int8 = round((x-0)/4   * 128)
berries_avail center=0.7   half_range=0.3     int8 = round((x-0.7)/0.3 * 128)
```

All inputs are clipped to `[center - half_range, center + half_range]`
before quantization. snail_pos comes from the KQuity feature
materializer's gold-symmetric `snail_x / SCREEN_WIDTH - 0.5`; the
+/-1 clip handles the rare snail_vel projection outliers seen in
the eval pool.

## Caveats

- The device is a deployable hardware-native control signal, not
  a calibrated probability replacement for the LightGBM oracle
  (AUC drop ~4 pp, Brier degradation ~1.5 pp; see
  `docs/kquity_hw_extraction.md` for the full fidelity table).
- The constants are folded with feature half-ranges and centers
  (`w_eff_i = w_i * hr_i`, `b_eff = b + sum_i w_i * c_i`). If
  Phase 4 refits, regenerate `kquity_constants.h` and
  `kquity_sigmoid_lut.h` from
  `experiments/kquity_hw/hls/kquity_pressure_ref.py` and rerun
  `tests/run_native_tests.sh` before reflashing.
- The model output is P(blue wins). The KQuity README says
  P(gold wins); the labels and all six monotonic probes confirm
  it is in fact P(blue). Documented in
  `experiments/kquity_hw/inventory.md`.
- USB CDC line-rate is unbounded by `pyserial`'s `baud` argument
  on RP2040; the parameter is accepted for compatibility but
  ignored by the device.
