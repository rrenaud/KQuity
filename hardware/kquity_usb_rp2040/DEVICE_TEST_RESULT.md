# RP2040 device test — first board (2026-05-21)

Hardware: Raspberry Pi Pico (RP2040), USB CDC at `/dev/ttyACM0`,
build host monster (Debian forky), ARM GNU Toolchain 13.3.Rel1 +
pico-sdk depth-1 main.

## Handshake

```
KQ1 PONG
KQ1 INFO kquity-rp2040 v0.1 logit_q=Q4.12 prob=uint16 features=int8_q1p7
```

## One-shot score (matches host kquity_pressure_ref.py exactly)

```
score 64 8 43 32 -32 84
-> logit_q4_12 = 8859    float_logit = +2.1628
-> prob_u16    = 58783   prob = 0.8970
```

## 152-vector bit-exact device test (128 random pool + 24 endpoint)

```
=== RP2040 device test ===
N = 152  (0.05s, 2938.6 vec/s)
logit mismatches: 0  (max abs err = 0 LSB Q4.12)
prob  mismatches: 0  (max abs err = 0 uint16 LSB)
PASS
```

Throughput on the Pi: **2938 scores/sec over USB CDC**. The
device handshake + ping round-trip dominates; raw arithmetic is
in the nanoseconds.

## Sizes (post-link, gcc-arm-none-eabi 13.3.Rel1, -O2)

```
text    data    bss     dec     hex     filename
46956   0       4412    51368   c8a8    kquity_usb_rp2040.elf

UF2: 86 KB (Pico has 2 MB flash; 4.2% used)
```

## End-to-end claim

The coin-sized USB stick returns the same P(blue wins) signal as
the LightGBM oracle would, bit-exact across all 152 golden vectors
(see `tests/run_native_tests.sh` for the desktop-side gate that
generates and validates these vectors). The path is:

```
6 raw game-state features
  -> int8 quantization via host/quantize_features.py
  -> 6 int8 over USB CDC line "SCORE e f s sol war ber"
  -> int8 dot product (7 params: 6 weights + 1 bias)
  -> int16 Q4.12 logit
  -> 1024-entry sigmoid LUT -> uint16 probability
  -> "KQ1 OK <logit> <prob>" over USB CDC
```

No floats on device. No malloc. No vibes. Phase 5 KQuity
objective-pressure primitive, bit-exact compiled to RP2040 silicon.
