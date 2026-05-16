# Case F — KQuity Phase 5 HLS report

The 7-parameter linear primitive extracted from the LightGBM
win-probability oracle (see Phase 2 / Phase 4) is implemented in
HLS C++ and synthesized for KV260 (xck26 Zynq UltraScale+ MPSoC).
The testbench is bit-exact against the python pure-int reference
(`kquity_pressure_ref.py`) on 152 vectors (128 random + 24
endpoint clamp probes).

## 1. Primitive summary

Inputs (six int8 features in Q1.7 after per-feature normalization
on host/PS; this is the *objective-pressure vector*):

```
egg_diff       Q1.7   center=0.0  half_range=4.0
food_diff      Q1.7   center=0.0  half_range=16.0
snail_pos      Q1.7   center=0.0  half_range=1.0
soldier_diff  Q1.7   center=0.0  half_range=4.0
warrior_diff  Q1.7   center=0.0  half_range=4.0
berries_norm  Q1.7   center=0.7  half_range=0.3
```

Folded weights (int8 Q3.5, half_range absorbed) and bias (int16
in accumulator scale Q4.12, feature centers absorbed):

```
W_EGG     =  92    /32 = +2.875    (float w_eff =  +2.869)
W_FOOD    = 114    /32 = +3.5625   (float w_eff =  +3.575)
W_SNAIL   =  27    /32 = +0.84375  (float w_eff =  +0.850)
W_SOLDIER =  63    /32 = +1.96875  (float w_eff =  +1.954)
W_WARRIOR =  36    /32 = +1.125    (float w_eff =  +1.115)
W_BERRIES =  -1    /32 = -0.03125  (float w_eff =  -0.025)
B_ACC     = 118    /4096 = +0.02881 (folded bias)
```

Output: int16 in Q4.12 (Sf*Sw = 4096), saturated to
[-32768, 32767]. This is the *objective-pressure score*; observed
test-pool logit range is roughly [-5, +5] in float, well inside
the saturation limit.

Optional probability wrapper: 1024-entry uint16 sigmoid LUT,
yielding `win-probability LUT` output in [0, 65535] (uint16 / 65535
= P(blue wins)).

## 2. Fixed-point parity recap

All quality loss is the linear-vs-LightGBM *model-class gap*, not
quantization:

```
quantity                                 metric
LightGBM oracle (100t x 100l, 52 feat)   AUC 0.7901, Brier 0.1868
                                          ECE10 0.0543
Float linear surrogate (6 feat, 7 par)   AUC 0.7495, Brier 0.2017
                                          ECE10 0.0519
                                          (vs oracle: -4.06 pp AUC,
                                           +1.50 pp Brier)
Int8 pure-int primitive (Phase 5 HLS)    AUC 0.7496, Brier 0.2017
                                          ECE10 0.0531
                                          (vs float surrogate:
                                           pRMSE 0.0012, AUC delta
                                           +0.0001)
```

The fixed-point primitive is statistically indistinguishable from
the float linear surrogate. Every percentage point of the 4.06 pp
AUC drop is the cost of going from a 100-tree LightGBM to a
6-feature linear; **zero** is the cost of going from float to
int8 hardware. That is the deployment story.

## 3. HLS implementation

```
experiments/kquity_hw/hls/
  kquity_pressure_hls.h           HLS interface (typedefs)
  kquity_pressure_hls.cpp          int dot product + LUT wrapper
  kquity_constants.h               generated from phase4 JSON
  kquity_sigmoid_lut.h             generated 1024-entry uint16 LUT
  kquity_pressure_ref.py           python pure-int reference
  gen_golden.py                    generates testbench golden vectors
  golden_kquity_pressure.h         152 inputs/outputs from python ref
  test_kquity_pressure.cpp         C++ testbench (bit-exact gate)
  run_kquity_hls.tcl               Vitis HLS 2025.2 csynth driver
  kquity_logit_only/...           csynth output, logit-only top
  kquity_with_lut/...             csynth output, logit + LUT top
```

Build/run:

```
python3 hls/kquity_pressure_ref.py     # regenerate constants header
python3 hls/gen_golden.py              # regenerate golden vectors
cd hls && /tools/Xilinx/2025.2/Vitis/bin/vitis_hls -f run_kquity_hls.tcl
```

The C++ also compiles with plain `g++`; running the testbench
natively confirms 0/152 logit and prob mismatches before invoking
the toolchain. That is the inspection path Ryan can run with no
Vitis license.

## 4. Resource tables

### 4.1 Logit-only (`kquity_logit_q8`)

```
                LUT     FF    DSP    BRAM_18K   URAM
DSP             -        -      2          -      -
Expression      165      0      -          -      -
Instance         40      0      0          -      -
Register         64    180      -          -      -
+--------------+------+------+------+---------+------+
Total           269    180      2          0      0
KV260 budget  117120 234240   1248        288     64
Utilization    0.23% 0.08%  0.16%       0.00%  0.00%
```

```
Latency:           4 cycles  (20.000 ns at 5 ns target clock)
Pipeline II:       1
Estimated Fmax:    288.9 MHz   (timing: 3.461 ns of 5.00 ns budget)
```

Vitis bound 5 of the 6 multiplies onto 2 DSP MACs
(`mac_muladd_8s_5ns_16s_16` and `mac_muladd_8s_7ns_16s_16`) and
fabric-mapped the 6th (`mul_8s_8ns_16` -> 40 LUT). The mix is
sane; HLS chose it without any `BIND_OP impl=fabric` hints. The
remaining LUTs are saturation logic + a few adders.

### 4.2 Logit + sigmoid LUT (`kquity_pressure`)

```
                LUT     FF    DSP    BRAM_18K   URAM
Expression      2        0      -          -      -
Instance       285    242      2          1      -
Register        -     41      -          -      -
+--------------+------+------+------+---------+------+
Total           287    283      2          1      0
KV260 budget  117120 234240   1248        288     64
Utilization    0.24% 0.12%  0.16%       0.35%  0.00%
```

```
Latency:           7 cycles  (35.000 ns total)
  logit subblock:  4 cycles
  prob subblock:   1 cycle
Pipeline II:       1
Estimated Fmax:    288.9 MHz
```

The sigmoid LUT (1024 entries x 16 bits = 16 Kbit) fits in 1
BRAM_18K tile. The wrapper costs ~18 extra LUT and ~100 extra FF
on top of the logit core. Adding the probability output is
essentially free.

## 5. Schematic for Ryan

```
  blue.eggs, blue.food, blue.n_sol, blue.n_war  (host)
  gold.eggs, gold.food, gold.n_sol, gold.n_war  (host)
  snail_pos, berries_avail                       (host)
              |
              v
     per-feature differential + normalize (host or PS)
              |
              v
              +------------------------------------+
              | int8 features (Q1.7):              |
              |   egg, food, snail, sol, war, ber  |
              +-----------+------------------------+
                          |
                          v
              +------------------------------+
              | kquity_logit_q8 (PL):        |
              |   acc = B_ACC                |
              |       + W_EGG  * egg         |
              |       + W_FOOD * food        |
              |       + W_SNAIL* snail       |
              |       + W_SOL  * sol         |
              |       + W_WAR  * war         |
              |       + W_BER  * ber         |
              |   return sat16(acc)          |
              +-----------+------------------+
                          |
              int16 Q4.12 logit  ("objective-pressure score")
              ~500 LUT / 2 DSP / 4 cyc / Fmax 289 MHz
                          |
              +-----------v------------------+
              | kquity_prob_lut (optional):  |
              |   idx = clip((logit+32768)>>6,|
              |              0, 1023)         |
              |   return lut[idx]             |
              +-----------+------------------+
                          |
              uint16 sigmoid probability
              +18 LUT / +1 BRAM / +1 cyc
```

KV260 budget left after fully populated: ~99.65% LUT, 99.92% FF,
99.84% DSP, 99.65% BRAM. The primitive is so small the chip is
essentially a coaster.

## 6. Logit-only vs probability-LUT comparison

```
config             cycles   ns@5ns    LUT    FF   DSP   BRAM    Fmax
logit-only           4       20.0    269   180    2      0    288.9 MHz
logit + prob LUT     7       35.0    287   283    2      1    288.9 MHz
```

The raw int16 logit is the real control signal: it is monotonic,
preserves ranking, and is what hardware would actually threshold
to take action. The sigmoid LUT wrapper exists to display a
human-friendly probability and costs essentially nothing if you
want it.

## 7. Testbench result

```
N = 152 (128 random pool draws + 24 endpoint clamp probes)
logit mismatches: 0 (max abs err = 0 LSB Q4.12)
prob  mismatches: 0 (max abs err = 0 uint16 LSB)
PASS
```

Bit-exact against `kquity_pressure_ref.py` on every vector
including the per-feature min/max/over-cap clamp probes (24
edge inputs). This is the only fidelity claim that matters at
this layer.

## 8. Caveats

- The primitive is a deployable hardware-native control signal,
  not a calibrated probability replacement for the LightGBM. AUC
  drop 4.06 pp; Brier degradation 1.5 pp; ECE essentially
  preserved. If a use case requires full LightGBM probability
  fidelity, this primitive is not it.
- Train/eval used the 16 `tests/benchmark_events_*.csv.gz` shards
  (~3,000 games, 233k events). The full encoded datasets referenced
  in `CLAUDE.md` are not local; if the surrogate's family/skill
  coverage matters for the friend demo, re-encode and re-fit (one
  CPU hour). Phase 4 fidelity should not move materially.
- KQuity README describes the model as "P(gold wins)"; both the
  labels and all six monotonic probes confirm it is P(blue wins).
  Recorded in `inventory.md`.
- HLS synthesis is post-`csynth` only; place-and-route/IP packaging
  numbers may shift slightly but the magnitudes are so small the
  schematic claim is robust.

## 9. What this is and what it is not

It is:

```
a hardware-native objective-pressure signal extracted from the
LightGBM win-probability oracle, costing ~500 LUT / 2 DSP /
4-7 cycles / Fmax >250 MHz on KV260, with 0% cost from
quantization and 4 pp AUC cost from the linear vs LightGBM
model-class simplification.
```

It is not:

```
a probability-faithful replacement for the LightGBM oracle. If
Ryan wants the full probability surface for analysis or display,
the LightGBM stays; the hardware primitive is for the
ranking/threshold/control path.
```

That distinction is the whole point of action-variable extraction.

## 10. Files / commits

```
branch case-f-hw-extraction
  07e8bed  Phase 0/1 baseline
  0133f14  Phase 2 surrogate ladder
  4cb1a57  Phase 2 tiny-LGBM diagnostic
  ca2a345  Phase 2.5 Route B interactions: NOT accepted
  e04953f  Phase 4 fixed-point parity
  (pending) Phase 5 HLS bundle + csynth artifacts
```
