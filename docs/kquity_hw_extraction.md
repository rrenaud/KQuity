# KQuity hardware extraction summary

A short writeup of a side experiment: extracting a tiny
hardware-native control signal from the KQuity win-probability
LightGBM. The result is a six-feature, seven-parameter int8
linear primitive that approximates the 100-tree / 100-leaf
oracle, synthesizes to ~270 LUT and 2 DSP on KV260, and is
bit-exact between a pure-int python reference and the Vitis HLS
C++ implementation.

This is **not** a drop-in calibrated probability replacement for
the LightGBM. It is a hardware-friendly objective-pressure
control signal extracted from the oracle.

## Contents

- [1. Starting oracle](#1-starting-oracle)
- [2. Extracted action variable](#2-extracted-action-variable)
- [3. Primitive](#3-primitive)
- [4. Fidelity table](#4-fidelity-table)
- [5. HLS resource table](#5-hls-resource-table)
- [6. Schematic](#6-schematic)
- [7. How to reproduce](#7-how-to-reproduce)
- [8. Caveats](#8-caveats)
- [9. Direct model-to-fabric baseline](#9-direct-model-to-fabric-baseline)

## 1. Starting oracle

`current_preferred_model.mdl` (symlink to
`model_experiments/combined_li_qf/qf_200k_symaug_100l_100t.mdl`)
is a LightGBM booster trained on quality-filtered game event
streams. It is `100 trees × 100 leaves × 52 input features`,
output is `P(blue wins)`.

**README mismatch note.** The repo README describes the model
as `P(gold wins)`. In `fast_materialize._process_game` the label
is `1 if last_vals[0] == 'Blue' else 0`, and all six monotonic
sanity probes (gold.eggs↑→p↓, blue.food↑→p↑, snail toward
gold→p↓, etc.) confirm the output is `P(blue wins)`. The probe
script is at `experiments/kquity_hw/baseline_eval.py` and the
finding is documented in `experiments/kquity_hw/inventory.md`.

Top features by gain on the preferred model (first 5 cover
**74.9%** of total gain, first 11 cover **91.1%**):

```
rank  feature              gain%   cum%
[ 0]  gold.eggs           19.84   19.84
[ 1]  blue.eggs           18.10   37.95
[ 2]  blue.food_count     13.37   51.32
[ 3]  gold.food_count     13.17   64.49
[ 4]  snail_pos           10.37   74.87
[ 5]  gold.n_soldiers      3.82   78.69
[ 6]  blue.n_soldiers      3.63   82.32
```

The three Killer Queen win conditions (eggs / berries / snail)
account for ~75% of model gain on their own.

## 2. Extracted action variable

The 52-feature input collapses cleanly to a six-feature
*objective-pressure vector*, defined as gold-vs-blue
differentials over the win conditions plus a combat
differential and a global resource pressure:

```
egg_diff       blue.eggs       - gold.eggs            (queen-lives lead)
food_diff      blue.food_count - gold.food_count      (berry lead)
snail_pos      gold-symmetric snail position
soldier_diff   blue.n_soldiers - gold.n_soldiers     (combat lead)
warrior_diff   blue.n_warriors - gold.n_warriors     (combat lead)
berries_avail  berries_avail   / 70.0                 (resource context)
```

`snail_pos` is the float feature
`(snail_x / SCREEN_WIDTH - 0.5) * gold_sym` already exported by
`fast_materialize`; positive means snail moving toward blue's
goal, regardless of left/right map orientation.

These six terms, fit linearly to the oracle's logit, achieve
within 5% of the prob-RMSE of the full 52-feature linear model
and exceed any depth-2/3/4/5 decision tree and any 3-D LUT
tested. They are the right primitive for this oracle.

A Route-B sweep with eleven candidate interaction terms
(`egg*food`, `egg*snail`, `food*snail`, `soldier*warrior`,
`soldier*snail`, `warrior*snail`, `ber*food`, `ber*egg`,
`phase*egg`, `phase*food`, `phase*snail`) tightens prob-RMSE by
5.8% at best and AUC/Brier essentially zero — far below the 25%
improvement gate needed to justify the added complexity. The
linear-in-6-differentials story is the right level of
simplification.

## 3. Primitive

Seven parameters: one bias plus six folded weights. Float and
int8 quantizations:

```
                 float w     float w*half_range     int8 w (Q3.5)
egg_diff        +0.71720       +2.8688                +92
food_diff       +0.22346       +3.5753               +114
snail_pos       +0.84968       +0.8497                +27
soldier_diff    +0.48848       +1.9539                +63
warrior_diff    +0.27873       +1.1149                +36
berries_avail   -0.08323       -0.0250                 -1

intercept       +0.08716   ->  +0.02890 (folded)   b_acc = +118
                                                   (int16, Q4.12)
```

`b_acc` is stored at accumulator scale `Sf*Sw = 128*32 = 4096`
(Q4.12). The bias detail matters because the feature centers
fold in: `b_eff = b + Σ w_i * center_i`. Only `berries_avail`
has a nonzero center (`0.7`), so the folding shifts the bias
from `+0.087` to `+0.029`. Mishandling this is exactly where
fixed-point primitives go quietly wrong.

Quantization plan, written out (HLS implementation is bit-exact
to this):

```
feature normalization (host or PS):
  x_norm = clip( (x - center) / half_range, -1, 1 )
  x_int8 = clip( round(x_norm * 128), -128, 127 )

per-feature half_range / center:
  egg_diff       hr=4    c=0.0
  food_diff      hr=16   c=0.0
  snail_pos      hr=1    c=0.0
  soldier_diff   hr=4    c=0.0
  warrior_diff   hr=4    c=0.0
  berries_avail  hr=0.3  c=0.7

weight quantization (compile time):
  w_eff_i = w_i * half_range_i          (fold hr into w)
  w_int_i = clip( round(w_eff_i * 32), -128, 127 )   (Q3.5)

bias quantization (compile time):
  b_eff   = b + Σ_i w_i * center_i      (fold centers into b)
  b_acc   = round( b_eff * 4096 )       (int16 in Q4.12)

forward pass:
  acc  = b_acc + Σ_i (w_int_i * x_int_i)              (int32)
  logit_q4_12 = saturate_int16(acc)                   (int16 output)

optional probability wrapper (1024-entry uint16 sigmoid LUT):
  idx = clip( (logit_q4_12 + 32768) >> 6, 0, 1023 )
  prob_u16 = sigmoid_lut[idx]
```

The output is the **objective-pressure score** in Q4.12 (divide
by 4096 to get the float logit). The optional uint16 wrapper is
the **win-probability LUT**; `prob_u16 / 65535 = P(blue wins)`.

## 4. Fidelity table

Event-level metrics on a held-out 600-game / 46k-event slice of
the test shards. The deployable claim is the bottom row:

```
                                         AUC      Brier     ECE10
LightGBM oracle (100t × 100l, 52 feat)   0.7901   0.1868   0.0543
Float linear surrogate (6 feat, 7 par)   0.7495   0.2017   0.0519
Int8 HLS primitive (Q1.7 / Q3.5)         0.7496   0.2017   0.0531

Delta oracle  -> float surrogate         -4.06pp  +1.50pp  -0.24pp
Delta float   -> int8 HLS                +0.01pp  +0.00pp  +0.12pp
```

The interesting decomposition: **the entire quality cost is the
linear-vs-100-tree-LightGBM model-class gap; the int8 hardware
quantization adds essentially zero on top.** The int8 HLS
primitive's pRMSE vs the float surrogate is `0.0016` — below the
LSB of any reasonable quantization. The pRMSE vs the oracle is
the same `0.1130` regardless of whether you use float or int8.

## 5. HLS resource table

Vitis HLS 2025.2, target `xck26-sfvc784-2LV-c` (KV260 Zynq
UltraScale+ MPSoC), 5 ns target clock. Numbers are post-csynth.

```
                       LUT     FF    DSP   BRAM_18K   cycles   Fmax
logit-only             269    180     2        0        4    288.9 MHz
logit + sigmoid LUT    287    283     2        1        7    288.9 MHz

KV260 budget        117120  234240  1248      288
Utilization (LUT)    0.23%   0.24%
```

Pipeline `II = 1` in both. Vitis bound 5 of 6 multiplies to two
DSP MACs (`mac_muladd_8s_5ns_16s_16` and
`mac_muladd_8s_7ns_16s_16`) and put the 6th in fabric (40 LUT).
The choice was automatic; no `BIND_OP impl=fabric` hints were
needed. The sigmoid LUT fits in a single 18 Kb BRAM tile
(`1024 × 16 bits = 16 Kbit`).

Latency for the logit-only path is `4 cycles × 5 ns = 20 ns`.
Adding the probability wrapper costs 3 more cycles. Both are
small enough that a fully-populated chip would have ~99.6% of
its resources left over.

The bit-exact testbench (`152` vectors: 128 random pool draws +
24 endpoint clamp probes covering each feature's min/max and
over-cap clamp) passes with `0/152` logit and `0/152` probability
mismatches on both native g++ and Vitis HLS C-simulation.

## 6. Schematic

```
   blue.eggs, blue.food, blue.n_sol, blue.n_war      (PS/host)
   gold.eggs, gold.food, gold.n_sol, gold.n_war      (PS/host)
   snail_pos, berries_avail                           (PS/host)
                 |
                 v   per-feature differential + normalize (PS or HLS)
                 v
                 +-----------------------------+
                 | int8 features (Q1.7):       |
                 |   egg, food, snail,         |
                 |   sol, war, ber             |
                 +-------------+---------------+
                               |
                               v
                 +-----------------------------+
                 | kquity_logit_q8 (PL):       |
                 |   acc = B_ACC               |
                 |       + W_EGG  * egg        |
                 |       + W_FOOD * food       |
                 |       + W_SNAIL* snail      |
                 |       + W_SOL  * sol        |
                 |       + W_WAR  * war        |
                 |       + W_BER  * ber        |
                 |   return sat16(acc)         |
                 +-------------+---------------+
                               |
                               v
                  int16 Q4.12 logit
                  ("objective-pressure score")
                  ~270 LUT / 2 DSP / 4 cyc / 289 MHz
                               |
                 +-------------v---------------+
                 | kquity_prob_lut (optional): |
                 |   idx = clip(               |
                 |     (logit + 32768) >> 6,   |
                 |     0, 1023                 |
                 |   )                         |
                 |   return lut[idx]           |
                 +-------------+---------------+
                               |
                               v
                  uint16 sigmoid probability
                  ("win-probability LUT" output)
                  +18 LUT / +1 BRAM / +1 cyc
```

## 7. How to reproduce

Environment:

```bash
python3 -m venv ~/envs/kquity
source ~/envs/kquity/bin/activate
pip install lightgbm numpy pandas scikit-learn scipy
```

Phases 0–4 (CPU only, <5 minutes total):

```bash
cd /path/to/KQuity
python3 experiments/kquity_hw/baseline_eval.py       # Phase 0/1
python3 experiments/kquity_hw/surrogate_ladder.py    # Phase 2
python3 experiments/kquity_hw/route_b_interactions.py # Phase 2.5
python3 experiments/kquity_hw/phase4_fixed_point.py  # Phase 4
```

Phase 5 HLS (regenerate constants + golden vectors + csynth):

```bash
cd experiments/kquity_hw/hls
python3 kquity_pressure_ref.py     # regenerate kquity_constants.h
python3 gen_golden.py              # regenerate golden_kquity_pressure.h

# Native g++ smoke test (no Vitis required):
g++ -O2 -I. -std=c++17 kquity_pressure_hls.cpp test_kquity_pressure.cpp \
    -o test_native
./test_native

# Vitis HLS 2025.2 csynth + C-simulation:
/tools/Xilinx/2025.2/Vitis/bin/vitis_hls -f run_kquity_hls.tcl
```

Reports land under `kquity_logit_only/solution1/syn/report/` and
`kquity_with_lut/solution1/syn/report/`. The full Phase 5 report
including resource detail tables is in
`experiments/kquity_hw/results/phase5_hls_report.md`.

## 8. Caveats

```
* Eval pool is the 16 tests/benchmark_events_*.csv.gz shards
  (~3,000 games, 233k events). The full encoded datasets
  referenced in CLAUDE.md (quality_filtered, logged_in,
  late_tournament) are not in the local checkout. The primitive
  is stable on the benchmark slice but production-grade
  coefficients should be refit on the full corpus.

* The model output is P(blue wins), not P(gold wins) as the
  README states. See section 1.

* The primitive is a hardware-native ranking/control signal, not
  a calibrated probability replacement. Brier degradation is
  1.5pp; AUC drop is 4.06pp. If a use case needs the full
  LightGBM probability surface, keep the LightGBM and use the
  hardware primitive only for the threshold/control path.

* HLS results are post-csynth (resource and timing estimates).
  Place-and-route + IP packaging may shift the numbers slightly,
  but the order of magnitude is robust given how small the
  design is.

* The primitive uses int8 Q1.7 features and int8 Q3.5 weights
  (half_range folded). A different but mathematically equivalent
  quantization could use raw integer counts for the four integer
  features (egg/food/sol/war differentials) and avoid the
  normalization step; the trade-off is more complex weight
  scaling per feature. The current pipeline keeps the data path
  uniform and is the one the HLS implements.

* No GPU was used in any phase. CPU + Vitis HLS only.
```

## 9. Direct model-to-fabric baseline

To frame the extracted primitive's hardware cost, we also ran a
direct `model -> fabric` compilation using
[Conifer](https://github.com/thesps/conifer). Same target
(`xck26-sfvc784-2LV-c`, KV260), same 5 ns target clock.

**The Conifer baseline is a comparable XGBoost BDT, not a
bit-exact compilation of `current_preferred_model.mdl`.** Conifer
1.8's `LightGBM → ONNX → Conifer` path failed with
`KeyError('base_values')` in its ONNX parser, so we trained a
same-budget XGBoost classifier (`n_estimators=50, max_depth=5`,
comparable to the 50-tree / 32-leaf small LightGBM diagnostic
from Phase 2) and used `convert_from_xgboost`. The reading is
"what cost class does keeping a tree ensemble in fabric land in?"
— not "what does the original LightGBM compile to?"

```
                                Extracted    Conifer XGB     Conifer XGB
                                Phase 5      50t/d5 16-bit   50t/d5 12-bit
Path                            extract      model-to-fabric model-to-fabric
Source model                    LightGBM     XGBoost         XGBoost
                                (oracle)     surrogate       surrogate
Input features                  6            52              52
Parameters                      7            ~25 n/tree × 50  same
Precision                       int8 / int8  ap_fixed<16,6>  ap_fixed<12,6>
LUT                             269          39,986          37,393
FF                              180          2,800           2,441
DSP                             2            0               0
BRAM_18K                        0            0               0
Latency cycles                  4            4               4
Fmax (MHz)                      288.9        274.4           275.3
KV260 LUT utilization           0.23%        34%             32%
AUC vs labels                   0.7496       0.7751          0.7751 †
Brier                           0.2017       0.1928          0.1928 †
pRMSE vs LightGBM oracle        0.113        0.094           0.094 †
```

† 12-bit fidelity is reported from the host-side XGBoost
classifier (identical between the two precision runs). Precision
only affects the synthesized comparator quantization, not the
python model. Bit-exact 16-bit vs 12-bit fabric output
divergence was not separately measured here.

The Conifer-compiled XGBoost surrogate keeps ~2.55 pp more AUC
and ~0.9 pp better Brier than the extracted linear primitive, at
the cost of **~148× more LUT** and 34% of the KV260 LUT budget
(16-bit). Reducing precision to 12-bit shaves only ~6.5% LUT
and ~13% FF: **the comparator topology is the cost driver,
not the bitwidth.** Latency and II are identical across all three
configurations.

The contrast is not "extraction is universally better" — it is a
clean hardware/fidelity tradeoff:

  the tree ensemble buys 2.5 pp of AUC at a 150× LUT premium;
  the extracted score buys interpretability and a microscopic circuit.

This is the orthogonal axis Case F exists to illustrate:

```
Path A: oracle-to-action extraction
  LightGBM oracle (or even XGBoost surrogate)
    -> 6-feature objective-pressure vector
    -> 7-parameter linear primitive
    -> tiny hardware

Path B: model-to-fabric compile
  Small BDT
    -> Conifer
    -> tree-comparator fabric
    -> compact hardware, but still ~150× the extracted primitive
```

Both are useful. Path A is what this writeup is about; Path B is
the right baseline to put next to it. The cost gradient (LUT,
roughly: extracted 269 → Conifer 50t/d5 ≈ 40k → full LightGBM
100t/100l would be larger again) is the schematic for "what
extraction buys".

See `experiments/kquity_hw/conifer/README.md` for the full
Conifer baseline notes and `results/conifer_baseline_xgb_50t_d5.json`
for the parsed numbers.

## Per-phase artifacts

For anyone wanting the full chain:

```
experiments/kquity_hw/
  README.md
  inventory.md                        52-feature decode + P(blue) note
  baseline_eval.py                    Phase 1 eval script
  surrogate_ladder.py                 Phase 2 surrogate ladder
  route_b_interactions.py             Phase 2.5 interaction probe
  phase4_fixed_point.py               Phase 4 fixed-point parity
  results/
    baseline_metrics.json
    calibration.csv
    phase_metrics.csv
    phase0_phase1_report.md
    surrogate_ladder.json
    phase2_report.md
    route_b_interactions.json
    phase4_fixed_point.json
    phase5_hls_report.md
  hls/
    kquity_pressure_hls.{h,cpp}
    kquity_constants.h
    kquity_sigmoid_lut.h
    kquity_pressure_ref.py
    gen_golden.py
    golden_kquity_pressure.h
    test_kquity_pressure.cpp
    run_kquity_hls.tcl
    README.md
  conifer/
    conifer_baseline.py
    README.md
    results/
      conifer_baseline_xgb_50t_d5.json
      conifer_xgb_50t_d5_p16_6/       full Conifer project + csynth
```

Branch: `case-f-hw-extraction` (local only; not pushed upstream).

## Status and follow-up

**Status: complete side demo.** Extracted primitive validated
bit-exact float / pure-int python / HLS C++ across 152 vectors;
HLS csynth done; Conifer model-to-fabric baseline run for
contrast; RP2040 USB firmware bundle prepared for Rob (board
arrives Monday per Phase 7 directive). No further phases queued.

### Rob handoff

Draft handoff message lives at
`experiments/kquity_hw/rob_handoff_draft.md` and includes:

- bit-exact int8 primitive description and weight constants
- KV260 csynth numbers (269 LUT / 2 DSP / 4 cyc, 289 MHz)
- Conifer XGBoost baseline contrast (~148× more LUT for
  +2.55 pp AUC)
- RP2040 USB firmware bundle for the demo board, with build
  and flash instructions

Awaiting user's go to send. Conifer 1.8 LightGBM→ONNX parser
bug (`KeyError('base_values')`) was hit during the baseline;
two minimal repros live at
`experiments/kquity_hw/conifer/repros/` but have not been
filed upstream at `thesps/conifer`. Filing is optional.

### What would extend Case F (not promote)

- Refit int8 coefficients on the full quality-filtered /
  logged-in / late-tournament KQuity corpus (current eval pool
  is the 16 benchmark shards, ~3k games / 233k events). Refit
  is unlikely to change the cost-class contrast but may
  improve the int8 coefficients for production deployment.
- Re-run Conifer at the *original* LightGBM oracle once the
  upstream ONNX parser is fixed (rather than the XGBoost
  comparable surrogate). This would tighten the
  ``model-to-fabric'' claim to bit-exact compilation rather
  than ``comparable tree ensemble.''
- Pack the LUT for sigmoid as URAM/distributed-RAM rather
  than BRAM\_18K (cosmetic; 1 BRAM is not a budget concern).

### What would NOT change the take-away

- More trees, deeper trees, or different XGBoost
  hyperparameters in the Conifer baseline. The contrast is
  cost-class scale (~$10^2$ vs ~$10^4$ LUT); intra-class
  tweaks do not move the schematic point.
- Higher precision in the extracted linear (the int8 quant
  loss is essentially zero against the float linear; the
  4.06 pp AUC gap to the LightGBM oracle is structural,
  not quantization).

### Paper citation

This work is cited as a mini-case in
`paper-hw-net/roadmap.tex` (`\subsection{Case~F}`) and is
mentioned in `paper-hw-net/limitations.tex` as the
``model-to-fabric vs.\ oracle-to-action'' contrast point.
Case F is not promoted to a main case study; the methodology
spine is carried by Cases A, B, C.
