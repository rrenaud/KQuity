# Extracted primitive vs. compiling the trees directly

There are two ways to put the KQuity win predictor onto hardware:

1. **Compile the tree ensemble directly.** Tools like
   [Conifer](https://github.com/thesps/conifer) take a boosted-tree
   model and emit HLS for an FPGA. Faithful, automatic — but it
   synthesizes every comparator in every tree.
2. **Extract the dominant decision structure** into a tiny
   hand-optimized primitive, and run *that*. This is what the firmware
   in this directory does.

This note compares the two on the same FPGA target so the tradeoff is
concrete.

## What gets deployed

| | source | what runs on hardware |
|---|---|---|
| Extracted primitive (this firmware) | LightGBM 100-tree win predictor | 6-feature int8 linear score + 1024-entry sigmoid LUT |
| Conifer baseline | XGBoost 50-tree / depth-5 surrogate of the same predictor | all 50 trees, compiled to HLS |

The 6 features are the "objective-pressure" differentials (eggs, food,
snail position, soldiers, warriors, berries). The extracted score is
the dominant linear structure of the full LightGBM model, quantized to
int8 and validated **bit-exact** against `tests/kquity_pressure_ref.py`
across 152 vectors. The Conifer baseline compiles an XGBoost surrogate
(a faithful tree approximation of the same predictor) for an
apples-to-apples FPGA cost.

## Hardware cost (KV260, Vitis HLS csynth)

| | LUT | DSP | latency | Fmax | notes |
|---|---|---|---|---|---|
| **Extracted primitive** | **~270** | 2 | 4 cyc / 20 ns | ~289 MHz | logit path; +3 cyc for the sigmoid LUT |
| Conifer XGBoost 50t/d5 | **~40,000** | — | — | — | ~34% of the KV260 LUT budget |

That is **~148× more LUT** for the direct tree compile.

## Accuracy cost

The Conifer-compiled trees keep **+2.55 pp AUC** and ~0.9 pp better
Brier than the extracted linear primitive. So the tradeoff is:

```
extracted primitive   ~270 LUT     baseline AUC      -> fits a $4 RP2040 MCU
Conifer 50t/d5      ~40,000 LUT     +2.55 pp AUC      -> needs a real FPGA
full LightGBM 100t   (CPU only)     +a bit more AUC   -> no hardware path
```

## Takeaway

If you want the win-probability signal on cheap hardware, the
extracted primitive is the better-optimized option by a wide margin:
it runs on a coin-sized $4 microcontroller (or ~270 LUT of an FPGA),
versus ~148× the silicon to compile the trees directly for 2.55 pp
more AUC. The automated route is the right call when you want maximum
fidelity and have the FPGA budget; the extracted primitive is the
right call when you want "good enough, basically free."
