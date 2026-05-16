# Conifer model-to-fabric baseline

Direct compilation of a small boosted decision tree (BDT) to FPGA
fabric using [Conifer](https://github.com/thesps/conifer). This
is the **baseline** against the extracted 7-parameter linear
primitive from Phase 5. It answers the orthogonal question
"what if we simply compile the model?", as a contrast to
"what action variable did the oracle reveal?"

## Headline

| | extracted (Phase 5) | Conifer 50t × d5 |
|---|---|---|
| approach | oracle-to-action extraction | model-to-fabric compile |
| input features | 6 (objective-pressure vector) | 52 (raw) |
| parameters | 7 (1 bias + 6 weights) | 50 trees × ~32 leaves |
| LUT | 269 | **39,986** |
| FF | 180 | 2,800 |
| DSP | 2 | 0 |
| BRAM_18K | 0 | 0 |
| latency cycles | 4 | 4 |
| Fmax (MHz) | 288.9 | 274.4 |
| KV260 LUT util | 0.23% | **34%** |
| AUC | 0.7496 | 0.7751 |
| Brier | 0.2017 | 0.1928 |
| pRMSE vs oracle | 0.113 | 0.094 |

The Conifer-compiled BDT retains 2.5 pp more AUC vs the extracted
linear primitive but uses **~148× more LUT** (34% of KV260 vs
0.23%). Same 4-cycle latency, but Fmax is slightly lower (274 vs
289 MHz) due to comparator-fan-out delay in the larger design.

## Conversion path notes

The directive was LightGBM → ONNX → Conifer. That **failed** with
`KeyError('base_values')` inside Conifer's onnx parser — Conifer
1.8 does not fully accept the schema that `onnxmltools` emits for
LightGBM classifiers. (Recorded in
`results/conifer_baseline_blocked.json`.)

Fell back to the documented XGBoost path: train an XGBoost
classifier with `n_estimators=50, max_depth=5` (≈32 leaves max
per tree) and use `conifer.converters.convert_from_xgboost`. This
worked end-to-end. Conifer 1.8 prints a warning that "xgboost ≥
2.0.0 are not yet fully supported" against installed xgboost 3.2.0,
but the produced HLS is correct (csynth succeeded with sensible
resource estimates).

## Repro

Environment:

```bash
python3 -m venv ~/envs/kquity_conifer
source ~/envs/kquity_conifer/bin/activate
pip install lightgbm numpy pandas scikit-learn scipy \
            onnx onnxmltools conifer xgboost
```

Run with Vitis HLS env sourced:

```bash
source /tools/Xilinx/2025.2/Vitis/settings64.sh
export XILINX_AP_INCLUDE=/tools/Xilinx/2025.2/Vitis/include
python3 experiments/kquity_hw/conifer/conifer_baseline.py \
        --source xgboost --n-estimators 50 --max-depth 5
```

Output project lands in
`experiments/kquity_hw/conifer/results/conifer_xgb_50t_d5_p16_6/`.
Top-level csynth report is at
`my_prj/solution1/syn/report/my_prj_csynth.rpt` (Conifer's bridge
hardcodes the project name `my_prj`, so we leave the auto_config
default unchanged).

## Files

```
conifer_baseline.py                  end-to-end driver
README.md                            this file
results/
  conifer_baseline_xgb_50t_d5.json   parsed top-level numbers
  conifer_xgb_50t_d5_p16_6/          full Conifer/Vitis HLS project
                                      (build.log, firmware/, my_prj/,
                                       *.tcl, *_test.cpp, bridge.cpp)
  conifer_baseline_blocked.json       (LightGBM->ONNX failure record)
```

## Caveats

- Conifer 1.8 + xgboost 3.2.0 path emits a soft "not yet fully
  supported" warning. Synthesis succeeded and produced sensible
  resource numbers, but if a future bug surfaces, downgrade to
  xgboost 1.7.x or use Conifer's TMVA/sklearn paths.
- Direct LightGBM→Conifer was blocked by the ONNX-parser
  `base_values` KeyError. A native LightGBM converter inside
  Conifer would be the cleanest fix; for this baseline the
  XGBoost surrogate is a fair stand-in (same task, same eval set,
  same precision, comparable tree budget).
- We only ran `ap_fixed<16,6>`. A `ap_fixed<12,6>` variant would
  shrink LUT further; not pursued because the comparison is
  already crisp at 16,6.
- AUC 0.7751 here is the small XGBoost model, NOT the original
  100t/100l LightGBM oracle (AUC 0.7901). The Conifer baseline
  exists to compare hardware costs at a *similar* fidelity, not
  to reproduce the full oracle.
