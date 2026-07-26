# Conifer 1.8 bug repros

Two issues hit during the Case F model-to-fabric baseline. Both
have minimal repros (no KQuity dependency) — copy each into a
fresh tmp dir, run, observe failure.

## 1. `KeyError('base_values')` on LightGBM → ONNX → Conifer

```
python repros/repro_lightgbm_onnx_base_values.py
```

Conifer 1.8's `convert_from_onnx` does not accept the schema
that `onnxmltools.convert_lightgbm` emits for binary
classifiers. The traceback ends inside Conifer's ONNX parser
with `KeyError('base_values')`.

Workaround used in `conifer_baseline.py`: train an equivalent
XGBoost classifier and use `convert_from_xgboost` instead.

## 2. `firmware/my_prj.h` hardcoded in `bridge.cpp`

```
source /tools/Xilinx/2025.2/Vitis/settings64.sh
export XILINX_AP_INCLUDE=/tools/Xilinx/2025.2/Vitis/include
python repros/repro_projectname_bridge_include.py
```

Setting `cfg["ProjectName"] = "anything_else"` causes
`bridge.cpp` to fail compilation because the firmware header
gets renamed to `firmware/anything_else.h` while `bridge.cpp`
still has `#include "firmware/my_prj.h"`.

Workaround: leave `ProjectName` at the default `my_prj` and
namespace projects via `OutputDir` alone.

## Environment

```
conifer        1.8
xgboost        3.2.0   (1.8 also warns "xgboost >= 2.0.0 not yet
                        fully supported" — synthesis still works,
                        but flagging in case it matters upstream)
lightgbm       4.6.0
onnxmltools    1.16.0
onnx           1.21.0
python         3.13
Vitis HLS      2025.2
```

## Filing status

Not filed upstream yet. The repros are clean and self-contained,
so a one-line `gh issue create` per bug would land them; this is
left as an optional follow-up, not blocking the Rob handoff.
