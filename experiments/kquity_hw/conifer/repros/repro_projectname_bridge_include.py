#!/usr/bin/env python3
"""Minimal repro: Conifer's xilinxhls bridge.cpp hardcodes
`#include "firmware/my_prj.h"` regardless of the `ProjectName`
value in the config. The compile step then fails with
"no such file or directory" because the actual firmware header
gets named after the configured ProjectName.

Environment (versions observed):
  python      3.13
  conifer     1.8
  xgboost     3.2.0

Repro:
  source /tools/Xilinx/2025.2/Vitis/settings64.sh
  export XILINX_AP_INCLUDE=/tools/Xilinx/2025.2/Vitis/include
  python repro_projectname_bridge_include.py

Expected (broken) behavior:
  bridge.cpp:3:10: fatal error: firmware/my_prj.h: No such file or directory
  Exception('Failed to compile project <custom_name>')

The workaround used in conifer_baseline.py is to leave
ProjectName at the default 'my_prj' and namespace via OutputDir.
The right fix would be for bridge.cpp's template to substitute
the configured project name in the #include line.
"""
import numpy as np
import xgboost as xgb
import conifer

rng = np.random.RandomState(0)
X = rng.randn(2000, 8).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

clf = xgb.XGBClassifier(
    n_estimators=10, max_depth=4, tree_method="hist",
    verbosity=0, eval_metric="logloss",
)
clf.fit(X, y)

cfg = conifer.backends.xilinxhls.auto_config()
cfg["ProjectName"] = "custom_proj_name"          # <-- trigger
cfg["OutputDir"] = "/tmp/conifer_repro_bridge"
cfg["Precision"] = "ap_fixed<16,6>"

model = conifer.converters.convert_from_xgboost(clf.get_booster(), cfg)
print("attempting model.compile() with ProjectName != 'my_prj'")
model.compile()
print("OK (unexpected — bug appears fixed)")
