#!/usr/bin/env python3
"""Minimal repro: LightGBM -> ONNX -> Conifer fails with
`KeyError('base_values')` inside Conifer's onnx parser.

Environment (versions observed):
  python      3.13
  conifer     1.8
  lightgbm    4.6.0
  onnxmltools 1.16.0
  onnx        1.21.0

Repro:
  python repro_lightgbm_onnx_base_values.py

Expected (broken) behavior:
  Traceback ending in
    KeyError: 'base_values'
  inside conifer.converters.convert_from_onnx.
"""
import numpy as np
import lightgbm as lgb
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxmltools.convert.lightgbm.convert import convert as convert_lightgbm
import conifer

# Tiny synthetic binary-classification toy.
rng = np.random.RandomState(0)
X = rng.randn(5000, 12).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.int32)

clf = lgb.LGBMClassifier(n_estimators=20, num_leaves=16, verbose=-1)
clf.fit(X, y)

initial_types = [("input", FloatTensorType([None, 12]))]
onnx_model = convert_lightgbm(
    clf, initial_types=initial_types, target_opset=12
)

cfg = conifer.backends.xilinxhls.auto_config()
cfg["OutputDir"] = "/tmp/conifer_repro_base_values"
cfg["Precision"] = "ap_fixed<16,6>"

print("attempting conifer.converters.convert_from_onnx(...)")
model = conifer.converters.convert_from_onnx(onnx_model, cfg)
print("OK (unexpected — bug appears fixed)")
