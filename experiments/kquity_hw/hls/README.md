# KQuity objective-pressure HLS

This directory holds the Vitis HLS C++ for the 7-parameter linear
primitive extracted from `current_preferred_model.mdl` in
Phase 2 / Phase 4. The HLS implementation is bit-exact against the
pure-int python reference. See `../results/phase5_hls_report.md`
for resource numbers, schematic, and the caveats.

## Files

```
kquity_pressure_hls.h          interface (typedefs)
kquity_pressure_hls.cpp         HLS implementation (6 int8 muls + LUT)
kquity_constants.h             generated: int8 weights + int16 bias
kquity_sigmoid_lut.h           generated: 1024-entry uint16 sigmoid LUT
kquity_pressure_ref.py         python pure-int reference
gen_golden.py                  generates testbench golden vectors
golden_kquity_pressure.h       generated: 152 vectors (inputs + outputs)
test_kquity_pressure.cpp       testbench (bit-exact gate)
run_kquity_hls.tcl              Vitis HLS 2025.2 csynth driver
```

## Reproducing

Regenerate constants (only if `../results/phase4_fixed_point.json`
changes):

```
python3 kquity_pressure_ref.py
python3 gen_golden.py
```

Quick native-g++ smoke test of the testbench (no Vitis required):

```
g++ -O2 -I. -std=c++17 kquity_pressure_hls.cpp test_kquity_pressure.cpp \
    -o test_native
./test_native
```

Vitis HLS 2025.2 synthesis:

```
/tools/Xilinx/2025.2/Vitis/bin/vitis_hls -f run_kquity_hls.tcl
```

Reports land under `kquity_logit_only/solution1/syn/report/` and
`kquity_with_lut/solution1/syn/report/`.

## Quantization

```
feature:  int8 Q1.7, scale Sf = 128
          x_int = clip(round((x - center)/half_range * 128), -128, 127)
weight:   int8 Q3.5, scale Sw = 32 (half_range folded into weight)
          w_int = clip(round(w * half_range * 32), -128, 127)
bias:     int16, accumulator scale Sf*Sw = 4096 (Q4.12)
          b_acc = round((b + sum_i w_i*center_i) * 4096)
output:   int16 Q4.12 logit, saturated to [-32768, 32767]
prob:     1024-entry uint16 sigmoid LUT
          idx = clip((logit + 32768) >> 6, 0, 1023)
```

Hardware estimate on KV260 (xck26):

```
config             cycles    LUT    FF   DSP   BRAM    Fmax
logit-only           4       269   180    2      0    288.9 MHz
logit + prob LUT     7       287   283    2      1    288.9 MHz
```

Both are <0.4% of KV260 in every resource class.
