// SPDX-License-Identifier: MIT
//
// KQuity objective-pressure primitive — HLS interface.
//
// The deployed primitive is a 6-feature linear dot product over
// pre-normalized int8 (Q1.7) features and int8 (Q3.5) folded weights,
// extracted from a 100-tree × 100-leaf LightGBM win-probability oracle
// (~1,400× parameter reduction at 4.0 pp AUC drop). The integer math
// here is bit-exact to the python reference in `kquity_pressure_ref.py`.
//
// Inputs (all int8, signed, pre-quantized on host/PS):
//
//   egg_diff_q     blue.eggs - gold.eggs           Q1.7, hr=4   c=0
//   food_diff_q    blue.food - gold.food           Q1.7, hr=16  c=0
//   snail_pos_q    gold-symmetric snail position   Q1.7, hr=1   c=0
//   soldier_diff_q blue.n_soldiers - gold ditto    Q1.7, hr=4   c=0
//   warrior_diff_q blue.n_warriors - gold ditto    Q1.7, hr=4   c=0
//   berries_q      berries_avail centered at 0.7   Q1.7, hr=0.3 c=0.7
//
// Outputs:
//
//   kquity_logit_q8(...) -> int16 Q4.12  ("objective-pressure score")
//     Logit divided by ACC_SCALE=4096 gives the float logit.
//     Saturated to [-32768, 32767]; observed test-pool range is
//     roughly [-5, +5] in float, well inside the saturation limit.
//
//   kquity_prob_lut(logit_q4_12) -> uint16 probability ∈ [0, 65535]
//     Optional sigmoid wrapper. uint16 / 65535 = P(blue wins) ∈ [0, 1].

#ifndef KQUITY_PRESSURE_HLS_H
#define KQUITY_PRESSURE_HLS_H

#include <stdint.h>

#ifdef __SYNTHESIS__
#include <ap_int.h>
typedef ap_int<8>    kq_q8_t;
typedef ap_int<16>   kq_q16_t;
typedef ap_int<32>   kq_q32_t;
typedef ap_uint<16>  kq_uq16_t;
#else
typedef int8_t       kq_q8_t;
typedef int16_t      kq_q16_t;
typedef int32_t      kq_q32_t;
typedef uint16_t     kq_uq16_t;
#endif

// Primary deployable primitive: 6-input -> int16 logit.
kq_q16_t kquity_logit_q8(
    kq_q8_t egg_diff_q,
    kq_q8_t food_diff_q,
    kq_q8_t snail_pos_q,
    kq_q8_t soldier_diff_q,
    kq_q8_t warrior_diff_q,
    kq_q8_t berries_q
);

// Optional probability wrapper.
kq_uq16_t kquity_prob_lut(kq_q16_t logit_q4_12);

#endif  // KQUITY_PRESSURE_HLS_H
