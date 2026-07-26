// SPDX-License-Identifier: MIT
//
// KQuity objective-pressure primitive — HLS implementation.
//
// Pure-int bit-exact equivalent of `kquity_pressure_ref.py`.
// Six int8 multiplies, one int32 accumulator, one int16 saturating
// output. Optional 1024-entry sigmoid LUT wrapper.

#include "kquity_pressure_hls.h"
#include "kquity_constants.h"
#include "kquity_sigmoid_lut.h"

// Saturate int32 to int16 [-32768, 32767].
static inline kq_q16_t kquity_sat16(kq_q32_t v) {
#pragma HLS INLINE
    if (v >  32767) v =  32767;
    if (v < -32768) v = -32768;
    return (kq_q16_t)v;
}

kq_q16_t kquity_logit_q8(
    kq_q8_t egg_diff_q,
    kq_q8_t food_diff_q,
    kq_q8_t snail_pos_q,
    kq_q8_t soldier_diff_q,
    kq_q8_t warrior_diff_q,
    kq_q8_t berries_q
) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    kq_q32_t acc = (kq_q32_t)KQUITY_B_ACC;
    acc += (kq_q32_t)KQUITY_W_EGG     * (kq_q32_t)egg_diff_q;
    acc += (kq_q32_t)KQUITY_W_FOOD    * (kq_q32_t)food_diff_q;
    acc += (kq_q32_t)KQUITY_W_SNAIL   * (kq_q32_t)snail_pos_q;
    acc += (kq_q32_t)KQUITY_W_SOLDIER * (kq_q32_t)soldier_diff_q;
    acc += (kq_q32_t)KQUITY_W_WARRIOR * (kq_q32_t)warrior_diff_q;
    acc += (kq_q32_t)KQUITY_W_BERRIES * (kq_q32_t)berries_q;
    return kquity_sat16(acc);
}

kq_uq16_t kquity_prob_lut(kq_q16_t logit_q4_12) {
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    // Bit-exact LUT index recipe (see kquity_pressure_ref.py).
    // half_int = 32768, but +32768 = -32768 in int16 (wraps). Use
    // wider int32 arithmetic, mirror the python's clamp range.
    kq_q32_t l = (kq_q32_t)logit_q4_12;
    if (l >  KQUITY_LUT_HALF - 1) l =  KQUITY_LUT_HALF - 1;
    if (l < -KQUITY_LUT_HALF    ) l = -KQUITY_LUT_HALF;
    kq_q32_t idx = (l + (kq_q32_t)KQUITY_LUT_HALF) >> KQUITY_LUT_SHIFT;
    if (idx < 0)                       idx = 0;
    if (idx > (KQUITY_LUT_SIZE - 1))   idx = KQUITY_LUT_SIZE - 1;
    return (kq_uq16_t)kquity_sigmoid_lut[idx];
}
