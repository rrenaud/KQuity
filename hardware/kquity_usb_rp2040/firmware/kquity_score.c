// SPDX-License-Identifier: Apache-2.0
//
// KQuity objective-pressure score: pure-int implementation.
// No floats, no malloc, no allocation, no globals beyond the
// const tables in the included headers.

#include "kquity_score.h"
#include "kquity_constants.h"
#include "kquity_sigmoid_lut.h"

static inline int16_t kquity_sat16(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

int16_t kquity_logit_q8(
    int8_t egg_q,
    int8_t food_q,
    int8_t snail_q,
    int8_t soldier_q,
    int8_t warrior_q,
    int8_t berries_q
) {
    int32_t acc = (int32_t)KQUITY_B_ACC;

    acc += (int32_t)KQUITY_W_EGG     * (int32_t)egg_q;
    acc += (int32_t)KQUITY_W_FOOD    * (int32_t)food_q;
    acc += (int32_t)KQUITY_W_SNAIL   * (int32_t)snail_q;
    acc += (int32_t)KQUITY_W_SOLDIER * (int32_t)soldier_q;
    acc += (int32_t)KQUITY_W_WARRIOR * (int32_t)warrior_q;
    acc += (int32_t)KQUITY_W_BERRIES * (int32_t)berries_q;

    return kquity_sat16(acc);
}

uint16_t kquity_prob_from_logit(int16_t logit_q412) {
    // Bit-exact LUT index recipe from kquity_pressure_ref.py
    //   idx = clip((logit + 32768) >> 6, 0, 1023)
    // For int16 logit the clip is theoretically redundant; we keep
    // the bounds check as a paranoid guardrail.
    int32_t idx = ((int32_t)logit_q412 + KQUITY_LUT_HALF) >> KQUITY_LUT_SHIFT;
    if (idx < 0)                    idx = 0;
    if (idx > (KQUITY_LUT_SIZE - 1)) idx = KQUITY_LUT_SIZE - 1;
    return kquity_sigmoid_lut[idx];
}

kquity_result_t kquity_score_q8(
    int8_t egg_q,
    int8_t food_q,
    int8_t snail_q,
    int8_t soldier_q,
    int8_t warrior_q,
    int8_t berries_q
) {
    kquity_result_t out;
    out.logit_q412 = kquity_logit_q8(
        egg_q, food_q, snail_q, soldier_q, warrior_q, berries_q
    );
    out.prob_u16 = kquity_prob_from_logit(out.logit_q412);
    return out;
}
