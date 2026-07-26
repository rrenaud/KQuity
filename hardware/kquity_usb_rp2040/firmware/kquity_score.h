// SPDX-License-Identifier: MIT
//
// KQuity objective-pressure score, RP2040 firmware interface.
//
// Pure-int port of the Phase 5 HLS C++. Bit-exact against:
//   experiments/kquity_hw/hls/kquity_pressure_hls.{h,cpp}
//   experiments/kquity_hw/hls/kquity_pressure_ref.py
//
// Inputs are six int8 features in Q1.7, normalized on host:
//   egg_q     blue.eggs - gold.eggs           hr=4    c=0
//   food_q    blue.food_count - gold.food_count   hr=16   c=0
//   snail_q   gold-symmetric snail position   hr=1    c=0  (clip)
//   sol_q     blue.n_soldiers - gold.n_soldiers  hr=4    c=0
//   war_q     blue.n_warriors - gold.n_warriors  hr=4    c=0
//   berries_q berries_avail centered at 0.7   hr=0.3  c=0.7
//
// Output:
//   int16 logit in Q4.12  (objective-pressure score; /4096 for float)
//   uint16 sigmoid probability  ([0, 65535]; /65535 = P(blue wins))

#ifndef KQUITY_SCORE_H
#define KQUITY_SCORE_H

#include <stdint.h>

typedef struct {
    int16_t  logit_q412;
    uint16_t prob_u16;
} kquity_result_t;

int16_t kquity_logit_q8(
    int8_t egg_q,
    int8_t food_q,
    int8_t snail_q,
    int8_t soldier_q,
    int8_t warrior_q,
    int8_t berries_q
);

uint16_t kquity_prob_from_logit(int16_t logit_q412);

kquity_result_t kquity_score_q8(
    int8_t egg_q,
    int8_t food_q,
    int8_t snail_q,
    int8_t soldier_q,
    int8_t warrior_q,
    int8_t berries_q
);

#endif  // KQUITY_SCORE_H
