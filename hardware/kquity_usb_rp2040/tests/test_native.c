// SPDX-License-Identifier: Apache-2.0
//
// Native (desktop) test. Compiles `kquity_score.c` with the host
// toolchain and verifies it produces bit-exact int16 logit and
// uint16 probability across the 152 committed golden vectors (see
// kquity_pressure_ref.py for the reference math). No Pico SDK
// required; run this before flashing hardware.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "kquity_score.h"

extern const int     kquity_golden_n;
extern const int8_t  kquity_golden_x[][6];
extern const int16_t kquity_golden_logit[];
extern const uint16_t kquity_golden_prob[];

int main(void) {
    int logit_mismatches = 0;
    int prob_mismatches = 0;
    int max_logit_abs_err = 0;
    int max_prob_abs_err = 0;

    for (int i = 0; i < kquity_golden_n; i++) {
        kquity_result_t r = kquity_score_q8(
            kquity_golden_x[i][0],
            kquity_golden_x[i][1],
            kquity_golden_x[i][2],
            kquity_golden_x[i][3],
            kquity_golden_x[i][4],
            kquity_golden_x[i][5]
        );
        int dl = (int)r.logit_q412 - (int)kquity_golden_logit[i];
        int dp = (int)r.prob_u16   - (int)kquity_golden_prob[i];
        if (dl) {
            int abs_dl = dl < 0 ? -dl : dl;
            if (abs_dl > max_logit_abs_err) max_logit_abs_err = abs_dl;
            if (++logit_mismatches <= 8) {
                printf("  [%3d] logit mismatch: got=%6d expected=%6d diff=%+d\n",
                       i, (int)r.logit_q412, (int)kquity_golden_logit[i], dl);
            }
        }
        if (dp) {
            int abs_dp = dp < 0 ? -dp : dp;
            if (abs_dp > max_prob_abs_err) max_prob_abs_err = abs_dp;
            if (++prob_mismatches <= 8) {
                printf("  [%3d] prob mismatch:  got=%5d expected=%5d diff=%+d\n",
                       i, (int)r.prob_u16, (int)kquity_golden_prob[i], dp);
            }
        }
    }

    printf("\n=== RP2040 firmware native test ===\n");
    printf("N = %d\n", kquity_golden_n);
    printf("logit mismatches: %d (max abs err = %d LSB Q4.12)\n",
           logit_mismatches, max_logit_abs_err);
    printf("prob  mismatches: %d (max abs err = %d uint16 LSB)\n",
           prob_mismatches, max_prob_abs_err);
    if (logit_mismatches == 0 && prob_mismatches == 0) {
        printf("PASS\n");
        return 0;
    }
    printf("FAIL\n");
    return 1;
}
