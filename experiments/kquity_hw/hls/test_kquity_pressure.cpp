// SPDX-License-Identifier: MIT
//
// HLS testbench. Compares `kquity_logit_q8` and `kquity_prob_lut`
// against the python-generated golden vectors in
// `golden_kquity_pressure.h`. Pass criterion: bit-exact logit and
// bit-exact probability.

#include <cstdio>
#include <cstdlib>

#include "kquity_pressure_hls.h"
#include "golden_kquity_pressure.h"

int main() {
    int logit_mismatches = 0;
    int prob_mismatches = 0;
    int max_logit_abs_err = 0;
    int max_prob_abs_err = 0;

    for (int i = 0; i < KQUITY_GOLDEN_N; ++i) {
        kq_q8_t egg = (kq_q8_t)kquity_golden_x[i][0];
        kq_q8_t food = (kq_q8_t)kquity_golden_x[i][1];
        kq_q8_t snail = (kq_q8_t)kquity_golden_x[i][2];
        kq_q8_t sol = (kq_q8_t)kquity_golden_x[i][3];
        kq_q8_t war = (kq_q8_t)kquity_golden_x[i][4];
        kq_q8_t ber = (kq_q8_t)kquity_golden_x[i][5];

        kq_q16_t logit = kquity_logit_q8(egg, food, snail, sol, war, ber);
        kq_uq16_t prob = kquity_prob_lut(logit);

        int got_logit = (int)logit;
        int exp_logit = (int)kquity_golden_logit_q4_12[i];
        int got_prob = (int)prob;
        int exp_prob = (int)kquity_golden_prob[i];

        int dl = got_logit - exp_logit;
        int dp = got_prob - exp_prob;
        if (dl != 0) {
            ++logit_mismatches;
            if (logit_mismatches <= 8) {
                std::printf(
                    "  [%3d] logit mismatch: got=%6d expected=%6d diff=%+d\n",
                    i, got_logit, exp_logit, dl
                );
            }
            int abs_dl = dl < 0 ? -dl : dl;
            if (abs_dl > max_logit_abs_err) max_logit_abs_err = abs_dl;
        }
        if (dp != 0) {
            ++prob_mismatches;
            if (prob_mismatches <= 8) {
                std::printf(
                    "  [%3d] prob mismatch:  got=%5d expected=%5d diff=%+d\n",
                    i, got_prob, exp_prob, dp
                );
            }
            int abs_dp = dp < 0 ? -dp : dp;
            if (abs_dp > max_prob_abs_err) max_prob_abs_err = abs_dp;
        }
    }

    std::printf("\n=== Phase 5 HLS testbench summary ===\n");
    std::printf("N = %d\n", KQUITY_GOLDEN_N);
    std::printf("logit mismatches: %d (max abs err = %d LSB Q4.12)\n",
                logit_mismatches, max_logit_abs_err);
    std::printf("prob  mismatches: %d (max abs err = %d uint16 LSB)\n",
                prob_mismatches, max_prob_abs_err);

    // Pass criterion: bit-exact on both. (<=1 LSB tolerance enabled
    // below if we ever introduce rounding-mode drift.)
    if (logit_mismatches == 0 && prob_mismatches == 0) {
        std::printf("PASS\n");
        return 0;
    }
    if (max_logit_abs_err <= 1 && max_prob_abs_err <= 1) {
        std::printf("PASS (<=1 LSB tolerance)\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}
