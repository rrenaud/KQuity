// SPDX-License-Identifier: MIT
//
// KQuity USB CDC firmware for Raspberry Pi Pico / RP2040.
// Line-oriented serial protocol over the Pico SDK stdio_usb path.
//
// Protocol (ASCII, newline-terminated commands and responses):
//
//   PING
//     -> KQ1 PONG
//
//   INFO
//     -> KQ1 INFO kquity-rp2040 v0.1 logit_q=Q4.12 prob=uint16
//
//   SCORE <egg> <food> <snail> <soldier> <warrior> <berries>
//     six signed int8 decimals in [-128, 127], pre-quantized on host
//     -> KQ1 OK <int16_logit> <uint16_prob>
//
//   Anything else
//     -> KQ1 ERR unknown_command   or   KQ1 ERR bad_args
//
// Bit-exact to Phase 5 HLS and to the python pure-int reference
// kquity_pressure_ref.py. The host should send the same int8
// features it would send to the FPGA top, and check identical
// logit/prob.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "pico/stdlib.h"
#include "kquity_score.h"

#define LINE_MAX_LEN 128

static int parse_i8(const char *s, int8_t *out) {
    if (!s || !*s) return 0;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0') return 0;
    if (v < -128 || v > 127) return 0;
    *out = (int8_t)v;
    return 1;
}

int main(void) {
    stdio_init_all();

    // Give the host a moment to enumerate /dev/ttyACM0 before we
    // start spamming the line.
    sleep_ms(1500);
    printf("KQ1 READY kquity-rp2040 v0.1\n");

    char line[LINE_MAX_LEN];

    for (;;) {
        if (!fgets(line, sizeof(line), stdin)) {
            tight_loop_contents();
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';

        if (line[0] == '\0') {
            continue;
        }

        if (strcmp(line, "PING") == 0) {
            printf("KQ1 PONG\n");
            continue;
        }

        if (strcmp(line, "INFO") == 0) {
            printf(
                "KQ1 INFO kquity-rp2040 v0.1 logit_q=Q4.12 "
                "prob=uint16 features=int8_q1p7\n"
            );
            continue;
        }

        char *tok = strtok(line, " ");
        if (!tok) {
            continue;
        }
        if (strcmp(tok, "SCORE") != 0) {
            printf("KQ1 ERR unknown_command\n");
            continue;
        }

        int8_t x[6];
        int ok = 1;
        for (int i = 0; i < 6; i++) {
            tok = strtok(NULL, " ");
            if (!tok || !parse_i8(tok, &x[i])) {
                ok = 0;
                break;
            }
        }
        if (!ok) {
            printf("KQ1 ERR bad_args\n");
            continue;
        }

        kquity_result_t r = kquity_score_q8(
            x[0], x[1], x[2], x[3], x[4], x[5]
        );
        printf("KQ1 OK %d %u\n", (int)r.logit_q412, (unsigned)r.prob_u16);
    }

    return 0;  // unreachable
}
