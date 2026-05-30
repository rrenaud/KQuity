#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Compile the firmware source with the host gcc and run the
# golden-vector test. This proves the int math is bit-exact before
# we touch a Pico.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
fw="${here}/../firmware"
out="${here}/build"
mkdir -p "${out}"

# golden_vectors.c is committed; the reference math lives in
# kquity_pressure_ref.py. Compile the firmware C and run the gate.
gcc -std=c11 -O2 -Wall -Wextra -Wshadow -Wpedantic \
    -I"${fw}" -I"${here}" \
    "${fw}/kquity_score.c" \
    "${here}/golden_vectors.c" \
    "${here}/test_native.c" \
    -o "${out}/test_native"

"${out}/test_native"
