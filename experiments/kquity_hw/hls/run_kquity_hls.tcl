# SPDX-License-Identifier: MIT
#
# Vitis HLS 2025.2 csynth script for the KQuity objective-pressure
# primitive. Run from this directory:
#
#   /tools/Xilinx/2025.2/Vitis/bin/vitis_hls -f run_kquity_hls.tcl
#
# Produces three artifacts:
#   - kquity_logit_only/      logit-only top (kquity_logit_q8)
#   - kquity_with_lut/        logit + sigmoid LUT top
#   - csynth.rpt summaries with LUT / FF / DSP / latency / II

set hls_root [pwd]
set src_files [list \
    "${hls_root}/kquity_pressure_hls.cpp" \
]

# KV260: Zynq UltraScale+ MPSoC, xck26-sfvc784-2LV-c
set fpga_part "xck26-sfvc784-2LV-c"
set clock_period_ns 5.0

# --- Project 1: logit-only ----------------------------------------------------
open_project -reset kquity_logit_only
set_top kquity_logit_q8
foreach f $src_files {
    add_files $f -cflags "-I${hls_root}"
}
add_files -tb "${hls_root}/test_kquity_pressure.cpp" -cflags "-I${hls_root}"
open_solution -reset "solution1" -flow_target vivado
set_part $fpga_part
create_clock -period ${clock_period_ns} -name default

csim_design
csynth_design
close_solution
close_project

# --- Project 2: logit + sigmoid LUT wrapper ---------------------------------
# Wrap into a single top that produces both logit and probability,
# for the "logit + LUT" resource number.

set wrap_src "${hls_root}/_kquity_with_lut_top.cpp"
set fp [open $wrap_src "w"]
puts $fp "// Generated wrapper: kquity_pressure (logit + sigmoid LUT)."
puts $fp "#include \"kquity_pressure_hls.h\""
puts $fp ""
puts $fp "void kquity_pressure(kq_q8_t egg, kq_q8_t food, kq_q8_t snail,"
puts $fp "                     kq_q8_t sol, kq_q8_t war, kq_q8_t ber,"
puts $fp "                     kq_q16_t *out_logit, kq_uq16_t *out_prob) {"
puts $fp "#pragma HLS PIPELINE II=1"
puts $fp "    kq_q16_t l = kquity_logit_q8(egg, food, snail, sol, war, ber);"
puts $fp "    *out_logit = l;"
puts $fp "    *out_prob  = kquity_prob_lut(l);"
puts $fp "}"
close $fp

open_project -reset kquity_with_lut
set_top kquity_pressure
foreach f $src_files {
    add_files $f -cflags "-I${hls_root}"
}
add_files $wrap_src -cflags "-I${hls_root}"
open_solution -reset "solution1" -flow_target vivado
set_part $fpga_part
create_clock -period ${clock_period_ns} -name default

csynth_design
close_solution
close_project

puts "================================================================"
puts " Phase 5 csynth done. Reports under:"
puts "   ${hls_root}/kquity_logit_only/solution1/syn/report/"
puts "   ${hls_root}/kquity_with_lut/solution1/syn/report/"
puts "================================================================"

exit
