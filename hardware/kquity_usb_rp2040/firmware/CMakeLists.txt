# SPDX-License-Identifier: MIT
#
# Pico SDK build for the KQuity USB firmware. Expects
# $PICO_SDK_PATH to point at a working pico-sdk checkout.
#
# Build:
#   cd hardware/kquity_usb_rp2040/firmware
#   cmake -S . -B build -DPICO_BOARD=pico
#   cmake --build build -j
# Output: build/kquity_usb_rp2040.uf2

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED ENV{PICO_SDK_PATH})
    message(FATAL_ERROR
        "PICO_SDK_PATH is not set. Clone "
        "https://github.com/raspberrypi/pico-sdk and "
        "export PICO_SDK_PATH=/path/to/pico-sdk first.")
endif()

include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(kquity_usb_rp2040 C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(kquity_usb_rp2040
    main.c
    kquity_score.c
)

target_include_directories(kquity_usb_rp2040 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(kquity_usb_rp2040
    pico_stdlib
)

# USB CDC stdio, no UART -- USB is the comms channel.
pico_enable_stdio_usb(kquity_usb_rp2040 1)
pico_enable_stdio_uart(kquity_usb_rp2040 0)

# Generate .uf2, .elf, .bin, .hex.
pico_add_extra_outputs(kquity_usb_rp2040)
