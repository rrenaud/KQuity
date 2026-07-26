# KQuity RP2040 firmware build on Debian forky/sid

Documented 2026-05-21 after a fresh build on monster. Captures the
two gotchas that hit on Debian forky (gcc-arm-none-eabi 15.x ships
picolibc instead of newlib; the system gcc breaks pico-sdk's
boot_stage2 and stdio glue).

## TL;DR — known-good recipe

```bash
# 1. ARM official toolchain (newlib, not Debian's picolibc one).
mkdir -p ~/devel/rp2040
cd ~/devel/rp2040

curl -sL -o arm-gnu.tar.xz \
  "https://developer.arm.com/-/media/Files/downloads/gnu/13.3.rel1/binrel/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi.tar.xz"
tar -xf arm-gnu.tar.xz
# -> ~/devel/rp2040/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/

# 2. pico-sdk
git clone --depth 1 --recurse-submodules -j4 \
  https://github.com/raspberrypi/pico-sdk.git

# 3. build the KQuity firmware
cd ~/devel/KQuity/hardware/kquity_usb_rp2040/firmware
rm -rf build
PICO_SDK_PATH=~/devel/rp2040/pico-sdk \
PICO_TOOLCHAIN_PATH=~/devel/rp2040/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi \
  cmake -S . -B build -DPICO_BOARD=pico

PICO_SDK_PATH=~/devel/rp2040/pico-sdk \
PICO_TOOLCHAIN_PATH=~/devel/rp2040/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi \
  cmake --build build -j 8

# output: build/kquity_usb_rp2040.uf2 (~86 KB)
```

## Why not the Debian gcc-arm-none-eabi?

`apt install gcc-arm-none-eabi` on Debian forky pulls v15.2.1, which
ships picolibc by default and does NOT include newlib. pico-sdk
hardcodes two things that picolibc lacks:

1. `--specs=nosys.specs` in
   `src/rp2040/boot_stage2/CMakeLists.txt` and the rp2350 sibling.
   nosys.specs is a newlib-only file.
2. `extern FILE *stdin, *stdout, *stderr;` semantics. In picolibc
   these are macros expanding to fixed addresses; pico_printf takes
   their address as if they were extern symbols. Link fails with
   "undefined reference to `stdin'".

A partial workaround (added to make a non-fatal build pass earlier)
is to append BSD-style printf-attribute macros to picolibc's
`/usr/arm-none-eabi/include/sys/cdefs.h`:

```c
#ifndef __printflike
#define __printflike(fmtarg, firstvararg) \
    __attribute__((__format__(__printf__, fmtarg, firstvararg)))
#endif
#ifndef __scanflike
#define __scanflike(fmtarg, firstvararg) \
    __attribute__((__format__(__scanf__, fmtarg, firstvararg)))
#endif
```

Even with that shim, the link step fails on stdin/stdout. Don't
chase it — just use the ARM toolchain tarball above.

## Flashing the Pico

1. Hold the **BOOTSEL** button on the Pico while plugging USB.
2. The board appears as `RPI-RP2` mass storage (usually under
   `/media/$USER/RPI-RP2/` on Debian/Ubuntu, or `/run/media/...`
   depending on auto-mount).
3. Copy the .uf2 onto it:

   ```bash
   cp ~/devel/KQuity/hardware/kquity_usb_rp2040/firmware/build/\
   kquity_usb_rp2040.uf2 /media/$USER/RPI-RP2/
   sync
   ```

4. The Pico reboots and re-enumerates as `/dev/ttyACM0`. Test:

   ```bash
   python3 ~/devel/KQuity/hardware/kquity_usb_rp2040/host/\
   kquity_usb_client.py --port /dev/ttyACM0 ping
   # -> KQ1 PONG

   python3 ~/devel/KQuity/hardware/kquity_usb_rp2040/host/\
   test_device.py --port /dev/ttyACM0 --ping-first
   # -> N = 152
   # -> logit mismatches: 0
   # -> prob mismatches: 0
   # -> PASS
   ```

If `/dev/ttyACM0` doesn't appear, check `dmesg --since "1 min ago"`
for the USB enumeration. The user might need to be in the `dialout`
group:

```bash
sudo usermod -aG dialout $USER
# log out and back in
```

## Sizes

Built ELF (Arm GNU 13.3.Rel1):

```
text    data    bss     dec     hex
46956   0       4412    51368   c8a8    kquity_usb_rp2040.elf
```

UF2 binary: ~86 KB. The Pico has 264 KB SRAM and 2 MB Flash, so the
firmware is comfortably tiny — well under 5% of either.

## What this firmware does

Bit-exact port of the Phase 5 KQuity objective-pressure primitive
(`kquity_score.c` / `kquity_constants.h` / `kquity_sigmoid_lut.h`).
USB CDC line protocol; 6 int8 features in, int16 Q4.12 logit +
uint16 sigmoid prob out. No floats, no malloc.

See `README.md` in this directory for the protocol and host-side
testing. See `tests/run_native_tests.sh` for the desktop golden-
vector validation (compiles the same C with host gcc, runs against
152 vectors).
