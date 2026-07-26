DO NOT SEND without owner approval. Draft message for Rob, with the
ChatGPT-Pro-polished framing plus the Conifer side panel and the
RP2040 firmware note. Saved here so it does not get lost between
sessions.

---

Hey Rob,

I ran a small hardware-extraction pass on KQuity for fun.

The full LightGBM win predictor mostly collapses to a six-feature
"objective-pressure" score over eggs, berries, snail position,
and combat differentials. The extracted primitive is a
7-parameter int8 linear score, basically a 6-MAC dot product
plus optional sigmoid lookup table.

It is not a drop-in calibrated replacement for the full
LightGBM: it loses about 4 pp AUC and 1.5 pp Brier. But the
hardware quantization adds essentially no extra loss, and the
Vitis HLS estimate on KV260 is tiny:

  logit-only:      269 LUT / 180 FF / 2 DSP / 0 BRAM / 4 cycles / 289 MHz
  logit + sigmoid: 287 LUT / 283 FF / 2 DSP / 1 BRAM / 7 cycles / 289 MHz

I also ran a direct model-to-fabric baseline with Conifer using
a comparable 50-tree / depth-5 XGBoost BDT. It keeps about
+2.5 pp AUC over the extracted primitive but costs ~40k LUT on
KV260 versus 269 LUT for the objective-pressure score. So the
tradeoff is pretty stark: the full tree-style model buys
fidelity, while the extracted score buys a ~150× smaller circuit
that you could fit anywhere.

Two caveats worth flagging:

  - The Conifer baseline is a comparable XGBoost BDT, not a
    bit-exact compile of your original LightGBM. Conifer's
    LightGBM→ONNX path hit a parser bug (`KeyError('base_values')`
    inside Conifer 1.8), so I trained a same-budget XGBoost
    classifier and used Conifer's native XGBoost converter.
    Reading: "what cost class does keeping a tree ensemble in
    fabric land in?" — not "what does your LightGBM compile to?"

  - Trained and evaluated on the 16 benchmark CSV shards in
    `tests/`, ~3000 games, not the full encoded datasets your
    `CLAUDE.md` points at. The primitive is stable on that slice;
    a refit on the full corpus might shift the int8 coefficients
    a little but won't change the headline.

There is also a tiny side artifact for the demo: I wrote
RP2040/Pico firmware that implements the exact same pure-int
primitive over USB CDC serial. Hand-built so it would be ready
before any hardware lands. Native (no-Pico) bit-exact test
passes 152/152 vectors. After your Pico arrives we flash a UF2
and the board returns the same int16 logit + uint16 probability
the FPGA design would, over a USB cable.

I have a local branch (`case-f-hw-extraction`, 10 commits) with:

  docs/kquity_hw_extraction.md             single-file summary
  experiments/kquity_hw/                    Phase 0-8 artifacts
    inventory.md                              52-feature decode
    baseline_eval.py / surrogate_ladder.py     extraction probe
    route_b_interactions.py                    rejected interaction tightening
    phase4_fixed_point.py                       int8 parity sweep
    hls/                                         Vitis HLS C++ + csynth
    conifer/                                     model-to-fabric baseline
    results/phase5_hls_report.md                 deep-dive resource report
  hardware/kquity_usb_rp2040/                 Pico firmware bundle

I can send it as a git bundle, a tarball of .patch files, open a
PR against your `main`, push a side branch, or just keep it as a
one-off artifact — your call. Nothing has been pushed upstream.

Want me to package it any particular way? Happy to walk through
the linear primitive or the Conifer contrast if useful, but the
single-file summary in `docs/kquity_hw_extraction.md` has the
whole picture.

---

Associated artifacts (when ready to share):

  branch case-f-hw-extraction on local KQuity
  commits 07e8bed -> (HEAD currently after Phase 8) on top of
  rrenaud/KQuity master at 70c92ad
  /tmp/kquity-case-f.bundle      git bundle (~110 KB)
  /tmp/kquity-case-f-patches/    format-patch tarball (9 files)

Nothing pushed upstream. No PR opened. Awaiting Rob's routing.
