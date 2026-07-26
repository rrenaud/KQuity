# KQuity → Hardware Primitive (Case F)

Action-variable extraction PoC on the KQuity Killer Queen win-probability
LightGBM oracle. Goal: produce a tiny hardware-shaped primitive that
approximates the 100-tree / 52-feature classifier closely enough to be a
fun friend-demo, not yet a paper claim.

## Pipeline (per ChatGPT Pro directive)

```
Phase 0  repo + artifact inventory                  (this directory)
Phase 1  baseline LightGBM oracle metrics
Phase 2  surrogate ladder + action-variable hypotheses
Phase 3  extraction target selection
Phase 4  fixed-point primitive
Phase 5  HLS sketch (Vitis HLS csynth)
Phase 6  Conifer/hls4ml model-to-fabric fallback (only if extraction fails)
```

We do **not** start with Conifer/hls4ml — that is the fallback baseline.
The thesis is the small extracted primitive.

## Files

```
README.md              this file
inventory.md           model + dataset inventory, 52-feature decode
baseline_eval.py       Phase 1 baseline metrics + sanity probes
results/               per-phase JSON outputs
```

## Environment

- Python venv at `~/envs/kquity/` with `lightgbm`, `numpy`, `pandas`,
  `scikit-learn`, `scipy`. CPU only.
- Source the venv before running scripts:
  ```
  source ~/envs/kquity/bin/activate
  ```

## Model and data

- Oracle: `current_preferred_model.mdl` →
  `model_experiments/combined_li_qf/qf_200k_symaug_100l_100t.mdl`
  (LightGBM, 100 trees × 100 leaves, 52 features, P(gold wins))
- Eval data: `tests/benchmark_events_*.csv.gz` (16 shards) — local
  CSV-encoded games. The compact `.bin` datasets referenced in
  `CLAUDE.md` are not present locally; the test shards are enough
  for Phase 0/1/2 fidelity work.

## Decisions log

- 2026-05-15: start. Inventory + Phase 1 baseline next.
