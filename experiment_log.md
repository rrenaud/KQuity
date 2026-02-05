# fast_materialize.py Optimization Experiment Log

Benchmark: 3000-game test suite, 233K output rows x 52 features.
Machine: Linux 6.8.0-90-generic, Python 3.11.14, numpy (system).
All timings: 5-run mean (with warmup), `time.perf_counter()`.

## Baseline (before any changes)

| Metric | Value |
|--------|-------|
| Mean time | 2.07s |
| Min time | 1.97s |
| Output dtype | float64 |
| Label dtype | int64 |
| States memory | 94,875 KB |
| Labels memory | 1,825 KB |
| **Total output memory** | **96,700 KB** |

## Experiment 1: All 7 changes at once (REVERTED partially)

Applied all 7 planned optimizations simultaneously:
1. Timestamps as floats (epoch seconds via `.timestamp()`)
2. Direct numpy indexed writes (`buf[idx, col+N] = val`)
3. float32 output buffer
4. int8 label buffer
5. Skip RNG when drop_prob=0
6. Pre-split values_str once per event
7. np.empty instead of np.zeros

| Metric | Value |
|--------|-------|
| Mean time | **2.63s (+27% SLOWER)** |
| Min time | 2.43s |
| Output dtype | float32 |
| Label dtype | int8 |
| States memory | 47,437 KB |
| Labels memory | 228 KB |
| **Total output memory** | **47,665 KB (-51%)** |

**Result: Memory halved but wall-clock regressed badly.**

### Root cause analysis

- **Change 2 (direct indexed writes) hurt performance.** Each `buf[idx, col] = val`
  is a separate Python->C boundary crossing with numpy indexing overhead.
  The old approach builds a 52-element Python list (cheap in CPython) and does
  a single bulk `buf[idx] = list` assignment, which numpy converts in one C loop.
  ~52 individual indexed writes per row x 233K rows = ~12M extra numpy index ops.

- **Change 1 (timestamp floats) hurt performance.** `fromisoformat()` is fast C code,
  but chaining `.timestamp()` adds timezone/epoch conversion overhead per parse.
  The old `(dt - gamestart_dt).total_seconds()` uses fast C-level datetime
  subtraction. Net effect: slower parsing, no gain on the subtraction side.

## Experiment 2: Keep only beneficial changes (FINAL)

Reverted changes 1 and 2, kept changes 3-7:
- ~~1. Timestamps as floats~~ (reverted — `.timestamp()` adds overhead)
- ~~2. Direct numpy indexed writes~~ (reverted — bulk list assignment is faster)
- 3. float32 output buffer
- 4. int8 label buffer
- 5. Skip RNG when drop_prob=0
- 6. Pre-split values_str once per event
- 7. np.empty instead of np.zeros

| Metric | Value |
|--------|-------|
| Mean time | **1.63s (-21% faster)** |
| Min time | 1.62s |
| Output dtype | float32 |
| Label dtype | int8 |
| States memory | 47,437 KB |
| Labels memory | 228 KB |
| **Total output memory** | **47,665 KB (-51%)** |

**Result: 21% faster AND memory halved. Best of both worlds.**

### Breakdown of kept changes

| Change | Effect |
|--------|--------|
| float32 output | Halves buffer memory; numpy bulk-writes 52 floats into smaller buffer slightly faster |
| int8 labels | 8x smaller label buffer; negligible time impact |
| Skip RNG when drop_prob=0 | Saves ~233K `rng.random()` calls in benchmark/test mode |
| Pre-split values_str | Eliminates redundant `[1:-1].split(',')` per event branch |
| np.empty vs np.zeros | Avoids zeroing memory that will be overwritten; minor |

## Key takeaway

In CPython, **bulk assignment** (`buf[idx] = python_list`) beats **element-wise indexed
writes** (`buf[idx, N] = val`) for numpy arrays. The per-element approach has too much
Python->C overhead. Building a temporary Python list is essentially free compared to
the numpy indexing machinery invoked 52 times per row.
