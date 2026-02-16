# quality_filtered/

Games filtered by a learned quality classifier. Source: all ~925 shards in `unfiltered_partitioned/`.

## How it was created

1. **Feature extraction** — 69 per-game features (duration, kill/carry/bless rates, temporal milestones, per-cabinet activity, etc.) computed by `game_quality_classifier/game_quality_features.py`.

2. **Model training** — A LightGBM binary classifier (`num_leaves=127`, `min_child_samples=75`, `lr=0.05`, AUC metric) trained to distinguish `logged_in_games` (positive) from raw `unfiltered_partitioned` (negative). Tournament games excluded to prevent leakage. Trained by `game_quality_classifier/train_quality_classifier.py`; saved to `quality_cache/quality_model.mdl`.

3. **Scoring** — Every game in `unfiltered_partitioned/` is scored. Cached in `quality_cache/game_scores.parquet`.

4. **Thresholding** — Keep games with `quality_score >= threshold_99` (0.3643), calibrated so 99% of tournament games pass. The threshold is stored in `quality_cache/threshold.json`.

5. **Resharding** — Passing games are sorted by `quality_score DESC` and written 1000 per partition to `gameevents_000.csv.gz`, `gameevents_001.csv.gz`, etc.

Script: `game_quality_classifier/apply_quality_filter.py`

## Sorting

Sorted by quality score descending. Partition 000 contains the highest-scored games.

## Encoding

`encode_datasets.py` converts all partitions into `encoded/all_games.bin` — a compact binary format (~2-3 bytes/event). Games with >60s event gaps or missing gamestart/mapstart are rejected during encoding.

## Size

~183K games across 183 partitions.
