# logged_in_games/

Games filtered to those with at least one logged-in user. Source: all ~925 shards in `unfiltered_partitioned/`.

## How it was created

1. **Login counting** — For each game, count rows in `unfiltered_partitioned/usergame.csv` with a non-empty `user_id`. Games with `login_count >= 1` are kept.

2. **Sorting** — Kept games are sorted by `login_count DESC`, then `start_time DESC` (from `unfiltered_partitioned/game.csv`). More logged-in users = earlier partition = higher quality signal.

3. **Resharding** — Sorted games are written 1000 per partition to `gameevents_000.csv.gz`, `gameevents_001.csv.gz`, etc.

Script: `create_logged_in_partitions.py`

## Sorting

Sorted by login count descending, then start time descending. Partition 000 contains games with the most logged-in users.

## Encoding

`encode_datasets.py` converts all partitions into `encoded/all_games.bin` — a compact binary format (~2-3 bytes/event). Games with >60s event gaps or missing gamestart/mapstart are rejected during encoding.

## Size

~199K games across 199 partitions.
