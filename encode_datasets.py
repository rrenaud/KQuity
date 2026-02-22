#!/usr/bin/env python3
"""One-time script to encode CSV.gz game data into packed binary format.

Processes one partition file at a time to keep memory bounded.
Peak memory ~ one partition's worth of CSV events (~1000 games).
"""

import csv
import datetime
import glob
import gzip
import os
import shutil
import time
from typing import Any

from event_codec import (
    encode_game, write_packed_games,
    SKIP_EVENTS, COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
)
from fast_materialize import _parse_ts


def encode_single_csv(csv_path: str) -> tuple[list[tuple[int, bytes]], int]:
    """Encode games from one CSV/gzip file.

    Returns list of (game_id, encoded_bytes) for successfully encoded games,
    and count of rejected games.
    """
    games: dict[int, list[tuple[datetime.datetime, str, str]]] = {}
    game_order: list[int] = []

    opener: Any = gzip.open if csv_path.endswith('.gz') else open
    with opener(csv_path, 'rt') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            event_type = row[COL_TYPE]
            if event_type in SKIP_EVENTS:
                continue
            game_id = int(row[COL_GAME_ID])
            if game_id not in games:
                games[game_id] = []
                game_order.append(game_id)
            games[game_id].append(
                (_parse_ts(row[COL_TS]), event_type, row[COL_VALUES]))

    entries: list[tuple[int, bytes]] = []
    rejected: int = 0
    for game_id in game_order:
        encoded = encode_game(games[game_id])
        if encoded is None:
            rejected += 1
            continue
        entries.append((game_id, encoded))
    # CSV data freed when function returns
    return entries, rejected


def encode_directory(csv_dir: str, out_path: str) -> tuple[int, int]:
    """Encode all CSV.gz partitions in a directory to a single packed binary file.

    Processes one partition at a time to keep memory bounded.
    """
    pattern: str = os.path.join(csv_dir, 'gameevents_*.csv.gz')
    csv_files: list[str] = sorted(glob.glob(pattern))
    if not csv_files:
        print(f"  No files matching {pattern}")
        return 0, 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_entries: list[tuple[int, bytes]] = []
    total_rejected: int = 0
    start: float = time.time()

    for i, csv_path in enumerate(csv_files):
        entries, rejected = encode_single_csv(csv_path)
        all_entries.extend(entries)
        total_rejected += rejected
        if (i + 1) % 20 == 0 or (i + 1) == len(csv_files):
            print(f"  [{i+1}/{len(csv_files)}] {len(all_entries):,} games encoded, "
                  f"{total_rejected:,} rejected")

    write_packed_games(all_entries, out_path)
    elapsed: float = time.time() - start
    size_mb: float = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Total: {len(all_entries):,} games, {total_rejected:,} rejected, "
          f"{size_mb:.1f} MB, {elapsed:.1f}s")
    return len(all_entries), total_rejected


def main() -> None:
    print("Encoding quality_filtered...")
    encode_directory('quality_filtered', 'quality_filtered/encoded/all_games.bin')

    print("\nEncoding logged_in_games...")
    encode_directory('logged_in_games', 'logged_in_games/encoded/all_games.bin')

    # Tournament games: copy from KQuity repo
    src = '/home/rrenaud/KQuity/late_tournament_games/encoded/all_games.bin'
    dst = 'late_tournament_games/encoded/all_games.bin'
    print(f"\nCopying tournament games: {src} -> {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"  {size_mb:.1f} MB copied")

    print("\nDone!")


if __name__ == '__main__':
    main()
