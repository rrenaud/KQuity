"""Extract every playerKill event for ALL quality-filtered games (any
map), with the game's quality rank and map.

Same one-pass strategy as extract_kills_full.py. Skips filtering by
night-map so we can compare across maps.
"""
import csv
import gzip
import glob
import os
import struct
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_PATH = os.path.join(REPO, "quality_filtered/encoded/all_games.bin")
CSV_GLOB = os.path.join(REPO, "unfiltered_partitioned/gameevents_*.csv.gz")
OUT_PATH = os.path.join(os.path.dirname(__file__), "all_kills.parquet")

MAP_NAMES = ["map_day", "map_night", "map_dusk", "map_twilight"]
SCREEN_WIDTH = 1920


def load_targets() -> dict[int, tuple[int, int, bool]]:
    """{game_id: (rank, map_idx, gold_on_left)} for ALL games in the bin."""
    out: dict[int, tuple[int, int, bool]] = {}
    with open(BIN_PATH, "rb") as f:
        (num_games,) = struct.unpack("<I", f.read(4))
        for rank in range(num_games):
            game_id, length = struct.unpack("<IH", f.read(6))
            header = f.read(1)[0]
            map_idx = (header >> 1) & 0x3
            gol = bool(header & 1)
            f.seek(length - 1, 1)
            out[game_id] = (rank, map_idx, gol)
    return out


def main() -> None:
    target = load_targets()
    print(f"Total games in bin: {len(target)}", file=sys.stderr)

    # Use every game in the quality-filtered set.
    by_map: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, midx, _ in target.values():
        by_map[midx] += 1
    keep: set[int] = set(target.keys())
    for midx, n in by_map.items():
        print(f"  map {MAP_NAMES[midx]}: {n} games kept",
              file=sys.stderr)

    rows: list[tuple] = []
    files = sorted(glob.glob(CSV_GLOB))
    for i, path in enumerate(files):
        with gzip.open(path, "rt") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[2] != "playerKill":
                    continue
                gid = int(row[4])
                if gid not in keep:
                    continue
                rank, midx, gol = target[gid]
                vals = row[3][1:-1].split(",")
                x = int(vals[0])
                y = int(vals[1])
                killer_pid = int(vals[2])
                killed_pid = int(vals[3])
                killed_cat = vals[4]
                x_canon = x if gol else SCREEN_WIDTH - x
                rows.append((gid, rank, midx, gol, x_canon, y,
                             killer_pid, killed_pid, killed_cat))
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  [{i + 1}/{len(files)}] kills so far: {len(rows):,}",
                  file=sys.stderr)

    df = pd.DataFrame(rows, columns=[
        "game_id", "rank", "map_idx", "gold_on_left",
        "x_canon", "y", "killer_pid", "killed_pid", "killed_cat",
    ])
    df = df.astype({
        "game_id": "int64", "rank": "int32",
        "map_idx": "int8", "gold_on_left": "bool",
        "x_canon": "int16", "y": "int16",
        "killer_pid": "int8", "killed_pid": "int8",
    })
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  ({len(df):,} kills)", file=sys.stderr)


if __name__ == "__main__":
    main()
