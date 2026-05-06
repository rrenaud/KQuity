"""Single CSV pass: extract every playerKill event for night-map games in
the quality-filtered set, with the game's quality rank (= position in
all_games.bin, which is sorted by quality_score DESC).

Columns:
    game_id        int64
    rank           int32   (0 = highest quality game)
    gold_on_left   bool
    x_canon        int16   (mirrored to gold-on-left frame)
    y              int16   (raw — Cartesian, y=0 at bottom of screen)
    killer_pid     int8
    killed_pid     int8
    killed_cat     str     (Queen/Soldier/Worker)
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
OUT_PATH = os.path.join(os.path.dirname(__file__), "night_kills.parquet")

MAP_NIGHT_IDX = 1
SCREEN_WIDTH = 1920


def load_night_targets() -> dict[int, tuple[int, bool]]:
    """{game_id: (rank, gold_on_left)} for night-map games."""
    out: dict[int, tuple[int, bool]] = {}
    with open(BIN_PATH, "rb") as f:
        (num_games,) = struct.unpack("<I", f.read(4))
        for rank in range(num_games):
            game_id, length = struct.unpack("<IH", f.read(6))
            header = f.read(1)[0]
            map_idx = (header >> 1) & 0x3
            gol = bool(header & 1)
            f.seek(length - 1, 1)
            if map_idx == MAP_NIGHT_IDX:
                out[game_id] = (rank, gol)
    return out


def main() -> None:
    target = load_night_targets()
    print(f"Night-map games: {len(target)}", file=sys.stderr)

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
                t = target.get(gid)
                if t is None:
                    continue
                rank, gol = t
                vals = row[3][1:-1].split(",")
                x = int(vals[0])
                y = int(vals[1])
                killer_pid = int(vals[2])
                killed_pid = int(vals[3])
                killed_cat = vals[4]
                x_canon = x if gol else SCREEN_WIDTH - x
                rows.append((gid, rank, gol, x_canon, y,
                             killer_pid, killed_pid, killed_cat))
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            print(f"  [{i + 1}/{len(files)}] kills so far: {len(rows):,}",
                  file=sys.stderr)

    df = pd.DataFrame(rows, columns=[
        "game_id", "rank", "gold_on_left",
        "x_canon", "y", "killer_pid", "killed_pid", "killed_cat",
    ])
    df = df.astype({
        "game_id": "int64", "rank": "int32", "gold_on_left": "bool",
        "x_canon": "int16", "y": "int16",
        "killer_pid": "int8", "killed_pid": "int8",
    })
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  ({len(df):,} kills)", file=sys.stderr)


if __name__ == "__main__":
    main()
