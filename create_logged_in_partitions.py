"""Create logged_in_games/ partitioned directory.

Filters game events to only include games with at least one logged-in user,
sorted by login count DESC then start_time DESC. This enables fast experiments
on high-quality data without scanning all 925 source partitions.
"""

import collections
import csv
import gzip
import os
import time


def compute_login_counts(usergame_csv_path):
    """Count non-empty user_id entries per game_id in usergame.csv."""
    counter = collections.Counter()
    with open(usergame_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['user_id']:
                counter[int(row['game_id'])] += 1
    return dict(counter)

SOURCE_DIR = 'new_data_partitioned'
OUTPUT_DIR = 'logged_in_games'
GAME_CSV = os.path.join(SOURCE_DIR, 'game.csv')
USERGAME_CSV = os.path.join(SOURCE_DIR, 'usergame.csv')
GAMES_PER_PARTITION = 1000
GAME_ID_COL = 4  # game_id is column index 4 in gameevents CSV


def build_partition_assignments():
    """Build game_id -> output_partition mapping.

    Games with login_count >= 1, sorted by (login_count DESC, start_time DESC).
    """
    print("Computing login counts...")
    login_counts = compute_login_counts(USERGAME_CSV)
    print(f"  Games with logins: {len(login_counts):,}")

    print("Reading game start times...")
    game_start_times = {}
    with open(GAME_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = int(row['id'])
            if game_id in login_counts:
                game_start_times[game_id] = row['start_time']

    print(f"  Games with logins AND in game.csv: {len(game_start_times):,}")

    # Sort by (login_count DESC, start_time DESC)
    sorted_games = sorted(
        game_start_times.keys(),
        key=lambda gid: (login_counts[gid], game_start_times[gid]),
        reverse=True,
    )

    game_to_partition = {
        gid: idx // GAMES_PER_PARTITION
        for idx, gid in enumerate(sorted_games)
    }
    num_partitions = (len(sorted_games) + GAMES_PER_PARTITION - 1) // GAMES_PER_PARTITION
    print(f"  Output partitions: {num_partitions}")

    # Show top games for verification
    for gid in sorted_games[:5]:
        print(f"    game {gid}: logins={login_counts[gid]}, start={game_start_times[gid]}")

    return game_to_partition, num_partitions


def stream_and_repartition(game_to_partition):
    """Stream source partitions and write events to output partitions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    open_writers = {}  # partition_num -> (gzip_file, csv_writer)
    header = None
    games_written = set()
    events_written = 0

    # Find all source partition files
    source_files = sorted(
        f for f in os.listdir(SOURCE_DIR)
        if f.startswith('gameevents_') and f.endswith('.csv.gz')
    )
    print(f"\nStreaming {len(source_files)} source partitions...")

    t0 = time.time()
    for file_idx, filename in enumerate(source_files):
        filepath = os.path.join(SOURCE_DIR, filename)

        with gzip.open(filepath, 'rt') as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header

            # Buffer events for one game at a time
            current_game_id = None
            current_buffer = []

            for row in reader:
                game_id = int(row[GAME_ID_COL])

                if game_id != current_game_id:
                    # Flush previous game's buffer
                    if current_buffer and current_game_id in game_to_partition:
                        _flush_game(
                            current_game_id, current_buffer,
                            game_to_partition, open_writers, header,
                        )
                        games_written.add(current_game_id)
                        events_written += len(current_buffer)

                    current_game_id = game_id
                    current_buffer = [row]
                else:
                    current_buffer.append(row)

            # Flush last game in file
            if current_buffer and current_game_id in game_to_partition:
                _flush_game(
                    current_game_id, current_buffer,
                    game_to_partition, open_writers, header,
                )
                games_written.add(current_game_id)
                events_written += len(current_buffer)

        if (file_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {file_idx + 1}/{len(source_files)} files "
                  f"({elapsed:.0f}s, {len(games_written):,} games, "
                  f"{events_written:,} events)")

    # Close all remaining open writers
    for part_num, (gz_file, _) in open_writers.items():
        gz_file.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Games written: {len(games_written):,}")
    print(f"  Events written: {events_written:,}")
    print(f"  Output files: {len(os.listdir(OUTPUT_DIR))}")


def _flush_game(game_id, buffer, game_to_partition, open_writers, header):
    """Write buffered events for a game to its output partition."""
    part_num = game_to_partition[game_id]

    if part_num not in open_writers:
        outpath = os.path.join(OUTPUT_DIR, f'gameevents_{part_num:03d}.csv.gz')
        gz_file = gzip.open(outpath, 'wt')
        writer = csv.writer(gz_file)
        writer.writerow(header)
        open_writers[part_num] = (gz_file, writer)

    _, writer = open_writers[part_num]
    writer.writerows(buffer)


def main():
    print("=== Phase 1: Build partition assignments ===")
    game_to_partition, num_partitions = build_partition_assignments()

    print("\n=== Phase 2: Stream and re-partition ===")
    stream_and_repartition(game_to_partition)

    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"Expected partitions: {num_partitions}")
    actual = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv.gz')])
    print(f"Actual partitions: {actual}")


if __name__ == '__main__':
    main()
