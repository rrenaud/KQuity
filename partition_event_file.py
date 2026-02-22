# Read a big csv file of game events, and partition it into smaller csv files such that
# events are grouped sequentially by game_id and sorted by timestamp. Individual games will
# be in single files.

import collections
import csv
import io
import os


def main() -> None:
    file_to_partition: str = 'validated_all_gameevent.csv'
    base_name: str = file_to_partition.split('.')[0]
    output_dir: str = f'{base_name}_partitioned'
    os.mkdir(output_dir)
    output_per_game_id: collections.Counter[int] = collections.Counter()
    with open(file_to_partition, 'r') as f:
        for row in csv.DictReader(f):
            output_per_game_id[int(row['game_id'])] += 1

    sorted_game_ids: list[int] = sorted(output_per_game_id.keys())
    partition_mapping: dict[int, int] = {}
    counts_per_partition: collections.Counter[int] = collections.Counter()
    for idx, game_id in enumerate(sorted_game_ids):
        partition: int = idx // 1000
        partition_mapping[game_id] = partition
        counts_per_partition[partition] += 1

    output_writers: dict[int, csv.DictWriter[str]] = {}
    output_files: dict[int, io.TextIOWrapper] = {}
    buffered_rows_per_game: collections.defaultdict[int, list[dict[str, str]]] = collections.defaultdict(list)

    with open(file_to_partition, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = int(row['game_id'])
            partition = partition_mapping[game_id]

            if partition not in output_files:
                output_file = open(f'{output_dir}/gameevents_{partition:03d}.csv', 'w')
                output_writers[partition] = csv.DictWriter(output_file, fieldnames=reader.fieldnames or [])
                output_writers[partition].writeheader()
                output_files[partition] = output_file

            buffered_rows_per_game[game_id].append(row)

            if len(buffered_rows_per_game[game_id]) == output_per_game_id[game_id]:
                buffered_rows_per_game[game_id].sort(key=lambda x: x['timestamp'])
                print('writing', game_id, partition)
                for output_row in buffered_rows_per_game[game_id]:
                    output_writers[partition].writerow(output_row)
                del partition_mapping[game_id]
                del buffered_rows_per_game[game_id]

                counts_per_partition[partition] -= 1
                if counts_per_partition[partition] == 0:
                    output_files[partition].close()
                    del output_files[partition]
                    del output_writers[partition]


if __name__ == '__main__':
    main()
