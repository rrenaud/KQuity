#!/usr/bin/env python3
"""Apply the quality classifier to all unfiltered shards.

Two-phase pipeline:
  --score:   Load trained model, score all 925 shards, write game_scores.parquet
  --reshard: Filter & sort by score, write quality_filtered/ partitions

Requires a trained model. Run this first:
    python -m game_quality_classifier.train_quality_classifier

Usage (from repo root):
    python -m game_quality_classifier.apply_quality_filter --score
    python -m game_quality_classifier.apply_quality_filter --reshard
    python -m game_quality_classifier.apply_quality_filter --score --reshard
"""

import argparse
import csv
import glob
import gzip
import json
import os
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .game_quality_features import compute_quality_features, FEATURE_COLUMNS
from .train_quality_classifier import CACHE_DIR, MODEL_PATH, THRESHOLD_PATH

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(REPO_ROOT, 'unfiltered_partitioned')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'quality_filtered')
SCORES_PATH = os.path.join(CACHE_DIR, 'game_scores.parquet')
GAMES_PER_PARTITION = 1000
GAME_ID_COL = 4  # game_id is column index 4 in gameevents CSV


def _load_model():
    """Load trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f'{MODEL_PATH} not found. '
            'Run `python -m game_quality_classifier.train_quality_classifier` first.')
    return lgb.Booster(model_file=MODEL_PATH)


def _load_thresholds():
    """Load thresholds from disk, return dict with threshold_99 and threshold_95."""
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f'{THRESHOLD_PATH} not found. '
            'Run `python -m game_quality_classifier.train_quality_classifier` first.')
    with open(THRESHOLD_PATH) as f:
        return json.load(f)


def score_all_shards(model):
    """Score all unfiltered shards one at a time, return DataFrame."""
    shard_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, 'gameevents_*.csv.gz')))
    print(f'\nScoring {len(shard_paths)} shards...')

    all_scores = []
    t0 = time.time()

    for i, shard_path in enumerate(shard_paths):
        df = compute_quality_features(shard_path)
        if len(df) == 0:
            continue
        scores = model.predict(df[FEATURE_COLUMNS].values)
        all_scores.append(pd.DataFrame({
            'game_id': df['game_id'].values,
            'quality_score': scores,
        }))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            total_scored = sum(len(s) for s in all_scores)
            print(f'  {i + 1}/{len(shard_paths)} shards '
                  f'({elapsed:.0f}s, {total_scored:,} games scored)')

    elapsed = time.time() - t0
    scores_df = pd.concat(all_scores, ignore_index=True)
    print(f'Scoring complete: {len(scores_df):,} games in {elapsed:.0f}s')

    return scores_df


def phase_score(threshold_pct=99):
    """Phase 1: Load trained model and score all shards."""
    print('=== Phase 1: Score ===')
    model = _load_model()
    thresholds = _load_thresholds()
    threshold = thresholds[f'threshold_{threshold_pct}']
    print(f'Loaded model from {MODEL_PATH}')
    print(f'Threshold ({threshold_pct}% tournament recall): {threshold:.4f}')

    scores_df = score_all_shards(model)

    pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
    scores_df.to_parquet(SCORES_PATH, index=False)
    print(f'\nWrote {SCORES_PATH}')

    # Summary stats
    print(f'\n=== Score Summary ===')
    print(f'Total games scored: {len(scores_df):,}')
    print(f'Score distribution:')
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f'  p{pct}: {np.percentile(scores_df["quality_score"], pct):.4f}')
    print(f'Pass rate at threshold {threshold:.4f}: '
          f'{(scores_df["quality_score"] >= threshold).mean():.1%}')

    return threshold


def phase_reshard(threshold=None, threshold_pct=99):
    """Phase 2: Filter and reshard by quality score."""
    print('=== Phase 2: Reshard ===')

    if not os.path.exists(SCORES_PATH):
        raise FileNotFoundError(
            f'{SCORES_PATH} not found. Run --score first.')

    scores_df = pd.read_parquet(SCORES_PATH)
    print(f'Loaded {len(scores_df):,} game scores')

    if threshold is None:
        thresholds = _load_thresholds()
        threshold = thresholds[f'threshold_{threshold_pct}']
        print(f'Loaded {threshold_pct}% threshold from {THRESHOLD_PATH}')

    # Filter and sort
    passing = scores_df[scores_df['quality_score'] >= threshold]
    passing = passing.sort_values('quality_score', ascending=False)
    print(f'Games passing threshold {threshold:.4f}: '
          f'{len(passing):,} ({len(passing)/len(scores_df):.1%})')

    # Build partition assignments
    game_to_partition = {
        row.game_id: idx // GAMES_PER_PARTITION
        for idx, row in enumerate(passing.itertuples())
    }
    num_partitions = (len(passing) + GAMES_PER_PARTITION - 1) // GAMES_PER_PARTITION
    print(f'Output partitions: {num_partitions}')

    # Stream and repartition
    _stream_and_repartition(game_to_partition)


def _stream_and_repartition(game_to_partition):
    """Stream source partitions and write qualifying games to output."""
    # Clear stale output files to ensure idempotency across re-runs
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stale = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv.gz')]
    if stale:
        print(f'  Clearing {len(stale)} existing output files in {OUTPUT_DIR}')
        for f in stale:
            os.remove(os.path.join(OUTPUT_DIR, f))

    open_writers = {}  # partition_num -> (gzip_file, csv_writer)
    header = None
    games_written = set()
    events_written = 0

    source_files = sorted(
        f for f in os.listdir(SOURCE_DIR)
        if f.startswith('gameevents_') and f.endswith('.csv.gz')
    )
    print(f'\nStreaming {len(source_files)} source partitions...')

    t0 = time.time()
    try:
        for file_idx, filename in enumerate(source_files):
            filepath = os.path.join(SOURCE_DIR, filename)

            with gzip.open(filepath, 'rt') as f:
                reader = csv.reader(f)
                file_header = next(reader)
                if header is None:
                    header = file_header

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
                print(f'  Processed {file_idx + 1}/{len(source_files)} files '
                      f'({elapsed:.0f}s, {len(games_written):,} games, '
                      f'{events_written:,} events)')
    finally:
        # Close all open writers even if an exception occurs
        for part_num, (gz_file, _) in open_writers.items():
            gz_file.close()

    # Log missing games (qualifying but not found in source files)
    missing = set(game_to_partition) - games_written
    if missing:
        print(f'\n  WARNING: {len(missing):,} qualifying games not found in source files')

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s')
    print(f'  Games written: {len(games_written):,}')
    print(f'  Events written: {events_written:,}')
    print(f'  Output files: '
          f'{len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv.gz")])}')


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
    parser = argparse.ArgumentParser(
        description='Apply quality classifier to filter/sort unfiltered data')
    parser.add_argument('--score', action='store_true',
                        help='Phase 1: Load trained model and score all shards')
    parser.add_argument('--reshard', action='store_true',
                        help='Phase 2: Filter by score and write quality_filtered/')
    parser.add_argument('--threshold-pct', type=int, default=99, choices=[95, 99],
                        help='Tournament recall percentile for threshold (default: 99)')
    args = parser.parse_args()

    if not args.score and not args.reshard:
        parser.error('Specify --score, --reshard, or both')

    threshold = None
    if args.score:
        threshold = phase_score(args.threshold_pct)
    if args.reshard:
        phase_reshard(threshold, args.threshold_pct)


if __name__ == '__main__':
    main()
