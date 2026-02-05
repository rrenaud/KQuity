#!/usr/bin/env python3
"""
Script to:
1. Validate and partition the new data export
2. Create train/test split
3. Train a new model on the new data
4. Compare new model vs repro model on the new test set
"""

import collections
import csv
import gzip
import os
import pathlib
import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import sklearn.metrics

from preprocess import (
    iterate_events_from_csv,
    iterate_events_by_game_and_normalize_time,
    iterate_game_events_with_state,
    is_valid_game,
    create_game_states_matrix,
    GameValidationError,
)
import map_structure
from fast_materialize import fast_materialize


def load_tournament_game_ids(game_csv_path: str) -> Set[int]:
    """Load the set of game IDs that are tournament games.

    Args:
        game_csv_path: Path to game.csv file

    Returns:
        Set of game IDs with non-empty tournament_match_id
    """
    tournament_ids = set()
    with open(game_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['tournament_match_id']:  # non-empty
                tournament_ids.add(int(row['id']))
    return tournament_ids


def get_tournament_train_test_split(
    game_csv_path: str,
    test_count: int = 2000
) -> Tuple[Set[int], Set[int]]:
    """Split tournament games into train/test sets by time.

    Args:
        game_csv_path: Path to game.csv file
        test_count: Number of most recent tournament games for test set

    Returns:
        Tuple of (train_game_ids, test_game_ids)
    """
    # Load tournament games with their start times
    tournament_games = []  # [(game_id, start_time), ...]
    with open(game_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['tournament_match_id']:  # non-empty
                tournament_games.append((int(row['id']), row['start_time']))

    # Sort by start_time descending (most recent first)
    tournament_games.sort(key=lambda x: x[1], reverse=True)

    # Take most recent test_count as test set
    test_game_ids = set(g[0] for g in tournament_games[:test_count])
    train_game_ids = set(g[0] for g in tournament_games[test_count:])

    return train_game_ids, test_game_ids


def validate_and_partition_data(
    input_csv: str,
    output_dir: str,
    games_per_partition: int = 1000,
    min_timestamp: str = '2022-09',
    max_games: int = None
):
    """Validate game events and partition into smaller files.

    Uses a memory-efficient streaming approach:
    - Pass 1: Count events per game (no buffering)
    - Pass 2: Stream and validate games one at a time
    - Pass 3: Stream and write validated games to partitions

    Args:
        max_games: If set, limit to processing this many games (for debugging)
    """
    print(f"Validating and partitioning {input_csv}...")
    print(f"Output directory: {output_dir}")
    if max_games:
        print(f"DEBUG MODE: Limiting to {max_games} games")
    import sys
    sys.stdout.flush()

    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    from preprocess import parse_event, normalize_times

    # Pass 1: Count events per game (memory-efficient - just counts)
    print("Pass 1: Counting events per game...")
    sys.stdout.flush()
    start = time.time()

    events_per_game_id = collections.Counter()
    row_count = 0

    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            if row_count % 100000 == 0:
                print(f"  Read {row_count:,} rows, {len(events_per_game_id):,} unique games...")
                sys.stdout.flush()

            if row['timestamp'] <= min_timestamp:
                continue

            game_id = int(row['game_id'])
            events_per_game_id[game_id] += 1

            # Early exit if we have enough games
            if max_games and len(events_per_game_id) >= max_games:
                print(f"  Reached {max_games} games limit, stopping pass 1 early")
                sys.stdout.flush()
                break

    print(f"  Total rows read: {row_count:,}")
    print(f"  Unique games: {len(events_per_game_id):,}")
    print(f"Pass 1 completed in {time.time() - start:.1f}s")
    sys.stdout.flush()

    # Pass 2: Validate games one at a time (streaming)
    print("\nPass 2: Validating games...")
    sys.stdout.flush()
    start = time.time()

    validated_game_ids = set()
    validation_errors = collections.Counter()
    map_structure_infos = map_structure.MapStructureInfos()
    buffered_rows = collections.defaultdict(list)
    games_checked = 0
    rows_read = 0

    # Only process games we counted in pass 1
    game_ids_to_process = set(events_per_game_id.keys())

    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_read += 1
            if rows_read % 100000 == 0:
                print(f"  Pass 2: {rows_read:,} rows, {games_checked:,} games checked, {len(buffered_rows):,} buffered...")
                sys.stdout.flush()

            if row['timestamp'] <= min_timestamp:
                continue

            game_id = int(row['game_id'])
            if game_id not in game_ids_to_process:
                continue

            buffered_rows[game_id].append(row)

            # When we have all rows for a game, validate and release memory
            if len(buffered_rows[game_id]) == events_per_game_id[game_id]:
                games_checked += 1
                if games_checked % 5000 == 0:
                    print(f"  Validated {games_checked:,} games, {len(validated_game_ids):,} valid, {len(buffered_rows):,} buffered...")
                    sys.stdout.flush()

                # Parse events
                events = []
                for r in buffered_rows[game_id]:
                    try:
                        event = parse_event(r)
                        if event is not None:
                            events.append(event)
                    except Exception:
                        pass

                # Immediately delete buffer to free memory
                del buffered_rows[game_id]

                if not events:
                    validation_errors['no_events'] += 1
                    continue

                # Normalize and validate
                try:
                    normalized = normalize_times(events)
                    error = is_valid_game(normalized, map_structure_infos)
                    if error:
                        validation_errors[str(error)] += 1
                    else:
                        validated_game_ids.add(game_id)
                except Exception as e:
                    validation_errors[str(e)] += 1

                # Stop if we've processed all games we wanted
                if games_checked >= len(game_ids_to_process):
                    print(f"  Processed all {games_checked} games, stopping pass 2")
                    sys.stdout.flush()
                    break

    print(f"  Valid games: {len(validated_game_ids):,}")
    print(f"  Validation errors: {validation_errors.most_common(10)}")
    print(f"Pass 2 completed in {time.time() - start:.1f}s")
    sys.stdout.flush()

    # Pass 3: Write validated games to partitioned files (streaming)
    print("\nPass 3: Writing partitioned files...")
    print(f"  Will write {len(validated_game_ids)} validated games")
    sys.stdout.flush()
    start = time.time()

    sorted_game_ids = sorted(validated_game_ids)
    partition_mapping = {gid: idx // games_per_partition for idx, gid in enumerate(sorted_game_ids)}
    counts_per_partition = collections.Counter()
    for gid in sorted_game_ids:
        counts_per_partition[partition_mapping[gid]] += 1

    output_writers = {}
    output_files = {}
    buffered_rows = collections.defaultdict(list)
    games_written = 0
    rows_read = 0

    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            rows_read += 1
            if rows_read % 100000 == 0:
                print(f"  Pass 3: {rows_read:,} rows, {games_written:,} games written, {len(buffered_rows):,} buffered...")
                sys.stdout.flush()

            if row['timestamp'] <= min_timestamp:
                continue

            game_id = int(row['game_id'])
            if game_id not in validated_game_ids:
                continue

            buffered_rows[game_id].append(row)

            # When we have all rows for a game, write and release memory
            if len(buffered_rows[game_id]) == events_per_game_id[game_id]:
                partition = partition_mapping[game_id]

                if partition not in output_files:
                    output_file = gzip.open(f'{output_dir}/gameevents_{partition:03d}.csv.gz', 'wt')
                    output_writers[partition] = csv.DictWriter(output_file, fieldnames=fieldnames)
                    output_writers[partition].writeheader()
                    output_files[partition] = output_file

                # Sort by timestamp and write
                buffered_rows[game_id].sort(key=lambda x: x['timestamp'])
                for output_row in buffered_rows[game_id]:
                    output_writers[partition].writerow(output_row)

                # Free memory immediately
                del buffered_rows[game_id]
                del partition_mapping[game_id]
                games_written += 1

                # Close partition file when all its games are written
                counts_per_partition[partition] -= 1
                if counts_per_partition[partition] == 0:
                    output_files[partition].close()
                    del output_files[partition]
                    del output_writers[partition]
                    print(f"  Closed partition {partition}")
                    sys.stdout.flush()

                if games_written % 5000 == 0:
                    print(f"  Written {games_written:,}/{len(validated_game_ids):,} games...")
                    sys.stdout.flush()

                # Stop if we've written all validated games
                if games_written >= len(validated_game_ids):
                    print(f"  Written all {games_written} games, stopping pass 3")
                    sys.stdout.flush()
                    break

    # Close any remaining files
    for f in output_files.values():
        f.close()

    num_partitions = len(set(idx // games_per_partition for idx in range(len(sorted_game_ids)))) if sorted_game_ids else 0
    print(f"  Written {games_written:,} games to {num_partitions} partition files")
    print(f"Pass 3 completed in {time.time() - start:.1f}s")
    sys.stdout.flush()

    return num_partitions, len(validated_game_ids)


def materialize_partition_range(
    input_dir: str,
    output_dir: str,
    start_partition: int,
    end_partition: int,
    drop_prob: float,
    allowed_game_ids: Optional[Set[int]] = None,
    use_fast_path: bool = True
):
    """Materialize game states from a range of partitions.

    Args:
        input_dir: Directory containing partitioned CSV files
        output_dir: Directory for output (unused but kept for API compatibility)
        start_partition: First partition index to process
        end_partition: Last partition index (exclusive)
        drop_prob: Probability of dropping each state
        allowed_game_ids: If provided, only include games with IDs in this set
        use_fast_path: If True, use fast_materialize (default). If False, use slow path.
    """
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    all_states = []
    all_labels = []

    for partition in range(start_partition, end_partition):
        csv_path = f'{input_dir}/gameevents_{partition:03d}.csv.gz'
        if not os.path.exists(csv_path):
            continue

        print(f"  Processing {csv_path}...")

        if use_fast_path:
            try:
                states, labels = fast_materialize(csv_path, drop_prob, allowed_game_ids=allowed_game_ids)
                if len(labels) > 0:
                    all_states.append(states)
                    all_labels.append(labels)
            except Exception as e:
                print(f"    Error processing {csv_path}: {e}")
        else:
            map_structure_infos = map_structure.MapStructureInfos()
            events = iterate_events_from_csv(csv_path)
            game_states_iter = iterate_game_events_with_state(events, map_structure_infos)

            try:
                states, labels = create_game_states_matrix(
                    game_states_iter, drop_prob, noisy=False, allowed_game_ids=allowed_game_ids
                )
                if states is not None and len(states) > 0:
                    all_states.append(states)
                    all_labels.append(labels)
            except Exception as e:
                print(f"    Error processing {csv_path}: {e}")

    if all_states:
        combined_states = np.vstack(all_states)
        combined_labels = np.concatenate(all_labels)
        return combined_states, combined_labels
    return None, None


def load_vectors(pattern: str):
    """Load state and label vectors from numpy files."""
    X = np.load(f'{pattern}_states.npy')
    y = np.load(f'{pattern}_labels.npy')
    return X, y


def train_lgb_model(train_X, train_y, num_leaves=100, num_trees=100, weights=None):
    """Train a LightGBM model."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': 0
    }
    train_data = lgb.Dataset(train_X, train_y, weight=weights)
    return lgb.train(param, train_data, num_boost_round=num_trees)


def evaluate_model(model, test_X, test_y, name: str):
    """Evaluate a model and return metrics."""
    predictions = model.predict(test_X)

    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    # Egg inversion test
    mask = test_X[:, 0] != 2
    eligible_X = test_X[mask]
    sample_size = min(2000, len(eligible_X))
    if sample_size > 0:
        indices = np.random.choice(len(eligible_X), sample_size, replace=False)
        sample_X = eligible_X[indices]
        orig_preds = model.predict(sample_X)
        modified_X = sample_X.copy()
        modified_X[:, 0] += 1
        mod_preds = model.predict(modified_X)
        inversions = (mod_preds < orig_preds).mean()
    else:
        inversions = 0.0

    print(f"\n{name} Results:")
    print(f"  Log Loss: {log_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")
    print(f"  Egg Inversions: {inversions:.4f} ({100*inversions:.2f}%)")

    return {
        'log_loss': log_loss,
        'accuracy': accuracy,
        'inversions': inversions
    }


def run_tournament_only_experiment(
    game_csv_path: str = 'new_data_partitioned/game.csv',
    partitioned_dir: str = 'new_data_partitioned',
    test_count: int = 2000,
    drop_prob: float = 0.9,
):
    """Run Experiment 1: Train and evaluate on tournament games only.

    This establishes a baseline for expert-level play by:
    1. Filtering to tournament games only (non-empty tournament_match_id)
    2. Using the 2000 most recent tournament games as test set
    3. Training on remaining ~35k tournament games

    Args:
        game_csv_path: Path to game.csv with tournament_match_id column
        partitioned_dir: Directory with partitioned event files
        test_count: Number of most recent tournament games for test set
        drop_prob: Probability of dropping states during training materialization
    """
    expt_dir = 'model_experiments/tournament_only'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("EXPERIMENT 1: Tournament-Only Training Baseline")
    print("=" * 60)

    # Step 1: Load tournament train/test split
    print("\nStep 1: Loading tournament game IDs and creating train/test split...")
    train_ids, test_ids = get_tournament_train_test_split(game_csv_path, test_count)
    print(f"  Training games: {len(train_ids):,}")
    print(f"  Test games: {len(test_ids):,}")

    # Verify no overlap
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test overlap detected: {len(overlap)} games")
    print("  Verified: no overlap between train and test sets")

    # Count partitions
    num_partitions = len([f for f in os.listdir(partitioned_dir) if f.endswith('.csv.gz')])
    print(f"  Partitioned data files: {num_partitions}")

    # Step 2: Materialize training data (tournament games only)
    print("\n" + "=" * 60)
    print("Step 2: Materializing tournament training data")
    print("=" * 60)

    train_states_file = f'{expt_dir}/train_states.npy'
    if not os.path.exists(train_states_file):
        print(f"Materializing training data ({100*drop_prob:.0f}% drop rate)...")
        train_X, train_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=drop_prob,
            allowed_game_ids=train_ids
        )
        np.save(f'{expt_dir}/train_states.npy', train_X)
        np.save(f'{expt_dir}/train_labels.npy', train_y)
        print(f"  Training samples: {len(train_y):,}")
    else:
        print("Loading existing training data...")
        train_X = np.load(f'{expt_dir}/train_states.npy')
        train_y = np.load(f'{expt_dir}/train_labels.npy')
        print(f"  Training samples: {len(train_y):,}")

    # Step 3: Materialize test data (tournament games only, no drop)
    print("\n" + "=" * 60)
    print("Step 3: Materializing tournament test data")
    print("=" * 60)

    test_states_file = f'{expt_dir}/test_states.npy'
    if not os.path.exists(test_states_file):
        print("Materializing test data (0% drop rate)...")
        test_X, test_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.0,
            allowed_game_ids=test_ids
        )
        np.save(f'{expt_dir}/test_states.npy', test_X)
        np.save(f'{expt_dir}/test_labels.npy', test_y)
        print(f"  Test samples: {len(test_y):,}")
    else:
        print("Loading existing test data...")
        test_X = np.load(f'{expt_dir}/test_states.npy')
        test_y = np.load(f'{expt_dir}/test_labels.npy')
        print(f"  Test samples: {len(test_y):,}")

    # Step 4: Train model
    print("\n" + "=" * 60)
    print("Step 4: Training model on tournament data")
    print("=" * 60)

    print(f"Training data shape: {train_X.shape}")
    print(f"Test data shape: {test_X.shape}")

    start = time.time()
    model = train_lgb_model(train_X, train_y, num_leaves=200, num_trees=200)
    print(f"Model trained in {time.time() - start:.1f}s")

    model_path = f'{expt_dir}/model.mdl'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("Step 5: Evaluating on tournament test set")
    print("=" * 60)

    metrics = evaluate_model(model, test_X, test_y, "Tournament-Only Model")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 SUMMARY: Tournament-Only Baseline")
    print("=" * 60)
    print(f"Training set: {len(train_ids):,} games, {len(train_y):,} states")
    print(f"Test set: {len(test_ids):,} games, {len(test_y):,} states")
    print()
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'Log Loss':<20} {metrics['log_loss']:<15.4f}")
    print(f"{'Accuracy':<20} {metrics['accuracy']:<15.4f} ({100*metrics['accuracy']:.1f}%)")
    print(f"{'Egg Inversions':<20} {metrics['inversions']:<15.4f} ({100*metrics['inversions']:.2f}%)")

    return metrics


def get_non_tournament_game_ids(game_csv_path: str, exclude_game_ids: Set[int] = None) -> Set[int]:
    """Get game IDs for non-tournament games, optionally excluding some IDs."""
    non_tournament_ids = set()
    with open(game_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['tournament_match_id']:  # empty = non-tournament
                game_id = int(row['id'])
                if exclude_game_ids is None or game_id not in exclude_game_ids:
                    non_tournament_ids.add(game_id)
    return non_tournament_ids


def compute_hivemind_login_counts(usergame_csv_path: str) -> Dict[int, int]:
    """Count non-null user_id entries per game_id in usergame.csv.

    Args:
        usergame_csv_path: Path to usergame.csv file

    Returns:
        Dict mapping game_id -> count of non-null user_id entries
    """
    login_counts = collections.Counter()
    with open(usergame_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['user_id']:  # non-empty = logged in
                login_counts[int(row['game_id'])] += 1
    return dict(login_counts)


def build_tiered_game_ids(
    game_csv_path: str,
    usergame_csv_path: str,
    clustering_path: str,
) -> Tuple[Set[int], List[Tuple[str, List[int]]]]:
    """Build holdout set and tiered training game IDs.

    Returns:
        (holdout_ids, [(tier_name, ordered_game_ids), ...])
    """
    from tournament_clustering import load_clustering
    from extract_late_tournament_games import get_late_tournament_game_ids

    # Load clustering data
    match_to_tournament, tournaments = load_clustering(clustering_path)

    # Load game.csv into DataFrame
    games_df = pd.read_csv(game_csv_path)

    # Get holdout: late tournament game IDs
    holdout_ids = get_late_tournament_game_ids(games_df, tournaments)
    print(f"Holdout (late tournament) games: {len(holdout_ids):,}")

    # Identify major tournaments (num_teams > 15)
    major_tournaments = [t for t in tournaments if t.num_teams > 15]
    major_tournament_ids = set(t.tournament_id for t in major_tournaments)
    print(f"Major tournaments (>15 teams): {len(major_tournaments)}")

    # Build set of match_ids (as floats) belonging to major tournaments
    major_match_ids = set()
    for match_id, tourn_id in match_to_tournament.items():
        if tourn_id in major_tournament_ids:
            major_match_ids.add(float(match_id))

    # Tier 1: Games whose tournament_match_id maps to a major tournament, minus holdout
    tier1_ids = []
    # Also build lookup structures for tier 2
    all_game_ids = set()
    tournament_game_ids = set()  # games with any tournament_match_id
    game_cabinet = {}  # game_id -> cabinet_id
    game_start_time = {}  # game_id -> start_time string

    for _, row in games_df.iterrows():
        game_id = int(row['id'])
        all_game_ids.add(game_id)
        game_cabinet[game_id] = row['cabinet_id']
        game_start_time[game_id] = row['start_time']

        tmid = row['tournament_match_id']
        if pd.notna(tmid) and str(tmid) != '':
            tournament_game_ids.add(game_id)
            if float(tmid) in major_match_ids and game_id not in holdout_ids:
                tier1_ids.append(game_id)

    tier1_set = set(tier1_ids)
    print(f"Tier 1 (other major tournament games): {len(tier1_ids):,}")

    # Tier 2: Non-tournament games on same cabinet within +/-2 days of a major tournament
    # Build major tournament time/cabinet ranges
    major_tournament_ranges = []
    for t in major_tournaments:
        for cab in t.cabinets:
            # Find cabinet_id from cabinet_name
            cab_ids = games_df[games_df['cabinet_name'] == cab]['cabinet_id'].unique()
            t_start = pd.Timestamp(t.start_time)
            t_end = pd.Timestamp(t.end_time)
            for cab_id in cab_ids:
                major_tournament_ranges.append((cab_id, t_start - pd.Timedelta(days=2), t_end + pd.Timedelta(days=2)))

    non_tournament_ids = all_game_ids - tournament_game_ids
    tier2_ids = []
    for game_id in non_tournament_ids:
        if game_id in holdout_ids:
            continue
        cab = game_cabinet[game_id]
        try:
            gt = pd.Timestamp(game_start_time[game_id])
        except Exception:
            continue
        for range_cab, range_start, range_end in major_tournament_ranges:
            if cab == range_cab and range_start <= gt <= range_end:
                tier2_ids.append(game_id)
                break

    tier2_set = set(tier2_ids)
    print(f"Tier 2 (non-tourn near major tournaments): {len(tier2_ids):,}")

    # Tier 3: All games with tournament_match_id not already in Tier 1 or holdout
    tier3_ids = []
    for game_id in tournament_game_ids:
        if game_id not in tier1_set and game_id not in holdout_ids:
            tier3_ids.append(game_id)
    print(f"Tier 3 (other tournament games): {len(tier3_ids):,}")

    # Compute hivemind login counts for tiers 4 and 5
    print("Computing hivemind login counts...")
    login_counts = compute_hivemind_login_counts(usergame_csv_path)

    # Remaining non-tournament games (not in tier 2)
    remaining_non_tourn = []
    for game_id in non_tournament_ids:
        if game_id not in holdout_ids and game_id not in tier2_set:
            remaining_non_tourn.append(game_id)

    # Tier 4: Remaining non-tournament with login_count > 0, sorted by count DESC
    tier4_ids = [(game_id, login_counts.get(game_id, 0)) for game_id in remaining_non_tourn
                 if login_counts.get(game_id, 0) > 0]
    tier4_ids.sort(key=lambda x: x[1], reverse=True)
    tier4_ids = [game_id for game_id, _ in tier4_ids]
    print(f"Tier 4 (non-tourn with logins): {len(tier4_ids):,}")

    # Tier 5: Remaining non-tournament with 0 logins
    tier5_ids = [game_id for game_id in remaining_non_tourn
                 if login_counts.get(game_id, 0) == 0]
    print(f"Tier 5 (non-tourn no logins): {len(tier5_ids):,}")

    tiers = [
        ('tier1_major_tourn', tier1_ids),
        ('tier2_near_major', tier2_ids),
        ('tier3_other_tourn', tier3_ids),
        ('tier4_logins', tier4_ids),
        ('tier5_no_logins', tier5_ids),
    ]

    # Verify no overlap between holdout and any tier
    all_tier_ids = set()
    for name, ids in tiers:
        tier_set = set(ids)
        overlap = tier_set & holdout_ids
        assert not overlap, f"Tier {name} overlaps holdout by {len(overlap)} games"
        all_tier_ids.update(tier_set)

    print(f"\nTotal games across all tiers: {len(all_tier_ids):,}")
    print(f"Total games in dataset: {len(all_game_ids):,}")
    print(f"Coverage: {len(all_tier_ids) + len(holdout_ids):,} / {len(all_game_ids):,}")

    return holdout_ids, tiers


def run_tiered_doubling_experiment():
    """Experiment: Per-tier quality comparison.

    Materializes each tier individually, then for each tier's sample count
    as a threshold, trains a model per tier (capped at that threshold).
    Produces an NxN grid showing which tier is best at each data budget.
    """
    print("=" * 70)
    print("EXPERIMENT: Per-Tier Quality Comparison")
    print("=" * 70)

    game_csv_path = 'new_data_partitioned/game.csv'
    usergame_csv_path = 'new_data_partitioned/usergame.csv'
    clustering_path = 'tournament_clustering.json'
    partitioned_dir = 'new_data_partitioned'
    expt_dir = 'model_experiments/tiered_doubling'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    num_partitions = len([f for f in os.listdir(partitioned_dir) if f.endswith('.csv.gz')])
    print(f"Found {num_partitions} partitions")

    holdout_ids, tiers = build_tiered_game_ids(game_csv_path, usergame_csv_path, clustering_path)

    # Materialize test set
    print("\nMaterializing test set...")
    test_X, test_y = materialize_partition_range(
        partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.0,
        allowed_game_ids=holdout_ids
    )
    print(f"Test samples: {len(test_y):,}")

    # Materialize tiers 1-4 individually (skip tier 5 / no-logins for speed)
    print("\nMaterializing each tier individually...")
    tier_data = []  # (name, X, y)
    for name, ids in tiers[:4]:
        print(f"  {name} ({len(ids):,} games)...")
        X, y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.9,
            allowed_game_ids=set(ids)
        )
        if X is not None and len(X) > 0:
            tier_data.append((name, X, y))
            print(f"    -> {len(y):,} samples")

    # Sort by sample count ascending for clean threshold progression
    tier_data.sort(key=lambda t: len(t[2]))

    for name, _, y in tier_data:
        print(f"  {name}: {len(y):,} samples")

    # For each tier as focal: threshold = focal's sample count, train all tiers
    print("\n" + "=" * 70)
    print("Training models: each tier at each threshold")
    print("=" * 70)

    results = []
    for focal_name, _, focal_y in tier_data:
        threshold = len(focal_y)
        print(f"\n--- Threshold: {threshold:,} (= {focal_name}) ---")

        for train_name, X, y in tier_data:
            n = min(len(y), threshold)
            start_t = time.time()
            model = train_lgb_model(X[:n], y[:n], num_leaves=200, num_trees=200)
            train_time = time.time() - start_t
            metrics = evaluate_model(model, test_X, test_y, f"{train_name}@{n:,}")
            results.append({
                'threshold': threshold,
                'threshold_tier': focal_name,
                'train_tier': train_name,
                'n_train': n,
                'train_time': train_time,
                **metrics,
            })

    # Print grid: rows=train tier, cols=threshold (ascending)
    print("\n" + "=" * 70)
    print("RESULTS: log_loss (rows=training tier, cols=sample threshold)")
    print("=" * 70)

    thresholds = sorted(set(r['threshold'] for r in results))
    train_tiers = [name for name, _, _ in tier_data]

    header = f"{'train_tier':<25}"
    for t in thresholds:
        header += f" {t:>10,}"
    print(header)
    print("-" * len(header))

    for tn in train_tiers:
        line = f"{tn:<25}"
        for t in thresholds:
            match = [r for r in results if r['train_tier'] == tn and r['threshold'] == t]
            if match:
                line += f" {match[0]['log_loss']:>10.4f}"
            else:
                line += f" {'':>10}"
        print(line)

    # Also show accuracy grid
    print(f"\nRESULTS: accuracy")
    print("-" * len(header))
    for tn in train_tiers:
        line = f"{tn:<25}"
        for t in thresholds:
            match = [r for r in results if r['train_tier'] == tn and r['threshold'] == t]
            if match:
                line += f" {match[0]['accuracy']:>10.4f}"
            else:
                line += f" {'':>10}"
        print(line)

    # Save results
    results_path = f'{expt_dir}/tier_comparison_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def run_weight_sweep_experiment():
    """Experiment 2: Weight sweep over non-tournament game weights.

    Tests different weights for non-tournament games when combining with
    tournament training data. Also tests adding an 'is_tournament' feature.
    """
    print("=" * 70)
    print("EXPERIMENT 2: Non-Tournament Weight Sweep")
    print("=" * 70)

    game_csv_path = 'new_data_partitioned/game.csv'
    partitioned_dir = 'new_data_partitioned'
    expt_dir = 'model_experiments/tournament_weight_sweep'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    # Count partitions
    num_partitions = len([f for f in os.listdir(partitioned_dir) if f.endswith('.csv.gz')])
    print(f"Found {num_partitions} partitions")

    # Step 1: Load tournament train/test split (same as experiment 1)
    print("\n" + "=" * 60)
    print("Step 1: Loading tournament train/test split")
    print("=" * 60)

    train_ids, test_ids = get_tournament_train_test_split(game_csv_path, test_count=2000)
    print(f"Tournament train games: {len(train_ids):,}")
    print(f"Tournament test games: {len(test_ids):,}")

    # Step 2: Get non-tournament game IDs (excluding test games)
    print("\n" + "=" * 60)
    print("Step 2: Getting non-tournament game IDs")
    print("=" * 60)

    non_tourn_ids = get_non_tournament_game_ids(game_csv_path, exclude_game_ids=test_ids)
    print(f"Non-tournament games: {len(non_tourn_ids):,}")

    # Step 3: Materialize datasets
    print("\n" + "=" * 60)
    print("Step 3: Materializing datasets")
    print("=" * 60)

    # Tournament train data (90% drop rate)
    tourn_train_file = f'{expt_dir}/tournament_train_states.npy'
    if not os.path.exists(tourn_train_file):
        print("Materializing tournament training data (90% drop rate)...")
        tourn_train_X, tourn_train_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.9,
            allowed_game_ids=train_ids
        )
        np.save(tourn_train_file, tourn_train_X)
        np.save(f'{expt_dir}/tournament_train_labels.npy', tourn_train_y)
        print(f"  Tournament train samples: {len(tourn_train_y):,}")
    else:
        print("Loading existing tournament training data...")
        tourn_train_X = np.load(tourn_train_file)
        tourn_train_y = np.load(f'{expt_dir}/tournament_train_labels.npy')
        print(f"  Tournament train samples: {len(tourn_train_y):,}")

    # Non-tournament train data (90% drop rate)
    non_tourn_train_file = f'{expt_dir}/non_tournament_train_states.npy'
    if not os.path.exists(non_tourn_train_file):
        print("Materializing non-tournament training data (90% drop rate)...")
        non_tourn_train_X, non_tourn_train_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.9,
            allowed_game_ids=non_tourn_ids
        )
        np.save(non_tourn_train_file, non_tourn_train_X)
        np.save(f'{expt_dir}/non_tournament_train_labels.npy', non_tourn_train_y)
        print(f"  Non-tournament train samples: {len(non_tourn_train_y):,}")
    else:
        print("Loading existing non-tournament training data...")
        non_tourn_train_X = np.load(non_tourn_train_file)
        non_tourn_train_y = np.load(f'{expt_dir}/non_tournament_train_labels.npy')
        print(f"  Non-tournament train samples: {len(non_tourn_train_y):,}")

    # Test data (tournament only, 0% drop)
    test_file = f'{expt_dir}/test_states.npy'
    if not os.path.exists(test_file):
        print("Materializing test data (0% drop rate)...")
        test_X, test_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.0,
            allowed_game_ids=test_ids
        )
        np.save(test_file, test_X)
        np.save(f'{expt_dir}/test_labels.npy', test_y)
        print(f"  Test samples: {len(test_y):,}")
    else:
        print("Loading existing test data...")
        test_X = np.load(test_file)
        test_y = np.load(f'{expt_dir}/test_labels.npy')
        print(f"  Test samples: {len(test_y):,}")

    # Step 4: Weight sweep experiments
    print("\n" + "=" * 60)
    print("Step 4: Running weight sweep experiments")
    print("=" * 60)

    # Weights to test (tournament weight is always 1.0)
    # Non-tournament weights range from very small (due to having ~30x more data)
    # to equal weight
    non_tourn_weights = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]

    results = []

    for non_tourn_weight in non_tourn_weights:
        for use_is_tournament_feature in [False, True]:
            print(f"\n--- Weight={non_tourn_weight}, is_tournament_feature={use_is_tournament_feature} ---")

            # Combine datasets
            if non_tourn_weight == 0.0:
                # Tournament only
                combined_X = tourn_train_X
                combined_y = tourn_train_y
                weights = None
            else:
                combined_X = np.vstack([tourn_train_X, non_tourn_train_X])
                combined_y = np.concatenate([tourn_train_y, non_tourn_train_y])
                weights = np.concatenate([
                    np.ones(len(tourn_train_y)),  # Tournament weight = 1.0
                    np.full(len(non_tourn_train_y), non_tourn_weight)
                ])

            # Add is_tournament feature if requested
            if use_is_tournament_feature:
                if non_tourn_weight == 0.0:
                    is_tourn_col = np.ones((len(combined_X), 1))
                else:
                    is_tourn_col = np.concatenate([
                        np.ones((len(tourn_train_X), 1)),
                        np.zeros((len(non_tourn_train_X), 1))
                    ])
                train_X_final = np.hstack([combined_X, is_tourn_col])
                test_X_final = np.hstack([test_X, np.ones((len(test_X), 1))])
            else:
                train_X_final = combined_X
                test_X_final = test_X

            # Train
            start = time.time()
            model = train_lgb_model(train_X_final, combined_y, num_leaves=200, num_trees=200, weights=weights)
            train_time = time.time() - start

            # Evaluate
            metrics = evaluate_model(model, test_X_final, test_y,
                                    f"w={non_tourn_weight}, is_tourn={use_is_tournament_feature}")

            results.append({
                'non_tourn_weight': non_tourn_weight,
                'is_tournament_feature': use_is_tournament_feature,
                'train_samples': len(combined_y),
                'train_time': train_time,
                **metrics
            })

            print(f"  Train samples: {len(combined_y):,}, Time: {train_time:.1f}s")
            print(f"  Log Loss: {metrics['log_loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    # Step 5: Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY: Weight Sweep Results")
    print("=" * 70)
    print(f"\nTest set: {len(test_y):,} samples from 2000 most recent tournament games")
    print(f"Tournament train: {len(tourn_train_y):,} samples from {len(train_ids):,} games")
    print(f"Non-tournament train: {len(non_tourn_train_y):,} samples from {len(non_tourn_ids):,} games")
    print()
    print(f"{'Weight':<8} {'is_tourn':<10} {'Train Size':<12} {'Log Loss':<10} {'Accuracy':<10} {'Egg Inv':<10}")
    print("-" * 70)

    best_result = min(results, key=lambda x: x['log_loss'])

    for r in results:
        is_best = " *" if r == best_result else ""
        print(f"{r['non_tourn_weight']:<8} {str(r['is_tournament_feature']):<10} {r['train_samples']:<12,} "
              f"{r['log_loss']:<10.4f} {r['accuracy']:<10.4f} {r['inversions']:<10.4f}{is_best}")

    print()
    print(f"Best configuration: weight={best_result['non_tourn_weight']}, "
          f"is_tournament_feature={best_result['is_tournament_feature']}")
    print(f"  Log Loss: {best_result['log_loss']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f} ({100*best_result['accuracy']:.2f}%)")
    print(f"  Egg Inversions: {best_result['inversions']:.4f} ({100*best_result['inversions']:.2f}%)")

    # Save results
    results_path = f'{expt_dir}/weight_sweep_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def run_doubling_experiment():
    """Experiment 3: Repeated doubling of non-tournament game states.

    Tests adding increasing amounts of non-tournament data to tournament training:
    - Start with tournament-only baseline
    - Add N non-tournament states (N = tournament training size)
    - Double to 2N, 4N, 8N, etc. until we use all non-tournament data

    Uses memory-efficient sampling via memmap to avoid OOM.
    """
    print("=" * 70)
    print("EXPERIMENT 3: Repeated Doubling of Non-Tournament States")
    print("=" * 70)

    game_csv_path = 'new_data_partitioned/game.csv'
    expt_dir = 'model_experiments/tournament_weight_sweep'
    doubling_dir = 'model_experiments/tournament_doubling'
    pathlib.Path(doubling_dir).mkdir(exist_ok=True, parents=True)

    # Load tournament training data
    print("\nStep 1: Loading tournament training data...")
    tourn_train_X = np.load(f'{expt_dir}/tournament_train_states.npy')
    tourn_train_y = np.load(f'{expt_dir}/tournament_train_labels.npy')
    n_tourn = len(tourn_train_y)
    print(f"  Tournament train samples: {n_tourn:,}")

    # Load test data (tournament only)
    print("\nStep 2: Loading tournament test data...")
    test_X = np.load(f'{expt_dir}/test_states.npy')
    test_y = np.load(f'{expt_dir}/test_labels.npy')
    print(f"  Test samples: {len(test_y):,}")

    # Memory-map the non-tournament data to avoid loading it all at once
    print("\nStep 3: Memory-mapping non-tournament data...")
    non_tourn_states_path = f'{expt_dir}/non_tournament_train_states.npy'
    non_tourn_labels_path = f'{expt_dir}/non_tournament_train_labels.npy'

    non_tourn_X_mmap = np.load(non_tourn_states_path, mmap_mode='r')
    non_tourn_y = np.load(non_tourn_labels_path)  # Labels are small, load fully
    n_non_tourn = len(non_tourn_y)
    print(f"  Non-tournament train samples available: {n_non_tourn:,}")
    print(f"  Non-tournament data shape: {non_tourn_X_mmap.shape}")

    # Calculate doubling schedule
    # Start with 0 (baseline), then N, 2N, 4N, ... up to all non-tournament data
    doubling_schedule = [0]
    current = n_tourn
    while current < n_non_tourn:
        doubling_schedule.append(current)
        current *= 2
    doubling_schedule.append(n_non_tourn)  # Include all data as final point

    print(f"\nDoubling schedule: {[f'{x:,}' for x in doubling_schedule]}")

    # Run experiments
    print("\n" + "=" * 60)
    print("Step 4: Running experiments")
    print("=" * 60)

    results = []
    np.random.seed(42)  # For reproducibility

    for i, n_non_tourn_to_use in enumerate(doubling_schedule):
        print(f"\n--- Run {i+1}/{len(doubling_schedule)}: {n_non_tourn_to_use:,} non-tournament states ---")

        import gc
        gc.collect()  # Clean up memory from previous iteration

        if n_non_tourn_to_use == 0:
            # Tournament only baseline
            combined_X = tourn_train_X
            combined_y = tourn_train_y
            print(f"  Using tournament data only: {len(combined_y):,} samples")
        else:
            # Sample non-tournament states
            print(f"  Sampling {n_non_tourn_to_use:,} non-tournament states...")
            if n_non_tourn_to_use >= n_non_tourn:
                # Use all non-tournament data
                indices = np.arange(n_non_tourn)
            else:
                indices = np.random.choice(n_non_tourn, n_non_tourn_to_use, replace=False)
                indices.sort()  # Sequential access is faster for memmap

            # Load sampled data from memmap
            sampled_X = np.array(non_tourn_X_mmap[indices])  # Copy into memory
            sampled_y = non_tourn_y[indices]

            # Combine with tournament data
            combined_X = np.vstack([tourn_train_X, sampled_X])
            combined_y = np.concatenate([tourn_train_y, sampled_y])
            print(f"  Combined training data: {len(combined_y):,} samples")

            # Free sampled arrays
            del sampled_X, sampled_y
            gc.collect()

        # Train model
        start = time.time()
        model = train_lgb_model(combined_X, combined_y, num_leaves=200, num_trees=200)
        train_time = time.time() - start
        print(f"  Model trained in {train_time:.1f}s")

        # Evaluate
        metrics = evaluate_model(model, test_X, test_y,
                                f"n_non_tourn={n_non_tourn_to_use:,}")

        results.append({
            'n_non_tourn': n_non_tourn_to_use,
            'total_train_samples': len(combined_y),
            'train_time': train_time,
            **metrics
        })

        # Save model
        model_path = f'{doubling_dir}/model_non_tourn_{n_non_tourn_to_use}.mdl'
        model.save_model(model_path)

        # Clean up
        if n_non_tourn_to_use > 0:
            del combined_X, combined_y
        del model
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY: Doubling Results")
    print("=" * 70)
    print(f"\nTournament train: {n_tourn:,} samples")
    print(f"Non-tournament available: {n_non_tourn:,} samples")
    print(f"Test set: {len(test_y):,} samples (tournament games)")
    print()
    print(f"{'Non-tourn':<12} {'Total Train':<14} {'Log Loss':<10} {'Accuracy':<10} {'Egg Inv':<10} {'Time':<8}")
    print("-" * 70)

    best_result = min(results, key=lambda x: x['log_loss'])

    for r in results:
        is_best = " *" if r == best_result else ""
        print(f"{r['n_non_tourn']:<12,} {r['total_train_samples']:<14,} "
              f"{r['log_loss']:<10.4f} {r['accuracy']:<10.4f} {r['inversions']:<10.4f} "
              f"{r['train_time']:<8.1f}{is_best}")

    print()
    print(f"Best configuration: {best_result['n_non_tourn']:,} non-tournament states")
    print(f"  Log Loss: {best_result['log_loss']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f} ({100*best_result['accuracy']:.2f}%)")
    print(f"  Egg Inversions: {best_result['inversions']:.4f} ({100*best_result['inversions']:.2f}%)")

    # Save results
    results_path = f'{doubling_dir}/doubling_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def run_non_tourn_weight_sweep():
    """Experiment 4: Sweep over non-tournament sample weights.

    Uses all non-tournament data but varies the weight applied to those samples
    during training. Tournament samples always have weight 1.0.

    This tests whether downweighting non-tournament data can improve performance
    on tournament test data.
    """
    print("=" * 70)
    print("EXPERIMENT 4: Non-Tournament Weight Sweep")
    print("=" * 70)

    expt_dir = 'model_experiments/tournament_weight_sweep'
    weight_dir = 'model_experiments/non_tourn_weight_sweep'
    pathlib.Path(weight_dir).mkdir(exist_ok=True, parents=True)

    # Load tournament training data
    print("\nStep 1: Loading tournament training data...")
    tourn_train_X = np.load(f'{expt_dir}/tournament_train_states.npy')
    tourn_train_y = np.load(f'{expt_dir}/tournament_train_labels.npy')
    n_tourn = len(tourn_train_y)
    print(f"  Tournament train samples: {n_tourn:,}")

    # Load test data (tournament only)
    print("\nStep 2: Loading tournament test data...")
    test_X = np.load(f'{expt_dir}/test_states.npy')
    test_y = np.load(f'{expt_dir}/test_labels.npy')
    print(f"  Test samples: {len(test_y):,}")

    # Load all non-tournament data
    print("\nStep 3: Loading non-tournament data...")
    non_tourn_X = np.load(f'{expt_dir}/non_tournament_train_states.npy')
    non_tourn_y = np.load(f'{expt_dir}/non_tournament_train_labels.npy')
    n_non_tourn = len(non_tourn_y)
    print(f"  Non-tournament train samples: {n_non_tourn:,}")

    # Combine datasets once (reuse across weight experiments)
    print("\nStep 4: Combining datasets...")
    combined_X = np.vstack([tourn_train_X, non_tourn_X])
    combined_y = np.concatenate([tourn_train_y, non_tourn_y])
    print(f"  Combined training samples: {len(combined_y):,}")

    # Free individual arrays
    del tourn_train_X, non_tourn_X
    import gc
    gc.collect()

    # Weight sweep
    # Test weights from very small to equal weight
    weights_to_test = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]

    print("\n" + "=" * 60)
    print("Step 5: Running weight sweep")
    print("=" * 60)
    print(f"Weights to test: {weights_to_test}")

    results = []

    for non_tourn_weight in weights_to_test:
        print(f"\n--- Weight={non_tourn_weight} ---")

        # Create sample weights
        sample_weights = np.concatenate([
            np.ones(n_tourn),  # Tournament weight = 1.0
            np.full(n_non_tourn, non_tourn_weight)
        ])

        # Effective weight ratio
        tourn_total_weight = n_tourn * 1.0
        non_tourn_total_weight = n_non_tourn * non_tourn_weight
        effective_ratio = tourn_total_weight / (tourn_total_weight + non_tourn_total_weight)
        print(f"  Effective tournament weight fraction: {effective_ratio:.2%}")

        # Train model
        start = time.time()
        model = train_lgb_model(combined_X, combined_y, num_leaves=200, num_trees=200,
                               weights=sample_weights)
        train_time = time.time() - start
        print(f"  Model trained in {train_time:.1f}s")

        # Evaluate
        metrics = evaluate_model(model, test_X, test_y, f"weight={non_tourn_weight}")

        results.append({
            'non_tourn_weight': non_tourn_weight,
            'effective_tourn_fraction': effective_ratio,
            'train_time': train_time,
            **metrics
        })

        # Save model
        model_path = f'{weight_dir}/model_weight_{non_tourn_weight}.mdl'
        model.save_model(model_path)

        del model, sample_weights
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 SUMMARY: Weight Sweep Results")
    print("=" * 70)
    print(f"\nTournament train: {n_tourn:,} samples (weight=1.0)")
    print(f"Non-tournament train: {n_non_tourn:,} samples (weight varies)")
    print(f"Test set: {len(test_y):,} samples (tournament games)")
    print()
    print(f"{'Weight':<8} {'Tourn Frac':<12} {'Log Loss':<10} {'Accuracy':<10} {'Egg Inv':<10} {'Time':<8}")
    print("-" * 70)

    best_result = min(results, key=lambda x: x['log_loss'])

    for r in results:
        is_best = " *" if r == best_result else ""
        print(f"{r['non_tourn_weight']:<8} {r['effective_tourn_fraction']:<12.2%} "
              f"{r['log_loss']:<10.4f} {r['accuracy']:<10.4f} {r['inversions']:<10.4f} "
              f"{r['train_time']:<8.1f}{is_best}")

    print()
    print(f"Best configuration: non_tourn_weight={best_result['non_tourn_weight']}")
    print(f"  Effective tournament fraction: {best_result['effective_tourn_fraction']:.2%}")
    print(f"  Log Loss: {best_result['log_loss']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f} ({100*best_result['accuracy']:.2f}%)")
    print(f"  Egg Inversions: {best_result['inversions']:.4f} ({100*best_result['inversions']:.2f}%)")

    # Save results
    results_path = f'{weight_dir}/weight_sweep_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def run_is_tournament_feature_experiment():
    """Experiment 5: Test whether adding an is_tournament feature helps.

    Compares model performance with and without an additional binary feature
    indicating whether the training sample came from a tournament game.
    At test time, this feature is always 1 (tournament).
    """
    print("=" * 70)
    print("EXPERIMENT 5: Is-Tournament Feature Test")
    print("=" * 70)

    expt_dir = 'model_experiments/tournament_weight_sweep'
    feature_dir = 'model_experiments/is_tournament_feature'
    pathlib.Path(feature_dir).mkdir(exist_ok=True, parents=True)

    # Load tournament training data
    print("\nStep 1: Loading tournament training data...")
    tourn_train_X = np.load(f'{expt_dir}/tournament_train_states.npy')
    tourn_train_y = np.load(f'{expt_dir}/tournament_train_labels.npy')
    n_tourn = len(tourn_train_y)
    print(f"  Tournament train samples: {n_tourn:,}")

    # Load test data (tournament only)
    print("\nStep 2: Loading tournament test data...")
    test_X = np.load(f'{expt_dir}/test_states.npy')
    test_y = np.load(f'{expt_dir}/test_labels.npy')
    print(f"  Test samples: {len(test_y):,}")

    # Load all non-tournament data
    print("\nStep 3: Loading non-tournament data...")
    non_tourn_X = np.load(f'{expt_dir}/non_tournament_train_states.npy')
    non_tourn_y = np.load(f'{expt_dir}/non_tournament_train_labels.npy')
    n_non_tourn = len(non_tourn_y)
    print(f"  Non-tournament train samples: {n_non_tourn:,}")

    # Combine datasets
    print("\nStep 4: Combining datasets...")
    combined_X = np.vstack([tourn_train_X, non_tourn_X])
    combined_y = np.concatenate([tourn_train_y, non_tourn_y])
    print(f"  Combined training samples: {len(combined_y):,}")

    # Free individual arrays
    del tourn_train_X, non_tourn_X
    import gc
    gc.collect()

    results = []

    # Experiment 1: Without is_tournament feature (baseline)
    print("\n" + "=" * 60)
    print("Test 1: Without is_tournament feature (baseline)")
    print("=" * 60)

    start = time.time()
    model_baseline = train_lgb_model(combined_X, combined_y, num_leaves=200, num_trees=200)
    train_time = time.time() - start
    print(f"  Model trained in {train_time:.1f}s")

    metrics = evaluate_model(model_baseline, test_X, test_y, "Without is_tournament")
    results.append({
        'is_tournament_feature': False,
        'train_time': train_time,
        **metrics
    })
    model_baseline.save_model(f'{feature_dir}/model_no_feature.mdl')
    del model_baseline
    gc.collect()

    # Experiment 2: With is_tournament feature
    print("\n" + "=" * 60)
    print("Test 2: With is_tournament feature")
    print("=" * 60)

    # Add is_tournament column to training data
    is_tourn_train = np.concatenate([
        np.ones((n_tourn, 1)),      # Tournament = 1
        np.zeros((n_non_tourn, 1))  # Non-tournament = 0
    ])
    combined_X_with_feature = np.hstack([combined_X, is_tourn_train])
    print(f"  Training data shape with feature: {combined_X_with_feature.shape}")

    # Add is_tournament column to test data (always 1 for tournament test set)
    test_X_with_feature = np.hstack([test_X, np.ones((len(test_X), 1))])
    print(f"  Test data shape with feature: {test_X_with_feature.shape}")

    start = time.time()
    model_with_feature = train_lgb_model(combined_X_with_feature, combined_y,
                                         num_leaves=200, num_trees=200)
    train_time = time.time() - start
    print(f"  Model trained in {train_time:.1f}s")

    metrics = evaluate_model(model_with_feature, test_X_with_feature, test_y,
                            "With is_tournament")
    results.append({
        'is_tournament_feature': True,
        'train_time': train_time,
        **metrics
    })
    model_with_feature.save_model(f'{feature_dir}/model_with_feature.mdl')

    # Check feature importance of is_tournament
    importance = model_with_feature.feature_importance()
    is_tourn_importance = importance[-1]  # Last feature is is_tournament
    total_importance = importance.sum()
    print(f"\n  is_tournament feature importance: {is_tourn_importance} "
          f"({100*is_tourn_importance/total_importance:.2f}% of total)")

    del model_with_feature
    gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 SUMMARY: Is-Tournament Feature Results")
    print("=" * 70)
    print(f"\nTraining: {n_tourn:,} tournament + {n_non_tourn:,} non-tournament samples")
    print(f"Test set: {len(test_y):,} samples (tournament games)")
    print()
    print(f"{'Feature':<25} {'Log Loss':<10} {'Accuracy':<10} {'Egg Inv':<10} {'Time':<8}")
    print("-" * 70)

    for r in results:
        feature_str = "With is_tournament" if r['is_tournament_feature'] else "Without is_tournament"
        print(f"{feature_str:<25} {r['log_loss']:<10.4f} {r['accuracy']:<10.4f} "
              f"{r['inversions']:<10.4f} {r['train_time']:<8.1f}")

    # Comparison
    baseline = results[0]
    with_feature = results[1]
    print()
    print("Difference (with - without):")
    print(f"  Log Loss:    {with_feature['log_loss'] - baseline['log_loss']:+.4f}")
    print(f"  Accuracy:    {with_feature['accuracy'] - baseline['accuracy']:+.4f}")
    print(f"  Egg Inv:     {with_feature['inversions'] - baseline['inversions']:+.4f}")

    # Save results
    results_path = f'{feature_dir}/is_tournament_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def load_late_tournament_game_ids(game_ids_path: str = 'late_tournament_game_events_game_ids.txt') -> Set[int]:
    """Load late tournament game IDs from file."""
    game_ids = set()
    with open(game_ids_path) as f:
        for line in f:
            line = line.strip()
            if line:
                game_ids.add(int(line))
    return game_ids


def get_late_tournament_train_test_split(
    game_csv_path: str,
    late_game_ids: Set[int],
    test_fraction: float = 0.25
) -> Tuple[Set[int], Set[int]]:
    """Split late tournament games into train/test sets by time.

    Args:
        game_csv_path: Path to game.csv file
        late_game_ids: Set of late tournament game IDs
        test_fraction: Fraction of games to use for test set

    Returns:
        Tuple of (train_game_ids, test_game_ids)
    """
    # Load games with their start times
    games = []
    with open(game_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            game_id = int(row['id'])
            if game_id in late_game_ids:
                games.append((game_id, row['start_time']))

    # Sort by start_time descending (most recent first)
    games.sort(key=lambda x: x[1], reverse=True)

    # Take most recent as test set
    test_count = int(len(games) * test_fraction)
    test_game_ids = set(g[0] for g in games[:test_count])
    train_game_ids = set(g[0] for g in games[test_count:])

    return train_game_ids, test_game_ids


def run_late_tournament_experiment():
    """Experiment: Train and evaluate on late tournament games only.

    Uses last 5 matches of tournaments with >15 teams as high-quality data.
    This represents elite-level play in tournament elimination rounds.
    """
    print("=" * 70)
    print("EXPERIMENT: Late Tournament Training (Last 5 matches, >15 teams)")
    print("=" * 70)

    game_csv_path = 'new_data_partitioned/game.csv'
    late_events_path = 'late_tournament_game_events.csv.gz'
    expt_dir = 'model_experiments/late_tournament'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    # Step 1: Load late tournament game IDs
    print("\nStep 1: Loading late tournament game IDs...")
    late_game_ids = load_late_tournament_game_ids()
    print(f"  Late tournament games: {len(late_game_ids):,}")

    # Step 2: Create train/test split (most recent 25% for test)
    print("\nStep 2: Creating train/test split...")
    train_ids, test_ids = get_late_tournament_train_test_split(
        game_csv_path, late_game_ids, test_fraction=0.25
    )
    print(f"  Training games: {len(train_ids):,}")
    print(f"  Test games: {len(test_ids):,}")

    # Verify no overlap
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"Train/test overlap detected: {len(overlap)} games")

    # Step 3: Materialize training data from late tournament events
    print("\n" + "=" * 60)
    print("Step 3: Materializing late tournament training data")
    print("=" * 60)

    train_states_file = f'{expt_dir}/train_states.npy'
    if not os.path.exists(train_states_file):
        print("Materializing training data (50% drop rate)...")
        map_structure_infos = map_structure.MapStructureInfos()

        events = iterate_events_from_csv(late_events_path)
        game_states_iter = iterate_game_events_with_state(events, map_structure_infos)

        train_X, train_y = create_game_states_matrix(
            game_states_iter, drop_state_probability=0.5, noisy=False, allowed_game_ids=train_ids
        )
        np.save(f'{expt_dir}/train_states.npy', train_X)
        np.save(f'{expt_dir}/train_labels.npy', train_y)
        print(f"  Training samples: {len(train_y):,}")
    else:
        print("Loading existing training data...")
        train_X = np.load(f'{expt_dir}/train_states.npy')
        train_y = np.load(f'{expt_dir}/train_labels.npy')
        print(f"  Training samples: {len(train_y):,}")

    # Step 4: Materialize test data
    print("\n" + "=" * 60)
    print("Step 4: Materializing late tournament test data")
    print("=" * 60)

    test_states_file = f'{expt_dir}/test_states.npy'
    if not os.path.exists(test_states_file):
        print("Materializing test data (0% drop rate)...")
        map_structure_infos = map_structure.MapStructureInfos()

        events = iterate_events_from_csv(late_events_path)
        game_states_iter = iterate_game_events_with_state(events, map_structure_infos)

        test_X, test_y = create_game_states_matrix(
            game_states_iter, drop_state_probability=0.0, noisy=False, allowed_game_ids=test_ids
        )
        np.save(f'{expt_dir}/test_states.npy', test_X)
        np.save(f'{expt_dir}/test_labels.npy', test_y)
        print(f"  Test samples: {len(test_y):,}")
    else:
        print("Loading existing test data...")
        test_X = np.load(f'{expt_dir}/test_states.npy')
        test_y = np.load(f'{expt_dir}/test_labels.npy')
        print(f"  Test samples: {len(test_y):,}")

    # Step 5: Train model on late tournament data only
    print("\n" + "=" * 60)
    print("Step 5: Training model on late tournament data")
    print("=" * 60)

    print(f"Training data shape: {train_X.shape}")
    print(f"Test data shape: {test_X.shape}")

    start = time.time()
    model = train_lgb_model(train_X, train_y, num_leaves=200, num_trees=200)
    print(f"Model trained in {time.time() - start:.1f}s")

    model_path = f'{expt_dir}/model.mdl'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Step 6: Evaluate
    print("\n" + "=" * 60)
    print("Step 6: Evaluating on late tournament test set")
    print("=" * 60)

    metrics = evaluate_model(model, test_X, test_y, "Late Tournament Model")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY: Late Tournament Baseline")
    print("=" * 60)
    print(f"Training set: {len(train_ids):,} games, {len(train_y):,} states")
    print(f"Test set: {len(test_ids):,} games, {len(test_y):,} states")
    print()
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'Log Loss':<20} {metrics['log_loss']:<15.4f}")
    print(f"{'Accuracy':<20} {metrics['accuracy']:<15.4f} ({100*metrics['accuracy']:.1f}%)")
    print(f"{'Egg Inversions':<20} {metrics['inversions']:<15.4f} ({100*metrics['inversions']:.2f}%)")

    return metrics


def run_late_tournament_weight_sweep():
    """Experiment: Weight sweep using late tournament data as test set.

    Tests different weights for non-tournament games when training, but
    evaluates on late tournament test set (highest quality data).
    """
    print("=" * 70)
    print("EXPERIMENT: Weight Sweep with Late Tournament Test Set")
    print("=" * 70)

    game_csv_path = 'new_data_partitioned/game.csv'
    partitioned_dir = 'new_data_partitioned'
    late_events_path = 'late_tournament_game_events.csv.gz'
    expt_dir = 'model_experiments/late_tournament_weight_sweep'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    # Count partitions
    num_partitions = len([f for f in os.listdir(partitioned_dir) if f.endswith('.csv.gz')])
    print(f"Found {num_partitions} partitions")

    # Step 1: Load late tournament game IDs for test set
    print("\nStep 1: Loading late tournament game IDs for test...")
    late_game_ids = load_late_tournament_game_ids()
    _, test_ids = get_late_tournament_train_test_split(
        game_csv_path, late_game_ids, test_fraction=0.25
    )
    print(f"  Late tournament test games: {len(test_ids):,}")

    # Step 2: Load tournament train IDs (excluding test games)
    print("\nStep 2: Loading tournament train IDs...")
    all_tournament_ids = load_tournament_game_ids(game_csv_path)
    tourn_train_ids = all_tournament_ids - test_ids - late_game_ids  # Exclude all late tournament games
    print(f"  Tournament train games (excluding late): {len(tourn_train_ids):,}")

    # Step 3: Load non-tournament game IDs
    print("\nStep 3: Loading non-tournament game IDs...")
    non_tourn_ids = get_non_tournament_game_ids(game_csv_path, exclude_game_ids=test_ids)
    print(f"  Non-tournament games: {len(non_tourn_ids):,}")

    # Step 4: Materialize datasets
    print("\n" + "=" * 60)
    print("Step 4: Materializing datasets")
    print("=" * 60)

    # Tournament train data
    tourn_train_file = f'{expt_dir}/tournament_train_states.npy'
    if not os.path.exists(tourn_train_file):
        print("Materializing tournament training data (90% drop rate)...")
        tourn_train_X, tourn_train_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.9,
            allowed_game_ids=tourn_train_ids
        )
        np.save(tourn_train_file, tourn_train_X)
        np.save(f'{expt_dir}/tournament_train_labels.npy', tourn_train_y)
        print(f"  Tournament train samples: {len(tourn_train_y):,}")
    else:
        print("Loading existing tournament training data...")
        tourn_train_X = np.load(tourn_train_file)
        tourn_train_y = np.load(f'{expt_dir}/tournament_train_labels.npy')
        print(f"  Tournament train samples: {len(tourn_train_y):,}")

    # Non-tournament train data
    non_tourn_train_file = f'{expt_dir}/non_tournament_train_states.npy'
    if not os.path.exists(non_tourn_train_file):
        print("Materializing non-tournament training data (90% drop rate)...")
        non_tourn_train_X, non_tourn_train_y = materialize_partition_range(
            partitioned_dir, expt_dir, 0, num_partitions, drop_prob=0.9,
            allowed_game_ids=non_tourn_ids
        )
        np.save(non_tourn_train_file, non_tourn_train_X)
        np.save(f'{expt_dir}/non_tournament_train_labels.npy', non_tourn_train_y)
        print(f"  Non-tournament train samples: {len(non_tourn_train_y):,}")
    else:
        print("Loading existing non-tournament training data...")
        non_tourn_train_X = np.load(non_tourn_train_file)
        non_tourn_train_y = np.load(f'{expt_dir}/non_tournament_train_labels.npy')
        print(f"  Non-tournament train samples: {len(non_tourn_train_y):,}")

    # Late tournament test data
    test_file = f'{expt_dir}/test_states.npy'
    if not os.path.exists(test_file):
        print("Materializing late tournament test data (0% drop rate)...")
        map_structure_infos = map_structure.MapStructureInfos()

        events = iterate_events_from_csv(late_events_path)
        game_states_iter = iterate_game_events_with_state(events, map_structure_infos)

        test_X, test_y = create_game_states_matrix(
            game_states_iter, drop_state_probability=0.0, noisy=False, allowed_game_ids=test_ids
        )
        np.save(test_file, test_X)
        np.save(f'{expt_dir}/test_labels.npy', test_y)
        print(f"  Test samples: {len(test_y):,}")
    else:
        print("Loading existing test data...")
        test_X = np.load(test_file)
        test_y = np.load(f'{expt_dir}/test_labels.npy')
        print(f"  Test samples: {len(test_y):,}")

    # Step 5: Weight sweep
    print("\n" + "=" * 60)
    print("Step 5: Running weight sweep experiments")
    print("=" * 60)

    non_tourn_weights = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    results = []

    for non_tourn_weight in non_tourn_weights:
        print(f"\n--- Weight={non_tourn_weight} ---")

        if non_tourn_weight == 0.0:
            combined_X = tourn_train_X
            combined_y = tourn_train_y
            weights = None
        else:
            combined_X = np.vstack([tourn_train_X, non_tourn_train_X])
            combined_y = np.concatenate([tourn_train_y, non_tourn_train_y])
            weights = np.concatenate([
                np.ones(len(tourn_train_y)),
                np.full(len(non_tourn_train_y), non_tourn_weight)
            ])

        start = time.time()
        model = train_lgb_model(combined_X, combined_y, num_leaves=200, num_trees=200, weights=weights)
        train_time = time.time() - start

        metrics = evaluate_model(model, test_X, test_y, f"w={non_tourn_weight}")

        results.append({
            'non_tourn_weight': non_tourn_weight,
            'train_samples': len(combined_y),
            'train_time': train_time,
            **metrics
        })

        print(f"  Train samples: {len(combined_y):,}, Time: {train_time:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY: Weight Sweep with Late Tournament Test")
    print("=" * 70)
    print(f"\nTest set: {len(test_y):,} samples from late tournament games")
    print(f"Tournament train: {len(tourn_train_y):,} samples")
    print(f"Non-tournament train: {len(non_tourn_train_y):,} samples")
    print()
    print(f"{'Weight':<8} {'Train Size':<12} {'Log Loss':<10} {'Accuracy':<10} {'Egg Inv':<10}")
    print("-" * 60)

    best_result = min(results, key=lambda x: x['log_loss'])

    for r in results:
        is_best = " *" if r == best_result else ""
        print(f"{r['non_tourn_weight']:<8} {r['train_samples']:<12,} "
              f"{r['log_loss']:<10.4f} {r['accuracy']:<10.4f} {r['inversions']:<10.4f}{is_best}")

    print()
    print(f"Best: weight={best_result['non_tourn_weight']}, log_loss={best_result['log_loss']:.4f}")

    # Save results
    results_path = f'{expt_dir}/weight_sweep_results.csv'
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {results_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow-and-verify', action='store_true',
                        help='Run both slow and fast paths, assert identical output')
    args = parser.parse_args()

    new_export_csv = 'export_20260115_210621/gameevent.csv'
    new_partitioned_dir = 'new_data_partitioned'
    new_expt_name = 'new_data_model'

    # DEBUG: Limit number of games for testing (set to None for full run)
    debug_max_games = None  # Set to a number like 5000 for faster testing

    # Step 1: Check if already partitioned, otherwise partition
    if not os.path.exists(new_partitioned_dir) or len(os.listdir(new_partitioned_dir)) == 0:
        print("=" * 60)
        print("Step 1: Validating and partitioning new data")
        print("=" * 60)
        num_partitions, num_games = validate_and_partition_data(
            new_export_csv,
            new_partitioned_dir,
            games_per_partition=1000,
            max_games=debug_max_games
        )
        print(f"\nCreated {num_partitions} partitions with {num_games} valid games")
    else:
        # Count existing partitions
        num_partitions = len([f for f in os.listdir(new_partitioned_dir) if f.endswith('.csv.gz')])
        print(f"Found existing partitioned data: {num_partitions} files")

    # Step 2: Create train/test split (80/20)
    print("\n" + "=" * 60)
    print("Step 2: Creating train/test split")
    print("=" * 60)

    train_end = int(num_partitions * 0.8)
    test_start = train_end
    print(f"Training partitions: 0-{train_end-1} ({train_end} files)")
    print(f"Test partitions: {test_start}-{num_partitions-1} ({num_partitions - test_start} files)")

    expt_dir = f'model_experiments/{new_expt_name}'
    pathlib.Path(expt_dir).mkdir(exist_ok=True, parents=True)

    # Materialize training data
    train_states_file = f'{expt_dir}/train_states.npy'
    if not os.path.exists(train_states_file):
        print("\nMaterializing training data (90% drop rate)...")
        if args.slow_and_verify:
            fast_X, fast_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, 0, train_end, drop_prob=0.9, use_fast_path=True
            )
            slow_X, slow_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, 0, train_end, drop_prob=0.9, use_fast_path=False
            )
            assert np.array_equal(fast_y, slow_y), "Training labels mismatch"
            assert np.allclose(fast_X, slow_X, atol=1e-5), f"Training states mismatch: {np.max(np.abs(fast_X - slow_X))}"
            print("  Verification passed: fast and slow paths match!")
            train_X, train_y = fast_X, fast_y
        else:
            train_X, train_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, 0, train_end, drop_prob=0.9
            )
        np.save(f'{expt_dir}/train_states.npy', train_X)
        np.save(f'{expt_dir}/train_labels.npy', train_y)
        print(f"  Training samples: {len(train_y):,}")
    else:
        print("Loading existing training data...")
        train_X = np.load(f'{expt_dir}/train_states.npy')
        train_y = np.load(f'{expt_dir}/train_labels.npy')
        print(f"  Training samples: {len(train_y):,}")

    # Materialize test data
    test_states_file = f'{expt_dir}/test_states.npy'
    if not os.path.exists(test_states_file):
        print("\nMaterializing test data (95% drop rate)...")
        if args.slow_and_verify:
            fast_X, fast_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, test_start, num_partitions, drop_prob=0.95, use_fast_path=True
            )
            slow_X, slow_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, test_start, num_partitions, drop_prob=0.95, use_fast_path=False
            )
            assert np.array_equal(fast_y, slow_y), "Test labels mismatch"
            assert np.allclose(fast_X, slow_X, atol=1e-5), f"Test states mismatch: {np.max(np.abs(fast_X - slow_X))}"
            print("  Verification passed: fast and slow paths match!")
            test_X, test_y = fast_X, fast_y
        else:
            test_X, test_y = materialize_partition_range(
                new_partitioned_dir, expt_dir, test_start, num_partitions, drop_prob=0.95
            )
        np.save(f'{expt_dir}/test_states.npy', test_X)
        np.save(f'{expt_dir}/test_labels.npy', test_y)
        print(f"  Test samples: {len(test_y):,}")
    else:
        print("Loading existing test data...")
        test_X = np.load(f'{expt_dir}/test_states.npy')
        test_y = np.load(f'{expt_dir}/test_labels.npy')
        print(f"  Test samples: {len(test_y):,}")

    # Step 3: Train new model
    print("\n" + "=" * 60)
    print("Step 3: Training new model on new data")
    print("=" * 60)

    print(f"Training data shape: {train_X.shape}")
    print(f"Test data shape: {test_X.shape}")

    start = time.time()
    new_model = train_lgb_model(train_X, train_y, num_leaves=200, num_trees=200)
    print(f"Model trained in {time.time() - start:.1f}s")

    new_model_path = f'{expt_dir}/model.mdl'
    new_model.save_model(new_model_path)
    print(f"New model saved to {new_model_path}")

    # Step 4: Load repro model and compare
    print("\n" + "=" * 60)
    print("Step 4: Comparing models on new test set")
    print("=" * 60)

    repro_model_path = 'model_experiments/new_data_model/model_100l_100t.mdl'
    if os.path.exists(repro_model_path):
        repro_model = lgb.Booster(model_file=repro_model_path)
        repro_metrics = evaluate_model(repro_model, test_X, test_y, "Current best Model")
    else:
        print(f"Warning: Repro model not found at {repro_model_path}")
        repro_metrics = None

    new_metrics = evaluate_model(new_model, test_X, test_y, "New Model (trained on new data)")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Test set: {len(test_y):,} samples from new data export")
    print()
    print(f"{'Metric':<20} {'CurChamp Model':<15} {'New Model':<15} {'Difference':<15}")
    print("-" * 65)

    if repro_metrics:
        for metric in ['log_loss', 'accuracy', 'inversions']:
            repro_val = repro_metrics[metric]
            new_val = new_metrics[metric]
            diff = new_val - repro_val
            sign = '+' if diff > 0 else ''
            print(f"{metric:<20} {repro_val:<15.4f} {new_val:<15.4f} {sign}{diff:<15.4f}")
    else:
        for metric in ['log_loss', 'accuracy', 'inversions']:
            print(f"{metric:<20} {'N/A':<15} {new_metrics[metric]:<15.4f}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'tournament':
            run_tournament_only_experiment()
        elif sys.argv[1] == 'weight_sweep':
            run_weight_sweep_experiment()
        elif sys.argv[1] == 'doubling':
            run_doubling_experiment()
        elif sys.argv[1] == 'non_tourn_weight':
            run_non_tourn_weight_sweep()
        elif sys.argv[1] == 'is_tourn_feature':
            run_is_tournament_feature_experiment()
        elif sys.argv[1] == 'late_tournament':
            run_late_tournament_experiment()
        elif sys.argv[1] == 'late_tournament_weight':
            run_late_tournament_weight_sweep()
        elif sys.argv[1] == 'tiered_doubling':
            run_tiered_doubling_experiment()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python3 train_model.py [tournament|weight_sweep|doubling|non_tourn_weight|is_tourn_feature|late_tournament|late_tournament_weight|tiered_doubling]")
    else:
        main()
