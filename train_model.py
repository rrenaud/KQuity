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
from typing import Optional, Set, Tuple
import numpy as np
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


def train_lgb_model(train_X, train_y, num_leaves=100, num_trees=100):
    """Train a LightGBM model."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': 0
    }
    train_data = lgb.Dataset(train_X, train_y)
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
    game_csv_path: str = 'export_20260115_210621/game.csv',
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
    if len(sys.argv) > 1 and sys.argv[1] == 'tournament':
        run_tournament_only_experiment()
    else:
        main()
