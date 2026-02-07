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
from symmetry import swap_teams


# Monotone constraints for LightGBM (52 features).
# +1 = increasing feature should increase Blue win prob
# -1 = increasing feature should decrease Blue win prob
#  0 = no constraint
#
# Feature layout (from vectorize_game_state):
#   Per team (20 features):
#     [0] eggs, [1] food_deposited, [2] num_vanilla_warriors, [3] num_speed_warriors
#     4 workers × [is_bot, has_food, has_speed, has_wings]
#   Blue team: indices 0-19 (good things = +1)
#   Gold team: indices 20-39 (good things = -1, bad for Blue)
#   40-44: maiden_control (x5, +1 = Blue holding)
#   45-48: map_one_hot (x4)
#   49: snail_position (+1, positive = Blue advancing)
#   50: snail_velocity (0)
#   51: berries_available (0)
_BLUE_WORKER = [0, 1, 1, 1]   # is_bot(0), has_food(+1), has_speed(+1), has_wings(+1)
_GOLD_WORKER = [0, -1, -1, -1]  # mirror for gold
MONOTONE_CONSTRAINTS = (
    [1, 1, 1, 1] + _BLUE_WORKER * 4 +   # blue: eggs, food, vanilla, speed_warriors, 4 workers
    [-1, -1, -1, -1] + _GOLD_WORKER * 4 +  # gold: mirror
    [1] * 5 +               # 5 maidens
    [0] * 4 +               # 4 map one-hot
    [1, 0] +                # snail_position(+1), snail_velocity(0)
    [0]                     # berries_available
)


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
    use_fast_path: bool = True
):
    """Materialize game states from a range of partitions."""
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
                states, labels = fast_materialize(csv_path, drop_prob)
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
                states, labels = create_game_states_matrix(game_states_iter, drop_prob, noisy=False)
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


def train_lgb_model(train_X, train_y, num_leaves=100, num_trees=100,
                    monotone_constraints=None):
    """Train a LightGBM model."""
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': 0
    }
    if monotone_constraints is not None:
        param['monotone_constraints'] = monotone_constraints
    train_data = lgb.Dataset(train_X, train_y)
    return lgb.train(param, train_data, num_boost_round=num_trees)


def _check_monotone_inversions(model, test_X, feature_idx, direction, sample_size=2000):
    """Check inversion rate for a single feature.

    direction: +1 means increasing feature should increase prediction,
               -1 means increasing feature should decrease prediction.
    Returns fraction of samples where the monotone relationship is violated.
    """
    eligible_X = test_X
    # For features with natural bounds (eggs at 3 max), filter out maxed-out rows
    if direction == 1:
        eligible_X = test_X[test_X[:, feature_idx] < test_X[:, feature_idx].max()]
    elif direction == -1:
        eligible_X = test_X[test_X[:, feature_idx] < test_X[:, feature_idx].max()]

    if len(eligible_X) == 0:
        return 0.0

    n = min(sample_size, len(eligible_X))
    indices = np.random.choice(len(eligible_X), n, replace=False)
    sample_X = eligible_X[indices]
    orig_preds = model.predict(sample_X)

    modified_X = sample_X.copy()
    if direction == 1:
        modified_X[:, feature_idx] += 0.1
        return (model.predict(modified_X) < orig_preds).mean()
    else:  # direction == -1
        modified_X[:, feature_idx] += 0.1
        return (model.predict(modified_X) > orig_preds).mean()


# Features to check for monotone inversions: (index, name, direction)
MONOTONE_CHECK_FEATURES = [
    (0, 'blue_eggs', 1),
    (1, 'blue_food', 1),
    (2, 'blue_vanilla', 1),
    (3, 'blue_spd_war', 1),
    (5, 'blue_w0_food', 1),
    (6, 'blue_w0_spd', 1),
    (20, 'gold_eggs', -1),
    (21, 'gold_food', -1),
    (22, 'gold_vanilla', -1),
    (23, 'gold_spd_war', -1),
    (25, 'gold_w0_food', -1),
    (26, 'gold_w0_spd', -1),
    (40, 'maiden_0', 1),
    (41, 'maiden_1', 1),
    (42, 'maiden_2', 1),
    (43, 'maiden_3', 1),
    (44, 'maiden_4', 1),
    (49, 'snail_pos', 1),
]


def evaluate_model(model, test_X, test_y, name: str, train_X=None, train_y=None):
    """Evaluate a model and return metrics."""
    predictions = model.predict(test_X)

    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    train_log_loss = None
    if train_X is not None and train_y is not None:
        train_preds = model.predict(train_X)
        train_log_loss = sklearn.metrics.log_loss(train_y, train_preds)

    # Per-feature monotone inversion checks
    inversion_rates = {}
    for feat_idx, feat_name, direction in MONOTONE_CHECK_FEATURES:
        rate = _check_monotone_inversions(model, test_X, feat_idx, direction)
        inversion_rates[feat_name] = rate

    print(f"\n{name} Results:")
    if train_log_loss is not None:
        print(f"  Train Log Loss: {train_log_loss:.4f}")
    print(f"  Test Log Loss: {log_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")
    print(f"  Monotone inversions:")
    for feat_name, rate in inversion_rates.items():
        print(f"    {feat_name:>12}: {rate:.4f} ({100*rate:.2f}%)")

    return {
        'log_loss': log_loss,
        'train_log_loss': train_log_loss,
        'accuracy': accuracy,
        'inversion_rates': inversion_rates,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow-and-verify', action='store_true',
                        help='Run both slow and fast paths, assert identical output')
    parser.add_argument('--symmetry-augment', action='store_true',
                        help='Double training data via Blue/Gold symmetry swap')
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

    # Symmetry augmentation (applied after loading, before training)
    if args.symmetry_augment:
        print("\nApplying symmetry augmentation...")
        swap_X, swap_y = swap_teams(train_X, train_y)
        train_X = np.vstack([train_X, swap_X])
        train_y = np.concatenate([train_y, swap_y])
        print(f"  Augmented training samples: {len(train_y):,} (2x original)")

    # Step 3: Hyperparameter sweep
    print("\n" + "=" * 60)
    print("Step 3: Hyperparameter sweep")
    print("=" * 60)

    print(f"Training data shape: {train_X.shape}")
    print(f"Test data shape: {test_X.shape}")

    assert len(MONOTONE_CONSTRAINTS) == train_X.shape[1], \
        f"Constraint vector length {len(MONOTONE_CONSTRAINTS)} != feature count {train_X.shape[1]}"

    leaves_grid = [200, 400, 800]
    trees_grid = [200, 400]
    sweep_results = []

    for num_leaves in leaves_grid:
        for num_trees in trees_grid:
            for mode in ['baseline', 'monotone']:
                constraints = MONOTONE_CONSTRAINTS if mode == 'monotone' else None
                label = f"{mode} L={num_leaves} T={num_trees}"

                start = time.time()
                model = train_lgb_model(
                    train_X, train_y,
                    num_leaves=num_leaves,
                    num_trees=num_trees,
                    monotone_constraints=constraints,
                )
                elapsed = time.time() - start
                print(f"Trained {label} in {elapsed:.1f}s")

                metrics = evaluate_model(
                    model, test_X, test_y, label,
                    train_X=train_X, train_y=train_y,
                )
                avg_inv = np.mean(list(metrics['inversion_rates'].values()))
                sweep_results.append({
                    'label': label,
                    'train_ll': metrics['train_log_loss'],
                    'test_ll': metrics['log_loss'],
                    'test_acc': metrics['accuracy'],
                    'avg_inv': avg_inv,
                })

    # Summary table
    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'Config':<28} | {'Train LL':>8} | {'Test LL':>8} | {'Test Acc':>8} | {'Avg Inv%':>8}")
    print("-" * 80)
    for r in sweep_results:
        print(f"{r['label']:<28} | {r['train_ll']:>8.4f} | {r['test_ll']:>8.4f} | {100*r['test_acc']:>7.1f}% | {100*r['avg_inv']:>7.2f}%")


if __name__ == '__main__':
    main()
