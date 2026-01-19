#!/usr/bin/env python3
"""Run scaling experiment: measure performance vs training data size."""

import glob
import json
import os
import sys
import traceback
from datetime import datetime
import numpy as np
import random
import concurrent.futures
import lightgbm as lgb
import sklearn.metrics
from preprocess import process_single_file

# Fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def materialize_shards_parallel(files, drop_prob, max_workers=8):
    """Materialize game states from files, in parallel."""
    if not files:
        print("  Warning: No files provided")
        return None, None

    all_states, all_labels = [], []
    args_list = [(f, drop_prob) for f in files]
    errors = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, args): args[0] for args in args_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            filepath = futures[future]
            try:
                _, states, labels = future.result(timeout=300)  # 5 min timeout per file
                all_states.append(states)
                all_labels.append(labels)
            except Exception as e:
                errors.append((filepath, str(e)))
                print(f"  ERROR processing {filepath}: {e}")
            completed += 1
            if completed % 10 == 0 or completed == len(files):
                print(f"  Processed {completed}/{len(files)} files...")

    if errors:
        print(f"  WARNING: {len(errors)} files failed to process")

    if not all_states:
        raise RuntimeError("No files processed successfully")

    print(f"  Completed {len(all_states)} files, {sum(len(l) for l in all_labels):,} samples")
    return np.vstack(all_states), np.concatenate(all_labels)


def extract_shard_num(filepath):
    """Extract shard number from filepath like 'new_data_partitioned/gameevents_123.csv.gz'."""
    basename = os.path.basename(filepath)  # gameevents_123.csv.gz
    return int(basename.split('_')[1].split('.')[0])  # 123


def get_shards_in_range(start_shard, end_shard):
    """Get shard files in range [start_shard, end_shard] inclusive."""
    all_files = sorted(glob.glob('new_data_partitioned/gameevents_*.csv.gz'))
    files = [f for f in all_files if start_shard <= extract_shard_num(f) <= end_shard]
    return sorted(files)


def run_scaling_experiment():
    # Set up logging to file
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = f'logs/scaling_experiment_{timestamp}.log'
    tee = Tee(log_path)
    sys.stdout = tee

    jsonl_path = f'logs/scaling_experiment_{timestamp}.jsonl'
    jsonl_file = open(jsonl_path, 'w')

    print("=" * 60)
    print("SCALING EXPERIMENT (debug mode)")
    print(f"Random seed: {SEED}")
    print(f"Log file: {log_path}")
    print(f"Results file: {jsonl_path}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Shard ranges:
    # Training: 000-829 (830 shards)
    # Test: 830-924 (95 shards)
    MAX_TRAIN_SHARD = 829
    MIN_TEST_SHARD = 830
    MAX_TEST_SHARD = 924
    MAX_TEST_SHARDS = 20  # Freeze test size at this

    results = []
    num_test_shards = 1
    num_train_shards = 1

    try:
        while True:
            # Calculate shard ranges
            # Test: take from end of test range
            test_start = MAX_TEST_SHARD - num_test_shards + 1
            test_end = MAX_TEST_SHARD

            # Train: take from end of train range
            train_start = MAX_TRAIN_SHARD - num_train_shards + 1
            train_end = MAX_TRAIN_SHARD

            print(f"\n{'=' * 60}")
            print(f"ITERATION: {num_train_shards} train shards, {num_test_shards} test shards")
            print(f"  Train range: {train_start:03d}-{train_end:03d}")
            print(f"  Test range: {test_start:03d}-{test_end:03d}")
            print("=" * 60)

            # Load test data
            print(f"\nMaterializing test data ({num_test_shards} shards, 90% drop)...")
            test_files = get_shards_in_range(test_start, test_end)
            print(f"  Found {len(test_files)} test files")
            test_X, test_y = materialize_shards_parallel(test_files, drop_prob=0.9)
            print(f"  Test samples: {len(test_y):,}")

            # Load training data
            print(f"\nMaterializing training data ({num_train_shards} shards, 90% drop)...")
            train_files = get_shards_in_range(train_start, train_end)
            print(f"  Found {len(train_files)} train files")
            train_X, train_y = materialize_shards_parallel(train_files, drop_prob=0.9)
            print(f"  Training samples: {len(train_y):,}, features: {train_X.shape[1]}")

            # Train model
            print("\nTraining LightGBM model...")
            param = {
                'num_leaves': 100,
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting': 'gbdt',
                'verbose': -1,
                'seed': SEED,
            }
            train_data = lgb.Dataset(train_X, train_y)
            model = lgb.train(param, train_data, num_boost_round=100)

            # Evaluate
            print("Evaluating...")
            predictions = model.predict(test_X)
            log_loss = sklearn.metrics.log_loss(test_y, predictions)
            accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

            result = {
                'train_shards': num_train_shards,
                'test_shards': num_test_shards,
                'train_samples': len(train_y),
                'test_samples': len(test_y),
                'log_loss': log_loss,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat(),
            }
            results.append(result)
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

            print(f"\n  Log Loss: {log_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")

            # Free memory
            del train_X, train_y, test_X, test_y, train_data, model, predictions
            import gc
            gc.collect()

            # Determine next iteration
            if num_test_shards < MAX_TEST_SHARDS:
                # Phase 1: Double both test and train
                num_test_shards = min(num_test_shards * 2, MAX_TEST_SHARDS)
                num_train_shards *= 2
            else:
                # Phase 2: Test frozen at MAX_TEST_SHARDS, double train only
                num_train_shards *= 2

            # Check if we've used all training shards
            if num_train_shards > MAX_TRAIN_SHARD + 1:
                print("\nReached maximum training shards.")
                break

    except Exception as e:
        print(f"\n\nERROR: {e}")
        print(traceback.format_exc())
    finally:
        # Final summary
        print(f"\n{'=' * 60}")
        print("SUMMARY: Scaling Results")
        print("=" * 60)
        print(f"{'Train':>8} {'Test':>6} {'Train Samples':>14} {'Log Loss':>10} {'Accuracy':>10}")
        print("-" * 52)
        for r in results:
            print(f"{r['train_shards']:>8} {r['test_shards']:>6} {r['train_samples']:>14,} {r['log_loss']:>10.4f} {r['accuracy']:>10.4f}")

        print(f"\nFinished: {datetime.now().isoformat()}")
        print(f"Results saved to: {jsonl_path}")

        jsonl_file.close()
        sys.stdout = tee.stdout
        tee.close()

    return results


if __name__ == '__main__':
    run_scaling_experiment()
