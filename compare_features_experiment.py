#!/usr/bin/env python3
"""Compare model performance with new features vs baseline."""

import numpy as np
import lightgbm as lgb
import sklearn.metrics
from preprocess import iterate_events_from_csv, iterate_game_events_with_state, create_game_states_matrix
import map_structure

def materialize_shards(start, end, drop_prob=0.0):
    """Materialize game states from shard range [start, end)."""
    map_infos = map_structure.MapStructureInfos()
    all_states, all_labels = [], []

    for shard in range(start, end):
        csv_path = f'new_data_partitioned/gameevents_{shard:03d}.csv.gz'
        events = iterate_events_from_csv(csv_path)
        game_states = iterate_game_events_with_state(events, map_infos)
        states, labels = create_game_states_matrix(game_states, drop_prob)
        all_states.append(states)
        all_labels.append(labels)
        print(f"  Shard {shard}: {len(labels)} samples")

    return np.vstack(all_states), np.concatenate(all_labels)

def train_and_evaluate():
    print("Materializing training data (shards 900-910)...")
    train_X, train_y = materialize_shards(900, 911, drop_prob=0.0)
    print(f"Training samples: {len(train_y):,}, features: {train_X.shape[1]}")

    print("\nMaterializing test data (shard 911)...")
    test_X, test_y = materialize_shards(911, 912, drop_prob=0.0)
    print(f"Test samples: {len(test_y):,}")

    print("\nTraining LightGBM model...")
    param = {'num_leaves': 100, 'objective': 'binary', 'metric': 'binary_logloss', 'boosting': 'gbdt', 'verbose': -1}
    train_data = lgb.Dataset(train_X, train_y)
    model = lgb.train(param, train_data, num_boost_round=100)

    print("\nEvaluating...")
    predictions = model.predict(test_X)
    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    print(f"\n{'='*40}")
    print(f"RESULTS (features: {train_X.shape[1]})")
    print(f"{'='*40}")
    print(f"Log Loss:  {log_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f} ({100*accuracy:.1f}%)")
    return log_loss, accuracy

if __name__ == '__main__':
    train_and_evaluate()
