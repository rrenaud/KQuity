#!/usr/bin/env python3
"""Ablation study to identify which new features help model performance."""

import numpy as np
import lightgbm as lgb
import sklearn.metrics
from preprocess import iterate_events_from_csv, iterate_game_events_with_state, create_game_states_matrix
import map_structure

# Feature layout constants
# Per-team structure:
#   - Team aggregate features (eggs, food, wing counts)
#   - Queen stats
#   - 4 workers, each with base features + running stats

NUM_TEAMS = 2
NUM_WORKERS_PER_TEAM = 4

# Team aggregate features
TEAM_EGGS = 0
TEAM_FOOD_DEPOSITED = 1
TEAM_WINGS_NO_SPEED = 2
TEAM_WINGS_AND_SPEED = 3
NUM_TEAM_AGGREGATE_FEATURES = 4

# Queen running stats (relative to team start + aggregate features)
QUEEN_KILLS = 0
QUEEN_DEATHS = 1
QUEEN_QUEEN_KILLS = 2
QUEEN_MILITARY_KILLS = 3
QUEEN_DRONE_KILLS = 4
NUM_QUEEN_STATS = 5

# Worker base features (relative to worker start)
WORKER_IS_BOT = 0
WORKER_HAS_FOOD = 1
WORKER_HAS_SPEED = 2
WORKER_HAS_WINGS = 3
NUM_WORKER_BASE_FEATURES = 4

# Worker running stats (relative to worker start + base features)
WORKER_KILLS = 0
WORKER_DEATHS = 1
WORKER_QUEEN_KILLS = 2
WORKER_MILITARY_KILLS = 3
WORKER_DRONE_KILLS = 4
WORKER_MILITARY_DEATHS = 5
WORKER_DRONE_DEATHS = 6
WORKER_BERRIES = 7
WORKER_SNAIL_PROGRESS = 8
NUM_WORKER_STATS = 9

# Computed sizes
NUM_WORKER_FEATURES = NUM_WORKER_BASE_FEATURES + NUM_WORKER_STATS  # 13
NUM_FEATURES_PER_TEAM = (NUM_TEAM_AGGREGATE_FEATURES + NUM_QUEEN_STATS +
                          NUM_WORKERS_PER_TEAM * NUM_WORKER_FEATURES)  # 61

# Team offsets in feature vector
BLUE_TEAM_OFFSET = 0
GOLD_TEAM_OFFSET = NUM_FEATURES_PER_TEAM
TEAM_OFFSETS = [BLUE_TEAM_OFFSET, GOLD_TEAM_OFFSET]


def get_queen_stat_index(team_offset, stat_idx):
    """Get absolute feature index for a queen stat."""
    return team_offset + NUM_TEAM_AGGREGATE_FEATURES + stat_idx


def get_worker_start(team_offset, worker_idx):
    """Get starting feature index for a worker."""
    return (team_offset + NUM_TEAM_AGGREGATE_FEATURES + NUM_QUEEN_STATS +
            worker_idx * NUM_WORKER_FEATURES)


def get_worker_stat_index(team_offset, worker_idx, stat_idx):
    """Get absolute feature index for a worker running stat."""
    worker_start = get_worker_start(team_offset, worker_idx)
    return worker_start + NUM_WORKER_BASE_FEATURES + stat_idx


def get_feature_groups():
    """Return dict mapping group name to list of feature indices to mask."""
    groups = {}

    # Queen stats by type
    queen_stat_names = ['kills', 'deaths', 'queen_kills', 'military_kills', 'drone_kills']
    for stat_idx, stat_name in enumerate(queen_stat_names):
        indices = [get_queen_stat_index(team_off, stat_idx) for team_off in TEAM_OFFSETS]
        groups[f'queen_{stat_name}'] = indices

    # All queen stats together
    groups['all_queen_stats'] = [
        get_queen_stat_index(team_off, stat_idx)
        for team_off in TEAM_OFFSETS
        for stat_idx in range(NUM_QUEEN_STATS)
    ]

    # Worker running stats by type
    worker_stat_names = ['kills', 'deaths', 'queen_kills', 'military_kills', 'drone_kills',
                         'military_deaths', 'drone_deaths', 'berries', 'snail_progress']
    for stat_idx, stat_name in enumerate(worker_stat_names):
        indices = [
            get_worker_stat_index(team_off, worker_idx, stat_idx)
            for team_off in TEAM_OFFSETS
            for worker_idx in range(NUM_WORKERS_PER_TEAM)
        ]
        groups[f'worker_{stat_name}'] = indices

    # All worker running stats together
    groups['all_worker_stats'] = [
        get_worker_stat_index(team_off, worker_idx, stat_idx)
        for team_off in TEAM_OFFSETS
        for worker_idx in range(NUM_WORKERS_PER_TEAM)
        for stat_idx in range(NUM_WORKER_STATS)
    ]

    # All new features (queen + worker stats)
    groups['all_new_features'] = groups['all_queen_stats'] + groups['all_worker_stats']

    return groups


def materialize_data():
    """Materialize training and test data."""
    map_infos = map_structure.MapStructureInfos()

    print("Materializing training data (shards 900-910)...")
    train_states, train_labels = [], []
    for shard in range(900, 911):
        csv_path = f'new_data_partitioned/gameevents_{shard:03d}.csv.gz'
        events = iterate_events_from_csv(csv_path)
        game_states = iterate_game_events_with_state(events, map_infos)
        states, labels = create_game_states_matrix(game_states, drop_state_probability=0.0)
        train_states.append(states)
        train_labels.append(labels)
        print(f"  Shard {shard}: {len(labels)} samples")

    train_X = np.vstack(train_states)
    train_y = np.concatenate(train_labels)
    print(f"Training: {len(train_y):,} samples, {train_X.shape[1]} features")

    print("\nMaterializing test data (shard 911)...")
    csv_path = 'new_data_partitioned/gameevents_911.csv.gz'
    events = iterate_events_from_csv(csv_path)
    game_states = iterate_game_events_with_state(events, map_infos)
    test_X, test_y = create_game_states_matrix(game_states, drop_state_probability=0.0)
    print(f"Test: {len(test_y):,} samples")

    return train_X, train_y, test_X, test_y


def train_and_eval(train_X, train_y, test_X, test_y, mask_indices=None):
    """Train model and return log loss. Optionally mask certain features."""
    X_train = train_X.copy()
    X_test = test_X.copy()

    if mask_indices:
        X_train[:, mask_indices] = 0
        X_test[:, mask_indices] = 0

    param = {'num_leaves': 100, 'objective': 'binary', 'metric': 'binary_logloss',
             'boosting': 'gbdt', 'verbose': -1}
    train_data = lgb.Dataset(X_train, train_y)
    model = lgb.train(param, train_data, num_boost_round=100)

    predictions = model.predict(X_test)
    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    return log_loss, accuracy


def run_ablation():
    train_X, train_y, test_X, test_y = materialize_data()

    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    # Baseline: all features
    baseline_loss, baseline_acc = train_and_eval(train_X, train_y, test_X, test_y)
    print(f"\nBaseline (all {train_X.shape[1]} features): Loss={baseline_loss:.4f}, Acc={baseline_acc:.4f}")

    groups = get_feature_groups()
    results = []

    print("\nTesting feature groups (masking each group)...")
    print("-" * 60)

    for group_name, indices in sorted(groups.items()):
        loss, acc = train_and_eval(train_X, train_y, test_X, test_y, mask_indices=indices)
        delta_loss = loss - baseline_loss
        # Positive delta = masking hurts (feature helps)
        # Negative delta = masking helps (feature hurts)
        status = "HELPS" if delta_loss > 0.0001 else ("HURTS" if delta_loss < -0.0001 else "NEUTRAL")
        results.append((group_name, len(indices), loss, delta_loss, status))
        print(f"{group_name:30s} ({len(indices):2d} feats): Loss={loss:.4f} (Δ={delta_loss:+.4f}) {status}")

    print("\n" + "="*60)
    print("SUMMARY: Features that HURT performance (should remove)")
    print("="*60)
    hurting = [(name, delta) for name, n, loss, delta, status in results if status == "HURTS"]
    if hurting:
        for name, delta in sorted(hurting, key=lambda x: x[1]):
            print(f"  {name}: Δ={delta:+.4f}")
    else:
        print("  None found - all features neutral or helpful")

    print("\n" + "="*60)
    print("SUMMARY: Features that HELP performance (should keep)")
    print("="*60)
    helping = [(name, delta) for name, n, loss, delta, status in results if status == "HELPS"]
    if helping:
        for name, delta in sorted(helping, key=lambda x: -x[1]):
            print(f"  {name}: Δ={delta:+.4f}")
    else:
        print("  None found - all features neutral or hurting")


if __name__ == '__main__':
    run_ablation()
