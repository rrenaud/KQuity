# Training the LightGBM Win Predictor Model

This document describes the complete pipeline for training the LightGBM win predictor model from Killer Queen game event data.

## Overview

The KQuity project builds a binary classification model that predicts match outcomes (Blue wins vs Gold wins) based on real-time game state extracted from Killer Queen arcade cabinet event logs.

**Model**: LightGBM Gradient Boosting Decision Trees
**Task**: Binary Classification (Blue win = 1, Gold win = 0)
**Current Best Accuracy**: ~70.4%

---

## Prerequisites

### Software Dependencies

```bash
pip install lightgbm numpy scikit-learn pandas
```

### Data Requirements

The training pipeline expects validated game event CSV files in:
```
/home/rrenaud/KQuity/validated_all_gameevent_partitioned/
```

These files follow the naming pattern `gameevents_000.csv` through `gameevents_091.csv`, with each partition containing ~1000 games worth of events.

**CSV Schema (Game Events)**:
```
id, timestamp, event_type, values, game_id
```

---

## Step 1: Data Validation (Optional - Already Done)

If working with raw game data, validate it first using `preprocess.py`:

```python
from preprocess import validate_game_data

validate_game_data(
    input_file='raw_gameevents.csv',
    output_file='validated_gameevents.csv'
)
```

Validation checks:
- Events occur after September 2022
- Games replay correctly through the state engine
- Victory conditions match actual game states (economic, military, or snail wins)

---

## Step 2: Configure the Experiment

In `preprocess.py`, set the experiment name (around line 23):

```python
expt_name = 'your_experiment_name'
```

This creates the output directory: `model_experiments/your_experiment_name/`

### Key Configuration Options

**Worker Ordering** (line ~610 in `vectorize_team()`):
```python
# Sort workers by power (strongest first) - RECOMMENDED
workers = sorted(workers, key=lambda w: -w.power())
```

**State Sampling** (line ~672 in `materialize_game_state_matrix()`):
```python
# Drop probability for training data balance
# Higher values = fewer samples but faster training
drop_prob = 0.9  # Drop 90% of states
```

**Minimum Game Time** (line ~670):
```python
# Only sample states after game has been running
if game_time < 5:  # Skip first 5 seconds
    continue
```

---

## Step 3: Materialize Feature Matrices

Convert raw event CSVs to numpy arrays for efficient training:

```python
from preprocess import materialize_game_state_matrix

# Process training files (e.g., files 0-79)
for i in range(80):
    filename = f'validated_all_gameevent_partitioned/gameevents_{i:03d}.csv'
    materialize_game_state_matrix(filename, drop_prob=0.9)

# Process test files (e.g., files 80-91) with no dropping
for i in range(80, 92):
    filename = f'validated_all_gameevent_partitioned/gameevents_{i:03d}.csv'
    materialize_game_state_matrix(filename, drop_prob=0.0)
```

This creates two files per input CSV:
- `gameevents_XXX.csv_states.npy` - Feature matrix (N samples × 49 features)
- `gameevents_XXX.csv_labels.npy` - Label vector (N samples, binary)

---

## Step 4: Feature Vector Structure

Each game state is encoded as a ~49-dimensional feature vector:

| Feature Range | Count | Description |
|---------------|-------|-------------|
| 0-19 | 20 | Blue team state |
| 20-39 | 20 | Gold team state |
| 40-44 | 5 | Maiden control states |
| 45-48 | 4 | Map one-hot encoding |
| 49 | 1 | Normalized berries available |
| 50-51 | 2 | Snail position and velocity |

### Team State Features (20 per team)

```
[0] eggs (queen health, 2 at start, -1 = dead)
[1] food_deposited (count toward economic win)
[2] vanilla_warriors (workers with wings only)
[3] speed_warriors (workers with wings + speed)
[4-7] worker_1: has_bot, has_food, has_speed, has_wings
[8-11] worker_2: has_bot, has_food, has_speed, has_wings
[12-15] worker_3: has_bot, has_food, has_speed, has_wings
[16-19] worker_4: has_bot, has_food, has_speed, has_wings
```

### Maiden Features

Encoded as: 0 (neutral), 1 (Blue control), -1 (Gold control)

### Snail Features

- **Position**: Normalized to [-0.5, 0.5] from center
- **Velocity**: Normalized by max snail speed
- Multiplied by symmetry factor to handle gold_on_left orientation

---

## Step 5: Train the Model

Use the `Train_LightGBM.ipynb` notebook or run directly:

```python
import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, accuracy_score, classification_report

# Load training data
def load_vectors(base_path, file_range):
    states_list, labels_list = [], []
    for i in file_range:
        states = np.load(f'{base_path}/gameevents_{i:03d}.csv_states.npy')
        labels = np.load(f'{base_path}/gameevents_{i:03d}.csv_labels.npy')
        states_list.append(states)
        labels_list.append(labels)
    return np.vstack(states_list), np.concatenate(labels_list)

expt_dir = 'model_experiments/your_experiment_name'
train_X, train_y = load_vectors(expt_dir, range(0, 80))
test_X, test_y = load_vectors(expt_dir, range(80, 92))

# Configure LightGBM
params = {
    'num_leaves': 100,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting': 'gbdt',
    'verbose': -1
}

# Train
train_data = lgb.Dataset(train_X, train_y)
model = lgb.train(params, train_data, num_boost_round=100)

# Save model
model.save_model(f'{expt_dir}/model.mdl')
```

---

## Step 6: Evaluate the Model

```python
# Predictions
predictions = model.predict(test_X)

# Metrics
print(f"Log Loss: {log_loss(test_y, predictions):.4f}")
print(f"Accuracy: {accuracy_score(test_y, predictions > 0.5):.4f}")
print(classification_report(test_y, predictions > 0.5,
                           target_names=['Gold Wins', 'Blue Wins']))
```

### Monotonicity Validation (Egg Inversion Test)

A well-trained model should predict higher Blue win probability when Blue has more eggs. The "egg inversion" metric measures violations:

```python
def compute_egg_inversions(model, test_X):
    """Measure % of cases where adding Blue eggs DECREASES predicted win prob."""
    original_preds = model.predict(test_X)

    # Boost Blue eggs (index 0) by 2
    modified_X = test_X.copy()
    modified_X[:, 0] += 2
    modified_preds = model.predict(modified_X)

    inversions = (modified_preds < original_preds).mean()
    return inversions

print(f"Egg Inversions: {compute_egg_inversions(model, test_X):.4f}")
# Goal: < 0.02 (less than 2% inversions)
```

---

## Hyperparameter Tuning

Experiments that have been tried:

| Experiment | num_leaves | num_trees | Drop Rate | Accuracy | Log Loss |
|------------|------------|-----------|-----------|----------|----------|
| baseline | 100 | 100 | 0% | ~68% | ~0.59 |
| drop_90 | 100 | 100 | 90% | ~70% | ~0.56 |
| more_leaves | 200 | 100 | 90% | ~70% | ~0.56 |
| power_sorted | 100 | 100 | 90% | 70.4% | 0.556 |

**Recommended Settings**:
- `num_leaves`: 100
- `num_boost_round`: 100
- Training drop rate: 90%
- Test drop rate: 0%
- Worker sorting: by power (strongest first)

---

## Full Training Pipeline Summary

```bash
# 1. Ensure validated data exists
ls validated_all_gameevent_partitioned/gameevents_*.csv

# 2. Edit preprocess.py to set experiment name and parameters

# 3. Run materialization (generates .npy files)
python -c "
from preprocess import materialize_game_state_matrix
for i in range(92):
    drop = 0.9 if i < 80 else 0.0
    materialize_game_state_matrix(
        f'validated_all_gameevent_partitioned/gameevents_{i:03d}.csv',
        drop_prob=drop
    )
"

# 4. Train in notebook or script
jupyter notebook Train_LightGBM.ipynb
```

---

## Using the Trained Model

```python
import lightgbm as lgb
from preprocess import GameState, vectorize_game_state

# Load model
model = lgb.Booster(model_file='model_experiments/your_experiment/model.mdl')

# Create game state from events (or real-time)
game_state = GameState(map_info)
for event in events:
    event.modify_game_state(game_state)

# Predict
feature_vector = vectorize_game_state(game_state)
win_probability = model.predict([feature_vector])[0]
print(f"Blue win probability: {win_probability:.2%}")
```

---

## Troubleshooting

### Common Issues

1. **Low accuracy**: Ensure workers are sorted by power, use 90% drop rate on training data
2. **High egg inversions**: Model may be overfitting to noise; reduce `num_leaves`
3. **Memory errors**: Process fewer files at once, or increase drop rate
4. **Validation failures**: Check that game events are properly formatted and ordered

### Data Quality Checks

```python
from preprocess import is_valid_game, iterate_events_by_game_and_normalize_time

# Count valid vs invalid games
valid, invalid = 0, 0
for game_id, events in iterate_events_by_game_and_normalize_time('gameevents.csv'):
    if is_valid_game(events):
        valid += 1
    else:
        invalid += 1
print(f"Valid: {valid}, Invalid: {invalid}")
```

---

## File Reference

| File | Purpose |
|------|---------|
| `preprocess.py` | Event parsing, game state tracking, feature vectorization |
| `constants.py` | Game enums (teams, victory conditions, maps) |
| `map_structure.py` | Map metadata (berry/maiden positions) |
| `map_structure_info.json` | Hardcoded map coordinates |
| `Train_LightGBM.ipynb` | Interactive training notebook |
| `validated_all_gameevent_partitioned/` | Input data directory |
| `model_experiments/` | Output models and feature matrices |
