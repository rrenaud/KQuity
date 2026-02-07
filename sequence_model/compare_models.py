"""Data-matched comparison of sequence model vs LightGBM win probability.

Both models are evaluated on the same test games. For each game, we compare
their win probability predictions.

The comparison uses per-game accuracy: for each complete game, we take each
model's prediction at various points and check if it correctly predicts the
winner.

For LightGBM: predictions on materialized feature-vector snapshots (1 per event)
For Seq model: predictions at each token position

We compare:
1. Overall accuracy and log loss on all test snapshots
2. Per-game-stage breakdown (early / mid / late)

Usage:
    # Evaluate on a single test CSV (with optional LGB retraining):
    python -m sequence_model.compare_models \
        --test-csv late_tournament_games/late_tournament_game_events.csv.gz \
        --lgb-train-csv logged_in_games/gameevents_000.csv.gz

    # Partition-range mode (original):
    python -m sequence_model.compare_models \
        --seq-checkpoint sequence_model/out/ckpt.pt \
        --lgb-model model_experiments/new_data_model/model.mdl
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import lightgbm as lgb
from fast_materialize import fast_materialize
from sequence_model.model import KQModel, GPTConfig
from sequence_model.tokenize_games import tokenize_single_game
from sequence_model.vocab import BOS, EOS
from preprocess import (
    iterate_events_from_csv,
    iterate_events_by_game_and_normalize_time,
    is_valid_game,
)
import map_structure


def load_seq_model(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    model = KQModel(config)
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_metrics(probs, labels):
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    accuracy = np.mean((probs > 0.5) == labels)
    log_loss = -np.mean(
        labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return accuracy, log_loss


def eval_lgb_on_partition(csv_path, lgb_model):
    """Return (predictions, labels) arrays from LightGBM on one partition."""
    states, labels = fast_materialize(csv_path, drop_state_probability=0.0)
    if states is None or len(labels) == 0:
        return np.array([]), np.array([])
    preds = lgb_model.predict(states)
    return preds, labels


def eval_seq_on_partition(csv_path, seq_model, map_infos, device, block_size,
                          sample_rate=0.1):
    """Return (predictions, labels) for seq model on one partition.

    For each game, tokenize it and sample ~sample_rate of token positions
    to evaluate win probability. This gives a manageable number of samples
    comparable to LightGBM's per-event snapshots.
    """
    all_probs = []
    all_labels = []

    events_iter = iterate_events_from_csv(csv_path)
    for game_id, game_events in iterate_events_by_game_and_normalize_time(events_iter):
        error = is_valid_game(game_events, map_infos)
        if error:
            continue

        result = tokenize_single_game(game_events, map_infos)
        if result is None:
            continue

        tokens, blue_wins = result

        # Truncate to block_size for the model
        if len(tokens) > block_size:
            tokens = tokens[:block_size]

        # Skip very short games
        if len(tokens) < 10:
            continue

        # Sample positions (skip BOS header of 3 tokens and EOS)
        n = len(tokens)
        # Sample roughly every 1/sample_rate tokens to match LightGBM density
        # LightGBM gets ~1 snapshot per event, events are ~3 tokens on average
        step = max(1, int(3 / sample_rate))
        indices = list(range(3, n - 1, step))
        if not indices:
            continue

        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            wp_probs = seq_model.estimate_win_probability(token_tensor)
        wp_probs = wp_probs[0].cpu().numpy()

        for idx in indices:
            all_probs.append(wp_probs[idx])
            all_labels.append(float(blue_wins))

    return np.array(all_probs), np.array(all_labels)


def eval_seq_by_stage(csv_path, seq_model, map_infos, device, block_size):
    """Return per-stage predictions for the seq model.

    For each game, split into early/mid/late thirds and sample one prediction
    per stage region.
    """
    stage_data = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}

    events_iter = iterate_events_from_csv(csv_path)
    for game_id, game_events in iterate_events_by_game_and_normalize_time(events_iter):
        error = is_valid_game(game_events, map_infos)
        if error:
            continue

        result = tokenize_single_game(game_events, map_infos)
        if result is None:
            continue

        tokens, blue_wins = result
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
        n = len(tokens)
        if n < 12:
            continue

        token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            wp_probs = seq_model.estimate_win_probability(token_tensor)
        wp_probs = wp_probs[0].cpu().numpy()

        label = float(blue_wins)
        # Content tokens are positions 3 to n-2 (skip BOS header and EOS)
        content_start = 3
        content_end = n - 1
        content_len = content_end - content_start

        # Sample ~every 3 tokens (matching LightGBM's 1-per-event density)
        for i in range(content_start, content_end, 3):
            frac = (i - content_start) / max(content_len - 1, 1)
            if frac < 0.25:
                stage = 'early'
            elif frac < 0.75:
                stage = 'mid'
            else:
                stage = 'late'
            stage_data[stage][0].append(wp_probs[i])
            stage_data[stage][1].append(label)

    return {k: (np.array(v[0]), np.array(v[1])) for k, v in stage_data.items()}


def eval_lgb_by_stage(csv_path, lgb_model):
    """Return per-stage predictions for LightGBM.

    fast_materialize returns one row per event. We group by game (contiguous
    blocks with same label) and split into early/mid/late.
    """
    states, labels = fast_materialize(csv_path, drop_state_probability=0.0)
    if states is None or len(labels) == 0:
        return {s: (np.array([]), np.array([])) for s in ['early', 'mid', 'late']}

    preds = lgb_model.predict(states)

    # Find game boundaries: label flips or non-contiguous blocks
    stage_data = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}
    game_start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[game_start]:
            # Game from game_start to i-1
            game_len = i - game_start
            for j in range(game_start, i):
                frac = (j - game_start) / max(game_len - 1, 1)
                if frac < 0.25:
                    stage = 'early'
                elif frac < 0.75:
                    stage = 'mid'
                else:
                    stage = 'late'
                stage_data[stage][0].append(preds[j])
                stage_data[stage][1].append(labels[j])
            game_start = i

    return {k: (np.array(v[0]), np.array(v[1])) for k, v in stage_data.items()}


def train_lgb_from_csv(csv_path, num_leaves=200, num_trees=200):
    """Train a LightGBM model from a CSV/gzip file.

    Uses fast_materialize to convert game events to feature vectors,
    then trains a LightGBM binary classifier.
    """
    print(f"  Materializing training data from {csv_path}...")
    states, labels = fast_materialize(csv_path, drop_state_probability=0.0)
    print(f"  Training samples: {len(labels):,}")
    param = {
        'num_leaves': num_leaves,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'verbose': 0,
    }
    train_data = lgb.Dataset(states, labels)
    model = lgb.train(param, train_data, num_boost_round=num_trees)
    return model


def print_comparison(all_lgb_preds, all_lgb_labels, all_seq_preds,
                     all_seq_labels, stage_lgb, stage_seq, data_desc):
    """Print overall and per-stage comparison results."""
    lgb_preds = np.concatenate(all_lgb_preds)
    lgb_labels = np.concatenate(all_lgb_labels)
    seq_preds = np.concatenate(all_seq_preds)
    seq_labels = np.concatenate(all_seq_labels)

    lgb_acc, lgb_ll = compute_metrics(lgb_preds, lgb_labels)
    seq_acc, seq_ll = compute_metrics(seq_preds, seq_labels)

    print(f"\n{'='*65}")
    print(f"OVERALL COMPARISON ({data_desc})")
    print(f"{'='*65}")
    print(f"  LightGBM:  {len(lgb_preds):>8,} snapshots | "
          f"Acc {100*lgb_acc:.1f}% | LogLoss {lgb_ll:.4f}")
    print(f"  Seq Model: {len(seq_preds):>8,} snapshots | "
          f"Acc {100*seq_acc:.1f}% | LogLoss {seq_ll:.4f}")

    print(f"\n{'='*65}")
    print("BY GAME STAGE (early <25% | mid 25-75% | late >75% of game)")
    print(f"{'='*65}")
    print(f"{'Stage':<8} {'Model':<12} {'N':>8} {'Accuracy':>10} {'LogLoss':>10}")
    print(f"{'-'*48}")
    for stage in ['early', 'mid', 'late']:
        for name, sd in [('LightGBM', stage_lgb), ('Seq Model', stage_seq)]:
            if sd[stage][0]:
                p = np.concatenate(sd[stage][0])
                l = np.concatenate(sd[stage][1])
                acc, ll = compute_metrics(p, l)
                print(f"{stage:<8} {name:<12} {len(p):>8,} {100*acc:>9.1f}% {ll:>10.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Data-matched comparison: sequence model vs LightGBM')
    parser.add_argument('--seq-checkpoint', type=str,
                        default='sequence_model/out/ckpt.pt')
    parser.add_argument('--lgb-model', type=str,
                        default='model_experiments/new_data_model/model.mdl')
    parser.add_argument('--data-dir', type=str, default='new_data_partitioned')
    parser.add_argument('--val-start', type=int, default=740)
    parser.add_argument('--val-end', type=int, default=925)
    parser.add_argument('--max-partitions', type=int, default=10,
                        help='Max val partitions to evaluate (for speed)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--block-size', type=int, default=2560)
    parser.add_argument('--test-csv', type=str, default=None,
                        help='Evaluate both models on this CSV file '
                             '(e.g. late_tournament_game_events.csv.gz)')
    parser.add_argument('--lgb-train-csv', type=str, default=None,
                        help='Train a fresh LightGBM model from this CSV '
                             'instead of loading --lgb-model')
    args = parser.parse_args()

    print(f"Loading sequence model from {args.seq_checkpoint}")
    seq_model = load_seq_model(args.seq_checkpoint, args.device)
    # Use the model's block_size if it's smaller than requested
    model_block_size = seq_model.config.block_size
    if model_block_size < args.block_size:
        print(f"  Note: model block_size={model_block_size}, using that instead of {args.block_size}")
        args.block_size = model_block_size

    if args.lgb_train_csv:
        print(f"\nTraining LightGBM from {args.lgb_train_csv}")
        lgb_model = train_lgb_from_csv(args.lgb_train_csv)
        print("  LightGBM training complete")
    else:
        print(f"Loading LightGBM model from {args.lgb_model}")
        lgb_model = lgb.Booster(model_file=args.lgb_model)

    map_infos = map_structure.MapStructureInfos()

    if args.test_csv:
        # Single-CSV test mode
        print(f"\nEvaluating on {args.test_csv}")

        all_lgb_preds, all_lgb_labels = [], []
        all_seq_preds, all_seq_labels = [], []
        stage_seq = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}
        stage_lgb = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}

        # LightGBM
        lp, ll = eval_lgb_on_partition(args.test_csv, lgb_model)
        if len(lp) > 0:
            all_lgb_preds.append(lp)
            all_lgb_labels.append(ll)

        # Seq model
        sp, sl = eval_seq_on_partition(args.test_csv, seq_model, map_infos,
                                       args.device, args.block_size)
        if len(sp) > 0:
            all_seq_preds.append(sp)
            all_seq_labels.append(sl)

        # Per-stage
        lgb_stages = eval_lgb_by_stage(args.test_csv, lgb_model)
        seq_stages = eval_seq_by_stage(args.test_csv, seq_model, map_infos,
                                       args.device, args.block_size)
        for stage in ['early', 'mid', 'late']:
            if len(lgb_stages[stage][0]) > 0:
                stage_lgb[stage][0].append(lgb_stages[stage][0])
                stage_lgb[stage][1].append(lgb_stages[stage][1])
            if len(seq_stages[stage][0]) > 0:
                stage_seq[stage][0].append(seq_stages[stage][0])
                stage_seq[stage][1].append(seq_stages[stage][1])

        print(f"  LGB {len(lp):,} snapshots, Seq {len(sp):,} snapshots")

        data_desc = f"test set: {os.path.basename(args.test_csv)}"
        print_comparison(all_lgb_preds, all_lgb_labels,
                         all_seq_preds, all_seq_labels,
                         stage_lgb, stage_seq, data_desc)
    else:
        # Partition-range mode (original behavior)
        partitions = list(range(args.val_start,
                                min(args.val_end, args.val_start + args.max_partitions)))
        print(f"\nEvaluating on {len(partitions)} val partitions "
              f"({partitions[0]}-{partitions[-1]})")

        all_lgb_preds, all_lgb_labels = [], []
        all_seq_preds, all_seq_labels = [], []
        stage_seq = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}
        stage_lgb = {'early': ([], []), 'mid': ([], []), 'late': ([], [])}

        for partition in partitions:
            csv_path = os.path.join(args.data_dir,
                                    f'gameevents_{partition:03d}.csv.gz')
            if not os.path.exists(csv_path):
                continue

            # LightGBM
            lp, ll = eval_lgb_on_partition(csv_path, lgb_model)
            if len(lp) > 0:
                all_lgb_preds.append(lp)
                all_lgb_labels.append(ll)

            # Seq model
            sp, sl = eval_seq_on_partition(csv_path, seq_model, map_infos,
                                           args.device, args.block_size)
            if len(sp) > 0:
                all_seq_preds.append(sp)
                all_seq_labels.append(sl)

            # Per-stage
            lgb_stages = eval_lgb_by_stage(csv_path, lgb_model)
            seq_stages = eval_seq_by_stage(csv_path, seq_model, map_infos,
                                           args.device, args.block_size)
            for stage in ['early', 'mid', 'late']:
                if len(lgb_stages[stage][0]) > 0:
                    stage_lgb[stage][0].append(lgb_stages[stage][0])
                    stage_lgb[stage][1].append(lgb_stages[stage][1])
                if len(seq_stages[stage][0]) > 0:
                    stage_seq[stage][0].append(seq_stages[stage][0])
                    stage_seq[stage][1].append(seq_stages[stage][1])

            print(f"  Partition {partition:03d}: "
                  f"LGB {len(lp):,} snapshots, Seq {len(sp):,} snapshots")

        data_desc = f"val partitions {partitions[0]}-{partitions[-1]}"
        print_comparison(all_lgb_preds, all_lgb_labels,
                         all_seq_preds, all_seq_labels,
                         stage_lgb, stage_seq, data_desc)


if __name__ == '__main__':
    main()
