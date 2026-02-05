"""
Evaluate the trained KQ sequence model.

Metrics:
1. Next-token perplexity
2. Win probability accuracy and log loss (vs LightGBM baseline)
3. Win probability calibration (reliability diagram)
4. Egg inversion test (monotonicity sanity check)

Usage:
    python -m sequence_model.evaluate
    python -m sequence_model.evaluate --checkpoint sequence_model/out/ckpt.pt
"""

import argparse
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sequence_model.model import KQModel, GPTConfig
from sequence_model.vocab import (
    VOCAB_SIZE, BOS, EOS, PAD,
    VICTORY_BLUE_MILITARY, VICTORY_BLUE_ECONOMIC, VICTORY_BLUE_SNAIL,
    VICTORY_GOLD_MILITARY, VICTORY_GOLD_ECONOMIC, VICTORY_GOLD_SNAIL,
    PLAYER_KILL, KILLED_QUEEN, TOKEN_NAMES,
    decode_tokens,
)


VICTORY_TOKENS = {
    VICTORY_BLUE_MILITARY, VICTORY_BLUE_ECONOMIC, VICTORY_BLUE_SNAIL,
    VICTORY_GOLD_MILITARY, VICTORY_GOLD_ECONOMIC, VICTORY_GOLD_SNAIL,
}
BLUE_VICTORY_TOKENS = {VICTORY_BLUE_MILITARY, VICTORY_BLUE_ECONOMIC, VICTORY_BLUE_SNAIL}


def load_model(checkpoint_path, device='cpu'):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = KQModel(config)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def compute_perplexity(model, data_dir, split, block_size, batch_size,
                       device, ctx, num_batches=100):
    """Compute next-token perplexity on a data split."""
    token_file = os.path.join(data_dir, f'{split}.bin')
    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')

    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(tokens[i:i + block_size].astype(np.int64)) for i in ix
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(tokens[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
        ]).to(device)

        with ctx:
            logits, _, _, _ = model(x, y)
            # Recompute loss manually to get proper token count
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1),
                ignore_index=-1, reduction='sum')

        total_loss += loss.item()
        total_tokens += (y != -1).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity, avg_loss


@torch.no_grad()
def compute_win_prob_metrics(model, data_dir, split, block_size, batch_size,
                             device, ctx, num_batches=200):
    """Compute win probability accuracy and log loss.

    We evaluate at every position (matching training), collecting predictions
    and labels.
    """
    token_file = os.path.join(data_dir, f'{split}.bin')
    label_file = os.path.join(data_dir, f'{split}_labels.bin')
    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
    labels = np.memmap(label_file, dtype=np.uint8, mode='r')

    all_probs = []
    all_labels = []

    for _ in range(num_batches):
        ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))
        x = torch.stack([
            torch.from_numpy(tokens[i:i + block_size].astype(np.int64)) for i in ix
        ]).to(device)
        wp = torch.stack([
            torch.from_numpy(labels[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
        ]).to(device)

        with ctx:
            _, wp_logits, _, _ = model(x)

        wp_mask = (wp != -1)
        if wp_mask.any():
            probs = torch.sigmoid(wp_logits[wp_mask]).float().cpu().numpy()
            labs = wp[wp_mask].float().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labs)

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Clip for numerical stability
    all_probs = np.clip(all_probs, 1e-7, 1 - 1e-7)

    # Accuracy
    accuracy = np.mean((all_probs > 0.5) == all_labels)

    # Log loss
    log_loss = -np.mean(
        all_labels * np.log(all_probs) +
        (1 - all_labels) * np.log(1 - all_probs))

    return {
        'accuracy': accuracy,
        'log_loss': log_loss,
        'num_samples': len(all_probs),
        'probs': all_probs,
        'labels': all_labels,
    }


def compute_calibration(probs, labels, n_bins=10):
    """Compute calibration statistics for a reliability diagram.

    Returns list of (bin_center, fraction_positive, count) tuples.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    calibration = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            frac_pos = labels[mask].mean()
            bin_center = (bins[i] + bins[i + 1]) / 2
            calibration.append((bin_center, frac_pos, mask.sum()))
    return calibration


def egg_inversion_test(model, data_dir, device, ctx, block_size=1024,
                       num_games=200):
    """Test that queen kills (egg losses) shift win probability correctly.

    For each game sequence, find positions where a playerKill targets a queen.
    The model should predict *lower* win probability for the team that lost a queen.

    Since we track events as tokens, we look for the pattern:
        playerKill player_X player_Y killed_queen

    After seeing this event, P(blue wins) should decrease if player_Y is on
    blue's team (odd position_ids are blue, even are gold... actually position_id
    % 2 == 0 is blue per the codebase constants).

    We test by comparing win prob before and after the kill event tokens.
    """
    token_file = os.path.join(data_dir, 'val.bin')
    label_file = os.path.join(data_dir, 'val_labels.bin')
    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
    labels = np.memmap(label_file, dtype=np.uint8, mode='r')

    correct = 0
    total = 0

    # Sample random positions and look for queen kill patterns
    for _ in range(num_games):
        start = np.random.randint(0, max(1, len(tokens) - block_size))
        chunk = tokens[start:start + block_size]

        # Find queen kills in this chunk: playerKill ... killed_queen
        for i in range(len(chunk) - 4):
            if chunk[i] == PLAYER_KILL and chunk[i + 3] == KILLED_QUEEN:
                # We found a queen kill at positions i..i+3
                # Get win prob before (at position i-1) and after (at position i+3)
                if i < 4 or i + 4 >= len(chunk):
                    continue

                # Context up to just before the kill
                pre_ctx = torch.from_numpy(
                    chunk[:i].astype(np.int64)).unsqueeze(0).to(device)
                # Context including the kill
                post_ctx = torch.from_numpy(
                    chunk[:i + 4].astype(np.int64)).unsqueeze(0).to(device)

                if pre_ctx.size(1) < 3 or post_ctx.size(1) < 3:
                    continue
                if pre_ctx.size(1) > block_size or post_ctx.size(1) > block_size:
                    continue

                with ctx:
                    pre_prob = model.estimate_win_probability(pre_ctx)[0, -1].item()
                    post_prob = model.estimate_win_probability(post_ctx)[0, -1].item()

                # Determine which team lost a queen
                # killed_position_id is encoded as player_N token at chunk[i+2]
                from sequence_model.vocab import PLAYER_1
                killed_player_token = chunk[i + 2]
                killed_position_id = killed_player_token - PLAYER_1 + 1
                # position_id % 2 == 0 means blue team
                blue_lost_queen = (killed_position_id % 2 == 0)

                if blue_lost_queen:
                    # Blue lost queen → P(blue wins) should decrease
                    if post_prob < pre_prob:
                        correct += 1
                elif not blue_lost_queen:
                    # Gold lost queen → P(blue wins) should increase
                    if post_prob > pre_prob:
                        correct += 1

                total += 1

    if total == 0:
        return {'accuracy': 0.0, 'total': 0}

    return {
        'accuracy': correct / total,
        'total': total,
        'correct': correct,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate KQ sequence model')
    parser.add_argument('--checkpoint', type=str,
                        default='sequence_model/out/ckpt.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                        default='sequence_model/data',
                        help='Directory with .bin data files')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-batches', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=1024)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    ptdtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    print(f"Loading model from {args.checkpoint}")
    model, checkpoint = load_model(args.checkpoint, args.device)
    print(f"  iter_num: {checkpoint.get('iter_num', 'N/A')}")
    print(f"  best_val_loss: {checkpoint.get('best_val_loss', 'N/A')}")

    # 1. Perplexity
    print("\n=== Next-Token Perplexity ===")
    for split in ['train', 'val']:
        ppl, avg_loss = compute_perplexity(
            model, args.data_dir, split, args.block_size, args.batch_size,
            args.device, ctx, num_batches=args.num_batches)
        print(f"  {split}: perplexity={ppl:.2f}, avg_loss={avg_loss:.4f}")

    # 2. Win Probability
    print("\n=== Win Probability Metrics ===")
    for split in ['train', 'val']:
        metrics = compute_win_prob_metrics(
            model, args.data_dir, split, args.block_size, args.batch_size,
            args.device, ctx, num_batches=args.num_batches)
        print(f"  {split}: accuracy={metrics['accuracy']:.4f} "
              f"({100 * metrics['accuracy']:.1f}%), "
              f"log_loss={metrics['log_loss']:.4f}, "
              f"n={metrics['num_samples']:,}")

    # LightGBM baseline comparison
    print("\n  LightGBM baseline: accuracy=70.4%, log_loss=0.556")

    # 3. Calibration
    print("\n=== Calibration (val split) ===")
    val_metrics = compute_win_prob_metrics(
        model, args.data_dir, 'val', args.block_size, args.batch_size,
        args.device, ctx, num_batches=args.num_batches)
    calibration = compute_calibration(val_metrics['probs'], val_metrics['labels'])
    print(f"  {'Bin Center':>10} {'Frac Pos':>10} {'Count':>10}")
    for center, frac, count in calibration:
        print(f"  {center:>10.2f} {frac:>10.3f} {count:>10,}")

    # 4. Egg inversion test
    print("\n=== Egg Inversion Test (queen kill direction) ===")
    egg_results = egg_inversion_test(
        model, args.data_dir, args.device, ctx,
        block_size=args.block_size)
    print(f"  Correct direction: {egg_results['accuracy']:.3f} "
          f"({egg_results.get('correct', 0)}/{egg_results['total']})")


if __name__ == '__main__':
    main()
