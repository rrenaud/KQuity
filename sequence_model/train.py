"""
Training script for Killer Queen sequence model.

Vendored from nanoGPT's train.py and modified for:
- Dual loss (next-token + win probability)
- Game-boundary-aware label loading via parallel label arrays
- Our data format (.bin token files + .bin label files)
- Single-GPU training (no DDP for now)

Usage:
    python -m sequence_model.train
    python -m sequence_model.train --compile=False  # for debugging
"""

import os
import sys
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sequence_model.model import KQModel, GPTConfig, FUTURE_PRED_FEATURE_INDICES
from sequence_model.vocab import VOCAB_SIZE, PAD, BOS as BOS_TOKEN
from fast_materialize import NUM_FEATURES

# -----------------------------------------------------------------------------
# Default config values — designed for our ~2M param KQ model
# -----------------------------------------------------------------------------

# I/O
out_dir = 'sequence_model/out'
eval_interval = 500
log_interval = 10
eval_iters = 200
save_interval = 5000
always_save_checkpoint = False

# Data
data_dir = 'sequence_model/data'

# Model
n_layer = 4
n_head = 4
n_embd = 128
block_size = 2560
dropout = 0.1
bias = False
vocab_size = VOCAB_SIZE

# Optimizer
learning_rate = 3e-4
max_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 3e-5

# Loss weighting
lambda_wp = 0.1

# System
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
do_compile = True

# Batch
batch_size = 64
gradient_accumulation_steps = 1

# Resume
init_from = 'scratch'  # 'scratch' or 'resume'


def parse_args():
    parser = argparse.ArgumentParser(description='Train KQ sequence model')
    parser.add_argument('--out-dir', type=str, default=out_dir)
    parser.add_argument('--data-dir', type=str, default=data_dir)
    parser.add_argument('--batch-size', type=int, default=batch_size)
    parser.add_argument('--block-size', type=int, default=block_size)
    parser.add_argument('--n-layer', type=int, default=n_layer)
    parser.add_argument('--n-head', type=int, default=n_head)
    parser.add_argument('--n-embd', type=int, default=n_embd)
    parser.add_argument('--max-iters', type=int, default=max_iters)
    parser.add_argument('--learning-rate', type=float, default=learning_rate)
    parser.add_argument('--lambda-wp', type=float, default=lambda_wp)
    parser.add_argument('--eval-interval', type=int, default=eval_interval)
    parser.add_argument('--log-interval', type=int, default=log_interval)
    parser.add_argument('--save-interval', type=int, default=save_interval)
    parser.add_argument('--compile', type=str, default=str(do_compile))
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--dtype', type=str, default=dtype)
    parser.add_argument('--init-from', type=str, default=init_from)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--gradient-accumulation-steps', type=int,
                        default=gradient_accumulation_steps)
    parser.add_argument('--eval-iters', type=int, default=eval_iters)
    parser.add_argument('--warmup-iters', type=int, default=warmup_iters)
    parser.add_argument('--lr-decay-iters', type=int, default=lr_decay_iters)
    parser.add_argument('--min-lr', type=float, default=min_lr)
    parser.add_argument('--model-type', type=str, default='transformer',
                        choices=['transformer', 'linear-attn', 'mamba'],
                        help='Sequence mixing layer type')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='kquity-seq',
                        help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    # LightGBM feature injection
    parser.add_argument('--use-lgb-features', action='store_true',
                        help='Load and inject 52D LightGBM features')
    parser.add_argument('--use-lgb-preds', action='store_true',
                        help='Load and inject LightGBM predictions')
    parser.add_argument('--lambda-distill', type=float, default=0.0,
                        help='Weight for distillation loss (default: 0.0)')
    parser.add_argument('--feature-dropout', type=float, default=0.0,
                        help='Probability of zeroing feature embeddings (default: 0.0)')
    parser.add_argument('--zero-tokens', action='store_true',
                        help='Replace all input tokens with PAD (ablation: features only, no sequence info)')
    # Future game state prediction
    parser.add_argument('--future-pred-horizon', type=int, default=0,
                        help='Token-based future prediction horizon K (0=disabled, 5≈1s, 28≈5s)')
    parser.add_argument('--lambda-future', type=float, default=0.0,
                        help='Weight for future prediction loss (default: 0.0)')
    return parser.parse_args()


# Cache for BOS offsets per split — built once, reused across get_batch calls
_bos_offset_cache = {}


def _get_bos_offsets(split, data_dir):
    """Scan .bin file for all positions where token == BOS. Cached per split."""
    key = (split, data_dir)
    if key not in _bos_offset_cache:
        token_file = os.path.join(data_dir, f'{split}.bin')
        tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
        offsets = np.where(tokens == BOS_TOKEN)[0]
        _bos_offset_cache[key] = offsets
        print(f"  Found {len(offsets)} games (BOS offsets) in {split}.bin")
    return _bos_offset_cache[key]


def get_batch(split, data_dir, block_size, batch_size, device, device_type,
              use_features=False, use_lgb_preds=False,
              future_pred_horizon=0, future_feature_indices=None):
    """Load a game-aligned random batch of token sequences.

    Each batch element starts at a BOS (game start) token. Games shorter than
    block_size are padded with PAD=0 for x and -1 for y/wp. Games longer than
    block_size are truncated (keeping the beginning).

    Returns:
        x: (B, T) input tokens
        y: (B, T) next-token targets (-1 = ignore)
        wp: (B, T) win probability labels (-1 = ignore)
        features: (B, T, 52) float32 feature vectors, or None
        lgb_preds: (B, T) float32 LGB predictions, or None
        future_features: (B, T, n_future_features) float32 future targets, or None
    """
    token_file = os.path.join(data_dir, f'{split}.bin')
    label_file = os.path.join(data_dir, f'{split}_labels.bin')

    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
    labels = np.memmap(label_file, dtype=np.uint8, mode='r')
    n = len(tokens)

    # Load feature/pred memmaps if needed
    feat_mmap = None
    pred_mmap = None
    if use_features:
        feat_file = os.path.join(data_dir, f'{split}_features.bin')
        feat_mmap = np.memmap(feat_file, dtype=np.float16, mode='r').reshape(-1, NUM_FEATURES)
    if use_lgb_preds:
        pred_file = os.path.join(data_dir, f'{split}_lgb_preds.bin')
        pred_mmap = np.memmap(pred_file, dtype=np.float16, mode='r')

    bos_offsets = _get_bos_offsets(split, data_dir)
    chosen = np.random.randint(0, len(bos_offsets), size=batch_size)

    x = np.zeros((batch_size, block_size), dtype=np.int64)  # PAD=0
    y = np.full((batch_size, block_size), -1, dtype=np.int64)
    wp = np.full((batch_size, block_size), -1, dtype=np.int64)
    feat_batch = np.zeros((batch_size, block_size, NUM_FEATURES),
                          dtype=np.float32) if use_features else None
    pred_batch = np.zeros((batch_size, block_size),
                          dtype=np.float32) if use_lgb_preds else None

    # Future feature targets: need features mmap even if not injecting features
    use_future = future_pred_horizon > 0 and future_feature_indices is not None
    future_feat_batch = None
    if use_future:
        n_future = len(future_feature_indices)
        future_feat_batch = np.zeros((batch_size, block_size, n_future), dtype=np.float32)
        if feat_mmap is None:
            feat_file = os.path.join(data_dir, f'{split}_features.bin')
            feat_mmap = np.memmap(feat_file, dtype=np.float16, mode='r').reshape(-1, NUM_FEATURES)

    for b, idx in enumerate(chosen):
        start = bos_offsets[idx]
        # Determine game length (until next BOS or end of file)
        if idx + 1 < len(bos_offsets):
            game_end = bos_offsets[idx + 1]
        else:
            game_end = n
        game_len = game_end - start

        # How many tokens to use (truncate if longer than block_size + 1)
        use_len = min(game_len, block_size + 1)

        chunk = tokens[start:start + use_len].astype(np.int64)
        label_chunk = labels[start:start + use_len].astype(np.int64)

        # x gets tokens[:-1], y gets tokens[1:], wp gets labels[1:]
        seq_len = use_len - 1  # number of positions in x/y
        if seq_len <= 0:
            continue
        actual_len = min(seq_len, block_size)
        x[b, :actual_len] = chunk[:actual_len]
        y[b, :actual_len] = chunk[1:actual_len + 1]
        wp[b, :actual_len] = label_chunk[1:actual_len + 1]

        if feat_mmap is not None:
            feat_batch[b, :actual_len] = feat_mmap[start:start + actual_len].astype(np.float32)
        if pred_mmap is not None:
            pred_batch[b, :actual_len] = pred_mmap[start:start + actual_len].astype(np.float32)

        # Future feature targets: features at position (start + K) within same game
        if use_future:
            K = future_pred_horizon
            # n_valid = positions where start+pos+K is still within this game
            n_valid = max(0, min(actual_len, game_end - start - K))
            if n_valid > 0:
                src = feat_mmap[start + K:start + K + n_valid]
                future_feat_batch[b, :n_valid] = src[:, future_feature_indices].astype(np.float32)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    wp = torch.from_numpy(wp)
    features_t = torch.from_numpy(feat_batch) if feat_batch is not None else None
    lgb_preds_t = torch.from_numpy(pred_batch) if pred_batch is not None else None
    future_features_t = torch.from_numpy(future_feat_batch) if future_feat_batch is not None else None

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        wp = wp.pin_memory().to(device, non_blocking=True)
        if features_t is not None:
            features_t = features_t.pin_memory().to(device, non_blocking=True)
        if lgb_preds_t is not None:
            lgb_preds_t = lgb_preds_t.pin_memory().to(device, non_blocking=True)
        if future_features_t is not None:
            future_features_t = future_features_t.pin_memory().to(device, non_blocking=True)
    else:
        x, y, wp = x.to(device), y.to(device), wp.to(device)
        if features_t is not None:
            features_t = features_t.to(device)
        if lgb_preds_t is not None:
            lgb_preds_t = lgb_preds_t.to(device)
        if future_features_t is not None:
            future_features_t = future_features_t.to(device)

    return x, y, wp, features_t, lgb_preds_t, future_features_t


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Cosine learning rate schedule with warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, device_type,
                  ctx, eval_iters, lw, use_features=False, use_lgb_preds=False,
                  lambda_distill=0.0, zero_tokens=False,
                  future_pred_horizon=0, lambda_future=0.0,
                  future_feature_indices=None):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        total_loss = 0.0
        total_lm = 0.0
        total_wp = 0.0
        total_distill = 0.0
        total_future = 0.0
        total_wp_correct = 0
        total_wp_count = 0
        for k in range(eval_iters):
            x, y, wp, features, lgb_preds, future_features = get_batch(
                split, data_dir, block_size, batch_size,
                device, device_type,
                use_features=use_features, use_lgb_preds=use_lgb_preds,
                future_pred_horizon=future_pred_horizon,
                future_feature_indices=future_feature_indices)
            if zero_tokens:
                x.zero_()
            with ctx:
                logits, wp_logits, loss, details = model(
                    x, y, wp, lambda_wp=lw,
                    features=features, lgb_preds=lgb_preds,
                    lambda_distill=lambda_distill,
                    future_features=future_features,
                    lambda_future=lambda_future)
            total_loss += loss.item()
            total_lm += details['lm_loss']
            total_wp += details['wp_loss']
            if 'distill_loss' in details:
                total_distill += details['distill_loss']
            if 'future_loss' in details:
                total_future += details['future_loss']
            # Win prob accuracy
            wp_mask = (wp != -1)
            if wp_mask.any():
                preds = (torch.sigmoid(wp_logits[wp_mask]) > 0.5).long()
                total_wp_correct += (preds == wp[wp_mask]).sum().item()
                total_wp_count += wp_mask.sum().item()

        n = eval_iters
        out[split] = {
            'loss': total_loss / n,
            'lm_loss': total_lm / n,
            'wp_loss': total_wp / n,
            'wp_acc': total_wp_correct / max(total_wp_count, 1),
        }
        if lambda_distill > 0:
            out[split]['distill_loss'] = total_distill / n
        if lambda_future > 0:
            out[split]['future_loss'] = total_future / n
    model.train()
    return out


def main():
    args = parse_args()

    do_compile = args.compile.lower() in ('true', '1', 'yes')

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in args.device else 'cpu'
    assert device_type == 'cuda' and torch.cuda.is_available(), \
        f"CUDA required but got device={args.device}, cuda.is_available()={torch.cuda.is_available()}"
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }[args.dtype]
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # Verify data exists
    train_bin = os.path.join(args.data_dir, 'train.bin')
    val_bin = os.path.join(args.data_dir, 'val.bin')
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print(f"Error: data files not found in {args.data_dir}")
        print("Run: python -m sequence_model.tokenize_games first")
        sys.exit(1)

    # Verify feature files exist if requested
    if args.use_lgb_features:
        for split in ['train', 'val']:
            feat_bin = os.path.join(args.data_dir, f'{split}_features.bin')
            if not os.path.exists(feat_bin):
                print(f"Error: feature file not found: {feat_bin}")
                print("Run: python -m sequence_model.tokenize_games --materialize first")
                sys.exit(1)
    if args.use_lgb_preds:
        for split in ['train', 'val']:
            pred_bin = os.path.join(args.data_dir, f'{split}_lgb_preds.bin')
            if not os.path.exists(pred_bin):
                print(f"Error: LGB prediction file not found: {pred_bin}")
                print("Run: python -m sequence_model.tokenize_games --materialize --lgb-model ... first")
                sys.exit(1)

    tokens_per_iter = (args.gradient_accumulation_steps * args.batch_size
                       * args.block_size)
    print(f"tokens per iteration: {tokens_per_iter:,}")

    # Model init
    iter_num = 0
    best_val_loss = 1e9

    # Determine future prediction config
    future_feature_indices = None
    if args.future_pred_horizon > 0:
        future_feature_indices = FUTURE_PRED_FEATURE_INDICES

    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, bias=bias, vocab_size=VOCAB_SIZE,
        dropout=args.dropout, model_type=args.model_type,
        n_lgb_features=NUM_FEATURES if args.use_lgb_features else 0,
        lgb_pred_dim=1 if args.use_lgb_preds else 0,
        feature_dropout=args.feature_dropout,
        n_future_features=len(FUTURE_PRED_FEATURE_INDICES) if args.future_pred_horizon > 0 else 0,
    )

    if args.init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = KQModel(gptconf)
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size',
                  'model_type', 'n_lgb_features', 'lgb_pred_dim', 'feature_dropout',
                  'n_future_features']:
            if k in checkpoint['model_args']:
                model_args[k] = checkpoint['model_args'][k]
            # old checkpoints may lack model_type/feature fields — use defaults
        gptconf = GPTConfig(**model_args)
        model = KQModel(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        raise ValueError(f"Unknown init_from: {args.init_from}")

    model.to(args.device)

    # GradScaler for float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.dtype == 'float16'))

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay, args.learning_rate,
        (beta1, beta2), device_type)
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpoint = None

    # Compile
    raw_model = model  # Keep reference to uncompiled model
    if do_compile:
        if args.model_type == 'mamba':
            print("WARNING: mamba sequential scan may cause graph breaks with torch.compile")
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Training loop
    x, y, wp, features, lgb_preds, future_features = get_batch(
        'train', args.data_dir, args.block_size,
        args.batch_size, args.device, device_type,
        use_features=args.use_lgb_features, use_lgb_preds=args.use_lgb_preds,
        future_pred_horizon=args.future_pred_horizon,
        future_feature_indices=future_feature_indices)
    if args.zero_tokens:
        x.zero_()
    t0 = time.time()
    train_start_time = time.time()
    local_iter_num = 0

    # Wandb init
    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("WARNING: --wandb requested but wandb not installed, skipping")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model_type': args.model_type,
                'n_layer': args.n_layer, 'n_head': args.n_head,
                'n_embd': args.n_embd, 'block_size': args.block_size,
                'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                'lambda_wp': args.lambda_wp, 'dropout': args.dropout,
                'max_iters': args.max_iters,
                'warmup_iters': args.warmup_iters,
                'lr_decay_iters': args.lr_decay_iters,
                'min_lr': args.min_lr,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'dtype': args.dtype, 'device': args.device,
                'params_M': raw_model.get_num_params() / 1e6,
                'use_lgb_features': args.use_lgb_features,
                'use_lgb_preds': args.use_lgb_preds,
                'lambda_distill': args.lambda_distill,
                'feature_dropout': args.feature_dropout,
                'future_pred_horizon': args.future_pred_horizon,
                'lambda_future': args.lambda_future,
            },
        )

    print(f"\nStarting training for {args.max_iters} iterations")
    print(f"  model_type = {args.model_type}")
    print(f"  lambda_wp = {args.lambda_wp}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  block_size = {args.block_size}")
    print(f"  warmup_iters = {args.warmup_iters}")
    print(f"  lr_decay_iters = {args.lr_decay_iters}")
    print(f"  min_lr = {args.min_lr}")
    print(f"  device = {args.device}")
    print(f"  dtype = {args.dtype}")
    if args.use_lgb_features or args.use_lgb_preds:
        print(f"  use_lgb_features = {args.use_lgb_features}")
        print(f"  use_lgb_preds = {args.use_lgb_preds}")
        print(f"  lambda_distill = {args.lambda_distill}")
        print(f"  feature_dropout = {args.feature_dropout}")
    if args.future_pred_horizon > 0:
        print(f"  future_pred_horizon = {args.future_pred_horizon}")
        print(f"  lambda_future = {args.lambda_future}")
    print()

    while True:
        # Set learning rate
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters,
                    args.lr_decay_iters, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, args.data_dir, args.block_size, args.batch_size,
                args.device, device_type, ctx, args.eval_iters, args.lambda_wp,
                use_features=args.use_lgb_features,
                use_lgb_preds=args.use_lgb_preds,
                lambda_distill=args.lambda_distill,
                zero_tokens=args.zero_tokens,
                future_pred_horizon=args.future_pred_horizon,
                lambda_future=args.lambda_future,
                future_feature_indices=future_feature_indices)
            distill_str = ''
            if 'distill_loss' in losses.get('train', {}):
                distill_str = (f", distill {losses['train']['distill_loss']:.4f}"
                               f" / {losses['val']['distill_loss']:.4f}")
            future_str = ''
            if 'future_loss' in losses.get('train', {}):
                future_str = (f", future {losses['train']['future_loss']:.4f}"
                              f" / {losses['val']['future_loss']:.4f}")
            print(f"step {iter_num}: "
                  f"train loss {losses['train']['loss']:.4f} "
                  f"(lm {losses['train']['lm_loss']:.4f}, "
                  f"wp {losses['train']['wp_loss']:.4f}, "
                  f"wp_acc {losses['train']['wp_acc']:.3f}{distill_str}{future_str}) | "
                  f"val loss {losses['val']['loss']:.4f} "
                  f"(lm {losses['val']['lm_loss']:.4f}, "
                  f"wp {losses['val']['wp_loss']:.4f}, "
                  f"wp_acc {losses['val']['wp_acc']:.3f})")
            if use_wandb:
                eval_log = {
                    'eval/train_loss': losses['train']['loss'],
                    'eval/train_lm_loss': losses['train']['lm_loss'],
                    'eval/train_wp_loss': losses['train']['wp_loss'],
                    'eval/train_wp_acc': losses['train']['wp_acc'],
                    'eval/val_loss': losses['val']['loss'],
                    'eval/val_lm_loss': losses['val']['lm_loss'],
                    'eval/val_wp_loss': losses['val']['wp_loss'],
                    'eval/val_wp_acc': losses['val']['wp_acc'],
                }
                if 'distill_loss' in losses.get('train', {}):
                    eval_log['eval/train_distill_loss'] = losses['train']['distill_loss']
                    eval_log['eval/val_distill_loss'] = losses['val']['distill_loss']
                if 'future_loss' in losses.get('train', {}):
                    eval_log['eval/train_future_loss'] = losses['train']['future_loss']
                    eval_log['eval/val_future_loss'] = losses['val']['future_loss']
                wandb.log(eval_log, step=iter_num)

            # Save checkpoint
            if (losses['val']['loss'] < best_val_loss or
                    (iter_num > 0 and iter_num % args.save_interval == 0)):
                if losses['val']['loss'] < best_val_loss:
                    best_val_loss = losses['val']['loss']
                if iter_num > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(args),
                    }
                    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
                    print(f"saving checkpoint to {ckpt_path}")
                    torch.save(ckpt, ckpt_path)

        # Forward/backward with gradient accumulation
        for micro_step in range(args.gradient_accumulation_steps):
            with ctx:
                logits, wp_logits, loss, details = model(
                    x, y, wp, lambda_wp=args.lambda_wp,
                    features=features, lgb_preds=lgb_preds,
                    lambda_distill=args.lambda_distill,
                    future_features=future_features,
                    lambda_future=args.lambda_future)
                loss = loss / args.gradient_accumulation_steps

            x, y, wp, features, lgb_preds, future_features = get_batch(
                'train', args.data_dir, args.block_size,
                args.batch_size, args.device, device_type,
                use_features=args.use_lgb_features,
                use_lgb_preds=args.use_lgb_preds,
                future_pred_horizon=args.future_pred_horizon,
                future_feature_indices=future_feature_indices)
            if args.zero_tokens:
                x.zero_()
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0:
            lossf = loss.item() * args.gradient_accumulation_steps
            if details:
                distill_part = ''
                if 'distill_loss' in details:
                    distill_part = f", distill {details['distill_loss']:.4f}"
                future_part = ''
                if 'future_loss' in details:
                    future_part = f", future {details['future_loss']:.4f}"
                print(f"iter {iter_num}: loss {lossf:.4f} "
                      f"(lm {details['lm_loss']:.4f}, wp {details['wp_loss']:.4f}"
                      f"{distill_part}{future_part}), "
                      f"time {dt * 1000:.2f}ms, lr {lr:.2e}")
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, "
                      f"time {dt * 1000:.2f}ms, lr {lr:.2e}")
            if use_wandb:
                log_dict = {
                    'train/loss': lossf,
                    'train/lr': lr,
                    'train/iter_time_ms': dt * 1000,
                }
                if details:
                    log_dict['train/lm_loss'] = details['lm_loss']
                    log_dict['train/wp_loss'] = details['wp_loss']
                    if 'distill_loss' in details:
                        log_dict['train/distill_loss'] = details['distill_loss']
                    if 'future_loss' in details:
                        log_dict['train/future_loss'] = details['future_loss']
                wandb.log(log_dict, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        if iter_num > args.max_iters:
            break

    elapsed = time.time() - train_start_time
    print(f"\nTraining complete. {iter_num} iters in {elapsed:.1f}s. "
          f"Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        wandb.log({'final/best_val_loss': best_val_loss,
                   'final/total_iters': iter_num,
                   'final/elapsed_s': elapsed}, step=iter_num)
        wandb.finish()


if __name__ == '__main__':
    main()
