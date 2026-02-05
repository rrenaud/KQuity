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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sequence_model.model import KQModel, GPTConfig
from sequence_model.vocab import VOCAB_SIZE, PAD

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
block_size = 1024
dropout = 0.1
bias = False
vocab_size = VOCAB_SIZE

# Optimizer
learning_rate = 3e-4
max_iters = 50000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
warmup_iters = 1000
lr_decay_iters = 50000
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
    parser.add_argument('--max-seconds', type=float, default=None,
                        help='Stop training after this many seconds')
    return parser.parse_args()


def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    """Load a random batch of token sequences and their win-probability labels.

    The token file and label file are parallel arrays of the same length.
    We sample random starting positions and extract (block_size) chunks.

    Returns:
        x: (B, T) input tokens
        y: (B, T) next-token targets
        wp: (B, T) win probability labels for each position in y
    """
    # Recreate memmap each call to avoid memory leak
    token_file = os.path.join(data_dir, f'{split}.bin')
    label_file = os.path.join(data_dir, f'{split}_labels.bin')

    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
    labels = np.memmap(label_file, dtype=np.uint8, mode='r')

    ix = torch.randint(len(tokens) - block_size - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy(tokens[i:i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(tokens[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    # Win prob labels aligned with the *target* positions (shifted by 1)
    wp = torch.stack([
        torch.from_numpy(labels[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix
    ])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        wp = wp.pin_memory().to(device, non_blocking=True)
    else:
        x, y, wp = x.to(device), y.to(device), wp.to(device)

    return x, y, wp


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
                  ctx, eval_iters, lw):
    """Estimate loss on train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        total_loss = 0.0
        total_lm = 0.0
        total_wp = 0.0
        total_wp_correct = 0
        total_wp_count = 0
        for k in range(eval_iters):
            x, y, wp = get_batch(split, data_dir, block_size, batch_size,
                                 device, device_type)
            with ctx:
                logits, wp_logits, loss, details = model(x, y, wp, lambda_wp=lw)
            total_loss += loss.item()
            total_lm += details['lm_loss']
            total_wp += details['wp_loss']
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

    tokens_per_iter = (args.gradient_accumulation_steps * args.batch_size
                       * args.block_size)
    print(f"tokens per iteration: {tokens_per_iter:,}")

    # Model init
    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, bias=bias, vocab_size=VOCAB_SIZE,
        dropout=args.dropout,
    )

    if args.init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = KQModel(gptconf)
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint['model_args'][k]
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
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # Training loop
    x, y, wp = get_batch('train', args.data_dir, args.block_size,
                         args.batch_size, args.device, device_type)
    t0 = time.time()
    train_start_time = time.time()
    local_iter_num = 0

    print(f"\nStarting training for {args.max_iters} iterations"
          + (f" or {args.max_seconds}s" if args.max_seconds else ""))
    print(f"  lambda_wp = {args.lambda_wp}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  block_size = {args.block_size}")
    print(f"  device = {args.device}")
    print(f"  dtype = {args.dtype}")
    print()

    while True:
        # Set learning rate
        lr = get_lr(iter_num, args.learning_rate, warmup_iters,
                    lr_decay_iters, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(
                model, args.data_dir, args.block_size, args.batch_size,
                args.device, device_type, ctx, args.eval_iters, args.lambda_wp)
            print(f"step {iter_num}: "
                  f"train loss {losses['train']['loss']:.4f} "
                  f"(lm {losses['train']['lm_loss']:.4f}, "
                  f"wp {losses['train']['wp_loss']:.4f}, "
                  f"wp_acc {losses['train']['wp_acc']:.3f}) | "
                  f"val loss {losses['val']['loss']:.4f} "
                  f"(lm {losses['val']['lm_loss']:.4f}, "
                  f"wp {losses['val']['wp_loss']:.4f}, "
                  f"wp_acc {losses['val']['wp_acc']:.3f})")

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
                    x, y, wp, lambda_wp=args.lambda_wp)
                loss = loss / args.gradient_accumulation_steps

            x, y, wp = get_batch('train', args.data_dir, args.block_size,
                                 args.batch_size, args.device, device_type)
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
                print(f"iter {iter_num}: loss {lossf:.4f} "
                      f"(lm {details['lm_loss']:.4f}, wp {details['wp_loss']:.4f}), "
                      f"time {dt * 1000:.2f}ms, lr {lr:.2e}")
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, "
                      f"time {dt * 1000:.2f}ms, lr {lr:.2e}")

        iter_num += 1
        local_iter_num += 1

        if iter_num > args.max_iters:
            break
        if args.max_seconds and (time.time() - train_start_time) > args.max_seconds:
            print(f"Time limit reached ({args.max_seconds}s)")
            break

    elapsed = time.time() - train_start_time
    print(f"\nTraining complete. {iter_num} iters in {elapsed:.1f}s. "
          f"Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
