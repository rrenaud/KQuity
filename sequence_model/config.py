"""Model and training hyperparameters for the sequence model."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 62       # From vocab.py VOCAB_SIZE
    block_size: int = 1024     # Context length — covers >99% of games
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False         # No bias in linear layers (nanoGPT default for new models)


@dataclass
class TrainConfig:
    # Data
    data_dir: str = 'sequence_model/data'
    train_bin: str = 'train.bin'
    val_bin: str = 'val.bin'
    train_labels_bin: str = 'train_labels.bin'
    val_labels_bin: str = 'val_labels.bin'

    # Training
    batch_size: int = 64
    block_size: int = 1024     # Must match ModelConfig.block_size
    max_iters: int = 50000
    eval_interval: int = 500
    eval_iters: int = 200

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # LR schedule
    warmup_iters: int = 1000
    lr_decay_iters: int = 50000  # Should be ~= max_iters
    min_lr: float = 3e-5         # ~10% of learning_rate

    # Loss weighting
    lambda_wp: float = 0.1      # Win probability loss weight

    # System
    device: str = 'cuda'
    dtype: str = 'bfloat16'     # 'float32', 'bfloat16', 'float16'
    compile: bool = True         # Use torch.compile (PyTorch 2.0+)

    # Logging
    log_interval: int = 10
    save_interval: int = 5000
    out_dir: str = 'sequence_model/out'
