"""
GPT model for Killer Queen game event sequences with win probability head.

Vendored from nanoGPT (https://github.com/karpathy/nanoGPT) and modified:
- Added win probability head (linear → sigmoid) at every position
- Dual loss: next-token cross-entropy + win-probability binary cross-entropy
- Removed from_pretrained (not needed for domain-specific training)
- Config defaults tuned for our tiny ~2M param model
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Top LGB features by importance (gain):
#   Eggs (36.6%), Food deposits (30.8%), Snail (13.2%), Maidens (5.1%), Berries (1.0%)
# Excludes worker-level features (sort order changes between timesteps) and static map one-hot.
FUTURE_PRED_FEATURE_INDICES = [0, 20, 1, 21, 49, 50, 40, 41, 42, 51]


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (PyTorch doesn't support bias=False directly)."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class CausalLinearAttention(nn.Module):
    """Causal linear attention with ELU+1 feature map.

    Same interface as CausalSelfAttention: (B, T, C) → (B, T, C).
    Uses cumulative sum recurrence for O(T·D²) causal attention instead of O(T²·D).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    @staticmethod
    def _elu_plus_1(x):
        """Non-negative feature map: φ(x) = ELU(x) + 1."""
        return F.elu(x) + 1.0

    def forward(self, x):
        B, T, C = x.size()
        D = C // self.n_head

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, D).transpose(1, 2)  # (B, H, T, D)
        k = k.view(B, T, self.n_head, D).transpose(1, 2)
        v = v.view(B, T, self.n_head, D).transpose(1, 2)

        # Apply feature map
        q = self._elu_plus_1(q)  # (B, H, T, D)
        k = self._elu_plus_1(k)

        # Causal linear attention via cumulative sums
        # S_t = Σ_{i≤t} φ(k_i)^T v_i  →  shape (B, H, T, D, D)
        kv = torch.einsum('bhti,bhtj->bhtij', k, v)  # (B, H, T, D, D)
        S = torch.cumsum(kv, dim=2)  # cumulative key-value outer products

        # z_t = Σ_{i≤t} φ(k_i)  →  shape (B, H, T, D)
        z = torch.cumsum(k, dim=2)

        # output_t = φ(q_t) S_t / (φ(q_t) · z_t)
        num = torch.einsum('bhti,bhtij->bhtj', q, S)  # (B, H, T, D)
        den = torch.einsum('bhti,bhti->bht', q, z).unsqueeze(-1)  # (B, H, T, 1)
        y = num / (den + 1e-6)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


@torch.jit.script
def _fast_ssm_scan(A_bar: torch.Tensor, Bu: torch.Tensor, C_t: torch.Tensor,
                   d_inner: int, d_state: int) -> torch.Tensor:
    """JIT-compiled sequential SSM scan — eliminates Python loop overhead.

    Args:
        A_bar: (B, T, d_inner, d_state) — discretized decay
        Bu: (B, T, d_inner, d_state) — discretized input
        C_t: (B, T, d_state) — output projection per timestep
        d_inner: inner dimension
        d_state: state dimension

    Returns:
        y: (B, T, d_inner) — output at each timestep
    """
    B = A_bar.shape[0]
    T = A_bar.shape[1]
    h = torch.zeros(B, d_inner, d_state, device=A_bar.device, dtype=A_bar.dtype)
    ys = torch.empty(B, T, d_inner, device=A_bar.device, dtype=A_bar.dtype)
    for t in range(T):
        h = A_bar[:, t] * h + Bu[:, t]
        # y_t = sum over state dim: h * C_t  (equivalent to einsum 'bds,bs->bd')
        ys[:, t] = (h * C_t[:, t].unsqueeze(1)).sum(dim=-1)
    return ys


class MambaLayerPure(nn.Module):
    """Pure PyTorch selective SSM (Mamba) — no CUDA kernels needed.

    Uses sequential scan (slow but correct). Architecture follows Mamba:
    input proj → SiLU gate → causal conv1d → selective SSM → gated output → output proj.
    """

    def __init__(self, config, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = config.n_embd
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = self.d_model * expand

        # Input projection: d_model → 2 * d_inner (split into x and gate)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=config.bias)

        # Causal conv1d on x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )

        # SSM parameters (input-dependent, hence "selective")
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)  # dt scalar → per-channel

        # Learnable SSM matrices
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.c_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.size()

        # Input projection and split into x and gate
        xz = self.in_proj(x)  # (B, T, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Causal conv1d on x branch
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_branch = self.conv1d(x_branch)[:, :, :T]  # causal: trim future
        x_branch = x_branch.transpose(1, 2)  # (B, T, d_inner)
        x_branch = F.silu(x_branch)

        # Compute input-dependent SSM parameters
        x_ssm = self.x_proj(x_branch)  # (B, T, d_state*2 + 1)
        B_t = x_ssm[:, :, :self.d_state]  # (B, T, d_state)
        C_t = x_ssm[:, :, self.d_state:2 * self.d_state]  # (B, T, d_state)
        dt = x_ssm[:, :, -1:]  # (B, T, 1)
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(2)  # (B, T, d_inner, d_state)

        # JIT-compiled sequential scan (eliminates Python loop overhead)
        Bu = B_bar * x_branch.unsqueeze(-1)  # (B, T, d_inner, d_state)
        y = _fast_ssm_scan(A_bar, Bu, C_t, self.d_inner, self.d_state)

        # Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch

        # Gate and output
        y = y * F.silu(z)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MambaLayer(nn.Module):
    """Mamba layer using the mamba-ssm package (fast CUDA kernels)."""

    def __init__(self, config):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=config.n_embd, d_state=16, d_conv=4, expand=2)
        self.c_proj = nn.Identity()  # compatibility: _init_weights checks c_proj
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.mamba(x))


def _make_mamba_layer(config):
    """Factory: try mamba-ssm package, fall back to pure PyTorch."""
    try:
        layer = MambaLayer(config)
        return layer
    except ImportError:
        import warnings
        warnings.warn(
            "mamba-ssm not installed, using pure PyTorch Mamba (slow sequential scan). "
            "Install with: pip install mamba-ssm causal-conv1d",
            stacklevel=2,
        )
        return MambaLayerPure(config)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.model_type == 'linear-attn':
            self.attn = CausalLinearAttention(config)
        elif config.model_type == 'mamba':
            self.attn = _make_mamba_layer(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 2560
    vocab_size: int = 185     # KQ game event vocabulary (52 base + 100 snail + 25 count + 8 time-gap)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False
    model_type: str = 'transformer'  # 'transformer' | 'linear-attn' | 'mamba'
    # LightGBM feature injection
    n_lgb_features: int = 0    # 0=disabled, 52=enabled
    lgb_pred_dim: int = 0      # 0=disabled, 1=enabled
    feature_dropout: float = 0.0  # probability of zeroing feature embeddings during training
    # Future game state prediction auxiliary task
    n_future_features: int = 0  # 0=disabled, 10=recommended (len(FUTURE_PRED_FEATURE_INDICES))
    # Context embeddings (hivemind: player/scene/cabinet)
    n_users: int = 0       # 0=disabled, N+1=enabled (N users + unknown)
    n_scenes: int = 0      # 0=disabled, N+1=enabled (N scenes + unknown)
    n_cabinets: int = 0    # 0=disabled, N+1=enabled (N cabinets + unknown)
    context_emb_dim: int = 16      # Embedding dimension for each context table
    context_dropout: float = 0.0   # Prob of zeroing entire context during training


class KQModel(nn.Module):
    """GPT-2 style transformer with an auxiliary win probability head.

    Outputs:
        logits: (B, T, vocab_size) — next-token prediction logits
        wp_logits: (B, T, 1) — win probability logits (pre-sigmoid)
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        # Mamba gets positional info from its recurrence, not learned embeddings
        if config.model_type != 'mamba':
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: embedding weights = lm_head weights
        self.transformer.wte.weight = self.lm_head.weight

        # Win probability head: hidden → 1 (sigmoid applied in loss, not here)
        self.wp_head = nn.Linear(config.n_embd, 1, bias=True)

        # LightGBM feature projection (optional)
        if config.n_lgb_features > 0:
            self.feature_proj = nn.Sequential(
                nn.Linear(config.n_lgb_features, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
            )
        if config.lgb_pred_dim > 0:
            self.lgb_pred_proj = nn.Linear(1, config.n_embd)

        # Context embeddings: player/scene/cabinet (hivemind)
        if config.n_users > 0:
            self.user_emb = nn.Embedding(config.n_users, config.context_emb_dim)
            self.scene_emb = nn.Embedding(config.n_scenes, config.context_emb_dim)
            self.cabinet_emb = nn.Embedding(config.n_cabinets, config.context_emb_dim)
            # Project summed context embedding to n_embd
            self.context_proj = nn.Sequential(
                nn.Linear(config.context_emb_dim, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd),
            )

        # Future game state prediction head (optional)
        if config.n_future_features > 0:
            self.future_head = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_future_features),
            )

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Near-zero init for feature projections so model starts ~= baseline
        if config.n_lgb_features > 0:
            with torch.no_grad():
                self.feature_proj[-1].weight.mul_(0.01)
                self.feature_proj[-1].bias.zero_()
        if config.lgb_pred_dim > 0:
            with torch.no_grad():
                self.lgb_pred_proj.weight.mul_(0.01)
                self.lgb_pred_proj.bias.zero_()
        if config.n_future_features > 0:
            with torch.no_grad():
                self.future_head[-1].weight.mul_(0.01)
                self.future_head[-1].bias.zero_()
        if config.n_users > 0:
            with torch.no_grad():
                self.context_proj[-1].weight.mul_(0.01)
                self.context_proj[-1].bias.zero_()

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, wp_labels=None, lambda_wp=0.1,
                features=None, lgb_preds=None, lambda_distill=0.0,
                future_features=None, lambda_future=0.0, context=None):
        """Forward pass with optional dual loss computation.

        Args:
            idx: (B, T) token indices
            targets: (B, T) next-token targets, -1 for ignore
            wp_labels: (B, T) win probability labels (0 or 1), -1 for ignore
            lambda_wp: weight for win probability loss
            features: (B, T, 52) LightGBM feature vectors, or None
            lgb_preds: (B, T) LightGBM win probability predictions, or None
            lambda_distill: weight for distillation loss (0 = disabled)
            future_features: (B, T, n_future_features) future state targets, or None
            lambda_future: weight for future prediction loss (0 = disabled)
            context: (B, 21) uint16 player/scene/cabinet context, or None

        Returns:
            logits: (B, T, vocab_size)
            wp_logits: (B, T) win probability logits (pre-sigmoid)
            loss: scalar total loss (if targets provided)
            loss_details: dict with 'lm_loss', 'wp_loss', and optionally 'distill_loss', 'future_loss'
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        tok_emb = self.transformer.wte(idx)
        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            tok_emb = tok_emb + self.transformer.wpe(pos)

        # Inject LightGBM features into embedding space
        if features is not None and hasattr(self, 'feature_proj'):
            feat_emb = self.feature_proj(features)
            # Feature dropout: randomly zero feature embeddings during training
            if self.training and self.config.feature_dropout > 0:
                mask = torch.rand(b, 1, 1, device=device) > self.config.feature_dropout
                feat_emb = feat_emb * mask
            tok_emb = tok_emb + feat_emb

        if lgb_preds is not None and hasattr(self, 'lgb_pred_proj'):
            tok_emb = tok_emb + self.lgb_pred_proj(lgb_preds.unsqueeze(-1))

        # Inject context embeddings (player/scene/cabinet)
        if context is not None and hasattr(self, 'user_emb'):
            user_ids = context[:, :10].long()      # (B, 10)
            scene_ids = context[:, 10:20].long()   # (B, 10)
            cab_ids = context[:, 20:21].long()     # (B, 1)

            # Sum embeddings across all 10 positions + cabinet
            ctx_emb = (self.user_emb(user_ids).sum(dim=1) +
                       self.scene_emb(scene_ids).sum(dim=1) +
                       self.cabinet_emb(cab_ids).squeeze(1))  # (B, context_emb_dim)

            ctx_proj = self.context_proj(ctx_emb)  # (B, n_embd)

            # Context dropout: zero entire context embedding per batch element
            if self.training and self.config.context_dropout > 0:
                mask = torch.rand(b, 1, device=device) > self.config.context_dropout
                ctx_proj = ctx_proj * mask

            tok_emb = tok_emb + ctx_proj.unsqueeze(1)  # broadcast over T

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Future game state prediction head (before lm_head detaches from x)
        future_preds = None
        if hasattr(self, 'future_head'):
            future_preds = self.future_head(x)  # (B, T, n_future_features)

        # Next-token prediction head
        logits = self.lm_head(x)
        # Win probability head
        wp_logits = self.wp_head(x).squeeze(-1)  # (B, T)

        loss = None
        loss_details = None

        if targets is not None:
            # Next-token cross-entropy loss
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1)

            # Win probability binary cross-entropy loss
            if wp_labels is not None:
                # Mask out positions where wp_labels == -1
                wp_mask = (wp_labels != -1)
                if wp_mask.any():
                    wp_loss = F.binary_cross_entropy_with_logits(
                        wp_logits[wp_mask],
                        wp_labels[wp_mask].float())
                else:
                    wp_loss = torch.tensor(0.0, device=device)
            else:
                wp_loss = torch.tensor(0.0, device=device)

            loss = lm_loss + lambda_wp * wp_loss
            loss_details = {
                'lm_loss': lm_loss.item(),
                'wp_loss': wp_loss.item(),
            }

            # Distillation loss: MSE between wp_logits and logit(lgb_preds)
            if lambda_distill > 0 and lgb_preds is not None:
                # Mask to positions where lgb_preds != 0 (skip early-game/padding)
                distill_mask = (lgb_preds != 0)
                if distill_mask.any():
                    lgb_p = lgb_preds[distill_mask].float().clamp(1e-7, 1 - 1e-7)
                    lgb_logits = torch.log(lgb_p / (1 - lgb_p))  # logit transform
                    distill_loss = F.mse_loss(wp_logits[distill_mask], lgb_logits)
                else:
                    distill_loss = torch.tensor(0.0, device=device)
                loss = loss + lambda_distill * distill_loss
                loss_details['distill_loss'] = distill_loss.item()

            # Future game state prediction loss
            if lambda_future > 0 and future_features is not None and future_preds is not None:
                # Mask: positions where future_features are all zero are padding/unavailable
                future_mask = (future_features.abs().sum(dim=-1) > 0)  # (B, T)
                if future_mask.any():
                    future_loss = F.mse_loss(
                        future_preds[future_mask], future_features[future_mask])
                else:
                    future_loss = torch.tensor(0.0, device=device)
                loss = loss + lambda_future * future_loss
                loss_details['future_loss'] = future_loss.item()

        return logits, wp_logits, loss, loss_details

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    @torch.no_grad()
    def estimate_win_probability(self, idx, features=None, lgb_preds=None,
                                  context=None):
        """Get win probability estimates at every position.

        Args:
            idx: (B, T) token indices
            features: (B, T, 52) LightGBM feature vectors, or None
            lgb_preds: (B, T) LightGBM predictions, or None
            context: (B, 21) player/scene/cabinet context, or None

        Returns:
            probs: (B, T) P(blue wins) at each position
        """
        _, wp_logits, _, _ = self(idx, features=features, lgb_preds=lgb_preds,
                                   context=context)
        return torch.sigmoid(wp_logits)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation (for sanity checking game dynamics)."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
