# Sequence Model Progression: LLM + Hand-Engineered Features + LGB Teacher

## The Goal

Build a transformer that predicts win probability from raw game event streams, closing the gap with (and eventually surpassing) a LightGBM baseline that uses 52 hand-engineered features.

---

## The LGB Baseline

The LightGBM model is a strong, cheap, and well-tuned baseline. It operates on a fundamentally different representation than the sequence model: at each game event, 52 hand-engineered features are extracted (berry counts, kill counts, snail position, maiden states, etc.), and LGB makes a single binary prediction per state.

**Best config**: 100 leaves, 100 trees, trained on 200K quality-filtered games with symmetry augmentation (~3M training states after 90% subsampling). Training takes ~75 seconds per seed.

**Performance** (10-seed mean on tournament holdout):
- **Accuracy: 69.0%**, Log loss: 0.567, AUC: 0.769
- Well-calibrated (ECE: 0.010), low symmetry deviation (0.013)

**Why it's hard to beat**: The top 3 feature families — eggs/queen kills (36.6% of gain), food deposits (30.8%), and snail position (13.2%) — account for 80.6% of predictive power. These are all *counting* features that LGB gets handed directly. The transformer must reconstruct these counts from raw event streams, which is fundamentally harder for attention-based architectures.

**Scaling behavior**: LGB accuracy plateaus quickly. Going from 1% to 100% of training data (90K → 8.4M states) only improves accuracy from 67.8% to 68.8%. The model is already near its ceiling at modest data sizes, suggesting the features themselves are the bottleneck, not data.

---

## Phase 1: Pure Sequence Model (Feb 5)

**Architecture**: GPT-2 style, 0.8M params (n_embd=128, 4 layers, 4 heads).
**Dual loss**: next-token cross-entropy + win probability BCE (lambda_wp=0.1).

**Result**: 56% on tournament holdout. Barely above chance.

**Problems identified**:
- Tiny training set (887 games from 1 shard)
- Random-offset batching spanning game boundaries
- No time information between events
- LR schedule never decayed (warmup=1000, only 2500 iters)
- Games exceed block_size (p99=1867 vs block_size=1024)

---

## Phase 2: Infrastructure Fixes (Feb 7)

Three critical changes:

1. **Game-aligned batching** — each batch element is one complete game, padded/truncated. Eliminates boundary-spanning.
2. **Time-gap tokens** — 8 empirically-bucketed tokens inserted between events. Boundaries: [0.05s, 0.15s, 0.35s, 0.65s, 1.0s, 1.5s, 2.5s].
3. **Block size 2560** — fits p99 games with time-gap overhead.

Also: snail position buckets (100 linear), LR schedule fix, bulk tokenization.

**Results** (100K games, 4.7M params): **62.5%** on val. Overfitting eliminated (0.6% train/val gap). Val loss still improving at 2K iters — not converged.

---

## Phase 3: Architecture Exploration (Feb 7)

Compared three architectures at matched param counts:
- **Transformer** (softmax attention)
- **Linear attention** (ELU+1 feature map, causal cumsum)
- **Mamba** (selective SSM, pure PyTorch)

Transformer remained the best choice for this scale. Linear attention and Mamba competitive but no clear advantage.

---

## Phase 4: Feature Injection + Distillation

The key insight: the transformer struggles to reconstruct counting features (eggs, food deposits) from raw event streams. These dominate LGB feature importance:
- Eggs (queen deaths): 36.6% of LGB gain
- Food deposits: 30.8%
- Snail position: 13.2%

**Feature injection**: Project the 52D LGB feature vector into embedding space and add to token embeddings. Near-zero init so model starts ≈ baseline.

**Distillation loss**: MSE between transformer wp_logits and logit(lgb_predictions). Masks early-game positions where LGB has no signal (t < 5s).

**Feature dropout**: Randomly zero feature embeddings during training (prob=feature_dropout) to prevent over-reliance.

**Result** (union_20k, 36K train, features+distill, 10K iters): **67% WP accuracy**, val loss 1.1499.

---

## Phase 5: Future Prediction Auxiliary Task

Hypothesis: predicting future game state K tokens ahead forces the model to build better internal representations.

Tested K=5 (~1s), K=15 (~3s) with lambda_future=0.02-0.05.

**Result**: No improvement. WP accuracy flat at ~67% across all configurations.

---

## Phase 6: Scaling Up

Doubled model size to n_embd=192 (1.87M params) with future prediction.

**Result**: Val loss 1.1149 (best overall), but WP accuracy still ~67%.

**Conclusion**: WP accuracy appears to hit a hard ceiling at ~67% regardless of model size or auxiliary tasks. The gap to LGB (69-70%) persists.

---

## Phase 7: Count Tokens + WP Boundary Masking (Current)

**Diagnosis**: The transformer must count events (berry deposits, queen kills) to reconstruct the two most important feature families. This is hard for attention.

**Three changes**:
1. **Count tokens** — insert explicit egg state (3x3=9 tokens) and food state (4x4=16 tokens) after each time-gap. Gives the model the counting features directly in the token stream without needing the feature injection pathway.
2. **WP boundary masking** — only compute WP loss at time-gap positions (one per event group). Reduces noise from sub-event tokens that share identical WP labels.
3. **Higher lambda_distill** — 0.3-0.5 to lean harder on the LGB teacher.

**Data**: 200K QF+LI union, symmetric augmentation → 400K games in data_v2. Token stats: mean=1064 (was ~700), p95=2068, p99=3017.

**Experiments planned**:
- A: count tokens + boundary WP + lambda_distill=0.3
- B: count tokens + boundary WP + lambda_distill=0.5
- C: count tokens + all-position WP + lambda_distill=0.3 (ablation)

---

## Phase 8: Hivemind Context Embeddings (Feb 17)

**Hypothesis**: Adding per-game player/venue identity might break the 67% ceiling. If certain players/venues correlate with play patterns, the model could adjust predictions accordingly.

**Implementation** (`context_lookup.py`, changes to `tokenize_games.py`, `model.py`, `train.py`):
- Per-game context vector of shape (21,): [10 user_ids, 10 scene_ids, 1 cabinet_id]
- Dense index mappings: 1,251 users, 32 scenes, 120 cabinets (index 0 = unknown/anonymous)
- Learned embeddings (dim=16) summed across all 21 slots → 2-layer MLP → added to token embeddings
- Context dropout (default 0.3): randomly zero entire context per batch element
- Near-zero init so model starts ≈ baseline
- Symmetry-aware: `swap_context()` swaps position pairs matching team swap

**Data coverage** (data_v3, 200K QF+LI union with symmetric aug → 400K games):
- 86% of games have at least one user_id entry
- 100% of games have cabinet_id
- Most positions in most games are anonymous (only 2,586/210K games have all 10 positions filled)

**Experiments**:

| Run | Params | Arch | Context | Iters | Val Loss | Val WP Acc | Wall Time |
|-----|--------|------|---------|-------|----------|------------|-----------|
| context_100 | 0.88M | 128d/4L/4H | Yes (drop=0.3) | 100 | 3.718 | 64.7% | 14min |
| context_1k | 0.88M | 128d/4L/4H | Yes (drop=0.3) | 1,000 | 1.149 | 66.2% | 17min |
| big_baseline | 4.85M | 256d/6L/8H | No | 80,000 | 0.713 | 67.1% | 6.3h |
| big_context | 4.94M | 256d/6L/8H | Yes (drop=0.3) | 80,000 | 0.712 | 66.9% | 6.3h |
| xl_context | 14.6M | 384d/8L/8H | Yes (drop=0.3) | 35,000 | 0.710 | 67.3% | 6.0h |

**Result**: Context conditioning provides no meaningful WP accuracy improvement. big_baseline (67.1%) vs big_context (66.9%) is within noise. The xl_context model at 67.3% is the highest observed across all experiments, but this is attributable to model scale, not context — the improvement over big_baseline is only +0.2pp.

**Secondary finding**: Scaling from 0.88M to 14.6M parameters and from 10K to 80K iters drives val loss much lower (1.15 → 0.71), showing the language modeling task continues to benefit from scale, but WP accuracy gains plateau completely.

---

## Phase 7b: 200K Data Scaling (Feb 10-12)

Scaled training data from 20K to 200K games (QF+LI union, symmetric aug → 400K). Tested larger models.

| Run | Params | Arch | Key Args | Val Loss | Val WP Acc |
|-----|--------|------|----------|----------|------------|
| 200k_k3_lam0_embd256 | ~2.5M | 256d/4L | no future pred | 1.126 | 66.9% |
| 200k_k3_lam01_embd256 | ~2.5M | 256d/4L | future_k=3, λ=0.01 | 1.126 | 67.0% |
| 200k_k3_lam01_embd192_8L | ~1.9M | 192d/8L | future_k=3, λ=0.01 | 1.127 | 66.5% |

**Result**: 10x more data did not break the ceiling. 67.0% best, consistent with 20K results.

---

## Summary Table

| Phase | Config | Params | Val WP Acc | Gap to LGB | Notes |
|-------|--------|--------|-----------|------------|-------|
| 1 | Pure seq, 887 games | 0.8M | 56% | -13pp | Broken infrastructure |
| 2 | Fixed batching, 100K games | 4.7M | 62.5% | -6.5pp | +6.5pp from infra fixes |
| 4 | Features+distill, 36K games | 0.8M | 67% | -2pp | +4.5pp from LGB injection |
| 5 | +Future prediction | 0.8M | 67% | -2pp | No improvement |
| 6 | 2x model size | 1.9M | 67% | -2pp | Lower val loss, same acc |
| 7b | 200K data, 256d model | 2.5M | 67% | -2pp | 10x data, no ceiling break |
| 8 | big_baseline (256d/6L) | 4.9M | 67.1% | -1.9pp | 80K iters, 6h training |
| 8 | xl_context (384d/8L) | 14.6M | 67.3% | -1.7pp | Largest model, highest acc |
| 8 | big_context (256d/6L+ctx) | 4.9M | 66.9% | -2.1pp | Context adds nothing |
| - | **LGB baseline** | **-** | **69.0%** | **0** | **52 features, 75s to train** |

The first 11pp of gap (56→67%) closed quickly with infrastructure fixes and feature injection. The last 2pp has resisted every attempt — 6 independent axes of improvement, 18x model scale, 40x training duration, and 30+ GPU-hours of experiments.

## What We've Ruled Out

The 67% WP accuracy ceiling has been tested against:

1. **Model scale**: 0.8M → 14.6M params (18x) — no improvement
2. **Training duration**: 2K → 80K iters (40x) — no improvement
3. **Data scale**: 20K → 200K games (10x) — no improvement
4. **Future prediction auxiliary task**: K=3,5,15 with various λ — no improvement
5. **Player/venue identity (hivemind context)**: 1,251 users, 32 scenes, 120 cabinets — no improvement
6. **Deeper models**: 4L → 8L — no improvement

## Open Questions

1. **Why the 67% ceiling?** The gap vs LGB (69-70%) persists. Hypotheses:
   - The transformer's token-level WP prediction is fundamentally harder than LGB's feature-level prediction
   - The 52D feature injection gives counting information but the model can't fully exploit positional relationships between features the way LGB can
   - WP signal at the token level is noisy — each event in a group gets the same label, diluting gradients
2. **Count tokens (Phase 7)**: Would explicit count tokens in the stream help the model reason about state without the 52D feature crutch? Still untested.
3. **WP boundary masking**: Computing WP loss only at time-gap positions might improve signal quality. Still untested.
4. **Is the LGB prediction target itself noisy?** The distillation loss compresses toward the LGB teacher — if LGB is wrong in systematic ways, the ceiling might be an artifact of the teacher.
