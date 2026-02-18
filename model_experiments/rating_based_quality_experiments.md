# Rating-Based Quality Classifier Experiments

**Date**: 2026-02-15

## Background: The Data Quality Problem in Killer Queen

### Why we need a quality classifier

KQuity is a win-probability model for Killer Queen arcade: given a partial game state (berry counts, snail position, kills, etc.), predict P(gold wins). The model is trained on game event streams — timestamped sequences of in-game actions logged by cabinets.

The problem is data quality. Cabinets log *every* game, including:
- Button-check games where someone tests the controls for 10 seconds
- 1v1 or 2v2 games with 6-8 empty positions
- Games where one side is AFK or a small child mashing buttons
- Practice/warmup games that don't reflect real competitive play

Training a win-probability model on this data teaches it to predict the outcome of garbage games, not competitive ones. We need a way to filter down to games that look like real, competitive Killer Queen — but we don't have explicit labels for "this was a real game."

### The labeling trick: logins as a proxy for quality

Killer Queen cabinets support optional player logins. Most casual/random games are fully anonymous. When players log in, it's usually because they're regulars or competing seriously. So games with many logged-in players are likely competitive.

This gives us a noisy but useful labeling scheme:
- **Positive examples**: Games from `logged_in_games/` — games with multiple logged-in players, sorted by login count (most logins first).
- **Negative examples**: Games from `unfiltered_partitioned/` — all games, including the junk. Since the vast majority of games are anonymous/casual, this is mostly negative.
- **Validation**: Tournament games (from organized competitive events) should almost all pass the quality filter. We anchor our thresholds to 99% and 95% tournament recall.

The classifier never sees login counts as a feature — it must learn to distinguish quality from the *gameplay itself*, so it generalizes to anonymous games too.

### Features: 69 per-game summaries

Each game's event stream is collapsed into 69 hand-crafted features that capture what competitive Killer Queen looks like:

- **Basic** (10): Duration, event count, bot count, victory condition (military/economic/snail), map.
- **Action counts** (12): Total kills, queen kills, berry carries/deposits/kick-ins, maiden blesses/uses, snail rides/eats/escapes.
- **Rates** (10): Actions per second — kill rate, carry rate, bless rate, etc. Competitive games have higher action density.
- **Temporal milestones** (7): Time to first kill, first carry, first bless, first maiden use, first snail ride. Real games hit these milestones quickly; button-check games may never.
- **Engagement** (4): Active player count, workers who never touched an objective, max event share (one player doing everything = not a real game), time until the last worker engages.
- **Gate coordination** (2): Minimum time window to bless 3 different maidens — a signature of coordinated play.
- **Per-cabinet activity** (10): Whether each of the 10 cabinet positions generated events. Empty positions signal an incomplete lobby.
- **Per-position maiden use** (10): First maiden use per team and per player position — competitive games have multiple players using maidens.
- **Milestone depth** (4): Time to 6th carry, 3rd/5th/10th bless — games that reach deep milestones are almost certainly real.

### Model and evaluation

The classifier is a LightGBM gradient-boosted tree (127 leaves, min 75 samples/leaf, lr=0.05). Trained on balanced positive/negative sets, validated with a stratified 80/20 split.

**Key evaluation metrics:**
- **AUC**: Discriminative power on the held-out validation set.
- **Logged-in pass rate**: What fraction of logged-in games pass? Should be high, since most logged-in games are genuinely competitive.
- **Unfiltered pass rate @ X% tournament recall**: The primary operational metric. Explained in detail below.

### Understanding the tournament-anchored threshold

We don't have ground-truth labels for "competitive game" across the full population. But we do have 693 tournament games — games from organized competitive events that are unambiguously high quality. These serve as our anchor.

The evaluation works like this:

1. **Score every game.** The classifier assigns a quality score (0 to 1) to every game: all tournament games, all unfiltered games, all logged-in games.

2. **Set the threshold using tournament recall.** We pick a recall target — say 95% — and find the score threshold where exactly 95% of tournament games pass. This gives us a threshold calibrated to real competitive games. At 99% recall, the threshold is more permissive (letting almost all tournament games through); at 95%, it's stricter (allowing a few tournament games to be filtered out in exchange for tighter filtering overall).

3. **Measure the unfiltered pass rate at that threshold.** Now apply that same threshold to the full unfiltered population (~1M games, mostly junk). The fraction that passes is the "unfiltered pass rate." Lower is better — it means the classifier is rejecting more junk while still catching real competitive games.

**Interpreting the numbers:** The best model in these experiments (>=9 logins, 10-seed average) achieved:
- **Unf@99%: 28.7%** — At a threshold where 99% of tournament games pass, 28.7% of all games also pass. We're keeping nearly every real competitive game but also letting through about 29% of the general population. Many of these are probably decent games that just happen to be anonymous.
- **Unf@95%: 20.6%** — At a stricter threshold where 95% of tournament games pass (sacrificing 5% of tournament games), only 20.6% of all games pass. This is a more aggressive filter that produces a cleaner dataset at the cost of slightly more false negatives on tournament-quality games.

**The precision-recall tradeoff:** Think of tournament games as confirmed positives. The 99% threshold prioritizes recall (miss almost no good games) at the cost of precision (more junk sneaks through). The 95% threshold trades 4pp of recall for a meaningful precision improvement — the filtered population shrinks from ~29% to ~21% of all games.

**Why this matters for the downstream win-probability model:** The quality-filtered dataset feeds into training the win-probability model. Every junk game in that dataset adds noise — the model wastes capacity learning to predict outcomes of games that don't resemble real play. Dropping unf@95% from 22% to 20% means the filtered dataset is ~10% cleaner, which compounds when you're training on hundreds of thousands of game states.

### The question these experiments address

The existing classifier uses login count as the sole ordering for positive examples (most logins first). Could we do better by incorporating player *skill ratings*? An OpenSkill (Plackett-Luce) rating system tracks each player's mu across games, with composite (user_id, role) keys for queen vs drone. If we select positive examples from games where the *rated skill* of players is high — not just the count of logins — we might get a purer set of competitive games.

**Hypothesis**: Skill-rated selection should outperform login-count sorting because it captures both "players cared enough to log in" and "the players who logged in are actually good."

**Result**: Ratings didn't help. The only thing that mattered was requiring more logins per game. All rating-based selection criteria performed at or below a simple login-count threshold.

---

## Experimental Setup

- **Positives**: Varied per experiment (different login thresholds, rating filters, and combinations).
- **Negatives**: Always 16K unfiltered games (time-strided shards across the hundreds digit for temporal coverage).
- **Rating system**: OpenSkill Plackett-Luce ratings computed chronologically from `logged_in_games/`. Each of 10 positions gets pre-game mu; anonymous positions default to ~25 mu. Output: `ratings_by_game: {game_id: np.array(10)}`.
- **`anon=min` imputation**: When computing avg_mu for a game, anonymous positions are assigned the minimum mu of the logged-in players in that game (rather than the default ~25).
- **Tournament holdout**: 693 tournament games excluded from all training sets. Used only for threshold calibration.

## Experiment 1: Rating-sorted positives (old cache, shards 000-019)

Initial cache had 18,906 logged-in games (all with >=7 logins from the first 20 shards sorted by login count).

| Variant | Positive N | AUC | Log Loss | Unf@99% | Unf@95% | Log@99% | Log@95% |
|---|---|---|---|---|---|---|---|
| **Baseline** (login sort) | 16,000 | 0.9073 | 0.3439 | 29.0% | 21.9% | 98.8% | 94.6% |
| **HR all logins** (sort by avg_mu) | 16,000 | 0.9094 | 0.3362 | 28.3% | 20.2% | 98.8% | 93.1% |
| **HR >=8, anon=min** | 12,318 | 0.9138 | 0.3405 | 27.9% | 22.8% | 98.2% | 94.7% |
| **HR >=6, anon=min** | 16,000* | 0.9109 | 0.3372 | 26.8% | 20.3% | 98.0% | 92.5% |

*>=6 filter passed all 18,663 games since cache only contained >=7; effectively just changed anon imputation + rating sort.

**Takeaway**: >=8 logins had the best AUC (0.9138), but this cache couldn't distinguish whether ratings or the login threshold drove the improvement.

## Experiment 2: Fair comparison on wider cache (shards 000-039)

Recomputed cache with 36,871 logged-in games (including games with 5-7 logins) for fair comparison.

| Variant | Positive N | AUC | Log Loss | Unf@99% | Unf@95% | Log@99% | Log@95% |
|---|---|---|---|---|---|---|---|
| **Baseline** (login sort) | 16,000 | 0.8865 | 0.3755 | 28.7% | 22.3% | 96.0% | 88.3% |
| **HR >=8, anon=min** | 12,318 | 0.9138 | 0.3405 | 27.9% | 22.8% | 94.8% | 88.6% |
| **HR >=6, anon=min** | 16,000 | 0.9073 | 0.3470 | 29.0% | 20.8% | 96.2% | 85.9% |

**Takeaway**: On the same cache, >=8 clearly beat baseline (+2.7pp AUC). >=6 was middling. But is the improvement from ratings or just requiring more logins?

## Experiment 3: Rating-based union filters

Tried combining login thresholds with queen rating filters to find games with skilled players even if not everyone logged in.

| Variant | Positive N | AUC | Log Loss | Unf@99% | Unf@95% |
|---|---|---|---|---|---|
| Baseline (login sort) | 16,000 | 0.8865 | 0.3755 | 28.7% | 22.3% |
| HR >=8 | 12,318 | 0.9138 | 0.3405 | 27.9% | 22.8% |
| **(>=6 & mu>25) \| (10 logins)** | 11,711 | 0.9093 | 0.3502 | 31.4% | 21.5% |
| **(>=6 & queen>25) \| (10 logins)** | 16,000 | 0.9081 | 0.3460 | 28.4% | 20.0% |
| **(>=6 & queen>30) \| (10 logins)** | 8,556 | 0.9114 | 0.3409 | 29.0% | 21.2% |
| **(>=8) \| (>=6 & queen>30)** | 16,000 | 0.9086 | 0.3451 | 28.7% | 21.7% |

**Takeaway**: Every union variant performed worse than the simple >=8 login filter. Adding games with 6-7 logins — even those with high queen ratings — diluted the positive set. Queen mu > 25 was too loose (most queens are near 25); queen mu > 30 was more selective but the selected games still underperformed pure >=8.

## Experiment 4: Does rating matter at all?

The >=8 login filter produces 12,318 games, which is below the 16K training cap. So **all** >=8 games are used regardless of rating sort order — the rating-based sorting has zero effect.

### Rating distribution analysis (>=8 login games):

| Percentile | avg_mu |
|---|---|
| p5 | 21.68 |
| p25 | 23.68 |
| p50 (median) | 24.69 |
| p75 | 25.74 |
| p95 | 27.28 |
| mean | 24.65 |
| std | 1.75 |

The drop from median to p25 is only **1.0 mu** — about 0.28 sigma per player. Averaging 10 positions compresses individual variation (std=3.59) to std=1.75 across games. The rating system doesn't have enough dynamic range to meaningfully differentiate lobbies within the >=8 login pool.

## Experiment 5: Login threshold is what matters (1K-20K sweep)

Swept positive set size from 1K to 20K in 1K steps, sorted by (login count desc, avg_mu desc).

| N | Min Logins | AUC | Unf@99% | Unf@95% |
|---|---|---|---|---|
| 1,000 | 10 | 0.9373 | 29.2% | 19.8% |
| 2,000 | 10 | 0.9318 | 30.2% | 20.5% |
| 3,000 | 9 | 0.9319 | 29.3% | 20.0% |
| 4,000 | 9 | 0.9283 | 28.8% | 20.7% |
| 5,000 | 9 | 0.9189 | 26.8% | 20.4% |
| 6,000 | 9 | 0.9193 | 28.6% | 20.6% |
| 7,000 | 8 | 0.9176 | 29.0% | 20.4% |
| 8,000 | 8 | 0.9187 | 30.1% | 21.6% |
| 9,000 | 8 | 0.9182 | 29.0% | 21.4% |
| 10,000 | 8 | 0.9153 | 28.7% | 22.3% |
| 11,000 | 8 | 0.9121 | 28.7% | 21.8% |
| 12,000 | 8 | 0.9115 | 29.0% | 23.3% |
| 13,000 | 7 | 0.9120 | 28.0% | 21.3% |
| 14,000 | 7 | 0.9111 | 28.3% | 21.3% |
| 15,000 | 7 | 0.9101 | 29.8% | 22.0% |
| 16,000 | 7 | 0.9088 | 28.7% | 20.8% |
| 17,000 | 7 | 0.9078 | 29.6% | 20.8% |
| 18,000 | 7 | 0.9069 | 29.7% | 21.7% |
| 19,000 | 7 | 0.9025 | 28.1% | 21.8% |
| 20,000 | 7 | 0.9015 | 28.5% | 22.9% |

**Takeaway**: AUC declines monotonically as noisier (fewer-login) games enter the positive set. Unf@99% is flat (~29%) regardless of size. Unf@95% is lowest (~20%) at 1-5K and drifts up. The inflated AUC at small sizes reflects class imbalance, not better generalization. Sweet spot is around 4-6K positives (>=9 logins).

## Experiment 6: Bootstrapping with quality-filtered games

Used 6,272 games with >=9 logins as a base, then added quality-filtered games (from the existing classifier, sorted by quality score desc) in doubling increments.

| QF Added | Total | AUC | Log Loss | Unf@99% | Unf@95% |
|---|---|---|---|---|---|
| 0 | 6,272 | 0.9204 | 0.3050 | 28.8% | 21.6% |
| +1,000 | 7,272 | 0.9234 | 0.3097 | 29.0% | 21.5% |
| +2,000 | 8,272 | 0.9196 | 0.3167 | 28.6% | 20.9% |
| +4,000 | 10,272 | 0.9147 | 0.3268 | 28.8% | 21.7% |
| +8,000 | 14,272 | 0.9234 | 0.3136 | 29.2% | 22.1% |
| +16,000 | 22,272 | 0.9050 | 0.3119 | 29.1% | 22.8% |
| +32,000 | 38,272 | 0.8858 | 0.2735 | 29.9% | 23.3% |
| +64,000 | 70,272 | 0.8969 | 0.1983 | 29.3% | 22.8% |

**Takeaway**: Adding QF games didn't help. AUC degraded past +4K. The +2K result (20.9% unf@95%) looked promising but turned out to be seed noise (see below). Circularity problem: QF games were selected by the current classifier, reinforcing existing biases rather than adding new signal.

## Experiment 7: Seed variance test (10 seeds)

Repeated the base (>=9 logins) vs +2K QF comparison across 10 random seeds to assess whether the +2K improvement was real.

| Config | AUC | Unf@99% | Unf@95% |
|---|---|---|---|
| **Base (>=9 logins)** | 0.9239 +/- 0.0018 | 28.7% +/- 0.6% | **20.6% +/- 0.5%** |
| **+2K quality-filtered** | 0.9235 +/- 0.0020 | 28.6% +/- 0.5% | 21.3% +/- 0.6% |

**Takeaway**: The distributions completely overlap on AUC and unf@99%. On unf@95%, the +2K variant is ~1 std worse. The earlier single-seed +2K result of 20.9% was within noise. Adding quality-filtered games provides no benefit.

## Experiment 8: Self-distillation — pruning low-scoring positives (10 seeds)

The existing quality classifier can identify suspicious games even among >=9 login positives. We scored all 6,272 base games with the existing model and removed the bottom 10% (628 games scoring below 0.129). These are likely button-checks or aborted games where a group of regulars happened to be logged in.

| Config | N | AUC | Unf@99% | Unf@95% |
|---|---|---|---|---|
| **Base (>=9 logins)** | 6,272 | 0.9239 +/- 0.0018 | 28.7% +/- 0.6% | 20.6% +/- 0.5% |
| **Pruned (drop bottom 10%)** | 5,644 | **0.9386 +/- 0.0025** | **28.2% +/- 0.5%** | **20.3% +/- 0.3%** |

**Takeaway**: First real improvement that survived multi-seed validation. AUC jumps +1.5pp with completely non-overlapping distributions (pruned won all 10 seeds). The operational metrics improve modestly but consistently, and variance on unf@95% shrinks from 0.5% to 0.3%. Removing mislabeled positives cleans up the training signal significantly.

## Experiment 9: Adding high-scoring 8-login games (10 seeds)

With a cleaner >=9 base, we tested whether high-confidence 8-login games (as scored by the existing classifier) could add useful training data. Scored all 6,046 8-login games, sorted by quality score descending, and added the top N.

| Config | N | AUC | Unf@99% | Unf@95% |
|---|---|---|---|---|
| Pruned >=9 only | 5,644 | 0.9386 +/- 0.0025 | 28.2% +/- 0.5% | 20.3% +/- 0.3% |
| +500 login8 | 6,144 | 0.9438 +/- 0.0026 | 28.6% +/- 0.9% | 20.5% +/- 0.5% |
| +1000 login8 | 6,644 | 0.9472 +/- 0.0026 | 28.8% +/- 0.6% | 20.5% +/- 0.3% |
| **+2000 login8** | **7,644** | **0.9508 +/- 0.0026** | **28.4% +/- 0.5%** | **20.5% +/- 0.4%** |
| +4000 login8 | 9,644 | 0.9457 +/- 0.0024 | 28.5% +/- 0.8% | 20.5% +/- 0.5% |

8-login quality scores: mean=0.37, p10=0.07, p50=0.33, p90=0.73.

**Takeaway**: +2000 is the sweet spot — AUC peaks at 0.9508 (+1.2pp over pruned-only), completely non-overlapping across seeds. Unf@95% holds flat at ~20.5%. At +4000, AUC drops back as we exhaust the high-confidence 8-login games (median score is only 0.33). Unlike the quality-filtered experiment (Exp 6), this works because we're cherry-picking from a different distribution (8-login games) rather than reinforcing the classifier's own output.

## Experiment 10: Bagged ensemble for free (10 models)

Since we were already training 10 models across seeds for variance estimation, we added 80% bootstrap bagging to each and averaged their predictions as an ensemble.

**Individual bagged models** (mean +/- std over 10 seeds):

| Metric | Individual Models | Ensemble of 10 |
|---|---|---|
| AUC | 0.9478 +/- 0.0029 | — |
| Unf@99% | 28.9% +/- 0.6% | **27.9%** |
| Unf@95% | 21.0% +/- 0.4% | **19.6%** |

The ensemble's averaged predictions beat every single individual model on both operational metrics. The unf@95% of **19.6%** is the best result across all experiments — a 1.4pp improvement over the individual model mean, achieved with zero additional training cost.

**Why it works**: Each bagged model sees a different 80% bootstrap sample of the training data, producing slightly different decision boundaries. Averaging smooths out per-model noise and produces sharper, better-calibrated score distributions. The tournament threshold becomes more precise when computed on averaged scores.

---

## Summary: What worked and what didn't

### Failed approaches (Experiments 1-7)
- **Skill ratings for positive selection**: No meaningful dynamic range. Averaging 10 positions compresses individual variation to std=1.75 across games, with only 1.0 mu separating median from p25.
- **Queen rating filters**: Too loose (mu>25) or too selective (mu>30), neither improved on login-count thresholds.
- **Union filters**: Combining login thresholds with rating filters diluted the positive set.
- **Adding quality-filtered games**: Circularity problem — games selected by the current classifier reinforce existing biases. Multi-seed testing revealed the apparent +2K improvement was seed noise.

### What worked (Experiments 8-10)

The final pipeline that achieved the best results:

1. **Start with >=9 login games** (6,272 games)
2. **Prune bottom 10%** by existing classifier score — removes 628 mislabeled positives (→ 5,644)
3. **Add top 2,000 8-login games** by classifier score — cherry-picks high-confidence games from a noisier pool (→ 7,644)
4. **Train 10 models** with 80% bootstrap bagging, different seeds
5. **Average predictions** across all 10 for the ensemble

**Final ensemble: 27.9% unf@99%, 19.6% unf@95%** (vs baseline: 28.7% unf@99%, 22.3% unf@95%).

### Key lessons

1. **Login count is the dominant positive signal.** Requiring >=9 logins was the single most impactful choice. Ratings added nothing beyond what login count already captured.

2. **Self-distillation works when the classifier is already decent.** Using the existing model to prune bad positives and select good 8-login games both helped, because the model is good enough at spotting junk even if it's imperfect.

3. **More data only helps if it's clean.** Adding 2K cherry-picked 8-login games helped; adding 4K (dipping into lower-quality ones) hurt. Adding quality-filtered games (circular) didn't help at all.

4. **Bagged ensembles are free.** If you're already training multiple models for variance estimation, averaging their predictions gives a strictly better classifier at no cost.

5. **Always validate with multiple seeds.** Several experiments showed apparent 1-2pp improvements that vanished under multi-seed testing (e.g., the +2K QF result in Exp 6). The seed variance on unf@95% is +/- 0.4-0.6%, so any claimed improvement under ~1pp needs multi-seed confirmation.
