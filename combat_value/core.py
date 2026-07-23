#!/usr/bin/env python3
"""Break-even combat value: state edits, matchup engine, and p* computation.

The decision model (locked design):

  A combat between attacker piece A (attacker's team) and defender piece B
  (opposing team) resolves as *exactly one death* — the defender dies with
  probability p, or the attacker dies with probability 1-p. There are no
  whiffs or trades: p is the share of engagements the attacker survives.

  Let V(.) be the win probability *from the attacker's team's perspective*.
    V(S)      = status-quo value (decline the fight)
    V_kill    = value after defender B dies   (S_kill)
    V_death   = value after attacker A dies    (S_death)

  Taking the fight is worth it iff  p*V_kill + (1-p)*V_death >= V(S), so the
  break-even success probability is

    p* = (V(S) - V_death) / (V_kill - V_death)

  Interpretation: p* is where V(S) sits on the segment [V_death, V_kill].
  Fight iff your true survival probability p exceeds p*.

State edits are the *exact* persistent consequences the win-prob model tracks
(preprocess.PlayerKillEvent.modify_game_state):
  - queen kill  -> that team's `eggs` decrements (2->1->0->-1 == military loss)
  - worker kill -> that worker loses wings/speed/food (warrior -> bare drone),
                   decrementing the team's n_vanilla / n_speed_warrior aggregate.

Feature layout (52 features), per team at offset 0 (blue) / 20 (gold):
  [0] eggs, [1] food_deposits, [2] n_vanilla_warrior, [3] n_speed_warrior
  [4:8]  worker0 (is_bot, has_food, has_speed, has_wings)
  [8:12] worker1, [12:16] worker2, [16:20] worker3
Workers are stored sorted ascending by power (preprocess.vectorize_team).
Global: [40:45] maidens, [45:49] map one-hot, [49:51] snail, [51] berries.

Model orientation: model.predict(X) == P(blue wins) (preprocess label is
1 iff blue won). So V(attacker) = pred if attacker=='blue' else 1-pred.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

NUM_FEATURES = 52

TEAM_OFFSET = {'blue': 0, 'gold': 20}
OPP = {'blue': 'gold', 'gold': 'blue'}

# team-relative aggregate feature indices
EGGS = 0
FOOD_DEPOSITS = 1
N_VANILLA = 2
N_SPEED = 3
WORKER_BASE = 4  # first worker block starts here

# per-worker field offsets within a 4-wide worker block
W_ISBOT, W_FOOD, W_SPEED, W_WINGS = 0, 1, 2, 3

# piece type -> (has_wings, has_speed)
WORKER_PIECES: dict[str, tuple[int, int]] = {
    'drone':           (0, 0),
    'speed_drone':     (0, 1),
    'vanilla_warrior': (1, 0),
    'speed_warrior':   (1, 1),
}
# 'queen' is a valid piece too, handled separately (eggs edit).
ALL_PIECES = tuple(WORKER_PIECES) + ('queen',)

FloatArr = npt.NDArray[np.float32]
BoolArr = npt.NDArray[np.bool_]


# --------------------------------------------------------------------------
# Low-level worker helpers (vectorized over N rows)
# --------------------------------------------------------------------------
def _worker_power(X: FloatArr, team: str) -> FloatArr:
    """(N, 4) power for each worker slot: wings + 0.5*speed + 0.25*food."""
    off = TEAM_OFFSET[team] + WORKER_BASE
    p = np.empty((len(X), 4), np.float32)
    for i in range(4):
        b = off + 4 * i
        p[:, i] = (X[:, b + W_WINGS]
                   + 0.5 * X[:, b + W_SPEED]
                   + 0.25 * X[:, b + W_FOOD])
    return p


def _resort_workers(X: FloatArr, team: str) -> None:
    """Re-sort a team's 4 worker blocks ascending by power, in place.

    Keeps edited states on-distribution: the model was trained on
    power-sorted worker blocks.
    """
    off = TEAM_OFFSET[team] + WORKER_BASE
    power = _worker_power(X, team)
    order = np.argsort(power, axis=1, kind='stable')  # ascending
    blocks = X[:, off:off + 16].reshape(len(X), 4, 4).copy()
    rows = np.arange(len(X))[:, None]
    X[:, off:off + 16] = blocks[rows, order].reshape(len(X), 16)


def worker_type_count(X: FloatArr, team: str, piece: str) -> npt.NDArray[np.int64]:
    """(N,) count of workers of the given piece type on the team."""
    wings, speed = WORKER_PIECES[piece]
    off = TEAM_OFFSET[team] + WORKER_BASE
    cnt = np.zeros(len(X), np.int64)
    for i in range(4):
        b = off + 4 * i
        m = ((X[:, b + W_WINGS] > 0.5) == bool(wings)) & \
            ((X[:, b + W_SPEED] > 0.5) == bool(speed))
        cnt += m
    return cnt


# --------------------------------------------------------------------------
# State edits: each returns (edited_copy, applicable_or_terminal_mask)
# --------------------------------------------------------------------------
def kill_queen(X: FloatArr, team: str) -> tuple[FloatArr, BoolArr]:
    """Kill `team`'s queen: eggs -= 1.

    Returns (X_edited, terminal) where terminal is True for rows whose eggs
    were 0 (now -1 == that team suffers a military loss).
    """
    out = X.copy()
    ei = TEAM_OFFSET[team] + EGGS
    terminal = out[:, ei] <= 0.5  # eggs == 0 -> -1 -> military loss
    out[:, ei] -= 1.0
    return out, terminal


def kill_worker(X: FloatArr, team: str, piece: str) -> tuple[FloatArr, BoolArr]:
    """Kill one worker of `piece` type on `team` (de-wing/de-upgrade to drone).

    Picks the lowest-power matching worker (deterministic). Returns
    (X_edited, applicable) where applicable is False for rows with no such
    worker (left unchanged; caller should mask them out).
    """
    if piece not in WORKER_PIECES:
        raise ValueError(f"kill_worker: {piece!r} is not a worker piece")
    out = X.copy()
    off = TEAM_OFFSET[team] + WORKER_BASE
    wings, speed = WORKER_PIECES[piece]
    N = len(out)

    power_if_match = np.full((N, 4), np.inf, np.float32)
    for i in range(4):
        b = off + 4 * i
        mi = ((out[:, b + W_WINGS] > 0.5) == bool(wings)) & \
             ((out[:, b + W_SPEED] > 0.5) == bool(speed))
        pw = (out[:, b + W_WINGS] + 0.5 * out[:, b + W_SPEED]
              + 0.25 * out[:, b + W_FOOD])
        power_if_match[:, i] = np.where(mi, pw, np.inf)

    applicable = np.isfinite(power_if_match).any(axis=1)
    rows = np.where(applicable)[0]
    if len(rows):
        pick = np.argmin(power_if_match[rows], axis=1)  # lowest-power match
        for f in (W_FOOD, W_SPEED, W_WINGS):
            out[rows, off + 4 * pick + f] = 0.0
        if wings:  # a warrior died -> decrement the team's warrior aggregate
            agg = N_VANILLA if not speed else N_SPEED
            ai = TEAM_OFFSET[team] + agg
            out[rows, ai] = np.maximum(out[rows, ai] - 1.0, 0.0)

    _resort_workers(out, team)
    return out, applicable


def _apply_kill(X: FloatArr, team: str, piece: str) -> tuple[FloatArr, BoolArr, BoolArr]:
    """Dispatch a kill of `piece` on `team`.

    Returns (X_edited, applicable, terminal):
      applicable: rows where the piece exists (worker) / always True (queen)
      terminal:   rows where this kill ends the game (queen at eggs 0)
    """
    if piece == 'queen':
        out, terminal = kill_queen(X, team)
        return out, np.ones(len(X), np.bool_), terminal
    out, applicable = kill_worker(X, team, piece)
    return out, applicable, np.zeros(len(X), np.bool_)


# --------------------------------------------------------------------------
# Win-prob orientation + break-even
# --------------------------------------------------------------------------
SNAIL_POS = 49
BERRIES = 51


def decision_features(X: FloatArr, attacker: str) -> dict[str, npt.NDArray]:
    """Decision-relevant scalars per state, oriented to the attacker.

    Used to bucket states for coarse-grained analysis.
      def_eggs      : defender queen lives remaining (0/1/2)
      atk_eggs      : attacker queen lives remaining
      net_warriors  : attacker warriors minus defender warriors (winged count)
      atk_warriors  : attacker winged count
      berries       : berries still available on the board
      snail_atk     : snail position from attacker's perspective
                      (>0 == snail heading toward the attacker's goal)
    """
    defender = OPP[attacker]
    ao, do = TEAM_OFFSET[attacker], TEAM_OFFSET[defender]
    atk_war = X[:, ao + N_VANILLA] + X[:, ao + N_SPEED]
    def_war = X[:, do + N_VANILLA] + X[:, do + N_SPEED]
    # snail_pos feature is stored blue-oriented (>0 favors gold-goal side per
    # the symmetry mult); flip sign for a gold attacker so + always means
    # "toward the attacker's win".
    snail = X[:, SNAIL_POS] * (1.0 if attacker == 'blue' else -1.0)
    return {
        'def_eggs': X[:, do + EGGS].astype(np.int64),
        'atk_eggs': X[:, ao + EGGS].astype(np.int64),
        'net_warriors': np.rint(atk_war - def_war).astype(np.int64),
        'atk_warriors': np.rint(atk_war).astype(np.int64),
        'berries': np.rint(X[:, BERRIES] * 70.0).astype(np.int64),
        'snail_atk': snail.astype(np.float32),
    }


def orient(pred_blue: FloatArr, attacker: str) -> FloatArr:
    """model.predict == P(blue wins); return P(attacker's team wins)."""
    return pred_blue if attacker == 'blue' else 1.0 - pred_blue


def break_even(vS: FloatArr, vK: FloatArr, vD: FloatArr,
               min_swing: float = 1e-3) -> FloatArr:
    """p* = (vS - vD) / (vK - vD); NaN where the swing |vK-vD| < min_swing.

    Degenerate (near-zero swing) combats barely matter and give unstable p*;
    they are returned as NaN so callers filter rather than plot them.
    """
    swing = vK - vD
    with np.errstate(divide='ignore', invalid='ignore'):
        pstar = (vS - vD) / swing
    pstar = np.where(np.abs(swing) < min_swing, np.nan, pstar)
    return pstar.astype(np.float32)


class MatchupResult:
    """Per-state break-even results for one matchup over N states."""

    def __init__(self, attacker: str, attacker_piece: str, defender_piece: str,
                 vS: FloatArr, vK: FloatArr, vD: FloatArr, pstar: FloatArr,
                 applicable: BoolArr, terminal_kill: BoolArr,
                 terminal_death: BoolArr,
                 feats: dict[str, npt.NDArray] | None = None,
                 game_ids: npt.NDArray | None = None,
                 timestamps: npt.NDArray | None = None):
        self.attacker = attacker
        self.attacker_piece = attacker_piece
        self.defender_piece = defender_piece
        self.vS = vS
        self.vK = vK
        self.vD = vD
        self.pstar = pstar
        self.applicable = applicable
        self.terminal_kill = terminal_kill
        self.terminal_death = terminal_death
        self.feats = feats or {}
        self.game_ids = game_ids
        self.timestamps = timestamps

    def label(self) -> str:
        defender = OPP[self.attacker] if self.attacker in OPP else 'opp'
        return (f"{self.attacker} {self.attacker_piece} "
                f"vs {defender} {self.defender_piece}")


def evaluate_matchup(
    X: FloatArr,
    predict,
    attacker: str,
    attacker_piece: str,
    defender_piece: str,
    game_ids: npt.NDArray | None = None,
    timestamps: npt.NDArray | None = None,
) -> MatchupResult:
    """Compute per-state break-even p* for one matchup.

    Args:
      X: (N, 52) feature matrix of real game states.
      predict: callable X -> P(blue wins), e.g. lgb.Booster.predict.
      attacker: 'blue' or 'gold' (whose piece initiates).
      attacker_piece / defender_piece: element of ALL_PIECES.

    Returns a MatchupResult. Rows where the required pieces are absent have
    applicable=False (their p* is still computed but should be filtered out).
    """
    defender = OPP[attacker]

    X_kill, applic_def, term_kill = _apply_kill(X, defender, defender_piece)
    X_death, applic_atk, term_death = _apply_kill(X, attacker, attacker_piece)
    applicable = applic_def & applic_atk

    # One batched predict for the three state sets.
    N = len(X)
    stacked = np.concatenate([X, X_kill, X_death], axis=0)
    preds = np.asarray(predict(stacked), dtype=np.float32)
    vS = orient(preds[:N], attacker)
    vK = orient(preds[N:2 * N], attacker).copy()
    vD = orient(preds[2 * N:], attacker).copy()

    # Terminal outcomes: model never sees eggs == -1, set exact endpoints.
    vK[term_kill] = 1.0    # defender's last queen died -> attacker wins
    vD[term_death] = 0.0   # attacker's last queen died -> attacker loses

    pstar = break_even(vS, vK, vD)
    return MatchupResult(attacker, attacker_piece, defender_piece,
                         vS, vK, vD, pstar, applicable, term_kill, term_death,
                         feats=decision_features(X, attacker),
                         game_ids=game_ids, timestamps=timestamps)


def evaluate_matchup_both_sides(
    X: FloatArr,
    predict,
    attacker_piece: str,
    defender_piece: str,
    game_ids: npt.NDArray | None = None,
    timestamps: npt.NDArray | None = None,
) -> MatchupResult:
    """Evaluate the matchup with blue-attacking and gold-attacking, stacked.

    Uses the full dataset symmetrically (every state contributes from both
    perspectives), like the two-team accumulation in worker_state_values.
    """
    blue = evaluate_matchup(X, predict, 'blue', attacker_piece, defender_piece,
                            game_ids, timestamps)
    gold = evaluate_matchup(X, predict, 'gold', attacker_piece, defender_piece,
                            game_ids, timestamps)

    def cat(a, b):
        return np.concatenate([a, b], axis=0)

    feats = {k: cat(blue.feats[k], gold.feats[k]) for k in blue.feats}
    gids = cat(blue.game_ids, gold.game_ids) if game_ids is not None else None
    ts = cat(blue.timestamps, gold.timestamps) if timestamps is not None else None
    return MatchupResult(
        'both', attacker_piece, defender_piece,
        cat(blue.vS, gold.vS), cat(blue.vK, gold.vK), cat(blue.vD, gold.vD),
        cat(blue.pstar, gold.pstar), cat(blue.applicable, gold.applicable),
        cat(blue.terminal_kill, gold.terminal_kill),
        cat(blue.terminal_death, gold.terminal_death),
        feats=feats, game_ids=gids, timestamps=ts)


def evaluate_matchups_shared(
    X: FloatArr,
    predict,
    matchups,
    game_ids: npt.NDArray | None = None,
    timestamps: npt.NDArray | None = None,
) -> dict[tuple[str, str], MatchupResult]:
    """Evaluate many matchups (both sides) while sharing model predictions.

    Every matchup needs win-prob for the identity state and for
    "kill piece P on team T" states. Those edits depend only on (P, team), not
    on the matchup, so a piece appearing in several matchups is killed-and-
    predicted once and reused. This returns *bit-for-bit* the same MatchupResult
    that ``evaluate_matchup_both_sides`` produces per matchup, but predicts only
    the unique (identity + one per (piece, team)) state sets instead of
    re-querying the model for every matchup.

    Args:
      matchups: iterable of tuples whose first two items are
        (attacker_piece, defender_piece); any extra items are ignored.

    Returns {(attacker_piece, defender_piece): MatchupResult}.
    """
    N = len(X)
    pairs = [(a, d) for a, d, *_ in matchups]
    pieces = sorted({p for pair in pairs for p in pair})

    # Predict the identity state and each unique (piece, team) kill exactly once.
    pred_id = np.asarray(predict(X), dtype=np.float32)
    edit_pred: dict[tuple[str, str], FloatArr] = {}
    edit_applic: dict[tuple[str, str], BoolArr] = {}
    edit_term: dict[tuple[str, str], BoolArr] = {}
    for piece in pieces:
        for team in ('blue', 'gold'):
            Xe, applic, term = _apply_kill(X, team, piece)
            edit_pred[(piece, team)] = np.asarray(predict(Xe), dtype=np.float32)
            edit_applic[(piece, team)] = applic
            edit_term[(piece, team)] = term
            del Xe

    feats = {'blue': decision_features(X, 'blue'),
             'gold': decision_features(X, 'gold')}

    def one_side(attacker: str, atk_piece: str, dfn_piece: str):
        defender = OPP[attacker]
        term_kill = edit_term[(dfn_piece, defender)]
        term_death = edit_term[(atk_piece, attacker)]
        vS = orient(pred_id, attacker)
        vK = orient(edit_pred[(dfn_piece, defender)], attacker).copy()
        vD = orient(edit_pred[(atk_piece, attacker)], attacker).copy()
        vK[term_kill] = 1.0
        vD[term_death] = 0.0
        applicable = (edit_applic[(dfn_piece, defender)]
                      & edit_applic[(atk_piece, attacker)])
        pstar = break_even(vS, vK, vD)
        return vS, vK, vD, pstar, applicable, term_kill, term_death

    def cat(a, b):
        return np.concatenate([a, b], axis=0)

    results: dict[tuple[str, str], MatchupResult] = {}
    for atk_piece, dfn_piece in pairs:
        b = one_side('blue', atk_piece, dfn_piece)
        g = one_side('gold', atk_piece, dfn_piece)
        mfeats = {k: cat(feats['blue'][k], feats['gold'][k]) for k in feats['blue']}
        gids = cat(game_ids, game_ids) if game_ids is not None else None
        ts = cat(timestamps, timestamps) if timestamps is not None else None
        results[(atk_piece, dfn_piece)] = MatchupResult(
            'both', atk_piece, dfn_piece,
            cat(b[0], g[0]), cat(b[1], g[1]), cat(b[2], g[2]),
            cat(b[3], g[3]), cat(b[4], g[4]), cat(b[5], g[5]), cat(b[6], g[6]),
            feats=mfeats, game_ids=gids, timestamps=ts)
    return results
