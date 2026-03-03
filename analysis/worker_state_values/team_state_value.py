#!/usr/bin/env python3
"""Team state value function via Bradley-Terry linearization.

Fits f(team_state) -> scalar where f(blue) - f(gold) predicts log-odds
of blue winning. A "team state" is (eggs, n_drone, n_speed_drone,
n_warrior, n_speed_warrior) -- only queen lives and military/speed
upgrades, not berries, maidens, snail, or bot status.

There are 3 * C(7,3) = 105 unique team states. We fit via OLS on
logit-transformed model predictions, using the existing classifier
as a teacher to marginalize over all other game state factors.
"""

import argparse
import json
import os
import sys
from typing import NamedTuple

# Ensure repo root is on the path so we can import event_codec, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import numpy.typing as npt


class TeamState(NamedTuple):
    eggs: int            # 0, 1, 2
    n_drone: int         # 0..4
    n_speed_drone: int   # 0..4 (has_speed, no has_wings)
    n_warrior: int       # 0..4 (has_wings, no has_speed)
    n_speed_warrior: int # 0..4 (has_wings + has_speed)
    # Constraint: n_drone + n_speed_drone + n_warrior + n_speed_warrior == 4


BASELINE_STATE = TeamState(2, 4, 0, 0, 0)


class WorkerState(NamedTuple):
    n_drone: int
    n_speed_drone: int
    n_warrior: int
    n_speed_warrior: int


BASELINE_WORKER_STATE = WorkerState(4, 0, 0, 0)

_CHARACTER_PRIORITY = ['skull', 'abs', 'stripes', 'checkers']


def enumerate_team_states() -> list[TeamState]:
    """Return all 105 canonical team states."""
    states = []
    for eggs in range(3):
        for n_speed_drone in range(5):
            for n_warrior in range(5):
                for n_speed_warrior in range(5):
                    n_drone = 4 - n_speed_drone - n_warrior - n_speed_warrior
                    if n_drone < 0:
                        continue
                    states.append(TeamState(eggs, n_drone, n_speed_drone,
                                           n_warrior, n_speed_warrior))
    return states


def build_state_index() -> dict[TeamState, int]:
    """Build reverse lookup: TeamState -> index."""
    return {s: i for i, s in enumerate(enumerate_team_states())}


def enumerate_worker_states() -> list[WorkerState]:
    """Return all 35 canonical worker states (no eggs)."""
    states = []
    for n_speed_drone in range(5):
        for n_warrior in range(5):
            for n_speed_warrior in range(5):
                n_drone = 4 - n_speed_drone - n_warrior - n_speed_warrior
                if n_drone < 0:
                    continue
                states.append(WorkerState(n_drone, n_speed_drone,
                                          n_warrior, n_speed_warrior))
    return states


def build_worker_state_index() -> dict[WorkerState, int]:
    """Build reverse lookup: WorkerState -> index."""
    return {s: i for i, s in enumerate(enumerate_worker_states())}


def _build_worker_lookup(
    state_index: dict[WorkerState, int],
) -> npt.NDArray[np.int32]:
    """Build (5, 5, 5) lookup: (n_speed_drone, n_warrior, n_speed_warrior) -> index.

    Invalid entries are -1. n_drone is derived as 4 - sum of others.
    """
    lookup = np.full((5, 5, 5), -1, dtype=np.int32)
    for state, idx in state_index.items():
        lookup[state.n_speed_drone,
               state.n_warrior, state.n_speed_warrior] = idx
    return lookup


def extract_worker_state_indices(
    X: npt.NDArray[np.float32],
    lookup: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Extract worker state indices (no eggs) for both teams from feature matrix.

    Same as extract_team_state_indices but ignores the eggs column,
    using the (5,5,5) worker lookup.

    Returns (blue_idx, gold_idx): arrays of shape (N,) with state indices.
    """
    indices = []
    for off in (0, 20):
        n_warrior = X[:, off + 2].astype(np.int32)
        n_speed_warrior = X[:, off + 3].astype(np.int32)

        # Count speed drones: has_speed & ~has_wings per worker
        n_speed_drone = np.zeros(len(X), dtype=np.int32)
        for i in range(4):
            has_speed = X[:, off + 4 + 4 * i + 2] > 0.5
            has_wings = X[:, off + 4 + 4 * i + 3] > 0.5
            n_speed_drone += (has_speed & ~has_wings).astype(np.int32)

        # Clamp to valid ranges
        n_speed_drone = np.clip(n_speed_drone, 0, 4)
        n_warrior = np.clip(n_warrior, 0, 4)
        n_speed_warrior = np.clip(n_speed_warrior, 0, 4)

        idx = lookup[n_speed_drone, n_warrior, n_speed_warrior]
        indices.append(idx)

    return indices[0], indices[1]


def _build_flat_lookup(
    state_index: dict[TeamState, int],
) -> npt.NDArray[np.int32]:
    """Build (3, 5, 5, 5) lookup: (eggs, n_speed_drone, n_warrior, n_speed_warrior) -> index.

    Invalid entries are -1. n_drone is derived as 4 - sum of others.
    """
    lookup = np.full((3, 5, 5, 5), -1, dtype=np.int32)
    for state, idx in state_index.items():
        lookup[state.eggs, state.n_speed_drone,
               state.n_warrior, state.n_speed_warrior] = idx
    return lookup


def extract_team_state_indices(
    X: npt.NDArray[np.float32],
    lookup: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Extract team state indices for both teams from feature matrix.

    Per-team feature layout (20 features at offset 0=blue, 20=gold):
      [0] eggs, [1] food_count, [2] n_vanilla_warrior, [3] n_speed_warrior
      [4:8] worker0 (is_bot, has_food, has_speed, has_wings), ...
      [8:12] worker1, [12:16] worker2, [16:20] worker3

    Returns (blue_idx, gold_idx): arrays of shape (N,) with state indices.
    """
    indices = []
    for off in (0, 20):
        eggs = X[:, off + 0].astype(np.int32)
        n_warrior = X[:, off + 2].astype(np.int32)
        n_speed_warrior = X[:, off + 3].astype(np.int32)

        # Count speed drones: has_speed & ~has_wings per worker
        n_speed_drone = np.zeros(len(X), dtype=np.int32)
        for i in range(4):
            has_speed = X[:, off + 4 + 4 * i + 2] > 0.5
            has_wings = X[:, off + 4 + 4 * i + 3] > 0.5
            n_speed_drone += (has_speed & ~has_wings).astype(np.int32)

        # Clamp to valid ranges
        eggs = np.clip(eggs, 0, 2)
        n_speed_drone = np.clip(n_speed_drone, 0, 4)
        n_warrior = np.clip(n_warrior, 0, 4)
        n_speed_warrior = np.clip(n_speed_warrior, 0, 4)

        idx = lookup[eggs, n_speed_drone, n_warrior, n_speed_warrior]
        indices.append(idx)

    return indices[0], indices[1]


def solve_team_state_values(
    blue_idx: npt.NDArray[np.int32],
    gold_idx: npt.NDArray[np.int32],
    logit_p: npt.NDArray[np.float64],
    n_states: int,
    baseline_idx: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Solve for team state values via normal equations.

    Model: logit_p ~= f(blue_state) - f(gold_state)
    Anchor: f(baseline_state) = 0

    Returns (values, counts): value per state and observation count per state.
    """
    # Filter out rows with invalid indices
    valid = (blue_idx >= 0) & (gold_idx >= 0)
    blue_idx = blue_idx[valid]
    gold_idx = gold_idx[valid]
    logit_p = logit_p[valid]

    # Build normal equations: XtX @ beta = Xty
    # Each row of the design matrix is e_{blue} - e_{gold}
    XtX = np.zeros((n_states, n_states), dtype=np.float64)
    Xty = np.zeros(n_states, dtype=np.float64)

    # Diagonal contributions
    np.add.at(XtX, (blue_idx, blue_idx), 1.0)
    np.add.at(XtX, (gold_idx, gold_idx), 1.0)
    # Off-diagonal cross-terms
    np.add.at(XtX, (blue_idx, gold_idx), -1.0)
    np.add.at(XtX, (gold_idx, blue_idx), -1.0)

    # Right-hand side
    np.add.at(Xty, blue_idx, logit_p)
    np.add.at(Xty, gold_idx, -logit_p)

    # Count observations per state
    counts = np.zeros(n_states, dtype=np.int64)
    np.add.at(counts, blue_idx, 1)
    np.add.at(counts, gold_idx, 1)

    # Drop baseline row/col to anchor f(baseline) = 0
    keep = np.ones(n_states, dtype=bool)
    keep[baseline_idx] = False
    reduced_XtX = XtX[np.ix_(keep, keep)]
    reduced_Xty = Xty[keep]

    # Solve 104x104 system
    reduced_beta = np.linalg.solve(reduced_XtX, reduced_Xty)

    # Reconstruct full beta vector
    values = np.zeros(n_states, dtype=np.float64)
    values[keep] = reduced_beta

    return values, counts


def compute_r_squared(
    blue_idx: npt.NDArray[np.int32],
    gold_idx: npt.NDArray[np.int32],
    logit_p: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
) -> float:
    """Compute R-squared of the team state model on logit(P)."""
    valid = (blue_idx >= 0) & (gold_idx >= 0)
    logit_p = logit_p[valid]
    pred = values[blue_idx[valid]] - values[gold_idx[valid]]
    ss_res = np.sum((logit_p - pred) ** 2)
    ss_tot = np.sum((logit_p - np.mean(logit_p)) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_structured_model(
    states: list,
    values: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    include_eggs: bool = True,
) -> dict:
    """Fit a linear decomposition of f on state values.

    With eggs (TeamState, 4 features):
      f ~ a*(eggs-2) + b*n_warrior + c*n_speed_warrior + d*n_speed_drone
    Without eggs (WorkerState, 3 features):
      f ~ b*n_warrior + c*n_speed_warrior + d*n_speed_drone

    Uses weighted OLS (weighted by observation count).
    Returns coefficients and R-squared.
    """
    n = len(states)
    n_features = 4 if include_eggs else 3
    A = np.zeros((n, n_features), dtype=np.float64)
    for i, s in enumerate(states):
        if include_eggs:
            A[i, 0] = s.eggs - 2
            A[i, 1] = s.n_warrior
            A[i, 2] = s.n_speed_warrior
            A[i, 3] = s.n_speed_drone
        else:
            A[i, 0] = s.n_warrior
            A[i, 1] = s.n_speed_warrior
            A[i, 2] = s.n_speed_drone

    y = values.copy()
    w = counts.astype(np.float64)
    w = np.maximum(w, 1.0)  # avoid zero weights

    # Weighted OLS: (A^T W A) beta = A^T W y
    sqrt_w = np.sqrt(w)
    Aw = A * sqrt_w[:, None]
    yw = y * sqrt_w

    beta = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    pred = A @ beta
    ss_res = np.sum(w * (y - pred) ** 2)
    wmean = np.average(y, weights=w)
    ss_tot = np.sum(w * (y - wmean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot

    result = {
        'warrior_value': beta[1] if include_eggs else beta[0],
        'speed_warrior_value': beta[2] if include_eggs else beta[1],
        'speed_drone_value': beta[3] if include_eggs else beta[2],
        'r_squared': r_squared,
    }
    if include_eggs:
        result['egg_value'] = beta[0]
    return result


def _assign_characters(
    state: WorkerState,
) -> list[tuple[str, bool, bool]]:
    """Map workers to characters by priority.

    Priority: skull > abs > stripes > checkers.
    Upgrade rank: speed_warrior > warrior > speed_drone > drone.
    Best upgrade goes to highest-priority character.

    Returns list of (char_type, is_warrior, is_speed) tuples, length 4.
    """
    upgrades: list[tuple[bool, bool]] = []
    for _ in range(state.n_speed_warrior):
        upgrades.append((True, True))
    for _ in range(state.n_warrior):
        upgrades.append((True, False))
    for _ in range(state.n_speed_drone):
        upgrades.append((False, True))
    for _ in range(state.n_drone):
        upgrades.append((False, False))
    return [
        (char, is_war, is_spd)
        for char, (is_war, is_spd) in zip(_CHARACTER_PRIORITY, upgrades)
    ]


def _compute_transitions(
    worker_states: list[WorkerState],
    values: npt.NDArray[np.float64],
    state_index: dict[WorkerState, int],
) -> list[dict]:
    """Compute valid transitions for each (state, character) pair.

    For each worker in each state, determines what happens on death,
    getting speed, or getting warrior wings, and computes the value delta.

    Returns list of {src, char, actions: [{type, target, delta, win_delta}]}.
    """
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    transitions = []
    for state in worker_states:
        src_idx = state_index[state]
        src_val = values[src_idx]
        src_win = _sigmoid(src_val) * 100.0
        assignments = _assign_characters(state)
        for ci, (_char_type, is_warrior, is_speed) in enumerate(assignments):
            actions = []

            if is_speed and is_warrior:
                # speed_warrior: death -> drone
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone,
                    state.n_warrior, state.n_speed_warrior - 1)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'death', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})
            elif is_warrior:
                # warrior: death -> drone
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone,
                    state.n_warrior - 1, state.n_speed_warrior)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'death', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})
            elif is_speed:
                # speed_drone: death -> drone
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone - 1,
                    state.n_warrior, state.n_speed_warrior)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'death', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})
                # speed_drone: get warrior -> speed_warrior
                target = WorkerState(
                    state.n_drone, state.n_speed_drone - 1,
                    state.n_warrior, state.n_speed_warrior + 1)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'warrior', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})
            else:
                # drone: death is no-op (stays same state), skip
                # drone: get speed -> speed_drone
                target = WorkerState(
                    state.n_drone - 1, state.n_speed_drone + 1,
                    state.n_warrior, state.n_speed_warrior)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'speed', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})
                # drone: get warrior -> warrior
                target = WorkerState(
                    state.n_drone - 1, state.n_speed_drone,
                    state.n_warrior + 1, state.n_speed_warrior)
                tidx = state_index[target]
                tgt_win = _sigmoid(values[tidx]) * 100.0
                actions.append({'type': 'warrior', 'target': tidx,
                                'delta': round(values[tidx] - src_val, 4),
                                'win_delta': round(tgt_win - src_win, 1)})

            if actions:
                transitions.append({
                    'src': src_idx, 'char': ci, 'actions': actions,
                    'src_warrior': is_warrior, 'src_speed': is_speed})
    return transitions


def _print_variant_results(
    variant_name: str,
    states: list,
    state_index: dict,
    baseline_idx: int,
    values: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    r2: float,
    include_eggs: bool,
    verbose: bool = False,
) -> None:
    """Print results for a single variant."""
    print(f"\n{'=' * 70}")
    print(f"  {variant_name}")
    print(f"{'=' * 70}")
    print(f"  R-squared: {r2:.4f} "
          f"({100 * r2:.1f}% of target variance explained)")

    # Structured decomposition
    struct = fit_structured_model(states, values, counts, include_eggs=include_eggs)
    print(f"\n  Structured linear decomposition (R²={struct['r_squared']:.4f}):")
    if include_eggs:
        print(f"    Per queen life:     {struct['egg_value']:+.3f}")
    print(f"    Per warrior:        {struct['warrior_value']:+.3f}")
    print(f"    Per speed warrior:  {struct['speed_warrior_value']:+.3f}")
    print(f"    Per speed drone:    {struct['speed_drone_value']:+.3f}")

    # Interpretable deltas
    print(f"\n  --- Interpretable deltas ---")

    if include_eggs:
        for e in range(2):
            s0 = TeamState(e, 4, 0, 0, 0)
            s1 = TeamState(e + 1, 4, 0, 0, 0)
            delta = values[state_index[s1]] - values[state_index[s0]]
            print(f"    Queen life (eggs {e}->{e + 1}, all drones): {delta:+.3f}")
        print()

    # Warrior upgrades
    for nw in range(4):
        if include_eggs:
            s_before = TeamState(2, 4 - nw, 0, nw, 0)
            s_after = TeamState(2, 3 - nw, 0, nw + 1, 0)
        else:
            s_before = WorkerState(4 - nw, 0, nw, 0)
            s_after = WorkerState(3 - nw, 0, nw + 1, 0)
        delta = values[state_index[s_after]] - values[state_index[s_before]]
        eggs_note = ", eggs=2" if include_eggs else ""
        print(f"    Warrior #{nw + 1} (drone->warrior{eggs_note}): {delta:+.3f}")

    # Speed on warrior
    print()
    for nsw in range(4):
        nw = 1
        nd = 4 - nw - nsw
        if nd < 0:
            break
        if include_eggs:
            s_warrior = TeamState(2, nd, 0, 1, nsw)
            s_speed = TeamState(2, nd, 0, 0, nsw + 1)
        else:
            s_warrior = WorkerState(nd, 0, 1, nsw)
            s_speed = WorkerState(nd, 0, 0, nsw + 1)
        delta = values[state_index[s_speed]] - values[state_index[s_warrior]]
        print(f"    Speed on warrior (war->sw, {nsw + 1} total upgraded): "
              f"{delta:+.3f}")

    # Speed on drone
    print()
    for nsd in range(4):
        nd = 4 - nsd
        if include_eggs:
            s_before = TeamState(2, nd, nsd, 0, 0)
            s_after = TeamState(2, nd - 1, nsd + 1, 0, 0)
        else:
            s_before = WorkerState(nd, nsd, 0, 0)
            s_after = WorkerState(nd - 1, nsd + 1, 0, 0)
        delta = values[state_index[s_after]] - values[state_index[s_before]]
        print(f"    Speed drone #{nsd + 1} (drone->sd): {delta:+.3f}")

    # Warrior vs speed drone
    print()
    if include_eggs:
        s_warrior = TeamState(2, 3, 0, 1, 0)
        s_sd = TeamState(2, 3, 1, 0, 0)
    else:
        s_warrior = WorkerState(3, 0, 1, 0)
        s_sd = WorkerState(3, 1, 0, 0)
    delta = values[state_index[s_warrior]] - values[state_index[s_sd]]
    print(f"    Warrior vs speed drone (1 upgrade): {delta:+.3f} "
          f"({'warrior' if delta > 0 else 'speed drone'} better)")

    # Full table (only for verbose or primary variant)
    if verbose:
        print(f"\n  {'State':<45} {'Value':>8} {'Count':>10}")
        print(f"  {'-' * 65}")
        sorted_indices = np.argsort(values)
        for i in sorted_indices:
            s = states[i]
            if include_eggs:
                label = (f"eggs={s.eggs} drone={s.n_drone} sd={s.n_speed_drone} "
                         f"war={s.n_warrior} sw={s.n_speed_warrior}")
            else:
                label = (f"drone={s.n_drone} sd={s.n_speed_drone} "
                         f"war={s.n_warrior} sw={s.n_speed_warrior}")
            marker = " *" if i == baseline_idx else ""
            print(f"    {label:<43} {values[i]:>8.3f} {counts[i]:>10,}{marker}")


def save_worker_stats_json(
    worker_states: list[WorkerState],
    values: npt.NDArray[np.float64],
    counts: npt.NDArray[np.int64],
    baseline_idx: int,
    path: str,
    metadata: dict | None = None,
) -> None:
    """Save worker state values and counts to JSON."""
    data: dict = {
        'states': [
            {'n_drone': s.n_drone, 'n_speed_drone': s.n_speed_drone,
             'n_warrior': s.n_warrior, 'n_speed_warrior': s.n_speed_warrior}
            for s in worker_states
        ],
        'values': values.tolist(),
        'counts': counts.tolist(),
        'baseline_idx': baseline_idx,
    }
    if metadata:
        data['metadata'] = metadata
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Worker state stats written to {path}")


def load_worker_stats_json(
    path: str,
) -> tuple[list[WorkerState], npt.NDArray[np.float64], npt.NDArray[np.int64], int]:
    """Load worker state values and counts from JSON."""
    with open(path) as f:
        data = json.load(f)
    states = [WorkerState(**s) for s in data['states']]
    values = np.array(data['values'], dtype=np.float64)
    counts = np.array(data['counts'], dtype=np.int64)
    return states, values, counts, data['baseline_idx']


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Team state value via Bradley-Terry linearization')
    parser.add_argument('--data', default='quality_filtered/encoded/all_games.bin',
                        help='Path to binary data file')
    parser.add_argument('--model', default='current_preferred_model.mdl',
                        help='Path to LightGBM model file')
    parser.add_argument('--max-games', type=int, default=50000,
                        help='Maximum number of games to load')
    parser.add_argument('--drop-prob', type=float, default=0.9,
                        help='Probability of dropping each state (subsampling)')
    parser.add_argument('--clip', type=float, default=0.01,
                        help='Clip predictions to [clip, 1-clip] before logit')
    parser.add_argument('--verbose', action='store_true',
                        help='Print full state tables for all variants')
    parser.add_argument('--html', type=str, nargs='?',
                        const='worker_state_values.html',
                        help='Generate HTML visualization (default: worker_state_values.html)')
    parser.add_argument('--json', type=str,
                        help='Save worker state stats to JSON file')
    parser.add_argument('--from-json', type=str,
                        help='Generate HTML from pre-computed JSON (skips computation)')
    args = parser.parse_args()

    # --- Fast path: generate HTML from pre-computed JSON ---
    if args.from_json:
        from analysis.worker_state_values.html_visualization import generate_html
        states, values, counts, baseline_idx = load_worker_stats_json(
            args.from_json)
        html_path = args.html or 'worker_state_values.html'
        generate_html(states, values, counts, baseline_idx, html_path)
        return

    # --- Load data ---
    from event_codec import fast_materialize_from_codec
    import lightgbm as lgb

    print(f"Loading data from {args.data}...")
    X, y, game_ids, timestamps = fast_materialize_from_codec(
        args.data,
        drop_state_probability=args.drop_prob,
        max_games=args.max_games,
    )
    print(f"  {X.shape[0]:,} snapshots from "
          f"{len(np.unique(game_ids)):,} games")

    # --- Model predictions ---
    print(f"\nLoading model from {args.model}...")
    model = lgb.Booster(model_file=args.model)

    p_blue = model.predict(X)
    p_clipped = np.clip(p_blue, args.clip, 1.0 - args.clip)
    logit_p = np.log(p_clipped / (1.0 - p_clipped))

    # --- Build state machinery (with eggs: 105 states) ---
    team_states = enumerate_team_states()
    assert len(team_states) == 105
    team_state_index = build_state_index()
    team_baseline_idx = team_state_index[BASELINE_STATE]
    team_lookup = _build_flat_lookup(team_state_index)

    # --- Build state machinery (without eggs: 35 states) ---
    worker_states = enumerate_worker_states()
    assert len(worker_states) == 35
    worker_state_index = build_worker_state_index()
    worker_baseline_idx = worker_state_index[BASELINE_WORKER_STATE]
    worker_lookup = _build_worker_lookup(worker_state_index)

    # --- Extract state indices ---
    print("\nExtracting state indices...")
    team_blue, team_gold = extract_team_state_indices(X, team_lookup)
    worker_blue, worker_gold = extract_worker_state_indices(X, worker_lookup)

    # --- 2x1 variant grid (with/without eggs, classifier logit) ---
    variants = [
        ("With eggs (105) + Classifier logit", True,
         team_blue, team_gold),
        ("Without eggs (35) + Classifier logit", False,
         worker_blue, worker_gold),
    ]

    worker_values = None
    worker_counts = None

    for variant_name, include_eggs, blue_idx, gold_idx in variants:
        if include_eggs:
            states = team_states
            state_index = team_state_index
            baseline_idx = team_baseline_idx
        else:
            states = worker_states
            state_index = worker_state_index
            baseline_idx = worker_baseline_idx

        values, counts = solve_team_state_values(
            blue_idx, gold_idx, logit_p, len(states), baseline_idx)

        r2 = compute_r_squared(blue_idx, gold_idx, logit_p, values)

        if not include_eggs:
            worker_values = values
            worker_counts = counts

        show_verbose = include_eggs or args.verbose
        _print_variant_results(
            variant_name, states, state_index, baseline_idx,
            values, counts, r2, include_eggs, verbose=show_verbose)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("  Summary: R-squared by variant")
    print(f"{'=' * 70}")
    for variant_name, include_eggs, blue_idx, gold_idx in variants:
        if include_eggs:
            states, state_index, baseline_idx = team_states, team_state_index, team_baseline_idx
        else:
            states, state_index, baseline_idx = worker_states, worker_state_index, worker_baseline_idx
        values, counts = solve_team_state_values(
            blue_idx, gold_idx, logit_p, len(states), baseline_idx)
        r2 = compute_r_squared(blue_idx, gold_idx, logit_p, values)
        print(f"  {variant_name:<45s} R²={r2:.4f}")

    # --- Save JSON ---
    if args.json and worker_values is not None:
        save_worker_stats_json(
            worker_states, worker_values, worker_counts,
            worker_baseline_idx, args.json,
            metadata={
                'dataset': args.data,
                'model': args.model,
                'max_games': args.max_games,
                'drop_prob': args.drop_prob,
                'n_snapshots': int(X.shape[0]),
                'n_games': int(len(np.unique(game_ids))),
            })

    # --- HTML visualization ---
    if args.html and worker_values is not None:
        from analysis.worker_state_values.html_visualization import generate_html
        generate_html(worker_states, worker_values, worker_counts,
                      worker_baseline_idx, args.html)


if __name__ == '__main__':
    main()
