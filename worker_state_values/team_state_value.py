#!/usr/bin/env python3
"""Worker state value analysis.

Computes empirical win probability per worker state by averaging actual game
outcomes over all observations where each state appeared, marginalizing over
opponents and all other game factors.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
from typing import NamedTuple

# Ensure repo root is on the path so we can import event_codec, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import numpy.typing as npt


class TeamState(NamedTuple):
    eggs: int            # 0, 1, 2
    n_drone: int         # 0..4
    n_speed_drone: int   # 0..4 (has_speed, no has_wings)
    n_warrior: int       # 0..4 (has_wings, no has_speed)
    n_speed_warrior: int # 0..4 (has_wings + has_speed)
    # Constraint: n_drone + n_speed_drone + n_warrior + n_speed_warrior == 4


class WorkerState(NamedTuple):
    n_drone: int
    n_speed_drone: int
    n_warrior: int
    n_speed_warrior: int


CHARACTER_PRIORITY = ['skull', 'abs', 'stripes', 'checkers']


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


MAP_NAMES = ['day', 'night', 'dusk', 'twilight']
MAP_FEATURE_OFFSET = 45  # X[:, 45:49] = map one-hot

BATCH_SIZE = 2000  # games per parallel batch

WEIGHTING_MODES = ('naive', 'time', 'unique')

# Per-worker state (set once by _init_worker, reused across tasks)
_worker_lookup: npt.NDArray[np.int32] | None = None
_worker_n_states: int = 0
_worker_drop_prob: float = 0.0


def _init_worker(
    worker_lookup: npt.NDArray[np.int32],
    n_states: int,
    drop_prob: float,
) -> None:
    """Initialize worker process state."""
    global _worker_lookup, _worker_n_states, _worker_drop_prob
    _worker_lookup = worker_lookup
    _worker_n_states = n_states
    _worker_drop_prob = drop_prob


def _make_empty_accum(
    n_states: int,
) -> tuple[
    npt.NDArray[np.float64],          # win_totals
    npt.NDArray[np.float64],          # win_ns (sum of weights)
    list[npt.NDArray[np.float64]],    # map_win_totals[4]
    list[npt.NDArray[np.float64]],    # map_win_ns[4]
]:
    """Create zeroed accumulator arrays for one weighting mode."""
    return (
        np.zeros(n_states, dtype=np.float64),
        np.zeros(n_states, dtype=np.float64),
        [np.zeros(n_states, dtype=np.float64) for _ in range(4)],
        [np.zeros(n_states, dtype=np.float64) for _ in range(4)],
    )


def _accumulate_weighted(
    accum: tuple,
    worker_blue: npt.NDArray[np.int32],
    worker_gold: npt.NDArray[np.int32],
    y_f: npt.NDArray[np.float64],
    valid: npt.NDArray[np.bool_],
    map_ids: npt.NDArray,
    blue_weights: npt.NDArray[np.float64],
    gold_weights: npt.NDArray[np.float64],
) -> None:
    """Add weighted observations into accumulator arrays (in-place)."""
    win_totals, win_ns, map_win_totals, map_win_ns = accum

    blue_mask = valid & (blue_weights > 0)
    gold_mask = valid & (gold_weights > 0)

    bw = blue_weights[blue_mask]
    gw = gold_weights[gold_mask]

    np.add.at(win_totals, worker_blue[blue_mask], y_f[blue_mask] * bw)
    np.add.at(win_ns, worker_blue[blue_mask], bw)
    np.add.at(win_totals, worker_gold[gold_mask], (1.0 - y_f[gold_mask]) * gw)
    np.add.at(win_ns, worker_gold[gold_mask], gw)

    for m_idx in range(4):
        bm = blue_mask & (map_ids == m_idx)
        gm = gold_mask & (map_ids == m_idx)
        bw_m = blue_weights[bm]
        gw_m = gold_weights[gm]
        np.add.at(map_win_totals[m_idx], worker_blue[bm], y_f[bm] * bw_m)
        np.add.at(map_win_ns[m_idx], worker_blue[bm], bw_m)
        np.add.at(map_win_totals[m_idx], worker_gold[gm], (1.0 - y_f[gm]) * gw_m)
        np.add.at(map_win_ns[m_idx], worker_gold[gm], gw_m)


# BatchResult: dict mapping mode name -> (win_totals, win_ns, map_win_totals, map_win_ns)
# plus an int n_snapshots
BatchResult = tuple[dict[str, tuple], int]


def _accumulate_batch(
    encoded_games: list[tuple[int, bytes]],
) -> BatchResult:
    """Materialize + accumulate empirical win stats for one batch of games.

    Computes all three weighting modes (naive, time, unique) simultaneously.
    Runs entirely in a worker process.
    """
    from event_codec import materialize_entries
    n_states = _worker_n_states

    X, y, gids, ts = materialize_entries(
        encoded_games, drop_state_probability=_worker_drop_prob)

    accums = {mode: _make_empty_accum(n_states) for mode in WEIGHTING_MODES}

    if len(X) == 0:
        return accums, 0

    y_f = y.astype(np.float64)
    worker_blue, worker_gold = extract_worker_state_indices(X, _worker_lookup)
    valid = (worker_blue >= 0) & (worker_gold >= 0)
    map_ids = np.argmax(X[:, MAP_FEATURE_OFFSET:MAP_FEATURE_OFFSET + 4], axis=1)

    n = len(X)

    # --- Naive: uniform weight = 1 ---
    ones = np.ones(n, dtype=np.float64)
    _accumulate_weighted(accums['naive'], worker_blue, worker_gold,
                         y_f, valid, map_ids, ones, ones)

    # --- Detect game boundaries (needed for time and unique) ---
    game_boundary = np.empty(n, dtype=np.bool_)
    game_boundary[0] = True
    game_boundary[1:] = gids[1:] != gids[:-1]

    # --- Time weights: duration until next event, 0 at end of game ---
    time_w = np.zeros(n, dtype=np.float64)
    time_w[:-1] = ts[1:].astype(np.float64) - ts[:-1].astype(np.float64)
    # Zero out cross-game durations
    next_is_boundary = np.empty(n, dtype=np.bool_)
    next_is_boundary[-1] = True
    next_is_boundary[:-1] = game_boundary[1:]
    time_w[next_is_boundary] = 0.0
    np.maximum(time_w, 0.0, out=time_w)

    _accumulate_weighted(accums['time'], worker_blue, worker_gold,
                         y_f, valid, map_ids, time_w, time_w)

    # --- Unique weights: 1 only when team's state changed (or first in game) ---
    blue_changed = np.ones(n, dtype=np.float64)
    gold_changed = np.ones(n, dtype=np.float64)
    blue_changed[1:] = np.where(
        game_boundary[1:] | (worker_blue[1:] != worker_blue[:-1]), 1.0, 0.0)
    gold_changed[1:] = np.where(
        game_boundary[1:] | (worker_gold[1:] != worker_gold[:-1]), 1.0, 0.0)

    _accumulate_weighted(accums['unique'], worker_blue, worker_gold,
                         y_f, valid, map_ids, blue_changed, gold_changed)

    return accums, len(X)


def assign_characters(
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
        for char, (is_war, is_spd) in zip(CHARACTER_PRIORITY, upgrades)
    ]


def compute_transitions(
    worker_states: list[WorkerState],
    win_probs: npt.NDArray[np.float64],
    state_index: dict[WorkerState, int],
) -> list[dict]:
    """Compute valid transitions for each (state, character) pair.

    For each worker in each state, determines what happens on death,
    getting speed, or getting warrior wings, and computes the win% delta.

    Returns list of {src, char, actions: [{type, target, win_delta}]}.
    """
    transitions = []
    for state in worker_states:
        src_idx = state_index[state]
        src_win = win_probs[src_idx] * 100.0
        assignments = assign_characters(state)
        for ci, (_char_type, is_warrior, is_speed) in enumerate(assignments):
            actions = []

            if is_speed and is_warrior:
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone,
                    state.n_warrior, state.n_speed_warrior - 1)
                tidx = state_index[target]
                actions.append({'type': 'death', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})
            elif is_warrior:
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone,
                    state.n_warrior - 1, state.n_speed_warrior)
                tidx = state_index[target]
                actions.append({'type': 'death', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})
            elif is_speed:
                target = WorkerState(
                    state.n_drone + 1, state.n_speed_drone - 1,
                    state.n_warrior, state.n_speed_warrior)
                tidx = state_index[target]
                actions.append({'type': 'death', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})
                target = WorkerState(
                    state.n_drone, state.n_speed_drone - 1,
                    state.n_warrior, state.n_speed_warrior + 1)
                tidx = state_index[target]
                actions.append({'type': 'warrior', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})
            else:
                target = WorkerState(
                    state.n_drone - 1, state.n_speed_drone + 1,
                    state.n_warrior, state.n_speed_warrior)
                tidx = state_index[target]
                actions.append({'type': 'speed', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})
                target = WorkerState(
                    state.n_drone - 1, state.n_speed_drone,
                    state.n_warrior + 1, state.n_speed_warrior)
                tidx = state_index[target]
                actions.append({'type': 'warrior', 'target': tidx,
                                'win_delta': round(win_probs[tidx] * 100.0 - src_win, 1)})

            if actions:
                transitions.append({
                    'src': src_idx, 'char': ci, 'actions': actions,
                    'src_warrior': is_warrior, 'src_speed': is_speed})
    return transitions


def save_worker_stats_json(
    worker_states: list[WorkerState],
    counts: npt.NDArray[np.int64],
    win_prob: npt.NDArray[np.float64],
    path: str,
    metadata: dict | None = None,
    win_prob_per_map: dict[str, list[float]] | None = None,
    counts_per_map: dict[str, list[int]] | None = None,
    weighting_modes: dict[str, dict] | None = None,
) -> None:
    """Save worker state counts and win probabilities to JSON.

    Top-level counts/win_prob are the naive mode (backward compatible).
    weighting_modes dict contains per-mode data for naive/time/unique.
    """
    data: dict = {
        'states': [
            {'n_drone': s.n_drone, 'n_speed_drone': s.n_speed_drone,
             'n_warrior': s.n_warrior, 'n_speed_warrior': s.n_speed_warrior}
            for s in worker_states
        ],
        'counts': counts.tolist(),
        'win_prob': win_prob.tolist(),
    }
    if metadata:
        data['metadata'] = metadata
    if win_prob_per_map is not None:
        data['win_prob_per_map'] = win_prob_per_map
    if counts_per_map is not None:
        data['counts_per_map'] = counts_per_map
    if weighting_modes is not None:
        data['weighting_modes'] = weighting_modes
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Worker state stats written to {path}")


def load_worker_stats_json(
    path: str,
) -> tuple[list[WorkerState], npt.NDArray[np.int64], npt.NDArray[np.float64], dict]:
    """Load worker state counts and win probs from JSON.

    Returns (states, counts, win_prob, extra) where extra may contain
    'win_prob_per_map', 'counts_per_map', and 'weighting_modes'.

    Supports old JSON format (empirical_win_prob / avg_win_prob keys).
    """
    with open(path) as f:
        data = json.load(f)
    states = [WorkerState(**s) for s in data['states']]
    counts = np.array(data['counts'], dtype=np.int64)
    # Support old format keys
    if 'win_prob' in data:
        win_prob = np.array(data['win_prob'], dtype=np.float64)
    elif 'empirical_win_prob' in data:
        win_prob = np.array(data['empirical_win_prob'], dtype=np.float64)
    else:
        win_prob = np.array(data['avg_win_prob'], dtype=np.float64)
    extra: dict = {}
    # Support old format keys for per-map data
    if 'win_prob_per_map' in data:
        extra['win_prob_per_map'] = data['win_prob_per_map']
    elif 'empirical_win_prob_per_map' in data:
        extra['win_prob_per_map'] = data['empirical_win_prob_per_map']
    if 'counts_per_map' in data:
        extra['counts_per_map'] = data['counts_per_map']
    if 'weighting_modes' in data:
        extra['weighting_modes'] = data['weighting_modes']
    return states, counts, win_prob, extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Worker state empirical win probability analysis')
    parser.add_argument('--data', default='quality_filtered/encoded/all_games.bin',
                        help='Path to binary data file')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Max games to load (default: all)')
    parser.add_argument('--drop-prob', type=float, default=0.0,
                        help='Probability of dropping each state (default: 0.0 = keep all)')
    parser.add_argument('--html', type=str, nargs='?',
                        const='worker_state_values.html',
                        help='Generate HTML visualization (default: worker_state_values.html)')
    parser.add_argument('--json', type=str,
                        help='Save worker state stats to JSON file')
    parser.add_argument('--from-json', type=str,
                        help='Generate HTML from pre-computed JSON (skips computation)')
    parser.add_argument('--n-workers', type=int, default=os.cpu_count() or 1,
                        help='Number of parallel worker processes (default: cpu_count)')
    args = parser.parse_args()

    # --- Fast path: generate HTML from pre-computed JSON ---
    if args.from_json:
        from worker_state_values.html_visualization import generate_html
        states, counts, win_prob, extra = load_worker_stats_json(args.from_json)
        html_path = args.html or 'worker_state_values.html'
        metadata = json.load(open(args.from_json)).get('metadata', {})
        generate_html(states, counts, win_prob, html_path,
                      win_prob_per_map=extra.get('win_prob_per_map'),
                      counts_per_map=extra.get('counts_per_map'),
                      n_games=metadata.get('n_games'),
                      weighting_modes=extra.get('weighting_modes'))
        return

    # --- Worker state machinery (35 states, no eggs) ---
    worker_states = enumerate_worker_states()
    assert len(worker_states) == 35
    worker_state_index = build_worker_state_index()
    worker_lookup = _build_worker_lookup(worker_state_index)
    n_states = len(worker_states)

    # --- Read encoded games into batches (encoded bytes are tiny, ~100 MB total) ---
    from event_codec import read_packed_games
    print(f"Reading games from {args.data}...")
    batches: list[list[tuple[int, bytes]]] = []
    batch: list[tuple[int, bytes]] = []
    n_games_read = 0
    for game_id, encoded in read_packed_games(args.data):
        if args.max_games is not None and n_games_read >= args.max_games:
            break
        batch.append((game_id, encoded))
        n_games_read += 1
        if len(batch) >= BATCH_SIZE:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    print(f"  {n_games_read:,} games in {len(batches)} batches")

    # --- Parallel streaming accumulation ---
    # Each worker materializes one batch locally, accumulates into O(35) arrays,
    # and returns those — X is never held in memory across batches.
    accums = {mode: _make_empty_accum(n_states) for mode in WEIGHTING_MODES}
    n_snapshots = 0

    n_workers = args.n_workers
    print(f"Processing with {n_workers} workers (batch_size={BATCH_SIZE})...")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(worker_lookup, n_states, args.drop_prob),
    ) as pool:
        futures = [pool.submit(_accumulate_batch, b) for b in batches]
        for i, fut in enumerate(as_completed(futures), 1):
            batch_accums, n_snap = fut.result()
            for mode in WEIGHTING_MODES:
                b_wt, b_wn, b_map_wt, b_map_wn = batch_accums[mode]
                accums[mode][0][:] += b_wt
                accums[mode][1][:] += b_wn
                for m in range(4):
                    accums[mode][2][m][:] += b_map_wt[m]
                    accums[mode][3][m][:] += b_map_wn[m]
            n_snapshots += n_snap
            print(f"\r  {i}/{len(batches)} batches — {n_snapshots:,} snapshots",
                  end='', flush=True)
    print()

    # --- Compute win probs for all weighting modes ---
    mode_results: dict[str, dict] = {}
    for mode in WEIGHTING_MODES:
        win_totals, win_ns, map_win_totals, map_win_ns = accums[mode]
        wp = win_totals / np.maximum(win_ns, 1e-12)
        wp_per_map = {
            m_name: (map_win_totals[m_idx] / np.maximum(map_win_ns[m_idx], 1e-12)).tolist()
            for m_idx, m_name in enumerate(MAP_NAMES)
        }
        wn_per_map = {
            m_name: map_win_ns[m_idx].tolist()
            for m_idx, m_name in enumerate(MAP_NAMES)
        }
        mode_results[mode] = {
            'win_prob': wp,
            'win_ns': win_ns,
            'win_prob_per_map': wp_per_map,
            'win_ns_per_map': wn_per_map,
        }

    # Use naive as the primary/backward-compatible result
    naive = mode_results['naive']
    win_prob = naive['win_prob']
    win_ns = naive['win_ns']
    worker_counts = win_ns.astype(np.int64)

    print("\nObservations per map (naive):")
    for m_idx, m_name in enumerate(MAP_NAMES):
        print(f"  {m_name}: {int(np.sum(accums['naive'][3][m_idx])):,}")

    for mode in WEIGHTING_MODES:
        wp = mode_results[mode]['win_prob']
        print(f"\nTop 5 states by {mode} win%:")
        for i in np.argsort(-wp)[:5]:
            s = worker_states[i]
            ns = mode_results[mode]['win_ns'][i]
            print(f"  {s.n_drone}d {s.n_speed_drone}sd {s.n_warrior}w {s.n_speed_warrior}sw: "
                  f"{wp[i]*100:.1f}%  (weight={ns:,.1f})")

    # --- Build weighting_modes dict for JSON ---
    weighting_modes_json = {}
    for mode in WEIGHTING_MODES:
        mr = mode_results[mode]
        weighting_modes_json[mode] = {
            'win_ns': mr['win_ns'].tolist(),
            'win_prob': mr['win_prob'].tolist(),
            'win_prob_per_map': mr['win_prob_per_map'],
            'win_ns_per_map': mr['win_ns_per_map'],
        }

    # --- Save JSON ---
    if args.json:
        save_worker_stats_json(
            worker_states, worker_counts, win_prob, args.json,
            metadata={
                'dataset': args.data,
                'max_games': args.max_games,
                'drop_prob': args.drop_prob,
                'n_snapshots': n_snapshots,
                'n_games': n_games_read,
            },
            win_prob_per_map=naive['win_prob_per_map'],
            counts_per_map={
                m_name: [int(x) for x in accums['naive'][3][m_idx].tolist()]
                for m_idx, m_name in enumerate(MAP_NAMES)
            },
            weighting_modes=weighting_modes_json)

    # --- HTML visualization ---
    if args.html:
        from worker_state_values.html_visualization import generate_html
        generate_html(worker_states, worker_counts, win_prob, args.html,
                      win_prob_per_map=naive['win_prob_per_map'],
                      counts_per_map={
                          m_name: [int(x) for x in accums['naive'][3][m_idx].tolist()]
                          for m_idx, m_name in enumerate(MAP_NAMES)
                      },
                      n_games=n_games_read,
                      weighting_modes=weighting_modes_json)


if __name__ == '__main__':
    main()
