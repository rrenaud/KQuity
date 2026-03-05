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


def _accumulate_batch(
    encoded_games: list[tuple[int, bytes]],
) -> tuple[
    npt.NDArray[np.int64],            # counts
    npt.NDArray[np.float64],          # emp_win_totals   (actual y)
    npt.NDArray[np.int64],            # emp_win_ns
    list[npt.NDArray[np.float64]],    # map_emp_win_totals[4]
    list[npt.NDArray[np.int64]],      # map_emp_win_ns[4]
    int,                              # n_snapshots
]:
    """Materialize + accumulate empirical win stats for one batch of games.

    Runs entirely in a worker process.  X is local and discarded after
    accumulation — memory usage is O(batch_size * snapshots * features),
    not O(total_snapshots).
    """
    from event_codec import materialize_entries
    n_states = _worker_n_states

    X, y, _gids, _ts = materialize_entries(
        encoded_games, drop_state_probability=_worker_drop_prob)

    counts = np.zeros(n_states, dtype=np.int64)
    emp_win_totals = np.zeros(n_states, dtype=np.float64)
    emp_win_ns = np.zeros(n_states, dtype=np.int64)
    map_emp_win_totals = [np.zeros(n_states, dtype=np.float64) for _ in range(4)]
    map_emp_win_ns = [np.zeros(n_states, dtype=np.int64) for _ in range(4)]

    if len(X) == 0:
        return (counts, emp_win_totals, emp_win_ns,
                map_emp_win_totals, map_emp_win_ns, 0)

    y_f = y.astype(np.float64)  # 1.0 = blue won, 0.0 = gold won
    worker_blue, worker_gold = extract_worker_state_indices(X, _worker_lookup)
    valid = (worker_blue >= 0) & (worker_gold >= 0)

    np.add.at(counts, worker_blue[valid], 1)
    np.add.at(counts, worker_gold[valid], 1)

    np.add.at(emp_win_totals, worker_blue[valid], y_f[valid])
    np.add.at(emp_win_ns, worker_blue[valid], 1)
    np.add.at(emp_win_totals, worker_gold[valid], 1.0 - y_f[valid])
    np.add.at(emp_win_ns, worker_gold[valid], 1)

    map_ids = np.argmax(X[:, MAP_FEATURE_OFFSET:MAP_FEATURE_OFFSET + 4], axis=1)
    for m_idx in range(4):
        mask = valid & (map_ids == m_idx)
        np.add.at(map_emp_win_totals[m_idx], worker_blue[mask], y_f[mask])
        np.add.at(map_emp_win_ns[m_idx], worker_blue[mask], 1)
        np.add.at(map_emp_win_totals[m_idx], worker_gold[mask], 1.0 - y_f[mask])
        np.add.at(map_emp_win_ns[m_idx], worker_gold[mask], 1)

    return (counts, emp_win_totals, emp_win_ns,
            map_emp_win_totals, map_emp_win_ns, len(X))


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
) -> None:
    """Save worker state counts and win probabilities to JSON."""
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
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Worker state stats written to {path}")


def load_worker_stats_json(
    path: str,
) -> tuple[list[WorkerState], npt.NDArray[np.int64], npt.NDArray[np.float64], dict]:
    """Load worker state counts and win probs from JSON.

    Returns (states, counts, win_prob, extra) where extra may contain
    'win_prob_per_map' and 'counts_per_map'.

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
                      n_games=metadata.get('n_games'))
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
    worker_counts = np.zeros(n_states, dtype=np.int64)
    emp_win_totals = np.zeros(n_states, dtype=np.float64)
    emp_win_ns = np.zeros(n_states, dtype=np.int64)
    map_emp_win_totals = [np.zeros(n_states, dtype=np.float64) for _ in range(4)]
    map_emp_win_ns = [np.zeros(n_states, dtype=np.int64) for _ in range(4)]
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
            (b_counts, b_emp_wt, b_emp_wn,
             b_map_ewt, b_map_ewn, n_snap) = fut.result()
            worker_counts += b_counts
            emp_win_totals += b_emp_wt
            emp_win_ns += b_emp_wn
            for m in range(4):
                map_emp_win_totals[m] += b_map_ewt[m]
                map_emp_win_ns[m] += b_map_ewn[m]
            n_snapshots += n_snap
            print(f"\r  {i}/{len(batches)} batches — {n_snapshots:,} snapshots",
                  end='', flush=True)
    print()

    win_prob = emp_win_totals / np.maximum(emp_win_ns, 1)
    win_prob_per_map = {
        m_name: (map_emp_win_totals[m_idx] / np.maximum(map_emp_win_ns[m_idx], 1)).tolist()
        for m_idx, m_name in enumerate(MAP_NAMES)
    }
    counts_per_map = {
        m_name: map_emp_win_ns[m_idx].tolist()
        for m_idx, m_name in enumerate(MAP_NAMES)
    }

    print("\nObservations per map:")
    for m_idx, m_name in enumerate(MAP_NAMES):
        print(f"  {m_name}: {int(np.sum(map_emp_win_ns[m_idx])):,}")

    print(f"\nTop 5 states by empirical win%:")
    for i in np.argsort(-win_prob)[:5]:
        s = worker_states[i]
        print(f"  {s.n_drone}d {s.n_speed_drone}sd {s.n_warrior}w {s.n_speed_warrior}sw: "
              f"{win_prob[i]*100:.1f}%  (n={worker_counts[i]:,})")

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
            win_prob_per_map=win_prob_per_map,
            counts_per_map=counts_per_map)

    # --- HTML visualization ---
    if args.html:
        from worker_state_values.html_visualization import generate_html
        generate_html(worker_states, worker_counts, win_prob, args.html,
                      win_prob_per_map=win_prob_per_map,
                      counts_per_map=counts_per_map,
                      n_games=n_games_read)


if __name__ == '__main__':
    main()
