"""Feature extraction for game quality classification.

Computes per-game summary features from CSV event streams.
Designed for fast iteration: modify features here, re-run classifier.
"""

import os
import sys
from collections import defaultdict
from itertools import combinations

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from preprocess import (
    iterate_events_from_csv,
    iterate_events_by_game_and_normalize_time,
    GameStartEvent, MapStartEvent, VictoryEvent, SpawnEvent,
    PlayerKillEvent, CarryFoodEvent, BerryDepositEvent, BerryKickInEvent,
    BlessMaidenEvent, UseMaidenEvent, GetOnSnailEvent, SnailEatEvent,
    GetOffSnailEvent, SnailEscapeEvent,
    position_id_to_team,
)
from constants import PlayerCategory, Map, VictoryCondition, Team
import map_structure

GAMEPLAY_EVENT_TYPES = (
    PlayerKillEvent, CarryFoodEvent, BerryDepositEvent, BerryKickInEvent,
    BlessMaidenEvent, UseMaidenEvent, GetOnSnailEvent, SnailEatEvent,
    GetOffSnailEvent, SnailEscapeEvent,
)

ALL_PIDS = range(1, 11)     # PIDs 1-10: all cabinet positions
WORKER_PIDS = range(3, 11)  # PIDs 3-10 are workers; 1-2 are queens


def _best_gate_triple_window(gate_bless_times):
    """Find minimum time window covering a bless at 3 different maidens.

    gate_bless_times: dict mapping maiden_index -> [timestamps]
    Returns the minimum time span, or float('inf') if no triple exists.
    """
    best = float('inf')
    maiden_indices = [idx for idx, times in gate_bless_times.items() if times]
    if len(maiden_indices) < 3:
        return best

    for triple in combinations(maiden_indices, 3):
        # Merge-style sliding window across 3 sorted timestamp lists
        lists = [gate_bless_times[idx] for idx in triple]
        # For each combination, find minimum window covering one from each list
        # Use pointer-based approach: advance the pointer with the smallest value
        ptrs = [0] * 3
        while True:
            times = [lists[i][ptrs[i]] for i in range(3)]
            window = max(times) - min(times)
            if window < best:
                best = window
            # Advance the pointer with the smallest time
            min_idx = times.index(min(times))
            ptrs[min_idx] += 1
            if ptrs[min_idx] >= len(lists[min_idx]):
                break
    return best


def compute_quality_features(csv_path):
    """Compute per-game quality features from a CSV event stream.

    Args:
        csv_path: Glob pattern or path to CSV/CSV.gz game event files.

    Returns:
        pd.DataFrame with one row per game and ~41 feature columns.
    """
    map_structure_infos = map_structure.MapStructureInfos()
    rows = []
    game_count = 0

    events = iterate_events_from_csv(csv_path)
    for game_id, game_events in iterate_events_by_game_and_normalize_time(events):
        game_count += 1

        # Find key structural events
        gamestart = None
        map_start = None
        victory = None

        for event in game_events:
            if isinstance(event, GameStartEvent) and gamestart is None:
                gamestart = event
            if isinstance(event, MapStartEvent) and map_start is None:
                map_start = event
            if isinstance(event, VictoryEvent):
                victory = event

        if gamestart is None or victory is None:
            continue

        duration = victory.timestamp  # already normalized
        if duration <= 0:
            continue

        # Resolve map info for maiden index lookup
        map_info = None
        if map_start is not None:
            try:
                map_info = map_structure_infos.get_map_info(
                    map_start.map, map_start.gold_on_left)
            except (KeyError, ValueError):
                pass

        # --- Accumulators ---
        bot_pids = set()
        total_kills = 0
        queen_kills = 0
        first_kill_t = None
        total_carry = 0
        first_carry_t = None
        total_deposit = 0
        total_kick_in = 0
        total_bless = 0
        first_bless_t = None
        total_use_maiden = 0
        first_maiden_use_t = None
        total_get_on_snail = 0
        first_snail_t = None
        total_snail_eat = 0
        total_snail_escape = 0
        first_snail_escape_t = None
        total_get_off_snail = 0
        first_get_off_snail_t = None
        gameplay_event_count = 0

        player_event_counts = defaultdict(int)
        active_pids = set()
        worker_first_objective_t = {}  # pid -> timestamp
        first_event_by_pid = {}  # pid -> timestamp of first gameplay event
        first_maiden_use_by_team = {}  # Team -> timestamp
        first_maiden_use_by_pid = {}   # pid -> timestamp
        carry_timestamps = []          # all CarryFood timestamps (for Nth pickup)
        bless_timestamps = []          # all BlessMaiden timestamps (for Nth bless)

        # Gate triple tracking: team -> maiden_index -> [timestamps]
        gate_bless_times = {
            Team.BLUE: defaultdict(list),
            Team.GOLD: defaultdict(list),
        }

        for event in game_events:
            if isinstance(event, SpawnEvent):
                if event.is_bot:
                    bot_pids.add(event.position_id)

            elif isinstance(event, PlayerKillEvent):
                total_kills += 1
                if event.killed_player_category == PlayerCategory.Queen:
                    queen_kills += 1
                if first_kill_t is None:
                    first_kill_t = event.timestamp
                active_pids.add(event.killer_position_id)
                active_pids.add(event.killed_position_id)
                player_event_counts[event.killer_position_id] += 1

            elif isinstance(event, CarryFoodEvent):
                total_carry += 1
                carry_timestamps.append(event.timestamp)
                if first_carry_t is None:
                    first_carry_t = event.timestamp
                pid = event.position_id
                active_pids.add(pid)
                player_event_counts[pid] += 1
                if pid in WORKER_PIDS and pid not in worker_first_objective_t:
                    worker_first_objective_t[pid] = event.timestamp

            elif isinstance(event, BerryDepositEvent):
                total_deposit += 1
                active_pids.add(event.position_id)
                player_event_counts[event.position_id] += 1

            elif isinstance(event, BerryKickInEvent):
                total_kick_in += 1
                active_pids.add(event.position_id)
                player_event_counts[event.position_id] += 1

            elif isinstance(event, BlessMaidenEvent):
                total_bless += 1
                bless_timestamps.append(event.timestamp)
                if first_bless_t is None:
                    first_bless_t = event.timestamp
                # Track gate bless times for triple window
                if map_info is not None:
                    try:
                        _, maiden_index = map_info.get_type_and_maiden_index(
                            event.maiden_x, event.maiden_y)
                        from constants import ContestableState
                        team = Team.BLUE if event.gate_color == ContestableState.BLUE else Team.GOLD
                        gate_bless_times[team][maiden_index].append(event.timestamp)
                    except ValueError:
                        pass

            elif isinstance(event, UseMaidenEvent):
                total_use_maiden += 1
                if first_maiden_use_t is None:
                    first_maiden_use_t = event.timestamp
                pid = event.position_id
                active_pids.add(pid)
                player_event_counts[pid] += 1
                team = position_id_to_team(pid)
                if team not in first_maiden_use_by_team:
                    first_maiden_use_by_team[team] = event.timestamp
                if pid not in first_maiden_use_by_pid:
                    first_maiden_use_by_pid[pid] = event.timestamp

            elif isinstance(event, GetOnSnailEvent):
                total_get_on_snail += 1
                if first_snail_t is None:
                    first_snail_t = event.timestamp
                pid = event.rider_position_id
                active_pids.add(pid)
                player_event_counts[pid] += 1
                if pid in WORKER_PIDS and pid not in worker_first_objective_t:
                    worker_first_objective_t[pid] = event.timestamp

            elif isinstance(event, SnailEatEvent):
                total_snail_eat += 1
                active_pids.add(event.rider_position_id)
                player_event_counts[event.rider_position_id] += 1

            elif isinstance(event, GetOffSnailEvent):
                total_get_off_snail += 1
                if first_get_off_snail_t is None:
                    first_get_off_snail_t = event.timestamp
                active_pids.add(event.position_id)
                player_event_counts[event.position_id] += 1

            elif isinstance(event, SnailEscapeEvent):
                total_snail_escape += 1
                if first_snail_escape_t is None:
                    first_snail_escape_t = event.timestamp
                active_pids.add(event.escaped_position_id)
                player_event_counts[event.escaped_position_id] += 1

            if isinstance(event, GAMEPLAY_EVENT_TYPES):
                gameplay_event_count += 1
                # Track first gameplay event per cabinet position
                pids_in_event = []
                if hasattr(event, 'position_id'):
                    pids_in_event.append(event.position_id)
                if hasattr(event, 'killer_position_id'):
                    pids_in_event.append(event.killer_position_id)
                if hasattr(event, 'killed_position_id'):
                    pids_in_event.append(event.killed_position_id)
                if hasattr(event, 'rider_position_id'):
                    pids_in_event.append(event.rider_position_id)
                if hasattr(event, 'eaten_position_id'):
                    pids_in_event.append(event.eaten_position_id)
                if hasattr(event, 'escaped_position_id'):
                    pids_in_event.append(event.escaped_position_id)
                for pid in pids_in_event:
                    if pid not in first_event_by_pid:
                        first_event_by_pid[pid] = event.timestamp

        # --- Derived features ---
        total_events = len(game_events)
        active_player_count = len(active_pids)
        total_player_events = sum(player_event_counts.values())
        max_player_share = (
            max(player_event_counts.values()) / total_player_events
            if total_player_events > 0 else 1.0
        )

        # Workers who never touched berry or snail
        objective_times = [worker_first_objective_t.get(pid) for pid in WORKER_PIDS]
        workers_never_touched = sum(1 for t in objective_times if t is None)
        workers_with_objective = [t for t in objective_times if t is not None]
        max_worker_first_objective = (
            max(workers_with_objective) if workers_with_objective else duration
        )

        # Gate triple window
        best_triple = min(
            _best_gate_triple_window(gate_bless_times[Team.BLUE]),
            _best_gate_triple_window(gate_bless_times[Team.GOLD]),
        )
        if best_triple == float('inf'):
            best_triple = 9999.0

        # NaN-fill for time_to_first_*: use duration if event never happened
        first_kill_t = first_kill_t if first_kill_t is not None else duration
        first_carry_t = first_carry_t if first_carry_t is not None else duration
        first_bless_t = first_bless_t if first_bless_t is not None else duration
        first_maiden_use_t = first_maiden_use_t if first_maiden_use_t is not None else duration
        first_snail_t = first_snail_t if first_snail_t is not None else duration
        first_snail_escape_t = first_snail_escape_t if first_snail_escape_t is not None else duration
        first_get_off_snail_t = first_get_off_snail_t if first_get_off_snail_t is not None else duration

        # Victory condition one-hot
        vc = victory.victory_condition
        # Map one-hot
        map_id = map_start.map if map_start else None

        # Per-team first maiden use (sorted so team identity doesn't matter)
        team_maiden_times = sorted([
            first_maiden_use_by_team.get(Team.BLUE, duration),
            first_maiden_use_by_team.get(Team.GOLD, duration),
        ])
        # Per-PID first maiden use (sorted ascending, 8 workers only — queens can't use maidens)
        pid_maiden_times = sorted(
            first_maiden_use_by_pid.get(pid, duration) for pid in WORKER_PIDS
        )

        # Time to Nth berry pickup (fill with duration if fewer than N pickups)
        time_to_6_carry = carry_timestamps[5] if len(carry_timestamps) >= 6 else duration

        # Time to Nth maiden bless (fill with duration if fewer than N blesses)
        def _time_to_nth(timestamps, n):
            return timestamps[n - 1] if len(timestamps) >= n else duration
        time_to_3_bless = _time_to_nth(bless_timestamps, 3)
        time_to_5_bless = _time_to_nth(bless_timestamps, 5)
        time_to_10_bless = _time_to_nth(bless_timestamps, 10)

        # Per-cab-position first event times, sorted ascending (team-agnostic)
        cab_first_times = sorted(
            first_event_by_pid.get(pid, duration) for pid in ALL_PIDS
        )

        row = {
            'game_id': game_id,
            # Basic (10)
            'duration_seconds': duration,
            'total_event_count': total_events,
            'bot_count': len(bot_pids),
            'vc_military': float(vc == VictoryCondition.military),
            'vc_economic': float(vc == VictoryCondition.economic),
            'vc_snail': float(vc == VictoryCondition.snail),
            'map_day': float(map_id == Map.map_day) if map_id else 0.0,
            'map_night': float(map_id == Map.map_night) if map_id else 0.0,
            'map_dusk': float(map_id == Map.map_dusk) if map_id else 0.0,
            'map_twilight': float(map_id == Map.map_twilight) if map_id else 0.0,
            # Counts (10)
            'total_kills': total_kills,
            'queen_kills': queen_kills,
            'total_carry_food': total_carry,
            'total_berry_deposits': total_deposit,
            'total_berry_kick_ins': total_kick_in,
            'total_bless_maiden': total_bless,
            'total_use_maiden': total_use_maiden,
            'total_get_on_snail': total_get_on_snail,
            'total_snail_eat': total_snail_eat,
            'total_snail_escape': total_snail_escape,
            'total_get_off_snail': total_get_off_snail,
            'gameplay_events': gameplay_event_count,
            # Rates (8)
            'gameplay_eps': gameplay_event_count / duration,
            'events_per_second': total_events / duration,
            'kill_rate': total_kills / duration,
            'carry_rate': total_carry / duration,
            'bless_rate': total_bless / duration,
            'deposit_rate': total_deposit / duration,
            'maiden_use_rate': total_use_maiden / duration,
            'snail_rate': total_get_on_snail / duration,
            'snail_escape_rate': total_snail_escape / duration,
            'get_off_snail_rate': total_get_off_snail / duration,
            # Temporal (7)
            'time_to_first_kill': first_kill_t,
            'time_to_first_carry': first_carry_t,
            'time_to_first_bless': first_bless_t,
            'time_to_first_maiden_use': first_maiden_use_t,
            'time_to_first_snail': first_snail_t,
            'time_to_first_snail_escape': first_snail_escape_t,
            'time_to_first_get_off_snail': first_get_off_snail_t,
            # Engagement (4)
            'active_player_count': active_player_count,
            'workers_never_touched_objective': workers_never_touched,
            'max_player_event_share': max_player_share,
            'max_worker_first_objective': max_worker_first_objective,
            # Gate triples (2)
            'best_gate_triple_window': best_triple,
            'best_gate_triple_window_frac': best_triple / duration if best_triple < 9999.0 else 9999.0,
        }
        # Per-cab first event (10) — sorted ascending
        for i, t in enumerate(cab_first_times):
            row[f'first_event_cab_{i+1:02d}'] = t
        # Per-team first maiden use (2) — sorted ascending
        row['first_maiden_use_team_1'] = team_maiden_times[0]
        row['first_maiden_use_team_2'] = team_maiden_times[1]
        # Per-PID first maiden use (8) — sorted ascending across workers
        for i, t in enumerate(pid_maiden_times):
            row[f'first_maiden_use_pid_{i+1:02d}'] = t
        # Time to 6 berry pickups (1)
        row['time_to_6_carry'] = time_to_6_carry
        # Time to Nth maiden bless (3)
        row['time_to_3_bless'] = time_to_3_bless
        row['time_to_5_bless'] = time_to_5_bless
        row['time_to_10_bless'] = time_to_10_bless
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f'{csv_path}: {game_count} games streamed, {len(df)} with gamestart+victory')
    return df


# Feature columns (excludes game_id)
FEATURE_COLUMNS = [
    # Basic (10)
    'duration_seconds', 'total_event_count', 'bot_count',
    'vc_military', 'vc_economic', 'vc_snail',
    'map_day', 'map_night', 'map_dusk', 'map_twilight',
    # Counts (10)
    'total_kills', 'queen_kills', 'total_carry_food',
    'total_berry_deposits', 'total_berry_kick_ins',
    'total_bless_maiden', 'total_use_maiden',
    'total_get_on_snail', 'total_snail_eat', 'total_snail_escape', 'total_get_off_snail',
    'gameplay_events',
    # Rates (8)
    'gameplay_eps', 'events_per_second', 'kill_rate', 'carry_rate',
    'bless_rate', 'deposit_rate', 'maiden_use_rate', 'snail_rate', 'snail_escape_rate',
    'get_off_snail_rate',
    # Temporal (7)
    'time_to_first_kill', 'time_to_first_carry', 'time_to_first_bless',
    'time_to_first_maiden_use', 'time_to_first_snail', 'time_to_first_snail_escape',
    'time_to_first_get_off_snail',
    # Engagement (4)
    'active_player_count', 'workers_never_touched_objective',
    'max_player_event_share', 'max_worker_first_objective',
    # Gate triples (2)
    'best_gate_triple_window', 'best_gate_triple_window_frac',
    # Per-cab first event (10)
] + [f'first_event_cab_{i:02d}' for i in range(1, 11)] + [
    # Per-team first maiden use (2)
    'first_maiden_use_team_1', 'first_maiden_use_team_2',
    # Per-PID first maiden use (8)
] + [f'first_maiden_use_pid_{i:02d}' for i in range(1, 9)] + [
    # Time to Nth (4)
    'time_to_6_carry',
    'time_to_3_bless', 'time_to_5_bless', 'time_to_10_bless',
]
