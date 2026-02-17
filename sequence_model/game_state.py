"""Game state tracker for producing 52D feature vectors from parsed event objects.

Mirrors the state replay logic in fast_materialize._process_game but consumes
the same parsed event objects that tokenize_single_game() uses (from preprocess.py).
Produces numerically identical 52D vectors to fast_materialize._vectorize_state.

Usage:
    tracker = GameStateTracker(map_start.map, map_start.gold_on_left, map_infos)
    for event in game_events:
        features = tracker.get_features(event.timestamp)  # shape (52,)
        tracker.apply_event(event)
"""

import numpy as np

from fast_materialize import (
    _vectorize_team, _MAP_ONE_HOT, _MAP_LOOKUPS,
    SCREEN_WIDTH, VANILLA_SNAIL_PPS, SPEED_SNAIL_PPS,
    MAP_INDEX, NUM_FEATURES,
)
from preprocess import (
    GameStartEvent, MapStartEvent, SpawnEvent, CarryFoodEvent,
    BerryDepositEvent, BerryKickInEvent, PlayerKillEvent,
    BlessMaidenEvent, UseMaidenEvent, GetOnSnailEvent, GetOffSnailEvent,
    SnailEatEvent, SnailEscapeEvent, VictoryEvent,
)
from constants import Team, ContestableState, PlayerCategory


def _snail_mult(gold_on_left, team):
    """Snail movement multiplier. team: Team enum."""
    if gold_on_left:
        return -1 if team == Team.GOLD else 1
    else:
        return 1 if team == Team.GOLD else -1


class GameStateTracker:
    """Tracks game state and produces 52D feature vectors matching fast_materialize.

    The feature vector layout (52 features):
      [0:20]   Blue team (20): eggs, food_count, num_vanilla_warriors, num_speed_warriors,
               then 4 workers sorted by power (each: is_bot, has_food, has_speed, has_wings)
      [20:40]  Gold team (20): same layout
      [40:45]  Maiden states (5): 0=neutral, 1=blue, -1=gold
      [45:49]  Map one-hot (4): day, night, dusk, twilight
      [49:52]  Snail: position, speed, berries_available
    """

    def __init__(self, map_name, gold_on_left, map_infos):
        """Initialize tracker for a game.

        Args:
            map_name: Map enum value (e.g. Map.map_day)
            gold_on_left: bool
            map_infos: MapStructureInfos instance (for berry/maiden lookups)
        """
        # Convert Map enum to string for _MAP_LOOKUPS key
        map_name_str = map_name.value  # e.g. 'map_day'
        self.map_lookup = _MAP_LOOKUPS[(map_name_str, gold_on_left)]
        self.map_idx = self.map_lookup['map_index']
        self.gold_on_left = gold_on_left
        self.gold_sym = 1.0 if gold_on_left else -1.0

        # map_infos used for maiden index lookup in BlessMaidenEvent
        self.map_info = map_infos.get_map_info(map_name, gold_on_left)
        self.berry_lookup = self.map_lookup['berry_lookup']
        self.total_berries = self.map_lookup['total_berries']

        # Worker state: w[team_int][worker_idx] = [is_bot, has_food, has_speed, has_wings]
        # team_int: 0=blue, 1=gold
        self.w = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
        self.eggs = [2, 2]
        self.food_dep = [[False] * 12, [False] * 12]
        self.food_count = [0, 0]
        self.maiden_states = [0, 0, 0, 0, 0]  # 0=neutral, 1=blue, -1=gold
        self.berries_avail = self.total_berries
        self.snail_x = float(SCREEN_WIDTH) / 2.0
        self.snail_vel = 0.0
        self.snail_last_ts = 0.0

    def get_features(self, event_timestamp):
        """Get 52D feature vector for the current state at given timestamp.

        Args:
            event_timestamp: float, seconds since game start (normalized)

        Returns:
            np.ndarray of shape (52,) with float32 values.
            Returns zeros if timestamp <= 5.0 (matching fast_materialize behavior).
        """
        if event_timestamp <= 5.0:
            return np.zeros(NUM_FEATURES, dtype=np.float32)

        # Compute inferred snail position
        inferred_pos = self.snail_x + (event_timestamp - self.snail_last_ts) * self.snail_vel
        snail_pos = (inferred_pos / SCREEN_WIDTH - 0.5) * self.gold_sym
        snail_spd = (self.snail_vel / SPEED_SNAIL_PPS) * self.gold_sym

        features = (
            _vectorize_team(self.w[0], self.eggs[0], self.food_count[0])
            + _vectorize_team(self.w[1], self.eggs[1], self.food_count[1])
            + [float(self.maiden_states[0]), float(self.maiden_states[1]),
               float(self.maiden_states[2]), float(self.maiden_states[3]),
               float(self.maiden_states[4])]
            + _MAP_ONE_HOT[self.map_idx]
            + [snail_pos, snail_spd, self.berries_avail / 70.0]
        )
        return np.array(features, dtype=np.float32)

    def apply_event(self, event):
        """Apply state mutation from a parsed event object.

        Mirrors fast_materialize._process_game state mutation logic.
        """
        if isinstance(event, SpawnEvent):
            pid = event.position_id
            team = pid % 2
            widx = (pid - 3) // 2
            self.w[team][widx][0] = event.is_bot

        elif isinstance(event, CarryFoodEvent):
            pid = event.position_id
            team = pid % 2
            widx = (pid - 3) // 2
            self.w[team][widx][1] = True

        elif isinstance(event, BerryDepositEvent):
            pid = event.position_id
            team = pid % 2
            widx = (pid - 3) // 2
            self.w[team][widx][1] = False
            bi = self.berry_lookup[(event.hole_x, event.hole_y)]
            if not self.food_dep[team][bi]:
                self.food_dep[team][bi] = True
                self.food_count[team] += 1
            self.berries_avail -= 1

        elif isinstance(event, BerryKickInEvent):
            pid = event.position_id
            team = pid % 2
            if not event.counts_for_own_team:
                team = 1 - team
            bi = self.berry_lookup[(event.hole_x, event.hole_y)]
            if not self.food_dep[team][bi]:
                self.food_dep[team][bi] = True
                self.food_count[team] += 1
            self.berries_avail -= 1

        elif isinstance(event, BlessMaidenEvent):
            color = 1 if event.gate_color == ContestableState.BLUE else -1
            try:
                _, midx = self.map_info.get_type_and_maiden_index(
                    event.maiden_x, event.maiden_y)
                self.maiden_states[midx] = color
            except ValueError:
                pass  # Skip invalid maiden coords

        elif isinstance(event, UseMaidenEvent):
            pid = event.position_id
            team = pid % 2
            widx = (pid - 3) // 2
            from constants import MaidenType
            if event.maiden_type == MaidenType.maiden_speed:
                self.w[team][widx][2] = True
            else:
                self.w[team][widx][3] = True
            self.w[team][widx][1] = False

        elif isinstance(event, PlayerKillEvent):
            killed_pid = event.killed_position_id
            team = killed_pid % 2
            if event.killed_player_category == PlayerCategory.Queen:
                self.eggs[team] -= 1
            else:
                widx = (killed_pid - 3) // 2
                self.w[team][widx][1] = False
                self.w[team][widx][2] = False
                self.w[team][widx][3] = False

        elif isinstance(event, GetOnSnailEvent):
            self.snail_x = float(event.snail_x)
            self.snail_last_ts = event.timestamp
            rider_pid = event.rider_position_id
            rider_team = rider_pid % 2
            rider_widx = (rider_pid - 3) // 2
            has_speed = self.w[rider_team][rider_widx][2]
            base_speed = SPEED_SNAIL_PPS if has_speed else VANILLA_SNAIL_PPS
            team_enum = Team.GOLD if rider_team == 1 else Team.BLUE
            self.snail_vel = base_speed * _snail_mult(self.gold_on_left, team_enum)

        elif isinstance(event, SnailEatEvent):
            self.snail_x = float(event.snail_x)
            self.snail_last_ts = event.timestamp
            rider_pid = event.rider_position_id
            rider_team = rider_pid % 2
            rider_widx = (rider_pid - 3) // 2
            has_speed = self.w[rider_team][rider_widx][2]
            base_speed = SPEED_SNAIL_PPS if has_speed else VANILLA_SNAIL_PPS
            team_enum = Team.GOLD if rider_team == 1 else Team.BLUE
            self.snail_vel = base_speed * _snail_mult(self.gold_on_left, team_enum)

        elif isinstance(event, GetOffSnailEvent):
            self.snail_x = float(event.snail_x)
            self.snail_last_ts = event.timestamp
            self.snail_vel = 0.0

        elif isinstance(event, SnailEscapeEvent):
            self.snail_x = float(event.snail_x)
            self.snail_last_ts = event.timestamp
            self.snail_vel = 0.0

        # GameStartEvent, MapStartEvent, VictoryEvent: no state mutation
