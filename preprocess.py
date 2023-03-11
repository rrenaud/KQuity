import csv

import collections
import datetime
import typing

import dateutil.parser
from typing import List, Optional, Dict, Union, Type

import constants
import map_structure
from constants import Team, ContestableState, VictoryCondition, PlayerCategory, MaidenType, Map

# https://kqhivemind.com/wiki/Stats_Socket_Events


def opposing_team(team: Team) -> Team:
    return Team.GOLD if team == Team.BLUE else Team.BLUE


class WorkerState:
    def __init__(self):
        self.has_speed = False
        self.has_wings = False
        self.has_food = False
        self.is_bot = False


class TeamState:
    def __init__(self):
        self.eggs = 2
        self.berries_deposited = [False for _ in range(12)]
        self.workers = [WorkerState() for _ in range(4)]


StartSnailEvent: Type = Union['GetOnSnailEvent', 'SnailEatEvent']
StopSnailEvent: Type = Union['GetOffSnailEvent', 'SnailEscapeEvent']


class InferredSnailState:
    VANILLA_SNAIL_PIXELS_PER_SECOND = 20.896215463
    SPEED_SNAIL_PIXELS_PER_SECOND = 28.209890875

    def __init__(self, game_state: 'GameState'):
        self.snail_x = constants.SCREEN_WIDTH / 2
        self.snail_velocity = 0
        self.last_touch_timestamp = 0.0
        self.game_state = game_state

    @staticmethod
    def snail_movement_multiplier(gold_on_left, team):
        if gold_on_left and team == Team.GOLD: return -1
        if gold_on_left and team == Team.BLUE: return 1
        if not gold_on_left and team == Team.GOLD: return 1
        if not gold_on_left and team == Team.BLUE: return -1

    def inferred_snail_position(self, current_ts: float) -> float:
        return self.snail_x + (current_ts - self.last_touch_timestamp) * self.snail_velocity

    def _compute_snail_velocity(self, rider_position_id) -> float:
        rider = self.game_state.get_worker_by_position_id(rider_position_id)
        velocity = self.SPEED_SNAIL_PIXELS_PER_SECOND if rider.has_speed else self.VANILLA_SNAIL_PIXELS_PER_SECOND
        velocity *= InferredSnailState.snail_movement_multiplier(self.game_state.map_info.gold_on_left,
                                                                 position_id_to_team(rider_position_id))

        # print('snail_x:', self.snail_x,
        #     'velocity:', velocity,
        #     'gold_on_left:', self.game_state.map_info.gold_on_left,
        #     'team:', position_id_to_team(rider_position_id))
        return velocity

    def _update_position_and_timestamp(self, snail_event):
        # if should_print:
        #     inferred_position = self.inferred_snail_position(snail_event.timestamp)
        #     diff = inferred_position - snail_event.snail_x
        #     print('inferred_position:', inferred_position, 'event_snail_x:', snail_event.snail_x, 'diff:', diff)
        self.last_touch_timestamp = snail_event.timestamp
        self.snail_x = snail_event.snail_x

    def start_snail(self, start_snail_event: StartSnailEvent):
        self.snail_velocity = self._compute_snail_velocity(start_snail_event.rider_position_id)
        self._update_position_and_timestamp(start_snail_event)

    def stop_snail(self, stop_snail_event: StopSnailEvent):
        self._update_position_and_timestamp(stop_snail_event)
        self.snail_velocity = 0


class GameState:
    def __init__(self, map_info: map_structure.MapStructureInfo):
        self.map_info = map_info
        self.teams = [TeamState() for _ in range(2)]
        self.berries_available = map_info.total_berries
        self.maiden_states = [ContestableState.NEUTRAL for _ in range(5)]
        self.snail_state = InferredSnailState(self)

    def get_team(self, team: Team) -> TeamState:
        return self.teams[team.value]

    def get_worker_by_position_id(self, position_id: int) -> WorkerState:
        team: Team = position_id_to_team(position_id)
        worker_index = position_id_to_worker_index(position_id)
        return self.get_team(team).workers[worker_index]

    def num_bots(self) -> int:
        return sum([worker.is_bot for team in self.teams for worker in team.workers])


class GameEvent:
    def __init__(self):
        self.timestamp = None
        self.game_id = None

    def modify_game_state(self, game_state: GameState):
        pass


GameEvents = List[GameEvent]


def position_id_to_team(position_id: int) -> Team:
    return Team(position_id % 2)


def position_id_to_worker_index(position_id: int) -> int:
    return (position_id - 3) // 2


def split_payload(payload: str) -> List[str]:
    assert payload.startswith('{')
    assert payload.endswith('}')
    return payload[1:-1].split(',')


class GameStartEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.map = Map[payload_values[0]]


class MapStartEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.map = Map[payload_values[0]]
        self.gold_on_left = payload_values[1] == 'True'
        self.game_version = payload_values[4]


class SpawnEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.position_id = int(payload_values[0])
        self.is_bot = payload_values[1] == 'True'

    def modify_game_state(self, game_state: GameState):
        game_state.get_worker_by_position_id(self.position_id).is_bot = self.is_bot


class BerryDepositEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.hole_x = int(payload_values[0])
        self.hole_y = int(payload_values[1])
        self.position_id = int(payload_values[2])

    def modify_game_state(self, game_state: GameState):
        team: Team = position_id_to_team(self.position_id)
        worker_index = position_id_to_worker_index(self.position_id)
        team_state: TeamState = game_state.get_team(team)
        team_state.workers[worker_index].has_food = False
        berry_index = game_state.map_info.get_berry_index(self.hole_x, self.hole_y)
        team_state.berries_deposited[berry_index] = True
        game_state.berries_available -= 1


class BerryKickInEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.hole_x = int(payload_values[0])
        self.hole_y = int(payload_values[1])
        self.position_id = int(payload_values[2])
        self.counts_for_own_team = payload_values[3] == 'True'

    def modify_game_state(self, game_state: GameState):
        team: Team = position_id_to_team(self.position_id)
        if not self.counts_for_own_team:
            team = opposing_team(team)

        berry_index = game_state.map_info.get_berry_index(self.hole_x, self.hole_y)
        game_state.get_team(team).berries_deposited[berry_index] = True
        game_state.berries_available -= 1


class BlessMaidenEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.maiden_x = int(payload_values[0])
        self.maiden_y = int(payload_values[1])
        self.gate_color = ContestableState.BLUE if payload_values[2] == 'Blue' else ContestableState.GOLD

    def modify_game_state(self, game_state: GameState):
        team: Team = Team.BLUE if self.gate_color == ContestableState.BLUE else ContestableState.GOLD
        _, maiden_index = game_state.map_info.get_type_and_maiden_index(self.maiden_x, self.maiden_y)
        game_state.maiden_states[maiden_index] = self.gate_color


class CarryFoodEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.position_id = int(payload_values[0])

    def modify_game_state(self, game_state: GameState):
        game_state.get_worker_by_position_id(self.position_id).has_food = True


class GetOnSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.rider_position_id = int(payload_values[2])

    def modify_game_state(self, game_state: GameState):
        game_state.snail_state.start_snail(self)


class SnailEatEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.rider_position_id = int(payload_values[2])
        self.eaten_position_id = int(payload_values[3])

    def modify_game_state(self, game_state: GameState):
        game_state.snail_state.start_snail(self)


class GetOffSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.position_id = int(payload_values[3])

    def modify_game_state(self, game_state: GameState):
        game_state.snail_state.stop_snail(self)


class SnailEscapeEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.escaped_position_id = int(payload_values[2])

    def modify_game_state(self, game_state: GameState):
        game_state.snail_state.stop_snail(self)


class GlanceEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        if len(payload_values) == 2:
            # Hack for older versions where the payload was missing the glance coordinates.
            payload_values = ['-1', '-1'] + payload_values
        self.glance_x = int(payload_values[0])
        self.glance_y = int(payload_values[1])
        self.position_ids = [int(payload_values[2]), int(payload_values[3])]


class PlayerKillEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.killer_x = int(payload_values[0])
        self.killer_y = int(payload_values[1])
        self.killer_position_id = int(payload_values[2])
        self.killed_position_id = int(payload_values[3])
        self.killed_player_category = PlayerCategory[payload_values[4]]

    def modify_game_state(self, game_state: GameState):
        team: Team = position_id_to_team(self.killed_position_id)
        if self.killed_player_category == PlayerCategory.Queen:
            game_state.get_team(team).eggs -= 1
        else:
            killed_worker = game_state.get_worker_by_position_id(self.killed_position_id)
            validate_condition(killed_worker.has_wings == (self.killed_player_category == PlayerCategory.Soldier),
                               'Worker has wings but is not a soldier')
            killed_worker.has_speed = killed_worker.has_food = killed_worker.has_wings = False


class GameValidationError(ValueError):
    def __init__(self, message: str):
        super().__init__(message)


def validate_condition(condition, message):
    if not condition:
        raise GameValidationError(message)


class UseMaidenEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.maiden_x = int(payload_values[0])
        self.maiden_y = int(payload_values[1])
        self.maiden_type = MaidenType[payload_values[2]]
        self.position_id = int(payload_values[3])

    def modify_game_state(self, game_state: GameState):
        _, maiden_index = game_state.map_info.get_type_and_maiden_index(self.maiden_x, self.maiden_y)
        worker_state = game_state.get_worker_by_position_id(self.position_id)

        validate_condition(worker_state.has_food, "Worker using maiden missing food")
        if self.maiden_type == MaidenType.maiden_speed:
            validate_condition(not worker_state.has_speed, "Worker using speed maiden already has speed")
            worker_state.has_speed = True
        elif self.maiden_type == MaidenType.maiden_wings:
            validate_condition(not worker_state.has_wings, "Worker using wings maiden already has wings")
            worker_state.has_wings = True

        worker_state.has_food = False


class VictoryEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.winning_team = Team.BLUE if payload_values[0] == 'Blue' else Team.GOLD
        self.victory_condition = VictoryCondition[payload_values[1]]

    def modify_game_state(self, game_state: GameState):
        if self.victory_condition == VictoryCondition.economic:
            validate_condition(sum(game_state.get_team(self.winning_team).berries_deposited) == 12,
                               "Econ victory team does not have 12 berries deposited")
        elif self.victory_condition == VictoryCondition.military:
            validate_condition(game_state.get_team(opposing_team(self.winning_team)).eggs == -1,
                               "Military loss team has a living queen?")
        elif self.victory_condition == VictoryCondition.snail:
            map_info = game_state.map_info
            snail_direction = InferredSnailState.snail_movement_multiplier(map_info.gold_on_left,
                                                                           self.winning_team)
            snail_post = constants.SCREEN_WIDTH / 2 + snail_direction * map_info.snail_track_width
            distance_from_goal = snail_post - game_state.snail_state.inferred_snail_position(self.timestamp)

            # This condition failed one in 65 games in a sample of otherwise validated snail victories.
            validate_condition(abs(distance_from_goal) < 100, 'inferred snail position far from goal')


def parse_event(raw_event_row) -> Optional[GameEvent]:
    skippable_events = {'gameend', 'playernames',
                        'reserveMaiden', 'unreserveMaiden',
                        'cabinetOnline', 'cabinetOffline',
                        'bracket', 'tstart', 'tournamentValidation', 'checkIfTournamentRunning'}
    dispatcher = {'berryDeposit': BerryDepositEvent,
                  'berryKickIn': BerryKickInEvent,
                  'carryFood': CarryFoodEvent,
                  'snailEat': SnailEatEvent,
                  'snailEscape': SnailEscapeEvent,
                  'getOnSnail': GetOnSnailEvent,
                  'getOffSnail': GetOffSnailEvent,
                  'glance': GlanceEvent,
                  'playerKill': PlayerKillEvent,
                  'blessMaiden': BlessMaidenEvent,
                  'useMaiden': UseMaidenEvent,
                  'spawn': SpawnEvent,
                  'gamestart': GameStartEvent,
                  'mapstart': MapStartEvent,
                  'victory': VictoryEvent,
                  }
    event_type = raw_event_row['event_type']
    if event_type in skippable_events:
        return None
    assert event_type in dispatcher, f'Unknown event type: {event_type}'

    payload_values = split_payload(raw_event_row['values'])
    event = dispatcher[event_type](payload_values)
    event.timestamp = dateutil.parser.isoparse(raw_event_row['timestamp'])
    event.game_id = int(raw_event_row['game_id'])
    return event


def group_events_by_game_and_normalize_time(events: GameEvents) -> Dict[int, GameEvents]:
    games = {}
    for event in events:
        if event.game_id not in games:
            games[event.game_id] = []
        games[event.game_id].append(event)
    for game_id, game_events in games.items():
        game_events.sort(key=lambda e: e.timestamp)
        start_ts: datetime.datetime = game_events[0].timestamp
        for game_event in game_events:
            game_event.timestamp = (game_event.timestamp - start_ts).total_seconds()
    return games


def debug_print_events(events: GameEvents):
    start_ts = None
    for event in events:
        if start_ts is None:
            start_ts = event.timestamp
        attrs = {a: getattr(event, a) for a in dir(event) if not a.startswith('_')}
        del attrs['game_id']
        attrs['timestamp'] = (event.timestamp - start_ts).total_seconds()
        print(event.__class__.__name__.removesuffix('Event'), attrs)


def get_map_start(events: GameEvents) -> MapStartEvent:
    for event in events:
        if isinstance(event, MapStartEvent):
            return event
    raise ValueError('No map start found in events')


def is_valid_game(events: GameEvents,
                  map_structure_infos: map_structure.MapStructureInfos) -> Optional[GameValidationError]:
    try:
        map_start: MapStartEvent = get_map_start(events)
        map_info: map_structure.MapStructureInfo = map_structure_infos.get_map_info(
            map_start.map, map_start.gold_on_left)

        game_state = GameState(map_info)
        for event in events:
            event.modify_game_state(game_state)
    except GameValidationError as e:
        return e
    return None


def read_events_from_csv(csv_path: str, skip_raw_events_fn=None) -> GameEvents:
    event_reader = csv.DictReader(open(csv_path))
    events = []
    for row in event_reader:
        try:
            if skip_raw_events_fn and skip_raw_events_fn(row):
                continue
            maybe_event = parse_event(row)
        except Exception as e:
            print(f'Failed to parse event: {row}, {e}')
            continue
        if maybe_event is not None:
            events.append(maybe_event)
    return events


def has_bots(events: GameEvents) -> bool:
    for event in events:
        if hasattr(event, 'is_bot') and event.is_bot:
            return True
    return False


def validate_game_data(csv_path):
    events = read_events_from_csv(csv_path)
    grouped_events = group_events_by_game_and_normalize_time(events)

    map_structure_infos = map_structure.MapStructureInfos()
    validation_errors = collections.Counter()
    validated_game_ids = set()
    for game_id, game_events in grouped_events.items():
        maybe_error = is_valid_game(game_events, map_structure_infos)
        # return
        if maybe_error:
            validation_errors[maybe_error.args[0]] += 1
        else:
            if game_events[-1].victory_condition == VictoryCondition.snail:
                validated_game_ids.add(game_id)
                validation_errors['valid'] += 1
        # print('new game')

    print(validation_errors.most_common())

    with open(csv_path, 'r') as f:
        event_reader = csv.DictReader(f)
        fieldnames = event_reader.fieldnames
        with open('snail_' + csv_path, 'w') as output_file:
            event_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            event_writer.writeheader()
            for row in event_reader:
                if int(row['game_id']) in validated_game_ids:
                    event_writer.writerow(row)


if __name__ == '__main__':
    validate_game_data('snail_validated_sampled_events.csv')

