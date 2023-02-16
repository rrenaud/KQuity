import csv
from enum import Enum
import datetime
import dateutil.parser
import numpy as np
from typing import List, Optional, Dict


# https://kqhivemind.com/wiki/Stats_Socket_Events


class GateState(Enum):
    BLUE = 0
    GOLD = 1
    NEUTRAL = 2


class Team(Enum):
    BLUE = 0
    GOLD = 1


def opposing_team(team: Team) -> Team:
    return Team.GOLD if team == Team.BLUE else Team.BLUE


class VictoryCondition(Enum):
    military = 'military'
    economic = 'economic'
    snail = 'snail'


class PlayerCategory(Enum):
    Queen = 'Queen'
    Soldier = 'Soldier'
    Worker = 'Worker'


class MaidenType(Enum):
    maiden_speed = 'maiden_speed'
    maiden_wings = 'maiden_wings'


class Map(Enum):
    map_day = 'map_day'
    map_night = 'map_night'
    map_dusk = 'map_dusk'
    map_twilight = 'map_twilight'


class DroneState:
    def __init__(self):
        self.has_speed = False
        self.has_wings = False
        self.has_berry = False


class TeamState:
    def __init__(self):
        self.eggs = 2
        self.berries_deposited = 0  # TODO: Better model easy/hard berries, eg on Dusk or Twilight.
        self.drones = [DroneState() for _ in range(4)]


class GameState:
    def __init__(self, map_name):
        self.teams = [TeamState() for _ in range(2)]
        self.berries_available = 0
        self.gate_states = [GateState.NEUTRAL for _ in range(5)]

    def get_team(self, team: Team) -> TeamState:
        return self.teams[team.value]


class GameEvent:
    def __init__(self):
        self.timestamp = None
        self.game_id = None

    def modify_game_state(self, game_state: GameState):
        pass


GameEvents = List[GameEvent]


def position_id_to_team(position_id: int) -> Team:
    return Team(position_id % 2)


def position_id_to_drone_index(position_id: int) -> int:
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
        self.game_version = payload_values[4]


class BerryDepositEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.drone_x = int(payload_values[0])
        self.drone_y = int(payload_values[1])
        self.position_id = int(payload_values[2])

    def modify_game_state(self, game_state: GameState):
        team: Team = position_id_to_team(self.position_id)
        drone_index = position_id_to_drone_index(self.position_id)
        team_state: TeamState = game_state.get_team(team)
        team_state.drones[drone_index].has_berry = False
        team_state.berries_deposited += 1
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

        game_state.get_team(team).berries_deposited += 1
        game_state.berries_available -= 1


class BlessMaidenEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.gate_x = int(payload_values[0])
        self.gate_y = int(payload_values[1])
        self.gate_color = GateState.BLUE if payload_values[2] == 'Blue' else GateState.GOLD


class CarryFoodEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.position_id = int(payload_values[0])


class GetOnSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.snail_y = int(payload_values[1])
        self.position_id = int(payload_values[2])


class GetOffSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.snail_x = int(payload_values[0])
        self.snail_y = int(payload_values[1])
        self.position_id = int(payload_values[3])


class GlanceEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
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


class UseMaidenEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.maiden_x = int(payload_values[0])
        self.maiden_y = int(payload_values[1])
        self.maiden_type = MaidenType[payload_values[2]]
        self.position_id = int(payload_values[3])


class VictoryEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.winning_team = Team.BLUE if payload_values[0] == 'Blue' else Team.GOLD
        self.victory_condition = VictoryCondition[payload_values[1]]


def parse_event(raw_event_row) -> Optional[GameEvent]:
    skippable_events = {'gameend', 'playernames', 'spawn',
                        'reserveMaiden', 'unreserveMaiden', 'snailEat',
                        'snailEscape'}
    dispatcher = {'berryDeposit': BerryDepositEvent,
                  'berryKickIn': BerryKickInEvent,
                  'carryFood': CarryFoodEvent,
                  'getOnSnail': GetOnSnailEvent,
                  'getOffSnail': GetOffSnailEvent,
                  'glance': GlanceEvent,
                  'playerKill': PlayerKillEvent,
                  'blessMaiden': BlessMaidenEvent,
                  'useMaiden': UseMaidenEvent,
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


def group_events_by_game(events: GameEvents) -> Dict[int, GameEvents]:
    games = {}
    for event in events:
        if event.game_id not in games:
            games[event.game_id] = []
        games[event.game_id].append(event)
    for game_id, game_events in games.items():
        game_events.sort(key=lambda e: e.timestamp)
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


def get_map(events: GameEvents) -> Map:
    for event in events:
        if hasattr(event, 'map_name'):
            return event.map_name
    raise ValueError('No map found in events')


def vectorize_game_states(events: GameEvents) -> np.ndarray:
    game_state = GameState(get_map(events))
    for event in events:
        event.modify_game_state(game_state)



def main():
    event_reader = csv.DictReader(open('match_data/export_20220928_183155_gdc6/gameevent.csv'))
    events = []
    for row in event_reader:
        maybe_event = parse_event(row)
        if maybe_event is not None:
            events.append(maybe_event)

    grouped_events = group_events_by_game(events)
    #debug_print_events(next(iter(grouped_events.values())))
    vectorize_events(next(iter(grouped_events.values())))

if __name__ == '__main__':
    main()
