import csv
from enum import Enum
import datetime
import dateutil.parser
from typing import List, Optional, Dict


# https://kqhivemind.com/wiki/Stats_Socket_Events


class GateState(Enum):
    NEUTRAL = 0
    BLUE = 1
    GOLD = 2


class VictoryCondition(Enum):
    military = 'military'
    economic = 'economic'
    snail = 'snail'


class Team(Enum):
    Blue = 'Blue'
    Gold = 'Gold'


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


class TeamState:
    def __init__(self):
        self.eggs = 2
        self.berries_deposited = 0  # TODO: Better model easy/hard berries, eg on Dusk or Twilight.
        self.drones = [DroneState() for _ in range(4)]


class GameState:
    def __init__(self, map_name):
        self.teams = [TeamState() for _ in range(2)]
        self.berries_available = 0



def split_payload(payload: str) -> List[str]:
    assert payload.startswith('{')
    assert payload.endswith('}')
    return payload[1:-1].split(',')


class GameEvent:
    def __init__(self):
        self.timestamp = None
        self.game_id = None


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


class BerryKickInEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        super().__init__()
        self.drone_x = int(payload_values[0])
        self.drone_y = int(payload_values[1])
        self.position_id = int(payload_values[2])
        self.counts_for_own_team = payload_values[3] == 'True'


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
        self.winning_team = Team[payload_values[0]]
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


def group_events_by_game(events: List[GameEvent]) -> Dict[int, List[GameEvent]]:
    games = {}
    for event in events:
        if event.game_id not in games:
            games[event.game_id] = []
        games[event.game_id].append(event)
    return games


def main():
    event_reader = csv.DictReader(open('match_data/export_20220928_183155_gdc6/gameevent.csv'))
    for row in event_reader:
        parse_event(row)



if __name__ == '__main__':
    main()
