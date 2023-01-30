import csv
from enum import Enum
from typing import List, Optional
# https://kqhivemind.com/wiki/Stats_Socket_Events


class GateState(Enum):
    NEUTRAL = 0
    BLUE = 1
    GOLD = 2



def split_payload(payload: str) -> List[str]:
    assert payload.startswith('{')
    assert payload.endswith('}')
    return payload[1:-1].split(',')


class GameEvent:
    pass


class BerryDepositEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.drone_x = int(payload_values[0])
        self.drone_y = int(payload_values[1])
        self.position_id = int(payload_values[2])


class BerryKickInEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.drone_x = int(payload_values[0])
        self.drone_y = int(payload_values[1])
        self.position_id = int(payload_values[2])
        self.counts_for_own_team = payload_values[3] == 'True'


class BlessMaidenEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.gate_x = int(payload_values[0])
        self.gate_y = int(payload_values[1])
        self.gate_color = GateState.BLUE if payload_values[2] == 'Blue' else GateState.GOLD


class CarryFoodEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.position_id = int(payload_values[0])


class GetOnSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.snail_x = int(payload_values[0])
        self.snail_y = int(payload_values[1])
        self.position_id = int(payload_values[2])


class GetOffSnailEvent(GameEvent):
    def __init__(self, payload_values: List[str]):
        self.snail_x = int(payload_values[0])
        self.snail_y = int(payload_values[1])
        self.position_id = int(payload_values[3])


def parse_event(raw_event_row) -> Optional[GameEvent]:
    skippable_events = {'gameend', 'gamestart'}
    dispatcher = {'berryDeposit': BerryDepositEvent,
                  'berryKickIn': BerryKickInEvent,
                  'blessMaiden': BlessMaidenEvent,
                  'carryFood': CarryFoodEvent,
                  'getOnSnail': GetOnSnailEvent,
                  'getOffSnail': GetOffSnailEvent,
                  }
    event_type = raw_event_row['event_type']
    if event_type in skippable_events:
        return None
    assert event_type in dispatcher
    payload_values = split_payload(raw_event_row['values'])
    return dispatcher[event_type](payload_values)


def main():
  event_reader = csv.DictReader(open('match_data/export_20220928_183155_gdc6/gameevent.csv'))
  for row in event_reader:
      print(row)
      break



if __name__ == '__main__':
    main()
