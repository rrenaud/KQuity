"""Compact binary encoding for game events.

Encodes game events into a compact binary format (~2-3 bytes/event).
Provides walk_game_states() to decode and apply mutations directly to GameState.

Binary format:
  [header: 1 byte] [event...]

Header byte: (map_idx << 1) | gold_on_left
  map_idx: 0=day, 1=night, 2=dusk, 3=twilight

Each event:
  [ts_delta: 1-2 bytes] [opcode+payload: 1-3 bytes]

Timestamp delta: centiseconds since previous event.
  < 128: 1 byte (high bit 0)
  else: 2 bytes (0x80 | hi7, lo8) for 15-bit value up to 32767cs (~327s)

Games with any event gap > 60 seconds are rejected (return None).

Opcode+payload packing:
  1-byte: [opcode:4][payload:4]
  2-byte: [opcode:4][payload_hi:4][payload_lo:8]  (12 bits payload)
  3-byte: [opcode:4][payload_hi:4][mid:8][lo:8]   (20 bits payload)
"""

import json
import os
from typing import Iterator, Tuple

from constants import Team, ContestableState, VictoryCondition, Map
from preprocess import GameState, position_id_to_team, position_id_to_worker_index
from fast_materialize import (
    _MAP_LOOKUPS, MAP_INDEX, MAP_NAMES, SCREEN_WIDTH,
    SKIP_EVENTS, VANILLA_SNAIL_PPS, SPEED_SNAIL_PPS, _snail_mult,
    _parse_ts, COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
)
import map_structure

# --- Opcodes ---
OP_GAMESTART = 0
OP_MAPSTART = 1
OP_SPAWN = 2
OP_CARRY_FOOD = 3
OP_BERRY_DEPOSIT = 4
OP_BERRY_KICK_IN = 5
OP_BLESS_MAIDEN = 6
OP_USE_MAIDEN = 7
OP_GET_ON_SNAIL = 8
OP_SNAIL_EAT = 9
OP_GET_OFF_SNAIL = 10
OP_SNAIL_ESCAPE = 11
OP_PLAYER_KILL = 12
OP_VICTORY = 13

_EVENT_TYPE_TO_OPCODE = {
    'gamestart': OP_GAMESTART,
    'mapstart': OP_MAPSTART,
    'spawn': OP_SPAWN,
    'carryFood': OP_CARRY_FOOD,
    'berryDeposit': OP_BERRY_DEPOSIT,
    'berryKickIn': OP_BERRY_KICK_IN,
    'blessMaiden': OP_BLESS_MAIDEN,
    'useMaiden': OP_USE_MAIDEN,
    'getOnSnail': OP_GET_ON_SNAIL,
    'snailEat': OP_SNAIL_EAT,
    'getOffSnail': OP_GET_OFF_SNAIL,
    'snailEscape': OP_SNAIL_ESCAPE,
    'playerKill': OP_PLAYER_KILL,
    'victory': OP_VICTORY,
}

# Bytes per opcode+payload (excluding timestamp delta)
_OPCODE_SIZES = [
    1,  # 0: gamestart
    1,  # 1: mapstart
    2,  # 2: spawn
    1,  # 3: carryFood
    2,  # 4: berryDeposit
    2,  # 5: berryKickIn
    1,  # 6: blessMaiden
    2,  # 7: useMaiden
    3,  # 8: getOnSnail
    3,  # 9: snailEat
    3,  # 10: getOffSnail
    3,  # 11: snailEscape
    2,  # 12: playerKill
    1,  # 13: victory
]

_VICTORY_COND_TO_INT = {'military': 0, 'economic': 1, 'snail': 2}
_INT_TO_VICTORY_COND = [
    VictoryCondition.military, VictoryCondition.economic, VictoryCondition.snail
]


def _build_maiden_type_lookup():
    json_path = os.path.join(os.path.dirname(__file__), 'map_structure_info.json')
    with open(json_path, 'rb') as f:
        raw = json.load(f)
    return {
        name: [m[0] for m in info['maiden_info']]
        for name, info in raw.items()
    }


_MAIDEN_TYPES = _build_maiden_type_lookup()

_MAP_STRUCTURE_INFOS = None


def _get_map_structure_infos():
    global _MAP_STRUCTURE_INFOS
    if _MAP_STRUCTURE_INFOS is None:
        _MAP_STRUCTURE_INFOS = map_structure.MapStructureInfos()
    return _MAP_STRUCTURE_INFOS


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_game(raw_events):
    """Encode a game's events into compact binary.

    Args:
        raw_events: list of (datetime, event_type, values_str) tuples
                    (same format as fast_materialize._process_game input)

    Returns:
        bytes, or None if game is invalid (missing gamestart/mapstart)
    """
    raw_events.sort(key=lambda x: x[0])

    gamestart_dt = None
    map_name = None
    gold_on_left = None

    for dt, event_type, values_str in raw_events:
        if event_type == 'gamestart' and gamestart_dt is None:
            gamestart_dt = dt
        if event_type == 'mapstart' and map_name is None:
            vals = values_str[1:-1].split(',')
            map_name = vals[0]
            gold_on_left = (vals[1] == 'True')

    if gamestart_dt is None or map_name is None:
        return None

    map_idx = MAP_INDEX[map_name]
    map_lookup = _MAP_LOOKUPS.get((map_name, gold_on_left))
    if map_lookup is None:
        return None

    berry_lookup = map_lookup['berry_lookup']
    maiden_lookup = map_lookup['maiden_lookup']

    buf = bytearray([(map_idx << 1) | int(gold_on_left)])
    last_cs = 0

    for dt, event_type, values_str in raw_events:
        opcode = _EVENT_TYPE_TO_OPCODE.get(event_type)
        if opcode is None:
            continue

        rel_ts = (dt - gamestart_dt).total_seconds()
        # Clamp to non-negative: pre-gamestart events all encode as t=0
        cs = max(0, int(round(rel_ts * 100)))
        delta = max(0, cs - last_cs)
        last_cs = cs

        if delta > 6000:  # > 60 seconds between events — reject anomalous game
            return None
        if delta < 128:
            buf.append(delta)
        else:
            buf.append(0x80 | (delta >> 8))
            buf.append(delta & 0xFF)

        vals = values_str[1:-1].split(',') if values_str and len(values_str) > 2 else []

        if opcode == OP_GAMESTART or opcode == OP_MAPSTART:
            payload = (map_idx << 1) | int(gold_on_left)
            buf.append((opcode << 4) | (payload & 0xF))

        elif opcode == OP_SPAWN:
            pid = int(vals[0])
            is_bot = int(vals[1] == 'True')
            payload = (pid << 1) | is_bot
            buf.append((opcode << 4) | ((payload >> 8) & 0xF))
            buf.append(payload & 0xFF)

        elif opcode == OP_CARRY_FOOD:
            pid = int(vals[0])
            buf.append((opcode << 4) | (pid & 0xF))

        elif opcode == OP_BERRY_DEPOSIT:
            hole_x, hole_y = int(vals[0]), int(vals[1])
            pid = int(vals[2])
            bi = berry_lookup[(hole_x, hole_y)]
            payload = (bi << 4) | pid
            buf.append((opcode << 4) | ((payload >> 8) & 0xF))
            buf.append(payload & 0xFF)

        elif opcode == OP_BERRY_KICK_IN:
            hole_x, hole_y = int(vals[0]), int(vals[1])
            pid = int(vals[2])
            own = int(vals[3] == 'True')
            bi = berry_lookup[(hole_x, hole_y)]
            payload = (bi << 5) | (pid << 1) | own
            buf.append((opcode << 4) | ((payload >> 8) & 0xF))
            buf.append(payload & 0xFF)

        elif opcode == OP_BLESS_MAIDEN:
            mx, my = int(vals[0]), int(vals[1])
            color = int(vals[2] == 'Blue')  # 1=blue, 0=gold
            _, midx = maiden_lookup[(mx, my)]
            payload = (midx << 1) | color
            buf.append((opcode << 4) | (payload & 0xF))

        elif opcode == OP_USE_MAIDEN:
            pid = int(vals[3])
            mx, my = int(vals[0]), int(vals[1])
            _, midx = maiden_lookup[(mx, my)]
            payload = (midx << 4) | pid
            buf.append((opcode << 4) | ((payload >> 8) & 0xF))
            buf.append(payload & 0xFF)

        elif opcode == OP_GET_ON_SNAIL:
            sx = int(vals[0])
            rider_pid = int(vals[2])
            payload = (sx << 4) | rider_pid
            buf.append((opcode << 4) | ((payload >> 16) & 0xF))
            buf.append((payload >> 8) & 0xFF)
            buf.append(payload & 0xFF)

        elif opcode == OP_SNAIL_EAT:
            sx = int(vals[0])
            rider_pid = int(vals[2])
            eaten_pid = int(vals[3])
            payload = (sx << 8) | (rider_pid << 4) | eaten_pid
            buf.append((opcode << 4) | ((payload >> 16) & 0xF))
            buf.append((payload >> 8) & 0xFF)
            buf.append(payload & 0xFF)

        elif opcode == OP_GET_OFF_SNAIL:
            sx = int(vals[0])
            pid = int(vals[3])
            payload = (sx << 4) | pid
            buf.append((opcode << 4) | ((payload >> 16) & 0xF))
            buf.append((payload >> 8) & 0xFF)
            buf.append(payload & 0xFF)

        elif opcode == OP_SNAIL_ESCAPE:
            sx = int(vals[0])
            escaped_pid = int(vals[2])
            payload = (sx << 4) | escaped_pid
            buf.append((opcode << 4) | ((payload >> 16) & 0xF))
            buf.append((payload >> 8) & 0xFF)
            buf.append(payload & 0xFF)

        elif opcode == OP_PLAYER_KILL:
            killer_pid = int(vals[2])
            killed_pid = int(vals[3])
            payload = (killer_pid << 4) | killed_pid
            buf.append((opcode << 4) | ((payload >> 8) & 0xF))
            buf.append(payload & 0xFF)

        elif opcode == OP_VICTORY:
            team_int = int(vals[0] == 'Gold')  # 0=blue, 1=gold
            cond = _VICTORY_COND_TO_INT[vals[1]]
            payload = (team_int << 2) | cond
            buf.append((opcode << 4) | (payload & 0xF))

    return bytes(buf)


# ---------------------------------------------------------------------------
# Decoder / walker
# ---------------------------------------------------------------------------

def walk_game_states(encoded_bytes):
    """Decode binary events and yield game states.

    Yields (rel_ts, game_state) BEFORE each event's mutation is applied,
    matching the vectorize-before-mutate pattern in fast_materialize.

    The game_state is the SAME object mutated each iteration; callers must
    copy or vectorize before advancing the iterator.
    """
    data = encoded_bytes
    pos = 0

    header = data[pos]; pos += 1
    map_idx = (header >> 1) & 0x3
    gold_on_left = bool(header & 1)
    map_name = MAP_NAMES[map_idx]

    map_infos = _get_map_structure_infos()
    map_info = map_infos.get_map_info(Map[map_name], gold_on_left)
    game_state = GameState(map_info)
    maiden_types = _MAIDEN_TYPES[map_name]

    current_cs = 0

    while pos < len(data):
        # Timestamp delta
        b = data[pos]; pos += 1
        if b & 0x80:
            b2 = data[pos]; pos += 1
            delta = ((b & 0x7F) << 8) | b2
        else:
            delta = b
        current_cs += delta
        rel_ts = current_cs / 100.0

        # Opcode + payload
        b0 = data[pos]; pos += 1
        opcode = b0 >> 4
        sz = _OPCODE_SIZES[opcode]

        if sz == 1:
            payload = b0 & 0xF
        elif sz == 2:
            b1 = data[pos]; pos += 1
            payload = ((b0 & 0xF) << 8) | b1
        else:
            b1 = data[pos]; pos += 1
            b2 = data[pos]; pos += 1
            payload = ((b0 & 0xF) << 16) | (b1 << 8) | b2

        # Yield state BEFORE mutation
        yield (rel_ts, game_state)

        # Apply mutation
        if opcode == OP_SPAWN:
            pid = payload >> 1
            is_bot = bool(payload & 1)
            team = pid % 2
            widx = (pid - 3) // 2
            game_state.teams[team].workers[widx].is_bot = is_bot

        elif opcode == OP_CARRY_FOOD:
            pid = payload
            team = pid % 2
            widx = (pid - 3) // 2
            game_state.teams[team].workers[widx].has_food = True

        elif opcode == OP_BERRY_DEPOSIT:
            bi = payload >> 4
            pid = payload & 0xF
            team = pid % 2
            widx = (pid - 3) // 2
            game_state.teams[team].workers[widx].has_food = False
            game_state.teams[team].food_deposited[bi] = True
            game_state.berries_available -= 1

        elif opcode == OP_BERRY_KICK_IN:
            bi = payload >> 5
            pid = (payload >> 1) & 0xF
            own = payload & 1
            team = pid % 2
            if not own:
                team = 1 - team
            game_state.teams[team].food_deposited[bi] = True
            game_state.berries_available -= 1

        elif opcode == OP_BLESS_MAIDEN:
            midx = payload >> 1
            is_blue = payload & 1
            game_state.maiden_states[midx] = (
                ContestableState.BLUE if is_blue else ContestableState.GOLD
            )

        elif opcode == OP_USE_MAIDEN:
            midx = payload >> 4
            pid = payload & 0xF
            team = pid % 2
            widx = (pid - 3) // 2
            worker = game_state.teams[team].workers[widx]
            if maiden_types[midx] == 'maiden_speed':
                worker.has_speed = True
            else:
                worker.has_wings = True
            worker.has_food = False

        elif opcode == OP_GET_ON_SNAIL:
            snail_x = payload >> 4
            rider_pid = payload & 0xF
            _apply_start_snail(game_state, snail_x, rider_pid, rel_ts, gold_on_left)

        elif opcode == OP_SNAIL_EAT:
            snail_x = payload >> 8
            rider_pid = (payload >> 4) & 0xF
            _apply_start_snail(game_state, snail_x, rider_pid, rel_ts, gold_on_left)

        elif opcode == OP_GET_OFF_SNAIL:
            snail_x = payload >> 4
            _apply_stop_snail(game_state, snail_x, rel_ts)

        elif opcode == OP_SNAIL_ESCAPE:
            snail_x = payload >> 4
            _apply_stop_snail(game_state, snail_x, rel_ts)

        elif opcode == OP_PLAYER_KILL:
            killed_pid = payload & 0xF
            team = killed_pid % 2
            if killed_pid <= 2:  # Queen
                game_state.teams[team].eggs -= 1
            else:
                widx = (killed_pid - 3) // 2
                w = game_state.teams[team].workers[widx]
                w.has_food = False
                w.has_speed = False
                w.has_wings = False

        # gamestart, mapstart, victory: no state mutation


def _apply_start_snail(game_state, snail_x, rider_pid, rel_ts, gold_on_left):
    rider_team = rider_pid % 2
    rider_widx = (rider_pid - 3) // 2
    has_speed = game_state.teams[rider_team].workers[rider_widx].has_speed
    base_speed = SPEED_SNAIL_PPS if has_speed else VANILLA_SNAIL_PPS
    game_state.snail_state.snail_x = float(snail_x)
    game_state.snail_state.snail_velocity = base_speed * _snail_mult(gold_on_left, rider_team)
    game_state.snail_state.last_touch_timestamp = rel_ts


def _apply_stop_snail(game_state, snail_x, rel_ts):
    game_state.snail_state.snail_x = float(snail_x)
    game_state.snail_state.snail_velocity = 0.0
    game_state.snail_state.last_touch_timestamp = rel_ts


# ---------------------------------------------------------------------------
# Multi-game binary file I/O
# ---------------------------------------------------------------------------
#
# Format: [num_games: uint32 LE]
#         per game: [game_id: uint32 LE] [length: uint16 LE] [binary payload]

import struct

_HEADER_FMT = '<I'       # num_games
_ENTRY_FMT = '<IH'       # game_id, payload_length
_ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)


def write_packed_games(entries, path):
    """Write a list of (game_id, encoded_bytes) to a packed binary file."""
    with open(path, 'wb') as f:
        f.write(struct.pack(_HEADER_FMT, len(entries)))
        for game_id, data in entries:
            f.write(struct.pack(_ENTRY_FMT, game_id, len(data)))
            f.write(data)


def read_packed_games(path):
    """Read a packed binary file, yielding (game_id, encoded_bytes) pairs."""
    with open(path, 'rb') as f:
        (num_games,) = struct.unpack(_HEADER_FMT, f.read(4))
        for _ in range(num_games):
            game_id, length = struct.unpack(_ENTRY_FMT, f.read(_ENTRY_SIZE))
            data = f.read(length)
            yield game_id, data


def encode_csv_to_packed(csv_path, out_path):
    """Encode all games from CSV/gzip files into a packed binary file.

    Returns (encoded_count, rejected_count).
    """
    import csv as csv_mod
    import glob
    import gzip

    games = {}
    game_order = []
    for filename in sorted(glob.glob(csv_path)):
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            reader = csv_mod.reader(f)
            next(reader)
            for row in reader:
                event_type = row[COL_TYPE]
                if event_type in SKIP_EVENTS:
                    continue
                game_id = int(row[COL_GAME_ID])
                if game_id not in games:
                    games[game_id] = []
                    game_order.append(game_id)
                games[game_id].append(
                    (_parse_ts(row[COL_TS]), event_type, row[COL_VALUES]))

    entries = []
    rejected = 0
    for game_id in game_order:
        encoded = encode_game(list(games[game_id]))
        if encoded is None:
            rejected += 1
            continue
        entries.append((game_id, encoded))

    write_packed_games(entries, out_path)
    return len(entries), rejected
