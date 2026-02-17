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

import glob
import json
import os
import struct
from typing import Iterator, Tuple

from constants import Team, ContestableState, VictoryCondition, Map
from preprocess import GameState, position_id_to_team, position_id_to_worker_index
from fast_materialize import (
    _MAP_LOOKUPS, MAP_INDEX, MAP_NAMES, SCREEN_WIDTH,
    SKIP_EVENTS, VANILLA_SNAIL_PPS, SPEED_SNAIL_PPS, _snail_mult,
    _parse_ts, COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
    _vectorize_state, NUM_FEATURES,
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

    try:
        return _encode_events(raw_events, gamestart_dt, map_idx, gold_on_left,
                              berry_lookup, maiden_lookup)
    except KeyError:
        return None


def _encode_events(raw_events, gamestart_dt, map_idx, gold_on_left,
                   berry_lookup, maiden_lookup):
    """Inner encoding loop. Raises KeyError on unmapped positions."""
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

    After the walk completes, game_state.winning_team and
    game_state.victory_condition are set (if a victory event was present).
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
        if opcode >= len(_OPCODE_SIZES):
            raise ValueError(f"Invalid opcode {opcode} at byte {pos - 1}")
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

        elif opcode == OP_VICTORY:
            team_int = payload >> 2
            cond_int = payload & 0x3
            # NB: This mutation happens AFTER the final yield, so callers
            # must fully exhaust the generator to get a valid winning_team.
            game_state.winning_team = Team.GOLD if team_int else Team.BLUE
            game_state.victory_condition = _INT_TO_VICTORY_COND[cond_int]

        # gamestart, mapstart: no state mutation


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

_HEADER_FMT = '<I'       # num_games
_ENTRY_FMT = '<IH'       # game_id (uint32, max ~4.3B), payload_length (uint16, max 65535 bytes)
_ENTRY_SIZE = struct.calcsize(_ENTRY_FMT)


def write_packed_games(entries, path):
    """Write a list of (game_id, encoded_bytes) to a packed binary file.

    Games with payloads exceeding uint16 max (65535 bytes) are skipped.
    """
    skipped_ids = []
    with open(path, 'wb') as f:
        # Write placeholder count, patch after
        count_pos = f.tell()
        f.write(struct.pack(_HEADER_FMT, 0))
        written = 0
        for game_id, data in entries:
            if len(data) > 65535:
                skipped_ids.append(game_id)
                continue
            f.write(struct.pack(_ENTRY_FMT, game_id, len(data)))
            f.write(data)
            written += 1
        # Patch actual count
        f.seek(count_pos)
        f.write(struct.pack(_HEADER_FMT, written))
    if skipped_ids:
        import sys
        print(f"  Warning: {len(skipped_ids)} games skipped (payload > 65535 bytes): "
              f"{skipped_ids}", file=sys.stderr)


def read_packed_games(path):
    """Read a packed binary file, yielding (game_id, encoded_bytes) pairs."""
    with open(path, 'rb') as f:
        (num_games,) = struct.unpack(_HEADER_FMT, f.read(4))
        for _ in range(num_games):
            game_id, length = struct.unpack(_ENTRY_FMT, f.read(_ENTRY_SIZE))
            data = f.read(length)
            yield game_id, data


def _read_csv_games(csv_glob):
    """Read games from CSV/gzip files into a dict keyed by game_id.

    Returns:
        games: dict {game_id: [(timestamp, event_type, values_str), ...]}.
        game_order: list of game_ids in first-seen order.
    """
    import csv as csv_mod
    import gzip

    games = {}
    game_order = []
    for filename in sorted(glob.glob(csv_glob)):
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
    return games, game_order


def _encode_games(games, game_order):
    """Encode parsed games into binary, returning (entries, rejected_count).

    entries is a list of (game_id, encoded_bytes) for successfully encoded games.
    """
    entries = []
    rejected = 0
    for game_id in game_order:
        encoded = encode_game(list(games[game_id]))
        if encoded is None:
            rejected += 1
            continue
        entries.append((game_id, encoded))
    return entries, rejected


def encode_csv_to_packed(csv_path, out_path):
    """Encode all games from CSV/gzip files into a packed binary file.

    Returns (encoded_count, rejected_count).
    """
    games, game_order = _read_csv_games(csv_path)
    entries, rejected = _encode_games(games, game_order)
    write_packed_games(entries, out_path)
    return len(entries), rejected


def reshard_packed_games(input_path, output_dir, num_shards):
    """Split a packed .bin file into N shard files using partial prefix reads.

    Intended for migrating existing single .bin files into the sharded format.
    For new data, prefer encode_csv_to_sharded_bins() which shards directly.

    Scans entry headers to build an offset table, then copies byte ranges
    directly without buffering all payloads in memory. Preserves input order,
    so sortedness of the original file is maintained within and across shards.

    Each shard is written as shard_NNN.bin in output_dir.
    Returns list of shard paths.
    """
    # Phase 1: scan headers to get game count and byte offsets of each entry
    entry_offsets = []  # (file_offset, entry_total_bytes) for each game
    with open(input_path, 'rb') as f:
        (num_games,) = struct.unpack(_HEADER_FMT, f.read(4))
        for _ in range(num_games):
            entry_start = f.tell()
            _game_id, length = struct.unpack(_ENTRY_FMT, f.read(_ENTRY_SIZE))
            f.seek(length, 1)  # skip payload
            entry_offsets.append((entry_start, _ENTRY_SIZE + length))

    os.makedirs(output_dir, exist_ok=True)
    # Round-robin (striped) assignment so each shard gets an even slice of
    # the input ordering.
    shard_buckets = [[] for _ in range(num_shards)]
    for idx, offset_entry in enumerate(entry_offsets):
        shard_buckets[idx % num_shards].append(offset_entry)
    shard_paths = []

    # Phase 2: copy byte ranges into shard files
    with open(input_path, 'rb') as f:
        for i in range(num_shards):
            if not shard_buckets[i]:
                continue
            path = os.path.join(output_dir, f'shard_{i:03d}.bin')
            with open(path, 'wb') as out:
                out.write(struct.pack(_HEADER_FMT, len(shard_buckets[i])))
                for file_offset, entry_bytes in shard_buckets[i]:
                    f.seek(file_offset)
                    out.write(f.read(entry_bytes))
            shard_paths.append(path)

    return shard_paths


def encode_csv_to_sharded_bins(csv_glob, output_dir, num_shards):
    """Encode all CSV partition files into N binary shards directly.

    Returns (total_encoded, total_rejected, shard_paths).
    """
    games, game_order = _read_csv_games(csv_glob)
    entries, rejected = _encode_games(games, game_order)

    os.makedirs(output_dir, exist_ok=True)
    # Round-robin (striped) assignment so each shard gets an even slice of
    # the quality ordering.  Game 0 → shard 0, game 1 → shard 1, etc.
    shard_buckets = [[] for _ in range(num_shards)]
    for idx, entry in enumerate(entries):
        shard_buckets[idx % num_shards].append(entry)
    shard_paths = []
    for i in range(num_shards):
        if not shard_buckets[i]:
            continue
        path = os.path.join(output_dir, f'shard_{i:03d}.bin')
        write_packed_games(shard_buckets[i], path)
        shard_paths.append(path)

    return len(entries), rejected, shard_paths


# ---------------------------------------------------------------------------
# Materialization from binary codec
# ---------------------------------------------------------------------------

def _game_state_to_vectorize_args(game_state):
    """Extract raw variables from GameState for _vectorize_state."""
    w = [[[False, False, False, False] for _ in range(4)] for _ in range(2)]
    eggs = [0, 0]
    food_count = [0, 0]

    for t in range(2):
        ts = game_state.teams[t]
        eggs[t] = ts.eggs
        food_count[t] = sum(ts.food_deposited)
        for wi in range(4):
            ws = ts.workers[wi]
            w[t][wi] = [ws.is_bot, ws.has_food, ws.has_speed, ws.has_wings]

    maiden_map = {
        ContestableState.NEUTRAL: 0,
        ContestableState.BLUE: 1,
        ContestableState.GOLD: -1,
    }
    maiden_states = [maiden_map[m] for m in game_state.maiden_states]
    map_idx = MAP_INDEX[game_state.map_info.map_id.value]
    gold_sym = 1.0 if game_state.map_info.gold_on_left else -1.0

    return (w, eggs, food_count, maiden_states, map_idx,
            game_state.snail_state.snail_x,
            game_state.snail_state.snail_velocity,
            game_state.snail_state.last_touch_timestamp,
            game_state.berries_available, gold_sym)


def materialize_entries(entries, drop_state_probability=0.0, seed=42):
    """Materialize features from an iterable of (game_id, encoded_bytes).

    This is the shared materialization core used by both
    fast_materialize_from_codec (file-backed) and callers that pre-load
    entries into memory.

    Args:
        entries: iterable of (game_id, encoded_bytes) tuples.
        drop_state_probability: Probability of dropping each eligible state.
        seed: RNG seed for state dropping (default: 42).

    Returns:
        (states, labels, game_ids, timestamps): numpy arrays.
    """
    import random
    import numpy as np

    capacity = 100_000
    output_buf = np.empty((capacity, NUM_FEATURES), dtype=np.float32)
    label_buf = np.empty(capacity, dtype=np.int8)
    game_id_buf = np.empty(capacity, dtype=np.int64)
    timestamp_buf = np.empty(capacity, dtype=np.float32)
    write_idx = 0

    rng = random.Random(seed)
    no_drop = (drop_state_probability == 0.0)
    n_failures = 0

    def _grow_buffers(needed):
        """Grow buffers by 50% (not 2x) to limit peak memory overhead."""
        nonlocal capacity, output_buf, label_buf, game_id_buf, timestamp_buf
        new_cap = max(capacity + capacity // 2, needed)
        new_out = np.empty((new_cap, NUM_FEATURES), dtype=np.float32)
        new_out[:write_idx] = output_buf[:write_idx]
        output_buf = new_out
        new_lbl = np.empty(new_cap, dtype=np.int8)
        new_lbl[:write_idx] = label_buf[:write_idx]
        label_buf = new_lbl
        new_gid = np.empty(new_cap, dtype=np.int64)
        new_gid[:write_idx] = game_id_buf[:write_idx]
        game_id_buf = new_gid
        new_ts = np.empty(new_cap, dtype=np.float32)
        new_ts[:write_idx] = timestamp_buf[:write_idx]
        timestamp_buf = new_ts
        capacity = new_cap

    for game_id, encoded in entries:
        start_idx = write_idx
        game_state = None

        try:
            for rel_ts, game_state in walk_game_states(encoded):
                if rel_ts > 5.0 and (no_drop or rng.random() > drop_state_probability):
                    if write_idx >= capacity:
                        _grow_buffers(write_idx + 1)
                    args = _game_state_to_vectorize_args(game_state)
                    (w, eggs, food_count, maiden_states, map_idx,
                     snail_x, snail_vel, snail_last_ts,
                     berries_avail, gold_sym) = args
                    _vectorize_state(output_buf, write_idx, w, eggs, food_count,
                                     maiden_states, map_idx, snail_x,
                                     snail_vel, snail_last_ts,
                                     rel_ts, berries_avail, gold_sym)
                    timestamp_buf[write_idx] = rel_ts
                    write_idx += 1
        except Exception as e:
            import sys
            print(f"  Warning: game {game_id} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            n_failures += 1
            write_idx = start_idx
            continue

        # Set labels from final game state
        if game_state is None or game_state.winning_team is None:
            write_idx = start_idx
            continue

        label = 1 if game_state.winning_team == Team.BLUE else 0
        label_buf[start_idx:write_idx] = label
        game_id_buf[start_idx:write_idx] = game_id

    if n_failures > 0:
        import sys
        print(f"  Warning: {n_failures} games failed during materialization",
              file=sys.stderr)

    return (output_buf[:write_idx], label_buf[:write_idx],
            game_id_buf[:write_idx], timestamp_buf[:write_idx])


def fast_materialize_from_codec(bin_path, drop_state_probability=0.0,
                                max_games=None, exclude_game_ids=None,
                                seed=42):
    """Materialize features from a packed binary file.

    Args:
        bin_path: Path to a packed binary file (from encode_csv_to_packed).
        drop_state_probability: Probability of dropping each eligible state.
        max_games: If set, only process this many games from the file.
            Excluded games do not count toward this limit.
        exclude_game_ids: Set of game IDs to skip entirely.
        seed: RNG seed for state dropping (default: 42).

    Returns:
        (states, labels, game_ids, timestamps): numpy arrays matching
        fast_materialize() output format.
    """
    def _iter_entries():
        games_processed = 0
        for game_id, encoded in read_packed_games(bin_path):
            if exclude_game_ids and game_id in exclude_game_ids:
                continue
            if max_games is not None and games_processed >= max_games:
                break
            games_processed += 1
            yield game_id, encoded

    return materialize_entries(_iter_entries(), drop_state_probability,
                              seed=seed)
