"""Convert CSV game events to token sequences for transformer training.

Reads partitioned CSV files via preprocess module, tokenizes each valid game,
and outputs memory-mapped .bin files (tokens + win-probability labels).

Usage:
    # Partition-range mode (original):
    python -m sequence_model.tokenize_games

    # Single-CSV mode with disjoint train/val/test splits:
    python -m sequence_model.tokenize_games \
        --train-csv logged_in_games/gameevents_000.csv.gz \
        --val-csv late_tournament_games/late_tournament_game_events.csv.gz

    # Directory mode: tokenize all shards, split 90/10:
    python -m sequence_model.tokenize_games \
        --train-dir logged_in_games/ \
        --val-csv late_tournament_games/late_tournament_game_events.csv.gz
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess import (
    iterate_events_from_csv,
    iterate_events_by_game_and_normalize_time,
    is_valid_game,
    get_map_start,
    GameStartEvent, MapStartEvent, SpawnEvent, CarryFoodEvent,
    BerryDepositEvent, BerryKickInEvent, PlayerKillEvent,
    BlessMaidenEvent, UseMaidenEvent, GetOnSnailEvent, GetOffSnailEvent,
    SnailEatEvent, SnailEscapeEvent, VictoryEvent,
)
from constants import Team, ContestableState
import map_structure

from sequence_model.vocab import (
    BOS, EOS, PAD, VOCAB_SIZE,
    tokenize_game_start, tokenize_spawn, tokenize_carry_food,
    tokenize_berry_deposit, tokenize_berry_kick_in, tokenize_player_kill,
    tokenize_bless_maiden, tokenize_use_maiden,
    tokenize_get_on_snail, tokenize_get_off_snail,
    tokenize_snail_eat, tokenize_snail_escape, tokenize_victory,
    snail_position_token, time_gap_token,
    decode_tokens,
)
from fast_materialize import NUM_FEATURES

def _contestable_to_team(state: ContestableState) -> Team:
    """Convert ContestableState.BLUE/GOLD to Team.BLUE/GOLD."""
    if state == ContestableState.BLUE:
        return Team.BLUE
    return Team.GOLD


def tokenize_single_game(game_events, map_infos):
    """Convert a list of game events to a token sequence.

    Args:
        game_events: Normalized event list for one game.
        map_infos: MapStructureInfos for maiden index lookup.

    Returns:
        (tokens, blue_wins) where tokens is a list of ints and
        blue_wins is 1 if blue won, 0 otherwise. Returns None if
        the game can't be tokenized (e.g. missing map start).
    """
    try:
        map_start = get_map_start(game_events)
    except Exception:
        return None

    try:
        map_info = map_infos.get_map_info(map_start.map, map_start.gold_on_left)
    except KeyError:
        return None

    tokens = tokenize_game_start(map_start.map, map_start.gold_on_left)
    current_map = map_start.map

    blue_wins = None
    last_timestamp = 0.0  # events have normalized timestamps (seconds from game start)

    for event in game_events:
        if isinstance(event, (GameStartEvent, MapStartEvent)):
            continue  # Already handled in header

        # Insert time-gap token before each event
        event_tokens = None
        if isinstance(event, SpawnEvent):
            event_tokens = tokenize_spawn(event.position_id, event.is_bot)
        elif isinstance(event, CarryFoodEvent):
            event_tokens = tokenize_carry_food(event.position_id)
        elif isinstance(event, BerryDepositEvent):
            event_tokens = tokenize_berry_deposit(event.position_id)
        elif isinstance(event, BerryKickInEvent):
            event_tokens = tokenize_berry_kick_in(
                event.position_id, event.counts_for_own_team)
        elif isinstance(event, PlayerKillEvent):
            event_tokens = tokenize_player_kill(
                event.killer_position_id, event.killed_position_id,
                event.killed_player_category)
        elif isinstance(event, BlessMaidenEvent):
            try:
                _, maiden_index = map_info.get_type_and_maiden_index(
                    event.maiden_x, event.maiden_y)
            except ValueError:
                continue  # Skip invalid maiden coords
            team = _contestable_to_team(event.gate_color)
            event_tokens = tokenize_bless_maiden(maiden_index, team)
        elif isinstance(event, UseMaidenEvent):
            event_tokens = tokenize_use_maiden(
                event.position_id, event.maiden_type)
        elif isinstance(event, GetOnSnailEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_get_on_snail(event.rider_position_id, spt)
        elif isinstance(event, GetOffSnailEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_get_off_snail(event.position_id, spt)
        elif isinstance(event, SnailEatEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_snail_eat(
                event.rider_position_id, event.eaten_position_id, spt)
        elif isinstance(event, SnailEscapeEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_snail_escape(event.escaped_position_id, spt)
        elif isinstance(event, VictoryEvent):
            event_tokens = tokenize_victory(
                event.winning_team, event.victory_condition)
            blue_wins = 1 if event.winning_team == Team.BLUE else 0
        # Other event types are silently skipped

        if event_tokens is not None:
            # Insert time-gap token before the event tokens
            elapsed = event.timestamp - last_timestamp
            tokens.append(time_gap_token(max(0.0, elapsed)))
            tokens.extend(event_tokens)
            last_timestamp = event.timestamp

    if blue_wins is None:
        return None  # No victory event

    tokens.append(EOS)
    return tokens, blue_wins


def tokenize_and_materialize_single_game(game_events, map_infos, lgb_model=None):
    """Convert a list of game events to tokens with aligned 52D feature vectors.

    Args:
        game_events: Normalized event list for one game.
        map_infos: MapStructureInfos for maiden index lookup.
        lgb_model: Optional LightGBM model for generating predictions.

    Returns:
        (tokens, blue_wins, features_per_token, lgb_preds_per_token) where:
        - tokens: list of ints
        - blue_wins: 1 if blue won, 0 otherwise
        - features_per_token: np.array shape (num_tokens, 52) float32
        - lgb_preds_per_token: np.array shape (num_tokens,) float32
        Returns None if the game can't be tokenized.
    """
    from sequence_model.game_state import GameStateTracker

    try:
        map_start = get_map_start(game_events)
    except Exception:
        return None

    try:
        map_info = map_infos.get_map_info(map_start.map, map_start.gold_on_left)
    except KeyError:
        return None

    tracker = GameStateTracker(map_start.map, map_start.gold_on_left, map_infos)

    tokens = tokenize_game_start(map_start.map, map_start.gold_on_left)
    current_map = map_start.map

    # BOS + header tokens get zero features
    zero_feat = np.zeros(NUM_FEATURES, dtype=np.float32)
    n_header = len(tokens)
    all_features = [zero_feat.copy() for _ in range(n_header)]  # 3 header tokens

    blue_wins = None
    last_timestamp = 0.0

    # Per-event tracking for batched LGB prediction
    # Each entry: (feat, num_tokens, timestamp)
    event_feat_map = []

    for event in game_events:
        if isinstance(event, (GameStartEvent, MapStartEvent)):
            continue

        event_tokens = None
        if isinstance(event, SpawnEvent):
            event_tokens = tokenize_spawn(event.position_id, event.is_bot)
        elif isinstance(event, CarryFoodEvent):
            event_tokens = tokenize_carry_food(event.position_id)
        elif isinstance(event, BerryDepositEvent):
            event_tokens = tokenize_berry_deposit(event.position_id)
        elif isinstance(event, BerryKickInEvent):
            event_tokens = tokenize_berry_kick_in(
                event.position_id, event.counts_for_own_team)
        elif isinstance(event, PlayerKillEvent):
            event_tokens = tokenize_player_kill(
                event.killer_position_id, event.killed_position_id,
                event.killed_player_category)
        elif isinstance(event, BlessMaidenEvent):
            try:
                _, maiden_index = map_info.get_type_and_maiden_index(
                    event.maiden_x, event.maiden_y)
            except ValueError:
                tracker.apply_event(event)
                continue
            team = _contestable_to_team(event.gate_color)
            event_tokens = tokenize_bless_maiden(maiden_index, team)
        elif isinstance(event, UseMaidenEvent):
            event_tokens = tokenize_use_maiden(
                event.position_id, event.maiden_type)
        elif isinstance(event, GetOnSnailEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_get_on_snail(event.rider_position_id, spt)
        elif isinstance(event, GetOffSnailEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_get_off_snail(event.position_id, spt)
        elif isinstance(event, SnailEatEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_snail_eat(
                event.rider_position_id, event.eaten_position_id, spt)
        elif isinstance(event, SnailEscapeEvent):
            spt = snail_position_token(event.snail_x, current_map)
            event_tokens = tokenize_snail_escape(event.escaped_position_id, spt)
        elif isinstance(event, VictoryEvent):
            event_tokens = tokenize_victory(
                event.winning_team, event.victory_condition)
            blue_wins = 1 if event.winning_team == Team.BLUE else 0

        if event_tokens is not None:
            # Get features BEFORE applying event (matches fast_materialize)
            feat = tracker.get_features(event.timestamp)

            # Insert time-gap token before the event tokens
            elapsed = event.timestamp - last_timestamp
            tokens.append(time_gap_token(max(0.0, elapsed)))
            tokens.extend(event_tokens)

            # All tokens from this event get the same feature vector
            num_new = 1 + len(event_tokens)
            all_features.extend([feat] * num_new)
            event_feat_map.append((feat, num_new, event.timestamp))

            last_timestamp = event.timestamp

        # Apply state mutation AFTER feature extraction
        tracker.apply_event(event)

    if blue_wins is None:
        return None

    tokens.append(EOS)
    all_features.append(zero_feat.copy())  # EOS gets zero features

    # Build LGB predictions via single batched predict call
    # Header (n_header) + EOS (1) get pred=0; events get batched predictions
    all_lgb_preds = np.zeros(len(tokens), dtype=np.float32)

    if lgb_model is not None and event_feat_map:
        # Collect features for events past t=5.0
        batch_feats = []
        batch_event_indices = []
        for i, (feat, num_tokens, ts) in enumerate(event_feat_map):
            if ts > 5.0:
                batch_feats.append(feat)
                batch_event_indices.append(i)

        if batch_feats:
            batch_preds = lgb_model.predict(
                np.array(batch_feats, dtype=np.float32))

            # Build index mapping: event index -> prediction
            pred_by_event = {}
            for j, ei in enumerate(batch_event_indices):
                pred_by_event[ei] = float(batch_preds[j])

            # Fill predictions into token positions
            tok_offset = n_header
            for i, (feat, num_tokens, ts) in enumerate(event_feat_map):
                pred = pred_by_event.get(i, 0.0)
                all_lgb_preds[tok_offset:tok_offset + num_tokens] = pred
                tok_offset += num_tokens

    assert len(all_features) == len(tokens)
    assert len(all_lgb_preds) == len(tokens)

    features_arr = np.array(all_features, dtype=np.float32)
    lgb_preds_arr = np.array(all_lgb_preds, dtype=np.float32)

    return tokens, blue_wins, features_arr, lgb_preds_arr


def tokenize_partition_range(input_dir, start_partition, end_partition,
                             map_infos, max_games=None, verbose=True):
    """Tokenize games from a range of partitions.

    Returns:
        List of (tokens, blue_wins) tuples for valid games.
    """
    all_games = []
    total_tokens = 0

    for partition in range(start_partition, end_partition):
        if max_games and len(all_games) >= max_games:
            break
        csv_path = os.path.join(input_dir, f'gameevents_{partition:03d}.csv.gz')
        if not os.path.exists(csv_path):
            continue

        events = iterate_events_from_csv(csv_path)
        game_count = 0
        for game_id, game_events in iterate_events_by_game_and_normalize_time(events):
            if max_games and len(all_games) >= max_games:
                break

            error = is_valid_game(game_events, map_infos)
            if error:
                continue

            result = tokenize_single_game(game_events, map_infos)
            if result is None:
                continue

            tokens, blue_wins = result
            all_games.append((tokens, blue_wins))
            total_tokens += len(tokens)
            game_count += 1

        if verbose:
            print(f'  Partition {partition:03d}: {game_count} games')

    _print_stats(all_games, total_tokens, verbose)
    return all_games


def tokenize_csv_file(csv_path, map_infos, max_games=None, verbose=True):
    """Tokenize games from a single CSV/gzip file.

    Returns:
        List of (tokens, blue_wins) tuples for valid games.
    """
    all_games = []
    total_tokens = 0

    events = iterate_events_from_csv(csv_path)
    for game_id, game_events in iterate_events_by_game_and_normalize_time(events):
        if max_games and len(all_games) >= max_games:
            break

        error = is_valid_game(game_events, map_infos)
        if error:
            continue

        result = tokenize_single_game(game_events, map_infos)
        if result is None:
            continue

        tokens, blue_wins = result
        all_games.append((tokens, blue_wins))
        total_tokens += len(tokens)

    _print_stats(all_games, total_tokens, verbose)
    return all_games


def tokenize_csv_file_with_features(csv_path, map_infos, lgb_model=None,
                                     max_games=None, verbose=True):
    """Tokenize games from a single CSV/gzip file with feature materialization.

    Returns:
        List of (tokens, blue_wins, features, lgb_preds) tuples for valid games.
    """
    all_games = []
    total_tokens = 0

    events = iterate_events_from_csv(csv_path)
    for game_id, game_events in iterate_events_by_game_and_normalize_time(events):
        if max_games and len(all_games) >= max_games:
            break

        error = is_valid_game(game_events, map_infos)
        if error:
            continue

        result = tokenize_and_materialize_single_game(
            game_events, map_infos, lgb_model=lgb_model)
        if result is None:
            continue

        tokens, blue_wins, features, lgb_preds = result
        all_games.append((tokens, blue_wins, features, lgb_preds))
        total_tokens += len(tokens)

    _print_stats(all_games, total_tokens, verbose)
    return all_games


def _game_tokens(g):
    """Extract token list from a game tuple (works for both plain and feature tuples)."""
    return g[0]


def _print_stats(all_games, total_tokens, verbose):
    if verbose:
        print(f'Total: {len(all_games)} games, {total_tokens} tokens')
        if all_games:
            lens = [len(_game_tokens(g)) for g in all_games]
            print(f'Token lengths: mean={np.mean(lens):.0f}, '
                  f'median={np.median(lens):.0f}, '
                  f'p95={np.percentile(lens, 95):.0f}, '
                  f'p99={np.percentile(lens, 99):.0f}, '
                  f'max={max(lens)}')


def write_bin_files(games, output_dir, prefix):
    """Write concatenated token and label arrays as memory-mapped .bin files.

    For the token file: all games are concatenated end-to-end. Each game
    already has <BOS> at start and <EOS> at end, so boundaries are marked.

    For the label file: same length as token file, each position holds the
    blue_wins label (0 or 1) for the game that token belongs to. This
    lets the training loop read (token_chunk, label_chunk) pairs of any
    alignment without needing to find game boundaries.

    Files:
        {prefix}.bin  — uint16 token IDs
        {prefix}_labels.bin — uint8 blue_wins labels (0 or 1)
    """
    os.makedirs(output_dir, exist_ok=True)

    total_tokens = sum(len(g[0]) for g in games)
    print(f'Writing {prefix}: {len(games)} games, {total_tokens} tokens')

    token_arr = np.empty(total_tokens, dtype=np.uint16)
    label_arr = np.empty(total_tokens, dtype=np.uint8)

    offset = 0
    for tokens, blue_wins in games:
        n = len(tokens)
        token_arr[offset:offset + n] = tokens
        label_arr[offset:offset + n] = blue_wins
        offset += n

    assert offset == total_tokens

    token_path = os.path.join(output_dir, f'{prefix}.bin')
    label_path = os.path.join(output_dir, f'{prefix}_labels.bin')

    token_arr.tofile(token_path)
    label_arr.tofile(label_path)

    print(f'  {token_path}: {os.path.getsize(token_path) / 1e6:.1f} MB')
    print(f'  {label_path}: {os.path.getsize(label_path) / 1e6:.1f} MB')


def write_feature_bin_files(games_with_features, output_dir, prefix):
    """Write feature and LGB prediction arrays as memory-mapped .bin files.

    games_with_features: list of (tokens, blue_wins, features, lgb_preds)
    where features is np.array(num_tokens, 52) and lgb_preds is np.array(num_tokens,).

    Files:
        {prefix}_features.bin   — float16[total_tokens, 52]
        {prefix}_lgb_preds.bin  — float16[total_tokens]
    """
    os.makedirs(output_dir, exist_ok=True)

    total_tokens = sum(len(g[0]) for g in games_with_features)
    print(f'Writing {prefix} features: {len(games_with_features)} games, '
          f'{total_tokens} tokens')

    feat_arr = np.empty((total_tokens, NUM_FEATURES), dtype=np.float16)
    pred_arr = np.empty(total_tokens, dtype=np.float16)

    offset = 0
    for tokens, blue_wins, features, lgb_preds in games_with_features:
        n = len(tokens)
        feat_arr[offset:offset + n] = features.astype(np.float16)
        pred_arr[offset:offset + n] = lgb_preds.astype(np.float16)
        offset += n

    assert offset == total_tokens

    feat_path = os.path.join(output_dir, f'{prefix}_features.bin')
    pred_path = os.path.join(output_dir, f'{prefix}_lgb_preds.bin')

    feat_arr.tofile(feat_path)
    pred_arr.tofile(pred_path)

    print(f'  {feat_path}: {os.path.getsize(feat_path) / 1e6:.1f} MB')
    print(f'  {pred_path}: {os.path.getsize(pred_path) / 1e6:.1f} MB')


def print_sample_game(games, index=0):
    """Print a sample game's tokens for sanity checking."""
    g = games[index]
    tokens, blue_wins = g[0], g[1]
    print(f'\n--- Sample game {index} ({len(tokens)} tokens, '
          f'blue_wins={blue_wins}) ---')
    names = decode_tokens(tokens)
    # Print in groups for readability
    line = []
    for name in names:
        line.append(name)
        if name in ('<EOS>', '<BOS>') or name.startswith('victory_'):
            print(' '.join(line))
            line = []
        elif len(line) >= 10:
            print(' '.join(line))
            line = []
    if line:
        print(' '.join(line))


_CODEC_MAP_NAMES = ['map_day', 'map_night', 'map_dusk', 'map_twilight']
_CODEC_OPCODE_SIZES = [1, 1, 2, 1, 2, 2, 1, 2, 3, 3, 3, 3, 2, 1]
_CODEC_INT_TO_VICTORY_COND = None  # lazy init

# Cache for reverse lookups: (map_name, gold_on_left) -> (berry_positions, maiden_positions)
_CODEC_LOOKUP_CACHE = {}


def _get_codec_reverse_lookups(map_infos, map_name, gold_on_left):
    """Get cached reverse lookups for codec decoding."""
    key = (map_name, gold_on_left)
    if key not in _CODEC_LOOKUP_CACHE:
        from constants import Map
        map_enum = Map[map_name]
        map_info = map_infos.get_map_info(map_enum, gold_on_left)

        berry_positions = {}
        for (bx, by), bi in map_info._gold_berries.items():
            berry_positions[bi] = (bx, by)
        for (bx, by), bi in map_info._blue_berries.items():
            berry_positions[bi] = (bx, by)

        maiden_positions = {}
        for (mx, my), (mtype, midx) in map_info._maidens.items():
            maiden_positions[midx] = (mx, my, mtype.name)

        _CODEC_LOOKUP_CACHE[key] = (berry_positions, maiden_positions)
    return _CODEC_LOOKUP_CACHE[key]


def decode_codec_to_events(encoded_bytes, game_id=None, map_infos=None):
    """Decode event_codec binary back to a list of GameEvent objects.

    This is the inverse of event_codec.encode_game(). Produces GameEvent objects
    compatible with tokenize_single_game() and tokenize_and_materialize_single_game().

    Args:
        encoded_bytes: compact binary-encoded game from event_codec
        game_id: optional game ID to attach to events
        map_infos: MapStructureInfos instance (reuse for performance)

    Returns:
        list of GameEvent objects, or None if decoding fails
    """
    global _CODEC_INT_TO_VICTORY_COND
    from constants import VictoryCondition, PlayerCategory

    if _CODEC_INT_TO_VICTORY_COND is None:
        _CODEC_INT_TO_VICTORY_COND = [
            VictoryCondition.military, VictoryCondition.economic,
            VictoryCondition.snail]
    data = encoded_bytes
    pos = 0

    header = data[pos]; pos += 1
    map_idx = (header >> 1) & 0x3
    gold_on_left = bool(header & 1)
    map_name = _CODEC_MAP_NAMES[map_idx]

    if map_infos is None:
        map_infos = map_structure.MapStructureInfos()

    berry_positions, maiden_positions = _get_codec_reverse_lookups(
        map_infos, map_name, gold_on_left)

    events = []
    current_cs = 0

    # Create gamestart + mapstart events
    gs_event = GameStartEvent([map_name, str(gold_on_left), '', '', 'v2.34b'])
    gs_event.timestamp = 0.0
    gs_event.game_id = game_id
    events.append(gs_event)

    ms_event = MapStartEvent([map_name, str(gold_on_left), '', '', 'v2.34b'])
    ms_event.timestamp = 0.0
    ms_event.game_id = game_id
    events.append(ms_event)

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
        if opcode >= len(_CODEC_OPCODE_SIZES):
            return None
        sz = _CODEC_OPCODE_SIZES[opcode]

        if sz == 1:
            payload = b0 & 0xF
        elif sz == 2:
            b1 = data[pos]; pos += 1
            payload = ((b0 & 0xF) << 8) | b1
        else:
            b1 = data[pos]; pos += 1
            b2 = data[pos]; pos += 1
            payload = ((b0 & 0xF) << 16) | (b1 << 8) | b2

        event = None

        if opcode == 0:  # gamestart - skip, already added
            continue
        elif opcode == 1:  # mapstart - skip, already added
            continue
        elif opcode == 2:  # spawn
            pid = payload >> 1
            is_bot = bool(payload & 1)
            event = SpawnEvent([str(pid), str(is_bot)])
        elif opcode == 3:  # carryFood
            pid = payload
            event = CarryFoodEvent([str(pid)])
        elif opcode == 4:  # berryDeposit
            bi = payload >> 4
            pid = payload & 0xF
            bx, by = berry_positions.get(bi, (0, 0))
            event = BerryDepositEvent([str(bx), str(by), str(pid)])
        elif opcode == 5:  # berryKickIn
            bi = payload >> 5
            pid = (payload >> 1) & 0xF
            own = payload & 1
            bx, by = berry_positions.get(bi, (0, 0))
            event = BerryKickInEvent([str(bx), str(by), str(pid), str(bool(own))])
        elif opcode == 6:  # blessMaiden
            midx = payload >> 1
            is_blue = payload & 1
            mx, my, _ = maiden_positions.get(midx, (0, 0, ''))
            color = 'Blue' if is_blue else 'Gold'
            event = BlessMaidenEvent([str(mx), str(my), color])
        elif opcode == 7:  # useMaiden
            midx = payload >> 4
            pid = payload & 0xF
            mx, my, mtype = maiden_positions.get(midx, (0, 0, 'maiden_speed'))
            event = UseMaidenEvent([str(mx), str(my), mtype, str(pid)])
        elif opcode == 8:  # getOnSnail
            sx = payload >> 4
            rider_pid = payload & 0xF
            event = GetOnSnailEvent([str(sx), '0', str(rider_pid)])
        elif opcode == 9:  # snailEat
            sx = payload >> 8
            rider_pid = (payload >> 4) & 0xF
            eaten_pid = payload & 0xF
            event = SnailEatEvent([str(sx), '0', str(rider_pid), str(eaten_pid)])
        elif opcode == 10:  # getOffSnail
            sx = payload >> 4
            pid = payload & 0xF
            event = GetOffSnailEvent([str(sx), '0', '0', str(pid)])
        elif opcode == 11:  # snailEscape
            sx = payload >> 4
            escaped_pid = payload & 0xF
            event = SnailEscapeEvent([str(sx), '0', str(escaped_pid)])
        elif opcode == 12:  # playerKill
            killer_pid = payload >> 4
            killed_pid = payload & 0xF
            # Determine category: queen if pid <= 2, else Worker/Soldier
            category = 'Queen' if killed_pid <= 2 else 'Worker'
            event = PlayerKillEvent(['0', '0', str(killer_pid), str(killed_pid),
                                     category])
        elif opcode == 13:  # victory
            team_int = payload >> 2
            cond_int = payload & 0x3
            team = 'Gold' if team_int else 'Blue'
            cond = _CODEC_INT_TO_VICTORY_COND[cond_int]
            event = VictoryEvent([team, cond.name])

        if event is not None:
            event.timestamp = rel_ts
            event.game_id = game_id
            events.append(event)

    return events


def build_interleaved_union(qf_bin_path, li_bin_path, max_games=None):
    """Build deduplicated interleaved union of QF and LI game pools.

    Returns:
        list of (game_id, encoded_bytes, source) tuples in interleaved order
    """
    import struct

    def read_packed_games(path):
        entries = []
        with open(path, 'rb') as f:
            (num_games,) = struct.unpack('<I', f.read(4))
            for _ in range(num_games):
                game_id, length = struct.unpack('<IH', f.read(6))
                payload = f.read(length)
                entries.append((game_id, payload))
        return entries

    qf_entries = read_packed_games(qf_bin_path)
    li_entries = read_packed_games(li_bin_path)

    seen = set()
    pool = []
    qi, li = 0, 0
    while qi < len(qf_entries) or li < len(li_entries):
        if qi < len(qf_entries):
            gid, data = qf_entries[qi]; qi += 1
            if gid not in seen:
                seen.add(gid)
                pool.append((gid, data, 'qf'))
        if li < len(li_entries):
            gid, data = li_entries[li]; li += 1
            if gid not in seen:
                seen.add(gid)
                pool.append((gid, data, 'li'))

    qf_count = sum(1 for _, _, s in pool if s == 'qf')
    li_count = sum(1 for _, _, s in pool if s == 'li')
    overlap = len(qf_entries) + len(li_entries) - len(pool)
    print(f'  QF: {len(qf_entries):,} games, LI: {len(li_entries):,} games')
    print(f'  Union: {len(pool):,} (overlap: {overlap:,})')

    if max_games:
        pool = pool[:max_games]
        qf_count = sum(1 for _, _, s in pool if s == 'qf')
        li_count = sum(1 for _, _, s in pool if s == 'li')
    print(f'  Selected: {len(pool):,} ({qf_count:,} QF + {li_count:,} LI)')

    return pool


def tokenize_codec_pool(pool, map_infos, lgb_model=None, symmetric=False):
    """Tokenize games from a codec binary pool with optional symmetric augmentation.

    Args:
        pool: list of (game_id, encoded_bytes, source) tuples
        map_infos: MapStructureInfos instance
        lgb_model: optional LightGBM model for predictions
        symmetric: if True, also tokenize team-swapped copies

    Returns:
        list of (tokens, blue_wins, features, lgb_preds) tuples
    """
    from symmetry import swap_event_stream

    all_games = []
    total_tokens = 0
    failed = 0

    t_start = time.time()
    for gi, (game_id, encoded_bytes, source) in enumerate(pool):
        if gi % 500 == 0:
            elapsed = time.time() - t_start
            rate = gi / elapsed if elapsed > 0 else 0
            eta = (len(pool) - gi) / rate if rate > 0 else 0
            print(f'  [{gi}/{len(pool)}] {elapsed:.0f}s elapsed, '
                  f'{rate:.1f} games/s, ETA {eta:.0f}s', flush=True)

        game_events = decode_codec_to_events(encoded_bytes, game_id, map_infos=map_infos)
        if game_events is None:
            failed += 1
            continue

        # Tokenize original
        if lgb_model is not None:
            result = tokenize_and_materialize_single_game(
                game_events, map_infos, lgb_model=lgb_model)
        else:
            result = tokenize_single_game(game_events, map_infos)
        if result is None:
            failed += 1
            continue

        if lgb_model is not None:
            tokens, blue_wins, features, lgb_preds = result
            all_games.append((tokens, blue_wins, features, lgb_preds))
        else:
            tokens, blue_wins = result
            all_games.append((tokens, blue_wins))
        total_tokens += len(tokens)

        # Symmetric augmentation: swap teams and tokenize
        if symmetric:
            swapped_events = swap_event_stream(game_events)
            if lgb_model is not None:
                result_sw = tokenize_and_materialize_single_game(
                    swapped_events, map_infos, lgb_model=lgb_model)
            else:
                result_sw = tokenize_single_game(swapped_events, map_infos)
            if result_sw is not None:
                if lgb_model is not None:
                    tokens_sw, bw_sw, feat_sw, lgb_sw = result_sw
                    all_games.append((tokens_sw, bw_sw, feat_sw, lgb_sw))
                else:
                    tokens_sw, bw_sw = result_sw
                    all_games.append((tokens_sw, bw_sw))
                total_tokens += len(tokens_sw if lgb_model is not None else result_sw[0])

    print(f'  Tokenized: {len(all_games)} games ({failed} failed)')
    _print_stats(all_games, total_tokens, verbose=True)
    return all_games


def main():
    parser = argparse.ArgumentParser(
        description='Tokenize Killer Queen game events for transformer training')
    parser.add_argument('--input-dir', default='new_data_partitioned',
                        help='Directory with partitioned CSV files')
    parser.add_argument('--output-dir', default='sequence_model/data',
                        help='Output directory for .bin files')
    parser.add_argument('--train-end', type=int, default=740,
                        help='Last partition for training (exclusive)')
    parser.add_argument('--val-start', type=int, default=740,
                        help='First partition for validation')
    parser.add_argument('--val-end', type=int, default=925,
                        help='Last partition for validation (exclusive)')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Max games to tokenize per split')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='Use a single CSV/gzip file for training; '
                             'games are split 90/10 into train.bin + val.bin')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='Tokenize all *.csv.gz files in directory; '
                             'split 90/10 into train.bin + val.bin')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Use a specific CSV/gzip file as the test set '
                             '(writes test.bin + test_labels.bin)')
    parser.add_argument('--sample', action='store_true',
                        help='Print sample tokenized games')
    parser.add_argument('--quick', action='store_true',
                        help='Only process first 5 partitions (for testing)')
    parser.add_argument('--materialize', action='store_true',
                        help='Also write feature and LGB prediction .bin files')
    parser.add_argument('--lgb-model', type=str, default=None,
                        help='Path to LightGBM model file for generating '
                             'predictions (only used with --materialize)')
    # QF+LI union mode
    parser.add_argument('--qf-bin', type=str, default=None,
                        help='Path to QF encoded binary (quality_filtered/encoded/all_games.bin)')
    parser.add_argument('--li-bin', type=str, default=None,
                        help='Path to LI encoded binary (logged_in_games/encoded/all_games.bin)')
    parser.add_argument('--symmetric', action='store_true',
                        help='Add team-swapped copies of all games (doubles data)')
    args = parser.parse_args()

    map_infos = map_structure.MapStructureInfos()

    # Load LGB model if materializing with predictions
    lgb_model = None
    if args.materialize and args.lgb_model:
        import lightgbm as lgb
        print(f'Loading LightGBM model from {args.lgb_model}')
        lgb_model = lgb.Booster(model_file=args.lgb_model)

    if args.qf_bin and args.li_bin:
        # QF+LI union mode: decode from codec binaries, interleave, tokenize
        print(f'=== Building QF+LI interleaved union ===')
        pool = build_interleaved_union(
            args.qf_bin, args.li_bin, max_games=args.max_games)

        print(f'\n=== Tokenizing {len(pool)} games ===')
        if args.symmetric:
            print('  (with symmetric augmentation)')
        if args.materialize:
            print('  (with feature materialization)')
        t0 = time.time()
        all_games = tokenize_codec_pool(
            pool, map_infos, lgb_model=lgb_model, symmetric=args.symmetric)
        print(f'Tokenization took {time.time() - t0:.1f}s\n')

        # Deterministic 90/10 split by game order
        n = len(all_games)
        split_idx = int(n * 0.9)
        train_games = all_games[:split_idx]
        val_games = all_games[split_idx:]
        print(f'Split: {len(train_games)} train, {len(val_games)} val '
              f'(from {n} total games)')

        has_features = args.materialize and len(all_games) > 0 and len(all_games[0]) == 4
        if has_features:
            write_bin_files(
                [(g[0], g[1]) for g in train_games], args.output_dir, 'train')
            write_bin_files(
                [(g[0], g[1]) for g in val_games], args.output_dir, 'val')
            write_feature_bin_files(train_games, args.output_dir, 'train')
            write_feature_bin_files(val_games, args.output_dir, 'val')
        else:
            write_bin_files(train_games, args.output_dir, 'train')
            write_bin_files(val_games, args.output_dir, 'val')

        # If --val-csv given, treat it as a held-out test set
        if args.val_csv:
            print(f'\n=== Tokenizing test data from {args.val_csv} ===')
            t0 = time.time()
            if args.materialize:
                test_games = tokenize_csv_file_with_features(
                    args.val_csv, map_infos, lgb_model=lgb_model)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(
                    [(g[0], g[1]) for g in test_games], args.output_dir, 'test')
                write_feature_bin_files(test_games, args.output_dir, 'test')
            else:
                test_games = tokenize_csv_file(args.val_csv, map_infos)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(test_games, args.output_dir, 'test')

    elif args.train_dir:
        # Directory mode: tokenize all CSV/gzip files, split 90/10
        csv_files = sorted(glob.glob(os.path.join(args.train_dir, '*.csv.gz')))
        print(f'=== Tokenizing {len(csv_files)} files from {args.train_dir} ===')
        if args.materialize:
            print('  (with feature materialization)')
        t0 = time.time()
        all_games = []
        total_tokens = 0
        for csv_path in csv_files:
            if args.materialize:
                games = tokenize_csv_file_with_features(
                    csv_path, map_infos, lgb_model=lgb_model, verbose=False)
            else:
                games = tokenize_csv_file(csv_path, map_infos, verbose=False)
            all_games.extend(games)
            total_tokens += sum(len(_game_tokens(g)) for g in games)
            print(f'  {os.path.basename(csv_path)}: {len(games)} games'
                  f' ({len(all_games)} total)')
            if args.max_games and len(all_games) >= args.max_games:
                all_games = all_games[:args.max_games]
                total_tokens = sum(len(_game_tokens(g)) for g in all_games)
                break
        _print_stats(all_games, total_tokens, verbose=True)
        print(f'Tokenization took {time.time() - t0:.1f}s\n')

        # Deterministic 90/10 split by game order
        n = len(all_games)
        split_idx = int(n * 0.9)
        train_games = all_games[:split_idx]
        val_games = all_games[split_idx:]
        print(f'Split: {len(train_games)} train, {len(val_games)} val '
              f'(from {n} total games)')

        if args.materialize:
            # Write both token files and feature files
            write_bin_files(
                [(g[0], g[1]) for g in train_games], args.output_dir, 'train')
            write_bin_files(
                [(g[0], g[1]) for g in val_games], args.output_dir, 'val')
            write_feature_bin_files(train_games, args.output_dir, 'train')
            write_feature_bin_files(val_games, args.output_dir, 'val')
        else:
            write_bin_files(train_games, args.output_dir, 'train')
            write_bin_files(val_games, args.output_dir, 'val')

        # If --val-csv given, treat it as a held-out test set
        if args.val_csv:
            print(f'\n=== Tokenizing test data from {args.val_csv} ===')
            t0 = time.time()
            if args.materialize:
                test_games = tokenize_csv_file_with_features(
                    args.val_csv, map_infos, lgb_model=lgb_model)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(
                    [(g[0], g[1]) for g in test_games], args.output_dir, 'test')
                write_feature_bin_files(test_games, args.output_dir, 'test')
            else:
                test_games = tokenize_csv_file(args.val_csv, map_infos)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(test_games, args.output_dir, 'test')

    elif args.train_csv:
        # Single-CSV mode: split one file into train/val, optionally write test
        print(f'=== Tokenizing from {args.train_csv} ===')
        if args.materialize:
            print('  (with feature materialization)')
        t0 = time.time()
        if args.materialize:
            all_games = tokenize_csv_file_with_features(
                args.train_csv, map_infos, lgb_model=lgb_model,
                max_games=args.max_games)
        else:
            all_games = tokenize_csv_file(args.train_csv, map_infos,
                                          max_games=args.max_games)
        print(f'Tokenization took {time.time() - t0:.1f}s\n')

        # Deterministic 90/10 split by game order
        n = len(all_games)
        split_idx = int(n * 0.9)
        train_games = all_games[:split_idx]
        val_games = all_games[split_idx:]
        print(f'Split: {len(train_games)} train, {len(val_games)} val '
              f'(from {n} total games)')

        if args.materialize:
            write_bin_files(
                [(g[0], g[1]) for g in train_games], args.output_dir, 'train')
            write_bin_files(
                [(g[0], g[1]) for g in val_games], args.output_dir, 'val')
            write_feature_bin_files(train_games, args.output_dir, 'train')
            write_feature_bin_files(val_games, args.output_dir, 'val')
        else:
            write_bin_files(train_games, args.output_dir, 'train')
            write_bin_files(val_games, args.output_dir, 'val')

        # If --val-csv given, treat it as a held-out test set
        if args.val_csv:
            print(f'\n=== Tokenizing test data from {args.val_csv} ===')
            t0 = time.time()
            if args.materialize:
                test_games = tokenize_csv_file_with_features(
                    args.val_csv, map_infos, lgb_model=lgb_model)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(
                    [(g[0], g[1]) for g in test_games], args.output_dir, 'test')
                write_feature_bin_files(test_games, args.output_dir, 'test')
            else:
                test_games = tokenize_csv_file(args.val_csv, map_infos)
                print(f'Test tokenization took {time.time() - t0:.1f}s\n')
                write_bin_files(test_games, args.output_dir, 'test')
    else:
        # Partition-range mode (original behavior)
        if args.quick:
            args.train_end = 4
            args.val_start = 4
            args.val_end = 5

        # Tokenize training data
        print('=== Tokenizing training data (partitions 0-{}{}) ==='.format(
            args.train_end - 1,
            f', max {args.max_games} games' if args.max_games else ''))
        t0 = time.time()
        train_games = tokenize_partition_range(
            args.input_dir, 0, args.train_end, map_infos,
            max_games=args.max_games)
        print(f'Training tokenization took {time.time() - t0:.1f}s\n')

        # Tokenize validation data
        if args.val_csv:
            print(f'=== Tokenizing validation data from {args.val_csv} ===')
            t0 = time.time()
            val_games = tokenize_csv_file(args.val_csv, map_infos)
        else:
            print('=== Tokenizing validation data (partitions {}-{}) ==='.format(
                args.val_start, args.val_end - 1))
            t0 = time.time()
            val_games = tokenize_partition_range(
                args.input_dir, args.val_start, args.val_end, map_infos)
        print(f'Validation tokenization took {time.time() - t0:.1f}s\n')

        write_bin_files(train_games, args.output_dir, 'train')
        write_bin_files(val_games, args.output_dir, 'val')

    # Print samples
    if args.sample or args.quick:
        if train_games:
            print_sample_game(train_games, 0)
        if len(train_games) > 1:
            print_sample_game(train_games, 1)

    print('\nDone!')


if __name__ == '__main__':
    main()
