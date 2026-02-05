"""Convert CSV game events to token sequences for transformer training.

Reads partitioned CSV files via preprocess module, tokenizes each valid game,
and outputs memory-mapped .bin files (tokens + win-probability labels).

Usage:
    python -m sequence_model.tokenize_games
"""

import argparse
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
    snail_position_token,
    decode_tokens,
)
from constants import SCREEN_WIDTH


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

    # Precompute snail track geometry for position discretization
    track_width = map_info.snail_track_width
    track_left = SCREEN_WIDTH / 2 - track_width / 2

    blue_wins = None

    for event in game_events:
        if isinstance(event, (GameStartEvent, MapStartEvent)):
            continue  # Already handled in header
        elif isinstance(event, SpawnEvent):
            tokens.extend(tokenize_spawn(event.position_id, event.is_bot))
        elif isinstance(event, CarryFoodEvent):
            tokens.extend(tokenize_carry_food(event.position_id))
        elif isinstance(event, BerryDepositEvent):
            tokens.extend(tokenize_berry_deposit(event.position_id))
        elif isinstance(event, BerryKickInEvent):
            tokens.extend(tokenize_berry_kick_in(
                event.position_id, event.counts_for_own_team))
        elif isinstance(event, PlayerKillEvent):
            tokens.extend(tokenize_player_kill(
                event.killer_position_id, event.killed_position_id,
                event.killed_player_category))
        elif isinstance(event, BlessMaidenEvent):
            try:
                _, maiden_index = map_info.get_type_and_maiden_index(
                    event.maiden_x, event.maiden_y)
            except ValueError:
                continue  # Skip invalid maiden coords
            team = _contestable_to_team(event.gate_color)
            tokens.extend(tokenize_bless_maiden(maiden_index, team))
        elif isinstance(event, UseMaidenEvent):
            tokens.extend(tokenize_use_maiden(
                event.position_id, event.maiden_type))
        elif isinstance(event, GetOnSnailEvent):
            spt = snail_position_token(event.snail_x, track_left, track_width)
            tokens.extend(tokenize_get_on_snail(event.rider_position_id, spt))
        elif isinstance(event, GetOffSnailEvent):
            spt = snail_position_token(event.snail_x, track_left, track_width)
            tokens.extend(tokenize_get_off_snail(event.position_id, spt))
        elif isinstance(event, SnailEatEvent):
            spt = snail_position_token(event.snail_x, track_left, track_width)
            tokens.extend(tokenize_snail_eat(
                event.rider_position_id, event.eaten_position_id, spt))
        elif isinstance(event, SnailEscapeEvent):
            spt = snail_position_token(event.snail_x, track_left, track_width)
            tokens.extend(tokenize_snail_escape(event.escaped_position_id, spt))
        elif isinstance(event, VictoryEvent):
            tokens.extend(tokenize_victory(
                event.winning_team, event.victory_condition))
            blue_wins = 1 if event.winning_team == Team.BLUE else 0
        # Other event types are silently skipped

    if blue_wins is None:
        return None  # No victory event

    tokens.append(EOS)
    return tokens, blue_wins


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


def _print_stats(all_games, total_tokens, verbose):
    if verbose:
        print(f'Total: {len(all_games)} games, {total_tokens} tokens')
        if all_games:
            lens = [len(g[0]) for g in all_games]
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


def print_sample_game(games, index=0):
    """Print a sample game's tokens for sanity checking."""
    tokens, blue_wins = games[index]
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
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Use a specific CSV/gzip file for validation '
                             'instead of partitions (e.g. tournament data)')
    parser.add_argument('--sample', action='store_true',
                        help='Print sample tokenized games')
    parser.add_argument('--quick', action='store_true',
                        help='Only process first 5 partitions (for testing)')
    args = parser.parse_args()

    map_infos = map_structure.MapStructureInfos()

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

    # Write binary files
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
