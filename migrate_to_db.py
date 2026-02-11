"""Migrate KQuity game data from CSV.gz partitions + JSONL cache into sharded SQLite.

Usage:
    python migrate_to_db.py --db-dir data/game_db/ --csv-dir unfiltered_partitioned/
    python migrate_to_db.py --db-dir data/game_db/ --jsonl-dir ../kq_stream_highlights/cache/game_events/
    python migrate_to_db.py --db-dir data/game_db/ --enrich  # add game.csv + usergame.csv metadata
    python migrate_to_db.py --db-dir data/game_db/ --ratings ratings_queen_drone.pkl
    python migrate_to_db.py --db-dir data/game_db/ --all     # full pipeline
"""

import argparse
import ctypes
import csv
import datetime
import glob
import gzip
import json
import os
import pickle
import resource
import sys

MEMORY_LIMIT_BYTES = 8 * 1024 ** 3  # 8 GB


def _set_resource_limits():
    """Set RLIMIT_DATA to 8 GB and lock pages into RAM to prevent swap thrashing."""
    # RLIMIT_DATA caps the data segment (heap) without counting mmap'd regions,
    # shared libraries, or gzip decompression buffers that inflate virtual size.
    # RLIMIT_AS is too aggressive — Python+gzip+SQLite virtual overhead can
    # exceed the limit even when actual heap usage is well within budget.
    resource.setrlimit(resource.RLIMIT_DATA,
                       (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    print(f"RLIMIT_DATA set to {MEMORY_LIMIT_BYTES / 1024**3:.0f} GB")

    # mlockall: lock all current and future pages into RAM
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    MCL_CURRENT = 1
    MCL_FUTURE = 2
    ret = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
    if ret != 0:
        errno = ctypes.get_errno()
        print(f"Warning: mlockall failed (errno={errno}). "
              "Run with CAP_IPC_LOCK or as root to prevent swap usage.",
              file=sys.stderr)
    else:
        print("mlockall succeeded — pages locked into RAM")

from game_db import (
    GameDocument, GameEvent, PlayerEntry, GameDB, ShardedGameDB,
    shard_id_for_game, GAMES_PER_SHARD,
)

# Skip events matching fast_materialize.py
SKIP_EVENTS = frozenset({
    'gameend', 'playernames',
    'reserveMaiden', 'unreserveMaiden',
    'cabinetOnline', 'cabinetOffline',
    'bracket', 'tstart', 'tournamentValidation', 'checkIfTournamentRunning',
    'glance',
    'enteredGameScreen', 'signInPlayer', 'signOutPlayer',
})

# CSV column indices
COL_TS = 1
COL_TYPE = 2
COL_VALUES = 3
COL_GAME_ID = 4


def _parse_values(values_str: str) -> list:
    """Parse '{8,False}' -> [8, False]. Preserves types for compact storage."""
    inner = values_str[1:-1]
    if not inner:
        return []
    parts = inner.split(',')
    result = []
    for p in parts:
        if p == 'True':
            result.append(True)
        elif p == 'False':
            result.append(False)
        else:
            try:
                result.append(int(p))
            except ValueError:
                try:
                    result.append(float(p))
                except ValueError:
                    result.append(p)
    return result


def _extract_game_info(raw_events):
    """Extract map_name, gold_on_left, gamestart_dt, victory info from raw events.

    raw_events: list of (datetime, event_type, values_str)
    Returns dict with extracted info, or None if game is invalid.
    """
    gamestart_dt = None
    map_name = None
    gold_on_left = None
    win_condition = None
    winning_team = None

    for dt, event_type, values_str in raw_events:
        if event_type == 'gamestart' and gamestart_dt is None:
            gamestart_dt = dt
        if event_type == 'mapstart' and map_name is None:
            vals = values_str[1:-1].split(',')
            map_name = vals[0]
            gold_on_left = (vals[1] == 'True')
        if event_type == 'victory':
            vals = values_str[1:-1].split(',')
            winning_team = vals[0]
            if len(vals) > 1:
                win_condition = vals[1]

    if gamestart_dt is None or map_name is None:
        return None

    return {
        'gamestart_dt': gamestart_dt,
        'map_name': map_name,
        'gold_on_left': gold_on_left,
        'winning_team': winning_team,
        'win_condition': win_condition,
    }


def _build_game_document(game_id, raw_events, game_info):
    """Convert raw CSV events into a GameDocument with compact relative-time events."""
    gamestart_dt = game_info['gamestart_dt']

    compact_events = []
    for dt, event_type, values_str in raw_events:
        if event_type in SKIP_EVENTS:
            continue
        rel_t = round((dt - gamestart_dt).total_seconds(), 4)
        evt = GameEvent(t=rel_t, type=event_type, vals=_parse_values(values_str))
        compact_events.append(evt)

    compact_events.sort(key=lambda e: e.t)

    last_dt = raw_events[-1][0] if raw_events else gamestart_dt
    duration = (last_dt - gamestart_dt).total_seconds()

    return GameDocument(
        game_id=game_id,
        map_name=game_info['map_name'],
        gold_on_left=game_info['gold_on_left'],
        start_time=gamestart_dt.isoformat(),
        end_time=last_dt.isoformat(),
        win_condition=game_info['win_condition'],
        winning_team=game_info['winning_team'],
        events=compact_events,
        duration_seconds=round(duration, 3),
    )


def _find_csv_files(csv_dir: str, csv_glob: str | None = None) -> list[str]:
    """Locate gameevents CSV files, preferring .csv.gz."""
    if csv_glob:
        return sorted(glob.glob(csv_glob))
    files = sorted(glob.glob(os.path.join(csv_dir, 'gameevents_*.csv.gz')))
    if not files:
        files = sorted(glob.glob(os.path.join(csv_dir, 'gameevents_*.csv')))
    return files



def _flush_games(game_ids, pending, sharded, stats):
    """Process pending games and write to shard DBs, removing them from pending."""
    batch_by_shard: dict[int, list[GameDocument]] = {}

    for game_id in game_ids:
        raw_events = pending.pop(game_id)
        raw_events.sort(key=lambda x: x[0])

        game_info = _extract_game_info(raw_events)
        if game_info is None:
            stats['skipped'] += 1
            continue

        doc = _build_game_document(game_id, raw_events, game_info)
        sid = shard_id_for_game(game_id)
        if sid not in batch_by_shard:
            batch_by_shard[sid] = []
        batch_by_shard[sid].append(doc)
        stats['migrated'] += 1

    for sid in sorted(batch_by_shard):
        db = sharded.get_or_create_shard(sid)
        db.insert_games_batch(batch_by_shard[sid])
        db.commit()


def migrate_csv_partitions(db_dir: str, csv_dir: str, verbose: bool = True,
                           csv_glob: str | None = None):
    """Single-pass, streaming CSV migration into sharded SQLite.

    Reads each partition file once. After each file, flushes games that
    did not appear in the current file (their events are complete).
    Games spanning consecutive files stay buffered until they stop appearing.
    Memory usage ≈ one partition file's worth of games (~1000 games).

    Args:
        db_dir: Output directory for shard DBs.
        csv_dir: Directory containing CSV files.
        verbose: Print progress.
        csv_glob: Optional explicit glob pattern for CSV files.
    """
    files = _find_csv_files(csv_dir, csv_glob)
    if not files:
        print(f"No CSV files found in {csv_dir}")
        return

    print(f"Found {len(files)} CSV files to migrate")
    os.makedirs(db_dir, exist_ok=True)

    sharded = ShardedGameDB(db_dir)
    pending: dict[int, list] = {}   # game_id -> raw events
    stats = {'migrated': 0, 'skipped': 0, 'rows': 0}

    for file_idx, filename in enumerate(files):
        if verbose:
            print(f"  [{file_idx+1}/{len(files)}] {os.path.basename(filename)}")

        seen_in_file: set[int] = set()
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                stats['rows'] += 1
                game_id = int(row[COL_GAME_ID])
                seen_in_file.add(game_id)
                if game_id not in pending:
                    pending[game_id] = []
                pending[game_id].append((
                    datetime.datetime.fromisoformat(row[COL_TS]),
                    row[COL_TYPE],
                    row[COL_VALUES],
                ))

        # Flush games not seen in this file — their events are complete
        flushable = [gid for gid in pending if gid not in seen_in_file]
        if flushable:
            _flush_games(flushable, pending, sharded, stats)

        if verbose and (file_idx + 1) % 100 == 0:
            print(f"    {stats['migrated']} migrated, {stats['skipped']} skipped, "
                  f"{len(pending)} pending in buffer")

    # Flush remaining games from the last file(s)
    _flush_games(list(pending.keys()), pending, sharded, stats)

    sharded.close()
    print(f"CSV migration complete: {stats['migrated']} games, "
          f"{stats['skipped']} skipped, {stats['rows']} rows read")


def enrich_with_metadata(db_dir: str, csv_dir: str, verbose: bool = True):
    """Step 3: Enrich games with metadata from game.csv and usergame.csv."""
    game_csv = os.path.join(csv_dir, 'game.csv')
    usergame_csv = os.path.join(csv_dir, 'usergame.csv')

    # Load game.csv: id,start_time,end_time,win_condition,winning_team,map_name,
    #                player_count,cabinet_id,cabinet_name,tournament_match_id,...
    print("Loading game.csv...")
    game_meta = {}
    with open(game_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            game_id = int(row[0])
            game_meta[game_id] = {
                'cabinet_name': row[8] or None,
                'tournament_match_id': int(row[9]) if row[9] else None,
                'player_count': int(row[6]) if row[6] else None,
                'scene_name': None,  # not in game.csv directly
            }
    print(f"  {len(game_meta)} games in game.csv")

    # Load usergame.csv: id,game_id,position_id,user_id,name,scene
    print("Loading usergame.csv...")
    usergame: dict[int, list[tuple]] = {}
    scenes_by_game: dict[int, str] = {}
    with open(usergame_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            game_id = int(row[1])
            position_id = int(row[2])
            user_id = int(row[3])
            name = row[4]
            scene = row[5] if len(row) > 5 else None
            if game_id not in usergame:
                usergame[game_id] = []
            usergame[game_id].append((position_id, user_id, name))
            if scene and game_id not in scenes_by_game:
                scenes_by_game[game_id] = scene
    print(f"  {len(usergame)} games with logged-in users")

    # Update shards
    sharded = ShardedGameDB(db_dir)
    updated_games = 0
    updated_players = 0

    # Group updates by shard
    shard_updates: dict[int, list[tuple[int, dict]]] = {}
    shard_players: dict[int, list[PlayerEntry]] = {}

    for game_id, meta in game_meta.items():
        sid = shard_id_for_game(game_id)
        if sid not in shard_updates:
            shard_updates[sid] = []
        login_count = len(usergame.get(game_id, []))
        scene = scenes_by_game.get(game_id)
        shard_updates[sid].append((game_id, meta, login_count, scene))

    for game_id, players in usergame.items():
        sid = shard_id_for_game(game_id)
        if sid not in shard_players:
            shard_players[sid] = []
        for position_id, user_id, name in players:
            role = 'queen' if position_id in (1, 2) else 'drone'
            shard_players[sid].append(PlayerEntry(
                game_id=game_id,
                position_id=position_id,
                user_id=user_id,
                user_name=name,
                role=role,
            ))

    for sid in sorted(set(shard_updates) | set(shard_players)):
        db = sharded.get_or_create_shard(sid)
        conn = db._get_conn()

        if sid in shard_updates:
            for game_id, meta, login_count, scene in shard_updates[sid]:
                conn.execute(
                    """UPDATE games SET
                       cabinet_name = COALESCE(?, cabinet_name),
                       tournament_match_id = COALESCE(?, tournament_match_id),
                       player_count = COALESCE(?, player_count),
                       scene_name = COALESCE(?, scene_name),
                       login_count = ?
                       WHERE game_id = ?""",
                    (meta['cabinet_name'], meta['tournament_match_id'],
                     meta['player_count'], scene, login_count, game_id))
                updated_games += 1

        if sid in shard_players:
            db.insert_players_batch(shard_players[sid])
            updated_players += len(shard_players[sid])

        db.commit()
        if verbose and (updated_games % 50000 < 1000):
            print(f"  Updated {updated_games} games, {updated_players} player entries...")

    sharded.close()
    print(f"Enrichment complete: {updated_games} games, {updated_players} players")


def migrate_ratings(db_dir: str, ratings_path: str, verbose: bool = True):
    """Step 4: Store ratings as game_metadata + denormalized max/avg mu."""
    print(f"Loading ratings from {ratings_path}...")
    with open(ratings_path, 'rb') as f:
        ratings_by_game = pickle.load(f)
    print(f"  {len(ratings_by_game)} games with ratings")

    sharded = ShardedGameDB(db_dir)

    # Group by shard
    shard_ratings: dict[int, list[tuple[int, list]]] = {}
    for game_id, mu_array in ratings_by_game.items():
        sid = shard_id_for_game(game_id)
        if sid not in shard_ratings:
            shard_ratings[sid] = []
        shard_ratings[sid].append((game_id, mu_array.tolist()))

    total = 0
    for sid in sorted(shard_ratings):
        db = sharded.get_or_create_shard(sid)
        conn = db._get_conn()

        for game_id, mu_list in shard_ratings[sid]:
            # Store as metadata
            conn.execute(
                """INSERT OR REPLACE INTO game_metadata (game_id, key, value, updated_at)
                   VALUES (?, 'ratings', ?, ?)""",
                (game_id, json.dumps(mu_list),
                 datetime.datetime.utcnow().isoformat()))

            # Denormalize max/avg mu
            max_mu = max(mu_list)
            avg_mu = sum(mu_list) / len(mu_list)
            conn.execute(
                """UPDATE games SET max_player_mu = ?, avg_player_mu = ?
                   WHERE game_id = ?""",
                (max_mu, avg_mu, game_id))
            total += 1

        db.commit()
        if verbose:
            print(f"  Shard {sid:05d}: {len(shard_ratings[sid])} ratings")

    sharded.close()
    print(f"Ratings migration complete: {total} games")


def migrate_jsonl_cache(db_dir: str, jsonl_dir: str, verbose: bool = True):
    """Step 5: Migrate JSONL cache files from kq_stream_highlights."""
    sharded = ShardedGameDB(db_dir)

    jsonl_files = sorted(glob.glob(os.path.join(jsonl_dir, '*.jsonl')))
    if not jsonl_files:
        print(f"No JSONL files found in {jsonl_dir}")
        return

    print(f"Found {len(jsonl_files)} JSONL cache files")

    migrated = 0
    skipped = 0

    for filepath in jsonl_files:
        game_id = int(os.path.basename(filepath).replace('.jsonl', ''))

        # Read all events from JSONL
        raw_events = []
        game_uuid = None
        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                event_type = item['event_type']
                game_uuid = item.get('game_uuid')
                dt = datetime.datetime.fromisoformat(item['timestamp'])
                # Reconstruct values_str from list format
                vals = item.get('values', [])
                values_str = '{' + ','.join(str(v) for v in vals) + '}'
                raw_events.append((dt, event_type, values_str, item.get('win_probability')))

        if not raw_events:
            skipped += 1
            continue

        # Find gamestart, mapstart
        gamestart_dt = None
        map_name = None
        gold_on_left = None
        winning_team = None
        win_condition = None

        for dt, event_type, values_str, wp in raw_events:
            if event_type == 'gamestart' and gamestart_dt is None:
                gamestart_dt = dt
            if event_type == 'mapstart' and map_name is None:
                vals = values_str[1:-1].split(',')
                map_name = vals[0]
                gold_on_left = (vals[1] == 'True')
            if event_type == 'victory':
                vals = values_str[1:-1].split(',')
                winning_team = vals[0]
                if len(vals) > 1:
                    win_condition = vals[1]

        if gamestart_dt is None or map_name is None:
            skipped += 1
            continue

        # Build compact events with win probability
        compact_events = []
        for dt, event_type, values_str, wp in raw_events:
            if event_type in SKIP_EVENTS:
                continue
            rel_t = round((dt - gamestart_dt).total_seconds(), 4)
            evt = GameEvent(
                t=rel_t, type=event_type,
                vals=_parse_values(values_str),
                wp=wp,
            )
            compact_events.append(evt)

        compact_events.sort(key=lambda e: e.t)

        last_dt = raw_events[-1][0]
        duration = (last_dt - gamestart_dt).total_seconds()

        doc = GameDocument(
            game_id=game_id,
            game_uuid=game_uuid,
            map_name=map_name,
            gold_on_left=gold_on_left,
            start_time=gamestart_dt.isoformat(),
            end_time=last_dt.isoformat(),
            win_condition=win_condition,
            winning_team=winning_team,
            events=compact_events,
            duration_seconds=round(duration, 3),
        )

        sid = shard_id_for_game(game_id)
        db = sharded.get_or_create_shard(sid)
        db.insert_game(doc)
        db.commit()
        migrated += 1

    sharded.close()
    print(f"JSONL migration complete: {migrated} games, {skipped} skipped")


def build_replicas(db_dir: str, tournament: bool = True,
                   high_skill: bool = True, min_mu: float = 30.0,
                   logged_in: bool = True, verbose: bool = True):
    """Step 6: Build replica DBs by scanning all shards."""
    from rebuild_replicas import rebuild_replicas
    rebuild_replicas(db_dir, tournament=tournament, high_skill=high_skill,
                     min_mu=min_mu, logged_in=logged_in, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description='Migrate game data to SQLite')
    parser.add_argument('--db-dir', required=True, help='Output directory for shard DBs')
    parser.add_argument('--csv-dir', default=None,
                        help='Directory with gameevents_*.csv.gz, game.csv, usergame.csv')
    parser.add_argument('--jsonl-dir', default=None,
                        help='Directory with JSONL cache files')
    parser.add_argument('--ratings', default=None,
                        help='Path to ratings pickle file')
    parser.add_argument('--enrich', action='store_true',
                        help='Enrich with game.csv + usergame.csv metadata')
    parser.add_argument('--replicas', action='store_true',
                        help='Build replica DBs')
    parser.add_argument('--min-mu', type=float, default=30.0,
                        help='Min mu threshold for high-skill replica')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline')
    args = parser.parse_args()
    _set_resource_limits()

    if args.all:
        if not args.csv_dir:
            args.csv_dir = os.path.join(os.path.dirname(__file__),
                                        'unfiltered_partitioned')
        migrate_csv_partitions(args.db_dir, args.csv_dir)
        enrich_with_metadata(args.db_dir, args.csv_dir)
        if args.ratings:
            migrate_ratings(args.db_dir, args.ratings)
        jsonl_dir = os.path.join(os.path.dirname(__file__),
                                 '..', 'kq_stream_highlights', 'cache', 'game_events')
        if os.path.isdir(jsonl_dir):
            migrate_jsonl_cache(args.db_dir, jsonl_dir)
        build_replicas(args.db_dir, min_mu=args.min_mu)
        return

    if args.csv_dir and not args.enrich:
        migrate_csv_partitions(args.db_dir, args.csv_dir)

    if args.enrich:
        csv_dir = args.csv_dir or os.path.join(os.path.dirname(__file__),
                                                'unfiltered_partitioned')
        enrich_with_metadata(args.db_dir, csv_dir)

    if args.jsonl_dir:
        migrate_jsonl_cache(args.db_dir, args.jsonl_dir)

    if args.ratings:
        migrate_ratings(args.db_dir, args.ratings)

    if args.replicas:
        build_replicas(args.db_dir, min_mu=args.min_mu)


if __name__ == '__main__':
    main()
