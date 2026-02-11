"""Build replica DBs from shards for fast access to interesting subsets.

Scans all shard_*.db files and copies matching game rows into dedicated
replica SQLite files. Each replica is self-contained with full game documents.

Usage:
    python rebuild_replicas.py --db-dir data/game_db/ --tournament --high-skill --min-mu 30.0
    python rebuild_replicas.py --db-dir data/game_db/ --all
"""

import argparse
import os

from game_db import GameDB, ShardedGameDB, init_db


def _copy_matching_games(sharded: ShardedGameDB, replica_path: str,
                         where: str, params: tuple = (),
                         verbose: bool = True) -> int:
    """Copy games matching a WHERE clause from all shards into a replica DB."""
    replica = GameDB(replica_path)
    replica.create_schema()
    conn_r = replica._get_conn()

    total = 0
    for shard_db in sharded.iter_shards():
        conn_s = shard_db._get_conn()

        # Copy matching game rows
        rows = conn_s.execute(
            f"SELECT * FROM games WHERE {where}", params).fetchall()
        if not rows:
            shard_db.close()
            continue

        game_ids = [r['game_id'] for r in rows]

        for row in rows:
            conn_r.execute(
                """INSERT OR REPLACE INTO games
                   (game_id, game_uuid, map_name, gold_on_left, cabinet_name,
                    scene_name, start_time, end_time, win_condition, winning_team,
                    player_count, tournament_match_id, events, event_count,
                    duration_seconds, login_count, max_player_mu, avg_player_mu)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(row))

        # Copy matching player rows
        placeholders = ','.join('?' * len(game_ids))
        player_rows = conn_s.execute(
            f"""SELECT * FROM game_players
                WHERE game_id IN ({placeholders})""",
            game_ids).fetchall()
        for pr in player_rows:
            conn_r.execute(
                """INSERT OR REPLACE INTO game_players
                   (game_id, position_id, user_id, user_name, role)
                   VALUES (?,?,?,?,?)""",
                tuple(pr))

        # Copy matching metadata rows
        meta_rows = conn_s.execute(
            f"""SELECT * FROM game_metadata
                WHERE game_id IN ({placeholders})""",
            game_ids).fetchall()
        for mr in meta_rows:
            conn_r.execute(
                """INSERT OR REPLACE INTO game_metadata
                   (game_id, key, value, updated_at)
                   VALUES (?,?,?,?)""",
                tuple(mr))

        total += len(rows)
        shard_db.close()

    replica.commit()
    replica.close()
    return total


def rebuild_replicas(db_dir: str, tournament: bool = True,
                     high_skill: bool = True, min_mu: float = 30.0,
                     logged_in: bool = True, verbose: bool = True):
    """Build all requested replica DBs from shards."""
    sharded = ShardedGameDB(db_dir)

    if tournament:
        path = os.path.join(db_dir, 'tournament_games.db')
        if os.path.exists(path):
            os.remove(path)
        print("Building tournament replica...")
        n = _copy_matching_games(
            sharded, path,
            "tournament_match_id IS NOT NULL")
        print(f"  tournament_games.db: {n} games")

    if high_skill:
        path = os.path.join(db_dir, 'high_skill_games.db')
        if os.path.exists(path):
            os.remove(path)
        print(f"Building high-skill replica (min_mu={min_mu})...")
        n = _copy_matching_games(
            sharded, path,
            "max_player_mu >= ?", (min_mu,))
        print(f"  high_skill_games.db: {n} games")

    if logged_in:
        path = os.path.join(db_dir, 'logged_in_games.db')
        if os.path.exists(path):
            os.remove(path)
        print("Building logged-in replica...")
        n = _copy_matching_games(
            sharded, path,
            "login_count >= 1")
        print(f"  logged_in_games.db: {n} games")

    sharded.close()


def main():
    parser = argparse.ArgumentParser(description='Build replica DBs from shards')
    parser.add_argument('--db-dir', required=True, help='Directory with shard DBs')
    parser.add_argument('--tournament', action='store_true')
    parser.add_argument('--high-skill', action='store_true')
    parser.add_argument('--logged-in', action='store_true')
    parser.add_argument('--min-mu', type=float, default=30.0)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        args.tournament = True
        args.high_skill = True
        args.logged_in = True

    rebuild_replicas(
        args.db_dir,
        tournament=args.tournament,
        high_skill=args.high_skill,
        min_mu=args.min_mu,
        logged_in=args.logged_in,
    )


if __name__ == '__main__':
    main()
