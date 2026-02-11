"""Build replica DBs from shards for fast access to interesting subsets.

Scans all shard_*.db files and copies matching game rows into dedicated
replica SQLite files. Each replica is self-contained with full game documents.

Usage:
    python rebuild_replicas.py --db-dir data/game_db/ --tournament --high-skill --min-mu 30.0
    python rebuild_replicas.py --db-dir data/game_db/ --all
"""

import argparse
import os

from game_db import GameDB, ShardedGameDB


def _copy_matching_games(sharded: ShardedGameDB, replica_path: str,
                         where: str, params: tuple = (),
                         verbose: bool = True) -> int:
    """Copy games matching a WHERE clause from all shards into a replica DB.

    Uses ATTACH DATABASE + INSERT INTO ... SELECT FROM for bulk copy,
    which is much faster than row-by-row Python iteration.
    """
    replica = GameDB(replica_path)
    replica.create_schema()
    conn_r = replica.conn

    total = 0
    for shard_db in sharded.iter_shards():
        shard_path = str(shard_db.db_path)

        # ATTACH the shard, bulk-copy matching rows, DETACH
        conn_r.execute("ATTACH DATABASE ? AS src", (shard_path,))

        # Count matching games first
        count = conn_r.execute(
            f"SELECT COUNT(*) FROM src.games WHERE {where}", params
        ).fetchone()[0]

        if count == 0:
            conn_r.execute("DETACH DATABASE src")
            continue

        conn_r.execute(
            f"""INSERT OR REPLACE INTO games
                SELECT * FROM src.games WHERE {where}""", params)

        # Copy player rows for matching games
        conn_r.execute(
            f"""INSERT OR REPLACE INTO game_players
                SELECT p.* FROM src.game_players p
                WHERE p.game_id IN (SELECT game_id FROM src.games WHERE {where})""",
            params)

        # Copy metadata rows for matching games
        conn_r.execute(
            f"""INSERT OR REPLACE INTO game_metadata
                SELECT m.* FROM src.game_metadata m
                WHERE m.game_id IN (SELECT game_id FROM src.games WHERE {where})""",
            params)

        conn_r.execute("DETACH DATABASE src")
        total += count

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
