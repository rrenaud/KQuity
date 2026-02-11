"""SQLite-backed document database for KQuity game events + metadata.

Unifies game events from CSV.gz partitions, JSONL cache files, and metadata
CSVs into a single queryable store with sharding by game_id.

Sharding: shard_id = game_id // GAMES_PER_SHARD
Each shard is a self-contained SQLite DB with identical schema.
Replica DBs (tournament, high-skill, logged-in) contain full game documents
for fast sequential access to interesting subsets.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from pydantic import BaseModel


# --- Sharding constants ---

GAMES_PER_SHARD = 17000


# --- Pydantic models ---

class GameEvent(BaseModel):
    t: float                    # relative seconds from gamestart
    type: str                   # 'spawn', 'playerKill', etc.
    vals: list[int | float | str | bool] = []
    wp: float | None = None     # win probability, if available


class GameDocument(BaseModel):
    game_id: int
    game_uuid: str | None = None
    map_name: str
    gold_on_left: bool
    cabinet_name: str | None = None
    scene_name: str | None = None
    start_time: str             # ISO 8601 absolute timestamp
    end_time: str | None = None
    win_condition: str | None = None
    winning_team: str | None = None
    player_count: int | None = None
    tournament_match_id: int | None = None
    events: list[GameEvent]
    duration_seconds: float | None = None
    login_count: int = 0
    max_player_mu: float | None = None
    avg_player_mu: float | None = None


class PlayerEntry(BaseModel):
    game_id: int
    position_id: int            # 1-10
    user_id: int | None = None
    user_name: str | None = None
    role: str                   # 'queen' or 'drone'


class GameMetadata(BaseModel):
    game_id: int
    key: str
    value: dict
    updated_at: str


# --- Schema ---

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
    game_id              INTEGER PRIMARY KEY,
    game_uuid            TEXT,
    map_name             TEXT NOT NULL,
    gold_on_left         INTEGER NOT NULL,
    cabinet_name         TEXT,
    scene_name           TEXT,
    start_time           TEXT NOT NULL,
    end_time             TEXT,
    win_condition        TEXT,
    winning_team         TEXT,
    player_count         INTEGER,
    tournament_match_id  INTEGER,
    events               TEXT NOT NULL,
    event_count          INTEGER NOT NULL,
    duration_seconds     REAL,
    login_count          INTEGER DEFAULT 0,
    max_player_mu        REAL,
    avg_player_mu        REAL
);

CREATE TABLE IF NOT EXISTS game_players (
    game_id      INTEGER NOT NULL REFERENCES games(game_id),
    position_id  INTEGER NOT NULL,
    user_id      INTEGER,
    user_name    TEXT,
    role         TEXT NOT NULL,
    PRIMARY KEY (game_id, position_id)
);

CREATE TABLE IF NOT EXISTS game_metadata (
    game_id    INTEGER NOT NULL REFERENCES games(game_id),
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (game_id, key)
);

CREATE INDEX IF NOT EXISTS idx_tournament ON games(tournament_match_id)
    WHERE tournament_match_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cabinet ON games(cabinet_name, start_time);
CREATE INDEX IF NOT EXISTS idx_scene ON games(scene_name, start_time);
CREATE INDEX IF NOT EXISTS idx_start_time ON games(start_time);
CREATE INDEX IF NOT EXISTS idx_max_mu ON games(max_player_mu)
    WHERE max_player_mu IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_player_user ON game_players(user_id)
    WHERE user_id IS NOT NULL;
"""


def init_db(conn: sqlite3.Connection):
    """Create schema tables and indexes."""
    conn.executescript(SCHEMA_SQL)


def shard_id_for_game(game_id: int) -> int:
    return game_id // GAMES_PER_SHARD


def shard_filename(shard_id: int) -> str:
    return f"shard_{shard_id:05d}.db"


# --- Serialization helpers ---

def _serialize_events(events: list[GameEvent]) -> str:
    return json.dumps([e.model_dump(exclude_none=True) for e in events])


def _deserialize_events(events_json: str) -> list[GameEvent]:
    return [GameEvent.model_validate(e) for e in json.loads(events_json)]


def _row_to_game_document(row: sqlite3.Row) -> GameDocument:
    return GameDocument(
        game_id=row['game_id'],
        game_uuid=row['game_uuid'],
        map_name=row['map_name'],
        gold_on_left=bool(row['gold_on_left']),
        cabinet_name=row['cabinet_name'],
        scene_name=row['scene_name'],
        start_time=row['start_time'],
        end_time=row['end_time'],
        win_condition=row['win_condition'],
        winning_team=row['winning_team'],
        player_count=row['player_count'],
        tournament_match_id=row['tournament_match_id'],
        events=_deserialize_events(row['events']),
        duration_seconds=row['duration_seconds'],
        login_count=row['login_count'] or 0,
        max_player_mu=row['max_player_mu'],
        avg_player_mu=row['avg_player_mu'],
    )


# --- GameDB class ---

class GameDB:
    """Interface to a single SQLite game database (shard or replica).

    For shard-routing across multiple files, use ShardedGameDB.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _get_conn(self) -> sqlite3.Connection:
        return self.conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def create_schema(self):
        init_db(self._get_conn())

    # --- Insert operations ---

    def insert_game(self, doc: GameDocument):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO games
               (game_id, game_uuid, map_name, gold_on_left, cabinet_name,
                scene_name, start_time, end_time, win_condition, winning_team,
                player_count, tournament_match_id, events, event_count,
                duration_seconds, login_count, max_player_mu, avg_player_mu)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (doc.game_id, doc.game_uuid, doc.map_name, int(doc.gold_on_left),
             doc.cabinet_name, doc.scene_name, doc.start_time, doc.end_time,
             doc.win_condition, doc.winning_team, doc.player_count,
             doc.tournament_match_id, _serialize_events(doc.events),
             len(doc.events), doc.duration_seconds, doc.login_count,
             doc.max_player_mu, doc.avg_player_mu))

    def insert_games_batch(self, docs: list[GameDocument]):
        conn = self._get_conn()
        rows = []
        for doc in docs:
            rows.append((
                doc.game_id, doc.game_uuid, doc.map_name, int(doc.gold_on_left),
                doc.cabinet_name, doc.scene_name, doc.start_time, doc.end_time,
                doc.win_condition, doc.winning_team, doc.player_count,
                doc.tournament_match_id, _serialize_events(doc.events),
                len(doc.events), doc.duration_seconds, doc.login_count,
                doc.max_player_mu, doc.avg_player_mu))
        conn.executemany(
            """INSERT OR REPLACE INTO games
               (game_id, game_uuid, map_name, gold_on_left, cabinet_name,
                scene_name, start_time, end_time, win_condition, winning_team,
                player_count, tournament_match_id, events, event_count,
                duration_seconds, login_count, max_player_mu, avg_player_mu)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows)

    def insert_player(self, player: PlayerEntry):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO game_players
               (game_id, position_id, user_id, user_name, role)
               VALUES (?,?,?,?,?)""",
            (player.game_id, player.position_id, player.user_id,
             player.user_name, player.role))

    def insert_players_batch(self, players: list[PlayerEntry]):
        conn = self._get_conn()
        conn.executemany(
            """INSERT OR REPLACE INTO game_players
               (game_id, position_id, user_id, user_name, role)
               VALUES (?,?,?,?,?)""",
            [(p.game_id, p.position_id, p.user_id, p.user_name, p.role)
             for p in players])

    def commit(self):
        if self._conn is not None:
            self._conn.commit()

    # --- Core lookups ---

    def get_game(self, game_id: int) -> GameDocument | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None
        return _row_to_game_document(row)

    def get_events(self, game_id: int) -> list[GameEvent] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT events FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_events(row['events'])

    def game_count(self) -> int:
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]

    # --- Metadata ---

    def get_metadata(self, game_id: int, key: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM game_metadata WHERE game_id = ? AND key = ?",
            (game_id, key)).fetchone()
        if row is None:
            return None
        return json.loads(row['value'])

    def set_metadata(self, game_id: int, key: str, value: dict):
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO game_metadata (game_id, key, value, updated_at)
               VALUES (?, ?, ?, ?)""",
            (game_id, key, json.dumps(value),
             datetime.now(timezone.utc).isoformat()))

    # --- Bulk iteration ---

    def iter_games(self, where: str = "1=1", params: tuple = ()) -> Iterator[GameDocument]:
        conn = self._get_conn()
        cursor = conn.execute(f"SELECT * FROM games WHERE {where}", params)
        for row in cursor:
            yield _row_to_game_document(row)

    def list_game_ids(self, where: str = "1=1", params: tuple = ()) -> list[int]:
        conn = self._get_conn()
        rows = conn.execute(
            f"SELECT game_id FROM games WHERE {where}", params).fetchall()
        return [r['game_id'] for r in rows]

    # --- Convenience queries ---

    def tournament_game_ids(self, tournament_id: int | None = None) -> list[int]:
        if tournament_id is not None:
            return self.list_game_ids(
                "tournament_match_id = ?", (tournament_id,))
        return self.list_game_ids("tournament_match_id IS NOT NULL")

    def high_skill_game_ids(self, min_mu: float = 30.0) -> list[int]:
        return self.list_game_ids("max_player_mu >= ?", (min_mu,))

    def games_by_cabinet(self, cabinet_name: str,
                         start: str | None = None,
                         end: str | None = None) -> list[int]:
        conditions = ["cabinet_name = ?"]
        params: list = [cabinet_name]
        if start is not None:
            conditions.append("start_time >= ?")
            params.append(start)
        if end is not None:
            conditions.append("start_time <= ?")
            params.append(end)
        return self.list_game_ids(" AND ".join(conditions), tuple(params))

    # --- Ratings integration ---

    def get_ratings_by_game(self, game_ids: list[int]) -> dict[int, np.ndarray]:
        """Get ratings metadata for multiple games.

        Returns {game_id: np.array(10)} of pre-game mu values.
        """
        conn = self._get_conn()
        result = {}
        # Process in chunks to avoid SQLite variable limit
        chunk_size = 500
        for i in range(0, len(game_ids), chunk_size):
            chunk = game_ids[i:i + chunk_size]
            placeholders = ','.join('?' * len(chunk))
            rows = conn.execute(
                f"""SELECT game_id, value FROM game_metadata
                    WHERE key = 'ratings' AND game_id IN ({placeholders})""",
                chunk).fetchall()
            for row in rows:
                mu_list = json.loads(row['value'])
                result[row['game_id']] = np.array(mu_list, dtype=np.float32)
        return result

    # --- Update helpers ---

    VALID_COLUMNS = frozenset({
        'game_uuid', 'map_name', 'gold_on_left', 'cabinet_name', 'scene_name',
        'start_time', 'end_time', 'win_condition', 'winning_team',
        'player_count', 'tournament_match_id', 'events', 'event_count',
        'duration_seconds', 'login_count', 'max_player_mu', 'avg_player_mu',
    })

    def update_game_field(self, game_id: int, field: str, value):
        """Update a single column on the games table."""
        if field not in self.VALID_COLUMNS:
            raise ValueError(f"Invalid column: {field!r}")
        conn = self._get_conn()
        conn.execute(
            f"UPDATE games SET {field} = ? WHERE game_id = ?",
            (value, game_id))


class ShardedGameDB:
    """Shard-routing wrapper over a directory of shard_*.db files + replicas."""

    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        self._shard_cache: dict[int, GameDB] = {}

    def _get_shard(self, game_id: int) -> GameDB:
        sid = shard_id_for_game(game_id)
        if sid not in self._shard_cache:
            path = self.db_dir / shard_filename(sid)
            self._shard_cache[sid] = GameDB(path)
        return self._shard_cache[sid]

    def get_or_create_shard(self, shard_id: int) -> GameDB:
        if shard_id not in self._shard_cache:
            path = self.db_dir / shard_filename(shard_id)
            db = GameDB(path)
            db.create_schema()
            self._shard_cache[shard_id] = db
        return self._shard_cache[shard_id]

    def get_game(self, game_id: int) -> GameDocument | None:
        return self._get_shard(game_id).get_game(game_id)

    def get_events(self, game_id: int) -> list[GameEvent] | None:
        return self._get_shard(game_id).get_events(game_id)

    def tournament_db(self) -> Path:
        return self.db_dir / "tournament_games.db"

    def high_skill_db(self) -> Path:
        return self.db_dir / "high_skill_games.db"

    def logged_in_db(self) -> Path:
        return self.db_dir / "logged_in_games.db"

    def iter_shards(self) -> Iterator[GameDB]:
        """Iterate over all shard DB files in the directory."""
        for path in sorted(self.db_dir.glob("shard_*.db")):
            yield GameDB(path)

    def close(self):
        for db in self._shard_cache.values():
            db.close()
        self._shard_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
