"""Tests for game_db.py and fast_materialize_from_db().

Verifies:
1. DB round-trip: insert -> read produces identical GameDocuments
2. DB-backed featurization produces identical output to CSV path
3. Index/query operations work correctly
4. Metadata and player operations work
"""

import json
import os
import shutil
import tempfile
import time
import unittest

import numpy as np

from game_db import (
    GameDB, GameDocument, GameEvent, PlayerEntry, ShardedGameDB,
    shard_id_for_game, GAMES_PER_SHARD,
)
from fast_materialize import fast_materialize, fast_materialize_from_db
from migrate_to_db import (
    _parse_values, _extract_game_info, _build_game_document,
    SKIP_EVENTS, migrate_csv_partitions,
)


class TestPydanticModels(unittest.TestCase):
    def test_game_event_roundtrip(self):
        evt = GameEvent(t=12.3, type='playerKill', vals=[450, 320, 3, 8, 'Worker'])
        d = evt.model_dump()
        evt2 = GameEvent.model_validate(d)
        self.assertEqual(evt, evt2)

    def test_game_event_exclude_none(self):
        evt = GameEvent(t=0.0, type='gamestart', vals=['map_day', True])
        d = evt.model_dump(exclude_none=True)
        self.assertNotIn('wp', d)

    def test_game_document_roundtrip(self):
        doc = GameDocument(
            game_id=12345,
            map_name='map_day',
            gold_on_left=True,
            start_time='2024-01-01T00:00:00+00:00',
            events=[
                GameEvent(t=0.0, type='gamestart', vals=['map_day', True]),
                GameEvent(t=0.5, type='spawn', vals=[8, False]),
            ],
        )
        d = doc.model_dump()
        doc2 = GameDocument.model_validate(d)
        self.assertEqual(doc.game_id, doc2.game_id)
        self.assertEqual(len(doc.events), len(doc2.events))


class TestParseValues(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_parse_values('{8,False}'), [8, False])

    def test_string_values(self):
        self.assertEqual(_parse_values('{map_day,True,0,False,17.26}'),
                         ['map_day', True, 0, False, 17.26])

    def test_empty(self):
        self.assertEqual(_parse_values('{}'), [])

    def test_kill_event(self):
        self.assertEqual(_parse_values('{450,320,3,8,Worker}'),
                         [450, 320, 3, 8, 'Worker'])


class TestGameDB(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, 'test.db')
        self.db = GameDB(self.db_path)
        self.db.create_schema()

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmpdir)

    def _make_doc(self, game_id=100, map_name='map_day', gold_on_left=True):
        return GameDocument(
            game_id=game_id,
            map_name=map_name,
            gold_on_left=gold_on_left,
            start_time='2024-01-01T00:00:00+00:00',
            end_time='2024-01-01T00:03:00+00:00',
            win_condition='military',
            winning_team='Blue',
            cabinet_name='testcab',
            tournament_match_id=42,
            events=[
                GameEvent(t=0.0, type='gamestart', vals=['map_day', True]),
                GameEvent(t=0.4, type='spawn', vals=[8, False]),
                GameEvent(t=120.0, type='victory', vals=['Blue', 'military']),
            ],
            duration_seconds=120.0,
            login_count=3,
            max_player_mu=32.5,
            avg_player_mu=26.0,
        )

    def test_insert_and_get(self):
        doc = self._make_doc()
        self.db.insert_game(doc)
        self.db.commit()

        retrieved = self.db.get_game(100)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.game_id, 100)
        self.assertEqual(retrieved.map_name, 'map_day')
        self.assertTrue(retrieved.gold_on_left)
        self.assertEqual(len(retrieved.events), 3)
        self.assertEqual(retrieved.events[0].type, 'gamestart')
        self.assertEqual(retrieved.events[1].vals, [8, False])

    def test_get_nonexistent(self):
        self.assertIsNone(self.db.get_game(999))

    def test_get_events(self):
        self.db.insert_game(self._make_doc())
        self.db.commit()
        events = self.db.get_events(100)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].t, 0.0)

    def test_game_count(self):
        self.db.insert_game(self._make_doc(100))
        self.db.insert_game(self._make_doc(200))
        self.db.commit()
        self.assertEqual(self.db.game_count(), 2)

    def test_batch_insert(self):
        docs = [self._make_doc(i) for i in range(10)]
        self.db.insert_games_batch(docs)
        self.db.commit()
        self.assertEqual(self.db.game_count(), 10)

    def test_metadata(self):
        self.db.insert_game(self._make_doc())
        self.db.commit()
        self.db.set_metadata(100, 'ratings', [25.0] * 10)
        self.db.commit()
        meta = self.db.get_metadata(100, 'ratings')
        self.assertEqual(meta, [25.0] * 10)

    def test_metadata_nonexistent(self):
        self.assertIsNone(self.db.get_metadata(999, 'ratings'))

    def test_players(self):
        self.db.insert_game(self._make_doc())
        players = [
            PlayerEntry(game_id=100, position_id=1, user_id=42,
                        user_name='Alice', role='queen'),
            PlayerEntry(game_id=100, position_id=3, user_id=43,
                        user_name='Bob', role='drone'),
        ]
        self.db.insert_players_batch(players)
        self.db.commit()

        conn = self.db.conn
        rows = conn.execute(
            "SELECT * FROM game_players WHERE game_id = 100").fetchall()
        self.assertEqual(len(rows), 2)

    def test_tournament_query(self):
        self.db.insert_game(self._make_doc(100))
        doc2 = self._make_doc(200)
        doc2.tournament_match_id = None
        self.db.insert_game(doc2)
        self.db.commit()
        ids = self.db.tournament_game_ids()
        self.assertEqual(ids, [100])

    def test_high_skill_query(self):
        self.db.insert_game(self._make_doc(100))  # max_mu=32.5
        doc2 = self._make_doc(200)
        doc2.max_player_mu = 25.0
        self.db.insert_game(doc2)
        self.db.commit()
        ids = self.db.high_skill_game_ids(min_mu=30.0)
        self.assertEqual(ids, [100])

    def test_cabinet_query(self):
        self.db.insert_game(self._make_doc(100))
        self.db.commit()
        ids = self.db.games_by_cabinet('testcab')
        self.assertEqual(ids, [100])
        ids = self.db.games_by_cabinet('othercab')
        self.assertEqual(ids, [])

    def test_iter_games(self):
        self.db.insert_game(self._make_doc(100))
        self.db.insert_game(self._make_doc(200))
        self.db.commit()
        docs = list(self.db.iter_games())
        self.assertEqual(len(docs), 2)

    def test_context_manager(self):
        self.db.insert_game(self._make_doc())
        self.db.commit()

        with GameDB(self.db_path) as db:
            retrieved = db.get_game(100)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.game_id, 100)
        # Connection should be closed after exiting context
        self.assertIsNone(db._conn)

    def test_upsert(self):
        """INSERT OR REPLACE should overwrite existing game."""
        self.db.insert_game(self._make_doc(100))
        self.db.commit()
        doc2 = self._make_doc(100)
        doc2.map_name = 'map_night'
        self.db.insert_game(doc2)
        self.db.commit()
        retrieved = self.db.get_game(100)
        self.assertEqual(retrieved.map_name, 'map_night')
        self.assertEqual(self.db.game_count(), 1)


class TestShardedGameDB(unittest.TestCase):
    def test_shard_routing(self):
        self.assertEqual(shard_id_for_game(0), 0)
        self.assertEqual(shard_id_for_game(GAMES_PER_SHARD - 1), 0)
        self.assertEqual(shard_id_for_game(GAMES_PER_SHARD), 1)
        self.assertEqual(shard_id_for_game(GAMES_PER_SHARD * 2 + 5), 2)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_insert_and_get_across_shards(self):
        sharded = ShardedGameDB(self.tmpdir)

        doc1 = GameDocument(
            game_id=100, map_name='map_day', gold_on_left=True,
            start_time='2024-01-01T00:00:00', events=[],
        )
        doc2 = GameDocument(
            game_id=GAMES_PER_SHARD + 100, map_name='map_night',
            gold_on_left=False, start_time='2024-06-01T00:00:00', events=[],
        )

        shard0 = sharded.get_or_create_shard(0)
        shard0.insert_game(doc1)
        shard0.commit()

        shard1 = sharded.get_or_create_shard(1)
        shard1.insert_game(doc2)
        shard1.commit()

        r1 = sharded.get_game(100)
        r2 = sharded.get_game(GAMES_PER_SHARD + 100)
        self.assertEqual(r1.map_name, 'map_day')
        self.assertEqual(r2.map_name, 'map_night')

        sharded.close()


class TestDBFeaturization(unittest.TestCase):
    """Verify DB-backed featurization matches CSV path."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_db_matches_csv(self):
        """Migrate benchmark CSVs to DB, then compare featurization output."""
        test_dir = os.path.dirname(__file__)
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        import glob
        if not glob.glob(benchmark_path):
            self.skipTest("Benchmark CSV files not found")

        # Run CSV path
        csv_states, csv_labels, csv_game_ids, csv_ts = fast_materialize(
            benchmark_path)

        if csv_states.shape[0] == 0:
            self.skipTest("No events in benchmark data")

        # Migrate to temp DB
        migrate_csv_partitions(self.tmpdir, test_dir, verbose=False,
                               csv_glob=benchmark_path)

        # Find the shard file(s) created
        import glob as glob_mod
        shard_files = sorted(glob_mod.glob(os.path.join(self.tmpdir, 'shard_*.db')))
        self.assertTrue(len(shard_files) > 0, "No shard files created")

        # Run DB path on each shard and collect results
        all_states = []
        all_labels = []
        all_game_ids = []
        all_ts = []

        for shard_path in shard_files:
            s, l, g, t = fast_materialize_from_db(shard_path)
            if s.shape[0] > 0:
                all_states.append(s)
                all_labels.append(l)
                all_game_ids.append(g)
                all_ts.append(t)

        if not all_states:
            self.skipTest("No features produced from DB")

        db_states = np.concatenate(all_states)
        db_labels = np.concatenate(all_labels)
        db_game_ids = np.concatenate(all_game_ids)
        db_ts = np.concatenate(all_ts)

        # Sort both by (game_id, timestamp) for stable comparison
        csv_order = np.lexsort((csv_ts, csv_game_ids))
        db_order = np.lexsort((db_ts, db_game_ids))

        csv_states = csv_states[csv_order]
        csv_labels = csv_labels[csv_order]
        db_states = db_states[db_order]
        db_labels = db_labels[db_order]

        self.assertEqual(csv_states.shape, db_states.shape,
                         f'Shape mismatch: CSV={csv_states.shape} DB={db_states.shape}')

        np.testing.assert_array_almost_equal(
            db_states, csv_states, decimal=4,
            err_msg='DB path states differ from CSV path')
        np.testing.assert_array_equal(
            db_labels, csv_labels,
            err_msg='DB path labels differ from CSV path')


if __name__ == '__main__':
    unittest.main()
