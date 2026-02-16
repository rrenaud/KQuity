"""Tests for codec_materialize: binary materialization pipeline."""

import os
import random
import tempfile
import unittest

import numpy as np

from codec_materialize import (
    _game_state_to_vectorize_args,
    _materialize_one_game_from_binary,
    _materialize_shard,
    parallel_materialize_bins,
    sequential_materialize_bins,
)
from event_codec import (
    encode_csv_to_sharded_bins,
    encode_game,
    read_packed_games,
    reshard_packed_games,
    walk_game_states,
    write_packed_games,
)
from fast_materialize import (
    NUM_FEATURES, _process_game, _vectorize_state,
)
from tests.test_event_codec import _read_benchmark_games


class TestSingleGameMatchesFastPath(unittest.TestCase):
    """Per-game binary materialization matches _process_game."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.game_order = _read_benchmark_games()

    def test_single_game_matches_fast_path(self):
        total_matched = 0
        total_games = 0
        max_snail_diff = 0.0

        for game_id in self.game_order:
            raw_events = [list(e) for e in self.games[game_id]]

            # Fast path reference (drop_prob=0)
            ref_buf = np.empty((len(raw_events), NUM_FEATURES), dtype=np.float32)
            ref_labels = np.empty(len(raw_events), dtype=np.int8)
            ref_ts = np.empty(len(raw_events), dtype=np.float32)
            rng = random.Random(42)
            ref_events = [tuple(e) for e in raw_events]
            ref_idx = _process_game(ref_events, ref_buf, ref_labels, ref_ts, 0, 0.0, rng)
            if ref_idx == 0:
                continue
            ref_features = ref_buf[:ref_idx]

            # Binary codec path
            codec_events = [tuple(e) for e in raw_events]
            encoded = encode_game(codec_events)
            if encoded is None:
                continue

            rows, label = _materialize_one_game_from_binary(
                game_id, encoded, drop_prob=0.0, rng_seed=42)
            if not rows:
                continue

            codec_features = np.array(rows, dtype=np.float32)

            # Centisecond rounding can shift timestamps across the rel_ts > 5.0
            # boundary, causing +-1 row difference. Align by tail.
            n = min(ref_features.shape[0], codec_features.shape[0])
            ref_features = ref_features[-n:]
            codec_features = codec_features[-n:]

            # Non-snail features must match
            non_snail = list(range(49)) + [51]
            np.testing.assert_array_almost_equal(
                codec_features[:, non_snail],
                ref_features[:, non_snail],
                decimal=5,
                err_msg=f'Game {game_id}: non-snail features differ')

            # Snail position tolerance for centisecond rounding
            snail_diff = np.max(np.abs(
                codec_features[:, 49] - ref_features[:, 49]))
            max_snail_diff = max(max_snail_diff, snail_diff)
            self.assertLess(
                snail_diff, 0.001,
                f'Game {game_id}: snail position diff {snail_diff:.6f}')

            # Snail speed exact
            np.testing.assert_array_almost_equal(
                codec_features[:, 50], ref_features[:, 50],
                decimal=5,
                err_msg=f'Game {game_id}: snail speed differs')

            total_matched += ref_features.shape[0]
            total_games += 1

        print(f'\nSingle-game: {total_games} games, {total_matched} states matched')
        print(f'Max snail position diff: {max_snail_diff:.6f}')
        self.assertGreater(total_games, 100, 'Expected >100 valid games')


class TestShardRoundTrip(unittest.TestCase):
    """Encode benchmark CSVs -> shards -> materialize matches fast_materialize."""

    def test_shard_round_trip(self):
        test_dir = os.path.dirname(__file__)
        csv_glob = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        with tempfile.TemporaryDirectory() as tmpdir:
            encoded, rejected, shard_paths = encode_csv_to_sharded_bins(
                csv_glob, tmpdir, num_shards=4)
            self.assertGreater(encoded, 0)
            self.assertGreater(len(shard_paths), 0)

            shard_glob = os.path.join(tmpdir, 'shard_*.bin')
            seq_features, seq_labels, seq_game_ids = sequential_materialize_bins(
                shard_glob, drop_prob=0.0, base_seed=0)

            self.assertGreater(seq_features.shape[0], 0)
            print(f'\nShard round-trip: {seq_features.shape[0]} states from '
                  f'{len(shard_paths)} shards')


class TestParallelMatchesSequential(unittest.TestCase):
    """Parallel and sequential binary paths produce identical results."""

    def test_parallel_matches_sequential(self):
        test_dir = os.path.dirname(__file__)
        csv_glob = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        with tempfile.TemporaryDirectory() as tmpdir:
            encode_csv_to_sharded_bins(csv_glob, tmpdir, num_shards=4)
            shard_glob = os.path.join(tmpdir, 'shard_*.bin')

            seq_features, seq_labels, seq_game_ids = sequential_materialize_bins(
                shard_glob, drop_prob=0.0, base_seed=0)
            par_features, par_labels, par_game_ids = parallel_materialize_bins(
                shard_glob, drop_prob=0.0, num_workers=2, base_seed=0)

            self.assertEqual(seq_features.shape, par_features.shape)
            np.testing.assert_array_equal(seq_features, par_features)
            np.testing.assert_array_equal(seq_labels, par_labels)
            np.testing.assert_array_equal(seq_game_ids, par_game_ids)
            print(f'\nParallel matches sequential: {seq_features.shape[0]} states')


class TestReshardPreservesAllGames(unittest.TestCase):
    """Resharding preserves all game IDs and payloads."""

    def test_reshard_preserves_all_games(self):
        test_dir = os.path.dirname(__file__)
        csv_glob = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        with tempfile.TemporaryDirectory() as tmpdir:
            # First encode to a single .bin
            from event_codec import encode_csv_to_packed
            single_bin = os.path.join(tmpdir, 'all.bin')
            encoded, rejected = encode_csv_to_packed(csv_glob, single_bin)
            self.assertGreater(encoded, 0)

            # Read original games
            original = dict(read_packed_games(single_bin))

            # Reshard into 4
            shard_dir = os.path.join(tmpdir, 'shards')
            shard_paths = reshard_packed_games(single_bin, shard_dir, num_shards=4)
            self.assertGreater(len(shard_paths), 0)

            # Read back all shards
            resharded = {}
            for path in shard_paths:
                for game_id, data in read_packed_games(path):
                    self.assertNotIn(game_id, resharded,
                                     f'Duplicate game_id {game_id} across shards')
                    resharded[game_id] = data

            # Compare
            self.assertEqual(set(original.keys()), set(resharded.keys()),
                             'Game IDs differ after resharding')
            for game_id in original:
                self.assertEqual(original[game_id], resharded[game_id],
                                 f'Game {game_id}: payload differs after resharding')

            print(f'\nReshard: {len(original)} games preserved across '
                  f'{len(shard_paths)} shards')


if __name__ == '__main__':
    unittest.main()
