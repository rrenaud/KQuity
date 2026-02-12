import csv
import glob
import gzip
import os
import random
import time
import unittest

import numpy as np

from constants import ContestableState, Team
from fast_materialize import (
    _MAP_LOOKUPS, _parse_ts, _process_game, _vectorize_state,
    MAP_INDEX, NUM_FEATURES, SCREEN_WIDTH, SKIP_EVENTS, SPEED_SNAIL_PPS,
    COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
)
from event_codec import encode_game, walk_game_states


def _read_benchmark_games():
    """Read benchmark CSVs and return dict {game_id: raw_events}."""
    test_dir = os.path.dirname(__file__)
    pattern = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

    games = {}
    game_order = []
    for filename in sorted(glob.glob(pattern)):
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                event_type = row[COL_TYPE]
                if event_type in SKIP_EVENTS:
                    continue
                game_id = int(row[COL_GAME_ID])
                if game_id not in games:
                    games[game_id] = []
                    game_order.append(game_id)
                games[game_id].append((_parse_ts(row[COL_TS]), event_type, row[COL_VALUES]))
    return games, game_order


def _game_state_to_vectorize_args(game_state):
    """Extract the raw variables from GameState for _vectorize_state."""
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


class TestEventCodecCorrectness(unittest.TestCase):
    """Compare binary codec path against fast_materialize for every benchmark game."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.game_order = _read_benchmark_games()

    def test_encode_walk_matches_fast_path(self):
        """Feature vectors from binary walk must match fast_materialize.

        Aligns events by index (not timestamp) to avoid centisecond boundary
        effects at the rel_ts > 5.0 threshold.
        """
        games = self.games
        game_order = self.game_order

        total_matched = 0
        total_games = 0
        max_snail_diff = 0.0

        for game_id in game_order:
            raw_events = [list(e) for e in games[game_id]]

            # --- fast path reference ---
            ref_buf = np.empty((len(raw_events), NUM_FEATURES), dtype=np.float32)
            ref_labels = np.empty(len(raw_events), dtype=np.int8)
            ref_ts = np.empty(len(raw_events), dtype=np.float32)
            rng = random.Random(42)
            ref_events = [tuple(e) for e in raw_events]
            ref_idx = _process_game(ref_events, ref_buf, ref_labels, ref_ts, 0, 0.0, rng)
            if ref_idx == 0:
                continue
            ref_features = ref_buf[:ref_idx]
            ref_timestamps = ref_ts[:ref_idx]

            # Find the exact rel_ts threshold: the minimum ref timestamp tells
            # us which event index the fast path started vectorizing at.
            # We use ref_idx to know how many events were vectorized.

            # --- binary codec path ---
            # Collect ALL walker states (not just > 5.0) so we can align by
            # event index. Both paths iterate events in the same sorted order.
            codec_events = [tuple(e) for e in raw_events]
            encoded = encode_game(codec_events)
            if encoded is None:
                continue  # anomalous game (e.g. >60s gap) rejected by encoder

            all_codec = []
            for rel_ts, gs in walk_game_states(encoded):
                args = _game_state_to_vectorize_args(gs)
                (w, eggs, food_count, maiden_states, map_idx,
                 snail_x, snail_vel, snail_last_ts,
                 berries_avail, gold_sym) = args
                row_buf = np.empty((1, NUM_FEATURES), dtype=np.float32)
                _vectorize_state(row_buf, 0, w, eggs, food_count,
                                 maiden_states, map_idx, snail_x,
                                 snail_vel, snail_last_ts,
                                 rel_ts, berries_avail, gold_sym)
                all_codec.append(row_buf[0].copy())

            # The fast path vectorized the LAST ref_idx events (events are
            # sorted by time; the first events have rel_ts <= 5.0).
            codec_features = np.array(all_codec[-ref_idx:], dtype=np.float32)

            self.assertEqual(
                ref_features.shape[0], codec_features.shape[0],
                f'Game {game_id}: row count mismatch '
                f'{ref_features.shape[0]} vs {codec_features.shape[0]}')

            # Non-snail features (indices 0-48, 51) must match exactly.
            non_snail = list(range(49)) + [51]
            np.testing.assert_array_almost_equal(
                codec_features[:, non_snail],
                ref_features[:, non_snail],
                decimal=5,
                err_msg=f'Game {game_id}: non-snail features differ')

            # Snail position (index 49) may differ due to centisecond rounding.
            snail_diff = np.max(np.abs(
                codec_features[:, 49] - ref_features[:, 49]))
            max_snail_diff = max(max_snail_diff, snail_diff)
            self.assertLess(
                snail_diff, 0.001,
                f'Game {game_id}: snail position diff {snail_diff:.6f}')

            # Snail speed (index 50) should match exactly (no timestamp dependence).
            np.testing.assert_array_almost_equal(
                codec_features[:, 50],
                ref_features[:, 50],
                decimal=5,
                err_msg=f'Game {game_id}: snail speed differs')

            total_matched += ref_features.shape[0]
            total_games += 1

        print(f'\nCorrectness: {total_games} games, {total_matched} states matched')
        print(f'Max snail position diff: {max_snail_diff:.6f}')
        self.assertGreater(total_games, 100, 'Expected >100 valid games')

    def test_victory_fields_set(self):
        """After walking a game, game_state.winning_team matches CSV label."""
        games = self.games
        game_order = self.game_order
        total_checked = 0

        for game_id in game_order:
            raw_events = games[game_id]

            # Find expected winner from the CSV victory event
            expected_team = None
            for _dt, etype, vals in raw_events:
                if etype == 'victory':
                    v = vals[1:-1].split(',')
                    expected_team = Team.BLUE if v[0] == 'Blue' else Team.GOLD
            if expected_team is None:
                continue

            encoded = encode_game(list(raw_events))
            if encoded is None:
                continue

            # Walk all states; after the loop gs has the final mutations applied
            gs = None
            for _rel_ts, gs in walk_game_states(encoded):
                pass

            self.assertIsNotNone(gs.winning_team,
                                 f'Game {game_id}: winning_team not set')
            self.assertEqual(gs.winning_team, expected_team,
                             f'Game {game_id}: expected {expected_team}, '
                             f'got {gs.winning_team}')
            self.assertIsNotNone(gs.victory_condition,
                                 f'Game {game_id}: victory_condition not set')
            total_checked += 1

        print(f'\nVictory fields: {total_checked} games checked')
        self.assertGreater(total_checked, 100, 'Expected >100 games with victory')


class TestEventCodecCompression(unittest.TestCase):
    """Measure compression ratio of binary encoding vs CSV."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.game_order = _read_benchmark_games()

    def test_compression_ratio(self):
        games = self.games
        total_binary = 0
        total_csv = 0
        encoded_count = 0

        for game_id in self.game_order:
            raw_events = games[game_id]

            csv_size = sum(
                len(f'{dt.isoformat()},{etype},{vals}')
                for dt, etype, vals in raw_events
            )

            encoded = encode_game(list(raw_events))
            if encoded is None:
                continue

            total_binary += len(encoded)
            total_csv += csv_size
            encoded_count += 1

        avg_binary = total_binary / encoded_count
        avg_csv = total_csv / encoded_count
        ratio = total_csv / total_binary

        print(f'\nCompression: {encoded_count} games')
        print(f'  Avg binary: {avg_binary:.0f} bytes/game')
        print(f'  Avg CSV:    {avg_csv:.0f} bytes/game')
        print(f'  Ratio:      {ratio:.1f}x')
        self.assertGreater(ratio, 5.0, 'Expected >5x compression')


class TestEventCodecSpeed(unittest.TestCase):
    """Benchmark walk_game_states vs fast_materialize._process_game."""

    @classmethod
    def setUpClass(cls):
        cls.games, cls.game_order = _read_benchmark_games()

    def test_speed_benchmark(self):
        """Both paths include featurization for a fair comparison."""
        games = self.games
        game_order = self.game_order

        # Pre-encode all games
        encoded_games = {}
        for game_id in game_order:
            enc = encode_game(list(games[game_id]))
            if enc is not None:
                encoded_games[game_id] = enc

        # Benchmark binary walk + vectorization
        row_buf = np.empty((1, NUM_FEATURES), dtype=np.float32)
        total_events_bin = 0
        start = time.perf_counter()
        for game_id, enc in encoded_games.items():
            for rel_ts, gs in walk_game_states(enc):
                if rel_ts > 5.0:
                    args = _game_state_to_vectorize_args(gs)
                    (w, eggs, food_count, maiden_states, map_idx,
                     snail_x, snail_vel, snail_last_ts,
                     berries_avail, gold_sym) = args
                    _vectorize_state(row_buf, 0, w, eggs, food_count,
                                     maiden_states, map_idx, snail_x,
                                     snail_vel, snail_last_ts,
                                     rel_ts, berries_avail, gold_sym)
                    total_events_bin += 1
        binary_elapsed = time.perf_counter() - start

        # Benchmark fast_materialize._process_game (includes vectorization)
        total_events_csv = 0
        buf = np.empty((5000, NUM_FEATURES), dtype=np.float32)
        lbl = np.empty(5000, dtype=np.int8)
        ts_buf = np.empty(5000, dtype=np.float32)
        start = time.perf_counter()
        for game_id in encoded_games:
            raw = list(games[game_id])
            rng = random.Random(42)
            n = _process_game(raw, buf, lbl, ts_buf, 0, 0.0, rng)
            total_events_csv += n
        csv_elapsed = time.perf_counter() - start

        bin_rate = total_events_bin / binary_elapsed
        csv_rate = total_events_csv / csv_elapsed

        print(f'\nSpeed benchmark ({len(encoded_games)} games, both include featurization):')
        print(f'  Binary walk: {binary_elapsed:.3f}s, {bin_rate:.0f} states/s')
        print(f'  CSV fast:    {csv_elapsed:.3f}s, {csv_rate:.0f} states/s')
        print(f'  Speedup:     {bin_rate / csv_rate:.2f}x')


if __name__ == '__main__':
    unittest.main()
