"""Tests for symmetry augmentation.

Verifies that:
1. Double swap is identity (roundtrip)
2. Spot checks on known vectors
3. Cross-verification: feature swap matches event stream swap + materialize
4. Label flip correctness
"""

import os
import unittest

import numpy as np

from symmetry import swap_teams, swap_event_stream, SWAP_PERM, SWAP_SIGN
from fast_materialize import (
    fast_materialize, _process_game, _parse_ts, NUM_FEATURES,
    SKIP_EVENTS, COL_TS, COL_TYPE, COL_VALUES, COL_GAME_ID,
)
import csv
import gzip


class TestSwapTeamsRoundtrip(unittest.TestCase):
    """swap(swap(X, y)) == (X, y)"""

    def test_roundtrip_random(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 52).astype(np.float32)
        y = rng.randint(0, 2, size=100).astype(np.int8)

        X2, y2 = swap_teams(X, y)
        X3, y3 = swap_teams(X2, y2)

        np.testing.assert_array_almost_equal(X3, X, decimal=6)
        np.testing.assert_array_equal(y3, y)

    def test_roundtrip_benchmark(self):
        """Roundtrip on actual benchmark data."""
        test_dir = os.path.join(os.path.dirname(__file__), 'tests')
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')
        expected_path = os.path.join(test_dir, 'benchmark_expected.npz')

        if not os.path.exists(expected_path):
            self.skipTest('Benchmark data not available')

        expected = np.load(expected_path)
        X = expected['states']
        y = expected['labels']

        X2, y2 = swap_teams(X, y)
        X3, y3 = swap_teams(X2, y2)

        np.testing.assert_array_almost_equal(X3, X, decimal=5)
        np.testing.assert_array_equal(y3, y)


class TestSwapTeamsSpotChecks(unittest.TestCase):
    """Verify specific feature positions are correctly swapped."""

    def _make_vector(self):
        """Create a recognizable feature vector."""
        X = np.zeros((1, 52), dtype=np.float32)
        # Blue team: eggs=2, food=3, vanilla=1, speed=0
        X[0, 0] = 2.0  # blue eggs
        X[0, 1] = 3.0  # blue food
        X[0, 2] = 1.0  # blue vanilla warriors
        X[0, 3] = 0.0  # blue speed warriors

        # Gold team: eggs=1, food=5, vanilla=0, speed=2
        X[0, 20] = 1.0  # gold eggs
        X[0, 21] = 5.0  # gold food
        X[0, 22] = 0.0  # gold vanilla warriors
        X[0, 23] = 2.0  # gold speed warriors

        # Maiden control: +1 (Blue), -1 (Gold), 0, +1, -1
        X[0, 40] = 1.0
        X[0, 41] = -1.0
        X[0, 42] = 0.0
        X[0, 43] = 1.0
        X[0, 44] = -1.0

        # Map one-hot: map_day
        X[0, 45] = 1.0

        # Snail
        X[0, 49] = 0.3   # snail position (Blue-favorable)
        X[0, 50] = 0.1   # snail velocity (Blue-favorable)

        # Berries
        X[0, 51] = 0.5

        return X

    def test_team_stats_swap(self):
        X = self._make_vector()
        y = np.array([1], dtype=np.int8)  # Blue wins

        X2, y2 = swap_teams(X, y)

        # Blue team stats should now have Gold's values
        self.assertAlmostEqual(X2[0, 0], 1.0)   # was gold eggs
        self.assertAlmostEqual(X2[0, 1], 5.0)   # was gold food
        self.assertAlmostEqual(X2[0, 2], 0.0)   # was gold vanilla
        self.assertAlmostEqual(X2[0, 3], 2.0)   # was gold speed

        # Gold team stats should now have Blue's values
        self.assertAlmostEqual(X2[0, 20], 2.0)  # was blue eggs
        self.assertAlmostEqual(X2[0, 21], 3.0)  # was blue food
        self.assertAlmostEqual(X2[0, 22], 1.0)  # was blue vanilla
        self.assertAlmostEqual(X2[0, 23], 0.0)  # was blue speed

    def test_maiden_control_flips(self):
        X = self._make_vector()
        y = np.array([1], dtype=np.int8)

        X2, _ = swap_teams(X, y)

        # Maiden control signs should flip
        self.assertAlmostEqual(X2[0, 40], -1.0)  # was +1
        self.assertAlmostEqual(X2[0, 41], 1.0)   # was -1
        self.assertAlmostEqual(X2[0, 42], 0.0)   # was 0 (stays 0)
        self.assertAlmostEqual(X2[0, 43], -1.0)  # was +1
        self.assertAlmostEqual(X2[0, 44], 1.0)   # was -1

    def test_map_unchanged(self):
        X = self._make_vector()
        y = np.array([1], dtype=np.int8)

        X2, _ = swap_teams(X, y)

        np.testing.assert_array_equal(X2[0, 45:49], X[0, 45:49])

    def test_snail_negated(self):
        X = self._make_vector()
        y = np.array([1], dtype=np.int8)

        X2, _ = swap_teams(X, y)

        self.assertAlmostEqual(X2[0, 49], -0.3)
        self.assertAlmostEqual(X2[0, 50], -0.1)

    def test_berries_unchanged(self):
        X = self._make_vector()
        y = np.array([1], dtype=np.int8)

        X2, _ = swap_teams(X, y)

        self.assertAlmostEqual(X2[0, 51], 0.5)

    def test_label_flips(self):
        X = self._make_vector()

        _, y2_blue = swap_teams(X, np.array([1], dtype=np.int8))
        _, y2_gold = swap_teams(X, np.array([0], dtype=np.int8))

        self.assertEqual(y2_blue[0], 0)  # Blue win -> Gold win
        self.assertEqual(y2_gold[0], 1)  # Gold win -> Blue win


class TestCrossVerification(unittest.TestCase):
    """Cross-verify: feature swap should match event-stream swap + materialize.

    For each game:
    1. Materialize normally -> swap feature vectors
    2. Swap event stream -> materialize
    3. Assert results match
    """

    def _load_games_from_benchmark(self, max_games=50):
        """Load raw events grouped by game from benchmark files."""
        test_dir = os.path.join(os.path.dirname(__file__), 'tests')
        benchmark_pattern = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        import glob
        games = {}
        game_order = []

        for filename in sorted(glob.glob(benchmark_pattern)):
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
                    games[game_id].append(
                        (_parse_ts(row[COL_TS]), event_type, row[COL_VALUES])
                    )
                    if len(game_order) > max_games:
                        break
            if len(game_order) > max_games:
                break

        return games, game_order[:max_games]

    def test_feature_swap_matches_event_swap(self):
        """For benchmark games, verify feature swap == event swap + materialize."""
        games, game_order = self._load_games_from_benchmark(max_games=50)

        if not game_order:
            self.skipTest('No benchmark games available')

        import random
        mismatches = 0
        total_games_tested = 0

        for game_id in game_order:
            raw_events = games[game_id]

            # Approach 1: Materialize normally, then swap features
            buf1 = np.empty((len(raw_events), NUM_FEATURES), dtype=np.float32)
            lbl1 = np.empty(len(raw_events), dtype=np.int8)
            rng1 = random.Random(0)
            try:
                n1 = _process_game(list(raw_events), buf1, lbl1, 0, 0.0, rng1)
            except Exception:
                continue

            if n1 == 0:
                continue

            orig_X = buf1[:n1].copy()
            orig_y = lbl1[:n1].copy()
            swapped_X, swapped_y = swap_teams(orig_X, orig_y)

            # Approach 2: Swap event stream, then materialize
            swapped_events = swap_event_stream(list(raw_events))
            buf2 = np.empty((len(swapped_events), NUM_FEATURES), dtype=np.float32)
            lbl2 = np.empty(len(swapped_events), dtype=np.int8)
            rng2 = random.Random(0)
            try:
                n2 = _process_game(swapped_events, buf2, lbl2, 0, 0.0, rng2)
            except Exception:
                continue

            if n2 == 0:
                continue

            event_X = buf2[:n2]
            event_y = lbl2[:n2]

            total_games_tested += 1

            # Verify they produce the same number of rows
            self.assertEqual(n1, n2,
                f'Game {game_id}: row count mismatch: '
                f'feature-swap={n1}, event-swap={n2}')

            # Verify labels match
            np.testing.assert_array_equal(
                swapped_y, event_y,
                err_msg=f'Game {game_id}: label mismatch')

            # Verify feature vectors match (within float32 precision)
            try:
                np.testing.assert_array_almost_equal(
                    swapped_X, event_X, decimal=4,
                    err_msg=f'Game {game_id}: feature mismatch')
            except AssertionError:
                mismatches += 1
                # Print diagnostic for first mismatch
                if mismatches <= 3:
                    diff = np.abs(swapped_X - event_X)
                    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
                    print(f'\nGame {game_id} mismatch at {max_diff_idx}: '
                          f'feature-swap={swapped_X[max_diff_idx]}, '
                          f'event-swap={event_X[max_diff_idx]}, '
                          f'diff={diff[max_diff_idx]}')
                raise

        self.assertGreater(total_games_tested, 0,
                          'No games were successfully tested')
        print(f'\nCross-verified {total_games_tested} games successfully')


class TestSwapPermConstants(unittest.TestCase):
    """Verify the swap constants are well-formed."""

    def test_perm_is_valid_permutation(self):
        self.assertEqual(sorted(SWAP_PERM), list(range(52)))

    def test_perm_is_involution(self):
        """Double-applying the permutation should be identity."""
        double = [SWAP_PERM[SWAP_PERM[i]] for i in range(52)]
        self.assertEqual(double, list(range(52)))

    def test_sign_is_involution(self):
        """SWAP_SIGN * SWAP_SIGN == 1 everywhere."""
        np.testing.assert_array_equal(SWAP_SIGN * SWAP_SIGN,
                                      np.ones(52, dtype=np.float32))

    def test_sign_values(self):
        """Only features 40-44 and 49-50 should be -1."""
        for i in range(52):
            if i in range(40, 45) or i in range(49, 51):
                self.assertEqual(SWAP_SIGN[i], -1.0, f'Feature {i} should be -1')
            else:
                self.assertEqual(SWAP_SIGN[i], 1.0, f'Feature {i} should be 1')


if __name__ == '__main__':
    unittest.main()
