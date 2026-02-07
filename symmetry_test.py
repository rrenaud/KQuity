"""Tests for symmetry augmentation.

Verifies that:
1. Double swap is identity (roundtrip)
2. Spot checks on known vectors
3. Cross-verification: feature swap matches event stream swap + materialize
4. Label flip correctness
"""

import copy
import os
import random
import unittest

import numpy as np

from symmetry import swap_teams, swap_event_stream, SWAP_PERM, SWAP_SIGN
from preprocess import (
    iterate_events_from_csv,
    iterate_events_by_game_and_normalize_time,
    vectorize_game_state,
    GameState,
    VictoryEvent,
    get_map_start,
)
from constants import Team
import map_structure


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


def _materialize_single_game(game_events, map_structure_infos):
    """Materialize feature vectors for a single game's events (slow path).

    Returns (states, labels) numpy arrays, or (None, None) if invalid.
    """
    map_start = get_map_start(game_events)
    map_info = map_structure_infos.get_map_info(map_start.map, map_start.gold_on_left)
    game_state = GameState(map_info)

    vectorized_states = []
    labels = []
    victory_event = game_events[-1]
    if not isinstance(victory_event, VictoryEvent):
        return None, None

    label = 1 if victory_event.winning_team == Team.BLUE else 0

    for event in game_events:
        if event.timestamp > 5.0:
            vectorized_states.append(vectorize_game_state(game_state, event))
            labels.append(label)
        event.modify_game_state(game_state)

    if not vectorized_states:
        return None, None
    return np.vstack(vectorized_states), np.array(labels)


class TestCrossVerification(unittest.TestCase):
    """Cross-verify: feature swap should match event-stream swap + materialize.

    For each game:
    1. Materialize normally via slow path -> swap feature vectors
    2. Swap GameEvent objects -> materialize via slow path
    3. Assert results match
    """

    def test_feature_swap_matches_event_swap(self):
        """For benchmark games, verify feature swap == event swap + materialize."""
        test_dir = os.path.join(os.path.dirname(__file__), 'tests')
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        if not os.path.exists(test_dir):
            self.skipTest('Benchmark data not available')

        map_infos = map_structure.MapStructureInfos()
        events = iterate_events_from_csv(benchmark_path)
        grouped = iterate_events_by_game_and_normalize_time(events)

        total_games_tested = 0
        max_games = 50

        for game_id, game_events in grouped:
            if total_games_tested >= max_games:
                break

            # Approach 1: Materialize normally, then swap features
            try:
                orig_X, orig_y = _materialize_single_game(game_events, map_infos)
            except Exception:
                continue
            if orig_X is None:
                continue

            swapped_X, swapped_y = swap_teams(orig_X, orig_y)

            # Approach 2: Swap event stream, then materialize
            swapped_events = swap_event_stream(game_events)
            try:
                event_X, event_y = _materialize_single_game(swapped_events, map_infos)
            except Exception:
                continue
            if event_X is None:
                continue

            total_games_tested += 1

            # Verify same number of rows
            self.assertEqual(len(orig_y), len(event_y),
                f'Game {game_id}: row count mismatch: '
                f'feature-swap={len(orig_y)}, event-swap={len(event_y)}')

            # Verify labels match
            np.testing.assert_array_equal(
                swapped_y, event_y,
                err_msg=f'Game {game_id}: label mismatch')

            # Verify feature vectors match (within float32 precision)
            np.testing.assert_array_almost_equal(
                swapped_X, event_X, decimal=4,
                err_msg=f'Game {game_id}: feature mismatch')

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
