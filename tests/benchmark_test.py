import os
import time
import unittest

import numpy as np

from preprocess import iterate_events_from_csv, iterate_game_events_with_state, create_game_states_matrix
from map_structure import MapStructureInfos
from fast_materialize import fast_materialize


class PipelineRegressionTest(unittest.TestCase):
    """Regression test for the event processing pipeline.

    This test processes ~3000 games across 16 sharded files and verifies
    the output matches expected results. Processing takes approximately 10 seconds.
    """

    def test_events_to_features_regression(self) -> None:
        """Verify event processing produces expected features and labels."""
        test_dir = os.path.dirname(__file__)
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')
        expected_path = os.path.join(test_dir, 'benchmark_expected.npz')

        # Load expected output
        expected = np.load(expected_path)
        expected_states = expected['states']
        expected_labels = expected['labels']

        # Process through pipeline
        map_infos = MapStructureInfos()
        start_time = time.time()
        events = iterate_events_from_csv(benchmark_path)
        game_states = iterate_game_events_with_state(events, map_infos)
        states, labels, _ = create_game_states_matrix(game_states, drop_state_probability=0.0)
        elapsed = time.time() - start_time

        print(f'\nProcessing time: {elapsed:.2f} seconds')
        print(f'States shape: {states.shape}, Labels shape: {labels.shape}')

        # Verify shapes match
        self.assertEqual(states.shape, expected_states.shape,
                        f'States shape mismatch: {states.shape} vs {expected_states.shape}')
        self.assertEqual(labels.shape, expected_labels.shape,
                        f'Labels shape mismatch: {labels.shape} vs {expected_labels.shape}')

        # Verify values match
        np.testing.assert_array_almost_equal(
            states, expected_states,
            decimal=10,
            err_msg='States matrix does not match expected output')
        np.testing.assert_array_equal(
            labels, expected_labels,
            err_msg='Labels do not match expected output')


    def test_fast_path_matches_expected(self) -> None:
        """Verify fast path produces identical output to expected benchmark."""
        test_dir = os.path.dirname(__file__)
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')
        expected_path = os.path.join(test_dir, 'benchmark_expected.npz')

        expected = np.load(expected_path)
        expected_states = expected['states']
        expected_labels = expected['labels']

        start_time = time.time()
        states, labels, _, _ = fast_materialize(benchmark_path)
        elapsed = time.time() - start_time

        print(f'\nFast path time: {elapsed:.2f} seconds')
        print(f'States shape: {states.shape}, Labels shape: {labels.shape}')

        self.assertEqual(states.shape, expected_states.shape,
                        f'States shape mismatch: {states.shape} vs {expected_states.shape}')
        self.assertEqual(labels.shape, expected_labels.shape,
                        f'Labels shape mismatch: {labels.shape} vs {expected_labels.shape}')

        np.testing.assert_array_almost_equal(
            states, expected_states,
            decimal=5,
            err_msg='Fast path states do not match expected output')
        np.testing.assert_array_equal(
            labels, expected_labels,
            err_msg='Fast path labels do not match expected output')


    def test_fast_path_with_ratings_matches_slow_path(self) -> None:
        """Verify fast and slow paths produce identical 62-feature output with ratings."""
        test_dir = os.path.dirname(__file__)
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')

        # First, run fast path without ratings to get game_ids
        _, _, game_ids, _ = fast_materialize(benchmark_path)
        unique_game_ids = set(game_ids)

        # Generate deterministic dummy ratings for each game
        ratings_by_game = {}
        for gid in unique_game_ids:
            # Use game_id as seed for reproducible ratings
            rng = np.random.RandomState(int(gid) % (2**31))
            ratings_by_game[gid] = rng.uniform(15.0, 35.0, size=10).astype(np.float32)

        # Fast path with ratings
        start_time = time.time()
        fast_states, fast_labels, fast_game_ids, _ = fast_materialize(
            benchmark_path, ratings_by_game=ratings_by_game)
        fast_elapsed = time.time() - start_time

        # Slow path with ratings
        map_infos = MapStructureInfos()
        events = iterate_events_from_csv(benchmark_path)
        game_states = iterate_game_events_with_state(events, map_infos)
        start_time = time.time()
        slow_states, slow_labels, slow_game_ids = create_game_states_matrix(
            game_states, drop_state_probability=0.0, ratings_by_game=ratings_by_game)
        slow_elapsed = time.time() - start_time

        print(f'\nRatings cross-verification:')
        print(f'  Fast path: {fast_elapsed:.2f}s, shape={fast_states.shape}')
        print(f'  Slow path: {slow_elapsed:.2f}s, shape={slow_states.shape}')

        self.assertEqual(fast_states.shape[1], 62, 'Expected 62 features with ratings')
        self.assertEqual(fast_states.shape, slow_states.shape,
                        f'Shape mismatch: fast={fast_states.shape} vs slow={slow_states.shape}')
        self.assertEqual(fast_labels.shape, slow_labels.shape)

        np.testing.assert_array_almost_equal(
            fast_states, slow_states,
            decimal=5,
            err_msg='Fast/slow path states mismatch with ratings')
        np.testing.assert_array_equal(
            fast_labels, slow_labels,
            err_msg='Fast/slow path labels mismatch with ratings')


if __name__ == '__main__':
    unittest.main()
