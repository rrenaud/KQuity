"""Tests for eval_metrics.py."""

import math
import os
import unittest

import numpy as np

from eval_metrics import (
    compute_standard_metrics,
    compute_calibration_metrics,
    compute_temporal_metrics,
    compute_kq_metrics,
)


class ConstantModel:
    """Mock model that always predicts the same probability."""
    def __init__(self, p):
        self.p = p

    def predict(self, X):
        return np.full(len(X), self.p)


class TestStandardMetrics(unittest.TestCase):

    def test_perfect_predictions(self):
        labels = np.array([1, 1, 0, 0], dtype=np.float64)
        # Near-perfect predictions (not exactly 0/1 to avoid log(0))
        predictions = np.array([0.999, 0.999, 0.001, 0.001])
        m = compute_standard_metrics(predictions, labels)
        self.assertAlmostEqual(m['accuracy'], 1.0)
        self.assertAlmostEqual(m['auc_roc'], 1.0)
        self.assertLess(m['log_loss'], 0.01)
        self.assertLess(m['brier_score'], 0.001)

    def test_uniform_predictions(self):
        labels = np.array([1, 0, 1, 0], dtype=np.float64)
        predictions = np.array([0.5, 0.5, 0.5, 0.5])
        m = compute_standard_metrics(predictions, labels)
        self.assertAlmostEqual(m['log_loss'], math.log(2), places=5)
        self.assertAlmostEqual(m['brier_score'], 0.25)
        # AUC is 0.5 for random predictions
        self.assertAlmostEqual(m['auc_roc'], 0.5)

    def test_single_class_labels_auc_nan(self):
        labels = np.array([1, 1, 1, 1], dtype=np.float64)
        predictions = np.array([0.6, 0.7, 0.8, 0.9])
        m = compute_standard_metrics(predictions, labels)
        self.assertTrue(math.isnan(m['auc_roc']))
        # Other metrics should still be valid
        self.assertAlmostEqual(m['accuracy'], 1.0)


class TestCalibrationMetrics(unittest.TestCase):

    def test_perfectly_calibrated(self):
        # Predictions exactly match empirical frequency in each bin
        rng = np.random.RandomState(42)
        n = 10000
        predictions = rng.uniform(0, 1, n).astype(np.float64)
        # Labels drawn from Bernoulli(p) for each prediction p
        labels = (rng.uniform(0, 1, n) < predictions).astype(np.float64)
        m = compute_calibration_metrics(predictions, labels, n_bins=10)
        # ECE should be small for a large sample
        self.assertLess(m['ece'], 0.05)

    def test_systematically_off(self):
        # Always predict 0.9 but true rate is 0.5
        predictions = np.full(1000, 0.9)
        labels = np.array([1, 0] * 500, dtype=np.float64)
        m = compute_calibration_metrics(predictions, labels, n_bins=10)
        self.assertGreater(m['ece'], 0.3)

    def test_bin_counts_sum_to_n(self):
        rng = np.random.RandomState(123)
        n = 500
        predictions = rng.uniform(0, 1, n)
        labels = rng.randint(0, 2, n).astype(np.float64)
        m = compute_calibration_metrics(predictions, labels, n_bins=5)
        self.assertEqual(sum(m['bin_counts']), n)


class TestTemporalMetrics(unittest.TestCase):

    def test_single_bin_matches_scalar(self):
        predictions = np.array([0.6, 0.7, 0.8, 0.3])
        labels = np.array([1, 1, 0, 0], dtype=np.float64)
        timestamps = np.array([10, 20, 30, 40], dtype=np.float32)

        # Single bin covering everything
        m = compute_temporal_metrics(predictions, labels, timestamps,
                                     time_bins=[float('inf')])
        scalar = compute_standard_metrics(predictions, labels)
        self.assertAlmostEqual(m['bin_brier_score'][0], scalar['brier_score'], places=5)
        self.assertAlmostEqual(m['bin_accuracy'][0], scalar['accuracy'], places=5)

    def test_bin_counts_sum_to_n(self):
        rng = np.random.RandomState(42)
        n = 300
        predictions = rng.uniform(0, 1, n)
        labels = rng.randint(0, 2, n).astype(np.float64)
        timestamps = rng.uniform(5, 300, n).astype(np.float32)
        m = compute_temporal_metrics(predictions, labels, timestamps)
        self.assertEqual(sum(m['bin_counts']), n)

    def test_empty_bin_gives_nan(self):
        predictions = np.array([0.5, 0.5])
        labels = np.array([1, 0], dtype=np.float64)
        timestamps = np.array([100.0, 200.0], dtype=np.float32)
        # First bin [0, 10) should be empty
        m = compute_temporal_metrics(predictions, labels, timestamps,
                                     time_bins=[10, float('inf')])
        self.assertTrue(math.isnan(m['bin_log_loss'][0]))
        self.assertEqual(m['bin_counts'][0], 0)
        self.assertEqual(m['bin_counts'][1], 2)


class TestKQMetrics(unittest.TestCase):

    def _make_states(self, n, eggs_val=1):
        """Create dummy 52-feature states with blue eggs at column 0."""
        rng = np.random.RandomState(42)
        states = rng.uniform(-1, 1, (n, 52)).astype(np.float32)
        states[:, 0] = eggs_val  # blue eggs
        return states

    def test_constant_model_zero_inversions(self):
        model = ConstantModel(0.5)
        states = self._make_states(200)
        labels = np.zeros(200, dtype=np.float64)
        m = compute_kq_metrics(model, states, labels, sample_size=100)
        self.assertEqual(m['egg_inversion_rate'], 0.0)

    def test_constant_half_symmetry_zero(self):
        # A constant 0.5 model has P(X) + P(swap(X)) = 0.5 + 0.5 = 1.0 exactly
        model = ConstantModel(0.5)
        states = self._make_states(100)
        labels = np.zeros(100, dtype=np.float64)
        m = compute_kq_metrics(model, states, labels)
        self.assertAlmostEqual(m['symmetry_deviation'], 0.0, places=5)

    def test_constant_biased_high_symmetry_deviation(self):
        # A constant 0.9 model has P(X)+P(swap(X)) = 0.9+0.9 = 1.8, deviation = 0.8
        model = ConstantModel(0.9)
        states = self._make_states(100)
        labels = np.zeros(100, dtype=np.float64)
        m = compute_kq_metrics(model, states, labels)
        self.assertAlmostEqual(m['symmetry_deviation'], 0.8, places=5)

    def test_maxed_eggs_excluded(self):
        model = ConstantModel(0.5)
        # All states have eggs=2, should all be excluded
        states = self._make_states(50, eggs_val=2)
        labels = np.zeros(50, dtype=np.float64)
        m = compute_kq_metrics(model, states, labels)
        self.assertEqual(m['egg_inversion_n'], 0)
        self.assertEqual(m['egg_inversion_rate'], 0.0)


class TestTimestampEmission(unittest.TestCase):

    def test_timestamps_returned(self):
        test_dir = os.path.join(os.path.dirname(__file__), 'tests')
        benchmark_path = os.path.join(test_dir, 'benchmark_events_*.csv.gz')
        if not os.path.exists(test_dir):
            self.skipTest('Benchmark data not available')

        from fast_materialize import fast_materialize
        states, labels, game_ids, timestamps = fast_materialize(benchmark_path)
        self.assertEqual(timestamps.shape, labels.shape)
        # All timestamps should be > 5.0 (the filter in _process_game)
        self.assertTrue((timestamps > 5.0).all(),
                        f'Min timestamp: {timestamps.min()}')


if __name__ == '__main__':
    unittest.main()
