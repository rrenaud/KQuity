"""Model evaluation metrics library for KQuity.

Computes standard, calibration, temporal, and KQ-specific domain metrics
from a model and game data (or raw predictions/labels).
"""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import sklearn.metrics

from fast_materialize import fast_materialize
from symmetry import swap_teams


class Predictor(Protocol):
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]: ...


# --- Standard metrics ---

def compute_standard_metrics(predictions: npt.NDArray[np.float64],
                             labels: npt.NDArray[np.float64]) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        predictions: array of predicted P(blue_wins), shape (N,)
        labels: array of 0/1 labels, shape (N,)

    Returns:
        dict with keys: log_loss, brier_score, auc_roc, accuracy
    """
    try:
        log_loss = float(sklearn.metrics.log_loss(labels, predictions))
    except ValueError:
        log_loss = float('nan')
    brier_score = float(np.mean((predictions - labels) ** 2))
    accuracy = float(sklearn.metrics.accuracy_score(labels, predictions > 0.5))
    try:
        auc_roc = float(sklearn.metrics.roc_auc_score(labels, predictions))
    except ValueError:
        auc_roc = float('nan')
    return {
        'log_loss': log_loss,
        'brier_score': brier_score,
        'auc_roc': auc_roc,
        'accuracy': accuracy,
    }


# --- Calibration metrics ---

def compute_calibration_metrics(predictions: npt.NDArray[np.float64],
                                labels: npt.NDArray[np.float64],
                                n_bins: int = 10) -> dict[str, Any]:
    """Compute expected calibration error (ECE) with equal-width bins.

    Args:
        predictions: array of predicted probabilities, shape (N,)
        labels: array of 0/1 labels, shape (N,)
        n_bins: number of equal-width bins on [0, 1]

    Returns:
        dict with keys: ece, bin_edges, bin_mean_predicted, bin_mean_observed, bin_counts
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mean_predicted = np.full(n_bins, np.nan)
    bin_mean_observed = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        else:
            # Last bin includes right edge
            mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_mean_predicted[i] = float(predictions[mask].mean())
            bin_mean_observed[i] = float(labels[mask].mean())

    # ECE: weighted average of |mean_pred - mean_observed| per bin
    total = bin_counts.sum()
    if total > 0:
        non_empty = bin_counts > 0
        ece = float(np.sum(
            bin_counts[non_empty] *
            np.abs(bin_mean_predicted[non_empty] - bin_mean_observed[non_empty])
        ) / total)
    else:
        ece = float('nan')

    return {
        'ece': ece,
        'bin_edges': bin_edges.tolist(),
        'bin_mean_predicted': bin_mean_predicted.tolist(),
        'bin_mean_observed': bin_mean_observed.tolist(),
        'bin_counts': bin_counts.tolist(),
    }


# --- Temporal metrics ---

DEFAULT_TIME_BINS: list[float] = [5, 30, 60, 90, 120, 180, 300, float('inf')]


def compute_temporal_metrics(predictions: npt.NDArray[np.float64],
                             labels: npt.NDArray[np.float64],
                             timestamps: npt.NDArray[np.float32],
                             time_bins: list[float] | None = None) -> dict[str, Any]:
    """Compute metrics binned by game time.

    Args:
        predictions: array of predicted probabilities, shape (N,)
        labels: array of 0/1 labels, shape (N,)
        timestamps: array of seconds-since-gamestart, shape (N,)
        time_bins: list of bin upper edges (default: DEFAULT_TIME_BINS).
            Each bin covers (previous_edge, edge]. First bin starts at 0.

    Returns:
        dict with keys: time_bins, bin_log_loss, bin_accuracy, bin_brier_score, bin_counts
    """
    if time_bins is None:
        time_bins = DEFAULT_TIME_BINS

    n_bins = len(time_bins)
    bin_log_loss = np.full(n_bins, np.nan)
    bin_accuracy = np.full(n_bins, np.nan)
    bin_brier_score = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    lower = 0.0
    for i, upper in enumerate(time_bins):
        if upper == float('inf'):
            mask = timestamps >= lower
        else:
            mask = (timestamps >= lower) & (timestamps < upper)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            p = predictions[mask]
            l = labels[mask]
            bin_brier_score[i] = float(np.mean((p - l) ** 2))
            bin_accuracy[i] = float((( p > 0.5) == l).mean())
            try:
                bin_log_loss[i] = float(sklearn.metrics.log_loss(l, p))
            except ValueError:
                bin_log_loss[i] = float('nan')
        lower = upper

    return {
        'time_bins': time_bins,
        'bin_log_loss': bin_log_loss.tolist(),
        'bin_accuracy': bin_accuracy.tolist(),
        'bin_brier_score': bin_brier_score.tolist(),
        'bin_counts': bin_counts.tolist(),
    }


# --- KQ-specific metrics ---

def compute_kq_metrics(model: Predictor,
                       states: npt.NDArray[np.float32],
                       labels: npt.NDArray[np.float64],
                       *, sample_size: int = 5000) -> dict[str, float | int]:
    """Compute KQ domain-specific metrics: egg inversions and symmetry deviation.

    Args:
        model: object with .predict(X) -> array of probabilities
        states: feature matrix, shape (N, 52) or (N, 62)
        labels: array of 0/1 labels, shape (N,)
        sample_size: max samples for egg inversion test

    Returns:
        dict with keys: egg_inversion_rate, egg_inversion_n, symmetry_deviation
    """
    # Egg inversions: adding a blue egg should increase P(blue_wins).
    # Feature index 0 is always blue eggs in both 52 and 62 layouts.
    mask = states[:, 0] != 2  # exclude states where eggs already maxed
    eligible = states[mask]
    n_sample = min(sample_size, len(eligible))
    if n_sample > 0:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(eligible), n_sample, replace=False)
        sample = eligible[indices]
        orig_preds = model.predict(sample)
        modified = sample.copy()
        modified[:, 0] += 1
        mod_preds = model.predict(modified)
        egg_inversion_rate = float((mod_preds < orig_preds).mean())
    else:
        egg_inversion_rate = 0.0

    # Symmetry deviation: |P(X) + P(swap(X)) - 1| should be 0 for perfect symmetry.
    swap_X, _ = swap_teams(states, labels)  # type: ignore[arg-type]
    p_orig = model.predict(states)
    p_swap = model.predict(swap_X)
    symmetry_deviation = float(np.mean(np.abs(p_orig + p_swap - 1.0)))

    return {
        'egg_inversion_rate': egg_inversion_rate,
        'egg_inversion_n': n_sample,
        'symmetry_deviation': symmetry_deviation,
    }


# --- Top-level orchestrator ---

def evaluate(model: Predictor, csv_pattern: str, *,
             ratings_by_game: dict[int, npt.NDArray[np.float32]] | None = None,
             drop_state_probability: float = 0.0, sample_size: int = 5000,
             n_calibration_bins: int = 10) -> dict[str, Any]:
    """Full evaluation: materialize data, predict, compute all metrics.

    Args:
        model: object with .predict(X) -> array of probabilities
        csv_pattern: glob pattern for CSV/gzip game event files
        ratings_by_game: optional dict {game_id: np.array(10)} for rated features
        drop_state_probability: fraction of states to drop (0.0 = keep all)
        sample_size: max samples for egg inversion test
        n_calibration_bins: number of calibration bins

    Returns:
        dict with keys: standard, calibration, temporal, kq, n_samples, n_games, num_features
    """
    states, labels, game_ids, timestamps = fast_materialize(
        csv_pattern, drop_state_probability, ratings_by_game)
    labels_float = labels.astype(np.float64)

    predictions = model.predict(states)

    return {
        'standard': compute_standard_metrics(predictions, labels_float),
        'calibration': compute_calibration_metrics(predictions, labels_float,
                                                   n_bins=n_calibration_bins),
        'temporal': compute_temporal_metrics(predictions, labels_float, timestamps),
        'kq': compute_kq_metrics(model, states, labels_float, sample_size=sample_size),
        'n_samples': len(labels),
        'n_games': len(np.unique(game_ids)),
        'num_features': states.shape[1],
    }


# --- Display utility ---

def print_metrics(metrics: dict[str, Any], name: str = '') -> None:
    """Print formatted summary of evaluation metrics."""
    header = f"Evaluation: {name}" if name else "Evaluation Results"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"Samples: {metrics['n_samples']:,}  |  Games: {metrics['n_games']:,}  |  Features: {metrics['num_features']}")

    std = metrics['standard']
    print(f"\n--- Standard Metrics ---")
    print(f"  Log Loss:    {std['log_loss']:.6f}")
    print(f"  Brier Score: {std['brier_score']:.6f}")
    print(f"  AUC-ROC:     {std['auc_roc']:.6f}")
    print(f"  Accuracy:    {std['accuracy']:.4f} ({100*std['accuracy']:.1f}%)")

    cal = metrics['calibration']
    print(f"\n--- Calibration ---")
    print(f"  ECE: {cal['ece']:.6f}")

    kq = metrics['kq']
    print(f"\n--- KQ Domain Metrics ---")
    print(f"  Egg Inversions: {kq['egg_inversion_rate']:.4f} ({100*kq['egg_inversion_rate']:.2f}%, n={kq['egg_inversion_n']})")
    print(f"  Symmetry Dev:   {kq['symmetry_deviation']:.6f}")

    temp = metrics['temporal']
    print(f"\n--- Temporal Breakdown ---")
    print(f"  {'Time Bin':>12}  {'Count':>7}  {'LogLoss':>8}  {'Accuracy':>8}  {'Brier':>8}")
    bins = temp['time_bins']
    lower = 0
    for i, upper in enumerate(bins):
        label = f"{lower}-{upper}" if upper != float('inf') else f"{lower}+"
        count = temp['bin_counts'][i]
        ll = temp['bin_log_loss'][i]
        acc = temp['bin_accuracy'][i]
        bs = temp['bin_brier_score'][i]
        ll_s = f"{ll:.4f}" if not np.isnan(ll) else "   nan"
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "   nan"
        bs_s = f"{bs:.4f}" if not np.isnan(bs) else "   nan"
        print(f"  {label:>12}  {count:>7}  {ll_s:>8}  {acc_s:>8}  {bs_s:>8}")
        lower = upper
    print()
