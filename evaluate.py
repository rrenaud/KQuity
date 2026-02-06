"""Shared evaluation metrics for KQuity game prediction models."""

import numpy as np
import sklearn.metrics


def evaluate_model(predict_fn, test_X, test_y, name: str):
    """Evaluate a binary prediction model and return metrics.

    Args:
        predict_fn: Callable taking (N, 52) float32 array, returning (N,)
                    array of blue-team win probabilities.
        test_X: (N, 52) float32 feature matrix.
        test_y: (N,) labels (0 or 1).
        name: Display name for printing.

    Returns:
        Dict with keys: log_loss, accuracy, inversions.
    """
    predictions = predict_fn(test_X)

    log_loss = sklearn.metrics.log_loss(test_y, predictions)
    accuracy = sklearn.metrics.accuracy_score(test_y, predictions > 0.5)

    # Egg inversion test: incrementing egg count (feature 0) should
    # increase predicted blue win probability, not decrease it.
    mask = test_X[:, 0] != 2
    eligible_X = test_X[mask]
    sample_size = min(2000, len(eligible_X))
    if sample_size > 0:
        indices = np.random.choice(len(eligible_X), sample_size, replace=False)
        sample_X = eligible_X[indices]
        orig_preds = predict_fn(sample_X)
        modified_X = sample_X.copy()
        modified_X[:, 0] += 1
        mod_preds = predict_fn(modified_X)
        inversions = (mod_preds < orig_preds).mean()
    else:
        inversions = 0.0

    print(f"\n{name} Results:")
    print(f"  Log Loss: {log_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({100*accuracy:.1f}%)")
    print(f"  Egg Inversions: {inversions:.4f} ({100*inversions:.2f}%)")

    return {
        'log_loss': log_loss,
        'accuracy': accuracy,
        'inversions': inversions
    }
