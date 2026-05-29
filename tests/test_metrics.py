"""Unit tests for the evaluation metrics (no dataset required)."""

import numpy as np

from edf import metrics


def test_per_column_rmse_simple():
    y_true = np.array([[0.0, 10.0], [0.0, 10.0]])
    y_pred = np.array([[1.0, 13.0], [-1.0, 7.0]])
    rmse = metrics.per_column_rmse(y_true, y_pred)
    np.testing.assert_allclose(rmse, [1.0, 3.0])


def test_challenge_score_is_sum_of_rmse():
    y_true = np.zeros((4, 3))
    y_pred = np.ones((4, 3))
    assert metrics.challenge_score(y_true, y_pred) == 3.0  # 1 + 1 + 1


def test_metrics_ignore_nan_in_truth():
    y_true = np.array([[0.0, np.nan], [0.0, 10.0]])
    y_pred = np.array([[2.0, 5.0], [-2.0, 10.0]])
    # column 0 RMSE = 2; column 1 ignores the NaN row -> error 0
    np.testing.assert_allclose(metrics.per_column_rmse(y_true, y_pred), [2.0, 0.0])
