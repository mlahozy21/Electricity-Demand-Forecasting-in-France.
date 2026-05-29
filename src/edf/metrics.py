"""Evaluation metrics, including the challenge score (NaN-aware).

Some target series (the métropoles) only start partway through the training
period, so the ground truth contains missing values. All metrics ignore the
positions where ``y_true`` is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_array(a):
    return a.to_numpy() if isinstance(a, (pd.DataFrame, pd.Series)) else np.asarray(a)


def per_column_rmse(y_true, y_pred) -> np.ndarray:
    """RMSE per target column, ignoring NaN positions in ``y_true``."""
    yt = _to_array(y_true).astype(float)
    yp = _to_array(y_pred).astype(float)
    mask = ~np.isnan(yt)
    se = np.where(mask, (yt - yp) ** 2, 0.0)
    counts = np.maximum(mask.sum(axis=0), 1)
    return np.sqrt(se.sum(axis=0) / counts)


def challenge_score(y_true, y_pred) -> float:
    """Codabench metric: sum of the per-column RMSEs across the 25 series."""
    return float(per_column_rmse(y_true, y_pred).sum())


def mae(y_true, y_pred) -> float:
    yt = _to_array(y_true).astype(float)
    yp = _to_array(y_pred).astype(float)
    mask = ~np.isnan(yt)
    return float(np.abs(np.where(mask, yt - yp, 0.0)).sum() / np.maximum(mask.sum(), 1))
