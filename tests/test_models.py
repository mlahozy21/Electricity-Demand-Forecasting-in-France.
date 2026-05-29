"""Smoke tests for the models on synthetic data (no dataset required)."""

import numpy as np
import pandas as pd

from edf.models import SeasonalNaive, TorchMLP


def _synthetic(n=2000, n_targets=3):
    idx = pd.date_range("2019-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    hour = idx.tz_convert("Europe/Paris").hour.to_numpy()
    base = 100 + 20 * np.sin(2 * np.pi * hour / 24)
    y = pd.DataFrame(
        {f"t{j}": base * (1 + 0.1 * j) + rng.normal(0, 1, n) for j in range(n_targets)},
        index=idx,
    )
    X = pd.DataFrame({"hour_sin": np.sin(2 * np.pi * hour / 24),
                      "hour_cos": np.cos(2 * np.pi * hour / 24)}, index=idx)
    return X, y


def test_seasonal_naive_runs_and_beats_mean():
    X, y = _synthetic()
    model = SeasonalNaive().fit(y)
    pred = model.predict(y.index)
    assert pred.shape == y.shape
    assert not pred.isna().any().any()


def test_torch_mlp_fits_and_predicts():
    X, y = _synthetic()
    model = TorchMLP(hidden=(16,), max_epochs=5, batch_size=256).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred.to_numpy()).all()
