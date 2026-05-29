"""High-level pipeline: build the modelling dataset and split it in time."""

from __future__ import annotations

import os

import pandas as pd

from . import data, features
from .models import GBMModel, SeasonalNaive, TorchMLP


def build_training_data(raw_dir: str):
    """Return (X, y, weather) for the training period from the raw files."""
    cons = data.load_consumption(os.path.join(raw_dir, "train.csv"))
    weather = data.build_region_weather(os.path.join(raw_dir, "meteo.parquet"))
    aligned = data.align_weather(cons.index, weather)
    X = features.build_feature_matrix(cons.index, aligned)
    y = cons[data.TARGETS]
    return X, y, weather


def build_test_data(raw_dir: str, weather: pd.DataFrame):
    """Return (X_test, test_index) for the 2022 horizon."""
    test_index = data.load_test_index(os.path.join(raw_dir, "test.csv"))
    aligned = data.align_weather(test_index, weather)
    X_test = features.build_feature_matrix(test_index, aligned)
    return X_test, test_index


def temporal_split(X: pd.DataFrame, y: pd.DataFrame, val_year: int = 2021):
    """Split chronologically: train on years < val_year, validate on val_year."""
    year = X.index.tz_convert("Europe/Paris").year
    tr, va = year < val_year, year == val_year
    return X[tr], y[tr], X[va], y[va]


MODELS = {
    "baseline": SeasonalNaive,
    "mlp": TorchMLP,
    "gbm": GBMModel,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by name (baseline | mlp | gbm)."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Choose from {list(MODELS)}.")
    return MODELS[name](**kwargs)
