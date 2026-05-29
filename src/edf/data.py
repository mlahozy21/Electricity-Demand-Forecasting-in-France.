"""Data loading and weather/consumption alignment.

The challenge provides:
  - train.csv : half-hourly electricity demand for France, 12 regions and
                12 métropoles (25 target series), 2017-2021.
  - test.csv  : the timestamps to predict (year 2022), no targets.
  - meteo.parquet : 3-hourly observations from 40 weather stations across the
                12 French regions, 2017-2022 (Météo-France SYNOP).

This module loads them, maps every target to its weather region, aggregates
stations to a per-region weather signal and resamples it to the half-hourly
grid of the consumption series.
"""

from __future__ import annotations

import pandas as pd

# The 25 target series (order preserved for the submission).
REGIONS = [
    "Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne",
    "Centre-Val de Loire", "Grand Est", "Hauts-de-France", "Normandie",
    "Nouvelle-Aquitaine", "Occitanie", "Pays de la Loire",
    "Provence-Alpes-Côte d'Azur", "Île-de-France",
]

# Map each target column to the weather region used as its predictor.
TARGET_TO_REGION = {
    "France": "France",  # national average of all regions
    **{r: r for r in REGIONS},
    "Montpellier Méditerranée Métropole": "Occitanie",
    "Métropole Européenne de Lille": "Hauts-de-France",
    "Métropole Grenoble-Alpes-Métropole": "Auvergne-Rhône-Alpes",
    "Métropole Nice Côte d'Azur": "Provence-Alpes-Côte d'Azur",
    "Métropole Rennes Métropole": "Bretagne",
    "Métropole Rouen Normandie": "Normandie",
    "Métropole d'Aix-Marseille-Provence": "Provence-Alpes-Côte d'Azur",
    "Métropole de Lyon": "Auvergne-Rhône-Alpes",
    "Métropole du Grand Nancy": "Grand Est",
    "Métropole du Grand Paris": "Île-de-France",
    "Nantes Métropole": "Pays de la Loire",
    "Toulouse Métropole": "Occitanie",
}

TARGETS = list(TARGET_TO_REGION.keys())

# Weather variables kept from the SYNOP feed (well populated, physically useful).
#   t   : air temperature (Kelvin -> converted to Celsius)
#   u   : relative humidity (%)
#   ff  : wind speed (m/s)
#   pmer: sea-level pressure (Pa)
WEATHER_VARS = ["t", "u", "ff", "pmer"]


def _read_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the `date` column to UTC and sort chronologically."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.sort_values("date").reset_index(drop=True)


def load_consumption(path: str) -> pd.DataFrame:
    """Load the consumption table indexed by UTC timestamp (targets as columns)."""
    df = _read_dates(pd.read_csv(path))
    return df.set_index("date")


def load_test_index(path: str) -> pd.DatetimeIndex:
    """Load the timestamps to predict (test.csv)."""
    df = _read_dates(pd.read_csv(path))
    return pd.DatetimeIndex(df["date"])


def build_region_weather(meteo_path: str) -> pd.DataFrame:
    """Aggregate stations to a per-region half-hourly weather frame.

    Steps: keep useful columns, convert temperature to Celsius, average the
    stations within each region, resample each region to 30 min with time
    interpolation, and add a national ("France") average.
    Returns a frame indexed by UTC timestamp with columns ``<var>_<region>``.
    """
    cols = ["numer_sta", "date", "nom_reg"] + WEATHER_VARS
    meteo = pd.read_parquet(meteo_path, columns=cols)
    meteo = _read_dates(meteo)
    meteo = meteo.dropna(subset=["nom_reg"])
    meteo["t"] = meteo["t"] - 273.15  # Kelvin -> Celsius

    # Average stations within a region at each timestamp.
    region_mean = (
        meteo.groupby(["nom_reg", "date"])[WEATHER_VARS].mean().reset_index()
    )

    frames = []
    for region, grp in region_mean.groupby("nom_reg"):
        s = grp.set_index("date")[WEATHER_VARS].sort_index()
        # 3-hourly -> 30-min via time interpolation, then forward/back fill edges.
        s = s.resample("30min").interpolate(method="time").ffill().bfill()
        s.columns = [f"{v}_{region}" for v in WEATHER_VARS]
        frames.append(s)

    weather = pd.concat(frames, axis=1)

    # National signal: mean across regions for each variable.
    for v in WEATHER_VARS:
        region_cols = [f"{v}_{r}" for r in REGIONS]
        weather[f"{v}_France"] = weather[region_cols].mean(axis=1)

    return weather


def align_weather(index: pd.DatetimeIndex, weather: pd.DataFrame) -> pd.DataFrame:
    """Reindex the weather frame onto the given timestamps (interpolate gaps)."""
    aligned = weather.reindex(weather.index.union(index)).interpolate(
        method="time"
    ).reindex(index)
    return aligned.ffill().bfill()
