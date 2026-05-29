# Electricity Demand Forecasting in France

Forecasting half-hourly electricity demand for **France, its 12 regions and 12 métropoles** (25 series) for the year 2022, from weather and calendar features. Built around the [Codabench data challenge](https://www.codabench.org/) metric: the **sum of the per-series RMSE**.

This repository is the **deep-learning forecasting** part of the project (by Marcos Lahoz), rebuilt from the original challenge notebook into a clean, installable Python package. The companion statistical-learning study (joint work with Samuel Molano and Mohamed Amine Grini) is summarised in `docs/statistical_learning_report.pdf` (code: https://github.com/SamuelMolano/ModelisationPredictive).

## Problem

- **Inputs**: half-hourly regional demand for 2017–2021 (`train.csv`), the timestamps to predict for 2022 (`test.csv`), and 3-hourly weather observations from 40 Météo-France stations across the 12 regions (`meteo.parquet`).
- **Targets**: 25 demand series. The 12 métropole series only start partway through the training period, so the targets contain missing values that the pipeline handles throughout.
- **Metric**: `challenge_score = Σ_series RMSE(series)` — implemented (NaN-aware) in `edf.metrics.challenge_score`. This is exactly the loss the neural network is trained on.

## Approach

**Feature engineering** (`edf.features`, no target leakage — every feature is available for the 2022 horizon):
- Calendar features in French local time: cyclical encodings of time-of-day, day-of-week and day-of-year, weekend and French public-holiday flags, and a year trend.
- Weather features: per-region temperature, humidity, wind and pressure (stations averaged per region and resampled from 3-hourly to the 30-minute grid), heating/cooling degree days (HDD/CDD), a national aggregate, and 24 h/48 h rolling temperatures for thermal inertia.

**Validation**: a strictly **temporal** split (train on years `< val_year`, validate on `val_year`, default 2021) — no random shuffling, which would leak future information in a time series.

**Models** (`edf.models`, shared `fit`/`predict` interface, all multi-output and NaN-aware):
- `SeasonalNaive` — climatology baseline (mean by month × weekday × half-hour).
- `TorchMLP` — the project's neural network, rebuilt: a 2-hidden-layer MLP with batch-norm and dropout, Xavier init, mini-batch training, early stopping on the validation metric, and trained directly on the **sum-of-RMSE loss** (the challenge metric); the network predicts in standardised output space for stable optimisation while the loss is computed on the raw scale.
- `GBMModel` — one `HistGradientBoostingRegressor` per series (scikit-learn), fitted in parallel.

On the 2021 hold-out, the MLP and gradient-boosting models clearly outperform the seasonal baseline (baseline `challenge_score ≈ 11541`). Run `scripts/evaluate.py` to reproduce the exact figures and the per-series RMSE figure.

## Project structure

```
.
├── pyproject.toml  requirements.txt  README.md  LICENSE  .gitignore
├── src/edf/                 # installable library package
│   ├── data.py              # load train/test/meteo, station→region aggregation, alignment
│   ├── features.py          # calendar + weather feature engineering
│   ├── metrics.py           # challenge metric (sum of per-series RMSE), NaN-aware
│   ├── models.py            # SeasonalNaive, TorchMLP, GBMModel
│   ├── pipeline.py          # build dataset + temporal split + model registry
│   └── utils.py             # reproducibility (set_seed)
├── scripts/                 # entry points
│   ├── prepare_data.py      # build and cache feature matrices
│   ├── evaluate.py          # temporal-validation comparison of the models
│   └── make_submission.py   # train on full history, predict 2022, write submission
├── tests/                   # unit/smoke tests (no dataset required)
└── docs/
    ├── statistical_learning_report.pdf
    └── weather_data_description.pdf
```

## Installation

```bash
git clone https://github.com/mlahozy21/Electricity-Demand-Forecasting-in-France.git
cd Electricity-Demand-Forecasting-in-France

python -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -e .                   # installs the edf package and its dependencies
```

Adjust the `torch` install to your platform following https://pytorch.org/get-started/locally/.

## Data

The challenge data is **not** versioned. Place the files under `data/raw/`:

```
data/raw/train.csv        # 2017–2021 half-hourly demand (25 series)
data/raw/test.csv         # 2022 timestamps to predict
data/raw/meteo.parquet    # Météo-France SYNOP observations
```

## Usage

Run from the repository root.

```bash
# 1. (optional) build and cache the feature matrices
python scripts/prepare_data.py --raw-dir data/raw --out-dir data/processed

# 2. compare the models with temporal validation (writes results/ + figure)
python scripts/evaluate.py --raw-dir data/raw --models baseline mlp gbm

# 3. train on the full history and generate the 2022 submission
python scripts/make_submission.py --raw-dir data/raw --model mlp --out submissions/submission.csv
```

`make_submission.py` accepts an optional `--template pred.csv` to match the official submission column names/order.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

Released under the MIT License — see `LICENSE`.
