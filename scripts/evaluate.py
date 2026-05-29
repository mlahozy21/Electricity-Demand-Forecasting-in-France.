"""Temporal-validation comparison of the forecasting models.

Trains on years < --val-year, evaluates on --val-year with the challenge
metric (sum of per-column RMSE), prints a table and saves results + figures.
"""

import argparse
import os

import pandas as pd

from edf import metrics, pipeline
from edf.data import TARGETS


def main():
    ap = argparse.ArgumentParser(description="Compare models with temporal validation.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--models", nargs="+", default=["baseline", "mlp", "gbm"],
                    choices=["baseline", "mlp", "gbm"])
    ap.add_argument("--val-year", type=int, default=2021)
    ap.add_argument("--epochs", type=int, default=200, help="Max epochs for the MLP.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y, _ = pipeline.build_training_data(args.raw_dir)
    Xtr, ytr, Xva, yva = pipeline.temporal_split(X, y, args.val_year)
    print(f"Train rows: {len(Xtr)} | Validation ({args.val_year}) rows: {len(Xva)}")

    rows, preds = [], {}
    for name in args.models:
        if name == "baseline":
            model = pipeline.get_model("baseline").fit(ytr)
            pred = model.predict(Xva.index)
        elif name == "mlp":
            model = pipeline.get_model("mlp", max_epochs=args.epochs).fit(Xtr, ytr, Xva, yva)
            pred = model.predict(Xva)
        else:
            model = pipeline.get_model("gbm").fit(Xtr, ytr)
            pred = model.predict(Xva)
        preds[name] = pred
        score = metrics.challenge_score(yva, pred)
        rows.append({"model": name, "challenge_score": score, "mae": metrics.mae(yva, pred)})
        print(f"  {name:10s} challenge_score={score:10.1f}  mae={metrics.mae(yva, pred):.1f}")

    table = pd.DataFrame(rows).sort_values("challenge_score")
    table.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    best = table.iloc[0]["model"]
    print(f"\nMetrics written to {args.out_dir}/metrics.csv (best model: {best})")

    # Figure: per-target RMSE for the best model (optional; skipped if matplotlib fails).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rmse = metrics.per_column_rmse(yva, preds[best])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(TARGETS, rmse, color="#1f77b4")
        ax.set_xlabel("RMSE")
        ax.set_title(f"Per-series RMSE on {args.val_year} — {best}")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "per_series_rmse.png"), dpi=150)
        print(f"Figure written to {args.out_dir}/per_series_rmse.png")
    except Exception as e:
        print(f"(Skipping figure — matplotlib unavailable: {e})")


if __name__ == "__main__":
    main()
