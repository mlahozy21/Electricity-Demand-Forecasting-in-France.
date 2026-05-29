"""Train a model on the full history and produce the 2022 submission file."""

import argparse
import os

from edf import pipeline


def main():
    ap = argparse.ArgumentParser(description="Generate the 2022 submission.")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--model", default="gbm", choices=["baseline", "mlp", "gbm"])
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--epochs", type=int, default=300, help="Max epochs for the MLP.")
    ap.add_argument("--template", default=None,
                    help="Optional submission template (e.g. pred.csv) to align columns.")
    args = ap.parse_args()

    X, y, weather = pipeline.build_training_data(args.raw_dir)
    X_test, test_index = pipeline.build_test_data(args.raw_dir, weather)

    if args.model == "baseline":
        model = pipeline.get_model("baseline").fit(y)
        pred = model.predict(test_index)
    elif args.model == "mlp":
        model = pipeline.get_model("mlp", max_epochs=args.epochs).fit(X, y)
        pred = model.predict(X_test)
    else:
        model = pipeline.get_model("gbm").fit(X, y)
        pred = model.predict(X_test)

    pred.insert(0, "date", test_index)

    if args.template:
        import pandas as pd
        tmpl_cols = list(pd.read_csv(args.template, nrows=0).columns)
        # Map our target columns onto the template's (handles a 'pred_' prefix).
        rename = {c: (f"pred_{c}" if f"pred_{c}" in tmpl_cols else c) for c in pred.columns}
        pred = pred.rename(columns=rename)
        keep = [c for c in tmpl_cols if c in pred.columns]
        pred = pred[keep]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pred.to_csv(args.out, index=False)
    print(f"Wrote submission with {pred.shape[0]} rows, {pred.shape[1]} cols to {args.out}")


if __name__ == "__main__":
    main()
