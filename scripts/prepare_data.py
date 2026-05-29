"""Build and cache the feature matrices (train + 2022 test) as parquet files."""

import argparse
import os

from edf import pipeline


def main():
    ap = argparse.ArgumentParser(description="Build and cache feature matrices.")
    ap.add_argument("--raw-dir", default="data/raw",
                    help="Folder with train.csv, test.csv, meteo.parquet.")
    ap.add_argument("--out-dir", default="data/processed",
                    help="Where to write the cached parquet files.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    X, y, weather = pipeline.build_training_data(args.raw_dir)
    X.to_parquet(os.path.join(args.out_dir, "X_train.parquet"))
    y.to_parquet(os.path.join(args.out_dir, "y_train.parquet"))

    X_test, test_index = pipeline.build_test_data(args.raw_dir, weather)
    X_test.to_parquet(os.path.join(args.out_dir, "X_test.parquet"))
    test_index.to_frame("date").to_parquet(os.path.join(args.out_dir, "test_index.parquet"))

    print(f"Cached X_train{tuple(X.shape)}, y_train{tuple(y.shape)}, "
          f"X_test{tuple(X_test.shape)} to {args.out_dir}")


if __name__ == "__main__":
    main()
