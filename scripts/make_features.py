# scripts/make_features.py
import argparse
import pandas as pd
import numpy as np
from src.features import windows_to_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/processed/windows.csv", help="CSV of segmented windows")
    ap.add_argument("--output", type=str, default="data/processed/features.csv", help="Output features CSV")
    args = ap.parse_args()

    # Load windows
    df = pd.read_csv(args.input)
    labels = df["label"].values if "label" in df.columns else None
    windows = df.drop(columns=["label"], errors="ignore").values

    # Convert to features
    feat_df = windows_to_features(windows, labels)

    # Save
    feat_df.to_csv(args.output, index=False)
    print(f"✅ Saved {feat_df.shape[0]} rows, {feat_df.shape[1]} columns → {args.output}")

if __name__ == "__main__":
    main()
