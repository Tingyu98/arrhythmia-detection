# scripts/make_features.py
import argparse
import os
import pandas as pd
import numpy as np
from src.features import windows_to_features

def main(input_path, output_path):
    # Load windows CSV
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        raise ValueError("Input windows file must contain a 'label' column")

    # Separate windows and labels
    labels = df["label"].values
    windows = df.drop(columns=["label"]).values

    # Extract features
    features = windows_to_features(windows)

    # Convert to DataFrame
    feat_df = pd.DataFrame(features)

    # Add labels back
    feat_df["label"] = labels

    # 🔑 Normalize all column names to lowercase
    feat_df.columns = [c.lower() for c in feat_df.columns]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df.to_csv(output_path, index=False)

    print(f"✅ Features extracted and saved to {output_path}")
    print(f"Shape: {feat_df.shape}")
    print(f"Columns: {list(feat_df.columns)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to windows.csv")
    ap.add_argument("--output", type=str, required=True, help="Path to save features.csv")
    args = ap.parse_args()

    main(args.input, args.output)
