# scripts/make_features.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
import os
import pandas as pd
import numpy as np
import joblib

from src.features import windows_to_features


def main(input_path, output_path, fs=360, model_path="out/baseline.joblib"):
    # Load windows CSV
    df = pd.read_csv(input_path)

    if "label" not in df.columns:
        raise ValueError("Input windows file must contain a 'label' column")

    # Separate windows and labels
    labels = df["label"].values
    windows = df.drop(columns=["label"]).values

    # Extract features
    feat_df = windows_to_features(windows, labels=labels, fs=fs)

    # Normalize colnames
    feat_df.columns = [c.lower() for c in feat_df.columns]

    # --- Align to trained model features if available ---
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            expected_features = [f.lower() for f in bundle.get("features", [])]

            if expected_features:
                # Add missing columns
                for col in expected_features:
                    if col not in feat_df.columns:
                        feat_df[col] = 0
                # Keep only expected + label
                feat_df = feat_df[[c for c in expected_features if c in feat_df.columns] + ["label"]]

                print(f"🔑 Aligned features to model: {len(expected_features)} features")
        except Exception as e:
            print(f"⚠️ Could not align features to model ({e}). Proceeding with raw extracted features.")

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
    ap.add_argument("--fs", type=int, default=360, help="Sampling frequency (Hz)")
    ap.add_argument("--model", type=str, default="out/baseline.joblib", help="Path to trained baseline model")
    args = ap.parse_args()

    main(args.input, args.output, fs=args.fs, model_path=args.model)
