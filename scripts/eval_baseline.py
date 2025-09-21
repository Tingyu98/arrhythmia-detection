# scripts/eval_baseline.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

def main(data_path, model_path):
    # Load CSV
    df = pd.read_csv(data_path)

    # Ensure labels exist
    if "label" not in df.columns:
        raise ValueError("❌ The dataset must contain a 'label' column for evaluation.")

    # Ensure labels are numeric (map strings → ints if needed)
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"Normal": 0, "Abnormal": 1})

    # Keep only numeric feature columns
    feature_cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].astype(float)
    y = df["label"]

    # Load trained baseline model
    clf = joblib.load(model_path)

    # Predict
    preds = clf.predict(X)

    # Classification report
    labels_present = unique_labels(y, preds)
    target_names = ["Normal", "Abnormal"]

    print("=== Baseline Model Evaluation ===")
    print(classification_report(
        y,
        preds,
        labels=labels_present,
        target_names=[target_names[i] for i in labels_present]
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.joblib)")
    args = parser.parse_args()

    main(args.data, args.model)
