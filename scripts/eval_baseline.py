# scripts/eval_baseline.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib

def normalize_labels(labels):
    """Convert labels to binary: Normal=0, Abnormal=1"""
    labels = labels.astype(str).str.upper().replace({
        "NORMAL": 0,
        "N": 0,
        "AF": 1,
        "AFIB": 1,
        "PVC": 1,
        "ABNORMAL": 1
    })
    return labels.apply(lambda x: 0 if str(x) == "0" else 1)

def main(data_path, model_path):
    # Load CSV
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("❌ The dataset must contain a 'label' column for evaluation.")

    # Normalize labels
    df["label"] = normalize_labels(df["label"])

    # Load model bundle (model + expected features)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    expected_features = [f.lower() for f in bundle["features"]]

    # --- Normalize feature names in dataset ---
    df.columns = [c.lower() for c in df.columns]

    # Select + align features
    X = df[expected_features].astype(float)
    y = df["label"].astype(int)

    # Predict
    preds = model.predict(X)

    # Report
    print("=== Baseline Model Evaluation ===")
    print(classification_report(
        y, preds, labels=[0, 1], target_names=["Normal", "Abnormal"]
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file (.joblib)")
    args = parser.parse_args()

    main(args.data, args.model)
