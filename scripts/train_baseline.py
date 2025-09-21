# scripts/train_baseline.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

def main(data_path, out_path):
    # Load CSV
    df = pd.read_csv(data_path)

    # Create dummy labels if missing
    if "label" not in df.columns:
        df["label"] = np.random.randint(0, 2, size=len(df))  # 2 classes: Normal, Abnormal

    # Define features and target
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    # Handle very small datasets
    if len(df) < 5:
        print("⚠️ Very small dataset — training on all data (no split).")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # Train baseline Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate (only if we have a test set)
    if len(X_test) > 0:
        preds = clf.predict(X_test)
        labels_present = unique_labels(y_test, preds)
        target_names = ["Normal", "Abnormal"]

        print("=== Classification Report (Binary: Normal=0, Abnormal=1) ===")
        print(classification_report(
            y_test,
            preds,
            labels=labels_present,
            target_names=[target_names[i] for i in labels_present]
        ))
    else:
        print("⚠️ No test set available, skipping evaluation.")

    # Save model with joblib
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"✅ Baseline model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--out", type=str, default="out/baseline.joblib", help="Output model path")
    args = parser.parse_args()

    main(args.data, args.out)
