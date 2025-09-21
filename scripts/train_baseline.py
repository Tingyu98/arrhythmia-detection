# scripts/train_baseline.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def main(data_path, out_path):
    # Load CSV
    df = pd.read_csv(data_path)

    # --- Normalize labels to binary ---
    if "label" not in df.columns:
        # If no labels, create dummy binary ones
        df["label"] = np.random.randint(0, 2, size=len(df))
    else:
        df["label"] = df["label"].astype(str).str.upper().replace({
            "NORMAL": 0,
            "N": 0,
            "AF": 1,
            "AFIB": 1,
            "PVC": 1,
            "ABNORMAL": 1
        })
        # Anything unmapped → Abnormal (1)
        df["label"] = df["label"].apply(lambda x: 0 if str(x) == "0" else 1)

    # --- Keep only numeric feature columns ---
    feature_cols = [c for c in df.columns if c != "label" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].astype(float)
    y = df["label"].astype(int)

    # 🔑 Normalize feature names to lowercase for consistency
    X.columns = [c.lower() for c in X.columns]
    feature_cols = list(X.columns)

    print(f"✅ Using {len(feature_cols)} features: {feature_cols}")

    # --- Train/test split ---
    if len(df) < 5:
        print("⚠️ Very small dataset — training on all data (no split).")
        X_train, y_train, X_test, y_test = X, y, X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    # --- Train model ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    if len(X_test) > 0:
        preds = clf.predict(X_test)
        print("=== Classification Report (Binary: Normal=0, Abnormal=1) ===")
        print(classification_report(
            y_test, preds, labels=[0, 1], target_names=["Normal", "Abnormal"]
        ))
    else:
        print("⚠️ No test set available, skipping evaluation.")

    # --- Save model + lowercase feature names ---
    model_bundle = {
        "model": clf,
        "features": feature_cols
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model_bundle, out_path)

    print(f"✅ Baseline model + lowercase features saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--out", type=str, default="out/baseline.joblib", help="Output model path")
    args = parser.parse_args()

    main(args.data, args.out)
