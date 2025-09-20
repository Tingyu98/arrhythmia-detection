import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import torch

def main(data_path, out_path):
    # Load CSV
    df = pd.read_csv(data_path)

    # Define features and target
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train baseline Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save model weights only (just like train_deep.py)
    torch.save(model.state_dict(), "out/baseline.pth")
    print("✅ Baseline model saved to out/baseline.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to feature CSV file")
    parser.add_argument("--out", type=str, default="out/baseline.joblib", help="Output model path")
    args = parser.parse_args()

    main(args.data, args.out)
