# scripts/eval_baseline.py
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
import os

def main(data_path, model_path, out_path):
    # Load features + labels
    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise ValueError("❌ Input CSV must contain a 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"]

    # Load trained baseline model
    model = joblib.load(model_path)

    # Predict
    preds = model.predict(X)

    # Define class names
    classes = ["Normal", "Abnormal"]

    # Adjust for cases where only one class is present in y
    present_classes = np.unique(y)
    labels = sorted(present_classes)
    target_names = [classes[i] for i in labels]

    # Generate report
    report = classification_report(y, preds, labels=labels, target_names=target_names)

    print("=== Baseline Model Evaluation ===")
    print(report)

    # Save report
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("=== Baseline Model Evaluation ===\n")
        f.write(report)

    print(f"✅ Evaluation report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline arrhythmia model")
    parser.add_argument("--data", type=str, required=True, help="Path to features CSV file")
    parser.add_argument("--model", type=str, required=True, help="Path to trained baseline model")
    parser.add_argument("--out", type=str, default="out/baseline_eval.txt", help="Output text file for report")
    args = parser.parse_args()

    main(args.data, args.model, args.out)
