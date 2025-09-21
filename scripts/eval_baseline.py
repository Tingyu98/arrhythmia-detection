# scripts/eval_baseline.py
import argparse
import pandas as pd
import joblib
from sklearn.metrics import classification_report

def main(data_path, model_path):
    # Load features
    df = pd.read_csv(data_path)
    if "label" not in df.columns:
        raise ValueError("Features CSV must contain a 'label' column")

    X = df.drop(columns=["label"])
    y = df["label"]

    # Load model
    clf = joblib.load(model_path)

    # Predict + report
    preds = clf.predict(X)
    print("=== Baseline Model Evaluation ===")
    print(classification_report(y, preds, target_names=["Normal", "Abnormal"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline model")
    parser.add_argument("--data", type=str, default="data/processed/features.csv", help="Path to features CSV")
    parser.add_argument("--model", type=str, default="out/baseline.joblib", help="Path to saved model")
    args = parser.parse_args()
    main(args.data, args.model)
