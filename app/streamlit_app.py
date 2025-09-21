import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from src.models import SimpleCNN

# --- Streamlit controls ---
st.sidebar.header("⚙️ Settings")
prob_threshold = st.sidebar.slider("Probability threshold", 0.0, 1.0, 0.6, 0.05)
n_consecutive = st.sidebar.slider("N consecutive abnormal windows", 1, 10, 3)
win_sec = st.sidebar.number_input("Window size (seconds)", 10, 60, 10)  # default 10s windows
fs = st.sidebar.number_input("Sampling frequency (Hz)", 100, 1000, 360)  # default 360Hz

# --- Load CNN model ---
MODEL_PATH = "out/cnn.pth"

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "state_dict" in checkpoint:
        input_len = checkpoint.get("input_len", 3000)
        model = SimpleCNN(input_len=input_len, n_classes=2)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = SimpleCNN(input_len=3000, n_classes=2)
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()

# --- Label normalization helper ---
def normalize_labels(labels):
    labels = labels.astype(str).str.upper()
    return labels.replace({
        "NORMAL": 0,
        "N": 0,
        "AF": 1,
        "AFIB": 1,
        "PVC": 1,
        "ABNORMAL": 1
    }).apply(lambda x: 0 if str(x) == "0" else 1)

# --- Streamlit UI ---
st.title("🫀 Arrhythmia Detection Demo: Normal vs Abnormal (PVC + AFib)")
st.markdown("Upload either a **feature CSV** (with `label`) or a **raw ECG CSV** (with `ecg`).")

uploaded_file = st.file_uploader("Upload ECG or feature CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- CASE 1: Feature dataset (baseline model) ---
    if "label" in df.columns:
        st.info("📊 Detected **feature dataset** → using baseline model (Random Forest).")
        bundle = joblib.load("out/baseline.joblib")
        baseline_model = bundle["model"]
        expected_features = [f.lower() for f in bundle["features"]]

        # Normalize labels
        df["label"] = normalize_labels(df["label"])

        # Lowercase all column names for consistency
        df.columns = [c.lower() for c in df.columns]

        # Extract + align features
        X = df.drop(columns=["label"])
        missing_features = [col for col in expected_features if col not in X.columns]
        extra_features = [col for col in X.columns if col not in expected_features + ["label"]]

        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features].astype(float)

        # 🔍 Sidebar diagnostics
        st.sidebar.subheader("📋 Feature Diagnostics")
        st.sidebar.write(f"**Expected features ({len(expected_features)}):** {expected_features}")
        st.sidebar.write(f"**Missing in file:** {missing_features if missing_features else 'None ✅'}")
        st.sidebar.write(f"**Extra in file (ignored):** {extra_features if extra_features else 'None ✅'}")

        # Predict probabilities
        probs = baseline_model.predict_proba(X)[:, 1]  # probability of Abnormal
        preds = (probs >= prob_threshold).astype(int)

        classes = ["Normal", "Abnormal"]
        pred_labels = [classes[p] for p in preds]

        st.subheader("Baseline Predictions")
        st.write(pred_labels)

        # Percentages
        counts = pd.Series(pred_labels).value_counts(normalize=True) * 100
        st.bar_chart(counts)

        # Confusion Matrix
        cm = confusion_matrix(df["label"], preds, labels=[0, 1])
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        cm_labels = ["Normal", "Abnormal"]
        cm_fig = ff.create_annotated_heatmap(
            z=np.round(cm_norm*100, 1),
            x=cm_labels,
            y=cm_labels,
            annotation_text=np.array([["{:.1f}%".format(v) for v in row] for row in cm_norm*100]),
            colorscale="Blues",
            showscale=True
        )
        cm_fig.update_layout(
            title="Confusion Matrix (Normalized %)",
            xaxis_title="Predicted",
            yaxis_title="True"
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # Metrics
        try:
            f1 = f1_score(df["label"], preds)
            auroc = roc_auc_score(df["label"], probs)
            false_alarms = ((df["label"] == 0) & (preds == 1)).sum()
            hours = len(df) * (win_sec / 3600.0)
            false_alarms_per_hour = false_alarms / hours if hours > 0 else 0

            st.metric("F1 Score", f"{f1:.3f}")
            st.metric("AUROC", f"{auroc:.3f}")
            st.metric("False Alarms/hour", f"{false_alarms_per_hour:.2f}")
        except Exception as e:
            st.warning(f"⚠️ Could not compute metrics: {e}")

        # Alert logic
        consec_count = 0
        alert_triggered = False
        for p in preds:
            if p == 1:
                consec_count += 1
                if consec_count >= n_consecutive:
                    alert_triggered = True
                    break
            else:
                consec_count = 0

        if alert_triggered:
            st.error(f"🚨 WARNING: {n_consecutive} consecutive abnormal windows detected! 🚨")
        elif "Abnormal" in pred_labels:
            st.warning("⚠️ Some abnormal windows detected (not consecutive).")
        else:
            st.success("✅ Normal rhythm detected across dataset", icon="💓")

    # --- CASE 2: Raw ECG signal (deep CNN, multi-window) ---
    elif "ecg" in df.columns:
        st.info("📈 Detected **raw ECG signal** → using deep CNN model.")

        signal = df["ecg"].values

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode="lines", name="ECG"))
        fig.update_layout(title="ECG Waveform", xaxis_title="Sample", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

        # Split into windows
        win_size = win_sec * fs
        n_windows = int(np.ceil(len(signal) / win_size))
        probs_all, preds_all = [], []

        for i in range(n_windows):
            start = i * win_size
            end = min((i + 1) * win_size, len(signal))
            window = signal[start:end]

            # Pad/truncate to 3000 samples
            max_len = 3000
            sig_proc = np.zeros(max_len)
            sig_proc[: min(len(window), max_len)] = window[:max_len]

            x = torch.tensor(sig_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                outputs = model(x)
                probs = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
                probs_all.append(probs[1])
                preds_all.append(1 if probs[1] >= prob_threshold else 0)

        # Display window-level probabilities
        st.subheader("CNN Window Predictions")
        st.line_chart(probs_all)

        # Alert logic
        consec_count = 0
        alert_triggered = False
        for p in preds_all:
            if p == 1:
                consec_count += 1
                if consec_count >= n_consecutive:
                    alert_triggered = True
                    break
            else:
                consec_count = 0

        if alert_triggered:
            st.error(f"🚨 CRITICAL ALERT: {n_consecutive} consecutive abnormal windows detected! 🚨")
        elif any(preds_all):
            st.warning("⚠️ Some abnormal windows detected (not consecutive).")
        else:
            st.success("✅ Normal rhythm detected", icon="💓")

    # --- CASE 3: Unsupported format ---
    else:
        st.error("CSV not recognized. Must contain either a `label` column (features) or an `ecg` column (raw signal).")
