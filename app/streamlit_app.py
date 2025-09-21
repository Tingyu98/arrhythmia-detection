import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go
from datetime import datetime

from src.models import SimpleCNN

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

# --- Streamlit UI ---
st.title("🫀 Arrhythmia Detection Demo: Normal vs Abnormal (PVC + AFib)")
st.markdown("Upload either a **feature CSV** (with `label`) or a **raw ECG CSV** (with `ecg`).")

uploaded_file = st.file_uploader("Upload ECG or feature CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- CASE 1: Feature dataset (baseline model) ---
    if "label" in df.columns:
        st.info("📊 Detected **feature dataset** → using baseline model (Random Forest).")
        baseline_model = joblib.load("out/baseline.joblib")

        X = df.drop(columns=["label"])
        preds = baseline_model.predict(X)

        classes = ["Normal", "Abnormal"]
        pred_labels = [classes[p] for p in preds]

        st.subheader("Baseline Predictions")
        st.write(pred_labels)

        # Show counts
        counts = pd.Series(pred_labels).value_counts()
        st.bar_chart(counts)

        # --- ALERT LOGIC ---
        if "Abnormal" in pred_labels:
            st.error("🚨 WARNING: Abnormal arrhythmia detected in dataset! 🚨")
        else:
            st.success("✅ Normal rhythm detected across dataset", icon="💓")

        # --- Save report to file (append mode) ---
        os.makedirs("out", exist_ok=True)
        report_path = os.path.join("out", "streamlit_eval.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(report_path, "a") as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"📅 Evaluation Run at {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write("Predictions:\n")
            f.write("\n".join(pred_labels) + "\n")
            f.write("\nClass counts:\n")
            f.write(str(counts) + "\n")

        st.success(f"📁 Evaluation report appended to {report_path}")

    # --- CASE 2: Raw ECG signal (deep CNN) ---
    elif "ecg" in df.columns:
        st.info("📈 Detected **raw ECG signal** → using deep CNN model.")

        signal = df["ecg"].values

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode="lines", name="ECG"))
        fig.update_layout(title="ECG Waveform", xaxis_title="Sample", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

        # Preprocess signal
        max_len = 3000
        sig_proc = np.zeros(max_len)
        sig_proc[: min(len(signal), max_len)] = signal[:max_len]

        x = torch.tensor(sig_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            predicted_class = np.argmax(probs)

        classes = ["Normal", "Abnormal"]
        pred_label = classes[predicted_class]

        st.subheader("CNN Prediction")
        st.write(f"**Class:** {pred_label}")
        st.write(f"**Confidence:** {probs[predicted_class]:.2f}")

        # Probabilities chart
        prob_fig = go.Figure([go.Bar(x=classes, y=probs)])
        prob_fig.update_layout(title="Class Probabilities")
        st.plotly_chart(prob_fig, use_container_width=True)

        # --- ALERT LOGIC ---
        if pred_label != "Normal":
            st.error("🚨 CRITICAL ALERT: Abnormal arrhythmia detected! 🚨")
        else:
            st.success("✅ Normal rhythm detected", icon="💓")

        # --- Save report to file (append mode) ---
        os.makedirs("out", exist_ok=True)
        report_path = os.path.join("out", "streamlit_eval.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(report_path, "a") as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"📅 CNN Evaluation Run at {timestamp}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Prediction: {pred_label}\n")
            f.write(f"Probabilities: {dict(zip(classes, probs))}\n")

        st.success(f"📁 CNN evaluation report appended to {report_path}")

    # --- CASE 3: Unsupported format ---
    else:
        st.error("CSV not recognized. Must contain either a `label` column (features) or an `ecg` column (raw signal).")
