import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go

from src.models import SimpleCNN

# --- Load model ---
MODEL_PATH = "out/cnn.pth"

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    if "state_dict" in checkpoint:
        # Use the input_len saved during training
        input_len = checkpoint.get("input_len", 3000)
        model = SimpleCNN(input_len=input_len, n_classes=3)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback for old checkpoints (no input_len stored)
        model = SimpleCNN(input_len=3000, n_classes=3)
        model.load_state_dict(checkpoint)

    model.eval()
    return model



model = load_model()

# --- Streamlit UI ---
st.title("🫀 Arrhythmia Detection Demo")
st.markdown("Upload an ECG CSV file with a column named **`ecg`** to analyze.")

uploaded_file = st.file_uploader("Upload ECG or feature CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- CASE 1: Feature dataset (baseline model) ---
    if "label" in df.columns:
        st.write("📊 Detected **feature dataset**. Using baseline model...")
        baseline_model = joblib.load("out/baseline.joblib")

        X = df.drop(columns=["label"])
        preds = baseline_model.predict(X)

        classes = ["Normal", "AFib", "PVC"]
        pred_labels = [classes[p] for p in preds]

        st.subheader("Baseline Predictions")
        st.write(pred_labels)

        # Show counts as a bar chart
        counts = pd.Series(pred_labels).value_counts()
        st.bar_chart(counts)

        # --- ALERT LOGIC ---
        if "PVC" in pred_labels:
            st.error("🚨 CRITICAL ALERT: PVC arrhythmia detected in dataset! 🚨")
        elif "AFib" in pred_labels:
            st.warning("⚠️ Warning: Possible AFib arrhythmia detected in dataset")
        else:
            st.success("✅ Normal rhythm detected across dataset", icon="💓")


    # --- CASE 2: Raw ECG signal (deep CNN) ---
    elif "ecg" in df.columns:
        st.write("📈 Detected **raw ECG signal**. Using deep CNN model...")

        signal = df["ecg"].values

        # Plot waveform
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=signal, mode="lines", name="ECG"))
        fig.update_layout(title="ECG Waveform", xaxis_title="Sample", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

        # Preprocess signal (pad/truncate)
        max_len = 3000
        sig_proc = np.zeros(max_len)
        sig_proc[: min(len(signal), max_len)] = signal[:max_len]

        x = torch.tensor(sig_proc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            predicted_class = np.argmax(probs)

        classes = ["Normal", "AFib", "PVC"]
        pred_label = classes[predicted_class]

        st.subheader("Prediction")
        st.write(f"**Class:** {pred_label}")
        st.write(f"**Confidence:** {probs[predicted_class]:.2f}")

        # Probabilities chart
        prob_fig = go.Figure([go.Bar(x=classes, y=probs)])
        prob_fig.update_layout(title="Class Probabilities")
        st.plotly_chart(prob_fig, use_container_width=True)

        # --- ALERT LOGIC ---
        if pred_label != "Normal":
            if pred_label == "PVC":
                st.error("🚨 CRITICAL ALERT: PVC arrhythmia detected! 🚨")
            else:
                st.warning(f"⚠️ Warning: Possible {pred_label} arrhythmia detected")
        else:
            st.success("✅ Normal rhythm detected", icon="💓")

    # --- CASE 3: Unsupported format ---
    else:
        st.error("CSV not recognized. Must contain either a `label` column (features) or an `ecg` column (raw signal).")
