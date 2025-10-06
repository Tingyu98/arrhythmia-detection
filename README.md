# Arrhythmia Detection Hackathon 

## Overview
This project leverages AI and Machine Learning in Python to detect cardiac arrhythmias from ECG signals. Using both feature-based machine learning models and a deep convolutional neural network (CNN), the goal is to identify abnormal heart rhythms such as atrial fibrillation (AFib) and premature ventricular contractions (PVC). A Streamlit web app was developed to visualize predictions in real time, allowing users to upload ECG data and monitor abnormal rhythm alerts interactively.

## Goal
* Build a machine learning pipeline capable of classifying ECG signals as Normal or Abnormal.
* Compare the performance of baseline models (Random Forest, Logistic Regression) with a 1D CNN trained on raw signals.
* Provide an interactive Streamlit demo for real-time detection, visualization, and model explainability.
* Evaluate and improve models for both binary and multi-class arrhythmia classification (Normal / PVC / AFib).

---

## Methodology
**Data Sources**
* MIT-BIH Arrhythmia Database (mitdb) – Annotated ECG signals containing various arrhythmia types.
  
**Preprocessing Pipeline**
* Band-pass filtering (0.5–40 Hz) to remove noise and baseline wander.
* Segmentation into 10-second windows (5-second stride).
* R-peak detection and extraction of RR intervals, HRV, and QRS features.
* Labeling logic:
   * AF → Atrial Fibrillation
   * PVC → Premature Ventricular Contraction (≥20% beats)
   * Other irregular rhythms → Abnormal
   * Normal beats → Normal

**Feature Engineering**
Extracted physiological and statistical features per ECG window:
   * Mean Heart Rate (mean_HR)
   * RR Interval Variability (CVRR, RMSSD)
   * QRS Width & R-Wave Amplitude
   * Signal Energy & R Count

**Modeling**
* Baseline Models: Logistic Regression, Random Forest (n_estimators=300, class_weight="balanced").
* Deep Learning Model: 1D CNN with 3 Conv1D layers → Global Average Pooling → Fully Connected layer.
* Evaluation Metrics: Accuracy, Macro-F1, AUROC, false alarms/hour.

**Visualization & Deployment**
* Developed a Streamlit app for real-time demo:
   * Upload ECG CSVs (raw signals or extracted features).
   * Adjustable thresholds and alert conditions.
   * Real-time probability bars, confusion matrix, and metrics.

* Tools: PyTorch, scikit-learn, NumPy, Pandas, Plotly, neurokit2.

## Results & Insights

**Binary Classification (Normal vs Abnormal)**

| Model               | Accuracy | Macro F1 | Key Observation                                          |
| ------------------- | -------- | -------- | -------------------------------------------------------- |
| Logistic Regression | 0.88     | 0.87     | Good baseline, but missed some abnormal cases.           |
| Random Forest       | **0.90** | **0.90** | Best overall performance; strong abnormal recall (0.91). |
| 1D CNN              | 0.892    | 0.891    | Best abnormal recall (92.8%), reduced missed detections. |

**Multi-Class Classification (Normal / PVC / AFib)**

| Model                 | Accuracy | Weighted F1 | Notes                                               |
| --------------------- | -------- | ----------- | --------------------------------------------------- |
| Random Forest + SMOTE | 0.89     | 0.89        | PVC detected well (F1 ≈ 0.9); AFib weaker (0.42).   |
| CNN + Oversampling    | ~0.50    | ~0.50       | Struggled due to class imbalance; more data needed. |

**Key Takeaways:**
* Random Forest – Reliable, interpretable, and high recall for abnormalities.
* CNN – Best suited for detecting subtle ECG pattern changes; high sensitivity to abnormal rhythms.
* AFib classification remains challenging; requires more data and architecture tuning.

## Application
* Clinical Use: Early arrhythmia detection from wearable or hospital ECG devices.
* Research: Extend to multi-lead ECG or 12-lead datasets for clinical validation.
* Interactive Demo: Streamlit app enables live testing and visualization for educational and diagnostic exploration.

## Project Structure
arrhythmia-detection-hackathon/
├── app/                    # Streamlit UI
│   └── streamlit_app.py
├── scripts/                # Training + preprocessing scripts
│   ├── make_windows.py
│   ├── make_features.py
│   ├── train_baseline.py
│   ├── eval_baseline.py
│   └── train_deep.py
├── src/                    # Models + feature extraction
│   ├── models.py
│   └── features.py
├── data/processed/         # Processed ECG data
├── test_data/              # Sample ECG CSVs
├── out/                    # Saved models (.pth, .joblib)
└── requirements.txt        # Dependencies



