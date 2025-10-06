<<<<<<< HEAD
# Arrhythmia Detection Hackathon 

## Overview
This project leverages AI and Machine Learning in Python to detect cardiac arrhythmias from ECG signals. Using both feature-based machine learning models and a deep convolutional neural network (CNN), the goal is to identify abnormal heart rhythms such as atrial fibrillation (AFib) and premature ventricular contractions (PVC). A Streamlit web app was developed to visualize predictions in real time, allowing users to upload ECG data and monitor abnormal rhythm alerts interactively.

## 🚀 Features
- Binary classification: **Normal vs Abnormal** (AFib + PVC combined).
- Baseline model (Random Forest) trained on extracted ECG features.
- Deep CNN trained directly on raw ECG signals.
- Streamlit app for live demo:
   - Upload **feature CSVs** or **raw ECG signals**.
   - Probability bars and class confidence.
   - Configurable alert logic: **N consecutive abnormal windows**.
   - Adjustable sliders:
      - Probability threshold  
      - N consecutive abnormal windows  
   - Baseline predictions shown as **percentages instead of counts**.
   - Metrics panel: **AUROC, F1, false alarms/hour**.
   - Confusion matrix visualization.

---

## 🧹 Preprocessing Pipeline (Day 1)

### Steps
1. **Load ECG Data**
   - Input CSV with a column named `ecg`  
   - Example: `test_data/test_ecg.csv`  

2. **Band-Pass Filtering**
   - Removes baseline wander and high-frequency noise  
   - Filter range: **0.5–40 Hz**

3. **Segmentation into Windows**
   - ECG split into **10-second windows** with **5-second step size**

4. **R-Peak Detection (Demo)**
   - QRS complexes identified  
   - Enables RR interval and HRV feature extraction  

5. **Save Processed Data**
   - Exported to `data/processed/windows.csv`

---

## 🧠 Day 2: Model Training & Evaluation

### Baseline (Random Forest)
- Trains on `features.csv`
- Evaluated with sklearn `classification_report`

### Deep CNN
- Input: raw ECG windows (length 3000)
- Output: binary classification
- Saved to `out/cnn.pth`

---

## 🎛️ Day 3: Integration & Streamlit Demo

Run the app:
```bash
streamlit run app/streamlit_app.py
````

## Inputs:
   - Raw ECG signal: CSV with column ecg.
   - Feature dataset: CSV with numeric features + label column.

## Alerts:
- ✅ Normal rhythm
- ⚠️ Warning when threshold exceeded
- 🚨 Critical alert if N consecutive abnormal windows cross threshold

## Summary Metrics
- F1 Score
- AUROC
- False alarms per hour

## Requirements:
- Python 3.10+
- PyTorch
- Streamlit
- NumPy, Pandas, scikit-learn, Plotly, neurokit2

## Instructions:
```bash
git clone https://github.com/EganO11/arrhythmia-detection-hackathon.git
cd arrhythmia-detection-hackathon
```
## Setup environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```
Usage Examples

Train baseline model:
````bash
python scripts/train_baseline.py --data data/processed/features.csv --out out/baseline.joblib
````

Evaluate baseline:
````bash
python scripts/eval_baseline.py --data data/processed/features.csv --model out/baseline.joblib
````

Train CNN:
````bash
python scripts/train_deep.py --epochs 3
````
## DEMO Instructions:

Run the streamlit app
```bash
streamlit run app/streamlit_app.py
```
Upload a CSV file:
- Raw ECG → must contain a column named ecg
- Feature dataset → must contain numeric features + label

Adjust sidebar sliders:
- Probability threshold
- N consecutive abnormal windows

View:
- Predicted classes and probabilities
- Bar chart of percentages
- Confusion matrix (baseline model)
- Metrics: AUROC, F1, false alarms/hour
- Alert messages when conditions are met

arrhythmia-detection-hackathon/
│
├── app/                 # Streamlit UI
│   └── streamlit_app.py
├── scripts/             # Training + preprocessing scripts
│   ├── make_windows.py
│   ├── make_features.py
│   ├── train_baseline.py
│   ├── eval_baseline.py
│   └── train_deep.py
├── src/                 # Models + feature extraction
│   ├── models.py
│   └── features.py
├── data/                # Processed data (not tracked in Git)
│   └── processed/
├── test_data/           # Sample ECG/test CSVs
├── out/                 # Saved models
├── requirements.txt     # Dependencies
└── README.md


## Partner Contribution – Modeling & Experimentation

This section summarizes additional work contributed via Google Colab and planning documents.  
The focus was on evaluating both **feature-based ML models** and a **deep learning CNN** using the MIT-BIH Arrhythmia Database.

### Dataset
- **MIT-BIH Arrhythmia Database (mitdb)** – annotated ECG signals with diverse arrhythmias.  
- (Optional extension) **AFDB** – atrial fibrillation-focused records.  
- Binary setup: *Normal* vs. *Abnormal* (AFib, PVC, and other irregularities).  
- Multi-class extension: *Normal*, *PVC*, *AFib*.  

### Preprocessing & Labeling
- **Filtering:** Bandpass 0.5–40 Hz to remove baseline wander & noise.  
- **Segmentation:** 10s windows, 5s stride.  
- **Labeling logic:**  
  - AF → "AF"  
  - PVC → "PVC" (if ≥20% of beats and ≥2 total)  
  - Other abnormal beats/rhythms → "Abnormal"  
  - Else → "Normal"  
- For **binary classification**, AF + PVC + other abnormal = "Abnormal".

### Feature Engineering
Extracted physiologically meaningful features per ECG window:
- Mean heart rate (`mean_HR`)  
- Coefficient of variation of RR intervals (`CVRR`)  
- RMSSD (parasympathetic activity)  
- QRS width (ventricular conduction)  
- R-wave amplitude (`R_amp`)  
- Signal energy  
- R count (beats per segment)  

### Modeling Results (Binary: Normal vs Abnormal)

**Logistic Regression**
- Accuracy: **0.88**, Macro F1: **0.87**  
- Good recall, but some abnormal cases misclassified as Normal.  

**Random Forest**
- Parameters: `n_estimators=300`, `class_weight="balanced"`  
- Accuracy: **0.90**, Macro F1: **0.90**  
- Strong recall for Abnormal rhythms (0.91).  

**1D CNN**
- Architecture: 3 Conv1D layers → GlobalAvgPool → Fully Connected.  
- Accuracy: **0.892**, Macro F1: **0.891**  
- Highest recall for Abnormal (92.8%), reducing missed detections.  

### Modeling Results (Multi-class: Normal / PVC / AFib)

**Random Forest + SMOTE**
- Accuracy: **0.89**, Weighted F1: **0.89**  
- PVC detected well (F1 ≈ 0.87–0.93).  
- AFib detection weaker (F1 ≈ 0.42).  

**CNN + Oversampling**
- Accuracy: ~0.50, F1: ~0.50.  
- Model struggled with AFib class due to imbalance.  

### Key Takeaways
- **Random Forest** → Strong balanced accuracy, interpretable features.  
- **1D CNN** → Best for abnormal recall, suitable for medical contexts.  
- **Logistic Regression** → Simple baseline, less effective for abnormalities.  
- **Multi-class setting** is harder, especially for AFib. More data or advanced architectures (e.g., ResNet/LSTM, focal loss) recommended.

---

