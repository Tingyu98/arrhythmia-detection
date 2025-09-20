<<<<<<< HEAD
# arrhythmia-detection
Using AI/ML in python to detect arrhythmias in the heart.
=======
# Arrhythmia Detection Hackathon (PyTorch)

This is a minimal scaffold for detecting arrhythmias from ECG signals.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training
```bash
python scripts/train_deep.py --records 100 101 --epochs 3
```

## Run the demo app
```bash
streamlit run app/streamlit_app.py
```
>>>>>>> 7bd28d4 (Removed .venv from ghub tracking)

## Instructions:
```bash
git clone https://github.com/EganO11/arrhythmia-detection-hackathon.git
cd arrhythmia-detection-hackathon
```
## Setup environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

## Train a demo model
```bash
python scripts/train_deep.py --epochs 3
```

## Run the streamlit app
```bash
streamlit run app/streamlit_app.py
```
arrhythmia-detection-hackathon/
│
├── app/                 # Streamlit UI
│   └── streamlit_app.py
├── scripts/             # Training scripts
│   ├── train_deep.py
│   └── train_baseline.py
├── src/                 # Models
│   └── models.py
├── test_data/           # Sample ECG/test CSVs
├── out/                 # Saved models
├── requirements.txt     # Dependencies
└── README.md

```bash
Input formats:
Raw ECG signal: CSV with a column named ecg
Feature dataset: CSV with extracted features and a label column

Alerts:
The app highlights predictions:
✅ Normal rhythm
⚠️ Warning for AFib
🚨 Critical alert for PVC

🛠 Requirements:
Python 3.10+
PyTorch
Streamlit
NumPy, Pandas, scikit-learn, Plotly
```
