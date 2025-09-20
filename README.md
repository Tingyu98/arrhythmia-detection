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
