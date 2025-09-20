<<<<<<< HEAD
# arrhythmia-detection
Using AI/ML in python to detect arrhythmias in the heart.
=======
# Arrhythmia Detection Hackathon (PyTorch)

This is a minimal scaffold for detecting arrhythmias from ECG signals.

## 🧹 Preprocessing Pipeline (Day 1)

Before training, we clean and segment the raw ECG signals so our models work with consistent, noise-free data.

### Steps:
1. **Load ECG Data**
   - Input CSV with a column named `ecg`  
   - Example: `test_data/test_ecg.csv`  

2. **Band-Pass Filtering**
   - Removes baseline wander and high-frequency noise  
   - Filter range: **0.5–40 Hz** (standard for arrhythmia detection)  

3. **Segmentation into Windows**
   - ECG is split into **10-second windows** with **5-second step size**  
   - Each window becomes one model input  

4. **R-Peak Detection (Demo)**
   - Peaks in the QRS complex are identified  
   - Enables feature extraction: RR intervals and heart rate variability (HRV)  

5. **Save Processed Data**
   - Exported to: `data/processed/windows.csv`  
   - Ready for baseline ML (logistic regression, random forest) or deep CNNs  

### Example Output
| Sample 0 | Sample 1 | ... | Sample N |
|----------|----------|-----|----------|
| 0.12     | 0.15     | ... | -0.07    |
| 0.10     | 0.18     | ... | -0.05    |
| ...      | ...      | ... | ...      |

Each row = one **10-second ECG window**.

---

✅ With this preprocessing in place, the project is ready for **Day 2: model training & evaluation**, followed by **Day 3: demo + alert logic**.


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
