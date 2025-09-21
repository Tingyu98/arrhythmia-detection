# scripts/make_windows.py
import argparse
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ----------------------------
# Bandpass filter helper
# ----------------------------
def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

# ----------------------------
# Window segmentation
# ----------------------------
def make_windows(signal, fs=360, win_sec=10, step_sec=5):
    win_size = int(win_sec * fs)
    step_size = int(step_sec * fs)
    windows = []

    if len(signal) < win_size:
        # Pad short signals with zeros to make one full window
        padded = np.zeros(win_size)
        padded[:len(signal)] = signal
        windows.append(padded)
    else:
        for start in range(0, len(signal) - win_size + 1, step_size):
            end = start + win_size
            windows.append(signal[start:end])

    return np.array(windows)

# ----------------------------
# Main CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="test_data/test_ecg.csv", help="CSV with column 'ecg'")
    ap.add_argument("--output", type=str, default="data/processed/windows.csv", help="Output CSV")
    ap.add_argument("--fs", type=int, default=360, help="Sampling frequency (Hz)")
    ap.add_argument("--win", type=float, default=10, help="Window length (s)")
    ap.add_argument("--step", type=float, default=5, help="Step size (s)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load ECG
    df = pd.read_csv(args.input)
    if "ecg" not in df.columns:
        raise ValueError("CSV must contain an 'ecg' column")
    raw_signal = df["ecg"].values

    # Filter
    filt_signal = bandpass_filter(raw_signal, fs=args.fs)

    # Segment
    windows = make_windows(filt_signal, fs=args.fs, win_sec=args.win, step_sec=args.step)

    # Save
    out_df = pd.DataFrame(windows)

    # --- Add dummy labels as strings (Normal / Abnormal) ---
    labels = np.random.choice(["Normal", "Abnormal"], size=windows.shape[0])
    out_df["label"] = labels

    out_df.to_csv(args.output, index=False)

    # Logs
    print(f"Raw signal length: {len(raw_signal)} samples")
    print(f"Window size: {args.win * args.fs} samples")
    print(f"Step size: {args.step * args.fs} samples")
    print(f"Number of windows created: {len(windows)}")
    print(f"✅ Saved {windows.shape[0]} windows with labels (Normal/Abnormal) to {args.output}")


if __name__ == "__main__":
    main()
