# src/features.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import neurokit2 as nk


def extract_features(window: np.ndarray, fs: int = 360) -> dict:
    """
    Extract statistical + HRV/morphology features from one ECG window.
    """
    feats = {}

    # --- Basic statistical features ---
    feats["mean"] = np.mean(window)
    feats["std"] = np.std(window)
    feats["min"] = np.min(window)
    feats["max"] = np.max(window)
    feats["skew"] = skew(window)
    feats["kurtosis"] = kurtosis(window)
    feats["energy"] = np.sum(window ** 2) / len(window)

    # --- ECG processing with NeuroKit2 ---
    try:
        # Clean and process
        ecg_cleaned = nk.ecg_clean(window, sampling_rate=fs)
        signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs)

        # Heart rate features
        feats["mean_hr"] = np.mean(signals["ECG_Rate"])
        feats["rmssd"] = nk.hrv_time(info, sampling_rate=fs)["HRV_RMSSD"].values[0]
        feats["cvrr"] = np.std(info["ECG_R_Peaks"]) / np.mean(info["ECG_R_Peaks"]) if np.mean(info["ECG_R_Peaks"]) > 0 else 0

        # Morphology features
        r_peaks = info["ECG_R_Peaks"]
        if len(r_peaks) > 1:
            qrs_durations = np.diff(r_peaks) / fs * 1000  # ms
            feats["qrs_width"] = np.mean(qrs_durations)
        else:
            feats["qrs_width"] = 0

        feats["r_amp"] = np.max(ecg_cleaned[r_peaks]) if len(r_peaks) > 0 else 0

    except Exception as e:
        # In case NeuroKit fails, fall back to safe defaults
        feats["mean_hr"] = 0
        feats["rmssd"] = 0
        feats["cvrr"] = 0
        feats["qrs_width"] = 0
        feats["r_amp"] = 0
        print(f"⚠️ Feature extraction error: {e}")

    return feats


def windows_to_features(
    windows: np.ndarray, labels: np.ndarray = None, fs: int = 360
) -> pd.DataFrame:
    """
    Convert a 2D array of windows into a DataFrame of features.
    """
    feature_dicts = [extract_features(w, fs=fs) for w in windows]
    df = pd.DataFrame(feature_dicts)
    if labels is not None:
        df["label"] = labels
    return df
