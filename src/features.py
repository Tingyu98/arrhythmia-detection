# src/features.py
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_features(window: np.ndarray) -> dict:
    """Extract simple statistical features from one ECG window."""
    feats = {}
    feats["mean"] = np.mean(window)
    feats["std"] = np.std(window)
    feats["min"] = np.min(window)
    feats["max"] = np.max(window)
    feats["skew"] = skew(window)
    feats["kurtosis"] = kurtosis(window)
    feats["energy"] = np.sum(window ** 2) / len(window)
    return feats

def windows_to_features(windows: np.ndarray, labels: np.ndarray = None) -> pd.DataFrame:
    """Convert a 2D array of windows into a DataFrame of features."""
    feature_dicts = [extract_features(w) for w in windows]
    df = pd.DataFrame(feature_dicts)
    if labels is not None:
        df["label"] = labels
    return df
