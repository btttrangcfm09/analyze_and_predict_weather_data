"""
Task: Fit a MinMaxScaler on all numeric feature columns, scale the DataFrame
      in-place to [0, 1], and persist both the scaler and the feature-column
      list to disk so inference can reproduce the exact transform.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

SCALER_PATH       = 'scaler_rain.pkl'
FEATURE_COLS_PATH = 'feature_cols_rain.pkl'

# Coordinates / time are identifiers, not model inputs → never scaled.
_EXCLUDE = {'latitude', 'longitude', 'valid_time'}


def fit_and_scale(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in _EXCLUDE]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    df[feature_cols] = scaled.astype(np.float32)

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLS_PATH, 'wb') as f:
        pickle.dump(feature_cols, f)

    print(f"  Scaler          -> {SCALER_PATH}")
    print(f"  Feature columns -> {FEATURE_COLS_PATH}  ({len(feature_cols)} cols)")

    return df, scaler, feature_cols
