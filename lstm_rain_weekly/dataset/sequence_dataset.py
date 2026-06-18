"""
Task: Turn the scaled DataFrame into per-coordinate numpy arrays and build
      lazy-loading sequence indices, split temporally per coordinate
      (80% train / 20% test) so there is zero geographic leakage.

Forecasting setup (Direct Multi-step):
  X : 168 past hours  × N features   → shape (168, N)
  y : 168 future hours of rainfall   → shape (168,)
      i.e. arr[start+seq_len : start+seq_len+predict_steps, target_idx]

Memory strategy: coordinate arrays are stored once; every sample is just a
(coord_idx, start) pointer. Sequences are sliced lazily in __getitem__, so we
never materialise the full 3-D tensor (which would blow up RAM for all of VN).
"""
import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 168   # look back 7 days (168 hours)
PREDICT_STEPS   = 168   # forecast the next 7 days (168 hours) at once


# ── Coordinate array builder ──────────────────────────────────────────────────────

def build_coordinate_arrays(df, feature_cols: list):
    """
    Return:
      arrays : list of float32 arrays, one per (lat, lon), sorted by coordinate.
      coords : parallel list of (lat, lon) tuples (used by the spatial error map).
    """
    arrays, coords = [], []
    for (lat, lon), grp in df.groupby(['latitude', 'longitude'], sort=True):
        arrays.append(grp[feature_cols].values.astype(np.float32))
        coords.append((float(lat), float(lon)))
    return arrays, coords


# ── Temporal split (per coordinate, no cross-boundary contamination) ─────────────

def build_split_indices(
    coordinate_arrays: list,
    seq_len: int       = SEQUENCE_LENGTH,
    predict_steps: int = PREDICT_STEPS,
    train_ratio: float = 0.8,
):
    """
    Split each coordinate's time axis at 80%.
      Train samples : the whole window [start, start+seq_len+predict_steps) < split.
      Test  samples : windows that start at/after split → no leakage from training.
    A sample needs seq_len + predict_steps consecutive hours to be valid.
    """
    span = seq_len + predict_steps
    train_idx, test_idx = [], []

    for cid, arr in enumerate(coordinate_arrays):
        n     = len(arr)
        split = int(n * train_ratio)

        # train: last index touched = start + span - 1 < split
        for start in range(0, max(0, split - span + 1)):
            train_idx.append((cid, start))

        # test: window starts at/after split, still fits inside n
        for start in range(split, max(split, n - span + 1)):
            test_idx.append((cid, start))

    return train_idx, test_idx


# ── Dataset ────────────────────────────────────────────────────────────────────

class WeatherSequenceDataset(Dataset):
    """
    Lazy multi-step sequence dataset. Each __getitem__ slices one (X, y) pair on
    the fly — no pre-materialised 3-D sequence tensor in RAM.

        X : (seq_len, n_features)
        y : (predict_steps,)        future rainfall (log1p + MinMax scaled)
    """

    def __init__(
        self,
        coordinate_arrays: list,
        indices: list,
        target_idx: int,
        seq_len: int       = SEQUENCE_LENGTH,
        predict_steps: int = PREDICT_STEPS,
    ):
        self.coordinate_arrays = coordinate_arrays
        self.indices           = indices
        self.target_idx        = target_idx
        self.seq_len           = seq_len
        self.predict_steps     = predict_steps

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        cid, start = self.indices[idx]
        arr = self.coordinate_arrays[cid]

        x_end = start + self.seq_len
        y_end = x_end + self.predict_steps

        # .copy() → independent buffers so DataLoader workers stay safe.
        x = arr[start:x_end].copy()
        y = arr[x_end:y_end, self.target_idx].copy()

        return torch.from_numpy(x), torch.from_numpy(y)
