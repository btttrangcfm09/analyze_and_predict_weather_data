"""
Task: Convert the scaled DataFrame into per-province numpy arrays and build
      lazy-loading sequence indices split temporally (80% train / 20% test)
      with zero geographic boundary overlap.

Memory strategy: store province arrays once; each sequence is a (prov_idx, start)
pointer — no pre-materialised copy of all sequences.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 24   # hours of context → predict T+1


# ── Province array builder ─────────────────────────────────────────────────────

def build_province_arrays(df, feature_cols: list) -> list:
    """Return a list of float32 arrays, one per (lat, lon) province, sorted."""
    arrays = []
    for _, grp in df.groupby(['latitude', 'longitude'], sort=True):
        arr = grp[feature_cols].values.astype(np.float32)
        arrays.append(arr)
    return arrays


# ── Temporal split (per province, no cross-boundary contamination) ─────────────

def build_split_indices(
    province_arrays: list,
    seq_len: int = SEQUENCE_LENGTH,
    train_ratio: float = 0.8,
):
    """
    For each province split at 80% of its time axis.
    Train sequences: x and y both fall inside [0, split).
    Test  sequences: x window starts at or after split → zero leakage.
    """
    train_idx, test_idx = [], []

    for prov_id, arr in enumerate(province_arrays):
        n     = len(arr)
        split = int(n * train_ratio)

        # last valid train start: start + seq_len < split  →  start < split - seq_len
        for i in range(split - seq_len):
            train_idx.append((prov_id, i))

        # first valid test start: x = arr[split : split+seq_len], y = arr[split+seq_len]
        for i in range(split, n - seq_len):
            test_idx.append((prov_id, i))

    return train_idx, test_idx


# ── Dataset ────────────────────────────────────────────────────────────────────

class WeatherSequenceDataset(Dataset):
    """
    Lazy sequence dataset: fetches one (X, y) pair per __getitem__ call
    without pre-materialising the entire sequence tensor in RAM.
    """

    def __init__(
        self,
        province_arrays: list,
        indices: list,
        target_idx: int,
        seq_len: int = SEQUENCE_LENGTH,
    ):
        self.province_arrays = province_arrays
        self.indices         = indices
        self.target_idx      = target_idx
        self.seq_len         = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        prov_id, start = self.indices[idx]
        arr = self.province_arrays[prov_id]

        # .copy() required so DataLoader workers get independent buffers
        x = arr[start : start + self.seq_len].copy()
        y = float(arr[start + self.seq_len, self.target_idx])

        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
        )
