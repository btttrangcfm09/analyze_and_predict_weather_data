"""
Task: Evaluate the trained model on the held-out test set and report errors in
      REAL rainfall units (mm/hour).

The target was trained as log1p(precip) then MinMax-scaled to [0, 1]. To get back
to mm/hour we invert in two steps:
    1. undo MinMax for the target column : x = scaled * data_range_ + data_min_
    2. undo the log                       : mm = np.expm1(x)
Both predictions and ground truth are inverted before computing RMSE / MAE.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dataset.sequence_dataset import WeatherSequenceDataset


# ── Target inverse transform (scaled-log  →  mm/hour) ────────────────────────────

def inverse_target(scaled, scaler, target_idx: int) -> np.ndarray:
    """Invert MinMax scaling then log1p for the rainfall target column."""
    scaled = np.asarray(scaled, dtype=np.float64)
    log_precip = scaled * scaler.data_range_[target_idx] + scaler.data_min_[target_idx]
    return np.expm1(log_precip)


# ── Test-set metrics (streaming, in mm/hour) ─────────────────────────────────────

def evaluate_test(model, test_dataset, device, scaler, target_idx: int,
                  batch_size: int = 2048) -> tuple:
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    model.eval()
    sse = sae = count = 0.0

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b  = X_b.to(device, non_blocking=True)
            out  = model(X_b).cpu().numpy()

            preds_mm   = inverse_target(out,          scaler, target_idx)
            targets_mm = inverse_target(y_b.numpy(),  scaler, target_idx)

            diff   = preds_mm - targets_mm
            sse   += float(np.sum(diff ** 2))
            sae   += float(np.sum(np.abs(diff)))
            count += diff.size

    mse  = sse / count
    rmse = float(np.sqrt(mse))
    mae  = sae / count

    w = 56
    print()
    print("=" * w)
    print(f"{'TEST SET EVALUATION  (real units, mm/hour)':^{w}}")
    print("=" * w)
    print(f"  {'Metric':<26} {'Value':>16}")
    print("-" * w)
    print(f"  {'Test MSE  (mm^2/h)':<26} {mse:>16.6f}")
    print(f"  {'Test RMSE (mm/hour)':<26} {rmse:>16.6f}")
    print(f"  {'Test MAE  (mm/hour)':<26} {mae:>16.6f}")
    print("=" * w)

    return mse, rmse, mae


# ── Prediction collector for visualisation (subsampled, in mm/hour) ──────────────

def collect_predictions(model, coordinate_arrays, indices, target_idx, scaler,
                        seq_len, predict_steps, device,
                        max_samples: int = 4000, batch_size: int = 2048):
    """
    Run inference over a random subset of `indices` and return real-unit arrays
    for plotting:
        preds_mm   : (M, predict_steps)
        targets_mm : (M, predict_steps)
        coord_ids  : (M,)  coordinate index of each sample
    Subsampled to keep the plotting buffers small while covering many regions.
    """
    rng = np.random.default_rng(42)
    if len(indices) > max_samples:
        pick = rng.choice(len(indices), size=max_samples, replace=False)
        pick.sort()
        sub_indices = [indices[i] for i in pick]
    else:
        sub_indices = list(indices)

    ds = WeatherSequenceDataset(coordinate_arrays, sub_indices, target_idx,
                                seq_len, predict_steps)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == 'cuda'))

    model.eval()
    preds_buf, tgts_buf = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            preds_buf.append(model(X_b).cpu().numpy())
            tgts_buf.append(y_b.numpy())

    preds_mm   = inverse_target(np.concatenate(preds_buf), scaler, target_idx)
    targets_mm = inverse_target(np.concatenate(tgts_buf),  scaler, target_idx)
    coord_ids  = np.array([cid for cid, _ in sub_indices], dtype=np.int64)

    return preds_mm, targets_mm, coord_ids
