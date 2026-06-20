"""
Task: Evaluate the trained Seq2Seq model on the held-out test set and report
      errors in REAL temperature units (°C).

Inference is AUTOREGRESSIVE (teacher_forcing=False): the decoder feeds its own
previous prediction back at each of the 168 future steps — the honest
deployment-time behaviour, unlike the teacher-forced monitor used in training.

The target was trained as MinMax-scaled temperature in [0, 1]. Because there is
NO log transform for temperature, inversion is a single step:
    °C = scaled * data_range_ + data_min_     (for the target column only)
Both predictions and ground truth are inverted before computing RMSE / MAE.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.sequence_dataset import WeatherSequenceDataset, DECODER_FEATURE_COLS


# ── Target inverse transform (scaled  →  °C) ─────────────────────────────────────

def inverse_target(scaled, scaler, target_idx: int) -> np.ndarray:
    """Invert MinMax scaling for the temperature target column (no log step)."""
    scaled = np.asarray(scaled, dtype=np.float64)
    return scaled * scaler.data_range_[target_idx] + scaler.data_min_[target_idx]


def _inverse_col(scaled, scaler, col_idx: int) -> np.ndarray:
    """Invert MinMax scaling for an arbitrary feature column."""
    scaled = np.asarray(scaled, dtype=np.float64)
    return scaled * scaler.data_range_[col_idx] + scaler.data_min_[col_idx]


# ── Test-set metrics (streaming, in °C) ──────────────────────────────────────────

def evaluate_test(model, test_dataset, device, scaler, target_idx: int,
                  batch_size: int = 2048) -> tuple:
    loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    model.eval()
    sse = sae = count = 0.0

    with torch.no_grad():
        for X_b, yf_b, y_b in loader:
            X_b  = X_b.to(device, non_blocking=True)
            yf_b = yf_b.to(device, non_blocking=True)
            out  = model(X_b, yf_b, teacher_forcing=False).cpu().numpy()

            preds_c   = inverse_target(out,         scaler, target_idx)
            targets_c = inverse_target(y_b.numpy(), scaler, target_idx)

            diff   = preds_c - targets_c
            sse   += float(np.sum(diff ** 2))
            sae   += float(np.sum(np.abs(diff)))
            count += diff.size

    mse  = sse / count
    rmse = float(np.sqrt(mse))
    mae  = sae / count

    w = 56
    print()
    print("=" * w)
    print(f"{'TEST SET EVALUATION  (real units, °C)':^{w}}")
    print("=" * w)
    print(f"  {'Metric':<26} {'Value':>16}")
    print("-" * w)
    print(f"  {'Test MSE  (°C^2)':<26} {mse:>16.6f}")
    print(f"  {'Test RMSE (°C)':<26} {rmse:>16.6f}")
    print(f"  {'Test MAE  (°C)':<26} {mae:>16.6f}")
    print("=" * w)

    return mse, rmse, mae


# ── Prediction collector for visualisation (subsampled, in °C) ───────────────────

def collect_predictions(model, coordinate_arrays, indices, target_idx, decoder_feat_idx,
                        feature_cols, scaler, seq_len, predict_steps, device,
                        max_samples: int = 4000, batch_size: int = 2048):
    """
    Run AUTOREGRESSIVE inference over a random subset of `indices` and return
    real-unit arrays for plotting:
        preds_c       : (M, predict_steps)   °C
        targets_c     : (M, predict_steps)   °C
        coord_ids     : (M,)                 coordinate index of each sample
        hours_of_day  : (M, predict_steps)   clock hour 0..23 of each forecast step
    Subsampled to keep the plotting buffers small while covering many regions.

    `hours_of_day` is recovered from the (scaled) hour_sin/hour_cos columns in
    y_feat — inverse-scaled, then hour = atan2(sin, cos) / 2π * 24.
    """
    rng = np.random.default_rng(42)
    if len(indices) > max_samples:
        pick = rng.choice(len(indices), size=max_samples, replace=False)
        pick.sort()
        sub_indices = [indices[i] for i in pick]
    else:
        sub_indices = list(indices)

    ds = WeatherSequenceDataset(coordinate_arrays, sub_indices, target_idx,
                                decoder_feat_idx, seq_len, predict_steps)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=(device.type == 'cuda'))

    # Locate hour_sin/hour_cos: position inside y_feat AND inside feature_cols.
    hs_feat = feature_cols.index('hour_sin')
    hc_feat = feature_cols.index('hour_cos')
    hs_col  = list(decoder_feat_idx).index(hs_feat)
    hc_col  = list(decoder_feat_idx).index(hc_feat)

    model.eval()
    preds_buf, tgts_buf, hour_buf = [], [], []
    with torch.no_grad():
        for X_b, yf_b, y_b in loader:
            X_b  = X_b.to(device, non_blocking=True)
            yf_d = yf_b.to(device, non_blocking=True)
            preds_buf.append(model(X_b, yf_d, teacher_forcing=False).cpu().numpy())
            tgts_buf.append(y_b.numpy())

            yf_np = yf_b.numpy()
            sin   = _inverse_col(yf_np[:, :, hs_col], scaler, hs_feat)
            cos   = _inverse_col(yf_np[:, :, hc_col], scaler, hc_feat)
            hours = (np.round(np.arctan2(sin, cos) / (2 * np.pi) * 24).astype(int)) % 24
            hour_buf.append(hours)

    preds_c      = inverse_target(np.concatenate(preds_buf), scaler, target_idx)
    targets_c    = inverse_target(np.concatenate(tgts_buf),  scaler, target_idx)
    coord_ids    = np.array([cid for cid, _ in sub_indices], dtype=np.int64)
    hours_of_day = np.concatenate(hour_buf)

    return preds_c, targets_c, coord_ids, hours_of_day
