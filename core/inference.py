"""
core/inference.py — Nạp mô hình Seq2Seq và suy luận autoregressive 168 giờ.
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.model               import LSTMModel
from core.features            import SEQUENCE_LENGTH, PREDICT_STEPS, TARGET_COL
from core.feature_engineering import engineer_features
from core.future_features     import build_future_index, build_decoder_features

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "lstm_temp_weekly", "results")
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
HISTORY_HOURS = 72   # số giờ thực tế trả về để vẽ kèm dự báo


def _resolve_ckpt() -> str:
    """Ưu tiên best checkpoint, fallback epoch_05."""
    for name in ("lstm_temp_weekly_model.pt", "lstm_temp_weekly_epoch_05.pt"):
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Không tìm thấy checkpoint trong {MODELS_DIR}.")


def load_model():
    """Nạp checkpoint + scaler + feature_cols. Trả (model, scaler, feat_cols, device)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nạp mô hình lên: {device}", flush=True)

    ckpt = torch.load(_resolve_ckpt(), map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    with open(os.path.join(RESULTS_DIR, cfg["scaler_name"]), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, cfg["feature_cols_name"]), "rb") as f:
        feat_cols = pickle.load(f)

    model = LSTMModel(
        input_size    = cfg["input_size"],
        dec_feat_size = cfg["dec_feat_size"],
        hidden_size   = cfg["hidden_size"],
        num_layers    = cfg["num_layers"],
        predict_steps = cfg["predict_steps"],
        dropout       = cfg["dropout"],
        target_idx    = cfg["target_idx"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Mô hình sẵn sàng — {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    return model, scaler, feat_cols, device


def _nearest_location(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Chuỗi thời gian đầy đủ của điểm lưới gần (lat, lon) nhất."""
    coords = df[["latitude", "longitude"]].drop_duplicates()
    d = (coords["latitude"] - lat) ** 2 + (coords["longitude"] - lon) ** 2
    near = coords.loc[d.idxmin()]
    loc = df[(df["latitude"] == near["latitude"]) & (df["longitude"] == near["longitude"])]
    return loc.sort_values("valid_time").reset_index(drop=True)


def _inverse_temp(scaled: np.ndarray, scaler, target_idx: int) -> np.ndarray:
    """Giải chuẩn hóa MinMax riêng cột nhiệt độ về °C."""
    return scaled * scaler.data_range_[target_idx] + scaler.data_min_[target_idx]


def _predict_core(lat, lon, model: nn.Module, scaler, feat_cols, device, df) -> dict:
    """
    Suy luận autoregressive 168 giờ cho (lat, lon).

    Trả về:
      coords          (lat, lon) lưới thực tế
      history_temps/_times   72 giờ thực tế cuối (°C)
      forecast_temps/_times  168 giờ dự báo (°C)
    """
    loc = _nearest_location(df, lat, lon)
    eng = engineer_features(loc.copy())
    if len(eng) < SEQUENCE_LENGTH:
        raise ValueError(f"Không đủ dữ liệu sau feature engineering: {len(eng)} < {SEQUENCE_LENGTH}.")

    # ── Encoder x: 168 hàng cuối × 30 đặc trưng đã scale ──
    seq = eng.tail(SEQUENCE_LENGTH)
    x_scaled = scaler.transform(seq[feat_cols].to_numpy(np.float32))
    x = torch.from_numpy(x_scaled).unsqueeze(0).to(device)

    # ── Decoder y_feat: 168 giờ tương lai × 10 đặc trưng đã scale ──
    last_time = pd.Timestamp(seq["valid_time"].iloc[-1])
    future_times = build_future_index(last_time, PREDICT_STEPS)
    monthly_clim = loc.groupby(loc["valid_time"].dt.month)[TARGET_COL].mean()
    y_feat = build_decoder_features(future_times, monthly_clim, feat_cols, scaler)
    y_feat_t = torch.from_numpy(y_feat).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(x, y_feat_t, teacher_forcing=False).cpu().numpy().reshape(-1)
    forecast = _inverse_temp(pred_scaled, scaler, feat_cols.index(TARGET_COL))

    hist = eng.tail(HISTORY_HOURS)
    fmt = "%Y-%m-%d %H:00"
    return {
        "coords":         (float(loc["latitude"].iloc[0]), float(loc["longitude"].iloc[0])),
        "history_temps":  hist[TARGET_COL].tolist(),
        "history_times":  [pd.Timestamp(t).strftime(fmt) for t in hist["valid_time"]],
        "forecast_temps": forecast.tolist(),
        "forecast_times": [t.strftime(fmt) for t in future_times],
    }
