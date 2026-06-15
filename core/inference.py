"""
core/inference.py — Nạp mô hình và suy luận LSTM nhiệt độ.
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.model    import LSTMModel
from core.features import SEQUENCE_LENGTH, TARGET_COL, _build_prediction_features

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CKPT_PATH  = os.path.join(MODELS_DIR, "lstm_weather_model_temp.pt")


def load_model():
    """
    Nạp checkpoint + scaler + feature_cols từ thư mục models/.
    Trả về (model, scaler, feat_cols, device).
    Tách riêng để app.py bọc @st.cache_resource.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nạp mô hình lên: {device}", flush=True)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {CKPT_PATH}\n"
            "Đặt lstm_weather_model_temp.pt vào thư mục models/."
        )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    scaler_path = os.path.join(MODELS_DIR, ckpt.get("scaler_name",       "scaler_temp.pkl"))
    feat_path   = os.path.join(MODELS_DIR, ckpt.get("feature_cols_name", "feature_cols_temp.pkl"))

    for p in (scaler_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy: {p}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)

    model = LSTMModel(
        input_size  = ckpt.get("input_size",  len(feat_cols)),
        hidden_size = ckpt.get("hidden_size", 64),
        num_layers  = ckpt.get("num_layers",  2),
        output_size = ckpt.get("output_size", 1),
        dropout     = ckpt.get("dropout",     0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Mô hình sẵn sàng — {sum(p.numel() for p in model.parameters()):,} params", flush=True)
    return model, scaler, feat_cols, device


def _predict_core(
    lat:       float,
    lon:       float,
    model:     nn.Module,
    scaler,
    feat_cols: list,
    device:    torch.device,
    df:        pd.DataFrame,
) -> dict:
    """
    Suy luận nhiệt độ T+1 cho tọa độ (lat, lon).

    Trả về:
      predicted_temp  float  — °C
      predict_time    str    — 'YYYY-MM-DD HH:00'
      coords          tuple  — (lat, lon) lưới thực tế
      history_temps   list   — 24 giá trị °C thực tế
      history_times   list   — 24 nhãn thời gian
    """
    # 1. Tìm tọa độ lưới gần nhất
    coords_df = df[["latitude", "longitude"]].drop_duplicates()
    dists     = np.sqrt(
        (coords_df["latitude"]  - lat) ** 2 +
        (coords_df["longitude"] - lon) ** 2
    )
    nearest  = coords_df.loc[dists.idxmin()]
    near_lat = float(nearest["latitude"])
    near_lon = float(nearest["longitude"])

    # 2. Lọc + lấy buffer cho lag6/roll6
    BUFFER = 30
    loc_df = (
        df[(df["latitude"] == near_lat) & (df["longitude"] == near_lon)]
        .sort_values("valid_time")
        .tail(SEQUENCE_LENGTH + BUFFER)
        .reset_index(drop=True)
    )

    if len(loc_df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Không đủ dữ liệu ({near_lat}, {near_lon}): "
            f"cần {SEQUENCE_LENGTH}, có {len(loc_df)}."
        )

    # 3. Tái tạo đặc trưng
    loc_feat = _build_prediction_features(loc_df)
    if len(loc_feat) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Sau feature engineering còn {len(loc_feat)} hàng "
            f"(cần {SEQUENCE_LENGTH}). Tăng buffer."
        )

    # 4. 24 hàng cuối → lịch sử hiển thị
    seq_df        = loc_feat.tail(SEQUENCE_LENGTH).reset_index(drop=True)
    history_temps = seq_df[TARGET_COL].values.tolist()
    history_times = [
        pd.Timestamp(t).strftime("%Y-%m-%d %H:00")
        for t in seq_df["valid_time"].tolist()
    ]

    # 5. Scale → tensor (1, 24, feat)
    try:
        seq_values = seq_df[feat_cols].values.astype(np.float32)
    except KeyError:
        missing = [c for c in feat_cols if c not in seq_df.columns]
        raise KeyError(f"Thiếu cột: {missing}")

    seq_scaled  = scaler.transform(seq_values)
    x           = torch.from_numpy(seq_scaled).unsqueeze(0).to(device)

    # 6. Suy luận
    with torch.no_grad():
        pred_scaled = float(model(x).cpu().numpy().flat[0])

    # 7. Nghịch đảo MinMaxScaler về °C
    temp_idx            = list(feat_cols).index(TARGET_COL)
    dummy               = np.zeros((1, len(feat_cols)), dtype=np.float32)
    dummy[0, temp_idx]  = pred_scaled
    predicted_temp      = float(scaler.inverse_transform(dummy)[0, temp_idx])

    # 8. Thời điểm T+1
    last_time    = pd.Timestamp(seq_df["valid_time"].iloc[-1])
    predict_time = (last_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:00")

    return {
        "predicted_temp": predicted_temp,
        "predict_time":   predict_time,
        "coords":         (near_lat, near_lon),
        "history_temps":  history_temps,
        "history_times":  history_times,
    }
