"""
core/future_features.py — Sinh đặc trưng decoder cho 168 giờ TƯƠNG LAI.

Decoder chỉ cần các đặc trưng xác định trước: 8 cyclic time + year_normalized
+ monthly_mean_temperature (tra climatology theo tháng của điểm lưới).

Để khớp scale lúc huấn luyện, ta dựng ma trận (168, n_feat) rồi scaler.transform
toàn bộ 30 cột (MinMax theo từng cột độc lập), sau đó rút đúng cột decoder.
"""
import numpy as np
import pandas as pd

from core.features import DECODER_FEATURE_COLS


def build_future_index(last_time: pd.Timestamp, steps: int) -> pd.DatetimeIndex:
    """168 mốc giờ kế tiếp sau giờ quan sát cuối cùng."""
    start = pd.Timestamp(last_time) + pd.Timedelta(hours=1)
    return pd.date_range(start=start, periods=steps, freq="h")


def _raw_decoder_values(times: pd.DatetimeIndex, monthly_clim: pd.Series) -> dict:
    """Giá trị THÔ (chưa scale) cho 10 đặc trưng decoder theo từng giờ tương lai."""
    return {
        "hour_sin":        np.sin(2 * np.pi * times.hour      / 24 ),
        "hour_cos":        np.cos(2 * np.pi * times.hour      / 24 ),
        "day_of_week_sin": np.sin(2 * np.pi * times.dayofweek / 7  ),
        "day_of_week_cos": np.cos(2 * np.pi * times.dayofweek / 7  ),
        "month_sin":       np.sin(2 * np.pi * times.month     / 12 ),
        "month_cos":       np.cos(2 * np.pi * times.month     / 12 ),
        "day_of_year_sin": np.sin(2 * np.pi * times.dayofyear / 365),
        "day_of_year_cos": np.cos(2 * np.pi * times.dayofyear / 365),
        "year_normalized": (times.year - 2020) / 6,
        # Tháng tương lai luôn có trong lịch sử nhiều năm; fallback mean phòng khuyết.
        "monthly_mean_temperature": (
            monthly_clim.reindex(times.month).fillna(monthly_clim.mean()).to_numpy()
        ),
    }


def build_decoder_features(
    times: pd.DatetimeIndex,
    monthly_clim: pd.Series,
    feature_cols: list,
    scaler,
) -> np.ndarray:
    """Trả y_feat đã scale, shape (steps, len(DECODER_FEATURE_COLS))."""
    raw = _raw_decoder_values(times, monthly_clim)

    full = np.zeros((len(times), len(feature_cols)), dtype=np.float32)
    for name, values in raw.items():
        full[:, feature_cols.index(name)] = values
    scaled = scaler.transform(full)

    dec_idx = [feature_cols.index(c) for c in DECODER_FEATURE_COLS]
    return scaled[:, dec_idx].astype(np.float32)
