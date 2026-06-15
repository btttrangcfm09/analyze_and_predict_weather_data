"""
core/features.py — Hằng số đặc trưng và pipeline tiền xử lý.

34 tỉnh thành khớp đúng với PROVINCES_COORDS trong notebook Kaggle
(lstm_temperature_kaggle.ipynb cell-3).
"""

import numpy as np
import pandas as pd

SEQUENCE_LENGTH = 24
TARGET_COL      = "temperature_celsius"

_METEO_COLS = [
    "temperature_celsius", "apparent_temperature",
    "relative_humidity", "wind_speed", "wind_direction",
    "total_precipitation", "total_cloud_cover",
    "mean_sea_level_pressure", "surface_pressure",
    "sea_surface_temperature", "air_density",
]

# 34 tọa độ khớp hoàn toàn với training (thứ tự giống PROVINCES_COORDS trong notebook)
VIETNAM_PROVINCES: dict[str, tuple[float, float]] = {
    "Hà Nội":                (21.0,   105.75),
    "Hải Phòng":             (20.75,  106.75),
    "Quảng Ninh":            (21.0,   107.25),
    "Bắc Giang":             (21.75,  106.75),
    "Cao Bằng":              (22.75,  106.25),
    "Hà Giang":              (22.5,   104.0),
    "Điện Biên":             (21.5,   103.0),
    "Lai Châu":              (21.25,  103.75),
    "Thái Nguyên":           (21.5,   105.75),
    "Vĩnh Phúc":             (21.25,  105.25),
    "Nam Định":              (20.25,  105.75),
    "Ninh Bình":             (20.0,   105.75),
    "Nghệ An":               (18.75,  105.75),
    "Hà Tĩnh":               (18.25,  105.75),
    "Quảng Trị":             (17.5,   106.5),
    "Thừa Thiên-Huế":        (16.75,  107.0),
    "Đà Nẵng":               (16.5,   107.5),
    "Quảng Nam":             (16.0,   108.25),
    "Quảng Ngãi":            (15.5,   108.5),
    "Bình Định":             (15.0,   108.75),
    "Phú Yên":               (13.75,  109.25),
    "Khánh Hòa (Nha Trang)": (13.0,   109.25),
    "Ninh Thuận":            (12.25,  109.25),
    "Gia Lai":               (14.25,  108.0),
    "Kon Tum":               (14.0,   108.0),
    "Đắk Lắk":              (12.5,   108.0),
    "Đắk Nông":              (12.0,   108.5),
    "Lâm Đồng (Đà Lạt)":    (11.0,   108.0),
    "Bình Dương":            (11.0,   106.75),
    "TP. Hồ Chí Minh":       (10.75,  106.75),
    "Bà Rịa-Vũng Tàu":      (10.5,   107.25),
    "Tiền Giang":            (10.0,   105.75),
    "Vĩnh Long":             (10.25,  105.25),
    "Cà Mau":                (9.25,   105.0),
}

# Set tọa độ để lọc nhanh
PROVINCES_COORDS_SET: frozenset[tuple[float, float]] = frozenset(VIETNAM_PROVINCES.values())


def _build_prediction_features(loc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tái tạo CHÍNH XÁC pipeline tiền xử lý lúc huấn luyện cho 1 tọa độ.

    Khớp với preprocessing/feature_engineering.py trên Kaggle:
      1. Nội suy tuyến tính các cột khí tượng bị khuyết
      2. 8 cyclic time features (sin/cos × hour/dayofweek/month/dayofyear)
      3. 3 lag features: lag1, lag3, lag6
      4. 2 rolling mean (shift(1) tránh leakage): roll3, roll6
      5. Xóa hàng NaN biên lag/rolling

    Lưu ý: Không áp dụng log1p cho total_precipitation — pipeline hourly không dùng.
    """
    loc = loc_df.copy().sort_values("valid_time").reset_index(drop=True)

    existing_meteo = [c for c in _METEO_COLS if c in loc.columns]
    loc[existing_meteo] = (
        loc[existing_meteo].interpolate(method="linear", limit_direction="both")
    )

    dt = loc["valid_time"]
    loc["hour_sin"]        = np.sin(2 * np.pi * dt.dt.hour       / 24 ).astype("float32")
    loc["hour_cos"]        = np.cos(2 * np.pi * dt.dt.hour       / 24 ).astype("float32")
    loc["day_of_week_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek  / 7  ).astype("float32")
    loc["day_of_week_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek  / 7  ).astype("float32")
    loc["month_sin"]       = np.sin(2 * np.pi * dt.dt.month      / 12 ).astype("float32")
    loc["month_cos"]       = np.cos(2 * np.pi * dt.dt.month      / 12 ).astype("float32")
    loc["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear  / 365).astype("float32")
    loc["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear  / 365).astype("float32")

    loc[f"{TARGET_COL}_lag1"] = loc[TARGET_COL].shift(1).astype("float32")
    loc[f"{TARGET_COL}_lag3"] = loc[TARGET_COL].shift(3).astype("float32")
    loc[f"{TARGET_COL}_lag6"] = loc[TARGET_COL].shift(6).astype("float32")

    loc[f"{TARGET_COL}_roll3"] = (
        loc[TARGET_COL].shift(1).rolling(3, min_periods=1).mean().astype("float32")
    )
    loc[f"{TARGET_COL}_roll6"] = (
        loc[TARGET_COL].shift(1).rolling(6, min_periods=1).mean().astype("float32")
    )

    return loc.dropna().reset_index(drop=True)
