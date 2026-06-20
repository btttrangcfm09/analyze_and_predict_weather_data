"""
core/features.py — Hằng số đặc trưng cho mô hình Seq2Seq dự báo tuần (168 giờ).

Khớp với lstm_temp_weekly: SEQUENCE_LENGTH = PREDICT_STEPS = 168, 30 đặc trưng
encoder và 10 đặc trưng decoder (đã-biết-trước) trong DECODER_FEATURE_COLS.
Danh sách Tỉnh/Thành không còn hardcode 34 tỉnh — sinh động từ GADM (core/provinces.py).
"""

SEQUENCE_LENGTH = 168   # nhìn lại 7 ngày
PREDICT_STEPS   = 168   # dự báo 7 ngày tới
TARGET_COL      = "temperature_celsius"

_METEO_COLS = [
    "temperature_celsius", "apparent_temperature",
    "relative_humidity", "wind_speed", "wind_direction",
    "total_precipitation", "total_cloud_cover",
    "mean_sea_level_pressure", "surface_pressure",
    "sea_surface_temperature", "air_density",
]

# Các đặc trưng mà decoder ĐÃ BIẾT TRƯỚC cho cửa sổ tương lai (lịch + khí hậu),
# đúng thứ tự như khi huấn luyện (config['decoder_feature_cols']).
DECODER_FEATURE_COLS = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "year_normalized", "monthly_mean_temperature",
]
