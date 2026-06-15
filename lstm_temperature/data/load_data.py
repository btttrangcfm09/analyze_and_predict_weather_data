"""
Task: Load raw weather Parquet dataset from Kaggle, filter to the 34 target
      provinces, and return a sorted DataFrame.
"""
import pandas as pd
import numpy as np
import os

DATA_PATH = '/kaggle/input/datasets/nguyentranggggg/vietnam-meteorological-weather-data-parquet/weather.parquet'

_COLS = [
    'latitude', 'longitude', 'valid_time',
    'temperature_celsius', 'apparent_temperature',
    'relative_humidity', 'wind_speed', 'wind_direction',
    'total_precipitation', 'total_cloud_cover',
    'mean_sea_level_pressure', 'surface_pressure',
    'sea_surface_temperature', 'air_density',
]

# 34 provinces spanning North → South Vietnam
# PROVINCE_COORDS: list[tuple[float, float]] = [
#     # Miền Bắc & Đồng bằng sông Hồng (11)
#     (21.00, 105.75),  # Hà Nội
#     (20.75, 106.75),  # Hải Phòng
#     (21.00, 107.25),  # Quảng Ninh  (Hạ Long)
#     (21.75, 106.75),  # Lạng Sơn
#     (22.75, 106.25),  # Cao Bằng
#     (22.50, 104.00),  # Lào Cai
#     (21.50, 103.00),  # Điện Biên
#     (21.25, 103.75),  # Sơn La
#     (21.50, 105.75),  # Thái Nguyên
#     (21.25, 105.25),  # Phú Thọ
#     (20.25, 105.75),  # Ninh Bình
#     # Bắc Trung Bộ (6)
#     (20.00, 105.75),  # Thanh Hóa
#     (18.75, 105.75),  # Nghệ An  (Vinh)
#     (18.25, 105.75),  # Hà Tĩnh
#     (17.50, 106.50),  # Quảng Bình (Đồng Hới)
#     (16.75, 107.00),  # Quảng Trị  (Đông Hà)
#     (16.50, 107.50),  # Thừa Thiên Huế
#     # Nam Trung Bộ & Tây Nguyên (11)
#     (16.00, 108.25),  # Đà Nẵng
#     (15.50, 108.50),  # Quảng Nam  (Tam Kỳ)
#     (15.00, 108.75),  # Quảng Ngãi
#     (13.75, 109.25),  # Bình Định  (Quy Nhơn)
#     (13.00, 109.25),  # Phú Yên    (Tuy Hòa)
#     (12.25, 109.25),  # Khánh Hòa  (Nha Trang)
#     (14.25, 108.00),  # Kon Tum
#     (14.00, 108.00),  # Gia Lai    (Pleiku)
#     (12.50, 108.00),  # Đắk Lắk   (Buôn Ma Thuột)
#     (12.00, 108.50),  # Lâm Đồng  (Đà Lạt)
#     (11.00, 108.00),  # Bình Thuận (Phan Thiết)
#     # Nam Bộ & Miền Tây (6)
#     (11.00, 106.75),  # Đồng Nai   (Biên Hòa)
#     (10.75, 106.75),  # TP. Hồ Chí Minh
#     (10.50, 107.25),  # Bà Rịa – Vũng Tàu
#     (10.00, 105.75),  # Cần Thơ
#     (10.25, 105.25),  # An Giang   (Long Xuyên)
#     ( 9.25, 105.00),  # Cà Mau
# ]
PROVINCES_COORDS = [
    (21.0, 105.75), (20.75, 106.75), (21.0, 107.25), (21.75, 106.75),
    (22.75, 106.25), (22.5, 104.0), (21.5, 103.0), (21.25, 103.75),
    (21.5, 105.75), (21.25, 105.25), (20.25, 105.75), (20.0, 105.75),
    (18.75, 105.75), (18.25, 105.75), (17.5, 106.5), (16.75, 107.0),
    (16.5, 107.5), (16.0, 108.25), (15.5, 108.5), (15.0, 108.75),
    (13.75, 109.25), (13.0, 109.25), (12.25, 109.25), (14.25, 108.0),
    (14.0, 108.0), (12.5, 108.0), (12.0, 108.5), (11.0, 108.0),
    (11.0, 106.75), (10.75, 106.75), (10.5, 107.25), (10.0, 105.75),
    (10.25, 105.25), (9.25, 105.0)
]

# Build a set for O(1) membership test
# _COORD_SET: set[tuple[float, float]] = set(PROVINCES_COORDS)


def load_data() -> pd.DataFrame:
    # 1. TỰ ĐỘNG DÒ ĐƯỜNG DẪN FILE/THƯ MỤC PARQUET TRONG KAGLLE
    data_path = 'D:/analyze_and_predict_weather_data/data/archive.zip/weather.parquet'  # Đường dẫn mặc định nếu chạy local
    # for root, dirs, files in os.walk("/kaggle/input"):
    #     # Kiểm tra nếu tên thư mục kết thúc bằng .parquet (Thư mục phân mảnh)
    #     if root.endswith(".parquet") and "weather" in root.lower():
    #         data_path = root
    #         break
    #     # Kiểm tra nếu là file đơn lẻ
    #     for file in files:
    #         if file.endswith(".parquet") and "weather" in file.lower():
    #             data_path = os.path.join(root, file)
    #             break
    #     if data_path:
    #         break

    if not data_path:
        raise FileNotFoundError("Không tìm thấy file hoặc thư mục Parquet nào chứa từ khóa 'weather' trong /kaggle/input!")

    print(f"Đang đọc dữ liệu thời tiết tự động từ đường dẫn: {data_path}")
    
    # Tách riêng danh sách vĩ độ và kinh độ để ép PyArrow chỉ đọc đúng tọa độ cần thiết từ đĩa
    unique_lats = list(set([coord[0] for coord in PROVINCES_COORDS]))
    unique_lons = list(set([coord[1] for coord in PROVINCES_COORDS]))

    print(f"Đang nạp dữ liệu thông minh trực tiếp từ đĩa cứng (RAM tiêu thụ < 200MB)...")
    df = pd.read_parquet(
        data_path,
        columns=_COLS,
        engine='auto',
        filters=[
            ('latitude', 'in', unique_lats),
            ('longitude', 'in', unique_lons)
        ]
    )

    # Drop partition columns nếu pyarrow tự nạp thêm
    df = df[[c for c in _COLS if c in df.columns]]

    # Lọc chính xác lại theo các cặp tọa độ (vĩ độ, kinh độ) của 34 tỉnh thành
    df.set_index(['latitude', 'longitude'], inplace=True)
    df = df.loc[df.index.isin(PROVINCES_COORDS)].reset_index()

    # Sắp xếp chuỗi thời gian liên tục
    df = df.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

    # Ép kiểu float64 → float32 sớm để tiết kiệm thêm một nửa RAM
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')

    print(f"Nạp dữ liệu thành công! Tổng số dòng sau khi lọc: {len(df):,}")
    return df
