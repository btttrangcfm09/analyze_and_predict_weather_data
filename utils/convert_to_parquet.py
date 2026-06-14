"""
convert_to_parquet.py  —  Bước 1.3: Chuyển Đổi Sang Parquet
=============================================================
- Đọc weather_YYYY.csv + rain_YYYY.csv (đã clean từ Bước 1.2)
- Merge theo (latitude, longitude, time)
- Tính các chỉ số phái sinh (wind_speed, humidity, ...)
- Lưu dạng Parquet phân vùng theo năm/tháng
- Sửa lại app.py để đọc từ Parquet thay vì CSV

Cách chạy: python convert_to_parquet.py
"""

import os
import math
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ETL_DIR    = r"d:\2025.2\analyze_and_predict_weather_data\ETL\code"
PARQUET_DIR = r"d:\2025.2\analyze_and_predict_weather_data\data\weather.parquet"
YEARS      = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]


# Ngưỡng outlier vật lý hợp lệ cho từng cột
PHYSICAL_BOUNDS = {
    "t2m":  (-60 + 273.15, 60 + 273.15),   # Kelvin
    "d2m":  (-80 + 273.15, 40 + 273.15),
    "msl":  (87000, 108000),                 # Pa
    "sp":   (50000, 110000),
    "u10":  (-80, 80),                       # m/s
    "v10":  (-80, 80),
    "tcc":  (0, 1),
    "tp":   (0, 0.5),                        # m/giờ
    "sst":  (270, 320),                      # Kelvin
}


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch DataFrame trên RAM: xử lý outlier → interpolate → mean imputation."""
    print("  Làm sạch data in-memory (không ghi đè CSV gốc)...", flush=True)

    # 1. Xử lý outlier
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            if mask.sum() > 0:
                df.loc[mask, col] = np.nan

    # 2. Linear Interpolation theo thời gian (nhóm theo lat/lon)
    time_col = "time"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print("    Sắp xếp dữ liệu...", flush=True)
    df = df.sort_values(["latitude", "longitude", time_col]).reset_index(drop=True)
    
    print("    Nội suy dữ liệu (tiết kiệm RAM)...", flush=True)
    # Dùng transform thay vì apply để tránh tràn RAM và tăng tốc x10
    df[num_cols] = df.groupby(["latitude", "longitude"])[num_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # 3. Mean Imputation cho các điểm còn sót (ví dụ SST trên đất liền)
    remaining = df[num_cols].isnull().sum()
    still_nan = remaining[remaining > 0]
    for col in still_nan.index:
        df[col] = df[col].fillna(df[col].mean())

    return df


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Tính các cột phái sinh giống hệt logic trong app.py."""
    # Đổi tên cột chuẩn (nếu cần)
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "valid_time"})

    # Đơn vị: Kelvin → Celsius
    df["temperature_celsius"]           = df["t2m"] - 273.15
    df["d2m_celsius"]                   = df["d2m"] - 273.15

    # Lượng mưa: m → mm
    df["total_precipitation"]           = df["tp"] * 1000

    # Tốc độ và hướng gió
    df["wind_speed"]                    = np.sqrt(df["u10"]**2 + df["v10"]**2)
    df["wind_direction"]                = (180 + (180 / math.pi) *
                                           np.arctan2(df["u10"], df["v10"])) % 360

    # Độ ẩm tương đối (Magnus formula)
    df["saturation_vapor_pressure"]     = 6.11 * (10 ** (7.5 * df["temperature_celsius"] /
                                                          (df["temperature_celsius"] + 237.3)))
    df["vapor_pressure"]                = 6.11 * (10 ** (7.5 * df["d2m_celsius"] /
                                                          (df["d2m_celsius"] + 237.3)))
    df["relative_humidity"]             = (df["vapor_pressure"] /
                                           df["saturation_vapor_pressure"]) * 100

    # Nhiệt độ cảm nhận (Australian BoM)
    df["apparent_temperature"]          = (df["temperature_celsius"]
                                           + 0.33 * df["vapor_pressure"]
                                           - 0.70 * df["wind_speed"]
                                           - 4.00)

    # Mật độ không khí (Ideal Gas Law)
    df["air_density"]                   = df["sp"] / (287.05 * df["t2m"])  # t2m vẫn Kelvin

    # Đổi tên cột áp suất, mây, SST
    df = df.rename(columns={
        "msl": "mean_sea_level_pressure",
        "sp":  "surface_pressure",
        "sst": "sea_surface_temperature",
        "tcc": "total_cloud_cover",
    })

    # Timestamp
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["year"]       = df["valid_time"].dt.year
    df["month"]      = df["valid_time"].dt.month

    # Loại bỏ cột trung gian không cần thiết
    drop_cols = ["t2m", "d2m", "u10", "v10", "tp",
                 "saturation_vapor_pressure", "vapor_pressure",
                 "d2m_celsius", "tciw", "tclw"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def process_year(year: str):
    w_path = os.path.join(ETL_DIR, f"weather_{year}.csv")
    r_path = os.path.join(ETL_DIR, f"rain_{year}.csv")

    if not os.path.exists(w_path) or not os.path.exists(r_path):
        print(f"  ⚠️  Thiếu file cho năm {year} — bỏ qua")
        return

    print(f"  Đang xử lý {year} theo từng phần nhỏ (Chunking) để không treo máy...", flush=True)

    chunk_size = 1_000_000
    w_iter = pd.read_csv(w_path, chunksize=chunk_size, low_memory=False)
    r_iter = pd.read_csv(r_path, chunksize=chunk_size, low_memory=False)

    chunk_idx = 1
    for df_w, df_r in zip(w_iter, r_iter):
        print(f"    -> Phần {chunk_idx}...", flush=True)
        # Chuẩn hóa cột thời gian (xử lý lỗi trùng lặp cột 'time')
        for df in [df_w, df_r]:
            if "time" in df.columns and "valid_time" in df.columns:
                df.drop(columns=["valid_time"], inplace=True)
            elif "valid_time" in df.columns:
                df.rename(columns={"valid_time": "time"}, inplace=True)

        df = pd.merge(df_w, df_r[["latitude", "longitude", "time", "tp"]],
                      on=["latitude", "longitude", "time"],
                      how="inner")
        df = df.drop_duplicates(subset=["latitude", "longitude", "time"])

        del df_w
        del df_r

        # Làm sạch và tính toán
        df = clean_df(df)
        df = compute_derived(df)

        # Lưu từng tháng thành các file Parquet riêng biệt trong partition
        for month, df_month in df.groupby("month"):
            out_dir = os.path.join(PARQUET_DIR, f"year={year}", f"month={month:02d}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"data_chunk_{chunk_idx}.parquet")
            df_month.drop(columns=["year", "month"]).to_parquet(
                out_path, index=False, compression="snappy"
            )

        chunk_idx += 1

    print(f"  ✅ Đã xử lý xong năm {year} ({chunk_idx - 1} phần)")


def main():
    print("=" * 55)
    print("  BƯỚC 1.3 — CHUYỂN ĐỔI SANG PARQUET (Siêu nhẹ RAM)")
    print("=" * 55)

    os.makedirs(PARQUET_DIR, exist_ok=True)

    for year in YEARS:
        print(f"\n[{year}] Đang bắt đầu...", flush=True)
        process_year(year)

    # Tính dung lượng thư mục Parquet
    total_size = sum(
        f.stat().st_size for f in
        __import__("pathlib").Path(PARQUET_DIR).rglob("*.parquet")
    ) / 1e9
    print(f"\n\n🎉 HOÀN TẤT Bước 1.3!")
    print(f"   Thư mục Parquet: {PARQUET_DIR}")
    print(f"   Tổng dung lượng: {total_size:.2f} GB")
    print(f"\n   👉 Bước tiếp theo: chạy 'python update_app.py' để app đọc Parquet")


if __name__ == "__main__":
    main()