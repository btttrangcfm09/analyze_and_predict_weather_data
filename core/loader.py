"""
core/loader.py — Nạp lịch sử thời tiết cho lưới ĐẤT LIỀN Việt Nam (500+ điểm).

Ưu tiên:
  1. data/weather_land_VNM.parquet   — cache lưới đất liền (dựng 1 lần từ zip)
  2. data/archive.zip                — stream trong RAM, lọc theo GADM, lưu cache
  3. Raise FileNotFoundError

Lọc đất liền: đọc unique toạ độ từ chunk đầu → polygon GADM L0 (buffer 0.05)
→ tập land_set. Mọi chunk lọc theo land_set (nhanh). Đồng thời sinh cache lưới
điểm-có-tỉnh (GADM L1) cho sidebar.
"""
import io
import os
import zipfile

import pandas as pd

from core.geo import BASE_DIR, filter_land_coords, load_vietnam_land_geometry

ARCHIVE_ZIP = os.path.join(BASE_DIR, "data", "archive.zip")
LAND_CACHE  = os.path.join(BASE_DIR, "data", "weather_land_VNM.parquet")

_NEEDED_COLS = [
    "latitude", "longitude", "valid_time",
    "temperature_celsius", "apparent_temperature",
    "relative_humidity", "wind_speed", "wind_direction",
    "total_precipitation", "total_cloud_cover",
    "mean_sea_level_pressure", "surface_pressure",
    "sea_surface_temperature", "air_density",
]


def _land_set(df_chunk: pd.DataFrame) -> set:
    """Tập (lat, lon) đất liền, suy ra 1 lần từ unique toạ độ của 1 chunk."""
    uniq = df_chunk[["latitude", "longitude"]].drop_duplicates()
    land = filter_land_coords(uniq, load_vietnam_land_geometry())
    return set(map(tuple, land[["latitude", "longitude"]].to_numpy()))


def _filter(df: pd.DataFrame, land: set) -> pd.DataFrame:
    keys = pd.MultiIndex.from_arrays([df["latitude"].to_numpy(), df["longitude"].to_numpy()])
    return df[keys.isin(list(land))]


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cột lịch GỌN NHẸ cho các trang phân tích: date (datetime64) + 3 categorical.

    Categorical thay cho object string → tiết kiệm RAM cực lớn trên ~23 triệu dòng,
    tránh MemoryError khi Streamlit pickle cache.
    """
    dt = df["valid_time"].dt
    df["date"]  = dt.floor("D")
    df["year"]  = pd.Categorical(dt.year.astype(str))
    df["month"] = pd.Categorical(dt.year.astype(str) + "-" + dt.month.astype(str).str.zfill(2))
    df["hour"]  = pd.Categorical(dt.hour.astype(str).str.zfill(2) + ":00")
    return df


def _read_from_zip_and_cache() -> pd.DataFrame:
    """Stream archive.zip trong RAM (không giải nén), lọc đất liền, lưu cache."""
    print("Đọc archive.zip (lần đầu, có thể mất vài phút)...", flush=True)
    frames, land = [], None
    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        files = sorted(
            f for f in zf.namelist()
            if f.endswith(".parquet") and "weather" in f.lower()
        )
        for i, fname in enumerate(files, 1):
            chunk = pd.read_parquet(io.BytesIO(zf.read(fname)), columns=_NEEDED_COLS)
            if land is None:
                land = _land_set(chunk)
                print(f"  {len(land):,} điểm lưới đất liền | {len(files)} file...", flush=True)
            frames.append(_filter(chunk, land))
            if i % 40 == 0 or i == len(files):
                print(f"  {i}/{len(files)} file đã xử lý...", flush=True)

    if not frames:
        raise ValueError("Không tìm thấy dữ liệu đất liền trong archive.zip.")

    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(LAND_CACHE), exist_ok=True)
    df.to_parquet(LAND_CACHE, index=False, compression="snappy")
    print(f"  Cache: {LAND_CACHE} ({os.path.getsize(LAND_CACHE) // 1024 // 1024} MB)", flush=True)

    from core.provinces import load_grid_points
    load_grid_points(df[["latitude", "longitude"]].drop_duplicates())
    return df


def load_history() -> pd.DataFrame:
    """Lịch sử thời tiết lưới đất liền; thêm cột date. Bọc @st.cache_data ở app.py."""
    if os.path.isfile(LAND_CACHE):
        df = pd.read_parquet(LAND_CACHE)
    elif os.path.isfile(ARCHIVE_ZIP):
        df = _read_from_zip_and_cache()
    else:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu.\n  Cache: {LAND_CACHE}\n  Zip: {ARCHIVE_ZIP}\n"
            "Đặt archive.zip vào thư mục data/."
        )

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    return _add_calendar(df)
