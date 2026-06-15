"""
core/loader.py

Priority:
  1. data/weather_34prov.parquet  -- 34-province cache (full 24h), built once from zip
  2. data/archive.zip             -- stream from zip, filter 34 provinces, save cache
  3. data/weather.parquet/        -- local crawled parquet (may be 6h/18h only)
  4. Raise FileNotFoundError
"""

import io
import os
import zipfile

import pandas as pd

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_WEATHER_DIR = os.path.join(BASE_DIR, "data", "weather.parquet")
ARCHIVE_ZIP       = os.path.join(BASE_DIR, "data", "archive.zip")
PROV_CACHE        = os.path.join(BASE_DIR, "data", "weather_34prov.parquet")

_NEEDED_COLS = [
    "latitude", "longitude", "valid_time",
    "temperature_celsius", "apparent_temperature",
    "relative_humidity", "wind_speed", "wind_direction",
    "total_precipitation", "total_cloud_cover",
    "mean_sea_level_pressure", "surface_pressure",
    "sea_surface_temperature", "air_density",
]


def _get_prov_set():
    from core.features import PROVINCES_COORDS_SET
    return PROVINCES_COORDS_SET


def _filter_provinces(df: pd.DataFrame) -> pd.DataFrame:
    prov_set = _get_prov_set()
    mask = pd.Series(
        zip(df["latitude"].values, df["longitude"].values)
    ).isin(prov_set).values
    return df[mask].reset_index(drop=True)


def _read_from_zip_and_cache() -> pd.DataFrame:
    """
    Stream parquet files from archive.zip without extracting to disk.
    Each chunk covers all Vietnam grid points (~986K rows); filter to 34 provinces immediately.
    Saves a small cache file so subsequent calls skip the zip.
    """
    print("Reading archive.zip (first time, may take a few minutes)...", flush=True)
    frames = []
    with zipfile.ZipFile(ARCHIVE_ZIP, "r") as zf:
        parquet_files = sorted(
            f for f in zf.namelist()
            if f.endswith(".parquet") and "weather" in f.lower()
        )
        total = len(parquet_files)
        print(f"  {total} partition files found...", flush=True)

        for i, fname in enumerate(parquet_files, 1):
            raw      = zf.read(fname)
            df_chunk = pd.read_parquet(io.BytesIO(raw), columns=_NEEDED_COLS)
            df_filt  = _filter_provinces(df_chunk)
            if not df_filt.empty:
                frames.append(df_filt)
            if i % 40 == 0 or i == total:
                print(f"  {i}/{total} files processed...", flush=True)

    if not frames:
        raise ValueError("No data for 34 provinces found in archive.zip.")

    df = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    df.to_parquet(PROV_CACHE, index=False, compression="snappy")
    size_mb = os.path.getsize(PROV_CACHE) // 1024 // 1024
    print(f"  Cache saved: {PROV_CACHE} ({size_mb} MB)", flush=True)
    return df


def _read_from_parquet_dir() -> pd.DataFrame:
    from core.features import PROVINCES_COORDS_SET
    unique_lats = list({c[0] for c in PROVINCES_COORDS_SET})
    unique_lons = list({c[1] for c in PROVINCES_COORDS_SET})

    df = pd.read_parquet(
        LOCAL_WEATHER_DIR,
        columns=_NEEDED_COLS,
        filters=[
            ("latitude",  "in", unique_lats),
            ("longitude", "in", unique_lons),
        ],
    )
    return _filter_provinces(df)


def load_history() -> pd.DataFrame:
    """
    Load 34-province hourly weather history.
    Returns DataFrame with valid_time (datetime) and date columns.
    Call this from app.py wrapped in @st.cache_data.
    """
    if os.path.isfile(PROV_CACHE):
        df = pd.read_parquet(PROV_CACHE)
    elif os.path.isfile(ARCHIVE_ZIP):
        df = _read_from_zip_and_cache()
    elif os.path.isdir(LOCAL_WEATHER_DIR):
        print("Warning: local parquet may only have 2 hours/day (6h, 18h).", flush=True)
        df = _read_from_parquet_dir()
    else:
        raise FileNotFoundError(
            "Weather data not found.\n"
            f"  Cache: {PROV_CACHE}\n"
            f"  Zip:   {ARCHIVE_ZIP}\n"
            f"  Dir:   {LOCAL_WEATHER_DIR}\n"
            "Place archive.zip in the data/ folder."
        )

    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["date"]       = df["valid_time"].dt.date

    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")

    return df
