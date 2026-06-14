"""
GIAI ĐOẠN 3 - utils/data_pipeline.py
Tải dữ liệu từ Copernicus CDS API (PHẦN 1) và xử lý + phát hiện bão (PHẦN 3).
Tách ra từ app.py. File dữ liệu kết quả nằm trong thư mục data/.
"""
import os
import math
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import cdsapi

from utils.storm import detect_storms

# ---- Đường dẫn TƯƠNG ĐỐI theo vị trí repo ----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
ETL_DIR = os.path.join(REPO_ROOT, "ETL", "code")
TMP_DIR = os.path.join(REPO_ROOT, "crawl_weather_temp")

DASHBOARD_CSV = os.path.join(DATA_DIR, "dashboard_main.csv")
STORMS_CSV = os.path.join(DATA_DIR, "storms.csv")
WEATHER_CSV = os.path.join(ETL_DIR, "weather_data.csv")
RAIN_CSV = os.path.join(ETL_DIR, "rain_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# ==============================================================
# PHẦN 1: TẢI DỮ LIỆU TỪ CDS API
# ==============================================================
def fetch_cds_data(start_date, end_date):
    os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
    os.environ["CDSAPI_KEY"] = "da23c1f9-21f4-4adc-96af-5e212d2f4440"

    client = cdsapi.Client(verify=False)
    dataset = "reanalysis-era5-single-levels"
    os.makedirs(ETL_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    delta = timedelta(days=1)

    # --- 1. TẢI WEATHER DATA ---
    st.info(f"Đang tải Dữ liệu Thời tiết từ {start_date.date()} đến {end_date.date()}...")
    all_weather = []
    curr = start_date
    progress_bar = st.progress(0)
    total_days = (end_date - start_date).days + 1
    day_count = 0

    while curr <= end_date:
        y, m, d = curr.strftime('%Y'), curr.strftime('%m'), curr.strftime('%d')
        target = os.path.join(TMP_DIR, f"weather_{d}_{m}_{y}.grib")
        client.retrieve(dataset, {
            'product_type': ['reanalysis'],
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'mean_sea_level_pressure', 'sea_surface_temperature', 'surface_pressure', 'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water'],
            'year': [y], 'month': [m], 'day': [d],
            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
            'data_format': 'grib', 'download_format': 'unarchived', 'area': [24, 102, 8, 112]
        }, target)

        ds = xr.open_dataset(target)
        all_weather.append(ds.to_dataframe().reset_index())
        curr += delta
        day_count += 1
        progress_bar.progress(day_count / total_days)

    pd.concat(all_weather, ignore_index=False).to_csv(WEATHER_CSV, index=False)
    st.success("Tải xong Dữ liệu Thời tiết!")

    # --- 2. TẢI RAIN DATA ---
    st.info("Đang tải Dữ liệu Lượng mưa...")
    all_rain = []
    curr = start_date
    progress_bar2 = st.progress(0)
    day_count = 0

    while curr <= end_date:
        y, m, d = curr.strftime('%Y'), curr.strftime('%m'), curr.strftime('%d')
        target = os.path.join(TMP_DIR, f"rain_{d}_{m}_{y}.grib")
        client.retrieve(dataset, {
            'product_type': ['reanalysis'],
            'variable': ['total_precipitation'],
            'year': [y], 'month': [m], 'day': [d],
            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
            'data_format': 'grib', 'download_format': 'unarchived', 'area': [24, 102, 8, 112]
        }, target)

        ds = xr.open_dataset(target)
        all_rain.append(ds.to_dataframe().reset_index())
        curr += delta
        day_count += 1
        progress_bar2.progress(day_count / total_days)

    pd.concat(all_rain, ignore_index=False).to_csv(RAIN_CSV, index=False)
    st.success("Tải xong Dữ liệu Lượng mưa!")

    # Xóa file grib tạm
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    st.success("Hoàn tất tải toàn bộ dữ liệu từ vệ tinh Copernicus!")


# ==============================================================
# PHẦN 3: XỬ LÝ DỮ LIỆU + PHÁT HIỆN BÃO (có cache Streamlit)
# ==============================================================
@st.cache_data(show_spinner="Đang xử lý dữ liệu bão...")
def process_and_load_data(force_reprocess=False):
    from predictor import LOCAL_STORMS_PATH
    if os.path.exists(LOCAL_STORMS_PATH):
        df_storms = pd.read_parquet(LOCAL_STORMS_PATH)
        return None, df_storms
    else:
        empty_storms = pd.DataFrame(columns=['time', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed', 'year', 'month', 'day', 'time_dt'])

