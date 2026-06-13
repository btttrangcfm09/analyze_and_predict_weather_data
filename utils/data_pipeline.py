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
@st.cache_data(show_spinner="Đang xử lý dữ liệu từ file thô (chỉ chạy lần đầu)...")
def process_and_load_data(force_reprocess=False):
    if force_reprocess or not os.path.exists(DASHBOARD_CSV) or not os.path.exists(STORMS_CSV):
        try:
            df_weather = pd.read_csv(WEATHER_CSV)
            df_rain = pd.read_csv(RAIN_CSV)
        except FileNotFoundError:
            st.error("Lỗi: Không tìm thấy file dữ liệu gốc ở thư mục ETL/code/. Vui lòng chạy các file crawl dữ liệu trước.")
            st.stop()

        if 'valid_time' in df_rain.columns:
            df_rain = df_rain.drop(columns=['time'], errors='ignore').rename(columns={'valid_time': 'time'})
            df_rain = df_rain[['time', 'latitude', 'longitude', 'tp']]

        if 'valid_time' in df_weather.columns:
            df_weather = df_weather.drop(columns=['time'], errors='ignore').rename(columns={'valid_time': 'time'})
            weather_cols = ['time', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sst', 'sp', 'tcc']
            df_weather = df_weather[[c for c in weather_cols if c in df_weather.columns]]

        df_joined = pd.merge(df_rain, df_weather, on=['latitude', 'longitude', 'time'], how='inner')
        df_joined = df_joined.drop_duplicates(subset=['latitude', 'longitude', 'time'])

        if df_joined.empty:
            st.error("Dữ liệu sau khi gộp bị rỗng. Vui lòng kiểm tra lại ngày tháng của 2 file crawl.")
            st.stop()

        df_joined['t2m'] = df_joined['t2m'] - 273.15
        df_joined['d2m'] = df_joined['d2m'] - 273.15
        df_joined['wind_speed'] = np.sqrt(df_joined['u10']**2 + df_joined['v10']**2)
        df_joined['saturation_vapor_pressure'] = 6.11 * (10 ** (7.5 * df_joined['t2m'] / (df_joined['t2m'] + 237.3)))
        df_joined['vapor_pressure'] = 6.11 * (10 ** (7.5 * df_joined['d2m'] / (df_joined['d2m'] + 237.3)))
        df_joined['relative_humidity'] = (df_joined['vapor_pressure'] / df_joined['saturation_vapor_pressure']) * 100
        df_joined['apparent_temperature'] = df_joined['t2m'] + 0.33 * df_joined['vapor_pressure'] - 0.70 * df_joined['wind_speed'] - 4.00
        df_joined['air_density'] = df_joined['sp'] / (287.05 * (df_joined['t2m'] + 273.15))
        df_joined['wind_direction'] = 180 + (180 / math.pi) * np.arctan2(df_joined['u10'], df_joined['v10'])
        df_joined['wind_direction'] = np.where(df_joined['wind_direction'] < 0, df_joined['wind_direction'] + 360, df_joined['wind_direction'])

        df_joined = df_joined[(df_joined['latitude'] >= 8) & (df_joined['latitude'] <= 24) &
                              (df_joined['longitude'] >= 102) & (df_joined['longitude'] <= 112)]

        dashboard_df = df_joined.rename(columns={
            'time': 'valid_time',
            'tp': 'total_precipitation',
            't2m': 'temperature_celsius',
            'msl': 'mean_sea_level_pressure',
            'sst': 'sea_surface_temperature_celsius',
            'sp': 'surface_pressure',
            'tcc': 'total_cloud_cover'
        })
        dashboard_df.to_csv(DASHBOARD_CSV, index=False)

        times = sorted(dashboard_df['valid_time'].unique())
        lons = np.arange(102, 112.25, 0.25)
        lats = np.arange(24, 7.75, -0.25)

        storm_rows = []
        storm_count = 0
        prev_lon, prev_lat = None, None

        for t in times:
            df_t = df_joined[df_joined['time'] == t]
            if df_t.empty:
                continue

            msl_pivot = df_t.pivot(index='latitude', columns='longitude', values='msl')
            ws_pivot = df_t.pivot(index='latitude', columns='longitude', values='wind_speed')
            msl_pivot = msl_pivot.reindex(index=lats, columns=lons).values
            ws_pivot = ws_pivot.reindex(index=lats, columns=lons).values

            lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, _ = detect_storms(
                msl_pivot, ws_pivot, lons, lats, res=0.25, order='topdown', Npix_min=9, Npix_max=6000,
                rel_amp_thresh=100, d_thresh=2500, cyc='cyclonic', cut_lon=1, cut_lat=1, globe=False
            )

            if len(lon_storms) > 0:
                for i in range(len(lon_storms)):
                    lon_storm, lat_storm = lon_storms[i], lat_storms[i]
                    if prev_lon is not None and np.sqrt((lon_storm - prev_lon)**2 + (lat_storm - prev_lat)**2) < 1:
                        storm_id = f"2024-{storm_count:02d}"
                    else:
                        storm_count += 1
                        storm_id = f"2024-{storm_count:02d}"
                    prev_lon, prev_lat = lon_storm, lat_storm
                    storm_rows.append({'time': t, 'id': storm_id, 'lon_storm': lon_storm, 'lat_storm': lat_storm,
                                       'amp_storm': amp_storms[i], 'max_wind_speed': max_wind_speeds[i], 'area_storm': area_storms[i]})

        if len(storm_rows) > 0:
            storms_df = pd.DataFrame(storm_rows)
        else:
            storms_df = pd.DataFrame(columns=['time', 'id', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed', 'area_storm'])

        storms_df.to_csv(STORMS_CSV, index=False)

    df_main = pd.read_csv(DASHBOARD_CSV)
    df_storms = pd.read_csv(STORMS_CSV)

    df_main['valid_time_dt'] = pd.to_datetime(df_main['valid_time'])
    df_main['day'] = df_main['valid_time_dt'].dt.date
    df_main['month'] = df_main['valid_time_dt'].dt.strftime('%Y-%m')
    df_main['year'] = df_main['valid_time_dt'].dt.strftime('%Y')
    df_main['hour'] = df_main['valid_time_dt'].dt.strftime('%H:%M')

    df_storms['time_dt'] = pd.to_datetime(df_storms['time'])
    df_storms['month'] = df_storms['time_dt'].dt.strftime('%Y-%m')
    df_storms['year'] = df_storms['time_dt'].dt.strftime('%Y')
    df_storms['day'] = df_storms['time_dt'].dt.date

    return df_main, df_storms
