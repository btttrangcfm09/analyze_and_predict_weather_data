import os
import pandas as pd
import numpy as np
import sys
from utils.storm import detect_storms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(BASE_DIR, "data", "weather.parquet")
STORMS_PARQUET = os.path.join(BASE_DIR, "data", "storms.parquet")

KAGGLE_DATASET = "nguyentranggggg/vietnam-meteorological-weather-data-2020-2026"

def get_kaggle_parquet():
    import kagglehub
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            root = kagglehub.dataset_download(KAGGLE_DATASET)
            return os.path.join(root, "weather.parquet")
        except Exception as e:
            print(f"Lỗi tải Kaggle (lần {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e

def generate_storms_cache():
    print("Generating storms...")
    target_parquet = PARQUET_DIR
    if not os.path.exists(target_parquet):
        print(f"Not found {target_parquet}, trying kaggle path...")
        try:
            target_parquet = get_kaggle_parquet()
        except Exception as e:
            print(f"Kaggle path error: {e}")
        
        if not os.path.exists(target_parquet):
            print("Not found kaggle path either!")
            return

    # Load all parquet files
    import pyarrow.parquet as pq
    df = pq.read_table(target_parquet).to_pandas()
    
    # We need time, latitude, longitude, msl, wind_speed
    df['time_dt'] = pd.to_datetime(df['valid_time'])
    df = df.sort_values(['time_dt', 'latitude', 'longitude'])
    
    times = df['valid_time'].unique()
    lons = np.arange(102, 112.25, 0.25)
    lats = np.arange(24, 7.75, -0.25)

    storm_rows = []
    storm_count = 0
    prev_lon, prev_lat = None, None

    print(f"Analyzing {len(times)} timeframes...")
    for i, t in enumerate(times):
        if i % 100 == 0:
            print(f"  Analyzed {i}/{len(times)}...")
        
        df_t = df[df['valid_time'] == t]
        if df_t.empty:
            continue

        msl_pivot = df_t.pivot(index='latitude', columns='longitude', values='mean_sea_level_pressure')
        ws_pivot = df_t.pivot(index='latitude', columns='longitude', values='wind_speed')
        
        msl_pivot = msl_pivot.reindex(index=lats, columns=lons).values
        ws_pivot = ws_pivot.reindex(index=lats, columns=lons).values

        lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, _ = detect_storms(
            msl_pivot, ws_pivot, lons, lats, res=0.25, order='topdown', Npix_min=9, Npix_max=6000,
            rel_amp_thresh=100, d_thresh=2500, cyc='cyclonic', cut_lon=1, cut_lat=1, globe=False
        )

        if len(lon_storms) > 0:
            for i_storm in range(len(lon_storms)):
                lon_storm, lat_storm = lon_storms[i_storm], lat_storms[i_storm]
                if prev_lon is not None and np.sqrt((lon_storm - prev_lon)**2 + (lat_storm - prev_lat)**2) < 1:
                    storm_id = f"storm-{storm_count:03d}"
                else:
                    storm_count += 1
                    storm_id = f"storm-{storm_count:03d}"
                prev_lon, prev_lat = lon_storm, lat_storm
                storm_rows.append({'time': pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S'), 'id': storm_id, 'lon_storm': lon_storm, 'lat_storm': lat_storm,
                                   'amp_storm': amp_storms[i_storm], 'max_wind_speed': max_wind_speeds[i_storm], 'area_storm': area_storms[i_storm]})

    if len(storm_rows) > 0:
        storms_df = pd.DataFrame(storm_rows)
    else:
        storms_df = pd.DataFrame(columns=['time', 'id', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed', 'area_storm'])

    # Format dataframe cho app.py (thêm year, month, day, time_dt)
    if not storms_df.empty:
        storms_df['time_dt'] = pd.to_datetime(storms_df['time'])
        storms_df['day'] = storms_df['time_dt'].dt.date
        storms_df['year'] = storms_df['time_dt'].dt.strftime('%Y')
        storms_df['month'] = storms_df['time_dt'].dt.strftime('%m')

    storms_df.to_parquet(STORMS_PARQUET, index=False, compression="snappy")
    print(f"Saved to {STORMS_PARQUET} (Total: {len(storms_df)} records)")

def append_new_storms(df_new: pd.DataFrame):
    """Phân tích bão trên dữ liệu mới cào về và nối vào cache storms.parquet hiện có."""
    if df_new.empty:
        return
        
    print("Bắt đầu quét bão trên dữ liệu mới...")
    df_new['time_dt'] = pd.to_datetime(df_new['valid_time'] if 'valid_time' in df_new.columns else df_new['time'])
    df_new = df_new.sort_values(['time_dt', 'latitude', 'longitude'])
    
    times = df_new['time_dt'].unique()
    lons = np.arange(102, 112.25, 0.25)
    lats = np.arange(24, 7.75, -0.25)

    storm_rows = []
    
    # Tìm id lớn nhất trong cache hiện tại để tăng dần
    storm_count = 0
    if os.path.exists(STORMS_PARQUET):
        old_storms = pd.read_parquet(STORMS_PARQUET)
        if not old_storms.empty and 'id' in old_storms.columns:
            last_id = old_storms['id'].max()
            if isinstance(last_id, str) and last_id.startswith("storm-"):
                try:
                    storm_count = int(last_id.split("-")[1])
                except ValueError:
                    pass
    
    prev_lon, prev_lat = None, None

    for i, t in enumerate(times):
        df_t = df_new[df_new['time_dt'] == t]
        if df_t.empty:
            continue

        # Convert back to string format for pivot if needed, or just use it
        time_str = pd.to_datetime(t).strftime('%Y-%m-%d %H:%M:%S')

        msl_pivot = df_t.pivot(index='latitude', columns='longitude', values='mean_sea_level_pressure' if 'mean_sea_level_pressure' in df_t.columns else 'msl')
        ws_pivot = df_t.pivot(index='latitude', columns='longitude', values='wind_speed')
        
        msl_pivot = msl_pivot.reindex(index=lats, columns=lons).values
        ws_pivot = ws_pivot.reindex(index=lats, columns=lons).values

        lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, _ = detect_storms(
            msl_pivot, ws_pivot, lons, lats, res=0.25, order='topdown', Npix_min=9, Npix_max=6000,
            rel_amp_thresh=100, d_thresh=2500, cyc='cyclonic', cut_lon=1, cut_lat=1, globe=False
        )

        if len(lon_storms) > 0:
            for i_storm in range(len(lon_storms)):
                lon_storm, lat_storm = lon_storms[i_storm], lat_storms[i_storm]
                if prev_lon is not None and np.sqrt((lon_storm - prev_lon)**2 + (lat_storm - prev_lat)**2) < 1:
                    storm_id = f"storm-{storm_count:03d}"
                else:
                    storm_count += 1
                    storm_id = f"storm-{storm_count:03d}"
                prev_lon, prev_lat = lon_storm, lat_storm
                storm_rows.append({'time': time_str, 'id': storm_id, 'lon_storm': lon_storm, 'lat_storm': lat_storm,
                                   'amp_storm': amp_storms[i_storm], 'max_wind_speed': max_wind_speeds[i_storm], 'area_storm': area_storms[i_storm]})

    if len(storm_rows) > 0:
        new_storms_df = pd.DataFrame(storm_rows)
        new_storms_df['time_dt'] = pd.to_datetime(new_storms_df['time'])
        new_storms_df['day'] = new_storms_df['time_dt'].dt.date
        new_storms_df['year'] = new_storms_df['time_dt'].dt.strftime('%Y')
        new_storms_df['month'] = new_storms_df['time_dt'].dt.strftime('%m')
        
        if os.path.exists(STORMS_PARQUET):
            old_storms = pd.read_parquet(STORMS_PARQUET)
            combined = pd.concat([old_storms, new_storms_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['time', 'lon_storm', 'lat_storm'], keep='last')
        else:
            combined = new_storms_df
            
        combined.to_parquet(STORMS_PARQUET, index=False, compression="snappy")
        print(f"Đã cập nhật {len(new_storms_df)} bản ghi bão mới vào {STORMS_PARQUET}.")
    else:
        print("Không phát hiện thêm cơn bão nào trong dữ liệu mới.")

if __name__ == "__main__":
    generate_storms_cache()
