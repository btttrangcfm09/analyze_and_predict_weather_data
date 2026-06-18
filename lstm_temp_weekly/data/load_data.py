"""
Task: Load the raw Vietnam weather Parquet dataset from Kaggle.

Notes:
  - Runs on the Kaggle GPU runtime, so we keep the FULL country (no province
    filtering, no spatial down-sampling) to give the LSTM maximum context.
  - 'valid_time' is cast to datetime and rows are sorted into continuous
    per-coordinate time series.
"""
import pandas as pd

DATA_PATH = '/kaggle/input/vietnam-meteorological-weather-data-parquet/weather.parquet'

# Columns we read off disk. temperature_celsius is the forecast target; the rest
# are exogenous meteorological drivers fed into the LSTM as input features.
_COLS = [
    'latitude', 'longitude', 'valid_time',
    'temperature_celsius', 'apparent_temperature',
    'relative_humidity', 'wind_speed', 'wind_direction',
    'total_precipitation', 'total_cloud_cover',
    'mean_sea_level_pressure', 'surface_pressure',
    'sea_surface_temperature', 'air_density',
]


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    print(f"Đang đọc dữ liệu thời tiết toàn Việt Nam từ: {data_path}")

    df = pd.read_parquet(data_path, columns=_COLS, engine='auto')

    # Keep only the columns we asked for (in case the parquet carries extras /
    # partition columns).
    df = df[[c for c in _COLS if c in df.columns]]

    # 'valid_time' must be a real datetime for the cyclic time features later.
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Continuous chronological series per coordinate — required for correct
    # lag / rolling / sequence construction.
    df = df.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

    # float64 -> float32 halves RAM with negligible precision loss.
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')

    n_locs = df.groupby(['latitude', 'longitude'], sort=False).ngroups
    print(f"Nạp dữ liệu thành công! {len(df):,} dòng  |  {n_locs:,} điểm tọa độ (toàn Việt Nam)")
    return df
