"""
Task: Add time-encoding, lag, and rolling features; interpolate original NaN values.

Order:
  1. Interpolate raw meteorological NaN (per province group)
  2. Add cyclic time encodings
  3. Add lag features for temperature
  4. Add rolling-mean features for temperature
  5. Drop the tiny residual NaN rows (lag/rolling boundaries)
"""
import numpy as np
import pandas as pd

TARGET_COL = 'temperature_celsius'

_METEO_COLS = [
    'temperature_celsius', 'apparent_temperature',
    'relative_humidity', 'wind_speed', 'wind_direction',
    'total_precipitation', 'total_cloud_cover',
    'mean_sea_level_pressure', 'surface_pressure',
    'sea_surface_temperature', 'air_density',
]


def _interpolate_raw(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in _METEO_COLS if c in df.columns]
    df[existing] = (
        df.groupby(['latitude', 'longitude'], sort=False)[existing]
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['valid_time']
    df['hour_sin']        = np.sin(2 * np.pi * dt.dt.hour       / 24 ).astype('float32')
    df['hour_cos']        = np.cos(2 * np.pi * dt.dt.hour       / 24 ).astype('float32')
    df['day_of_week_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek  / 7  ).astype('float32')
    df['day_of_week_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek  / 7  ).astype('float32')
    df['month_sin']       = np.sin(2 * np.pi * dt.dt.month      / 12 ).astype('float32')
    df['month_cos']       = np.cos(2 * np.pi * dt.dt.month      / 12 ).astype('float32')
    df['day_of_year_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear  / 365).astype('float32')
    df['day_of_year_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear  / 365).astype('float32')
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    df[f'{TARGET_COL}_lag1'] = grp.shift(1).astype('float32')
    df[f'{TARGET_COL}_lag3'] = grp.shift(3).astype('float32')
    df[f'{TARGET_COL}_lag6'] = grp.shift(6).astype('float32')
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    # shift(1) ensures we never use the current value → no leakage
    df[f'{TARGET_COL}_roll3'] = (
        grp.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .astype('float32')
    )
    df[f'{TARGET_COL}_roll6'] = (
        grp.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        .astype('float32')
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _interpolate_raw(df)
    df = _add_time_features(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    return df
