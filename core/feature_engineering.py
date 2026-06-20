"""
core/feature_engineering.py — Dựng 30 đặc trưng encoder cho mô hình tuần.

Copy y hệt pipeline huấn luyện (lstm_temp_weekly/preprocessing/feature_engineering.py)
để khớp tuyệt đối thứ tự cột trong feature_cols_temp.pkl:
  interpolate → year_normalized → monthly climatology → lag → rolling → cyclic → dropna.
"""
import numpy as np
import pandas as pd

from core.features import TARGET_COL, _METEO_COLS

_LAG_HOURS  = [1, 3, 6, 24, 168]
_ROLL_HOURS = [3, 6, 24, 168]


def _interpolate_raw(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in _METEO_COLS if c in df.columns]
    df[existing] = (
        df.groupby(['latitude', 'longitude'], sort=False)[existing]
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    return df


def _add_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['year_normalized'] = ((df['valid_time'].dt.year - 2020) / 6).astype('float32')
    return df


def _add_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    month = df['valid_time'].dt.month
    df['monthly_mean_temperature'] = (
        df.groupby(['latitude', 'longitude', month], sort=False)[TARGET_COL]
        .transform('mean')
        .astype('float32')
    )
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    for h in _LAG_HOURS:
        df[f'{TARGET_COL}_lag{h}'] = grp.shift(h).astype('float32')
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    for h in _ROLL_HOURS:
        # shift(1) đảm bảo giờ hiện tại không rò rỉ vào chính đặc trưng của nó.
        df[f'{TARGET_COL}_roll{h}'] = (
            grp.transform(lambda x: x.shift(1).rolling(h, min_periods=1).mean())
            .astype('float32')
        )
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df['valid_time']
    df['hour_sin']        = np.sin(2 * np.pi * dt.dt.hour      / 24 ).astype('float32')
    df['hour_cos']        = np.cos(2 * np.pi * dt.dt.hour      / 24 ).astype('float32')
    df['day_of_week_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7  ).astype('float32')
    df['day_of_week_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7  ).astype('float32')
    df['month_sin']       = np.sin(2 * np.pi * dt.dt.month     / 12 ).astype('float32')
    df['month_cos']       = np.cos(2 * np.pi * dt.dt.month     / 12 ).astype('float32')
    df['day_of_year_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365).astype('float32')
    df['day_of_year_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365).astype('float32')
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _interpolate_raw(df)
    df = _add_year_feature(df)
    df = _add_monthly_climatology(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_time_features(df)
    df = df.dropna().reset_index(drop=True)
    return df
