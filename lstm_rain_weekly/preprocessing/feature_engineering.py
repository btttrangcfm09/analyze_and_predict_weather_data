"""
Task: Build the feature matrix for weekly rainfall forecasting.

Pipeline order (each step depends on the previous one):
  1. Interpolate raw meteorological NaN (per coordinate group).
  2. log1p transform of total_precipitation — rainfall is extremely right-skewed
     (lots of zeros, rare heavy bursts); log1p compresses the tail so the LSTM
     trains on a near-symmetric target.
  3. YEARLY feature   : 'year_normalized' = (year - 2020) / 6  → long-term climate drift.
  4. MONTHLY feature  : 'monthly_mean_precipitation' = historical mean rainfall per
                         (lat, lon, month) → a climatology baseline telling the model
                         which months are the rainy season of each region.
  5. WEEKLY context   : lag (1, 3, 6, 24, 168h) and rolling-mean (3, 6, 24, 168h)
                         features → the model "remembers" the past week.
  6. Cyclic time encodings (hour, day_of_week, month, day_of_year as sin/cos).
  7. Drop residual NaN rows created by the lag/rolling boundaries.

IMPORTANT: every rainfall-derived feature below is computed AFTER the log1p
transform, so the model and all baselines live in a single consistent log space.
"""
import numpy as np
import pandas as pd

TARGET_COL = 'total_precipitation'

_METEO_COLS = [
    'temperature_celsius', 'apparent_temperature',
    'relative_humidity', 'wind_speed', 'wind_direction',
    'total_precipitation', 'total_cloud_cover',
    'mean_sea_level_pressure', 'surface_pressure',
    'sea_surface_temperature', 'air_density',
]

_LAG_HOURS  = [1, 3, 6, 24, 168]
_ROLL_HOURS = [3, 6, 24, 168]


# ── 1. Interpolate raw NaN ──────────────────────────────────────────────────────

def _interpolate_raw(df: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in _METEO_COLS if c in df.columns]
    df[existing] = (
        df.groupby(['latitude', 'longitude'], sort=False)[existing]
        .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    )
    return df


# ── 2. log1p transform of rainfall ──────────────────────────────────────────────

def _log_transform_target(df: pd.DataFrame) -> pd.DataFrame:
    # Clip tiny negative noise to 0 before log1p (precip can never be negative).
    df[TARGET_COL] = np.log1p(df[TARGET_COL].clip(lower=0)).astype('float32')
    return df


# ── 3. Yearly climate-drift feature ─────────────────────────────────────────────

def _add_year_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['year_normalized'] = ((df['valid_time'].dt.year - 2020) / 6).astype('float32')
    return df


# ── 4. Monthly climatology baseline ──────────────────────────────────────────────

def _add_monthly_climatology(df: pd.DataFrame) -> pd.DataFrame:
    month = df['valid_time'].dt.month
    df['monthly_mean_precipitation'] = (
        df.groupby(['latitude', 'longitude', month], sort=False)[TARGET_COL]
        .transform('mean')
        .astype('float32')
    )
    return df


# ── 5. Weekly context: lags + rolling means ──────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    for h in _LAG_HOURS:
        df[f'{TARGET_COL}_lag{h}'] = grp.shift(h).astype('float32')
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(['latitude', 'longitude'], sort=False)[TARGET_COL]
    for h in _ROLL_HOURS:
        # shift(1) guarantees the current hour never leaks into its own feature.
        df[f'{TARGET_COL}_roll{h}'] = (
            grp.transform(lambda x: x.shift(1).rolling(h, min_periods=1).mean())
            .astype('float32')
        )
    return df


# ── 6. Cyclic time encodings ──────────────────────────────────────────────────────

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


# ── Orchestrator ──────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _interpolate_raw(df)
    df = _log_transform_target(df)
    df = _add_year_feature(df)
    df = _add_monthly_climatology(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_time_features(df)
    df = df.dropna().reset_index(drop=True)
    return df
