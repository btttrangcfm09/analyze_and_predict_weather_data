"""
GIAI ĐOẠN 3 - predictor.py
Hàm dự đoán NHIỆT ĐỘ ngày mai (T+1) cho dashboard Streamlit.

Đọc 3 artifact của Giai đoạn 2 từ models/ và kho dữ liệu archive/weather.parquet:
  - lstm_weather_model_temp.pt   (checkpoint + metadata, lưu TÊN FILE tương đối)
  - scaler_temp.pkl              (MinMaxScaler đã fit)
  - feature_cols_temp.pkl        (danh sách 27 cột đặc trưng)

File này KHÔNG import streamlit để còn test được bằng CLI. Phần cache được đặt ở app.py:
  - app.py bọc load_model() bằng @st.cache_resource và load_history() bằng @st.cache_data
  - rồi gọi _predict_core(...) với các object đã cache.
"""
import os

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ---- Đường dẫn TƯƠNG ĐỐI (di động giữa các máy) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "archive", "weather.parquet")
CKPT_PATH = os.path.join(MODELS_DIR, "lstm_weather_model_temp.pt")

# Phải khớp với lúc train (xem models/train_lstm_temperature.py)
SEQUENCE_LENGTH = 30
TARGET_COLS = ['temperature_celsius']
FEATURE_COLS_RAW = [
    'temperature_celsius', 'total_precipitation', 'wind_speed', 'wind_direction',
    'relative_humidity', 'mean_sea_level_pressure', 'surface_pressure',
    'total_cloud_cover', 'apparent_temperature', 'air_density',
]


# ---- Kiến trúc model: PHẢI giống hệt lúc train ----
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out)


def load_model():
    """Dựng lại model + scaler + feat_cols từ checkpoint.
    Tách riêng để app.py bọc @st.cache_resource (chỉ chạy 1 lần/phiên)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # Resolve tên file tương đối theo MODELS_DIR; fallback cho checkpoint cũ (*_path tuyệt đối)
    scaler_name = ckpt.get('scaler_name') or os.path.basename(ckpt['scaler_path'])
    feat_name = ckpt.get('feature_cols_name') or os.path.basename(ckpt['feature_cols_path'])
    scaler = joblib.load(os.path.join(MODELS_DIR, scaler_name))
    feat_cols = joblib.load(os.path.join(MODELS_DIR, feat_name))

    model = LSTMModel(ckpt['input_size'], ckpt['hidden_size'], ckpt['num_layers'],
                      ckpt['output_size'], ckpt['dropout']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, scaler, feat_cols, device


def load_history():
    """Đọc parquet 1 lần. Tách riêng để app.py bọc @st.cache_data."""
    df = pd.read_parquet(DATA_PATH)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['date'] = df['valid_time'].dt.date
    return df


def _build_features(loc, feat_cols, scaler):
    """Tái tạo ĐÚNG pipeline đặc trưng lúc train cho 1 tọa độ."""
    loc = loc.copy()
    for col in FEATURE_COLS_RAW:
        loc[col] = loc[col].interpolate('linear').bfill().ffill()
    loc['total_precipitation'] = np.log1p(loc['total_precipitation'])      # log mưa
    loc['day_of_week'] = loc['valid_time'].dt.dayofweek
    loc['day_of_year'] = loc['valid_time'].dt.dayofyear
    loc['month_num'] = loc['valid_time'].dt.month
    loc['season'] = loc['month_num'].apply(
        lambda m: 0 if m in (2, 3, 4) else 1 if m in (5, 6, 7) else 2 if m in (8, 9, 10) else 3)
    loc['day_of_week_sin'] = np.sin(2 * np.pi * loc['day_of_week'] / 7)
    loc['day_of_week_cos'] = np.cos(2 * np.pi * loc['day_of_week'] / 7)
    loc['month_sin'] = np.sin(2 * np.pi * loc['month_num'] / 12)
    loc['month_cos'] = np.cos(2 * np.pi * loc['month_num'] / 12)
    loc['day_of_year_sin'] = np.sin(2 * np.pi * loc['day_of_year'] / 365)
    loc['day_of_year_cos'] = np.cos(2 * np.pi * loc['day_of_year'] / 365)

    agg = {c: 'mean' for c in FEATURE_COLS_RAW}
    agg.update({k: 'first' for k in ['day_of_week', 'day_of_year', 'month_num', 'season',
                'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
                'day_of_year_sin', 'day_of_year_cos']})
    daily = loc.groupby(['latitude', 'longitude', 'date']).agg(agg).reset_index()

    for col in ['temperature_celsius', 'total_precipitation']:
        for lag in (1, 3, 7):
            daily[f'{col}_lag{lag}'] = daily.groupby(['latitude', 'longitude'])[col].shift(lag)
        for w in (3, 7):
            daily[f'{col}_roll{w}'] = daily.groupby(['latitude', 'longitude'])[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean())
    daily = daily.dropna()

    numeric_feats = [c for c in feat_cols if c != 'season']
    daily[numeric_feats] = scaler.transform(daily[numeric_feats])
    return daily, numeric_feats


def _predict_core(lat, lon, model, scaler, feat_cols, device, df):
    """Phần lõi dự đoán, nhận sẵn model/df đã cache. Dùng chung cho CLI và web."""
    # Tọa độ lưới gần nhất
    coords = df[['latitude', 'longitude']].drop_duplicates()
    dists = np.sqrt((coords['latitude'] - lat) ** 2 + (coords['longitude'] - lon) ** 2)
    near = coords.loc[dists.idxmin()]
    loc = df[(df['latitude'] == near['latitude']) & (df['longitude'] == near['longitude'])]

    daily, numeric_feats = _build_features(loc, feat_cols, scaler)
    if len(daily) < SEQUENCE_LENGTH:
        raise ValueError(f"Không đủ {SEQUENCE_LENGTH} ngày dữ liệu cho tọa độ này.")

    seq = daily[feat_cols].values[-SEQUENCE_LENGTH:]
    x = torch.FloatTensor(np.asarray(seq, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    # Inverse transform về °C
    result = {}
    for i, name in enumerate(TARGET_COLS):
        d = np.zeros((1, len(numeric_feats)))
        idx = numeric_feats.index(name)
        d[0, idx] = pred[i]
        val = scaler.inverse_transform(d)[0, idx]
        if name == 'total_precipitation':
            val = np.expm1(val)
        result[name] = float(val)

    last_date = pd.to_datetime(daily['date'].iloc[-1])
    result['predict_date'] = (last_date + pd.Timedelta(days=1)).date().isoformat()
    result['coords'] = (float(near['latitude']), float(near['longitude']))
    return result


def predict_tomorrow(lat, lon):
    """
    Dự đoán nhiệt độ ngày mai (°C) tại (lat, lon).
    Phiên bản tự load model + dữ liệu (dùng cho CLI/test). Web nên dùng bản cached ở app.py.
    Trả về dict: {temperature_celsius, predict_date, coords}
    """
    model, scaler, feat_cols, device = load_model()
    df = load_history()
    return _predict_core(lat, lon, model, scaler, feat_cols, device, df)


if __name__ == '__main__':
    # Test nhanh từ CLI: python predictor.py
    for name, (la, lo) in {'Hà Nội': (21.0, 105.8), 'TP.HCM': (10.8, 106.7),
                           'Đà Nẵng': (16.0, 108.2)}.items():
        print(name, '->', predict_tomorrow(la, lo))
