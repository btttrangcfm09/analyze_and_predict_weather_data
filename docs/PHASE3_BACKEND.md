# GIAI ĐOẠN 3 — KỸ SƯ BACKEND & TÍCH HỢP (QUY TRÌNH CHI TIẾT)

> **✅ TRẠNG THÁI: ĐÃ THỰC THI XONG.** Theo 2 quyết định đã chốt: (1) *sửa tận gốc* đường dẫn
> checkpoint — `export_model` lưu `scaler_name`/`feature_cols_name` (tên file) và checkpoint
> hiện có đã được resave; (2) *dời* `dashboard_main.csv` & `storms.csv` vào `data/`.
> Thực thi gồm: `utils/storm.py`, `utils/data_pipeline.py`, `predictor.py`, refactor `app.py`
> (caching), cập nhật `requirements.txt`. Đã test: `python predictor.py` + chạy thử dashboard.

> Mục tiêu: lắp "bộ não" LSTM (4 file `_temp` từ Giai đoạn 2) vào dashboard Streamlit,
> tổ chức lại code theo kiến trúc trong [PROJECT_WORKFLOW.md](../PROJECT_WORKFLOW.md), và
> tối ưu để web phản hồi tức thì. Tài liệu này viết để **làm theo từng bước, copy được code**.

**Đầu vào bắt buộc (đã có sẵn trong `models/`):**
- `lstm_weather_model_temp.pt` — checkpoint + metadata (input_size, hidden_size, …).
- `scaler_temp.pkl` — `MinMaxScaler` đã fit cho 26 cột số.
- `feature_cols_temp.pkl` — danh sách 27 cột đặc trưng (có `season`).
- `archive/weather.parquet` — kho dữ liệu lịch sử để lấy 30 ngày gần nhất.
- (`training_results_temp.png` — chỉ để hiển thị/minh họa, không dùng khi suy luận.)

**Hợp đồng mô hình** (phải khớp tuyệt đối — nhắc lại từ CLAUDE.md §4.2):
`SEQUENCE_LENGTH = 30`, target = `temperature_celsius`, có `log1p` cho mưa, lag 1/3/7 +
rolling 3/7, chuẩn hóa bằng đúng `scaler_temp.pkl`, thứ tự cột theo đúng `feature_cols_temp.pkl`.

---

## Tổng quan 5 bước
1. **Bước 3.0** — Chuẩn bị môi trường & làm cho checkpoint "di động" (đường dẫn tương đối).
2. **Bước 3.1** — Refactor: tách `app.py` thành `utils/` (+ thư mục `data/`, `models/`).
3. **Bước 3.2** — Viết `predictor.py` với `predict_tomorrow(lat, lon)`.
4. **Bước 3.3** — Tối ưu bộ nhớ: `@st.cache_resource` (model) + `@st.cache_data` (dữ liệu).
5. **Bước 3.4** — Kiểm thử & bàn giao cho Giai đoạn 4.

---

## BƯỚC 3.0 — Chuẩn bị & làm checkpoint di động

### 3.0.1. Bổ sung phụ thuộc
Thêm vào `requirements.txt`:
```
torch
scikit-learn
joblib
pyarrow
matplotlib
```
Cài: `pip install -r requirements.txt`.

### 3.0.2. Vấn đề đường dẫn tuyệt đối (PHẢI xử lý)
Checkpoint hiện lưu **đường dẫn tuyệt đối** của máy train:
```python
'scaler_path':       'D:\\...\\models\\scaler_temp.pkl',
'feature_cols_path': 'D:\\...\\models\\feature_cols_temp.pkl',
```
và `DATA_PATH` trong script cũng tuyệt đối. Khi chạy ở máy/đường dẫn khác sẽ lỗi
`FileNotFoundError`. **Giải pháp ở Giai đoạn 3:** *bỏ qua* đường dẫn lưu trong checkpoint,
luôn tự resolve theo vị trí file `predictor.py` (xem code Bước 3.2). Không cần train lại.

> (Tùy chọn) Nếu muốn sửa tận gốc: ở `export_model()` lưu **tên file** thay vì đường dẫn đầy
> đủ, rồi train lại. Không bắt buộc cho Giai đoạn 3.
-> Hãy sửa tận gốc

---

## BƯỚC 3.1 — Refactor cấu trúc thư mục

Theo PROJECT_WORKFLOW Bước 3.1, tách logic khỏi `app.py`. Cấu trúc đích:

```
analyze_and_predict_weather_data/
├── app.py                 # CHỈ còn phần UI + điều phối (mỏng đi)
├── predictor.py           # MỚI: hàm dự đoán AI (Bước 3.2)
├── utils/
│   ├── __init__.py
│   ├── storm.py           # detect_storms, spatial_filter, distance_matrix, len_deg_lon, nanmean
│   └── data_pipeline.py   # fetch_cds_data, process_and_load_data
├── models/                # 4 file _temp (đã có)
├── data/                  # (tùy chọn) dời dashboard_main.csv, storms.csv vào đây
└── archive/weather.parquet
```

### 3.1.1. Tạo `utils/storm.py`
Chuyển nguyên các hàm thuật toán bão từ `app.py` (PHẦN 2, dòng ~97–182):
`distance_matrix`, `nanmean`, `len_deg_lon`, `spatial_filter`, `detect_storms`.
Chúng thuần NumPy/SciPy, không phụ thuộc Streamlit nên copy nguyên là chạy.

### 3.1.2. Tạo `utils/data_pipeline.py`
Chuyển `fetch_cds_data` (PHẦN 1) và `process_and_load_data` (PHẦN 3) vào đây.
Lưu ý `process_and_load_data` gọi `detect_storms` → thêm `from utils.storm import detect_storms`.
Giữ nguyên decorator `@st.cache_data` trên `process_and_load_data`.

### 3.1.3. Rút gọn `app.py`
Đầu file chỉ còn import + điều phối UI:
```python
from utils.data_pipeline import fetch_cds_data, process_and_load_data
from predictor import predict_tomorrow      # dùng ở Giai đoạn 4
```
Xóa các định nghĩa hàm đã chuyển đi. **Không đổi logic UI** ở bước này — refactor thuần túy
để mỗi commit dễ review. Chạy lại `streamlit run app.py` xác nhận dashboard vẫn y hệt.

> Nếu muốn giảm rủi ro, có thể **bỏ qua việc dời file dữ liệu** vào `data/` ở lần đầu (giữ
> `dashboard_main.csv`, `storms.csv` ở gốc) và chỉ tách code. Dời file là tùy chọn.
-> Cứ dời đi

---

## BƯỚC 3.2 — Viết `predictor.py`

File cốt lõi của Giai đoạn 3. Đọc 4 file `_temp`, lấy 30 ngày gần nhất tại (lat, lon),
tái tạo **đúng** pipeline đặc trưng lúc train, rồi trả về nhiệt độ dự đoán ngày mai.

Logic lấy từ `predict_tomorrow` mẫu trong
[models/train_lstm_temperature.py](../models/train_lstm_temperature.py#L337) nhưng được:
(a) **dùng đường dẫn tương đối**, (b) **tách phần load model/dữ liệu để cache** (Bước 3.3),
(c) trả thêm metadata hữu ích cho UI (ngày dự đoán, nhiệt độ hôm nay).

```python
# predictor.py
"""
GIAI ĐOẠN 3 - Hàm dự đoán cho Streamlit.
Đọc 4 file _temp từ models/ và archive/weather.parquet để dự đoán nhiệt độ ngày mai.
"""
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# ---- Đường dẫn TƯƠNG ĐỐI (di động giữa các máy) ----
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "archive", "weather.parquet")
CKPT_PATH  = os.path.join(MODELS_DIR, "lstm_weather_model_temp.pt")
SCALER_PATH       = os.path.join(MODELS_DIR, "scaler_temp.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_cols_temp.pkl")

# Phải khớp với lúc train (xem train_lstm_temperature.py)
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
    """Dựng lại model từ checkpoint. Tách riêng để Bước 3.3 bọc @st.cache_resource."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = LSTMModel(ckpt['input_size'], ckpt['hidden_size'], ckpt['num_layers'],
                      ckpt['output_size'], ckpt['dropout']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    scaler = joblib.load(SCALER_PATH)            # bỏ qua path tuyệt đối trong ckpt
    feat_cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, feat_cols, device


def load_history():
    """Đọc parquet 1 lần. Tách riêng để Bước 3.3 bọc @st.cache_data."""
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
    loc['month_num']   = loc['valid_time'].dt.month
    loc['season'] = loc['month_num'].apply(
        lambda m: 0 if m in (2, 3, 4) else 1 if m in (5, 6, 7) else 2 if m in (8, 9, 10) else 3)
    loc['day_of_week_sin'] = np.sin(2*np.pi*loc['day_of_week']/7)
    loc['day_of_week_cos'] = np.cos(2*np.pi*loc['day_of_week']/7)
    loc['month_sin'] = np.sin(2*np.pi*loc['month_num']/12)
    loc['month_cos'] = np.cos(2*np.pi*loc['month_num']/12)
    loc['day_of_year_sin'] = np.sin(2*np.pi*loc['day_of_year']/365)
    loc['day_of_year_cos'] = np.cos(2*np.pi*loc['day_of_year']/365)

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


def predict_tomorrow(lat, lon):
    """
    Dự đoán nhiệt độ ngày mai (°C) tại (lat, lon).
    Trả về dict: {temperature_celsius, predict_date, coords}
    """
    model, scaler, feat_cols, device = load_model()
    df = load_history()

    # Tọa độ lưới gần nhất
    coords = df[['latitude', 'longitude']].drop_duplicates()
    dists = np.sqrt((coords['latitude']-lat)**2 + (coords['longitude']-lon)**2)
    near = coords.loc[dists.idxmin()]
    loc = df[(df['latitude'] == near['latitude']) & (df['longitude'] == near['longitude'])]

    daily, numeric_feats = _build_features(loc, feat_cols, scaler)
    if len(daily) < SEQUENCE_LENGTH:
        raise ValueError(f"Không đủ {SEQUENCE_LENGTH} ngày dữ liệu cho tọa độ này.")

    seq = daily[feat_cols].values[-SEQUENCE_LENGTH:]
    x = torch.FloatTensor(seq).unsqueeze(0).to(device)
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


if __name__ == '__main__':
    # Test nhanh từ CLI: python predictor.py
    print(predict_tomorrow(21.0, 105.8))   # Hà Nội
```

**Kiểm tra nhanh:** `python predictor.py` phải in ra một dict với nhiệt độ hợp lý
(khoảng 15–35 °C tùy mùa), không lỗi đường dẫn.

---

## BƯỚC 3.3 — Tối ưu bộ nhớ (Caching)

Vấn đề: `load_model()` (đọc checkpoint ~0.9 MB) và `load_history()` (đọc cả parquet) sẽ chạy
**mỗi lần** người dùng bấm nút → web giật. Bọc cache để chỉ chạy 1 lần.

> `predictor.py` cố ý **không import streamlit** để còn test được bằng CLI. Ta thêm một lớp
> cache mỏng ngay trong `app.py` (hoặc `predictor.py` với import streamlit tùy chọn).

Cách gọn nhất — bọc trong `app.py`:
```python
import streamlit as st
import predictor

@st.cache_resource(show_spinner="Đang nạp mô hình AI (chỉ 1 lần)...")
def get_model():
    return predictor.load_model()       # model + scaler + feat_cols nằm trong RAM, dùng lại

@st.cache_data(show_spinner="Đang nạp dữ liệu lịch sử...")
def get_history():
    return predictor.load_history()
```

Rồi viết một `predict` mỏng dùng đối tượng đã cache (không gọi lại `load_*` bên trong):
```python
def predict_cached(lat, lon):
    model, scaler, feat_cols, device = get_model()
    df = get_history()
    # ... lặp lại phần thân predict_tomorrow nhưng dùng model/df đã cache ...
```

Để tránh lặp code, cách sạch hơn: tách phần lõi của `predict_tomorrow` thành
`_predict_core(lat, lon, model, scaler, feat_cols, device, df)` trong `predictor.py`, rồi:
- `predict_tomorrow` gọi `load_model()/load_history()` (cho CLI/test),
- `app.py` gọi `_predict_core(..., *get_model(), get_history())` (cho web, có cache).

**Nguyên tắc bắt buộc:**
- Model/scaler → `@st.cache_resource` (object không serialize được, sống suốt phiên).
- DataFrame parquet → `@st.cache_data` (Streamlit hash & lưu).
- Tuyệt đối **không** đặt `load_model()` trong thân nút bấm/vòng lặp.

---

## BƯỚC 3.4 — Kiểm thử & bàn giao

Checklist trước khi giao cho Giai đoạn 4:
- [ ] `python predictor.py` chạy được trên máy sạch (không lỗi đường dẫn tuyệt đối).
- [ ] `streamlit run app.py` vẫn hiển thị 3 tab cũ y như trước refactor.
- [ ] Gọi `predict_tomorrow` cho vài tọa độ (Hà Nội 21.0/105.8, TP.HCM 10.8/106.7,
      Đà Nẵng 16.0/108.2) đều trả nhiệt độ hợp lý.
- [ ] Lần gọi dự đoán thứ 2 nhanh hẳn (chứng tỏ cache hoạt động).
- [ ] Đã thêm `torch`, `scikit-learn`, `joblib`, `pyarrow`, `matplotlib` vào `requirements.txt`.

**Bàn giao cho Giai đoạn 4 (Frontend):** chỉ cần dùng một hàm duy nhất —
`predict_tomorrow(lat, lon)` (hoặc bản cached trong `app.py`) — trả về:
```python
{'temperature_celsius': 28.3, 'predict_date': '2024-10-15', 'coords': (21.0, 105.75)}
```
Giai đoạn 4 vẽ line chart nét đứt và alert box từ giá trị này.

---

## Phụ lục — Khác biệt cần biết so với PROJECT_WORKFLOW.md
- Workflow gốc nói model dùng `.h5` (Keras) và "7 ngày gần nhất". Thực tế bản `_temp` dùng
  **PyTorch (`.pt`)** và **30 ngày** (`SEQUENCE_LENGTH = 30`). Theo code, không theo văn bản cũ.
- Workflow gốc nói `predict_tomorrow` trả "nhiệt độ **và** lượng mưa". Bản `_temp` hiện chỉ
  train **nhiệt độ**. Nếu cần dự đoán mưa, phải train thêm model riêng (đổi `TARGET_COLS`).
