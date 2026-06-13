# CLAUDE.md

Tài liệu hướng dẫn cho Claude Code (và thành viên mới) khi làm việc trong repo này.

---

## 1. Dự án là gì?

**Analyze & Predict Weather Data** — hệ thống phân tích và dự đoán thời tiết Việt Nam,
gồm 2 khối:

1. **Dashboard phân tích** (`app.py`, Streamlit): tải dữ liệu ERA5 từ Copernicus CDS API,
   xử lý, phát hiện tâm bão/áp thấp và vẽ heatmap/biểu đồ theo Ngày / Tháng / Năm.
2. **Mô hình AI** (`models/`, PyTorch LSTM): học từ dữ liệu lịch sử để **dự đoán nhiệt độ
   ngày mai (T+1)** tại một tọa độ (lat, lon).

Dự án được chia làm 4 giai đoạn, mỗi giai đoạn 1 thành viên — xem
[PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md) (lộ trình gốc) và mục
[§5 Quy trình thực hiện các phần](#5-quy-trình-thực-hiện-các-phần) bên dưới.

> Ngôn ngữ làm việc: **tiếng Việt** (UI, comment, commit message đều tiếng Việt). Giữ nguyên
> phong cách này khi viết code/tài liệu mới.

---

## 2. Cấu trúc thư mục

```
analyze_and_predict_weather_data/
├── app.py                    # Streamlit dashboard (UI + điều phối, đã refactor mỏng)
├── predictor.py              # Giai đoạn 3: predict_tomorrow(lat, lon) + load_model/load_history
├── PROJECT_WORKFLOW.md       # Lộ trình 4 giai đoạn (tài liệu gốc của nhóm)
├── reproduce.md              # Hướng dẫn dựng lại hạ tầng GCP/Mage/Spark/Kafka (bản cũ)
├── requirements.txt          # Phụ thuộc (dashboard + AI/ML)
├── utils/                    # Giai đoạn 3: code tách khỏi app.py
│   ├── storm.py              #   - thuật toán phát hiện bão (NumPy/SciPy)
│   └── data_pipeline.py      #   - fetch_cds_data + process_and_load_data (+ hằng đường dẫn)
├── ETL/code/                 # Giai đoạn 1: crawl dữ liệu
│   ├── crawl_weather.py      #   - cào biến thời tiết (trừ mưa)
│   ├── crawl_rain.py         #   - cào lượng mưa
│   ├── weather_data.csv      #   - dữ liệu thô (đã cào)
│   └── rain_data.csv
│                             # (KHÔNG còn thư mục archive/ — dữ liệu LSTM lấy từ Kaggle)
├── models/                   # Giai đoạn 2: AI/ML
│   ├── train_lstm_temperature.py   # Script train LSTM dự đoán NHIỆT ĐỘ (bản mới nhất)
│   ├── lstm_weather_model_temp.pt  # Checkpoint model + metadata (lưu TÊN FILE tương đối)
│   ├── scaler_temp.pkl             # MinMaxScaler đã fit (26 đặc trưng số)
│   ├── feature_cols_temp.pkl       # Danh sách 27 cột đặc trưng (kèm 'season')
│   ├── training_results_temp.png   # Biểu đồ loss + thực tế vs dự đoán
│   └── train_lstm.py / *.pkl / *.pt  # Bản CŨ (đa biến, giữ để tham khảo)
├── data/                     # Dữ liệu đã xử lý cho dashboard (Giai đoạn 3 dời vào đây)
│   ├── dashboard_main.csv    #   - dữ liệu đã xử lý
│   └── storms.csv            #   - kết quả phát hiện bão (sinh ra từ pipeline)
└── docs/
    ├── PHASE3_BACKEND.md     # *** Quy trình CHI TIẾT Giai đoạn 3 (đọc kỹ) ***
    └── *.png                 # Ảnh kiến trúc, review
```

**Lưu ý về cặp file `_temp`:** Bản train mới nhất (`*_temp.*`) chỉ dự đoán **nhiệt độ**
(`TARGET_COLS = ['temperature_celsius']`) và là bản **được dùng làm đầu vào cho Giai đoạn 3**.
Bản không có hậu tố `_temp` là bản cũ — đừng nhầm.

---

## 3. Lệnh thường dùng

```powershell
# Cài phụ thuộc dashboard
pip install -r requirements.txt

# Chạy dashboard (cổng tùy ý)
streamlit run app.py --server.port 8505

# Train lại mô hình LSTM nhiệt độ (tự tải dữ liệu từ Kaggle, ghi models/*_temp.*)
python models/train_lstm_temperature.py
```

Phụ thuộc cho phần AI đã có sẵn trong `requirements.txt`:
`torch`, `scikit-learn`, `joblib`, `pyarrow`, `matplotlib`.

```powershell
# Test nhanh hàm dự đoán (in nhiệt độ ngày mai cho Hà Nội/TP.HCM/Đà Nẵng)
python predictor.py
```

---

## 4. Kiến trúc dữ liệu & mô hình (cần nắm trước khi sửa)

### 4.1. Luồng dữ liệu
```
Copernicus CDS API
   └─(crawl)→ ETL/code/*.csv  ─(xử lý/đổi tên cột)→ data/dashboard_main.csv  (cho dashboard)

Kaggle dataset (weather.parquet, partition year=/month=)  ──(kagglehub, tự cache)→  LSTM
   nguyentranggggg/vietnam-meteorological-weather-data-2020-2026
```

`dashboard_main.csv` và `weather.parquet` dùng **chung tên cột đã chuẩn hóa**
(`temperature_celsius`, `total_precipitation`, `wind_speed`, `relative_humidity`,
`mean_sea_level_pressure`, `surface_pressure`, `total_cloud_cover`, `apparent_temperature`,
`air_density`, `wind_direction`). Vùng dữ liệu: lat 8–24, lon 102–112 (Việt Nam).

### 4.2. Hợp đồng (contract) của mô hình LSTM — RẤT QUAN TRỌNG
Đây là các hằng số mà bất kỳ code dự đoán nào (Giai đoạn 3) **phải khớp tuyệt đối** với lúc train:

| Thành phần | Giá trị | Nguồn |
|---|---|---|
| Độ dài chuỗi đầu vào | **30 ngày** (không phải 7) | `SEQUENCE_LENGTH = 30` |
| Mục tiêu dự đoán | `temperature_celsius` (T+1) | `TARGET_COLS` |
| Số đặc trưng | 27 cột (gồm `season`); scaler chỉ fit 26 cột số (loại `season`) | `feature_cols_temp.pkl` |
| Tiền xử lý mưa | `np.log1p(total_precipitation)` | bước feature engineering |
| Lag/rolling | lag 1/3/7 và rolling 3/7 cho temp & precip | `make_sequences` |
| Chuẩn hóa | `MinMaxScaler` đã fit sẵn | `scaler_temp.pkl` |
| Kiến trúc | LSTM(128, 2 lớp) → FC(64) → ReLU → Dropout → FC(1) | `LSTMModel` |

Mọi sai khác (ví dụ chỉ lấy 7 ngày, quên `log1p`, đảo thứ tự cột) sẽ cho kết quả vô nghĩa.
Hàm `predict_tomorrow(lat, lon)` mẫu đã có sẵn ở cuối
[models/train_lstm_temperature.py](models/train_lstm_temperature.py#L337) — dùng nó làm chuẩn.

### 4.3. Tính di động (portability) — đã xử lý ở Giai đoạn 3
Checkpoint trước đây lưu **đường dẫn tuyệt đối** tới scaler/feature_cols. Nay đã sửa tận gốc:
checkpoint chỉ lưu **tên file** (`scaler_name`, `feature_cols_name`), và mọi đường dẫn
(`DATA_PATH`, `MODELS_DIR`, …) được resolve **tương đối** theo vị trí file `.py`. `predictor.py`
còn có fallback đọc `scaler_path`/`feature_cols_path` cho checkpoint cũ. Repo chạy được trên
máy khác mà không cần sửa đường dẫn.

---

## 5. Quy trình thực hiện các phần

Tóm tắt từ [PROJECT_WORKFLOW.md](PROJECT_WORKFLOW.md). Trạng thái hiện tại ghi trong ngoặc.

### Giai đoạn 1 — Kỹ sư dữ liệu *(phần lớn đã xong)*
1. Cào dữ liệu lịch sử 5 năm bằng `crawl_weather.py` / `crawl_rain.py` (đổi `start/end_date`).
2. Làm sạch bằng pandas (interpolation / mean imputation).
3. Lưu dạng **Parquet** → đã đẩy lên Kaggle (`weather.parquet`, partition `year=/month=`,
   2020–2026); code dùng `kagglehub` tải/cache, không lưu trong repo.
4. (Tùy chọn) Tự động hóa cào hằng ngày bằng Task Scheduler/Cron.

### Giai đoạn 2 — Kỹ sư AI/ML *(đã xong, ra 4 file `_temp`)*
1. Feature engineering: biến thời gian (sin/cos), mùa, lag 1/3/7, rolling 3/7.
2. Scaling bằng `MinMaxScaler` → `scaler_temp.pkl`.
3. Xây LSTM (PyTorch, thay TensorFlow vì Python 3.14).
4. Train (80%) / Test (20%), chấm bằng RMSE & MAE.
5. Export: `lstm_weather_model_temp.pt`, `scaler_temp.pkl`, `feature_cols_temp.pkl`.
   → **Đây là đầu vào cho Giai đoạn 3.**

### Giai đoạn 3 — Kỹ sư Backend & Tích hợp *(ĐÃ XONG)*
1. ✅ Refactor: tách `app.py` → `utils/storm.py`, `utils/data_pipeline.py`; dời dữ liệu vào `data/`.
2. ✅ `predictor.py` với `predict_tomorrow(lat, lon)` đọc 3 file `_temp` + parquet.
3. ✅ Caching: `@st.cache_resource` (model) + `@st.cache_data` (dữ liệu) trong `app.py`.
4. ✅ Sửa tận gốc đường dẫn checkpoint (xem §4.3).

→ **Quy trình chi tiết, từng bước, kèm code mẫu: [docs/PHASE3_BACKEND.md](docs/PHASE3_BACKEND.md)**

### Giai đoạn 4 — Kỹ sư UI/UX *(sau Giai đoạn 3)*
1. Hoàn thiện tab Monthly/Yearly cho đủ 5 năm + thống kê tóm tắt.
2. Tạo tab mới **"🤖 AI Weather Prediction"** (giao diện hiện đại).
3. Gọi `predict_tomorrow` từ tab này, vẽ line chart nét đứt cho dữ liệu tương lai.
4. Alert box đổi màu (nóng→đỏ, lạnh→xanh; cảnh báo mưa lớn).

---

## 6. Quy ước khi viết code

- **Ngôn ngữ:** comment, label UI, thông báo lỗi → tiếng Việt; tên biến/hàm → tiếng Anh.
- **Streamlit caching:** dữ liệu nặng dùng `@st.cache_data`; tài nguyên (model PyTorch, scaler)
  dùng `@st.cache_resource`. Không bao giờ load model trong thân vòng lặp hay mỗi lần bấm nút.
- **Đường dẫn:** dùng tương đối/`os.path` để chạy được trên máy khác (xem §4.3).
- **Bí mật:** `app.py` đang hardcode `CDSAPI_KEY` (dòng ~26) và `ETL/code/.cdsapirc` chứa key.
  Đừng commit key mới; nếu refactor, chuyển sang biến môi trường / `st.secrets`.
- **Đừng phá hợp đồng mô hình** ở §4.2 khi đụng tới code dự đoán.

---

## 7. Trạng thái Git
- Nhánh chính: `main`.
- File `_temp.*` trong `models/`, `PROJECT_WORKFLOW.md` hiện chưa commit.
- File dữ liệu lớn (`*.csv`, `*.parquet`, `*.grib`) không nên đẩy lên Git — cân nhắc thêm
  vào `.gitignore` trước khi commit.
