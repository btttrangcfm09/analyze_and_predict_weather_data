# 🌤️ Modern Weather Dashboard — Dự báo nhiệt độ 7 ngày bằng Seq2Seq LSTM

Ứng dụng Streamlit phân tích thời tiết Việt Nam và **dự báo nhiệt độ 168 giờ (7 ngày)
liên tiếp** cho từng điểm lưới đất liền, dùng mô hình **Seq2Seq LSTM Encoder-Decoder**.

## ✨ Tính năng

- **Daily / Monthly / Yearly Analysis** — trực quan hóa lịch sử thời tiết.
- **AI Weather Prediction** — dự báo sóng nhiệt độ ngày/đêm 7 ngày tới:
  - 3 khối chỉ số: nhiệt độ trung bình / cao nhất / thấp nhất tuần tới.
  - Đồ thị Plotly nối liền 72 giờ thực tế + 168 giờ dự báo (đường hồng nét đứt).
  - Bảng chi tiết 168 giờ, cho phép tải CSV.
- **Lưới đất liền 500+ điểm** lọc từ ranh giới GADM, chọn theo **Tỉnh/Thành → điểm lưới**.

## 🧠 Kiến trúc mô hình

`forward(x, y_feat, y=None, teacher_forcing=True)`:

| Tensor   | Shape           | Ý nghĩa                                              |
|----------|-----------------|-----------------------------------------------------|
| `x`      | `(B, 168, 30)`  | 168 giờ quá khứ × 30 đặc trưng encoder              |
| `y_feat` | `(B, 168, 10)`  | 168 giờ tương lai × 10 đặc trưng decoder (đã-biết)  |
| output   | `(B, 168)`      | 168 nhiệt độ dự báo                                 |

Suy luận web chạy **autoregressive** (`teacher_forcing=False`): decoder dùng chính
dự đoán bước trước, kết hợp đặc trưng lịch/khí hậu tương lai để giữ dao động ngày/đêm.

## 📁 Cấu trúc

```
app.py                     # Router Streamlit: cache + sidebar + định tuyến
core/
  model.py                 # Seq2Seq LSTMModel
  features.py              # Hằng số: SEQUENCE_LENGTH=168, DECODER_FEATURE_COLS
  feature_engineering.py   # 30 đặc trưng encoder (khớp lúc train)
  future_features.py       # 10 đặc trưng decoder cho 168 giờ tương lai
  geo.py                   # Lọc toạ độ đất liền (GADM L0 + buffer 0.05)
  provinces.py             # Spatial join GADM L1 → gắn tên Tỉnh cho điểm lưới
  loader.py                # Stream archive.zip → cache lưới đất liền
  inference.py             # load_model() + _predict_core() autoregressive
pages/                     # daily / monthly / yearly / ai_prediction
components/styles.py       # CSS sci-fi
lstm_temp_weekly/          # Pipeline huấn luyện + checkpoint/scaler trong results/
```

## 🚀 Chạy

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8505
```

### Dữ liệu cần đặt trong `data/`
- `archive.zip` — dữ liệu thời tiết Parquet (2020–2026, ~hàng nghìn điểm lưới).
- `gadm41_VNM.gpkg` — ranh giới hành chính (layer 0 = cả nước, layer 1 = tỉnh).

### Mô hình (đã có sẵn)
`lstm_temp_weekly/results/`: `lstm_temp_weekly_model.pt`, `scaler_temp.pkl`,
`feature_cols_temp.pkl`.

> **Lần chạy đầu** sẽ stream `archive.zip` trong RAM (không giải nén), lọc lưới đất liền
> và lưu cache `data/weather_land_VNM.parquet` + `data/grid_points_VNM.parquet`.
> Có thể mất vài phút; các lần sau đọc cache nên rất nhanh.
