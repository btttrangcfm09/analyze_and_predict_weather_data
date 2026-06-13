# QUY TRÌNH TRIỂN KHAI DỰ ÁN: ANALYZE & PREDICT WEATHER DATA
*Tài liệu này là lộ trình từng bước (Step-by-Step) dành cho nhóm 4 người để hoàn thiện toàn bộ hệ thống từ lấy dữ liệu đến dự đoán AI và hiển thị Web.*

---

## GIAI ĐOẠN 1: KỸ SƯ DỮ LIỆU (DATA ENGINEERING)
**👉 Người phụ trách: Thành viên 1**
**Mục tiêu:** Cung cấp "thức ăn" (dữ liệu) dồi dào và sạch sẽ cho cả đội.

- [ ] **Bước 1.1 - Cào Dữ Liệu Lịch Sử:** Sửa lại file `crawl_weather.py` và `crawl_rain.py`, thay đổi `start_date` và `end_date` để tải về toàn bộ dữ liệu thời tiết Việt Nam trong **5 năm qua (ví dụ: 01/01/2019 - 31/12/2023)**. Lưu ý: có thể chia nhỏ ra tải từng năm để tránh đứt kết nối API.
- [ ] **Bước 1.2 - Làm Sạch Dữ Liệu (Data Cleaning):** Viết script dùng thư viện `pandas` để kiểm tra toàn bộ data vừa tải. Xử lý các dòng bị lỗi (Missing values) bằng phương pháp nội suy (Interpolation) hoặc điền giá trị trung bình (Mean imputation).
- [ ] **Bước 1.3 - Chuyển Đổi Lưu Trữ:** File CSV 5 năm sẽ cực kỳ nặng (hàng GB). Chuyển đổi định dạng lưu trữ từ `.csv` sang file **Parquet** (`.parquet`) hoặc đưa vào Database mini như **SQLite**. Tốc độ đọc của Web sẽ nhanh gấp 10 lần.
- [ ] **Bước 1.4 - Tự Động Hóa (Tùy chọn):** Viết 1 file script nhỏ và dùng Cronjob (hoặc Windows Task Scheduler) để cứ 12h đêm tự động kích hoạt tải dữ liệu thời tiết của ngày hôm qua đắp vào kho lưu trữ.

---

## GIAI ĐOẠN 2: KỸ SƯ AI & MACHINE LEARNING
**👉 Người phụ trách: Thành viên 2**
**Mục tiêu:** Tạo ra "Bộ não" thông minh (Mô hình LSTM) để dự đoán tương lai.

- [ ] **Bước 2.1 - Feature Engineering (Tạo đặc trưng):** Nhận kho dữ liệu Parquet từ Thành viên 1. Tạo thêm các cột mới: 
  - Biến thời gian: Ngày trong tuần, Tháng, Mùa (xuân/hạ/thu/đông).
  - Biến độ trễ (Lag): Nhiệt độ/Lượng mưa của 1 ngày trước, 3 ngày trước, 7 ngày trước.
- [ ] **Bước 2.2 - Chuẩn hóa (Scaling):** Tuyệt đối không quên dùng `MinMaxScaler` hoặc `StandardScaler` (từ thư viện `scikit-learn`) ép tất cả dữ liệu về khoảng [0, 1] để mô hình LSTM không bị nổ gradient.
- [ ] **Bước 2.3 - Xây Dựng LSTM:** Dùng `TensorFlow/Keras` hoặc `PyTorch` viết kiến trúc mạng LSTM (ví dụ: 1 lớp Input -> 2 lớp LSTM -> 1 lớp Dense/Output). Target là Nhiệt độ hoặc Lượng mưa của ngày mai (T+1).
- [ ] **Bước 2.4 - Huấn luyện & Backtesting:** Chia data thành Tập Train (2019-2022) và Tập Test (2023). Dạy mô hình bằng tập Train và chấm điểm nó bằng tập Test (Dùng hàm RMSE, MAE).
- [ ] **Bước 2.5 - Đóng gói AI:** Sau khi ưng ý với độ chính xác, export mô hình ra thành 1 file (ví dụ: `lstm_weather_model.h5` và cục `scaler.pkl`) để gửi cho Thành viên 3.

---

## GIAI ĐOẠN 3: KỸ SƯ BACKEND & TÍCH HỢP
**👉 Người phụ trách: Thành viên 3**
**Mục tiêu:** Lắp ráp mô hình AI vào trang Web và tối ưu hệ thống chạy mượt mà.

- [ ] **Bước 3.1 - Tái cấu trúc (Refactor):** Không nhét chung mọi thứ vào `app.py`. Tách code ra thành các thư mục:
  - `/data` (Chứa file dữ liệu của TV 1)
  - `/models` (Chứa file `.h5` và `.pkl` của TV 2)
  - `/utils` (Chứa các hàm tính toán bão, hàm helper)
- [ ] **Bước 3.2 - Viết Hàm Dự Đoán (`predictor.py`):** Viết 1 function tên là `predict_tomorrow(lat, lon)`. Hàm này sẽ đọc file `.h5`, lôi dữ liệu 7 ngày gần nhất của tọa độ (lat, lon) đó ra, đưa qua mô hình, và `return` ra con số nhiệt độ dự đoán ngày mai.
- [ ] **Bước 3.3 - Tối ưu Bộ Nhớ (Caching):** Tích hợp kỹ thuật `@st.cache_resource` của Streamlit để mô hình AI chỉ load đúng 1 lần khi bật Web lên, chứ không load lại mỗi khi người dùng bấm nút, giúp trang Web phản hồi ngay tức khắc (Real-time).

---

## GIAI ĐOẠN 4: KỸ SƯ GIAO DIỆN UI/UX (FRONTEND)
**👉 Người phụ trách: Thành viên 4**
**Mục tiêu:** Mặc áo mới cho Web, thiết kế Tab hiển thị kết quả của AI khiến người xem WOW.

- [ ] **Bước 4.1 - Nâng cấp "Monthly/Yearly Analytic":** Sau khi TV 1 đổ data 5 năm vào hệ thống, kiểm tra lại 2 Tab này để đảm bảo biểu đồ vẽ ra liền mạch, đẹp mắt cho toàn bộ các năm và các tháng. Bổ sung các thông số tóm tắt (Tổng mưa trong năm lớn nhất, Ngày nóng nhất...).
- [ ] **Bước 4.2 - Tạo Tab Mới "🤖 AI Weather Prediction":** Code thêm 1 Tab hoàn toàn mới trên Sidebar. Trong Tab này thiết kế giao diện theo hướng tương lai (Sci-fi/Modern).
- [ ] **Bước 4.3 - Gọi AI lên Giao Diện:** Khi người dùng chọn tọa độ trên bản đồ và bấm nút "Dự Đoán", Web sẽ gọi hàm `predict_tomorrow` (của TV 3), lấy kết quả và vẽ lên một biểu đồ Đường nét đứt (Line chart) để biểu thị đây là dữ liệu của Tương lai.
- [ ] **Bước 4.4 - Giao Diện Cảnh Báo (Alerts):** Đổi màu sắc sinh động (nóng -> đỏ, lạnh -> xanh). Thêm khung thông báo (Alert Box) ví dụ: *"🔴 Cảnh báo: AI dự đoán ngày mai sẽ có mưa rất to (>100mm) tại khu vực này!"*.

---

## 🚀 KẾ HOẠCH GIT & HỢP TÁC (QUAN TRỌNG)
- **TUẦN 1:** Thành viên 1 và 2 làm độc lập trên thư mục riêng (Jupyter Notebook) để tạo Data và code AI. Thành viên 3 và 4 làm độc lập trên file `app.py` với data giả (Mock data).
- **TUẦN 2:** Thành viên 2 chuyển file AI cho Thành viên 3. Thành viên 1 đẩy toàn bộ CSDL cho Thành viên 3. Thành viên 3 tích hợp mọi thứ lại.
- **TUẦN 3:** Thành viên 4 đắp giao diện cuối cùng, chải chuốt đồ thị. Cả team Review, bắt Bugs và Chuẩn bị Slide báo cáo.
