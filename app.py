"""
app.py — Router chính. Nhiệm vụ duy nhất: cấu hình trang, cache, điều hướng.

Mọi logic suy luận: core/
Mọi hiển thị giao diện tab: pages/
Mọi CSS/style: components/
"""

import streamlit as st

import core
from pages import ai_prediction, daily, monthly, yearly

# ─── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modern Weather Dashboard",
    layout="wide",
    page_icon="🌤️",
)


# ─── Cache ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang nạp mô hình Seq2Seq LSTM (chỉ 1 lần/phiên)...")
def get_model():
    return core.load_model()


# cache_resource (không pickle) cho khung ~23 triệu dòng → tránh nhân đôi RAM & MemoryError.
# Các trang chỉ ĐỌC df_main (daily lọc rồi .copy()), nên chia sẻ chung một đối tượng là an toàn.
@st.cache_resource(show_spinner="Đang nạp dữ liệu lịch sử thời tiết đất liền...")
def get_history():
    # Đã kèm sẵn cột lịch gọn nhẹ (date/year/month/hour) trong core.load_history().
    return core.load_history()


@st.cache_data(show_spinner="Đang gắn nhãn Tỉnh/Thành cho lưới điểm...")
def get_grid():
    return core.load_grid_points()


def predict_next_week(lat: float, lon: float) -> dict:
    model, scaler, feat_cols, device = get_model()
    df = get_history()
    return core._predict_core(lat, lon, model, scaler, feat_cols, device, df)


# ─── Nạp dữ liệu trước (sinh cache lưới điểm) ─────────────────────────────────
df_main = get_history()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
try:
    st.sidebar.image("utils/678310.png", width="stretch")
except Exception:
    pass

st.sidebar.markdown(
    "<h1 style='text-align:center;color:#1E88E5;'>WEATHER DASHBOARD</h1>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Chọn Loại Phân Tích",
    ["📅 Daily Analysis", "🗓️ Monthly Analysis", "📆 Yearly Analysis", "🤖 AI Weather Prediction"],
)

# Selectbox Tỉnh → điểm lưới chỉ hiện ở tab AI
lat_ai = lon_ai = None
selected_province = selected_point = None
if page == "🤖 AI Weather Prediction":
    grid = get_grid()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Địa Điểm Dự Báo AI")

    selected_province = st.sidebar.selectbox(
        "Chọn Tỉnh/Thành Phố:", sorted(grid["province"].unique()), key="ai_province",
    )
    pts = grid[grid["province"] == selected_province].reset_index(drop=True)
    labels = [f"{r.latitude:.2f}°N, {r.longitude:.2f}°E" for r in pts.itertuples()]
    idx = labels.index(
        st.sidebar.selectbox("Chọn điểm lưới:", labels, key="ai_point")
    )
    lat_ai, lon_ai = float(pts.latitude[idx]), float(pts.longitude[idx])
    selected_point = labels[idx]
    st.sidebar.info(f"📍 **{selected_province}**\nĐiểm lưới: {selected_point}")

# ─── Định tuyến sang page ─────────────────────────────────────────────────────
if page == "📅 Daily Analysis":
    daily.render_daily_page(df_main)

elif page == "🗓️ Monthly Analysis":
    monthly.render_monthly_page(df_main)

elif page == "📆 Yearly Analysis":
    yearly.render_yearly_page(df_main)

elif page == "🤖 AI Weather Prediction":
    ai_prediction.render_ai_page(predict_next_week, selected_province, lat_ai, lon_ai)
