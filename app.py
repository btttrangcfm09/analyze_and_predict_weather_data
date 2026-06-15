"""
app.py — Router chính. Nhiệm vụ duy nhất: cấu hình trang, cache, điều hướng.

Mọi logic suy luận: core/
Mọi hiển thị giao diện tab: pages/
Mọi CSS/style: components/
"""

import streamlit as st

import core
from core.features import VIETNAM_PROVINCES
from pages import ai_prediction, daily, monthly, yearly

# ─── Cấu hình trang ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modern Weather Dashboard",
    layout="wide",
    page_icon="🌤️",
)


# ─── Cache ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang nạp mô hình AI LSTM (chỉ 1 lần/phiên)...")
def get_model():
    return core.load_model()


@st.cache_data(show_spinner="Đang nạp dữ liệu lịch sử thời tiết...")
def get_history():
    return core.load_history()


@st.cache_data(show_spinner="Đang chuẩn bị dữ liệu phân tích...")
def get_processed_history():
    df = get_history()
    df = df.copy()
    dt          = df["valid_time"].dt
    df["month"] = dt.year.astype(str) + "-" + dt.month.astype(str).str.zfill(2)
    df["year"]  = dt.year.astype(str)
    df["hour"]  = dt.hour.astype(str).str.zfill(2) + ":00"
    return df


def predict_next_hour(lat: float, lon: float) -> dict:
    model, scaler, feat_cols, device = get_model()
    df = get_history()
    return core._predict_core(lat, lon, model, scaler, feat_cols, device, df)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
try:
    st.sidebar.image("utils/678310.png", use_container_width=True)
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

# Selectbox tỉnh/thành chỉ hiện khi ở tab AI
selected_province = None
lat_ai = lon_ai = None
if page == "🤖 AI Weather Prediction":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Địa Điểm Dự Báo AI")
    selected_province = st.sidebar.selectbox(
        "Chọn Tỉnh/Thành Phố:",
        list(VIETNAM_PROVINCES.keys()),
        key="ai_province",
    )
    lat_ai, lon_ai = VIETNAM_PROVINCES[selected_province]
    st.sidebar.info(f"📍 **{selected_province}**\nVĩ độ: {lat_ai} | Kinh độ: {lon_ai}")

# ─── Định tuyến sang page ─────────────────────────────────────────────────────
df_main = get_processed_history()

if page == "📅 Daily Analysis":
    daily.render_daily_page(df_main)

elif page == "🗓️ Monthly Analysis":
    monthly.render_monthly_page(df_main)

elif page == "📆 Yearly Analysis":
    yearly.render_yearly_page(df_main)

elif page == "🤖 AI Weather Prediction":
    ai_prediction.render_ai_page(predict_next_hour, selected_province, lat_ai, lon_ai)
