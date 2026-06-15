"""components/styles.py — CSS và helper UI dùng chung."""

import streamlit as st

SCIFI_CSS = """
<style>
.scifi-container {
    background: linear-gradient(135deg, #0d1117, #161b22, #0f2027);
    border: 1px solid rgba(0, 191, 255, 0.6);
    border-radius: 14px;
    padding: 28px 36px;
    margin: 16px 0;
    box-shadow:
        0 0 20px rgba(0, 191, 255, 0.3),
        0 0 60px rgba(0, 191, 255, 0.1),
        inset 0 0 30px rgba(0, 0, 0, 0.4);
    text-align: center;
    color: #c9d1d9;
    position: relative;
    overflow: hidden;
}
.scifi-container::before {
    content: "";
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(45deg,
        transparent 30%, rgba(0,191,255,0.15) 50%, transparent 70%
    );
    border-radius: 15px;
    z-index: 0;
}
.scifi-province {
    font-size: 1.4em; font-weight: 600; color: #8b949e;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 6px; position: relative; z-index: 1;
}
.scifi-time {
    font-size: 0.95em; color: rgba(0,191,255,0.7);
    margin-bottom: 20px; letter-spacing: 1px;
    position: relative; z-index: 1;
}
.scifi-temp {
    font-size: 5em; font-weight: 900; color: #00bfff;
    text-shadow: 0 0 8px #00bfff, 0 0 20px #00bfff,
        0 0 40px rgba(0,191,255,0.6), 0 0 80px rgba(0,191,255,0.3);
    line-height: 1.1; position: relative; z-index: 1; letter-spacing: -2px;
}
.scifi-unit {
    font-size: 1.5em; color: rgba(0,191,255,0.8);
    vertical-align: super; font-weight: 400;
}
.scifi-label {
    font-size: 0.9em; color: #8b949e; margin-top: 8px;
    letter-spacing: 3px; text-transform: uppercase;
    position: relative; z-index: 1;
}
.scifi-title {
    font-size: 2em; font-weight: 700; color: #00bfff;
    text-shadow: 0 0 10px #00bfff, 0 0 25px rgba(0,191,255,0.5);
    text-align: center; margin-bottom: 6px; letter-spacing: 2px;
}
.scifi-subtitle {
    color: rgba(0,191,255,0.6); text-align: center;
    font-size: 0.9em; margin-bottom: 24px; letter-spacing: 1px;
}
</style>
"""


def inject_scifi_styles() -> None:
    st.markdown(SCIFI_CSS, unsafe_allow_html=True)


def province_selector(provinces: dict, label: str = "📍 Chọn Tỉnh/Thành Phố:", key: str = "prov") -> tuple[str, float, float]:
    """Selectbox chọn tỉnh, trả về (tên, lat, lon)."""
    name = st.selectbox(label, list(provinces.keys()), key=key)
    lat, lon = provinces[name]
    st.info(f"Đang lấy dữ liệu cho: **{name}** (Vĩ độ {lat}, Kinh độ {lon})")
    return name, lat, lon
