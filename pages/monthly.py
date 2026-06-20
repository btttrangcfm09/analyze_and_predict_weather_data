"""pages/monthly.py — Tab phân tích theo tháng."""

import plotly.graph_objects as go
import streamlit as st

from components.styles import province_selector
from core.provinces    import province_points


def render_monthly_page(df_main):
    st.title("📅 Monthly Weather Data Analysis")

    years          = list(range(2020, 2027))
    selected_year  = st.selectbox("Chọn Năm",   years, index=years.index(2024))
    selected_month = st.selectbox(
        "Chọn Tháng", range(1, 13), format_func=lambda x: str(x).zfill(2)
    )
    _, lat_sel, lon_sel = province_selector(province_points(), key="monthly_prov")

    month_str = f"{selected_year}-{selected_month:02d}"
    df_month  = df_main[
        (df_main["month"]     == month_str) &
        (df_main["latitude"]  == lat_sel) &
        (df_main["longitude"] == lon_sel)
    ]

    if df_month.empty:
        st.warning("Không có dữ liệu cho tháng/tỉnh này!")
        return

    df_daily = df_month.groupby("date").agg({
        "temperature_celsius": ["max", "min"],
        "total_precipitation": "sum",
    })
    df_daily.columns = ["max_temperature", "min_temperature", "total_precipitation"]
    df_daily         = df_daily.reset_index()

    max_temp   = df_daily["max_temperature"].max()
    total_rain = df_daily["total_precipitation"].sum() * 1000
    max_diff   = (df_daily["max_temperature"] - df_daily["min_temperature"]).max()

    st.markdown("### 📊 Tổng Quan Tháng")
    m1, m2, m3 = st.columns(3)
    m1.metric("Nhiệt độ cao nhất",              f"{max_temp:.1f} °C")
    m2.metric("Tổng lượng mưa",                 f"{total_rain:.1f} mm")
    m3.metric("Chênh lệch nhiệt độ lớn nhất",   f"{max_diff:.1f} °C")

    days = df_daily["date"].dt.strftime("%Y-%m-%d").tolist()
    fig  = go.Figure()
    fig.add_trace(go.Bar(
        x=days, y=(df_daily["total_precipitation"] * 1000).tolist(),
        name="Lượng mưa (mm)", marker=dict(color="#2196F3"), yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=df_daily["max_temperature"].tolist(),
        mode="lines+markers", name="Nhiệt độ cao nhất", line=dict(color="#F44336"),
    ))
    fig.add_trace(go.Scatter(
        x=days, y=df_daily["min_temperature"].tolist(),
        mode="lines+markers", name="Nhiệt độ thấp nhất", line=dict(color="#4CAF50"),
    ))
    fig.update_layout(
        title=f"Tổng Hợp Thời Tiết Tháng ({month_str})",
        xaxis=dict(title="Ngày", tickangle=-45),
        yaxis=dict(title="Nhiệt độ (°C)"),
        yaxis2=dict(title="Lượng mưa (mm)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=1.1, orientation="h"),
        template="plotly_white",
    )
    pie = go.Figure(go.Pie(
        labels=days,
        values=(df_daily["total_precipitation"] * 1000).tolist(),
        hole=0.4, textinfo="none",
    ))
    pie.update_layout(title="Phân Bố Lượng Mưa Theo Ngày")

    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(pie, width="stretch")
