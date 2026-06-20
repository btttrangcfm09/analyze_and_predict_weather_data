"""pages/yearly.py — Tab phân tích theo năm (không có storm)."""

import plotly.graph_objects as go
import streamlit as st

from components.styles import province_selector
from core.provinces    import province_points


def render_yearly_page(df_main):
    st.title("📆 Yearly Weather Data Analysis")

    years         = list(range(2020, 2027))
    selected_year = st.selectbox("Chọn Năm", years, index=years.index(2024))
    _, lat_sel, lon_sel = province_selector(province_points(), key="yearly_prov")

    df_year = df_main[
        (df_main["year"]      == str(selected_year)) &
        (df_main["latitude"]  == lat_sel) &
        (df_main["longitude"] == lon_sel)
    ]

    if df_year.empty:
        st.warning("Không có dữ liệu cho năm/tỉnh này!")
        return

    df_daily = df_year.groupby("date").agg({
        "temperature_celsius": ["max", "min"],
        "total_precipitation": "sum",
    })
    df_daily.columns       = ["max_temperature", "min_temperature", "total_rain"]
    df_daily["daily_diff"] = df_daily["max_temperature"] - df_daily["min_temperature"]
    df_daily["month"]      = df_daily.index.strftime("%Y-%m")

    df_monthly = df_daily.groupby("month").agg({
        "max_temperature": "max",
        "min_temperature": "min",
        "daily_diff":      "max",
        "total_rain":      "sum",
    }).reset_index()

    max_temp_year   = df_monthly["max_temperature"].max()
    total_rain_year = df_monthly["total_rain"].sum() * 1000
    max_rain_month  = (
        df_monthly.loc[df_monthly["total_rain"].idxmax(), "month"]
        if not df_monthly.empty else "N/A"
    )

    st.markdown("### 📊 Tổng Quan Năm")
    m1, m2, m3 = st.columns(3)
    m1.metric("Nhiệt độ cao nhất",    f"{max_temp_year:.1f} °C")
    m2.metric("Tổng lượng mưa",       f"{total_rain_year:.1f} mm")
    m3.metric("Tháng mưa nhiều nhất", max_rain_month)

    months = df_monthly["month"].tolist()
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=months, y=(df_monthly["total_rain"] * 1000).tolist(),
        name="Lượng mưa (mm)", marker=dict(color="#2196F3"), yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=df_monthly["max_temperature"].tolist(),
        mode="lines+markers", name="Nhiệt độ cao nhất", line=dict(color="#F44336"),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=df_monthly["min_temperature"].tolist(),
        mode="lines+markers", name="Nhiệt độ thấp nhất", line=dict(color="#4CAF50"),
    ))
    fig.add_trace(go.Scatter(
        x=months, y=df_monthly["daily_diff"].tolist(),
        mode="lines+markers", name="Chênh lệch nhiệt độ", line=dict(color="#FF9800"),
    ))
    fig.update_layout(
        title=f"Tổng Hợp Thời Tiết Năm {selected_year}",
        xaxis=dict(title="Tháng"),
        yaxis=dict(title="Nhiệt độ (°C)"),
        yaxis2=dict(title="Lượng mưa (mm)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=1.1, orientation="h"),
        template="plotly_white",
    )

    pie = go.Figure(go.Pie(
        labels=months,
        values=(df_monthly["total_rain"] * 1000).tolist(),
        hole=0.4,
    ))
    pie.update_layout(title="Phân Bố Lượng Mưa Theo Tháng")

    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(pie, width="stretch")
