"""pages/daily.py — Tab phân tích theo ngày, theo giờ cho 34 tỉnh thành."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.styles import province_selector
from core.provinces    import province_points


def render_daily_page(df_main):
    st.title("🌡️ Daily Weather Data Analysis")

    dates = sorted(df_main["date"].unique())
    if not dates:
        st.warning("Không có dữ liệu!")
        st.stop()

    selected_date = st.selectbox(
        "Chọn Ngày", dates, index=len(dates) - 1,
        format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"),
    )
    df_day = df_main[df_main["date"] == selected_date].copy()

    attr_mapping = {
        "Temperature (°C)":              "temperature_celsius",
        "Apparent Temp (°C)":            "apparent_temperature",
        "Mean Sea Level Pressure (hPa)": "mean_sea_level_pressure",
        "Total Cloud Cover (%)":         "total_cloud_cover",
        "Total Precipitation (mm)":      "total_precipitation",
        "Surface Pressure (hPa)":        "surface_pressure",
        "Sea Surface Temp (°C)":         "sea_surface_temperature",
        "Wind Speed (m/s)":              "wind_speed",
        "Relative Humidity (%)":         "relative_humidity",
        "Air Density (kg/m3)":           "air_density",
    }

    selected_attr_name = st.selectbox("Chọn Thuộc Tính Hiển Thị", list(attr_mapping.keys()), index=0)
    attr_col = attr_mapping[selected_attr_name]

    # ── Heatmap animation theo giờ ────────────────────────────────────────────
    st.markdown(f"### Bản Đồ Nhiệt Động: {selected_attr_name} ({selected_date})")
    df_day_sorted = df_day.sort_values("valid_time")

    if attr_col == "total_precipitation":
        df_day_sorted["_plot"] = df_day_sorted[attr_col] * 1000
        color_scale = "Blues"
    elif "temperature" in attr_col:
        df_day_sorted["_plot"] = df_day_sorted[attr_col]
        color_scale = "Turbo"
    else:
        df_day_sorted["_plot"] = df_day_sorted[attr_col]
        color_scale = "Viridis"

    fig_map = px.density_map(
        df_day_sorted,
        lat="latitude", lon="longitude", z="_plot",
        animation_frame="hour",
        radius=20,
        center=dict(lat=16, lon=106), zoom=4,
        map_style="carto-positron",
        color_continuous_scale=color_scale,
        title=f"{selected_attr_name} theo giờ — {selected_date}",
    )
    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_map, width="stretch")

    # ── Biểu đồ từng giờ theo tỉnh thành ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📍 Phân Tích Từng Tỉnh Theo Giờ")
    col1, col2 = st.columns([1, 2])

    with col1:
        _, lat_sel, lon_sel = province_selector(province_points(), key="daily_prov")

    with col2:
        df_point = df_day_sorted[
            (df_day_sorted["latitude"]  == lat_sel) &
            (df_day_sorted["longitude"] == lon_sel)
        ].sort_values("valid_time")

        if not df_point.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_point["hour"], y=df_point["temperature_celsius"],
                mode="lines+markers", name="Nhiệt độ (°C)",
                line=dict(color="red"),
            ))
            fig.add_trace(go.Scatter(
                x=df_point["hour"], y=df_point["apparent_temperature"],
                mode="lines+markers", name="Nhiệt độ cảm nhận (°C)",
                line=dict(color="orange"),
            ))
            fig.add_trace(go.Bar(
                x=df_point["hour"],
                y=df_point["total_precipitation"] * 1000,
                name="Lượng mưa (mm)",
                marker=dict(color="blue"), yaxis="y2",
            ))
            fig.update_layout(
                title=f"Xu Hướng Theo Giờ — {selected_date}",
                xaxis=dict(title="Giờ (0–23)", categoryorder="array",
                           categoryarray=[f"{h:02d}:00" for h in range(24)]),
                yaxis=dict(title="Nhiệt độ (°C)"),
                yaxis2=dict(title="Lượng mưa (mm)", overlaying="y", side="right"),
                legend=dict(x=0.05, y=1.2, orientation="h"),
                template="plotly_white", barmode="relative",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Không có dữ liệu cho tỉnh/tọa độ này trong ngày đã chọn.")
