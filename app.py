import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import traceback
import os
from datetime import datetime

import predictor
from utils.data_pipeline import (
    fetch_cds_data, process_and_load_data, DASHBOARD_CSV, STORMS_CSV,
)

st.set_page_config(page_title='Modern Weather Dashboard', layout='wide', page_icon='🌤️')

# ==============================================================
# TÍCH HỢP AI (Giai đoạn 3) - CACHING
#   - Mô hình LSTM nạp 1 lần/phiên bằng @st.cache_resource
#   - Dữ liệu lịch sử (parquet) nạp 1 lần bằng @st.cache_data
# ==============================================================
@st.cache_resource(show_spinner="Đang nạp mô hình AI (chỉ 1 lần)...")
def get_model():
    return predictor.load_model()


@st.cache_data(show_spinner="Đang nạp dữ liệu lịch sử cho AI...")
def get_history():
    return predictor.load_history()


def predict_tomorrow(lat, lon):
    """Bản dự đoán có cache cho web: model + dữ liệu chỉ load 1 lần.
    Giai đoạn 4 gọi hàm này để vẽ kết quả AI lên giao diện."""
    model, scaler, feat_cols, device = get_model()
    df = get_history()
    return predictor._predict_core(lat, lon, model, scaler, feat_cols, device, df)

# ==============================================================
# PHẦN 4: GIAO DIỆN DASHBOARD VÀ ĐIỀU KHIỂN
# ==============================================================
try:
    st.sidebar.image("https://th.bing.com/th/id/OIG2.1aEaYj0.6DqDq0a95V7H?pid=ImgGn", use_column_width=True)
    st.sidebar.markdown(
        "<h1 style='text-align: center; color: #1E88E5;'>WEATHER DASHBOARD</h1>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ☁️ Tải Dữ liệu Mới (Crawl)")
    st.sidebar.info("Chọn khoảng ngày để cào dữ liệu từ Copernicus API. Quá trình này sẽ mất một lúc do tải file vệ tinh.")
    col_d1, col_d2 = st.sidebar.columns(2)
    start_d = col_d1.date_input("Từ ngày", value=datetime(2024, 10, 13))
    end_d = col_d2.date_input("Đến ngày", value=datetime(2024, 10, 14))

    if st.sidebar.button("Cào Dữ Liệu Ngay", use_container_width=True, type="primary"):
        with st.spinner(""):
            s_dt = datetime.combine(start_d, datetime.min.time())
            e_dt = datetime.combine(end_d, datetime.min.time())
            fetch_cds_data(s_dt, e_dt)
            # Sau khi tải xong, xóa file CSV kết quả cũ để ép chạy lại pipeline
            if os.path.exists(DASHBOARD_CSV): os.remove(DASHBOARD_CSV)
            if os.path.exists(STORMS_CSV): os.remove(STORMS_CSV)
            process_and_load_data.clear() # Xóa cache của Streamlit
            st.rerun()

    st.sidebar.markdown("---")
    page = st.sidebar.radio("Select Analysis Type", ["Daily Analysis", "Monthly Analysis", "Yearly Analysis"])

    df_main, df_storms = process_and_load_data()

    attr_mapping = {
        'Temperature (°C)': 'temperature_celsius',
        'Mean Sea Level Pressure (hPa)': 'mean_sea_level_pressure',
        'Total Cloud Cover (%)': 'total_cloud_cover',
        'Total Precipitation (mm)': 'total_precipitation',
        'Surface Pressure (hPa)': 'surface_pressure',
        'Sea Surface Temp (°C)': 'sea_surface_temperature_celsius',
        'Wind Speed (m/s)': 'wind_speed',
        'Apparent Temp (°C)': 'apparent_temperature',
        'Relative Humidity (%)': 'relative_humidity',
        'Air Density (kg/m3)': 'air_density'
    }

    if page == "Daily Analysis":
        st.title("🌡️ Daily Weather Data Analysis")

        dates = sorted(df_main['day'].unique())
        if dates:
            selected_date = st.selectbox("Select Date", dates, index=0)
        else:
            st.warning("Không có dữ liệu!")
            st.stop()

        df_day = df_main[df_main['day'] == selected_date].copy()
        storm_df = df_storms[df_storms['day'] == selected_date].copy()
        any_storms_detected = not storm_df.empty

        selected_attr_name = st.selectbox("Select Attribute to View", list(attr_mapping.keys()), index=0)
        attr_col = attr_mapping[selected_attr_name]

        st.markdown(f"### Heatmap Animation: {selected_attr_name} ({selected_date})")
        df_day = df_day.sort_values(by="valid_time_dt")

        if attr_col == 'total_precipitation':
            df_day['Plot Value'] = df_day[attr_col] * 1000  # Convert to mm
            color_scale = "Blues"
        elif 'temperature' in attr_col:
            df_day['Plot Value'] = df_day[attr_col]
            color_scale = "Turbo"
        else:
            df_day['Plot Value'] = df_day[attr_col]
            color_scale = "Viridis"

        fig_map = px.density_mapbox(
            df_day,
            lat='latitude', lon='longitude', z='Plot Value',
            animation_frame='hour',
            radius=20,
            center=dict(lat=16, lon=106), zoom=4,
            mapbox_style="carto-positron",
            color_continuous_scale=color_scale,
            title=f"Animated {selected_attr_name} over Vietnam"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("---")

        st.markdown("### 📍 Single Location Analysis")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.info("Select coordinates to view the daily trend.")
            lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25)
            lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25)

        with col2:
            df_point = df_day[(df_day['latitude'] == lat_slider) & (df_day['longitude'] == lon_slider)]

            if not df_point.empty:
                df_point = df_point.sort_values('valid_time')

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_point["hour"], y=df_point["temperature_celsius"], mode='lines+markers', name='Temperature (°C)', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df_point["hour"], y=df_point["apparent_temperature"], mode='lines+markers', name='Apparent Temp (°C)', line=dict(color='orange')))
                fig.add_trace(go.Bar(x=df_point["hour"], y=df_point["total_precipitation"]*1000, name='Precipitation (mm)', marker=dict(color='blue'), yaxis='y2'))

                fig.update_layout(
                    title=f"Trends at Lat: {lat_slider}, Lon: {lon_slider}",
                    xaxis=dict(title="Time (Hour)"),
                    yaxis=dict(title=dict(text="Temperature (°C)", font=dict(color="black")), tickfont=dict(color="black")),
                    yaxis2=dict(title=dict(text="Rainfall (mm)", font=dict(color="blue")), tickfont=dict(color="blue"), overlaying='y', side='right'),
                    legend=dict(x=0.05, y=1.2, orientation='h'),
                    template="plotly_white", barmode='relative'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không có dữ liệu cho tọa độ này!")

        st.markdown("---")
        st.subheader("🌀 Detected Storm Path")
        col3, col4 = st.columns([1, 1])

        with col3:
            if any_storms_detected:
                st.dataframe(storm_df[['time', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed']], use_container_width=True)
            else:
                st.success("Không phát hiện tâm bão/áp thấp trong ngày hôm nay.")

        with col4:
            if any_storms_detected:
                valid_storms = storm_df.dropna(subset=['lat_storm', 'lon_storm']).copy()
                valid_storms['hour'] = valid_storms['time_dt'].dt.strftime('%H:%M')

                fig_storm = go.Figure()

                fig_storm.add_trace(go.Scattermapbox(
                    mode="lines",
                    lon=valid_storms['lon_storm'],
                    lat=valid_storms['lat_storm'],
                    marker={'size': 10},
                    line=dict(width=2, color='red'),
                    name="Storm Path"
                ))

                fig_storm.add_trace(go.Scattermapbox(
                    mode="markers+text",
                    lon=valid_storms['lon_storm'],
                    lat=valid_storms['lat_storm'],
                    text=valid_storms['hour'],
                    textposition="top right",
                    marker={'size': 12, 'color': 'red', 'symbol': 'circle'},
                    name="Storm Centers"
                ))

                fig_storm.update_layout(
                    mapbox=dict(
                        style="carto-positron",
                        center=dict(lat=16, lon=106),
                        zoom=4
                    ),
                    margin={"r":0,"t":0,"l":0,"b":0},
                    showlegend=False
                )
                st.plotly_chart(fig_storm, use_container_width=True)

    elif page == "Monthly Analysis":
        st.title("📅 Monthly Weather Data Analysis")
        years = list(range(2000, 2031))
        selected_year = st.selectbox("Select Year", years, index=years.index(2024))
        selected_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: str(x).zfill(2))

        lat, lon = st.columns([1, 1])
        with lat:
            lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25)
        with lon:
            lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25)

        month_str = f"{selected_year}-{selected_month:02d}"
        df_month = df_main[(df_main['month'] == month_str) &
                           (df_main['latitude'] == lat_slider) &
                           (df_main['longitude'] == lon_slider)]

        if not df_month.empty:
            df_daily = df_month.groupby('day').agg({
                'temperature_celsius': ['max', 'min'],
                'total_precipitation': 'sum'
            })
            df_daily.columns = ['max_temperature', 'min_temperature', 'total_precipitation']
            df_daily = df_daily.reset_index()

            df_dict = {
                "day": df_daily['day'].astype(str).tolist(),
                "max_tem": df_daily['max_temperature'].tolist(),
                "min_tem": df_daily['min_temperature'].tolist(),
                "tp": (df_daily['total_precipitation']*1000).tolist()
            }
        else:
            df_dict = {"day":[], "max_tem":[], "min_tem":[], "tp":[]}

        if df_dict["day"]:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_dict["day"], y=df_dict["tp"], name='Total Precipitation', marker=dict(color='#2196F3'), yaxis='y2'))
            fig.add_trace(go.Scatter(x=df_dict["day"], y=df_dict["max_tem"], mode='lines+markers', name='Max Temperature', line=dict(color='#F44336')))
            fig.add_trace(go.Scatter(x=df_dict["day"], y=df_dict["min_tem"], mode='lines+markers', name='Min Temperature', line=dict(color='#4CAF50')))

            fig.update_layout(
                title=f"Monthly Weather Summary ({month_str})",
                xaxis=dict(title="Date", tickangle=-45),
                yaxis=dict(title="Temperature (°C)"),
                yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'),
                legend=dict(x=0.01, y=1.1, orientation='h'),
                template="plotly_white"
            )

            pie_chart = go.Figure(go.Pie(labels=df_dict["day"], values=df_dict["tp"], hole=0.4, textinfo='none'))
            pie_chart.update_layout(title="Rainfall Distribution by Day")

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.warning("Không có dữ liệu cho tháng/tọa độ này!")

    elif page == "Yearly Analysis":
        st.title("📆 Yearly Weather Data Analysis")
        years = list(range(2000, 2031))
        selected_year = st.selectbox("Select Year", years, index=years.index(2024))

        lat, lon = st.columns([1,1])
        with lat:
            lat_slider = st.slider("Select Latitude", 8.0, 24.0, 16.0, step=0.25)
        with lon:
            lon_slider = st.slider("Select Longitude", 102.0, 112.0, 106.0, step=0.25)

        df_year = df_main[(df_main['year'] == str(selected_year)) &
                          (df_main['latitude'] == lat_slider) &
                          (df_main['longitude'] == lon_slider)]

        if not df_year.empty:
            df_daily = df_year.groupby('day').agg({
                'temperature_celsius': ['max', 'min'],
                'total_precipitation': 'sum'
            })
            df_daily.columns = ['max_temperature', 'min_temperature', 'total_rain']
            df_daily['daily_difference'] = df_daily['max_temperature'] - df_daily['min_temperature']
            df_daily['month'] = pd.to_datetime(df_daily.index).strftime('%Y-%m')

            df_monthly = df_daily.groupby('month').agg({
                'max_temperature': 'max',
                'min_temperature': 'min',
                'daily_difference': 'max',
                'total_rain': 'sum'
            }).reset_index()

            df_dict = {
                "month": df_monthly['month'].tolist(),
                "max_tem": df_monthly['max_temperature'].tolist(),
                "min_tem": df_monthly['min_temperature'].tolist(),
                "max_dif_tem": df_monthly['daily_difference'].tolist(),
                "tp": (df_monthly['total_rain']*1000).tolist()
            }

            storm_year = df_storms[df_storms['year'] == str(selected_year)]
            storms = storm_year['month'].value_counts().to_dict()
            all_months = [f"{selected_year}-{str(m).zfill(2)}" for m in range(1, 13)]
            storm_counts = [storms.get(m, 0) for m in all_months]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_dict["month"], y=df_dict["tp"], name='Total Precipitation', marker=dict(color='#2196F3'), yaxis='y2'))
            fig.add_trace(go.Scatter(x=df_dict["month"], y=df_dict["max_tem"], mode='lines+markers', name='Max Temperature', line=dict(color='#F44336')))
            fig.add_trace(go.Scatter(x=df_dict["month"], y=df_dict["min_tem"], mode='lines+markers', name='Min Temperature', line=dict(color='#4CAF50')))
            fig.add_trace(go.Scatter(x=df_dict["month"], y=df_dict["max_dif_tem"], mode='lines+markers', name='Temp Difference', line=dict(color='#FF9800')))

            fig.update_layout(
                title=f"Yearly Weather Summary ({selected_year})",
                xaxis=dict(title="Month"),
                yaxis=dict(title="Temperature (°C)"),
                yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'),
                legend=dict(x=0.01, y=1.1, orientation='h'),
                template="plotly_white"
            )

            fig2 = go.Figure(go.Bar(x=all_months, y=storm_counts, name='Number of Storms', marker_color='#9C27B0'))
            fig2.update_layout(title='Number of Storms by Month', xaxis_title='Month', yaxis_title='Count', template='plotly_white')

            pie_chart = go.Figure(go.Pie(labels=df_dict["month"], values=df_dict["tp"], hole=0.4))
            pie_chart.update_layout(title="Rainfall Distribution by Month")

            st.plotly_chart(fig, use_container_width=True)
            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(fig2, use_container_width=True)
            with colB:
                st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.warning("Không có dữ liệu cho năm/tọa độ này!")

except Exception as e:
    st.error(f"Lỗi hệ thống: {str(e)}")
    st.code(traceback.format_exc())
