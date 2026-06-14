import torch
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

@st.cache_data(show_spinner="Đang xử lý giao diện hiển thị...")
def get_processed_history():
    df = get_history()  # Streamlit cache đã tự copy an toàn
    
    # Sử dụng numpy/pandas vectorization thay vì strftime (nhanh hơn x10 lần)
    df['day'] = df['date']
    
    # Tạo chuỗi nhanh hơn strftime
    # valid_time là datetime64, lấy năm, tháng, giờ rất nhanh bằng .dt
    dt_accessor = df['valid_time'].dt
    
    # Format YYYY-MM
    df['month'] = dt_accessor.year.astype(str) + '-' + dt_accessor.month.astype(str).str.zfill(2)
    df['year'] = dt_accessor.year.astype(str)
    
    # Format HH:MM (vì dữ liệu là theo giờ tròn)
    df['hour'] = dt_accessor.hour.astype(str).str.zfill(2) + ':00'
    
    return df


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
    st.sidebar.image("utils/678310.png", width='stretch')
    st.sidebar.markdown(
        "<h1 style='text-align: center; color: #1E88E5;'>WEATHER DASHBOARD</h1>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ☁️ Tải Dữ liệu Mới (Crawl)")
    st.sidebar.info("Chọn khoảng ngày để cào dữ liệu từ Copernicus API. Quá trình này sẽ mất một lúc do tải file vệ tinh.")
    with st.sidebar.form("crawl_form"):
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("Từ ngày", value=datetime(2026, 6, 14))
        end_d = col_d2.date_input("Đến ngày", value=datetime(2024, 6, 15))
        submitted = st.form_submit_button("Cào Dữ Liệu Ngay", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Đang tải file từ vệ tinh..."):
            s_dt = datetime.combine(start_d, datetime.min.time())
            e_dt = datetime.combine(end_d, datetime.min.time())
            fetch_cds_data(s_dt, e_dt)
            
            # --- XỬ LÝ LƯU LOCAL PARQUET ---
            from utils.convert_to_parquet import clean_df, compute_derived
            from utils.data_pipeline import WEATHER_CSV, RAIN_CSV
            from predictor import LOCAL_WEATHER_DIR
            
            st.info("Đang xử lý và nối dữ liệu vào kho Parquet local...")
            try:
                # Đảm bảo thư mục local tồn tại bằng cách gọi get_data_path()
                from predictor import get_data_path
                get_data_path()
                
                df_w = pd.read_csv(WEATHER_CSV)
                df_r = pd.read_csv(RAIN_CSV)
                
                # Sửa cột time/valid_time
                for df in [df_w, df_r]:
                    if "time" in df.columns and "valid_time" in df.columns:
                        df.drop(columns=["valid_time"], inplace=True)
                    elif "valid_time" in df.columns:
                        df.rename(columns={"valid_time": "time"}, inplace=True)

                df_merged = pd.merge(df_w, df_r[["latitude", "longitude", "time", "tp"]],
                                     on=["latitude", "longitude", "time"], how="inner")
                df_merged = df_merged.drop_duplicates(subset=["latitude", "longitude", "time"])
                
                df_cleaned = clean_df(df_merged)
                df_final = compute_derived(df_cleaned)
                
                # Lưu vào parquet phân vùng
                os.makedirs(LOCAL_WEATHER_DIR, exist_ok=True)
                for (year, month), df_chunk in df_final.groupby(['year', 'month']):
                    out_dir = os.path.join(LOCAL_WEATHER_DIR, f"year={year}", f"month={month:02d}")
                    os.makedirs(out_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(out_dir, f"update_{ts}.parquet")
                    df_chunk.drop(columns=["year", "month"]).to_parquet(out_path, index=False, compression="snappy")
                
                st.success("Đã nối dữ liệu thành công vào kho lưu trữ!")
                
                st.info("Đang phân tích bão trên dữ liệu mới...")
                from utils.generate_storms import append_new_storms
                append_new_storms(df_final)
                st.success("Đã cập nhật dữ liệu bão!")
            except Exception as e:
                st.error(f"Lỗi khi nối dữ liệu: {e}")
                traceback.print_exc()
                
            # Dọn dẹp CSV tạm
            if os.path.exists(WEATHER_CSV): os.remove(WEATHER_CSV)
            if os.path.exists(RAIN_CSV): os.remove(RAIN_CSV)
            
            # Xóa cache
            get_history.clear()
            get_processed_history.clear()
            process_and_load_data.clear()
            
            import time
            st.success("🎉 Hoàn tất mọi thao tác! Hệ thống sẽ tự làm mới giao diện sau 2 giây...")
            time.sleep(2)
            st.rerun()

    if st.sidebar.button("Đồng bộ dữ liệu lên Kaggle", width='stretch', type="secondary"):
        from utils.kaggle_sync import push_to_kaggle
        from predictor import BASE_DIR
        data_dir = os.path.join(BASE_DIR, "data")
        push_to_kaggle(data_dir)

    st.sidebar.markdown("---")

    page = st.sidebar.radio("Select Analysis Type", ["Daily Analysis", "Monthly Analysis", "Yearly Analysis", "🤖 AI Weather Prediction"])

    VIETNAM_CITIES = {
        'Hà Nội': (21.0, 105.75),
        'Hải Phòng': (20.75, 106.75),
        'Vinh (Nghệ An)': (18.75, 105.75),
        'Huế': (16.5, 107.5),
        'Đà Nẵng': (16.0, 108.25),
        'Nha Trang': (12.25, 109.25),
        'Đà Lạt': (12.0, 108.5),
        'TP. Hồ Chí Minh': (10.75, 106.75),
        'Cần Thơ': (10.0, 105.75)
    }
    
    def render_city_selector(key):
        city = st.selectbox('📍 Chọn Tỉnh/Thành Phố để phân tích:', list(VIETNAM_CITIES.keys()), key=f"city_{key}")
        lat, lon = VIETNAM_CITIES[city]
        st.info(f'Đang lấy dữ liệu cho: {city} (Vĩ độ {lat}, Kinh độ {lon})')
        return lat, lon


    _, df_storms = process_and_load_data()
    
    # Sử dụng bộ dữ liệu 5 năm đã được cache và tiền xử lý
    df_main = get_processed_history()

    attr_mapping = {
        'Temperature (°C)': 'temperature_celsius',
        'Mean Sea Level Pressure (hPa)': 'mean_sea_level_pressure',
        'Total Cloud Cover (%)': 'total_cloud_cover',
        'Total Precipitation (mm)': 'total_precipitation',
        'Surface Pressure (hPa)': 'surface_pressure',
        'Sea Surface Temp (°C)': 'sea_surface_temperature',
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
        df_day = df_day.sort_values(by="valid_time")

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
        st.plotly_chart(fig_map, width='stretch')

        st.markdown("---")

        st.markdown("### 📍 Single Location Analysis")
        col1, col2 = st.columns([1, 2])

        with col1:
            lat_slider, lon_slider = render_city_selector("daily")

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

                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Không có dữ liệu cho tọa độ này!")

        st.markdown("---")
        st.subheader("🌀 Detected Storm Path")
        col3, col4 = st.columns([1, 1])

        with col3:
            if any_storms_detected:
                st.dataframe(storm_df[['time', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed']], width='stretch')
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
                st.plotly_chart(fig_storm, width='stretch')

    elif page == "Monthly Analysis":
        st.title("📅 Monthly Weather Data Analysis")
        years = list(range(2000, 2031))
        selected_year = st.selectbox("Select Year", years, index=years.index(2024))
        selected_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: str(x).zfill(2))

        lat_slider, lon_slider = render_city_selector("monthly")

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
            # Thêm metrics
            max_temp = df_daily['max_temperature'].max()
            total_rain = (df_daily['total_precipitation'].sum()) * 1000
            max_diff = (df_daily['max_temperature'] - df_daily['min_temperature']).max()

            st.markdown("### 📊 Tổng Quan Tháng")
            met1, met2, met3 = st.columns(3)
            met1.metric("Nhiệt độ cao nhất", f"{max_temp:.1f} °C")
            met2.metric("Tổng lượng mưa", f"{total_rain:.1f} mm")
            met3.metric("Chênh lệch nhiệt độ lớn nhất", f"{max_diff:.1f} °C")

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

            st.plotly_chart(fig, width='stretch')
            st.plotly_chart(pie_chart, width='stretch')
        else:
            st.warning("Không có dữ liệu cho tháng/tọa độ này!")

    elif page == "Yearly Analysis":
        st.title("📆 Yearly Weather Data Analysis")
        years = list(range(2000, 2031))
        selected_year = st.selectbox("Select Year", years, index=years.index(2024))

        lat_slider, lon_slider = render_city_selector("monthly")

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

            # Thêm metrics
            max_temp_year = df_monthly['max_temperature'].max()
            total_rain_year = (df_monthly['total_rain'].sum()) * 1000
            max_rain_month = df_monthly.loc[df_monthly['total_rain'].idxmax()]['month'] if not df_monthly.empty else "N/A"
            total_storms_year = sum(storm_counts)

            st.markdown("### 📊 Tổng Quan Năm")
            met1, met2, met3, met4 = st.columns(4)
            met1.metric("Nhiệt độ cao nhất", f"{max_temp_year:.1f} °C")
            met2.metric("Tổng lượng mưa", f"{total_rain_year:.1f} mm")
            met3.metric("Tháng mưa nhiều nhất", max_rain_month)
            met4.metric("Số cơn bão", str(total_storms_year))

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

            st.plotly_chart(fig, width='stretch')
            colA, colB = st.columns(2)
            with colA:
                st.plotly_chart(fig2, width='stretch')
            with colB:
                st.plotly_chart(pie_chart, width='stretch')
        else:
            st.warning("Không có dữ liệu cho năm/tọa độ này!")

    elif page == "🤖 AI Weather Prediction":
        st.markdown("""
        <style>
        .sci-fi-container {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border: 1px solid #0ff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 0 15px #0ff;
            color: #e0f7fa;
            text-align: center;
        }
        .glowing-text {
            color: #0ff;
            text-shadow: 0 0 10px #0ff, 0 0 20px #0ff;
        }
        .metric-box {
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #0ff;
            padding: 15px;
            border-radius: 8px;
            display: inline-block;
            margin-top: 15px;
        }
        .metric-value {
            font-size: 3em;
            font-weight: bold;
            color: #0ff;
            text-shadow: 0 0 10px #0ff;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<h1 class='glowing-text'>🤖 Tương Lai: AI Weather Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<div style='color:#0ff; margin-bottom:20px;'>Sử dụng mạng nơ-ron hồi quy (LSTM) để dự báo nhiệt độ dựa trên chuỗi thời gian 30 ngày.</div>", unsafe_allow_html=True)

        lat_slider, lon_slider = render_city_selector("monthly")

        if st.button("🚀 KÍCH HOẠT MÔ HÌNH DỰ ĐOÁN", type="primary", width='stretch'):
            with st.spinner("Đang khởi tạo Neural Network... Phân tích Pattern..."):
                try:
                    # Gọi hàm predict_tomorrow
                    pred_res = predict_tomorrow(lat_slider, lon_slider)
                    pred_temp = pred_res['temperature_celsius']
                    pred_date = pred_res['predict_date']
                    
                    # Lấy dữ liệu 14 ngày gần nhất
                    df_loc = df_main[(df_main['latitude'] == lat_slider) & (df_main['longitude'] == lon_slider)].copy()
                    if not df_loc.empty:
                        df_loc['date'] = pd.to_datetime(df_loc['valid_time']).dt.date
                        df_daily = df_loc.groupby('date').agg({'temperature_celsius': 'mean'}).reset_index()
                        df_daily = df_daily.sort_values('date').tail(14)
                        
                        dates = df_daily['date'].tolist()
                        temps = df_daily['temperature_celsius'].tolist()
                        
                        # Điểm nối
                        last_date = dates[-1]
                        last_temp = temps[-1]
                        
                        future_dates = [last_date, pd.to_datetime(pred_date).date()]
                        future_temps = [last_temp, pred_temp]
                        
                        fig = go.Figure()
                        # Quá khứ
                        fig.add_trace(go.Scatter(x=dates, y=temps, mode='lines+markers', name='Quá khứ',
                                                 line=dict(color='#0ff', width=3)))
                        # Tương lai (nét đứt)
                        fig.add_trace(go.Scatter(x=future_dates, y=future_temps, mode='lines+markers', name='Dự đoán AI',
                                                 line=dict(color='#f0f', width=4, dash='dash'),
                                                 marker=dict(size=10, symbol='star')))
                        
                        fig.update_layout(
                            title="<span style='color:#0ff'>Đường Cong Nhiệt Độ (14 Ngày Qua + Dự Đoán)</span>",
                            paper_bgcolor='#0f2027',
                            plot_bgcolor='#0f2027',
                            font=dict(color='#0ff'),
                            xaxis=dict(showgrid=True, gridcolor='#203a43'),
                            yaxis=dict(showgrid=True, gridcolor='#203a43', title="Temperature (°C)"),
                            legend=dict(x=0.01, y=0.99)
                        )
                        
                        st.markdown(f"""
                        <div class="sci-fi-container">
                            <h2>Kết quả cho Tọa độ ({lat_slider}, {lon_slider})</h2>
                            <p>Ngày dự báo (T+1): <b>{pred_date}</b></p>
                            <div class="metric-box">
                                <div class="metric-value">{pred_temp:.2f} °C</div>
                                <div>Nhiệt độ Tương lai</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning("Không có dữ liệu quá khứ cho tọa độ này để vẽ biểu đồ.")
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")

except Exception as e:
    st.error(f"Lỗi hệ thống: {str(e)}")
    st.code(traceback.format_exc())
