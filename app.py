import streamlit as st
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
import math
import plotly.express as px
import plotly.graph_objects as go
import traceback
import os
import cdsapi
import urllib3
import xarray as xr
import glob
from datetime import datetime, timedelta
import shutil

urllib3.disable_warnings()

st.set_page_config(page_title='Modern Weather Dashboard', layout='wide', page_icon='🌤️')

# ==============================================================
# PHẦN 1: TẢI DỮ LIỆU TỪ CDS API
# ==============================================================
def fetch_cds_data(start_date, end_date):
    os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
    os.environ["CDSAPI_KEY"] = "da23c1f9-21f4-4adc-96af-5e212d2f4440"
    
    client = cdsapi.Client(verify=False)
    dataset = "reanalysis-era5-single-levels"
    os.makedirs("ETL/code", exist_ok=True)
    os.makedirs("crawl_weather_temp", exist_ok=True)
    
    delta = timedelta(days=1)
    
    # --- 1. TẢI WEATHER DATA ---
    st.info(f"Đang tải Dữ liệu Thời tiết từ {start_date.date()} đến {end_date.date()}...")
    all_weather = []
    curr = start_date
    progress_bar = st.progress(0)
    total_days = (end_date - start_date).days + 1
    day_count = 0
    
    while curr <= end_date:
        y, m, d = curr.strftime('%Y'), curr.strftime('%m'), curr.strftime('%d')
        target = f"crawl_weather_temp/weather_{d}_{m}_{y}.grib"
        client.retrieve(dataset, {
            'product_type': ['reanalysis'],
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'mean_sea_level_pressure', 'sea_surface_temperature', 'surface_pressure', 'total_cloud_cover', 'total_column_cloud_ice_water', 'total_column_cloud_liquid_water'],
            'year': [y], 'month': [m], 'day': [d],
            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
            'data_format': 'grib', 'download_format': 'unarchived', 'area': [24, 102, 8, 112]
        }, target)
        
        ds = xr.open_dataset(target)
        all_weather.append(ds.to_dataframe().reset_index())
        curr += delta
        day_count += 1
        progress_bar.progress(day_count / total_days)
        
    pd.concat(all_weather, ignore_index=False).to_csv("ETL/code/weather_data.csv", index=False)
    st.success("Tải xong Dữ liệu Thời tiết!")
    
    # --- 2. TẢI RAIN DATA ---
    st.info(f"Đang tải Dữ liệu Lượng mưa...")
    all_rain = []
    curr = start_date
    progress_bar2 = st.progress(0)
    day_count = 0
    
    while curr <= end_date:
        y, m, d = curr.strftime('%Y'), curr.strftime('%m'), curr.strftime('%d')
        target = f"crawl_weather_temp/rain_{d}_{m}_{y}.grib"
        client.retrieve(dataset, {
            'product_type': ['reanalysis'],
            'variable': ['total_precipitation'],
            'year': [y], 'month': [m], 'day': [d],
            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
            'data_format': 'grib', 'download_format': 'unarchived', 'area': [24, 102, 8, 112]
        }, target)
        
        ds = xr.open_dataset(target)
        all_rain.append(ds.to_dataframe().reset_index())
        curr += delta
        day_count += 1
        progress_bar2.progress(day_count / total_days)
        
    pd.concat(all_rain, ignore_index=False).to_csv("ETL/code/rain_data.csv", index=False)
    st.success("Tải xong Dữ liệu Lượng mưa!")
    
    # Xóa file grib tạm
    shutil.rmtree("crawl_weather_temp", ignore_errors=True)
    st.success("Hoàn tất tải toàn bộ dữ liệu từ vệ tinh Copernicus!")

# ==============================================================
# PHẦN 2: CÁC HÀM XỬ LÝ DỮ LIỆU VÀ NHẬN DIỆN BÃO (TỪ PIPELINE CŨ)
# ==============================================================
def distance_matrix(lons,lats):
    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    d = np.zeros((X,X))
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[c<1,i2] = EARTH_RADIUS * np.arccos(c[c<1])
    return d

def nanmean(array, axis=None):
    return np.mean(np.ma.masked_array(array, np.isnan(array)), axis)

def len_deg_lon(lat):
    R = 6371. # Radius of Earth [km]
    return (np.pi/180.) * R * np.cos( lat * np.pi/180. )

def spatial_filter(msl, lon, lat, res, cut_lon, cut_lat):
    msl_filt = np.zeros(msl.shape)
    sig_lon = (cut_lon/5.) / res
    sig_lat = (cut_lat/5.) / res
    land = np.isnan(msl)
    msl[land] = nanmean(msl)
    msl_filt =  ndimage.gaussian_filter(msl, [sig_lat, sig_lon])
    msl_filt[land] = np.nan
    return msl_filt

def detect_storms(msl, wind_speed, lon, lat, res, order, Npix_min, Npix_max, rel_amp_thresh, d_thresh, cyc, cut_lon, cut_lat, globe=False):
    len_deg_lat = 111.325
    msl = spatial_filter(msl, lon, lat, res, cut_lon, cut_lat)
    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])
    area_storms = np.array([])
    max_wind_speeds = np.array([])
    regions_storm = []
    ssh_crits = np.array([100000]) # 1000hPa

    for ssh_crit in ssh_crits:
        regions, nregions = ndimage.label( (msl<ssh_crit).astype(int) )
        for iregion in range(nregions):
            region = (regions==iregion+1).astype(int)
            region_Npix = region.sum()
            storm_area_within_limits = ((region_Npix >= Npix_min) * (region_Npix <= Npix_max))
            interior = ndimage.binary_erosion(region)
            exterior = np.logical_xor(region.astype(bool), interior)
            if interior.sum() == 0:
                continue
            
            amp_abs = msl[interior].min()
            amp = msl[exterior].mean() - amp_abs
            is_tall_storm = amp >= rel_amp_thresh
            lon_ext = llon[exterior]
            lat_ext = llat[exterior]
            d = distance_matrix(lon_ext, lat_ext)
            is_small_storm = d.max() < d_thresh

            if (storm_area_within_limits * is_tall_storm * is_small_storm):
                storm_object_with_mass = msl * region
                storm_object_with_mass[np.isnan(storm_object_with_mass)] = 0
                min_pressure_pos = np.unravel_index(np.argmin(msl), msl.shape)
                j_min, i_min = min_pressure_pos
                lon_cen = np.interp(i_min, range(0,len(lon)), lon)
                lat_cen = np.interp(j_min, range(0,len(lat)), lat)

                lon_storms = np.append(lon_storms, lon_cen)
                lat_storms = np.append(lat_storms, lat_cen)
                amp_storms = np.append(amp_storms, amp_abs)
                area = region_Npix * res**2 * len_deg_lat * len_deg_lon(lat_cen)
                area_storms = np.append(area_storms, area)
                
                storm_mask = np.ones(msl.shape)
                storm_mask[interior.astype(int)==1] = np.nan
                storm_mask = msl * storm_mask
                region_storm = np.isnan(storm_mask).astype(int)
                regions_storm.append(region_storm)
                max_wind_speed = (wind_speed*region_storm).max()
                max_wind_speeds = np.append(max_wind_speeds, max_wind_speed)

    return lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, regions_storm 

# ==============================================================
# PHẦN 3: TÍCH HỢP PIPELINE VÀO STREAMLIT CACHE
# ==============================================================
@st.cache_data(show_spinner="Đang xử lý dữ liệu từ file thô (chỉ chạy lần đầu)...")
def process_and_load_data(force_reprocess=False):
    if force_reprocess or not os.path.exists("dashboard_main.csv") or not os.path.exists("storms.csv"):
        try:
            df_weather = pd.read_csv("ETL/code/weather_data.csv")
            df_rain = pd.read_csv("ETL/code/rain_data.csv")
        except FileNotFoundError as e:
            st.error(f"Lỗi: Không tìm thấy file dữ liệu gốc ở thư mục ETL/code/. Vui lòng chạy các file crawl dữ liệu trước.")
            st.stop()

        if 'valid_time' in df_rain.columns:
            df_rain = df_rain.drop(columns=['time'], errors='ignore').rename(columns={'valid_time': 'time'})
            df_rain = df_rain[['time', 'latitude', 'longitude', 'tp']]
            
        if 'valid_time' in df_weather.columns:
            df_weather = df_weather.drop(columns=['time'], errors='ignore').rename(columns={'valid_time': 'time'})
            weather_cols = ['time', 'latitude', 'longitude', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sst', 'sp', 'tcc']
            df_weather = df_weather[[c for c in weather_cols if c in df_weather.columns]]
        
        df_joined = pd.merge(df_rain, df_weather, on=['latitude', 'longitude', 'time'], how='inner')
        df_joined = df_joined.drop_duplicates(subset=['latitude', 'longitude', 'time'])
        
        if df_joined.empty:
            st.error("Dữ liệu sau khi gộp bị rỗng. Vui lòng kiểm tra lại ngày tháng của 2 file crawl.")
            st.stop()
            
        df_joined['t2m'] = df_joined['t2m'] - 273.15
        df_joined['d2m'] = df_joined['d2m'] - 273.15
        df_joined['wind_speed'] = np.sqrt(df_joined['u10']**2 + df_joined['v10']**2)
        df_joined['saturation_vapor_pressure'] = 6.11 * (10 ** (7.5 * df_joined['t2m'] / (df_joined['t2m'] + 237.3)))
        df_joined['vapor_pressure'] = 6.11 * (10 ** (7.5 * df_joined['d2m'] / (df_joined['d2m'] + 237.3)))
        df_joined['relative_humidity'] = (df_joined['vapor_pressure'] / df_joined['saturation_vapor_pressure']) * 100
        df_joined['apparent_temperature'] = df_joined['t2m'] + 0.33 * df_joined['vapor_pressure'] - 0.70 * df_joined['wind_speed'] - 4.00
        df_joined['air_density'] = df_joined['sp'] / (287.05 * (df_joined['t2m'] + 273.15))
        df_joined['wind_direction'] = 180 + (180 / math.pi) * np.arctan2(df_joined['u10'], df_joined['v10'])
        df_joined['wind_direction'] = np.where(df_joined['wind_direction'] < 0, df_joined['wind_direction'] + 360, df_joined['wind_direction'])
        
        df_joined = df_joined[(df_joined['latitude'] >= 8) & (df_joined['latitude'] <= 24) & 
                              (df_joined['longitude'] >= 102) & (df_joined['longitude'] <= 112)]
        
        dashboard_df = df_joined.rename(columns={
            'time': 'valid_time',
            'tp': 'total_precipitation',
            't2m': 'temperature_celsius',
            'msl': 'mean_sea_level_pressure',
            'sst': 'sea_surface_temperature_celsius',
            'sp': 'surface_pressure',
            'tcc': 'total_cloud_cover'
        })
        dashboard_df.to_csv("dashboard_main.csv", index=False)
        
        times = sorted(dashboard_df['valid_time'].unique())
        lons = np.arange(102, 112.25, 0.25)
        lats = np.arange(24, 7.75, -0.25)
        
        storm_rows = []
        storm_count = 0
        prev_lon, prev_lat = None, None
        
        for t in times:
            df_t = df_joined[df_joined['time'] == t]
            if df_t.empty: continue
            
            msl_pivot = df_t.pivot(index='latitude', columns='longitude', values='msl')
            ws_pivot = df_t.pivot(index='latitude', columns='longitude', values='wind_speed')
            msl_pivot = msl_pivot.reindex(index=lats, columns=lons).values
            ws_pivot = ws_pivot.reindex(index=lats, columns=lons).values
            
            lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, _ = detect_storms(
                msl_pivot, ws_pivot, lons, lats, res=0.25, order='topdown', Npix_min=9, Npix_max=6000, 
                rel_amp_thresh=100, d_thresh=2500, cyc='cyclonic', cut_lon=1, cut_lat=1, globe=False
            )
            
            if len(lon_storms) > 0:
                for i in range(len(lon_storms)):
                    lon_storm, lat_storm = lon_storms[i], lat_storms[i]
                    if prev_lon is not None and np.sqrt((lon_storm - prev_lon)**2 + (lat_storm - prev_lat)**2) < 1:
                        storm_id = f"2024-{storm_count:02d}"
                    else:
                        storm_count += 1
                        storm_id = f"2024-{storm_count:02d}"
                    prev_lon, prev_lat = lon_storm, lat_storm
                    storm_rows.append({'time': t, 'id': storm_id, 'lon_storm': lon_storm, 'lat_storm': lat_storm,
                                       'amp_storm': amp_storms[i], 'max_wind_speed': max_wind_speeds[i], 'area_storm': area_storms[i]})

        if len(storm_rows) > 0:
            storms_df = pd.DataFrame(storm_rows)
        else:
            storms_df = pd.DataFrame(columns=['time', 'id', 'lon_storm', 'lat_storm', 'amp_storm', 'max_wind_speed', 'area_storm'])
            
        storms_df.to_csv("storms.csv", index=False)

    df_main = pd.read_csv("dashboard_main.csv")
    df_storms = pd.read_csv("storms.csv")
    
    df_main['valid_time_dt'] = pd.to_datetime(df_main['valid_time'])
    df_main['day'] = df_main['valid_time_dt'].dt.date
    df_main['month'] = df_main['valid_time_dt'].dt.strftime('%Y-%m')
    df_main['year'] = df_main['valid_time_dt'].dt.strftime('%Y')
    df_main['hour'] = df_main['valid_time_dt'].dt.strftime('%H:%M')
    
    df_storms['time_dt'] = pd.to_datetime(df_storms['time'])
    df_storms['month'] = df_storms['time_dt'].dt.strftime('%Y-%m')
    df_storms['year'] = df_storms['time_dt'].dt.strftime('%Y')
    df_storms['day'] = df_storms['time_dt'].dt.date
    
    return df_main, df_storms

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
            if os.path.exists("dashboard_main.csv"): os.remove("dashboard_main.csv")
            if os.path.exists("storms.csv"): os.remove("storms.csv")
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
