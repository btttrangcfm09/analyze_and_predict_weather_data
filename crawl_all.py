"""
crawl_all.py
Tải toàn bộ dữ liệu thời tiết ERA5 khu vực Việt Nam từ 2021 đến 2026.
Cách chạy: python crawl_all.py

- Tải theo từng tháng, lưu ra ổ D NGAY sau mỗi tháng (tiết kiệm RAM)
- Tự ngăn Windows Sleep trong khi chạy (màn hình tắt vẫn OK)
- Mỗi tháng append vào file yearly: weather_YYYY.csv và rain_YYYY.csv
"""

import os
import ctypes
import cdsapi
import xarray as xr
import pandas as pd
import urllib3

urllib3.disable_warnings()

# ── Ngăn Windows Sleep (màn hình vẫn có thể tắt, máy không ngủ) ──────────────
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED
ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)

# ── Cấu hình ─────────────────────────────────────────────────────────────────
os.environ["CDSAPI_URL"] = "https://cds.climate.copernicus.eu/api"
os.environ["CDSAPI_KEY"] = "da23c1f9-21f4-4adc-96af-5e212d2f4440"

OUTPUT_DIR = r"d:\2025.2\analyze_and_predict_weather_data\ETL\code"
AREA       = [24, 102, 8, 112]  # [North, West, South, East] — Việt Nam
DAYS       = [f"{d:02d}" for d in range(1, 32)]
HOURS      = [f"{h:02d}:00" for h in range(24)]

WEATHER_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "total_cloud_cover",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
]

# Lịch tải: năm → danh sách tháng cần tải
# 2021-2024 đã tải đủ
# ERA5 có độ trễ ~5 ngày, nên 2026 chỉ tải đến tháng 5 cho chắc
SCHEDULE = {
    "2020": [f"{m:02d}" for m in range(1, 13)],
    "2025": [f"{m:02d}" for m in range(1, 13)],
    "2026": [f"{m:02d}" for m in range(1, 6)],  # Jan–May 2026
}

# ── Hàm tải 1 tháng ──────────────────────────────────────────────────────────
def fetch_month(client, dataset, variable_list, year, month, tmp_path):
    client.retrieve(dataset, {
        "product_type": ["reanalysis"],
        "variable":     variable_list,
        "year":         [year],
        "month":        [month],
        "day":          DAYS,
        "time":         HOURS,
        "data_format":  "grib",
        "download_format": "unarchived",
        "area":         AREA,
    }, tmp_path)
    ds = xr.open_dataset(tmp_path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    df = ds.to_dataframe().reset_index()
    ds.close()
    os.remove(tmp_path)
    return df

# ── Vòng lặp chính ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
client  = cdsapi.Client(verify=False)
dataset = "reanalysis-era5-single-levels"

for year, months in SCHEDULE.items():
    print(f"\n{'='*55}")
    print(f"  NĂM {year}  —  {len(months)} tháng")
    print(f"{'='*55}")

    out_w = os.path.join(OUTPUT_DIR, f"weather_{year}.csv")
    out_r = os.path.join(OUTPUT_DIR, f"rain_{year}.csv")

    # Xóa file cũ nếu tồn tại (bắt đầu năm mới)
    for f in [out_w, out_r]:
        if os.path.exists(f):
            os.remove(f)

    for month in months:
        print(f"\n  [{year}-{month}] Đang tải WEATHER...", flush=True)
        tmp = os.path.join(OUTPUT_DIR, "_tmp_weather.grib")
        df_w = fetch_month(client, dataset, WEATHER_VARS, year, month, tmp)
        # Ghi ngay ra ổ D (append nếu đã có header, write header nếu chưa)
        df_w.to_csv(out_w, mode="a", header=not os.path.exists(out_w), index=False)
        del df_w  # Giải phóng RAM ngay

        print(f"  [{year}-{month}] Đang tải RAIN...", flush=True)
        tmp = os.path.join(OUTPUT_DIR, "_tmp_rain.grib")
        df_r = fetch_month(client, dataset, ["total_precipitation"], year, month, tmp)
        df_r.to_csv(out_r, mode="a", header=not os.path.exists(out_r), index=False)
        del df_r  # Giải phóng RAM ngay

        print(f"  [{year}-{month}] ✓ Đã ghi ra ổ D", flush=True)

    print(f"\n  ✅ Hoàn tất năm {year}: weather_{year}.csv  +  rain_{year}.csv")

# ── Khôi phục trạng thái Sleep bình thường ───────────────────────────────────
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

print("\n\n🎉 HOÀN TẤT! Toàn bộ dữ liệu 2021–2026 đã được tải xong.")
print(f"   File đầu ra: {OUTPUT_DIR}")
