"""
core/provinces.py — Gắn nhãn Tỉnh/Thành cho từng điểm lưới (GADM Level-1).

Spatial join các điểm đất liền với polygon tỉnh (NAME_1) → bảng
(latitude, longitude, province). Cache nhẹ data/grid_points_VNM.parquet.
"""
import os

import geopandas as gpd
import pandas as pd

from core.geo import BASE_DIR, GPKG_PATH

GRID_CACHE = os.path.join(BASE_DIR, "data", "grid_points_VNM.parquet")


def _land_coords_from_weather_cache() -> pd.DataFrame:
    """Suy lưới đất liền từ cache thời tiết khi grid cache bị thiếu (import trễ tránh vòng lặp)."""
    from core.loader import LAND_CACHE
    if not os.path.isfile(LAND_CACHE):
        raise FileNotFoundError(
            f"Chưa có {GRID_CACHE} lẫn {LAND_CACHE}. Chạy load_history() trước."
        )
    return pd.read_parquet(LAND_CACHE, columns=["latitude", "longitude"]).drop_duplicates()


_METRIC_CRS = 32648   # UTM 48N — chiếu hệ mét để sjoin_nearest tính khoảng cách đúng


def label_points_with_province(land_coords: pd.DataFrame, gpkg_path: str = GPKG_PATH) -> pd.DataFrame:
    """sjoin điểm lưới × tỉnh GADM L1 → cột 'province'. Điểm ngoài tỉnh → 'Khác'."""
    provinces = gpd.read_file(gpkg_path, layer=1)[["NAME_1", "geometry"]].to_crs(epsg=_METRIC_CRS)

    points = gpd.GeoDataFrame(
        land_coords.copy(),
        geometry=gpd.points_from_xy(
            land_coords["longitude"].to_numpy(), land_coords["latitude"].to_numpy()
        ),
        crs="EPSG:4326",
    ).to_crs(epsg=_METRIC_CRS)
    # sjoin_nearest (hệ mét): gắn tỉnh gần nhất → điểm ven biển/đảo cũng có tên tỉnh thực.
    joined = gpd.sjoin_nearest(points, provinces, how="left")
    joined = joined.drop_duplicates(subset=["latitude", "longitude"])

    out = joined[["latitude", "longitude", "NAME_1"]].rename(columns={"NAME_1": "province"})
    out["province"] = out["province"].fillna("Khác")
    return out.reset_index(drop=True)


def load_grid_points(land_coords: pd.DataFrame | None = None) -> pd.DataFrame:
    """Đọc cache bảng điểm-lưới-có-tỉnh; nếu chưa có thì dựng từ land_coords rồi lưu."""
    if os.path.isfile(GRID_CACHE):
        return pd.read_parquet(GRID_CACHE)
    if land_coords is None:
        land_coords = _land_coords_from_weather_cache()
    grid = label_points_with_province(land_coords)
    os.makedirs(os.path.dirname(GRID_CACHE), exist_ok=True)
    grid.to_parquet(GRID_CACHE, index=False)
    return grid


def province_points() -> dict:
    """{tên Tỉnh: (lat, lon)} — 1 điểm lưới đại diện (gần tâm tỉnh nhất) cho trang phân tích."""
    grid = load_grid_points()
    reps: dict = {}
    for name, g in grid.groupby("province"):
        lat = g["latitude"].to_numpy()
        lon = g["longitude"].to_numpy()
        i = int((((lat - lat.mean()) ** 2) + ((lon - lon.mean()) ** 2)).argmin())
        reps[str(name)] = (float(lat[i]), float(lon[i]))
    return dict(sorted(reps.items()))
