"""
core/geo.py — Ranh giới đất liền Việt Nam từ GADM (GeoPackage).

Đọc GADM Level-0 (cả nước), union + buffer 0.05 độ, rồi lọc các toạ độ lưới
nằm trong lãnh thổ. Rút gọn từ lstm_temp_weekly/data/load_data.py.
"""
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPKG_PATH = os.path.join(BASE_DIR, "data", "gadm41_VNM.gpkg")
LAND_BUFFER_DEGREES = 0.05
_COORD_COLS = ["latitude", "longitude"]


def load_vietnam_land_geometry(gpkg_path: str = GPKG_PATH, buffer_degrees: float = LAND_BUFFER_DEGREES):
    """Polygon Việt Nam (GADM level-0) đã union + buffer, ở EPSG:4326."""
    level0 = gpd.read_file(gpkg_path, layer=0)
    if level0.empty:
        raise ValueError("GeoPackage không có hình học Việt Nam ở layer=0.")
    if level0.crs is not None and level0.crs.to_epsg() != 4326:
        level0 = level0.to_crs(epsg=4326)
    geom = level0.geometry.union_all()
    if geom.is_empty:
        raise ValueError("Không tạo được polygon Việt Nam từ GeoPackage.")
    return geom.buffer(buffer_degrees)


def filter_land_coords(unique_coords: pd.DataFrame, land_area) -> pd.DataFrame:
    """Giữ các (latitude, longitude) nằm trong polygon đất liền. Point dùng (lon, lat)."""
    mask = [
        land_area.contains(Point(lon, lat))
        for lat, lon in unique_coords[_COORD_COLS].itertuples(index=False, name=None)
    ]
    land = unique_coords.loc[mask, _COORD_COLS].reset_index(drop=True)
    if land.empty:
        raise ValueError("Không có điểm lưới nào nằm trong polygon Việt Nam.")
    return land
