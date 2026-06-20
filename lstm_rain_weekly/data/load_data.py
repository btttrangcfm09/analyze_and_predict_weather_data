"""
Nap du lieu thoi tiet cho cac diem luoi nam trong lanh tho Viet Nam.

Luon loc toa do truoc khi doc day du du lieu:
  1. Chi doc hai cot latitude/longitude de lay danh sach toa do duy nhat.
  2. Dung GeoPackage GADM level-0 de giu cac toa do nam trong polygon Viet Nam
     da buffer 0.05 do.
  3. Moi doc day du _COLS tu Parquet voi filters theo cac toa do hop le.
  4. Sap xep chuoi thoi gian va ep float64 -> float32.
"""
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DATA_PATH = '/kaggle/input/datasets/nguyentranggggg/vietnam-meteorological-weather-data-parquet/weather.parquet'
GPKG_PATH = '/kaggle/input/datasets/nglan271204/vnm-gpkg/gadm41_VNM.gpkg'
LOCAL_GPKG_PATH = './data/gadm41_VNM.gpkg'
LAND_BUFFER_DEGREES = 0.05

target_cols = ['total_precipitation']

_COORD_COLS = ['latitude', 'longitude']

_COLS = [
    'latitude', 'longitude', 'valid_time',
    'temperature_celsius', 'apparent_temperature',
    'relative_humidity', 'wind_speed', 'wind_direction',
    'total_precipitation', 'total_cloud_cover',
    'mean_sea_level_pressure', 'surface_pressure',
    'sea_surface_temperature', 'air_density',
]


def _resolve_gpkg_path(gpkg_path: str) -> str:
    """Uu tien duong dan Kaggle, fallback ve ./data khi test local."""
    if Path(gpkg_path).exists():
        return gpkg_path
    if Path(LOCAL_GPKG_PATH).exists():
        return LOCAL_GPKG_PATH
    return gpkg_path


def _union_geometries(geometry):
    """Gop tat ca polygon Viet Nam thanh mot hinh hoc duy nhat."""
    return geometry.union_all()


def _load_vietnam_land_geometry(gpkg_path: str = GPKG_PATH):
    """Doc polygon quoc gia Viet Nam tu GeoPackage GADM level-0."""
    gpkg_path = _resolve_gpkg_path(gpkg_path)
    print(f"Dang doc ranh gioi Viet Nam tu GeoPackage: {gpkg_path}")
    vietnam_level0 = gpd.read_file(gpkg_path, layer=0)

    if vietnam_level0.empty:
        raise ValueError("GeoPackage khong co hinh hoc Viet Nam o layer=0.")

    # Du lieu thoi tiet la kinh/vi do, nen dua polygon ve EPSG:4326 neu can.
    if vietnam_level0.crs is not None and vietnam_level0.crs.to_epsg() != 4326:
        vietnam_level0 = vietnam_level0.to_crs(epsg=4326)

    vietnam_geom = _union_geometries(vietnam_level0.geometry)

    if vietnam_geom.is_empty:
        raise ValueError("Khong tao duoc polygon Viet Nam tu GeoPackage.")

    return vietnam_geom


def _read_unique_coordinates(data_path: str) -> pd.DataFrame:
    """
    Chi doc hai cot toa do tu Parquet.

    Buoc nay khong doc cac cot khi tuong lon nhu nhiet do/mua/ap suat, nen RAM
    nhe hon rat nhieu so voi doc full _COLS truoc roi moi loc.
    """
    print("Dang doc nhe 2 cot latitude/longitude de lay toa do duy nhat ...")
    coords = pd.read_parquet(data_path, columns=_COORD_COLS, engine='auto')
    unique_coords = coords.drop_duplicates().reset_index(drop=True)
    print(f"So diem toa do duy nhat truoc khi loc: {len(unique_coords):,}")
    return unique_coords


def _find_vietnam_land_coordinates(
    unique_coords: pd.DataFrame,
    vietnam_geom,
    buffer_degrees: float = LAND_BUFFER_DEGREES,
) -> pd.DataFrame:
    """
    Loc toa do nam trong polygon Viet Nam da buffer.

    Chi tao Point va contains() cho danh sach toa do duy nhat, tuyet doi khong
    chay phep toan hinh hoc tren hang trieu dong du lieu thoi gian.
    """
    vietnam_area = vietnam_geom.buffer(buffer_degrees)

    # Shapely Point dung thu tu (x, y) = (longitude, latitude).
    points = [
        Point(lon, lat)
        for lat, lon in unique_coords[_COORD_COLS].itertuples(index=False, name=None)
    ]
    inside_mask = [vietnam_area.contains(point) for point in points]

    land_coords = unique_coords.loc[inside_mask, _COORD_COLS].copy()
    print(
        "So diem toa do nam trong lanh tho Viet Nam sau khi loc "
        f"(buffer={buffer_degrees} do): {len(land_coords):,}"
    )

    if land_coords.empty:
        raise ValueError(
            "Khong co diem luoi nao nam trong polygon Viet Nam. "
            "Hay kiem tra CRS/toa do hoac duong dan GeoPackage."
        )

    return land_coords


def _build_coordinate_filters(keep_coords: pd.DataFrame) -> list:
    """
    Tao Parquet filters theo cap toa do hop le.

    Dung OR theo tung latitude, trong moi latitude dung longitude in [...]
    de filter dung cap (latitude, longitude) nhung ngan gon hon hang nghin
    dieu kien OR rieng le.
    """
    coords = keep_coords[_COORD_COLS].astype('float64')
    lon_by_lat: dict[float, list[float]] = {}

    for lat_value, lon_value in coords.itertuples(index=False, name=None):
        lat = float(cast(Any, lat_value))
        lon = float(cast(Any, lon_value))
        lon_by_lat.setdefault(lat, []).append(lon)

    return [
        [
            ('latitude', '==', lat),
            ('longitude', 'in', lons),
        ]
        for lat, lons in lon_by_lat.items()
    ]


def _read_filtered_weather_data(data_path: str, keep_coords: pd.DataFrame) -> pd.DataFrame:
    """Doc day du _COLS chi cho cac toa do dat lien da duoc loc truoc."""
    filters = _build_coordinate_filters(keep_coords)
    print(f"Dang doc Parquet voi filters cho {len(keep_coords):,} diem toa do hop le ...")
    return pd.read_parquet(
        data_path,
        columns=_COLS,
        engine='auto',
        filters=filters,
    )


def load_data(
    data_path: str = DATA_PATH,
    gpkg_path: str = GPKG_PATH,
    buffer_degrees: float = LAND_BUFFER_DEGREES,
) -> pd.DataFrame:
    vietnam_geom = _load_vietnam_land_geometry(gpkg_path)
    unique_coords = _read_unique_coordinates(data_path)
    land_coords = _find_vietnam_land_coordinates(unique_coords, vietnam_geom, buffer_degrees)
    df = _read_filtered_weather_data(data_path, land_coords)

    # valid_time phai la datetime de tao feature thoi gian dung ve sau.
    df['valid_time'] = pd.to_datetime(df['valid_time'])

    # Sap xep de moi cap toa do co chuoi thoi gian lien tuc truoc khi tao sequence.
    df = df.sort_values(['latitude', 'longitude', 'valid_time']).reset_index(drop=True)

    # float64 -> float32 giup giam RAM khi feature engineering va train LSTM.
    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('float32')

    n_locs = df.groupby(['latitude', 'longitude'], sort=False).ngroups
    print(
        "Nap du lieu dat lien Viet Nam thanh cong! "
        f"{len(df):,} dong | {n_locs:,} diem toa do | target_cols={target_cols}"
    )
    return df
