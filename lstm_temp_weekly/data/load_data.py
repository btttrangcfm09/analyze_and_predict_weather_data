"""
Nap du lieu thoi tiet va loc TRON VEN cac diem luoi nam tren lanh tho
dat lien Viet Nam bang GeoPackage GADM level-0.

Y tuong hieu nang:
  1. Doc Parquet mot lan voi dung cac cot can cho mo hinh.
  2. Rut gon DataFrame lon thanh danh sach toa do duy nhat.
  3. Chi chay phep toan hinh hoc Shapely tren danh sach toa do duy nhat.
  4. Dung MultiIndex.isin() de lay lai toan bo chuoi thoi gian tu DataFrame goc.
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

DATA_PATH = '/kaggle/input/datasets/nguyentranggggg/vietnam-meteorological-weather-data-parquet/weather.parquet'
GPKG_PATH = '/kaggle/input/datasets/nglan271204/vnm-gpkg/gadm41_VNM.gpkg'
LOCAL_GPKG_PATH = './data/gadm41_VNM.gpkg'
LAND_BUFFER_DEGREES = 0.05

target_cols = ['temperature_celsius']

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
    """
    Uu tien duong dan Kaggle. Khi test local, tu dong fallback ve ./data.
    """
    if Path(gpkg_path).exists():
        return gpkg_path
    if Path(LOCAL_GPKG_PATH).exists():
        return LOCAL_GPKG_PATH
    return gpkg_path


def _union_geometries(geometry):
    """
    Gop tat ca polygon Viet Nam thanh mot hinh hoc duy nhat.
    """
    return geometry.union_all()


def _load_vietnam_land_geometry(gpkg_path: str = GPKG_PATH):
    """
    Doc polygon quoc gia Viet Nam tu GADM level-0.

    Khac voi bai toan "sat bien", lan nay ta giu polygon dat lien day du, sau
    do buffer nhe polygon de khong mat cac diem luoi nam ngay sat bien gioi
    hoac sat duong bo bien.
    """
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


def _find_vietnam_land_coordinates(
    df: pd.DataFrame,
    vietnam_geom,
    buffer_degrees: float = LAND_BUFFER_DEGREES,
) -> pd.DataFrame:
    """
    Loc toa do nam trong polygon Viet Nam da buffer.

    Luu y quan trong ve RAM: chi tao Point va contains() cho cac cap toa do
    duy nhat, khong tinh hinh hoc tren tung dong thoi gian cua DataFrame lon.
    """
    unique_coords = df[_COORD_COLS].drop_duplicates().reset_index(drop=True)
    print(f"So diem toa do duy nhat truoc khi loc: {len(unique_coords):,}")

    # Buffer polygon Viet Nam them 0.05 do (~5 km) de giu diem sat bien/bo bien.
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


def _filter_rows_by_coordinates(df: pd.DataFrame, keep_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Dung MultiIndex.isin() de so khop dung cap (latitude, longitude).

    Cach nay chi tao mask theo toa do, khong tao Point cho hang trieu dong du
    lieu goc, nen tiet kiem RAM hon rat nhieu so voi loc hinh hoc truc tiep.
    """
    keep_index = pd.MultiIndex.from_frame(keep_coords[_COORD_COLS])
    df_index = pd.MultiIndex.from_arrays(
        [df['latitude'], df['longitude']],
        names=_COORD_COLS,
    )
    return df.loc[df_index.isin(keep_index)].copy()


def load_data(
    data_path: str = DATA_PATH,
    gpkg_path: str = GPKG_PATH,
    buffer_degrees: float = LAND_BUFFER_DEGREES,
) -> pd.DataFrame:
    print(f"Dang doc du lieu thoi tiet tu: {data_path}")
    df = pd.read_parquet(data_path, columns=_COLS, engine='auto')

    vietnam_geom = _load_vietnam_land_geometry(gpkg_path)
    land_coords = _find_vietnam_land_coordinates(df, vietnam_geom, buffer_degrees)
    df = _filter_rows_by_coordinates(df, land_coords)

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
