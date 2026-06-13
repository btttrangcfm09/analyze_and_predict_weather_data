"""
GIAI ĐOẠN 3 - utils/storm.py
Thuật toán phát hiện tâm bão/áp thấp từ trường áp suất mực biển (MSL).
Thuần NumPy/SciPy, không phụ thuộc Streamlit (tách ra từ app.py - PHẦN 2).
"""
import numpy as np
import scipy.ndimage as ndimage


def distance_matrix(lons, lats):
    EARTH_RADIUS = 6378.1
    X = len(lons)
    d = np.zeros((X, X))
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons - loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[c < 1, i2] = EARTH_RADIUS * np.arccos(c[c < 1])
    return d


def nanmean(array, axis=None):
    return np.mean(np.ma.masked_array(array, np.isnan(array)), axis)


def len_deg_lon(lat):
    R = 6371.  # Radius of Earth [km]
    return (np.pi / 180.) * R * np.cos(lat * np.pi / 180.)


def spatial_filter(msl, lon, lat, res, cut_lon, cut_lat):
    msl_filt = np.zeros(msl.shape)
    sig_lon = (cut_lon / 5.) / res
    sig_lat = (cut_lat / 5.) / res
    land = np.isnan(msl)
    msl[land] = nanmean(msl)
    msl_filt = ndimage.gaussian_filter(msl, [sig_lat, sig_lon])
    msl_filt[land] = np.nan
    return msl_filt


def detect_storms(msl, wind_speed, lon, lat, res, order, Npix_min, Npix_max, rel_amp_thresh,
                  d_thresh, cyc, cut_lon, cut_lat, globe=False):
    len_deg_lat = 111.325
    msl = spatial_filter(msl, lon, lat, res, cut_lon, cut_lat)
    llon, llat = np.meshgrid(lon, lat)

    lon_storms = np.array([])
    lat_storms = np.array([])
    amp_storms = np.array([])
    area_storms = np.array([])
    max_wind_speeds = np.array([])
    regions_storm = []
    ssh_crits = np.array([100000])  # 1000hPa

    for ssh_crit in ssh_crits:
        regions, nregions = ndimage.label((msl < ssh_crit).astype(int))
        for iregion in range(nregions):
            region = (regions == iregion + 1).astype(int)
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
                lon_cen = np.interp(i_min, range(0, len(lon)), lon)
                lat_cen = np.interp(j_min, range(0, len(lat)), lat)

                lon_storms = np.append(lon_storms, lon_cen)
                lat_storms = np.append(lat_storms, lat_cen)
                amp_storms = np.append(amp_storms, amp_abs)
                area = region_Npix * res**2 * len_deg_lat * len_deg_lon(lat_cen)
                area_storms = np.append(area_storms, area)

                storm_mask = np.ones(msl.shape)
                storm_mask[interior.astype(int) == 1] = np.nan
                storm_mask = msl * storm_mask
                region_storm = np.isnan(storm_mask).astype(int)
                regions_storm.append(region_storm)
                max_wind_speed = (wind_speed * region_storm).max()
                max_wind_speeds = np.append(max_wind_speeds, max_wind_speed)

    return lon_storms, lat_storms, amp_storms, max_wind_speeds, area_storms, regions_storm
