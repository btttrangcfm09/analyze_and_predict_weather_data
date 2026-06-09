import io
import pandas as pd
import requests
import glob
import cdsapi
import xarray as xr
import os
from datetime import datetime, timedelta
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def load_data_from_api(*args, **kwargs):

    """
    Template for loading data from API
    """
    import urllib3
    urllib3.disable_warnings()
    client = cdsapi.Client(verify=False)
    dataset = "reanalysis-era5-single-levels"
    year = 2024
    
    # Đặt start_date là ngày 24/10/2024
    # start_date = datetime.now() - timedelta(days=8)
    # end_date = datetime.now() - timedelta(days=8)
    start_date = datetime(2024, 10, 13)
    end_date = datetime(2024, 10, 14)
    
    # Tạo thư mục crawl_weather nếu chưa tồn tại
    os.makedirs("crawl_weather", exist_ok=True)
    
    delta = timedelta(days=1)
    all_data = []
    current_date = start_date
    while current_date <= end_date:
        year = current_date.strftime('%Y')
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')
        request = {
            'product_type': ['reanalysis'],
            'variable': ['total_precipitation'],
            'year': [year],
            'month': [month],
            'day': [day],
            'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
            'data_format': 'grib',
            'download_format': 'unarchived',
            'area': [24, 102, 8, 112]
        }
        target = f"crawl_weather/rain_{day}_{month}_{year}.grib"
        client.retrieve(dataset, request, target)
        ds = xr.open_dataset(target)
        df = ds.to_dataframe().reset_index()
        all_data.append(df)
        current_date += delta
        folder_path = "crawl_weather"
        grib_files = glob.glob(os.path.join(folder_path, "*.grib"))
        idx_files = glob.glob(os.path.join(folder_path, "*.idx"))
        for file in grib_files:
            os.remove(file)

        for file in idx_files:
            os.remove(file)

        print(current_date)
    
    combined_df = pd.concat(all_data, ignore_index=False)
    return combined_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'

if __name__ == '__main__':
    print("Bắt đầu tải dữ liệu thời tiết...")
    df = load_data_from_api()
    df.to_csv("rain_data.csv", index=False)
    print("Xong! Đã lưu toàn bộ dữ liệu vào file rain_data.csv")
