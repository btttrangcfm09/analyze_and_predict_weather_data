## Deploy ETL pipeline lên Cloud Run

2 phần chính:
Deploy Mage lên Cloud Run và tạo ETL pipeline và lập lịch trên Mage

### Deploy Mage lên Cloud Run

Link hướng dẫn: https://docs.mage.ai/production/deploying-to-cloud/gcp/setup

### Tạo ETL pipeline và lập lịch trên Mage

#### Tạo service account với quyền Owner và tạo key json trên service account đó

#### Copy file key json vào Mage

#### Config file io_config.yaml (Mẫu trong code/io_config.yaml)

#### Copy file .cdsapirc vào thư mục root của Mage

#### Tạo 2 pipeline:

    - Pipeline weather_to_gcs: gồm crawl_weather.py và weather_to_gcs_partitioned.py. Tạo trigger lập lịch chạy hằng ngày.
    - Pipeline weather_to_bq: gồm load_weather_from_gcs.py và weahter_bq.py. Chỉ thực hiện khi cần đưa dữ liệu lên Bigquery
