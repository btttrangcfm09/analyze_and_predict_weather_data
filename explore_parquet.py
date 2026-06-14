import zipfile
import io
import pathlib
import pyarrow.parquet as pq
import pandas as pd

ZIP_PATH = pathlib.Path("data/archive.zip")

# --- 1. Tìm file Parquet bên trong ZIP ---
with zipfile.ZipFile(ZIP_PATH) as zf:
    parquet_names = [n for n in zf.namelist() if n.endswith(".parquet")]
    if not parquet_names:
        raise FileNotFoundError("No .parquet file found inside ZIP.")

    print(f"Found {len(parquet_names)} Parquet file(s):")
    for n in parquet_names:
        print(f"  - {n}")

    # Pick first file
    target = parquet_names[0]
    print(f"\n=== Reading: {target} ===\n")

    with zf.open(target) as f:
        buf = io.BytesIO(f.read())

# --- 2. Đọc schema (không tải toàn bộ dữ liệu vào RAM) ---
pf = pq.ParquetFile(buf)
schema = pf.schema_arrow

print("SCHEMA (columns & data types):")
print("-" * 40)
for field in schema:
    print(f"  {field.name:<35} {field.type}")

# --- 3. In 5 dòng đầu ---
print("\nFIRST 5 ROWS:")
print("-" * 40)
df_head = pf.read_row_group(0).to_pandas().head(5)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df_head.to_string(index=False))
