import os
import json
import subprocess
import streamlit as st

def push_to_kaggle(parquet_dir: str):
    """Đẩy toàn bộ thư mục parquet_dir lên Kaggle dataset."""
    st.info("Chuẩn bị đồng bộ dữ liệu lên Kaggle...")
    
    metadata = {
      "title": "vietnam-meteorological-weather-data-2020-2026",
      "id": "nguyentranggggg/vietnam-meteorological-weather-data-2020-2026",
      "licenses": [{"name": "CC0-1.0"}]
    }
    
    metadata_path = os.path.join(parquet_dir, 'dataset-metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    try:
        # Chạy lệnh đẩy dữ liệu bằng thư viện kaggle
        cmd = ["kaggle", "datasets", "version", "-p", parquet_dir, "-m", "Daily automated update from web app", "--dir-mode", "zip"]
        st.write("Đang tải file lên máy chủ Kaggle (có thể mất vài phút tuỳ dung lượng)...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        st.success("✅ Đã đồng bộ thành công lên Kaggle!")
        st.text(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error("❌ Lỗi khi đồng bộ lên Kaggle. Vui lòng kiểm tra lại cấu hình ~/.kaggle/kaggle.json")
        st.code(e.stderr)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy công cụ `kaggle`. Vui lòng chắc chắn bạn đã cài đặt pip install kaggle.")
