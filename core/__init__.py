"""core — Backend dự báo nhiệt độ LSTM (không import streamlit)."""

from .model     import LSTMModel
from .loader    import load_history, LOCAL_WEATHER_DIR, BASE_DIR
from .inference import load_model, _predict_core
from .features  import VIETNAM_PROVINCES, SEQUENCE_LENGTH, TARGET_COL
