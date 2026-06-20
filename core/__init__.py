"""core — Backend dự báo nhiệt độ Seq2Seq LSTM (không import streamlit)."""

from .model       import LSTMModel
from .loader      import load_history, BASE_DIR
from .provinces   import load_grid_points, province_points
from .inference   import load_model, _predict_core
from .features    import SEQUENCE_LENGTH, PREDICT_STEPS, TARGET_COL
