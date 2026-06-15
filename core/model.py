"""
core/model.py — Kiến trúc mô hình LSTM nhiệt độ.

Khớp CHÍNH XÁC với lstm_temperature/model/lstm_model.py trên Kaggle.
nn.Sequential cho head đảm bảo state_dict keys:
  head.0.weight/bias  (Linear 64→32)
  head.3.weight/bias  (Linear 32→1)
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        output_size: int   = 1,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        return self.head(out).squeeze(-1)
