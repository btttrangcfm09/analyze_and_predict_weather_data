"""
Task: Define the LSTM model architecture.

Architecture:
  LSTM(input_size, hidden=64, layers=2, dropout=0.1)
  → take last timestep hidden state
  → Linear(64, 32) → ReLU → Dropout(0.1) → Linear(32, 1)
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
            # PyTorch applies dropout between LSTM layers (not after the last)
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
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)       # (batch, seq_len, hidden)
        out    = out[:, -1, :]      # last timestep → (batch, hidden)
        return self.head(out).squeeze(-1)   # (batch,)
