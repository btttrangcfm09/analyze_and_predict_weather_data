"""
Task: Define the multi-step LSTM model.

Architecture (Direct Multi-step forecasting):
  LSTM(input_size, hidden=128, layers=2, dropout=0.1, batch_first)
  → take the hidden state of the LAST timestep
  → Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, 168)

The final layer emits all 168 future hours at once, so prediction errors are not
fed back into the model — this avoids the error accumulation of recursive
single-step forecasting.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        output_size: int   = 168,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            # PyTorch applies dropout between stacked LSTM layers only.
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)      # (batch, seq_len, hidden)
        out    = out[:, -1, :]     # last timestep → (batch, hidden)
        return self.head(out)      # (batch, output_size=168)
