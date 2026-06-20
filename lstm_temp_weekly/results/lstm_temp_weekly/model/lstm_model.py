"""
Task: Define the multi-step LSTM model.

Architecture (Seq2Seq Encoder-Decoder — LSTM only, no GRU):

  Encoder : nn.LSTM over the 168 past hours → final (h, c) = context vector.
  Decoder : a SEPARATE nn.LSTM initialised with the encoder's (h, c). At every
            future step its input is
                [ temperature of the previous step ]  ⊕  [ known future features ]
            and a Linear(hidden, 1) head emits the next temperature.

Two forward modes share one set of weights:

  • Vectorized Teacher Forcing (training, ratio = 1.0):
      The whole ground-truth target, shifted one step and prefixed with the last
      observed temperature, is fed to the decoder LSTM in a SINGLE call. PyTorch
      unrolls the recurrence on the GPU in parallel → no Python for-loop, fast.

  • Autoregressive (evaluation / inference):
      The decoder consumes its OWN previous prediction at each step, looping over
      the 168 horizon — the honest deployment-time behaviour.

Feeding the known future calendar features to the decoder is what preserves the
day/night diurnal peaks that the old Direct (flat Linear) head smoothed away.
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size:    int,
        dec_feat_size: int,
        hidden_size:   int   = 128,
        num_layers:    int   = 2,
        predict_steps: int   = 168,
        dropout:       float = 0.1,
        target_idx:    int   = 0,
    ):
        super().__init__()
        self.predict_steps = predict_steps
        self.target_idx    = target_idx

        # PyTorch applies dropout between stacked LSTM layers only.
        layer_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = layer_dropout,
            batch_first = True,
        )
        # decoder input per step = [prev temperature (1)] ⊕ [known future features]
        self.decoder = nn.LSTM(
            input_size  = 1 + dec_feat_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = layer_dropout,
            batch_first = True,
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, y_feat, y=None, teacher_forcing=True):
        # x:      (batch, seq_len, input_size)
        # y_feat: (batch, predict_steps, dec_feat_size)
        # y:      (batch, predict_steps)            ground-truth target (TF only)
        _, (h, c) = self.encoder(x)                          # context vector
        last_temp = x[:, -1, self.target_idx].unsqueeze(1)   # (B, 1) start token

        if teacher_forcing:
            # ── Vectorized Teacher Forcing: one parallel decoder call ──
            prev   = torch.cat([last_temp, y[:, :-1]], dim=1)         # (B, T) shifted
            dec_in = torch.cat([prev.unsqueeze(-1), y_feat], dim=-1)  # (B, T, 1+F)
            dec_out, _ = self.decoder(dec_in, (h, c))                 # (B, T, hidden)
            return self.out(dec_out).squeeze(-1)                      # (B, T)

        # ── Autoregressive: feed own prediction back, step by step ──
        prev  = last_temp                                    # (B, 1)
        preds = []
        for t in range(self.predict_steps):
            step_in = torch.cat([prev, y_feat[:, t]], dim=-1).unsqueeze(1)  # (B,1,1+F)
            out_t, (h, c) = self.decoder(step_in, (h, c))
            prev = self.out(out_t).squeeze(1)               # (B, 1)
            preds.append(prev)
        return torch.cat(preds, dim=1)                      # (B, T)
