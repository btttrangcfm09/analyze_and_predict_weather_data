"""
core/model.py — Seq2Seq Encoder-Decoder LSTM (dự báo 168 giờ).

Y hệt kiến trúc huấn luyện (lstm_temp_weekly/model/lstm_model.py):
  Encoder : nn.LSTM trên 168 giờ quá khứ → (h, c) = context vector.
  Decoder : nn.LSTM riêng, khởi tạo bằng (h, c). Mỗi bước tương lai nhận
            [ nhiệt độ bước trước ] ⊕ [ đặc trưng tương lai đã biết ],
            Linear(hidden, 1) sinh ra nhiệt độ kế tiếp.

Suy luận web dùng teacher_forcing=False → autoregressive từng bước.
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

        layer_dropout = dropout if num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = layer_dropout,
            batch_first = True,
        )
        # decoder input mỗi bước = [nhiệt độ trước (1)] ⊕ [đặc trưng tương lai]
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
        # y:      (batch, predict_steps) — ground-truth (chỉ dùng khi teacher forcing)
        _, (h, c) = self.encoder(x)                          # context vector
        last_temp = x[:, -1, self.target_idx].unsqueeze(1)   # (B, 1) start token

        if teacher_forcing:
            assert y is not None, "teacher_forcing=True cần ground-truth y"
            prev   = torch.cat([last_temp, y[:, :-1]], dim=1)         # (B, T)
            dec_in = torch.cat([prev.unsqueeze(-1), y_feat], dim=-1)  # (B, T, 1+F)
            dec_out, _ = self.decoder(dec_in, (h, c))                 # (B, T, hidden)
            return self.out(dec_out).squeeze(-1)                      # (B, T)

        # ── Autoregressive: lấy chính dự đoán bước trước làm đầu vào ──
        prev  = last_temp                                    # (B, 1)
        preds = []
        for t in range(self.predict_steps):
            step_in = torch.cat([prev, y_feat[:, t]], dim=-1).unsqueeze(1)  # (B,1,1+F)
            out_t, (h, c) = self.decoder(step_in, (h, c))
            prev = self.out(out_t).squeeze(1)               # (B, 1)
            preds.append(prev)
        return torch.cat(preds, dim=1)                      # (B, T)
