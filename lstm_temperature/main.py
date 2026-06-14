"""
Task: Orchestrate the full pipeline — load → engineer → scale → build sequences
      → train LSTM → evaluate → export artifacts.

Run on Kaggle:
    !python /kaggle/working/lstm_temperature/main.py
"""
import os
import sys

# Allow sibling-package imports regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.load_data import load_data
from preprocessing.feature_engineering import engineer_features
from preprocessing.scaling import fit_and_scale
from dataset.sequence_dataset import (
    build_province_arrays,
    build_split_indices,
    WeatherSequenceDataset,
    SEQUENCE_LENGTH,
)
from model.lstm_model import LSTMModel
from training.trainer import train_model
from training.evaluator import evaluate_test
from visualization.plot_loss import plot_loss_curve, save_history

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 1024
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.1
LR          = 0.001
TARGET_COL  = 'temperature_celsius'
MODEL_PATH  = 'lstm_weather_model_temp.pt'


def main() -> None:
    # ── Device ──────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    use_pin = device.type == 'cuda'

    # ── 1. Load ─────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    df = load_data()
    n_locs = df['latitude'].nunique()
    print(f"  {len(df):>10,} rows  |  {n_locs} unique locations")

    # ── 2. Feature engineering ───────────────────────────────────────────────────
    print("\n[2/6] Engineering features ...")
    df = engineer_features(df)
    print(f"  {len(df):>10,} rows  |  {len(df.columns)} columns")

    # ── 3. Scale ─────────────────────────────────────────────────────────────────
    print("\n[3/6] Scaling ...")
    df, _, feature_cols = fit_and_scale(df)
    target_idx = feature_cols.index(TARGET_COL)

    # ── 4. Sequences ─────────────────────────────────────────────────────────────
    print("\n[4/6] Building sequences ...")
    province_arrays = build_province_arrays(df, feature_cols)
    del df   # release ~180 MB

    train_idx, test_idx = build_split_indices(province_arrays, SEQUENCE_LENGTH)
    print(f"  Train: {len(train_idx):,}  |  Test: {len(test_idx):,}  "
          f"(seq_len={SEQUENCE_LENGTH})")

    train_ds = WeatherSequenceDataset(province_arrays, train_idx, target_idx, SEQUENCE_LENGTH)
    test_ds  = WeatherSequenceDataset(province_arrays, test_idx,  target_idx, SEQUENCE_LENGTH)

    # The val_loader re-uses the test split for epoch-level monitoring
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )
    val_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )

    # ── 5. Model ─────────────────────────────────────────────────────────────────
    input_size = len(feature_cols)
    model      = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, 1, DROPOUT).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    criterion  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[5/6] Model ready  |  input_size={input_size}  |  params={n_params:,}")

    # ── 6. Train ─────────────────────────────────────────────────────────────────
    print(f"\n[6/6] Training  ({EPOCHS} epochs, batch={BATCH_SIZE}) ...")
    print("-" * 95)
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, EPOCHS
    )
    print("-" * 95)

    # ── Evaluate on independent test set ────────────────────────────────────────
    evaluate_test(model, test_ds, device, BATCH_SIZE)

    # ── Export artifacts ─────────────────────────────────────────────────────────
    print("\n[Export]")
    plot_loss_curve(history)
    save_history(history)

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_size':       input_size,
            'hidden_size':      HIDDEN_SIZE,
            'num_layers':       NUM_LAYERS,
            'output_size':      1,
            'dropout':          DROPOUT,
            'sequence_length':  SEQUENCE_LENGTH,
            'target_cols':      [TARGET_COL],
            'scaler_name':      'scaler_temp.pkl',
            'feature_cols_name':'feature_cols_temp.pkl',
        },
        MODEL_PATH,
    )
    print(f"  Checkpoint      -> {MODEL_PATH}")
    print("\n[Done]")


if __name__ == '__main__':
    main()
