"""
Task: Orchestrate the full weekly-rainfall pipeline:
      Load → Engineer → Scale → Sequences → Train → Evaluate → 8 Plots → Checkpoint.

Run on Kaggle (GPU):
    !python /kaggle/working/lstm_rain_weekly/main.py
"""
import os
import sys

# Allow sibling-package imports regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.load_data import load_data
from preprocessing.feature_engineering import engineer_features, TARGET_COL
from preprocessing.scaling import fit_and_scale, SCALER_PATH, FEATURE_COLS_PATH
from dataset.sequence_dataset import (
    build_coordinate_arrays,
    build_split_indices,
    WeatherSequenceDataset,
    SEQUENCE_LENGTH,
    PREDICT_STEPS,
)
from model.lstm_model import LSTMModel
from training.trainer import train_model
from training.evaluator import evaluate_test, collect_predictions
from visualization.plot_loss import save_history, save_all_plots

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 2048      # large batch to saturate the Kaggle T4 GPU
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.1
LR          = 0.001
OUTPUT_SIZE = PREDICT_STEPS          # 168 future hours (Direct Multi-step)

MODELS_DIR  = 'models'
MODEL_PATH  = os.path.join(MODELS_DIR, 'lstm_rain_weekly_model.pt')


def main() -> None:
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    use_pin = device.type == 'cuda'

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    df = load_data()

    # Keep a raw-rainfall sample BEFORE log1p — needed for the distribution plot.
    raw_precip = df[TARGET_COL].sample(
        n=min(100_000, len(df)), random_state=0
    ).to_numpy()

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/7] Engineering features ...")
    df = engineer_features(df)
    print(f"  {len(df):>12,} rows  |  {len(df.columns)} columns")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    print("\n[3/7] Scaling ...")
    df, scaler, feature_cols = fit_and_scale(df)
    target_idx = feature_cols.index(TARGET_COL)

    # Sample of the scaled frame for the correlation heatmap (before we drop df).
    df_sample = df[feature_cols].sample(
        n=min(50_000, len(df)), random_state=0
    ).copy()

    # ── 4. Sequences ─────────────────────────────────────────────────────────
    print("\n[4/7] Building sequences ...")
    coordinate_arrays, coords = build_coordinate_arrays(df, feature_cols)
    del df   # release the big frame; arrays + sample are all we still need

    train_idx, test_idx = build_split_indices(
        coordinate_arrays, SEQUENCE_LENGTH, PREDICT_STEPS
    )
    print(f"  Train: {len(train_idx):,}  |  Test: {len(test_idx):,}  "
          f"(seq_len={SEQUENCE_LENGTH}, predict_steps={PREDICT_STEPS})")

    train_ds = WeatherSequenceDataset(
        coordinate_arrays, train_idx, target_idx, SEQUENCE_LENGTH, PREDICT_STEPS
    )
    test_ds = WeatherSequenceDataset(
        coordinate_arrays, test_idx, target_idx, SEQUENCE_LENGTH, PREDICT_STEPS
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )
    # The test split doubles as the per-epoch validation monitor.
    val_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=use_pin, persistent_workers=True,
    )

    # ── 5. Model ─────────────────────────────────────────────────────────────
    input_size = len(feature_cols)
    model      = LSTMModel(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, DROPOUT).to(device)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    criterion  = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[5/7] Model ready  |  input_size={input_size}  |  "
          f"output_size={OUTPUT_SIZE}  |  params={n_params:,}")

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print(f"\n[6/7] Training  ({EPOCHS} epochs, batch={BATCH_SIZE}) ...")
    print("-" * 100)
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device, EPOCHS
    )
    print("-" * 100)

    # ── 7. Evaluate (real units) + visualise + export ────────────────────────
    print("\n[7/7] Evaluating on the test set ...")
    evaluate_test(model, test_ds, device, scaler, target_idx, BATCH_SIZE)

    print("\n[Export]")
    save_history(history)

    preds_mm, targets_mm, coord_ids = collect_predictions(
        model, coordinate_arrays, test_idx, target_idx, scaler,
        SEQUENCE_LENGTH, PREDICT_STEPS, device,
        max_samples=4000, batch_size=BATCH_SIZE,
    )
    save_all_plots(
        history, targets_mm, preds_mm, coord_ids, coords,
        df_sample, feature_cols, raw_precip, target_col=TARGET_COL,
    )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(
        {
            'model_state_dict':  model.state_dict(),
            'input_size':        input_size,
            'hidden_size':       HIDDEN_SIZE,
            'num_layers':        NUM_LAYERS,
            'output_size':       OUTPUT_SIZE,        # 168
            'dropout':           DROPOUT,
            'sequence_length':   SEQUENCE_LENGTH,    # 168
            'target_cols':       [TARGET_COL],       # ['total_precipitation']
            'scaler_name':       SCALER_PATH,        # 'scaler_rain.pkl'
            'feature_cols_name': FEATURE_COLS_PATH,  # 'feature_cols_rain.pkl'
        },
        MODEL_PATH,
    )
    print(f"  Checkpoint       -> {MODEL_PATH}")
    print("\n[Done]")


if __name__ == '__main__':
    main()
