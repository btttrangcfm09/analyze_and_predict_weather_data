"""
Task: Orchestrate the full weekly-TEMPERATURE pipeline:
      Load → Engineer → Scale → Sequences → Train → Evaluate → Plots → Checkpoint.

Model: Seq2Seq Encoder-Decoder LSTM (vectorized teacher forcing in training,
       autoregressive at inference), with AMP, early stopping, and a checkpoint
       saved every epoch to models/lstm_temp_weekly_epoch_NN.pt.

Run on Kaggle (GPU):
    !python /kaggle/working/lstm_temp_weekly/main.py
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
    resolve_decoder_feat_idx,
    WeatherSequenceDataset,
    DECODER_FEATURE_COLS,
    SEQUENCE_LENGTH,
    PREDICT_STEPS,
)
from model.lstm_model import LSTMModel
from training.trainer import train_model
from training.evaluator import evaluate_test, collect_predictions
from visualization.plot_loss import save_history, save_all_plots

# ── Hyper-parameters ───────────────────────────────────────────────────────────
EPOCHS      = 5         # max epochs; early stopping usually halts sooner
PATIENCE    = 2         # stop when val MSE doesn't improve for 2 epochs
BATCH_SIZE  = 2048      # large batch to saturate the Kaggle T4 GPU
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.1
LR          = 0.001
USE_AMP     = True      # mixed precision on CUDA

MODELS_DIR  = 'models'
CKPT_PREFIX = 'lstm_temp_weekly'
MODEL_PATH  = os.path.join(MODELS_DIR, 'lstm_temp_weekly_model.pt')


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

    # Keep a raw-temperature sample for the distribution plot (it stays in °C
    # the whole pipeline, but we grab it before scaling for a clean axis).
    raw_temp = df[TARGET_COL].sample(
        n=min(100_000, len(df)), random_state=0
    ).to_numpy()

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/7] Engineering features ...")
    df = engineer_features(df)
    print(f"  {len(df):>12,} rows  |  {len(df.columns)} columns")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    print("\n[3/7] Scaling ...")
    df, scaler, feature_cols = fit_and_scale(df)
    target_idx       = feature_cols.index(TARGET_COL)
    decoder_feat_idx = resolve_decoder_feat_idx(feature_cols)
    dec_feat_size    = len(decoder_feat_idx)

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
          f"(seq_len={SEQUENCE_LENGTH}, predict_steps={PREDICT_STEPS}, "
          f"dec_feat={dec_feat_size})")

    train_ds = WeatherSequenceDataset(
        coordinate_arrays, train_idx, target_idx, decoder_feat_idx,
        SEQUENCE_LENGTH, PREDICT_STEPS,
    )
    test_ds = WeatherSequenceDataset(
        coordinate_arrays, test_idx, target_idx, decoder_feat_idx,
        SEQUENCE_LENGTH, PREDICT_STEPS,
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
    model = LSTMModel(
        input_size, dec_feat_size, HIDDEN_SIZE, NUM_LAYERS,
        PREDICT_STEPS, DROPOUT, target_idx,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Full network config — enough to rebuild the model from any checkpoint.
    config = {
        'input_size':        input_size,
        'dec_feat_size':     dec_feat_size,
        'hidden_size':       HIDDEN_SIZE,
        'num_layers':        NUM_LAYERS,
        'predict_steps':     PREDICT_STEPS,        # 168
        'output_size':       PREDICT_STEPS,        # alias (back-compat)
        'dropout':           DROPOUT,
        'target_idx':        target_idx,
        'sequence_length':   SEQUENCE_LENGTH,      # 168
        'decoder_feature_cols': DECODER_FEATURE_COLS,
        'target_cols':       [TARGET_COL],         # ['temperature_celsius']
        'scaler_name':       SCALER_PATH,          # 'scaler_temp.pkl'
        'feature_cols_name': FEATURE_COLS_PATH,    # 'feature_cols_temp.pkl'
    }

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[5/7] Seq2Seq model ready  |  input_size={input_size}  |  "
          f"dec_feat_size={dec_feat_size}  |  predict_steps={PREDICT_STEPS}  |  "
          f"params={n_params:,}")

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print(f"\n[6/7] Training  (max {EPOCHS} epochs, patience={PATIENCE}, "
          f"batch={BATCH_SIZE}) ...")
    print("-" * 100)
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=EPOCHS, patience=PATIENCE, config=config,
        models_dir=MODELS_DIR, ckpt_prefix=CKPT_PREFIX, use_amp=USE_AMP,
    )
    print("-" * 100)

    # ── 7. Evaluate (real units) + visualise + export ────────────────────────
    print("\n[7/7] Evaluating on the test set (autoregressive) ...")
    evaluate_test(model, test_ds, device, scaler, target_idx, BATCH_SIZE)

    print("\n[Export]")
    save_history(history)

    preds_c, targets_c, coord_ids, hours_of_day = collect_predictions(
        model, coordinate_arrays, test_idx, target_idx, decoder_feat_idx,
        feature_cols, scaler, SEQUENCE_LENGTH, PREDICT_STEPS, device,
        max_samples=4000, batch_size=BATCH_SIZE,
    )
    save_all_plots(
        history, targets_c, preds_c, coord_ids, coords,
        df_sample, feature_cols, raw_temp, hours_of_day, target_col=TARGET_COL,
    )

    # ── Final checkpoint (best epoch's weights) ───────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'config':           config,
            'best_epoch':       history.get('best_epoch'),
            # back-compat top-level keys
            'input_size':        input_size,
            'hidden_size':       HIDDEN_SIZE,
            'num_layers':        NUM_LAYERS,
            'output_size':       PREDICT_STEPS,
            'dropout':           DROPOUT,
            'sequence_length':   SEQUENCE_LENGTH,
            'target_cols':       [TARGET_COL],
            'scaler_name':       SCALER_PATH,
            'feature_cols_name': FEATURE_COLS_PATH,
        },
        MODEL_PATH,
    )
    print(f"  Checkpoint       -> {MODEL_PATH}")
    print("\n[Done]")


if __name__ == '__main__':
    main()
