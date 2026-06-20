"""
Task: Run the Seq2Seq training loop — one epoch of gradient updates over the
      train set, then inference over the val set, logging MSE / RMSE / MAE.

Speed & regularisation:
  • Mixed Precision (torch.amp) shrinks T4 step time and memory.
  • Early Stopping (patience) halts as soon as val_loss stops improving — the old
    Direct model overfit by epoch 1-2, so this saves hours.
  • A checkpoint is written every epoch to models/<prefix>_epoch_NN.pt (full
    state_dict + optimizer + network config + epoch + metrics), and the BEST
    epoch's weights are restored into the model before returning.

Both train and val passes use TEACHER FORCING here (a single parallel decoder
call) so the per-epoch monitor is fast. The honest autoregressive °C error is
reported separately by the evaluator. Metrics here live in the model's working
space (MinMax-scaled [0, 1]), matching the optimisation objective.
"""
import copy
import os

import numpy as np
import torch


# ── Single epoch pass ─────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, criterion, device, train, scaler=None):
    model.train(train)
    use_amp = scaler is not None and scaler.is_enabled()
    sse = sae = count = 0.0   # streaming accumulators → no giant pred buffers

    with torch.set_grad_enabled(train):
        for X_b, yf_b, y_b in loader:
            X_b  = X_b.to(device, non_blocking=True)
            yf_b = yf_b.to(device, non_blocking=True)
            y_b  = y_b.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out  = model(X_b, yf_b, y_b, teacher_forcing=True)   # (B, 168)
                    loss = criterion(out, y_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    out = model(X_b, yf_b, y_b, teacher_forcing=True)

            diff   = (out.float() - y_b).detach()
            sse   += torch.sum(diff ** 2).item()
            sae   += torch.sum(torch.abs(diff)).item()
            count += diff.numel()

    mse  = sse / count
    rmse = float(np.sqrt(mse))
    mae  = sae / count
    return mse, rmse, mae


# ── Full training loop ─────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                epochs: int = 5, patience: int = 2, config: dict = None,
                models_dir: str = 'models', ckpt_prefix: str = 'lstm_temp_weekly',
                use_amp: bool = True) -> dict:
    history = {k: [] for k in [
        'epoch',
        'train_mse', 'train_rmse', 'train_mae',
        'val_mse',   'val_rmse',   'val_mae',
    ]}

    os.makedirs(models_dir, exist_ok=True)
    amp_on = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)
    if amp_on:
        print("  Mixed precision (AMP) : ENABLED")

    best_val   = float('inf')
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_mse, tr_rmse, tr_mae = _run_epoch(
            model, train_loader, optimizer, criterion, device, True, scaler
        )
        vl_mse, vl_rmse, vl_mae = _run_epoch(
            model, val_loader, None, criterion, device, False, scaler
        )

        for key, val in zip(history.keys(), [
            epoch,
            tr_mse, tr_rmse, tr_mae,
            vl_mse, vl_rmse, vl_mae,
        ]):
            history[key].append(val)

        # ── Per-epoch checkpoint ──────────────────────────────────────────────
        ckpt_path = os.path.join(models_dir, f"{ckpt_prefix}_epoch_{epoch:02d}.pt")
        torch.save(
            {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config':               config or {},
                'metrics': {
                    'train_mse': tr_mse, 'train_rmse': tr_rmse, 'train_mae': tr_mae,
                    'val_mse':   vl_mse, 'val_rmse':   vl_rmse, 'val_mae':   vl_mae,
                },
            },
            ckpt_path,
        )

        improved = vl_mse < best_val
        if improved:
            best_val, best_epoch = vl_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch [{epoch:02d}/{epochs}]  "
            f"Train MSE: {tr_mse:.5f}, Val MSE: {vl_mse:.5f}  |  "
            f"Val RMSE: {vl_rmse:.5f}, Val MAE: {vl_mae:.5f}  "
            f"{'  <-- best' if improved else f'  (no improve {no_improve}/{patience})'}"
            f"  -> {ckpt_path}"
        )

        # ── Early stopping ────────────────────────────────────────────────────
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best epoch {best_epoch}, val MSE {best_val:.5f}).")
            break

    # Restore best weights so downstream evaluation uses the best epoch.
    model.load_state_dict(best_state)
    print(f"  Restored best weights from epoch {best_epoch} (val MSE {best_val:.5f}).")
    history['best_epoch'] = best_epoch
    return history
