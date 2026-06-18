"""
Task: Run the training loop — one epoch of gradient updates over the train set,
      followed by inference over the val set, logging MSE / RMSE / MAE for both.

Metrics here are computed in the model's working space (MinMax-scaled [0, 1]),
which matches the optimisation objective. The real-unit (°C) error is reported
separately by the evaluator via the MinMax inverse transform.
"""
import numpy as np
import torch


# ── Metric helper ─────────────────────────────────────────────────────────────

def _metrics(preds: np.ndarray, targets: np.ndarray):
    mse  = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - targets)))
    return mse, rmse, mae


# ── Single epoch pass ─────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    sse = sae = count = 0.0   # streaming accumulators → no giant pred buffers

    with torch.set_grad_enabled(train):
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            out  = model(X_b)              # (batch, 168)
            loss = criterion(out, y_b)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            diff   = (out - y_b).detach()
            sse   += torch.sum(diff ** 2).item()
            sae   += torch.sum(torch.abs(diff)).item()
            count += diff.numel()

    mse  = sse / count
    rmse = float(np.sqrt(mse))
    mae  = sae / count
    return mse, rmse, mae


# ── Full training loop ─────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs: int = 15) -> dict:
    history = {k: [] for k in [
        'epoch',
        'train_mse', 'train_rmse', 'train_mae',
        'val_mse',   'val_rmse',   'val_mae',
    ]}

    for epoch in range(1, epochs + 1):
        tr_mse, tr_rmse, tr_mae = _run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        vl_mse, vl_rmse, vl_mae = _run_epoch(
            model, val_loader, None, criterion, device, train=False
        )

        for key, val in zip(history.keys(), [
            epoch,
            tr_mse, tr_rmse, tr_mae,
            vl_mse, vl_rmse, vl_mae,
        ]):
            history[key].append(val)

        print(
            f"Epoch [{epoch:02d}/{epochs}]  "
            f"Train MSE: {tr_mse:.5f}, Val MSE: {vl_mse:.5f}  |  "
            f"Train RMSE: {tr_rmse:.5f}, Val RMSE: {vl_rmse:.5f}  |  "
            f"Train MAE: {tr_mae:.5f}, Val MAE: {vl_mae:.5f}"
        )

    return history
