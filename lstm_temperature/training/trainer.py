"""
Task: Run the training loop — one epoch of gradient updates on train set,
      followed by inference on val set, then log MSE / RMSE / MAE for both.
"""
import numpy as np
import torch


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _metrics(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float, float]:
    mse  = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - targets)))
    return mse, rmse, mae


# ── Single epoch pass ──────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    preds_buf, tgts_buf = [], []

    with torch.set_grad_enabled(train):
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            out  = model(X_b)
            loss = criterion(out, y_b)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds_buf.append(out.detach().cpu().numpy())
            tgts_buf.append(y_b.detach().cpu().numpy())

    preds   = np.concatenate(preds_buf)
    targets = np.concatenate(tgts_buf)
    return _metrics(preds, targets)


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
            f"Train MSE: {tr_mse:.4f}, Val MSE: {vl_mse:.4f}  |  "
            f"Train RMSE: {tr_rmse:.4f}, Val RMSE: {vl_rmse:.4f}  |  "
            f"Train MAE: {tr_mae:.4f}, Val MAE: {vl_mae:.4f}"
        )

    return history
