"""
Task: Evaluate the trained model on the held-out test Dataset and
      print a formatted summary table of Test MSE / RMSE / MAE.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_test(model, test_dataset, device, batch_size: int = 1024) -> tuple[float, float, float]:
    loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'),
    )

    model.eval()
    preds_buf, tgts_buf = [], []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device, non_blocking=True)
            out = model(X_b)
            preds_buf.append(out.cpu().numpy())
            tgts_buf.append(y_b.numpy())

    preds   = np.concatenate(preds_buf)
    targets = np.concatenate(tgts_buf)

    mse  = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(preds - targets)))

    w = 54
    print()
    print("=" * w)
    print(f"{'TEST SET EVALUATION RESULTS':^{w}}")
    print("=" * w)
    print(f"  {'Metric':<22} {'Value':>16}")
    print("-" * w)
    print(f"  {'Test MSE':<22} {mse:>16.6f}")
    print(f"  {'Test RMSE (deg C)':<22} {rmse:>16.6f}")
    print(f"  {'Test MAE  (deg C)':<22} {mae:>16.6f}")
    print("=" * w)

    return mse, rmse, mae
