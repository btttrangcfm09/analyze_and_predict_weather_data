"""
Task: (1) Plot Train vs Val MSE loss curve and save as 'loss_curve.png'.
      (2) Save the full per-epoch history (MSE, RMSE, MAE) to 'training_history.csv'.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for Kaggle notebooks
import matplotlib.pyplot as plt

LOSS_CURVE_PATH = 'loss_curve.png'
HISTORY_CSV_PATH = 'training_history.csv'


def plot_loss_curve(history: dict, save_path: str = LOSS_CURVE_PATH) -> None:
    epochs = history['epoch']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_mse'], 'o-',  label='Train MSE', linewidth=2)
    ax.plot(epochs, history['val_mse'],   's--', label='Val MSE',   linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('LSTM Temperature Model — Train vs Val MSE Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve      -> {save_path}")


def save_history(history: dict, save_path: str = HISTORY_CSV_PATH) -> None:
    pd.DataFrame(history).to_csv(save_path, index=False)
    print(f"  Training history -> {save_path}")
