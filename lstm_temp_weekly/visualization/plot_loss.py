"""
Task: Persist the training history CSV and render the report-grade figures into
      the 'plots/' directory.

All temperature quantities passed in here are already in REAL units (°C) — the
caller inverts MinMax via training.evaluator.inverse_target first.

Figures
  1-3. plots/loss_mse.png, loss_rmse.png, loss_mae.png  — Train vs Val curves.
  4.   plots/scatter_actual_vs_pred.png                 — actual vs predicted (+ R²).
  5.   plots/sample_prediction_comparison.png           — one coord, 168h actual vs forecast.
  6.   plots/residuals_histogram.png                    — distribution of (actual − pred).
  7.   plots/feature_correlation_heatmap.png            — input-feature correlation matrix.
  8.   plots/spatial_error_heatmap.png                  — per-coordinate MAE map of VN.
  9.   plots/feature_distributions.png                  — temperature frequency distribution.
  10.  plots/residuals_vs_fitted.png                    — residuals vs fitted values.
"""
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # headless backend — safe inside Kaggle notebooks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

PLOTS_DIR        = 'plots'
HISTORY_CSV_PATH = 'training_history_temp.csv'

sns.set_theme(style='whitegrid')
_RNG = np.random.default_rng(7)


# ── Infrastructure ────────────────────────────────────────────────────────────

def ensure_plots_dir(path: str = PLOTS_DIR) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_history(history: dict, save_path: str = HISTORY_CSV_PATH) -> None:
    pd.DataFrame(history).to_csv(save_path, index=False)
    print(f"  Training history -> {save_path}")


def _subsample_flat(*arrays, n: int = 25000):
    """Flatten then take a shared random subset of points (for scatter/hist)."""
    flat = [a.ravel() for a in arrays]
    total = flat[0].size
    if total > n:
        pick = _RNG.choice(total, size=n, replace=False)
        flat = [f[pick] for f in flat]
    return flat


# ── 1-3. Loss curves ─────────────────────────────────────────────────────────

def plot_loss_curves(history: dict, out_dir: str = PLOTS_DIR) -> None:
    epochs = history['epoch']
    specs = [
        ('mse',  'MSE',  'loss_mse.png'),
        ('rmse', 'RMSE', 'loss_rmse.png'),
        ('mae',  'MAE',  'loss_mae.png'),
    ]
    for key, label, fname in specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, history[f'train_{key}'], 'o-',  label=f'Train {label}', linewidth=2)
        ax.plot(epochs, history[f'val_{key}'],   's--', label=f'Val {label}',   linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'{label} (scaled [0, 1] space)')
        ax.set_title(f'LSTM Weekly Temperature — Train vs Val {label}')
        ax.legend()
        ax.set_xticks(epochs)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Loss curve       -> {os.path.join(out_dir, fname)}")


# ── 4. Scatter actual vs predicted ───────────────────────────────────────────

def plot_scatter_actual_vs_pred(targets_c, preds_c, out_dir: str = PLOTS_DIR) -> None:
    a, p = _subsample_flat(targets_c, preds_c)
    r2 = r2_score(targets_c.ravel(), preds_c.ravel())

    lo = min(a.min(), p.min())
    hi = max(a.max(), p.max())
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a, p, s=6, alpha=0.25, edgecolors='none')
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Ideal  y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Actual temperature (°C)')
    ax.set_ylabel('Predicted temperature (°C)')
    ax.set_title(f'Actual vs Predicted Temperature  (R² = {r2:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scatter_actual_vs_pred.png'), dpi=150)
    plt.close(fig)
    print(f"  Scatter A/P      -> {os.path.join(out_dir, 'scatter_actual_vs_pred.png')}")


# ── 5. Sample 168h forecast vs actual (one coordinate) ───────────────────────

def plot_sample_prediction(targets_c, preds_c, coord_ids, coords,
                           out_dir: str = PLOTS_DIR) -> None:
    i = int(_RNG.integers(len(preds_c)))
    lat, lon = coords[coord_ids[i]]
    hours = np.arange(targets_c.shape[1])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, targets_c[i], color='royalblue', linewidth=2,
            label='Thực tế (actual)')
    ax.plot(hours, preds_c[i], color='deeppink', linestyle='--', linewidth=2,
            label='Dự báo AI (forecast)')
    ax.set_xlabel('Giờ tương lai (0 → 168h)')
    ax.set_ylabel('Nhiệt độ (°C)')
    ax.set_title(f'Dự báo nhiệt độ 1 tuần tới tại tọa độ (lat={lat:.2f}, lon={lon:.2f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sample_prediction_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  Sample forecast  -> {os.path.join(out_dir, 'sample_prediction_comparison.png')}")


# ── 6. Residuals histogram ───────────────────────────────────────────────────

def plot_residuals_histogram(targets_c, preds_c, out_dir: str = PLOTS_DIR) -> None:
    resid = (targets_c - preds_c).ravel()
    (resid,) = _subsample_flat(resid, n=60000)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(resid, bins=80, kde=True, ax=ax, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Residual = Actual − Predicted (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution  (mean = {resid.mean():.4f} °C)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'residuals_histogram.png'), dpi=150)
    plt.close(fig)
    print(f"  Residual hist    -> {os.path.join(out_dir, 'residuals_histogram.png')}")


# ── 7. Feature correlation heatmap ───────────────────────────────────────────

def plot_feature_correlation_heatmap(df_sample, feature_cols,
                                     target_col: str = 'temperature_celsius',
                                     out_dir: str = PLOTS_DIR) -> None:
    cols = [c for c in feature_cols if c in df_sample.columns]
    corr = df_sample[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                linewidths=0.4, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title(f'Input Feature Correlation Matrix (target: {target_col})')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_correlation_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"  Corr heatmap     -> {os.path.join(out_dir, 'feature_correlation_heatmap.png')}")


# ── 8. Spatial error heatmap ─────────────────────────────────────────────────

def plot_spatial_error_heatmap(targets_c, preds_c, coord_ids, coords,
                               out_dir: str = PLOTS_DIR) -> None:
    abs_err = np.abs(targets_c - preds_c).mean(axis=1)   # per-sample MAE

    lats, lons, maes = [], [], []
    for cid in np.unique(coord_ids):
        mask = coord_ids == cid
        lat, lon = coords[cid]
        lats.append(lat); lons.append(lon)
        maes.append(float(abs_err[mask].mean()))

    fig, ax = plt.subplots(figsize=(8, 10))
    sc = ax.scatter(lons, lats, c=maes, cmap='YlOrRd', s=60,
                    edgecolors='k', linewidths=0.3)
    fig.colorbar(sc, ax=ax, label='Mean Absolute Error (°C)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Spatial Temperature Forecast Error Across Vietnam')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'spatial_error_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"  Spatial error    -> {os.path.join(out_dir, 'spatial_error_heatmap.png')}")


# ── 9. Temperature distribution (natural bell shape) ─────────────────────────

def plot_feature_distributions(raw_temp, out_dir: str = PLOTS_DIR) -> None:
    temp = np.asarray(raw_temp, dtype=np.float64)
    temp = temp[np.isfinite(temp)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(temp, bins=80, kde=True, ax=ax, color='darkorange')
    ax.axvline(temp.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {temp.mean():.2f} °C')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Actual Temperature '
                 '(naturally bell-shaped — no log transform needed)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_distributions.png'), dpi=150)
    plt.close(fig)
    print(f"  Distributions    -> {os.path.join(out_dir, 'feature_distributions.png')}")


# ── 10. Residuals vs fitted ──────────────────────────────────────────────────

def plot_residuals_vs_fitted(targets_c, preds_c, out_dir: str = PLOTS_DIR) -> None:
    fitted, resid = _subsample_flat(preds_c, targets_c - preds_c)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fitted, resid, s=6, alpha=0.25, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted / predicted temperature (°C)')
    ax.set_ylabel('Residual = Actual − Predicted (°C)')
    ax.set_title('Residuals vs Fitted Values')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'residuals_vs_fitted.png'), dpi=150)
    plt.close(fig)
    print(f"  Resid vs fitted  -> {os.path.join(out_dir, 'residuals_vs_fitted.png')}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def save_all_plots(history, targets_c, preds_c, coord_ids, coords,
                   df_sample, feature_cols, raw_temp,
                   target_col: str = 'temperature_celsius',
                   out_dir: str = PLOTS_DIR) -> None:
    ensure_plots_dir(out_dir)
    plot_loss_curves(history, out_dir)
    plot_scatter_actual_vs_pred(targets_c, preds_c, out_dir)
    plot_sample_prediction(targets_c, preds_c, coord_ids, coords, out_dir)
    plot_residuals_histogram(targets_c, preds_c, out_dir)
    plot_feature_correlation_heatmap(df_sample, feature_cols, target_col, out_dir)
    plot_spatial_error_heatmap(targets_c, preds_c, coord_ids, coords, out_dir)
    plot_feature_distributions(raw_temp, out_dir)
    plot_residuals_vs_fitted(targets_c, preds_c, out_dir)
