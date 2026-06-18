"""
Task: Persist the training history CSV and render the 8 report-grade figures into
      the 'plots/' directory.

All rainfall quantities passed in here are already in REAL units (mm/hour) — the
caller inverts log1p + MinMax via training.evaluator.inverse_target first.

Figures
  1-3. plots/loss_mse.png, loss_rmse.png, loss_mae.png  — Train vs Val curves.
  4.   plots/scatter_actual_vs_pred.png                 — actual vs predicted (+ R²).
  5.   plots/sample_prediction_comparison.png           — one coord, 168h actual vs forecast.
  6.   plots/residuals_histogram.png                    — distribution of (actual − pred).
  7.   plots/feature_correlation_heatmap.png            — input-feature correlation matrix.
  8.   plots/spatial_error_heatmap.png                  — per-coordinate MAE map of VN.
  9.   plots/feature_distributions.png                  — rainfall raw vs log1p.
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
HISTORY_CSV_PATH = 'training_history_rain.csv'

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
        ax.set_ylabel(f'{label} (scaled-log space)')
        ax.set_title(f'LSTM Weekly Rain — Train vs Val {label}')
        ax.legend()
        ax.set_xticks(epochs)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Loss curve       -> {os.path.join(out_dir, fname)}")


# ── 4. Scatter actual vs predicted ───────────────────────────────────────────

def plot_scatter_actual_vs_pred(targets_mm, preds_mm, out_dir: str = PLOTS_DIR) -> None:
    a, p = _subsample_flat(targets_mm, preds_mm)
    r2 = r2_score(targets_mm.ravel(), preds_mm.ravel())

    lim = max(a.max(), p.max()) * 1.02
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a, p, s=6, alpha=0.25, edgecolors='none')
    ax.plot([0, lim], [0, lim], 'r--', linewidth=2, label='Ideal  y = x')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('Actual rainfall (mm/hour)')
    ax.set_ylabel('Predicted rainfall (mm/hour)')
    ax.set_title(f'Actual vs Predicted Rainfall  (R² = {r2:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'scatter_actual_vs_pred.png'), dpi=150)
    plt.close(fig)
    print(f"  Scatter A/P      -> {os.path.join(out_dir, 'scatter_actual_vs_pred.png')}")


# ── 5. Sample 168h forecast vs actual (one coordinate) ───────────────────────

def plot_sample_prediction(targets_mm, preds_mm, coord_ids, coords,
                           out_dir: str = PLOTS_DIR) -> None:
    i = int(_RNG.integers(len(preds_mm)))
    lat, lon = coords[coord_ids[i]]
    hours = np.arange(targets_mm.shape[1])

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, targets_mm[i], color='royalblue', linewidth=2,
            label='Thực tế (actual)')
    ax.plot(hours, preds_mm[i], color='deeppink', linestyle='--', linewidth=2,
            label='Dự báo AI (forecast)')
    ax.set_xlabel('Giờ tương lai (0 → 168h)')
    ax.set_ylabel('Lượng mưa (mm/hour)')
    ax.set_title(f'Dự báo 1 tuần tới tại tọa độ (lat={lat:.2f}, lon={lon:.2f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'sample_prediction_comparison.png'), dpi=150)
    plt.close(fig)
    print(f"  Sample forecast  -> {os.path.join(out_dir, 'sample_prediction_comparison.png')}")


# ── 6. Residuals histogram ───────────────────────────────────────────────────

def plot_residuals_histogram(targets_mm, preds_mm, out_dir: str = PLOTS_DIR) -> None:
    resid = (targets_mm - preds_mm).ravel()
    (resid,) = _subsample_flat(resid, n=60000)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(resid, bins=80, kde=True, ax=ax, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Residual = Actual − Predicted (mm/hour)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution  (mean = {resid.mean():.4f} mm/h)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'residuals_histogram.png'), dpi=150)
    plt.close(fig)
    print(f"  Residual hist    -> {os.path.join(out_dir, 'residuals_histogram.png')}")


# ── 7. Feature correlation heatmap ───────────────────────────────────────────

def plot_feature_correlation_heatmap(df_sample, feature_cols,
                                     target_col: str = 'total_precipitation',
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

def plot_spatial_error_heatmap(targets_mm, preds_mm, coord_ids, coords,
                               out_dir: str = PLOTS_DIR) -> None:
    abs_err = np.abs(targets_mm - preds_mm).mean(axis=1)   # per-sample MAE

    lats, lons, maes = [], [], []
    for cid in np.unique(coord_ids):
        mask = coord_ids == cid
        lat, lon = coords[cid]
        lats.append(lat); lons.append(lon)
        maes.append(float(abs_err[mask].mean()))

    fig, ax = plt.subplots(figsize=(8, 10))
    sc = ax.scatter(lons, lats, c=maes, cmap='YlOrRd', s=60,
                    edgecolors='k', linewidths=0.3)
    fig.colorbar(sc, ax=ax, label='Mean Absolute Error (mm/hour)')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_title('Spatial Forecast Error Across Vietnam')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'spatial_error_heatmap.png'), dpi=150)
    plt.close(fig)
    print(f"  Spatial error    -> {os.path.join(out_dir, 'spatial_error_heatmap.png')}")


# ── 9. Rainfall distribution: raw vs log1p ───────────────────────────────────

def plot_feature_distributions(raw_precip, out_dir: str = PLOTS_DIR) -> None:
    raw = np.asarray(raw_precip, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    raw = np.clip(raw, 0, None)
    logged = np.log1p(raw)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(raw, bins=80, ax=axes[0], color='indianred')
    axes[0].set_title('Raw total_precipitation (right-skewed)')
    axes[0].set_xlabel('mm/hour'); axes[0].set_ylabel('Frequency')

    sns.histplot(logged, bins=80, ax=axes[1], color='seagreen')
    axes[1].set_title('log1p(total_precipitation) (compressed tail)')
    axes[1].set_xlabel('log1p(mm/hour)'); axes[1].set_ylabel('Frequency')

    fig.suptitle('Why we log-transform rainfall before training')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'feature_distributions.png'), dpi=150)
    plt.close(fig)
    print(f"  Distributions    -> {os.path.join(out_dir, 'feature_distributions.png')}")


# ── 10. Residuals vs fitted ──────────────────────────────────────────────────

def plot_residuals_vs_fitted(targets_mm, preds_mm, out_dir: str = PLOTS_DIR) -> None:
    fitted, resid = _subsample_flat(preds_mm, targets_mm - preds_mm)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(fitted, resid, s=6, alpha=0.25, edgecolors='none')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Fitted / predicted rainfall (mm/hour)')
    ax.set_ylabel('Residual = Actual − Predicted (mm/hour)')
    ax.set_title('Residuals vs Fitted Values')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'residuals_vs_fitted.png'), dpi=150)
    plt.close(fig)
    print(f"  Resid vs fitted  -> {os.path.join(out_dir, 'residuals_vs_fitted.png')}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def save_all_plots(history, targets_mm, preds_mm, coord_ids, coords,
                   df_sample, feature_cols, raw_precip,
                   target_col: str = 'total_precipitation',
                   out_dir: str = PLOTS_DIR) -> None:
    ensure_plots_dir(out_dir)
    plot_loss_curves(history, out_dir)
    plot_scatter_actual_vs_pred(targets_mm, preds_mm, out_dir)
    plot_sample_prediction(targets_mm, preds_mm, coord_ids, coords, out_dir)
    plot_residuals_histogram(targets_mm, preds_mm, out_dir)
    plot_feature_correlation_heatmap(df_sample, feature_cols, target_col, out_dir)
    plot_spatial_error_heatmap(targets_mm, preds_mm, coord_ids, coords, out_dir)
    plot_feature_distributions(raw_precip, out_dir)
    plot_residuals_vs_fitted(targets_mm, preds_mm, out_dir)
