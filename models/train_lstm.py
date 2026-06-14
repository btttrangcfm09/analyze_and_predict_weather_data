"""
GIAI ĐOẠN 2: KỸ SƯ AI & MACHINE LEARNING
===========================================
Mô hình LSTM dự đoán thời tiết (Nhiệt độ & Lượng mưa) cho Việt Nam.
Sử dụng PyTorch (thay thế TensorFlow do chưa hỗ trợ Python 3.14).

Bước 2.1 - Feature Engineering
Bước 2.2 - Chuẩn hóa (Scaling)
Bước 2.3 - Xây dựng LSTM
Bước 2.4 - Huấn luyện & Backtesting
Bước 2.5 - Đóng gói AI (export model)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH
# ============================================================
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dữ liệu lịch sử lấy trực tiếp từ Kaggle (kagglehub tự cache, không lưu trong repo)
import kagglehub
KAGGLE_DATASET = "nguyentranggggg/vietnam-meteorological-weather-data-2020-2026"
DATA_PATH = os.path.join(kagglehub.dataset_download(KAGGLE_DATASET), "weather.parquet")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
TRAIN_RATIO = 0.8

FEATURE_COLS_RAW = [
    'temperature_celsius', 'total_precipitation', 'wind_speed', 'wind_direction',
    'relative_humidity', 'mean_sea_level_pressure', 'surface_pressure',
    'total_cloud_cover', 'apparent_temperature', 'air_density',
]
TARGET_COLS = ['temperature_celsius']

# ============================================================
# BƯỚC 2.1 - FEATURE ENGINEERING
# ============================================================
def load_and_preprocess_data():
    print("=" * 60)
    print("BƯỚC 2.1 - FEATURE ENGINEERING")
    print("=" * 60)

    print("\n[1/4] Đang tải dữ liệu...")
    df = pd.read_parquet(DATA_PATH)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['date'] = df['valid_time'].dt.date
    df = df[(df['latitude'] >= 8) & (df['latitude'] <= 24)]
    df = df[(df['longitude'] >= 102) & (df['longitude'] <= 112)]
    print(f"  Kich thuoc: {df.shape}")

    print("\n[2/4] Xu ly missing values...")
    for col in FEATURE_COLS_RAW:
        df[col] = df.groupby(['latitude', 'longitude'])[col].transform(
            lambda x: x.interpolate(method='linear').bfill().ffill()
        )
    print(f"  Missing con lai: {df[FEATURE_COLS_RAW].isnull().sum().sum()}")

    print("\n[2.5/4] Log transform luong mua (giam nhieu)...")
    # Biến đổi logarit để trị các đỉnh mưa đột biến (spikes)
    df['total_precipitation'] = np.log1p(df['total_precipitation'])

    print("\n[3/4] Tao dac trung thoi gian...")
    df['day_of_week'] = df['valid_time'].dt.dayofweek
    df['day_of_year'] = df['valid_time'].dt.dayofyear
    df['month_num'] = df['valid_time'].dt.month

    def get_season(m):
        if m in [2,3,4]: return 0
        if m in [5,6,7]: return 1
        if m in [8,9,10]: return 2
        return 3
    df['season'] = df['month_num'].apply(get_season)
    df['day_of_week_sin'] = np.sin(2*np.pi*df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2*np.pi*df['day_of_week']/7)
    df['month_sin'] = np.sin(2*np.pi*df['month_num']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month_num']/12)
    df['day_of_year_sin'] = np.sin(2*np.pi*df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2*np.pi*df['day_of_year']/365)

    print("\n[4/4] Tao du lieu ngay + lag features...")
    agg_dict = {c: 'mean' for c in FEATURE_COLS_RAW}
    agg_dict.update({'day_of_week':'first','day_of_year':'first','month_num':'first',
                     'season':'first','day_of_week_sin':'first','day_of_week_cos':'first',
                     'month_sin':'first','month_cos':'first','day_of_year_sin':'first','day_of_year_cos':'first'})
    daily = df.groupby(['latitude','longitude','date']).agg(agg_dict).reset_index()

    for col in ['temperature_celsius','total_precipitation']:
        for lag in [1,3,7]:
            daily[f'{col}_lag{lag}'] = daily.groupby(['latitude','longitude'])[col].shift(lag)
        for w in [3,7]:
            daily[f'{col}_roll{w}'] = daily.groupby(['latitude','longitude'])[col].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
    daily = daily.dropna()
    print(f"  Du lieu ngay: {daily.shape}")
    return daily

# ============================================================
# BƯỚC 2.2 - SCALING
# ============================================================
def scale_data(daily):
    print("\n" + "="*60)
    print("BƯỚC 2.2 - CHUAN HOA (MinMaxScaler)")
    print("="*60)

    all_feats = FEATURE_COLS_RAW + [
        'day_of_week_sin','day_of_week_cos','month_sin','month_cos',
        'day_of_year_sin','day_of_year_cos','season'
    ]
    for col in ['temperature_celsius','total_precipitation']:
        for lag in [1,3,7]: all_feats.append(f'{col}_lag{lag}')
        for w in [3,7]: all_feats.append(f'{col}_roll{w}')

    numeric_feats = [c for c in all_feats if c != 'season']
    scaler = MinMaxScaler()
    daily[numeric_feats] = scaler.fit_transform(daily[numeric_feats])

    joblib.dump(scaler, os.path.join(OUTPUT_DIR,'scaler_temp.pkl'))
    joblib.dump(all_feats, os.path.join(OUTPUT_DIR,'feature_cols_temp.pkl'))
    print(f"  So dac trung: {len(all_feats)}")
    print(f"  Da luu scaler.pkl va feature_cols.pkl")
    return daily, all_feats, scaler

# ============================================================
# BƯỚC 2.3 - MO HINH LSTM
# ============================================================
class WeatherDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.drop(self.relu(self.fc1(out)))
        return self.fc2(out)

def make_sequences(data, feat_cols, tgt_cols, seq_len):
    X, y = [], []
    fd = data[feat_cols].values
    td = data[tgt_cols].values
    for i in range(len(data)-seq_len):
        X.append(fd[i:i+seq_len])
        y.append(td[i+seq_len])
    return np.array(X), np.array(y)

# ============================================================
# BƯỚC 2.4 - HUAN LUYEN & BACKTESTING
# ============================================================
def train_and_evaluate(daily, feat_cols, scaler):
    print("\n" + "="*60)
    print("BƯỚC 2.3 & 2.4 - XAY DUNG LSTM + HUAN LUYEN")
    print("="*60)

    # Chon toa do Ha Noi (21.0, 105.8)
    lat, lon = 21.0, 105.8
    coords = daily[['latitude','longitude']].drop_duplicates()
    dists = np.sqrt((coords['latitude']-lat)**2 + (coords['longitude']-lon)**2)
    near = coords.loc[dists.idxmin()]
    loc = daily[(daily['latitude']==near['latitude'])&(daily['longitude']==near['longitude'])]
    loc = loc.sort_values('date').reset_index(drop=True)
    print(f"  Toa do: ({near['latitude']}, {near['longitude']})")
    print(f"  So mau: {len(loc)}")

    X, y = make_sequences(loc, feat_cols, TARGET_COLS, SEQUENCE_LENGTH)
    print(f"  X: {X.shape}, y: {y.shape}")

    split = int(len(X)*TRAIN_RATIO)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    print(f"  Train: {len(Xtr)}, Test: {len(Xte)}")

    train_loader = DataLoader(WeatherDS(Xtr,ytr), BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(WeatherDS(Xte,yte), BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = LSTMModel(Xtr.shape[2], HIDDEN_SIZE, NUM_LAYERS, len(TARGET_COLS), DROPOUT).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Kien truc: {model}")
    print(f"  Tong tham so: {total_p:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    train_losses, test_losses = [], []
    best_loss = float('inf')
    best_state = None

    print(f"\n  Bat dau huan luyen {EPOCHS} epochs...")
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)
        train_losses.append(tl)

        model.eval()
        vl = 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                vl += criterion(model(bx), by).item()
        vl /= len(test_loader)
        test_losses.append(vl)
        scheduler.step(vl)

        if vl < best_loss:
            best_loss = vl
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        if (ep+1)%10==0 or ep==0:
            print(f"  Epoch {ep+1:3d}/{EPOCHS} | Train: {tl:.6f} | Test: {vl:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    model.load_state_dict(best_state)
    print(f"\n  Mo hinh tot nhat - Test Loss: {best_loss:.6f}")

    # Danh gia
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            tgts.append(by.numpy())
    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)

    print("\n  +-----------------------------------------------------+")
    print("  |           KET QUA DANH GIA MO HINH                  |")
    print("  +-----------------------------------------------------+")
    numeric_feats = [c for c in feat_cols if c != 'season']
    for i, name in enumerate(TARGET_COLS):
        p, t = preds[:,i], tgts[:,i]
        # inverse transform - use numeric features only
        d = np.zeros((len(p), len(numeric_feats)))
        idx = numeric_feats.index(name)
        d[:,idx] = p; po = scaler.inverse_transform(d)[:,idx]
        d2 = np.zeros((len(t), len(numeric_feats)))
        d2[:,idx] = t; to = scaler.inverse_transform(d2)[:,idx]
        
        if name == 'total_precipitation':
            po = np.expm1(po)
            to = np.expm1(to)
            
        rmse = np.sqrt(mean_squared_error(to, po))
        mae = mean_absolute_error(to, po)
        print(f"  |  {name}")
        print(f"  |    RMSE: {rmse:.4f}  |  MAE: {mae:.4f}")
    print("  +-----------------------------------------------------+")

    # Ve bieu do (Chi 1 dong 2 cot cho Nhiet do)
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    fig.suptitle('Ket qua LSTM - Du doan Nhiet do Viet Nam', fontsize=14, fontweight='bold')

    axes[0].plot(train_losses, label='Train', color='blue', alpha=0.7)
    axes[0].plot(test_losses, label='Test', color='red', alpha=0.7)
    axes[0].set_title('Training & Test Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(tgts[:200,0], label='Thuc te', color='blue', alpha=0.7)
    axes[1].plot(preds[:200,0], label='Du doan', color='red', alpha=0.7, ls='--')
    axes[1].set_title('Nhiet do: Thuc te vs Du doan'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,'training_results_temp.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Da luu training_results.png")

    return model

# ============================================================
# BƯỚC 2.5 - EXPORT MODEL
# ============================================================
def export_model(model):
    print("\n" + "="*60)
    print("BƯỚC 2.5 - DONG GOI AI")
    print("="*60)
    path = os.path.join(OUTPUT_DIR, 'lstm_weather_model_temp.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': model.lstm.input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'output_size': model.fc2.out_features,
        'dropout': DROPOUT,
        'sequence_length': SEQUENCE_LENGTH,
        'target_cols': TARGET_COLS,
        # Chỉ lưu TÊN FILE, không lưu đường dẫn tuyệt đối -> resolve theo vị trí .pt khi load
        'scaler_name': 'scaler_temp.pkl',
        'feature_cols_name': 'feature_cols_temp.pkl',
    }, path)
    print(f"  Da luu model: {path}")
    print(f"  Da luu scaler: {os.path.join(OUTPUT_DIR, 'scaler_temp.pkl')}")
    print(f"  Da luu feature_cols: {os.path.join(OUTPUT_DIR, 'feature_cols_temp.pkl')}")

# ============================================================
# HAM DU DOAN CHO STREAMLIT (BƯỚC 3)
# ============================================================
def predict_tomorrow(lat, lon):
    """
    Du doan nhiet do va luong mua ngay mai cho toa do (lat, lon).
    Su dung 30 ngay gan nhat lam input.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(os.path.join(OUTPUT_DIR,'lstm_weather_model_temp.pt'), map_location=device, weights_only=False)
    # Resolve theo OUTPUT_DIR; fallback cho checkpoint cũ (lưu *_path tuyệt đối)
    feat_name = ckpt.get('feature_cols_name') or os.path.basename(ckpt['feature_cols_path'])
    scaler_name = ckpt.get('scaler_name') or os.path.basename(ckpt['scaler_path'])
    feat_cols = joblib.load(os.path.join(OUTPUT_DIR, feat_name))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, scaler_name))

    # Rebuild model
    model = LSTMModel(ckpt['input_size'], ckpt['hidden_size'], ckpt['num_layers'],
                      ckpt['output_size'], ckpt['dropout']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load data
    df = pd.read_parquet(DATA_PATH)
    df['valid_time'] = pd.to_datetime(df['valid_time'])
    df['date'] = df['valid_time'].dt.date

    # Find nearest coords
    coords = df[['latitude','longitude']].drop_duplicates()
    dists = np.sqrt((coords['latitude']-lat)**2+(coords['longitude']-lon)**2)
    near = coords.loc[dists.idxmin()]
    loc = df[(df['latitude']==near['latitude'])&(df['longitude']==near['longitude'])]

    # Feature engineering (same as training)
    for col in FEATURE_COLS_RAW:
        loc[col] = loc[col].interpolate('linear').bfill().ffill()
    loc['total_precipitation'] = np.log1p(loc['total_precipitation'])
    loc['day_of_week'] = loc['valid_time'].dt.dayofweek
    loc['day_of_year'] = loc['valid_time'].dt.dayofyear
    loc['month_num'] = loc['valid_time'].dt.month
    loc['season'] = loc['month_num'].apply(lambda m: 0 if m in[2,3,4] else 1 if m in[5,6,7] else 2 if m in[8,9,10] else 3)
    loc['day_of_week_sin'] = np.sin(2*np.pi*loc['day_of_week']/7)
    loc['day_of_week_cos'] = np.cos(2*np.pi*loc['day_of_week']/7)
    loc['month_sin'] = np.sin(2*np.pi*loc['month_num']/12)
    loc['month_cos'] = np.cos(2*np.pi*loc['month_num']/12)
    loc['day_of_year_sin'] = np.sin(2*np.pi*loc['day_of_year']/365)
    loc['day_of_year_cos'] = np.cos(2*np.pi*loc['day_of_year']/365)

    agg = {c:'mean' for c in FEATURE_COLS_RAW}
    agg.update({'day_of_week':'first','day_of_year':'first','month_num':'first',
                'season':'first','day_of_week_sin':'first','day_of_week_cos':'first',
                'month_sin':'first','month_cos':'first','day_of_year_sin':'first','day_of_year_cos':'first'})
    daily = loc.groupby(['latitude','longitude','date']).agg(agg).reset_index()

    for col in ['temperature_celsius','total_precipitation']:
        for lag in [1,3,7]: daily[f'{col}_lag{lag}'] = daily.groupby(['latitude','longitude'])[col].shift(lag)
        for w in [3,7]: daily[f'{col}_roll{w}'] = daily.groupby(['latitude','longitude'])[col].transform(lambda x: x.rolling(w,min_periods=1).mean())
    daily = daily.dropna()

    # Scale
    numeric_feats = [c for c in feat_cols if c != 'season']
    daily[numeric_feats] = scaler.transform(daily[numeric_feats])

    # Take last SEQUENCE_LENGTH days
    seq = daily[feat_cols].values[-SEQUENCE_LENGTH:]
    x = torch.FloatTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    # Inverse transform
    numeric_feats = [c for c in feat_cols if c != 'season']
    result = {}
    for i, name in enumerate(TARGET_COLS):
        d = np.zeros((1, len(numeric_feats)))
        idx = numeric_feats.index(name)
        d[0, idx] = pred[i]
        val = scaler.inverse_transform(d)[0, idx]
        if name == 'total_precipitation':
            val = np.expm1(val)
        result[name] = val

    return result

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("BAT DAU GIAI DOAN 2 - AI & MACHINE LEARNING")
    print("=" * 60)

    daily = load_and_preprocess_data()
    daily, feat_cols, scaler = scale_data(daily)
    model = train_and_evaluate(daily, feat_cols, scaler)
    export_model(model)

    print("\n" + "="*60)
    print("HOAN THANH GIAI DOAN 2!")
    print("="*60)
    print(f"Cac file da tao:")
    for f in ['lstm_weather_model_temp.pt', 'scaler_temp.pkl', 'feature_cols_temp.pkl', 'training_results_temp.png']:
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fp):
            print(f"  [OK] {f} ({os.path.getsize(fp):,} bytes)")
        else:
            print(f"  [MISSING] {f}")
            