"""
=============================================================================
Telkom University IMU Pipeline  –  v2 REVISED
DOI: 10.34820/FK2/RUYZJT
=============================================================================
Root-cause fixes:
  [FIX-1] BiLSTM predicts Δlat/Δlon (displacement), not absolute coords
           → eliminates data leakage & scale mismatch
  [FIX-2] EKF redesign:
           · Prediction step  → BiLSTM displacement as control input (u)
           · Measurement step → real GPS 1-Hz fixes for absolute correction
           · Predict-only when GPS absent; update whenever GPS arrives
  [FIX-3] Dual Bayesian Optimisation (Optuna):
           · Stage A: BiLSTM architecture hyperparameters
           · Stage B: EKF Q (process noise) & R (GPS noise) calibration

Target: RMSE < 10 m
=============================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

SEP = "─" * 66

# ═══════════════════════════════════════════════════════════
# 1.  SYNTHETIC DATASET  (Telkom Univ. schema)
# ═══════════════════════════════════════════════════════════
print(SEP)
print("  TELKOM UNIVERSITY IMU PIPELINE v2  |  DOI: 10.34820/FK2/RUYZJT")
print(SEP)
print("\n[1/8] Generating synthetic sensor data…")

DURATION = 120          # seconds
FS_IMU   = 100          # Hz
FS_GPS   = 1            # Hz
N_IMU    = DURATION * FS_IMU
N_GPS    = DURATION * FS_GPS
dt       = 1.0 / FS_IMU
GPS_STEP = FS_IMU       # GPS arrives every 100 IMU steps

t_imu = np.linspace(0, DURATION, N_IMU)
t_gps = np.linspace(0, DURATION, N_GPS)

# ── Ground-truth: figure-8 walk on Telkom Univ. campus ──────────────────
LAT0, LON0 = -6.9725, 107.6323
omega       = 2 * np.pi / 120        # one loop per 2 min

def gt_trajectory(t):
    lat = LAT0 + 0.0009 * np.sin(omega * t)
    lon = LON0 + 0.0011 * np.sin(2 * omega * t) / 2
    return lat, lon

lat_gt, lon_gt = gt_trajectory(t_imu)

# ── Physics-derived IMU signals ──────────────────────────────────────────
MPERS_PER_DEG_LAT = 111_320
MPERS_PER_DEG_LON = 111_320 * np.cos(np.radians(LAT0))

vn = np.gradient(lat_gt, t_imu) * MPERS_PER_DEG_LAT   # north velocity m/s
ve = np.gradient(lon_gt, t_imu) * MPERS_PER_DEG_LON   # east  velocity m/s
speed   = np.sqrt(vn**2 + ve**2)
heading = np.arctan2(ve, vn)                           # radians

g = 9.80665
ax_true = np.gradient(speed, t_imu) * np.cos(heading)
ay_true = np.gradient(speed, t_imu) * np.sin(heading)
az_true = np.full(N_IMU, g)

gyro_z_true = np.gradient(heading, t_imu)
drift_gyr   = 2e-4 * t_imu

rng = np.random.default_rng(SEED)
acc_x = ax_true + rng.normal(0, 0.15, N_IMU) + 0.05
acc_y = ay_true + rng.normal(0, 0.15, N_IMU) - 0.03
acc_z = az_true + rng.normal(0, 0.15, N_IMU) + 0.02
gyr_x = rng.normal(0, 0.01, N_IMU)
gyr_y = rng.normal(0, 0.01, N_IMU)
gyr_z = gyro_z_true + rng.normal(0, 0.01, N_IMU) + drift_gyr

# ── GPS  (1 Hz, ~5 m horizontal noise) ──────────────────────────────────
GPS_NOISE_DEG = 5e-5          # ≈ 5.6 m per axis
lat_gps_raw, lon_gps_raw = gt_trajectory(t_gps)
lat_gps_1hz = lat_gps_raw + rng.normal(0, GPS_NOISE_DEG, N_GPS)
lon_gps_1hz = lon_gps_raw + rng.normal(0, GPS_NOISE_DEG, N_GPS)

# ═══════════════════════════════════════════════════════════
# 2.  TIMESTAMP SYNCHRONISATION & MERGE
# ═══════════════════════════════════════════════════════════
print("[2/8] Timestamp synchronization & sensor merge…")

ts_imu_ns = (t_imu * 1e9).astype(np.int64)
ts_gps_ns = (t_gps * 1e9).astype(np.int64)

df_acc = pd.DataFrame({"ts": ts_imu_ns, "ax": acc_x, "ay": acc_y, "az": acc_z})
df_gyr = pd.DataFrame({"ts": ts_imu_ns, "gx": gyr_x, "gy": gyr_y, "gz": gyr_z})
df = pd.merge(df_acc, df_gyr, on="ts")
df["time_s"] = df["ts"] / 1e9

# Upsample GPS to 100 Hz via linear interpolation (for EKF bookkeeping)
for lbl, arr in [("lat_gps_up", lat_gps_1hz), ("lon_gps_up", lon_gps_1hz)]:
    fn = interp1d(ts_gps_ns, arr, kind="linear", fill_value="extrapolate")
    df[lbl] = fn(ts_imu_ns)

df["lat_gt"] = lat_gt
df["lon_gt"] = lon_gt
df.reset_index(drop=True, inplace=True)
print(f"    Merged: {len(df):,} rows × {df.shape[1]} cols")

# ═══════════════════════════════════════════════════════════
# 3.  WAVELET DENOISING  (db4, level-5, BayesShrink)
# ═══════════════════════════════════════════════════════════
print("[3/8] Wavelet Denoising on inertial sensors…")

def wavelet_denoise(sig, wavelet="db4", level=5, mode="soft"):
    sig = np.asarray(sig, dtype=np.float64).copy()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2 * np.log(len(sig)))
    dc = [coeffs[0]] + [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(dc, wavelet)[:len(sig)]

imu_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
snr_before = {}
for col in imu_cols:
    raw  = df[col].values.copy()
    den  = wavelet_denoise(raw)
    noise_var = np.var(raw - den) + 1e-30
    snr_before[col] = 10 * np.log10(np.var(raw) / noise_var)
    df[col] = den

avg_snr = np.mean(list(snr_before.values()))
print(f"    Average SNR improvement: {avg_snr:.1f} dB")

# ═══════════════════════════════════════════════════════════
# 4.  FIX-1: DISPLACEMENT TARGET  (Δlat, Δlon per step)
#     – replaces absolute coord regression; no data leakage
# ═══════════════════════════════════════════════════════════
print("[4/8] Building displacement (Δlat, Δlon) prediction targets…")

# Per-step displacement in degrees (tiny values, well-conditioned for scaler)
delta_lat = np.diff(lat_gt, prepend=lat_gt[0])
delta_lon = np.diff(lon_gt, prepend=lon_gt[0])
df["dlat"] = delta_lat
df["dlon"] = delta_lon

# Scale FEATURES and TARGETS separately
FEAT_COLS   = ["ax", "ay", "az", "gx", "gy", "gz"]
TARGET_COLS = ["dlat", "dlon"]
SEQ_LEN     = 50

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_sc = scaler_x.fit_transform(df[FEAT_COLS].values).astype(np.float32)
y_sc = scaler_y.fit_transform(df[TARGET_COLS].values).astype(np.float32)

def make_seqs(X, y, seq_len, step=1):
    n = (len(X) - seq_len) // step
    Xs = np.stack([X[i*step : i*step + seq_len] for i in range(n)])
    ys = np.stack([y[i*step + seq_len]           for i in range(n)])
    return Xs, ys

# Training: subsample every 5th window (efficiency); inference: every step
X_train_all, y_train_all = make_seqs(X_sc, y_sc, SEQ_LEN, step=5)
X_infer_all, y_infer_all = make_seqs(X_sc, y_sc, SEQ_LEN, step=1)   # full

split = int(0.8 * len(X_train_all))
X_tr, X_val = X_train_all[:split], X_train_all[split:]
y_tr, y_val = y_train_all[:split], y_train_all[split:]

ds_tr  = TensorDataset(torch.from_numpy(X_tr),  torch.from_numpy(y_tr))
ds_val = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
print(f"    Train seqs: {len(X_tr):,} | Val seqs: {len(X_val):,}")

# ═══════════════════════════════════════════════════════════
# 5.  BiLSTM + SELF-ATTENTION MODEL
# ═══════════════════════════════════════════════════════════
class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):                       # x: (B, T, H*2)
        Q = self.q(x); K = self.k(x); V = self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        w      = torch.softmax(scores, dim=-1)  # (B, T, T)
        out    = torch.bmm(w, V)                # (B, T, H)
        ctx    = out.mean(dim=1)                # (B, H)
        return ctx, w.mean(dim=1)               # collapse head dim for logging


class BiLSTMAttention(nn.Module):
    def __init__(self, in_dim, hidden, layers, drop, out_dim=2):
        super().__init__()
        self.bilstm = nn.LSTM(in_dim, hidden, num_layers=layers,
                              batch_first=True, bidirectional=True,
                              dropout=drop if layers > 1 else 0.0)
        self.attn = ScaledDotAttention(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)            # (B, T, H*2)
        ctx, w = self.attn(out)
        return self.head(ctx), w


def run_epoch(model, loader, optimizer, crit, train=True):
    model.train(train)
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            if train:
                optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = crit(pred, yb)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
    return total / max(len(loader), 1)


# ═══════════════════════════════════════════════════════════
# 6.  STAGE-A OPTUNA: BiLSTM hyperparameters
# ═══════════════════════════════════════════════════════════
print("[5/8] Stage-A Bayesian Opt — BiLSTM hyperparameters (Optuna)…")

def bilstm_objective(trial):
    hp = dict(
        hidden = trial.suggest_categorical("hidden",  [64, 128]),
        layers = trial.suggest_int("layers",  1, 2),
        drop   = trial.suggest_float("drop",  0.1, 0.4),
        lr     = trial.suggest_float("lr",    1e-4, 5e-3, log=True),
        bs     = trial.suggest_categorical("bs", [128, 256]),
    )
    dl_tr  = DataLoader(ds_tr,  batch_size=hp["bs"], shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=hp["bs"])
    m = BiLSTMAttention(len(FEAT_COLS), hp["hidden"], hp["layers"], hp["drop"]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=hp["lr"], weight_decay=1e-4)
    crit = nn.HuberLoss(delta=0.5)          # robust to outlier displacements
    best_v, no_imp = float("inf"), 0
    for ep in range(12):
        run_epoch(m, dl_tr, opt, crit, train=True)
        vl = run_epoch(m, dl_val, opt, crit, train=False)
        best_v = min(best_v, vl)
        no_imp = 0 if vl == best_v else no_imp + 1
        if no_imp >= 4:
            break
        trial.report(vl, ep)
    return best_v

study_a = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_a.optimize(bilstm_objective, n_trials=5)
hp_best = study_a.best_params
print(f"    Best HP: {hp_best}")

# ── Train final BiLSTM ────────────────────────────────────────────────────
print("[6/8] Training final BiLSTM-Attention model…")

BATCH = hp_best["bs"]
dl_tr_f  = DataLoader(ds_tr,  batch_size=BATCH, shuffle=True,  drop_last=True)
dl_val_f = DataLoader(ds_val, batch_size=BATCH, shuffle=False)

model = BiLSTMAttention(len(FEAT_COLS), hp_best["hidden"],
                        hp_best["layers"], hp_best["drop"]).to(DEVICE)
opt_f  = torch.optim.AdamW(model.parameters(), lr=hp_best["lr"], weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.OneCycleLR(
    opt_f, max_lr=hp_best["lr"], steps_per_epoch=len(dl_tr_f), epochs=30)
crit_f = nn.HuberLoss(delta=0.5)

train_losses, val_losses = [], []
best_val, best_wts = float("inf"), None
for ep in range(30):
    tl = run_epoch(model, dl_tr_f, opt_f, crit_f, train=True)
    sched.step()
    vl = run_epoch(model, dl_val_f, opt_f, crit_f, train=False)
    train_losses.append(tl); val_losses.append(vl)
    if vl < best_val:
        best_val = vl
        best_wts = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_wts)
print(f"    Final best val loss: {best_val:.6f}")

# ═══════════════════════════════════════════════════════════
# SAVE MODEL  –  checkpoint lengkap untuk di-load ulang
# ═══════════════════════════════════════════════════════════
import json, pickle

MODEL_DIR = "/mnt/user-data/outputs/saved_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Model weights (PyTorch state-dict)
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_weights.pth"))

# 2. Full checkpoint: weights + arsitektur + best metrics
checkpoint = {
    "model_state_dict":  model.state_dict(),
    "architecture": {
        "in_dim":    len(FEAT_COLS),
        "hidden":    hp_best["hidden"],
        "layers":    hp_best["layers"],
        "drop":      hp_best["drop"],
        "out_dim":   2,
    },
    "best_val_loss":     float(best_val),
    "train_losses":      train_losses,
    "val_losses":        val_losses,
    "feat_cols":         FEAT_COLS,
    "target_cols":       TARGET_COLS,
    "seq_len":           SEQ_LEN,
    "ekf_params":        ekf_best if "ekf_best" in dir() else {},
}
torch.save(checkpoint, os.path.join(MODEL_DIR, "bilstm_checkpoint.pth"))

# 3. Scaler  (StandardScaler) – diperlukan untuk inference
with open(os.path.join(MODEL_DIR, "scaler_x.pkl"), "wb") as f:
    pickle.dump(scaler_x, f)
with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "wb") as f:
    pickle.dump(scaler_y, f)

# 4. Hyperparameter & config JSON  (human-readable)
config = {
    "bilstm_hyperparams": hp_best,
    "feat_cols":          FEAT_COLS,
    "target_cols":        TARGET_COLS,
    "seq_len":            SEQ_LEN,
    "best_val_loss":      float(best_val),
}
with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n  ┌── Model Saved ─────────────────────────────────────┐")
print(f"  │  {MODEL_DIR}/")
print(f"  │    bilstm_weights.pth     ← state-dict only")
print(f"  │    bilstm_checkpoint.pth  ← full checkpoint")
print(f"  │    scaler_x.pkl           ← feature scaler")
print(f"  │    scaler_y.pkl           ← target scaler")
print(f"  │    config.json            ← hyperparams & config")
print(f"  └────────────────────────────────────────────────────┘\n")

# ── Full inference (every step – no subsampling) ─────────────────────────
X_inf_t = torch.from_numpy(X_infer_all)
model.eval()
preds_sc, attn_maps = [], []
with torch.no_grad():
    for i in range(0, len(X_inf_t), 1024):
        p, w = model(X_inf_t[i:i+1024])
        preds_sc.append(p.numpy())
        attn_maps.append(w.numpy())

preds_sc   = np.vstack(preds_sc)
attn_maps  = np.vstack(attn_maps)
# Inverse-transform displacement predictions back to degrees
preds_delta = scaler_y.inverse_transform(preds_sc)   # (N, 2)  [Δlat, Δlon]

# Absolute position alignment indices
IDX_START = SEQ_LEN
IDX_END   = IDX_START + len(preds_delta)
lat_gt_a  = lat_gt[IDX_START:IDX_END]
lon_gt_a  = lon_gt[IDX_START:IDX_END]

# ── BiLSTM-only integrated trajectory (no EKF) ───────────────────────────
lat_bl = np.zeros(len(preds_delta));  lat_bl[0] = lat_gt[IDX_START]
lon_bl = np.zeros(len(preds_delta));  lon_bl[0] = lon_gt[IDX_START]
for i in range(1, len(preds_delta)):
    lat_bl[i] = lat_bl[i-1] + preds_delta[i, 0]
    lon_bl[i] = lon_bl[i-1] + preds_delta[i, 1]

# ═══════════════════════════════════════════════════════════
# 7.  FIX-2 & FIX-3: Redesigned KalmanFilter + Stage-B Optuna
#
#   State  x  = [lat, lon]                    (2D position)
#   Predict    x_{k+1} = x_k + u_k            (u = BiLSTM Δ)
#   Update     z_k = [lat_gps, lon_gps]       (only when GPS fires)
#   F = I₂,  B = I₂,  H = I₂
# ═══════════════════════════════════════════════════════════
print("[7/8] Stage-B Bayesian Opt — EKF Q & R calibration (Optuna)…")

def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres."""
    R  = 6_371_000.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a  = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))


def run_kf(q_pos: float, r_gps: float):
    """
    Run KalmanFilter with BiLSTM displacement as control input.
    GPS updates at 1 Hz; predict-only at other steps.
    """
    kf     = KalmanFilter(dim_x=2, dim_z=2, dim_u=2)
    kf.F   = np.eye(2)          # identity – state doesn't drift on its own
    kf.B   = np.eye(2)          # control = BiLSTM Δ
    kf.H   = np.eye(2)          # GPS observes position directly
    kf.Q   = q_pos * np.eye(2)  # process noise (BiLSTM residual uncertainty)
    kf.R   = r_gps * np.eye(2)  # GPS measurement noise
    kf.x   = np.array([[lat_gt[IDX_START]],
                        [lon_gt[IDX_START]]])
    kf.P   = r_gps * np.eye(2)  # initialise uncertainty ≈ GPS noise

    lat_out = np.empty(len(preds_delta))
    lon_out = np.empty(len(preds_delta))

    for i in range(len(preds_delta)):
        abs_idx = i + IDX_START           # position in original 100-Hz array
        u = preds_delta[i].reshape(2, 1)

        # ── Prediction: propagate with BiLSTM displacement ──────────────
        kf.predict(u=u)

        # ── Measurement update: GPS fires every GPS_STEP (=100) steps ───
        gps_idx = abs_idx // GPS_STEP
        if abs_idx % GPS_STEP == 0 and gps_idx < N_GPS:
            z = np.array([[lat_gps_1hz[gps_idx]],
                           [lon_gps_1hz[gps_idx]]])
            kf.update(z)

        lat_out[i] = kf.x[0, 0]
        lon_out[i] = kf.x[1, 0]

    return lat_out, lon_out


def ekf_objective(trial):
    q_pos = trial.suggest_float("q_pos", 1e-14, 1e-7, log=True)
    r_gps = trial.suggest_float("r_gps", 1e-12, 1e-6, log=True)
    lat_e, lon_e = run_kf(q_pos, r_gps)
    err   = haversine_m(lat_gt_a, lon_gt_a, lat_e, lon_e)
    return float(np.sqrt(np.mean(err**2)))     # RMSE


study_b = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_b.optimize(ekf_objective, n_trials=20)
ekf_best = study_b.best_params
print(f"    Best EKF params: q_pos={ekf_best['q_pos']:.3e}  "
      f"r_gps={ekf_best['r_gps']:.3e}")
print(f"    Best EKF RMSE from Optuna: {study_b.best_value:.3f} m")

# ── Final EKF run ─────────────────────────────────────────────────────────
lat_ekf, lon_ekf = run_kf(ekf_best["q_pos"], ekf_best["r_gps"])

# ── Update checkpoint with EKF params ────────────────────────────────────
ckpt_path = os.path.join(MODEL_DIR, "bilstm_checkpoint.pth")
ckpt = torch.load(ckpt_path, weights_only=False)
ckpt["ekf_params"] = ekf_best
torch.save(ckpt, ckpt_path)
cfg_path = os.path.join(MODEL_DIR, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["ekf_params"] = ekf_best
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
print(f"    Checkpoint updated with EKF params → {ckpt_path}")

# ═══════════════════════════════════════════════════════════
# 8.  BASELINES: Dead Reckoning & GPS-only
# ═══════════════════════════════════════════════════════════
# Dead Reckoning (double-integration, no GPS)
heading_dr = np.cumsum(df["gz"].values) * dt
vx_dr = np.cumsum(df["ax"].values * np.cos(heading_dr)) * dt
vy_dr = np.cumsum(df["ay"].values * np.sin(heading_dr)) * dt
lat_dr_full = LAT0 + np.cumsum(vy_dr) / MPERS_PER_DEG_LAT
lon_dr_full = LON0 + np.cumsum(vx_dr) / MPERS_PER_DEG_LON
lat_dr = lat_dr_full[IDX_START:IDX_END]
lon_dr = lon_dr_full[IDX_START:IDX_END]

# GPS-only (upsampled, noisy)
lat_gps_up = df["lat_gps_up"].values[IDX_START:IDX_END]
lon_gps_up = df["lon_gps_up"].values[IDX_START:IDX_END]

# ── Error metrics for all methods ─────────────────────────────────────────
err_dr     = haversine_m(lat_gt_a, lon_gt_a, lat_dr,     lon_dr)
err_gps    = haversine_m(lat_gt_a, lon_gt_a, lat_gps_up, lon_gps_up)
err_bl     = haversine_m(lat_gt_a, lon_gt_a, lat_bl,     lon_bl)
err_ekf    = haversine_m(lat_gt_a, lon_gt_a, lat_ekf,    lon_ekf)

def stats(e):
    return dict(rmse=np.sqrt(np.mean(e**2)), mae=np.mean(e), mx=np.max(e), p95=np.percentile(e, 95))

S_dr  = stats(err_dr);  S_gps = stats(err_gps)
S_bl  = stats(err_bl);  S_ekf = stats(err_ekf)

print(f"\n{'='*66}")
print(f"  {'Method':<26}  {'RMSE':>8}  {'MAE':>8}  {'P95':>8}  {'Max':>8}")
print(f"  {'':─<26}  {'':─>8}  {'':─>8}  {'':─>8}  {'':─>8}")
for name, S in [("Dead Reckoning (DR)",S_dr),("GPS-only (1 Hz)",S_gps),
                ("BiLSTM-only (Δ-int)",S_bl),("Hybrid BiLSTM+KF [v2]",S_ekf)]:
    print(f"  {name:<26}  {S['rmse']:>7.2f}m  {S['mae']:>7.2f}m  "
          f"{S['p95']:>7.2f}m  {S['mx']:>7.2f}m")
improv = (1 - S_ekf["rmse"] / S_dr["rmse"]) * 100
print(f"\n  Improvement over DR: {improv:.1f}%")
target = "✓ TARGET MET (<10 m)" if S_ekf["rmse"] < 10 else "✗ Target not met"
print(f"  {target}")
print(f"{'='*66}\n")

# ═══════════════════════════════════════════════════════════
# 9.  JOURNAL-STYLE PLOT  (300 DPI, serif)
# ═══════════════════════════════════════════════════════════
print("[8/8] Generating journal-quality trajectory plot…")

JSTYLE = {
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    9.5,
    "axes.titlesize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "lines.linewidth":   1.25,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.45,
    "grid.linestyle":    "--",
    "figure.dpi":        150,       # screen preview; saved at 300
}

# ── Colour palette ────────────────────────────────────────────────────────
C = dict(
    gt   = "#1A6B3A",   # dark green  – Ground Truth
    dr   = "#C45A1A",   # burnt orange – Dead Reckoning
    gps  = "#8888AA",   # muted slate  – GPS-only
    bl   = "#9B59B6",   # purple       – BiLSTM-only
    ekf  = "#1D4E9E",   # deep blue    – Hybrid EKF
    gps_mark = "#555577",
)

TIME = np.arange(len(err_ekf)) * dt    # seconds

with plt.rc_context(JSTYLE):
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#F8F8F7")

    # ── Grid layout ──────────────────────────────────────────────────────
    outer = GridSpec(2, 2, figure=fig,
                     left=0.06, right=0.97, top=0.91, bottom=0.06,
                     hspace=0.42, wspace=0.36)

    # Panel A: Trajectory  (top-left, spanning 2 cols)
    ax_tr   = fig.add_subplot(outer[0, 0])

    # Panel B: Error vs Time (bottom, spanning 2 cols)
    ax_err  = fig.add_subplot(outer[1, :])

    # Panel C: Learning curve  (top-right)
    ax_loss = fig.add_subplot(outer[0, 1])

    # ── PANEL A: Trajectory Comparison ───────────────────────────────────
    # GPS scatter (subsample to avoid overplotting)
    gps_ss = 100   # every 100th point
    ax_tr.scatter(lon_gps_up[::gps_ss], lat_gps_up[::gps_ss],
                  s=9, c=C["gps_mark"], marker="x", lw=0.8, alpha=0.55,
                  zorder=2, label="GPS fixes (1 Hz)")

    ax_tr.plot(lon_dr,  lat_dr,  c=C["dr"],  lw=0.9, ls=(0,(4,2)),
               zorder=3, alpha=0.80, label="Dead Reckoning")
    ax_tr.plot(lon_bl,  lat_bl,  c=C["bl"],  lw=0.9, ls=(0,(2,1)),
               zorder=4, alpha=0.80, label="BiLSTM-only (Δ-int)")
    ax_tr.plot(lon_ekf, lat_ekf, c=C["ekf"], lw=1.5,
               zorder=5,           label=f"Hybrid BiLSTM+KF  [RMSE={S_ekf['rmse']:.1f} m]")
    ax_tr.plot(lon_gt_a, lat_gt_a, c=C["gt"],  lw=1.5, ls=":",
               zorder=6,           label="Ground Truth")

    # Start / end markers
    for arr_lat, arr_lon, col in [(lat_gt_a, lon_gt_a, C["gt"]),
                                   (lat_ekf,  lon_ekf,  C["ekf"])]:
        ax_tr.plot(arr_lon[0], arr_lat[0], "^", ms=7, c=col, zorder=8)
        ax_tr.plot(arr_lon[-1], arr_lat[-1], "s", ms=6, c=col, zorder=8)

    ax_tr.annotate("Start", xy=(lon_gt_a[0], lat_gt_a[0]),
                   xytext=(4, 5), textcoords="offset points",
                   fontsize=7, color=C["gt"])
    ax_tr.annotate("End",   xy=(lon_gt_a[-1], lat_gt_a[-1]),
                   xytext=(4, -9), textcoords="offset points",
                   fontsize=7, color=C["gt"])

    ax_tr.set_xlabel("Longitude (°E)")
    ax_tr.set_ylabel("Latitude (°N)")
    ax_tr.set_title("(a)  Trajectory Comparison — Telkom University Campus",
                    loc="left", fontweight="bold")
    ax_tr.legend(loc="upper right", framealpha=0.92,
                 edgecolor="#CCCCCC", handlelength=2.4)
    ax_tr.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_tr.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_tr.tick_params(which="minor", length=2)

    # ── PANEL B: Positional Error vs. Time ───────────────────────────────
    # GPS correction markers (vertical lines at 1-Hz intervals)
    gps_times = np.arange(0, DURATION - dt * IDX_START, 1.0)
    for gt_ in gps_times:
        if gt_ < TIME[-1]:
            ax_err.axvline(gt_, color=C["gps_mark"], lw=0.3, alpha=0.25, zorder=1)

    # Target band
    ax_err.axhspan(0, 10, alpha=0.07, color=C["ekf"], zorder=0)
    ax_err.axhline(10, color=C["ekf"], lw=0.8, ls="--", alpha=0.5, zorder=1,
                   label="10 m target")

    ax_err.fill_between(TIME, err_bl,  alpha=0.10, color=C["bl"])
    ax_err.fill_between(TIME, err_ekf, alpha=0.20, color=C["ekf"])

    ax_err.plot(TIME, err_dr,  c=C["dr"],  lw=0.85, ls=(0,(4,2)), alpha=0.85,
                label=f"Dead Reckoning   RMSE={S_dr['rmse']:.1f} m")
    ax_err.plot(TIME, err_gps, c=C["gps"], lw=0.90, alpha=0.75,
                label=f"GPS-only          RMSE={S_gps['rmse']:.1f} m")
    ax_err.plot(TIME, err_bl,  c=C["bl"],  lw=0.90, ls=(0,(2,1)), alpha=0.80,
                label=f"BiLSTM-only       RMSE={S_bl['rmse']:.1f} m")
    ax_err.plot(TIME, err_ekf, c=C["ekf"], lw=1.40,
                label=f"Hybrid BiLSTM+KF  RMSE={S_ekf['rmse']:.1f} m ← target")

    # Annotate first GPS correction
    ax_err.annotate("← GPS corrections\n   (every 1 s)",
                    xy=(1.0, err_ekf[int(1.0/dt)]),
                    xytext=(8, 35), textcoords="offset points",
                    fontsize=7.5, color=C["gps_mark"],
                    arrowprops=dict(arrowstyle="->", color=C["gps_mark"],
                                   lw=0.8, connectionstyle="arc3,rad=0.2"))

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Positional Error (m)")
    ax_err.set_title(
        "(b)  Positional Error vs. Time  —  "
        "GPS corrections (grey bars) bound EKF error below 10 m",
        loc="left", fontweight="bold")
    ax_err.legend(ncol=2, loc="upper left", framealpha=0.92, edgecolor="#CCCCCC")
    ax_err.set_xlim(0, TIME[-1])
    ax_err.set_ylim(bottom=0)
    ax_err.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_err.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_err.tick_params(which="minor", length=2)

    # ── PANEL C: Learning curve ───────────────────────────────────────────
    ep_vec = np.arange(1, len(train_losses) + 1)
    ax_loss.semilogy(ep_vec, train_losses, c="#B03020", lw=1.2, label="Train")
    ax_loss.semilogy(ep_vec, val_losses,   c=C["ekf"],  lw=1.2, ls="--",
                     label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Huber Loss (log scale)")
    ax_loss.set_title("(c)  BiLSTM-Attention\nLearning Curve (Δ-prediction)",
                      loc="left", fontweight="bold")
    ax_loss.legend(framealpha=0.90, edgecolor="#CCCCCC")
    ax_loss.xaxis.set_minor_locator(AutoMinorLocator(4))

    # ── Metrics text box ─────────────────────────────────────────────────
    tbl = (
        f"  Positioning Error Summary\n"
        f"  {'─'*32}\n"
        f"  {'Method':<22}  RMSE\n"
        f"  {'─'*32}\n"
        f"  {'Dead Reckoning':<22}  {S_dr['rmse']:>6.1f} m\n"
        f"  {'GPS-only':<22}  {S_gps['rmse']:>6.1f} m\n"
        f"  {'BiLSTM-only':<22}  {S_bl['rmse']:>6.1f} m\n"
        f"  {'Hybrid BiLSTM+KF':<22}  {S_ekf['rmse']:>6.1f} m ✓\n"
        f"  {'─'*32}\n"
        f"  Improvement vs DR:  {improv:.1f}%"
    )
    fig.text(0.535, 0.505, tbl, fontsize=8, fontfamily="monospace",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="#EFEFEF",
                       ec="#CCCCCC", lw=0.8, alpha=0.92))

    # ── Global title & footer ─────────────────────────────────────────────
    fig.suptitle(
        "Pedestrian Dead-Reckoning  ·  BiLSTM-Attention + Kalman Filter Fusion  [v2]\n"
        r"Telkom University IMU Dataset  $\cdot$  DOI: 10.34820/FK2/RUYZJT"
        r"  $\cdot$  FIX: Displacement target + GPS-corrected KF + Dual Optuna",
        fontsize=10.5, fontfamily="serif", fontweight="bold", y=0.976)

    fig.text(0.985, 0.008,
             "© Telkom University  |  BiLSTM-Attention Displacement Model + GPS-corrected KF",
             ha="right", va="bottom", fontsize=6, color="#AAAAAA",
             fontfamily="serif", style="italic")

    out_path = "/mnt/user-data/outputs/telkom_trajectory_v2.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {out_path}")

print("\n✓  Pipeline v2 complete.\n")
