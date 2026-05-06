"""
=============================================================================
telkom_inference.py  –  Inference on Real Test Dataset
=============================================================================
Memuat model BiLSTM-Attention yang sudah disimpan, lalu menjalankan:
  1. Parsing & preprocessing Test.csv (Telkom Univ. schema)
  2. Wavelet Denoising
  3. BiLSTM displacement inference
  4. GPS-corrected Kalman Filter fusion
  5. Trajectory plot 300 DPI (journal style)

Usage:
    python3 telkom_inference.py

Requirements (sama dengan pipeline utama):
    pip install PyWavelets torch filterpy scikit-learn pandas matplotlib scipy
=============================================================================
"""

import os, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import AutoMinorLocator

# ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")

MODEL_DIR  = "/mnt/user-data/outputs/saved_model"
TEST_CSV   = "/mnt/user-data/uploads/Test.csv"
OUT_DIR    = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SEP = "─" * 66

# ═══════════════════════════════════════════════════════════
# STEP 1 — LOAD SAVED MODEL & CONFIG
# ═══════════════════════════════════════════════════════════
print(SEP)
print("  TELKOM UNIVERSITY  –  BiLSTM+KF INFERENCE ON TEST.CSV")
print(SEP)
print("\n[1/6] Loading saved checkpoint…")

# Load config JSON
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)

FEAT_COLS   = cfg["feat_cols"]           # ["ax","ay","az","gx","gy","gz"]
TARGET_COLS = cfg["target_cols"]         # ["dlat","dlon"]
SEQ_LEN     = cfg["seq_len"]            # 50
HP          = cfg["bilstm_hyperparams"]
EKF_PARAMS  = cfg.get("ekf_params", {"q_pos": 2.7e-10, "r_gps": 2.0e-8})

# Load scalers
with open(os.path.join(MODEL_DIR, "scaler_x.pkl"), "rb") as f:
    scaler_x = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler_y.pkl"), "rb") as f:
    scaler_y = pickle.load(f)

# ── Rebuild model architecture ────────────────────────────────────────────
class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim * 2, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.q(x); K = self.k(x); V = self.v(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        w      = torch.softmax(scores, dim=-1)
        out    = torch.bmm(w, V)
        ctx    = out.mean(dim=1)
        return ctx, w.mean(dim=1)


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
        out, _ = self.bilstm(x)
        ctx, w = self.attn(out)
        return self.head(ctx), w


model = BiLSTMAttention(
    in_dim = len(FEAT_COLS),
    hidden = HP["hidden"],
    layers = HP["layers"],
    drop   = HP["drop"],
).to(DEVICE)

ckpt = torch.load(
    os.path.join(MODEL_DIR, "bilstm_checkpoint.pth"),
    map_location=DEVICE,
    weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"  ✓ Model loaded  (hidden={HP['hidden']}, layers={HP['layers']}, "
      f"drop={HP['drop']:.2f})")
print(f"  ✓ Best val loss from training: {ckpt['best_val_loss']:.6f}")
print(f"  ✓ EKF params  q={EKF_PARAMS['q_pos']:.2e}  r={EKF_PARAMS['r_gps']:.2e}")

# ═══════════════════════════════════════════════════════════
# STEP 2 — PARSE TEST.CSV  (Telkom Univ. real sensor format)
# ═══════════════════════════════════════════════════════════
print("\n[2/6] Parsing Test.csv…")

raw = pd.read_csv(TEST_CSV, skiprows=1, header=None)
raw.columns = [
    "time_server", "device_ts",
    "ax_raw", "ay_raw", "az_raw",
    "gx", "gy", "gz",
    "mx", "my", "mz",
    "euler_x", "euler_y", "euler_z",
    "lat", "lon", "alt", "gps_acc",
    "gmaps", "drop"
]
raw.drop(columns=["drop", "gmaps", "mx", "my", "mz",
                  "euler_x", "euler_y", "euler_z"], inplace=True)

# Fix European decimal separator (comma → dot)
for col in ["gx", "gy", "gz", "lat", "lon", "alt"]:
    raw[col] = raw[col].astype(str).str.replace(",", ".", regex=False)
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# Accelerometer raw values are in 100× m/s² → convert
raw["ax"] = raw["ax_raw"] / 100.0
raw["ay"] = raw["ay_raw"] / 100.0
raw["az"] = raw["az_raw"] / 100.0
raw.drop(columns=["ax_raw", "ay_raw", "az_raw"], inplace=True)

# Parse server timestamp → elapsed seconds
raw["time_server"] = pd.to_datetime(
    raw["time_server"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
)
raw.dropna(subset=["time_server"], inplace=True)
raw["time_s"] = (raw["time_server"] - raw["time_server"].iloc[0]).dt.total_seconds()
raw.sort_values("time_s", inplace=True)
raw.reset_index(drop=True, inplace=True)

# Keep only valid GPS (Indonesia bounding box)
mask_valid_gps = (
    (raw["lat"] < -5) & (raw["lat"] > -8) &
    (raw["lon"] > 106) & (raw["lon"] < 109)
)
df_gps_valid = raw[mask_valid_gps].copy()

# Use ALL rows for IMU (zero-GPS rows still have valid IMU)
df_imu = raw.copy()
# Fill missing GPS by forward-fill (used only for the KF measurement step)
df_imu["lat"] = df_imu["lat"].where(mask_valid_gps)
df_imu["lon"] = df_imu["lon"].where(mask_valid_gps)

print(f"  Total rows       : {len(df_imu):,}")
print(f"  Valid GPS rows   : {mask_valid_gps.sum():,}")
print(f"  Time range       : {df_imu['time_s'].iloc[0]:.0f} – "
      f"{df_imu['time_s'].iloc[-1]:.0f} s  "
      f"({df_imu['time_s'].iloc[-1]/60:.1f} min)")
print(f"  GPS area         : lat [{df_gps_valid['lat'].min():.4f}, "
      f"{df_gps_valid['lat'].max():.4f}]  "
      f"lon [{df_gps_valid['lon'].min():.4f}, {df_gps_valid['lon'].max():.4f}]")

# ═══════════════════════════════════════════════════════════
# STEP 3 — WAVELET DENOISING
# ═══════════════════════════════════════════════════════════
print("\n[3/6] Wavelet denoising (db4 level-5)…")

def wavelet_denoise(sig, wavelet="db4", level=5, mode="soft"):
    sig = np.asarray(sig, dtype=np.float64).copy()
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma  = np.median(np.abs(coeffs[-1])) / 0.6745
    thr    = sigma * np.sqrt(2 * np.log(max(len(sig), 2)))
    dc = [coeffs[0]] + [pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]]
    return pywt.waverec(dc, wavelet)[:len(sig)]

for col in FEAT_COLS:
    df_imu[col] = wavelet_denoise(df_imu[col].values)

print(f"  ✓ Denoised: {FEAT_COLS}")

# ═══════════════════════════════════════════════════════════
# STEP 4 — BiLSTM INFERENCE  (predict Δlat, Δlon)
# ═══════════════════════════════════════════════════════════
print("\n[4/6] Running BiLSTM displacement inference…")

X_raw = df_imu[FEAT_COLS].values.astype(np.float32)
X_sc  = scaler_x.transform(X_raw)

# Build sliding-window sequences (stride=1)
N_SEQ = len(X_sc) - SEQ_LEN
if N_SEQ <= 0:
    raise ValueError(f"Dataset too short ({len(X_sc)} rows) for SEQ_LEN={SEQ_LEN}")

X_seq = np.stack([X_sc[i:i+SEQ_LEN] for i in range(N_SEQ)], axis=0)
X_t   = torch.from_numpy(X_seq)

preds_sc, attn_maps = [], []
with torch.no_grad():
    for i in range(0, len(X_t), 1024):
        p, w = model(X_t[i:i+1024])
        preds_sc.append(p.numpy())
        attn_maps.append(w.numpy())

preds_sc   = np.vstack(preds_sc)
attn_maps  = np.vstack(attn_maps)
preds_delta = scaler_y.inverse_transform(preds_sc)   # (N, 2) → [Δlat, Δlon]
print(f"  ✓ Predicted {len(preds_delta):,} displacement steps")
print(f"  Δlat std={preds_delta[:,0].std():.2e}   "
      f"Δlon std={preds_delta[:,1].std():.2e}")

# ── Integrate to trajectory (BiLSTM-only) ────────────────────────────────
IDX_START   = SEQ_LEN
time_aligned = df_imu["time_s"].values[IDX_START: IDX_START + len(preds_delta)]
lat_start   = df_gps_valid["lat"].iloc[0]
lon_start   = df_gps_valid["lon"].iloc[0]

lat_bl = np.empty(len(preds_delta)); lat_bl[0] = lat_start
lon_bl = np.empty(len(preds_delta)); lon_bl[0] = lon_start
for i in range(1, len(preds_delta)):
    lat_bl[i] = lat_bl[i-1] + preds_delta[i, 0]
    lon_bl[i] = lon_bl[i-1] + preds_delta[i, 1]

# ═══════════════════════════════════════════════════════════
# STEP 5 — GPS-CORRECTED KALMAN FILTER
# ═══════════════════════════════════════════════════════════
print("\n[5/6] Running GPS-corrected Kalman Filter…")

q_pos = EKF_PARAMS["q_pos"]
r_gps = EKF_PARAMS["r_gps"]

kf   = KalmanFilter(dim_x=2, dim_z=2, dim_u=2)
kf.F = np.eye(2)
kf.B = np.eye(2)
kf.H = np.eye(2)
kf.Q = q_pos * np.eye(2)
kf.R = r_gps * np.eye(2)
kf.x = np.array([[lat_start], [lon_start]])
kf.P = r_gps * np.eye(2)

# Precompute GPS lookup: for each aligned row, nearest valid GPS
#   GPS fires whenever we have a real lat/lon measurement
gps_times   = df_gps_valid["time_s"].values
gps_lats    = df_gps_valid["lat"].values
gps_lons    = df_gps_valid["lon"].values

# Build set of GPS timestamps (for fast lookup)
gps_time_set = set(gps_times.tolist())

# For each IMU time in aligned window, check if GPS available (±0.5 s tolerance)
def nearest_gps(t, tol=1.5):
    diffs = np.abs(gps_times - t)
    idx   = np.argmin(diffs)
    if diffs[idx] <= tol:
        return gps_lats[idx], gps_lons[idx]
    return None, None

lat_ekf = np.empty(len(preds_delta))
lon_ekf = np.empty(len(preds_delta))
gps_used_mask = np.zeros(len(preds_delta), dtype=bool)
prev_gps_time = -999.0

for i in range(len(preds_delta)):
    u = preds_delta[i].reshape(2, 1)
    kf.predict(u=u)

    t = time_aligned[i]
    # Update only when GPS fires (avoid repeated update at same second)
    if t - prev_gps_time >= 1.0:
        g_lat, g_lon = nearest_gps(t)
        if g_lat is not None:
            kf.update(np.array([[g_lat], [g_lon]]))
            gps_used_mask[i] = True
            prev_gps_time = t

    lat_ekf[i] = kf.x[0, 0]
    lon_ekf[i] = kf.x[1, 0]

n_gps_updates = gps_used_mask.sum()
print(f"  ✓ KF finished  |  GPS updates applied: {n_gps_updates:,} times")

# ═══════════════════════════════════════════════════════════
# DEAD RECKONING BASELINE (for comparison)
# ═══════════════════════════════════════════════════════════
# Estimate dt from time column (irregular sampling)
times_full = df_imu["time_s"].values
dt_arr     = np.diff(times_full, prepend=times_full[0])
dt_arr[0]  = dt_arr[1]

heading_dr = np.cumsum(df_imu["gz"].values * dt_arr)
vx_dr      = np.cumsum(df_imu["ax"].values * np.cos(heading_dr) * dt_arr)
vy_dr      = np.cumsum(df_imu["ay"].values * np.sin(heading_dr) * dt_arr)

MPERS_LAT = 111_320
MPERS_LON = 111_320 * np.cos(np.radians(lat_start))

lat_dr_full = lat_start + np.cumsum(vy_dr) / MPERS_LAT
lon_dr_full = lon_start + np.cumsum(vx_dr) / MPERS_LON
lat_dr = lat_dr_full[IDX_START: IDX_START + len(preds_delta)]
lon_dr = lon_dr_full[IDX_START: IDX_START + len(preds_delta)]

# ═══════════════════════════════════════════════════════════
# METRICS vs GPS ground reference
# ═══════════════════════════════════════════════════════════
def haversine_m(lat1, lon1, lat2, lon2):
    R  = 6_371_000.0
    φ1 = np.radians(lat1); φ2 = np.radians(lat2)
    Δφ = np.radians(lat2 - lat1)
    Δλ = np.radians(lon2 - lon1)
    a  = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))

# Build GPS reference trajectory aligned to preds window
#   Use interpolated GPS as reference (forward-fill from valid GPS)
gps_interp_lat = interp1d(gps_times, gps_lats, kind="linear",
                           fill_value="extrapolate")(time_aligned)
gps_interp_lon = interp1d(gps_times, gps_lons, kind="linear",
                           fill_value="extrapolate")(time_aligned)

err_dr  = haversine_m(gps_interp_lat, gps_interp_lon, lat_dr,  lon_dr)
err_bl  = haversine_m(gps_interp_lat, gps_interp_lon, lat_bl,  lon_bl)
err_ekf = haversine_m(gps_interp_lat, gps_interp_lon, lat_ekf, lon_ekf)

def stats(e):
    return dict(rmse=np.sqrt(np.mean(e**2)), mae=np.mean(e),
                p50=np.median(e), p95=np.percentile(e,95), mx=np.max(e))

S_dr  = stats(err_dr)
S_bl  = stats(err_bl)
S_ekf = stats(err_ekf)
improv = (1 - S_ekf["rmse"] / max(S_dr["rmse"], 1e-9)) * 100

print(f"\n{'='*66}")
print(f"  {'Method':<26} {'RMSE':>7} {'MAE':>7} {'P50':>7} "
      f"{'P95':>7} {'Max':>7}")
print(f"  {'':─<26} {'':─>7} {'':─>7} {'':─>7} {'':─>7} {'':─>7}")
for name, S in [("Dead Reckoning (DR)", S_dr),
                ("BiLSTM-only (Δ-int)", S_bl),
                ("Hybrid BiLSTM+KF",    S_ekf)]:
    print(f"  {name:<26} {S['rmse']:>6.1f}m {S['mae']:>6.1f}m "
          f"{S['p50']:>6.1f}m {S['p95']:>6.1f}m {S['mx']:>6.1f}m")
target_ok = "✓ TARGET MET (<10 m)" if S_ekf["rmse"] < 10 else "⚠ Above 10 m target"
print(f"\n  {target_ok}   |   Improvement vs DR: {improv:.1f}%")
print(f"{'='*66}\n")

# ═══════════════════════════════════════════════════════════
# STEP 6 — JOURNAL-STYLE PLOT  (300 DPI)
# ═══════════════════════════════════════════════════════════
print("[6/6] Generating journal-style plot…")

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
    "lines.linewidth":   1.25,
    "axes.grid":         True,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.45,
    "grid.linestyle":    "--",
}

C = dict(
    gps = "#555577",
    dr  = "#C45A1A",
    bl  = "#9B59B6",
    ekf = "#1D4E9E",
)

TIME = time_aligned - time_aligned[0]   # relative seconds

with plt.rc_context(JSTYLE):
    fig = plt.figure(figsize=(15, 11))
    fig.patch.set_facecolor("#F8F8F7")
    gs = GridSpec(2, 2, figure=fig,
                  left=0.07, right=0.97, top=0.91, bottom=0.06,
                  hspace=0.42, wspace=0.36)

    ax_tr  = fig.add_subplot(gs[0, 0])   # Trajectory
    ax_err = fig.add_subplot(gs[1, :])   # Error vs Time
    ax_attn= fig.add_subplot(gs[0, 1])   # Attention heatmap

    # ── Panel A: Trajectory ───────────────────────────────────────────────
    # GPS scatter (subsample)
    ss = max(1, len(gps_interp_lat) // 300)
    ax_tr.scatter(gps_interp_lon[::ss], gps_interp_lat[::ss],
                  s=7, c=C["gps"], marker="x", lw=0.7, alpha=0.50,
                  zorder=2, label="GPS Reference")

    ax_tr.plot(lon_dr,  lat_dr,  c=C["dr"],  lw=0.9, ls=(0,(4,2)),
               zorder=3, alpha=0.80, label="Dead Reckoning")
    ax_tr.plot(lon_bl,  lat_bl,  c=C["bl"],  lw=0.9, ls=(0,(2,1)),
               zorder=4, alpha=0.80, label="BiLSTM-only (Δ-int)")
    ax_tr.plot(lon_ekf, lat_ekf, c=C["ekf"], lw=1.6,
               zorder=5, label=f"Hybrid BiLSTM+KF  [RMSE={S_ekf['rmse']:.1f} m]")

    # GPS update markers
    gps_upd_lon = lon_ekf[gps_used_mask]
    gps_upd_lat = lat_ekf[gps_used_mask]
    ax_tr.scatter(gps_upd_lon[::max(1,len(gps_upd_lon)//40)],
                  gps_upd_lat[::max(1,len(gps_upd_lat)//40)],
                  s=18, c="red", marker="o", alpha=0.35, zorder=6,
                  label="KF GPS update point")

    # Start / end markers
    ax_tr.plot(lon_ekf[0],  lat_ekf[0],  "^", ms=8, c=C["ekf"], zorder=9)
    ax_tr.plot(lon_ekf[-1], lat_ekf[-1], "s", ms=7, c=C["ekf"], zorder=9)
    ax_tr.annotate("Start", (lon_ekf[0],  lat_ekf[0]),
                   xytext=(5, 5), textcoords="offset points",
                   fontsize=7, color=C["ekf"])
    ax_tr.annotate("End",   (lon_ekf[-1], lat_ekf[-1]),
                   xytext=(5, -9), textcoords="offset points",
                   fontsize=7, color=C["ekf"])

    ax_tr.set_xlabel("Longitude (°E)")
    ax_tr.set_ylabel("Latitude (°N)")
    ax_tr.set_title("(a)  Estimated Trajectory — Real Test Data",
                    loc="left", fontweight="bold")
    ax_tr.legend(loc="best", framealpha=0.92,
                 edgecolor="#CCCCCC", handlelength=2.2, fontsize=7.5)
    ax_tr.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax_tr.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_tr.tick_params(which="minor", length=2)

    # ── Panel B: Error vs Time ────────────────────────────────────────────
    # GPS correction tick marks
    gps_update_times = TIME[gps_used_mask]
    for gt_ in gps_update_times[::max(1, len(gps_update_times)//60)]:
        ax_err.axvline(gt_, color=C["gps"], lw=0.25, alpha=0.2, zorder=1)

    # 10 m target band
    ax_err.axhspan(0, 10, alpha=0.07, color=C["ekf"], zorder=0)
    ax_err.axhline(10, c=C["ekf"], lw=0.9, ls="--", alpha=0.55, zorder=1,
                   label="10 m target")

    ax_err.fill_between(TIME, err_bl,  alpha=0.08, color=C["bl"])
    ax_err.fill_between(TIME, err_ekf, alpha=0.20, color=C["ekf"])

    ax_err.plot(TIME, err_dr,  c=C["dr"],  lw=0.85, ls=(0,(4,2)), alpha=0.85,
                label=f"Dead Reckoning   RMSE={S_dr['rmse']:.1f} m")
    ax_err.plot(TIME, err_bl,  c=C["bl"],  lw=0.85, ls=(0,(2,1)), alpha=0.80,
                label=f"BiLSTM-only      RMSE={S_bl['rmse']:.1f} m")
    ax_err.plot(TIME, err_ekf, c=C["ekf"], lw=1.40,
                label=f"Hybrid BiLSTM+KF RMSE={S_ekf['rmse']:.1f} m")

    # Annotate GPS corrections
    if len(gps_update_times) > 0:
        t_ann = gps_update_times[len(gps_update_times)//4]
        idx_ann = np.argmin(np.abs(TIME - t_ann))
        ax_err.annotate(
            f"GPS corrections\n({n_gps_updates} updates)",
            xy=(t_ann, err_ekf[idx_ann]),
            xytext=(30, 40), textcoords="offset points",
            fontsize=7.5, color=C["gps"],
            arrowprops=dict(arrowstyle="->", color=C["gps"], lw=0.8,
                            connectionstyle="arc3,rad=0.25")
        )

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error vs GPS (m)")
    ax_err.set_title(
        "(b)  Positional Error vs. Time  —  Real Test.csv  "
        f"({len(df_imu):,} rows, {df_imu['time_s'].iloc[-1]/60:.0f} min)",
        loc="left", fontweight="bold")
    ax_err.legend(ncol=2, loc="upper left",
                  framealpha=0.92, edgecolor="#CCCCCC")
    ax_err.set_xlim(0, TIME[-1])
    ax_err.set_ylim(bottom=0)
    ax_err.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_err.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax_err.tick_params(which="minor", length=2)

    # ── Panel C: Attention Heatmap ────────────────────────────────────────
    n_show  = min(50, len(attn_maps))
    indices = np.linspace(0, len(attn_maps)-1, n_show, dtype=int)
    attn_show = attn_maps[indices]
    # attn_show may be (N, T, T) or (N, T) depending on averaging
    if attn_show.ndim == 3:
        attn_show = attn_show.mean(axis=-1)        # (N, T)

    im = ax_attn.imshow(attn_show, aspect="auto", cmap="plasma",
                        interpolation="nearest", origin="upper")
    plt.colorbar(im, ax=ax_attn, pad=0.02, shrink=0.85).set_label(
        "Attention Weight", fontsize=8, fontfamily="serif")
    ax_attn.set_xlabel("Time Step in Window")
    ax_attn.set_ylabel("Sequence Sample Index")
    ax_attn.set_title("(c)  BiLSTM Attention\nHeatmap (Real Test Data)",
                      loc="left", fontweight="bold")

    # ── Metrics box ───────────────────────────────────────────────────────
    tbl = (
        f"  Positioning Error (vs GPS)\n"
        f"  {'─'*34}\n"
        f"  {'Method':<22}  RMSE       MAE\n"
        f"  {'─'*34}\n"
        f"  {'Dead Reckoning':<22}  {S_dr['rmse']:>6.1f} m  {S_dr['mae']:>6.1f} m\n"
        f"  {'BiLSTM-only':<22}  {S_bl['rmse']:>6.1f} m  {S_bl['mae']:>6.1f} m\n"
        f"  {'Hybrid BiLSTM+KF':<22}  {S_ekf['rmse']:>6.1f} m  {S_ekf['mae']:>6.1f} m\n"
        f"  {'─'*34}\n"
        f"  GPS updates: {n_gps_updates:,}   Improv: {improv:.1f}%"
    )
    fig.text(0.515, 0.72, tbl, fontsize=7.8, fontfamily="monospace",
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc="#EFEFEF",
                       ec="#CCCCCC", lw=0.8, alpha=0.93))

    # ── Global title ──────────────────────────────────────────────────────
    fig.suptitle(
        "Inference on Real Test Dataset  ·  BiLSTM-Attention + GPS-corrected KF\n"
        r"Telkom University IMU  $\cdot$  DOI: 10.34820/FK2/RUYZJT"
        r"  $\cdot$  Model loaded from saved checkpoint",
        fontsize=10.5, fontfamily="serif", fontweight="bold", y=0.975)
    fig.text(0.985, 0.007,
             "© Telkom University  |  Inference Script  |  Pre-trained BiLSTM+KF",
             ha="right", va="bottom", fontsize=6, color="#AAAAAA",
             fontfamily="serif", style="italic")

    out_path = os.path.join(OUT_DIR, "telkom_inference_result.png")
    fig.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Plot saved: {out_path}")

# ── Save result CSV ───────────────────────────────────────────────────────
result_df = pd.DataFrame({
    "time_s":     time_aligned,
    "lat_ekf":    lat_ekf,
    "lon_ekf":    lon_ekf,
    "lat_bl":     lat_bl,
    "lon_bl":     lon_bl,
    "lat_dr":     lat_dr,
    "lon_dr":     lon_dr,
    "lat_gps_ref":gps_interp_lat,
    "lon_gps_ref":gps_interp_lon,
    "err_ekf_m":  err_ekf,
    "err_bl_m":   err_bl,
    "err_dr_m":   err_dr,
    "gps_update": gps_used_mask.astype(int),
})
csv_path = os.path.join(OUT_DIR, "telkom_inference_result.csv")
result_df.to_csv(csv_path, index=False)
print(f"  ✓ Results CSV: {csv_path}")

print(f"\n{'='*66}")
print(f"  INFERENCE COMPLETE")
print(f"  Hybrid BiLSTM+KF RMSE  : {S_ekf['rmse']:.2f} m")
print(f"  GPS updates applied     : {n_gps_updates:,}")
print(f"  Total positions inferred: {len(lat_ekf):,}")
print(f"{'='*66}\n")
