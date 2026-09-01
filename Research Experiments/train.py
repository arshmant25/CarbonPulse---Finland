#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# train_15min.py  (now frequency-dynamic — see ROUND 3 CHANGES)
#
# Standalone training script for CSC Roihu. Trains RNN, GRU, LSTM
# + VAR + SARIMA baselines on a pre-aggregated feature set at
# whichever frequency you point --data-dir at (15min / hourly /
# daily / weekly / monthly / yearly), evaluates on the held-out
# test set, and produces:
#   - loss curves (train vs val) per DL model
#   - actual-vs-predicted / scatter / residual plots, saved
#     SEPARATELY per model (GRU/RNN/LSTM/VAR/SARIMA), 3 files each
#   - a "next forecast" summary per DL model: last known timestamp,
#     next predicted timestamp (frequency-aware step), and
#     predicted values in original units (gCO2/kWh)
#
# ── CHANGES IN THIS VERSION (round 1) ───────────────────────
# 1. EARLY STOPPING, 2. LR SCHEDULER, 3. UNIFIED EPOCH BUDGET,
# 4. PERSISTENCE BASELINE — see previous version's changelog,
#    all unchanged and still active.
#
# ── ROUND 2 CHANGES ──────────────────────────────────────────
# 6. BIAS CORRECTION (fit on VAL, applied at eval + saved to
#    bias_corrections.json), 7. --loss-fn {mse,huber},
# 8. --drop-features ablation — all unchanged and still active.
#
# ── ROUND 3 CHANGES (this version) — frequency dynamism ─────
# 9.  --frequency {15min,hourly,daily,weekly,monthly,yearly}.
#     Drives SEQ_LEN, HORIZON, SARIMA seasonal order, VAR/SARIMA
#     resample rule, and plot window length from one flag, per
#     the table you gave me. --seq-len/--horizon still override
#     the frequency default if you pass them explicitly.
# 10. SARIMA BASELINE ADDED: run_sarima_baseline() fits
#     SARIMAX(1,0,1)(1,0,1,seasonal_period) — seasonal_period is
#     frequency-dependent (96/24/7/52/12), or a plain (1,0,1) with
#     no seasonal term for yearly. IMPORTANT METHODOLOGY FIX per
#     your note: SARIMA now trains/tests on the SAME chronological
#     train/test date boundaries as the DL models (not a fixed
#     demo Jan/Feb 2024 window) — see train_start/train_end/
#     test_start/test_end below. For sub-daily frequencies
#     (15min/hourly) with 6 years of history, fitting SARIMAX on
#     the full multi-year train period is computationally
#     infeasible, so --sarima-max-train-obs (default 20,000)
#     caps the TRAINING window to the most recent N observations
#     ending exactly at the train/val boundary — the TEST period
#     and comparison to the DL models' test period stays identical,
#     only the SARIMA training depth is capped. Set to 0 to disable
#     capping (only advisable for daily/weekly/monthly/yearly).
# 11. VAR IS NOW FREQUENCY-DYNAMIC: previously hardcoded to
#     resample("1h") regardless of the experiment. Now resamples
#     to the same pandas frequency string as the current
#     --frequency, and its train/test split uses the same
#     chronological boundaries as everything else. maxlags is
#     also frequency-scaled (see var_maxlags) since 48 lags makes
#     sense hourly but not yearly.
# 12. PLOTTING REWRITTEN: plot_predicted_vs_actual() (all models
#     overlaid on 3 shared panels) is REMOVED. Replaced with
#     save_model_plots(), called once per model (GRU/RNN/LSTM/VAR/
#     SARIMA), writing 3 separate files into
#     <output_dir>/<MODEL_NAME>/{actual_vs_predicted,scatter,
#     residual}_<target>.png. Window length for the overlay panel
#     comes from FREQUENCY_CONFIGS[freq]["plot_steps"], not a
#     hardcoded 7-day/672 assumption.
# 13. HORIZON policy: per your note, HORIZON=1 (one-step-ahead) is
#     kept constant across every frequency by default — "1 step"
#     just means a different wall-clock gap depending on
#     --frequency. This keeps the research question identical
#     ("how well does each model predict the very next observation
#     at this resolution") across all experiments.
# 14. print_next_forecast() next-timestamp math is now
#     frequency-aware via forecast_delta_for() instead of a
#     hardcoded 15-minute Timedelta.
# 15. YEARLY IS A SPECIAL CASE: the table you gave has
#     SEQ_LEN=None / HORIZON=None for yearly, because resampling
#     2018-2026 to yearly leaves ~8-9 rows total — nowhere near
#     enough for a sequence model window. When --frequency yearly
#     is selected, RNN/GRU/LSTM training is SKIPPED automatically
#     (with a printed explanation) and only VAR + non-seasonal
#     SARIMA(1,0,1) run.
# 16. VAR and SARIMA metrics are now saved to
#     var_results.json / sarima_results.json in --output-dir,
#     alongside the existing gru/lstm/rnn_results.json.
#
# ── IMPORTANT — NOT COVERED BY THIS FILE ─────────────────────
# This script only trains/evaluates on an ALREADY-BUILT
# df_model.parquet + column_config.json for the chosen frequency.
# It does NOT re-run the upstream feature-engineering pipeline
# (Stage 1-7 in your notebook / build_features_from_db() in
# main.py) at a different resolution. LAG_STEPS, ROLL_WINDOWS, and
# the 3-min-clean -> aggregate step are still hardcoded to 15-min
# assumptions in that upstream code. To actually produce a
# df_model.parquet at hourly/daily/weekly/monthly/yearly
# resolution, that feature-engineering stage needs the same kind
# of --frequency parameterization applied to it separately — see
# the folder-naming answer for how to organize this once you do.
# ─────────────────────────────────────────────────────────────

import argparse
import copy
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════
# FREQUENCY CONFIG — the single source of truth for Task 1
# ════════════════════════════════════════════════════════════
#
# seq_len / horizon     : sequence model window / prediction horizon
# sarima_seasonal        : SARIMA seasonal period (None = no seasonal term)
# pandas_freq            : resample rule used for VAR + SARIMA
# plot_steps              : how many points the "overlay" plot panel shows
# var_maxlags             : VAR max lag order to try (statsmodels select_order)
#
# All other training hyperparameters (BATCH_SIZE, HIDDEN_SIZE,
# NUM_LAYERS, DROPOUT, LR, MAX_EPOCHS, PATIENCE) stay constant
# across frequencies per your table — those are just the existing
# --batch-size/--hidden-size/... CLI defaults, already set to
# 64/64/2/0.2/5e-4/60/6 below, unchanged by --frequency.

FREQUENCY_CONFIGS = {
    "15min":   dict(seq_len=96,  horizon=1, sarima_seasonal=96,  pandas_freq="15min",
                     plot_steps=672, var_maxlags=96),
    "hourly":  dict(seq_len=168, horizon=1, sarima_seasonal=24,  pandas_freq="1h",
                     plot_steps=168, var_maxlags=48),
    "daily":   dict(seq_len=90,  horizon=1, sarima_seasonal=7,   pandas_freq="1D",
                     plot_steps=90,  var_maxlags=14),
    "weekly":  dict(seq_len=52,  horizon=1, sarima_seasonal=52,  pandas_freq="1W",
                     plot_steps=52,  var_maxlags=12),
    "monthly": dict(seq_len=24,  horizon=1, sarima_seasonal=12,  pandas_freq="1MS",
                     plot_steps=24,  var_maxlags=12),
    "yearly":  dict(seq_len=None, horizon=None, sarima_seasonal=None, pandas_freq="1YS",
                     plot_steps=5,   var_maxlags=2),
}


def forecast_delta_for(frequency: str, last_ts: pd.Timestamp):
    """Returns the timestamp of the 'next' observation after last_ts,
    for the given frequency. DateOffset (month/year) handles calendar
    month/year-length variability correctly; Timedelta is used for
    fixed-length steps."""
    deltas = {
        "15min":  pd.Timedelta(minutes=15),
        "hourly": pd.Timedelta(hours=1),
        "daily":  pd.Timedelta(days=1),
        "weekly": pd.Timedelta(weeks=1),
        "monthly": pd.DateOffset(months=1),
        "yearly":  pd.DateOffset(years=1),
    }
    return last_ts + deltas[frequency]


# ════════════════════════════════════════════════════════════
# CLI ARGS
# ════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="CarbonPulse multi-frequency training (RNN/GRU/LSTM/VAR/SARIMA)")

    p.add_argument("--frequency", choices=list(FREQUENCY_CONFIGS.keys()), default="15min",
                   help="Which resolution's df_model.parquet/column_config.json you're pointing "
                        "--data-dir at. Drives SEQ_LEN, HORIZON, SARIMA seasonal order, VAR/SARIMA "
                        "resample rule, and plot window length (see FREQUENCY_CONFIGS).")

    p.add_argument("--data-dir", default="test",
                   help="Dir containing df_model.parquet and column_config.json for this frequency")
    p.add_argument("--output-dir", default="test",
                   help="Dir to write results, plots, and json metrics")
    p.add_argument("--model-dir", default="test",
                   help="Dir to write model .pt/.pkl artifacts")

    # NEW: default None -> resolved from FREQUENCY_CONFIGS unless explicitly overridden
    p.add_argument("--seq-len", type=int, default=None,
                   help="Overrides the --frequency default SEQ_LEN. Leave unset to use the table value.")
    p.add_argument("--horizon", type=int, default=None,
                   help="Overrides the --frequency default HORIZON (normally always 1 -- see changelog #13).")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Initial LR; ReduceLROnPlateau lowers it automatically once val loss plateaus")

    p.add_argument("--epochs", type=int, default=60,
                   help="Max epochs ceiling for all models (early stopping decides the actual stop point)")
    p.add_argument("--rnn-epochs", type=int, default=None, help="Override max epochs for RNN")
    p.add_argument("--gru-epochs", type=int, default=None, help="Override max epochs for GRU")
    p.add_argument("--lstm-epochs", type=int, default=None, help="Override max epochs for LSTM")

    p.add_argument("--patience", type=int, default=6,
                   help="Early stopping patience (epochs with no val-loss improvement)")
    p.add_argument("--min-delta", type=float, default=1e-5,
                   help="Minimum val-loss improvement to reset patience counter")

    p.add_argument("--skip-var", action="store_true", help="Skip the VAR baseline")
    p.add_argument("--skip-sarima", action="store_true", help="Skip the SARIMA baseline")

    p.add_argument("--sarima-order", type=int, nargs=3, default=[1, 0, 1],
                   metavar=("p", "d", "q"), help="Non-seasonal SARIMA order (default 1 0 1)")
    p.add_argument("--sarima-max-train-obs", type=int, default=20000,
                   help="Cap SARIMA training window to the most recent N observations ending at "
                        "the train/val boundary (0 = no cap). Sub-daily frequencies need this; "
                        "daily/weekly/monthly/yearly can usually set this to 0.")

    p.add_argument("--forecast-steps", type=int, default=1,
                   help="Reserved for future multi-step forecasting in print_next_forecast(); "
                        "current implementation is always single-step (matches HORIZON=1).")

    p.add_argument("--loss-fn", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber-delta", type=float, default=1.0)

    p.add_argument("--drop-features", nargs="*", default=[],
                   help="Feature name substrings to exclude from FEATURE_COLS before training")

    return p.parse_args()


# ════════════════════════════════════════════════════════════
# MODELS  (identical to notebook cell 15 — unchanged)
# ════════════════════════════════════════════════════════════

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          nonlinearity="tanh", dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out[:, -1, :]))


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


# ════════════════════════════════════════════════════════════
# DATASET  (identical to notebook cell 13 — unchanged)
# ════════════════════════════════════════════════════════════

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int = 1):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_samples = len(X) - seq_len - horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_target = self.y[idx + self.seq_len + self.horizon - 1]
        return x_seq, y_target


# ════════════════════════════════════════════════════════════
# PERSISTENCE BASELINE (unchanged)
# ════════════════════════════════════════════════════════════

def print_persistence_baseline(X_test_raw, y_test_raw, FEATURE_COLS, TARGET_COLS):
    print("\n" + "=" * 60)
    print(" PERSISTENCE BASELINE (predict = last known value)")
    print("=" * 60)

    for i, col in enumerate(TARGET_COLS):
        lag1_col = f"{col}_lag1"
        if lag1_col not in FEATURE_COLS:
            print(f"  [SKIP] {col}: no matching {lag1_col} feature found")
            continue

        lag1_idx = FEATURE_COLS.index(lag1_col)
        naive_pred = X_test_raw[:, lag1_idx]
        actual = y_test_raw[:, i]

        naive_mae = np.mean(np.abs(naive_pred - actual))
        naive_rmse = np.sqrt(np.mean((naive_pred - actual) ** 2))
        naive_mape = np.mean(np.abs((naive_pred - actual) / (actual + 1e-8))) * 100

        print(f"  {col}:")
        print(f"    MAE={naive_mae:.4f}  RMSE={naive_rmse:.4f}  MAPE={naive_mape:.2f}%")
    print()


# ════════════════════════════════════════════════════════════
# TRAIN / EVAL — early stopping + LR scheduler (unchanged)
# ════════════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader, epochs=60, lr=5e-4,
                 model_name="Model", device=None, patience=6, min_delta=1e-5,
                 loss_fn="mse", huber_delta=1.0):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Training on device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if loss_fn == "huber":
        criterion = nn.HuberLoss(delta=huber_delta)
        print(f"[INFO] {model_name}: using HuberLoss(delta={huber_delta})")
    else:
        criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[INFO] ===== {model_name} Epoch {epoch}/{epochs} =====")

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            if epoch == 1 and i == 0:
                print(f"[DEBUG] Train batch shape: {xb.shape}, device: {xb.device}")

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            if not torch.isfinite(loss):
                print(f"[STOP] NaN detected at epoch {epoch}, batch {i}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                return train_losses, val_losses

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if i % 100 == 0:
                print(f"[TRAIN] Epoch {epoch} Batch {i} Loss: {loss.item():.6f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        print("[INFO] Running validation...")
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for i, (xb, yb) in enumerate(val_loader):
                xb = xb.to(device)
                yb = yb.to(device)

                if epoch == 1 and i == 0:
                    print(f"[DEBUG] Val batch shape: {xb.shape}, device: {xb.device}")

                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(n_val_batches, 1)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[RESULT] {model_name} Epoch {epoch}: "
              f"Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f} | LR={current_lr:.2e}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] {model_name}: no val improvement for {patience} epochs. "
                      f"Best epoch was {best_epoch} (val={best_val_loss:.6f}). Stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[INFO] {model_name}: restored weights from best epoch {best_epoch} (val={best_val_loss:.6f})")

    return train_losses, val_losses


def compute_bias_correction(model, val_loader, scaler_y, target_cols, model_name, device):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            all_preds.append(pred)
            all_trues.append(yb.numpy())

    pred_real = scaler_y.inverse_transform(np.concatenate(all_preds, axis=0))
    true_real = scaler_y.inverse_transform(np.concatenate(all_trues, axis=0))

    bias = np.mean(pred_real - true_real, axis=0)

    print(f"\n[BIAS CORRECTION] {model_name} (computed on VALIDATION set):")
    for i, col in enumerate(target_cols):
        print(f"  {col}: {bias[i]:+.4f} gCO2/kWh (subtract this from raw predictions at inference)")

    return bias


def evaluate_model(model, test_loader, scaler_y, target_cols, model_name, device,
                    bias_correction=None):
    model.eval()
    all_preds, all_trues = [], []

    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            all_preds.append(pred.detach().cpu().numpy())
            all_trues.append(yb.detach().cpu().numpy())
            del xb, yb, pred
            if i % 500 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[EVAL] {model_name} Batch {i} / {len(test_loader)}")

    pred_scaled = np.concatenate(all_preds, axis=0)
    true_scaled = np.concatenate(all_trues, axis=0)
    del all_preds, all_trues
    gc.collect()

    pred_real_raw = scaler_y.inverse_transform(pred_scaled)
    true_real = scaler_y.inverse_transform(true_scaled)

    if bias_correction is not None:
        pred_real = pred_real_raw - bias_correction[np.newaxis, :]
    else:
        pred_real = pred_real_raw

    print(f"\n[{model_name}] Evaluation Results:")
    results = {}
    for i, col in enumerate(target_cols):
        mae_raw = np.mean(np.abs(pred_real_raw[:, i] - true_real[:, i]))
        mape_raw = np.mean(np.abs((pred_real_raw[:, i] - true_real[:, i]) / (true_real[:, i] + 1e-8))) * 100

        mae = np.mean(np.abs(pred_real[:, i] - true_real[:, i]))
        rmse = np.sqrt(np.mean((pred_real[:, i] - true_real[:, i]) ** 2))
        mape = np.mean(np.abs((pred_real[:, i] - true_real[:, i]) / (true_real[:, i] + 1e-8))) * 100
        bias = np.mean(pred_real[:, i] - true_real[:, i])

        results[col] = {
            "MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape), "Bias": float(bias),
            "true": true_real[:, i], "pred": pred_real[:, i],
        }

        if bias_correction is not None:
            print(f"  {col}: RAW      MAE={mae_raw:.4f}  MAPE={mape_raw:.2f}%")
            print(f"  {col}: CORRECTED MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  Bias={bias:+.4f}")
        else:
            print(f"  {col}: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  Bias={bias:+.4f}")

    return results


# ════════════════════════════════════════════════════════════
# VAR BASELINE — now frequency-dynamic (Task 4)
# ════════════════════════════════════════════════════════════

def run_var_baseline(df_model, frequency, freq_cfg, train_start, train_end,
                      test_start, test_end, target_cols):
    """
    Resamples to the SAME frequency as the current experiment
    (previously hardcoded to "1h" regardless of --frequency), and
    splits on the same chronological boundaries as the DL models
    and SARIMA. Returns a dict shaped like evaluate_model()'s
    output so it can go straight into save_model_plots() and the
    results json.
    """
    from statsmodels.tsa.vector_ar.var_model import VAR

    var_cols = [
        "co2_intensity_simple_avg", "consumption_co2_intens_simple_avg",
        "wind_mw_mean", "nuclear_mw_mean", "total_production_mw_mean",
        "chp_district_mw_mean", "chp_industrial_mw_mean",
        "hydro_mw_mean", "consumption_elec_mw_mean",
    ]
    var_cols = [c for c in var_cols if c in df_model.columns]

    df_var = df_model[var_cols].resample(freq_cfg["pandas_freq"]).mean().dropna()

    var_train = df_var[train_start:train_end]
    var_test = df_var[test_start:test_end]

    maxlags = min(freq_cfg["var_maxlags"], max(1, len(var_train) // 3 - 1))
    if maxlags < 1:
        print(f"[VAR] Not enough training rows ({len(var_train)}) at frequency={frequency}. Skipping VAR.")
        return None
    if len(var_test) == 0:
        print(f"[VAR] Test period is empty at frequency={frequency}. Skipping VAR.")
        return None

    print(f"[VAR] frequency={frequency}  train_rows={len(var_train):,}  test_rows={len(var_test):,}  "
          f"maxlags={maxlags}")

    var_model = VAR(var_train)
    lag_order = var_model.select_order(maxlags=maxlags)
    var_fit = var_model.fit(lag_order.aic)

    n_lags = max(lag_order.aic, 1)
    var_pred = var_fit.forecast(var_train.values[-n_lags:], steps=len(var_test))
    var_pred_df = pd.DataFrame(var_pred, index=var_test.index, columns=var_test.columns)

    results = {}
    for target in target_cols:
        if target not in var_test.columns:
            continue
        true = var_test[target].values
        pred = var_pred_df[target].values
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100
        print(f"  VAR {target}: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  "
              f"(resolution={frequency})")
        results[target] = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape),
                            "true": true, "pred": pred}

    return results


# ════════════════════════════════════════════════════════════
# SARIMA BASELINE — new (Task 2)
# ════════════════════════════════════════════════════════════

def run_sarima_baseline(df_model, frequency, freq_cfg, train_start, train_end,
                         test_start, test_end, target_cols, sarima_order=(1, 0, 1),
                         max_train_obs=20000):
    """
    Fits SARIMAX(order)(seasonal_order) per target, where the
    seasonal period comes from freq_cfg["sarima_seasonal"]
    (96/24/7/52/12, or None -> no seasonal term for yearly).

    Uses the SAME chronological train/test date boundaries as the
    DL models and VAR (train_start..train_end / test_start..test_end)
    -- this replaces the original demo's fixed Jan/Feb-2024 window,
    per your methodology correction.

    max_train_obs caps the TRAINING window length only (most recent
    N observations up to train_end) because fitting SARIMAX with a
    seasonal period of 96 or 24 over several years of sub-daily data
    is not computationally tractable. The TEST period is never
    capped, so the comparison against DL/VAR test-set metrics stays
    apples-to-apples. Set max_train_obs=0 to disable capping
    (fine for daily/weekly/monthly/yearly, where full-period fits
    are cheap).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    seasonal_period = freq_cfg["sarima_seasonal"]
    seasonal_order = (1, 0, 1, seasonal_period) if seasonal_period else (0, 0, 0, 0)

    results = {}
    for target in target_cols:
        if target not in df_model.columns:
            print(f"[SARIMA] {target} not found in df_model columns. Skipping.")
            continue

        series = df_model[target].resample(freq_cfg["pandas_freq"]).mean().dropna()

        s_train = series[train_start:train_end]
        s_test = series[test_start:test_end]

        if max_train_obs and len(s_train) > max_train_obs:
            print(f"[SARIMA] {target}: capping train window {len(s_train):,} -> "
                  f"last {max_train_obs:,} obs (test period unaffected)")
            s_train = s_train.iloc[-max_train_obs:]

        if len(s_train) < 10 or len(s_test) == 0:
            print(f"[SARIMA] {target}: not enough data (train={len(s_train)}, "
                  f"test={len(s_test)}) at frequency={frequency}. Skipping.")
            continue

        try:
            fit = SARIMAX(
                s_train, order=tuple(sarima_order), seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)

            pred = np.asarray(fit.forecast(steps=len(s_test)))
            true = s_test.values

            mae = np.mean(np.abs(pred - true))
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            mape = np.mean(np.abs((pred - true) / (true + 1e-8))) * 100

            print(f"  SARIMA{tuple(sarima_order)}{seasonal_order} {target}: "
                  f"MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%  "
                  f"(train={len(s_train):,} obs, test={len(s_test):,} obs, freq={frequency})")

            results[target] = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape),
                                "true": true, "pred": pred}
        except Exception as e:
            print(f"[SARIMA] {target} fit failed: {e}")

    return results if results else None


# ════════════════════════════════════════════════════════════
# PLOTTING — one model at a time, 3 separate files (Task 3 / 3.1)
# ════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "GRU": "#2a9d8f", "LSTM": "#e76f51", "RNN": "#e9c46a",
    "VAR": "#6d6875", "SARIMA": "#264653",
}


def save_model_plots(model_name, results, target_cols, output_dir, plot_steps):
    """
    Writes, per target column:
      <output_dir>/<model_name>/actual_vs_predicted_<target>.png
      <output_dir>/<model_name>/scatter_<target>.png
      <output_dir>/<model_name>/residual_<target>.png

    results: {target: {"true": np.ndarray, "pred": np.ndarray, ...}}
    plot_steps: window length for the overlay panel, from
                FREQUENCY_CONFIGS[freq]["plot_steps"]
                (NOT a hardcoded 7-day/672 assumption — Task 3.1).
    """
    if not results:
        print(f"[PLOTS] {model_name}: no results to plot, skipping.")
        return

    color = MODEL_COLORS.get(model_name, "#2a9d8f")
    model_out_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    for target in target_cols:
        if target not in results:
            continue
        true = np.asarray(results[target]["true"])
        pred = np.asarray(results[target]["pred"])
        safe = target.replace("/", "_").replace(" ", "_")

        # ── 1. Actual vs predicted overlay ───────────────────
        n = min(plot_steps, len(true))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(true[:n], color="#264653", lw=1.2, label="Actual", zorder=5)
        ax.plot(pred[:n], color=color, lw=1.0, linestyle="--", alpha=0.9, label=model_name)
        ax.set_title(f"{model_name} — Actual vs Predicted — {target}\n"
                     f"({n} steps shown)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("gCO2/kWh")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        p1 = os.path.join(model_out_dir, f"actual_vs_predicted_{safe}.png")
        plt.savefig(p1, dpi=130, bbox_inches="tight")
        plt.close(fig)

        # ── 2. Scatter ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 6))
        n_sample = min(5000, len(true))
        idx = np.random.choice(len(true), n_sample, replace=False)
        ax.scatter(true[idx], pred[idx], alpha=0.2, s=6, color=color)
        mn = min(true.min(), pred.min())
        mx = max(true.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="Perfect prediction", zorder=10)
        ax.set_xlabel("Actual (gCO2/kWh)")
        ax.set_ylabel("Predicted (gCO2/kWh)")
        ax.set_title(f"{model_name} — Scatter — {target}\n(closer to diagonal = better)", fontsize=10)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        p2 = os.path.join(model_out_dir, f"scatter_{safe}.png")
        plt.savefig(p2, dpi=130, bbox_inches="tight")
        plt.close(fig)

        # ── 3. Residual distribution ──────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        residuals = pred - true
        ax.hist(residuals, bins=80, alpha=0.75, color=color, density=True)
        ax.axvline(0, color="black", lw=1.5, linestyle="--")
        ax.set_xlabel("Residual (Predicted - Actual) gCO2/kWh")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name} — Residual Distribution — {target}\n"
                     f"(centred at 0 = unbiased)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        p3 = os.path.join(model_out_dir, f"residual_{safe}.png")
        plt.savefig(p3, dpi=130, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {p1}\nSaved: {p2}\nSaved: {p3}")


# ════════════════════════════════════════════════════════════
# NEXT-STEP FORECAST SUMMARY — frequency-aware (Task 6)
# ════════════════════════════════════════════════════════════

def print_next_forecast(models_dict, X_test_s, test_index, scaler_y,
                         target_cols, seq_len, frequency, device=None):
    print("\n" + "=" * 60)
    print(" NEXT-STEP FORECAST (sanity check on last test window)")
    print("=" * 60)

    if len(X_test_s) < seq_len:
        print(f"[WARN] Test set has only {len(X_test_s)} rows, need >= {seq_len} for one window. Skipping.")
        return

    last_ts = test_index[-1]
    next_ts = forecast_delta_for(frequency, last_ts)

    print(f"Frequency                         : {frequency}")
    print(f"Last known timestamp in test set  : {last_ts}")
    print(f"Next timestamp being predicted     : {next_ts}")
    print()

    seed = X_test_s[-seq_len:]

    for model_name, model in models_dict.items():
        model.eval()
        x = torch.FloatTensor(seed).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(x).detach().cpu().numpy()[0]

        pred_real = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]

        print(f"[{model_name}]")
        for i, col in enumerate(target_cols):
            print(f"   {col:40s} -> {round(float(pred_real[i]), 3):>10} gCO2/kWh")
        print()


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model_dir = args.model_dir or args.data_dir
    freq_cfg = FREQUENCY_CONFIGS[args.frequency]

    # Resolve seq_len / horizon: explicit CLI flag wins, else the
    # frequency table default (Task 1).
    seq_len = args.seq_len if args.seq_len is not None else freq_cfg["seq_len"]
    horizon = args.horizon if args.horizon is not None else freq_cfg["horizon"]

    run_dl = seq_len is not None and horizon is not None
    if not run_dl:
        print(f"\n[INFO] --frequency {args.frequency}: SEQ_LEN/HORIZON are undefined for this "
              f"frequency (yearly resampling leaves too few rows for a sequence window). "
              f"Skipping RNN/GRU/LSTM training. Only VAR + non-seasonal SARIMA will run.\n")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Frequency: {args.frequency}  |  SEQ_LEN={seq_len}  HORIZON={horizon}  "
          f"SARIMA seasonal={freq_cfg['sarima_seasonal']}  pandas_freq={freq_cfg['pandas_freq']}")

    # ── Load data + config ──────────────────────────────────
    df_model = pd.read_parquet(os.path.join(args.data_dir, "df_model.parquet"))

    with open(os.path.join(args.data_dir, "column_config.json")) as f:
        cfg = json.load(f)

    FEATURE_COLS = cfg["FEATURE_COLS"]
    TARGET_COLS = cfg["TARGET_COLS"]
    TRAIN_RATIO = cfg["TRAIN_RATIO"]
    VAL_RATIO = cfg["VAL_RATIO"]
    TEST_RATIO = cfg["TEST_RATIO"]
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    if args.drop_features:
        before = len(FEATURE_COLS)
        dropped = [c for c in FEATURE_COLS if any(sub in c for sub in args.drop_features)]
        FEATURE_COLS = [c for c in FEATURE_COLS if c not in dropped]
        print(f"\n[INFO] --drop-features {args.drop_features}: removed {len(dropped)}/{before} features")
        for c in dropped:
            print(f"  - dropped: {c}")
        print(f"[INFO] {len(FEATURE_COLS)} features remaining\n")

    # ── Time-based split — never shuffle time series ────────
    df_model = df_model.sort_index()
    n = len(df_model)
    train_end_idx = int(n * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(n * VAL_RATIO)

    train = df_model.iloc[:train_end_idx].copy()
    val = df_model.iloc[train_end_idx:val_end_idx].copy()
    test = df_model.iloc[val_end_idx:].copy()

    # These boundaries are reused by VAR and SARIMA so every model
    # in this run is compared on the IDENTICAL chronological split
    # (Task 2's methodology correction).
    train_start, train_end = train.index.min(), train.index.max()
    test_start, test_end = test.index.min(), test.index.max()

    print("=== TRAIN / VAL / TEST SPLIT ===")
    print(f"Train: {len(train):>10,} rows  ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"Val:   {len(val):>10,} rows  ({val.index.min().date()} -> {val.index.max().date()})")
    print(f"Test:  {len(test):>10,} rows  ({test.index.min().date()} -> {test.index.max().date()})")

    # ══════════════════════════════════════════════════════
    # Deep-learning models (skipped entirely for --frequency yearly)
    # ══════════════════════════════════════════════════════
    gru_results = lstm_results = rnn_results = None
    gru_model = rnn_model = lstm_model = None

    if run_dl:
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_raw = train[FEATURE_COLS].values
        X_val_raw = val[FEATURE_COLS].values
        X_test_raw = test[FEATURE_COLS].values

        y_train_raw = train[TARGET_COLS].values
        y_val_raw = val[TARGET_COLS].values
        y_test_raw = test[TARGET_COLS].values

        scaler_X.fit(X_train_raw)
        scaler_y.fit(y_train_raw)

        X_train_s = scaler_X.transform(X_train_raw)
        X_val_s = scaler_X.transform(X_val_raw)
        X_test_s = scaler_X.transform(X_test_raw)

        y_train_s = scaler_y.transform(y_train_raw)
        y_val_s = scaler_y.transform(y_val_raw)
        y_test_s = scaler_y.transform(y_test_raw)

        if "time_index" in FEATURE_COLS:
            ti = FEATURE_COLS.index("time_index")
            print("\n=== TIME_INDEX SCALER CHECK ===")
            print("Raw train time_index:", X_train_raw[:, ti].min(), "->", X_train_raw[:, ti].max())
            print("Scaled train time_index:", X_train_s[:, ti].min(), "->", X_train_s[:, ti].max())
            print("Raw test time_index:", X_test_raw[:, ti].min(), "->", X_test_raw[:, ti].max())
            print("Scaled test time_index:", X_test_s[:, ti].min(), "->", X_test_s[:, ti].max())

        print_persistence_baseline(X_test_raw, y_test_raw, FEATURE_COLS, TARGET_COLS)

        joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X.pkl"))
        joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y.pkl"))

        with open(os.path.join(model_dir, "column_config_effective.json"), "w") as f:
            json.dump({
                "FEATURE_COLS": FEATURE_COLS, "TARGET_COLS": TARGET_COLS,
                "dropped_feature_substrings": args.drop_features,
                "frequency": args.frequency, "seq_len": seq_len, "horizon": horizon,
            }, f, indent=2)
        print(f"[INFO] Saved effective feature config: "
              f"{os.path.join(model_dir, 'column_config_effective.json')}")

        print("[INFO] Creating datasets...")
        train_ds = TimeSeriesWindowDataset(X_train_s, y_train_s, seq_len, horizon)
        val_ds = TimeSeriesWindowDataset(X_val_s, y_val_s, seq_len, horizon)
        test_ds = TimeSeriesWindowDataset(X_test_s, y_test_s, seq_len, horizon)

        print(f"Train windows available: {len(train_ds):,}")
        print(f"Val windows available:   {len(val_ds):,}")
        print(f"Test windows available:  {len(test_ds):,}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        n_features = X_train_s.shape[1]

        rnn_epochs = args.rnn_epochs or args.epochs
        gru_epochs = args.gru_epochs or args.epochs
        lstm_epochs = args.lstm_epochs or args.epochs

        rnn_model = RNNModel(n_features, hidden_size=args.hidden_size,
                             num_layers=args.num_layers, dropout=args.dropout).to(device)
        rnn_train_losses, rnn_val_losses = train_model(
            rnn_model, train_loader, val_loader, epochs=rnn_epochs, lr=args.lr,
            model_name="RNN", device=device, patience=args.patience, min_delta=args.min_delta,
            loss_fn=args.loss_fn, huber_delta=args.huber_delta
        )
        torch.save(rnn_model.state_dict(), os.path.join(model_dir, "rnn_model.pt"))

        gru_model = GRUModel(n_features, hidden_size=args.hidden_size,
                            num_layers=args.num_layers, dropout=args.dropout).to(device)
        gru_train_losses, gru_val_losses = train_model(
            gru_model, train_loader, val_loader, epochs=gru_epochs, lr=args.lr,
            model_name="GRU", device=device, patience=args.patience, min_delta=args.min_delta,
            loss_fn=args.loss_fn, huber_delta=args.huber_delta
        )
        torch.save(gru_model.state_dict(), os.path.join(model_dir, "gru_model.pt"))

        lstm_model = LSTMModel(n_features, hidden_size=args.hidden_size,
                              num_layers=args.num_layers, dropout=args.dropout).to(device)
        lstm_train_losses, lstm_val_losses = train_model(
            lstm_model, train_loader, val_loader, epochs=lstm_epochs, lr=args.lr,
            model_name="LSTM", device=device, patience=args.patience, min_delta=args.min_delta,
            loss_fn=args.loss_fn, huber_delta=args.huber_delta
        )
        torch.save(lstm_model.state_dict(), os.path.join(model_dir, "lstm_model.pt"))

        rnn_bias = compute_bias_correction(rnn_model, val_loader, scaler_y, TARGET_COLS, "RNN", device)
        gru_bias = compute_bias_correction(gru_model, val_loader, scaler_y, TARGET_COLS, "GRU", device)
        lstm_bias = compute_bias_correction(lstm_model, val_loader, scaler_y, TARGET_COLS, "LSTM", device)

        bias_corrections = {
            "RNN": rnn_bias.tolist(), "GRU": gru_bias.tolist(), "LSTM": lstm_bias.tolist(),
            "target_cols": TARGET_COLS,
        }
        with open(os.path.join(model_dir, "bias_corrections.json"), "w") as f:
            json.dump(bias_corrections, f, indent=2)
        print(f"\n[INFO] Saved bias corrections: {os.path.join(model_dir, 'bias_corrections.json')}")

        print("\n=== FINAL TEST RESULTS (bias-corrected) ===")
        rnn_results = evaluate_model(rnn_model, test_loader, scaler_y, TARGET_COLS, "RNN", device,
                                      bias_correction=rnn_bias)
        gru_results = evaluate_model(gru_model, test_loader, scaler_y, TARGET_COLS, "GRU", device,
                                      bias_correction=gru_bias)
        lstm_results = evaluate_model(lstm_model, test_loader, scaler_y, TARGET_COLS, "LSTM", device,
                                       bias_correction=lstm_bias)

        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError

        def results_for_json(results):
            return {col: {"MAE": v["MAE"], "RMSE": v["RMSE"], "MAPE": v["MAPE"], "Bias": v["Bias"]}
                    for col, v in results.items()}

        with open(os.path.join(args.output_dir, "gru_results.json"), "w") as f:
            json.dump(results_for_json(gru_results), f, indent=4, default=convert)
        with open(os.path.join(args.output_dir, "lstm_results.json"), "w") as f:
            json.dump(results_for_json(lstm_results), f, indent=4, default=convert)
        with open(os.path.join(args.output_dir, "rnn_results.json"), "w") as f:
            json.dump(results_for_json(rnn_results), f, indent=4, default=convert)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, tl, vl, name in [
            (axes[0], gru_train_losses, gru_val_losses, "GRU"),
            (axes[1], lstm_train_losses, lstm_val_losses, "LSTM"),
            (axes[2], rnn_train_losses, rnn_val_losses, "RNN"),
        ]:
            ax.plot(tl, label="Train loss", color="#264653")
            ax.plot(vl, label="Val loss", color="#e76f51", linestyle="--")
            ax.set_title(f"{name} — Training vs Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(f"{'Huber' if args.loss_fn == 'huber' else 'MSE'} Loss (scaled)")
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "loss_curves_gru_lstm_rnn.png"), dpi=130)
        plt.close(fig)
        print(f"Saved: {os.path.join(args.output_dir, 'loss_curves_gru_lstm_rnn.png')}")

    # ══════════════════════════════════════════════════════
    # VAR baseline — frequency-dynamic (Task 4)
    # ══════════════════════════════════════════════════════
    var_results = None
    if not args.skip_var:
        try:
            var_results = run_var_baseline(
                df_model, args.frequency, freq_cfg,
                train_start, train_end, test_start, test_end, TARGET_COLS,
            )
        except Exception as e:
            print(f"[WARN] VAR baseline failed/skipped: {e}")

    if var_results:
        with open(os.path.join(args.output_dir, "var_results.json"), "w") as f:
            json.dump({col: {"MAE": v["MAE"], "RMSE": v["RMSE"], "MAPE": v["MAPE"]}
                      for col, v in var_results.items()}, f, indent=4)
        print(f"[INFO] Saved: {os.path.join(args.output_dir, 'var_results.json')}")

    # ══════════════════════════════════════════════════════
    # SARIMA baseline — new, frequency-dynamic (Task 2)
    # ══════════════════════════════════════════════════════
    sarima_results = None
    if not args.skip_sarima:
        try:
            sarima_results = run_sarima_baseline(
                df_model, args.frequency, freq_cfg,
                train_start, train_end, test_start, test_end, TARGET_COLS,
                sarima_order=tuple(args.sarima_order),
                max_train_obs=args.sarima_max_train_obs,
            )
        except Exception as e:
            print(f"[WARN] SARIMA baseline failed/skipped: {e}")

    if sarima_results:
        with open(os.path.join(args.output_dir, "sarima_results.json"), "w") as f:
            json.dump({col: {"MAE": v["MAE"], "RMSE": v["RMSE"], "MAPE": v["MAPE"]}
                      for col, v in sarima_results.items()}, f, indent=4)
        print(f"[INFO] Saved: {os.path.join(args.output_dir, 'sarima_results.json')}")

    # ══════════════════════════════════════════════════════
    # Plots — separate per model (Task 3 / 3.1)
    # ══════════════════════════════════════════════════════
    plot_steps = freq_cfg["plot_steps"]

    if gru_results:
        save_model_plots("GRU", gru_results, TARGET_COLS, args.output_dir, plot_steps)
    if rnn_results:
        save_model_plots("RNN", rnn_results, TARGET_COLS, args.output_dir, plot_steps)
    if lstm_results:
        save_model_plots("LSTM", lstm_results, TARGET_COLS, args.output_dir, plot_steps)
    if var_results:
        save_model_plots("VAR", var_results, TARGET_COLS, args.output_dir, plot_steps)
    if sarima_results:
        save_model_plots("SARIMA", sarima_results, TARGET_COLS, args.output_dir, plot_steps)

    # ══════════════════════════════════════════════════════
    # Next-step forecast summary — DL models only, frequency-aware (Task 6)
    # ══════════════════════════════════════════════════════
    if run_dl:
        print_next_forecast(
            {"RNN": rnn_model, "GRU": gru_model, "LSTM": lstm_model},
            X_test_s, test.index, scaler_y, TARGET_COLS,
            seq_len=seq_len, frequency=args.frequency, device=device,
        )

    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()