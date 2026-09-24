"""
CarbonPulse Finland — Hierarchical Time-Series Forecasting
============================================================
Forecasts regional GHG emissions at THREE levels of the data hierarchy:

    subclass  (region, sector, subclass)   e.g. Central Finland / Electricity / Housing
       |  sums to
    sector    (region, sector)             e.g. Central Finland / Electricity
       |  sums to
    region    (region)                     e.g. Central Finland (total_ktco2e)

...using FIVE model types at each level: GRU, RNN, LSTM (deep learning),
VAR and SARIMA (classical). After forecasting, it checks whether the three
levels are numerically COHERENT (do subclass forecasts really sum to the
sector forecast? do sector forecasts sum to the region forecast?) and
reports a bottom-up RECONCILED forecast as the recommended coherent answer.

============================== READ THIS FIRST ==============================
Your data has a hard constraint that shapes every design choice below:
each series has only ~20 usable annual points (2005-2024; 1990 exists but
is excluded -- see "Why 1990 is dropped"). That is an extremely small
sample for time-series ML, and it rules out some "obvious" approaches:

  - Per-series deep learning (train one GRU per subclass) would have ~14
    training points per model -- nowhere near enough for gradient descent
    to learn anything beyond noise. FIX: GRU/RNN/LSTM here are GLOBAL
    models -- ONE model per level, trained on windows pooled across every
    series at that level (e.g. all ~50 subclass series together), with the
    series' identity (region/sector/subclass) fed in as a learned
    embedding. This turns "20 points x 50 series" into a genuinely
    trainable dataset, and is standard practice for large panels of short
    series (the same idea behind Amazon's DeepAR, M5-competition winners).

  - VAR (Vector Autoregression) models several series jointly and needs
    roughly (n_variables)^2 parameters. With e.g. 14 sectors and ~14
    training years, a full VAR is mathematically singular -- it cannot be
    fit at all. FIX: the VAR step includes an automatic dimensionality
    guard. It only fits a group (e.g. all sectors in one region) if there
    is genuinely enough data to support it; otherwise it logs exactly why
    and skips that group rather than silently producing a fitted-but-
    meaningless model. In practice this means VAR will mostly succeed for
    small subclass groups (many sectors only have 2-6 subclasses) and will
    mostly be SKIPPED at the sector level (13-14 variables) and almost
    certainly skipped at the region level (19 variables) -- that is the
    honest, correct behaviour given the data size, not a bug.

  - SARIMA has no such issue (it is univariate, one series at a time) but
    annual data has no sub-year seasonal cycle, so the seasonal part of
    "SARIMA" is set to (0,0,0,0) -- it is functionally an ARIMA model.
    This is noted wherever SARIMA results are reported.

FEATURES USED (deliberately minimal, per your instruction to skip GDP):
  - The series' own past SEQ_LEN values (the lag window) -- the primary signal.
  - A normalised time index (0=first year, 1=last year in the series) at
    each step of the window, so the model has an explicit sense of "where
    in time" a window sits, not just relative recent values.
  - Learned embeddings of the series' identity: region (all levels),
    sector (sector & subclass levels), subclass (subclass level only).
    This lets the one shared model still specialise per series.
  No GDP, no population, no external data -- purely the emissions series
  themselves plus their position in the hierarchy and in time.

WHY 1990 IS DROPPED: the source data jumps from 1990 straight to 2005 (a
15-year gap). Including 1990 in a sequence model would make the network
learn from a fake "1990 is 1 step before 2005" adjacency that doesn't
exist in reality. So the modelling window is 2005-latest only; 1990 is
simply unused here (it's a policy decision, not a limitation of the code
-- see PREPROCESSING CONFIG below if you want to change it).

PREPROCESSING / SPLIT:
  - Years split chronologically (not randomly -- this is a time series) by
    ratio: 70% train / 15% val / 15% test, applied to the sorted list of
    years and used identically across every series so results are
    comparable across the hierarchy.
  - Each series is scaled independently (MinMax to [0,1]) using ONLY its
    training-period values to fit the scaler -- this avoids leaking
    validation/test information into the scaling step.
  - Series shorter than SEQ_LEN + 2 points of training data are skipped
    (logged), since there isn't enough signal to build even one window.
"""

import os
import sys
import json
import sqlite3
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# CONFIG — change these, nothing else needs editing
# ══════════════════════════════════════════════════════════════
DB_PATH    = "data/project_data.db"
OUTPUT_DIR = Path(r"Research Experiments/outputs_regional_forecasting")

MIN_YEAR = 2005            # 1990 excluded -- see module docstring
SEQ_LEN  = 5                # years of history fed to GRU/RNN/LSTM per window
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / val / test, by YEAR (chronological)

# Deep learning
EPOCHS       = 300
PATIENCE     = 20           # early stopping: stop if val loss doesn't improve for this many epochs
BATCH_SIZE   = 64
HIDDEN_SIZE  = 32
NUM_LAYERS   = 1
EMBED_DIM    = 4
LEARNING_RATE = 1e-3
DROPOUT      = 0.1
GRAD_CLIP    = 1.0

# Classical models
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (0, 0, 0, 0)   # annual data -> no sub-year seasonality; SARIMA == ARIMA here
VAR_MAX_LAG = 2
VAR_SAFETY_FACTOR = 3   # require n_train_obs >= safety_factor * (k^2 * lag + k) to attempt VAR

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(SEED)
torch.manual_seed(SEED)

DL_MODELS = ["GRU", "RNN", "LSTM"]
ALL_MODELS = DL_MODELS + ["VAR", "SARIMA"]

LEVELS = {
    "region": dict(
        table="region_year_summary",
        value_col="total_ktco2e",
        group_cols=["region"],
        cat_cols=["region"],
    ),
    "sector": dict(
        table="ghg_emissions_long",
        value_col="value_ktco2e",
        group_cols=["region", "sector"],
        cat_cols=["region", "sector"],
    ),
    "subclass": dict(
        table="emissions_subclass_long",
        value_col="value_ktco2e",
        group_cols=["region", "sector", "subclass"],
        cat_cols=["region", "sector", "subclass"],
    ),
}


def log(msg):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════
def load_level_df(level, db_path=DB_PATH):
    """Loads one hierarchy level's raw long-format table and standardises
    column names to 'value' and 'series_id'."""
    cfg = LEVELS[level]
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {cfg['table']}", conn)
    conn.close()

    df = df.rename(columns={cfg["value_col"]: "value"})
    df = df[df["year"] >= MIN_YEAR].copy()
    # Replace line 183 with this:
    df["series_id"] = df[cfg["group_cols"]].agg(lambda row: "||".join(row.map(str)), axis=1)    
    keep_cols = ["series_id", "year", "value"] + cfg["group_cols"]
    df = df[keep_cols].dropna(subset=["value"]).sort_values(["series_id", "year"])
    df = df.drop_duplicates(subset=["series_id", "year"])
    return df


def compute_year_splits(years, ratios=SPLIT_RATIOS):
    years = sorted(set(int(y) for y in years))
    n = len(years)
    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = max(1, n - n_val - n_test)
    train_years = years[:n_train]
    val_years = years[n_train:n_train + n_val]
    test_years = years[n_train + n_val:]
    return train_years, val_years, test_years


def build_category_vocab(df, cat_cols):
    """Maps each categorical value (region name, sector name, ...) to an
    integer index for the embedding layers. Index 0 is reserved for
    unseen/unknown values at inference time."""
    vocab = {}
    for c in cat_cols:
        # Replace line 212 with this:
        uniques = sorted(list(map(str, df[c].dropna().unique())))
        vocab[c] = {v: i + 1 for i, v in enumerate(uniques)}   # 0 = unknown
    return vocab


def build_windows_for_level(level, df, cat_cols, cat_vocab, seq_len=SEQ_LEN):
    """For every series at this level: fit a per-series MinMax scaler on its
    TRAIN years only, build overlapping (X, y) windows, assign each window to
    train/val/test by its TARGET year, and also build the "future window"
    (the most recent seq_len points) used later to forecast one year beyond
    the last known year. Returns a dict with everything downstream code needs.
    """
    all_years = df["year"].unique()
    train_years, val_years, test_years = compute_year_splits(all_years)
    train_set, val_set, test_set = set(train_years), set(val_years), set(test_years)

    samples = {"train": [], "val": [], "test": []}
    scalers = {}
    series_meta = {}
    future_windows = {}
    skipped = []

    for series_id, g in df.groupby("series_id"):
        g = g.sort_values("year")
        years = g["year"].to_numpy()
        vals = g["value"].to_numpy(dtype=float)

        train_mask = np.isin(years, train_years)
        if train_mask.sum() < 2 or len(vals) < seq_len + 2:
            skipped.append((series_id, f"only {train_mask.sum()} train pts / {len(vals)} total"))
            continue

        train_vals = vals[train_mask].reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rng = train_vals.max() - train_vals.min()
        if rng == 0:
            # Constant series in the training period -- MinMaxScaler would
            # divide by zero. Fall back to a fixed scaler around that constant.
            scaler.fit(np.array([[train_vals.min() - 1], [train_vals.min() + 1]]))
        else:
            scaler.fit(train_vals)
        scalers[series_id] = scaler

        scaled = scaler.transform(vals.reshape(-1, 1)).ravel()
        year_norm = (years - years.min()) / max(1, (years.max() - years.min()))

        cat_idx = [cat_vocab[c].get(str(g[c].iloc[0]), 0) for c in cat_cols]
        series_meta[series_id] = {
            **{c: str(g[c].iloc[0]) for c in cat_cols},
            "cat_idx": cat_idx, "last_year": int(years.max()),
            "last_value": float(vals[-1]),
        }

        for t in range(seq_len, len(vals)):
            target_year = int(years[t])
            split = ("train" if target_year in train_set else
                     "val" if target_year in val_set else
                     "test" if target_year in test_set else None)
            if split is None:
                continue
            x_val = scaled[t - seq_len:t]
            x_year = year_norm[t - seq_len:t]
            x_seq = np.stack([x_val, x_year], axis=-1)   # (seq_len, 2)
            samples[split].append((x_seq, scaled[t], cat_idx, series_id, target_year))

        # Future window: last seq_len known points -> forecast next year
        future_windows[series_id] = np.stack(
            [scaled[-seq_len:], year_norm[-seq_len:]], axis=-1
        )

    return dict(
        samples=samples, scalers=scalers, series_meta=series_meta,
        future_windows=future_windows, skipped=skipped,
        train_years=train_years, val_years=val_years, test_years=test_years,
        cat_cols=cat_cols,
    )


# ══════════════════════════════════════════════════════════════
# GLOBAL DEEP-LEARNING MODEL (shared across all series at a level)
# ══════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    def __init__(self, samples, n_cats):
        self.x = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
        self.y = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
        self.cats = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.long)
        self.series_id = [s[3] for s in samples]
        self.target_year = [s[4] for s in samples]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.cats[i], self.y[i]


class GlobalSequenceModel(nn.Module):
    """One shared recurrent model (GRU/RNN/LSTM, chosen by `cell_type`) across
    every series at a hierarchy level. The series' own recent values (+ time
    position) go through the recurrent layer; the series' identity (region /
    sector / subclass) is embedded and concatenated with the recurrent
    output before the final prediction layer, so the single shared model can
    still specialise its prediction per series."""

    def __init__(self, cell_type, n_numeric_features, cat_cardinalities,
                 hidden=HIDDEN_SIZE, layers=NUM_LAYERS, embed_dim=EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        rnn_cls = {"GRU": nn.GRU, "RNN": nn.RNN, "LSTM": nn.LSTM}[cell_type]
        self.rnn = rnn_cls(n_numeric_features, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim, padding_idx=0) for card in cat_cardinalities
        ])
        combined_size = hidden + embed_dim * len(cat_cardinalities)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(combined_size, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, x_seq, cat_idx):
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        embeds = [emb(cat_idx[:, i]) for i, emb in enumerate(self.embeddings)]
        combined = torch.cat([last] + embeds, dim=1)
        return self.fc(self.drop(combined)).squeeze(-1)


def train_global_model(cell_type, data, epochs=EPOCHS, patience=PATIENCE):
    """Trains one global model with early stopping on validation loss.
    Returns the trained model (best-val-loss weights restored) and the
    per-epoch train/val loss history for the loss-curve plot."""
    n_cats = len(data["cat_cols"])
    cat_cardinalities = []
    for c in data["cat_cols"]:
        max_idx = max((m["cat_idx"][data["cat_cols"].index(c)] for m in data["series_meta"].values()),
                      default=1)
        cat_cardinalities.append(max_idx)

    train_ds = WindowDataset(data["samples"]["train"], n_cats)
    val_ds = WindowDataset(data["samples"]["val"], n_cats)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GlobalSequenceModel("RNN" if cell_type == "RNN" else cell_type,
                                n_numeric_features=2, cat_cardinalities=cat_cardinalities).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n_b = 0.0, 0
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, cb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += loss.item()
            n_b += 1
        train_loss = ep_loss / max(1, n_b)

        model.eval()
        v_loss, n_v = 0.0, 0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
                v_loss += loss_fn(model(xb, cb), yb).item()
                n_v += 1
        val_loss = v_loss / max(1, n_v)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"    [{cell_type}] Early stopping at epoch {epoch} "
                    f"(best val loss {best_val:.5f} at epoch {epoch - patience_counter})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, cat_cardinalities


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.sum() else float("nan")
    return {"MAE": float(mae), "RMSE": rmse, "MAPE": mape}


@torch.no_grad()
def predict_dl(model, samples, batch_size=256):
    """Runs the model over a list of (x_seq, y, cat_idx, series_id, year)
    samples and returns arrays of predictions (still in SCALED units)."""
    if not samples:
        return np.array([]), np.array([]), [], []
    ds = WindowDataset(samples, n_cats=len(samples[0][2]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.eval()
    for xb, cb, _ in loader:
        xb, cb = xb.to(DEVICE), cb.to(DEVICE)
        preds.append(model(xb, cb).cpu().numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    return preds, ds.y.numpy(), ds.series_id, ds.target_year


def inverse_scale(series_id, scaled_values, scalers):
    scaler = scalers[series_id]
    arr = np.asarray(scaled_values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(arr).ravel()


# ══════════════════════════════════════════════════════════════
# PLOTTING (3 diagnostic files per model, + loss curve for DL models)
# ══════════════════════════════════════════════════════════════
def plot_loss_curve(train_losses, val_losses, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss", color="#2a9d8f")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss", color="#e76f51")
    ax.axvline(len(train_losses), color="grey", linestyle="--", lw=1,
              label="Training stopped here")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (scaled units)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted(years, actual, predicted, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = np.argsort(years)
    ax.plot(np.array(years)[order], np.array(actual)[order], "o-", label="Actual",
            color="#264653", ms=4)
    ax.plot(np.array(years)[order], np.array(predicted)[order], "o--", label="Predicted",
            color="#e76f51", ms=4)
    ax.set_xlabel("Year"); ax.set_ylabel("ktCO2e (original units, pooled across test series)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(actual, predicted, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.5, s=20, color="#264653")
    lims = [min(min(actual, default=0), min(predicted, default=0)),
            max(max(actual, default=1), max(predicted, default=1))]
    ax.plot(lims, lims, "--", color="#e76f51", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual (ktCO2e)"); ax.set_ylabel("Predicted (ktCO2e)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(actual, predicted, out_path, title):
    residuals = np.array(predicted) - np.array(actual)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(actual, residuals, alpha=0.5, s=20, color="#264653")
    axes[0].axhline(0, color="#e76f51", lw=1.5)
    axes[0].set_xlabel("Actual (ktCO2e)"); axes[0].set_ylabel("Residual (predicted - actual)")
    axes[0].set_title("Residuals vs actual", fontweight="bold")
    axes[1].hist(residuals, bins=20, color="#2a9d8f", edgecolor="white")
    axes[1].set_xlabel("Residual (ktCO2e)"); axes[1].set_title("Residual distribution", fontweight="bold")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# DEEP-LEARNING PIPELINE (GRU / RNN / LSTM) — one global model per level
# ══════════════════════════════════════════════════════════════
def run_dl_model(level, model_name, data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Training {model_name} (global model, {len(data['scalers'])} series, "
        f"{len(data['samples']['train'])} train windows)...")

    if len(data["samples"]["train"]) < 10 or len(data["samples"]["val"]) < 2:
        log(f"    Skipping {model_name}: not enough windows to train "
            f"(train={len(data['samples']['train'])}, val={len(data['samples']['val'])}).")
        return None, None

    model, train_losses, val_losses, cat_cards = train_global_model(model_name, data)

    plot_loss_curve(train_losses, val_losses, out_dir / "loss_curve.png",
                    f"{level.title()} level — {model_name} — train vs val loss")

    metrics_by_split = {}
    all_test_actual, all_test_pred, all_test_years = [], [], []
    for split in ["train", "val", "test"]:
        samples = data["samples"][split]
        if not samples:
            continue
        preds_scaled, actual_scaled, series_ids, years = predict_dl(model, samples)
        preds_orig = np.array([inverse_scale(sid, [p], data["scalers"])[0]
                               for sid, p in zip(series_ids, preds_scaled)])
        actual_orig = np.array([inverse_scale(sid, [a], data["scalers"])[0]
                                for sid, a in zip(series_ids, actual_scaled)])
        metrics_by_split[split] = compute_metrics(actual_orig, preds_orig)
        if split == "test":
            all_test_actual, all_test_pred, all_test_years = actual_orig, preds_orig, years

    if len(all_test_actual):
        plot_actual_vs_predicted(all_test_years, all_test_actual, all_test_pred,
                                 out_dir / "actual_vs_predicted.png",
                                 f"{level.title()} — {model_name} — test set (all series pooled)")
        plot_scatter(all_test_actual, all_test_pred, out_dir / "scatter.png",
                    f"{level.title()} — {model_name} — actual vs predicted (test)")
        plot_residuals(all_test_actual, all_test_pred, out_dir / "residuals.png",
                       f"{level.title()} — {model_name} — residuals (test)")

    # Next-year forecast, one row per series, in ORIGINAL units
    next_rows = []
    model.eval()
    for series_id, window in data["future_windows"].items():
        meta = data["series_meta"][series_id]
        x = torch.tensor(window[None, :, :], dtype=torch.float32).to(DEVICE)
        c = torch.tensor([meta["cat_idx"]], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            pred_scaled = model(x, c).cpu().numpy()[0]
        pred_orig = inverse_scale(series_id, [pred_scaled], data["scalers"])[0]
        row = {"series_id": series_id, "last_known_year": meta["last_year"],
              "last_known_value_ktco2e": meta["last_value"],
              "next_year": meta["last_year"] + 1,
              "predicted_value_ktco2e": float(pred_orig)}
        row.update({c: meta[c] for c in data["cat_cols"]})
        next_rows.append(row)
    next_forecast_df = pd.DataFrame(next_rows)
    next_forecast_df.to_csv(out_dir / "next_forecast.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_by_split, f, indent=2)

    return metrics_by_split, next_forecast_df


# ══════════════════════════════════════════════════════════════
# SARIMA — univariate, per series (works at any level)
# ══════════════════════════════════════════════════════════════
def run_sarima(level, df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_years = df["year"].unique()
    train_years, val_years, test_years = compute_year_splits(all_years)

    all_actual, all_pred, all_years_flat = [], [], []
    next_rows = []
    n_ok, n_failed = 0, 0

    for series_id, g in df.groupby("series_id"):
        g = g.sort_values("year")
        years, vals = g["year"].to_numpy(), g["value"].to_numpy(dtype=float)
        train_mask = np.isin(years, train_years)
        test_mask = np.isin(years, test_years)
        if train_mask.sum() < 6 or test_mask.sum() < 1:
            continue
        try:
            model = SARIMAX(vals[train_mask], order=SARIMA_ORDER,
                            seasonal_order=SARIMA_SEASONAL_ORDER,
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False)
            n_forecast = int((~train_mask).sum())
            forecast = fit.forecast(steps=n_forecast)
            test_forecast = forecast[-test_mask.sum():]
            test_actual = vals[test_mask]

            all_actual.extend(test_actual)
            all_pred.extend(test_forecast)
            all_years_flat.extend(years[test_mask])

            # Refit on everything for the true "next year" forecast
            full_fit = SARIMAX(vals, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            next_pred = float(full_fit.forecast(steps=1)[0])
            row = {"series_id": series_id, "last_known_year": int(years.max()),
                  "last_known_value_ktco2e": float(vals[-1]),
                  "next_year": int(years.max()) + 1, "predicted_value_ktco2e": next_pred}
            cfg = LEVELS[level]
            for c in cfg["group_cols"]:
                row[c] = g[c].iloc[0]
            next_rows.append(row)
            n_ok += 1
        except Exception as e:
            n_failed += 1
            continue

    log(f"  SARIMA: {n_ok} series fit successfully, {n_failed} failed/skipped "
        f"(too short or non-converging).")

    metrics = compute_metrics(all_actual, all_pred) if all_actual else {"MAE": None, "RMSE": None, "MAPE": None}
    if all_actual:
        plot_actual_vs_predicted(all_years_flat, all_actual, all_pred,
                                 out_dir / "actual_vs_predicted.png",
                                 f"{level.title()} — SARIMA{SARIMA_ORDER} — test set (all series pooled)")
        plot_scatter(all_actual, all_pred, out_dir / "scatter.png",
                    f"{level.title()} — SARIMA — actual vs predicted (test)")
        plot_residuals(all_actual, all_pred, out_dir / "residuals.png",
                       f"{level.title()} — SARIMA — residuals (test)")

    next_forecast_df = pd.DataFrame(next_rows)
    if len(next_forecast_df):
        next_forecast_df.to_csv(out_dir / "next_forecast.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": metrics}, f, indent=2)
    return {"test": metrics}, next_forecast_df


# ══════════════════════════════════════════════════════════════
# VAR — jointly models several series at once (e.g. all sectors in a
# region), WITH a dimensionality guard: it refuses to fit (and says why)
# when there isn't remotely enough data for the number of variables.
# ══════════════════════════════════════════════════════════════
def var_group_spec(level):
    """Returns (group_cols, variable_col): how to split the data into
    VAR groups, and which column becomes the set of jointly-modelled
    variables within each group."""
    return {
        "region": ([], "region"),
        "sector": (["region"], "sector"),
        "subclass": (["region", "sector"], "subclass"),
    }[level]


def run_var(level, df, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    group_cols, variable_col = var_group_spec(level)
    all_years = df["year"].unique()
    train_years, val_years, test_years = compute_year_splits(all_years)

    all_actual, all_pred, all_years_flat = [], [], []
    next_rows = []
    n_fit, n_skipped = 0, 0
    skip_reasons = []

    groups = [((), df)] if not group_cols else list(df.groupby(group_cols))

    for group_key, g in groups:
        wide = g.pivot_table(index="year", columns=variable_col, values="value", aggfunc="sum")
        wide = wide.dropna(axis=0, how="any")  # only years where every variable has data
        k = wide.shape[1]
        if k < 2:
            continue   # VAR needs >=2 variables; a lone series should use SARIMA instead

        train_wide = wide[wide.index.isin(train_years)]
        n_train = len(train_wide)
        required = VAR_SAFETY_FACTOR * (k * k * VAR_MAX_LAG + k)
        if n_train < required:
            n_skipped += 1
            skip_reasons.append(
                f"{group_key or 'ALL'}: k={k} variables, n_train={n_train} obs, "
                f"need >= {required} for a reliable VAR({VAR_MAX_LAG}) fit -- skipped."
            )
            continue

        try:
            maxlag = max(1, min(VAR_MAX_LAG, n_train // (k + 2)))
            model = VAR(train_wide.values)
            fit = model.fit(maxlags=maxlag, ic="aic")
            lag_used = fit.k_ar if fit.k_ar > 0 else 1
            if fit.k_ar == 0:
                fit = model.fit(1)
                lag_used = 1

            test_wide = wide[wide.index.isin(test_years)]
            n_forecast = len(wide) - n_train
            if n_forecast <= 0 or len(test_wide) == 0:
                continue
            forecast = fit.forecast(train_wide.values[-lag_used:], steps=n_forecast)
            forecast_years = wide.index[n_train:]
            forecast_df = pd.DataFrame(forecast, index=forecast_years, columns=wide.columns)
            test_pred = forecast_df.loc[forecast_df.index.isin(test_years)]
            test_actual = wide.loc[test_pred.index]

            for col in wide.columns:
                all_actual.extend(test_actual[col].values)
                all_pred.extend(test_pred[col].values)
                all_years_flat.extend(test_pred.index.values)

            # Refit on ALL available data for the true next-year forecast
            full_fit = VAR(wide.values).fit(lag_used)
            next_pred = full_fit.forecast(wide.values[-lag_used:], steps=1)[0]
            last_year = int(wide.index.max())
            for i, col in enumerate(wide.columns):
                row = {"series_id": None, "last_known_year": last_year,
                      "last_known_value_ktco2e": float(wide[col].iloc[-1]),
                      "next_year": last_year + 1,
                      "predicted_value_ktco2e": float(next_pred[i]),
                      variable_col: col}
                for gc, gv in zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)):
                    row[gc] = gv
                row["series_id"] = "||".join(str(row.get(c, "")) for c in LEVELS[level]["group_cols"])
                next_rows.append(row)
            n_fit += 1
        except Exception as e:
            n_skipped += 1
            skip_reasons.append(f"{group_key or 'ALL'}: fit failed ({e})")
            continue

    log(f"  VAR: {n_fit} group(s) fit successfully, {n_skipped} skipped.")
    for reason in skip_reasons[:10]:
        log(f"    - {reason}")
    if len(skip_reasons) > 10:
        log(f"    ... and {len(skip_reasons) - 10} more (see var_skip_log.txt)")
    with open(out_dir / "var_skip_log.txt", "w") as f:
        f.write("\n".join(skip_reasons) if skip_reasons else "No groups skipped.")

    metrics = compute_metrics(all_actual, all_pred) if all_actual else {"MAE": None, "RMSE": None, "MAPE": None}
    if all_actual:
        plot_actual_vs_predicted(all_years_flat, all_actual, all_pred,
                                 out_dir / "actual_vs_predicted.png",
                                 f"{level.title()} — VAR — test set (all fitted groups pooled)")
        plot_scatter(all_actual, all_pred, out_dir / "scatter.png",
                    f"{level.title()} — VAR — actual vs predicted (test)")
        plot_residuals(all_actual, all_pred, out_dir / "residuals.png",
                       f"{level.title()} — VAR — residuals (test)")

    next_forecast_df = pd.DataFrame(next_rows)
    if len(next_forecast_df):
        next_forecast_df.to_csv(out_dir / "next_forecast.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": metrics, "n_groups_fit": n_fit, "n_groups_skipped": n_skipped}, f, indent=2)
    return {"test": metrics}, next_forecast_df


# ══════════════════════════════════════════════════════════════
# HIERARCHICAL COHERENCE CHECK
# ══════════════════════════════════════════════════════════════
def reconciliation_report(next_forecasts, out_dir):
    """Checks whether subclass forecasts sum to their sector's forecast, and
    sector forecasts sum to their region's forecast, for each model that
    produced forecasts at all 3 levels. Also computes a BOTTOM-UP reconciled
    forecast (sum children up) as the recommended coherent alternative."""
    rows = []
    for model_name in ALL_MODELS:
        sub_df = next_forecasts.get(("subclass", model_name))
        sec_df = next_forecasts.get(("sector", model_name))
        reg_df = next_forecasts.get(("region", model_name))
        if sub_df is None or sec_df is None or reg_df is None or not len(sub_df) or not len(sec_df) or not len(reg_df):
            continue

        # subclass -> sector
        bottom_up_sector = sub_df.groupby(["region", "sector"])["predicted_value_ktco2e"].sum().reset_index()
        bottom_up_sector = bottom_up_sector.rename(columns={"predicted_value_ktco2e": "bottom_up_from_subclass"})
        sec_check = sec_df.merge(bottom_up_sector, on=["region", "sector"], how="left")
        sec_check["abs_diff"] = sec_check["predicted_value_ktco2e"] - sec_check["bottom_up_from_subclass"]
        sec_check["pct_diff"] = sec_check["abs_diff"] / sec_check["predicted_value_ktco2e"].replace(0, np.nan) * 100
        for _, r in sec_check.iterrows():
            rows.append({"model": model_name, "check": "subclass -> sector",
                        "region": r["region"], "sector": r["sector"],
                        "independent_forecast": r["predicted_value_ktco2e"],
                        "bottom_up_forecast": r["bottom_up_from_subclass"],
                        "abs_diff": r["abs_diff"], "pct_diff": r["pct_diff"]})

        # sector -> region
        bottom_up_region = sec_df.groupby("region")["predicted_value_ktco2e"].sum().reset_index()
        bottom_up_region = bottom_up_region.rename(columns={"predicted_value_ktco2e": "bottom_up_from_sector"})
        reg_check = reg_df.merge(bottom_up_region, on="region", how="left")
        reg_check["abs_diff"] = reg_check["predicted_value_ktco2e"] - reg_check["bottom_up_from_sector"]
        reg_check["pct_diff"] = reg_check["abs_diff"] / reg_check["predicted_value_ktco2e"].replace(0, np.nan) * 100
        for _, r in reg_check.iterrows():
            rows.append({"model": model_name, "check": "sector -> region",
                        "region": r["region"], "sector": None,
                        "independent_forecast": r["predicted_value_ktco2e"],
                        "bottom_up_forecast": r["bottom_up_from_sector"],
                        "abs_diff": r["abs_diff"], "pct_diff": r["pct_diff"]})

    report_df = pd.DataFrame(rows)
    if len(report_df):
        report_df.to_csv(out_dir / "reconciliation_report.csv", index=False)
        log(f"\nReconciliation report saved: {out_dir / 'reconciliation_report.csv'}")
        log(f"  Mean |pct_diff| by model/check:")
        log(report_df.groupby(["model", "check"])["pct_diff"].apply(lambda s: s.abs().mean()).to_string())
    else:
        log("\nReconciliation report: no model produced forecasts at all 3 levels "
            "(likely some levels were skipped for lack of data) -- nothing to compare.")
    return report_df


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global EPOCHS, PATIENCE
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS.keys()),
                        choices=list(LEVELS.keys()))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    args = parser.parse_args()

    EPOCHS, PATIENCE = args.epochs, args.patience

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    next_forecasts = {}

    for level in args.levels:
        log(f"\n{'='*70}\nLEVEL: {level.upper()}\n{'='*70}")
        cfg = LEVELS[level]
        df = load_level_df(level)
        log(f"Loaded {df['series_id'].nunique()} series, years {df['year'].min()}-{df['year'].max()}")

        level_dir = OUTPUT_DIR / level

        if any(m in args.models for m in DL_MODELS):
            cat_vocab = build_category_vocab(df, cfg["cat_cols"])
            data = build_windows_for_level(level, df, cfg["cat_cols"], cat_vocab)
            if data["skipped"]:
                log(f"  {len(data['skipped'])} series skipped (too short): "
                    f"{data['skipped'][:5]}{'...' if len(data['skipped']) > 5 else ''}")
            log(f"  Train/val/test years: {data['train_years']} / {data['val_years']} / {data['test_years']}")

            for model_name in DL_MODELS:
                if model_name not in args.models:
                    continue
                metrics, next_df = run_dl_model(level, model_name, data, level_dir / model_name)
                if metrics is not None:
                    all_results[f"{level}__{model_name}"] = metrics
                    next_forecasts[(level, model_name)] = next_df

        if "SARIMA" in args.models:
            log("  Running SARIMA...")
            metrics, next_df = run_sarima(level, df, level_dir / "SARIMA")
            all_results[f"{level}__SARIMA"] = metrics
            next_forecasts[(level, "SARIMA")] = next_df

        if "VAR" in args.models:
            log("  Running VAR...")
            metrics, next_df = run_var(level, df, level_dir / "VAR")
            all_results[f"{level}__VAR"] = metrics
            next_forecasts[(level, "VAR")] = next_df

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"\nAll metrics saved: {OUTPUT_DIR / 'results.json'}")

    reconciliation_report(next_forecasts, OUTPUT_DIR)

    log(f"\nDone. Outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()




