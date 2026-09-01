# api/main.py
# ─────────────────────────────────────────────────────────────
# FastAPI service that serves CO2 intensity forecasts.
# Feature engineering here EXACTLY matches fingrid_15min_pipeline.ipynb:
#   - Same SOURCES, same PHYSICAL_BOUNDS
#   - Same clean_source_3min() logic
#   - Same add_bucket_cols(), aggregate_mw_source(), aggregate_intensity_source()
#   - Same feature columns: hour_sin/cos, dow_sin/cos, month_sin/cos,
#     is_weekend, time_index, diff1/diff4 rate-of-change, lags [1,4,96,672],
#     rolling [4,96], renewable_share, net_load, fossil_share
#   - Same EXCLUDE list (counts, flags, intermediates)
#   - Same model: GRU hidden=64 layers=2 dropout=0.2 output=2
#   - Same SEQ_LEN=80, HORIZON=1
#
# ── CHANGES IN THIS VERSION (synced to notebook + train_15min.py) ──
# 1. ROLL_COLS FIX: restored missing comma between chp_industrial_mw_mean
#    and nuclear_mw_mean. Previously these silently concatenated into
#    one bad string, so nuclear_mw_mean never got rolling features --
#    same bug you found and fixed in the EDA notebook.
# 2. TIME_INDEX FIX: now computed as (index - index.min()) / (index.max()
#    - index.min()), matching the notebook's current Stage 7 formula
#    exactly. This replaces the old fixed 2018-2026 absolute-range
#    version (t_ref/t_end globals removed). See the inline comment in
#    build_features_from_db() for a note on what this means for
#    consistency across different API call window sizes.
# 3. RATE-OF-CHANGE FEATURES ADDED: co2_intensity_simple_avg_diff1/diff4
#    and consumption_co2_intens_simple_avg_diff1/diff4, matching the
#    notebook. These flow into FEATURE_COLS automatically via
#    column_config.json, no EXCLUDE-list change needed.
# 4. EFFECTIVE FEATURE CONFIG: startup now prefers
#    column_config_effective.json (written by train_15min.py's
#    --drop-features ablation) over column_config.json if present, so
#    the served feature list always matches what the loaded scaler/
#    model were actually fit on.
# 5. BIAS CORRECTION: loads bias_corrections.json (written by
#    train_15min.py's compute_bias_correction, fit on the VALIDATION
#    set) if present, and subtracts the GRU correction from the FINAL
#    forecast values only -- never fed back into the autoregressive
#    lag-1 loop, to avoid pushing the sequence off the distribution
#    the model was trained on.
# ─────────────────────────────────────────────────────────────

import gc
import os
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config (from environment variables) ───────────────────────
DB_PATH    = os.getenv("DB_PATH",    "/data/project_data.db")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/gru_model.pt")
SCALER_X   = os.getenv("SCALER_X",  "/data/scaler_X.pkl")
SCALER_Y   = os.getenv("SCALER_Y",  "/data/scaler_y.pkl")
COL_CFG    = os.getenv("COL_CFG",   "/data/column_config.json")
SEQ_LEN    = int(os.getenv("SEQ_LEN", "80"))

# ── Constants that MUST match fingrid_15min_pipeline.ipynb ─────
ZSCORE_WINDOW = 480
ZSCORE_THRESH = 4.0
FILL_LIMIT    = 5

PHYSICAL_BOUNDS = {
    "co2_intensity":              (0,   500),
    "consumption_co2_intensity":  (0,   600),
    "wind":                       (0,  8000),
    "nuclear":                    (0,  5000),
    "hydro":                      (0,  3500),
    "chp_district":               (0,  5000),
    "chp_industrial":             (0,  3000),
    "total_production":           (0, 20000),
    "consumption_electricity":    (0, 20000),
}

SOURCES = [
    ("co2_intensity",             "co2_intensity",          "intensity"),
    ("consumption_co2_intensity", "consumption_co2_intens", "intensity"),
    ("chp_district",              "chp_district_mw",        "mean"),
    ("chp_industrial",            "chp_industrial_mw",      "mean"),
    ("hydro",                     "hydro_mw",               "mean"),
    ("nuclear",                   "nuclear_mw",             "mean"),
    ("wind",                      "wind_mw",                "mean"),
    ("total_production",          "total_production_mw",    "mean"),
    ("consumption_electricity",   "consumption_elec_mw",    "mean"),
]

INTENSITY_WEIGHTS = {
    "co2_intensity":             "total_production",
    "consumption_co2_intensity": "consumption_electricity",
}

TARGET_COLS = [
    "co2_intensity_simple_avg",
    "consumption_co2_intens_simple_avg",
]

KEY_COLS = ["timestamp_utc", "year", "month", "day", "hour", "minute_bucket"]

LAG_STEPS    = [1, 4, 96, 672]
ROLL_WINDOWS = [4, 96]
ROLL_COLS = ["co2_intensity_simple_avg","consumption_co2_intens_simple_avg","wind_mw_mean", "hydro_mw_mean","chp_district_mw_mean","chp_industrial_mw_mean",
             "nuclear_mw_mean","total_production_mw_mean","consumption_elec_mw_mean"]

# ── Model definition (MUST match training notebook) ─────────────
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2,
                 dropout=0.2, out=2):
        super().__init__()
        self.gru  = nn.GRU(input_size, hidden, layers,
                           batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, out)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(self.drop(o[:, -1, :]))


# ── App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="CarbonPulse Finland — Forecast API",
    description=(
        "Real-time CO2 intensity forecasting for Finland's electricity grid. "
        "Loads cleaned 15-min data from SQLite, applies the same preprocessing "
        "as the training pipeline, and returns GRU predictions."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals loaded at startup ────────────────────────────────────
model            = None
scaler_X         = None
scaler_y         = None
feat_cols        = None
bias_correction  = None   # NEW: per-target array, subtracted from final output only


@app.on_event("startup")
def load_model():
    global model, scaler_X, scaler_y, feat_cols, bias_correction

    scaler_X = joblib.load(SCALER_X)
    scaler_y = joblib.load(SCALER_Y)

    import json

    # NEW: prefer column_config_effective.json if it exists next to
    # COL_CFG -- this is written by train_15min.py's --drop-features
    # ablation and reflects the ACTUAL feature list the currently
    # loaded scaler/model were fit on. Falling back to plain
    # column_config.json keeps this working for runs that didn't use
    # --drop-features. Using the wrong list here is exactly the class
    # of bug that caused the original scaler mismatch, so this is
    # intentionally explicit rather than silently assumed.
    effective_cfg_path = Path(COL_CFG).with_name("column_config_effective.json")
    if effective_cfg_path.exists():
        with open(effective_cfg_path) as f:
            cfg = json.load(f)
        log.info("Loaded EFFECTIVE feature config: %s", effective_cfg_path)
    else:
        with open(COL_CFG) as f:
            cfg = json.load(f)
        log.info("Loaded feature config: %s", COL_CFG)

    feat_cols = cfg["FEATURE_COLS"]

    n_features = len(feat_cols)
    model = GRUModel(input_size=n_features)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    # NEW: optional bias correction from train_15min.py's Round 2
    # changes (compute_bias_correction, fit on the VALIDATION set).
    # Applied ONLY to the final reported values -- never fed back into
    # the autoregressive lag features, since the model's internal
    # dynamics were trained on raw (uncorrected) lag inputs and mixing
    # corrected values into the feedback loop would shift it off the
    # distribution the model actually learned.
    bias_path = os.getenv("BIAS_CORRECTIONS",
                          str(Path(MODEL_PATH).with_name("bias_corrections.json")))
    if Path(bias_path).exists():
        with open(bias_path) as f:
            bc = json.load(f)
        bias_correction = np.array(bc["GRU"], dtype=np.float32)
        log.info("Loaded bias correction (GRU): %s", bias_correction.tolist())
    else:
        bias_correction = None
        log.info("No bias_corrections.json found at %s -- serving uncorrected predictions", bias_path)

    log.info("Model loaded. n_features=%d  SEQ_LEN=%d", n_features, SEQ_LEN)


# ── Data loading and preprocessing ──────────────────────────────

def load_raw_source(table_name: str, n_rows: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            f'SELECT start_time, value FROM "{table_name}" '
            f'ORDER BY start_time DESC LIMIT {n_rows}',
            conn
        )
    finally:
        conn.close()
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = (df.drop_duplicates(subset=["start_time"])
            .sort_values("start_time")
            .reset_index(drop=True))
    return df


def clean_source_3min(table_name: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Identical to clean_source_3min() in the pipeline notebook."""
    df = df_raw.rename(columns={"value": table_name}).set_index("start_time").copy()
    lo, hi = PHYSICAL_BOUNDS[table_name]

    df.loc[df[table_name] < lo, table_name] = np.nan
    df.loc[df[table_name] > hi, table_name] = np.nan

    roll_mean = (df[table_name]
                 .rolling(ZSCORE_WINDOW, center=True,
                          min_periods=int(ZSCORE_WINDOW * 0.5))
                 .mean())
    roll_std = (df[table_name]
                .rolling(ZSCORE_WINDOW, center=True,
                         min_periods=int(ZSCORE_WINDOW * 0.5))
                .std()
                .replace(0, np.nan))
    z    = (df[table_name] - roll_mean).abs() / roll_std
    mask = z > ZSCORE_THRESH
    df.loc[mask, table_name] = np.nan

    filled = df[table_name].ffill(limit=FILL_LIMIT)
    df[table_name] = filled
    return df


def add_bucket_cols(df: pd.DataFrame) -> pd.DataFrame:
    bucket             = df.index.tz_convert("UTC").floor("15min")
    df["year"]         = bucket.year
    df["month"]        = bucket.month
    df["day"]          = bucket.day
    df["hour"]         = bucket.hour
    df["minute_bucket"]= bucket.minute
    df["timestamp_utc"]= bucket.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df


def aggregate_mw_source(df: pd.DataFrame, col: str,
                         label: str) -> pd.DataFrame:
    df = add_bucket_cols(df.copy())
    agg = (df.groupby(KEY_COLS)[col]
             .agg(mean_val="mean", count_val="count")
             .reset_index())
    return agg.rename(columns={"mean_val":  f"{label}_mean",
                                "count_val": f"{label}_count"})


def aggregate_intensity_source(df_int: pd.DataFrame, col_int: str,
                                label: str,
                                df_weight: pd.DataFrame,
                                col_weight: str) -> pd.DataFrame:
    df_int = add_bucket_cols(df_int.copy())

    simple = (df_int.groupby(KEY_COLS)[col_int]
                    .agg(simple_avg="mean", count_val="count")
                    .reset_index()
                    .rename(columns={"simple_avg": f"{label}_simple_avg",
                                     "count_val":  f"{label}_count"}))

    if df_weight is None or df_weight.empty:
        simple[f"{label}_weighted_avg"] = np.nan
        return simple

    df_int["b3"] = df_int.index.tz_convert("UTC").floor("3min")
    df_w = df_weight[[col_weight]].copy()
    df_w["b3"] = df_weight.index.tz_convert("UTC").floor("3min")

    merged = df_int[[col_int, "b3"] + KEY_COLS].reset_index().merge(
        df_w.reset_index()[["b3", col_weight]], on="b3", how="left"
    )
    merged["ix_w"] = merged[col_int] * merged[col_weight]

    w_agg = (merged.groupby(KEY_COLS)
                   .apply(lambda g: pd.Series({
                       "ws":  g["ix_w"].sum(skipna=True),
                       "ws2": g[col_weight].sum(skipna=True),
                   }), include_groups=False)
                   .reset_index())
    w_agg[f"{label}_weighted_avg"] = np.where(
        w_agg["ws2"] > 0, w_agg["ws"] / w_agg["ws2"], np.nan
    )
    return simple.merge(
        w_agg[KEY_COLS + [f"{label}_weighted_avg"]], on=KEY_COLS, how="left"
    )


def build_features_from_db(n_raw_rows: int = 15000) -> pd.DataFrame:
    """
    Load, clean, aggregate, and feature-engineer the latest rows
    from the SQLite DB. Mirrors the pipeline notebook exactly.
    n_raw_rows: how many 3-min rows to load per source (covers
                SEQ_LEN=80 × 15min plus lag-672 warmup).
    """
    # Step 1: load and clean each source
    raw_data = {}
    cleaned  = {}
    for table_name, label, agg_type in SOURCES:
        df_raw = load_raw_source(table_name, n_raw_rows)
        raw_data[table_name] = df_raw
        cleaned[table_name]  = clean_source_3min(table_name, df_raw)

    # Step 2: aggregate to 15-min
    frames = []
    for table_name, label, agg_type in SOURCES:
        df_c = cleaned[table_name]
        if agg_type == "mean":
            agg = aggregate_mw_source(df_c, table_name, label)
        else:
            wt     = INTENSITY_WEIGHTS.get(table_name)
            df_wt  = cleaned.get(wt)
            col_wt = wt if df_wt is not None else None
            agg = aggregate_intensity_source(df_c, table_name, label,
                                              df_wt, col_wt)
        frames.append(agg)

    # Step 3: inner join (same as pipeline)
    df_merged = frames[0]
    for f in frames[1:]:
        df_merged = df_merged.merge(f, on=KEY_COLS, how="inner")
    df_merged = df_merged.sort_values(KEY_COLS).reset_index(drop=True)

    # Drop count columns not needed for features
    drop_cols = [c for c in df_merged.columns
                 if c.endswith("_sum") or c.endswith("_mwh")]
    df_merged = df_merged.drop(columns=drop_cols, errors="ignore")

    # Step 4: feature engineering (mirrors Stage 7 of pipeline notebook)
    df = df_merged.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc").sort_index()

    new_cols = {}
    new_cols["hour_sin"]   = np.sin(2 * np.pi * df["hour"]              / 24)
    new_cols["hour_cos"]   = np.cos(2 * np.pi * df["hour"]              / 24)
    new_cols["dow_sin"]    = np.sin(2 * np.pi * df.index.dayofweek      / 7)
    new_cols["dow_cos"]    = np.cos(2 * np.pi * df.index.dayofweek      / 7)
    new_cols["month_sin"]  = np.sin(2 * np.pi * df["month"]             / 12)
    new_cols["month_cos"]  = np.cos(2 * np.pi * df["month"]             / 12)
    new_cols["is_weekend"] = (df.index.dayofweek >= 5).astype(np.float32)

    # NEW / CORRECTED: time_index now matches the notebook's Stage 7
    # formula exactly -- relative to THIS window's own min/max, not a
    # fixed 2018-2026 absolute range. This also fixes the earlier
    # pandas datetime-resolution bug for good, since it never divides
    # by a hardcoded 1e9 (ns) assumption -- Timestamp arithmetic here
    # is resolution-agnostic by construction, same principle as the
    # debug_pipeline.py fix, but now using the notebook's own relative
    # convention instead of the fixed-epoch one.
    #
    # NOTE: because this call only loads the latest n_raw_rows (roughly
    # the last few weeks), df.index.min()/max() here span a MUCH
    # shorter window than training's full multi-year history. That
    # means time_index's scale/meaning differs between training and
    # inference -- it's a relative "position within loaded data" here,
    # not "position within all of 2018-2026" per se. This matches what
    # the notebook currently computes, so it's implemented as-is, but
    # it's worth knowing: if predictions from this feature look
    # unstable across different --n_raw_rows / API window sizes, this
    # is why, and increasing n_raw to cover a fixed longer span (e.g.
    # always exactly 1 year) would make it consistent between calls.
    new_cols["time_index"] = (
        (df.index - df.index.min()) / (df.index.max() - df.index.min())
    ).astype(np.float32)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # NEW: rate-of-change features (added in the notebook to give the
    # model explicit "how fast is this moving right now" signal,
    # addressing the lag/over-smoothing behaviour seen in early
    # actual-vs-predicted plots). diff1 = change over last 15 min,
    # diff4 = change over last 1 hour. Must run BEFORE lag/rolling
    # feature blocks since column ordering doesn't matter for those,
    # but this keeps it identical to notebook cell order.
    for col in ["co2_intensity_simple_avg", "consumption_co2_intens_simple_avg"]:
        df[f"{col}_diff1"] = df[col].diff(1)
        df[f"{col}_diff4"] = df[col].diff(4)

    # Lag features
    MEAN_COLS = [c for c in df.columns if c.endswith("_mean")]
    INT_COLS  = [c for c in df.columns
                 if c.endswith("_simple_avg") or c.endswith("_weighted_avg")]
    SIG_COLS  = MEAN_COLS + INT_COLS

    lag_frames = {}
    for col in SIG_COLS:
        if col in df.columns:
            for lag in LAG_STEPS:
                lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
    df = pd.concat([df, pd.DataFrame(lag_frames, index=df.index)], axis=1)

    # Rolling stats
    roll_frames = {}
    for col in ROLL_COLS:
        if col not in df.columns:
            continue
        for w in ROLL_WINDOWS:
            mp = max(1, int(w * 0.5))
            roll_frames[f"{col}_rmean{w}"] = df[col].rolling(w, min_periods=mp).mean()
            roll_frames[f"{col}_rstd{w}"]  = df[col].rolling(w, min_periods=mp).std()
    df = pd.concat([df, pd.DataFrame(roll_frames, index=df.index)], axis=1)

    # Physics features
    phys = {}
    if all(c in df.columns for c in
           ["wind_mw_mean", "nuclear_mw_mean", "hydro_mw_mean"]):
        phys["renewable_mw"] = (df["wind_mw_mean"]
                                + df["nuclear_mw_mean"]
                                + df["hydro_mw_mean"])
        if "total_production_mw_mean" in df.columns:
            phys["renewable_share"] = (phys["renewable_mw"]
                                       / (df["total_production_mw_mean"] + 1e-6))
            phys["net_load"]        = (df["total_production_mw_mean"]
                                       - phys["renewable_mw"])
    if all(c in df.columns for c in
           ["chp_district_mw_mean", "chp_industrial_mw_mean"]):
        phys["fossil_chp_mw"] = (df["chp_district_mw_mean"]
                                  + df["chp_industrial_mw_mean"])
        if "total_production_mw_mean" in df.columns:
            phys["fossil_share"] = (phys["fossil_chp_mw"]
                                    / (df["total_production_mw_mean"] + 1e-6))
    df = pd.concat([df, pd.DataFrame(phys, index=df.index)], axis=1)

    return df


def get_seed_window(df_feat: pd.DataFrame) -> np.ndarray:
    """
    Extract the last SEQ_LEN rows of feature columns,
    matching the exact EXCLUDE logic from the pipeline notebook.
    """
    TARGET_COLS_WEIGHTED = [
        "co2_intensity_weighted_avg",
        "consumption_co2_intens_weighted_avg",
    ]
    EXCLUDE = (
        ["year", "month", "day", "hour", "minute_bucket",
         "renewable_mw", "fossil_chp_mw"]
        + TARGET_COLS
        + TARGET_COLS_WEIGHTED
        + [c for c in df_feat.columns if c.endswith("_count")]
        + [c for c in df_feat.columns if c.endswith("_flag")]
    )
    # Use the feature columns from training (loaded from column_config.json)
    available = [c for c in feat_cols if c in df_feat.columns]

    df_clean = df_feat[available].dropna()
    if len(df_clean) < SEQ_LEN:
        raise ValueError(
            f"Not enough clean rows after feature engineering: "
            f"need {SEQ_LEN}, got {len(df_clean)}. "
            f"Try increasing n_raw_rows."
        )

    return df_clean[available].values[-SEQ_LEN:]


# ── Response models ──────────────────────────────────────────────

class ForecastPoint(BaseModel):
    timestamp:                        str
    co2_intensity_gco2kwh:            float
    consumption_co2_intensity_gco2kwh: float


class ForecastResponse(BaseModel):
    generated_at:          str
    last_known_timestamp:  str
    horizon_steps:         int
    horizon_hours:         float
    resolution_minutes:    int
    forecasts:             list[ForecastPoint]


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "n_features":   len(feat_cols) if feat_cols else 0,
        "seq_len":      SEQ_LEN,
        "bias_correction_active": bias_correction is not None,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }


@app.get("/current")
def current():
    """Return the most recent co2_intensity reading from the DB."""
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        'SELECT start_time, value FROM co2_intensity '
        'ORDER BY start_time DESC LIMIT 1'
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="No data available")
    return {
        "timestamp":           row[0],
        "co2_intensity_gco2kwh": row[1],
        "unit":                "gCO2/kWh",
        "source":              "Fingrid dataset 266 (production side)",
    }


@app.get("/forecast", response_model=ForecastResponse)
def forecast(steps: int = 96):
    """
    Generate autoregressive forecast for the next `steps` 15-min periods.
    Default: 96 steps = 24 hours. Max: 192 steps = 48 hours.

    Pipeline:
      1. Load latest ~15,000 3-min rows per source from SQLite
      2. Clean each source (physical bounds + rolling z-score + ffill)
      3. Aggregate to 15-min (same method as training)
      4. Feature engineer (same columns as training)
      5. Scale with training scalers
      6. Autoregressive GRU prediction (each step feeds back as lag-1 input)
    """
    steps = min(max(steps, 1), 192)

    # Load enough rows to cover SEQ_LEN + lag-672 warmup
    # lag-672 = 7 days at 15-min = 672 × 5 = 3360 raw 3-min rows minimum
    # Add buffer: 4000 + SEQ_LEN × 5 = enough for full warmup
    n_raw = max(15000, (SEQ_LEN + 700) * 5)

    try:
        df_feat = build_features_from_db(n_raw_rows=n_raw)
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"Feature engineering failed: {e}")

    try:
        seed = get_seed_window(df_feat)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    last_ts = df_feat.index[-1]

    # Scale seed using training scaler
    seed_scaled = scaler_X.transform(seed)   # (SEQ_LEN, n_features)
    window      = list(seed_scaled)

    # Find lag-1 column indices for autoregressive feedback
    co2_lag1_idx  = next(
        (i for i, c in enumerate(feat_cols)
         if "co2_intensity_simple_avg_lag1" in c), None
    )
    cons_lag1_idx = next(
        (i for i, c in enumerate(feat_cols)
         if "consumption_co2_intens_simple_avg_lag1" in c), None
    )

    predictions = []
    with torch.no_grad():
        for _ in range(steps):
            x    = torch.FloatTensor(np.array(window)).unsqueeze(0)
            pred = model(x).numpy()[0]    # shape (2,)
            predictions.append(pred)

            # Feed this prediction back as the lag-1 feature for next step
            next_row = window[-1].copy()
            if co2_lag1_idx is not None:
                next_row[co2_lag1_idx]  = pred[0]
            if cons_lag1_idx is not None:
                next_row[cons_lag1_idx] = pred[1]
            window.pop(0)
            window.append(next_row)

    pred_arr  = np.array(predictions)                       # (steps, 2)
    pred_real = scaler_y.inverse_transform(pred_arr)        # back to gCO2/kWh

    # NEW: apply bias correction (from train_15min.py, fit on VAL set)
    # to the FINAL reported values only. The autoregressive loop above
    # already ran to completion using raw (uncorrected) predictions fed
    # back as lag-1 -- that's intentional, since the model's internal
    # dynamics were trained on raw lag inputs, and feeding corrected
    # values back in would push the sequence off-distribution.
    if bias_correction is not None:
        pred_real = pred_real - bias_correction[np.newaxis, :]

    future_times = pd.date_range(
        start=last_ts + pd.Timedelta(minutes=15),
        periods=steps, freq="15min", tz="UTC"
    )

    return ForecastResponse(
        generated_at=datetime.now(timezone.utc).isoformat(),
        last_known_timestamp=str(last_ts),
        horizon_steps=steps,
        horizon_hours=round(steps * 15 / 60, 2),
        resolution_minutes=15,
        forecasts=[
            ForecastPoint(
                timestamp=t.isoformat(),
                co2_intensity_gco2kwh=round(float(p[0]), 3),
                consumption_co2_intensity_gco2kwh=round(float(p[1]), 3),
            )
            for t, p in zip(future_times, pred_real)
        ],
    )
