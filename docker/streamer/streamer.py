# streamer/streamer.py
# Incremental Fingrid fetch — runs every FETCH_INTERVAL_MIN minutes.
# Identical retry/pagination logic to fetch_fingrid_datasets.py.
# Fetches only rows newer than the latest stored timestamp per dataset.
#
# ── CHANGES IN THIS VERSION ──────────────────────────────────
# NEW: after every raw fetch cycle, incrementally rebuilds the LAST
# HOURLY_MERGE_LOOKBACK_HOURS of the `fingrid_hourly_merged` table
# (the table streamlit_app.py actually reads). Previously the
# streamer only ever wrote to the raw per-source 3-min tables --
# fingrid_hourly_merged was only ever populated by manually running
# build_hourly_aggregates.py, so the dashboard's merged-data views
# went stale the moment nobody ran that script by hand.
#
# This does NOT replace build_hourly_aggregates.py -- you still need
# to run that once (or whenever you want a full historical rebuild)
# to seed the table with your full history. This addition only keeps
# the last few days rolling-fresh automatically on every cycle, using
# INSERT OR REPLACE keyed on timestamp_utc so it never touches rows
# outside the lookback window.
# ─────────────────────────────────────────────────────────────

import os
import time
import logging
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

API_KEY            = os.environ["FINGRID_API_KEY"]
BASE_URL           = "https://data.fingrid.fi/api"
DB_PATH            = os.getenv("DB_PATH", "/data/project_data.db")
FETCH_INTERVAL_MIN = int(os.getenv("FETCH_INTERVAL_MIN", "15"))
LOOKBACK_MIN       = int(os.getenv("LOOKBACK_MIN", "60"))
MAX_RETRIES        = 5
RETRY_BASE_S       = 5

# NEW: how many hours back to recompute in fingrid_hourly_merged on
# every cycle. 72h (3 days) comfortably covers FETCH_INTERVAL_MIN gaps,
# retries, and any late-arriving corrected readings, without having to
# re-aggregate your full multi-year history every 15 minutes.
HOURLY_MERGE_LOOKBACK_HOURS = int(os.getenv("HOURLY_MERGE_LOOKBACK_HOURS", "72"))

DATASETS = {
    "co2_intensity":             266,
    "consumption_co2_intensity": 265,
    "chp_district":              201,
    "chp_industrial":            202,
    "hydro":                     191,
    "nuclear":                   188,
    "wind":                      181,
    "total_production":          192,
    "consumption_electricity":   193,
}

# NEW: same source definitions as build_hourly_aggregates.py, needed
# here so the incremental hourly merge uses identical aggregation
# logic (mean vs. simple+weighted intensity average).
SOURCE_AGG_TYPES = {
    "co2_intensity":             "intensity",
    "consumption_co2_intensity": "intensity",
    "chp_district":              "mean",
    "chp_industrial":            "mean",
    "hydro":                     "mean",
    "nuclear":                   "mean",
    "wind":                      "mean",
    "total_production":          "mean",
    "consumption_electricity":   "mean",
}
SOURCE_LABELS = {
    "co2_intensity":             "co2_intensity",
    "consumption_co2_intensity": "consumption_co2_intens",
    "chp_district":              "chp_district_mw",
    "chp_industrial":            "chp_industrial_mw",
    "hydro":                     "hydro_mw",
    "nuclear":                   "nuclear_mw",
    "wind":                      "wind_mw",
    "total_production":          "total_production_mw",
    "consumption_electricity":   "consumption_elec_mw",
}
INTENSITY_WEIGHTS = {
    "co2_intensity":             "total_production",
    "consumption_co2_intensity": "consumption_electricity",
}


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_table(name):
    conn = get_conn()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS "{name}" (
            start_time TEXT PRIMARY KEY,
            end_time   TEXT NOT NULL,
            value      REAL,
            fetched_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_latest(name) -> datetime:
    conn = get_conn()
    row = conn.execute(
        f'SELECT MAX(start_time) FROM "{name}"'
    ).fetchone()
    conn.close()
    if row and row[0]:
        dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        return dt - timedelta(minutes=LOOKBACK_MIN)
    return datetime(2018, 1, 1, tzinfo=timezone.utc)


def fetch_page(dataset_id, name, params):
    headers = {"x-api-key": API_KEY}
    wait = RETRY_BASE_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(f"{BASE_URL}/datasets/{dataset_id}/data",
                             headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                log.warning("[%s] 429 attempt %d — waiting %ds", name, attempt, wait)
                time.sleep(wait); wait *= 2; continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning("[%s] Error attempt %d: %s", name, attempt, e)
            time.sleep(wait); wait *= 2
    return None


def fetch_window(dataset_id, name, start, end):
    all_rows, page = [], 1
    while True:
        params = {
            "startTime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime":   end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json", "oneRowPerTimePeriod": False,
            "page": page, "pageSize": 10000, "locale": "en",
        }
        data = fetch_page(dataset_id, name, params)
        if not data: break
        rows = data.get("data", [])
        if not rows: break
        all_rows.extend(rows)
        if len(rows) < 10000: break
        page += 1
        time.sleep(2.5)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"startTime": "start_time", "endTime": "end_time"})
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["end_time"]   = pd.to_datetime(df["end_time"],   utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df["value"]      = pd.to_numeric(df["value"], errors="coerce")
    return df[["start_time", "end_time", "value"]]


def insert(name, df):
    if df.empty: return
    now = datetime.now(timezone.utc).isoformat()
    rows = [(r["start_time"], r["end_time"], r["value"], now)
            for _, r in df.iterrows()]
    conn = get_conn()
    conn.executemany(
        f'INSERT OR REPLACE INTO "{name}" '
        f'(start_time, end_time, value, fetched_at) VALUES (?,?,?,?)',
        rows
    )
    conn.commit()
    conn.close()
    log.info("[%s] Upserted %d rows", name, len(rows))


def run_cycle():
    now = datetime.now(timezone.utc)
    log.info("=== Fetch cycle %s ===", now.strftime("%Y-%m-%d %H:%M UTC"))
    for name, did in DATASETS.items():
        ensure_table(name)
        start = get_latest(name)
        df = fetch_window(did, name, start, now)
        insert(name, df)
    log.info("=== Cycle complete ===")

    # NEW: keep fingrid_hourly_merged rolling-fresh for the last
    # HOURLY_MERGE_LOOKBACK_HOURS. This is what fixes the dashboard
    # showing stale data -- previously nothing ever rebuilt this table
    # automatically.
    try:
        update_hourly_merged(now)
    except Exception as e:
        log.error("Hourly merge update failed: %s", e)


# ════════════════════════════════════════════════════════════
# NEW: INCREMENTAL HOURLY MERGE
# Same aggregation logic as build_hourly_aggregates.py, but only
# over the last HOURLY_MERGE_LOOKBACK_HOURS, and upserted via
# INSERT OR REPLACE keyed on timestamp_utc instead of replacing the
# whole table. Run build_hourly_aggregates.py once by hand to seed
# full history -- this only keeps recent hours fresh afterwards.
# ════════════════════════════════════════════════════════════

def ensure_hourly_table(conn):
    """Creates fingrid_hourly_merged with timestamp_utc as PRIMARY KEY
    if it doesn't already exist, so INSERT OR REPLACE can upsert by
    hour without duplicating rows. If the table already exists from a
    prior build_hourly_aggregates.py run (which uses to_sql 'replace'
    and has no explicit primary key), this CREATE TABLE IF NOT EXISTS
    is a no-op and the existing table/rows are left untouched."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fingrid_hourly_merged (
            timestamp_utc TEXT PRIMARY KEY,
            year INTEGER, month INTEGER, day INTEGER, hour INTEGER
        )
    """)
    conn.commit()


def load_recent(conn, table_name, since: datetime) -> pd.DataFrame:
    try:
        df = pd.read_sql(
            f'SELECT start_time, value FROM "{table_name}" '
            f'WHERE start_time >= ? ORDER BY start_time',
            conn, params=[since.strftime("%Y-%m-%dT%H:%M:%SZ")]
        )
    except Exception as e:
        log.warning("[hourly-merge] Could not load '%s': %s", table_name, e)
        return pd.DataFrame(columns=["start_time", "value"])

    if df.empty:
        return df
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df


def add_time_columns(df):
    utc = df["start_time"].dt.tz_convert("UTC")
    df["year"], df["month"], df["day"], df["hour"] = (
        utc.dt.year, utc.dt.month, utc.dt.day, utc.dt.hour
    )
    return df


def aggregate_mean(df, label):
    if df.empty:
        return pd.DataFrame()
    df = add_time_columns(df)
    agg = (df.groupby(["year", "month", "day", "hour"])["value"]
             .agg(mean_val="mean", count_val="count")
             .reset_index())
    return agg.rename(columns={"mean_val": label, "count_val": f"{label}_count"})


def aggregate_intensity(df_intensity, df_weight, label):
    if df_intensity.empty:
        return pd.DataFrame()
    df_intensity = add_time_columns(df_intensity)

    simple = (df_intensity.groupby(["year", "month", "day", "hour"])["value"]
                          .agg(simple_avg="mean", count_val="count")
                          .reset_index()
                          .rename(columns={"simple_avg": f"{label}_simple_avg",
                                           "count_val":  f"{label}_count"}))

    if df_weight is None or df_weight.empty:
        simple[f"{label}_weighted_avg"] = np.nan
        return simple

    df_weight = add_time_columns(df_weight.copy())
    df_intensity = df_intensity.rename(columns={"value": "intensity"})
    df_weight    = df_weight.rename(columns={"value": "weight"})[
        ["start_time", "weight"]
    ]

    # Align on nearest 3-min bucket (both grids share the same minute
    # bucket after floor to hour, so a direct hour-level join is
    # sufficient here -- matches the notebook/EDA logic).
    merged = df_intensity.merge(
        df_weight, on="start_time", how="left"
    )
    merged["intensity_x_weight"] = merged["intensity"] * merged["weight"]

    weighted = (merged.groupby(["year", "month", "day", "hour"])
                       .apply(lambda g: pd.Series({
                           "weighted_sum": g["intensity_x_weight"].sum(skipna=True),
                           "weight_sum":   g["weight"].sum(skipna=True),
                       }), include_groups=False)
                       .reset_index())
    weighted[f"{label}_weighted_avg"] = np.where(
        weighted["weight_sum"] > 0,
        weighted["weighted_sum"] / weighted["weight_sum"],
        np.nan,
    )
    weighted = weighted[["year", "month", "day", "hour", f"{label}_weighted_avg"]]

    return simple.merge(weighted, on=["year", "month", "day", "hour"], how="left")


def update_hourly_merged(now: datetime):
    since = now - timedelta(hours=HOURLY_MERGE_LOOKBACK_HOURS)
    log.info("[hourly-merge] Rebuilding last %dh (since %s)",
             HOURLY_MERGE_LOOKBACK_HOURS, since.strftime("%Y-%m-%d %H:%M UTC"))

    conn = get_conn()
    ensure_hourly_table(conn)

    raw = {name: load_recent(conn, name, since) for name in DATASETS}

    frames = []
    for name, agg_type in SOURCE_AGG_TYPES.items():
        label = SOURCE_LABELS[name]
        if agg_type == "mean":
            agg = aggregate_mean(raw[name], label)
        else:
            weight_name = INTENSITY_WEIGHTS[name]
            agg = aggregate_intensity(raw[name], raw.get(weight_name), label)
        if not agg.empty:
            frames.append(agg)

    if not frames:
        log.info("[hourly-merge] No recent rows to merge")
        conn.close()
        return

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["year", "month", "day", "hour"], how="outer")
    merged = merged.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)

    merged["timestamp_utc"] = (
        pd.to_datetime(merged[["year", "month", "day", "hour"]]
                       .assign(minute=0, second=0))
        .dt.tz_localize("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Ensure every column this row could supply exists in the table
    # (ALTER TABLE ADD COLUMN is a no-op-safe way to keep schema in
    # sync if a source/column was added after the table was created).
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(fingrid_hourly_merged)")}
    for col in merged.columns:
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE fingrid_hourly_merged ADD COLUMN "{col}" REAL')
    conn.commit()

    cols = list(merged.columns)
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(f'"{c}"' for c in cols)
    rows = [tuple(r) for r in merged[cols].itertuples(index=False, name=None)]

    conn.executemany(
        f'INSERT OR REPLACE INTO fingrid_hourly_merged ({col_list}) VALUES ({placeholders})',
        rows
    )
    conn.commit()
    conn.close()

    log.info("[hourly-merge] Upserted %d hourly rows (%s -> %s)",
             len(merged), merged["timestamp_utc"].min(), merged["timestamp_utc"].max())


if __name__ == "__main__":
    import os
    os.makedirs("/data", exist_ok=True)
    log.info("Streamer starting — interval %d min", FETCH_INTERVAL_MIN)

    # NEW: create fingrid_hourly_merged immediately, before the first
    # fetch cycle runs. On a first-ever run, run_cycle() has to pull
    # full history (since 2018) for 9 datasets, which can take a while.
    # Previously the table was only created at the END of that first
    # cycle (inside update_hourly_merged), so if the dashboard queried
    # the DB anytime during that window, it got:
    #   "no such table: fingrid_hourly_merged"
    # Creating an (empty) table up front means the dashboard's
    # `SELECT * FROM fingrid_hourly_merged` succeeds immediately —
    # it'll just return 0 rows until the first cycle finishes and
    # actually populates it.
    _conn = get_conn()
    ensure_hourly_table(_conn)
    _conn.close()

    run_cycle()
    while True:
        time.sleep(FETCH_INTERVAL_MIN * 60)
        try:
            run_cycle()
        except Exception as e:
            log.error("Cycle failed: %s — retrying next interval", e)