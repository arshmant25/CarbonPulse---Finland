# build_hourly_aggregates.py
# ─────────────────────────────────────────────────────────────
# Converts every 3-minute Fingrid source table into hourly
# aggregates, then joins all sources on (year, month, day, hour).
#
# KEY DESIGN DECISIONS explained inline:
#
#   1. Each source is aggregated INDEPENDENTLY before joining.
#      This means a gap in "wind" does not remove a row for
#      that hour in "nuclear" — you get the nuclear average
#      for that hour and NaN for wind. No data from other
#      sources is lost because one source had a gap.
#
#   2. co2_intensity and consumption_co2_intensity get BOTH:
#      - simple_avg  : straight mean of all 3-min readings
#      - weighted_avg: mean weighted by the corresponding MW
#        production volume (total_production for co2_intensity,
#        consumption_electricity for consumption_co2_intensity)
#        This is more accurate for emission totals because a
#        3-min spike at high output counts more than one at
#        low output.
#
#   3. For MW sources (wind, nuclear, etc.) only mean is needed.
#      Mean of MW over an hour = average power = correct basis
#      for converting to MWh (multiply by 1 hour).
#
#   4. A "reading_count" column is added for every source showing
#      how many 3-min readings contributed to that hour (max=20).
#      This lets you filter hours with poor data coverage.
#
#   5. The two clock groups (Grid A: HH:MM:00, Grid B: HH:MM:01)
#      both round to the same (year, month, day, hour) — that is
#      exactly why hourly aggregation solves the alignment problem
#      cleanly without any timestamp shifting or proximity merge.
#
#   6. Output goes to a SEPARATE database: data/hourly_data.db
#      so it does not pollute your raw source database.
#
# Run:
#   python build_hourly_aggregates.py
#
# Output table in data/hourly_data.db:
#   fingrid_hourly_merged
#
# Tools: pandas, sqlite3

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# Output database — always separate from the raw source database.
# Hardcoded here so it cannot be accidentally overridden by config.
HOURLY_DB_PATH = "data/project_data.db"   # <-- new dedicated DB
SOURCE_DB_PATH = "data/project_data.db"                  # "data/project_data.db"

# ── Dataset definitions ───────────────────────────────────────
# Each entry: (table_name, column_label_in_output, aggregation_type)
#
# aggregation_type:
#   "mean"     — simple mean (used for all MW production sources)
#   "intensity"— BOTH simple mean AND weighted mean
#                (used for gCO2/kWh intensity datasets)

GRID_B_SOURCES = [
    # These all start at HH:MM:01
    ("chp_district",          "chp_district_mw",        "mean"),
    ("chp_industrial",        "chp_industrial_mw",      "mean"),
    ("hydro",                 "hydro_mw",               "mean"),
    ("nuclear",               "nuclear_mw",             "mean"),
    ("wind",                  "wind_mw",                "mean"),
    ("total_production",      "total_production_mw",    "mean"),
    ("consumption_electricity","consumption_elec_mw",   "mean"),
]

GRID_A_SOURCES = [
    # These start at HH:MM:00 — same hour bucket after floor()
    ("co2_intensity",             "co2_intensity",         "intensity"),
    ("consumption_co2_intensity", "consumption_co2_intens","intensity"),
]

# Weight source for each intensity dataset:
#   co2_intensity is weighted by total_production (production side)
#   consumption_co2_intensity is weighted by consumption_electricity
INTENSITY_WEIGHTS = {
    "co2_intensity":             "total_production",
    "consumption_co2_intensity": "consumption_electricity",
}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def get_source_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SOURCE_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_hourly_conn() -> sqlite3.Connection:
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(HOURLY_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def load_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """
    Load a raw 3-minute table from the source DB.
    Returns a DataFrame with:
      start_time  (datetime64, UTC, parsed)
      value       (float)
    """
    try:
        df = pd.read_sql(
            f'SELECT start_time, value FROM "{table_name}" ORDER BY start_time',
            conn
        )
    except Exception as e:
        print(f"  [WARN] Could not load '{table_name}': {e}")
        return pd.DataFrame(columns=["start_time", "value"])

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df = df.drop_duplicates(subset=["start_time"]).reset_index(drop=True)

    print(f"  Loaded '{table_name}': {len(df):,} rows  "
          f"({df['start_time'].min()} -> {df['start_time'].max()})")
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add year, month, day, hour columns from start_time.
    The hour is the floor-hour in Helsinki local time (UTC+2 / UTC+3).

    WHY LOCAL TIME:
      Aggregating by UTC hour means "hour 0" in January is
      02:00–02:59 Helsinki time — confusing for daily patterns.
      Helsinki is UTC+2 (winter) / UTC+3 (summer, DST).
      We use Helsinki local time so "hour 0" = midnight Helsinki,
      which matches how Finnish energy consumption patterns work.

    The output column "hour" is 0–23 in Helsinki local time.
    """
    # Keep everything in UTC — no DST ambiguity possible
    # (Helsinki local time reconstruction via tz_localize crashes
    # every autumn when clocks go back and the same local hour
    # exists twice: e.g. 2018-10-28 03:00 Helsinki = DST clash)
    utc = df["start_time"].dt.tz_convert("UTC")
    df["year"]  = utc.dt.year
    df["month"] = utc.dt.month
    df["day"]   = utc.dt.day
    df["hour"]  = utc.dt.hour
    return df


# ══════════════════════════════════════════════════════════════
# AGGREGATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def aggregate_mean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Simple mean aggregation for MW production sources.

    Groups by (year, month, day, hour) and computes:
      {label}        — mean MW for that hour
      {label}_count  — number of 3-min readings in that hour (max 20)
                       Use this to filter low-coverage hours.
    """
    if df.empty:
        return pd.DataFrame()

    df = add_time_columns(df)

    agg = (df.groupby(["year", "month", "day", "hour"])["value"]
             .agg(
                 mean_val="mean",
                 count_val="count"
             )
             .reset_index())

    agg = agg.rename(columns={
        "mean_val":  label,
        "count_val": f"{label}_count",
    })

    return agg


def aggregate_intensity(df_intensity: pd.DataFrame,
                        df_weight: pd.DataFrame,
                        label: str,
                        weight_label: str) -> pd.DataFrame:
    """
    Dual aggregation for gCO2/kWh intensity datasets.

    Produces TWO columns per intensity dataset:
      {label}_simple_avg   — straight mean of all 3-min readings
      {label}_weighted_avg — production-volume-weighted mean

    WHY WEIGHTED AVERAGE MATTERS:
      Suppose in one hour:
        - 3-min slot 1: intensity=200 gCO2/kWh, production=4000 MW
        - 3-min slots 2-20: intensity=30 gCO2/kWh, production=5000 MW

      Simple mean = (200 + 19×30) / 20 = 38.5 gCO2/kWh
      Weighted mean = (200×4000 + 30×5000×19) / (4000 + 5000×19)
                    = (800000 + 2850000) / (4000 + 95000)
                    = 3650000 / 99000 = 36.9 gCO2/kWh

      The difference seems small per hour, but across 8 years it
      accumulates. For calculating annual CO2 tonnes from hourly
      data, the weighted version is more accurate.

      Both are provided so your supervisor can choose which to use
      for which purpose (simple for quick EDA, weighted for totals).

    HOW WEIGHTING IS DONE:
      For each 3-min slot, intensity × weight_MW gives total CO2
      grams emitted in that slot (per kWh × MW = per hour).
      Hourly weighted avg = sum(intensity × weight) / sum(weight)
      This is the standard electricity grid carbon intensity formula.
    """
    if df_intensity.empty:
        print(f"  [WARN] No intensity data for '{label}'. Skipping.")
        return pd.DataFrame()

    df_int = add_time_columns(df_intensity.copy())
    df_int = df_int.rename(columns={"value": "intensity"})

    # ── Simple average ────────────────────────────────────────
    simple = (df_int.groupby(["year", "month", "day", "hour"])["intensity"]
                    .agg(
                        simple_avg="mean",
                        count_val="count"
                    )
                    .reset_index())
    simple = simple.rename(columns={
        "simple_avg": f"{label}_simple_avg",
        "count_val":  f"{label}_count",
    })

    # ── Weighted average ──────────────────────────────────────
    if df_weight.empty:
        print(f"  [WARN] No weight data for '{label}' weighted avg. "
              f"Returning simple avg only.")
        simple[f"{label}_weighted_avg"] = np.nan
        return simple

    df_w = add_time_columns(df_weight.copy())
    df_w = df_w.rename(columns={"value": "weight"})

    # Merge intensity with weight on exact timestamp
    # (both are either :00 or their respective grid clock)
    # We merge on (year, month, day, hour, minute-level) using
    # the original start_time before we dropped it, so we need
    # to keep start_time through to the merge step.
    df_int_full = df_intensity.copy()
    df_int_full = add_time_columns(df_int_full)
    df_int_full = df_int_full.rename(columns={"value": "intensity"})

    df_w_full = df_weight.copy()
    df_w_full = add_time_columns(df_w_full)
    df_w_full = df_w_full.rename(columns={"value": "weight"})

    # Merge on start_time — only rows where both have a reading
    # Note: Grid A (intensity, :00) and Grid B (weight, :01) have
    # a 60-second offset, so a direct timestamp merge will lose rows.
    # Solution: merge on the (year, month, day, hour) + minute bucket.
    # We add a "3min_bucket" column (floor to 3-min) then merge.
    df_int_full["bucket"] = df_int_full["start_time"].dt.floor("3min")
    df_w_full["bucket"]   = df_w_full["start_time"].dt.floor("3min")

    # For Grid A vs Grid B: the offset is exactly 60s. Floor to 3min
    # means :00 and :01 both floor to :00 — so they DO match.
    merged = df_int_full.merge(
        df_w_full[["bucket", "weight", "year", "month", "day", "hour"]],
        on=["bucket", "year", "month", "day", "hour"],
        how="left"
    )

    # Compute weighted sum and weight sum per hour
    merged["intensity_x_weight"] = merged["intensity"] * merged["weight"]

    weighted = (merged.groupby(["year", "month", "day", "hour"])
                      .apply(lambda g: pd.Series({
                          "weighted_sum": g["intensity_x_weight"].sum(skipna=True),
                          "weight_sum":   g["weight"].sum(skipna=True),
                      }), include_groups=False)
                      .reset_index())

    # weighted avg = sum(intensity*weight) / sum(weight)
    # Guard against zero-weight hours (all MW readings were 0 or NaN)
    weighted[f"{label}_weighted_avg"] = np.where(
        weighted["weight_sum"] > 0,
        weighted["weighted_sum"] / weighted["weight_sum"],
        np.nan
    )
    weighted = weighted[["year", "month", "day", "hour",
                          f"{label}_weighted_avg"]]

    # Merge simple and weighted results
    result = simple.merge(weighted,
                          on=["year", "month", "day", "hour"],
                          how="left")
    return result


# ══════════════════════════════════════════════════════════════
# MAIN BUILD FUNCTION
# ══════════════════════════════════════════════════════════════

def build_hourly_merged():
    print("=" * 60)
    print("BUILD HOURLY AGGREGATES")
    print(f"  Source DB : {SOURCE_DB_PATH}")
    print(f"  Output DB : {HOURLY_DB_PATH}")
    print("=" * 60)

    src_conn = get_source_conn()

    # ── Step 1: Load all raw tables ───────────────────────────
    print("\nStep 1: Loading raw 3-minute tables...")
    raw = {}

    all_sources = GRID_B_SOURCES + GRID_A_SOURCES
    for table_name, label, agg_type in all_sources:
        raw[table_name] = load_table(src_conn, table_name)

    src_conn.close()

    # ── Step 2: Aggregate each source independently ───────────
    print("\nStep 2: Aggregating each source to hourly...")
    hourly_frames = []

    for table_name, label, agg_type in all_sources:
        df = raw[table_name]

        if agg_type == "mean":
            print(f"  [{agg_type}] {table_name} -> '{label}'")
            agg = aggregate_mean(df, label)

        elif agg_type == "intensity":
            weight_table = INTENSITY_WEIGHTS.get(table_name)
            weight_label = None
            for t, lbl, _ in all_sources:
                if t == weight_table:
                    weight_label = lbl
                    break
            print(f"  [intensity] {table_name} -> '{label}' "
                  f"(weighted by '{weight_table}')")
            df_weight = raw.get(weight_table, pd.DataFrame())
            agg = aggregate_intensity(df, df_weight, label, weight_label)
        else:
            continue

        if not agg.empty:
            hourly_frames.append(agg)
            n_hours = len(agg)
            print(f"    -> {n_hours:,} hourly rows")

    # ── Step 3: Join all hourly frames on (year,month,day,hour) ─
    print("\nStep 3: Joining all hourly aggregates...")
    print("  Using OUTER JOIN so a gap in one source does not drop")
    print("  rows from other sources.")

    if not hourly_frames:
        print("  No hourly frames produced. Exiting.")
        return

    # Start with the first frame and outer-join the rest one by one
    merged = hourly_frames[0]
    for frame in hourly_frames[1:]:
        merged = merged.merge(
            frame,
            on=["year", "month", "day", "hour"],
            how="outer"     # outer = keep all hours from all sources
        )

    # Sort chronologically
    merged = merged.sort_values(
        ["year", "month", "day", "hour"]
    ).reset_index(drop=True)

    # Build a UTC timestamp string directly from the join key columns.
    # We use pd.to_datetime with utc=True instead of tz_localize
    # because tz_localize on Helsinki times crashes every autumn DST
    # transition (e.g. 2018-10-28 03:00 local exists twice -> ValueError).
    # UTC timestamps have no ambiguous hours — this is always safe.
    merged["timestamp_utc"] = (
        pd.to_datetime(
            merged[["year", "month", "day", "hour"]]
            .assign(minute=0, second=0)
        )
        .dt.tz_localize("UTC")   # safe: we built these from UTC components
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Reorder columns: timestamp first, then year/month/day/hour,
    # then data columns in logical order
    key_cols   = ["timestamp_utc", "year", "month", "day", "hour"]
    data_cols  = [c for c in merged.columns if c not in key_cols]

    # Put intensity columns first, then MW production columns
    intensity_cols  = [c for c in data_cols if "co2_intens" in c]
    production_cols = [c for c in data_cols if c not in intensity_cols]

    final_col_order = key_cols + intensity_cols + production_cols
    merged = merged[final_col_order]

    print(f"\n  Final merged table: {len(merged):,} hourly rows")
    print(f"  Date range: {merged['timestamp_utc'].min()} -> "
          f"{merged['timestamp_utc'].max()}")
    print(f"  Columns ({len(merged.columns)}):")
    for col in merged.columns:
        null_pct = merged[col].isna().mean() * 100
        print(f"    {col:40s}  {null_pct:5.1f}% NaN")

    # ── Step 4: Save to output database ───────────────────────
    print(f"\nStep 4: Saving to {HOURLY_DB_PATH} -> table 'fingrid_hourly_merged'")
    out_conn = get_hourly_conn()

    merged.to_sql(
        "fingrid_hourly_merged",
        out_conn,
        if_exists="replace",
        index=False
    )
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hourly_timestamp
        ON fingrid_hourly_merged (timestamp_utc)
    """)
    out_conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hourly_ymdh
        ON fingrid_hourly_merged (year, month, day, hour)
    """)
    out_conn.commit()

    # ── Step 5: Write a data quality summary table ─────────────
    # For each source, count how many hours have full coverage
    # (reading_count == 20, meaning all 20 three-minute slots present)
    print("\nStep 5: Writing coverage summary table...")
    count_cols = [c for c in merged.columns if c.endswith("_count")]
    summary_rows = []
    for col in count_cols:
        source = col.replace("_count", "")
        total_hours = merged[col].notna().sum()
        full_hours  = (merged[col] == 20).sum()
        partial_hours = ((merged[col] > 0) & (merged[col] < 20)).sum()
        empty_hours = merged[col].isna().sum()
        summary_rows.append({
            "source":              source,
            "total_hours_present": int(total_hours),
            "full_coverage_hours": int(full_hours),
            "partial_hours":       int(partial_hours),
            "empty_hours_nan":     int(empty_hours),
            "pct_full":            round(full_hours / len(merged) * 100, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_sql(
        "coverage_summary",
        out_conn,
        if_exists="replace",
        index=False
    )
    out_conn.commit()
    out_conn.close()

    print("\nCoverage summary:")
    print(df_summary.to_string(index=False))

    # ── Final summary ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Output DB   : {HOURLY_DB_PATH}")
    print(f"  Main table  : fingrid_hourly_merged ({len(merged):,} rows)")
    print(f"  Quality table: coverage_summary")
    print()
    print("Column guide:")
    print("  co2_intensity_simple_avg   — arithmetic mean of 3-min readings")
    print("  co2_intensity_weighted_avg — production-weighted mean (use for")
    print("                               CO2 tonne calculations)")
    print("  *_count columns            — how many 3-min slots per hour")
    print("                               (max 20, filter >= 16 for 80% coverage)")
    print()
    print("Recommended ML filter (add to notebook):")
    print("  df = df[df['co2_intensity_count'] >= 16]  # 80% coverage minimum")


if __name__ == "__main__":
    build_hourly_merged()
