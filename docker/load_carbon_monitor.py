#!/usr/bin/env python3
# load_carbon_monitor.py
# ─────────────────────────────────────────────────────────────
# One-time (or re-run-when-updated) loader: reads your local Carbon
# Monitor Excel file and writes it into the `carbon_monitor_finland`
# table in project_data.db, which is what streamlit_app.py's Page 3
# ("Monitoring Multiscale Carbon Emission") reads via load_carbon_monitor().
#
# WHY THIS EXISTS: streamlit_app.py expects a table with (at least)
# columns: date, sector, and a value column whose name contains "Mt"
# or "value". Your data currently only exists as a local Excel file,
# so Page 3 shows "Carbon Monitor data not found in the database."
#
# ── I DON'T KNOW YOUR EXACT FILE STRUCTURE ──────────────────────
# I haven't seen the actual Excel file, so this script:
#   1. Reads it and PRINTS a preview (columns, dtypes, first rows)
#      before writing anything, so you can sanity-check the mapping.
#   2. Tries to auto-detect the standard Carbon Monitor EU column
#      names (they typically ship as: date, country, sector,
#      MtCO2/day or similar). Common variants are handled below.
#   3. Filters to Finland only if a country column exists (Carbon
#      Monitor EU files are usually multi-country).
#
# If the auto-detected mapping is wrong, the printed preview will
# make it obvious -- edit COLUMN_MAP below to match your actual
# column names and re-run. Paste me the preview output if you'd
# rather I fix the mapping directly.
#
# Usage:
#   python load_carbon_monitor.py --xlsx "C:\...\data\your_file.xlsx"
#   python load_carbon_monitor.py --xlsx "..." --sheet "Sheet1"
#   python load_carbon_monitor.py --xlsx "..." --db project_data.db
# ─────────────────────────────────────────────────────────────

import argparse
import sqlite3
import sys

import pandas as pd


# EDIT THIS if auto-detection below doesn't match your file.
# Left side = what your Excel column is actually called.
# Right side = what streamlit_app.py expects.
COLUMN_MAP_CANDIDATES = {
    "date":            ["date", "Date", "DATE", "day"],
    "country":         ["country", "Country", "COUNTRY", "region"],
    "sector":          ["sector", "Sector", "SECTOR"],
    "value_mtco2":     ["MtCO2", "MtCO2/day", "value", "Value", "emission",
                         "emissions", "MtCO2 per day", "co2_mt"],
}


def detect_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", required=True, help="Path to the Carbon Monitor Excel file")
    p.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
    p.add_argument("--db", default="data/project_data.db", help="Path to project_data.db")
    p.add_argument("--country-filter", default="Finland",
                   help="Value to filter the country column on, if one exists "
                        "(set to '' to disable filtering)")
    p.add_argument("--dry-run", action="store_true",
                   help="Only print the preview, don't write to the database")
    args = p.parse_args()

    print("=" * 60)
    print("LOADING CARBON MONITOR EXCEL FILE")
    print(f"  File:  {args.xlsx}")
    print(f"  Sheet: {args.sheet}")
    print("=" * 60)

    try:
        raw = pd.read_excel(args.xlsx, sheet_name=args.sheet)
    except Exception as e:
        print(f"[ERROR] Could not read Excel file: {e}")
        sys.exit(1)

    if isinstance(raw, dict):
        # sheet_name resolved to multiple sheets -- list them and exit
        print("\n[INFO] Multiple sheets found. Re-run with --sheet <name>:")
        for name, sheet_df in raw.items():
            print(f"  '{name}'  shape={sheet_df.shape}  columns={list(sheet_df.columns)[:8]}")
        sys.exit(0)

    print(f"\nLoaded {len(raw):,} rows, {len(raw.columns)} columns.")
    print("\nColumns found:")
    for c in raw.columns:
        print(f"  {c!r}  (dtype={raw[c].dtype})")

    print("\nFirst 5 rows:")
    print(raw.head().to_string())

    # ── Detect columns ──────────────────────────────────────────
    date_col    = detect_column(raw, COLUMN_MAP_CANDIDATES["date"])
    country_col = detect_column(raw, COLUMN_MAP_CANDIDATES["country"])
    sector_col  = detect_column(raw, COLUMN_MAP_CANDIDATES["sector"])
    value_col   = detect_column(raw, COLUMN_MAP_CANDIDATES["value_mtco2"])

    print("\n" + "=" * 60)
    print("AUTO-DETECTED MAPPING")
    print("=" * 60)
    print(f"  date column:    {date_col!r}")
    print(f"  country column: {country_col!r}")
    print(f"  sector column:  {sector_col!r}")
    print(f"  value column:   {value_col!r}")

    if date_col is None or sector_col is None or value_col is None:
        print("\n[ERROR] Could not auto-detect required columns (date/sector/value).")
        print("Edit COLUMN_MAP_CANDIDATES at the top of this script to match your")
        print("actual column names (see 'Columns found' list above), then re-run.")
        sys.exit(1)

    df = raw.copy()

    # ── Filter to Finland if a country column exists ────────────
    if country_col and args.country_filter:
        before = len(df)
        matches = df[country_col].astype(str).str.contains(
            args.country_filter, case=False, na=False
        )
        df = df[matches]
        print(f"\n[INFO] Filtered to country contains '{args.country_filter}': "
              f"{before:,} -> {len(df):,} rows")
        if df.empty:
            print(f"[WARN] No rows matched '{args.country_filter}' in column "
                  f"'{country_col}'. Unique values found:")
            print(raw[country_col].unique()[:20])
            sys.exit(1)
    elif country_col is None:
        print("\n[INFO] No country column detected -- assuming file is Finland-only already.")

    # ── Build final table matching streamlit_app.py's expectations ──
    out = pd.DataFrame({
        "date":   pd.to_datetime(df[date_col], utc=True, dayfirst=True),
        "sector": df[sector_col].astype(str),
        "MtCO2":  pd.to_numeric(df[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["MtCO2"]).sort_values("date").reset_index(drop=True)

    print(f"\nFinal table: {len(out):,} rows")
    print(f"Date range: {out['date'].min()} -> {out['date'].max()}")
    print(f"Sectors found: {sorted(out['sector'].unique().tolist())}")
    print("\nPreview:")
    print(out.head(10).to_string())

    if args.dry_run:
        print("\n[DRY RUN] Not writing to database. Remove --dry-run to actually load.")
        return

    # ── Write to DB ──────────────────────────────────────────────
    print(f"\nWriting to {args.db} -> table 'carbon_monitor_finland' ...")
    conn = sqlite3.connect(args.db)
    out_for_sql = out.copy()
    out_for_sql["date"] = out_for_sql["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_for_sql.to_sql("carbon_monitor_finland", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cm_date ON carbon_monitor_finland (date)"
    )
    conn.commit()
    conn.close()

    print("Done. Restart/refresh the Streamlit app -- Page 3 should now show data.")


if __name__ == "__main__":
    main()
