"""
CarbonPulse Finland — Regional Consolidation Script
====================================================
Reads every downloaded region .xlsx file from paastot.hiilineutraalisuomi.fi
(the "GHG emissions" tab) and stacks them into ONE tidy master dataset.

USAGE:
    1. Put all 19 downloaded .xlsx files into RAW_DIR (any filenames, doesn't matter).
    2. Run: python 01_consolidate_regions.py
    3. Outputs land in OUT_DIR as CSVs, AND get written into the shared
       SQLite database at DB_PATH (tables: ghg_emissions_long, region_year_summary)
       so the rest of the pipeline (and any other notebooks) can just query the DB.

Works with 1 file today and scales automatically to 19 — no code changes needed
as you add more files to RAW_DIR.
"""

import re
import glob
import os
import sqlite3
import pandas as pd

RAW_DIR = os.path.join("region_data", "raw")
OUT_DIR = os.path.join("region_data", "processed")
DB_PATH = os.path.join("data", "project_data.db")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

SECTOR_ROWS = [
    "Electricity", "Electric heating", "District heating", "Oil heating",
    "Other heating", "Industry", "Machinery", "Road transport",
    "Rail transport", "Water transport", "Agriculture",
    "Waste treatment", "F-gases", "Emission credits",
]

def clean_region_name(raw: str) -> str:
    """Strip the <FONT DIR="AUTO" ...>...</FONT> wrapper the site exports and
    normalise to title case, e.g. 'CENTRAL FINLAND' -> 'Central Finland'."""
    text = re.sub(r"<[^>]+>", "", str(raw))
    text = text.strip()
    return text.title()

def parse_one_file(path: str):
    """Return (region_name, long_df, summary_df) for one region's GHG emissions tab."""
    header_cell = pd.read_excel(path, sheet_name="GHG emissions", header=None, nrows=1).iloc[0, 0]
    region_name = clean_region_name(header_cell)

    df = pd.read_excel(path, sheet_name="GHG emissions", header=0)
    df = df.rename(columns={df.columns[0]: "row_label"})
    year_cols = [c for c in df.columns if c != "row_label"]

    df["row_label_clean"] = df["row_label"].astype(str).str.strip()

    sector_mask = df["row_label_clean"].isin(SECTOR_ROWS)
    sectors = df[sector_mask].copy()

    long_rows = []
    for _, row in sectors.iterrows():
        for yc in year_cols:
            val = row[yc]
            if pd.notna(val):
                long_rows.append({
                    "region": region_name,
                    "sector": row["row_label_clean"].strip(),
                    "year": int(yc),
                    "value_ktco2e": float(val),
                })
    long_df = pd.DataFrame(long_rows)

    def get_row(label):
        r = df[df["row_label_clean"].str.startswith(label)]
        return r.iloc[0] if len(r) else None

    total_row = get_row("total emissions")
    percap_row = get_row("per person")
    pop_row = get_row("population")

    summary_rows = []
    for yc in year_cols:
        summary_rows.append({
            "region": region_name,
            "year": int(yc),
            "total_ktco2e": float(total_row[yc]) if total_row is not None and pd.notna(total_row[yc]) else None,
            "per_capita_tco2e": float(percap_row[yc]) if percap_row is not None and pd.notna(percap_row[yc]) else None,
            "population": float(pop_row[yc]) if pop_row is not None and pd.notna(pop_row[yc]) else None,
        })
    summary_df = pd.DataFrame(summary_rows)

    return region_name, long_df, summary_df

def main():
    files = glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
    if not files:
        print(f"No .xlsx files found in '{RAW_DIR}/'. Put your downloaded region files there.")
        return

    all_long, all_summary, found = [], [], []

    for path in sorted(files):
        try:
            region_name, long_df, summary_df = parse_one_file(path)
            all_long.append(long_df)
            all_summary.append(summary_df)
            found.append({"file": os.path.basename(path), "region": region_name,
                          "sector_rows": len(long_df), "years": long_df["year"].nunique()})
            print(f"  OK   {os.path.basename(path):45s} -> {region_name}")
        except Exception as e:
            print(f"  FAIL {os.path.basename(path):45s} -> {e}")

    master_long = pd.concat(all_long, ignore_index=True)
    master_summary = pd.concat(all_summary, ignore_index=True)
    found_df = pd.DataFrame(found)

    # ── Save as CSV ──────────────────────────────────────────
    master_long.to_csv(os.path.join(OUT_DIR, "ghg_emissions_long.csv"), index=False)
    master_summary.to_csv(os.path.join(OUT_DIR, "region_year_summary.csv"), index=False)
    found_df.to_csv(os.path.join(OUT_DIR, "regions_found.csv"), index=False)

    # ── Save into the shared SQLite database ────────────────
    conn = sqlite3.connect(DB_PATH)
    master_long.to_sql("ghg_emissions_long", conn, if_exists="replace", index=False)
    master_summary.to_sql("region_year_summary", conn, if_exists="replace", index=False)
    conn.close()

    n_regions = master_long["region"].nunique()
    print(f"\nParsed {len(files)} files -> {n_regions} unique regions.")
    print(f"Master long table   : {master_long.shape}  -> {OUT_DIR}/ghg_emissions_long.csv")
    print(f"Region-year summary : {master_summary.shape}  -> {OUT_DIR}/region_year_summary.csv")
    print(f"Also written to DB  : {DB_PATH} (tables: ghg_emissions_long, region_year_summary)")
    if n_regions < 19:
        print(f"\nNOTE: expecting 19 regions, found {n_regions}. "
              f"Add the remaining files to '{RAW_DIR}/' and re-run — nothing else needs to change.")

if __name__ == "__main__":
    main()
