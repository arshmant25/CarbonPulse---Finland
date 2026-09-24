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

def parse_subclass_blocks(path: str, region_name: str) -> pd.DataFrame:
    """Parse the 'emissions and energy' sheet, which holds one or more
    stacked tables (blocks) of the form:
        class | subclass | unit | 1990 | 2005 | 2006 | ... | 2024
    Each region's real file has (at least) two blocks: one with
    unit='ktCO2e' (emissions by class/subclass -- what we want for the
    sectoral breakdown) and one with unit='GWh' (underlying energy use by
    the same class/subclass -- kept too, useful for "did emissions per unit
    of energy fall" questions later). Any further blocks the site adds are
    picked up automatically since we detect blocks generically by their
    header row rather than assuming exactly two.
    """
    df = pd.read_excel(path, sheet_name="emissions and energy", header=None)

    header_rows = df.index[
        (df[0].astype(str).str.strip().str.lower() == "class")
        & (df[1].astype(str).str.strip().str.lower() == "subclass")
    ].tolist()

    records = []
    for h in header_rows:
        header = df.loc[h]
        year_cols = {}
        for c in range(3, df.shape[1]):
            val = header[c]
            if pd.notna(val):
                try:
                    year_cols[c] = int(float(val))
                except (ValueError, TypeError):
                    continue

        r = h + 1
        while r < len(df):
            cls = df.loc[r, 0]
            if pd.isna(cls):
                break  # blank row ends this block
            subclass = df.loc[r, 1]
            unit = df.loc[r, 2]
            for c, year in year_cols.items():
                val = df.loc[r, c]
                if pd.notna(val):
                    records.append({
                        "region": region_name,
                        "sector": str(cls).strip(),
                        "subclass": str(subclass).strip() if pd.notna(subclass) else None,
                        "unit": str(unit).strip() if pd.notna(unit) else None,
                        "year": year,
                        "value": float(val),
                    })
            r += 1

    return pd.DataFrame(records)

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
    all_subclass_emissions, all_subclass_energy = [], []

    for path in sorted(files):
        try:
            region_name, long_df, summary_df = parse_one_file(path)
            all_long.append(long_df)
            all_summary.append(summary_df)
            found.append({"file": os.path.basename(path), "region": region_name,
                          "sector_rows": len(long_df), "years": long_df["year"].nunique()})

            sub_df = parse_subclass_blocks(path, region_name)
            if not sub_df.empty:
                is_emissions = sub_df["unit"].str.lower() == "ktco2e"
                all_subclass_emissions.append(sub_df[is_emissions].drop(columns=["unit"])
                                               .rename(columns={"value": "value_ktco2e"}))
                all_subclass_energy.append(sub_df[~is_emissions])

            print(f"  OK   {os.path.basename(path):45s} -> {region_name}"
                  f"  (+{len(sub_df):,} subclass rows)")
        except Exception as e:
            print(f"  FAIL {os.path.basename(path):45s} -> {e}")

    master_long = pd.concat(all_long, ignore_index=True)
    master_summary = pd.concat(all_summary, ignore_index=True)
    found_df = pd.DataFrame(found)

    master_subclass_emissions = (pd.concat(all_subclass_emissions, ignore_index=True)
                                  if all_subclass_emissions else pd.DataFrame())
    master_subclass_energy = (pd.concat(all_subclass_energy, ignore_index=True)
                               if all_subclass_energy else pd.DataFrame())

    # ── Save as CSV ──────────────────────────────────────────
    master_long.to_csv(os.path.join(OUT_DIR, "ghg_emissions_long.csv"), index=False)
    master_summary.to_csv(os.path.join(OUT_DIR, "region_year_summary.csv"), index=False)
    found_df.to_csv(os.path.join(OUT_DIR, "regions_found.csv"), index=False)
    if not master_subclass_emissions.empty:
        master_subclass_emissions.to_csv(
            os.path.join(OUT_DIR, "emissions_subclass_long.csv"), index=False)
    if not master_subclass_energy.empty:
        master_subclass_energy.to_csv(
            os.path.join(OUT_DIR, "energy_subclass_long.csv"), index=False)

    # ── Save into the shared SQLite database ────────────────
    conn = sqlite3.connect(DB_PATH)
    master_long.to_sql("ghg_emissions_long", conn, if_exists="replace", index=False)
    master_summary.to_sql("region_year_summary", conn, if_exists="replace", index=False)
    if not master_subclass_emissions.empty:
        master_subclass_emissions.to_sql("emissions_subclass_long", conn,
                                          if_exists="replace", index=False)
    if not master_subclass_energy.empty:
        master_subclass_energy.to_sql("energy_subclass_long", conn,
                                       if_exists="replace", index=False)
    conn.close()

    n_regions = master_long["region"].nunique()
    print(f"\nParsed {len(files)} files -> {n_regions} unique regions.")
    print(f"Master long table    : {master_long.shape}  -> {OUT_DIR}/ghg_emissions_long.csv")
    print(f"Region-year summary  : {master_summary.shape}  -> {OUT_DIR}/region_year_summary.csv")
    if not master_subclass_emissions.empty:
        print(f"Subclass emissions   : {master_subclass_emissions.shape}  "
              f"-> {OUT_DIR}/emissions_subclass_long.csv"
              f"  (units found: {sorted(master_subclass_emissions['sector'].unique())[:3]}...)")
    if not master_subclass_energy.empty:
        print(f"Subclass energy      : {master_subclass_energy.shape}  "
              f"-> {OUT_DIR}/energy_subclass_long.csv"
              f"  (units found: {sorted(master_subclass_energy['unit'].dropna().unique())})")
    print(f"Also written to DB   : {DB_PATH}")
    print("  tables: ghg_emissions_long, region_year_summary, "
          "emissions_subclass_long, energy_subclass_long")
    if n_regions < 19:
        print(f"\nNOTE: expecting 19 regions, found {n_regions}. "
              f"Add the remaining files to '{RAW_DIR}/' and re-run — nothing else needs to change.")

if __name__ == "__main__":
    main()
