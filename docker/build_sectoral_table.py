# build_sectoral_table.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE (or whenever you re-download the source CSVs):
#
#   python build_sectoral_table.py
#
# What it does:
#   1. Loads sectoral_emissions_total.csv  (gas = "Total", already CO2e)
#   2. Loads sectoral_emissions_co2.csv    (gas = "Carbon dioxide (CO2)")
#   3. Joins them on (year, sector_code) so each row shows both
#      total_kt_co2e AND co2_kt -- the difference is "other gases"
#      (CH4, N2O, F-gases combined, already in CO2e from the source)
#   4. Attaches the full hierarchy: level, group, parent_code,
#      parent_name, parent_l1 (top-level ancestor)
#   5. Verifies parent/child sums (prints a report, does not fail --
#      source data has intentional gaps for confidentiality reasons)
#   6. Writes everything to data/project_data.db as table
#      "sectoral_emissions"
#
# Input files expected in data/:
#   data/sectoral_emissions_total.csv
#   data/sectoral_emissions_co2.csv
#
# Tools used: pandas, sqlite3, re

import re
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

from sector_hierarchy import get_hierarchy_dataframe, verify_hierarchy

DATA_DIR = Path("data")
DB_PATH  = DATA_DIR / "project_data.db"

TOTAL_CSV = DATA_DIR / "sectoral_emissions_total.csv"
CO2_CSV   = DATA_DIR / "sectoral_emissions_co2.csv"


# ══════════════════════════════════════════════════════════════
# STEP 1: Parse raw CSVs
# ══════════════════════════════════════════════════════════════

# Special category names that don't follow the "CODE Name" pattern
SPECIAL_CODES = {
    "Emissions without LULUCF": "TOTAL_NO_LULUCF",
    "Emissions with LULUCF":    "TOTAL_WITH_LULUCF",
    "Indirect CO2 emission":    "INDCO2",
}


def parse_sector_code(full_name: str) -> str:
    """
    Extract the IPCC sector code from the full category string.

    Examples:
      "1A1 Energy industries"            -> "1A1"
      "1A2(-1A2gvii) Manufacturing..."   -> "1A2(-1A2gvii)"
      "Emissions without LULUCF"         -> "TOTAL_NO_LULUCF"
      "Indirect CO2 emission"            -> "INDCO2"
    """
    if full_name in SPECIAL_CODES:
        return SPECIAL_CODES[full_name]
    match = re.match(r'^(\S+)\s+(.*)$', full_name)
    if match:
        return match.group(1)
    return full_name


def load_raw_csv(path: Path, value_col_name: str) -> pd.DataFrame:
    """
    Load one Statistics Finland CSV and return a clean DataFrame
    with columns: year, sector_code, sector_full, <value_col_name>

    Handles:
      - "*" suffix on years (preliminary estimate flag)
      - "-" meaning literal zero
      - "." meaning data suppressed / not available -> NaN
      - thousands separators / decimal commas if present
    """
    df = pd.read_csv(path, dtype=str)
    df.columns = ["year", "sector_full", "gas", "value"]

    # Year: strip "*" (preliminary marker), convert to int
    df["is_preliminary"] = df["year"].str.contains(r"\*", regex=True)
    df["year"] = (df["year"]
                   .str.replace("*", "", regex=False)
                   .astype(int))

    # Value: "-" -> 0 (true zero), "." -> NaN (suppressed/unavailable)
    df["value"] = (df["value"]
                   .astype(str)
                   .str.strip()
                   .str.replace(" ", "", regex=False)    # thousands sep
                   .str.replace(",", ".", regex=False))  # decimal comma
    df["value"] = df["value"].replace("-", "0")
    df["value"] = df["value"].replace(".", np.nan)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Extract IPCC sector code
    df["sector_code"] = df["sector_full"].apply(parse_sector_code)

    df = df.rename(columns={"value": value_col_name})

    return df[["year", "sector_code", "sector_full",
               "is_preliminary", value_col_name]]


# ══════════════════════════════════════════════════════════════
# STEP 2: Join total + co2 on (year, sector_code)
# ══════════════════════════════════════════════════════════════

def build_joined_table() -> pd.DataFrame:
    """
    Returns one row per (year, sector_code) with:
      - total_kt_co2e : value from the "Total" gas file (already CO2e)
      - co2_kt        : value from the "Carbon dioxide (CO2)" file
      - other_gases_kt: total_kt_co2e - co2_kt
                        (= combined CH4 + N2O + F-gases in CO2e)
      - co2_share_pct : what % of this sector's emissions is CO2
                        (low % => sector dominated by CH4/N2O,
                         e.g. agriculture)
    """
    df_total = load_raw_csv(TOTAL_CSV, "total_kt_co2e")
    df_co2   = load_raw_csv(CO2_CSV,   "co2_kt")

    df_co2_slim = df_co2[["year", "sector_code", "co2_kt"]]

    df = df_total.merge(
        df_co2_slim,
        on=["year", "sector_code"],
        how="left"
    )

    # other gases = total - co2 (can be negative for LULUCF
    # sub-categories where signs are mixed -- that's expected)
    df["other_gases_kt"] = df["total_kt_co2e"] - df["co2_kt"]

    # co2 share: guard against division by zero / NaN
    with np.errstate(divide="ignore", invalid="ignore"):
        df["co2_share_pct"] = np.where(
            (df["total_kt_co2e"].notna()) & (df["total_kt_co2e"] != 0),
            (df["co2_kt"] / df["total_kt_co2e"]) * 100,
            np.nan
        )

    return df


# ══════════════════════════════════════════════════════════════
# STEP 3: Attach hierarchy (parent/child/subchild + groups)
# ══════════════════════════════════════════════════════════════

def attach_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge in level, group, parent_code, parent_name, parent_l1
    from sector_hierarchy.py

    Adds human-readable depth labels:
      level 0 -> "total"
      level 1 -> "parent"      (main IPCC sector, e.g. 1, 2, 3, 4, 5)
      level 2 -> "child"       (subsector, e.g. 1A1, 1A2, 1A3)
      level 3 -> "subchild"    (sub-subsector, e.g. 1A3a, 1A3b)
      level 4 -> "subsubchild" (deepest level, e.g. 1A3bi)
    """
    hier = get_hierarchy_dataframe()

    code_to_name = hier.set_index("sector_code")["sector_name"].to_dict()

    df = df.merge(hier, on="sector_code", how="left")

    df["parent_name"] = df["parent_code"].map(code_to_name)

    depth_labels = {
        0: "total",
        1: "parent",
        2: "child",
        3: "subchild",
        4: "subsubchild",
    }
    df["depth_label"] = df["level"].map(depth_labels)

    return df


# ══════════════════════════════════════════════════════════════
# STEP 4: Verification
# ══════════════════════════════════════════════════════════════

def run_verification(df: pd.DataFrame):
    """
    Print a verification report:
      - row count, year range, sector count
      - special check: TOTAL_WITH_LULUCF = TOTAL_NO_LULUCF + sector 4
      - general parent/child sum check for the latest year
    """
    print("=" * 60)
    print("SECTORAL DATA VERIFICATION REPORT")
    print("=" * 60)

    print(f"Total rows        : {len(df):,}")
    print(f"Years covered     : {df['year'].min()} - {df['year'].max()}")
    print(f"Unique sectors    : {df['sector_code'].nunique()}")
    print(f"Rows with NaN total_kt_co2e : {df['total_kt_co2e'].isna().sum()}")
    print(f"Rows with NaN co2_kt        : {df['co2_kt'].isna().sum()}")

    latest_year = df["year"].max()
    sub = df[df["year"] == latest_year].set_index("sector_code")

    no_lulucf   = sub.loc["TOTAL_NO_LULUCF", "total_kt_co2e"]
    with_lulucf = sub.loc["TOTAL_WITH_LULUCF", "total_kt_co2e"]
    sector_4    = sub.loc["4", "total_kt_co2e"]

    print(f"\n--- LULUCF check for {latest_year} ---")
    print(f"  TOTAL without LULUCF : {no_lulucf:>10,.0f} kt CO2e")
    print(f"  Sector 4 (LULUCF)    : {sector_4:>10,.0f} kt CO2e")
    print(f"  Sum                  : {no_lulucf + sector_4:>10,.0f} kt CO2e")
    print(f"  TOTAL with LULUCF    : {with_lulucf:>10,.0f} kt CO2e")
    diff = (no_lulucf + sector_4) - with_lulucf
    print(f"  Difference           : {diff:>10,.1f} kt CO2e "
          f"{'OK' if abs(diff) < 5 else 'CHECK THIS'}")

    print(f"\n--- Parent/child sum check for {latest_year} ---")
    check_df = verify_hierarchy(
        df[df["year"] == latest_year].rename(columns={"total_kt_co2e": "value_total"}),
        "value_total"
    )
    check_df = check_df.dropna(subset=["children_sum"])
    check_df = check_df[check_df["children_sum"] != 0]
    # TOTAL_WITH_LULUCF is verified separately above (it's an additive
    # relationship: TOTAL_NO_LULUCF + sector 4, not a sum-of-children)
    check_df = check_df[check_df["parent_code"] != "TOTAL_WITH_LULUCF"]
    big_diff = check_df[check_df["difference"].abs() > 5]
    if big_diff.empty:
        print("  All parent/child sums match within rounding tolerance.")
    else:
        print("  Differences > 5 kt found (often due to suppressed")
        print("  subcategory data for confidentiality -- not necessarily an error):")
        print(big_diff.to_string(index=False))

    print("=" * 60)


# ══════════════════════════════════════════════════════════════
# STEP 5: Save to database
# ══════════════════════════════════════════════════════════════

def save_to_db(df: pd.DataFrame):
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    cols = [
        "year", "sector_code", "sector_full", "sector_name",
        "level", "depth_label", "group",
        "parent_code", "parent_name", "parent_l1",
        "total_kt_co2e", "co2_kt", "other_gases_kt", "co2_share_pct",
        "is_preliminary",
    ]
    cols = [c for c in cols if c in df.columns]
    df_out = df[cols].sort_values(["year", "sector_code"]).reset_index(drop=True)

    df_out.to_sql("sectoral_emissions", conn, if_exists="replace", index=False)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sectoral_year_code
        ON sectoral_emissions (year, sector_code)
    """)
    conn.commit()
    conn.close()

    print(f"\nSaved {len(df_out):,} rows to {DB_PATH} -> table 'sectoral_emissions'")
    print(f"Columns: {', '.join(cols)}")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading and joining sectoral CSVs...")
    df = build_joined_table()

    print("Attaching hierarchy (parent/child/subchild + groups)...")
    df = attach_hierarchy(df)

    run_verification(df)

    save_to_db(df)

    print("\nDone. Example query to try later:")
    print("""
    SELECT year, sector_code, sector_name, depth_label, "group",
           total_kt_co2e, co2_kt, other_gases_kt, co2_share_pct
    FROM   sectoral_emissions
    WHERE  sector_code = '1A1' AND year >= 2018
    ORDER  BY year;
    """)
