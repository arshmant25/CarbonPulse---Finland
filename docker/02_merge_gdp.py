"""
Parse Statistics Finland's "GDP per capita by area" export and merge it
into region_year_summary.

YOUR FILE: region_data/gdp_raw.xlsx
    "Gross domestic product per capita by area, annually by Area, Year and Information"

This is a PxWeb export with a 2-row header: row 1 has the year (merged
across 3 columns), row 2 repeats "At current prices, euro" / "Population" /
"Volume series, reference year 2015" for each year block. Column A holds
the area label, which mixes several region levels in one file:
    WHOLE COUNTRY, MA... (mainland), SA... (major regions),
    MK... (regions / maakunta)   <-- THE ONES WE WANT
    HVA... (wellbeing services counties), SK... (sub-regions)

We keep only rows starting with "MK##" and treat the rest of that label as
the region name (e.g. "MK01 Uusimaa" -> "Uusimaa").

The "At current prices, euro" column is GDP PER CAPITA (not total GDP).
We multiply it by the "Population" column from the same file to recover
total regional GDP in euros — that's what you compare emissions against.
The "Volume series, reference year 2015" column is an inflation-adjusted
(real) per-capita index; kept too, in case you want a real-terms trend
that isn't distorted by inflation.

USAGE:
    python 02_merge_gdp.py
Outputs:
    region_data/processed/gdp_by_region_year.csv        (tidy GDP table alone)
    region_data/processed/region_year_with_gdp.csv       (emissions summary + GDP, merged)
    Both tables are also written into data/project_data.db
"""

import os
import re
import sqlite3
import pandas as pd

GDP_RAW_PATH = os.path.join("region_data", "gdp_raw.xlsx")
OUT_DIR = os.path.join("region_data", "processed")
DB_PATH = os.path.join("data", "project_data.db")

os.makedirs(OUT_DIR, exist_ok=True)

MK_ROW_RE = re.compile(r"^MK\d+\s+(.*)$")

def normalize(name: str) -> str:
    """Normalise a region name so GDP-file spelling and emissions-file
    spelling can be matched reliably (diacritics, hyphens, 'South(ern)')."""
    name = name.strip()
    trans = str.maketrans("åäöÅÄÖ", "aaoAAO")
    name = name.translate(trans)
    name = name.replace("-", " ")
    name = re.sub(r"\s+", " ", name).lower()
    name = name.replace("southern ostrobothnia", "south ostrobothnia")
    return name

def parse_gdp_file(path: str) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)

    # Row 0 = title. Row 1 = years (sparse, only first col of each 3-col block).
    # Row 2 = sub-column labels, repeating every 3 columns.
    year_row = raw.iloc[2].ffill()
    subcol_row = raw.iloc[3]

    data = raw.iloc[4:].reset_index(drop=True)
    data = data.rename(columns={0: "area_label"})

    records = []
    for col in raw.columns[1:]:
        year_val = year_row[col]
        subcol_val = subcol_row[col]
        if pd.isna(year_val) or pd.isna(subcol_val):
            continue
        try:
            year = int(float(year_val))
        except (ValueError, TypeError):
            continue

        for i, row in data.iterrows():
            label = str(row["area_label"]).strip()
            m = MK_ROW_RE.match(label)
            if not m:
                continue
            region = m.group(1).strip()
            val = row[col]
            if pd.isna(val):
                continue
            records.append({
                "region_raw": region,
                "year": year,
                "field": str(subcol_val).strip(),
                "value": float(val),
            })

    long_df = pd.DataFrame(records)

    print("Available columns in long_df:", long_df.columns.tolist())
    print(long_df.head(2))

    wide = long_df.pivot_table(index=["region_raw", "year"], columns="field",
                                values="value", aggfunc="first").reset_index()
    wide.columns.name = None

    wide = wide.rename(columns={
        "At current prices, euro": "gdp_per_capita_eur",
        "Population": "population_gdp_source",
        "Volume series, reference year 2015": "gdp_per_capita_real_2015eur",
    })

    if "gdp_per_capita_eur" in wide.columns and "population_gdp_source" in wide.columns:
        wide["gdp_total_eur"] = wide["gdp_per_capita_eur"] * wide["population_gdp_source"]

    wide["region_key"] = wide["region_raw"].apply(normalize)
    return wide

def main():
    if not os.path.exists(GDP_RAW_PATH):
        print(f"Could not find {GDP_RAW_PATH}. Save your download there first.")
        return

    gdp = parse_gdp_file(GDP_RAW_PATH)
    print(f"Parsed GDP rows: {gdp.shape}")
    print("Regions found in GDP file:", sorted(gdp['region_raw'].unique()))

    gdp.to_csv(os.path.join(OUT_DIR, "gdp_by_region_year.csv"), index=False)

    summary_path = os.path.join(OUT_DIR, "region_year_summary.csv")
    if not os.path.exists(summary_path):
        print(f"\n{summary_path} not found — run 01_consolidate_regions.py first. "
              f"Saved GDP table alone for now.")
        return

    summary = pd.read_csv(summary_path)
    summary["region_key"] = summary["region"].apply(normalize)

    merged = summary.merge(
        gdp.drop(columns=["region_raw"]), on=["region_key", "year"], how="left"
    ).drop(columns=["region_key"])

    if "gdp_total_eur" in merged.columns:
        merged["co2_tonnes_per_million_eur_gdp"] = (
            merged["total_ktco2e"] * 1000 / (merged["gdp_total_eur"] / 1e6)
        )

    unmatched_regions = sorted(
        set(summary.loc[~summary["region_key"].isin(gdp["region_key"]), "region"])
    )
    if unmatched_regions:
        print(f"\nWARNING: no GDP match for: {unmatched_regions}")
        print("Check the region's spelling in gdp_raw.xlsx vs the emissions file, "
              "or add a fix to normalize()/an explicit alias.")

    out_path = os.path.join(OUT_DIR, "region_year_with_gdp.csv")
    merged.to_csv(out_path, index=False)

    conn = sqlite3.connect(DB_PATH)
    gdp.to_sql("gdp_by_region_year", conn, if_exists="replace", index=False)
    merged.to_sql("region_year_with_gdp", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\nSaved: {out_path}  {merged.shape}")
    print(f"Also written to DB: {DB_PATH} (tables: gdp_by_region_year, region_year_with_gdp)")

if __name__ == "__main__":
    main()
