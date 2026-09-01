import sqlite3
import pandas as pd
import numpy as np
from old_versions.config import DB_PATH, DATASETS

# Step 1: Aggregate Fingrid hourly data to daily
conn = sqlite3.connect("data/project_data.db")
df_hourly = pd.read_sql(
    "SELECT * FROM fingrid_hourly_merged ORDER BY timestamp_utc", conn
)
conn.close()

df_hourly["timestamp_utc"] = pd.to_datetime(df_hourly["timestamp_utc"], utc=True)
df_hourly = df_hourly.set_index("timestamp_utc")

# Daily mean intensity and total production
df_daily_fingrid = df_hourly.resample("D").agg({
    "co2_intensity_simple_avg":   "mean",    # mean intensity across the day
    "co2_intensity_weighted_avg": "mean",
    "total_production_mw":       "sum",     # sum = total MWh / 1000 = GWh
    "wind_mw":               "mean",
    "nuclear_mw":            "mean",
})
# Convert production to daily CO2 in Mt:
# intensity (gCO2/kWh) × production (MWh) ÷ 1e9 = Mt CO2
df_daily_fingrid["daily_co2_mt"] = (
    df_daily_fingrid["co2_intensity_weighted_avg"] *
    df_daily_fingrid["total_production_mw"] /
    1e9   # g→t is ÷1e6, but MWh×gCO2/kWh = gCO2×1000, net ÷1e6
)

# Step 2: Load Carbon Monitor Finland Power sector
# Download from eu.carbonmonitor.org — filter country=FI, sector=Power
df_cm = pd.read_excel("data/carbon_monitor_finland.xlsx")
df_cm.columns = df_cm.columns.str.strip()
df_cm["date"] = pd.to_datetime(df_cm["date"], dayfirst=True).dt.tz_localize("UTC")
df_cm_power   = df_cm[df_cm["sector"] == "Power"].set_index("date")

# Step 3: Join on date for the overlapping period 2019-2026
df_compare = df_daily_fingrid.join(
    df_cm_power[["MtCO2 per day"]].rename(columns={"MtCO2 per day": "cm_power_mt"}),
    how="inner"
)

# Step 4: Correlation and validation
r = df_compare[["daily_co2_mt","cm_power_mt"]].corr().iloc[0,1]
mae = np.mean(np.abs(df_compare["daily_co2_mt"] - df_compare["cm_power_mt"]))
print(f"Correlation (Fingrid-derived vs Carbon Monitor Power): {r:.3f}")
print(f"MAE: {mae:.4f} Mt CO2/day")