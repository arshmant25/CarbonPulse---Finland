# causality_analysis.py
import json
import sqlite3
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

# Define the unified output path for JSON and graphs
# Change this line in your script:
OUTPUT_DIR = Path("Research Experiments/outputs_fingrid_multiscale_forecasting/causality_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────
conn = sqlite3.connect("data/project_data.db")
df = pd.read_sql(
    "SELECT * FROM fingrid_hourly_merged ORDER BY timestamp_utc", conn
)
conn.close()

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.set_index("timestamp_utc")

# Use hourly data 2019-2024 (stationary period, avoid structural break
# of Olkiluoto 3 coming online in 2023 which shifts the mean)
target_col = "co2_intensity_weighted_avg"
source_cols = [
    "wind_mw",
    "nuclear_mw",
    "hydro_mw",
    "chp_district_mw",
    "chp_industrial_mw",
]

# Resample to daily for Granger (hourly is too noisy, too many lags)
df_daily = df[[target_col] + source_cols].resample("D").mean().dropna()

# ── Granger causality: does each source Granger-cause intensity? ──
print("=== GRANGER CAUSALITY TEST ===")
print("H0: Source does NOT Granger-cause CO2 intensity")
print(
    "Reject H0 (p < 0.05): source IS a significant predictor of future intensity"
)
print()

max_lag = 7  # test up to 7-day lags
results_summary = []

for col in source_cols:
    test_data = df_daily[[target_col, col]].dropna()
    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

    # Extract minimum p-value across all lags
    min_p = min(
        gc_result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
    )

    # Correct extraction of best lag value (the key integer, e.g., 3)
    best_lag = int(
        min(gc_result, key=lambda l: gc_result[l][0]["ssr_ftest"][1])
    )

    results_summary.append(
        {
            "source": col,
            "min_p_value": float(round(min_p, 4)),
            "best_lag_days": best_lag,
            "granger_causes": (
                "YES ***"
                if min_p < 0.001
                else ("YES *" if min_p < 0.05 else "NO")
            ),
        }
    )
    print(
        f"  {col:25s}: p={min_p:.4f}  lag={best_lag}d  → {results_summary[-1]['granger_causes']}"
    )

df_gc = pd.DataFrame(results_summary)
print()
print(df_gc.to_string(index=False))

# ── Save Results to JSON ───────────────────────────────────────
json_path = OUTPUT_DIR / "granger_causality_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=4)
print(f"\n[INFO] JSON metrics saved to: {json_path}")

# ── Visualise: Granger causality p-values ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = [
    "#2a9d8f" if r["min_p_value"] < 0.05 else "#e76f51"
    for _, r in df_gc.iterrows()
]
bars = ax.barh(
    df_gc["source"], -np.log10(df_gc["min_p_value"]), color=colors, alpha=0.85
)
ax.axvline(
    -np.log10(0.05), color="black", lw=1.5, linestyle="--", label="p=0.05 threshold"
)
ax.axvline(
    -np.log10(0.001), color="grey", lw=1, linestyle=":", label="p=0.001 threshold"
)
ax.set_xlabel("-log₁₀(p-value)  [higher = stronger causality]")
ax.set_title(
    "Granger Causality: Do generation sources predict CO₂ intensity?\n"
    "Teal = significant (p<0.05), Coral = not significant",
    fontweight="bold",
)
ax.legend()
plt.tight_layout()

# Save using unified pathlib object
graph1_path = OUTPUT_DIR / "granger_causality.png"
plt.savefig(graph1_path, dpi=130, bbox_inches="tight")
print(f"[INFO] Granger plot saved to: {graph1_path}")
plt.show()

# ── Lagged cross-correlation ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(source_cols):
    ax = axes[i]
    xcorr = [
        df_daily[target_col].corr(df_daily[col].shift(lag))
        for lag in range(-7, 8)
    ]
    ax.bar(
        range(-7, 8),
        xcorr,
        color=["#2a9d8f" if x < 0 else "#e76f51" for x in xcorr],
        alpha=0.8,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.5, linestyle=":")
    ax.set_title(f"{col}\nvs co2_intensity", fontsize=9, fontweight="bold")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Pearson r")
    ax.set_xticks(range(-7, 8))

for j in range(len(source_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle(
    "Lagged Cross-Correlation: Generation Sources vs CO₂ Intensity\n"
    "Teal = negative correlation (source reduces intensity), Coral = positive",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()

# Save using unified pathlib object
graph2_path = OUTPUT_DIR / "lagged_crosscorrelation.png"
plt.savefig(graph2_path, dpi=130, bbox_inches="tight")
print(f"[INFO] Cross-correlation plot saved to: {graph2_path}")
plt.show()
