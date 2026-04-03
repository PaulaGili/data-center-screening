# EDA on the merged dataset: distributions, missingness, outliers, correlations.
# Outputs go to reports/ so we can reference them in the report without re-running.
# Input:  data/processed/merged_country_dataset.csv

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_here = Path.cwd().resolve()
for _root in [_here, *_here.parents]:
    if (_root / "src" / "paths.py").exists():
        PROJECT_ROOT = _root
        break
else:
    raise FileNotFoundError("Run from the project root.")
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import MERGED_CSV, FIGURES_DIR, TABLES_DIR

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

FIGURES_DIR = FIGURES_DIR / "02_eda"
TABLES_DIR = TABLES_DIR / "02_eda"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load data
df = pd.read_csv(MERGED_CSV)
education_cols = [col for col in df.columns if "education_expenditure" in col.lower()]
df = df.drop(columns=education_cols, errors="ignore")
print("Shape:", df.shape, "| Unique ISO3:", df["iso3"].nunique())


# 2. Subset to core screening variables
core_vars = [
    "country", "iso3", "region",
    "has_data_center_record", "total_data_centers",
    "avg_mean_temp_c_2021_2023", "max_daily_max_temp_c_2021_2023",
    "cooling_degree_days_65f_2021_2023", "days_heat_index_gt35c_2021_2023",
    "water_stress_pct_latest", "freshwater_m3_per_capita_latest",
    "worldrisk_exposure_2025", "worldrisk_index_2025",
    "political_stability_score_0_100_avg_3y",
    "electricity_business_usd_kwh", "renewable_share_pct",
    "internet_penetration_percent",
]
df_core = df[core_vars].copy()


# 3. Clean internet penetration and add log data center count
def extract_number(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.lower() in ["unknown", "none", "nan", ""]:
        return np.nan
    value = value.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else np.nan

df_core["internet_penetration_pct_clean"] = df_core["internet_penetration_percent"].apply(extract_number)
df_core.loc[df_core["internet_penetration_pct_clean"] > 100, "internet_penetration_pct_clean"] = (
    df_core.loc[df_core["internet_penetration_pct_clean"] > 100, "internet_penetration_pct_clean"] / 100
)
df_core["log_total_data_centers"] = np.log1p(df_core["total_data_centers"])


# 4. Save summary statistics and missingness tables
summary_vars = [
    "total_data_centers", "log_total_data_centers",
    "avg_mean_temp_c_2021_2023", "max_daily_max_temp_c_2021_2023",
    "cooling_degree_days_65f_2021_2023", "days_heat_index_gt35c_2021_2023",
    "water_stress_pct_latest", "freshwater_m3_per_capita_latest",
    "worldrisk_exposure_2025", "worldrisk_index_2025",
    "political_stability_score_0_100_avg_3y",
    "electricity_business_usd_kwh", "renewable_share_pct",
    "internet_penetration_pct_clean",
]
df_core[summary_vars].describe().T.to_csv(TABLES_DIR / "summary_statistics.csv")

missing_table = pd.DataFrame({
    "column": df_core.columns,
    "missing_values": df_core.isna().sum().values,
    "missing_pct": (df_core.isna().mean() * 100).round(2).values,
}).sort_values("missing_pct", ascending=False)
missing_table.to_csv(TABLES_DIR / "missing_values.csv", index=False)


# 5. Figure 1: distributions of key screening variables (3x3 grid)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Distributions of key screening variables", fontsize=13)

plot_vars = [
    ("avg_mean_temp_c_2021_2023",           "Avg mean temperature (°C)"),
    ("cooling_degree_days_65f_2021_2023",    "Cooling degree days"),
    ("days_heat_index_gt35c_2021_2023",      "Days heat index > 35°C"),
    ("water_stress_pct_latest",              "Water stress (%)"),
    ("worldrisk_index_2025",                 "WorldRisk Index 2025"),
    ("political_stability_score_0_100_avg_3y","Political stability (0-100)"),
    ("electricity_business_usd_kwh",         "Electricity price (USD/kWh)"),
    ("renewable_share_pct",                  "Renewable share (%)"),
    ("internet_penetration_pct_clean",       "Internet penetration (%)"),
]

for ax, (col, label) in zip(axes.flat, plot_vars):
    ax.hist(df_core[col].dropna(), bins=25, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_title(label, fontsize=9)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_distributions.png")


# 6. Figure 2: scatterplots of key trade-offs (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Key trade-offs between screening variables", fontsize=13)

scatter_pairs = [
    ("avg_mean_temp_c_2021_2023",          "cooling_degree_days_65f_2021_2023",
     "Avg temperature (°C)",               "Cooling degree days"),
    ("cooling_degree_days_65f_2021_2023",  "water_stress_pct_latest",
     "Cooling degree days",                "Water stress (%)"),
    ("electricity_business_usd_kwh",       "renewable_share_pct",
     "Electricity price (USD/kWh)",        "Renewable share (%)"),
    ("political_stability_score_0_100_avg_3y", "worldrisk_index_2025",
     "Political stability (0-100)",        "WorldRisk Index 2025"),
]

for ax, (x, y, xlabel, ylabel) in zip(axes.flat, scatter_pairs):
    data = df_core[[x, y]].dropna()
    ax.scatter(data[x], data[y], s=18, alpha=0.6, color="steelblue")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_scatterplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_scatterplots.png")


# 7. Figure 3: missingness bar chart
plt.figure(figsize=(9, 5))
top_missing = missing_table[missing_table["missing_pct"] > 0].sort_values("missing_pct", ascending=True)
plt.barh(top_missing["column"], top_missing["missing_pct"], color="steelblue")
plt.xlabel("Missing values (%)")
plt.title("Missing values per variable (core screening set)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_missingness.png")


# 8. Spearman correlation matrix
corr_vars = [c for c in summary_vars if c in df_core.columns]
df_core[corr_vars].corr(method="spearman").to_csv(TABLES_DIR / "spearman_correlation.csv")
print("Saved: spearman_correlation.csv")


# 9. Outlier detection: IQR method across core numeric variables
outlier_vars = [
    "avg_mean_temp_c_2021_2023",
    "cooling_degree_days_65f_2021_2023",
    "days_heat_index_gt35c_2021_2023",
    "water_stress_pct_latest",
    "freshwater_m3_per_capita_latest",
    "worldrisk_index_2025",
    "political_stability_score_0_100_avg_3y",
    "electricity_business_usd_kwh",
    "renewable_share_pct",
    "internet_penetration_pct_clean",
]

outlier_flags = df_core[["country", "iso3"]].copy()
outlier_summary = []

for col in outlier_vars:
    s = df_core[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flag = ((df_core[col] < lower) | (df_core[col] > upper))
    outlier_flags[f"outlier_{col}"] = flag
    outlier_summary.append({
        "variable": col,
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_fence": round(lower, 3),
        "upper_fence": round(upper, 3),
        "n_outliers": int(flag.sum()),
    })

outlier_summary_df = pd.DataFrame(outlier_summary)
outlier_summary_df.to_csv(TABLES_DIR / "outlier_summary.csv", index=False)
print("\nOutlier detection (IQR method):")
print(outlier_summary_df[["variable", "lower_fence", "upper_fence", "n_outliers"]].to_string(index=False))

# Countries flagged as outlier in at least one variable
outlier_flag_cols = [c for c in outlier_flags.columns if c.startswith("outlier_")]
outlier_flags["n_outlier_flags"] = outlier_flags[outlier_flag_cols].sum(axis=1)
flagged = outlier_flags[outlier_flags["n_outlier_flags"] > 0].sort_values("n_outlier_flags", ascending=False)
flagged.to_csv(TABLES_DIR / "outlier_flags.csv", index=False)
print(f"\nCountries with at least one outlier flag: {len(flagged)}")

# Figure 4: boxplots for outlier visualization (2x5 grid)
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle("Outlier detection: boxplots of core screening variables", fontsize=12)

for ax, col in zip(axes.flat, outlier_vars):
    data = df_core[col].dropna()
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6),
               medianprops=dict(color="black", linewidth=1.5),
               flierprops=dict(marker="o", markersize=3, alpha=0.5, color="tomato"))
    ax.set_title(col.replace("_", " ").replace("2021 2023", "21-23"), fontsize=7)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_outlier_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 04_outlier_boxplots.png")


# 10. Range validation: check known valid ranges for each variable
VALID_RANGES = {
    "political_stability_score_0_100_avg_3y": (0, 100),
    "renewable_share_pct": (0, 100),
    "internet_penetration_pct_clean": (0, 100),
    "water_stress_pct_latest": (0, None),   # must be >= 0
    "worldrisk_index_2025": (0, 100),
    "electricity_business_usd_kwh": (0, None),   # must be >= 0
    "avg_mean_temp_c_2021_2023": (-90, 60),   # plausible global range
    "cooling_degree_days_65f_2021_2023": (0, None),
}

range_issues = []
for col, (lo, hi) in VALID_RANGES.items():
    if col not in df_core.columns:
        continue
    s = df_core[col].dropna()
    below = int((s < lo).sum()) if lo is not None else 0
    above = int((s > hi).sum()) if hi is not None else 0
    range_issues.append({
        "variable":      col,
        "expected_min":  lo,
        "expected_max":  hi,
        "actual_min":    round(float(s.min()), 4),
        "actual_max":    round(float(s.max()), 4),
        "n_below_min":   below,
        "n_above_max":   above,
        "status":        "OK" if (below + above) == 0 else "FLAG",
    })

range_df = pd.DataFrame(range_issues)
range_df.to_csv(TABLES_DIR / "range_validation.csv", index=False)
print("\nRange validation:")
print(range_df[["variable", "expected_min", "expected_max", "actual_min", "actual_max", "n_below_min", "n_above_max", "status"]].to_string(index=False))

# Figure 5: actual data range vs expected bounds per variable
fig, ax = plt.subplots(figsize=(11, 5))

labels = range_df["variable"].str.replace("_", " ").str[:40].tolist()
n = len(labels)

for i, row in range_df.iterrows():
    color = "tomato" if row["status"] == "FLAG" else "steelblue"
    ax.barh(i, row["actual_max"] - row["actual_min"], left=row["actual_min"],
            height=0.4, color=color, alpha=0.7, label="actual range" if i == 0 else "")
    if row["expected_min"] is not None:
        ax.axvline(x=row["expected_min"], color="gray", linestyle="--", linewidth=0.5)
    if pd.notna(row["expected_max"]):
        ax.axvline(x=row["expected_max"], color="gray", linestyle="--", linewidth=0.5)

ax.set_yticks(list(range(n)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Value range")
ax.set_title("Range validation: actual data range vs expected bounds (dashed lines are limits)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "05_range_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 05_range_validation.png")

print(f"\nDone. Figures: {FIGURES_DIR}")
print(f"      Tables:  {TABLES_DIR}")
