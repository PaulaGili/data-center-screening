# Three suitability definitions. We explored these before settling on the min-max
# approach in step 04. Results are directionally comparable but not numerically identical
# (percentile rank here vs min-max in step 04).
#
# Definition 1: hard thresholds (pass/fail per pillar)
# Definition 2: average percentile rank across all pillars
# Definition 3: strategy-specific weighted scores (same weights as step 04 presets)
#
# Input:  data/processed/merged_country_dataset.csv
# Output: reports/figures/03_suitability/, reports/tables/03_suitability/

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

FIGURES_DIR = FIGURES_DIR / "03_suitability"
TABLES_DIR = TABLES_DIR / "03_suitability"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load data
df = pd.read_csv(MERGED_CSV)
education_cols = [col for col in df.columns if "education_expenditure" in col.lower()]
df = df.drop(columns=education_cols, errors="ignore")
print("Shape:", df.shape)


# 2. Clean internet penetration and subset to core variables
def extract_number(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip()
    if value.lower() in ["unknown", "none", "nan", ""]:
        return np.nan
    value = value.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group()) if match else np.nan

df["internet_penetration_pct_clean"] = df["internet_penetration_percent"].apply(extract_number)
df.loc[df["internet_penetration_pct_clean"] > 100, "internet_penetration_pct_clean"] = (
    df.loc[df["internet_penetration_pct_clean"] > 100, "internet_penetration_pct_clean"] / 100
)

keep_cols = [
    "country", "iso3", "region",
    "has_data_center_record", "total_data_centers",
    "cooling_degree_days_65f_2021_2023",
    "water_stress_pct_latest",
    "worldrisk_index_2025",
    "political_stability_score_0_100_avg_3y",
    "electricity_business_usd_kwh",
    "renewable_share_pct",
    "internet_penetration_pct_clean",
    "avg_mean_temp_c_2021_2023",
]
df_core = df[keep_cols].copy()


# Definition 1: minimum feasibility
# 75th percentile as cutoff for "lower is better" vars, 25th for "higher is better"
# we tried median but it was too strict, excluded too many viable countries
thresholds = pd.DataFrame({
    "variable": [
        "cooling_degree_days_65f_2021_2023",
        "water_stress_pct_latest",
        "worldrisk_index_2025",
        "political_stability_score_0_100_avg_3y",
        "electricity_business_usd_kwh",
        "renewable_share_pct",
        "internet_penetration_pct_clean",
    ],
    "direction": [
        "lower is better", "lower is better", "lower is better", "higher is better",
        "lower is better", "higher is better", "higher is better",
    ],
    "threshold": [
        df_core["cooling_degree_days_65f_2021_2023"].quantile(0.75),
        df_core["water_stress_pct_latest"].quantile(0.75),
        df_core["worldrisk_index_2025"].quantile(0.75),
        df_core["political_stability_score_0_100_avg_3y"].quantile(0.25),
        df_core["electricity_business_usd_kwh"].quantile(0.75),
        df_core["renewable_share_pct"].quantile(0.25),
        df_core["internet_penetration_pct_clean"].quantile(0.25),
    ],
})
thresholds.to_csv(TABLES_DIR / "definition1_thresholds.csv", index=False)

feasible = df_core.dropna(subset=thresholds["variable"].tolist()).copy()
feasible["pass_cooling"] = feasible["cooling_degree_days_65f_2021_2023"] <= thresholds.loc[0, "threshold"]
feasible["pass_water"] = feasible["water_stress_pct_latest"] <= thresholds.loc[1, "threshold"]
feasible["pass_risk"] = feasible["worldrisk_index_2025"] <= thresholds.loc[2, "threshold"]
feasible["pass_stability"] = feasible["political_stability_score_0_100_avg_3y"] >= thresholds.loc[3, "threshold"]
feasible["pass_electricity"] = feasible["electricity_business_usd_kwh"] <= thresholds.loc[4, "threshold"]
feasible["pass_renewables"] = feasible["renewable_share_pct"] >= thresholds.loc[5, "threshold"]
feasible["pass_internet"] = feasible["internet_penetration_pct_clean"] >= thresholds.loc[6, "threshold"]

pass_cols = ["pass_cooling", "pass_water", "pass_risk", "pass_stability",
             "pass_electricity", "pass_renewables", "pass_internet"]
feasible["n_conditions_passed"] = feasible[pass_cols].sum(axis=1)
feasible["passes_minimum_feasibility"] = feasible["n_conditions_passed"] == len(pass_cols)

definition1_summary = pd.DataFrame({
    "metric": [
        "Countries before Definition 1",
        "Countries evaluated in Definition 1",
        "Countries excluded (missing values)",
        "Countries passing Definition 1",
        "Countries failing Definition 1",
    ],
    "value": [
        len(df_core), len(feasible), len(df_core) - len(feasible),
        int(feasible["passes_minimum_feasibility"].sum()),
        int((~feasible["passes_minimum_feasibility"]).sum()),
    ],
})
definition1_summary.to_csv(TABLES_DIR / "definition1_summary.csv", index=False)
print("\nDefinition 1 summary:")
print(definition1_summary.to_string(index=False))


# 4. Definition 2: balanced suitability (percentile scores, then averaged)
balanced = df_core.copy()
balanced["score_cooling"] = 1 - balanced["cooling_degree_days_65f_2021_2023"].rank(pct=True)
balanced["score_water"] = 1 - balanced["water_stress_pct_latest"].rank(pct=True)
balanced["score_risk"] = 1 - balanced["worldrisk_index_2025"].rank(pct=True)
balanced["score_electricity"] = 1 - balanced["electricity_business_usd_kwh"].rank(pct=True)
balanced["score_stability"] = balanced["political_stability_score_0_100_avg_3y"].rank(pct=True)
balanced["score_renewables"] = balanced["renewable_share_pct"].rank(pct=True)
balanced["score_internet"] = balanced["internet_penetration_pct_clean"].rank(pct=True)

score_cols = ["score_cooling", "score_water", "score_risk", "score_electricity",
              "score_stability", "score_renewables", "score_internet"]
balanced["n_scores_used"] = balanced[score_cols].notna().sum(axis=1)
balanced["balanced_suitability_score"] = balanced[score_cols].mean(axis=1, skipna=True)

top_balanced = (
    balanced.loc[balanced["n_scores_used"] >= 5,
                 ["country", "iso3", "region", "balanced_suitability_score",
                  "n_scores_used", "has_data_center_record", "total_data_centers"]]
    .sort_values("balanced_suitability_score", ascending=False)
    .head(20)
)
top_balanced.to_csv(TABLES_DIR / "definition2_top20_balanced.csv", index=False)
print("\nDefinition 2, top 20 balanced:")
print(top_balanced.to_string(index=False))


# 5. Definition 3: strategy-specific scores
# Sustainability weights and pillars match the preset in dc_scoring.py (step 04)
# so results from both steps are directly comparable.
# HDD proxy: max(0, 18°C minus mean annual temp) x 365; higher means colder climate,
# favourable for waste-heat reuse (same formula as step 04, percentile-ranked here).
strategy = balanced.copy()
t_mean = pd.to_numeric(strategy["avg_mean_temp_c_2021_2023"], errors="coerce")
strategy["score_hdd"] = (np.maximum(0.0, 18.0 - t_mean) * 365.0).rank(pct=True)

strategy["cost_score"] = (
    0.45 * strategy["score_electricity"] +
    0.35 * strategy["score_cooling"] +
    0.20 * strategy["score_internet"]
)
strategy["sustainability_score"] = (
    0.30 * strategy["score_renewables"] +
    0.20 * strategy["score_cooling"] +
    0.20 * strategy["score_water"] +
    0.30 * strategy["score_hdd"]
)
strategy["resilience_score"] = (
    0.40 * strategy["score_stability"] +
    0.35 * strategy["score_risk"] +
    0.15 * strategy["score_water"] +
    0.10 * strategy["score_internet"]
)

strategy_score_cols = ["cost_score", "sustainability_score", "resilience_score"]
top_strategies = (
    strategy[["country", "iso3", "region"] + strategy_score_cols]
    .dropna(subset=strategy_score_cols)
    .sort_values("cost_score", ascending=False)
    .head(15)
)
top_strategies.to_csv(TABLES_DIR / "definition3_top15_strategies.csv", index=False)


# 6. Figure 1: top 20 countries by balanced suitability score
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_balanced["country"][::-1], top_balanced["balanced_suitability_score"][::-1], color="steelblue")
ax.set_xlabel("Balanced suitability score (avg percentile rank, min 5/7 dimensions)")
ax.set_title("Top 20 countries by balanced suitability (Definition 2)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_top20_balanced.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 01_top20_balanced.png")


# 7. Figure 2: top 10 across the three strategy scores
top10_base = strategy.dropna(subset=strategy_score_cols).nlargest(10, "balanced_suitability_score")
melted = top10_base.melt(
    id_vars=["country"], value_vars=strategy_score_cols,
    var_name="strategy", value_name="score"
)
melted["strategy"] = melted["strategy"].str.replace("_score", "").str.replace("_", " ")

fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=melted, x="country", y="score", hue="strategy", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_ylabel("Strategy score")
ax.set_title("Top 10 countries: comparison across strategy-specific scores (Definition 3)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_top10_strategies.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_top10_strategies.png")

print(f"\nDone. Figures: {FIGURES_DIR}")
print(f"      Tables:  {TABLES_DIR}")
