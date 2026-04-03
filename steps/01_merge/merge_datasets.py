# Joins the six raw sources into a single country-level table.
# Climate dataset is the base because it has the broadest coverage.
# All joins are left on ISO3; missing values stay as NaN rather than dropping rows.

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path.cwd().resolve()
for _root in [_here, *_here.parents]:
    if (_root / "src" / "paths.py").exists():
        PROJECT_ROOT = _root
        break
else:
    raise FileNotFoundError("Run from the project root.")
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import DATA_RAW, DATA_PROCESSED

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

climate_path = DATA_RAW / "ai_datacenter_climate_country_screening.csv"
dc_path = DATA_RAW / "data-center-dataset-kaggle-with-iso3.csv"
political_path = DATA_RAW / "PoliticalStability_WorldBank.csv"
petrol_path = DATA_RAW / "globalpetrol_energy_raw.csv"
ember_path = DATA_RAW / "ember_energy_raw.csv"
devindi_path = DATA_RAW / "DevIndi_WorldBank.csv"


# 1. Load datasets
climate = pd.read_csv(climate_path)
data_center = pd.read_csv(dc_path)
political = pd.read_csv(political_path, sep=";")
petrol = pd.read_csv(petrol_path, sep=";")
devindi = pd.read_csv(devindi_path)

print("climate:", climate.shape)
print("data_center:", data_center.shape)
print("political:", political.shape)
print("petrol:", petrol.shape)
print("devindi:", devindi.shape)


# 2. Standardize ISO3 keys
data_center = data_center.drop(columns=["Unnamed: 19", "Unnamed: 20"], errors="ignore")

climate["iso3"] = climate["iso3"].astype(str).str.strip().str.upper()
data_center["iso3"] = data_center["iso3_country_code"].astype(str).str.strip().str.upper()
political["iso3"] = political["Economy (code)"].astype(str).str.strip().str.upper()
petrol["iso3"] = petrol["iso3"].astype(str).str.strip().str.upper()
devindi["iso3"] = devindi["Country Code"].astype(str).str.strip().str.upper()


# 3. Parse Ember file manually
# A few country names contain an internal semicolon (e.g. "Palestine; State of"),
# which breaks a standard CSV read. We handle those rows explicitly.
rows = []
with open(str(ember_path), "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f, delimiter=";")
    next(reader)  # skip header
    for row in reader:
        if not row or all(x == "" for x in row):
            continue
        if row[-1] == "":
            row = row[:-1]
        if len(row) == 4:
            country_name, iso3, year, renewable_share_pct = row
        elif len(row) == 5:
            country_name = row[0] + ";" + row[1]
            iso3, year, renewable_share_pct = row[2], row[3], row[4]
        else:
            country_name = ";".join(row[:-3])
            iso3, year, renewable_share_pct = row[-3], row[-2], row[-1]
        rows.append([country_name.strip(), iso3.strip().upper(), year, renewable_share_pct])

ember = pd.DataFrame(rows, columns=["country_name", "iso3", "ember_year", "renewable_share_pct"])
ember["ember_year"] = pd.to_numeric(ember["ember_year"], errors="coerce")
ember["renewable_share_pct"] = pd.to_numeric(ember["renewable_share_pct"], errors="coerce")
print("ember:", ember.shape)


# 4. Prepare one row per country

# Political stability: average over the 3 most recent years
political["Year"] = pd.to_numeric(political["Year"], errors="coerce")
latest_year = int(political["Year"].max())
recent_years = [latest_year - 2, latest_year - 1, latest_year]

political_recent = political[political["Year"].isin(recent_years)].copy()
for col in ["Governance estimate (approx. -2.5 to +2.5)", "Governance score (0-100)"]:
    political_recent[col] = pd.to_numeric(political_recent[col], errors="coerce")

political_avg = (
    political_recent.groupby("iso3", as_index=False)
    .agg(
        political_stability_year_start=("Year", "min"),
        political_stability_year_end=("Year", "max"),
        political_stability_years_used=("Year", "nunique"),
        political_stability_estimate_avg_3y=("Governance estimate (approx. -2.5 to +2.5)", "mean"),
        political_stability_score_0_100_avg_3y=("Governance score (0-100)", "mean"),
    )
)

# Data center: keep relevant columns
dc_cols = [
    "iso3", "total_data_centers", "hyperscale_data_centers", "colocation_data_centers",
    "floor_space_sqft_total", "power_capacity_MW_total", "average_renewable_energy_usage_percent",
    "tier_distribution", "key_operators", "cloud_provider", "internet_penetration_percent",
    "avg_latency_to_global_hubs_ms", "number_of_fiber_connections",
    "growth_rate_of_data_centers_percent_per_year", "cooling_technologies_common",
    "regulatory_challenges_or_limits", "disaster_recovery_sites_common",
    "green_dc_initiatives_description", "source_of_data",
]
dc_selected = data_center[dc_cols].copy()
petrol_selected = petrol[["iso3", "electricity_household_usd_kwh", "electricity_business_usd_kwh"]].copy()
ember_selected = ember[["iso3", "ember_year", "renewable_share_pct"]].copy()

# Development indicators: keep country rows only, rename year columns
devindi_clean = devindi[
    devindi["Country Name"].notna() & devindi["iso3"].str.len().eq(3)
].copy()

devindi_selected = devindi_clean[[
    "iso3", "Series Name",
    "2016 [YR2016]", "2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]",
    "2021 [YR2021]", "2022 [YR2022]", "2023 [YR2023]", "2024 [YR2024]", "2025 [YR2025]",
]].rename(columns={
    "Series Name": "devindi_series_name",
    "2016 [YR2016]": "education_expenditure_pct_2016",
    "2017 [YR2017]": "education_expenditure_pct_2017",
    "2018 [YR2018]": "education_expenditure_pct_2018",
    "2019 [YR2019]": "education_expenditure_pct_2019",
    "2020 [YR2020]": "education_expenditure_pct_2020",
    "2021 [YR2021]": "education_expenditure_pct_2021",
    "2022 [YR2022]": "education_expenditure_pct_2022",
    "2023 [YR2023]": "education_expenditure_pct_2023",
    "2024 [YR2024]": "education_expenditure_pct_2024",
    "2025 [YR2025]": "education_expenditure_pct_2025",
}).copy()

for col in devindi_selected.columns:
    if col.startswith("education_expenditure_pct_"):
        s = devindi_selected[col].astype(str).str.strip().mask(lambda x: x == "..", np.nan)
        devindi_selected[col] = pd.to_numeric(s, errors="coerce")


# 5. Merge everything on ISO3
merged = climate.merge(dc_selected, on="iso3", how="left", indicator="dc_merge_status")
merged["has_data_center_record"] = merged["dc_merge_status"].eq("both")

# Fill count-like DC columns with 0 for countries with no record
zero_fill_cols = [
    "total_data_centers", "hyperscale_data_centers", "colocation_data_centers",
    "floor_space_sqft_total", "power_capacity_MW_total",
]
for col in zero_fill_cols:
    merged[col] = merged[col].astype(object)
    merged.loc[~merged["has_data_center_record"], col] = "0"

merged["total_data_centers"] = pd.to_numeric(merged["total_data_centers"], errors="coerce").fillna(0).astype(int)

merged = (
    merged
    .drop(columns=["dc_merge_status"])
    .merge(political_avg[["iso3", "political_stability_year_start", "political_stability_year_end",
                           "political_stability_years_used", "political_stability_estimate_avg_3y",
                           "political_stability_score_0_100_avg_3y"]], on="iso3", how="left")
    .merge(petrol_selected, on="iso3", how="left")
    .merge(ember_selected, on="iso3", how="left")
    .merge(devindi_selected, on="iso3", how="left")
    .dropna(axis=1, how="all")
)


# 6. Quality checks
print("\nFinal shape:", merged.shape)
print("Unique ISO3:", merged["iso3"].nunique())
print("Duplicate ISO3:", merged["iso3"].duplicated().any())
print("Countries without DC record:", (~merged["has_data_center_record"]).sum())
print("\nTop missing columns:")
print(merged.isna().sum().sort_values(ascending=False).head(10))


# 7. Save
output_path = DATA_PROCESSED / "merged_country_dataset.csv"
merged.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
