# Fetches education expenditure data from the World Bank Indicators API.
# SE.XPD.CTOT.ZS = current education expenditure (% of total public institution spending)
# Output: data/raw/DevIndi_WorldBank.csv

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.paths import DATA_RAW

OUTPUT_FILE = DATA_RAW / "DevIndi_WorldBank.csv"
INDICATOR = "SE.XPD.CTOT.ZS"
SERIES_NAME = "Current education expenditure, total (% of total expenditure in public institutions)"
YEARS = "2016:2025"
WB_API = f"https://api.worldbank.org/v2/country/all/indicator/{INDICATOR}"
YEAR_RANGE = range(2016, 2026)


def fetch_indicator():
    params = {"format": "json", "per_page": 1000, "date": YEARS}
    rows, page = [], 1

    while True:
        params["page"] = page
        resp = requests.get(WB_API, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break

        meta, records = data
        for r in records:
            rows.append({
                "Country Name": r["country"]["value"],
                "Country Code": r["country"]["id"],
                "year": int(r["date"]),
                "value": r["value"],
            })

        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(0.3)

    return pd.DataFrame(rows)


def pivot_wide(df):
    # reshape to one row per country, year columns like "2016 [YR2016]" to match WB export format
    wide = df.pivot_table(
        index=["Country Name", "Country Code"],
        columns="year",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns = [f"{c} [YR{c}]" if isinstance(c, int) else c for c in wide.columns]
    wide.insert(0, "Series Name", SERIES_NAME)
    wide.insert(1, "Series Code", INDICATOR)

    year_cols = [f"{y} [YR{y}]" for y in YEAR_RANGE]
    for col in year_cols:
        if col not in wide.columns:
            wide[col] = None
        wide[col] = wide[col].apply(lambda x: ".." if pd.isna(x) else x)

    return (
        wide[["Series Name", "Series Code", "Country Name", "Country Code"] + year_cols]
        .sort_values("Country Name")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print(f"Fetching {INDICATOR} ({YEARS})...")
    df_wide = pivot_wide(fetch_indicator())
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df_wide)} countries to {OUTPUT_FILE}")
