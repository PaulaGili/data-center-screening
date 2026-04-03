# Fetches political stability scores from the World Bank Governance Indicators API.
# PV.EST = governance estimate (-2.5 to +2.5)
# PV.PER.RNK = percentile rank (0-100)
# Output: data/raw/PoliticalStability_WorldBank.csv

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.paths import DATA_RAW

OUTPUT_FILE = DATA_RAW / "PoliticalStability_WorldBank.csv"
YEARS = "2020:2024"
WB_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}"

INDICATORS = {
    "PV.EST": "Governance estimate (approx. -2.5 to +2.5)",
    "PV.PER.RNK": "Governance score (0-100)",
}


def fetch_indicator(code):
    url = WB_API.format(indicator=code)
    params = {"format": "json", "per_page": 1000, "date": YEARS}
    rows, page = [], 1

    while True:
        params["page"] = page
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            break

        meta, records = data
        for r in records:
            if r.get("value") is None:
                continue
            rows.append({
                "iso3": r["country"]["id"],
                "country_name": r["country"]["value"],
                "year": int(r["date"]),
                "value": float(r["value"]),
            })

        if page >= meta.get("pages", 1):
            break
        page += 1
        time.sleep(0.3)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    frames = {}
    for code, col_name in INDICATORS.items():
        print(f"Fetching {code}...")
        frames[col_name] = fetch_indicator(code)

    est = frames["Governance estimate (approx. -2.5 to +2.5)"].rename(
        columns={"value": "Governance estimate (approx. -2.5 to +2.5)"}
    )
    rnk = frames["Governance score (0-100)"][["iso3", "year", "value"]].rename(
        columns={"value": "Governance score (0-100)"}
    )

    df = (
        est.merge(rnk, on=["iso3", "year"], how="outer")
        .rename(columns={"iso3": "Economy (code)", "country_name": "Economy (name)", "year": "Year"})
        .assign(**{"Governance dimension": "pv"})
    )
    df = df[[
        "Economy (code)", "Economy (name)", "Year", "Governance dimension",
        "Governance estimate (approx. -2.5 to +2.5)", "Governance score (0-100)",
    ]].sort_values(["Economy (code)", "Year"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"Saved {len(df)} rows to {OUTPUT_FILE}")
