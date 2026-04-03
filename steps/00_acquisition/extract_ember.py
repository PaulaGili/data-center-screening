# Filters renewable electricity share per country from the Ember Climate bulk CSV.
# The input file needs to be downloaded manually from ember-climate.org and saved
# as data/raw/yearly_full_release_long_format.csv
# Output: data/raw/ember_energy_raw.csv

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.paths import DATA_RAW

INPUT_FILE = DATA_RAW / "yearly_full_release_long_format.csv"
OUTPUT_FILE = DATA_RAW / "ember_energy_raw.csv"
LATEST_YEAR = 2025


if __name__ == "__main__":
    print(f"Loading {INPUT_FILE.name}...")
    df = pd.read_csv(INPUT_FILE)

    mask = (
        (df["Category"] == "Electricity generation")
        & (df["Variable"] == "Renewables")
        & (df["Unit"] == "%")
        & (df["Year"] == LATEST_YEAR)
        & (df["Area type"] == "Country or economy")
    )

    out = (
        df[mask][["Area", "ISO 3 code", "Year", "Value"]]
        .copy()
        .rename(columns={"Area": "country_name", "ISO 3 code": "iso3", "Value": "renewable_share_pct"})
        .dropna(subset=["renewable_share_pct", "iso3"])
    )
    out["renewable_share_pct"] = out["renewable_share_pct"].round(2)
    out = out.sort_values("renewable_share_pct", ascending=False).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out)} countries to {OUTPUT_FILE}")
    print(out[["country_name", "iso3", "renewable_share_pct"]].head(10).to_string(index=False))
