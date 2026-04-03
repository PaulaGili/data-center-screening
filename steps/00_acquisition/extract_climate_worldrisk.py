# Collects climate variables, water stress and disaster risk data for all countries.
# Sources: World Bank CCKP ERA5 (bulk CSV), World Bank WDI API, WorldRiskIndex (scraped)
# Output:  data/raw/ai_datacenter_climate_country_screening.csv
#          data/raw/ai_datacenter_climate_data_dictionary.csv

import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.paths import DATA_RAW




TARGET_YEARS = [2021, 2022, 2023]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0 Safari/537.36"
    )
}

CCKP_URL = "https://data360files.worldbank.org/data360-data/data/WB_CCKP/WB_CCKP_WIDEF.csv"
WRI_PAGE_URL = "https://weltrisikobericht.de/worldriskreport/"
WRI_FALLBACK = "https://weltrisikobericht.de/download/4568/?tmstv=1758617984"

CCKP_LOCAL = DATA_RAW / "WB_CCKP_WIDEF.csv"
WRI_LOCAL = DATA_RAW / "WorldRiskIndex_latest.xlsx"
OUTPUT_CSV = DATA_RAW / "ai_datacenter_climate_country_screening.csv"
OUTPUT_DICT = DATA_RAW / "ai_datacenter_climate_data_dictionary.csv"

# CCKP indicator codes → output column names
CCKP_VARIABLES = {
    "WB_CCKP_TAS": "avg_mean_temp_c_2021_2023",
    "WB_CCKP_TXX": "max_daily_max_temp_c_2021_2023",
    "WB_CCKP_CDD65": "cooling_degree_days_65f_2021_2023",
    "WB_CCKP_HI35": "days_heat_index_gt35c_2021_2023",
    "WB_CCKP_PR": "precip_mm_2021_2023",
    "WB_CCKP_CDD": "consecutive_dry_days_2021_2023",
    "WB_CCKP_R20MM": "days_precip_gt20mm_2021_2023",
    "WB_CCKP_RX5DAY": "largest_5day_precip_mm_2021_2023",
    "WB_CCKP_WSDI": "warm_spell_duration_days_2021_2023",
    "WB_CCKP_HURS": "relative_humidity_pct_2021_2023",
}

# World Bank WDI indicator codes → output column names
WDI_INDICATORS = {
    "ER.H2O.FWST.ZS": "water_stress_pct_latest",
    "ER.H2O.INTR.PC": "freshwater_m3_per_capita_latest",
}

WRI_COLUMN_MAP = {
    "Country": "country_wri",
    "ISO3": "iso3",
    "WorldRiskIndex": "worldrisk_index_2025",
    "Exposure": "worldrisk_exposure_2025",
    "Vulnerability": "worldrisk_vulnerability_2025",
    "Susceptibility": "worldrisk_susceptibility_2025",
    "Lack of Coping Capacities": "worldrisk_lack_of_coping_2025",
    "Lack of Adaptive Capacities": "worldrisk_lack_of_adaptation_2025",
}

FINAL_COLUMNS = [
    "iso3", "iso2", "country", "region", "income_level", "lending_type", "capital", "lat", "lon",
    "avg_mean_temp_c_2021_2023", "max_daily_max_temp_c_2021_2023", "cooling_degree_days_65f_2021_2023",
    "days_heat_index_gt35c_2021_2023", "precip_mm_2021_2023", "consecutive_dry_days_2021_2023",
    "days_precip_gt20mm_2021_2023", "largest_5day_precip_mm_2021_2023",
    "warm_spell_duration_days_2021_2023", "relative_humidity_pct_2021_2023",
    "water_stress_pct_latest", "water_stress_pct_latest_year",
    "freshwater_m3_per_capita_latest", "freshwater_m3_per_capita_latest_year",
    "worldrisk_exposure_2025", "worldrisk_index_2025", "worldrisk_vulnerability_2025",
    "worldrisk_susceptibility_2025", "worldrisk_lack_of_coping_2025", "worldrisk_lack_of_adaptation_2025",
]

session = requests.Session()




def download_file(url: str, dest: Path, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, headers=HEADERS, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def get_json(url: str, params: dict | None = None, timeout: int = 120, max_retries: int = 5) -> list:
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait = min(30, 2 ** attempt)
            print(f"  retry {attempt + 1}/{max_retries} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"All retries failed for {url}: {last_err}")


def scrape_wri_download_url() -> str:
    """Try to find the latest WorldRiskIndex .xlsx link on the official page."""
    try:
        html = session.get(WRI_PAGE_URL, headers=HEADERS, timeout=120).text
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True).lower()
            href = urljoin(WRI_PAGE_URL, a["href"])
            if "worldriskindex" in text and ".xlsx" in text:
                return href
            if "worldriskindex" in href.lower() and ".xlsx" in href.lower():
                return href
    except Exception as e:
        print(f"  could not scrape WRI page ({e}), using fallback URL")
    return WRI_FALLBACK


def extract_cckp_indicator(wide_df: pd.DataFrame, code: str, out_col: str) -> pd.DataFrame:
    year_cols = [str(y) for y in TARGET_YEARS if str(y) in wide_df.columns]
    sub = wide_df.loc[wide_df["INDICATOR"] == code].copy()
    if sub.empty:
        raise ValueError(f"Indicator not found in WIDEF file: {code}")
    for col in year_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    agg = sub.groupby(
        ["REF_AREA", "REF_AREA_LABEL", "INDICATOR", "INDICATOR_LABEL", "UNIT_MEASURE_LABEL"],
        as_index=False,
    )[year_cols].mean()
    agg[out_col] = agg[year_cols].mean(axis=1, skipna=True)
    return agg.rename(columns={"REF_AREA": "iso3", "REF_AREA_LABEL": "country"})[
        ["iso3", "country", "INDICATOR", "INDICATOR_LABEL", "UNIT_MEASURE_LABEL", out_col]
    ]


def fetch_worldbank_countries() -> pd.DataFrame:
    rows, page = [], 1
    while True:
        data = get_json(
            "https://api.worldbank.org/v2/country",
            params={"format": "json", "per_page": 400, "page": page},
        )
        meta, values = data
        for r in values:
            if r.get("region", {}).get("value") == "Aggregates":
                continue
            rows.append({
                "iso3": r.get("id"),
                "iso2": r.get("iso2Code"),
                "country_wb": r.get("name"),
                "region": r.get("region", {}).get("value"),
                "income_level": r.get("incomeLevel", {}).get("value"),
                "lending_type": r.get("lendingType", {}).get("value"),
                "capital": r.get("capitalCity"),
                "lat": pd.to_numeric(r.get("latitude"), errors="coerce"),
                "lon": pd.to_numeric(r.get("longitude"), errors="coerce"),
            })
        if page >= meta["pages"]:
            break
        page += 1
    return pd.DataFrame(rows)


def fetch_wdi_latest(indicator_code: str, out_col: str) -> pd.DataFrame:
    data = get_json(
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}",
        params={"format": "json", "per_page": 20000, "mrnev": 1, "gapfill": "Y"},
    )
    _, values = data
    rows = []
    for r in values:
        iso3 = r.get("countryiso3code") or r.get("country", {}).get("id")
        value = r.get("value")
        if not iso3 or value is None:
            continue
        rows.append({"iso3": iso3, out_col: value, f"{out_col}_year": r.get("date")})
    return pd.DataFrame(rows).drop_duplicates(subset=["iso3"])




if __name__ == "__main__":

    # 1. Download source files
    print("Downloading World Bank CCKP WIDEF...")
    download_file(CCKP_URL, CCKP_LOCAL)
    print(f"  → {CCKP_LOCAL.name}")

    print("Scraping WorldRiskIndex download link...")
    wri_url = scrape_wri_download_url()
    download_file(wri_url, WRI_LOCAL)
    print(f"  → {WRI_LOCAL.name}")

    # 2. Extract climate variables from CCKP
    print(f"Extracting {len(CCKP_VARIABLES)} CCKP indicators...")
    cckp_wide = pd.read_csv(CCKP_LOCAL)
    cckp_frames, dict_rows = [], []
    for code, out_col in CCKP_VARIABLES.items():
        part = extract_cckp_indicator(cckp_wide, code, out_col)
        cckp_frames.append(part[["iso3", out_col]])
        row_meta = part[["INDICATOR", "INDICATOR_LABEL", "UNIT_MEASURE_LABEL"]].iloc[0]
        dict_rows.append({
            "variable": out_col,
            "source_dataset": "World Bank CCKP ERA5 WIDEF",
            "source_code": code,
            "source_label": row_meta["INDICATOR_LABEL"],
            "unit": row_meta["UNIT_MEASURE_LABEL"],
            "definition_used": f"Mean of {TARGET_YEARS}",
        })
    climate_df = cckp_frames[0]
    for frame in cckp_frames[1:]:
        climate_df = climate_df.merge(frame, on="iso3", how="outer")
    print(f"  → {len(climate_df)} countries")

    # 3. World Bank country metadata (region, income level, lat/lon)
    print("Fetching World Bank country list...")
    countries = fetch_worldbank_countries()
    print(f"  → {len(countries)} countries")

    # 4. Water indicators via WDI API
    print("Fetching water indicators (WDI API)...")
    wdi_frames = [fetch_wdi_latest(code, col) for code, col in WDI_INDICATORS.items()]
    water_df = wdi_frames[0]
    for frame in wdi_frames[1:]:
        water_df = water_df.merge(frame, on="iso3", how="outer")
    print(f"  → {len(water_df)} countries")

    # 5. WorldRiskIndex from downloaded workbook
    print("Loading WorldRiskIndex workbook...")
    wri = pd.read_excel(WRI_LOCAL).rename(columns=WRI_COLUMN_MAP)
    wri = wri[[c for c in WRI_COLUMN_MAP.values() if c in wri.columns]]
    print(f"  → {len(wri)} countries")

    # 6. Merge all sources on ISO3
    print("Merging all sources...")
    df = (
        countries
        .merge(climate_df, on="iso3", how="left")
        .merge(water_df, on="iso3", how="left")
        .merge(wri, on="iso3", how="left")
    )
    df["country"] = df["country_wb"].fillna(df.get("country")).fillna(df.get("country_wri"))
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]].copy()
    print(f"  → final shape: {df.shape}")

    # 7. Data dictionary
    extra_dict_rows = [
        {"variable": "water_stress_pct_latest", "source_dataset": "World Bank Indicators API", "source_code": "ER.H2O.FWST.ZS", "source_label": "Level of water stress", "unit": "Percent", "definition_used": "Most recent non-empty value"},
        {"variable": "freshwater_m3_per_capita_latest", "source_dataset": "World Bank Indicators API", "source_code": "ER.H2O.INTR.PC", "source_label": "Renewable internal freshwater per capita", "unit": "m³/person", "definition_used": "Most recent non-empty value"},
        {"variable": "worldrisk_exposure_2025", "source_dataset": "WorldRiskIndex 2025", "source_code": "Exposure", "source_label": "Exposure to natural hazards", "unit": "Index", "definition_used": "2025 workbook"},
        {"variable": "worldrisk_index_2025", "source_dataset": "WorldRiskIndex 2025", "source_code": "WorldRiskIndex", "source_label": "Overall WorldRiskIndex score", "unit": "Index", "definition_used": "2025 workbook"},
    ]
    data_dict = pd.concat([pd.DataFrame(dict_rows), pd.DataFrame(extra_dict_rows)], ignore_index=True)

    # 8. Save outputs
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    data_dict.to_csv(OUTPUT_DICT, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_DICT}")
