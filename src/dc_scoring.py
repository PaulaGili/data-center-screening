# Shared scoring logic used by steps 04 through 06 and the Streamlit app.
# Min-max normalization across all countries, direction-aware.

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


def read_csv_robust(path: Union[str, Path]) -> pd.DataFrame:
    # pd.read_csv was throwing a numpy Index bug in some environments,
    # so we use stdlib csv instead. Bit verbose but reliable.
    path_str = os.fspath(Path(path).expanduser().resolve())
    with open(path_str, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        rows = list(reader)
    if not header:
        return pd.DataFrame()
    width = len(header)
    header_strs = [str(h) for h in header]
    norm_rows: List[List[str]] = []
    for r in rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        norm_rows.append(r)
    df = pd.DataFrame(norm_rows, columns=header_strs)
    df = df.replace("", np.nan)
    for c in df.columns:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        unparsable = s.notna() & coerced.isna()
        if not unparsable.any():
            df[c] = coerced
    return df


# (raw column, direction, normalized column)
# "lower" → low raw = good (e.g. electricity price, cooling demand)
# "higher" → high raw = good (e.g. renewables, stability)
FEATURE_CONFIG: List[Tuple[str, str, str]] = [
    ("cooling_degree_days_65f_2021_2023", "lower", "norm_cooling"),
    ("water_stress_pct_latest", "lower", "norm_water"),
    ("worldrisk_index_2025", "lower", "norm_risk"),
    ("political_stability_score_0_100_avg_3y", "higher", "norm_stability"),
    ("electricity_business_usd_kwh", "lower", "norm_electricity"),
    ("renewable_share_pct", "higher", "norm_renewables"),
    ("internet_penetration_pct_clean", "higher", "norm_internet"),
    ("heating_degree_days_proxy", "higher", "norm_hdd"),
]

NORM_COLUMNS: List[str] = [t[2] for t in FEATURE_CONFIG]

# 5 of 8; we tried 4 but it let in too many data-poor countries
MIN_VALID_NORMS_EQUAL = 5

# 18°C is the standard balance-point temperature (EN ISO 15927-6)
HDD_BASE_TEMP_C = 18.0


def extract_internet_penetration(value) -> float:
    # Some rows have values like "6190%" which are clearly per-mille mistakes
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in ("unknown", "none", "nan", ""):
        return np.nan
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return np.nan
    x = float(m.group())
    if x > 100:
        print(f"[data fix] internet_penetration: {value!r} looks like a per-mille value, dividing by 100 -> {x/100:.1f}%")
        x = x / 100.0
    return x


def load_merged_dataset(csv_path: str) -> pd.DataFrame:
    df = read_csv_robust(csv_path)
    edu_cols = [c for c in df.columns if "education_expenditure" in c.lower()]
    df = df.drop(columns=edu_cols, errors="ignore")
    return df


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["internet_penetration_pct_clean"] = out["internet_penetration_percent"].apply(
        extract_internet_penetration
    )
    # HDD proxy: colder mean annual temp → more heating demand → better for waste-heat reuse
    t_mean = pd.to_numeric(out["avg_mean_temp_c_2021_2023"], errors="coerce")
    out["heating_degree_days_proxy"] = np.maximum(0.0, HDD_BASE_TEMP_C - t_mean) * 365.0
    return out


def minmax_normalize_series(series: pd.Series, direction: str) -> pd.Series:
    # 1 = best for siting, 0 = worst
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if hi == lo:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out[s.notna()] = 1.0
        return out
    linear = (s - lo) / (hi - lo)
    linear = linear.clip(0.0, 1.0)
    if direction == "lower":
        return 1.0 - linear
    return linear


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw_col, direction, norm_col in FEATURE_CONFIG:
        out[norm_col] = minmax_normalize_series(out[raw_col], direction)
    return out


def count_valid_norms(row: pd.Series) -> int:
    return int(sum(pd.notna(row[c]) for c in NORM_COLUMNS))


def weighted_composite(row: pd.Series, weights: Dict[str, float]) -> float:
    # Weighted average over available pillars only.
    num = 0.0
    den = 0.0
    for col, w in weights.items():
        if w <= 0 or col not in NORM_COLUMNS:
            continue
        v = row[col]
        if pd.notna(v):
            num += float(w) * float(v)
            den += float(w)
    if den <= 0:
        return np.nan
    return num / den


# Preset weight configurations
# Weights don't need to sum to 1, the composite normalizes automatically
PRESET_WEIGHTS: Dict[str, Dict[str, float]] = {
    "equal": {c: 1.0 for c in NORM_COLUMNS},
    "cost": {
        "norm_electricity": 0.45,
        "norm_cooling": 0.35,
        "norm_internet": 0.20,
    },
    "sustainability": {
        # renewables and HDD equally dominant: green energy and waste-heat reuse narrative
        "norm_renewables": 0.30,
        "norm_cooling":    0.20,
        "norm_water":      0.20,
        "norm_hdd":        0.30,
    },
    "resilience": {
        "norm_stability": 0.40,
        "norm_risk": 0.35,
        "norm_water": 0.15,
        "norm_internet": 0.10,
    },
}


def score_equal_balanced(row: pd.Series) -> float:
    vals = [row[c] for c in NORM_COLUMNS if pd.notna(row[c])]
    if len(vals) < MIN_VALID_NORMS_EQUAL:
        return np.nan
    return float(np.mean(vals))


def add_all_preset_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_equal_balanced"] = out.apply(score_equal_balanced, axis=1)
    for name, wmap in PRESET_WEIGHTS.items():
        if name == "equal":
            continue

        def _row_score(r: pd.Series, wm=wmap) -> float:
            needed = [c for c, w in wm.items() if w > 0]
            present = sum(1 for c in needed if pd.notna(r[c]))
            if present < max(1, (len(needed) + 1) // 2):
                return np.nan
            return weighted_composite(r, wm)

        out[f"score_{name}"] = out.apply(_row_score, axis=1)
    return out


def build_scored_dataframe(merged_csv_path: str) -> pd.DataFrame:
    df = load_merged_dataset(merged_csv_path)
    df = add_engineered_columns(df)
    df = add_normalized_columns(df)
    df = add_all_preset_scores(df)
    df["n_valid_norms"] = df.apply(count_valid_norms, axis=1)
    return df


def export_scored_csv(df: pd.DataFrame, path: str) -> None:
    # Slim version for Streamlit and downstream scripts, only the columns we actually use
    id_cols = ["iso3", "iso2", "country", "region", "income_level", "lat", "lon"]
    raw_extra = [
        "cooling_degree_days_65f_2021_2023",
        "water_stress_pct_latest",
        "worldrisk_index_2025",
        "political_stability_score_0_100_avg_3y",
        "electricity_business_usd_kwh",
        "renewable_share_pct",
        "internet_penetration_pct_clean",
        "heating_degree_days_proxy",
        "avg_mean_temp_c_2021_2023",
        "total_data_centers",
        "n_valid_norms",
    ]
    score_cols = [
        "score_equal_balanced",
        "score_cost",
        "score_sustainability",
        "score_resilience",
    ]
    cols = [c for c in id_cols + raw_extra + NORM_COLUMNS + score_cols if c in df.columns]
    df[cols].to_csv(path, index=False)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.paths import MERGED_CSV, SCORED_CSV

    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored = build_scored_dataframe(str(MERGED_CSV))
    export_scored_csv(scored, str(SCORED_CSV))
    print("Wrote", SCORED_CSV, "rows=", len(scored))
