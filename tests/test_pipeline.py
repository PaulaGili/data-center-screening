# Basic integrity checks for the scoring pipeline.
# Run from project root: python tests/test_pipeline.py
# Or with pytest:        pytest tests/

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import SCORED_CSV, MERGED_CSV
from src.dc_scoring import NORM_COLUMNS


def test_scored_csv_exists():
    assert SCORED_CSV.exists(), (
        f"scored_countries.csv not found at {SCORED_CSV}. "
        "Run: python steps/04_scoring/scoring.py"
    )


def test_scored_csv_columns():
    df = pd.read_csv(SCORED_CSV)
    required = ["iso3", "country"] + NORM_COLUMNS
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing columns in scored_countries.csv: {missing}"


def test_norm_columns_in_range():
    df = pd.read_csv(SCORED_CSV)
    for col in NORM_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        assert (values >= 0).all() and (values <= 1).all(), (
            f"{col} has values outside [0, 1]: min={values.min():.4f}, max={values.max():.4f}"
        )


def test_iso3_unique_in_scored():
    df = pd.read_csv(SCORED_CSV)
    duplicates = df["iso3"].duplicated().sum()
    assert duplicates == 0, f"scored_countries.csv has {duplicates} duplicate ISO3 values"


def test_scored_csv_row_count():
    df = pd.read_csv(SCORED_CSV)
    assert len(df) >= 100, f"Expected at least 100 countries, got {len(df)}"


def test_merged_csv_exists():
    assert MERGED_CSV.exists(), (
        f"merged_country_dataset.csv not found at {MERGED_CSV}. "
        "Run: python steps/01_merge/merge_datasets.py"
    )


def test_merged_iso3_unique():
    df = pd.read_csv(MERGED_CSV)
    duplicates = df["iso3"].duplicated().sum()
    assert duplicates == 0, f"merged_country_dataset.csv has {duplicates} duplicate ISO3 values"


def test_merged_row_count():
    df = pd.read_csv(MERGED_CSV)
    assert len(df) >= 100, f"Expected at least 100 countries in merged dataset, got {len(df)}"


if __name__ == "__main__":
    tests = [
        test_scored_csv_exists,
        test_scored_csv_columns,
        test_norm_columns_in_range,
        test_iso3_unique_in_scored,
        test_scored_csv_row_count,
        test_merged_csv_exists,
        test_merged_iso3_unique,
        test_merged_row_count,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
