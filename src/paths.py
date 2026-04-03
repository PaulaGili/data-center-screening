# Central path constants for the data center screening project.

from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MERGED_CSV = DATA_PROCESSED / "merged_country_dataset.csv"
SCORED_CSV = DATA_PROCESSED / "scored_countries.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
