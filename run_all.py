# Reproduces the full pipeline from raw data to final outputs.
# Run from the project root: python run_all.py
# Skips step 00 (acquisition) by default because it requires internet access
# and Selenium. Pass --acquire to include it.
#
# Usage:
#   python run_all.py            # merge through research questions
#   python run_all.py --acquire  # full pipeline including data download

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("01 - Merge datasets",          "steps/01_merge/merge_datasets.py"),
    ("02 - EDA",                      "steps/02_eda/eda.py"),
    ("03 - Suitability definitions",  "steps/03_suitability/suitability_definitions.py"),
    ("04 - Scoring",                  "steps/04_scoring/scoring.py"),
    ("05 - Clustering",               "steps/05_clustering/clustering.py"),
    ("06 - Research questions",       "steps/06_research_questions/research_questions.py"),
]

ACQUISITION_STEPS = [
    ("00 - Climate and WorldRisk",    "steps/00_acquisition/extract_climate_worldrisk.py"),
    ("00 - GlobalPetrol prices",      "steps/00_acquisition/extract_globalpetrol.py"),
    ("00 - Ember renewables",         "steps/00_acquisition/extract_ember.py"),
    ("00 - Political stability",      "steps/00_acquisition/extract_political_stability.py"),
    ("00 - Development indicators",   "steps/00_acquisition/extract_devindi.py"),
]


def run_step(label: str, script: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\nFailed: {script} (exit code {result.returncode})")
        return False
    print(f"\nDone in {elapsed:.1f}s")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data center screening pipeline.")
    parser.add_argument("--acquire", action="store_true",
                        help="Also run data acquisition scripts (step 00).")
    args = parser.parse_args()

    steps = (ACQUISITION_STEPS if args.acquire else []) + STEPS
    t_start = time.time()

    for label, script in steps:
        if not run_step(label, script):
            print("\nPipeline aborted.")
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All {len(steps)} steps completed in {total:.0f}s")
    print(f"  Outputs: data/processed/, reports/figures/, reports/tables/")
    print(f"  To explore results: python -m streamlit run app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
