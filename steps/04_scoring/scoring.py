# Runs the scoring pipeline and saves outputs.
# Heavy lifting is in src/dc_scoring.py; this script just calls it and makes the plots.
# Input:  data/processed/merged_country_dataset.csv
# Output: data/processed/scored_countries.csv, reports/figures/04_scoring/, reports/tables/04_scoring/

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

_here = Path.cwd().resolve()
for _root in [_here, *_here.parents]:
    if (_root / "src" / "paths.py").exists():
        PROJECT_ROOT = _root
        break
else:
    raise FileNotFoundError("Run from the project root.")
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import MERGED_CSV, SCORED_CSV, FIGURES_DIR, TABLES_DIR
from src.dc_scoring import NORM_COLUMNS, build_scored_dataframe, export_scored_csv

FIGURES_DIR = FIGURES_DIR / "04_scoring"
TABLES_DIR = TABLES_DIR / "04_scoring"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# 1. Run scoring pipeline
scored = build_scored_dataframe(str(MERGED_CSV))
export_scored_csv(scored, str(SCORED_CSV))
print("Shape:", scored.shape)
print("Saved:", SCORED_CSV)


# quick sanity check, make sure HDD proxy and normalization look right
sanity_cols = [
    "country",
    "avg_mean_temp_c_2021_2023",
    "heating_degree_days_proxy",
    "norm_hdd",
    "cooling_degree_days_65f_2021_2023",
    "norm_cooling",
]
print("\nSanity check (highest HDD countries):")
print(scored.sort_values("heating_degree_days_proxy", ascending=False)[sanity_cols].head(8).to_string(index=False))


# 3. Preset scores overview
preset_cols = [c for c in scored.columns if c.startswith("score_")]
print("\nTop 15 by equal-balanced score:")
print(
    scored[["country", "iso3", "n_valid_norms"] + preset_cols]
    .sort_values("score_equal_balanced", ascending=False)
    .head(15)
    .to_string(index=False)
)


# 4. Bar chart: Top 15 by equal-balanced score
top15 = scored.dropna(subset=["score_equal_balanced"]).nlargest(15, "score_equal_balanced")

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top15["country"][::-1], top15["score_equal_balanced"][::-1], color="steelblue")
ax.set_xlabel("Equal-balanced composite score (min 5/8 pillars)")
ax.set_title("Top 15 countries by equal-balanced score")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_top15_equal_balanced.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 01_top15_equal_balanced.png")


# 5. Grouped bar chart: Top 10 across all four scoring presets
preset_score_cols = ["score_equal_balanced", "score_cost", "score_sustainability", "score_resilience"]
top10 = scored.dropna(subset=preset_score_cols).nlargest(10, "score_equal_balanced")
melted = top10.melt(id_vars=["country"], value_vars=preset_score_cols, var_name="preset", value_name="score")
melted["preset"] = melted["preset"].str.replace("score_", "").str.replace("_", " ")

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=melted, x="country", y="score", hue="preset", ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_ylabel("Composite score")
ax.set_title("Top 10 countries: comparison across scoring presets")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_top10_presets_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_top10_presets_comparison.png")


# 6. Radar chart: Top 10, all eight normalized pillars (saved as interactive HTML)
top10_radar = scored.dropna(subset=["score_equal_balanced"]).nlargest(10, "score_equal_balanced")
labels = [c.replace("norm_", "") for c in NORM_COLUMNS]

fig = go.Figure()
for _, row in top10_radar.iterrows():
    vals = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in NORM_COLUMNS]
    vals_closed = vals + [vals[0]]
    theta_closed = labels + [labels[0]]
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=theta_closed, fill="toself", name=row["country"]
    ))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    title="Top 10 countries: normalized pillars (equal-balanced ranking)",
)
fig.write_html(str(FIGURES_DIR / "03_radar_top10_pillars.html"))
print("Saved: 03_radar_top10_pillars.html")

# Save full ranking table
preset_cols = [c for c in scored.columns if c.startswith("score_")]
ranking = (
    scored[["country", "iso3", "region", "income_level", "n_valid_norms"] + preset_cols]
    .sort_values("score_equal_balanced", ascending=False)
    .reset_index(drop=True)
)
ranking.index += 1
ranking.to_csv(TABLES_DIR / "country_ranking.csv")
print(f"\nSaved: country_ranking.csv ({len(ranking)} countries)")


# Model validation: do known DC markets rank in the top quartile?
# If the scoring is directionally correct, established hubs should score well.
# A country ranking outside the top 25% is worth investigating.
KNOWN_HUBS = ["IRL", "NLD", "SGP", "USA", "DEU", "SWE", "FIN", "DNK"]

valid_scores = scored.dropna(subset=["score_equal_balanced"])
total_ranked = len(valid_scores)
sorted_scores = valid_scores.sort_values("score_equal_balanced", ascending=False).reset_index(drop=True)
sorted_scores["rank"] = sorted_scores.index + 1

hub_rows = sorted_scores[sorted_scores["iso3"].isin(KNOWN_HUBS)].copy()
hub_rows["top_pct"] = (hub_rows["rank"] / total_ranked * 100).round(1)
hub_rows[["country", "iso3", "rank", "top_pct", "score_equal_balanced", "n_valid_norms"]].to_csv(
    TABLES_DIR / "model_validation_known_hubs.csv", index=False
)

print(f"\nModel validation: known DC markets (n={total_ranked} countries ranked)")
print(hub_rows[["country", "iso3", "rank", "top_pct", "score_equal_balanced"]].to_string(index=False))
top25_cutoff = int(total_ranked * 0.25)
n_in_top25 = int((hub_rows["rank"] <= top25_cutoff).sum())
print(f"  {n_in_top25} of {len(hub_rows)} known hubs land in the top 25% (rank <= {top25_cutoff})")

print(f"\nDone. Figures: {FIGURES_DIR}")
print(f"      Tables:  {TABLES_DIR}")
