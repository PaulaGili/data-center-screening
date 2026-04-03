# Answers the three main research questions using the merged dataset.
# RQ1: Does lower electricity cost correlate with higher data center density?
# RQ2: Do countries with lower cooling demand also tend to be more politically stable?
# RQ3: Which countries combine high renewable electricity share with low DC saturation?
# Input:  data/processed/merged_country_dataset.csv
# Output: steps/06_research_questions/figures/  (PNG + HTML)
#         steps/06_research_questions/tables/   (CSV)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats

_here = Path.cwd().resolve()
for _root in [_here, *_here.parents]:
    if (_root / "src" / "paths.py").exists():
        PROJECT_ROOT = _root
        break
else:
    raise FileNotFoundError("Run from the project root.")
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import MERGED_CSV, FIGURES_DIR, TABLES_DIR
from src.dc_scoring import add_engineered_columns, load_merged_dataset

FIGURES_DIR = FIGURES_DIR / "06_research_questions"
TABLES_DIR = TABLES_DIR / "06_research_questions"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# Load and engineer features
raw = load_merged_dataset(str(MERGED_CSV))
raw = add_engineered_columns(raw)
print("Shape:", raw.shape)


# RQ1: Power cost vs infrastructure density
# Hypothesis: a "sweet spot" exists, relatively cheap electricity with an unsaturated market
sub1 = raw.dropna(subset=["total_data_centers", "electricity_business_usd_kwh"]).copy()
sub1["log_dc"] = np.log1p(sub1["total_data_centers"])

# Spearman rather than Pearson because both variables are skewed
rho_rq1, p_rq1 = stats.spearmanr(sub1["electricity_business_usd_kwh"], sub1["log_dc"])
print(f"RQ1 Spearman rho = {rho_rq1:.3f}, p = {p_rq1:.4f}, n = {len(sub1)}")

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    sub1["electricity_business_usd_kwh"],
    sub1["log_dc"],
    c=sub1["renewable_share_pct"],
    cmap="viridis",
    alpha=0.75,
    s=36,
)
plt.colorbar(sc, ax=ax, label="Renewable share (%, Ember)")
ax.set_xlabel("Business electricity price (USD/kWh)")
ax.set_ylabel("log(1 + total data centers)")
ax.set_title("RQ1: Power cost vs infrastructure density")
ax.annotate(
    f"Spearman ρ = {rho_rq1:.3f}  (p = {p_rq1:.3f}, n = {len(sub1)})",
    xy=(0.98, 0.05), xycoords="axes fraction", ha="right", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_rq1_cost_vs_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_rq1_cost_vs_density.png")

# Sweet spot: below-median electricity price + below-median DC count (cheap + unsaturated)
thr_price = sub1["electricity_business_usd_kwh"].median()
thr_dc_rq1 = sub1["total_data_centers"].median()
sweet_spot = sub1[
    (sub1["electricity_business_usd_kwh"] <= thr_price) &
    (sub1["total_data_centers"] <= thr_dc_rq1)
][["country", "iso3", "region", "electricity_business_usd_kwh", "total_data_centers", "renewable_share_pct"]]
sweet_spot = sweet_spot.sort_values("electricity_business_usd_kwh").reset_index(drop=True)
sweet_spot.to_csv(TABLES_DIR / "rq1_cost_sweet_spot.csv", index=False)
print(f"RQ1 sweet spot: {len(sweet_spot)} countries")


# RQ2: Cooling demand vs political stability
# Countries with low cooling demand and high stability are the most favorable climate+governance combination
sub2 = raw.dropna(
    subset=["cooling_degree_days_65f_2021_2023", "political_stability_score_0_100_avg_3y"]
).copy()

rho_rq2, p_rq2 = stats.spearmanr(
    sub2["cooling_degree_days_65f_2021_2023"],
    sub2["political_stability_score_0_100_avg_3y"],
)
print(f"RQ2 Spearman rho = {rho_rq2:.3f}, p = {p_rq2:.4f}, n = {len(sub2)}")

fig = px.scatter(
    sub2,
    x="cooling_degree_days_65f_2021_2023",
    y="political_stability_score_0_100_avg_3y",
    hover_name="country",
    color="region",
    title=f"RQ2: Cooling degree days vs political stability  (Spearman ρ = {rho_rq2:.3f}, p = {p_rq2:.3f})",
    labels={
        "cooling_degree_days_65f_2021_2023": "Cooling degree days (65°F baseline, 2021-23 mean)",
        "political_stability_score_0_100_avg_3y": "Political stability score (0-100, 3-year mean)",
    },
)
fig.update_traces(marker=dict(size=8, opacity=0.7))
fig.write_html(str(FIGURES_DIR / "02_rq2_cooling_vs_stability.html"))
print("Saved: 02_rq2_cooling_vs_stability.html")

# Countries in the favorable quadrant: low cooling demand + high stability
low_cdd = sub2["cooling_degree_days_65f_2021_2023"] <= sub2["cooling_degree_days_65f_2021_2023"].median()
high_stab = sub2["political_stability_score_0_100_avg_3y"] >= sub2["political_stability_score_0_100_avg_3y"].median()
quadrant = sub2.loc[low_cdd & high_stab, ["country", "iso3", "region",
             "cooling_degree_days_65f_2021_2023", "political_stability_score_0_100_avg_3y"]]
quadrant = quadrant.sort_values("political_stability_score_0_100_avg_3y", ascending=False)
quadrant.to_csv(TABLES_DIR / "rq2_favorable_quadrant.csv", index=False)
print(f"RQ2 favorable quadrant: {len(quadrant)} countries")


# RQ3: Renewable electricity frontier
# High renewable share (top quartile) + low DC count (below median) = potential green expansion markets
sub3 = raw.dropna(subset=["renewable_share_pct", "total_data_centers"]).copy()
thr_ren = sub3["renewable_share_pct"].quantile(0.75)
thr_dc = sub3["total_data_centers"].quantile(0.50)
frontier = sub3[(sub3["renewable_share_pct"] >= thr_ren) & (sub3["total_data_centers"] <= thr_dc)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sub3["renewable_share_pct"], sub3["total_data_centers"],
           alpha=0.35, color="steelblue", s=28, label="All countries")
ax.scatter(frontier["renewable_share_pct"], frontier["total_data_centers"],
           color="crimson", s=40, label=f"Frontier (top-quartile renewables, below-median DC)")
ax.axvline(thr_ren, color="gray", ls="--", lw=1)
ax.axhline(thr_dc, color="gray", ls="--", lw=1)
ax.set_xlabel("Renewable electricity share (%, Ember)")
ax.set_ylabel("Total data centers")
ax.set_title("RQ3: Renewable electricity vs data center count")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "03_rq3_renewables_vs_dc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03_rq3_renewables_vs_dc.png")

frontier_out = (
    frontier[["country", "iso3", "region", "renewable_share_pct", "total_data_centers"]]
    .sort_values("renewable_share_pct", ascending=False)
    .reset_index(drop=True)
)
frontier_out.to_csv(TABLES_DIR / "rq3_renewable_frontier.csv", index=False)
print(f"RQ3 frontier countries: {len(frontier_out)}")
print(frontier_out.head(12).to_string(index=False))


# Supplementary: heating-demand proxy choropleth (waste-heat reuse narrative)
sub4 = raw.dropna(subset=["iso3", "heating_degree_days_proxy"])
fig = px.choropleth(
    sub4,
    locations="iso3",
    color="heating_degree_days_proxy",
    hover_name="country",
    color_continuous_scale="Blues",
    title="Heating-demand proxy: max(0, 18°C minus mean temp) x 365",
)
fig.update_layout(height=550)
fig.write_html(str(FIGURES_DIR / "04_hdd_proxy_choropleth.html"))
print("Saved: 04_hdd_proxy_choropleth.html")

# Correlation summary table for both RQs
rq_stats = pd.DataFrame([
    {
        "rq": "RQ1",
        "var_x": "electricity_business_usd_kwh",
        "var_y": "log_total_data_centers",
        "spearman_rho": round(rho_rq1, 3),
        "p_value": round(p_rq1, 4),
        "n": len(sub1),
    },
    {
        "rq": "RQ2",
        "var_x": "cooling_degree_days_65f_2021_2023",
        "var_y": "political_stability_score_0_100_avg_3y",
        "spearman_rho": round(rho_rq2, 3),
        "p_value": round(p_rq2, 4),
        "n": len(sub2),
    },
])
rq_stats.to_csv(TABLES_DIR / "rq_correlation_stats.csv", index=False)
print("Saved: rq_correlation_stats.csv")

print(f"\nDone. Figures: {FIGURES_DIR}")
print(f"      Tables:  {TABLES_DIR}")
