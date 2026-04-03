# Clusters countries by their normalized pillar scores using KMeans.
# k is selected automatically via silhouette score (we also checked elbow, both agreed on k=3).
# Input:  data/processed/scored_countries.csv
# Output: reports/figures/05_clustering/, reports/tables/05_clustering/

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

_here = Path.cwd().resolve()
for _root in [_here, *_here.parents]:
    if (_root / "src" / "paths.py").exists():
        PROJECT_ROOT = _root
        break
else:
    raise FileNotFoundError("Run from the project root.")
sys.path.insert(0, str(PROJECT_ROOT))
from src.paths import SCORED_CSV, FIGURES_DIR, TABLES_DIR
from src.dc_scoring import NORM_COLUMNS, read_csv_robust

FIGURES_DIR = FIGURES_DIR / "05_clustering"
TABLES_DIR = TABLES_DIR / "05_clustering"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# load and impute; median felt safer than mean given some skewed distributions
scored = read_csv_robust(SCORED_CSV)
X = scored[NORM_COLUMNS].values
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)
print(f"Countries: {len(scored)} | Pillars: {len(NORM_COLUMNS)}")


# try k from 2 to 10, pick the one with the best silhouette
# (k=3 consistently won; k=4 gave one cluster with only ~15 countries)
Ks = range(2, 11)
inertias, silhouettes = [], []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_imp)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_imp, labels))

best_k = list(Ks)[int(np.argmax(silhouettes))]
print(f"Best k by silhouette: {best_k}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(Ks), inertias, marker="o", color="steelblue")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow curve")

axes[1].plot(list(Ks), silhouettes, marker="o", color="darkorange")
axes[1].axvline(x=best_k, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Silhouette score by k")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "01_elbow_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 01_elbow_silhouette.png")


K_FINAL = best_k  # 3 in our runs
km = KMeans(n_clusters=K_FINAL, random_state=42, n_init="auto")
scored["cluster"] = km.fit_predict(X_imp)

cluster_counts = scored["cluster"].value_counts().sort_index()
print(f"\nCluster sizes (k={K_FINAL}):")
print(cluster_counts.to_string())


# 4. Cluster profiles: mean normalized pillar per cluster
profile = scored.groupby("cluster")[NORM_COLUMNS].mean()
profile.columns = [c.replace("norm_", "") for c in profile.columns]
profile.to_csv(TABLES_DIR / "cluster_profiles.csv")
print("\nCluster profiles (mean normalized scores):")
print(profile.round(3).to_string())

# Save cluster assignments
scored[["country", "iso3", "region", "cluster", "score_equal_balanced"]].sort_values(
    ["cluster", "score_equal_balanced"], ascending=[True, False]
).to_csv(TABLES_DIR / "cluster_assignments.csv", index=False)
print("Saved: cluster_assignments.csv")


# 5. Figure 2: cluster centre heatmap
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(profile.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(len(profile.columns)))
ax.set_xticklabels(profile.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(profile)))
ax.set_yticklabels([f"Cluster {i}" for i in profile.index])
ax.set_title(f"Cluster centres: mean normalized pillars (k={K_FINAL})")
plt.colorbar(im, ax=ax, label="Mean normalized score (0=worst, 1=best)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "02_cluster_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02_cluster_heatmap.png")


# 6. Figure 3: PCA 2D projection coloured by cluster (interactive HTML)
pca = PCA(n_components=2, random_state=42)
XY = pca.fit_transform(X_imp)
scored["pc1"] = XY[:, 0]
scored["pc2"] = XY[:, 1]

var_explained = pca.explained_variance_ratio_ * 100
fig = px.scatter(
    scored,
    x="pc1", y="pc2",
    color=scored["cluster"].astype(str),
    hover_name="country",
    hover_data={"score_equal_balanced": ":.3f", "region": True},
    labels={"pc1": f"PC1 ({var_explained[0]:.1f}%)", "pc2": f"PC2 ({var_explained[1]:.1f}%)"},
    title=f"PCA projection: countries coloured by cluster (k={K_FINAL})",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.write_html(str(FIGURES_DIR / "03_pca_clusters.html"))
print("Saved: 03_pca_clusters.html")


# 7. Figure 4: choropleth world map coloured by cluster (interactive HTML)
fig = px.choropleth(
    scored,
    locations="iso3",
    color=scored["cluster"].astype(str),
    hover_name="country",
    hover_data={"score_equal_balanced": ":.3f"},
    title=f"Country clusters (k={K_FINAL})",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(height=550)
fig.write_html(str(FIGURES_DIR / "04_choropleth_clusters.html"))
print("Saved: 04_choropleth_clusters.html")

print(f"\nDone. Figures: {FIGURES_DIR}")
print(f"      Tables:  {TABLES_DIR}")
