# Data Center Location Screening

A country-level screening framework that ranks nations as potential data center locations using eight public indicators. Built as an end-to-end data pipeline with an interactive Streamlit app for real-time weight adjustment.

**[Live demo →](https://data-center-screening.streamlit.app)**

---

## What it does

Choosing a country for a new data center means balancing electricity costs, climate, water availability, political stability, renewable energy access, and connectivity. This project builds a reproducible scoring model that quantifies all of these dimensions at once and makes the trade-offs visible.

The pipeline covers the full workflow: data acquisition from six public sources, cleaning and merging, exploratory analysis, three scoring approaches, KMeans clustering, and three research questions with statistical tests. The Streamlit app lets anyone adjust the relative importance of each factor and instantly see how the global ranking changes.

---

## Features

- **Eight-pillar composite score**: cooling demand, water stress, disaster risk, political stability, electricity price, renewable share, internet penetration, and a cold-climate proxy for waste-heat reuse
- **Four strategic presets**: equally balanced, cost-oriented, sustainability-oriented, resilience-oriented
- **Interactive world map**: choropleth that updates in real time as sliders move; can display any individual pillar or the composite score
- **Country explorer**: score distribution with percentile marker, pillar profile vs global median, head-to-head comparison between two countries
- **Green frontier tab**: identifies countries with top-quartile renewables and below-median data center count
- **Gradient ranking table** with CSV export
- **Model validation**: eight established DC markets (Ireland, Netherlands, Singapore, US, Germany, Sweden, Finland, Denmark) all land in the top quartile

---

## Key findings

Under the equally balanced preset, **Iceland** (0.894), **Bhutan** (0.869), and **Norway** (0.867) lead the global ranking. Nordic and Northern European countries dominate across all four strategic presets, combining cold climates, high renewable electricity share, and strong political institutions.

KMeans clustering with k=3 cleanly separates a high-suitability group (Nordic/Northern European), a mixed mid-tier group, and a low-suitability group where multiple pillars score below average simultaneously.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data acquisition | Python, Requests, Selenium, BeautifulSoup |
| Data processing | pandas, NumPy, pycountry |
| Analysis | scikit-learn (KMeans, PCA), SciPy (Spearman) |
| Visualisation | Plotly, Matplotlib, Seaborn |
| App | Streamlit |

---

## Data sources

All sources are public and free:

- **Climate and water:** AI Datacenter Climate Country Screening dataset
- **Data center infrastructure:** Kaggle Data Center Locations dataset
- **Political stability:** World Bank Worldwide Governance Indicators
- **Electricity prices:** GlobalPetrolPrices.com (scraped with Selenium; robots.txt permits crawling)
- **Renewable electricity:** Ember Global Electricity Review
- **Development indicators:** World Bank Development Indicators API

---

## Methodology

Each variable is min-max normalised across all countries to [0, 1], with direction inverted where a lower raw value is preferable (electricity price, cooling demand, water stress). The composite score is a weighted average over available pillars; missing pillars are excluded from numerator and denominator so effective weights always sum to one.

The heating-degree-day proxy is computed as `max(0, 18°C − mean annual temperature) × 365`, following the balance-point temperature in EN ISO 15927-6.

---

## How to run

```bash
git clone https://github.com/PaulaGili/data-center-screening.git
cd data-center-screening
pip install -r requirements.txt
streamlit run app.py
```

All outputs are included in the repository. To reproduce the full pipeline from raw data:

```bash
python run_all.py
```

---

## Project structure

```
├── app.py                  # Streamlit app
├── run_all.py              # One-command pipeline runner
├── src/
│   ├── dc_scoring.py       # Shared scoring logic
│   └── paths.py            # Path constants
├── steps/
│   ├── 00_acquisition/     # Data download scripts
│   ├── 01_merge/           # Dataset integration
│   ├── 02_eda/             # Exploratory analysis
│   ├── 03_suitability/     # Three suitability definitions
│   ├── 04_scoring/         # Min-max scoring and presets
│   ├── 05_clustering/      # KMeans clustering
│   └── 06_research_questions/
├── data/processed/         # Merged and scored datasets
├── reports/                # Figures and tables (all pre-generated)
└── tests/                  # Pipeline integrity checks
```
