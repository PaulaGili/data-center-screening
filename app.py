# Interactive data center country screening tool.
# Run from project root: python -m streamlit run app.py
# Requires data/processed/scored_countries.csv; generate with: python steps/04_scoring/scoring.py

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dc_scoring import NORM_COLUMNS, PRESET_WEIGHTS, read_csv_robust, weighted_composite
from src.paths import SCORED_CSV

PILLAR_LABELS: dict[str, str] = {
    "norm_cooling":     "Cooling demand",
    "norm_water":       "Water stress",
    "norm_risk":        "Disaster risk",
    "norm_stability":   "Political stability",
    "norm_electricity": "Electricity price",
    "norm_renewables":  "Renewable share",
    "norm_internet":    "Internet penetration",
    "norm_hdd":         "Cold climate (HDD)",
}

SLIDER_LABELS = [
    "Cooling demand (CDD): lower is better",
    "Water stress: lower is better",
    "WorldRisk index: lower is better",
    "Political stability: higher is better",
    "Business electricity price: lower is better",
    "Renewable electricity share: higher is better",
    "Internet penetration: higher is better",
    "HDD proxy (cold climate, heating demand): higher favours waste-heat reuse",
]

PRESET_DISPLAY = {
    "Custom (use sliders)": None,
    "Equally balanced (8 factors)": "equal",
    "Cost-oriented": "cost",
    "Sustainability (+ district heat proxy)": "sustainability",
    "Resilience-oriented": "resilience",
}

PRESET_SCORE_COLS = [
    "score_equal_balanced",
    "score_cost",
    "score_sustainability",
    "score_resilience",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not SCORED_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SCORED_CSV}. Run: python steps/04_scoring/scoring.py"
        )
    df = read_csv_robust(SCORED_CSV)

    # Pre-compute robustness: how many preset top-10s does each country appear in?
    available_preset_cols = [c for c in PRESET_SCORE_COLS if c in df.columns]
    n_top10 = pd.Series(0, index=df.index, dtype=int)
    for col in available_preset_cols:
        top10_iso3 = df.dropna(subset=[col]).nlargest(10, col)["iso3"].values
        n_top10 += df["iso3"].isin(top10_iso3).astype(int)
    df["n_top10_presets"] = n_top10

    return df


def preset_to_slider_dict(preset_key: str) -> dict[str, float]:
    wmap = PRESET_WEIGHTS[preset_key]
    m = max(wmap.values()) if wmap else 1.0
    if m <= 0:
        m = 1.0
    return {col: (wmap.get(col, 0.0) / m) * 100.0 for col in NORM_COLUMNS}


def pillar_bar_chart(country_row: pd.Series, global_median: pd.Series) -> go.Figure:
    # Bars are green when the country beats the global median, red when below.
    labels = [PILLAR_LABELS[c] for c in NORM_COLUMNS]
    values = [float(country_row[c]) if pd.notna(country_row[c]) else None for c in NORM_COLUMNS]
    medians = [float(global_median[c]) if pd.notna(global_median[c]) else 0.0 for c in NORM_COLUMNS]

    colors = [
        "seagreen" if (v is not None and v >= m)
        else ("tomato" if v is not None else "lightgray")
        for v, m in zip(values, medians)
    ]
    bar_x = [v if v is not None else 0.0 for v in values]
    bar_text = [f"{v:.2f}" if v is not None else "n/a" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=bar_x, orientation="h",
        name=str(country_row["country"]),
        marker_color=colors,
        text=bar_text, textposition="inside", insidetextanchor="middle",
        textfont=dict(color="white", size=11),
    ))
    fig.add_trace(go.Scatter(
        y=labels, x=medians, mode="markers",
        name="Global median",
        marker=dict(symbol="line-ns", size=14, color="black",
                    line=dict(width=2.5, color="black")),
    ))
    fig.update_layout(
        height=340,
        margin=dict(l=0, r=10, t=50, b=0),
        xaxis=dict(range=[0, 1], title="Normalized score (0 = worst, 1 = best)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"{country_row['country']}: pillar scores vs global median",
    )
    return fig


def distribution_chart(scores: pd.Series, highlight_score: float | None,
                        country_name: str) -> go.Figure:
    valid = scores.dropna()
    fig = px.histogram(valid, nbins=28, color_discrete_sequence=["steelblue"])
    fig.update_traces(opacity=0.75)

    if highlight_score is not None and pd.notna(highlight_score):
        pct = float((valid <= highlight_score).mean() * 100)
        fig.add_vline(
            x=highlight_score, line_color="crimson", line_width=2.5,
            annotation_text=f"{country_name} ({pct:.0f}th pct.)",
            annotation_position="top right",
            annotation_font_color="crimson", annotation_font_size=11,
        )
    fig.update_layout(
        height=280, margin=dict(l=0, r=10, t=40, b=0),
        xaxis_title="Composite score", yaxis_title="Number of countries",
        title="Score distribution", showlegend=False,
    )
    return fig


def comparison_chart(row_a: pd.Series, row_b: pd.Series) -> go.Figure:
    # Grouped bars are easier to read than radar charts for 8 pillars.
    labels = [PILLAR_LABELS[c] for c in NORM_COLUMNS]
    vals_a = [float(row_a[c]) if pd.notna(row_a[c]) else 0.0 for c in NORM_COLUMNS]
    vals_b = [float(row_b[c]) if pd.notna(row_b[c]) else 0.0 for c in NORM_COLUMNS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=vals_a, orientation="h",
        name=str(row_a["country"]),
        marker_color="steelblue", opacity=0.88,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=vals_b, orientation="h",
        name=str(row_b["country"]),
        marker_color="crimson", opacity=0.88,
    ))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(l=0, r=10, t=50, b=0),
        xaxis=dict(range=[0, 1], title="Normalized score (0 = worst, 1 = best)"),
        title=f"{row_a['country']} vs {row_b['country']}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def frontier_chart(df: pd.DataFrame) -> go.Figure:
    # Replicates RQ3 interactively: high renewables + low market saturation.
    sub = df.dropna(subset=["renewable_share_pct", "total_data_centers"]).copy()
    if sub.empty:
        return go.Figure()
    thr_ren = float(sub["renewable_share_pct"].quantile(0.75))
    thr_dc = float(sub["total_data_centers"].quantile(0.50))
    sub["frontier"] = (
        (sub["renewable_share_pct"] >= thr_ren) &
        (sub["total_data_centers"] <= thr_dc)
    )
    sub["label"] = sub["frontier"].map({True: "Green frontier", False: "Other"})

    fig = px.scatter(
        sub, x="renewable_share_pct", y="total_data_centers",
        color="label", hover_name="country",
        hover_data={"composite": ":.3f", "renewable_share_pct": ":.1f",
                    "total_data_centers": True, "label": False},
        color_discrete_map={"Green frontier": "crimson", "Other": "steelblue"},
        opacity=0.75,
        labels={
            "renewable_share_pct": "Renewable electricity share (%)",
            "total_data_centers": "Total data centers",
        },
        title="Green frontier: top-quartile renewables with below-median data center count",
    )
    fig.add_vline(x=thr_ren, line_dash="dash", line_color="gray",
                  annotation_text=f"75th pct. ({thr_ren:.0f}%)",
                  annotation_position="bottom right")
    fig.add_hline(y=thr_dc, line_dash="dash", line_color="gray",
                  annotation_text=f"Median DCs ({thr_dc:.0f})",
                  annotation_position="top left")
    fig.update_layout(height=500, legend_title_text="")
    return fig


CSS = """<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="collapsedControl"] {visibility: visible !important;}
</style>"""


def main() -> None:
    st.set_page_config(
        page_title="DC Location Screening",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("Data center location screening")
    st.caption(
        "Composite score: weighted average of eight min-max normalized pillars. "
        "Adjust weights and filters in the sidebar."
    )

    df = load_data()
    global_median = df[NORM_COLUMNS].median()

    with st.sidebar:
        st.header("Scoring")
        preset_name = st.selectbox("Preset", list(PRESET_DISPLAY.keys()))
        preset_internal = PRESET_DISPLAY[preset_name]

        if st.session_state.get("_last_preset") != preset_name:
            st.session_state["_last_preset"] = preset_name
            new_defaults = (
                preset_to_slider_dict(preset_internal)
                if preset_internal
                else {c: 12.5 for c in NORM_COLUMNS}
            )
            for col in NORM_COLUMNS:
                st.session_state[col] = new_defaults[col]

        for col in NORM_COLUMNS:
            if col not in st.session_state:
                defaults = (
                    preset_to_slider_dict(preset_internal)
                    if preset_internal
                    else {c: 12.5 for c in NORM_COLUMNS}
                )
                st.session_state[col] = float(defaults[col])

        with st.expander("Adjust weights (0 to 100)", expanded=True):
            weights_ui: dict[str, float] = {}
            for col, label in zip(NORM_COLUMNS, SLIDER_LABELS):
                if col == "norm_hdd":
                    st.caption(
                        "HDD proxy: max(0, 18°C minus mean temp) x 365. "
                        "Higher values signal colder climates with waste-heat reuse potential."
                    )
                weights_ui[col] = float(
                    st.slider(label.split(":")[0], 0.0, 100.0, key=col)
                )

        with st.expander("Filter countries"):
            min_valid = st.slider("Minimum valid pillars (out of 8)", 0, 8, 5)
            regions = ["(all)"] + sorted(
                df["region"].dropna().astype(str).str.strip().unique().tolist()
            )
            region_pick = st.selectbox("Region", regions)
            incomes = ["(all)"] + sorted(df["income_level"].dropna().unique().tolist())
            income_pick = st.selectbox("Income level", incomes)

    w_final = {c: weights_ui[c] for c in NORM_COLUMNS}

    plot_df = df.copy()
    if region_pick != "(all)":
        plot_df = plot_df[plot_df["region"].astype(str).str.strip() == region_pick]
    if income_pick != "(all)":
        plot_df = plot_df[plot_df["income_level"] == income_pick]
    if min_valid > 0:
        plot_df = plot_df[plot_df["n_valid_norms"] >= min_valid]
    plot_df["composite"] = plot_df.apply(lambda r: weighted_composite(r, w_final), axis=1)
    plot_df = plot_df.dropna(subset=["composite"])

    if plot_df.empty:
        st.warning("No countries match the current filters. Try relaxing them.")
        return

    country_list = ["(none)"] + sorted(plot_df["country"].tolist())

    tab_map, tab_explorer, tab_ranking, tab_frontier = st.tabs([
        "World map", "Country explorer", "Ranking", "Green frontier"
    ])

    with tab_map:
        top3 = plot_df.nlargest(3, "composite").reset_index(drop=True)
        c1, c2, c3 = st.columns(3)
        for col, medal, (_, row) in zip([c1, c2, c3], ["1st", "2nd", "3rd"], top3.iterrows()):
            col.metric(medal, row["country"], f"score {row['composite']:.3f}")

        st.caption(
            f"{len(plot_df)} countries shown  |  "
            f"median score {plot_df['composite'].median():.3f}  |  "
            f"range {plot_df['composite'].min():.3f} to {plot_df['composite'].max():.3f}"
        )

        hover_labels = {c: PILLAR_LABELS[c] for c in NORM_COLUMNS}
        fig_map = px.choropleth(
            plot_df,
            locations="iso3",
            color="composite",
            hover_name="country",
            hover_data={
                "iso3": True,
                "composite": ":.3f",
                "n_valid_norms": True,
                **{c: ":.3f" for c in NORM_COLUMNS},
            },
            labels={"composite": "Score", **hover_labels},
            color_continuous_scale="RdYlGn",
            title="Composite suitability score",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=580)
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_explorer:
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Score distribution")
            highlight_pick = st.selectbox("Highlight a country", country_list, key="highlight")
            hl_score, hl_name = None, ""
            if highlight_pick != "(none)" and highlight_pick in plot_df["country"].values:
                hl_score = float(
                    plot_df.loc[plot_df["country"] == highlight_pick, "composite"].iloc[0]
                )
                hl_name = highlight_pick
            st.plotly_chart(
                distribution_chart(plot_df["composite"], hl_score, hl_name),
                use_container_width=True,
            )

        with right_col:
            st.subheader("Country profile")
            profile_default = (
                country_list.index(highlight_pick)
                if highlight_pick in country_list
                else 0
            )
            profile_pick = st.selectbox(
                "Select country", country_list, index=profile_default, key="profile"
            )
            if profile_pick != "(none)" and profile_pick in plot_df["country"].values:
                profile_row = plot_df[plot_df["country"] == profile_pick].iloc[0]
                st.plotly_chart(
                    pillar_bar_chart(profile_row, global_median), use_container_width=True
                )
            else:
                st.caption("Select a country above to see its pillar scores vs the global median.")

        st.divider()
        st.subheader("Head-to-head comparison")
        cmp_a, cmp_b = st.columns(2)
        with cmp_a:
            country_a = st.selectbox("Country A", country_list, index=1, key="cmp_a")
        with cmp_b:
            country_b = st.selectbox("Country B", country_list, index=2, key="cmp_b")

        if (
            country_a != "(none)" and country_b != "(none)"
            and country_a in plot_df["country"].values
            and country_b in plot_df["country"].values
        ):
            row_a = plot_df[plot_df["country"] == country_a].iloc[0]
            row_b = plot_df[plot_df["country"] == country_b].iloc[0]
            st.plotly_chart(comparison_chart(row_a, row_b), use_container_width=True)
        else:
            st.caption("Select two countries above to see their radar comparison.")

    with tab_ranking:
        n_robust = int((plot_df["n_top10_presets"] == len(PRESET_SCORE_COLS)).sum())
        if n_robust > 0:
            robust_names = ", ".join(
                plot_df[plot_df["n_top10_presets"] == len(PRESET_SCORE_COLS)]["country"].tolist()
            )
            st.info(
                f"{n_robust} {'country appears' if n_robust == 1 else 'countries appear'} "
                f"in the top 10 under all four scoring presets: **{robust_names}**"
            )

        rank_cols = (
            ["country", "iso3", "composite", "n_top10_presets", "n_valid_norms"]
            + NORM_COLUMNS
        )
        rank_cols = [c for c in rank_cols if c in plot_df.columns]
        rank_df = (
            plot_df.sort_values("composite", ascending=False)[rank_cols]
            .head(50)
            .reset_index(drop=True)
        )
        rank_df.index += 1

        gradient_cols = [c for c in ["composite"] + NORM_COLUMNS if c in rank_df.columns]
        styled = (
            rank_df.style
                   .background_gradient(subset=gradient_cols, cmap="RdYlGn", vmin=0, vmax=1)
                   .format({c: "{:.3f}" for c in gradient_cols})
        )
        if "n_top10_presets" in rank_df.columns:
            styled = styled.background_gradient(
                subset=["n_top10_presets"], cmap="RdYlGn", vmin=0, vmax=len(PRESET_SCORE_COLS)
            )
        if "n_valid_norms" in rank_df.columns:
            styled = styled.background_gradient(
                subset=["n_valid_norms"], cmap="RdYlGn", vmin=0, vmax=len(NORM_COLUMNS)
            )
        st.dataframe(styled, use_container_width=True)

        st.download_button(
            label="Download full ranking as CSV",
            data=plot_df.sort_values("composite", ascending=False)[rank_cols].to_csv(index=False),
            file_name="dc_screening_ranking.csv",
            mime="text/csv",
        )

    with tab_frontier:
        st.markdown(
            "Countries in the **green frontier** combine top-quartile renewable electricity share "
            "with below-median data center count. They offer clean energy infrastructure without "
            "the market saturation seen in established hubs."
        )
        if "renewable_share_pct" in plot_df.columns and "total_data_centers" in plot_df.columns:
            st.plotly_chart(frontier_chart(plot_df), use_container_width=True)

            frontier_sub = plot_df.dropna(
                subset=["renewable_share_pct", "total_data_centers"]
            ).copy()
            thr_ren = float(frontier_sub["renewable_share_pct"].quantile(0.75))
            thr_dc = float(frontier_sub["total_data_centers"].quantile(0.50))
            frontier_countries = frontier_sub[
                (frontier_sub["renewable_share_pct"] >= thr_ren) &
                (frontier_sub["total_data_centers"] <= thr_dc)
            ][["country", "iso3", "region", "composite",
               "renewable_share_pct", "total_data_centers"]].sort_values(
                "renewable_share_pct", ascending=False
            ).reset_index(drop=True)
            frontier_countries.index += 1

            st.subheader(f"Frontier countries ({len(frontier_countries)} total)")
            st.dataframe(
                frontier_countries.style.format(
                    {"composite": "{:.3f}", "renewable_share_pct": "{:.1f}"}
                ),
                use_container_width=True,
            )
        else:
            st.warning("Renewable share or data center count not available in the scored dataset.")


if __name__ == "__main__":
    main()
