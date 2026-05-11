import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import branca.colormap as cm


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Texas Bird Diversity Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.title(" Texas Bird Diversity and Citizen-Science Effort Dashboard")

st.markdown(
    """
    This dashboard explores whether measured bird diversity in Texas citizen-science data
    is related to observation effort, population density, and land cover.

    **Main question:**  
    Do some areas appear more bird-diverse because they receive more observations?
    """
)

# CSS to keep Folium legend from getting cut off
st.markdown(
    """
    <style>
    .leaflet-control {
        margin-right: 30px !important;
        margin-top: 30px !important;
    }
    .legend {
        max-width: 260px !important;
        white-space: normal !important;
        font-size: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Load GeoJSON without GeoPandas
# --------------------------------------------------

@st.cache_data
def load_data():
    geojson_file = "tract_streamlit_observed.geojson"

    with open(geojson_file, "r") as f:
        geojson_data = json.load(f)

    records = []

    for feature in geojson_data["features"]:
        props = feature["properties"].copy()
        records.append(props)

    df = pd.DataFrame(records)

    # Make sure GEOID is string
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.replace(".0", "", regex=False)
    else:
        df["GEOID"] = df.index.astype(str)

    # Required numeric columns
    numeric_cols = [
        "observations",
        "diversity",
        "population",
        "population_density",
        "total_birds"
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Required categorical column
    if "nlcd_class" not in df.columns:
        df["nlcd_class"] = "Unknown"

    df["nlcd_class"] = df["nlcd_class"].fillna("Unknown").astype(str)

    # Log columns
    df["log_observations"] = np.log1p(df["observations"])
    df["log_diversity"] = np.log1p(df["diversity"])
    df["log_population_density"] = np.log1p(df["population_density"])

    # Clean GeoJSON feature GEOIDs too
    for feature in geojson_data["features"]:
        if "GEOID" in feature["properties"]:
            feature["properties"]["GEOID"] = str(feature["properties"]["GEOID"]).replace(".0", "")

    return df, geojson_data


tracts, geojson_data = load_data()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

available_landcovers = sorted(tracts["nlcd_class"].dropna().unique())

selected_landcovers = st.sidebar.multiselect(
    "Select land-cover types",
    options=available_landcovers,
    default=available_landcovers
)
# Avoid huge slider problems by using log observations for filter
min_log_obs = float(tracts["log_observations"].min())
max_log_obs = float(tracts["log_observations"].max())

log_obs_range = st.sidebar.slider(
    "Log observation effort range",
    min_value=min_log_obs,
    max_value=max_log_obs,
    value=(min_log_obs, max_log_obs)
)

map_variable = st.sidebar.selectbox(
    "Choose map variable",
    options=[
        "log_observations",
        "log_diversity",
        "log_population_density",
        "observations",
        "diversity",
        "population_density",
        "nlcd_class"
    ],
    index=0
)
filtered = tracts[
    (tracts["nlcd_class"].isin(selected_landcovers)) &
    (tracts["log_observations"] >= log_obs_range[0]) &
    (tracts["log_observations"] <= log_obs_range[1])
].copy()

# --------------------------------------------------
# Build filtered GeoJSON
# --------------------------------------------------

def make_filtered_geojson(full_geojson, filtered_df):
    filtered_geoids = set(filtered_df["GEOID"].astype(str))

    filtered_features = []

    for feature in full_geojson["features"]:
        feature_geoid = str(feature["properties"].get("GEOID", "")).replace(".0", "")

        if feature_geoid in filtered_geoids:
            filtered_features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }


filtered_geojson = make_filtered_geojson(geojson_data, filtered)


# --------------------------------------------------
# Interactive map
# --------------------------------------------------
# --------------------------------------------------
# Side-by-side interactive maps
# --------------------------------------------------

st.subheader("Side-by-Side Map Comparison")

st.markdown(
    """
    Compare two spatial patterns side by side. For example, compare **observation effort**
    with **measured bird diversity**, or compare **population density** with **observation effort**.
    """
)

map_options = [
    "log_observations",
    "log_diversity",
    "log_population_density",
    "observations",
    "diversity",
    "population_density",
    "nlcd_class"
]

left_col, right_col = st.columns(2)

with left_col:
    left_map_variable = st.selectbox(
        "Left Map",
        options=map_options,
        index=0,
        key="left_map_variable"
    )

with right_col:
    right_map_variable = st.selectbox(
        "Right Map",
        options=map_options,
        index=1,
        key="right_map_variable"
    )


def make_map(map_variable, filtered_df, filtered_geojson):
    m = folium.Map(
        location=[31.0, -99.0],
        zoom_start=5,
        tiles="cartodbpositron"
    )

    if len(filtered_geojson["features"]) == 0:
        return m

    # Continuous variable map
    if map_variable != "nlcd_class":
        values = filtered_df[map_variable].replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) == 0:
            vmin, vmax = 0, 1
        else:
            vmin = float(values.quantile(0.05))
            vmax = float(values.quantile(0.95))

            if vmin == vmax:
                vmin = float(values.min())
                vmax = float(values.max())

            if vmin == vmax:
                vmax = vmin + 1

        colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)

        caption_names = {
            "log_observations": "Log Observations",
            "log_diversity": "Log Diversity",
            "log_population_density": "Log Pop. Density",
            "observations": "Observations",
            "diversity": "Diversity",
            "population_density": "Pop. Density"
        }

        colormap.caption = caption_names.get(map_variable, map_variable)
        colormap.add_to(m)

        def style_function(feature):
            value = feature["properties"].get(map_variable)

            try:
                value = float(value)
                fill_color = colormap(value)
            except Exception:
                fill_color = "#d9d9d9"

            return {
                "fillColor": fill_color,
                "color": "black",
                "weight": 0.15,
                "fillOpacity": 0.65
            }

    # Categorical land-cover map
    else:
        landcover_colors = {
            "Water": "#2b83ba",
            "Developed": "#d7191c",
            "Barren": "#bdbdbd",
            "Forest": "#1a9641",
            "Shrub/Scrub": "#a6d96a",
            "Grassland": "#ffffbf",
            "Agriculture": "#fdae61",
            "Wetlands": "#74add1",
            "Other": "#969696",
            "Unknown": "#d9d9d9"
        }

        def style_function(feature):
            landcover = feature["properties"].get("nlcd_class", "Unknown")
            fill_color = landcover_colors.get(landcover, "#d9d9d9")

            return {
                "fillColor": fill_color,
                "color": "black",
                "weight": 0.15,
                "fillOpacity": 0.65
            }

    tooltip_fields = [
        "GEOID",
        "observations",
        "diversity",
        "population_density",
        "nlcd_class"
    ]

    tooltip_aliases = [
        "Tract GEOID:",
        "Observation effort:",
        "Measured diversity:",
        "Population density:",
        "Land cover:"
    ]

    folium.GeoJson(
        filtered_geojson,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True
        )
    ).add_to(m)

    return m


if len(filtered_geojson["features"]) == 0:
    st.warning("No tracts match the selected filters.")
else:
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"### Left Map: `{left_map_variable}`")
        left_map = make_map(left_map_variable, filtered, filtered_geojson)
        st_folium(left_map, use_container_width=True, height=550, key="left_map")

    with right_col:
        st.markdown(f"### Right Map: `{right_map_variable}`")
        right_map = make_map(right_map_variable, filtered, filtered_geojson)
        st_folium(right_map, use_container_width=True, height=550, key="right_map")
# --------------------------------------------------
# Charts
# --------------------------------------------------

st.subheader("Exploratory Charts")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Scatterplots",
        "Boxplots",
        "Frequency Charts",
        "Correlation Heatmap"
    ]
)


# -------------------------
# Scatterplots
# -------------------------

with tab1:
    st.markdown("### Scatterplots")

    scatter_choice = st.selectbox(
        "Choose scatterplot",
        [
            "Log observation effort vs measured diversity",
            "Log population density vs log observation effort",
            "Log population density vs measured diversity"
        ]
    )

    if len(filtered) == 0:
        st.warning("No data available for the selected filters.")
    else:
        if scatter_choice == "Log observation effort vs measured diversity":
            fig = px.scatter(
                filtered,
                x="log_observations",
                y="diversity",
                color="nlcd_class",
                trendline="ols",
                hover_data=["GEOID", "observations", "population_density"],
                title="Log Observation Effort vs Measured Bird Diversity"
            )

            fig.update_layout(
                xaxis_title="Log Observation Effort",
                yaxis_title="Measured Bird Diversity"
            )

        elif scatter_choice == "Log population density vs log observation effort":
            fig = px.scatter(
                filtered,
                x="log_population_density",
                y="log_observations",
                color="nlcd_class",
                trendline="ols",
                hover_data=["GEOID", "observations", "population_density"],
                title="Log Population Density vs Log Observation Effort"
            )

            fig.update_layout(
                xaxis_title="Log Population Density",
                yaxis_title="Log Observation Effort"
            )

        else:
            fig = px.scatter(
                filtered,
                x="log_population_density",
                y="diversity",
                color="nlcd_class",
                trendline="ols",
                hover_data=["GEOID", "observations", "population_density"],
                title="Log Population Density vs Measured Bird Diversity"
            )

            fig.update_layout(
                xaxis_title="Log Population Density",
                yaxis_title="Measured Bird Diversity"
            )

        st.plotly_chart(fig, use_container_width=True)


# -------------------------
# Boxplots
# -------------------------

with tab2:
    st.markdown("### Boxplots by Land Cover")

    box_choice = st.selectbox(
        "Choose boxplot",
        [
            "Measured diversity by land cover",
            "Log observation effort by land cover",
            "Log population density by land cover"
        ]
    )

    if len(filtered) == 0:
        st.warning("No data available for the selected filters.")
    else:
        if box_choice == "Measured diversity by land cover":
            y_col = "diversity"
            title = "Measured Bird Diversity by Land Cover"
            y_label = "Measured Bird Diversity"

        elif box_choice == "Log observation effort by land cover":
            y_col = "log_observations"
            title = "Log Observation Effort by Land Cover"
            y_label = "Log Observation Effort"

        else:
            y_col = "log_population_density"
            title = "Log Population Density by Land Cover"
            y_label = "Log Population Density"

        fig = px.box(
            filtered,
            x="nlcd_class",
            y=y_col,
            color="nlcd_class",
            points="outliers",
            title=title
        )

        fig.update_layout(
            xaxis_title="Dominant NLCD Land Cover",
            yaxis_title=y_label,
            showlegend=False,
            xaxis_tickangle=-30
        )

        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Correlation heatmap
# -------------------------

with tab4:
    st.markdown("### Correlation Heatmap")

    if len(filtered) < 3:
        st.warning("Not enough data for a correlation heatmap.")
    else:
        heatmap_data = filtered.copy()

        landcover_dummies = pd.get_dummies(
            heatmap_data["nlcd_class"],
            prefix="LC"
        )

        numeric_cols = [
            "observations",
            "log_observations",
            "diversity",
            "population_density",
            "log_population_density"
        ]

        heatmap_table = pd.concat(
            [
                heatmap_data[numeric_cols],
                landcover_dummies
            ],
            axis=1
        )

        corr_matrix = heatmap_table.corr()

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            fmt=".2f",
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap: Effort, Diversity, Population Density, and Land Cover"
        )

        st.pyplot(fig)

# --------------------------------------------------
# Interpretation
# --------------------------------------------------

st.subheader("Interpretation Guide")

st.markdown(
    """
    - **Observation effort** is the number of bird records in each Census tract.
    - **Measured diversity** is the number of unique bird species recorded in each tract.
    - **Population density** is tract population divided by tract area in square kilometers.
    - **Land cover** is the dominant NLCD land-cover class in each tract.
    - Log-transformed variables are used because observation counts and population density are highly skewed.

    A tract with high measured diversity may truly have many species, but it may also have more recorded species
    because it received more observation effort.
    """
)
