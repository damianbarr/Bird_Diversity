import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
import json


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="Texas Bird Diversity Dashboard",
    layout="wide"
)

st.title("Texas Bird Diversity and Citizen-Science Effort Dashboard")

st.markdown(
    """
    This dashboard explores whether measured bird diversity in Texas citizen-science data
    is related to observation effort, population density, and land cover.

    **Main question:**  
    Do some areas appear more bird-diverse because they receive more observations?
    """
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    with open("tract_bird_population_landcover_map.geojson", "r") as f:
        geojson_data = json.load(f)

    records = []

    for feature in geojson_data["features"]:
        props = feature["properties"].copy()
        records.append(props)

    df = pd.DataFrame(records)

    df["GEOID"] = df["GEOID"].astype(str).str.replace(".0", "", regex=False)

    # Clean numeric columns
    numeric_cols = [
        "observations",
        "diversity",
        "population",
        "population_density",
        "total_birds"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Create log columns
    df["log_observations"] = np.log1p(df["observations"])
    df["log_diversity"] = np.log1p(df["diversity"])
    df["log_population_density"] = np.log1p(df["population_density"])

    if "nlcd_class" not in df.columns:
        df["nlcd_class"] = "Unknown"

    df["GEOID"] = df["GEOID"].astype(str)

    return df, geojson_data


tracts, geojson_data = load_data()

st.write("Rows loaded:", len(tracts))
st.write("GeoJSON features loaded:", len(geojson_data["features"]))
st.write(tracts.head())


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.header("Dashboard Controls")

available_landcovers = sorted(tracts["nlcd_class"].dropna().unique())

selected_landcovers = st.sidebar.multiselect(
    "Select land-cover types",
    options=available_landcovers,
    default=available_landcovers
)

min_obs, max_obs = int(tracts["observations"].min()), int(tracts["observations"].max())

obs_range = st.sidebar.slider(
    "Observation count range",
    min_value=min_obs,
    max_value=max_obs,
    value=(min_obs, max_obs)
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
    (tracts["observations"] >= obs_range[0]) &
    (tracts["observations"] <= obs_range[1])
].copy()

# --------------------------------------------------
# Interactive map
# --------------------------------------------------

st.subheader("Interactive Map")

st.markdown(
    """
    Use the sidebar to change the mapped variable.  
    Log-transformed variables are useful because observation effort and population density are highly skewed.
    """
)

# Center map on Texas
m = folium.Map(
    location=[31.0, -99.0],
    zoom_start=6,
    tiles="cartodbpositron"
)

# Continuous variable map
if map_variable != "nlcd_class":

    values = filtered[map_variable].replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) > 0:
        vmin = float(values.quantile(0.05))
        vmax = float(values.quantile(0.95))

        if vmin == vmax:
            vmin = float(values.min())
            vmax = float(values.max())

        colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
        colormap.caption = map_variable.replace("_", " ").title()
        colormap.add_to(m)

        def style_function(feature):
            value = feature["properties"].get(map_variable)

            if value is None:
                color = "#d9d9d9"
            else:
                try:
                    value = float(value)
                    color = colormap(value)
                except Exception:
                    color = "#d9d9d9"

            return {
                "fillColor": color,
                "color": "black",
                "weight": 0.2,
                "fillOpacity": 0.65
            }

    else:
        def style_function(feature):
            return {
                "fillColor": "#d9d9d9",
                "color": "black",
                "weight": 0.2,
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
        landcover = feature["properties"].get("nlcd_class")
        color = landcover_colors.get(landcover, "#d9d9d9")

        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.2,
            "fillOpacity": 0.65
        }


tooltip_fields = [
    "GEOID",
    "observations",
    "diversity",
    "population_density",
    "nlcd_class"
]

tooltip_fields = [field for field in tooltip_fields if field in filtered.columns]

tooltip_aliases = [
    "Tract GEOID:",
    "Observation effort:",
    "Measured diversity:",
    "Population density:",
    "Land cover:"
][:len(tooltip_fields)]

filtered_geoids = set(filtered["GEOID"].astype(str).str.replace(".0", "", regex=False))

filtered_features = []

for feature in geojson_data["features"]:
    feature_geoid = str(feature["properties"].get("GEOID")).replace(".0", "")
    
    if feature_geoid in filtered_geoids:
        filtered_features.append(feature)

filtered_geojson = {
    "type": "FeatureCollection",
    "features": filtered_features
}

st.write("Features shown on map:", len(filtered_features))

folium.GeoJson(
    filtered_geojson,
    style_function=style_function,
    tooltip=folium.GeoJsonTooltip(
        fields=tooltip_fields,
        aliases=tooltip_aliases,
        localize=True
    )
).add_to(m)

if len(filtered_features) > 0:
    bounds = []

    for feature in filtered_features:
        geom = feature["geometry"]
        
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            for lon, lat in coords:
                bounds.append([lat, lon])
        
        elif geom["type"] == "MultiPolygon":
            for polygon in geom["coordinates"]:
                coords = polygon[0]
                for lon, lat in coords:
                    bounds.append([lat, lon])

    if bounds:
        m.fit_bounds(bounds)

st_folium(m, use_container_width=True, height=650)


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
    st.markdown("### Observation Effort, Population Density, and Measured Diversity")

    scatter_choice = st.selectbox(
        "Choose scatterplot",
        [
            "Log observation effort vs measured diversity",
            "Log population density vs log observation effort",
            "Log population density vs measured diversity"
        ]
    )

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

    ax.set_title("Correlation Heatmap: Effort, Diversity, Population Density, and Land Cover")

    st.pyplot(fig)



# --------------------------------------------------
# Interpretation
# --------------------------------------------------

st.subheader("Interpretation Guide")

st.markdown(
    """
    - **Observation effort** is the number of bird records in each Census tract.
    - **Measured diversity** is the number of unique bird species recorded in each tract.
    - **Population density** is Census tract population divided by tract area in square kilometers.
    - **Land cover** comes from the dominant NLCD class in each tract.

    A tract with high measured diversity may truly have many species, but it may also have more recorded species
    because it received more observation effort.
    """
)
