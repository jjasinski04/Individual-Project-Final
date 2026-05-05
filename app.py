import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Seattle Airbnb Pricing Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

FEATURES = [
    "price",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "neigh_price_proxy",
    "room_type_enc",
    "amenity_count"
]

TIER_NAMES = {
    0: "Budget",
    1: "Standard",
    2: "Premium",
    3: "Luxury"
}

TIER_COLORS = {
    "Budget": "#4CAF50",
    "Standard": "#2196F3",
    "Premium": "#FF9800",
    "Luxury": "#9C27B0"
}

TIER_EMOJIS = {
    "Budget": "🟢",
    "Standard": "🔵",
    "Premium": "🟠",
    "Luxury": "🟣"
}

TIER_DESC = {
    "Budget": "Lower-priced listings, often smaller spaces or private rooms, suited for cost-conscious travelers.",
    "Standard": "Mid-market listings with balanced price, capacity, and amenities.",
    "Premium": "Higher-value listings with stronger location, capacity, and amenity profiles.",
    "Luxury": "Top-tier listings with the highest price levels, larger spaces, and premium market positioning."
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    df = pd.read_excel("airbnb_raw.xlsx", sheet_name="listings")

    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )

    df = df[(df["price"] > 0) & (df["price"] < 1000)].copy()

    df["amenity_count"] = df["amenities"].astype(str).str.count(",") + 1

    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    room_map = {
        "Entire home/apt": 3,
        "Private room": 2,
        "Shared room": 1
    }

    df["room_type_enc"] = df["room_type"].map(room_map).fillna(1)

    neigh_proxy = (
        df.groupby("neighbourhood_cleansed")["price"]
        .median()
        .rename("neigh_price_proxy")
    )

    df = df.join(neigh_proxy, on="neighbourhood_cleansed")

    df["host_is_superhost"] = (
        df["host_is_superhost"]
        .map({"t": True, "f": False, True: True, False: False})
        .fillna(False)
    )

    df["instant_bookable"] = (
        df["instant_bookable"]
        .map({"t": True, "f": False, True: True, False: False})
        .fillna(False)
    )

    df = df.dropna(subset=FEATURES)

    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    cluster_means = {
        c: df["price"].values[labels == c].mean()
        for c in range(4)
    }

    order = sorted(cluster_means, key=cluster_means.get)
    remap = {old: new for new, old in enumerate(order)}

    df = df.copy()
    df["cluster"] = [remap[label] for label in labels]
    df["tier"] = df["cluster"].map(TIER_NAMES)

    silhouette = silhouette_score(X_scaled, labels)
    inertia = kmeans.inertia_

    return df, scaler, kmeans, remap, silhouette, inertia


DF_RAW = load_and_prepare()
DF, SCALER, KMEANS, REMAP, SILHOUETTE, INERTIA = train_model(DF_RAW)
NEIGHBOURHOODS = sorted(DF["neighbourhood_cleansed"].unique())

# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def fig_price_distribution():
    fig = px.histogram(
        DF,
        x="price",
        color="tier",
        color_discrete_map=TIER_COLORS,
        nbins=60,
        barmode="overlay",
        opacity=0.75,
        title="Price Distribution by Tier",
        labels={"price": "Nightly Price", "tier": "Tier"},
        category_orders={"tier": ["Budget", "Standard", "Premium", "Luxury"]}
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def fig_cluster_scatter():
    sample = DF.sample(min(1500, len(DF)), random_state=1)

    fig = px.scatter(
        sample,
        x="accommodates",
        y="price",
        color="tier",
        size="bedrooms",
        color_discrete_map=TIER_COLORS,
        hover_data=["neighbourhood_cleansed", "room_type", "amenity_count"],
        title="Cluster Map: Guest Capacity vs Nightly Price",
        labels={"accommodates": "Guest Capacity", "price": "Nightly Price"},
        category_orders={"tier": ["Budget", "Standard", "Premium", "Luxury"]}
    )
    fig.update_layout(template="plotly_white", height=520)
    return fig


def fig_neighbourhood_pricing():
    neigh_stats = (
        DF.groupby("neighbourhood_cleansed")["price"]
        .agg(median_price="median", listing_count="count")
        .reset_index()
        .sort_values("median_price", ascending=False)
        .head(25)
    )

    fig = px.bar(
        neigh_stats,
        x="median_price",
        y="neighbourhood_cleansed",
        orientation="h",
        color="median_price",
        color_continuous_scale="Viridis",
        text="median_price",
        title="Top 25 Neighbourhoods by Median Nightly Price",
        labels={
            "median_price": "Median Nightly Price",
            "neighbourhood_cleansed": "Neighbourhood"
        }
    )

    fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=650,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed")
    )

    return fig


def fig_price_by_room_type():
    fig = px.box(
        DF,
        x="room_type",
        y="price",
        color="room_type",
        title="Price Distribution by Room Type",
        labels={"price": "Nightly Price", "room_type": "Room Type"}
    )
    fig.update_layout(template="plotly_white", height=420, showlegend=False)
    return fig


def fig_tier_radar():
    categories = [
        "Median Price",
        "Avg Accommodates",
        "Avg Bedrooms",
        "Avg Bathrooms",
        "Avg Amenities"
    ]

    fig = go.Figure()

    for tier_id, tier_name in TIER_NAMES.items():
        sub = DF[DF["tier"] == tier_name]

        values = [
            sub["price"].median() / DF["price"].median(),
            sub["accommodates"].mean() / DF["accommodates"].mean(),
            sub["bedrooms"].mean() / DF["bedrooms"].mean(),
            sub["bathrooms"].mean() / DF["bathrooms"].mean(),
            sub["amenity_count"].mean() / DF["amenity_count"].mean()
        ]

        values += [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=tier_name,
                line_color=TIER_COLORS[tier_name]
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])),
        showlegend=True,
        template="plotly_white",
        title="Normalized Tier Profiles",
        height=500
    )

    return fig


def fig_elbow():
    X = SCALER.transform(DF[FEATURES].values)
    ks = list(range(2, 8))
    inertias = []
    silhouettes = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=inertias,
            mode="lines+markers",
            name="Inertia"
        )
    )

    fig.add_vline(
        x=4,
        line_dash="dash",
        annotation_text="Selected k = 4",
        annotation_position="top right"
    )

    fig.update_layout(
        title="Elbow Method: Inertia by Number of Clusters",
        xaxis_title="Number of Clusters",
        yaxis_title="Inertia",
        template="plotly_white",
        height=420
    )

    return fig


def fig_silhouette():
    X = SCALER.transform(DF[FEATURES].values)
    ks = list(range(2, 8))
    scores = []

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X)
        scores.append(silhouette_score(X, labels))

    fig = px.bar(
        x=ks,
        y=scores,
        text=[f"{score:.3f}" for score in scores],
        title="Silhouette Score by Number of Clusters",
        labels={"x": "Number of Clusters", "y": "Silhouette Score"}
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(template="plotly_white", height=420)

    return fig


def cluster_profiles_table():
    rows = []

    for tier_id in range(4):
        tier_name = TIER_NAMES[tier_id]
        sub = DF[DF["tier"] == tier_name]

        top_neigh = (
            sub["neighbourhood_cleansed"]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )

        rows.append({
            "Tier": f"{TIER_EMOJIS[tier_name]} {tier_name}",
            "Listings": f"{len(sub):,}",
            "Median Price": f"${sub['price'].median():.0f}",
            "IQR Price Range": f"${sub['price'].quantile(0.25):.0f}–${sub['price'].quantile(0.75):.0f}",
            "Median Guests": f"{sub['accommodates'].median():.0f}",
            "Median Bedrooms": f"{sub['bedrooms'].median():.0f}",
            "Median Bathrooms": f"{sub['bathrooms'].median():.1f}",
            "Avg Amenities": f"{sub['amenity_count'].mean():.0f}",
            "Top Room Type": sub["room_type"].mode()[0],
            "Top Neighbourhoods": ", ".join(top_neigh)
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PRICING RECOMMENDATION
# ─────────────────────────────────────────────

def recommend_price(
    room_type,
    accommodates,
    bedrooms,
    bathrooms,
    neighbourhood,
    amenity_count,
    is_superhost,
    instant_book
):
    room_enc = {
        "Entire home/apt": 3,
        "Private room": 2,
        "Shared room": 1
    }[room_type]

    neigh_proxy = (
        DF.groupby("neighbourhood_cleansed")["price"]
        .median()
        .get(neighbourhood, DF["price"].median())
    )

    user_row = pd.DataFrame([{
        "price": neigh_proxy,
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "neigh_price_proxy": neigh_proxy,
        "room_type_enc": room_enc,
        "amenity_count": amenity_count
    }])

    X_user = SCALER.transform(user_row[FEATURES].values)

    raw_label = KMEANS.predict(X_user)[0]
    cluster_id = REMAP[raw_label]
    tier_name = TIER_NAMES[cluster_id]

    tier_df = DF[DF["tier"] == tier_name]

    p25 = tier_df["price"].quantile(0.25)
    p75 = tier_df["price"].quantile(0.75)
    median_p = tier_df["price"].median()

    top_neigh = (
        tier_df["neighbourhood_cleansed"]
        .value_counts()
        .head(3)
        .index
        .tolist()
    )

    return f"""
## {TIER_EMOJIS[tier_name]} Recommended Segment: **{tier_name}**

{TIER_DESC[tier_name]}

### Pricing Recommendation

| Metric | Value |
|---|---|
| Suggested Anchor Price | **${median_p:.0f} per night** |
| Competitive Range | **${p25:.0f}–${p75:.0f} per night** |
| Comparable Listings | **{len(tier_df):,} listings** |

### Why This Recommendation Makes Sense

- **Room Type:** {room_type}
- **Capacity:** {accommodates} guests
- **Bedrooms:** {bedrooms}
- **Bathrooms:** {bathrooms}
- **Amenities:** {amenity_count}
- **Neighbourhood:** {neighbourhood}
- **Common neighbourhoods in this tier:** {", ".join(top_neigh)}

### Decision Guidance

Use the anchor price as a starting point. Price closer to the lower end if the listing is new, has limited reviews, or lacks premium photos. Price closer to the upper end if the listing has strong reviews, a desirable location, high amenity count, or instant booking enabled.
"""


# ─────────────────────────────────────────────
# APP HEADER
# ─────────────────────────────────────────────

st.title("🏠 Seattle Airbnb Pricing Intelligence")
st.markdown(
    """
An **unsupervised clustering solution** that segments Seattle Airbnb listings into pricing tiers and helps hosts make more data-driven pricing decisions.
"""
)

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Executive Briefing",
    "📊 Market Analysis",
    "🔍 Cluster Explorer",
    "💰 Pricing Tool",
    "🧪 Model Diagnostics",
    "📖 PDID & Deployment"
])

# ─────────────────────────────────────────────
# TAB 1: EXECUTIVE BRIEFING
# ─────────────────────────────────────────────

with tab1:
    st.header("📌 Executive Briefing")

    st.markdown(
        """
### Business Problem

Airbnb hosts often price listings using guesswork, competitor browsing, or personal judgment. This can lead to underpricing, overpricing, or poor positioning in the market.

### Project Goal

The goal of this project is to create a deployed analytics tool that helps Airbnb hosts understand their market position and receive a data-driven pricing recommendation based on comparable Seattle listings.

### Research Questions

1. What natural pricing segments exist in the Seattle Airbnb market?
2. Which listing characteristics separate budget, standard, premium, and luxury listings?
3. How can a host use market data to set a more competitive nightly price?
4. How can clustering results be translated into a simple tool for non-technical decision makers?
"""
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Listings Analyzed", f"{len(DF):,}")
    c2.metric("Median Price", f"${DF['price'].median():.0f}")
    c3.metric("Average Price", f"${DF['price'].mean():.0f}")
    c4.metric("Clusters", "4")

    st.subheader("Decision-Maker Value")

    st.success(
        """
This app helps hosts quickly compare their listing against the market, understand which pricing tier they belong to, and choose a realistic nightly price range. Instead of manually reviewing hundreds of competing listings, the host receives an immediate recommendation based on market segments.
"""
    )

# ─────────────────────────────────────────────
# TAB 2: MARKET ANALYSIS
# ─────────────────────────────────────────────

with tab2:
    st.header("📊 Market Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Unique Neighbourhoods", f"{DF['neighbourhood_cleansed'].nunique()}")
    col2.metric("Entire Home/Apt Share", f"{(DF['room_type'] == 'Entire home/apt').mean() * 100:.0f}%")
    col3.metric("Superhost Share", f"{DF['host_is_superhost'].mean() * 100:.0f}%")
    col4.metric("Instant Bookable Share", f"{DF['instant_bookable'].mean() * 100:.0f}%")

    st.plotly_chart(fig_price_distribution(), use_container_width=True)
    st.plotly_chart(fig_price_by_room_type(), use_container_width=True)
    st.plotly_chart(fig_neighbourhood_pricing(), use_container_width=True)

# ─────────────────────────────────────────────
# TAB 3: CLUSTER EXPLORER
# ─────────────────────────────────────────────

with tab3:
    st.header("🔍 Cluster Explorer")

    st.markdown(
        """
The clustering model groups listings into four interpretable pricing tiers. These tiers are ordered by average price and labeled as Budget, Standard, Premium, and Luxury.
"""
    )

    st.plotly_chart(fig_cluster_scatter(), use_container_width=True)
    st.plotly_chart(fig_tier_radar(), use_container_width=True)

    st.subheader("Cluster Profile Summary")
    st.dataframe(cluster_profiles_table(), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# TAB 4: PRICING TOOL
# ─────────────────────────────────────────────

with tab4:
    st.header("💰 Pricing Tool")

    st.markdown(
        """
Use this tool as the live demonstration portion of the project. Enter a hypothetical Airbnb listing and the app will classify it into a pricing tier and recommend a competitive nightly price range.
"""
    )

    left_col, right_col = st.columns([1, 2])

    with left_col:
        room_type_in = st.selectbox(
            "Room Type",
            ["Entire home/apt", "Private room", "Shared room"]
        )

        accommodates_in = st.slider(
            "Guests Accommodated",
            min_value=1,
            max_value=16,
            value=3,
            step=1
        )

        bedrooms_in = st.slider(
            "Bedrooms",
            min_value=0,
            max_value=10,
            value=1,
            step=1
        )

        bathrooms_in = st.slider(
            "Bathrooms",
            min_value=0.5,
            max_value=8.0,
            value=1.0,
            step=0.5
        )

        neighbourhood_in = st.selectbox(
            "Neighbourhood",
            NEIGHBOURHOODS
        )

        amenity_count_in = st.slider(
            "Number of Amenities",
            min_value=1,
            max_value=80,
            value=20,
            step=1
        )

        superhost_in = st.checkbox("Superhost", value=False)
        instant_in = st.checkbox("Instant Bookable", value=True)

        get_price = st.button("Generate Pricing Recommendation", type="primary")

    with right_col:
        if get_price:
            st.markdown(
                recommend_price(
                    room_type_in,
                    accommodates_in,
                    bedrooms_in,
                    bathrooms_in,
                    neighbourhood_in,
                    amenity_count_in,
                    superhost_in,
                    instant_in
                )
            )
        else:
            st.info("Enter listing details and click the button to generate a recommendation.")

# ─────────────────────────────────────────────
# TAB 5: MODEL DIAGNOSTICS
# ─────────────────────────────────────────────

with tab5:
    st.header("🧪 Model Diagnostics")

    st.markdown(
        """
### Analytical Approach

This project uses an **unsupervised clustering approach** because the goal is not to predict a known label, but to discover natural pricing groups in the market.
"""
    )

    d1, d2, d3 = st.columns(3)

    d1.metric("Algorithm", "K-Means")
    d2.metric("Selected k", "4")
    d3.metric("Silhouette Score", f"{SILHOUETTE:.3f}")

    st.markdown(
        f"""
### Key Model Design Decisions

| Design Choice | Value |
|---|---|
| Algorithm | K-Means Clustering |
| Number of Clusters | 4 |
| Initialization | n_init = 20 |
| Random State | 42 |
| Scaling Method | StandardScaler |
| Inertia | {INERTIA:,.0f} |

### Features Used

| Feature | Meaning |
|---|---|
| price | Nightly price |
| accommodates | Guest capacity |
| bedrooms | Property size |
| bathrooms | Property size |
| neigh_price_proxy | Median price of the listing's neighbourhood |
| room_type_enc | Encoded room type |
| amenity_count | Count of listed amenities |
"""
    )

    st.plotly_chart(fig_elbow(), use_container_width=True)
    st.plotly_chart(fig_silhouette(), use_container_width=True)

# ─────────────────────────────────────────────
# TAB 6: PDID & DEPLOYMENT
# ─────────────────────────────────────────────

with tab6:
    st.header("📖 PDID Framework & Deployment")

    st.markdown(
        """
## Problem → Data → Insights → Deployment

### 1. Problem

The project addresses the challenge Airbnb hosts face when setting competitive nightly prices. Hosts need a simple way to understand where their listing fits in the market.

### 2. Data

The dataset contains Seattle Airbnb listings with price, room type, capacity, bedrooms, bathrooms, amenities, neighbourhood, host characteristics, and review-related fields. The data was cleaned by removing extreme prices, imputing missing bedroom and bathroom values, encoding room type, and engineering amenity count and neighbourhood price proxy.

### 3. Insights

The K-Means model segments listings into four pricing tiers. These clusters reveal how price relates to listing size, location, capacity, room type, and amenities. The results are translated into a practical recommendation engine for hosts.

### 4. Deployment

The solution is deployed as a Streamlit app. The app allows non-technical users to explore market patterns, inspect cluster profiles, evaluate model quality, and generate a personalized pricing recommendation.

---

## Final Prompt Engineered for the Solution

Create a deployed Streamlit analytics application that analyzes Seattle Airbnb listing data using an unsupervised clustering approach. The app should clean and prepare the data, engineer relevant pricing features, apply K-Means clustering, label clusters as pricing tiers, visualize the market, provide model diagnostics, and include an interactive pricing recommendation tool for Airbnb hosts. The app should be designed for non-technical decision makers and structured around the Problem–Data–Insights–Deployment framework.

---

## How a Non-Technical Stakeholder Uses This App

1. Start with the Executive Briefing to understand the business problem.
2. Use Market Analysis to understand Seattle pricing patterns.
3. Review Cluster Explorer to see the four pricing tiers.
4. Use the Pricing Tool to enter listing details.
5. Interpret the recommended price range as a decision-support estimate, not a guaranteed optimal price.

---

## Main Learnings

### Technical Learnings
- How to clean real-world listing data
- How to engineer features for clustering
- How to evaluate unsupervised models with inertia and silhouette score
- How to deploy an analytics solution using Streamlit

### Analytical Learnings
- Pricing is influenced by multiple interacting factors, not just bedrooms or location.
- Clustering can simplify a complex market into understandable decision categories.
- A model is most valuable when its results are translated into clear business actions.

### Challenges
- Cleaning inconsistent price and amenity fields
- Choosing an interpretable number of clusters
- Making model results understandable for non-technical users
- Turning analysis into a deployed decision-support app
"""
    )

    st.info("Remember to include your deployed app URL in the slide deck and submit app.py, requirements.txt, the dataset, and the final prompt engineered for the solution.")
