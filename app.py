import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# STREAMLIT PAGE SETUP
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
    "Budget": "Mostly private rooms for 1–2 guests in modest neighbourhoods. Great for budget-conscious travellers.",
    "Standard": "Entire homes for 2–3 guests with solid amenity counts — the backbone of the Seattle market.",
    "Premium": "Spacious entire homes for 3–4 guests in desirable areas with above-average amenities.",
    "Luxury": "Large entire homes for 5+ guests in prime neighbourhoods with premium features."
}

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    df = pd.read_excel("airbnb_raw.xlsx", sheet_name="listings")

    # Clean price
    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )

    df = df[(df["price"] > 0) & (df["price"] < 1000)].copy()

    # Amenity count
    df["amenity_count"] = df["amenities"].str.count(",") + 1

    # Impute missing values
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # Room type encoding
    room_map = {
        "Entire home/apt": 3,
        "Private room": 2,
        "Shared room": 1
    }

    df["room_type_enc"] = df["room_type"].map(room_map).fillna(1)

    # Neighbourhood price proxy
    neigh_proxy = (
        df.groupby("neighbourhood_cleansed")["price"]
        .median()
        .rename("neigh_price_proxy")
    )

    df = df.join(neigh_proxy, on="neighbourhood_cleansed")

    # Clean booleans
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

    # Order clusters by average price so 0 = Budget and 3 = Luxury
    cluster_medians = {
        c: df["price"].values[labels == c].mean()
        for c in range(4)
    }

    order = sorted(cluster_medians, key=cluster_medians.get)
    remap = {old: new for new, old in enumerate(order)}

    df = df.copy()
    df["cluster"] = [remap[label] for label in labels]
    df["tier"] = df["cluster"].map(TIER_NAMES)

    return df, scaler, kmeans, remap


# ─────────────────────────────────────────────
# LOAD DATA AND MODEL
# ─────────────────────────────────────────────

DF_RAW = load_and_prepare()
DF, SCALER, KMEANS, REMAP = train_model(DF_RAW)
NEIGHBOURHOODS = sorted(DF["neighbourhood_cleansed"].unique())

# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
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
        labels={
            "price": "Nightly Price (USD)",
            "tier": "Tier"
        },
        title="Price Distribution by Tier",
        category_orders={
            "tier": ["Budget", "Standard", "Premium", "Luxury"]
        }
    )

    fig.update_layout(
        template="plotly_white",
        legend_title="Tier",
        title_font_size=16,
        height=400
    )

    return fig


def fig_cluster_scatter():
    sample = DF.sample(min(1500, len(DF)), random_state=1)

    fig = px.scatter(
        sample,
        x="accommodates",
        y="price",
        color="tier",
        color_discrete_map=TIER_COLORS,
        size="bedrooms",
        hover_data=[
            "neighbourhood_cleansed",
            "room_type",
            "amenity_count"
        ],
        labels={
            "price": "Nightly Price (USD)",
            "accommodates": "Guests"
        },
        title="Pricing Clusters: Capacity vs Price",
        category_orders={
            "tier": ["Budget", "Standard", "Premium", "Luxury"]
        },
        opacity=0.7
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        title_font_size=16
    )

    return fig


def fig_neighbourhood_pricing():
    neigh_stats = (
        DF.groupby("neighbourhood_cleansed")["price"]
        .agg(median_price="median", count="count")
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
        labels={
            "median_price": "Median Nightly Price (USD)",
            "neighbourhood_cleansed": "Neighbourhood"
        },
        title="Top 25 Neighbourhoods by Median Price",
        text="median_price"
    )

    fig.update_traces(
        texttemplate="$%{text:.0f}",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        height=650,
        title_font_size=16,
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
        color_discrete_sequence=[
            "#4CAF50",
            "#2196F3",
            "#FF9800"
        ],
        labels={
            "price": "Nightly Price (USD)",
            "room_type": "Room Type"
        },
        title="Price Distribution by Room Type",
        category_orders={
            "room_type": [
                "Entire home/apt",
                "Private room",
                "Shared room"
            ]
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False,
        title_font_size=16
    )

    return fig


def fig_tier_radar():
    metrics = []

    for tier_id, tier_name in TIER_NAMES.items():
        sub = DF[DF["tier"] == tier_name]

        if "review_scores_rating" in DF.columns:
            avg_review_score = sub["review_scores_rating"].mean() / DF["review_scores_rating"].mean()
        else:
            avg_review_score = 1

        metrics.append({
            "Tier": tier_name,
            "Median Price": sub["price"].median() / DF["price"].median(),
            "Avg Accommodates": sub["accommodates"].mean() / DF["accommodates"].mean(),
            "Avg Bedrooms": sub["bedrooms"].mean() / DF["bedrooms"].mean(),
            "Avg Bathrooms": sub["bathrooms"].mean() / DF["bathrooms"].mean(),
            "Avg Amenities": sub["amenity_count"].mean() / DF["amenity_count"].mean(),
            "Avg Review Score": avg_review_score
        })

    categories = [
        "Median Price",
        "Avg Accommodates",
        "Avg Bedrooms",
        "Avg Bathrooms",
        "Avg Amenities",
        "Avg Review Score"
    ]

    fig = go.Figure()

    for metric in metrics:
        values = [metric[category] for category in categories]
        values += [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=metric["Tier"],
                line_color=TIER_COLORS[metric["Tier"]],
                opacity=0.7
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2.5]
            )
        ),
        showlegend=True,
        title="Tier Profiles: Normalized Metrics",
        template="plotly_white",
        height=500,
        title_font_size=16
    )

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

        if "cancellation_policy" in sub.columns:
            cancel_top = sub["cancellation_policy"].value_counts().index[0]
        else:
            cancel_top = "N/A"

        if "review_scores_rating" in sub.columns:
            avg_review_score = f"{sub['review_scores_rating'].mean():.1f}"
        else:
            avg_review_score = "N/A"

        rows.append({
            "Tier": f"{TIER_EMOJIS[tier_name]} {tier_name}",
            "Listings": f"{len(sub):,}",
            "Median Price": f"${sub['price'].median():.0f}",
            "Price Range IQR": f"${sub['price'].quantile(0.25):.0f}–${sub['price'].quantile(0.75):.0f}",
            "Median Guests": f"{sub['accommodates'].median():.0f}",
            "Median Beds": f"{sub['bedrooms'].median():.0f}",
            "Top Room Type": sub["room_type"].mode()[0],
            "Avg Amenities": f"{sub['amenity_count'].mean():.0f}",
            "Avg Review Score": avg_review_score,
            "Superhost %": f"{sub['host_is_superhost'].mean() * 100:.0f}%",
            "Instant Book %": f"{sub['instant_bookable'].mean() * 100:.0f}%",
            "Top Cancellation": cancel_top,
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
    count = len(tier_df)

    top_neigh = (
        tier_df["neighbourhood_cleansed"]
        .value_counts()
        .head(3)
        .index
        .tolist()
    )

    emoji = TIER_EMOJIS[tier_name]

    tier_median_acc = tier_df["accommodates"].median()
    tier_median_bed = tier_df["bedrooms"].median()
    tier_median_amen = tier_df["amenity_count"].median()

    neigh_rank = (
        DF.groupby("neighbourhood_cleansed")["price"]
        .median()
        .rank(ascending=False)
    )

    rank = int(neigh_rank.get(neighbourhood, len(neigh_rank)))

    result = f"""
## {emoji} You're in the **{tier_name}** Tier

{TIER_DESC[tier_name]}

---

### 💰 Pricing Recommendation

| Metric | Recommendation |
|---|---|
| Suggested anchor price | **${median_p:.0f} / night** |
| Competitive price range | **${p25:.0f} – ${p75:.0f} / night** |
| Listings in this tier | **{count:,}** |

---

### 📊 Why this tier?

- **Room type:** `{room_type}` aligns with the listing profile of the **{tier_name}** tier.
- **Capacity:** `{accommodates}` guests compared with the tier median of `{tier_median_acc:.0f}` guests.
- **Bedrooms:** `{bedrooms:.0f}` bedrooms compared with the tier median of `{tier_median_bed:.0f}`.
- **Amenities:** `{amenity_count}` amenities compared with the tier median of `{tier_median_amen:.0f}`.
- **Neighbourhood:** `{neighbourhood}` ranks **#{rank}** out of **{len(neigh_rank)}** neighbourhoods by median price.

---

### 🗺️ Top Neighbourhoods in the {tier_name} Tier

{", ".join(top_neigh)}

---

### 💡 Tips to Move Up a Tier
"""

    if tier_name != "Luxury":
        next_tier = TIER_NAMES[cluster_id + 1]
        next_df = DF[DF["tier"] == next_tier]

        result += f"""
To reach **{next_tier}** territory, where the median price is about **${next_df["price"].median():.0f}/night**, consider:

- Increasing guest capacity to around **{next_df["accommodates"].median():.0f}+ guests**
- Adding amenities to reach around **{next_df["amenity_count"].median():.0f}+ amenities**
- Improving listing quality through better photos, stronger descriptions, and flexible booking
- Listing an entire home/apartment when possible
"""
    else:
        result += """
You're already in the top tier. Focus on reviews, instant bookability, professional photos, and maintaining premium guest experience.
"""

    return result


# ─────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────

st.title("🏠 Seattle Airbnb Pricing Intelligence")
st.markdown(
    """
**Unsupervised market segmentation & data-driven pricing recommendations**  
Built on real Seattle Airbnb listings using K-Means clustering into four pricing tiers.
"""
)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Market Overview",
    "🔍 Cluster Explorer",
    "💰 Get My Price",
    "📖 Methodology"
])

# ─────────────────────────────────────────────
# TAB 1: MARKET OVERVIEW
# ─────────────────────────────────────────────

with tab1:
    st.header("📈 Seattle Airbnb Market Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Listings Analysed", f"{len(DF):,}")
    col2.metric("Median Nightly Price", f"${DF['price'].median():.0f}")
    col3.metric("Average Nightly Price", f"${DF['price'].mean():.0f}")
    col4.metric("Unique Neighbourhoods", f"{DF['neighbourhood_cleansed'].nunique()}")

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Entire Home/Apt Share",
        f"{(DF['room_type'] == 'Entire home/apt').mean() * 100:.0f}%"
    )

    col6.metric(
        "Superhost Share",
        f"{DF['host_is_superhost'].mean() * 100:.0f}%"
    )

    col7.metric(
        "Instant Bookable Share",
        f"{DF['instant_bookable'].mean() * 100:.0f}%"
    )

    st.subheader("Price Distribution by Tier")
    st.plotly_chart(fig_price_distribution(), use_container_width=True)

    st.subheader("Price Distribution by Room Type")
    st.plotly_chart(fig_price_by_room_type(), use_container_width=True)

    st.subheader("Top Neighbourhoods by Median Price")
    st.plotly_chart(fig_neighbourhood_pricing(), use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2: CLUSTER EXPLORER
# ─────────────────────────────────────────────

with tab2:
    st.header("🔍 Cluster Explorer")

    st.markdown(
        """
This section visualizes how listings are grouped into Budget, Standard, Premium, and Luxury tiers.
The clusters are based on price, capacity, property size, neighbourhood pricing, room type, and amenities.
"""
    )

    st.subheader("Capacity vs Price by Tier")
    st.plotly_chart(fig_cluster_scatter(), use_container_width=True)

    st.subheader("Tier Profiles")
    st.plotly_chart(fig_tier_radar(), use_container_width=True)

    st.subheader("Cluster Profiles Table")
    st.dataframe(
        cluster_profiles_table(),
        use_container_width=True,
        hide_index=True
    )


# ─────────────────────────────────────────────
# TAB 3: PRICING TOOL
# ─────────────────────────────────────────────

with tab3:
    st.header("💰 Get My Price")

    st.markdown(
        """
Enter listing details below to receive a personalized pricing recommendation based on the closest Airbnb market segment.
"""
    )

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("🏡 Property Details")

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

        superhost_in = st.checkbox(
            "Superhost",
            value=False
        )

        instant_in = st.checkbox(
            "Instant Bookable",
            value=True
        )

        get_price = st.button(
            "🔍 Get Pricing Recommendation",
            type="primary"
        )

    with right_col:
        if get_price:
            recommendation = recommend_price(
                room_type_in,
                accommodates_in,
                bedrooms_in,
                bathrooms_in,
                neighbourhood_in,
                amenity_count_in,
                superhost_in,
                instant_in
            )

            st.markdown(recommendation)
        else:
            st.info("Fill in your listing details and click the button to see your recommendation.")


# ─────────────────────────────────────────────
# TAB 4: METHODOLOGY
# ─────────────────────────────────────────────

with tab4:
    st.header("📖 Methodology")

    st.markdown(
        f"""
## How It Works

### 1. Data Preparation

- **{len(DF_RAW):,} Seattle listings** were filtered to prices between **$1 and $999 per night**
- `amenity_count` was derived from the comma-separated amenities field
- Missing `bedrooms` and `bathrooms` values were filled using median imputation
- `room_type` was encoded numerically:
    - Entire home/apt = 3
    - Private room = 2
    - Shared room = 1
- `neigh_price_proxy` was calculated as the median nightly price within each neighbourhood

### 2. Clustering Features

| Feature | Purpose |
|---|---|
| `price` | Pricing anchor |
| `accommodates` | Guest capacity |
| `bedrooms` | Property size |
| `bathrooms` | Property size |
| `neigh_price_proxy` | Location desirability proxy |
| `room_type_enc` | Listing format |
| `amenity_count` | Listing quality |

### 3. Modeling

- Features were standardized using **StandardScaler**
- **K-Means clustering** was fit with:
    - `k = 4`
    - `n_init = 20`
    - `random_state = 42`
- Clusters were relabeled in ascending price order:
    - Budget
    - Standard
    - Premium
    - Luxury

### 4. Recommendation Engine

- User inputs are transformed using the same fitted scaler
- The fitted K-Means model predicts the closest cluster
- The app returns the tier median price and the 25th–75th percentile price range

### Data Source

Seattle Airbnb listings scraped in 2016, originally from Inside Airbnb.
"""
    )
