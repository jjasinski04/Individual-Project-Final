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
    "price","accommodates","bedrooms","bathrooms",
    "neigh_price_proxy","room_type_enc","amenity_count"
]

TIER_NAMES = {0:"Budget",1:"Standard",2:"Premium",3:"Luxury"}

TIER_COLORS = {
    "Budget":"#4CAF50","Standard":"#2196F3",
    "Premium":"#FF9800","Luxury":"#9C27B0"
}

TIER_EMOJIS = {"Budget":"🟢","Standard":"🔵","Premium":"🟠","Luxury":"🟣"}

TIER_DESC = {
    "Budget":"Lower-priced listings, often smaller or private rooms.",
    "Standard":"Balanced listings with moderate price and amenities.",
    "Premium":"Higher-end listings with stronger location and features.",
    "Luxury":"Top-tier listings with premium pricing and features."
}

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_and_prepare():
    df = pd.read_excel("airbnb_raw.xlsx", sheet_name="listings")

    if df["price"].dtype == object:
        df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)

    df = df[(df["price"] > 0) & (df["price"] < 1000)]

    df["amenity_count"] = df["amenities"].astype(str).str.count(",") + 1

    df["bedrooms"].fillna(df["bedrooms"].median(), inplace=True)
    df["bathrooms"].fillna(df["bathrooms"].median(), inplace=True)

    room_map = {"Entire home/apt":3,"Private room":2,"Shared room":1}
    df["room_type_enc"] = df["room_type"].map(room_map)

    neigh = df.groupby("neighbourhood_cleansed")["price"].median()
    df["neigh_price_proxy"] = df["neighbourhood_cleansed"].map(neigh)

    return df.dropna(subset=FEATURES)


@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xs)

    means = {c: df["price"][labels==c].mean() for c in range(4)}
    order = sorted(means, key=means.get)
    remap = {old:new for new, old in enumerate(order)}

    df["cluster"] = [remap[l] for l in labels]
    df["tier"] = df["cluster"].map(TIER_NAMES)

    return df, scaler, kmeans, remap


DF_RAW = load_and_prepare()
DF, SCALER, KMEANS, REMAP = train_model(DF_RAW)
NEIGHBOURHOODS = sorted(DF["neighbourhood_cleansed"].unique())

# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def fig_price_distribution():
    return px.histogram(DF, x="price", color="tier",
                        color_discrete_map=TIER_COLORS,
                        title="Price Distribution by Tier")

def fig_cluster_scatter():
    return px.scatter(
        DF.sample(min(1500,len(DF))),
        x="accommodates", y="price",
        color="tier", size="bedrooms",
        title="Clusters: Capacity vs Price"
    )

def fig_neighbourhood():
    df2 = DF.groupby("neighbourhood_cleansed")["price"].median().sort_values(ascending=False).head(20)
    return px.bar(df2, orientation="h", title="Top Neighbourhood Prices")

def fig_roomtype():
    return px.box(DF, x="room_type", y="price", title="Room Type Pricing")

# ─────────────────────────────────────────────
# RECOMMENDATION
# ─────────────────────────────────────────────

def recommend_price(room_type, accommodates, bedrooms, bathrooms, neighbourhood, amenities):

    enc = {"Entire home/apt":3,"Private room":2,"Shared room":1}[room_type]
    neigh_price = DF.groupby("neighbourhood_cleansed")["price"].median()[neighbourhood]

    row = pd.DataFrame([{
        "price":neigh_price,
        "accommodates":accommodates,
        "bedrooms":bedrooms,
        "bathrooms":bathrooms,
        "neigh_price_proxy":neigh_price,
        "room_type_enc":enc,
        "amenity_count":amenities
    }])

    cluster = REMAP[KMEANS.predict(SCALER.transform(row[FEATURES]))[0]]
    tier = TIER_NAMES[cluster]

    sub = DF[DF["tier"]==tier]

    return f"""
## {TIER_EMOJIS[tier]} {tier}

Suggested Price: **${sub["price"].median():.0f}**

Range: ${sub["price"].quantile(0.25):.0f} - ${sub["price"].quantile(0.75):.0f}

{TIER_DESC[tier]}
"""

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("🏠 Airbnb Pricing Intelligence")
st.markdown("Unsupervised clustering tool for pricing decisions")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Executive",
    "📊 Market",
    "🔍 Clusters",
    "💰 Pricing Tool",
    "🧪 Model"
])

# EXEC
with tab1:
    st.header("Executive Summary")
    st.write("Helps Airbnb hosts price listings using clustering instead of guesswork.")

    c1,c2,c3 = st.columns(3)
    c1.metric("Listings", len(DF))
    c2.metric("Median Price", f"${DF['price'].median():.0f}")
    c3.metric("Clusters", 4)

# MARKET
with tab2:
    st.plotly_chart(fig_price_distribution(), use_container_width=True)
    st.plotly_chart(fig_roomtype(), use_container_width=True)
    st.plotly_chart(fig_neighbourhood(), use_container_width=True)

# CLUSTERS
with tab3:
    st.plotly_chart(fig_cluster_scatter(), use_container_width=True)

# TOOL
with tab4:
    col1,col2 = st.columns([1,2])

    with col1:
        rt = st.selectbox("Room Type",["Entire home/apt","Private room","Shared room"])
        acc = st.slider("Guests",1,10,3)
        bed = st.slider("Bedrooms",0,5,1)
        bath = st.slider("Bathrooms",1.0,4.0,1.0)
        neigh = st.selectbox("Neighbourhood", NEIGHBOURHOODS)
        amen = st.slider("Amenities",1,50,10)

        btn = st.button("Get Price")

    with col2:
        if btn:
            st.markdown(recommend_price(rt,acc,bed,bath,neigh,amen))

# MODEL
with tab5:
    st.header("Model Info")
    st.write("K-Means clustering with 4 clusters on standardized features.")

    X = SCALER.transform(DF[FEATURES])
    score = silhouette_score(X, DF["cluster"])

    st.metric("Silhouette Score", f"{score:.3f}")
