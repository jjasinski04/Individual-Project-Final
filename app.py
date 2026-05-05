import pandas as pd
import numpy as np
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_and_prepare():
    df = pd.read_excel("listings.xlsx", sheet_name="listings")

    # Clean price
    if df["price"].dtype == object:
        df["price"] = df["price"].str.replace(r"[$,]", "", regex=True).astype(float)
    df = df[(df["price"] > 0) & (df["price"] < 1000)].copy()

    # Amenity count
    df["amenity_count"] = df["amenities"].str.count(",") + 1

    # Impute
    df["bedrooms"] = df["bedrooms"].fillna(df["bedrooms"].median())
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())

    # Room type encoding
    room_map = {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}
    df["room_type_enc"] = df["room_type"].map(room_map).fillna(1)

    # Neighbourhood price proxy
    neigh_proxy = df.groupby("neighbourhood_cleansed")["price"].median().rename("neigh_price_proxy")
    df = df.join(neigh_proxy, on="neighbourhood_cleansed")

    # Clean booleans
    df["host_is_superhost"] = df["host_is_superhost"].map({"t": True, "f": False, True: True, False: False}).fillna(False)
    df["instant_bookable"] = df["instant_bookable"].map({"t": True, "f": False, True: True, False: False}).fillna(False)

    df = df.dropna(subset=FEATURES)
    return df


FEATURES = ["price", "accommodates", "bedrooms", "bathrooms",
            "neigh_price_proxy", "room_type_enc", "amenity_count"]

TIER_NAMES   = {0: "Budget", 1: "Standard", 2: "Premium", 3: "Luxury"}
TIER_COLORS  = {"Budget": "#4CAF50", "Standard": "#2196F3",
                "Premium": "#FF9800", "Luxury": "#9C27B0"}
TIER_EMOJIS  = {"Budget": "🟢", "Standard": "🔵", "Premium": "🟠", "Luxury": "🟣"}
TIER_DESC    = {
    "Budget":   "Mostly private rooms for 1–2 guests in modest neighbourhoods. Great for budget-conscious travellers.",
    "Standard": "Entire homes for 2–3 guests with solid amenity counts — the backbone of the Seattle market.",
    "Premium":  "Spacious entire homes for 3–4 guests in desirable areas with above-average amenities.",
    "Luxury":   "Large entire homes for 5+ guests in prime neighbourhoods with premium features.",
}

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(df):
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    # Order clusters by median price so 0=Budget … 3=Luxury
    cluster_medians = {c: df["price"].values[labels == c].mean() for c in range(4)}
    order = sorted(cluster_medians, key=cluster_medians.get)
    remap = {old: new for new, old in enumerate(order)}
    df = df.copy()
    df["cluster"] = [remap[l] for l in labels]
    df["tier"]    = df["cluster"].map(TIER_NAMES)

    # Re-fit on reordered labels (for consistent centroid ordering)
    return df, scaler, kmeans, remap

# ─────────────────────────────────────────────
# INITIALISE GLOBALS
# ─────────────────────────────────────────────

print("Loading data …")
DF_RAW = load_and_prepare()
DF, SCALER, KMEANS, REMAP = train_model(DF_RAW)
NEIGHBOURHOODS = sorted(DF["neighbourhood_cleansed"].unique())
print(f"Ready — {len(DF):,} listings in {len(NEIGHBOURHOODS)} neighbourhoods.")

# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def fig_price_distribution():
    fig = px.histogram(
        DF, x="price", color="tier",
        color_discrete_map=TIER_COLORS,
        nbins=60, barmode="overlay", opacity=0.75,
        labels={"price": "Nightly Price (USD)", "tier": "Tier"},
        title="Price Distribution by Tier",
        category_orders={"tier": ["Budget", "Standard", "Premium", "Luxury"]},
    )
    fig.update_layout(template="plotly_white", legend_title="Tier",
                      title_font_size=16, height=380)
    return fig


def fig_cluster_scatter():
    sample = DF.sample(min(1500, len(DF)), random_state=1)
    fig = px.scatter(
        sample, x="accommodates", y="price", color="tier",
        color_discrete_map=TIER_COLORS, size="bedrooms",
        hover_data=["neighbourhood_cleansed", "room_type", "amenity_count"],
        labels={"price": "Nightly Price (USD)", "accommodates": "Guests"},
        title="Pricing Clusters: Capacity vs Price",
        category_orders={"tier": ["Budget", "Standard", "Premium", "Luxury"]},
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white", height=420, title_font_size=16)
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
        neigh_stats, x="median_price", y="neighbourhood_cleansed",
        orientation="h", color="median_price",
        color_continuous_scale="Viridis",
        labels={"median_price": "Median Nightly Price (USD)",
                "neighbourhood_cleansed": "Neighbourhood"},
        title="Top 25 Neighbourhoods by Median Price",
        text="median_price",
    )
    fig.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
    fig.update_layout(template="plotly_white", height=600, title_font_size=16,
                      coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    return fig


def fig_price_by_room_type():
    fig = px.box(
        DF, x="room_type", y="price", color="room_type",
        color_discrete_sequence=["#4CAF50", "#2196F3", "#FF9800"],
        labels={"price": "Nightly Price (USD)", "room_type": "Room Type"},
        title="Price Distribution by Room Type",
        category_orders={"room_type": ["Entire home/apt", "Private room", "Shared room"]},
    )
    fig.update_layout(template="plotly_white", height=380,
                      showlegend=False, title_font_size=16)
    return fig


def fig_tier_radar():
    metrics = []
    for tier_id, tier_name in TIER_NAMES.items():
        sub = DF[DF["tier"] == tier_name]
        metrics.append({
            "Tier": tier_name,
            "Median Price (norm)": sub["price"].median() / DF["price"].median(),
            "Avg Accommodates": sub["accommodates"].mean() / DF["accommodates"].mean(),
            "Avg Bedrooms":     sub["bedrooms"].mean() / DF["bedrooms"].mean(),
            "Avg Bathrooms":    sub["bathrooms"].mean() / DF["bathrooms"].mean(),
            "Avg Amenities":    sub["amenity_count"].mean() / DF["amenity_count"].mean(),
            "Avg Review Score": (sub["review_scores_rating"].mean() /
                                 DF["review_scores_rating"].mean()
                                 if "review_scores_rating" in DF.columns else 1),
        })

    cats = ["Median Price (norm)", "Avg Accommodates", "Avg Bedrooms",
            "Avg Bathrooms", "Avg Amenities", "Avg Review Score"]
    fig = go.Figure()
    for m in metrics:
        vals = [m[c] for c in cats] + [m[cats[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]],
            fill="toself", name=m["Tier"],
            line_color=TIER_COLORS[m["Tier"]], opacity=0.7,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 2.5])),
        showlegend=True, title="Tier Profiles (Normalised Metrics)",
        template="plotly_white", height=420, title_font_size=16,
    )
    return fig


def cluster_profiles_table():
    rows = []
    for tier_id in range(4):
        tier_name = TIER_NAMES[tier_id]
        sub = DF[DF["tier"] == tier_name]
        top_neigh = sub["neighbourhood_cleansed"].value_counts().head(3).index.tolist()
        cancel_top = sub["cancellation_policy"].value_counts().index[0]
        rows.append({
            "Tier": f"{TIER_EMOJIS[tier_name]} {tier_name}",
            "Listings": f"{len(sub):,}",
            "Median Price": f"${sub['price'].median():.0f}",
            "Price Range (IQR)": f"${sub['price'].quantile(0.25):.0f}–${sub['price'].quantile(0.75):.0f}",
            "Median Guests": f"{sub['accommodates'].median():.0f}",
            "Median Beds": f"{sub['bedrooms'].median():.0f}",
            "Top Room Type": sub["room_type"].mode()[0],
            "Avg Amenities": f"{sub['amenity_count'].mean():.0f}",
            "Avg Review Score": f"{sub['review_scores_rating'].mean():.1f}" if "review_scores_rating" in sub else "N/A",
            "Superhost %": f"{sub['host_is_superhost'].mean()*100:.0f}%",
            "Instant Book %": f"{sub['instant_bookable'].mean()*100:.0f}%",
            "Top Cancellation": cancel_top,
            "Top Neighbourhoods": ", ".join(top_neigh),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# PRICING RECOMMENDATION
# ─────────────────────────────────────────────

def recommend_price(room_type, accommodates, bedrooms, bathrooms,
                    neighbourhood, amenity_count, is_superhost, instant_book):

    room_enc = {"Entire home/apt": 3, "Private room": 2, "Shared room": 1}[room_type]
    neigh_proxy = DF.groupby("neighbourhood_cleansed")["price"].median().get(neighbourhood, DF["price"].median())

    user_row = pd.DataFrame([{
        "price":            neigh_proxy,   # placeholder; not used for prediction
        "accommodates":     accommodates,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "neigh_price_proxy": neigh_proxy,
        "room_type_enc":    room_enc,
        "amenity_count":    amenity_count,
    }])

    X_user = SCALER.transform(user_row[FEATURES].values)
    raw_label = KMEANS.predict(X_user)[0]
    cluster_id = REMAP[raw_label]
    tier_name  = TIER_NAMES[cluster_id]

    tier_df   = DF[DF["tier"] == tier_name]
    p25       = tier_df["price"].quantile(0.25)
    p75       = tier_df["price"].quantile(0.75)
    median_p  = tier_df["price"].median()
    count     = len(tier_df)
    top_neigh = tier_df["neighbourhood_cleansed"].value_counts().head(3).index.tolist()

    emoji = TIER_EMOJIS[tier_name]
    color = TIER_COLORS[tier_name]

    # Build markdown output
    result = f"""
## {emoji} You're in the **{tier_name}** Tier

{TIER_DESC[tier_name]}

---

### 💰 Pricing Recommendation

| | |
|---|---|
| **Suggested anchor price** | **${median_p:.0f} / night** |
| **Competitive price range** | ${p25:.0f} – ${p75:.0f} / night |
| **Listings in this tier** | {count:,} |

---

### 📊 Why this tier?

"""
    insights = []
    all_median_acc  = DF["accommodates"].median()
    all_median_bed  = DF["bedrooms"].median()
    all_median_bath = DF["bathrooms"].median()
    all_median_amen = DF["amenity_count"].median()
    tier_median_acc  = tier_df["accommodates"].median()
    tier_median_bed  = tier_df["bedrooms"].median()
    tier_median_amen = tier_df["amenity_count"].median()

    # Room type
    insights.append(f"**Room type** — '{room_type}' is the dominant listing type in the {tier_name} tier "
                    f"({tier_df['room_type'].value_counts(normalize=True).iloc[0]*100:.0f}% of tier).")

    # Capacity
    cmp = "above" if accommodates > tier_median_acc else ("below" if accommodates < tier_median_acc else "at")
    insights.append(f"**Capacity ({accommodates} guests)** — {cmp} the {tier_name} tier median of {tier_median_acc:.0f} guests.")

    # Bedrooms
    cmp = "above" if bedrooms > tier_median_bed else ("below" if bedrooms < tier_median_bed else "at")
    insights.append(f"**Bedrooms ({bedrooms:.0f})** — {cmp} the {tier_name} median of {tier_median_bed:.0f}.")

    # Amenities
    cmp = "above" if amenity_count > tier_median_amen else ("below" if amenity_count < tier_median_amen else "at")
    insights.append(f"**Amenities ({amenity_count})** — {cmp} the {tier_name} median of {tier_median_amen:.0f}.")

    # Neighbourhood
    neigh_rank = DF.groupby("neighbourhood_cleansed")["price"].median().rank(ascending=False)
    rank = int(neigh_rank.get(neighbourhood, len(neigh_rank)))
    insights.append(f"**Neighbourhood ({neighbourhood})** — ranks #{rank} out of {len(neigh_rank)} neighbourhoods by median price.")

    for ins in insights:
        result += f"- {ins}\n"

    result += f"""
---

### 🗺️ Top Neighbourhoods in {tier_name} Tier

{', '.join(top_neigh)}

---

### 💡 Tips to Move Up a Tier

"""
    if tier_name != "Luxury":
        next_tier = TIER_NAMES[cluster_id + 1]
        next_df   = DF[DF["tier"] == next_tier]
        result += (f"To reach **{next_tier}** (median ${next_df['price'].median():.0f}/night), consider:\n"
                   f"- Increasing guest capacity to {next_df['accommodates'].median():.0f}+\n"
                   f"- Adding amenities to reach {next_df['amenity_count'].median():.0f}+\n"
                   f"- If possible, listing an entire home/apt rather than a private room\n")
    else:
        result += "You're already in the top tier! Focus on **reviews** and **instant bookability** to maximise occupancy.\n"

    return result


# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

# Pre-build static figures
FIG_DIST      = fig_price_distribution()
FIG_SCATTER   = fig_cluster_scatter()
FIG_NEIGH     = fig_neighbourhood_pricing()
FIG_ROOMTYPE  = fig_price_by_room_type()
FIG_RADAR     = fig_tier_radar()
PROFILES_DF   = cluster_profiles_table()

MARKET_STATS_MD = f"""
### 📈 Seattle Airbnb Market Overview (2016)

| Metric | Value |
|---|---|
| Total listings analysed | {len(DF):,} |
| Median nightly price | ${DF['price'].median():.0f} |
| Average nightly price | ${DF['price'].mean():.0f} |
| Price range (filtered) | $20 – $999 |
| Unique neighbourhoods | {DF['neighbourhood_cleansed'].nunique()} |
| Entire home/apt share | {(DF['room_type']=='Entire home/apt').mean()*100:.0f}% |
| Superhost share | {DF['host_is_superhost'].mean()*100:.0f}% |
| Instant bookable share | {DF['instant_bookable'].mean()*100:.0f}% |
"""

CSS = """
.gr-markdown h2 { color: #1a1a2e; border-bottom: 2px solid #e91e63; padding-bottom:6px; }
.gr-markdown h3 { color: #16213e; margin-top:16px; }
.tier-header { font-size:1.4rem; font-weight:700; }
footer { display:none !important; }
"""

with gr.Blocks(
    title="Seattle Airbnb Pricing Intelligence",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="pink"),
    css=CSS,
) as demo:

    gr.Markdown("""
# 🏠 Seattle Airbnb Pricing Intelligence
**Unsupervised market segmentation & data-driven pricing recommendations**  
*Built on 3,818 real Seattle listings · K-Means clustering into 4 pricing tiers*
""")

    with gr.Tabs():

        # ── TAB 1: MARKET OVERVIEW ──────────────────────────────────
        with gr.TabItem("📊 Market Overview"):
            gr.Markdown(MARKET_STATS_MD)
            with gr.Row():
                gr.Plot(FIG_DIST,    label="Price Distribution")
                gr.Plot(FIG_ROOMTYPE, label="Price by Room Type")
            gr.Plot(FIG_NEIGH, label="Neighbourhood Pricing")

        # ── TAB 2: CLUSTER EXPLORER ─────────────────────────────────
        with gr.TabItem("🔍 Cluster Explorer"):
            gr.Markdown("### Pricing clusters visualised by guest capacity and nightly rate")
            gr.Plot(FIG_SCATTER, label="Cluster Scatter")
            with gr.Row():
                gr.Plot(FIG_RADAR, label="Tier Radar Chart")
            gr.Markdown("### 📋 Cluster Profiles — All Four Tiers Side by Side")
            gr.Dataframe(
                PROFILES_DF,
                label="Tier Profiles",
                wrap=True,
                interactive=False,
            )

        # ── TAB 3: PRICING TOOL ─────────────────────────────────────
        with gr.TabItem("💰 Get My Price"):
            gr.Markdown("""
### Enter your listing details to receive a personalised pricing recommendation
*All fields are used to match your listing to the most similar market segment.*
""")
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("#### 🏡 Property Details")
                    room_type_in  = gr.Dropdown(
                        ["Entire home/apt", "Private room", "Shared room"],
                        value="Entire home/apt", label="Room Type"
                    )
                    accommodates_in = gr.Slider(1, 16, value=3, step=1, label="Guests (accommodates)")
                    bedrooms_in     = gr.Slider(0, 10, value=1, step=1, label="Bedrooms")
                    bathrooms_in    = gr.Slider(0.5, 8, value=1, step=0.5, label="Bathrooms")
                    neighbourhood_in = gr.Dropdown(
                        NEIGHBOURHOODS, value=NEIGHBOURHOODS[0], label="Neighbourhood"
                    )
                    amenity_count_in = gr.Slider(1, 80, value=20, step=1, label="Number of Amenities")
                    superhost_in   = gr.Checkbox(label="Superhost", value=False)
                    instant_in     = gr.Checkbox(label="Instant Bookable", value=True)

                    btn = gr.Button("🔍 Get Pricing Recommendation", variant="primary", size="lg")

                with gr.Column(scale=2):
                    recommendation_out = gr.Markdown(
                        value="*Fill in your listing details and click the button to see your recommendation.*"
                    )

            btn.click(
                fn=recommend_price,
                inputs=[room_type_in, accommodates_in, bedrooms_in, bathrooms_in,
                        neighbourhood_in, amenity_count_in, superhost_in, instant_in],
                outputs=recommendation_out,
            )

        # ── TAB 4: METHODOLOGY ──────────────────────────────────────
        with gr.TabItem("📖 Methodology"):
            gr.Markdown(f"""
## How It Works

### 1. Data Preparation
- {len(DF_RAW):,} Seattle listings filtered to prices $1–$999/night
- `amenity_count` derived from comma-separated amenity strings
- Missing `bedrooms` and `bathrooms` imputed with column medians
- `room_type` ordinally encoded: Entire home/apt=3, Private room=2, Shared room=1
- `neigh_price_proxy` = median nightly price of all listings in the same neighbourhood (location wealth proxy)

### 2. Clustering Features
| Feature | Purpose |
|---|---|
| `price` | Target variable & clustering anchor |
| `accommodates` | Guest capacity |
| `bedrooms` | Property size |
| `bathrooms` | Property size |
| `neigh_price_proxy` | Location desirability proxy |
| `room_type_enc` | Listing format |
| `amenity_count` | Listing quality |

### 3. Modelling
- Features standardised with **StandardScaler** (zero mean, unit variance)
- **K-Means** fit with k=4, n_init=20, random_state=42
- Clusters re-labelled in ascending price order: Budget → Standard → Premium → Luxury

### 4. Recommendation Engine
- User inputs are transformed with the **same fitted scaler**
- The fitted K-Means model predicts the closest cluster centroid
- Tier statistics (25th–75th percentile, median) are returned as the price recommendation

### Data Source
Seattle Airbnb listings, scraped 2016 — originally from [Inside Airbnb](http://insideairbnb.com).
""")

demo.launch(share=False)
