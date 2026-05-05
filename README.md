---
title: Seattle Airbnb Pricing Intelligence
emoji: 🏠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Seattle Airbnb Pricing Intelligence

An interactive, data-driven tool that segments the Seattle Airbnb market into four pricing tiers using **K-Means clustering** and provides personalised nightly pricing recommendations.

## Features

- 📊 **Market Overview** — price distributions, room-type breakdowns, neighbourhood rankings
- 🔍 **Cluster Explorer** — interactive scatter plots, radar charts, and a full tier-profiles table
- 💰 **Pricing Tool** — enter your listing details and receive a tier assignment plus recommended price range
- 📖 **Methodology** — full walkthrough of the data pipeline and modelling approach

## How to Run Locally

```bash
pip install -r requirements.txt
# Place listings.xlsx in the same folder
python app.py
```

## Dataset
Seattle Airbnb listings (2016) — 3,818 listings, 92 features — sourced from [Inside Airbnb](http://insideairbnb.com).
