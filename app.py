"""
Sales Forecasting & Demand Intelligence — Streamlit Dashboard
Author: Hrithika S.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

To deploy: push this folder to a public GitHub repo, then go to
https://share.streamlit.io -> New app -> point it at app.py in that repo.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data loading & caching
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv", encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


@st.cache_data
def get_monthly(df, filter_col=None, filter_val=None):
    d = df if filter_col is None else df[df[filter_col] == filter_val]
    return d.set_index("Order Date").resample("MS")["Sales"].sum()


@st.cache_data
def get_weekly(df):
    return df.set_index("Order Date").resample("W")["Sales"].sum()


@st.cache_resource
def fit_sarima(series):
    m = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return m


def forecast_with_metrics(series, steps=3):
    train, test = series.iloc[:-steps], series.iloc[-steps:]
    model = fit_sarima(train)
    fc = model.get_forecast(steps=steps)
    pred = fc.predicted_mean
    ci = fc.conf_int()
    mae = mean_absolute_error(test.values, pred.values)
    rmse = np.sqrt(mean_squared_error(test.values, pred.values))
    return pred, ci, mae, rmse


@st.cache_data
def compute_clusters(df):
    subcat = df.groupby("Sub-Category").agg(
        total_sales=("Sales", "sum"),
        avg_order_value=("Sales", "mean"),
    ).reset_index()
    yearly_sub = df.groupby(["Sub-Category", df["Order Date"].dt.year])["Sales"].sum().unstack()
    yoy_growth = yearly_sub.pct_change(axis=1).mean(axis=1) * 100
    volatility = df.groupby([df["Order Date"].dt.to_period("M"), "Sub-Category"])["Sales"].sum().unstack().std()

    subcat = subcat.set_index("Sub-Category")
    subcat["yoy_growth_pct"] = yoy_growth
    subcat["volatility"] = volatility
    subcat = subcat.fillna(0)

    feat_cols = ["total_sales", "yoy_growth_pct", "volatility", "avg_order_value"]
    X = StandardScaler().fit_transform(subcat[feat_cols])
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    subcat["cluster"] = kmeans.fit_predict(X)

    profile = subcat.groupby("cluster")[feat_cols].mean()

    def label_cluster(row):
        if row["total_sales"] >= profile["total_sales"].median() and row["volatility"] <= profile["volatility"].median():
            return "High Volume, Stable Demand"
        if row["yoy_growth_pct"] > profile["yoy_growth_pct"].median() and row["yoy_growth_pct"] > 0:
            return "Growing Demand"
        if row["yoy_growth_pct"] < 0:
            return "Declining Demand"
        return "Low Volume, High Volatility"

    labels = {c: label_cluster(profile.loc[c]) for c in profile.index}
    subcat["cluster_label"] = subcat["cluster"].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    subcat["pca1"], subcat["pca2"] = coords[:, 0], coords[:, 1]
    return subcat.reset_index()


@st.cache_data
def compute_anomalies(df):
    weekly = get_weekly(df)
    feat = weekly.to_frame(name="Sales")
    feat["rolling_mean"] = weekly.rolling(4, min_periods=1).mean()
    feat["rolling_std"] = weekly.rolling(4, min_periods=1).std().fillna(0)

    iso = IsolationForest(contamination=0.05, random_state=42)
    feat["iso_anomaly"] = iso.fit_predict(feat[["Sales", "rolling_mean", "rolling_std"]])

    roll_mean_prior = weekly.shift(1).rolling(4, min_periods=3).mean()
    roll_std_prior = weekly.shift(1).rolling(4, min_periods=3).std()
    feat["rolling_mean"] = roll_mean_prior
    z = (weekly - roll_mean_prior) / roll_std_prior
    feat["zscore"] = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    feat["z_anomaly"] = feat["zscore"].abs() > 2
    return feat


df = load_data()

st.title("📊 Sales Forecasting & Demand Intelligence Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

# ---------------------------------------------------------------------------
# Page 1 — Sales Overview
# ---------------------------------------------------------------------------
if page == "Sales Overview":
    st.header("Sales Overview")

    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year",
                     color_discrete_sequence=["#2E5EAA"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        monthly = get_monthly(df)
        fig = px.line(monthly.reset_index(), x="Order Date", y="Sales",
                      title="Monthly Sales Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filter by Region & Category")
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    cats = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
    filtered = df[df["Region"].isin(regions) & df["Category"].isin(cats)]

    col3, col4 = st.columns(2)
    with col3:
        by_region = filtered.groupby("Region")["Sales"].sum().reset_index()
        st.plotly_chart(px.bar(by_region, x="Region", y="Sales", title="Sales by Region",
                                color_discrete_sequence=["#4C9F70"]), use_container_width=True)
    with col4:
        by_cat = filtered.groupby("Category")["Sales"].sum().reset_index()
        st.plotly_chart(px.bar(by_cat, x="Category", y="Sales", title="Sales by Category",
                                color_discrete_sequence=["#E08E45"]), use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2 — Forecast Explorer
# ---------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.header("Forecast Explorer (SARIMA — best-performing model)")

    dim = st.selectbox("Forecast for:", ["Overall", "Category", "Region"])
    if dim == "Category":
        val = st.selectbox("Select Category", sorted(df["Category"].unique()))
        series = get_monthly(df, "Category", val)
    elif dim == "Region":
        val = st.selectbox("Select Region", sorted(df["Region"].unique()))
        series = get_monthly(df, "Region", val)
    else:
        series = get_monthly(df)

    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    with st.spinner("Fitting SARIMA model..."):
        pred, ci, mae, rmse = forecast_with_metrics(series, steps=3)
        pred = pred.iloc[:horizon]
        ci = ci.iloc[:horizon]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual", line=dict(color="#2E5EAA")))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, name="Forecast",
                              line=dict(color="#B0413E"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=list(ci.index) + list(ci.index[::-1]),
                              y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                              fill="toself", fillcolor="rgba(176,65,62,0.15)",
                              line=dict(color="rgba(255,255,255,0)"), name="Confidence Interval"))
    fig.update_layout(title=f"{dim} Sales Forecast — next {horizon} month(s)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("MAE (3-month backtest)", f"${mae:,.0f}")
    c2.metric("RMSE (3-month backtest)", f"${rmse:,.0f}")
    st.caption("MAE/RMSE computed by holding out the last 3 actual months and comparing to SARIMA's forecast for them.")

# ---------------------------------------------------------------------------
# Page 3 — Anomaly Report
# ---------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.header("Anomaly Report — Weekly Sales")
    feat = compute_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feat.index, y=feat["Sales"], name="Weekly Sales", line=dict(color="#2E5EAA")))
    iso_anoms = feat[feat["iso_anomaly"] == -1]
    fig.add_trace(go.Scatter(x=iso_anoms.index, y=iso_anoms["Sales"], mode="markers",
                              name="Isolation Forest Anomaly", marker=dict(color="red", size=9)))
    z_anoms = feat[feat["z_anomaly"]]
    fig.add_trace(go.Scatter(x=z_anoms.index, y=z_anoms["Sales"], mode="markers",
                              name="Z-Score Anomaly", marker=dict(color="orange", size=7, symbol="diamond")))
    fig.update_layout(title="Detected Anomalies in Weekly Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Isolation Forest Anomalies")
    st.dataframe(iso_anoms[["Sales"]].rename_axis("Week").reset_index())

    st.subheader("Z-Score Anomalies (>2 std from rolling mean)")
    st.dataframe(z_anoms[["Sales", "zscore"]].rename_axis("Week").reset_index())

# ---------------------------------------------------------------------------
# Page 4 — Product Demand Segments
# ---------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.header("Product Demand Segments (K-Means Clustering)")
    subcat = compute_clusters(df)

    fig = px.scatter(subcat, x="pca1", y="pca2", color="cluster_label", text="Sub-Category",
                      title="Sub-Category Demand Clusters (PCA projection)", size="total_sales")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sub-Category → Cluster Table")
    st.dataframe(
        subcat[["Sub-Category", "cluster_label", "total_sales", "yoy_growth_pct", "volatility", "avg_order_value"]]
        .sort_values("cluster_label")
        .round(1)
    )

    st.info(
        "**Stocking guidance:** High Volume/Stable → steady reorder-point replenishment. "
        "Growing Demand → build buffers ahead of season. Declining Demand → cut commitments, "
        "consider clearance. Low Volume/High Volatility → smaller, more frequent orders with wider safety stock."
    )
