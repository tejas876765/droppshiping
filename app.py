import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io

st.set_page_config(page_title="Dropshipping Analytics (Upgraded)", layout="wide", initial_sidebar_state="expanded")

# ---------- Helpers ----------
def infer_columns(df):
    lower = [c.lower() for c in df.columns]
    def find(hints):
        for h in hints:
            for i,c in enumerate(lower):
                if h in c:
                    return df.columns[i]
        return None
    info = {}
    info["date"] = find(["date","order_date","created","timestamp","day"])
    info["product"] = find(["product","title","name","sku","asin"])
    info["category"] = find(["category","type"])
    info["orders"] = find(["qty","quantity","units","orders","sold","sales_count","order"])
    info["revenue"] = find(["revenue","sales","gmv","amount","turnover","total"])
    info["price"] = find(["price","selling_price","sale_price"])
    info["cost"] = find(["cost","cogs","buy_cost"])
    info["returns"] = find(["return","refund"])
    info["rating"] = find(["rating","review","stars"])
    info["region"] = find(["country","region","market","city","state","geo"])
    info["stock"] = find(["stock","inventory","on_hand"])
    info["numeric_cols"] = df.select_dtypes(include=[np.number]).columns.tolist()
    return info

def load_dataset(uploaded_file):
    try:    df = pd.read_csv(uploaded_file)
    except: upload.seek(0); df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
    return df

def safe_sum(series):
    try: return float(series.sum())
    except: return np.nan

# UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home","KPIs","Trends","Products","Categories","Regions","Forecast","Settings"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

use_sample = False
sample_path = Path("dropshipping_model_dataset_1000.csv")
if not uploaded and sample_path.exists():
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)

if uploaded: df = load_dataset(uploaded)
elif use_sample: df = pd.read_csv(sample_path)
else:
    st.title("📦 Dropshipping Analytics (Upgraded)")
    st.info("Upload CSV from sidebar or include sample dataset.")
    st.stop()

info = infer_columns(df)

# Coerce date column
if info["date"] and info["date"] in df.columns:
    df[info["date"]] = pd.to_datetime(df[info["date"]], errors="coerce")
else:
    df["_row_id_"] = np.arange(len(df))

# Sidebar filters
st.sidebar.markdown("### Filters")
if info["date"] and info["date"] in df.columns:
    min_d, max_d = df[info["date"]].min(), df[info["date"]].max()
    start_date, end_date = st.sidebar.date_input("Date range", [min_d.date(), max_d.date()])
    df = df[(df[info["date"]] >= pd.to_datetime(start_date)) & (df[info["date"]] <= pd.to_datetime(end_date))]

if info["category"] and info["category"] in df.columns:
    cats = ["All"] + sorted(df[info["category"]].dropna().unique().astype(str).tolist())
    s_cat = st.sidebar.selectbox("Category", cats)
    if s_cat != "All": df = df[df[info["category"]].astype(str) == s_cat]

if info["region"] and info["region"] in df.columns:
    regs = ["All"] + sorted(df[info["region"]].dropna().unique().astype(str).tolist())
    s_reg = st.sidebar.selectbox("Region", regs)
    if s_reg != "All": df = df[df[info["region"]].astype(str) == s_reg]

# Calculations
rev_c = info["revenue"]
ord_c = info["orders"]
cos_c = info["cost"]

total_revenue = safe_sum(df[rev_c]) if rev_c else np.nan
total_orders  = safe_sum(df[ord_c]) if ord_c else np.nan
total_cost    = safe_sum(df[cos_c]) if cos_c else np.nan
profit = total_revenue - total_cost if np.isfinite(total_revenue) and np.isfinite(total_cost) else np.nan
avg_order = total_revenue/total_orders if total_orders and np.isfinite(total_orders) else np.nan

# ============================= Pages =============================

# HOME
if page == "Home":
    st.title("📦 Dropshipping Analytics — Home")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Revenue", f"{total_revenue:,.2f}")
    c2.metric("Orders", f"{total_orders:,.0f}")
    c3.metric("Profit", f"{profit:,.2f}")
    c4.metric("AOV", f"{avg_order:,.2f}")
    st.write("### Dataset Preview")
    st.dataframe(df.head(100))

# KPIs
if page == "KPIs":
    st.title("📊 KPIs Overview")
    c1,c2,c3 = st.columns(3)
    c1.metric("Revenue", f"{total_revenue:,.2f}")
    c2.metric("Orders", f"{total_orders:,.0f}")
    c3.metric("Profit", f"{profit:,.2f}")

# Trends
if page == "Trends":
    st.title("📈 Trends Over Time")
    if info["date"] and rev_c:
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_c].sum().reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()
        fig = px.line(ts, x=info["date"], y=rev_c, title="Revenue Trend", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid date column detected.")

# Products
if page == "Products":
    st.title("🛍️ Product Analysis")
    prod = info["product"]
    if prod and rev_c:
        top = df.groupby(prod)[rev_c].sum().sort_values(ascending=False).head(20).reset_index()
        st.dataframe(top)
        fig = px.bar(top, x=prod, y=rev_c, text=rev_c, title="Top Products")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product column detected.")

# Categories
if page == "Categories":
    st.title("📂 Category Analysis")
    cat = info["category"]
    if cat and rev_c:
        mix = df.groupby(cat)[rev_c].sum().reset_index()
        fig = px.pie(mix, names=cat, values=rev_c)
        st.plotly_chart(fig, use_container_width=True)

# Regions
if page == "Regions":
    st.title("🌍 Region Analysis")
    reg = info["region"]
    if reg and rev_c:
        r = df.groupby(reg)[rev_c].sum().reset_index()
        fig = px.bar(r, x=reg, y=rev_c, title="Region Performance")
        st.plotly_chart(fig, use_container_width=True)

# Forecast
if page == "Forecast":
    st.title("🔮 Forecasting")
    if info["date"] and rev_c:
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_c].sum().reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()
        ts["t"] = np.arange(len(ts))
        X = ts[["t"]].values; y = ts[rev_c].values
        model = LinearRegression().fit(X,y)
        fut = np.arange(len(ts), len(ts)+30).reshape(-1,1)
        pred = model.predict(fut)
        fut_dates = pd.date_range(ts[info["date"]].iloc[-1] + pd.Timedelta(days=1), periods=30)
        pred_df = pd.DataFrame({info["date"]: fut_dates, "forecast": pred})
        fig = px.line(ts, x=info["date"], y=rev_c)
        fig.add_scatter(x=pred_df[info["date"]], y=pred_df["forecast"], mode='lines', name='Forecast')
        st.plotly_chart(fig, use_container_width=True)

# Settings
if page == "Settings":
    st.title("⚙️ Settings")
    st.write("Column inference:")
    st.json(info)
