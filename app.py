import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from sklearn.linear_model import LinearRegression
import io

# ------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Dropshipping Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBAL STYLES
st.markdown("""
    <style>
        .metric-card {
            background: #ffffff;
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        }
        .stDataFrame { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# HELPERS
# ------------------------------------------------

def infer_columns(df):
    """
    Intelligent column detection: finds most meaningful column names.
    """
    lower = [c.lower() for c in df.columns]

    def find(hints):
        for h in hints:
            for i, c in enumerate(lower):
                if h in c:
                    return df.columns[i]
        return None

    return {
        "date": find(["date", "order_date", "timestamp"]),
        "product": find(["product", "title", "item", "sku"]),
        "category": find(["category", "type"]),
        "orders": find(["orders", "qty", "quantity", "units"]),
        "revenue": find(["revenue", "sales", "gmv", "amount"]),
        "price": find(["price", "selling_price"]),
        "cost": find(["cost", "cogs", "buy_cost"]),
        "returns": find(["return", "refund"]),
        "rating": find(["rating", "review"]),
        "region": find(["country", "region", "market", "geo"]),
    }


def load_dataset(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except:
        uploaded_file.seek(0)
        raw = uploaded_file.getvalue().decode("utf-8")
        return pd.read_csv(io.StringIO(raw))


def safe_sum(series):
    return float(series.sum()) if series is not None else np.nan


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("📦 Dropshipping Dashboard")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sample_path = Path("dropshipping_model_dataset_1000.csv")

use_sample = False
if not uploaded and sample_path.exists():
    use_sample = st.sidebar.checkbox("Use sample dataset", value=True)

# Load data
if uploaded:
    df = load_dataset(uploaded)
elif use_sample:
    df = pd.read_csv(sample_path)
else:
    st.title("📦 Dropshipping Analytics Dashboard")
    st.info("Upload a CSV file or enable the sample dataset from sidebar.")
    st.stop()

info = infer_columns(df)

# Parse date
if info["date"] in df.columns:
    df[info["date"]] = pd.to_datetime(df[info["date"]], errors="coerce")

# Sidebar Filters
st.sidebar.markdown("### Filters")

if info["date"]:
    min_d, max_d = df[info["date"]].min(), df[info["date"]].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_d, max_d] if not pd.isna(min_d) else []
    )
    if len(date_range) == 2:
        df = df[(df[info["date"]] >= pd.to_datetime(date_range[0])) &
                (df[info["date"]] <= pd.to_datetime(date_range[1]))]

if info["category"]:
    cats = ["All"] + sorted(df[info["category"]].dropna().astype(str).unique())
    s_cat = st.sidebar.selectbox("Category", cats)
    if s_cat != "All":
        df = df[df[info["category"]].astype(str) == s_cat]

if info["region"]:
    regs = ["All"] + sorted(df[info["region"]].dropna().astype(str).unique())
    s_reg = st.sidebar.selectbox("Region", regs)
    if s_reg != "All":
        df = df[df[info["region"]].astype(str) == s_reg]


# ------------------------------------------------
# KPIs
# ------------------------------------------------
rev_col = info["revenue"]
ord_col = info["orders"]
cost_col = info["cost"]

total_revenue = safe_sum(df[rev_col]) if rev_col else np.nan
total_orders = safe_sum(df[ord_col]) if ord_col else np.nan
total_cost = safe_sum(df[cost_col]) if cost_col else np.nan

profit = total_revenue - total_cost if all(np.isfinite([total_revenue, total_cost])) else np.nan
aov = total_revenue / total_orders if np.isfinite(total_orders) and total_orders > 0 else np.nan


# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    ["Home", "KPIs", "Trends", "Products", "Categories", "Regions", "Forecast", "Settings"]
)


# ------------------------------------------------
# PAGES
# ------------------------------------------------

# HOME
if page == "Home":
    st.title("📦 Dropshipping Analytics — Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Revenue", f"{total_revenue:,.2f}")
    with c2: st.metric("Total Orders", f"{total_orders:,.0f}")
    with c3: st.metric("Profit", f"{profit:,.2f}")
    with c4: st.metric("AOV", f"{aov:,.2f}")

    st.markdown("### 📄 Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)


# KPIs
elif page == "KPIs":
    st.title("📊 Key Performance Indicators")

    k1, k2, k3 = st.columns(3)
    k1.metric("Revenue", f"{total_revenue:,.2f}")
    k2.metric("Orders", f"{total_orders:,.0f}")
    k3.metric("Profit", f"{profit:,.2f}")


# TRENDS
elif page == "Trends":
    st.title("📈 Revenue Trends Over Time")

    if info["date"] and rev_col:
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_col].sum()
        ts = ts.reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()

        fig = px.line(
            ts,
            x=info["date"],
            y=rev_col,
            title="Daily Revenue Trend",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid date or revenue column detected.")


# PRODUCTS
elif page == "Products":
    st.title("🛍️ Product Performance")

    prod = info["product"]
    if prod and rev_col:
        top = df.groupby(prod)[rev_col].sum().sort_values(ascending=False).head(20).reset_index()

        st.dataframe(top, use_container_width=True)

        fig = px.bar(top, x=prod, y=rev_col, text=rev_col, title="Top Selling Products")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Product or revenue column missing.")


# CATEGORIES
elif page == "Categories":
    st.title("📂 Category Overview")

    if info["category"] and rev_col:
        mix = df.groupby(info["category"])[rev_col].sum().reset_index()

        fig = px.pie(mix, names=info["category"], values=rev_col)
        st.plotly_chart(fig, use_container_width=True)


# REGIONS
elif page == "Regions":
    st.title("🌍 Regional Performance")

    if info["region"] and rev_col:
        r = df.groupby(info["region"])[rev_col].sum().reset_index()
        fig = px.bar(r, x=info["region"], y=rev_col, title="Revenue by Region")
        st.plotly_chart(fig, use_container_width=True)


# FORECAST
elif page == "Forecast":
    st.title("🔮 Revenue Forecast (Next 30 Days)")

    if info["date"] and rev_col:
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_col].sum().reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()

        ts["t"] = np.arange(len(ts))
        X, y = ts[["t"]].values, ts[rev_col].values

        model = LinearRegression().fit(X, y)

        future = np.arange(len(ts), len(ts) + 30).reshape(-1, 1)
        predictions = model.predict(future)

        future_dates = pd.date_range(ts[info["date"]].iloc[-1] + pd.Timedelta(days=1),
                                     periods=30)

        pred_df = pd.DataFrame({info["date"]: future_dates, "forecast": predictions})

        fig = px.line(ts, x=info["date"], y=rev_col, title="Historical Revenue")
        fig.add_scatter(x=pred_df[info["date"]], y=pred_df["forecast"],
                        mode="lines", name="Forecast")

        st.plotly_chart(fig, use_container_width=True)


# SETTINGS
elif page == "Settings":
    st.title("⚙️ Settings & Column Mapping")
    st.json(info)
