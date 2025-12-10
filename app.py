import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
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
    except Exception:
        uploaded_file.seek(0)
        raw = uploaded_file.getvalue().decode("utf-8")
        return pd.read_csv(io.StringIO(raw))


def safe_sum(series):
    try:
        return float(series.sum())
    except Exception:
        return np.nan


def generate_trend_insights(ts, date_col, rev_col):
    """
    Given a dataframe ts with a date column and a revenue column (daily aggregated),
    returns a list of human-readable insights. This function is defensive and will
    still return insights when the series is short.
    """
    insights = []
    if ts.empty:
        return ["No data available for insights."]

    # Ensure sorted
    ts = ts.sort_values(date_col).reset_index(drop=True)

    # Basic metrics
    total = ts[rev_col].sum()
    avg = ts[rev_col].mean()
    latest_date = ts[date_col].iloc[-1]
    latest_value = ts[rev_col].iloc[-1]

    insights.append(f"**Total observed revenue:** {total:,.0f} over {len(ts)} days.")
    insights.append(f"**Average daily revenue:** {avg:,.2f}.")
    insights.append(f"**Latest date:** {pd.to_datetime(latest_date).date()} with revenue **{latest_value:,.0f}**.")

    # Recent trend (compare last 7 days to previous 7 days when available)
    if len(ts) >= 14:
        last7 = ts[rev_col].iloc[-7:].sum()
        prev7 = ts[rev_col].iloc[-14:-7].sum()
        if prev7 == 0:
            pct = np.nan
        else:
            pct = (last7 - prev7) / prev7 * 100
        insights.append(f"**7‑day change:** Revenue for the last 7 days is {last7:,.0f}; change vs previous 7 days: {pct:+.1f}%.")
    elif len(ts) >= 7:
        last7 = ts[rev_col].iloc[-7:].sum()
        insights.append(f"**Recent 7 days revenue:** {last7:,.0f} (not enough history for direct comparison).")

    # Highest and fastest growing periods
    top_day = ts.loc[ts[rev_col].idxmax()]
    insights.append(f"**Top performing day:** {pd.to_datetime(top_day[date_col]).date()} with revenue {top_day[rev_col]:,.0f}.")

    # Compute simple linear growth rate (slope) normalized by mean
    try:
        t = np.arange(len(ts))
        slope = np.polyfit(t, ts[rev_col].values, 1)[0]
        rel_slope = slope / (avg if avg != 0 else 1)
        if rel_slope > 0.01:
            trend_desc = "strong upward"
        elif rel_slope > 0.001:
            trend_desc = "moderate upward"
        elif rel_slope < -0.01:
            trend_desc = "strong downward"
        elif rel_slope < -0.001:
            trend_desc = "moderate downward"
        else:
            trend_desc = "flat/seasonal"
        insights.append(f"**Overall trend:** {trend_desc} (slope ≈ {slope:.2f} revenue units/day).")
    except Exception:
        pass

    return insights


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

if info["date"] and df[info["date"]].notna().any():
    min_d, max_d = df[info["date"]].min(), df[info["date"]].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        [min_d.date(), max_d.date()] if not pd.isna(min_d) else []
    )
    if len(date_range) == 2:
        df = df[(df[info["date"]] >= pd.to_datetime(date_range[0])) &
                (df[info["date"]] <= pd.to_datetime(date_range[1]))]

if info["category"] and df[info["category"]].notna().any():
    cats = ["All"] + sorted(df[info["category"]].dropna().astype(str).unique())
    s_cat = st.sidebar.selectbox("Category", cats)
    if s_cat != "All":
        df = df[df[info["category"]].astype(str) == s_cat]

if info["region"] and df[info["region"]].notna().any():
    regs = ["All"] + sorted(df[info["region"]].dropna().astype(str).unique())
    s_reg = st.sidebar.selectbox("Region", regs)
    if s_reg != "All":
        df = df[df[info["region"]].astype(str) == s_reg]


# ------------------------------------------------
# KPIs
# ------------------------------------------------
rev_col = info.get("revenue")
ord_col = info.get("orders")
cost_col = info.get("cost")

total_revenue = safe_sum(df[rev_col]) if rev_col in df.columns else np.nan
total_orders = safe_sum(df[ord_col]) if ord_col in df.columns else np.nan
total_cost = safe_sum(df[cost_col]) if cost_col in df.columns else np.nan

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


# TRENDS (with professional chart + written insight)
elif page == "Trends":
    st.title("📈 Revenue Trends & Insights")

    if info["date"] and rev_col and info["date"] in df.columns and rev_col in df.columns:
        # aggregate to daily revenue
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_col].sum().reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()
        ts = ts.sort_values(info["date"]).reset_index(drop=True)

        # Create moving average and trendline
        ts['ma7'] = ts[rev_col].rolling(window=7, min_periods=1).mean()
        ts['ma30'] = ts[rev_col].rolling(window=30, min_periods=1).mean()

        # Professional combined chart: bars for daily, lines for MA and forecast placeholder
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ts[info["date"]],
            y=ts[rev_col],
            name='Daily Revenue',
            marker={'opacity': 0.85}
        ))
        fig.add_trace(go.Scatter(
            x=ts[info["date"]],
            y=ts['ma7'],
            mode='lines',
            name='7‑day MA',
            line={'width': 2, 'dash': 'dash'}
        ))
        fig.add_trace(go.Scatter(
            x=ts[info["date"]],
            y=ts['ma30'],
            mode='lines',
            name='30‑day MA',
            line={'width': 2}
        ))

        fig.update_layout(
            title='Daily Revenue with 7‑day & 30‑day Moving Averages',
            xaxis_title='Date',
            yaxis_title='Revenue',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True, height=500)

        # Generate automated insights
        insights = generate_trend_insights(ts, info["date"], rev_col)

        st.markdown("### 🔍 Automated Insights")
        for s in insights:
            st.markdown(f"- {s}")

        # Add an expandable section with recommended actions based on insights
        with st.expander("Recommended next actions (AI suggestions)"):
            st.write("Based on the current trend and recent performance, consider the following actions:")
            st.write("1. If revenue shows a downward trend, run targeted promotions for your top 3 products to recover sales.")
            st.write("2. If recent 7‑day revenue is strong, ensure inventory levels are sufficient for the next 14 days.")
            st.write("3. Investigate the top performing day(s) to identify marketing or external events that drove conversions.")

    else:
        st.warning("No valid date or revenue column detected for trend analysis.")


# PRODUCTS
elif page == "Products":
    st.title("🛍️ Product Performance")

    prod = info.get("product")
    if prod and rev_col and prod in df.columns and rev_col in df.columns:
        top = df.groupby(prod)[rev_col].sum().sort_values(ascending=False).head(20).reset_index()

        st.dataframe(top, use_container_width=True)

        fig = px.bar(top, x=prod, y=rev_col, text=rev_col, title="Top Selling Products")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Product or revenue column missing.")


# CATEGORIES
elif page == "Categories":
    st.title("📂 Category Overview")

    if info.get("category") and rev_col and info.get("category") in df.columns:
        mix = df.groupby(info["category"])[rev_col].sum().reset_index()

        fig = px.pie(mix, names=info["category"], values=rev_col)
        st.plotly_chart(fig, use_container_width=True)


# REGIONS
elif page == "Regions":
    st.title("🌍 Regional Performance")

    if info.get("region") and rev_col and info.get("region") in df.columns:
        r = df.groupby(info["region"])[rev_col].sum().reset_index()
        fig = px.bar(r, x=info["region"], y=rev_col, title="Revenue by Region")
        st.plotly_chart(fig, use_container_width=True)


# FORECAST
elif page == "Forecast":
    st.title("🔮 Revenue Forecast (Next 30 Days)")

    if info["date"] and rev_col and info["date"] in df.columns and rev_col in df.columns:
        ts = df.groupby(df[info["date"]].dt.to_period("D"))[rev_col].sum().reset_index()
        ts[info["date"]] = ts[info["date"]].dt.to_timestamp()
        ts = ts.sort_values(info["date"]).reset_index(drop=True)

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
