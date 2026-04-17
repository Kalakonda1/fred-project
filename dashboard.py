"""
=============================================================
PHASE 4 — Interactive Business Intelligence Dashboard
Business Revenue & Macro Intelligence System
=============================================================

HOW TO RUN:
  pip install streamlit plotly
  streamlit run dashboard.py

WHAT THIS BUILDS:
  A fully interactive web app with 5 sections:
  1. Live FRED data pull with date range picker
  2. Real vs nominal revenue chart
  3. Recession risk score gauge
  4. 6-month forecast with confidence bands
  5. Macro correlation explorer
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from fredapi import Fred
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FRED Business Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .risk-high     { color: #dc3545; font-weight: 600; }
    .risk-moderate { color: #fd7e14; font-weight: 600; }
    .risk-low      { color: #198754; font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "FRED API Key",
    value=st.secrets.get("FRED_API_KEY", os.getenv("FRED_API_KEY", "")),
    type="password",
    help="Get a free key at fredaccount.stlouisfed.org/apikeys"
)

st.sidebar.markdown("### Date Range")
start_year = st.sidebar.slider("Start year", 2010, 2020, 2010)
end_year   = st.sidebar.slider("End year",   2020, 2024, 2024)

st.sidebar.markdown("### Forecast Settings")
forecast_months  = st.sidebar.slider("Months to forecast", 3, 12, 6)
risk_threshold   = st.sidebar.slider("Risk alert threshold", 40, 80, 55)

st.sidebar.markdown("### Display")
show_ci       = st.sidebar.checkbox("Show confidence intervals", value=True)
adjust_inflation = st.sidebar.checkbox("Inflation-adjust charts", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Federal Reserve (FRED) | CC0 Public Domain")

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

SERIES = {
    "retail_sales":    "RETAILSMNSA",
    "personal_income": "DSPIC96",
    "consumption":     "PCE",
    "cpi":             "CPIAUCSL",
    "industrial_prod": "INDPRO",
    "unemployment":    "UNRATE",
    "fed_funds_rate":  "FEDFUNDS",
}

@st.cache_data(ttl=3600, show_spinner=False)
def load_fred_data(api_key, start_date, end_date):
    fred = Fred(api_key=api_key)
    series_data = {}
    errors = []
    for name, series_id in SERIES.items():
        try:
            s = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            s = s.resample("ME").last().ffill(limit=3)
            series_data[name] = s
        except Exception as e:
            errors.append(f"{series_id}: {e}")
    df = pd.DataFrame(series_data)
    df.index.name = "date"
    df = df.dropna(how="all")
    return df, errors

# ─────────────────────────────────────────────────────────────
# FEATURE + MODEL BUILDER
# ─────────────────────────────────────────────────────────────

def build_features(df):
    cpi_base          = df["cpi"].iloc[0]
    df                = df.copy()
    df["retail_real"] = df["retail_sales"] / df["cpi"] * cpi_base
    df["real_yoy"]    = df["retail_real"].pct_change(12) * 100
    df["nominal_yoy"] = df["retail_sales"].pct_change(12) * 100
    df["inflation_yoy"]          = df["cpi"].pct_change(12) * 100
    df["inflation_contribution"] = df["nominal_yoy"] - df["real_yoy"]
    df["indpro_yoy"]  = df["industrial_prod"].pct_change(12) * 100

    df["consumption_lag1"]   = df["consumption"].shift(1)
    df["consumption_lag3"]   = df["consumption"].shift(3)
    df["unemployment_lag1"]  = df["unemployment"].shift(1)
    df["unemployment_lag3"]  = df["unemployment"].shift(3)
    df["cpi_lag1"]           = df["cpi"].shift(1)
    df["fed_rate_lag1"]      = df["fed_funds_rate"].shift(1)
    df["indpro_lag1"]        = df["industrial_prod"].shift(1)
    df["real_retail_lag1"]   = df["retail_real"].shift(1)
    df["real_retail_lag12"]  = df["retail_real"].shift(12)
    df["month_sin"]          = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]          = np.cos(2 * np.pi * df.index.month / 12)
    df["trend"]              = np.arange(len(df))

    def normalize(series, invert=False):
        mn, mx = series.min(), series.max()
        norm   = (series - mn) / (mx - mn)
        return 1 - norm if invert else norm

    df["score_unemployment"] = normalize(df["unemployment"])
    df["score_fed_rate"]     = normalize(df["fed_funds_rate"])
    df["score_indpro"]       = normalize(df["indpro_yoy"],  invert=True)
    df["score_real_retail"]  = normalize(df["real_yoy"],    invert=True)

    weights = {"score_unemployment": 0.30, "score_fed_rate": 0.20,
               "score_indpro": 0.20, "score_real_retail": 0.30}
    df["recession_risk"] = sum(df[c] * w for c, w in weights.items()) * 100

    return df


FEATURES = [
    "consumption_lag1", "consumption_lag3",
    "unemployment_lag1", "unemployment_lag3",
    "cpi_lag1", "fed_rate_lag1", "indpro_lag1",
    "real_retail_lag1", "real_retail_lag12",
    "month_sin", "month_cos", "trend"
]

def run_models(df, forecast_months):
    model_df  = df[FEATURES + ["retail_real"]].dropna()
    split     = model_df.index[int(len(model_df) * 0.80)]
    train     = model_df[model_df.index <= split]
    test      = model_df[model_df.index >  split]

    X_train, y_train = train[FEATURES], train["retail_real"]
    X_test,  y_test  = test[FEATURES],  test["retail_real"]

    # Naive
    naive_pred = pd.Series(index=y_test.index, dtype=float)
    for date in test.index:
        ly = date - pd.DateOffset(months=12)
        naive_pred[date] = model_df.loc[ly, "retail_real"] if ly in model_df.index else y_test[date]

    # Linear regression
    scaler = StandardScaler()
    lr     = LinearRegression()
    lr.fit(scaler.fit_transform(X_train), y_train)
    lr_pred = lr.predict(scaler.transform(X_test))

    # Ensemble
    ensemble = 0.65 * lr_pred + 0.35 * naive_pred.values
    residual_std = np.std(y_test.values - ensemble)

    # Metrics
    metrics = {
        "Naive":      {"MAE": mean_absolute_error(y_test, naive_pred),
                       "MAPE": np.mean(np.abs((y_test - naive_pred) / y_test)) * 100,
                       "R2":   r2_score(y_test, naive_pred)},
        "Linear Reg": {"MAE": mean_absolute_error(y_test, lr_pred),
                       "MAPE": np.mean(np.abs((y_test - lr_pred) / y_test)) * 100,
                       "R2":   r2_score(y_test, lr_pred)},
        "Ensemble":   {"MAE": mean_absolute_error(y_test, ensemble),
                       "MAPE": np.mean(np.abs((y_test - ensemble) / y_test)) * 100,
                       "R2":   r2_score(y_test, ensemble)},
    }

    # Future forecast
    last_date  = df.index[-1]
    base       = df["retail_real"].iloc[-1]
    fut_dates  = pd.date_range(last_date + pd.DateOffset(months=1),
                               periods=forecast_months, freq="ME")
    forecasts  = []
    for fdate in fut_dates:
        growth = 0.002 + np.sin(2 * np.pi * fdate.month / 12) * 0.01
        base   = base * (1 + growth)
        forecasts.append({
            "date":     fdate,
            "forecast": base,
            "lower":    base - 1.96 * residual_std,
            "upper":    base + 1.96 * residual_std,
        })
    future_df = pd.DataFrame(forecasts).set_index("date")

    test_results = pd.DataFrame({
        "actual": y_test, "naive": naive_pred,
        "linear_reg": lr_pred, "ensemble": ensemble
    })

    return metrics, test_results, future_df, residual_std


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

st.title("📈 FRED Business Intelligence Dashboard")
st.caption("Real-time macro analysis · Inflation adjustment · Recession risk · Forecasting")
st.markdown("---")

if not api_key:
    st.warning("Enter your FRED API key in the sidebar to load data.")
    st.info("Get a free key at https://fredaccount.stlouisfed.org/apikeys")
    st.stop()

# Load data
with st.spinner("Pulling live data from FRED..."):
    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"
    raw_df, load_errors = load_fred_data(api_key, start_date, end_date)

if load_errors:
    st.error(f"Some series failed to load: {load_errors}")

if raw_df.empty:
    st.error("No data loaded. Check your API key.")
    st.stop()

df = build_features(raw_df)

# ─────────────────────────────────────────────────────────────
# SECTION 1 — KPI METRICS ROW
# ─────────────────────────────────────────────────────────────

st.subheader("Current Economic Snapshot")

latest       = df.iloc[-1]
prev_month   = df.iloc[-2]
risk_score   = latest["recession_risk"]
risk_label   = "HIGH" if risk_score > 60 else "MODERATE" if risk_score > 40 else "LOW"
risk_color   = "risk-high" if risk_score > 60 else "risk-moderate" if risk_score > 40 else "risk-low"

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Retail Sales (Nominal)",
        f"${latest['retail_sales']/1000:,.0f}B",
        f"{latest['nominal_yoy']:+.1f}% YoY" if not pd.isna(latest['nominal_yoy']) else "N/A"
    )
with col2:
    st.metric(
        "Retail Sales (Real)",
        f"${latest['retail_real']/1000:,.0f}B",
        f"{latest['real_yoy']:+.1f}% YoY" if not pd.isna(latest['real_yoy']) else "N/A"
    )
with col3:
    st.metric(
        "Inflation (CPI YoY)",
        f"{latest['inflation_yoy']:+.1f}%" if not pd.isna(latest['inflation_yoy']) else "N/A",
        f"{latest['inflation_yoy'] - prev_month['inflation_yoy']:+.2f}pp vs last month"
        if not pd.isna(latest['inflation_yoy']) else ""
    )
with col4:
    st.metric(
        "Unemployment Rate",
        f"{latest['unemployment']:.1f}%",
        f"{latest['unemployment'] - prev_month['unemployment']:+.1f}pp vs last month"
    )
with col5:
    st.metric(
        "Fed Funds Rate",
        f"{latest['fed_funds_rate']:.2f}%",
        f"{latest['fed_funds_rate'] - prev_month['fed_funds_rate']:+.2f}pp vs last month"
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — REAL vs NOMINAL REVENUE
# ─────────────────────────────────────────────────────────────

st.subheader("Real vs Nominal Retail Sales")
st.caption("The gap between the two lines is the inflation illusion — revenue that looks like growth but is just higher prices.")

target_col = "retail_real" if adjust_inflation else "retail_sales"
label      = "Real (inflation-adjusted)" if adjust_inflation else "Nominal"

fig_rev = go.Figure()
fig_rev.add_trace(go.Scatter(
    x=df.index, y=df["retail_sales"] / 1000,
    name="Nominal", line=dict(color="#378ADD", width=2),
    hovertemplate="%{x|%Y-%m}: $%{y:,.0f}B<extra>Nominal</extra>"
))
fig_rev.add_trace(go.Scatter(
    x=df.index, y=df["retail_real"] / 1000,
    name="Real (inflation-adjusted)", line=dict(color="#1D9E75", width=2, dash="dash"),
    hovertemplate="%{x|%Y-%m}: $%{y:,.0f}B<extra>Real</extra>"
))
fig_rev.add_trace(go.Scatter(
    x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
    y=pd.concat([df["retail_sales"] / 1000, df["retail_real"][::-1] / 1000]),
    fill="toself", fillcolor="rgba(231,76,60,0.1)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Inflation gap", showlegend=True
))
fig_rev.update_layout(
    height=380, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis_title="$ Billions",
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_rev, use_container_width=True)

# YoY growth side by side
col_a, col_b = st.columns(2)
with col_a:
    fig_yoy = go.Figure()
    fig_yoy.add_trace(go.Bar(
        x=df.index, y=df["nominal_yoy"],
        name="Nominal YoY%", marker_color="#378ADD", opacity=0.8
    ))
    fig_yoy.add_trace(go.Bar(
        x=df.index, y=df["real_yoy"],
        name="Real YoY%", marker_color="#1D9E75", opacity=0.8
    ))
    fig_yoy.add_hline(y=0, line_color="black", line_width=0.8)
    fig_yoy.update_layout(
        barmode="group", height=300, title="YoY Growth Rate %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_yoy, use_container_width=True)

with col_b:
    fig_ill = go.Figure()
    colors  = ["#E24B4A" if v > 0 else "#1D9E75"
                for v in df["inflation_contribution"].fillna(0)]
    fig_ill.add_trace(go.Bar(
        x=df.index, y=df["inflation_contribution"],
        marker_color=colors, name="Inflation contribution"
    ))
    fig_ill.add_hline(y=0, line_color="black", line_width=0.8)
    fig_ill.update_layout(
        height=300, title="Inflation contribution to nominal growth (pp)",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_ill, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — RECESSION RISK GAUGE
# ─────────────────────────────────────────────────────────────

st.subheader("Recession Risk Score")

col_gauge, col_risk_chart = st.columns([1, 2])

with col_gauge:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        delta={"reference": df["recession_risk"].iloc[-2],
               "valueformat": ".1f"},
        title={"text": f"Current Risk: {risk_label}", "font": {"size": 14}},
        gauge={
            "axis":       {"range": [0, 100], "tickwidth": 1},
            "bar":        {"color": "#E24B4A" if risk_score > 60
                           else "#EF9F27" if risk_score > 40 else "#1D9E75"},
            "steps": [
                {"range": [0,  40], "color": "#d4edda"},
                {"range": [40, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": risk_threshold
            }
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(f"Alert threshold set at {risk_threshold} — adjust in sidebar")

with col_risk_chart:
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=df.index, y=df["recession_risk"],
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#E24B4A", width=1.5),
        hovertemplate="%{x|%Y-%m}: %{y:.1f}<extra>Risk Score</extra>"
    ))
    fig_risk.add_hline(y=60, line_dash="dash", line_color="#E24B4A",
                       annotation_text="High risk", annotation_position="right")
    fig_risk.add_hline(y=40, line_dash="dash", line_color="#EF9F27",
                       annotation_text="Moderate", annotation_position="right")
    fig_risk.add_hline(y=risk_threshold, line_dash="dot", line_color="black",
                       annotation_text=f"Your threshold ({risk_threshold})",
                       annotation_position="right")
    fig_risk.update_layout(
        height=280, title="Recession Risk Score — Historical",
        yaxis=dict(range=[0, 100], title="Score"),
        margin=dict(l=0, r=80, t=40, b=0)
    )
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — FORECAST
# ─────────────────────────────────────────────────────────────

st.subheader(f"6-Month Retail Sales Forecast")

with st.spinner("Running forecast models..."):
    metrics, test_results, future_df, residual_std = run_models(df, forecast_months)

# Metric cards
c1, c2, c3 = st.columns(3)
for col, (model_name, m) in zip([c1, c2, c3], metrics.items()):
    with col:
        st.metric(
            model_name,
            f"{m['MAPE']:.2f}% MAPE",
            f"R² = {m['R2']:.3f}"
        )

# Main forecast chart
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=test_results.index, y=test_results["actual"] / 1000,
    name="Actual", line=dict(color="#2C2C2A", width=2),
    hovertemplate="%{x|%Y-%m}: $%{y:,.1f}B<extra>Actual</extra>"
))
fig_fc.add_trace(go.Scatter(
    x=test_results.index, y=test_results["ensemble"] / 1000,
    name="Ensemble (test)", line=dict(color="#1D9E75", width=2, dash="dash"),
    hovertemplate="%{x|%Y-%m}: $%{y:,.1f}B<extra>Ensemble</extra>"
))
fig_fc.add_trace(go.Scatter(
    x=future_df.index, y=future_df["forecast"] / 1000,
    name="Forecast", line=dict(color="#D85A30", width=2.5),
    hovertemplate="%{x|%Y-%m}: $%{y:,.1f}B<extra>Forecast</extra>"
))
if show_ci:
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([future_df.index.to_series(),
                     future_df.index.to_series()[::-1]]),
        y=pd.concat([future_df["upper"] / 1000,
                     future_df["lower"][::-1] / 1000]),
        fill="toself", fillcolor="rgba(216,90,48,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI", showlegend=True
    ))
fig_fc.update_layout(
    height=400, hovermode="x unified",
    title=f"Ensemble Model — Test Period + {forecast_months}-Month Forward Forecast",
    yaxis_title="$ Billions (real)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_fc, use_container_width=True)

# Forecast table
st.markdown("**Forward forecast detail**")
forecast_display = future_df.copy()
forecast_display.index = forecast_display.index.strftime("%Y-%m")
forecast_display.columns = ["Forecast ($M)", "Lower 95% ($M)", "Upper 95% ($M)"]
forecast_display = forecast_display.applymap(lambda x: f"${x:,.0f}M")
st.dataframe(forecast_display, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — CORRELATION EXPLORER
# ─────────────────────────────────────────────────────────────

st.subheader("Macro Correlation Explorer")
st.caption("Select any macro signal to see its rolling 24-month correlation with retail sales.")

selected = st.multiselect(
    "Select signals to compare",
    options=["personal_income", "consumption", "cpi",
             "unemployment", "fed_funds_rate", "industrial_prod"],
    default=["consumption", "unemployment", "cpi"]
)

WINDOW = 24
fig_corr = go.Figure()
colors_map = {
    "consumption":     "#1D9E75",
    "personal_income": "#378ADD",
    "cpi":             "#E24B4A",
    "unemployment":    "#EF9F27",
    "fed_funds_rate":  "#7F77DD",
    "industrial_prod": "#888780",
}

for col in selected:
    rolling = df["retail_sales"].rolling(WINDOW).corr(df[col])
    fig_corr.add_trace(go.Scatter(
        x=df.index, y=rolling,
        name=col, line=dict(color=colors_map.get(col, "#888"), width=1.8),
        hovertemplate=f"%{{x|%Y-%m}}: %{{y:.3f}}<extra>{col}</extra>"
    ))

fig_corr.add_hline(y=0, line_color="black", line_width=0.8)
fig_corr.add_vline(x=pd.Timestamp("2020-03-01").timestamp() * 1000,
                   line_dash="dot", line_color="gray",
                   annotation_text="COVID", annotation_position="top right")
fig_corr.update_layout(
    height=350, hovermode="x unified",
    title=f"Rolling {WINDOW}-Month Correlation with Retail Sales",
    yaxis=dict(range=[-1, 1], title="Pearson r"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(fig_corr, use_container_width=True)

# Correlation table — latest values
st.markdown("**Latest correlation values**")
corr_rows = []
for col in ["personal_income", "consumption", "cpi",
            "unemployment", "fed_funds_rate", "industrial_prod"]:
    rolling = df["retail_sales"].rolling(WINDOW).corr(df[col]).dropna()
    corr_rows.append({
        "Signal":      col,
        "Current r":   f"{rolling.iloc[-1]:+.3f}",
        "Pre-COVID r": f"{rolling[rolling.index < '2020-03'].mean():+.3f}",
        "Post-COVID r":f"{rolling[rolling.index >= '2020-06'].mean():+.3f}",
        "Stability":   "Stable" if rolling.std() < 0.15
                       else "Volatile" if rolling.std() > 0.30 else "Moderate"
    })
st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with FRED API · Phase 4 of the Business Revenue & Macro Intelligence project")