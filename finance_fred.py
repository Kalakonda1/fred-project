"""
=============================================================
BUSINESS REVENUE & MACRO INTELLIGENCE SYSTEM
Using FRED (Federal Reserve Economic Data)
=============================================================

REAL-WORLD PROBLEM WE'RE SOLVING:
  "Why did our revenue drop in Q3? Was it us — or the economy?"

  This is the #1 question every CFO and business analyst faces.
  By layering FRED macro data against business performance,
  we can separate internal problems from external headwinds.

PHASES IN THIS FILE:
  Phase 1 — Setup & API connection
  Phase 2 — Pull & clean 7 macro series from FRED
  Phase 3 — Revenue trend analysis + seasonality
  Phase 4 — Correlation: which macro signals drive revenue?
  Phase 5 — Anomaly detection (z-score method)
  Phase 6 — Export dashboard-ready data

PREREQUISITES:
  pip install fredapi pandas numpy matplotlib seaborn scipy

API KEY:
  Get yours free at: https://fredaccount.stlouisfed.org/apikeys
  Replace 'YOUR_API_KEY_HERE' below with your actual key.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from fredapi import Fred
import warnings
import os

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:,.2f}".format)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — edit this section
# ─────────────────────────────────────────────────────────────

from dotenv import load_dotenv
import os

load_dotenv()                          # reads the .env file
API_KEY = os.getenv("FRED_API_KEY")   # pulls the key into a variable

if not API_KEY:
    raise ValueError("FRED_API_KEY not found. Check your .env file.")  # <-- paste your FRED key here
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"
OUTPUT_DIR = "fred_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# PHASE 1 — CONNECT TO FRED
# ─────────────────────────────────────────────────────────────
# ANALYST NOTE: Always test connection before running long pulls.
# FRED has rate limits (~120 req/min) — we batch our series
# to avoid hitting them.

print("=" * 60)
print("PHASE 1 — Connecting to FRED API")
print("=" * 60)

fred = Fred(api_key=API_KEY)

# Define all series we want — this is our "data catalog"
SERIES = {
    # Revenue / demand signals
    "retail_sales":       ("RETAILSMNSA", "Retail Sales ($ Millions, NSA)"),
    "personal_income":    ("DSPIC96",     "Real Disposable Personal Income (Bil $)"),
    "consumption":        ("PCE",         "Personal Consumption Expenditure (Bil $)"),

    # Cost / supply signals
    "cpi":                ("CPIAUCSL",    "Consumer Price Index (1982-84=100)"),
    "industrial_prod":    ("INDPRO",      "Industrial Production Index"),

    # Economic environment
    "unemployment":       ("UNRATE",      "Unemployment Rate (%)"),
    "fed_funds_rate":     ("FEDFUNDS",    "Federal Funds Rate (%)"),
}

# ─────────────────────────────────────────────────────────────
# PHASE 2 — PULL & CLEAN DATA
# ─────────────────────────────────────────────────────────────
# KEY CONCEPT: FRED data comes at different frequencies.
# Retail sales = monthly, some others = quarterly.
# We resample everything to MONTHLY and forward-fill gaps.
# This is standard practice when merging mixed-frequency data.

print("\nPHASE 2 — Pulling macro series from FRED...")
print("-" * 40)

series_data = {}

for name, (series_id, description) in SERIES.items():
    try:
        s = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE
        )
        # Resample to month-end, forward fill up to 3 months
        s = s.resample("ME").last().ffill(limit=3)
        series_data[name] = s
        print(f"  ✓ {description}")
        print(f"    → {len(s)} monthly observations | "
              f"{s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')}")
        print(f"    → Latest value: {s.iloc[-1]:,.2f}  |  "
              f"Min: {s.min():,.2f}  Max: {s.max():,.2f}")
    except Exception as e:
        print(f"  ✗ Failed to fetch {series_id}: {e}")

# Combine into a single DataFrame aligned on date index
df = pd.DataFrame(series_data)
df.index.name = "date"
df = df.dropna(how="all")

print(f"\nMaster DataFrame: {df.shape[0]} months × {df.shape[1]} series")
print(f"Date range: {df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}")
print(f"\nMissing values per series:")
print(df.isnull().sum())

# ─────────────────────────────────────────────────────────────
# PHASE 3 — REVENUE TREND & SEASONALITY ANALYSIS
# ─────────────────────────────────────────────────────────────
# We use Retail Sales as our primary revenue proxy.
# ANALYST SKILLS: YoY growth, rolling averages, seasonal decomposition

print("\n" + "=" * 60)
print("PHASE 3 — Revenue Trend & Seasonality Analysis")
print("=" * 60)

df["retail_yoy_pct"] = df["retail_sales"].pct_change(12) * 100
df["retail_mom_pct"] = df["retail_sales"].pct_change(1) * 100
df["retail_ma12"]    = df["retail_sales"].rolling(12).mean()
df["retail_ma3"]     = df["retail_sales"].rolling(3).mean()

# Seasonality: average retail sales by calendar month
df["month_num"] = df.index.month
monthly_seasonality = df.groupby("month_num")["retail_sales"].mean()
monthly_seasonality_idx = monthly_seasonality / monthly_seasonality.mean() * 100

print("\nSeasonal Index by Month (100 = average):")
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
for m, idx in zip(month_names, monthly_seasonality_idx):
    bar = "█" * int(idx / 5)
    flag = " ◄ peak" if idx == monthly_seasonality_idx.max() else \
           " ◄ trough" if idx == monthly_seasonality_idx.min() else ""
    print(f"  {m:>3}: {idx:6.1f}  {bar}{flag}")

# ─────────────────────────────────────────────────────────────
# PHASE 4 — CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────
# KEY QUESTION: Which macro signal is the strongest leading
# indicator of retail revenue?
# We test both same-month and lagged correlations (1-6 months).
# ANALYST SKILL: Lag correlation is critical — the economy
# affects business with a delay, not instantly.

print("\n" + "=" * 60)
print("PHASE 4 — Macro Correlation Analysis")
print("=" * 60)

target = "retail_sales"
predictors = [c for c in series_data.keys() if c != target]

print(f"\nCorrelation with {target} (Pearson r):")
print(f"{'Series':<22} {'Same month':>12} {'Lag 1m':>10} {'Lag 3m':>10} {'Lag 6m':>10}")
print("-" * 66)

lag_corr_results = {}
for predictor in predictors:
    row = {}
    for lag in [0, 1, 3, 6]:
        shifted = df[predictor].shift(lag)
        valid   = df[[target]].join(shifted.rename("x")).dropna()
        r, p    = stats.pearsonr(valid[target], valid["x"])
        row[lag] = (r, p)
    lag_corr_results[predictor] = row

    r0 = lag_corr_results[predictor][0][0]
    r1 = lag_corr_results[predictor][1][0]
    r3 = lag_corr_results[predictor][3][0]
    r6 = lag_corr_results[predictor][6][0]
    print(f"  {predictor:<20} {r0:>+11.3f} {r1:>+10.3f} {r3:>+10.3f} {r6:>+10.3f}")

# ─────────────────────────────────────────────────────────────
# PHASE 5 — ANOMALY DETECTION (Z-SCORE METHOD)
# ─────────────────────────────────────────────────────────────
# ANALYST SKILL: Flag months where retail sales deviated
# sharply from what the macro environment would predict.
# These are your "investigate further" signals.
# We use a rolling z-score (window=24 months) to adapt to trends.

print("\n" + "=" * 60)
print("PHASE 5 — Anomaly Detection on Retail Sales")
print("=" * 60)

rolling_mean = df["retail_sales"].rolling(24, min_periods=12).mean()
rolling_std  = df["retail_sales"].rolling(24, min_periods=12).std()
df["zscore"] = (df["retail_sales"] - rolling_mean) / rolling_std

THRESHOLD = 2.0
anomalies = df[df["zscore"].abs() > THRESHOLD][["retail_sales", "retail_yoy_pct", "zscore"]]

print(f"\nAnomaly threshold: |z| > {THRESHOLD}")
print(f"Anomalies found: {len(anomalies)}")
print("\nAnomalous months:")
for date, row in anomalies.iterrows():
    direction = "SPIKE ▲" if row["zscore"] > 0 else "DROP  ▼"
    print(f"  {date.strftime('%Y-%m')}  {direction}  "
          f"Sales: ${row['retail_sales']:,.0f}M  "
          f"YoY: {row['retail_yoy_pct']:+.1f}%  "
          f"z={row['zscore']:.2f}")

# ─────────────────────────────────────────────────────────────
# PHASE 6 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 6 — Generating Charts")
print("=" * 60)

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Business Revenue & Macro Intelligence Dashboard\n(FRED Data)", 
             fontsize=16, fontweight="bold", y=0.98)

# --- Chart 1: Retail Sales Trend with anomalies ---
ax1 = fig.add_subplot(4, 2, (1, 2))
ax1.plot(df.index, df["retail_sales"] / 1000, color="#1D9E75", linewidth=1.5, label="Retail Sales")
ax1.plot(df.index, df["retail_ma12"] / 1000,  color="#533AB7", linewidth=2, linestyle="--", label="12-month MA")
# Shade anomaly months
for date in anomalies.index:
    ax1.axvspan(date - pd.DateOffset(days=15), date + pd.DateOffset(days=15),
                alpha=0.25, color="#E24B4A", zorder=0)
ax1.set_title("US Retail Sales — Monthly with Anomalies Highlighted", fontsize=12)
ax1.set_ylabel("$ Billions")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
ax1.legend(fontsize=9)

# --- Chart 2: YoY Growth Rate ---
ax2 = fig.add_subplot(4, 2, 3)
colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in df["retail_yoy_pct"].dropna()]
ax2.bar(df["retail_yoy_pct"].dropna().index, df["retail_yoy_pct"].dropna(), color=colors, width=25)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Retail Sales — Year-over-Year Growth %", fontsize=11)
ax2.set_ylabel("YoY %")

# --- Chart 3: Seasonal Index ---
ax3 = fig.add_subplot(4, 2, 4)
bars = ax3.bar(month_names, monthly_seasonality_idx.values,
               color=["#D85A30" if v >= 105 else "#1D9E75" if v <= 95 else "#888780"
                      for v in monthly_seasonality_idx.values])
ax3.axhline(100, color="black", linewidth=1, linestyle="--")
ax3.set_title("Seasonal Revenue Index by Month\n(100 = average month)", fontsize=11)
ax3.set_ylabel("Index")

# --- Chart 4: Unemployment vs Revenue ---
ax4 = fig.add_subplot(4, 2, 5)
ax4b = ax4.twinx()
ax4.plot(df.index, df["retail_sales"] / 1000, color="#1D9E75", linewidth=1.5, label="Retail Sales (L)")
ax4b.plot(df.index, df["unemployment"], color="#E24B4A", linewidth=1.5, linestyle="--", label="Unemployment % (R)")
ax4.set_title("Retail Sales vs Unemployment Rate", fontsize=11)
ax4.set_ylabel("Sales $ Billions", color="#1D9E75")
ax4b.set_ylabel("Unemployment %", color="#E24B4A")
ax4.legend(loc="upper left", fontsize=8)
ax4b.legend(loc="upper right", fontsize=8)

# --- Chart 5: CPI inflation vs Retail Sales Growth ---
ax5 = fig.add_subplot(4, 2, 6)
df["cpi_yoy"] = df["cpi"].pct_change(12) * 100
ax5.scatter(df["cpi_yoy"].dropna(), 
            df["retail_yoy_pct"].reindex(df["cpi_yoy"].dropna().index),
            alpha=0.5, color="#378ADD", s=20)
ax5.set_xlabel("CPI Inflation YoY %")
ax5.set_ylabel("Retail Sales Growth YoY %")
ax5.set_title("Inflation vs Revenue Growth\n(scatter — each dot = 1 month)", fontsize=11)
# Trend line
valid_scatter = df[["cpi_yoy", "retail_yoy_pct"]].dropna()
z = np.polyfit(valid_scatter["cpi_yoy"], valid_scatter["retail_yoy_pct"], 1)
p = np.poly1d(z)
x_line = np.linspace(valid_scatter["cpi_yoy"].min(), valid_scatter["cpi_yoy"].max(), 100)
ax5.plot(x_line, p(x_line), color="#E24B4A", linewidth=1.5, linestyle="--")

# --- Chart 6: Correlation heatmap ---
ax6 = fig.add_subplot(4, 2, (7, 8))
corr_df = df[list(series_data.keys())].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax6, square=True,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
ax6.set_title("Cross-Series Correlation Matrix", fontsize=11)
ax6.tick_params(axis="x", rotation=30)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/fred_dashboard.png", dpi=150, bbox_inches="tight")
print(f"  ✓ Dashboard saved → {OUTPUT_DIR}/fred_dashboard.png")

# ─────────────────────────────────────────────────────────────
# PHASE 7 — EXPORT CLEAN DATA
# ─────────────────────────────────────────────────────────────

df.drop(columns=["month_num"], inplace=True, errors="ignore")
df.to_csv(f"{OUTPUT_DIR}/fred_macro_data.csv")
anomalies.to_csv(f"{OUTPUT_DIR}/anomalies.csv")

print(f"  ✓ Master data saved → {OUTPUT_DIR}/fred_macro_data.csv")
print(f"  ✓ Anomalies saved   → {OUTPUT_DIR}/anomalies.csv")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"  Series pulled       : {len(series_data)}")
print(f"  Date range          : {START_DATE} → {END_DATE}")
print(f"  Total observations  : {df.shape[0]} months × {df.shape[1]} columns")
print(f"  Anomalies detected  : {len(anomalies)}")
print(f"\nTop insight from correlation analysis:")

# Find best predictor at any lag
best_r, best_pred, best_lag = 0, "", 0
for pred, lags in lag_corr_results.items():
    for lag, (r, p) in lags.items():
        if abs(r) > abs(best_r):
            best_r, best_pred, best_lag = r, pred, lag

print(f"  → '{best_pred}' at lag {best_lag}m has strongest correlation "
      f"with retail sales (r={best_r:.3f})")
print(f"\nNext step: Phase 2 — EDA deep dive on the exported data")
print(f"Files in ./{OUTPUT_DIR}/")