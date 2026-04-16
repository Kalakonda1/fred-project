"""
=============================================================
PHASE 2 — Deep EDA on FRED Macro Data
Business Revenue & Macro Intelligence System
=============================================================

REAL ANALYST QUESTIONS WE ARE ANSWERING:
  Q1. Is retail sales growth real or just inflation?
  Q2. Can we build a recession risk score from macro signals?
  Q3. Does the relationship between income and retail sales
      change over time and did COVID break it?

PREREQUISITES:
  - Run finance_fred.py first to generate fred_output/fred_macro_data.csv
  - pip install -r requirements.txt

OUTPUT FILES saved to fred_output/:
  - real_vs_nominal.csv
  - recession_signals.csv
  - rolling_correlations.csv
  - phase2_dashboard.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:,.2f}".format)

INPUT_FILE = "fred_output/fred_macro_data.csv"
OUTPUT_DIR = "fred_output"

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 2 — Deep EDA on FRED Macro Data")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, index_col="date", parse_dates=True)

print(f"\nLoaded: {df.shape[0]} months x {df.shape[1]} columns")
print(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 1 — REAL vs NOMINAL RETAIL SALES
# ─────────────────────────────────────────────────────────────
# CONCEPT: Nominal = raw dollar number. Real = after removing
# inflation. If CPI rose 8% and retail sales rose 8%, real
# growth is actually 0. We are stripping out the price illusion.
# Formula: Real Sales = (Nominal Sales / CPI) x 100

print("\n" + "=" * 60)
print("ANALYSIS 1 — Real vs Nominal Retail Sales")
print("=" * 60)

cpi_base = df["cpi"].iloc[0]
df["cpi_indexed"]             = df["cpi"] / cpi_base * 100
df["retail_real"]             = df["retail_sales"] / df["cpi"] * cpi_base
df["nominal_yoy"]             = df["retail_sales"].pct_change(12) * 100
df["real_yoy"]                = df["retail_real"].pct_change(12) * 100
df["inflation_yoy"]           = df["cpi"].pct_change(12) * 100
df["inflation_contribution"]  = df["nominal_yoy"] - df["real_yoy"]

print("\nNominal vs Real Retail Sales Growth YoY — selected years:")
print(f"\n{'Year':<8} {'Nominal%':>10} {'Real%':>10} {'Inflation%':>12} {'Illusion%':>12}")
print("-" * 54)

annual = df.resample("YE").last()[
    ["nominal_yoy", "real_yoy", "inflation_yoy", "inflation_contribution"]
].dropna()

for date, row in annual.iterrows():
    flag = " <- COVID"           if date.year == 2020 else \
           " <- Stimulus"        if date.year == 2021 else \
           " <- Inflation surge" if date.year == 2022 else ""
    print(f"  {date.year:<6} {row['nominal_yoy']:>+10.1f} {row['real_yoy']:>+10.1f} "
          f"{row['inflation_yoy']:>+12.1f} {row['inflation_contribution']:>+12.1f}{flag}")

avg_illusion = df["inflation_contribution"].mean()
print(f"\nAverage inflation contribution to nominal growth: {avg_illusion:+.1f}% per year")
print(f"On average {avg_illusion:.1f}% of reported revenue growth was just price increases")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 2 — RECESSION RISK SCORE
# ─────────────────────────────────────────────────────────────
# CONCEPT: No single indicator predicts recessions perfectly.
# We combine 4 signals into a composite score from 0 to 100.
#   - Unemployment:          rising = bad
#   - Fed funds rate:        very high = tightening = bad
#   - Industrial production: falling = bad
#   - Real retail sales:     falling = bad
# Each is normalized 0-1, weighted, and combined.

print("\n" + "=" * 60)
print("ANALYSIS 2 — Recession Risk Score")
print("=" * 60)

df["indpro_yoy"] = df["industrial_prod"].pct_change(12) * 100

def normalize(series, invert=False):
    mn, mx = series.min(), series.max()
    norm = (series - mn) / (mx - mn)
    return 1 - norm if invert else norm

df["score_unemployment"] = normalize(df["unemployment"])
df["score_fed_rate"]     = normalize(df["fed_funds_rate"])
df["score_indpro"]       = normalize(df["indpro_yoy"],  invert=True)
df["score_real_retail"]  = normalize(df["real_yoy"],    invert=True)

weights = {
    "score_unemployment": 0.30,
    "score_fed_rate":     0.20,
    "score_indpro":       0.20,
    "score_real_retail":  0.30,
}

df["recession_risk"] = sum(df[col] * w for col, w in weights.items()) * 100

print("\nTop 15 highest risk months:")
print(f"{'Month':<10} {'Risk Score':>12}   {'Context'}")
print("-" * 48)

for date, score in df["recession_risk"].nlargest(15).items():
    context = ""
    if date.year == 2020: context = "<- COVID recession"
    elif date.year == 2022: context = "<- Rate hike cycle"
    elif date.year == 2023: context = "<- Tightening peak"
    print(f"  {date.strftime('%Y-%m'):<8} {score:>12.1f}   {context}")

current_risk  = df["recession_risk"].iloc[-1]
risk_level    = "HIGH" if current_risk > 60 else "MODERATE" if current_risk > 40 else "LOW"
print(f"\nCurrent risk score: {current_risk:.1f} / 100 — {risk_level}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS 3 — ROLLING CORRELATION
# ─────────────────────────────────────────────────────────────
# CONCEPT: A static correlation tells you the average over 15
# years. But relationships change. Rolling correlation shows
# HOW the relationship evolved month by month.
# Window = 24 months so each point uses 2 years of data.

print("\n" + "=" * 60)
print("ANALYSIS 3 — Rolling 24-Month Correlations with Retail Sales")
print("=" * 60)

WINDOW = 24
predictors = ["personal_income", "consumption", "cpi",
              "unemployment", "fed_funds_rate", "industrial_prod"]

rolling_corr = pd.DataFrame(index=df.index)

for col in predictors:
    rolling_corr[col] = (
        df["retail_sales"]
        .rolling(WINDOW)
        .corr(df[col])
    )

print(f"\nRolling {WINDOW}-month correlation — latest values:")
print(f"{'Series':<22} {'Current r':>10} {'Min r':>10} {'Max r':>10} {'Stability'}")
print("-" * 65)

for col in predictors:
    series  = rolling_corr[col].dropna()
    current = series.iloc[-1]
    mn      = series.min()
    mx      = series.max()
    spread  = mx - mn
    stability = "Stable" if spread < 0.3 else "Volatile" if spread > 0.6 else "Moderate"
    print(f"  {col:<20} {current:>+10.3f} {mn:>+10.3f} {mx:>+10.3f}   {stability}")

# COVID impact: compare pre vs post correlation
pre_covid  = rolling_corr[rolling_corr.index < "2020-03"]
post_covid = rolling_corr[rolling_corr.index >= "2020-06"]

print("\nCOVID impact on correlations (pre vs post):")
print(f"{'Series':<22} {'Pre-COVID r':>13} {'Post-COVID r':>13} {'Change':>10}")
print("-" * 60)

for col in predictors:
    pre  = pre_covid[col].mean()
    post = post_covid[col].mean()
    chg  = post - pre
    arrow = "UP" if chg > 0.05 else "DOWN" if chg < -0.05 else "flat"
    print(f"  {col:<20} {pre:>+13.3f} {post:>+13.3f} {chg:>+10.3f}  {arrow}")

# ─────────────────────────────────────────────────────────────
# EXPORT ANALYSIS DATA
# ─────────────────────────────────────────────────────────────

real_nominal_cols = [
    "retail_sales", "retail_real", "nominal_yoy",
    "real_yoy", "inflation_yoy", "inflation_contribution"
]
df[real_nominal_cols].to_csv(f"{OUTPUT_DIR}/real_vs_nominal.csv")

recession_cols = [
    "unemployment", "fed_funds_rate", "indpro_yoy",
    "real_yoy", "recession_risk"
]
df[recession_cols].to_csv(f"{OUTPUT_DIR}/recession_signals.csv")
rolling_corr.to_csv(f"{OUTPUT_DIR}/rolling_correlations.csv")

print(f"\nExported:")
print(f"  -> fred_output/real_vs_nominal.csv")
print(f"  -> fred_output/recession_signals.csv")
print(f"  -> fred_output/rolling_correlations.csv")

# ─────────────────────────────────────────────────────────────
# PHASE 2 DASHBOARD — 6 PANELS
# ─────────────────────────────────────────────────────────────

print("\nGenerating Phase 2 dashboard...")

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 20))
fig.suptitle("Phase 2 — Deep EDA Dashboard\nReal Growth, Recession Risk & Rolling Correlations",
             fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Nominal vs Real retail sales level
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(df.index, df["retail_sales"] / 1000, color="#378ADD",
         linewidth=1.5, label="Nominal retail sales")
ax1.plot(df.index, df["retail_real"] / 1000, color="#1D9E75",
         linewidth=1.5, linestyle="--", label="Real retail sales (inflation adjusted)")
ax1.fill_between(df.index,
                 df["retail_real"] / 1000,
                 df["retail_sales"] / 1000,
                 alpha=0.15, color="#E24B4A", label="Inflation illusion")
ax1.set_title("Nominal vs Real Retail Sales — the inflation gap", fontsize=12)
ax1.set_ylabel("$ Billions")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
ax1.legend(fontsize=9)

# Panel 2: Nominal vs Real YoY growth bars
ax2 = fig.add_subplot(3, 2, 3)
width = 20
ax2.bar(df["nominal_yoy"].dropna().index - pd.Timedelta(days=12),
        df["nominal_yoy"].dropna(), width=width,
        color="#378ADD", alpha=0.8, label="Nominal YoY%")
ax2.bar(df["real_yoy"].dropna().index + pd.Timedelta(days=12),
        df["real_yoy"].dropna(), width=width,
        color="#1D9E75", alpha=0.8, label="Real YoY%")
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Nominal vs Real Growth Rate YoY %", fontsize=11)
ax2.set_ylabel("YoY %")
ax2.legend(fontsize=9)

# Panel 3: Inflation contribution
ax3 = fig.add_subplot(3, 2, 4)
colors = ["#E24B4A" if v > 0 else "#1D9E75"
          for v in df["inflation_contribution"].dropna()]
ax3.bar(df["inflation_contribution"].dropna().index,
        df["inflation_contribution"].dropna(),
        color=colors, width=25)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_title("Inflation contribution to nominal growth\n(red = inflation inflating the number)",
              fontsize=11)
ax3.set_ylabel("Percentage points")

# Panel 4: Recession risk score
ax4 = fig.add_subplot(3, 2, 5)
risk = df["recession_risk"].dropna()
ax4.fill_between(risk.index, risk, alpha=0.3, color="#E24B4A")
ax4.plot(risk.index, risk, color="#E24B4A", linewidth=1.5)
ax4.axhline(60, color="#E24B4A", linewidth=1, linestyle="--", alpha=0.6)
ax4.axhline(40, color="#EF9F27", linewidth=1, linestyle="--", alpha=0.6)
ax4.text(risk.index[-1], 62, "High risk", fontsize=8, color="#E24B4A")
ax4.text(risk.index[-1], 42, "Moderate", fontsize=8, color="#EF9F27")
ax4.set_title("Composite Recession Risk Score (0-100)", fontsize=11)
ax4.set_ylabel("Risk Score")
ax4.set_ylim(0, 100)

# Panel 5: Rolling correlations
ax5 = fig.add_subplot(3, 2, 6)
colors_rc = ["#378ADD", "#1D9E75", "#E24B4A", "#EF9F27", "#7F77DD", "#888780"]
for col, color in zip(["consumption", "personal_income", "unemployment",
                        "cpi", "fed_funds_rate", "industrial_prod"], colors_rc):
    ax5.plot(rolling_corr.index, rolling_corr[col],
             linewidth=1.2, label=col, color=color)
ax5.axhline(0, color="black", linewidth=0.8)
ax5.axvline(pd.Timestamp("2020-03"), color="gray",
            linewidth=1, linestyle="--", alpha=0.7)
ax5.text(pd.Timestamp("2020-04"), -0.9, "COVID", fontsize=8, color="gray")
ax5.set_title(f"Rolling {WINDOW}-month Correlation with Retail Sales", fontsize=11)
ax5.set_ylabel("Pearson r")
ax5.set_ylim(-1, 1)
ax5.legend(fontsize=8, loc="lower left")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/phase2_dashboard.png", dpi=150, bbox_inches="tight")

print(f"  -> fred_output/phase2_dashboard.png")

print("\n" + "=" * 60)
print("PHASE 2 COMPLETE")
print("=" * 60)
print("\nKey findings:")
print(f"  1. Avg inflation illusion in nominal growth: {avg_illusion:.1f}% per year")
print(f"  2. Current recession risk score: {current_risk:.1f}/100 ({risk_level})")
print(f"  3. Strongest rolling predictor of retail sales: consumption")
print("\nNext: Phase 3 — Forecasting future retail sales")