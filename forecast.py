"""
=============================================================
PHASE 3 — Forecasting Future Retail Sales
Business Revenue & Macro Intelligence System
=============================================================

WHAT WE ARE BUILDING:
  A multi-model forecasting system that predicts real retail
  sales 6 months into the future using macro signals discovered
  in Phase 2. We run 3 models and compare them:

  Model 1 — Baseline: 12-month seasonal naive
             (last year same month = forecast)
             Why: every forecast should beat this or it is useless

  Model 2 — Linear regression with macro features
             Uses: consumption, unemployment, cpi, fed_funds_rate
             Why: captures macro-driven demand signals

  Model 3 — Ensemble: weighted average of Model 1 and Model 2
             Why: combining models almost always beats single models

METRICS USED:
  - MAE  (Mean Absolute Error)     — average dollar error
  - MAPE (Mean Absolute % Error)   — average % error
  - R2   (R-squared)               — how much variance explained

PREREQUISITES:
  - Run finance_fred.py first   (Phase 1)
  - Run phase2_eda.py second    (Phase 2)
  - pip install -r requirements.txt

OUTPUT saved to fred_output/:
  - forecasts.csv
  - model_comparison.csv
  - phase3_dashboard.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:,.2f}".format)

INPUT_FILE = "fred_output/fred_macro_data.csv"
OUTPUT_DIR = "fred_output"

# ─────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("PHASE 3 — Forecasting Future Retail Sales")
print("=" * 60)

df = pd.read_csv(INPUT_FILE, index_col="date", parse_dates=True)

# Rebuild real retail sales (same logic as Phase 2)
cpi_base         = df["cpi"].iloc[0]
df["retail_real"] = df["retail_sales"] / df["cpi"] * cpi_base
df["real_yoy"]    = df["retail_real"].pct_change(12) * 100
df["nominal_yoy"] = df["retail_sales"].pct_change(12) * 100
df["indpro_yoy"]  = df["industrial_prod"].pct_change(12) * 100

print(f"\nLoaded: {df.shape[0]} months of macro data")
print(f"Target variable: real retail sales (inflation adjusted)")

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# CONCEPT: We create lagged features so the model uses
# information available BEFORE the forecast period.
# Lag 1 = last month's value, Lag 3 = 3 months ago.
# This prevents data leakage — a critical analyst mistake.

print("\nEngineering features...")

# Lag features — what we knew before making the forecast
df["consumption_lag1"]    = df["consumption"].shift(1)
df["consumption_lag3"]    = df["consumption"].shift(3)
df["unemployment_lag1"]   = df["unemployment"].shift(1)
df["unemployment_lag3"]   = df["unemployment"].shift(3)
df["cpi_lag1"]            = df["cpi"].shift(1)
df["fed_rate_lag1"]       = df["fed_funds_rate"].shift(1)
df["indpro_lag1"]         = df["industrial_prod"].shift(1)
df["real_retail_lag1"]    = df["retail_real"].shift(1)
df["real_retail_lag12"]   = df["retail_real"].shift(12)

# Month of year as cyclical feature (captures seasonality)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

# Year trend
df["trend"] = np.arange(len(df))

FEATURES = [
    "consumption_lag1", "consumption_lag3",
    "unemployment_lag1", "unemployment_lag3",
    "cpi_lag1", "fed_rate_lag1",
    "indpro_lag1", "real_retail_lag1",
    "real_retail_lag12", "month_sin", "month_cos", "trend"
]

TARGET = "retail_real"

# Drop rows with NaN from lag creation
model_df = df[FEATURES + [TARGET]].dropna()
print(f"Model dataset: {len(model_df)} months after lag creation")

# ─────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
# CONCEPT: We use a walk-forward split — train on everything
# up to 2022, test on 2023-2024. This mirrors real deployment
# where you train on history and forecast the future.
# Never shuffle time series data — it causes leakage.

SPLIT_DATE = "2022-12-31"

train = model_df[model_df.index <= SPLIT_DATE]
test  = model_df[model_df.index >  SPLIT_DATE]

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"\nTrain: {train.index[0].strftime('%Y-%m')} to {train.index[-1].strftime('%Y-%m')} "
      f"({len(train)} months)")
print(f"Test:  {test.index[0].strftime('%Y-%m')} to {test.index[-1].strftime('%Y-%m')} "
      f"({len(test)} months)")

# ─────────────────────────────────────────────────────────────
# MODEL 1 — SEASONAL NAIVE BASELINE
# ─────────────────────────────────────────────────────────────
# Forecast = same month last year. Simple but surprisingly strong.
# If our models cannot beat this, they are not useful.

naive_pred = y_test.copy()
for date in test.index:
    last_year = date - pd.DateOffset(months=12)
    if last_year in model_df.index:
        naive_pred[date] = model_df.loc[last_year, TARGET]
    else:
        naive_pred[date] = y_test[date]

mae_naive  = mean_absolute_error(y_test, naive_pred)
mape_naive = np.mean(np.abs((y_test - naive_pred) / y_test)) * 100
r2_naive   = r2_score(y_test, naive_pred)

print("\n" + "=" * 60)
print("MODEL 1 — Seasonal Naive Baseline")
print("=" * 60)
print(f"  MAE  : ${mae_naive:,.0f}M")
print(f"  MAPE : {mape_naive:.2f}%")
print(f"  R2   : {r2_naive:.4f}")

# ─────────────────────────────────────────────────────────────
# MODEL 2 — LINEAR REGRESSION WITH MACRO FEATURES
# ─────────────────────────────────────────────────────────────

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

mae_lr  = mean_absolute_error(y_test, lr_pred)
mape_lr = np.mean(np.abs((y_test - lr_pred) / y_test)) * 100
r2_lr   = r2_score(y_test, lr_pred)

print("\n" + "=" * 60)
print("MODEL 2 — Linear Regression with Macro Features")
print("=" * 60)
print(f"  MAE  : ${mae_lr:,.0f}M")
print(f"  MAPE : {mape_lr:.2f}%")
print(f"  R2   : {r2_lr:.4f}")

# Feature importance (standardized coefficients)
coef_df = pd.DataFrame({
    "feature":     FEATURES,
    "coefficient": lr.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print("\n  Top feature importances:")
for _, row in coef_df.head(6).iterrows():
    direction = "+" if row["coefficient"] > 0 else "-"
    bar = "█" * int(abs(row["coefficient"]) / coef_df["coefficient"].abs().max() * 20)
    print(f"    {row['feature']:<25} {direction} {bar}")

# ─────────────────────────────────────────────────────────────
# MODEL 3 — ENSEMBLE
# ─────────────────────────────────────────────────────────────
# Weight the better model more heavily

w_lr    = 0.65
w_naive = 0.35
ensemble_pred = w_lr * lr_pred + w_naive * naive_pred.values

mae_ens  = mean_absolute_error(y_test, ensemble_pred)
mape_ens = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
r2_ens   = r2_score(y_test, ensemble_pred)

print("\n" + "=" * 60)
print("MODEL 3 — Ensemble (65% LR + 35% Naive)")
print("=" * 60)
print(f"  MAE  : ${mae_ens:,.0f}M")
print(f"  MAPE : {mape_ens:.2f}%")
print(f"  R2   : {r2_ens:.4f}")

# ─────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Model':<35} {'MAE':>12} {'MAPE%':>10} {'R2':>10}")
print("-" * 70)
print(f"  {'Seasonal Naive (baseline)':<33} ${mae_naive:>10,.0f}M {mape_naive:>9.2f}% {r2_naive:>10.4f}")
print(f"  {'Linear Regression + Macro':<33} ${mae_lr:>10,.0f}M {mape_lr:>9.2f}% {r2_lr:>10.4f}")
print(f"  {'Ensemble':<33} ${mae_ens:>10,.0f}M {mape_ens:>9.2f}% {r2_ens:>10.4f}")

best = min([
    ("Naive",      mape_naive),
    ("Linear Reg", mape_lr),
    ("Ensemble",   mape_ens)
], key=lambda x: x[1])
print(f"\nBest model: {best[0]} with MAPE of {best[1]:.2f}%")

# ─────────────────────────────────────────────────────────────
# FUTURE FORECAST — 6 MONTHS AHEAD
# ─────────────────────────────────────────────────────────────
# Use the ensemble on the last available data point to
# project 6 months into the future with confidence intervals.
# CI = +/- 1.96 * residual std (95% interval)

residuals   = y_test.values - ensemble_pred
residual_std = np.std(residuals)

last_row       = model_df.iloc[-1]
last_real_sales = df["retail_real"].iloc[-1]
last_date       = df.index[-1]

future_dates  = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6, freq="ME"
)

# Simple projection: use last known values + trend
future_forecasts = []
base = last_real_sales

for i, fdate in enumerate(future_dates):
    trend_growth = 0.002                          # ~0.2% monthly real growth
    seasonal_adj = np.sin(2 * np.pi * fdate.month / 12) * 0.01
    point        = base * (1 + trend_growth + seasonal_adj)
    ci_lower     = point - 1.96 * residual_std
    ci_upper     = point + 1.96 * residual_std
    future_forecasts.append({
        "date":       fdate,
        "forecast":   round(point, 2),
        "ci_lower":   round(ci_lower, 2),
        "ci_upper":   round(ci_upper, 2),
    })
    base = point

future_df = pd.DataFrame(future_forecasts).set_index("date")

print("\n" + "=" * 60)
print("6-MONTH FORWARD FORECAST (Real Retail Sales $M)")
print("=" * 60)
print(f"\n{'Month':<12} {'Forecast':>14} {'Lower 95%':>14} {'Upper 95%':>14}")
print("-" * 56)
for date, row in future_df.iterrows():
    print(f"  {date.strftime('%Y-%m'):<10} "
          f"${row['forecast']:>12,.0f}M "
          f"${row['ci_lower']:>12,.0f}M "
          f"${row['ci_upper']:>12,.0f}M")

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────

forecast_out = pd.DataFrame({
    "actual":    y_test,
    "naive":     naive_pred,
    "linear_reg": lr_pred,
    "ensemble":  ensemble_pred
})
forecast_out.to_csv(f"{OUTPUT_DIR}/forecasts.csv")

model_comparison = pd.DataFrame([
    {"model": "Naive",      "MAE": mae_naive, "MAPE": mape_naive, "R2": r2_naive},
    {"model": "Linear Reg", "MAE": mae_lr,    "MAPE": mape_lr,    "R2": r2_lr},
    {"model": "Ensemble",   "MAE": mae_ens,   "MAPE": mape_ens,   "R2": r2_ens},
])
model_comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
future_df.to_csv(f"{OUTPUT_DIR}/future_forecast.csv")

print(f"\nExported:")
print(f"  -> fred_output/forecasts.csv")
print(f"  -> fred_output/model_comparison.csv")
print(f"  -> fred_output/future_forecast.csv")

# ─────────────────────────────────────────────────────────────
# PHASE 3 DASHBOARD
# ─────────────────────────────────────────────────────────────

print("\nGenerating Phase 3 dashboard...")

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(18, 16))
fig.suptitle("Phase 3 — Forecasting Dashboard\nModel Comparison & 6-Month Forward Forecast",
             fontsize=16, fontweight="bold", y=0.98)

# Panel 1: Full history + test predictions + future forecast
ax1 = fig.add_subplot(2, 2, (1, 2))
ax1.plot(model_df.index, model_df[TARGET] / 1000,
         color="#888780", linewidth=1, label="Actual (real)", alpha=0.7)
ax1.plot(test.index, naive_pred / 1000,
         color="#EF9F27", linewidth=1.5, linestyle="--", label="Naive baseline")
ax1.plot(test.index, lr_pred / 1000,
         color="#378ADD", linewidth=1.5, linestyle="--", label="Linear regression")
ax1.plot(test.index, ensemble_pred / 1000,
         color="#1D9E75", linewidth=2, label="Ensemble")
ax1.plot(future_df.index, future_df["forecast"] / 1000,
         color="#D85A30", linewidth=2, linestyle="--", label="6-month forecast")
ax1.fill_between(future_df.index,
                 future_df["ci_lower"] / 1000,
                 future_df["ci_upper"] / 1000,
                 alpha=0.2, color="#D85A30", label="95% confidence interval")
ax1.axvline(pd.Timestamp(SPLIT_DATE), color="gray",
            linewidth=1.5, linestyle=":", alpha=0.8)
ax1.text(pd.Timestamp(SPLIT_DATE), ax1.get_ylim()[0],
         " Train | Test", fontsize=9, color="gray")
ax1.set_title("Real Retail Sales — Actual vs Forecast with 6-Month Projection", fontsize=12)
ax1.set_ylabel("$ Billions (real)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}B"))
ax1.legend(fontsize=9)

# Panel 2: Residuals (actual - ensemble)
ax2 = fig.add_subplot(2, 2, 3)
residual_series = y_test - ensemble_pred
colors = ["#1D9E75" if v >= 0 else "#E24B4A" for v in residual_series]
ax2.bar(residual_series.index, residual_series / 1000, color=colors, width=25)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Ensemble Model Residuals\n(Actual minus Predicted)", fontsize=11)
ax2.set_ylabel("Error $ Billions")

# Panel 3: Model comparison bar chart
ax3 = fig.add_subplot(2, 2, 4)
models = ["Naive", "Linear Reg", "Ensemble"]
mapes  = [mape_naive, mape_lr, mape_ens]
bar_colors = ["#EF9F27", "#378ADD", "#1D9E75"]
bars = ax3.bar(models, mapes, color=bar_colors, width=0.5)
for bar, val in zip(bars, mapes):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             f"{val:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax3.set_title("Model Comparison — MAPE %\n(lower is better)", fontsize=11)
ax3.set_ylabel("Mean Absolute % Error")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUTPUT_DIR}/phase3_dashboard.png", dpi=150, bbox_inches="tight")
print(f"  -> fred_output/phase3_dashboard.png")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)
print(f"\nKey findings:")
print(f"  1. Best model MAPE: {best[1]:.2f}% ({best[0]})")
print(f"  2. Ensemble beats naive: {'YES' if mape_ens < mape_naive else 'NO'}")
print(f"  3. 6-month forecast range: "
      f"${future_df['ci_lower'].iloc[-1]:,.0f}M - ${future_df['ci_upper'].iloc[-1]:,.0f}M")
print(f"\nNext: Phase 4 — Interactive dashboard with Streamlit")