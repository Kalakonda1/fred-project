# FRED Business Intelligence Dashboard

> A full-stack data analytics project built on real Federal Reserve economic data — featuring inflation analysis, recession risk scoring, machine learning forecasting, and a live interactive dashboard.

🚀 **Live App**: [https://fred-project-fmofzjlj2sctsctrav7rkr.streamlit.app/](https://fred-project-fmofzjlj2sctsctrav7rkr.streamlit.app/)

📁 **Repository**: [https://github.com/Kalakonda1/fred-project](https://github.com/Kalakonda1/fred-project)

---

## What This Project Does

This project pulls 7 live macro series from the Federal Reserve Economic Data (FRED) API and builds a complete business intelligence pipeline answering the real-world question:

> *"Is our revenue growth real — or just inflation? And what does the macro environment say about the next 6 months?"*

### Features

- **Live data pipeline** — pulls real US economic data directly from the Federal Reserve
- **Inflation adjustment** — separates real growth from nominal (price-driven) growth
- **Recession risk scoring** — composite 0-100 risk score built from 4 macro signals
- **Machine learning forecasting** — 3 models compared, ensemble delivers 1.54% MAPE
- **Interactive dashboard** — live Streamlit app with date range picker, sliders, and charts
- **Rolling correlation explorer** — see how macro relationships shifted before and after COVID

---

## Project Structure

```
fred_project/
├── finance_fred.py        <- 1: FRED data pipeline + anomaly detection
├── eda.py                 <- 2: Deep EDA — real vs nominal, recession risk
├── forecast.py            <- 3: Forecasting — baseline, linear regression, ensemble
├── dashboard.py           <- 4: Live Streamlit dashboard
├── requirements.txt       <- Package dependencies
├── .env.example           <- API key template
├── .env                   <- Your real API key (never committed)
├── .gitignore             <- Protects secrets and output files
└── README.md              <- This file
```

---

## Data Sources

| Series ID | Description | Frequency |
|---|---|---|
| `RETAILSMNSA` | US Retail Sales | Monthly |
| `DSPIC96` | Real Disposable Personal Income | Monthly |
| `PCE` | Personal Consumption Expenditure | Monthly |
| `CPIAUCSL` | Consumer Price Index (Inflation) | Monthly |
| `INDPRO` | Industrial Production Index | Monthly |
| `UNRATE` | Unemployment Rate | Monthly |
| `FEDFUNDS` | Federal Funds Interest Rate | Monthly |

All data sourced from the St. Louis Federal Reserve (FRED). License: CC0 Public Domain.

---

## Key Findings

- **Inflation illusion**: On average 2.7% of nominal retail growth per year was purely price increases — not real demand
- **COVID anomaly**: April 2020 scored 80/100 on the recession risk model — correctly identified without hardcoding any dates
- **2022 reality check**: Nominal growth looked like +3.8% but real growth was actually -2.4% — the business was shrinking while prices masked it
- **Best forecast model**: Ensemble (65% Linear Regression + 35% Seasonal Naive) achieved 1.54% MAPE on 24-month test period
- **Post-COVID shift**: Personal income correlation with retail sales dropped sharply after COVID — what people earn now matters less than what they choose to spend

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kalakonda1/fred-project.git
cd fred-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free FRED API key

Go to [fredaccount.stlouisfed.org/apikeys](https://fredaccount.stlouisfed.org/apikeys), register with your email and receive your key instantly.

### 4. Configure your API key

```bash
copy .env.example .env
```

Open `.env` and replace the placeholder:

```
FRED_API_KEY=your_actual_key_here
```

### 5. Run the project phases in order

```bash
# 1 — Pull data and detect anomalies
python finance_fred.py

# 2 — Deep EDA
python eda.py

# 3 — Forecasting
python forecast.py

# 4 — Launch dashboard locally
streamlit run dashboard.py
```

---

## Breakdown

### 1 — Data Foundation (finance_fred.py)
Pulls 7 FRED series across 180 months (2010-2024). Runs seasonality analysis, lag correlations, and z-score anomaly detection. Outputs clean CSVs to fred_output/.

### 2 — Deep EDA (peda.py)
Answers: Is the growth real? Deflates retail sales by CPI to find real vs nominal growth. Builds a composite recession risk score. Computes rolling 24-month correlations to track how macro relationships evolved over time.

### 3 — Forecasting (forecast.py)
Builds 3 models — seasonal naive baseline, linear regression with macro features, and an ensemble. Uses walk-forward train/test split to prevent data leakage. Generates a 6-month forward forecast with 95% confidence intervals.

### 4 — Dashboard (dashboard.py)
Live Streamlit app with 5 interactive sections: economic snapshot KPIs, real vs nominal revenue charts, recession risk gauge, forecast with confidence bands, and a macro correlation explorer. Deployed publicly on Streamlit Cloud.

---

## Problems Encountered & How We Fixed Them

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | .gitignore not staging in Git | File saved with wrong UTF-8 BOM encoding on Windows | Rewrote using Set-Content with -Encoding utf8 in PowerShell |
| 2 | pandas==2.2.2 install failed | Python 3.13 had no prebuilt wheel — required C compilation with missing Visual Studio | Switched to unpinned requirements.txt letting pip choose compatible versions |
| 3 | .env.example not created by echo | PowerShell echo behaves differently from bash | Created file manually in VS Code with Ctrl+N |
| 4 | Stray New Text Document.txt in repo | Windows auto-created file in project folder | Deleted via VS Code Explorer |
| 5 | Plotly add_vline TypeError | Plotly bug with string dates — x="2020-03-01" caused integer + string crash | Converted to Unix timestamp: pd.Timestamp("2020-03-01").timestamp() * 1000 |
| 6 | pywin32==308 broke Streamlit Cloud | pip freeze captured Windows-only packages that don't exist on Linux | Replaced full freeze output with a minimal 9-package requirements.txt |
| 7 | cffi==1.17.1 failed to build on Cloud | Python 3.14 had no prebuilt wheel — required missing ffi.h system header | Forced Python 3.11 via Streamlit Cloud UI settings |
| 8 | runtime.txt ignored by Streamlit Cloud | Newer Streamlit Cloud infrastructure ignores runtime.txt | Set Python version directly in the deploy UI dropdown |
| 9 | KeyError: consumption in dashboard | FRED series failed to load silently, crashing build_features | Added column existence check with np.nan fallback |
| 10 | Deployment stuck 15+ minutes | Python 3.14 compiled heavy packages from source — no prebuilt wheels for scipy, scikit-learn | Forced Python 3.11 which has full wheel support — deploy time under 2 minutes |

---

## Requirements

- Python 3.11+
- See requirements.txt for full package list
- Free FRED API key from fredaccount.stlouisfed.org

---

## License

MIT — free to use, modify, and distribute with attribution.

---

*Built as a hands-on data analytics learning project covering real-world data pipelines, EDA, machine learning forecasting, and cloud deployment.*