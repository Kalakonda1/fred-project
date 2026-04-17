# Business Revenue & Macro Intelligence System
> Analyzing US economic trends using real Federal Reserve (FRED) data

---

## What this project does

Pulls 7 live macro series from the Federal Reserve Economic Data (FRED) API
and builds a full business intelligence pipeline:

- **Revenue trend analysis** — US retail sales with seasonality decomposition
- **Macro correlation analysis** — which economic signals drive revenue, and with what lag
- **Anomaly detection** — z-score method flags unusual months automatically
- **Dashboard** — 6-panel chart saved as a PNG

---

## Project structure

```
fred_project/
├── fred_business_intelligence.py   ← main analysis script
├── requirements.txt                ← all dependencies
├── .env.example                    ← environment variable template
├── .env                            ← your real secrets (not committed)
├── .gitignore                      ← protects .env and output files
└── README.md                       ← this file
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/fred-project.git
cd fred-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key

Copy the example env file:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Then open `.env` and replace the placeholder with your real FRED API key:

```
FRED_API_KEY=your_actual_key_here
```

Get a free key at: https://fredaccount.stlouisfed.org/apikeys
(Instant — just needs an email address)

### 4. Run the project

```bash
python fred_business_intelligence.py
```

Output files will be saved to `fred_output/`:
- `fred_macro_data.csv` — cleaned master dataset (168 months × 7 series)
- `anomalies.csv` — flagged anomalous months
- `fred_dashboard.png` — full 6-panel visual dashboard

---

## Data sources (FRED series used)

| Series ID      | Description                          | Frequency |
|----------------|--------------------------------------|-----------|
| `RETAILSMNSA`  | US Retail Sales                      | Monthly   |
| `DSPIC96`      | Real Disposable Personal Income      | Monthly   |
| `PCE`          | Personal Consumption Expenditure     | Monthly   |
| `CPIAUCSL`     | Consumer Price Index (Inflation)     | Monthly   |
| `INDPRO`       | Industrial Production Index          | Monthly   |
| `UNRATE`       | Unemployment Rate                    | Monthly   |
| `FEDFUNDS`     | Federal Funds Interest Rate          | Monthly   |

All data is sourced from the St. Louis Federal Reserve (FRED).
License: Public domain / CC0.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full package list

---

## License

MIT — free to use, modify, and distribute.

## Deep EDA

Run after Setup:

    python eda.py

Produces:
- `fred_output/real_vs_nominal.csv` — inflation adjusted retail sales
- `fred_output/recession_signals.csv` — monthly risk scores  
- `fred_output/rolling_correlations.csv` — 24 month rolling correlations
- `fred_output/phase2_dashboard.png` — 6 panel dashboard

Key findings:
- Average 2.7% of nominal growth per year is pure inflation illusion
- COVID April 2020 scored 80/100 on recession risk (model validation)
- Post-COVID consumer behaviour shifted — spending patterns now matter more than income levels


##  Forecasting

Run after EDA:

    python forecast.py

Builds 3 models and compares them:
- Seasonal naive baseline
- Linear regression with macro features
- Ensemble (weighted average)

Produces a 6-month forward forecast with 95% confidence intervals.

##  Interactive Dashboard

Run the live dashboard:

    streamlit run dashboard.py

Features:
- Live FRED data pull with date range picker
- Real vs nominal revenue with inflation gap
- Recession risk gauge with custom alert threshold
- 6-month forecast with 95% confidence bands
- Rolling correlation explorer with COVID impact