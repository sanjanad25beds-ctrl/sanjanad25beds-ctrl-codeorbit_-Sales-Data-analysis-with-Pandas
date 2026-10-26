# SalesPulse AI

**AI-Powered Sales Intelligence and Business Decision Platform**

> From Raw Sales Data to Smarter Business Decisions.

SalesPulse AI converts raw sales datasets into meaningful business insights:
it cleans messy data, builds an executive dashboard, segments customers with
RFM analysis, forecasts future sales, detects anomalies, and produces
data-grounded business recommendations — plus a downloadable executive PDF
report.

Built for internship project evaluation, college demonstrations, portfolio /
GitHub presentation, and as a foundation for a future commercial SaaS product.

## Features

| Module | What it does |
|---|---|
| Smart Data Upload | CSV/Excel upload, preview, dtypes, missing values, duplicates, field validation. Sample dataset loads by default. |
| Automated Data Cleaning | Duplicates, mixed date formats, invalid rows, category fixes, missing-value fills, quality score 0–100, full transformation log. Raw data is never mutated. |
| Sales Intelligence Dashboard | KPIs (revenue, profit, orders, AOV, units, margin, best product/region) + interactive Plotly charts with date/category/region filters. |
| Product Performance | Top 10, most purchased, highest profit, lowest performers, declining products. |
| Customer Intelligence | RFM scoring (1–5) and 5 segments: High-value, Loyal, Potential, At-risk, Low-engagement, with revenue contribution. |
| AI Sales Forecasting | Monthly revenue model (linear trend + annual seasonality); 1/3/6-month horizons; 95% uncertainty band. Refuses to forecast when history is insufficient (<60 days / <6 months). |
| Anomaly Detection | Rolling z-score + Isolation Forest on daily revenue/transactions; severity levels and explanations. |
| AI Business Advisor | Rule-based recommendations grounded in computed metrics: problem, supporting data, business impact, suggested action, potential outcome. No guaranteed outcomes, no external LLM calls. |
| Executive Reports | One-click PDF report (ReportLab) covering all modules. |

## Quick start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the sample dataset
python scripts/generate_sample_data.py

# 4. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Running the tests

```bash
python -m pytest tests/ -q
```

## Folder structure

```
SalesPulse_AI/
├── app.py                    # Streamlit entry point (10-page sidebar nav)
├── salespulse/
│   ├── data.py               # upload loading, validation, profiling
│   ├── cleaning.py           # cleaning engine + quality score
│   ├── analytics.py          # KPIs, trends, product/region aggregations
│   ├── customers.py          # RFM scoring + segmentation
│   ├── forecasting.py        # time-series forecasting + uncertainty band
│   ├── anomalies.py          # z-score + Isolation Forest detection
│   ├── advisor.py            # rule-based, data-grounded recommendations
│   └── report.py             # ReportLab PDF generation
├── data/sample_sales.csv     # seeded sample dataset (1,500+ transactions)
├── scripts/generate_sample_data.py
├── tests/test_core.py        # pytest: KPIs, cleaning, RFM, forecast, advisor
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map and data flow.

## Honesty rules built into the app

- **No hardcoded metrics.** Every KPI and chart is computed from the active dataset.
- **No fabricated forecasts.** If history is under 60 days or spans fewer than
  6 months, the app shows an "insufficient data" message instead of predictions.
- **No guaranteed outcomes.** The Advisor frames results as *potential* and
  cites the supporting numbers.
- **No silent changes.** Cleaning produces a cleaned copy plus a log; the
  uploaded file is never modified.
