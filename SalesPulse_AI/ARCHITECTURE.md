# SalesPulse AI — Architecture

## Data flow

```
RAW DATA  →  DATA CLEANING  →  BUSINESS INTELLIGENCE  →  MACHINE LEARNING
     (CSV/Excel upload          (cleaned copy + log)      (KPIs, trends, product/
      or sample dataset)                                   region/customer views)

     →  ANOMALY DETECTION  →  AI RECOMMENDATIONS  →  BUSINESS DECISIONS
        (z-score +            (rule-based advisor,       (executive PDF report,
         Isolation Forest)      grounded in metrics)       dashboard actions)
```

## Module map

| File | Responsibility | Inputs → Outputs |
|---|---|---|
| `app.py` | Streamlit UI, sidebar navigation, page rendering | session state → pages |
| `salespulse/data.py` | Load CSV/Excel, validate columns, profile uploads | file → DataFrame, profile dict |
| `salespulse/cleaning.py` | Duplicates, date parsing, numeric validation, category fixes, missing-value fills, quality score, derived Revenue/Profit | raw DataFrame → `{cleaned, log, before, after, rows_removed}` |
| `salespulse/analytics.py` | KPIs, filters, monthly trend, category/region/product aggregations, customer overview | cleaned DataFrame → KPI dicts / DataFrames |
| `salespulse/customers.py` | RFM scoring (quintiles 1–5) + 5 segments + revenue contribution | cleaned DataFrame → `{rfm, segment_summary}` |
| `salespulse/forecasting.py` | Monthly aggregation, LinearRegression on trend + annual seasonality, 95% residual band, insufficient-data guardrail | cleaned DataFrame → `{insufficient, forecast, history, r2, residual_std}` |
| `salespulse/anomalies.py` | Rolling z-score + Isolation Forest on daily revenue/transactions, severity + explanations | cleaned DataFrame → `{daily, anomalies, count}` |
| `salespulse/advisor.py` | Rule-based recommendations: problem, supporting data, impact, action, potential outcome | cleaned DataFrame (+ RFM summary) → list of recommendation dicts |
| `salespulse/report.py` | ReportLab Platypus PDF: overview → cleaning → KPIs → products → regions → customers → forecast → anomalies → recommendations → executive summary | context dict → PDF file |
| `scripts/generate_sample_data.py` | Seeded generator for `data/sample_sales.csv` (1,500+ rows, intentional messiness) | — → CSV |

## Key design decisions

1. **Raw vs cleaned separation.** The uploaded DataFrame lives in
   `st.session_state["raw"]` and is never mutated. `clean_data` returns a
   new DataFrame plus a log; every downstream module reads the cleaned copy.
2. **One source of truth for numbers.** KPIs, charts, advisor rules and the
   PDF all call the same functions in `analytics.py` / `customers.py`, so the
   dashboard and the report can never disagree.
3. **Guardrails over fabrication.** Forecasting refuses short histories;
   the advisor only emits rules whose supporting numbers were computed;
   anomaly severity comes from measured deviation, not vibes.
4. **Beginner-friendly.** Each module is self-contained, commented and
   independently testable (`tests/test_core.py`).

## State flow in the app

```
first load ──► sample CSV ──► clean_data ──► session_state["clean_result"]
upload ──────► load_file ──► profile ──► [Use this dataset] ──► clean_data
pages ───────► read cleaned DataFrame + module functions ──► charts / tables
Reports ─────► build context dict ──► generate_pdf ──► download button
```

## Dependencies

Streamlit (UI) · Pandas/NumPy (data) · Plotly/Matplotlib (charts) ·
scikit-learn (forecasting regression, Isolation Forest) ·
ReportLab (PDF) · openpyxl (Excel upload) · pytest (tests).
