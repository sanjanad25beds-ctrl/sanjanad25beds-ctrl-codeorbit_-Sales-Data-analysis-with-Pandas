"""SalesPulse AI — Streamlit entry point.

Navigation (sidebar): Overview, Data Upload, Data Cleaning, Sales Analytics,
Product Intelligence, Customer Intelligence, Sales Forecasting,
Anomaly Detection, AI Business Advisor, Reports.

The app always shows the *sample dataset* on first load so it works
immediately; uploading a file replaces the active dataset. Raw uploads are
never mutated — cleaning produces a separate cleaned copy.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from salespulse import analytics as an
from salespulse import advisor as adv
from salespulse import anomalies as ano
from salespulse import cleaning as cl
from salespulse import customers as cu
from salespulse import data as dt
from salespulse import forecasting as fc
from salespulse import report as rep

st.set_page_config(page_title="SalesPulse AI", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

NAVY = "#16294A"
GOLD = "#D4A017"

# --- styling: dark-navy sidebar, clean white KPI cards --------------------
st.markdown(f"""
<style>
section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: white; }}
.kpi-card {{
    background: white; border: 1px solid #DCE3EC; border-radius: 14px;
    padding: 16px 18px; box-shadow: 0 2px 8px rgba(22,41,74,.08);
}}
.kpi-label {{ font-size: 13px; color: #5A6B84; margin-bottom: 4px; }}
.kpi-value {{ font-size: 26px; font-weight: 700; color: {NAVY}; }}
.kpi-sub {{ font-size: 13px; color: #8A99B0; }}
.hero {{ background: linear-gradient(90deg, {NAVY}, #24467A);
    border-radius: 16px; padding: 28px 32px; color: white; }}
.hero h1 {{ color: white; margin: 0; }}
.hero p {{ color: #C9D6EA; margin: 6px 0 0 0; }}
.rec-card {{ background: white; border-left: 5px solid {GOLD};
    border-radius: 10px; padding: 16px 18px; margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(22,41,74,.06); }}
</style>
""", unsafe_allow_html=True)


# --- session state ----------------------------------------------------------
def _init_state() -> None:
    if "raw" not in st.session_state:
        st.session_state["raw"] = dt.load_sample_data()
        st.session_state["source"] = "Sample dataset (data/sample_sales.csv)"
    if "clean_result" not in st.session_state:
        st.session_state["clean_result"] = cl.clean_data(st.session_state["raw"])


_init_state()
cleaned: pd.DataFrame = st.session_state["clean_result"]["cleaned"]


# --- helpers ------------------------------------------------------------------
def kpi_cards(kpis: dict) -> None:
    cols = st.columns(4)
    cards = [
        ("Total Revenue", f"{kpis['total_revenue']:,.0f}", "across all orders"),
        ("Total Profit", f"{kpis['total_profit']:,.0f}", f"margin {kpis['profit_margin']:.1f}%"),
        ("Total Orders", f"{kpis['total_orders']:,}", f"AOV {kpis['avg_order_value']:,.0f}"),
        ("Products Sold", f"{kpis['total_products_sold']:,}", f"best: {kpis['best_product']}"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )
    cols2 = st.columns(4)
    cards2 = [
        ("Average Order Value", f"{kpis['avg_order_value']:,.0f}", "revenue per order"),
        ("Profit Margin", f"{kpis['profit_margin']:.1f}%", "profit / revenue"),
        ("Best Product", kpis["best_product"], f"revenue {kpis['best_product_revenue']:,.0f}"),
        ("Best Region", kpis["best_region"], f"revenue {kpis['best_region_revenue']:,.0f}"),
    ]
    for col, (label, value, sub) in zip(cols2, cards2):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )


def money_axis(fig):
    fig.update_layout(template="plotly_white", height=380,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# --- sidebar ------------------------------------------------------------------
st.sidebar.markdown("## 📊 SalesPulse AI")
st.sidebar.caption("From Raw Sales Data to Smarter Business Decisions.")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Upload", "Data Cleaning", "Sales Analytics",
     "Product Intelligence", "Customer Intelligence", "Sales Forecasting",
     "Anomaly Detection", "AI Business Advisor", "Reports"],
)
st.sidebar.divider()
st.sidebar.caption(f"Active dataset: **{st.session_state['source']}**")
st.sidebar.caption(f"Rows (raw): {len(st.session_state['raw']):,} | "
                   f"Rows (cleaned): {len(cleaned):,}")
st.sidebar.caption("Pipeline: Raw Data → Cleaning → BI → ML → "
                   "Anomaly → Recommendations → Decisions")


# --- pages ---------------------------------------------------------------------
def page_overview():
    st.markdown(
        '<div class="hero"><h1>SalesPulse AI</h1>'
        "<p>AI-Powered Sales Intelligence and Business Decision Platform</p></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    kpi_cards(an.compute_kpis(cleaned))
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Revenue Trend")
        trend = an.monthly_trend(cleaned)
        fig = px.line(trend, x="Month", y="Revenue", markers=True)
        st.plotly_chart(money_axis(fig), use_container_width=True)
    with col2:
        st.subheader("Revenue by Region")
        reg = an.revenue_by_region(cleaned)
        fig = px.bar(reg, x="Region", y="Revenue", color="Region")
        st.plotly_chart(money_axis(fig), use_container_width=True)
    st.info(
        "How it works: upload (or use the sample) → the cleaning engine produces "
        "a cleaned copy with a full log → analytics, RFM, forecasting, anomaly "
        "detection and the rule-based AI Advisor all read from that cleaned data. "
        "Every number on screen comes from the active dataset."
    )


def page_upload():
    st.header("📤 Data Upload")
    st.write("Upload a CSV or Excel sales file. The raw file is never modified — "
             "cleaning always produces a separate cleaned copy.")
    uploaded = st.file_uploader("Choose a .csv or .xlsx file", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            df = dt.load_file(uploaded, uploaded.name)
        except ValueError as exc:
            st.error(str(exc))
            return
        prof = dt.profile(df)
        st.subheader("Preview — first 10 rows")
        st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{prof['rows']:,}")
        c2.metric("Columns", prof["columns"])
        c3.metric("Duplicate rows", prof["duplicate_rows"])
        st.subheader("Columns and data types")
        st.dataframe(pd.DataFrame(
            {"Column": prof["column_names"],
             "Data type": [prof["dtypes"][c] for c in prof["column_names"]]}),
            use_container_width=True)
        st.subheader("Missing values")
        if prof["missing_values"]:
            st.dataframe(pd.DataFrame(
                {"Column": list(prof["missing_values"]),
                 "Missing": list(prof["missing_values"].values())}),
                use_container_width=True)
        else:
            st.success("No missing values found.")
        for w in prof["warnings"]:
            st.warning(w)
        if st.button("✅ Use this dataset", type="primary"):
            st.session_state["raw"] = df
            st.session_state["source"] = f"Uploaded file: {uploaded.name}"
            st.session_state["clean_result"] = cl.clean_data(df)
            st.success("Dataset applied and cleaned. Open Data Cleaning to see the log.")
            st.rerun()
    if st.button("📦 Load sample dataset"):
        st.session_state["raw"] = dt.load_sample_data()
        st.session_state["source"] = "Sample dataset (data/sample_sales.csv)"
        st.session_state["clean_result"] = cl.clean_data(st.session_state["raw"])
        st.success("Sample dataset loaded.")
        st.rerun()


def page_cleaning():
    st.header("🧹 Data Cleaning")
    res = st.session_state["clean_result"]
    before, after = res["before"], res["after"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows before → after", f"{before['rows']} → {after['rows']}")
    c2.metric("Missing cells before → after", f"{before['missing_cells']} → {after['missing_cells']}")
    c3.metric("Quality score before", before["quality_score"])
    c4.metric("Quality score after", after["quality_score"])
    st.subheader("Before / after statistics")
    st.dataframe(pd.DataFrame(
        {"Metric": ["Rows", "Columns", "Missing cells", "Duplicate rows", "Quality score (0–100)"],
         "Before": [before["rows"], before["columns"], before["missing_cells"],
                    before["duplicate_rows"], before["quality_score"]],
         "After": [after["rows"], after["columns"], after["missing_cells"],
                   after["duplicate_rows"], after["quality_score"]]}),
        use_container_width=True)
    st.subheader("Transformation log")
    for line in res["log"]:
        st.write("• " + line)
    st.subheader("Cleaned data — first 10 rows")
    st.dataframe(cleaned.head(10), use_container_width=True)
    st.download_button("⬇ Download cleaned data (CSV)",
                       cleaned.to_csv(index=False).encode("utf-8"),
                       file_name="cleaned_sales.csv", mime="text/csv")


def _filter_widgets(key_prefix: str):
    st.sidebar.divider()
    st.sidebar.subheader("Filters")
    dates = cleaned["Order Date"]
    start = st.sidebar.date_input("Start date", dates.min().date(), key=key_prefix + "_s")
    end = st.sidebar.date_input("End date", dates.max().date(), key=key_prefix + "_e")
    cats = st.sidebar.multiselect("Category", sorted(cleaned["Category"].unique()),
                                  default=sorted(cleaned["Category"].unique()),
                                  key=key_prefix + "_c")
    regs = st.sidebar.multiselect("Region", sorted(cleaned["Region"].unique()),
                                  default=sorted(cleaned["Region"].unique()),
                                  key=key_prefix + "_r")
    return an.apply_filters(cleaned, pd.Timestamp(start), pd.Timestamp(end), cats, regs)


def page_analytics():
    st.header("📈 Sales Analytics")
    filt = _filter_widgets("an")
    if filt.empty:
        st.warning("No rows match the current filters.")
        return
    kpi_cards(an.compute_kpis(filt))
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Monthly Revenue Trend")
        fig = px.line(an.monthly_trend(filt), x="Month", y="Revenue", markers=True)
        st.plotly_chart(money_axis(fig), use_container_width=True)
    with col2:
        st.subheader("Revenue vs Profit (Monthly)")
        trend = an.monthly_trend(filt)
        fig = px.bar(trend, x="Month", y=["Revenue", "Profit"], barmode="group")
        st.plotly_chart(money_axis(fig), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Product Category")
        fig = px.pie(an.revenue_by_category(filt), names="Category", values="Revenue",
                     hole=0.35)
        st.plotly_chart(money_axis(fig), use_container_width=True)
    with col4:
        st.subheader("Regional Performance")
        fig = px.bar(an.revenue_by_region(filt), x="Region", y="Revenue", color="Region")
        st.plotly_chart(money_axis(fig), use_container_width=True)
    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Top 10 Products")
        fig = px.bar(an.top_products(filt, 10), x="Revenue", y="Product Name",
                     orientation="h", color="Revenue")
        st.plotly_chart(money_axis(fig), use_container_width=True)
    with col6:
        st.subheader("Customer Behaviour — Monthly Active Customers")
        fig = px.line(an.customer_overview(filt), x="Month", y="Active Customers",
                      markers=True)
        st.plotly_chart(money_axis(fig), use_container_width=True)


def page_products():
    st.header("🏆 Product Intelligence")
    top = an.top_products(cleaned, 10)
    st.subheader("Top 10 Revenue-Generating Products")
    st.dataframe(top, use_container_width=True)
    st.plotly_chart(money_axis(px.bar(top, x="Revenue", y="Product Name",
                                      orientation="h", color="Revenue")),
                    use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Frequently Purchased (by quantity)")
        st.dataframe(top.sort_values("Quantity", ascending=False).head(5),
                     use_container_width=True)
        st.subheader("Highest-Profit Products")
        st.dataframe(top.sort_values("Profit", ascending=False).head(5),
                     use_container_width=True)
    with col2:
        st.subheader("Lowest-Performing Products")
        st.dataframe(an.worst_products(cleaned, 5), use_container_width=True)
        st.subheader("Recommendations")
        st.write("• Promote the top-revenue product with an above-average margin.")
        st.write("• Review the lowest performers: check stock, price and demand "
                 "before discounting further.")
        st.write("• Bundle a top seller with a mid-tier product to lift both.")


def page_customers():
    st.header("👥 Customer Intelligence (RFM)")
    res = cu.rfm_analysis(cleaned)
    st.subheader("Segments and Revenue Contribution")
    st.dataframe(res["segment_summary"], use_container_width=True)
    fig = px.bar(res["segment_summary"], x="Segment", y="Revenue", color="Segment")
    st.plotly_chart(money_axis(fig), use_container_width=True)
    st.subheader("How segments are defined")
    st.write("**High-value** — recent, frequent, big spenders (R≥4, F≥4, M≥4). "
             "**Loyal** — buy often with solid spend. **Potential** — recent but "
             "infrequent. **At-risk** — bought often before, gone quiet. "
             "**Low-engagement** — everyone else.")
    st.subheader("Per-Customer RFM Table")
    st.dataframe(res["rfm"].sort_values("Monetary", ascending=False),
                 use_container_width=True)


def page_forecast():
    st.header("🔮 Sales Forecasting")
    st.write("Monthly revenue forecast: linear trend + annual seasonality, "
             "fitted on monthly totals (daily revenue is mostly noise).")
    horizon = st.radio("Forecast horizon", [1, 3, 6], index=1, horizontal=True,
                       format_func=lambda m: f"Next {m} month{'s' if m > 1 else ''}")
    result = fc.forecast_revenue(cleaned, horizon)
    if result["insufficient"]:
        st.warning(result["message"])
        st.write("Forecasting needs at least 60 days of history across 6 distinct "
                 "months. Upload a longer dataset or use the sample data.")
        return
    st.info(result["message"])
    hist, fore = result["history"], result["forecast"]
    st.subheader("Historical and Predicted Revenue (Monthly)")
    fig = px.line(hist, x="Month", y="Revenue", title="Monthly revenue")
    fig.add_scatter(x=fore["Month"], y=fore["Predicted Revenue"], mode="lines+markers",
                    name="Forecast", line=dict(color="orange", dash="dash"))
    fig.add_scatter(x=fore["Month"], y=fore["Upper"], mode="lines", name="Upper (95%)",
                    line=dict(color="grey", width=0.5))
    fig.add_scatter(x=fore["Month"], y=fore["Lower"], mode="lines", name="Lower (95%)",
                    line=dict(color="grey", width=0.5), fill="tonexty",
                    fillcolor="rgba(200,200,200,0.25)")
    st.plotly_chart(money_axis(fig), use_container_width=True)
    st.subheader("Forecasted Totals")
    st.metric("Predicted revenue (total)", f"{fore['Predicted Revenue'].sum():,.0f}")
    st.dataframe(fore, use_container_width=True)
    st.caption("Limitations: the model is a linear trend + annual seasonality. "
               "It cannot foresee promotions, stock-outs or market shocks — treat "
               "the band as a range, not a promise.")


def page_anomalies():
    st.header("🚨 Anomaly Detection")
    res = ano.detect_anomalies(cleaned)
    st.metric("Anomalies detected", res["count"])
    if res["count"]:
        st.subheader("Detected Anomalies")
        st.dataframe(res["anomalies"], use_container_width=True)
    else:
        st.success("No anomalies detected in the current data.")
    st.subheader("Daily Revenue with Anomalies")
    daily = res["daily"]
    fig = px.line(daily, x="Date", y="Revenue")
    flagged = daily[daily["Is Anomaly"]]
    if not flagged.empty:
        fig.add_scatter(x=flagged["Date"], y=flagged["Revenue"], mode="markers",
                        name="Anomaly", marker=dict(color="red", size=9, symbol="x"))
    st.plotly_chart(money_axis(fig), use_container_width=True)
    st.caption("Methods: rolling z-score (7-day window) + Isolation Forest on daily "
               "revenue and transaction count. Severity: High ≥50% deviation or |z|≥3; "
               "Medium ≥30% or |z|≥2; otherwise Low.")


def page_advisor():
    st.header("🤖 AI Business Advisor")
    st.write("Rule-based recommendations computed from the active dataset. "
             "Outcomes are potential, never guaranteed.")
    summary = cu.rfm_analysis(cleaned)["segment_summary"]
    recs = adv.generate_recommendations(cleaned, summary)
    questions = ["All recommendations"] + [r["question"] for r in recs]
    choice = st.selectbox("Ask the advisor", questions)
    shown = recs if choice == "All recommendations" else [r for r in recs if r["question"] == choice]
    for rec in shown:
        st.markdown(
            f'<div class="rec-card"><h4>{rec["question"]}</h4>'
            f"<p><b>Problem:</b> {rec['problem']}</p>"
            f"<p><b>Supporting data:</b> {rec['supporting_data']}</p>"
            f"<p><b>Business impact:</b> {rec['business_impact']}</p>"
            f"<p><b>Suggested action:</b> {rec['suggested_action']}</p>"
            f"<p><b>Potential outcome:</b> {rec['potential_outcome']}</p></div>",
            unsafe_allow_html=True,
        )


def page_reports():
    st.header("📄 Executive Reports")
    st.write("Generate a manager-ready PDF covering overview, cleaning, KPIs, "
             "products, regions, customers, forecast, anomalies and recommendations.")
    if st.button("Generate PDF report", type="primary"):
        kpis = an.compute_kpis(cleaned)
        rfm = cu.rfm_analysis(cleaned)
        fore = fc.forecast_revenue(cleaned, 3)
        anoms = ano.detect_anomalies(cleaned)
        recs = adv.generate_recommendations(cleaned, rfm["segment_summary"])
        res = st.session_state["clean_result"]
        if fore["insufficient"]:
            fore_text = fore["message"]
            fore_rows = []
        else:
            fore_text = fore["message"]
            fore_rows = fore["forecast"].copy()
            fore_rows["Month"] = fore_rows["Month"].dt.date.astype(str)
            fore_rows = fore_rows.to_dict("records")
        ctx = {
            "kpis": kpis,
            "rows": len(cleaned),
            "date_range": (
                f"{cleaned['Order Date'].min().date()} to {cleaned['Order Date'].max().date()}"
                if not cleaned.empty else "—"),
            "source": st.session_state["source"],
            "overview": (
                "SalesPulse AI converts raw sales data into business decisions. "
                "This report summarises the active dataset after automated cleaning, "
                "the key performance indicators, and data-grounded recommendations."),
            "cleaning_summary": (
                f"Cleaning removed {res['rows_removed']} row(s) "
                f"({res['before']['rows']} → {res['after']['rows']} rows); "
                f"quality score improved from {res['before']['quality_score']} "
                f"to {res['after']['quality_score']}."),
            "cleaning_log": res["log"],
            "top_products": an.top_products(cleaned, 10).to_dict("records"),
            "regions": an.revenue_by_region(cleaned).to_dict("records"),
            "segments": rfm["segment_summary"].to_dict("records"),
            "forecast_text": fore_text,
            "forecast_rows": fore_rows,
            "anomaly_text": f"{anoms['count']} anomalie(s) detected.",
            "anomaly_rows": anoms["anomalies"].to_dict("records") if anoms["count"] else [],
            "recommendations": recs,
            "executive_summary": (
                f"Total revenue {kpis['total_revenue']:,.0f} at a "
                f"{kpis['profit_margin']:.1f}% margin across {kpis['total_orders']} "
                f"orders. Best product: {kpis['best_product']}; best region: "
                f"{kpis['best_region']}. The recommendations in this report are "
                "derived from these numbers — follow the suggested actions to "
                "protect revenue and improve profitability."),
        }
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            rep.generate_pdf(tmp.name, ctx)
            pdf_bytes = Path(tmp.name).read_bytes()
        st.success("Report generated.")
        st.download_button("⬇ Download Executive Report (PDF)", pdf_bytes,
                           file_name="SalesPulse_AI_Executive_Report.pdf",
                           mime="application/pdf")


PAGES = {
    "Overview": page_overview,
    "Data Upload": page_upload,
    "Data Cleaning": page_cleaning,
    "Sales Analytics": page_analytics,
    "Product Intelligence": page_products,
    "Customer Intelligence": page_customers,
    "Sales Forecasting": page_forecast,
    "Anomaly Detection": page_anomalies,
    "AI Business Advisor": page_advisor,
    "Reports": page_reports,
}
PAGES[page]()
