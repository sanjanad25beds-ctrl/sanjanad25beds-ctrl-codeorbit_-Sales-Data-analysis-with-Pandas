"""Executive PDF report generation (ReportLab).

``generate_pdf`` builds a manager-ready report from a context dictionary of
already-computed results: KPIs, cleaning summary, product / region tables,
customer segments, forecast summary, anomalies and recommendations. It
never computes business metrics itself — the numbers come from the same
modules that power the dashboard, so the PDF always agrees with the app.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#16294A")
GOLD = colors.HexColor("#D4A017")
LIGHT = colors.HexColor("#EEF2F7")
WHITE = colors.white


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title2", parent=base["Title"], textColor=NAVY,
                                fontSize=24, leading=28, spaceAfter=4),
        "h1": ParagraphStyle("H1x", parent=base["Heading1"], textColor=NAVY,
                             fontSize=16, leading=20, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("H2x", parent=base["Heading2"], textColor=NAVY,
                             fontSize=12, leading=15, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("BodyX", parent=base["BodyText"], fontSize=10,
                               leading=14, spaceAfter=4),
        "small": ParagraphStyle("SmallX", parent=base["BodyText"], fontSize=9,
                                leading=12, textColor=colors.HexColor("#555555")),
        "cell": ParagraphStyle("Cell", parent=base["BodyText"], fontSize=9,
                               leading=11),
        "cell_b": ParagraphStyle("CellB", parent=base["BodyText"], fontSize=9,
                                 leading=11, fontName="Helvetica-Bold"),
    }


def _table(headers: list[str], rows: list[list], s) -> Table:
    data = [[Paragraph(f"<b>{h}</b>", s["cell_b"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), s["cell"]) for c in row])
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9C4D4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _money(x: float) -> str:
    return f"{x:,.2f}"


def generate_pdf(path: str, ctx: dict) -> str:
    """Write the executive report to ``path``; return ``path``."""
    s = _styles()
    story = []
    kpis = ctx["kpis"]
    story.append(Paragraph("SalesPulse AI — Executive Report", s["title"]))
    story.append(Paragraph(
        "AI-Powered Sales Intelligence and Business Decision Platform",
        s["small"]))
    story.append(Spacer(1, 6))

    # 1. Overview ------------------------------------------------------------
    story.append(Paragraph("1. Project and Dataset Overview", s["h1"]))
    story.append(Paragraph(ctx.get("overview", ""), s["body"]))
    story.append(Paragraph(
        f"Rows analysed: {ctx.get('rows', '—')} &nbsp;|&nbsp; "
        f"Date range: {ctx.get('date_range', '—')} &nbsp;|&nbsp; "
        f"Source: {ctx.get('source', '—')}", s["small"]))

    # 2. Cleaning --------------------------------------------------------------
    story.append(Paragraph("2. Data Cleaning Summary", s["h1"]))
    story.append(Paragraph(ctx.get("cleaning_summary", ""), s["body"]))
    if ctx.get("cleaning_log"):
        for line in ctx["cleaning_log"]:
            story.append(Paragraph(f"• {line}", s["body"]))

    # 3. KPIs ------------------------------------------------------------------
    story.append(Paragraph("3. Sales KPI Summary", s["h1"]))
    story.append(_table(
        ["KPI", "Value"],
        [
            ["Total Revenue", _money(kpis["total_revenue"])],
            ["Total Profit", _money(kpis["total_profit"])],
            ["Total Orders", str(kpis["total_orders"])],
            ["Average Order Value", _money(kpis["avg_order_value"])],
            ["Total Products Sold", str(kpis["total_products_sold"])],
            ["Profit Margin", f"{kpis['profit_margin']:.1f}%"],
            ["Best Product", f"{kpis['best_product']} ({_money(kpis['best_product_revenue'])})"],
            ["Best Region", f"{kpis['best_region']} ({_money(kpis['best_region_revenue'])})"],
        ], s))

    # 4. Product performance -----------------------------------------------------
    story.append(Paragraph("4. Product Performance (Top 10 by Revenue)", s["h1"]))
    story.append(_table(
        ["Product", "Revenue", "Quantity", "Profit"],
        [[r["Product Name"], _money(r["Revenue"]), f"{r['Quantity']:,.0f}", _money(r["Profit"])]
         for r in ctx.get("top_products", [])], s))

    # 5. Regional analysis ---------------------------------------------------------
    story.append(Paragraph("5. Regional Analysis", s["h1"]))
    story.append(_table(
        ["Region", "Revenue", "Profit", "Orders"],
        [[r["Region"], _money(r["Revenue"]), _money(r["Profit"]), str(int(r["Orders"]))]
         for r in ctx.get("regions", [])], s))

    # 6. Customer insights -----------------------------------------------------------
    story.append(Paragraph("6. Customer Insights (RFM Segments)", s["h1"]))
    story.append(_table(
        ["Segment", "Customers", "Revenue", "Revenue Share"],
        [[r["Segment"], str(int(r["Customers"])), _money(r["Revenue"]),
          f"{r['Revenue Share %']:.1f}%"]
         for r in ctx.get("segments", [])], s))

    # 7. Forecasting --------------------------------------------------------------------
    story.append(Paragraph("7. Forecasting Results", s["h1"]))
    story.append(Paragraph(ctx.get("forecast_text", ""), s["body"]))
    if ctx.get("forecast_rows"):
        story.append(_table(
            ["Month", "Predicted Revenue", "Lower (95%)", "Upper (95%)"],
            [[r["Month"], _money(r["Predicted Revenue"]), _money(r["Lower"]), _money(r["Upper"])]
             for r in ctx["forecast_rows"][:14]], s))

    # 8. Anomalies ---------------------------------------------------------------------------
    story.append(Paragraph("8. Anomaly Detection Results", s["h1"]))
    story.append(Paragraph(ctx.get("anomaly_text", ""), s["body"]))
    if ctx.get("anomaly_rows"):
        story.append(_table(
            ["Date", "Revenue", "Deviation", "Severity"],
            [[str(r["Date"]), _money(r["Revenue"]), f"{r['Deviation %']}%", r["Severity"]]
             for r in ctx["anomaly_rows"][:15]], s))

    # 9. Recommendations ------------------------------------------------------------------------
    story.append(Paragraph("9. AI Business Recommendations", s["h1"]))
    for rec in ctx.get("recommendations", []):
        story.append(Paragraph(f"<b>{rec['question']}</b>", s["h2"]))
        story.append(Paragraph(f"<b>Problem:</b> {rec['problem']}", s["body"]))
        story.append(Paragraph(f"<b>Supporting data:</b> {rec['supporting_data']}", s["body"]))
        story.append(Paragraph(f"<b>Business impact:</b> {rec['business_impact']}", s["body"]))
        story.append(Paragraph(f"<b>Suggested action:</b> {rec['suggested_action']}", s["body"]))
        story.append(Paragraph(f"<b>Potential outcome:</b> {rec['potential_outcome']}", s["body"]))

    # 10. Executive summary -----------------------------------------------------------------------
    story.append(Paragraph("10. Executive Summary", s["h1"]))
    story.append(Paragraph(ctx.get("executive_summary", ""), s["body"]))

    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="SalesPulse AI — Executive Report")
    doc.build(story)
    return path
