"""Core tests for SalesPulse AI.

Covers: revenue/profit/AOV correctness, the cleaning engine (including
that the raw input is never mutated), RFM segmentation, the forecasting
insufficient-data guardrail, advisor grounding (no "guaranteed" claims),
and quality-score bounds.
"""

import pandas as pd
import pytest

from salespulse import advisor as adv
from salespulse import analytics as an
from salespulse import cleaning as cl
from salespulse import customers as cu
from salespulse import forecasting as fc


def _toy_df() -> pd.DataFrame:
    """Small deterministic dataset: 10 orders, 2 customers, 2 products."""
    rows = []
    for i in range(10):
        rows.append({
            "Order ID": f"ORD-{i}",
            "Order Date": f"2025-01-{i + 1:02d}",
            "Customer ID": "C-1" if i < 7 else "C-2",
            "Customer Name": "Alice" if i < 7 else "Bob",
            "Product ID": "P-1" if i % 2 == 0 else "P-2",
            "Product Name": "Widget" if i % 2 == 0 else "Gadget",
            "Category": "Electronics",
            "Quantity": 2,
            "Unit Price": 100.0,
            "Unit Cost": 60.0,
            "Discount": 0.10,
            "Region": "North",
            "Sales Representative": "Rep 1",
        })
    return pd.DataFrame(rows)


def test_revenue_profit_aov():
    cleaned = cl.clean_data(_toy_df())["cleaned"]
    kpis = an.compute_kpis(cleaned)
    # Per order: revenue 2*100*0.9 = 180, profit 180 - 2*60 = 60.
    assert kpis["total_revenue"] == pytest.approx(1800.0)
    assert kpis["total_profit"] == pytest.approx(600.0)
    assert kpis["total_orders"] == 10
    assert kpis["avg_order_value"] == pytest.approx(180.0)
    assert kpis["total_products_sold"] == 20
    assert kpis["profit_margin"] == pytest.approx(600 / 1800 * 100)


def test_cleaning_removes_duplicates_and_invalid_and_never_mutates():
    df = _toy_df()
    df = pd.concat([df, df.iloc[[0]], df.iloc[[1]]], ignore_index=True)  # 2 dupes
    df.loc[len(df)] = df.iloc[0].copy()
    df.loc[len(df) - 1, "Quantity"] = -5  # invalid row
    df.loc[3, "Category"] = None          # missing category -> "Unknown"
    df.loc[4, "Region"] = "EAST"           # casing fix
    original = df.copy()
    result = cl.clean_data(df)
    pd.testing.assert_frame_equal(df, original)  # raw input untouched
    cleaned = result["cleaned"]
    assert len(cleaned) == 10
    assert cleaned["Category"].isna().sum() == 0
    assert "Unknown" in set(cleaned["Category"])
    assert "East" in set(cleaned["Region"])
    assert result["rows_removed"] == 3  # 2 duplicates + 1 invalid
    assert result["after"]["quality_score"] == 100.0


def test_cleaning_handles_empty_and_missing_columns():
    result = cl.clean_data(pd.DataFrame(columns=["Order ID"]))
    assert result["cleaned"].empty
    assert result["after"]["quality_score"] == 0.0


def test_quality_score_bounds_and_improvement():
    messy = _toy_df()
    messy.loc[0, "Quantity"] = None
    messy = pd.concat([messy, messy.iloc[[0]]], ignore_index=True)
    assert 0.0 <= cl.quality_score(messy) <= 100.0
    assert cl.quality_score(messy) < cl.quality_score(_toy_df())


def test_rfm_segments_and_revenue_share():
    cleaned = cl.clean_data(_toy_df())["cleaned"]
    res = cu.rfm_analysis(cleaned)
    rfm = res["rfm"]
    assert set(rfm["Segment"]).issubset(
        {"High-value", "Loyal", "Potential", "At-risk", "Low-engagement"})
    summary = res["segment_summary"]
    assert summary["Revenue"].sum() == pytest.approx(cleaned["Revenue"].sum())
    assert summary["Revenue Share %"].sum() == pytest.approx(100.0, abs=0.2)


def test_forecast_guardrail_on_short_history():
    cleaned = cl.clean_data(_toy_df())["cleaned"]  # only 10 days of history
    result = fc.forecast_revenue(cleaned, 3)
    assert result["insufficient"] is True
    assert "Insufficient data" in result["message"]
    assert result["forecast"].empty


def test_forecast_runs_on_sample_data():
    from salespulse.data import load_sample_data
    cleaned = cl.clean_data(load_sample_data())["cleaned"]
    result = fc.forecast_revenue(cleaned, 3)
    assert result["insufficient"] is False
    assert len(result["forecast"]) == 3
    assert (result["forecast"]["Predicted Revenue"] >= 0).all()
    assert (result["forecast"]["Lower"] <= result["forecast"]["Predicted Revenue"]).all()
    assert (result["forecast"]["Predicted Revenue"] <= result["forecast"]["Upper"]).all()
    assert result["r2"] is not None and result["r2"] > 0.1  # monthly signal exists


def test_advisor_grounded_no_guarantees():
    from salespulse.data import load_sample_data
    cleaned = cl.clean_data(load_sample_data())["cleaned"]
    rfm = cu.rfm_analysis(cleaned)["segment_summary"]
    recs = adv.generate_recommendations(cleaned, rfm)
    assert recs, "advisor should produce at least one recommendation"
    for rec in recs:
        for key in ("question", "problem", "supporting_data", "business_impact",
                    "suggested_action", "potential_outcome"):
            assert rec[key], f"recommendation missing {key}"
        joined = " ".join(str(v) for v in rec.values()).lower()
        assert "guarantee" not in joined
        assert "guaranteed" not in joined


def test_kpis_empty_frame():
    assert an.compute_kpis(pd.DataFrame())["total_revenue"] == 0.0
