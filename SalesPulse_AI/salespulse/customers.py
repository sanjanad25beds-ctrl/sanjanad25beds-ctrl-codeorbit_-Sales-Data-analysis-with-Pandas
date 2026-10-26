"""Customer intelligence: RFM analysis and segmentation.

RFM (Recency, Frequency, Monetary) is computed per customer against the
latest order date in the dataset:
    Recency  = days since the customer's most recent order
    Frequency = number of orders
    Monetary  = total revenue from the customer

Each dimension is scored 1–5 by quintile (5 = best). Segments:

    High-value     R>=4 and F>=4 and M>=4   — recent, frequent, big spenders
    Loyal          F>=4 and M>=3             — buy often, solid spend
    Potential      R>=4 and F<=2             — recent but infrequent
    At-risk        R<=2 and F>=3             — used to buy often, gone quiet
    Low-engagement everything else

Quintiles use rank-based binning so tied values cannot crash the scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _score(series: pd.Series, ascending: bool) -> pd.Series:
    """Quintile score 1–5. ``ascending``: True if larger values are better."""
    ranks = series.rank(method="first")
    try:
        bins = pd.qcut(ranks, 5, labels=False, duplicates="drop")
    except ValueError:  # fewer than 5 distinct ranks
        bins = pd.cut(ranks, 5, labels=False, include_lowest=True)
    bins = bins.fillna(0).astype(int)  # 0..4
    if ascending:
        return bins + 1
    return 5 - bins


def _segment(row: pd.Series) -> str:
    r, f, m = row["R Score"], row["F Score"], row["M Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "High-value"
    if f >= 4 and m >= 3:
        return "Loyal"
    if r >= 4 and f <= 2:
        return "Potential"
    if r <= 2 and f >= 3:
        return "At-risk"
    return "Low-engagement"


def rfm_analysis(df: pd.DataFrame) -> dict:
    """Return per-customer RFM table and a segment summary.

    Keys: ``rfm`` (DataFrame, one row per customer),
    ``segment_summary`` (DataFrame: customers, revenue, share per segment).
    """
    empty = {
        "rfm": pd.DataFrame(columns=[
            "Customer ID", "Customer Name", "Recency", "Frequency",
            "Monetary", "R Score", "F Score", "M Score", "RFM Score", "Segment"]),
        "segment_summary": pd.DataFrame(
            columns=["Segment", "Customers", "Revenue", "Revenue Share %"]),
    }
    if df.empty or "Customer ID" not in df.columns or "Order Date" not in df.columns:
        return empty

    reference = df["Order Date"].max() + pd.Timedelta(days=1)
    grouped = df.groupby(["Customer ID", "Customer Name"]).agg(
        Recency=("Order Date", lambda s: (reference - s.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Revenue", "sum"),
    ).reset_index()

    grouped["R Score"] = _score(grouped["Recency"], ascending=False)
    grouped["F Score"] = _score(grouped["Frequency"], ascending=True)
    grouped["M Score"] = _score(grouped["Monetary"], ascending=True)
    grouped["RFM Score"] = (
        grouped["R Score"] + grouped["F Score"] + grouped["M Score"]
    )
    grouped["Segment"] = grouped.apply(_segment, axis=1)

    total = float(grouped["Monetary"].sum())
    summary = (
        grouped.groupby("Segment")
        .agg(Customers=("Customer ID", "nunique"), Revenue=("Monetary", "sum"))
        .reset_index()
    )
    summary["Revenue Share %"] = (
        summary["Revenue"] / total * 100.0 if total else 0.0
    ).round(1)
    order = ["High-value", "Loyal", "Potential", "At-risk", "Low-engagement"]
    summary["Segment"] = pd.Categorical(summary["Segment"], order, ordered=True)
    summary = summary.sort_values("Segment").reset_index(drop=True)
    return {"rfm": grouped, "segment_summary": summary}
