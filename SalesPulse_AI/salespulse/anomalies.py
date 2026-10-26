"""Anomaly detection on daily revenue.

Two complementary methods:
    * Rolling z-score: each day's revenue vs the centred 7-day rolling
      median / standard deviation — catches sudden spikes and drops.
    * Isolation Forest on [daily revenue, daily transaction count]
      (standardised) — catches multivariate outliers the z-score may miss.

A day is flagged when either method fires. Severity comes from the
deviation from the rolling median:

    |deviation| >= 50% or |z| >= 3  -> High
    |deviation| >= 30% or |z| >= 2  -> Medium
    otherwise                          -> Low

Every anomaly carries a plain-English explanation grounded in the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def _severity(deviation_pct: float, z: float) -> str:
    if abs(deviation_pct) >= 50 or abs(z) >= 3:
        return "High"
    if abs(deviation_pct) >= 30 or abs(z) >= 2:
        return "Medium"
    return "Low"


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Return daily series and a table of flagged anomalies.

    Keys: ``daily`` (DataFrame with Revenue, Transactions, rolling stats,
    z-score, isolation flag), ``anomalies`` (flagged rows with severity and
    explanation), ``count`` (int).
    """
    empty_daily = pd.DataFrame(columns=[
        "Date", "Revenue", "Transactions", "Rolling Median", "Z Score",
        "Isolation Anomaly", "Is Anomaly"])
    if df.empty or "Order Date" not in df.columns:
        return {"daily": empty_daily, "anomalies": pd.DataFrame(), "count": 0}

    daily = (
        df.groupby(df["Order Date"].dt.normalize())
        .agg(Revenue=("Revenue", "sum"), Transactions=("Order ID", "nunique"))
        .reset_index()
        .rename(columns={"Order Date": "Date"})
        .sort_values("Date")
    )
    full = pd.DataFrame({"Date": pd.date_range(daily["Date"].min(), daily["Date"].max())})
    daily = full.merge(daily, on="Date", how="left").fillna({"Revenue": 0.0, "Transactions": 0})

    # Rolling statistics (centred so a spike does not hide inside its own window).
    roll = daily["Revenue"].rolling(window=7, center=True, min_periods=3)
    daily["Rolling Median"] = roll.median()
    roll_std = roll.std().replace(0, np.nan)
    daily["Z Score"] = (daily["Revenue"] - daily["Rolling Median"]) / roll_std
    daily["Z Score"] = daily["Z Score"].fillna(0.0)

    # Isolation Forest on [revenue, transactions].
    X = StandardScaler().fit_transform(daily[["Revenue", "Transactions"]].to_numpy())
    if len(daily) >= 20:
        iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
        daily["Isolation Anomaly"] = iso.predict(X) == -1
    else:  # too few points for a forest: rely on the z-score alone
        daily["Isolation Anomaly"] = False

    daily["Is Anomaly"] = (daily["Z Score"].abs() >= 2) | daily["Isolation Anomaly"]
    flagged = daily[daily["Is Anomaly"]].copy()
    rows = []
    for _, r in flagged.iterrows():
        median = r["Rolling Median"] if pd.notna(r["Rolling Median"]) else 0.0
        deviation = ((r["Revenue"] - median) / median * 100.0) if median else 0.0
        direction = "spike" if r["Revenue"] >= median else "drop"
        rows.append({
            "Date": r["Date"].date(),
            "Revenue": float(r["Revenue"]),
            "Expected (7-day median)": float(median),
            "Deviation %": round(float(deviation), 1),
            "Z Score": round(float(r["Z Score"]), 2),
            "Severity": _severity(deviation, r["Z Score"]),
            "Explanation": (
                f"Revenue {direction} of {abs(deviation):.0f}% vs the 7-day "
                f"median ({median:,.0f}); {int(r['Transactions'])} transactions "
                f"that day."
            ),
        })
    anomalies = pd.DataFrame(rows)
    if not anomalies.empty:
        order = {"High": 0, "Medium": 1, "Low": 2}
        anomalies["_ord"] = anomalies["Severity"].map(order)
        anomalies = anomalies.sort_values(["_ord", "Date"]).drop(columns="_ord")
    return {"daily": daily, "anomalies": anomalies, "count": int(len(anomalies))}
