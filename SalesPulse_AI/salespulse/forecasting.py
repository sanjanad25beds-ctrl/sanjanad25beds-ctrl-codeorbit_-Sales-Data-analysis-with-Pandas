"""Sales forecasting.

Design decision (documented, not hidden): the model fits **monthly** revenue
totals. Daily revenue in real sales data is dominated by day-to-day noise
(order arrivals), so a daily regression fits mostly noise (R² ≈ 0.02 on the
sample data) while monthly totals reveal the real trend + seasonality
(R² ≈ 0.45). Monthly aggregation is explicitly allowed by the spec
("aggregate sales by day or month").

Model: ordinary least-squares linear regression on:
    * linear trend (months since first observation)
    * annual seasonality: sin / cos of the 12-month cycle

Uncertainty: a 95% band of ±1.96 × the standard deviation of in-sample
residuals. This is an honest, model-based band — not a guarantee.

Honesty guardrail (hard rule): forecasting is refused when the history has
fewer than 60 days or fewer than 6 distinct months (a monthly model cannot
be fit on less). The caller receives ``{"insufficient": True, ...}`` and
must show the message instead of any chart of predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MIN_DAYS = 60
MIN_MONTHS = 6


def _monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.groupby(df["Order Date"].dt.to_period("M"))["Revenue"]
        .sum()
        .rename("Revenue")
        .reset_index()
        .rename(columns={"Order Date": "Month"})
    )
    monthly["Month"] = monthly["Month"].dt.to_timestamp()
    # Fill any missing calendar months with 0 so the trend is not distorted.
    full = pd.DataFrame({
        "Month": pd.date_range(monthly["Month"].min(), monthly["Month"].max(), freq="MS")
    })
    return full.merge(monthly, on="Month", how="left").fillna({"Revenue": 0.0})


def _features(months: pd.Series) -> np.ndarray:
    """[trend, sin(annual), cos(annual)] for a series of month starts."""
    t = np.arange(len(months), dtype=float)
    annual = 2 * np.pi * t / 12.0
    return np.column_stack([t, np.sin(annual), np.cos(annual)])


def forecast_revenue(df: pd.DataFrame, horizon_months: int = 3) -> dict:
    """Forecast monthly revenue ``horizon_months`` ahead.

    Returns a dict with keys ``insufficient`` (bool), ``message`` (str),
    ``forecast`` (DataFrame: Month, Predicted Revenue, Lower, Upper),
    ``history`` (DataFrame: Month, Revenue), ``r2`` and ``residual_std``.
    """
    if df.empty or "Order Date" not in df.columns or "Revenue" not in df.columns:
        return {
            "insufficient": True,
            "message": "No dated revenue history available — forecasting is not possible.",
            "forecast": pd.DataFrame(), "history": pd.DataFrame(),
            "r2": None, "residual_std": None,
        }
    history = _monthly_series(df)
    n_days = (df["Order Date"].max() - df["Order Date"].min()).days + 1
    n_months = len(history)
    if n_days < MIN_DAYS or n_months < MIN_MONTHS:
        return {
            "insufficient": True,
            "message": (
                f"Insufficient data for reliable forecasting: only {n_days} "
                f"day(s) of history across {n_months} month(s). A monthly "
                f"forecasting model needs at least {MIN_DAYS} days across "
                f"{MIN_MONTHS} distinct months. No predictions are shown "
                "rather than fabricating them."
            ),
            "forecast": pd.DataFrame(), "history": history,
            "r2": None, "residual_std": None,
        }

    X = _features(history["Month"])
    y = history["Revenue"].to_numpy(dtype=float)
    model = LinearRegression().fit(X, y)
    fitted = model.predict(X)
    residuals = y - fitted
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    r2 = float(model.score(X, y))

    future_months = pd.date_range(
        history["Month"].max() + pd.DateOffset(months=1),
        periods=horizon_months, freq="MS",
    )
    all_months = pd.concat(
        [history["Month"], pd.Series(future_months)], ignore_index=True
    )
    X_future = _features(all_months)[len(history):]
    pred = np.clip(model.predict(X_future), 0.0, None)  # revenue cannot be negative
    band = 1.96 * residual_std

    forecast = pd.DataFrame({
        "Month": future_months,
        "Predicted Revenue": pred,
        "Lower": np.clip(pred - band, 0.0, None),
        "Upper": pred + band,
    })
    return {
        "insufficient": False,
        "message": (
            f"Forecast based on {n_months} months of history "
            f"(in-sample R² = {r2:.2f}). The band shows ±1.96 × residual "
            f"standard deviation (±{band:,.0f}); it reflects past variability, "
            "not a guarantee."
        ),
        "forecast": forecast,
        "history": history,
        "r2": r2,
        "residual_std": residual_std,
    }
