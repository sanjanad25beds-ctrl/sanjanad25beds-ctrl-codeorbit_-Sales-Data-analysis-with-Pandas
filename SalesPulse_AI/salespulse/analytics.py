"""Analytics: KPIs, trends and product / region aggregations.

Every function derives its numbers from the DataFrame it is given — nothing
is hardcoded. All functions tolerate an empty DataFrame and return zeros /
empty frames instead of raising.
"""

from __future__ import annotations

import pandas as pd


def _empty_kpis() -> dict:
    return {
        "total_revenue": 0.0,
        "total_profit": 0.0,
        "total_orders": 0,
        "avg_order_value": 0.0,
        "total_products_sold": 0,
        "profit_margin": 0.0,
        "best_product": "—",
        "best_product_revenue": 0.0,
        "best_region": "—",
        "best_region_revenue": 0.0,
    }


def compute_kpis(df: pd.DataFrame) -> dict:
    """Core KPIs from a *cleaned* DataFrame."""
    if df.empty or "Revenue" not in df.columns:
        return _empty_kpis()
    revenue = float(df["Revenue"].sum())
    profit = float(df["Profit"].sum()) if "Profit" in df.columns else 0.0
    orders = int(df["Order ID"].nunique()) if "Order ID" in df.columns else len(df)
    qty = float(df["Quantity"].sum()) if "Quantity" in df.columns else 0.0

    best_product, best_product_rev = "—", 0.0
    if "Product Name" in df.columns:
        by_product = df.groupby("Product Name")["Revenue"].sum()
        if not by_product.empty:
            best_product = str(by_product.idxmax())
            best_product_rev = float(by_product.max())

    best_region, best_region_rev = "—", 0.0
    if "Region" in df.columns:
        by_region = df.groupby("Region")["Revenue"].sum()
        if not by_region.empty:
            best_region = str(by_region.idxmax())
            best_region_rev = float(by_region.max())

    return {
        "total_revenue": revenue,
        "total_profit": profit,
        "total_orders": orders,
        "avg_order_value": revenue / orders if orders else 0.0,
        "total_products_sold": int(qty),
        "profit_margin": (profit / revenue * 100.0) if revenue else 0.0,
        "best_product": best_product,
        "best_product_revenue": best_product_rev,
        "best_region": best_region,
        "best_region_revenue": best_region_rev,
    }


def apply_filters(
    df: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    categories: list[str] | None = None,
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """Return ``df`` restricted by date range, categories and regions."""
    out = df
    if "Order Date" in out.columns:
        if start is not None:
            out = out[out["Order Date"] >= start]
        if end is not None:
            out = out[out["Order Date"] <= end]
    if categories and "Category" in out.columns:
        out = out[out["Category"].isin(categories)]
    if regions and "Region" in out.columns:
        out = out[out["Region"].isin(regions)]
    return out


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly Revenue / Profit / Orders, sorted by month."""
    if df.empty or "Order Date" not in df.columns:
        return pd.DataFrame(columns=["Month", "Revenue", "Profit", "Orders"])
    work = df.copy()
    work["Month"] = work["Order Date"].dt.to_period("M").dt.to_timestamp()
    agg = (
        work.groupby("Month")
        .agg(Revenue=("Revenue", "sum"),
             Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .reset_index()
        .sort_values("Month")
    )
    return agg


def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue / Profit / Orders per category, sorted by revenue desc."""
    if df.empty or "Category" not in df.columns:
        return pd.DataFrame(columns=["Category", "Revenue", "Profit", "Orders"])
    return (
        df.groupby("Category")
        .agg(Revenue=("Revenue", "sum"),
             Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )


def revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue / Profit / Orders per region, sorted by revenue desc."""
    if df.empty or "Region" not in df.columns:
        return pd.DataFrame(columns=["Region", "Revenue", "Profit", "Orders"])
    return (
        df.groupby("Region")
        .agg(Revenue=("Revenue", "sum"),
             Profit=("Profit", "sum"),
             Orders=("Order ID", "nunique"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top ``n`` products by revenue, with quantity, profit and margin."""
    if df.empty or "Product Name" not in df.columns:
        return pd.DataFrame(
            columns=["Product Name", "Revenue", "Quantity", "Profit", "Profit Margin %"]
        )
    agg = (
        df.groupby("Product Name")
        .agg(Revenue=("Revenue", "sum"),
             Quantity=("Quantity", "sum"),
             Profit=("Profit", "sum"))
        .reset_index()
    )
    agg["Profit Margin %"] = (agg["Profit"] / agg["Revenue"] * 100).fillna(0.0)
    return agg.sort_values("Revenue", ascending=False).head(n)


def worst_products(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Lowest ``n`` products by revenue (candidates for review)."""
    top = top_products(df, n=len(df["Product Name"].unique()) if not df.empty else 0)
    return top.sort_values("Revenue").head(n) if not top.empty else top


def customer_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly active customers and average orders per customer."""
    if df.empty or "Order Date" not in df.columns:
        return pd.DataFrame(columns=["Month", "Active Customers", "Orders"])
    work = df.copy()
    work["Month"] = work["Order Date"].dt.to_period("M").dt.to_timestamp()
    return (
        work.groupby("Month")
        .agg(**{"Active Customers": ("Customer ID", "nunique"),
                "Orders": ("Order ID", "nunique")})
        .reset_index()
        .sort_values("Month")
    )
