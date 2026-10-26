"""AI Business Advisor — rule-based, data-grounded recommendations.

There is no external LLM call. Every recommendation is produced by a rule
that reads metrics computed from the active dataset, so each claim can be
traced back to a number. Outcomes are always framed as *potential*, never
guaranteed.

Each recommendation is a dict with:
    question, problem, supporting_data, business_impact,
    suggested_action, potential_outcome
"""

from __future__ import annotations

import pandas as pd

from .analytics import compute_kpis, monthly_trend, revenue_by_region, top_products


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def generate_recommendations(
    df: pd.DataFrame,
    rfm_summary: pd.DataFrame | None = None,
    forecast: dict | None = None,
) -> list[dict]:
    """Generate recommendations from the active (cleaned) dataset."""
    recs: list[dict] = []
    if df.empty or "Revenue" not in df.columns:
        return recs

    kpis = compute_kpis(df)
    monthly = monthly_trend(df)
    regions = revenue_by_region(df)
    products = top_products(df, n=len(df["Product Name"].unique()) if "Product Name" in df else 0)

    # 1. Revenue movement: last month vs the month before ------------------
    if len(monthly) >= 2:
        last, prev = monthly.iloc[-1], monthly.iloc[-2]
        change = (last["Revenue"] - prev["Revenue"]) / prev["Revenue"] * 100 if prev["Revenue"] else 0.0
        if change < -5:
            recs.append({
                "question": "Why did revenue decrease this month?",
                "problem": (
                    f"Revenue fell {abs(change):.1f}% month over month, from "
                    f"{_fmt(prev['Revenue'])} to {_fmt(last['Revenue'])}."
                ),
                "supporting_data": (
                    f"Last month: revenue {_fmt(last['Revenue'])}, "
                    f"profit {_fmt(last['Profit'])}, orders {int(last['Orders'])}. "
                    f"Previous month: revenue {_fmt(prev['Revenue'])}, "
                    f"profit {_fmt(prev['Profit'])}, orders {int(prev['Orders'])}."
                ),
                "business_impact": (
                    "A sustained drop at this rate reduces quarterly revenue "
                    "and signals weakening demand or lost customers."
                ),
                "suggested_action": (
                    "Compare the two months by product and region to isolate "
                    "the drop, then re-engage customers who stopped ordering."
                ),
                "potential_outcome": (
                    "If the cause is a few lapsed customers or one weak product, "
                    "targeted follow-up could recover part of the lost revenue."
                ),
            })
        elif change > 5:
            recs.append({
                "question": "Why did revenue decrease this month?",
                "problem": "Revenue did not decrease — it grew.",
                "supporting_data": (
                    f"Revenue rose {change:.1f}% month over month, from "
                    f"{_fmt(prev['Revenue'])} to {_fmt(last['Revenue'])}."
                ),
                "business_impact": (
                    "Growth is healthy, but it is worth checking whether it "
                    "came from one product or region (concentration risk)."
                ),
                "suggested_action": (
                    "Confirm which products and regions drove the increase and "
                    "protect their stock and promotion."
                ),
                "potential_outcome": (
                    "Understanding the driver helps repeat the growth next month."
                ),
            })

    # 2. Product to promote -------------------------------------------------
    if not products.empty:
        top = products.iloc[0]
        avg_margin = float(products["Profit Margin %"].mean())
        recs.append({
            "question": "Which product should the company promote?",
            "problem": "Promotion budget should follow proven revenue and margin.",
            "supporting_data": (
                f"{top['Product Name']} is the top product by revenue "
                f"({_fmt(top['Revenue'])}, {_fmt(top['Quantity'])} units, "
                f"margin {top['Profit Margin %']:.1f}%) vs an average product "
                f"margin of {avg_margin:.1f}%."
            ),
            "business_impact": (
                "Promoting a proven, profitable product converts spend into "
                "revenue more reliably than promoting an untested one."
            ),
            "suggested_action": (
                f"Feature {top['Product Name']} in the next campaign and "
                "bundle it with a complementary mid-tier product."
            ),
            "potential_outcome": (
                "A focused campaign could lift sales of an already-winning "
                "product without discounting it heavily."
            ),
        })

    # 3. Weakest region ------------------------------------------------------
    if len(regions) >= 2:
        worst = regions.iloc[-1]
        best = regions.iloc[0]
        recs.append({
            "question": "Which region needs improvement?",
            "problem": f"{worst['Region']} is the weakest region by revenue.",
            "supporting_data": (
                f"{worst['Region']}: revenue {_fmt(worst['Revenue'])}, "
                f"profit {_fmt(worst['Profit'])}, orders {int(worst['Orders'])}. "
                f"Best region {best['Region']}: revenue {_fmt(best['Revenue'])}."
            ),
            "business_impact": (
                "A large gap between regions means untapped demand or weak "
                "coverage in the lagging region."
            ),
            "suggested_action": (
                f"Review pricing, assortment and sales-rep coverage in "
                f"{worst['Region']}, and run a small local promotion to test demand."
            ),
            "potential_outcome": (
                f"Closing even part of the gap to {best['Region']} could add "
                "meaningful revenue from an existing market."
            ),
        })

    # 4. At-risk customers ----------------------------------------------------
    if rfm_summary is not None and not rfm_summary.empty:
        at_risk = rfm_summary[rfm_summary["Segment"] == "At-risk"]
        if not at_risk.empty:
            row = at_risk.iloc[0]
            recs.append({
                "question": "Which customers are at risk of leaving?",
                "problem": (
                    f"{int(row['Customers'])} customer(s) are classified At-risk."
                ),
                "supporting_data": (
                    f"At-risk customers contributed {_fmt(row['Revenue'])} "
                    f"({row['Revenue Share %']:.1f}% of total revenue). "
                    "They bought frequently in the past but have not ordered recently."
                ),
                "business_impact": (
                    "Losing these customers would remove a proven revenue base "
                    "that is cheaper to retain than to replace."
                ),
                "suggested_action": (
                    "Send a win-back offer or check-in to the At-risk list "
                    "within the next two weeks."
                ),
                "potential_outcome": (
                    "Reactivating even a fraction of the At-risk segment could "
                    "protect a measurable share of revenue."
                ),
            })

    # 5. Discount / margin pressure -------------------------------------------
    if "Discount" in df.columns and "Profit Margin" in df.columns:
        high_disc = df[df["Discount"] >= 0.15]
        low_disc = df[df["Discount"] < 0.05]
        if not high_disc.empty and not low_disc.empty:
            hm = float(high_disc["Profit Margin"].mean() * 100)
            lm = float(low_disc["Profit Margin"].mean() * 100)
            if lm - hm > 5:
                recs.append({
                    "question": "What actions can increase profitability?",
                    "problem": "Heavy discounts are compressing margins.",
                    "supporting_data": (
                        f"Orders with discount >= 15% average {hm:.1f}% margin, "
                        f"vs {lm:.1f}% for orders with discount < 5% "
                        f"({len(high_disc)} high-discount orders in the data)."
                    ),
                    "business_impact": (
                        "Discounts buy volume but give away profit; the gap "
                        "shows how much margin is being traded for sales."
                    ),
                    "suggested_action": (
                        "Cap routine discounts at 10% and reserve deeper "
                        "discounts for clearance or win-back campaigns."
                    ),
                    "potential_outcome": (
                        "Tighter discount discipline could raise average margin "
                        "without necessarily losing volume."
                    ),
                })

    # 6. Declining products -----------------------------------------------------
    if len(monthly) >= 2 and "Product Name" in df.columns:
        last_month = df["Order Date"].max().to_period("M")
        prev_month = (df["Order Date"].max() - pd.offsets.MonthBegin(1)).to_period("M")
        cur = df[df["Order Date"].dt.to_period("M") == last_month]
        prv = df[df["Order Date"].dt.to_period("M") == prev_month]
        cur_rev = cur.groupby("Product Name")["Revenue"].sum()
        prv_rev = prv.groupby("Product Name")["Revenue"].sum()
        both = pd.DataFrame({"cur": cur_rev, "prv": prv_rev}).dropna()
        both = both[both["prv"] > 0]
        if not both.empty:
            both["change"] = (both["cur"] - both["prv"]) / both["prv"] * 100
            worst = both.sort_values("change").iloc[0]
            if worst["change"] < -20:
                recs.append({
                    "question": "Which products have declining demand?",
                    "problem": f"{worst.name} shows a sharp recent decline.",
                    "supporting_data": (
                        f"{worst.name}: revenue { _fmt(worst['prv'])} -> "
                        f"{_fmt(worst['cur'])} ({worst['change']:.0f}%) "
                        "between the last two months."
                    ),
                    "business_impact": (
                        "A steep drop can mean fading demand, stock issues or "
                        "a better competitor offer."
                    ),
                    "suggested_action": (
                        f"Check stock, price and reviews for {worst.name} before "
                        "deciding whether to promote, reprice or phase it out."
                    ),
                    "potential_outcome": (
                        "Catching a decline early leaves time to fix it before "
                        "it drags the category down."
                    ),
                })

    # 7. Overall margin health ---------------------------------------------------
    if kpis["profit_margin"] < 15:
        recs.append({
            "question": "What actions can increase profitability?",
            "problem": f"Overall profit margin is low at {kpis['profit_margin']:.1f}%.",
            "supporting_data": (
                f"Total revenue {_fmt(kpis['total_revenue'])}, total profit "
                f"{_fmt(kpis['total_profit'])} — margin "
                f"{kpis['profit_margin']:.1f}%."
            ),
            "business_impact": (
                "A thin margin leaves little room for discounts, returns or "
                "rising supplier costs."
            ),
            "suggested_action": (
                "Review unit costs on the lowest-margin products and test a "
                "small price increase on the best sellers."
            ),
            "potential_outcome": (
                "Even a 1–2 point margin improvement could add meaningful "
                "profit at current revenue."
            ),
        })

    if not recs:
        recs.append({
            "question": "What actions can increase profitability?",
            "problem": "No clear problem was detected in the current data.",
            "supporting_data": (
                f"Revenue {_fmt(kpis['total_revenue'])}, profit "
                f"{_fmt(kpis['total_profit'])}, margin "
                f"{kpis['profit_margin']:.1f}% — all within normal ranges "
                "for this dataset."
            ),
            "business_impact": "There is no urgent issue to fix.",
            "suggested_action": (
                "Keep monitoring monthly trends and the At-risk customer list."
            ),
            "potential_outcome": "Steady performance is maintained.",
        })
    return recs
