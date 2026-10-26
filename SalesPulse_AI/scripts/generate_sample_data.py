"""Generate the bundled sample sales dataset (data/sample_sales.csv).

The dataset is seeded (reproducible) and deliberately realistic:
- ~1,500 transactions over 12 months with a gentle growth trend and
  seasonal uplift in November–December.
- 24 products across 4 categories, 120 customers, 4 regions, 8 sales reps.
- A small amount of intentional messiness so the cleaning engine has
  something to show: a few missing values, 5 exact duplicate rows, mixed
  date formats and 2 invalid rows (negative quantity / negative price).

Revenue = Quantity × Unit Price × (1 − Discount)
Profit  = Revenue − Quantity × Unit Cost
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ORDERS = 1500

PRODUCTS = [
    # (Product ID, Product Name, Category, Unit Price, Unit Cost)
    ("P-001", "Laptop Pro 14", "Electronics", 899.00, 620.00),
    ("P-002", "Wireless Mouse", "Electronics", 29.99, 12.00),
    ("P-003", "Mechanical Keyboard", "Electronics", 89.00, 45.00),
    ("P-004", "4K Monitor 27in", "Electronics", 329.00, 210.00),
    ("P-005", "USB-C Hub", "Electronics", 45.00, 18.00),
    ("P-006", "Noise-Cancelling Headphones", "Electronics", 199.00, 110.00),
    ("P-007", "Standing Desk", "Furniture", 449.00, 260.00),
    ("P-008", "Ergonomic Chair", "Furniture", 299.00, 160.00),
    ("P-009", "Bookshelf", "Furniture", 129.00, 65.00),
    ("P-010", "Filing Cabinet", "Furniture", 179.00, 95.00),
    ("P-011", "Desk Lamp", "Furniture", 39.00, 15.00),
    ("P-012", "Conference Table", "Furniture", 899.00, 520.00),
    ("P-013", "Notebook Set", "Office Supplies", 19.99, 7.00),
    ("P-014", "Gel Pen Pack", "Office Supplies", 12.50, 4.00),
    ("P-015", "Printer Paper A4", "Office Supplies", 24.00, 11.00),
    ("P-016", "Stapler Pro", "Office Supplies", 22.00, 9.00),
    ("P-017", "Sticky Notes", "Office Supplies", 8.99, 3.00),
    ("P-018", "Desk Organiser", "Office Supplies", 27.00, 10.00),
    ("P-019", "Laptop Stand", "Accessories", 59.00, 24.00),
    ("P-020", "Phone Case", "Accessories", 24.99, 8.00),
    ("P-021", "Cable Set", "Accessories", 19.00, 6.00),
    ("P-022", "Webcam HD", "Accessories", 79.00, 34.00),
    ("P-023", "External SSD 1TB", "Accessories", 119.00, 68.00),
    ("P-024", "Wireless Charger", "Accessories", 35.00, 13.00),
]

REGIONS = ["North", "South", "East", "West"]
REGION_WEIGHTS = [0.34, 0.26, 0.22, 0.18]
REPS = [f"Rep {i}" for i in range(1, 9)]
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Arjun",
         "Meera", "Karan", "Divya", "Nikhil", "Ishita", "Aditya", "Kavya",
         "Manish", "Ritu", "Sanjay", "Pooja", "Varun", "Neha"]
LAST = ["Sharma", "Patel", "Iyer", "Khan", "Reddy", "Nair", "Gupta", "Verma",
        "Singh", "Das", "Menon", "Kulkarni", "Joshi", "Bose", "Chopra",
        "Agarwal", "Rao", "Desai", "Mehta", "Pillai"]


def _format_date(dt: pd.Timestamp, style: int) -> str:
    """Render a date in one of three formats (intentional messiness)."""
    if style == 0:
        return dt.strftime("%Y-%m-%d")          # ISO
    if style == 1:
        return dt.strftime("%d/%m/%Y")          # day-first
    return dt.strftime("%b %d, %Y")             # e.g. Jun 20, 2024


def generate(seed: int = SEED, n_orders: int = N_ORDERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Customers: 120, with a skew so some buy often (RFM spread).
    customers = []
    for i in range(1, 121):
        customers.append({
            "Customer ID": f"C-{i:03d}",
            "Customer Name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "Region": rng.choice(REGIONS, p=REGION_WEIGHTS),
            "activity": float(rng.choice([0.4, 1.0, 2.2, 4.0], p=[0.35, 0.35, 0.2, 0.1])),
        })
    weights = np.array([c["activity"] for c in customers], dtype=float)
    weights /= weights.sum()

    start, end = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")
    all_days = pd.date_range(start, end)
    # Gentle growth + seasonal uplift in Nov/Dec, dip in Feb.
    month_factor = np.array([
        1.00, 0.88, 1.02, 1.05, 1.08, 1.06, 1.10, 1.12, 1.08, 1.15, 1.30, 1.42
    ])
    day_weights = np.array([month_factor[d.month - 1] for d in all_days])
    day_weights /= day_weights.sum()

    rows = []
    for n in range(n_orders):
        day = all_days[rng.choice(len(all_days), p=day_weights)]
        cust = customers[rng.choice(len(customers), p=weights)]
        pid, pname, cat, price, cost = PRODUCTS[rng.integers(0, len(PRODUCTS))]
        qty = int(rng.choice([1, 1, 2, 2, 3, 4, 5, 8, 10], p=[.25, .2, .15, .12, .1, .08, .05, .03, .02]))
        discount = float(rng.choice([0.0, 0.0, 0.05, 0.10, 0.15, 0.20],
                                    p=[.4, .2, .15, .12, .08, .05]))
        rows.append({
            "Order ID": f"ORD-{1000 + n}",
            "Order Date": _format_date(day, int(rng.choice([0, 0, 0, 1, 2]))),
            "Customer ID": cust["Customer ID"],
            "Customer Name": cust["Customer Name"],
            "Product ID": pid,
            "Product Name": pname,
            "Category": cat,
            "Quantity": qty,
            "Unit Price": price,
            "Unit Cost": cost,
            "Discount": discount,
            "Region": cust["Region"],
            "Sales Representative": rng.choice(REPS),
        })

    df = pd.DataFrame(rows)

    # Intentional messiness (cleaning engine input) -------------------------
    # 5 exact duplicate rows.
    df = pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)
    # A few missing values.
    for col, idxs in [("Category", [3, 41]), ("Discount", [17]), ("Region", [55])]:
        df.loc[idxs, col] = np.nan
    # A couple of invalid rows the cleaner must drop.
    df.loc[len(df)] = df.iloc[0].copy()
    df.loc[len(df) - 1, "Quantity"] = -2
    df.loc[len(df)] = df.iloc[1].copy()
    df.loc[len(df) - 1, "Unit Price"] = -50.0
    # A padded / oddly-cased product name and region.
    df.loc[7, "Product Name"] = "  " + df.loc[7, "Product Name"] + " "
    df.loc[9, "Region"] = "EAST"
    df.loc[12, "Category"] = "ELECTRONICS"
    return df.reset_index(drop=True)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Wrote {out} — {len(df)} rows, {len(df.columns)} columns.")


if __name__ == "__main__":
    main()
