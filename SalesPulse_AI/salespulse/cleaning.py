"""Automated data cleaning engine.

``clean_data`` never mutates its input: it works on a copy and returns the
cleaned DataFrame together with a transformation log, before/after statistics
and a data-quality score. Every change is explained in the log so the user
can audit exactly what happened.

Cleaning steps:
    1. Drop exact duplicate rows.
    2. Parse Order Date from mixed formats; drop rows whose date is
       unparseable (they cannot be placed on a timeline).
    3. Coerce Quantity / Unit Price / Unit Cost / Discount to numbers;
       drop rows with non-positive Quantity or negative prices/costs.
    4. Standardise Category / Region / Product Name casing and whitespace.
    5. Fill remaining missing values (never invents business facts):
       Category/Region -> "Unknown", Discount -> 0,
       Customer Name -> "Unknown Customer",
       Quantity / Unit Price / Unit Cost -> column median.
       Rows with a *present but invalid* Quantity (<= 0), negative price /
       cost, or Discount outside 0–1 are dropped, because no defensible
       fill exists for them.
    6. Re-derive Revenue, Profit and Profit Margin from the cleaned values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical spellings for the categories used by the sample dataset.
# Any other value is title-cased; unknown values stay readable, never lost.
CATEGORY_MAP = {
    "electronics": "Electronics",
    "furniture": "Furniture",
    "office supplies": "Office Supplies",
    "office supply": "Office Supplies",
    "accessories": "Accessories",
    "accessory": "Accessories",
    "stationery": "Stationery",
}


def _clean_text(value) -> str:
    """Trim whitespace and collapse internal double spaces."""
    return " ".join(str(value).strip().split())


def _canonical_category(value) -> str:
    text = _clean_text(value)
    return CATEGORY_MAP.get(text.lower(), text.title())


def quality_score(df: pd.DataFrame) -> float:
    """Data-quality score in [0, 100] for any DataFrame.

    Penalties (weights sum to 100):
      * missing cells: up to 50 points, proportional to missing-cell rate
      * duplicate rows: up to 30 points, proportional to duplicate rate
      * unparseable dates: up to 20 points, proportional to bad-date rate
    A fully clean dataset scores 100.
    """
    if df.empty:
        return 0.0
    n_cells = df.shape[0] * df.shape[1]
    missing_rate = float(df.isna().sum().sum()) / max(n_cells, 1)
    dup_rate = float(df.duplicated().sum()) / len(df)
    if "Order Date" in df.columns:
        bad_dates = float(
            pd.to_datetime(df["Order Date"], errors="coerce", format="mixed").isna().sum()
        ) / len(df)
    else:
        bad_dates = 0.0
    score = 100.0 - 50.0 * missing_rate - 30.0 * dup_rate - 20.0 * bad_dates
    return round(max(0.0, min(100.0, score)), 1)


def _stats(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_score": quality_score(df),
    }


def clean_data(df: pd.DataFrame) -> dict:
    """Clean ``df`` (input untouched) and return a result dictionary.

    Keys: ``cleaned`` (DataFrame), ``log`` (list[str] of human-readable
    steps), ``before`` / ``after`` (stats dicts), ``rows_removed`` (int).
    """
    raw = df.copy()
    before = _stats(raw)
    log: list[str] = []
    removed = 0

    work = raw.copy()
    work.columns = [str(c).strip() for c in work.columns]

    # 1. Exact duplicates -------------------------------------------------
    dupes = int(work.duplicated().sum())
    if dupes:
        work = work.drop_duplicates().reset_index(drop=True)
        removed += dupes
        log.append(
            f"Removed {dupes} exact duplicate row(s) "
            f"({before['rows']} -> {len(work)} rows)."
        )
    else:
        log.append("No exact duplicate rows found.")

    # 2. Dates -------------------------------------------------------------
    if "Order Date" in work.columns:
        parsed = pd.to_datetime(work["Order Date"], errors="coerce", format="mixed")
        bad = int(parsed.isna().sum())
        work["Order Date"] = parsed
        if bad:
            work = work[work["Order Date"].notna()].reset_index(drop=True)
            removed += bad
            log.append(
                f"Dropped {bad} row(s) with unparseable Order Date "
                "(they cannot be placed on a timeline)."
            )
        else:
            log.append("All Order Date values parsed (mixed formats normalised).")
    else:
        log.append("Order Date column missing — date-based modules will be empty.")

    # 3. Numeric validation -----------------------------------------------
    for col in ["Quantity", "Unit Price", "Unit Cost", "Discount"]:
        if col in work.columns:
            before_missing = int(work[col].isna().sum())
            work[col] = pd.to_numeric(work[col], errors="coerce")
            coerced_bad = int(work[col].isna().sum()) - before_missing
            if coerced_bad:
                log.append(
                    f"Converted {coerced_bad} non-numeric value(s) in "
                    f"{col!r} to missing (filled below)."
                )

    # Only *present* values can be invalid. Missing numerics are not
    # dropped here — they are filled with documented defaults in step 5.
    invalid = pd.Series(False, index=work.index)
    if "Quantity" in work.columns:
        invalid |= work["Quantity"].notna() & (work["Quantity"] <= 0)
    for col in ["Unit Price", "Unit Cost"]:
        if col in work.columns:
            invalid |= work[col].notna() & (work[col] < 0)
    if "Discount" in work.columns:
        invalid |= (
            work["Discount"].notna()
            & ((work["Discount"] < 0) | (work["Discount"] > 1))
        )
    n_invalid = int(invalid.sum())
    if n_invalid:
        work = work[~invalid].reset_index(drop=True)
        removed += n_invalid
        log.append(
            f"Dropped {n_invalid} invalid sales record(s): non-positive "
            "Quantity, negative price/cost, or Discount outside 0–1."
        )
    else:
        log.append("No invalid sales records (Quantity, prices, Discount all valid).")

    # 4. Text standardisation ---------------------------------------------
    for col in ["Category", "Region"]:
        if col in work.columns:
            work[col] = work[col].astype("string").str.strip()
    if "Category" in work.columns:
        work["Category"] = work["Category"].map(
            lambda v: _canonical_category(v) if pd.notna(v) else v
        )
        log.append("Standardised Category casing/whitespace (e.g. 'ELECTRONICS' -> 'Electronics').")
    if "Region" in work.columns:
        work["Region"] = work["Region"].map(
            lambda v: _clean_text(v).title() if pd.notna(v) else v
        )
        log.append("Standardised Region casing/whitespace (e.g. 'EAST' -> 'East').")
    if "Product Name" in work.columns:
        work["Product Name"] = work["Product Name"].map(
            lambda v: _clean_text(v) if pd.notna(v) else v
        )

    # 5. Missing-value fills (documented, never invented business facts) --
    fills = []
    for col, fill in [("Category", "Unknown"), ("Region", "Unknown"),
                      ("Customer Name", "Unknown Customer")]:
        if col in work.columns:
            n = int(work[col].isna().sum())
            if n:
                work[col] = work[col].fillna(fill)
                fills.append(f"{col}: {n} -> {fill!r}")
    for col in ["Quantity", "Unit Price", "Unit Cost"]:
        if col in work.columns:
            n = int(work[col].isna().sum())
            if n:
                median = float(work[col].median())
                work[col] = work[col].fillna(median)
                fills.append(f"{col}: {n} -> column median ({median:,.2f})")
    if "Discount" in work.columns:
        n = int(work["Discount"].isna().sum())
        if n:
            work["Discount"] = work["Discount"].fillna(0.0)
            fills.append(f"Discount: {n} -> 0 (no discount assumed)")
    if fills:
        log.append("Filled missing values: " + "; ".join(fills) + ".")
    else:
        log.append("No missing values left to fill.")

    # 6. Derived metrics (recomputed from cleaned values) ------------------
    if {"Quantity", "Unit Price", "Discount"}.issubset(work.columns):
        work["Revenue"] = (
            work["Quantity"] * work["Unit Price"] * (1.0 - work["Discount"])
        )
    if {"Revenue", "Quantity", "Unit Cost"}.issubset(work.columns):
        work["Profit"] = work["Revenue"] - work["Quantity"] * work["Unit Cost"]
    if "Revenue" in work.columns:
        work["Profit Margin"] = np.where(
            work["Revenue"] != 0, work["Profit"] / work["Revenue"], 0.0
        )
    log.append("Re-derived Revenue, Profit and Profit Margin from cleaned values.")

    after = _stats(work)
    log.append(
        f"Cleaning complete: {before['rows']} -> {after['rows']} rows "
        f"({removed} removed); quality score "
        f"{before['quality_score']} -> {after['quality_score']}."
    )
    return {
        "cleaned": work,
        "log": log,
        "before": before,
        "after": after,
        "rows_removed": removed,
    }
