"""Data loading, validation and profiling.

Every function here works on a pandas DataFrame and never mutates its input.
The application always keeps the uploaded (raw) data separate from the
cleaned copy produced by :mod:`salespulse.cleaning`.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd

# Canonical column names the application understands. "Order ID" and
# "Order Date" are required; the rest are expected but missing ones are
# reported instead of silently assumed.
REQUIRED_COLUMNS = [
    "Order ID",
    "Order Date",
    "Customer ID",
    "Customer Name",
    "Product ID",
    "Product Name",
    "Category",
    "Quantity",
    "Unit Price",
    "Unit Cost",
    "Discount",
    "Region",
    "Sales Representative",
]

NUMERIC_COLUMNS = ["Quantity", "Unit Price", "Unit Cost", "Discount"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names; return a copy."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def load_file(file: BinaryIO, filename: str) -> pd.DataFrame:
    """Load an uploaded CSV or Excel file into a DataFrame.

    Raises a clear ValueError for unsupported formats or unreadable files.
    """
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            raise ValueError(
                f"Unsupported file type: {filename!r}. "
                "Please upload a .csv or .xlsx file."
            )
    except ValueError:
        raise
    except Exception as exc:  # unreadable / malformed file
        raise ValueError(f"Could not read {filename!r}: {exc}") from exc
    return standardize_columns(df)


def load_sample_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled sample dataset (data/sample_sales.csv)."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"
    df = pd.read_csv(path)
    return standardize_columns(df)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of required columns missing from ``df``."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def profile(df: pd.DataFrame) -> dict:
    """Summarise a raw DataFrame for the upload page.

    Returns rows/columns, dtypes, missing-value counts, duplicate count and
    a list of human-readable validation warnings.
    """
    warnings: list[str] = []
    missing_cols = validate_columns(df)
    if missing_cols:
        warnings.append(f"Missing expected column(s): {', '.join(missing_cols)}")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    dupes = int(df.duplicated().sum())

    if df.empty:
        warnings.append("The dataset is empty — no rows to analyse.")

    # Date parse check (informational only; cleaning does the real work).
    if "Order Date" in df.columns and not df.empty:
        parsed = pd.to_datetime(df["Order Date"], errors="coerce", format="mixed")
        bad = int(parsed.isna().sum())
        if bad:
            warnings.append(
                f"{bad} Order Date value(s) could not be parsed as dates."
            )

    for col in NUMERIC_COLUMNS:
        if col in df.columns and not df.empty:
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = int(coerced.isna().sum() - int(df[col].isna().sum()))
            if bad:
                warnings.append(f"{bad} non-numeric value(s) in {col!r}.")

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in missing.items()},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": dupes,
        "missing_columns": missing_cols,
        "warnings": warnings,
    }
