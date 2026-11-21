"""Leakage-aware analytics for cleaned UCI Online Retail transactions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


FEATURES = [
    "recency", "frequency", "monetary", "tenure", "avg_order_value", "total_quantity"
]
REQUIRED = {"InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"}


def clean_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize types, deduplicate, and retain identifiable positive purchases.

    Reason counts are computed after deduplication and can overlap. Identical
    normalized source rows are treated as duplicates; revenue is recomputed.
    """
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    df = raw.drop(columns="revenue", errors="ignore").copy()
    for col in ("InvoiceNo", "CustomerID"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", errors="coerce")
    for col in ("Quantity", "UnitPrice"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_raw = len(df)
    df = df.drop_duplicates()
    duplicate_rows = n_raw - len(df)
    with np.errstate(over="ignore", invalid="ignore"):
        revenue = df["Quantity"] * df["UnitPrice"]
    reasons = {
        "missing_customer": df["CustomerID"].isna(),
        "missing_invoice": df["InvoiceNo"].isna(),
        "cancelled": df["InvoiceNo"].str.upper().str.startswith("C", na=False),
        "invalid_date": df["InvoiceDate"].isna(),
        "invalid_quantity": ~(df["Quantity"].gt(0) & np.isfinite(df["Quantity"])).fillna(False),
        "invalid_price": ~(df["UnitPrice"].gt(0) & np.isfinite(df["UnitPrice"])).fillna(False),
        "invalid_revenue": ~np.isfinite(revenue).fillna(False),
    }
    invalid = pd.DataFrame(reasons).any(axis=1)
    audit = {
        "input_rows": int(n_raw),
        "duplicate_rows": int(duplicate_rows),
        "invalid_rows": int(invalid.sum()),
        "reason_counts": {k: int(v.sum()) for k, v in reasons.items()},
        "output_rows": int((~invalid).sum()),
    }
    df = df.loc[~invalid].copy()
    df["revenue"] = revenue.loc[~invalid]
    return df.sort_values("InvoiceDate", kind="stable").reset_index(drop=True), audit


def features(df: pd.DataFrame, cutoff, window_days: int = 90) -> pd.DataFrame:
    """Features for customers active in [cutoff - window_days, cutoff).

    Recency and tenure are fractional days. Tenure means time since the first
    purchase in this window, not lifetime tenure. Frequency counts invoices.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    cutoff = pd.Timestamp(cutoff)
    history = df.loc[
        df["InvoiceDate"].ge(cutoff - pd.Timedelta(days=window_days))
        & df["InvoiceDate"].lt(cutoff)
    ]
    grouped = history.groupby("CustomerID", observed=True)
    result = grouped.agg(
        first=("InvoiceDate", "min"), last=("InvoiceDate", "max"),
        frequency=("InvoiceNo", "nunique"), monetary=("revenue", "sum"),
        total_quantity=("Quantity", "sum"),
    )
    result["recency"] = (cutoff - result["last"]).dt.total_seconds() / 86400
    result["tenure"] = (cutoff - result["first"]).dt.total_seconds() / 86400
    result["avg_order_value"] = result["monetary"] / result["frequency"]
    return result.reindex(columns=FEATURES).astype(float)


def _observed_end(df: pd.DataFrame) -> pd.Timestamp:
    if df.empty:
        raise ValueError("No cleaned transactions available")
    # Coverage is inferred at day granularity, not from the last sale's time.
    return df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)


def snapshot(df: pd.DataFrame, cutoff, horizon: int = 30):
    """Return 90-day features and a repeat-purchase label in [cutoff, end)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon)
    if end > _observed_end(df):
        raise ValueError(f"Insufficient future coverage for label window ending {end.date()}")
    X = features(df, cutoff)
    future = df.loc[df["InvoiceDate"].ge(cutoff) & df["InvoiceDate"].lt(end)]
    y = pd.Series(X.index.isin(future["CustomerID"]), index=X.index,
                  name="repurchased", dtype="int64")
    return X, y


def cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly purchase retention, with incomplete/unobserved months excluded.

    A cohort is the customer's first observed purchase month, not necessarily
    their true acquisition month. Data coverage is inferred from sale dates.
    Complete observed cells without purchases are zero; future cells are NaN.
    """
    if df.empty:
        return pd.DataFrame()
    start, end = df["InvoiceDate"].min().normalize(), _observed_end(df)
    activity = df[["CustomerID", "InvoiceDate"]].copy()
    activity["month"] = activity["InvoiceDate"].dt.to_period("M")
    first = activity.groupby("CustomerID", observed=True)["month"].min()
    complete = [m for m in pd.period_range(start, end, freq="M")
                if m.start_time >= start and (m + 1).start_time <= end]
    cohorts = sorted(set(first) & set(complete))
    if not cohorts:
        return pd.DataFrame()
    last = complete[-1]
    width = last.ordinal - cohorts[0].ordinal + 1
    result = pd.DataFrame(np.nan, index=[str(c) for c in cohorts], columns=range(width))
    result.index.name, result.columns.name = "cohort", "month_offset"
    activity["cohort"] = activity["CustomerID"].map(first)
    counts = activity.groupby(["cohort", "month"], observed=True)["CustomerID"].nunique()
    sizes = first.value_counts()
    for cohort in cohorts:
        for offset in range(last.ordinal - cohort.ordinal + 1):
            result.loc[str(cohort), offset] = float(
                counts.get((cohort, cohort + offset), 0) / sizes[cohort]
            )
    return result


def _metrics(y: pd.Series, scores) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    rate = float(y.mean())
    k = max(1, math.ceil(len(y) * 0.2))
    selected = np.argsort(-scores, kind="stable")[:k]
    precision = float(y[selected].mean())
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else 0.0,
        "precision_at_20": precision,
        "lift_at_20": precision / rate if rate else None,
        "positive_rate": rate,
    }


def evaluate(df: pd.DataFrame) -> dict:
    """Select C by September AP, then evaluate once on November.

    All transforms and candidates fit July only. Customers may appear in later
    snapshots, matching the intended scoring of existing retail customers.
    Top-20% metrics use ceil(0.2 * n); score ties preserve CustomerID order.
    """
    dates = {"train": "2011-07-01", "validation": "2011-09-01", "test": "2011-11-01"}
    splits = {name: snapshot(df, date) for name, date in dates.items()}
    for name, (X, y) in splits.items():
        if X.empty:
            raise ValueError(f"No eligible customers in {name} window")
        if y.nunique() < 2:
            raise ValueError(f"{name.capitalize()} labels must include both repeat and non-repeat customers")
    X_train, y_train = splits["train"]
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]
    best, best_ap, chosen = None, -1.0, None
    validation_scores = {}
    for C in (0.1, 1.0, 10.0):
        model = make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=42),
        )
        model.fit(X_train, y_train)
        score = _metrics(y_val, model.predict_proba(X_val)[:, 1])["average_precision"]
        validation_scores[str(C)] = score
        if score > best_ap:
            best, best_ap, chosen = model, score, C
    split_sizes = {}
    for name, (X, y) in splits.items():
        cutoff = pd.Timestamp(dates[name])
        split_sizes[name] = {
            "customers": int(len(X)), "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "feature_start": str((cutoff - pd.Timedelta(days=90)).date()),
            "cutoff": str(cutoff.date()),
            "label_end": str((cutoff + pd.Timedelta(days=30)).date()),
        }
    return {
        "model_metrics": _metrics(y_test, best.predict_proba(X_test)[:, 1]),
        "baseline_metrics": _metrics(y_test, -X_test["recency"]),
        "split_sizes": split_sizes,
        "chosen_C": float(chosen),
        "validation_ap_by_C": validation_scores,
        "feature_coefficients": dict(zip(FEATURES, best[-1].coef_[0].tolist())),
    }
