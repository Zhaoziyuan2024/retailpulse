import json

import numpy as np
import pandas as pd
import pytest

from retailpulse import core


def transactions(rows):
    """Rows are customer, date, quantity, price, invoice (optional)."""
    return pd.DataFrame([
        {"CustomerID": r[0], "InvoiceDate": r[1], "Quantity": r[2],
         "UnitPrice": r[3], "InvoiceNo": r[4] if len(r) > 4 else f"I{i}"}
        for i, r in enumerate(rows)
    ])


def cleaned(rows):
    return core.clean_transactions(transactions(rows))[0]


def test_clean_filters_bad_rows_and_counts_duplicate_once():
    raw = transactions([
        ("A", "2011-01-01", 2, 3, "good"),
        ("A", "2011-01-01", 2, 3, "good"),
        (None, "2011-01-02", 1, 1),
        ("B", "2011-01-02", 1, 1, "c123"),
        ("B", "bad date", 1, 1),
        ("B", "2011-01-02", -1, 1),
        ("B", "2011-01-02", 1, 0),
        ("B", "2011-01-02", np.inf, 1),
        ("  ", "2011-01-02", 1, 1),
    ])
    df, audit = core.clean_transactions(raw)
    assert len(df) == 1
    assert df.iloc[0]["revenue"] == 6
    assert audit["duplicate_rows"] == 1
    assert audit["invalid_rows"] == 7
    assert audit["input_rows"] == audit["duplicate_rows"] + audit["invalid_rows"] + audit["output_rows"]
    json.dumps(audit, allow_nan=False)
    assert "revenue" not in raw  # No mutation of the caller's data.


def test_features_are_invariant_to_future_data_and_obey_window():
    df = cleaned([
        ("A", "2011-01-31", 1000, 1000),  # Outside 90-day history.
        ("A", "2011-02-01", 2, 5, "first"),
        ("A", "2011-04-30", 1, 5, "last"),
        ("A", "2011-04-30", 1, 2, "last"),
        ("A", "2011-05-01", 999999, 999999),  # At cutoff: label only.
        ("future-only", "2011-05-02", 1, 1),
    ])
    actual = core.features(df, "2011-05-02")
    # A separate exact-boundary check with an explicit 90-day start.
    cutoff = pd.Timestamp("2011-05-02")
    before = df[df.InvoiceDate < cutoff]
    future = cleaned([("A", "2011-05-03", 1e9, 1e9)])
    pd.testing.assert_frame_equal(actual, core.features(pd.concat([before, future]), cutoff))
    boundary = cleaned([
        ("A", cutoff - pd.Timedelta(days=90, seconds=1), 1000, 1),
        ("A", cutoff - pd.Timedelta(days=90), 2, 5, "one"),
        ("A", cutoff - pd.Timedelta(days=1), 1, 5, "two"),
        ("A", cutoff - pd.Timedelta(days=1), 1, 2, "two"),
        ("A", cutoff, 999999, 999999),
    ])
    X = core.features(boundary, cutoff)
    assert X.loc["A"].to_dict() == {
        "recency": 1.0, "frequency": 2.0, "monetary": 17.0,
        "tenure": 90.0, "avg_order_value": 8.5, "total_quantity": 4.0,
    }


def test_snapshot_labels_are_half_open_and_future_only_buyers_excluded():
    df = cleaned([
        ("A", "2011-04-01", 1, 1), ("B", "2011-04-01", 1, 1),
        ("A", "2011-05-01", 1, 1),  # Included: label start.
        ("B", "2011-05-31", 1, 1),  # Excluded: label end.
        ("new", "2011-05-02", 1, 1),
    ])
    X, y = core.snapshot(df, "2011-05-01")
    assert y.to_dict() == {"A": 1, "B": 0}
    assert X.index.equals(y.index)
    assert X.loc["A", "monetary"] == 1
    with pytest.raises(ValueError, match="Insufficient future coverage"):
        core.snapshot(df, "2011-05-03")


def test_snapshot_accepts_coverage_through_last_day():
    df = cleaned([("A", "2011-04-01", 1, 1), ("A", "2011-05-30 08:00", 1, 1)])
    assert core.snapshot(df, "2011-05-01")[1].loc["A"] == 1


def test_clean_rejects_nullable_numbers_and_overflowed_revenue():
    raw = transactions([
        ("A", "2011-01-01", "2", "3"),
        ("B", "2011-01-02", "bad", "1"),
        ("C", "2011-01-03", "1", None),
        ("D", "2011-01-04", "1e308", "1e308"),
        ("E", "2011-01-05", "1", "inf"),
    ])
    raw["Quantity"] = raw["Quantity"].astype("string")
    raw["UnitPrice"] = raw["UnitPrice"].astype("string")
    df, audit = core.clean_transactions(raw)
    assert df.CustomerID.tolist() == ["A"]
    assert df.revenue.tolist() == [6.0]
    assert audit["invalid_rows"] == 4


def test_empty_data_has_defined_boundaries():
    empty = cleaned([("A", "2011-01-01", -1, 1)])
    assert core.features(empty, "2011-05-01").empty
    assert core.cohort_retention(empty).empty
    with pytest.raises(ValueError, match="No cleaned transactions"):
        core.snapshot(empty, "2011-05-01")
    with pytest.raises(ValueError, match="Missing required columns"):
        core.clean_transactions(pd.DataFrame())


def test_cohorts_distinguish_zero_from_unobserved_and_drop_partial_month():
    df = cleaned([
        ("A", "2011-01-01", 1, 1), ("B", "2011-01-02", 1, 1),
        ("A", "2011-02-01", 1, 1), ("C", "2011-02-01", 1, 1),
        ("D", "2011-03-31", 1, 1),
        ("E", "2011-04-10", 1, 1),  # April is incomplete.
    ])
    retained = core.cohort_retention(df)
    assert retained.index.tolist() == ["2011-01", "2011-02", "2011-03"]
    assert retained.loc["2011-01"].tolist() == [1.0, 0.5, 0.0]
    assert retained.loc["2011-02", 1] == 0.0
    assert np.isnan(retained.loc["2011-02", 2])
    assert np.isnan(retained.loc["2011-03", 1])
    # January starts mid-month, so its cohort is excluded, not reassigned.
    partial = df.copy()
    partial.loc[partial.CustomerID == "A", "InvoiceDate"] = pd.to_datetime(
        ["2011-01-15", "2011-02-01"]
    )
    assert "2011-01" not in core.cohort_retention(partial).index


def test_evaluation_fits_preprocessing_only_on_training_and_is_json_safe(monkeypatch):
    rows = []
    for month in (6, 8, 10):
        for i in range(8):
            rows.append((str(i), f"2011-{month:02d}-{i + 1:02d}", i + 1, month))
            if i % 2 == 0:
                rows.append((str(i), f"2011-{month + 1:02d}-10", 1, 1))
    rows.append(("coverage", "2011-11-30", 1, 1))
    df = cleaned(rows)
    expected = np.log1p(core.snapshot(df, "2011-07-01")[0].to_numpy())
    seen = []
    original = core.StandardScaler.fit

    def capture_fit(self, X, y=None, **kwargs):
        seen.append(np.asarray(X).copy())
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(core.StandardScaler, "fit", capture_fit)
    report = core.evaluate(df)
    assert len(seen) == 3
    for fitted in seen:
        np.testing.assert_allclose(fitted, expected)
    assert report["chosen_C"] in [0.1, 1.0, 10.0]
    assert report["split_sizes"]["test"]["label_end"] == "2011-12-01"
    assert set(report["feature_coefficients"]) == set(core.FEATURES)
    assert report["model_metrics"]["positive_rate"] == 0.5
    assert report["baseline_metrics"]["positive_rate"] == 0.5
    json.dumps(report, allow_nan=False)
    # A mature test window with no repeat purchases cannot support AUC.
    no_test_positives = df.loc[
        ~df.InvoiceDate.between("2011-11-01", "2011-11-29 23:59:59")
    ]
    with pytest.raises(ValueError, match="Test labels must include both"):
        core.evaluate(no_test_positives)
