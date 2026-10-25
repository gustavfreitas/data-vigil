import numpy as np
import pandas as pd
import pytest

from src.data import add_features, amount_from_log, drop_duplicates, load_raw, prepare


def test_drop_duplicates_removes_repeated_rows(raw_df):
    doubled = pd.concat([raw_df.head(10), raw_df.head(10)], ignore_index=True)
    cleaned = drop_duplicates(doubled)
    assert len(cleaned) == 10
    assert list(cleaned.index) == list(range(10))


def test_add_features_creates_and_removes_columns(raw_df):
    out = add_features(raw_df)
    assert {"Amount_log", "Hour"} <= set(out.columns)
    assert "Amount" not in out.columns and "Time" not in out.columns
    assert out["Hour"].between(0, 23).all()
    assert "Amount" in raw_df.columns  # a função não altera o original


def test_amount_roundtrip(raw_df):
    out = add_features(raw_df)
    np.testing.assert_allclose(amount_from_log(out["Amount_log"]), raw_df["Amount"], rtol=1e-9)


def test_prepare_split_is_stratified_and_disjoint(raw_df):
    X_train, X_test, y_train, y_test = prepare(raw_df)
    assert len(X_train) + len(X_test) == len(raw_df)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.005
    assert "Class" not in X_train.columns


def test_prepare_is_reproducible(raw_df):
    first = prepare(raw_df)[1].index
    second = prepare(raw_df)[1].index
    assert list(first) == list(second)


def test_load_raw_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nao_existe.csv")
