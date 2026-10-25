"""Dados sintéticos no mesmo formato do creditcard.csv (os testes não dependem do Kaggle)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=6000,
        n_features=28,
        n_informative=10,
        n_redundant=2,
        weights=[0.97, 0.03],
        class_sep=1.5,
        flip_y=0,
        random_state=0,
    )
    rng = np.random.default_rng(0)
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.integers(0, 172_800, len(df))).astype(float))
    df["Amount"] = rng.gamma(2.0, 40.0, len(df)).round(2)
    df["Class"] = y
    return df
