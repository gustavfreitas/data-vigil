"""Carga, limpeza, engenharia de atributos e split treino/teste."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_RAW, SEED, TARGET, TEST_SIZE


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    """Lê o CSV original. Dá uma mensagem clara se o arquivo não existir."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. "
            "Rode `python -m scripts.download_data` ou baixe o creditcard.csv "
            "do Kaggle (mlg-ulb/creditcardfraud) e coloque em data/."
        )
    return pd.read_csv(path)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas ANTES do split (evita a mesma linha no treino e no teste)."""
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria `Amount_log` e `Hour` e remove `Amount` e `Time` originais."""
    out = df.copy()
    out["Amount_log"] = np.log1p(out["Amount"])
    out["Hour"] = (out["Time"] % (24 * 3600)) // 3600
    return out.drop(columns=["Amount", "Time"])


def amount_from_log(amount_log) -> np.ndarray:
    """Recupera o valor original da transação a partir de `Amount_log`."""
    return np.expm1(np.asarray(amount_log, dtype=float))


def prepare(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Limpa, cria atributos e separa treino/teste (estratificado).

    Retorna X_train, X_test, y_train, y_test.
    """
    df = add_features(drop_duplicates(df))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)
