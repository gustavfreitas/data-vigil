"""Definição dos modelos e ajuste de hiperparâmetros."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import SEED

PARAM_GRID = {
    "max_depth": [3, 5],
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
}


def scale_pos_weight(y) -> float:
    """Razão entre negativos e positivos (peso da classe rara no XGBoost)."""
    return float((y == 0).sum() / (y == 1).sum())


def make_xgb(y, seed: int = SEED, **params) -> XGBClassifier:
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight(y),
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        **params,
    )


def build_models(y_train, seed: int = SEED) -> dict:
    """Os três modelos 'simples': baseline linear, Random Forest e XGBoost."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),  # fica no Pipeline: só aprende no treino
                (
                    "model",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "XGBoost (baseline)": make_xgb(y_train, seed),
    }


def tune_xgboost(X_train, y_train, seed: int = SEED, param_grid=None, cv: int = 3):
    """GridSearchCV (métrica PR-AUC) usando APENAS o treino.

    Retorna (melhor_modelo_já_treinado, melhores_parametros).
    """
    grid = GridSearchCV(
        make_xgb(y_train, seed),
        param_grid or PARAM_GRID,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
