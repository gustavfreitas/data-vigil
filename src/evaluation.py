"""Métricas, escolha de threshold (por F1 e por custo) e incerteza (bootstrap).

Regra de ouro deste módulo: thresholds e seleção de modelo são calculados com
probabilidades *out-of-fold* do TREINO. O teste só é usado para medir.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import CV_FOLDS, REVIEW_COST, RISK_BINS, RISK_LABELS, SEED

# ---------------------------------------------------------------------------
# Probabilidades out-of-fold e comparação entre modelos
# ---------------------------------------------------------------------------


def oof_probabilities(model, X, y, cv: int = CV_FOLDS, seed: int = SEED) -> np.ndarray:
    """Probabilidade de fraude de cada linha, prevista por um modelo que NÃO a viu no treino."""
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]


def compare_models(models: dict, X, y, cv: int = CV_FOLDS, seed: int = SEED):
    """Compara modelos por PR-AUC e ROC-AUC com validação cruzada (sem tocar no teste).

    Retorna (tabela_ordenada_por_PR-AUC, {nome: probabilidades_oof}).
    """
    rows, oof = [], {}
    for name, model in models.items():
        probs = oof_probabilities(model, X, y, cv=cv, seed=seed)
        oof[name] = probs
        rows.append(
            {
                "Modelo": name,
                "PR-AUC (CV)": average_precision_score(y, probs),
                "ROC-AUC (CV)": roc_auc_score(y, probs),
            }
        )
    table = pd.DataFrame(rows).sort_values("PR-AUC (CV)", ascending=False)
    return table.reset_index(drop=True), oof


# ---------------------------------------------------------------------------
# Métricas num threshold fixo
# ---------------------------------------------------------------------------


def classification_metrics(y, probs, threshold: float) -> dict:
    """Precision, recall, F1 e matriz de confusão para um threshold dado."""
    y = np.asarray(y)
    flagged = np.asarray(probs) >= threshold
    tp = int(((y == 1) & flagged).sum())
    fp = int(((y == 0) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    tn = int(((y == 0) & ~flagged).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def ranking_metrics(y, probs) -> dict:
    """Métricas que não dependem de threshold."""
    return {
        "pr_auc": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
    }


# ---------------------------------------------------------------------------
# Escolha de threshold
# ---------------------------------------------------------------------------


def best_threshold_f1(y, probs) -> float:
    """Threshold que maximiza o F1 (escolha puramente estatística)."""
    precision, recall, thresholds = precision_recall_curve(y, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])


def total_cost(y, flagged, amounts, review_cost: float = REVIEW_COST) -> float:
    """Custo total = valor das fraudes não detectadas + custo de revisar cada alerta."""
    y = np.asarray(y)
    flagged = np.asarray(flagged, dtype=bool)
    amounts = np.asarray(amounts, dtype=float)
    missed = amounts[(y == 1) & ~flagged].sum()
    return float(missed + review_cost * flagged.sum())


def best_threshold_cost(y, probs, amounts, review_cost: float = REVIEW_COST):
    """Threshold que minimiza o custo total. Retorna (threshold, custo).

    Se nem sinalizar a transação de maior score compensa, o threshold devolvido
    fica acima de todos os scores (ou seja, "não sinalizar nada").
    """
    y = np.asarray(y)
    probs = np.asarray(probs, dtype=float)
    amounts = np.asarray(amounts, dtype=float)

    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    fraud_amount = (amounts * y)[order]

    total_fraud = fraud_amount.sum()
    caught = np.cumsum(fraud_amount)
    n_flagged = np.arange(1, len(sorted_probs) + 1)
    cost = (total_fraud - caught) + review_cost * n_flagged

    # Só dá para cortar entre scores diferentes (empates entram juntos).
    is_cut_point = np.append(sorted_probs[:-1] != sorted_probs[1:], True)
    cost = np.where(is_cut_point, cost, np.inf)

    k = int(np.argmin(cost))
    if total_fraud <= cost[k]:  # não sinalizar nada sai mais barato
        return float(np.nextafter(sorted_probs[0], np.inf)), float(total_fraud)
    return float(sorted_probs[k]), float(cost[k])


def cost_curve(y, probs, amounts, review_cost: float = REVIEW_COST, n_points: int = 200):
    """Custo total para uma grade de thresholds (para plotar)."""
    thresholds = np.linspace(0, 1, n_points)
    probs = np.asarray(probs)
    costs = [total_cost(y, probs >= t, amounts, review_cost) for t in thresholds]
    return thresholds, np.array(costs)


def review_cost_sensitivity(
    oof_train, y_train, amounts_train, probs_test, y_test, amounts_test, threshold_f1, review_costs
) -> list:
    """Como o threshold ótimo e o custo mudam conforme o custo de revisar um alerta.

    Para cada custo de revisão, o threshold é REESCOLHIDO só com o treino
    (probabilidades out-of-fold); o teste apenas mede. Compara com o threshold
    de máx. F1, que ignora custos.
    """
    y_test = np.asarray(y_test)
    probs_test = np.asarray(probs_test)
    amounts_test = np.asarray(amounts_test, dtype=float)
    cost_flag_nothing = float(amounts_test[y_test == 1].sum())

    rows = []
    for review_cost in review_costs:
        threshold, _ = best_threshold_cost(y_train, oof_train, amounts_train, review_cost)
        m = classification_metrics(y_test, probs_test, threshold)
        cost_at_cost = total_cost(y_test, probs_test >= threshold, amounts_test, review_cost)
        cost_at_f1 = total_cost(y_test, probs_test >= threshold_f1, amounts_test, review_cost)
        rows.append(
            {
                "review_cost": float(review_cost),
                "threshold_cost": threshold,
                "alerts": m["tp"] + m["fp"],
                "frauds_caught": m["tp"],
                "precision": m["precision"],
                "recall": m["recall"],
                "cost_at_cost_threshold": cost_at_cost,
                "cost_at_f1_threshold": cost_at_f1,
                "cost_flag_nothing": cost_flag_nothing,
                "saving_vs_nothing": (
                    1 - cost_at_cost / cost_flag_nothing if cost_flag_nothing else 0.0
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Incerteza e faixas de risco
# ---------------------------------------------------------------------------


def bootstrap_ci(y, probs, threshold: float, n_boot: int = 500, seed: int = SEED, alpha=0.05):
    """Intervalo de confiança (bootstrap) de precision, recall, F1 e PR-AUC.

    Com poucas fraudes no teste (~140), os intervalos são largos — e isso é
    informação importante para quem lê o resultado.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    probs = np.asarray(probs)
    n = len(y)
    draws = {"precision": [], "recall": [], "f1": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y[idx], probs[idx]
        if y_b.sum() == 0:
            continue
        m = classification_metrics(y_b, p_b, threshold)
        for key in ("precision", "recall", "f1"):
            draws[key].append(m[key])
        draws["pr_auc"].append(average_precision_score(y_b, p_b))
    low, high = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        k: (float(np.percentile(v, low)), float(np.percentile(v, high))) for k, v in draws.items()
    }


def risk_band(scores) -> pd.Series:
    """Converte o score em faixa operacional (LOW / MEDIUM / HIGH / CRITICAL)."""
    bins = [-np.inf, *RISK_BINS[1:], np.inf]
    return pd.cut(pd.Series(np.asarray(scores)), bins=bins, labels=RISK_LABELS, right=False).astype(
        str
    )
