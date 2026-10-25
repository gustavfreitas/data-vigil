import numpy as np
import pytest

from src.evaluation import (
    best_threshold_cost,
    best_threshold_f1,
    bootstrap_ci,
    classification_metrics,
    review_cost_sensitivity,
    risk_band,
    total_cost,
)


def test_classification_metrics_counts():
    y = [1, 1, 1, 0, 0, 0, 0]
    probs = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1]
    m = classification_metrics(y, probs, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_classification_metrics_no_alerts_does_not_divide_by_zero():
    m = classification_metrics([1, 0, 0], [0.1, 0.1, 0.1], threshold=0.9)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_best_threshold_f1_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_threshold_f1(y, probs)
    assert classification_metrics(y, probs, thr)["f1"] == pytest.approx(1.0)


def test_total_cost():
    y = [1, 1, 0, 0]
    amounts = [100, 50, 10, 10]
    flagged = [True, False, True, False]
    # perdeu a fraude de 50; revisou 2 alertas a 5 cada
    assert total_cost(y, flagged, amounts, review_cost=5) == pytest.approx(60)


def test_best_threshold_cost_flags_only_what_pays_off():
    # fraude grande (1000) vale revisar; fraude pequena (2) não vale (revisão custa 5)
    y = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.95, 0.90, 0.30, 0.20, 0.10])
    amounts = np.array([1000, 2, 30, 30, 30])
    thr, cost = best_threshold_cost(y, probs, amounts, review_cost=5)
    assert 0.90 < thr <= 0.95  # sinaliza só a transação de maior score
    assert cost == pytest.approx(5 + 2)  # 1 revisão + fraude pequena perdida


def test_best_threshold_cost_flag_nothing_when_review_too_expensive():
    y = np.array([1, 0, 0])
    probs = np.array([0.9, 0.2, 0.1])
    amounts = np.array([1, 10, 10])
    thr, cost = best_threshold_cost(y, probs, amounts, review_cost=100)
    assert thr > probs.max()
    assert cost == pytest.approx(1.0)


def test_best_threshold_cost_handles_tied_scores():
    y = np.array([1, 0, 0, 0])
    probs = np.array([0.5, 0.5, 0.5, 0.1])
    thr, _ = best_threshold_cost(y, probs, np.array([100, 1, 1, 1]), review_cost=1)
    flagged = probs >= thr
    assert flagged.sum() in (0, 3)  # empates entram juntos, nunca "metade"


def test_risk_band_boundaries():
    bands = risk_band([0.0, 0.39, 0.40, 0.69, 0.70, 0.89, 0.90, 1.0])
    assert list(bands) == ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "CRITICAL", "CRITICAL"]


def test_bootstrap_ci_is_ordered_and_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.05).astype(int)
    probs = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    first = bootstrap_ci(y, probs, 0.6, n_boot=50, seed=7)
    second = bootstrap_ci(y, probs, 0.6, n_boot=50, seed=7)
    assert first == second
    for low, high in first.values():
        assert low <= high


def _synthetic_scores(seed=3, n=4000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    probs = np.clip(0.15 + y * 0.5 + rng.normal(0, 0.2, n), 0, 1)
    amounts = rng.gamma(2.0, 50.0, n)
    return y, probs, amounts


def test_higher_review_cost_never_flags_more():
    """Revisar fica mais caro => o threshold ótimo só sobe e os alertas só diminuem."""
    y, probs, amounts = _synthetic_scores()
    thresholds, alerts = [], []
    for review_cost in (0.5, 5, 50, 500, 5000):
        thr, _ = best_threshold_cost(y, probs, amounts, review_cost)
        thresholds.append(thr)
        alerts.append(int((probs >= thr).sum()))
    assert thresholds == sorted(thresholds)
    assert alerts == sorted(alerts, reverse=True)


def test_review_cost_sensitivity_rows():
    y, probs, amounts = _synthetic_scores()
    y_test, probs_test, amounts_test = _synthetic_scores(seed=4)
    rows = review_cost_sensitivity(
        probs, y, amounts, probs_test, y_test, amounts_test,
        threshold_f1=0.5, review_costs=(1, 10, 1000),
    )
    assert [r["review_cost"] for r in rows] == [1.0, 10.0, 1000.0]
    for r in rows:
        assert r["alerts"] >= r["frauds_caught"] >= 0
        assert r["saving_vs_nothing"] <= 1.0
    # com revisão muito cara, sinalizar quase nada é o esperado
    assert rows[-1]["alerts"] <= rows[0]["alerts"]
