import json

import joblib
import pytest

from src.data import prepare
from src.pipeline import evaluate_on_test, fit_and_choose, run

FAST_GRID = {"max_depth": [3], "n_estimators": [20], "learning_rate": [0.1]}


@pytest.fixture(scope="module")
def chosen_and_split(raw_df):
    X_train, X_test, y_train, y_test = prepare(raw_df)
    chosen = fit_and_choose(X_train, y_train, cv=3, param_grid=FAST_GRID)
    return chosen, X_train, X_test, y_train, y_test


def test_fit_and_choose_returns_valid_decisions(chosen_and_split):
    chosen, *_ = chosen_and_split
    assert chosen["selected"] in set(chosen["comparison"]["Modelo"])
    assert 0 < chosen["threshold_f1"] <= 1
    # a tabela vem ordenada: o selecionado é o de maior PR-AUC no CV
    pr_aucs = list(chosen["comparison"]["PR-AUC (CV)"])
    assert pr_aucs == sorted(pr_aucs, reverse=True)


def test_selected_model_beats_trivial_baseline(chosen_and_split):
    chosen, _, X_test, _, y_test = chosen_and_split
    result = evaluate_on_test(chosen, X_test, y_test, n_boot=20)
    # num dataset com 3% de fraude, PR-AUC aleatório seria ~0,03
    assert result["ranking"]["pr_auc"] > 0.3


def test_sensitivity_is_part_of_the_test_evaluation(chosen_and_split):
    chosen, _, X_test, _, y_test = chosen_and_split
    rows = evaluate_on_test(chosen, X_test, y_test, n_boot=20, sensitivity_costs=(1, 50))[
        "sensitivity"
    ]
    assert [r["review_cost"] for r in rows] == [1.0, 50.0]


def test_cost_never_worse_than_doing_nothing(chosen_and_split):
    chosen, _, X_test, _, y_test = chosen_and_split
    cost = evaluate_on_test(chosen, X_test, y_test, n_boot=20)["cost"]
    # o threshold de custo é escolhido no treino; no teste deve pelo menos não
    # custar mais que "não fazer nada" por uma margem grande
    assert cost["at_cost"] <= cost["flag_nothing"] * 1.10


def test_run_writes_all_artifacts(raw_df, tmp_path):
    summary = run(
        raw_df,
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        cv=3,
        param_grid=FAST_GRID,
        n_boot=20,
        verbose=False,
    )
    reports = tmp_path / "reports"
    for name in ("final_metrics.json", "cv_comparison.csv", "results.md"):
        assert (reports / name).exists(), name
    for fig in (
        "v2_pr_curves_oof.png",
        "v2_confusion_matrix.png",
        "v2_cost_curve.png",
        "v2_sensitivity.png",
    ):
        assert (reports / "figures" / fig).exists(), fig

    saved = joblib.load(tmp_path / "models" / "final_model.joblib")
    assert {"model", "threshold_f1", "threshold_cost", "features"} <= set(saved)

    on_disk = json.loads((reports / "final_metrics.json").read_text(encoding="utf-8"))
    assert on_disk["selected_model"] == summary["selected_model"]
    results_md = (reports / "results.md").read_text(encoding="utf-8")
    assert "Resultados" in results_md
    assert "Sensibilidade ao custo de revisão" in results_md
    assert len(on_disk["sensitivity"]) == len(summary["sensitivity"]) > 0
