"""Pipeline ponta a ponta: dados -> modelos -> thresholds -> teste -> relatórios.

O desenho separa duas fases de propósito:

1. `fit_and_choose` recebe SÓ o treino. Escolhe o modelo e os thresholds usando
   probabilidades out-of-fold. Não existe como o teste vazar para cá.
2. `evaluate_on_test` mede, uma única vez, o que foi decidido na fase 1.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    REPORTS_DIR,
    REVIEW_COST,
    SEED,
    SENSITIVITY_REVIEW_COSTS,
)
from src.data import amount_from_log, prepare
from src.evaluation import (
    best_threshold_cost,
    best_threshold_f1,
    bootstrap_ci,
    classification_metrics,
    compare_models,
    cost_curve,
    ranking_metrics,
    review_cost_sensitivity,
    risk_band,
    total_cost,
)
from src.explain import shap_importance_plot
from src.models import build_models, tune_xgboost
from src.reporting import (
    plot_confusion,
    plot_cost_curve,
    plot_pr_curves,
    plot_sensitivity,
    render_results_md,
)


def fit_and_choose(
    X_train, y_train, cv=CV_FOLDS, param_grid=None, seed=SEED, review_cost=REVIEW_COST
):
    """Fase 1 — usa apenas o treino."""
    models = build_models(y_train, seed)
    tuned, best_params = tune_xgboost(X_train, y_train, seed=seed, param_grid=param_grid)
    models["XGBoost (tuned)"] = tuned

    table, oof = compare_models(models, X_train, y_train, cv=cv, seed=seed)
    selected = table.loc[0, "Modelo"]

    amounts_train = amount_from_log(X_train["Amount_log"])
    threshold_f1 = best_threshold_f1(y_train, oof[selected])
    threshold_cost, _ = best_threshold_cost(y_train, oof[selected], amounts_train, review_cost)

    final_model = clone(models[selected]).fit(X_train, y_train)
    return {
        "model": final_model,
        "selected": selected,
        "best_params": best_params,
        "comparison": table,
        "oof": oof,
        "threshold_f1": threshold_f1,
        "threshold_cost": threshold_cost,
        # usados só na análise de sensibilidade (continuam sendo dados de TREINO)
        "oof_selected": oof[selected],
        "y_train": y_train,
        "amounts_train": amounts_train,
    }


def evaluate_on_test(
    chosen,
    X_test,
    y_test,
    review_cost=REVIEW_COST,
    n_boot=500,
    seed=SEED,
    sensitivity_costs=SENSITIVITY_REVIEW_COSTS,
):
    """Fase 2 — mede no teste as decisões da fase 1 (nada é reajustado aqui)."""
    probs = chosen["model"].predict_proba(X_test)[:, 1]
    amounts = amount_from_log(X_test["Amount_log"])
    thr_f1, thr_cost = chosen["threshold_f1"], chosen["threshold_cost"]

    flagged_f1 = probs >= thr_f1
    flagged_cost = probs >= thr_cost
    total_fraud = float(amounts[y_test.to_numpy() == 1].sum())

    return {
        "probs": probs,
        "amounts": amounts,
        "ranking": ranking_metrics(y_test, probs),
        "at_default": classification_metrics(y_test, probs, 0.5),
        "at_f1": classification_metrics(y_test, probs, thr_f1),
        "at_cost": classification_metrics(y_test, probs, thr_cost),
        "cost": {
            "review_cost": review_cost,
            "flag_nothing": total_fraud,
            "flag_everything": float(review_cost * len(y_test)),
            "at_f1": total_cost(y_test, flagged_f1, amounts, review_cost),
            "at_cost": total_cost(y_test, flagged_cost, amounts, review_cost),
        },
        "ci_at_f1": bootstrap_ci(y_test, probs, thr_f1, n_boot=n_boot, seed=seed),
        "risk_bands": risk_band(probs).value_counts().to_dict(),
        "sensitivity": review_cost_sensitivity(
            chosen["oof_selected"],
            chosen["y_train"],
            chosen["amounts_train"],
            probs,
            y_test,
            amounts,
            thr_f1,
            sensitivity_costs,
        ),
    }


def run(
    df: pd.DataFrame,
    reports_dir: Path = REPORTS_DIR,
    models_dir: Path = MODELS_DIR,
    cv: int = CV_FOLDS,
    param_grid=None,
    n_boot: int = 500,
    review_cost: float = REVIEW_COST,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """Roda tudo e grava modelo, métricas, figuras e `results.md`. Devolve o resumo."""
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    X_train, X_test, y_train, y_test = prepare(df, seed=seed)
    log(f"Treino: {len(X_train)} linhas ({int(y_train.sum())} fraudes) | "
        f"Teste: {len(X_test)} linhas ({int(y_test.sum())} fraudes)")

    log("Fase 1/2: escolhendo modelo e thresholds só com o treino...")
    chosen = fit_and_choose(X_train, y_train, cv=cv, param_grid=param_grid, seed=seed,
                            review_cost=review_cost)
    log(f"Modelo selecionado: {chosen['selected']}")

    log("Fase 2/2: medindo no teste (uma única vez)...")
    test = evaluate_on_test(chosen, X_test, y_test, review_cost=review_cost, n_boot=n_boot,
                            seed=seed)

    summary = {
        "selected_model": chosen["selected"],
        "best_params_xgb": {k: (v.item() if hasattr(v, "item") else v)
                            for k, v in chosen["best_params"].items()},
        "cv_folds": cv,
        "data": {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "frauds_train": int(y_train.sum()),
            "frauds_test": int(y_test.sum()),
        },
        "cv_comparison": chosen["comparison"].to_dict("records"),
        "thresholds": {"f1": chosen["threshold_f1"], "cost": chosen["threshold_cost"],
                       "default": 0.5},
        "test": {k: test[k] for k in ("ranking", "at_default", "at_f1", "at_cost")},
        "cost": test["cost"],
        "ci_at_f1": test["ci_at_f1"],
        "risk_bands": test["risk_bands"],
        "sensitivity": test["sensitivity"],
    }

    # --- artefatos -------------------------------------------------------
    joblib.dump(
        {
            "model": chosen["model"],
            "model_name": chosen["selected"],
            "threshold_f1": chosen["threshold_f1"],
            "threshold_cost": chosen["threshold_cost"],
            "features": list(X_train.columns),
        },
        models_dir / "final_model.joblib",
    )
    chosen["comparison"].to_csv(reports_dir / "cv_comparison.csv", index=False)
    (reports_dir / "final_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "results.md").write_text(render_results_md(summary), encoding="utf-8")

    plot_pr_curves(y_train, chosen["oof"], figures_dir / "v2_pr_curves_oof.png")
    plot_confusion(test["at_f1"], f"Matriz de confusão — {chosen['selected']}",
                   figures_dir / "v2_confusion_matrix.png")
    grid, costs = cost_curve(y_test, test["probs"], test["amounts"], review_cost)
    plot_cost_curve(
        grid, costs,
        marks={
            "Máx. F1": (chosen["threshold_f1"], test["cost"]["at_f1"]),
            "Mín. custo": (chosen["threshold_cost"], test["cost"]["at_cost"]),
        },
        baselines={"Não sinalizar nada": test["cost"]["flag_nothing"]},
        path=figures_dir / "v2_cost_curve.png",
    )
    plot_sensitivity(test["sensitivity"], chosen["threshold_f1"],
                     figures_dir / "v2_sensitivity.png")
    try:
        if shap_importance_plot(chosen["model"], X_test, figures_dir / "v2_shap_importance.png"):
            log("Gráfico SHAP salvo.")
        else:
            log("SHAP pulado (pacote `shap` ausente ou modelo não é XGBoost).")
    except Exception as exc:  # explicabilidade é complementar: não derruba o pipeline
        log(f"SHAP falhou e foi pulado: {exc}")

    log(f"Pronto. Veja {reports_dir / 'results.md'}")
    return summary
