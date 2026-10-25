"""Explicabilidade com SHAP (opcional: precisa do pacote `shap`)."""

from pathlib import Path

from src.config import SEED


def shap_importance_plot(model, X, path: Path, n: int = 500, seed: int = SEED) -> bool:
    """Salva o gráfico de importância SHAP. Devolve False se não for possível gerar.

    Só funciona para o XGBoost; para outros modelos o pipeline simplesmente pula esta etapa.
    """
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        return False
    if not hasattr(model, "get_booster"):
        return False

    sample = X.sample(min(n, len(X)), random_state=seed)
    values = shap.TreeExplainer(model).shap_values(sample)
    shap.summary_plot(values, sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()
    return True
