"""Gráficos e relatório em Markdown gerados pelo pipeline."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem precisar de janela

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402


def plot_pr_curves(y, oof_probs: dict, path: Path) -> None:
    """Curvas Precision-Recall (out-of-fold, treino) de todos os modelos."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probs in oof_probs.items():
        precision, recall, _ = precision_recall_curve(y, probs)
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curvas Precision-Recall (validação cruzada no treino)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(metrics: dict, title: str, path: Path) -> None:
    """Matriz de confusão a partir de tp/fp/fn/tn."""
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(matrix, cmap="Blues")
    labels = ["Normal", "Fraude"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            text = f"{matrix[i, j]:,}".replace(",", ".")
            ax.text(j, i, text, ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_cost_curve(thresholds, costs, marks: dict, baselines: dict, path: Path) -> None:
    """Custo total no teste em função do threshold, com as escolhas marcadas."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, costs, label="Modelo")
    for label, value in baselines.items():
        ax.axhline(value, linestyle="--", color="gray", alpha=0.7)
        ax.text(0.02, value, label, va="bottom", fontsize=8, color="gray")
    for label, (thr, cost) in marks.items():
        ax.scatter([thr], [cost], zorder=3, label=f"{label} (thr={thr:.3f})")
    if baselines:  # sem isso, o custo de "sinalizar tudo" (threshold 0) achata o resto
        ax.set_ylim(0, 1.5 * max(baselines.values()))
    ax.set_xlabel("Threshold de decisão")
    ax.set_ylabel("Custo total no teste")
    ax.set_title("Custo total vs. threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _dec(value: float, digits: int = 3) -> str:
    """Número com vírgula decimal (formato brasileiro)."""
    return f"{value:.{digits}f}".replace(".", ",")


def _int(value) -> str:
    """Inteiro com ponto de milhar."""
    return f"{value:,.0f}".replace(",", ".")


def _pct(value: float) -> str:
    return _dec(100 * value, 1) + "%"


def _ci(interval) -> str:
    return f"{_dec(interval[0])}–{_dec(interval[1])}"


def render_results_md(summary: dict) -> str:
    """Texto pronto para colar/linkar no README (todos os números vêm do pipeline)."""
    d, t, c = summary["data"], summary["test"], summary["cost"]
    ci = summary["ci_at_f1"]
    f1, cost_t = t["at_f1"], t["at_cost"]

    lines = [
        "# Resultados (gerado por `python main.py`)",
        "",
        "Protocolo: modelo e thresholds escolhidos só com o **treino** (validação cruzada "
        f"de {summary['cv_folds']} folds); o teste é usado uma única vez, para medir.",
        "",
        f"- Treino: {_int(d['n_train'])} transações ({d['frauds_train']} fraudes)",
        f"- Teste: {_int(d['n_test'])} transações ({d['frauds_test']} fraudes)",
        "",
        "## Seleção do modelo (validação cruzada no treino)",
        "",
        "| Modelo | PR-AUC (CV) | ROC-AUC (CV) |",
        "|---|---:|---:|",
    ]
    for row in summary["cv_comparison"]:
        mark = " ✅" if row["Modelo"] == summary["selected_model"] else ""
        lines.append(
            f"| {row['Modelo']}{mark} | {_dec(row['PR-AUC (CV)'])} | {_dec(row['ROC-AUC (CV)'])} |"
        )
    lines += [
        "",
        "## Desempenho no teste",
        "",
        f"Modelo: **{summary['selected_model']}** · PR-AUC {_dec(t['ranking']['pr_auc'])} "
        f"(IC 95%: {_ci(ci['pr_auc'])}) · ROC-AUC {_dec(t['ranking']['roc_auc'])}",
        "",
        "| Threshold | Valor | Precision | Recall | F1 | Alertas | Fraudes pegas |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, m in (
        ("Padrão (0,5)", t["at_default"]),
        ("Máx. F1 (escolhido no treino)", f1),
        ("Mín. custo (escolhido no treino)", cost_t),
    ):
        lines.append(
            f"| {label} | {_dec(m['threshold'])} | {_pct(m['precision'])} | {_pct(m['recall'])} "
            f"| {_dec(m['f1'])} | {m['tp'] + m['fp']} | {m['tp']} de {m['tp'] + m['fn']} |"
        )
    lines += [
        "",
        f"Incerteza (bootstrap, 95%) no threshold de máx. F1: precision {_ci(ci['precision'])}, "
        f"recall {_ci(ci['recall'])}, F1 {_ci(ci['f1'])}. Com {d['frauds_test']} fraudes no teste, "
        "diferenças pequenas entre modelos ou thresholds não são conclusivas.",
        "",
        "## Custo de negócio (no teste)",
        "",
        f"Premissa: revisar um alerta custa {_dec(c['review_cost'], 2)}; uma fraude não "
        "detectada custa o valor da transação.",
        "",
        "| Estratégia | Custo total |",
        "|---|---:|",
        f"| Não sinalizar nada | {_int(c['flag_nothing'])} |",
        f"| Sinalizar tudo | {_int(c['flag_everything'])} |",
        f"| Modelo, threshold máx. F1 | {_int(c['at_f1'])} |",
        f"| Modelo, threshold mín. custo | {_int(c['at_cost'])} |",
        "",
        "## Faixas de risco (no teste)",
        "",
        "| Faixa | Transações |",
        "|---|---:|",
    ]
    for band in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        lines.append(f"| {band} | {_int(summary['risk_bands'].get(band, 0))} |")
    lines.append("")
    return "\n".join(lines)
