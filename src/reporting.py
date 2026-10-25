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


def plot_sensitivity(rows: list, threshold_f1: float, path: Path) -> None:
    """Como o threshold ótimo e o custo total mudam com o custo de revisar um alerta."""
    review_costs = [r["review_cost"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    thresholds = [min(r["threshold_cost"], 1.0) for r in rows]
    ax1.plot(review_costs, thresholds, marker="o", label="Threshold ótimo por custo")
    ax1.axhline(threshold_f1, linestyle="--", color="gray", label="Threshold de máx. F1")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Custo de revisar um alerta (escala log)")
    ax1.set_ylabel("Threshold")
    ax1.set_title("Threshold ótimo vs. custo de revisão")
    ax1.legend()

    ax2.plot(review_costs, [r["cost_at_cost_threshold"] for r in rows], marker="o",
             label="Threshold por custo")
    ax2.plot(review_costs, [r["cost_at_f1_threshold"] for r in rows], marker="s",
             label="Threshold de máx. F1")
    ax2.plot(review_costs, [r["cost_flag_nothing"] for r in rows], linestyle="--",
             color="gray", label="Não sinalizar nada")
    ax2.set_xscale("log")
    ax2.set_xlabel("Custo de revisar um alerta (escala log)")
    ax2.set_ylabel("Custo total no teste")
    ax2.set_title("Custo total por estratégia")
    ax2.legend()

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
    percent = 100 * value
    if abs(percent) < 0.05:  # evita "-0,0%"
        percent = 0.0
    return _dec(percent, 1) + "%"


def _ci(interval) -> str:
    return f"{_dec(interval[0])}–{_dec(interval[1])}"


def _sensitivity_section(rows) -> list:
    """Tabela de sensibilidade ao custo de revisão (vazia se o resumo não tiver a análise)."""
    if not rows:
        return []
    lines = [
        "## Sensibilidade ao custo de revisão (no teste)",
        "",
        "Para cada custo de revisão, o threshold de custo é reescolhido só com o treino; "
        "o teste apenas mede. O threshold de máx. F1 ignora custos.",
        "",
        "| Custo de revisão | Threshold (custo) | Alertas | Fraudes pegas | Custo (thr. custo) "
        "| Custo (thr. F1) | Economia vs. não sinalizar |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        threshold = _dec(r["threshold_cost"]) if r["threshold_cost"] <= 1 else "não sinalizar"
        lines.append(
            f"| {_dec(r['review_cost'], 0)} | {threshold} | {r['alerts']} | {r['frauds_caught']} "
            f"| {_int(r['cost_at_cost_threshold'])} | {_int(r['cost_at_f1_threshold'])} "
            f"| {_pct(r['saving_vs_nothing'])} |"
        )
    lines.append("")
    return lines


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
    ]
    lines += _sensitivity_section(summary.get("sensitivity"))
    lines += [
        "## Faixas de risco (no teste)",
        "",
        "| Faixa | Transações |",
        "|---|---:|",
    ]
    for band in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        lines.append(f"| {band} | {_int(summary['risk_bands'].get(band, 0))} |")
    lines.append("")
    return "\n".join(lines)
