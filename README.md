# Data-Vigil

Sistema experimental de **detecção de anomalias em transações de cartão de crédito**, construído como um pipeline reprodutível de ciência de dados — da exploração dos dados brutos até a seleção e explicabilidade do modelo final.

> Anomalias são tratadas aqui como **sinais potenciais** de fraude ou comportamento suspeito, não como veredito automático. O projeto tem finalidade educacional/portfólio; nenhuma decisão financeira real é tomada por este sistema.

---

## Resultado final

| Modelo | Precision (fraude) | Recall (fraude) | F1 (fraude) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **XGBoost (tuned)** — modelo selecionado | 0,872 | 0,768 | 0,816 | 0,966 | **0,814** |
| XGBoost (baseline) | 0,908 | 0,761 | 0,828 | 0,976 | 0,810 |
| Random Forest | 0,876 | 0,746 | 0,806 | 0,945 | 0,808 |
| Logistic Regression (baseline) | 0,051 | 0,887 | 0,096 | 0,963 | 0,691 |

Com o threshold de decisão ajustado para maximizar F1 (score ≥ 0,96), o modelo final atinge **97,2% de precisão e 73,9% de recall** na detecção de fraude no conjunto de teste (85.443 transações, das quais 142 fraudes após remoção de duplicatas).

*Por que não comparar por acurácia: com ~0,17% de fraude no dataset, um modelo que sempre prevê "transação normal" acerta ~99,8% e é inútil. A métrica de seleção usada foi PR-AUC (Precision-Recall AUC), mais informativa que ROC-AUC para classes extremamente raras.*

---

## O problema

> Como identificar transações que fogem do comportamento esperado e podem representar fraude ou comprometimento de conta, minimizando tanto fraudes não detectadas quanto bloqueios indevidos de clientes legítimos?

O pipeline não classifica direto como "fraude = 1". Ele gera um **score de anomalia**, converte esse score em uma **faixa de risco** (LOW / MEDIUM / HIGH / CRITICAL) e trata isso como recomendação de ação — não como decisão automática de bloqueio.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB/Kaggle, uso educacional): 284.807 transações realizadas por portadores de cartão europeus em setembro de 2013, com variáveis `V1`–`V28` anonimizadas via PCA, além de `Time`, `Amount` e o rótulo `Class` (1 = fraude). Apenas 492 transações (0,173%) são fraude.

O CSV **não é versionado neste repositório** (ver `.gitignore`) — baixe em `data/creditcard.csv` antes de rodar os notebooks.

## Pipeline

```
creditcard.csv
      │
      ▼
01_eda.ipynb            → exploração e validação (integridade, distribuição de classes, correlações)
      │
      ▼
02_preprocessing.ipynb  → remoção de duplicatas, engenharia de atributos, split treino/teste
      │                    (scaler ajustado apenas no treino — sem vazamento de dados)
      ▼
03_modelagem.ipynb      → Logistic Regression → Random Forest → XGBoost → XGBoost + GridSearchCV
      │
      ▼
04_avaliacao.ipynb      → comparação por PR-AUC, seleção do modelo, ajuste de threshold,
      │                    matriz de confusão, faixas de risco, explicabilidade (SHAP)
      ▼
models/final_model.pkl  → modelo + threshold + lista de features, serializado
```

## Estrutura do repositório

```
data-vigil/
├── data/                    # dataset (não versionado)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
├── models/                  # modelos serializados (.pkl)
├── reports/
│   ├── figures/              # gráficos exportados pelos notebooks
│   ├── model_comparison.csv
│   └── final_metrics.json
├── src/                      # (próximo passo: funções reutilizáveis extraídas dos notebooks)
├── requirements.txt
└── README.md
```

## Como rodar

```bash
git clone https://github.com/gustavfreitas/data-vigil.git
cd data-vigil
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Baixe o `creditcard.csv` do Kaggle e coloque em `data/creditcard.csv`. Execute os notebooks em ordem (01 → 04); cada um lê os artefatos salvos pelo anterior.

## Segurança e limitações

- **Anomalia ≠ fraude**: o modelo sinaliza desvio de comportamento; a confirmação de fraude exigiria contexto adicional (investigação humana, regras de negócio).
- **Ataques adversariais**: um agente com conhecimento do modelo poderia, em tese, desenhar transações abaixo do threshold de detecção. Modelos baseados em árvore/ensemble dificultam — mas não eliminam — esse risco.
- **Concept drift**: o padrão de fraude muda com o tempo. Este modelo reflete o comportamento capturado no período do dataset (2013); em produção seria necessário monitoramento contínuo e re-treino periódico.
- **Dados**: dataset público e anonimizado via PCA, usado estritamente para fins educacionais.

## Próximos passos (V2)

Este projeto cobre o ciclo completo de ciência de dados sobre um dataset estático. Uma evolução natural seria:

```
API de transações (própria/simulada) → ETL em Python → PostgreSQL (bronze/silver/gold)
→ feature engineering comportamental (velocidade de transações, dispositivo novo, IP novo)
→ modelo + risk engine → FastAPI (endpoint /predict) + dashboard (Power BI)
→ monitoramento de drift → Docker Compose → CI/CD
```

## Autor

Gustavo Freitas — [LinkedIn](https://www.linkedin.com/in/gs-freitas/) · [Portfólio](https://portfolio-tawny-one-99.vercel.app/)

## Licença

MIT
