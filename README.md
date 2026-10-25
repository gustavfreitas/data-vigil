# Data-Vigil

[![CI](https://github.com/gustavfreitas/data-vigil/actions/workflows/ci.yml/badge.svg)](https://github.com/gustavfreitas/data-vigil/actions/workflows/ci.yml)

Sistema experimental de **detecção de anomalias em transações de cartão de crédito**, construído como um pipeline reprodutível de ciência de dados — da exploração dos dados brutos até a seleção, o custo de negócio e a explicabilidade do modelo final.

> Anomalias são tratadas aqui como **sinais potenciais** de fraude ou comportamento suspeito, não como veredito automático. O projeto tem finalidade educacional/portfólio; nenhuma decisão financeira real é tomada por este sistema.

---

## Resultados

Os números da versão atual são **gerados pelo código** e ficam em [`reports/results.md`](reports/results.md) depois de rodar `python main.py` (veja [Como rodar](#como-rodar)). Eles seguem este protocolo:

1. **Treino/teste separados antes de tudo** (70/30, estratificado, semente fixa).
2. **Modelo e thresholds escolhidos só com o treino**, usando probabilidades *out-of-fold* (validação cruzada de 5 folds). O teste é usado **uma única vez**, para medir.
3. **Três pontos de operação comparados**: threshold padrão (0,5), threshold que maximiza F1 e threshold que **minimiza o custo de negócio** (fraude não detectada custa o valor da transação; revisar um alerta custa um valor configurável).
4. **Incerteza explícita**: intervalos de confiança por bootstrap. Com ~140 fraudes no teste, diferenças pequenas entre modelos não são conclusivas — o relatório diz isso.

*Por que não comparar por acurácia: com ~0,17% de fraude no dataset, um modelo que sempre prevê "transação normal" acerta ~99,8% e é inútil. A métrica de seleção é a PR-AUC (Precision-Recall AUC), mais informativa que a ROC-AUC para classes extremamente raras.*

### Nota sobre a primeira versão (notebooks)

Os notebooks `03` e `04` registram a primeira versão do experimento. Nela, o **threshold (0,96) e o modelo final foram escolhidos olhando o próprio conjunto de teste**, então a precisão de 97,2% e o recall de 73,9% daquela versão são otimistas. Além disso, a vantagem do XGBoost "tuned" sobre o baseline era de 0,814 contra 0,810 de PR-AUC, uma diferença dentro do ruído com 142 fraudes. Os notebooks continuam no repositório como registro exploratório; **a fonte de verdade agora é o código em `src/`**.

| Modelo (v1, notebooks) | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost (tuned) | 0,872 | 0,768 | 0,816 | 0,966 | 0,814 |
| XGBoost (baseline) | 0,908 | 0,761 | 0,828 | 0,976 | 0,810 |
| Random Forest | 0,876 | 0,746 | 0,806 | 0,945 | 0,808 |
| Logistic Regression (baseline) | 0,051 | 0,887 | 0,096 | 0,963 | 0,691 |

*(precision/recall/F1 no threshold padrão; PR-AUC e ROC-AUC calculadas no teste)*

---

## O problema

> Como identificar transações que fogem do comportamento esperado e podem representar fraude ou comprometimento de conta, minimizando tanto fraudes não detectadas quanto bloqueios indevidos de clientes legítimos?

O pipeline não classifica direto como "fraude = 1". Ele gera um **score de anomalia**, converte esse score em uma **faixa de risco** (LOW / MEDIUM / HIGH / CRITICAL) e trata isso como recomendação de ação — não como decisão automática de bloqueio.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB/Kaggle, uso educacional): 284.807 transações realizadas por portadores de cartão europeus em setembro de 2013, com variáveis `V1`–`V28` anonimizadas via PCA, além de `Time`, `Amount` e o rótulo `Class` (1 = fraude). Apenas 492 transações (0,173%) são fraude.

O CSV **não é versionado neste repositório** (ver `.gitignore`). Baixe com `python -m scripts.download_data` (precisa do token do Kaggle, veja [`SETUP.md`](SETUP.md)) ou manualmente em `data/creditcard.csv`.

## Pipeline

```
creditcard.csv
      │
      ▼
src/data.py        → remove duplicatas, cria Amount_log e Hour, split treino/teste
      │
      ▼
src/pipeline.py    FASE 1 (só o treino)
      │            ├─ Logistic Regression, Random Forest, XGBoost, XGBoost + GridSearchCV
      │            ├─ comparação por PR-AUC com validação cruzada (probabilidades out-of-fold)
      │            └─ escolha do modelo e dos thresholds (F1 e custo) nas previsões out-of-fold
      │
      │            FASE 2 (teste, uma única vez)
      │            ├─ precision / recall / F1 / PR-AUC com intervalos de confiança (bootstrap)
      │            ├─ custo de negócio vs. "não sinalizar nada" e "sinalizar tudo"
      │            └─ faixas de risco e explicabilidade (SHAP)
      ▼
models/final_model.joblib   → modelo + thresholds + lista de features
reports/results.md          → tabelas prontas para o README
reports/final_metrics.json  → todos os números (para o que for consumir)
```

Os notebooks `01_eda.ipynb` a `04_avaliacao.ipynb` documentam a exploração e a primeira versão do experimento.

## Estrutura do repositório

```
data-vigil/
├── data/                    # dataset (não versionado)
├── notebooks/               # EDA e primeira versão do experimento
├── src/
│   ├── config.py            # caminhos, semente, premissas de negócio (custo de revisão)
│   ├── data.py              # carga, limpeza, atributos, split
│   ├── models.py            # modelos e ajuste de hiperparâmetros
│   ├── evaluation.py        # métricas, thresholds (F1 e custo), bootstrap, faixas de risco
│   ├── explain.py           # SHAP (opcional)
│   ├── reporting.py         # gráficos e results.md
│   └── pipeline.py          # fase 1 (escolhe) e fase 2 (mede)
├── scripts/download_data.py # baixa o dataset via Kaggle API
├── tests/                   # testes com dados sintéticos (não dependem do Kaggle)
├── models/                  # modelos serializados
├── reports/                 # figuras, métricas e results.md
├── main.py                  # roda o pipeline completo
├── requirements.txt         # ambiente completo (com Jupyter e SHAP)
├── requirements-dev.txt     # mínimo para pipeline, testes e CI
└── pyproject.toml           # configuração do ruff e do pytest
```

## Como rodar

```bash
git clone https://github.com/gustavfreitas/data-vigil.git
cd data-vigil
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.download_data   # ou coloque o creditcard.csv em data/
python main.py                    # roda tudo e gera models/ e reports/
```

Opções úteis: `python main.py --review-cost 10` muda a premissa de custo de revisão de um alerta (padrão: 5, na mesma unidade de `Amount`).

Testes e estilo (usam dados sintéticos, rodam em segundos):

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

A cada push e pull request, o GitHub Actions roda `ruff` e `pytest` em Python 3.11 e 3.12.

## Segurança e limitações

- **Anomalia ≠ fraude**: o modelo sinaliza desvio de comportamento; a confirmação de fraude exigiria contexto adicional (investigação humana, regras de negócio).
- **Poucas fraudes no teste (~140)**: as métricas têm incerteza grande (veja os intervalos em `reports/results.md`). Diferenças de décimos de ponto entre modelos não devem ser lidas como superioridade.
- **Custo de negócio é uma premissa**: o custo de revisão (padrão 5) é ilustrativo. Em um cenário real, viria da operação antifraude, e o threshold ótimo mudaria junto.
- **Ajuste de hiperparâmetros e comparação usam a mesma validação cruzada**: isso favorece levemente o modelo "tuned". O teste final, que não participa de nenhuma escolha, é a medida que vale.
- **Ataques adversariais**: um agente com conhecimento do modelo poderia, em tese, desenhar transações abaixo do threshold de detecção. Modelos baseados em árvore/ensemble dificultam — mas não eliminam — esse risco.
- **Concept drift**: o padrão de fraude muda com o tempo. Este modelo reflete o comportamento capturado no período do dataset (2013); em produção seria necessário monitoramento contínuo e re-treino periódico. O split aleatório também não simula esse efeito: um split temporal seria mais realista.
- **Dados**: dataset público e anonimizado via PCA, usado estritamente para fins educacionais.

## Evolução futura (produção)

Este projeto cobre o ciclo completo de ciência de dados sobre um dataset estático. Uma evolução natural seria:

```
API de transações (própria/simulada) → ETL em Python → PostgreSQL (bronze/silver/gold)
→ feature engineering comportamental (velocidade de transações, dispositivo novo, IP novo)
→ modelo + risk engine → FastAPI (endpoint /predict) + dashboard (Power BI)
→ monitoramento de drift → Docker Compose → CD
```

## Autor

Gustavo Freitas — [LinkedIn](https://www.linkedin.com/in/gs-freitas/) · [Portfólio](https://portfolio-tawny-one-99.vercel.app/)
