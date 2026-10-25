# Resultados (gerado por `python main.py`)

Protocolo: modelo e thresholds escolhidos só com o **treino** (validação cruzada de 5 folds); o teste é usado uma única vez, para medir.

- Treino: 198.608 transações (331 fraudes)
- Teste: 85.118 transações (142 fraudes)

## Seleção do modelo (validação cruzada no treino)

| Modelo | PR-AUC (CV) | ROC-AUC (CV) |
|---|---:|---:|
| XGBoost (baseline) ✅ | 0,852 | 0,981 |
| XGBoost (tuned) | 0,850 | 0,981 |
| Random Forest | 0,827 | 0,971 |
| Logistic Regression | 0,736 | 0,979 |

## Desempenho no teste

Modelo: **XGBoost (baseline)** · PR-AUC 0,810 (IC 95%: 0,748–0,870) · ROC-AUC 0,976

| Threshold | Valor | Precision | Recall | F1 | Alertas | Fraudes pegas |
|---|---:|---:|---:|---:|---:|---:|
| Padrão (0,5) | 0,500 | 90,8% | 76,1% | 0,828 | 119 | 108 de 142 |
| Máx. F1 (escolhido no treino) | 0,452 | 90,0% | 76,1% | 0,824 | 120 | 108 de 142 |
| Mín. custo (escolhido no treino) | 0,187 | 85,8% | 76,8% | 0,810 | 127 | 109 de 142 |

Incerteza (bootstrap, 95%) no threshold de máx. F1: precision 0,840–0,950, recall 0,686–0,825, F1 0,772–0,869. Com 142 fraudes no teste, diferenças pequenas entre modelos ou thresholds não são conclusivas.

## Custo de negócio (no teste)

Premissa: revisar um alerta custa 5,00; uma fraude não detectada custa o valor da transação.

| Estratégia | Custo total |
|---|---:|
| Não sinalizar nada | 19.878 |
| Sinalizar tudo | 425.590 |
| Modelo, threshold máx. F1 | 4.537 |
| Modelo, threshold mín. custo | 4.568 |

## Sensibilidade ao custo de revisão (no teste)

Para cada custo de revisão, o threshold de custo é reescolhido só com o treino; o teste apenas mede. O threshold de máx. F1 ignora custos.

| Custo de revisão | Threshold (custo) | Alertas | Fraudes pegas | Custo (thr. custo) | Custo (thr. F1) | Economia vs. não sinalizar |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,000 | 1994 | 127 | 4.149 | 4.057 | 79,1% |
| 5 | 0,187 | 127 | 109 | 4.568 | 4.537 | 77,0% |
| 25 | 0,638 | 115 | 108 | 6.812 | 6.937 | 65,7% |
| 100 | 1,000 | 21 | 21 | 20.393 | 15.937 | -2,6% |
| 500 | 1,000 | 3 | 3 | 20.948 | 63.937 | -5,4% |

## Faixas de risco (no teste)

| Faixa | Transações |
|---|---:|
| LOW | 84.997 |
| MEDIUM | 7 |
| HIGH | 4 |
| CRITICAL | 110 |
