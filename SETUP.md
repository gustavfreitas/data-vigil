# Setup do ambiente — data-vigil

Passo a passo para preparar a máquina **antes de clonar/rodar o repositório**, do item mais crítico (sem ele, nada funciona) ao mais opcional (dispensável, só conveniência).

---

## 1. Python — a versão certa (crítico)

O projeto foi testado de ponta a ponta com **Python 3.12.3**. Recomendo instalar/usar essa versão (ou 3.13; o 3.11 **não** funciona, porque o `xgboost` fixado exige 3.12 ou mais novo) em vez da 3.14 que aparecia no seu ambiente Windows anterior.

**Por quê**: 3.14 é uma versão muito recente (lançada há poucos meses) — bibliotecas como XGBoost e SHAP costumam demorar a publicar builds totalmente validadas para versões novas do Python. Funcionou no seu caso, mas para um projeto de portfólio que outras pessoas vão clonar e rodar, uma versão amplamente suportada reduz o risco de alguém encontrar um erro de instalação que não é problema do *seu* código.

```bash
python --version   # confirme 3.12.x ou 3.13.x
```

Se não tiver, instale pelo [python.org](https://www.python.org/downloads/) ou via `pyenv`/Microsoft Store (Windows).

## 2. Git (crítico)

Necessário para clonar o repositório e (se for usar Git LFS, ver item 5) para baixar o `.gitattributes` corretamente.

```bash
git --version
```

## 3. Ambiente virtual (muito importante)

Isola as dependências do projeto do resto da sua máquina — evita que uma versão de biblioteca de outro projeto quebre este (ou vice-versa).

```bash
python -m venv .venv

# ativar:
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (cmd)
.venv\Scripts\activate.bat
# Linux/Mac
source .venv/bin/activate
```

## 4. Instalar as dependências (crítico)

Com o ambiente virtual ativado:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

O `requirements.txt` já está organizado por prioridade (Tier 1 = indispensável, Tier 7 = opcional) — se algo falhar na instalação, veja o comentário acima da biblioteca para saber o quão crítica ela é antes de tentar contornar o erro.

## 5. Dataset (crítico para rodar os notebooks)

O `creditcard.csv` **não é versionado no Git** (está no `.gitignore`, e o `.gitattributes` configura Git LFS caso você opte por versionar CSVs no futuro).

1. Baixe em: [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) (precisa de conta gratuita no Kaggle).
2. Coloque o arquivo em `data/creditcard.csv` (não em `notebooks/` — os notebooks leem o caminho relativo `../data/creditcard.csv`).

Alternativa automática: `python -m scripts.download_data` (veja o item 10 para configurar o token do Kaggle).

## 6. Registrar o kernel Jupyter (importante)

Para o Jupyter/VS Code enxergarem o ambiente virtual como opção de kernel:

```bash
python -m ipykernel install --user --name data-vigil --display-name "Python (data-vigil)"
```

Ao abrir os notebooks, selecione o kernel **Python (data-vigil)**.

## 7. Rodar o pipeline

```bash
python main.py
```

Gera `models/final_model.joblib`, `reports/results.md`, `reports/final_metrics.json` e as figuras `reports/figures/v2_*.png`. Para conferir que o código está saudável (sem precisar do dataset): `pytest` e `ruff check .`.

## 7b. Rodar os notebooks (opcional, na ordem)

```
01_eda.ipynb → 02_preprocessing.ipynb → 03_modelagem.ipynb → 04_avaliacao.ipynb
```

Cada notebook lê artefatos salvos pelo anterior (`data/processed/`, `models/`) — rodar fora de ordem vai gerar erro de arquivo não encontrado.

## 8. Verificação de segurança das dependências (recomendável, não crítico)

Boa prática antes de publicar o repositório — checa se alguma das bibliotecas fixadas tem vulnerabilidade conhecida (CVE):

```bash
pip install pip-audit
pip-audit
```

## 9. Editor/IDE (opcional)

Qualquer um funciona (Jupyter Lab, VS Code + extensão Jupyter, PyCharm). Nenhuma configuração especial é exigida além do kernel do passo 6.

## 10. Kaggle API (opcional, conveniência)

Só relevante se você quiser automatizar o download do dataset em vez de baixar manualmente pelo site:

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

Ou, com o token configurado, simplesmente: `python -m scripts.download_data`.

Exige configurar um token de API do Kaggle (`~/.kaggle/kaggle.json`) — dispensável para rodar o projeto uma única vez.
