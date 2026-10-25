"""Caminhos, sementes e premissas de negócio do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 42
TEST_SIZE = 0.3
CV_FOLDS = 5
TARGET = "Class"

# Premissa de negócio (ajuste conforme o cenário): quanto custa um analista
# revisar um alerta, na mesma unidade de `Amount`. Uma fraude NÃO detectada
# custa o valor da própria transação.
REVIEW_COST = 5.0

# Faixas de risco operacionais (limites inferiores de cada faixa).
RISK_BINS = [0.0, 0.40, 0.70, 0.90]
RISK_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
