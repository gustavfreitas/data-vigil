"""Roda o pipeline completo: python main.py"""

import argparse

from src.config import DATA_RAW, REVIEW_COST
from src.data import load_raw
from src.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-Vigil: pipeline de detecção de fraude")
    parser.add_argument("--data", default=str(DATA_RAW), help="caminho do creditcard.csv")
    parser.add_argument(
        "--review-cost",
        type=float,
        default=REVIEW_COST,
        help="custo de revisar um alerta, na unidade de Amount (premissa de negócio)",
    )
    parser.add_argument("--n-boot", type=int, default=500, help="reamostragens do bootstrap")
    args = parser.parse_args()

    run(load_raw(args.data), review_cost=args.review_cost, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
