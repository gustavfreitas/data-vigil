"""Baixa o creditcard.csv do Kaggle para data/.

Uso:  python -m scripts.download_data

Requer `pip install kaggle` e um token em ~/.kaggle/kaggle.json
(Kaggle > Settings > API > Create New Token).
"""

import subprocess
import sys

from src.config import DATA_RAW


def main() -> None:
    if DATA_RAW.exists():
        print(f"{DATA_RAW} já existe. Nada a fazer.")
        return
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "mlg-ulb/creditcardfraud",
        "-p", str(DATA_RAW.parent),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("Comando `kaggle` não encontrado. Rode: pip install kaggle (e configure o token).")
    except subprocess.CalledProcessError:
        sys.exit("Falha no download. Confira o token em ~/.kaggle/kaggle.json.")
    print(f"Pronto: {DATA_RAW}")


if __name__ == "__main__":
    main()
