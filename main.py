from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from miopia_nlp import phenotype_course


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Analiza una frase ficticia con el baseline docente de miopía."
    )
    parser.add_argument(
        "texto",
        nargs="?",
        default="No se descarta miopía; pendiente de refracción.",
        help="Texto ficticio que se desea inspeccionar.",
    )
    args = parser.parse_args()
    print(json.dumps(phenotype_course(args.texto), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
