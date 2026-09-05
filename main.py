import argparse
import json
import sys

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
