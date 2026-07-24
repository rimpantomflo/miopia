"""Prepara JSONL ficticio para un proyecto NER de Doccano.

Ejemplo:
    uv run python scripts/preparar_doccano.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_nlp_course import dictionary_suggestions, make_doccano_record


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte cursos JSONL a JSONL preanotado para Doccano."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "nefrologia_sintetica.jsonl",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "data" / "conceptos_renales_sinteticos.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "doccano_nefrologia_sintetica.jsonl",
    )
    parser.add_argument("--version", default="renal-dictionary-v1")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    concepts = json.loads(args.concepts.read_text(encoding="utf-8"))
    source_rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    suggestion_count = 0
    for row in source_rows:
        suggestions = dictionary_suggestions(row["text"], concepts)
        suggestion_count += len(suggestions)
        records.append(
            make_doccano_record(
                document_id=row["course_id"],
                text=row["text"],
                suggestions=suggestions,
                suggestion_version=args.version,
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        f"{len(records)} documentos y {suggestion_count} sugerencias -> "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
