"""Genera un benchmark multiclase ficticio, balanceado y sin PHI."""

from __future__ import annotations

import json
from pathlib import Path

from clinical_nlp_course import generate_renal_classification_rows

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "renal_classification_synthetic.jsonl"

if __name__ == "__main__":
    records = generate_renal_classification_rows()
    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"{OUTPUT}: {len(records)} filas sintéticas")
