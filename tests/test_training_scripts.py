from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_nlp_course import (
    generate_renal_classification_rows,
    read_token_classification_jsonl,
)


def test_advanced_generator_is_deterministic_and_patient_safe() -> None:
    first = generate_renal_classification_rows(seed=17)
    second = generate_renal_classification_rows(seed=17)
    assert first == second
    assert len(first) == 320
    assert {row["label"] for row in first} == {
        "HEMODIALYSIS",
        "PERITONEAL_DIALYSIS",
        "TRANSPLANT",
        "NO_REPLACEMENT",
    }
    patient_splits: dict[str, set[str]] = {}
    for row in first:
        patient_splits.setdefault(row["patient_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in patient_splits.values())


def test_transformer_reader_validates_alignment_and_leakage(tmp_path: Path) -> None:
    rows = [
        {
            "document_id": "D1",
            "patient_id": "P1",
            "split": "train",
            "tokens": ["ERC", "G5"],
            "ner_tags": ["B-DISEASE", "I-DISEASE"],
        },
        {
            "document_id": "D2",
            "patient_id": "P2",
            "split": "test",
            "tokens": ["Estable"],
            "ner_tags": ["O"],
        },
    ]
    path = tmp_path / "ner.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert read_token_classification_jsonl(path) == rows

    rows[1]["patient_id"] = "P1"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="fuga"):
        read_token_classification_jsonl(path)
