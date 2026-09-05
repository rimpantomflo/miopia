from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from clinical_nlp_course import (
    ASSESSMENT_RUBRIC,
    InputContract,
    build_run_manifest,
    canonical_sha256,
    load_brat,
    load_entity_tsv,
    operation_key,
    population_stability_index,
    safe_batch_event,
    score_assessment,
    validate_capstone_evidence,
)
from clinical_nlp_course.cli import main as course_cli
from miopia_nlp.cli import main as phenotype_cli


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "SYN-P1",
                "course_id": "SYN-C1",
                "date": "2025-01-01",
                "language": "es",
                "text": "Texto ficticio.",
            }
        ]
    )


def test_input_contract_and_operation_key() -> None:
    contract = InputContract()
    assert contract.validate(valid_frame()) == []
    broken = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    assert "course_id duplicado" in contract.validate(broken)
    wrong_type = valid_frame().astype({"patient_id": object})
    wrong_type.loc[0, "patient_id"] = 123
    assert "patient_id debe contener strings" in contract.validate(wrong_type)
    first = operation_key("C1", "texto A", "cfg", pipeline_version="1")
    assert first == operation_key("C1", "texto A", "cfg", pipeline_version="1")
    assert first != operation_key("C1", "texto B", "cfg", pipeline_version="1")


def test_psi_keeps_outliers_and_safe_event() -> None:
    assert population_stability_index([1, 2, 3, 4], [1, 2, 3, 40], bins=[2, 3, 10]) > 0
    with pytest.raises(ValueError):
        population_stability_index([], [1])
    event = safe_batch_event(
        run_id="RUN-X",
        batch_number=1,
        rows=5,
        errors=0,
        duration_ms=12.34567,
        model_version="1.0",
    )
    assert "text" not in event
    assert event["duration_ms"] == 12.346


def test_hash_and_manifest(tmp_path: Path) -> None:
    data = tmp_path / "synthetic.txt"
    data.write_text("ficticio", encoding="utf-8")
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    manifest = build_run_manifest(
        name="test",
        configuration={"threshold": 0.5},
        data_files={"synthetic": str(data)},
    )
    assert manifest["run_id"].startswith("test-")
    assert len(manifest["data_sha256"]["synthetic"]) == 64


def test_public_corpus_adapters(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text("Hemodiálisis estable.", encoding="utf-8")
    (tmp_path / "doc1.ann").write_text(
        "T1\tTREATMENT 0 12\tHemodiálisis\n", encoding="utf-8"
    )
    brat = load_brat(tmp_path)
    assert brat[0]["entities"][0]["evidence"] == "Hemodiálisis"

    tsv = tmp_path / "entities.tsv"
    tsv.write_text(
        "document_id\tlabel\tstart\tend\ttext\nD1\tTREATMENT\t0\t13\tHemodiálisis\n",
        encoding="utf-8",
    )
    assert load_entity_tsv(tsv)[0]["document_id"] == "D1"


def test_clis() -> None:
    runner = CliRunner()
    result = runner.invoke(phenotype_cli, ["analyze", "Miopía magna.", "--compact"])
    assert result.exit_code == 0
    assert json.loads(result.output)["ever_myopia"] is True
    doctor = runner.invoke(course_cli, ["doctor"])
    assert doctor.exit_code == 0
    assert json.loads(doctor.output)["core"]["spacy"] is True

    template = Path("projects/capstone_template/submission.example.json")
    capstone = runner.invoke(course_cli, ["check-capstone", str(template)])
    assert capstone.exit_code == 1
    assert "falta evidencia" in capstone.output


def test_assessment_is_exactly_100_and_critical_failures_block() -> None:
    assert sum(ASSESSMENT_RUBRIC.values()) == 100
    perfect = score_assessment(ASSESSMENT_RUBRIC)
    assert perfect == {
        "score": 100,
        "maximum": 100,
        "critical_failures": [],
        "passed": True,
    }
    blocked = score_assessment(ASSESSMENT_RUBRIC, critical_failures=["privacy"])
    assert blocked["passed"] is False
    issues = validate_capstone_evidence({"safeguards": {}})
    assert "falta separación por paciente" in issues
