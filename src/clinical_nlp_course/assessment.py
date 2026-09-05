"""Rúbrica verificable y controles mínimos del capstone."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ASSESSMENT_RUBRIC = {
    "concepts": 20,
    "code_offsets": 5,
    "code_dictionary": 5,
    "code_ner": 10,
    "code_split": 5,
    "code_llm": 10,
    "code_rag": 5,
    "clinical_design": 20,
    "technology_selection": 10,
    "capstone": 10,
}

REQUIRED_CAPSTONE_EVIDENCE = (
    "intended_use",
    "excluded_uses",
    "data_card",
    "annotation_protocol",
    "baseline_report",
    "candidate_report",
    "locked_test_report",
    "error_analysis",
    "model_card",
    "deployment_plan",
    "monitoring_plan",
    "rollback_plan",
)


def rubric_total(rubric: Mapping[str, int] = ASSESSMENT_RUBRIC) -> int:
    if any(not isinstance(points, int) or points < 0 for points in rubric.values()):
        raise ValueError("la rúbrica solo admite enteros no negativos")
    return sum(rubric.values())


def score_assessment(
    awarded: Mapping[str, int],
    *,
    critical_failures: Sequence[str] = (),
    passing_score: int = 80,
) -> dict[str, Any]:
    """Valida límites por sección; un fallo crítico impide aprobar."""

    unknown = set(awarded) - set(ASSESSMENT_RUBRIC)
    missing = set(ASSESSMENT_RUBRIC) - set(awarded)
    if unknown or missing:
        raise ValueError(
            f"secciones desconocidas={sorted(unknown)}, ausentes={sorted(missing)}"
        )
    for section, points in awarded.items():
        if not isinstance(points, int) or not 0 <= points <= ASSESSMENT_RUBRIC[section]:
            raise ValueError(f"puntuación inválida en {section}: {points}")
    total = sum(awarded.values())
    return {
        "score": total,
        "maximum": rubric_total(),
        "critical_failures": list(critical_failures),
        "passed": total >= passing_score and not critical_failures,
    }


def validate_capstone_evidence(evidence: Mapping[str, Any]) -> list[str]:
    """Evita declarar completo un capstone con entregables vacíos."""

    issues: list[str] = []
    for field in REQUIRED_CAPSTONE_EVIDENCE:
        value = evidence.get(field)
        if value is None or value == "" or value == [] or value == {}:
            issues.append(f"falta evidencia: {field}")
    safeguards = evidence.get("safeguards", {})
    if not safeguards.get("patient_level_split"):
        issues.append("falta separación por paciente")
    if not safeguards.get("real_data_authorized"):
        issues.append("datos reales no constan como autorizados")
    if safeguards.get("phi_sent_to_unapproved_service"):
        issues.append("fallo crítico: PHI enviada a servicio no aprobado")
    if not safeguards.get("locked_test"):
        issues.append("test no consta como bloqueado")
    return issues
