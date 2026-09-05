"""Baseline clínico auditable para detectar evidencia de miopía.

El objetivo de este módulo es docente: cada resultado conserva la regla, el
fragmento y los offsets que lo originaron. No sustituye la validación local ni
debe utilizarse para tomar decisiones clínicas sin revisión.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

MYOPIA_THRESHOLD_D = -0.50
MYOPIA_HIGH_THRESHOLD_D = -6.00

# Lista deliberadamente conservadora. Las erratas se añaden tras observarlas
# en el corpus; usar "miop*" sin límites genera falsos positivos evitables.
MYOPIA_RE = re.compile(
    r"\b(?:"
    r"miopía\s+magna|miopia\s+magna|alta\s+miopía|alta\s+miopia|"
    r"gran\s+miop|"
    r"miopía|miopia|miope|miopes|miop|"
    r"mipía|mipia"
    r")\b",
    flags=re.IGNORECASE,
)

REFRACTION_RE = re.compile(
    r"""
    \b(?P<eye>OD|OI|AO)\b
    \s*[:=]?\s*
    (?:(?:esf|sph)\.?\s*)?
    (?P<sphere>[+\-−–]\s*\d{1,2}(?:[.,]\d{1,2})?)
    \s*(?:D|dpt)?
    (?:
        \s*[,;/]?\s*
        (?:(?:cil|cyl)\.?\s*)?
        (?P<cylinder>[+\-−–]\s*\d{1,2}(?:[.,]\d{1,2})?)
        \s*(?:D|dpt)?
        (?:\s*(?:x|a)\s*(?P<axis>\d{1,3})\s*°?)?
    )?
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Mention:
    text: str
    start: int
    end: int
    sentence: str
    assertion: str
    experiencer: str
    temporality: str
    context: str
    rule_id: str


@dataclass(frozen=True)
class Refraction:
    text: str
    start: int
    end: int
    eye: str
    sphere_d: float
    cylinder_d: float | None
    axis_deg: int | None
    spherical_equivalent_d: float
    completeness: str
    refractive_class: str
    rule_id: str


def _fold(text: str) -> str:
    """Minúsculas sin diacríticos para comparar contexto, no para offsets."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Segmentación mínima y determinista para contexto local."""
    left_candidates = [
        text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")
    ]
    left = max(left_candidates) + 1
    right_candidates = [
        position
        for mark in (".", "!", "?", ";", "\n")
        if (position := text.find(mark, end)) != -1
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return left, right


def _classify_context(
    sentence: str,
    mention_start_in_sentence: int,
) -> tuple[str, str, str, str]:
    folded = _fold(sentence)
    prefix = folded[max(0, mention_start_in_sentence - 90) : mention_start_in_sentence]

    educational_markers = (
        "se explica que",
        "informacion sobre",
        "folleto",
        "aumenta el riesgo",
        "educacion sanitaria",
    )
    context = (
        "educational"
        if any(marker in folded for marker in educational_markers)
        else "clinical"
    )

    family_re = re.compile(
        r"(?:\b(?:madre|padre|herman[oa]s?|abuel[oa]s?|hij[oa]s?|"
        r"familiares?|antecedentes?\s+familiares?)\b|\baf\s*:)"
    )
    experiencer = "family" if family_re.search(folded) else "patient"

    # "No se descarta" expresa incertidumbre, no negación.
    possible_re = re.compile(
        r"\b(?:posible|possible|probable|sospecha|sospita|"
        r"descartar|no\s+se\s+descarta|no\s+puede\s+descartarse)\b"
    )
    negated_re = re.compile(
        r"\b(?:no|niega|nega|sin|sense|ausencia\s+de|"
        r"descartad[ao]s?|no\s+presenta)\b"
    )
    if possible_re.search(prefix) or "?" in sentence:
        assertion = "possible"
    elif negated_re.search(prefix) or re.search(r"\bdescartad[ao]\b", folded):
        assertion = "negated"
    else:
        assertion = "affirmed"

    historical_re = re.compile(
        r"\b(?:antecedentes?|historia\s+de|intervenid[oa]|operad[oa]|"
        r"lasik|prk|cirugia\s+refractiva|previamente|en\s+(?:19|20)\d{2})\b"
    )
    temporality = "historical" if historical_re.search(folded) else "current"
    return assertion, experiencer, temporality, context


def extract_mentions(text: str) -> list[dict[str, Any]]:
    """Extrae menciones léxicas y clasifica cuatro dimensiones de contexto."""
    if not isinstance(text, str) or not text.strip():
        return []

    mentions: list[dict[str, Any]] = []
    for match in MYOPIA_RE.finditer(text):
        sent_start, sent_end = _sentence_bounds(text, match.start(), match.end())
        sentence = text[sent_start:sent_end].strip()
        mention_start_in_sentence = match.start() - sent_start
        assertion, experiencer, temporality, context = _classify_context(
            sentence,
            mention_start_in_sentence,
        )
        rule_id = (
            "MYOPIA_TYPO_WHITELIST"
            if _fold(match.group()) == "mipia"
            else "MYOPIA_LEXICON"
        )
        mention = Mention(
            text=match.group(),
            start=match.start(),
            end=match.end(),
            sentence=sentence,
            assertion=assertion,
            experiencer=experiencer,
            temporality=temporality,
            context=context,
            rule_id=rule_id,
        )
        mentions.append(asdict(mention))
    return mentions


def _parse_signed_number(value: str) -> float:
    return float(
        value.replace("−", "-").replace("–", "-").replace(" ", "").replace(",", ".")
    )


def _refractive_class(spherical_equivalent: float) -> str:
    if spherical_equivalent <= MYOPIA_HIGH_THRESHOLD_D:
        return "high_myopia_numeric"
    if spherical_equivalent <= MYOPIA_THRESHOLD_D:
        return "myopia_numeric"
    return "not_myopic_numeric"


def parse_refractions(text: str) -> list[dict[str, Any]]:
    """Extrae refracciones compactas del tipo ``OD esf -3 cil -1 x 90``.

    Si no se documenta cilindro, se conserva la esfera como aproximación y se
    marca ``sphere_only``. El dato numérico es evidencia, no un diagnóstico.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    measurements: list[dict[str, Any]] = []
    for match in REFRACTION_RE.finditer(text):
        sphere = _parse_signed_number(match.group("sphere"))
        cylinder_raw = match.group("cylinder")
        cylinder = _parse_signed_number(cylinder_raw) if cylinder_raw else None
        axis_raw = match.group("axis")
        axis = int(axis_raw) if axis_raw else None
        spherical_equivalent = sphere + (cylinder / 2 if cylinder is not None else 0)
        completeness = "sphere_and_cylinder" if cylinder is not None else "sphere_only"
        measurement = Refraction(
            text=match.group().strip(),
            start=match.start(),
            end=match.end(),
            eye=match.group("eye").upper(),
            sphere_d=sphere,
            cylinder_d=cylinder,
            axis_deg=axis,
            spherical_equivalent_d=round(spherical_equivalent, 2),
            completeness=completeness,
            refractive_class=_refractive_class(spherical_equivalent),
            rule_id="REFRACTION_EYE_SIGNED",
        )
        measurements.append(asdict(measurement))
    return measurements


def phenotype_course(text: str) -> dict[str, Any]:
    """Resume evidencia de un curso sin perder las extracciones originales."""
    mentions = extract_mentions(text)
    refractions = parse_refractions(text)

    patient_clinical = [
        mention
        for mention in mentions
        if mention["experiencer"] == "patient" and mention["context"] == "clinical"
    ]
    affirmed = [m for m in patient_clinical if m["assertion"] == "affirmed"]
    possible = [m for m in patient_clinical if m["assertion"] == "possible"]
    current_positive = [m for m in affirmed if m["temporality"] == "current"]
    current_negative = [
        m
        for m in patient_clinical
        if m["assertion"] == "negated" and m["temporality"] == "current"
    ]
    numeric_myopia = [
        measurement
        for measurement in refractions
        if measurement["refractive_class"] in {"myopia_numeric", "high_myopia_numeric"}
    ]

    if affirmed or numeric_myopia:
        document_status = "confirmed"
    elif possible:
        document_status = "possible"
    else:
        document_status = "not_supported"

    if current_positive or numeric_myopia:
        current_status = "confirmed"
    elif current_negative:
        current_status = "negated"
    elif possible:
        current_status = "possible"
    else:
        current_status = "unknown"

    return {
        "document_status": document_status,
        "ever_myopia": bool(affirmed or numeric_myopia),
        "current_status": current_status,
        "high_myopia_numeric": any(
            m["refractive_class"] == "high_myopia_numeric" for m in refractions
        ),
        "mentions": mentions,
        "refractions": refractions,
        "evidence_count": len(mentions) + len(refractions),
    }


def process_courses(
    rows: Iterable[Mapping[str, Any]],
    *,
    text_key: str = "texto",
) -> list[dict[str, Any]]:
    """Procesa registros conservando sus metadatos y añadiendo el fenotipo."""
    output: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        result.update(phenotype_course(str(row.get(text_key, "") or "")))
        output.append(result)
    return output


def _date_sort_key(value: Any) -> tuple[int, str]:
    if isinstance(value, datetime):
        return (1, value.isoformat())
    if isinstance(value, date):
        return (1, value.isoformat())
    if value is None:
        return (0, "")
    return (1, str(value))


def aggregate_patient(
    courses: Sequence[Mapping[str, Any]],
    *,
    date_key: str = "fecha",
) -> dict[str, Any]:
    """Agrega cursos de un paciente con una política longitudinal explícita."""
    processed = [
        dict(course)
        if "document_status" in course
        else {**dict(course), **phenotype_course(str(course.get("texto", "") or ""))}
        for course in courses
    ]
    ordered = sorted(processed, key=lambda item: _date_sort_key(item.get(date_key)))
    ever_myopia = any(bool(item["ever_myopia"]) for item in ordered)
    high_myopia_numeric = any(bool(item["high_myopia_numeric"]) for item in ordered)

    informative = [
        item
        for item in ordered
        if item["current_status"] in {"confirmed", "negated", "possible"}
    ]
    latest = informative[-1] if informative else None
    current_status = latest["current_status"] if latest else "unknown"

    return {
        "ever_myopia": ever_myopia,
        "current_status": current_status,
        "high_myopia_numeric": high_myopia_numeric,
        "n_courses": len(ordered),
        "n_courses_with_evidence": sum(item["evidence_count"] > 0 for item in ordered),
        "latest_evidence_date": latest.get(date_key) if latest else None,
    }


def binary_metrics(
    y_true: Sequence[bool | int],
    y_pred: Sequence[bool | int],
) -> dict[str, float | int]:
    """Calcula métricas binarias transparentes, incluida especificidad."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud")
    if not y_true:
        raise ValueError("Se necesita al menos un ejemplo")

    pairs = [
        (bool(truth), bool(prediction)) for truth, prediction in zip(y_true, y_pred)
    ]
    tp = sum(truth and prediction for truth, prediction in pairs)
    tn = sum(not truth and not prediction for truth, prediction in pairs)
    fp = sum(not truth and prediction for truth, prediction in pairs)
    fn = sum(truth and not prediction for truth, prediction in pairs)

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else math.nan

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    ppv = divide(tp, tp + fp)
    npv = divide(tn, tn + fn)
    if math.isnan(ppv) or math.isnan(sensitivity):
        f1 = math.nan
    elif ppv == 0 and sensitivity == 0:
        f1 = 0.0
    else:
        f1 = divide(2 * ppv * sensitivity, ppv + sensitivity)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "accuracy": divide(tp + tn, len(pairs)),
    }


def pseudonymize_id(identifier: str, secret: bytes, *, length: int = 20) -> str:
    """Crea un seudónimo determinista con HMAC-SHA256.

    La clave debe residir fuera del código y de los resultados. Seudonimizar no
    anonimiza: los datos resultantes continúan siendo datos personales.
    """
    if not secret or len(secret) < 16:
        raise ValueError("La clave debe tener al menos 16 bytes aleatorios")
    digest = hmac.new(secret, str(identifier).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:length]
