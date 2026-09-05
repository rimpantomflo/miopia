"""Contratos e indicadores de deriva que nunca necesitan registrar texto."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InputContract:
    required_columns: tuple[str, ...] = (
        "patient_id",
        "course_id",
        "date",
        "language",
        "text",
    )
    allowed_languages: tuple[str, ...] = ("es", "ca")
    maximum_text_length: int = 200_000
    minimum_date: str = "1900-01-01"
    maximum_date: str = "2100-01-01"

    def validate(self, frame: pd.DataFrame) -> list[str]:
        issues: list[str] = []
        missing = set(self.required_columns) - set(frame.columns)
        if missing:
            return [f"columnas ausentes: {sorted(missing)}"]
        if frame.empty:
            issues.append("lote vacío")
        for column in ("patient_id", "course_id", "language", "text"):
            if frame[column].isna().any():
                issues.append(f"{column} contiene nulos")
        for column in ("patient_id", "course_id", "language", "text"):
            non_null = frame[column].dropna()
            if not non_null.map(lambda value: isinstance(value, str)).all():
                issues.append(f"{column} debe contener strings")
        if frame["course_id"].astype(str).duplicated().any():
            issues.append("course_id duplicado")
        if frame["course_id"].astype(str).str.strip().eq("").any():
            issues.append("course_id vacío")
        lengths = frame["text"].fillna("").astype(str).str.len()
        if lengths.gt(self.maximum_text_length).any():
            issues.append("texto supera longitud permitida")
        languages = set(frame["language"].dropna().astype(str))
        unknown = languages - set(self.allowed_languages)
        if unknown:
            issues.append(f"idiomas no previstos: {sorted(unknown)}")
        parsed_dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        if parsed_dates.isna().any():
            issues.append("fecha inválida o nula")
        else:
            lower = pd.Timestamp(self.minimum_date, tz="UTC")
            upper = pd.Timestamp(self.maximum_date, tz="UTC")
            if ((parsed_dates < lower) | (parsed_dates > upper)).any():
                issues.append("fecha fuera del intervalo permitido")
        return issues


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def operation_key(
    record_id: str,
    source_text: str,
    config_hash: str,
    *,
    pipeline_version: str,
) -> str:
    """Distingue texto o versión nuevos sin exponer el contenido en la clave."""

    payload = {
        "record_id": record_id,
        "content_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "config_hash": config_hash,
        "pipeline_version": pipeline_version,
    }
    return canonical_sha256(payload)


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: Sequence[float] | None = None,
    quantiles: int = 10,
) -> float:
    """PSI con colas abiertas para no perder observaciones fuera de rango."""

    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    if not len(reference_array) or not len(current_array):
        raise ValueError("reference y current no pueden estar vacíos")
    if not (np.isfinite(reference_array).all() and np.isfinite(current_array).all()):
        raise ValueError("PSI no admite NaN o infinito")
    if bins is None:
        if quantiles < 2:
            raise ValueError("quantiles debe ser >= 2")
        internal = np.quantile(reference_array, np.linspace(0, 1, quantiles + 1)[1:-1])
    else:
        internal = np.asarray(list(bins), dtype=float)
    edges = np.unique(np.concatenate(([-math.inf], internal, [math.inf])))
    if len(edges) < 3:
        raise ValueError("se necesitan al menos dos bins distintos")
    reference_counts, _ = np.histogram(reference_array, bins=edges)
    current_counts, _ = np.histogram(current_array, bins=edges)
    epsilon = 1e-6
    reference_pct = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_pct = np.clip(current_counts / current_counts.sum(), epsilon, None)
    return float(
        np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))
    )


def safe_batch_event(
    *,
    run_id: str,
    batch_number: int,
    rows: int,
    errors: int,
    duration_ms: float,
    model_version: str,
) -> Mapping[str, Any]:
    """Lista positiva de campos: no acepta texto ni identificadores clínicos."""

    return {
        "run_id": run_id,
        "event": "batch_complete",
        "batch_number": int(batch_number),
        "rows": int(rows),
        "errors": int(errors),
        "duration_ms": round(float(duration_ms), 3),
        "model_version": model_version,
    }
