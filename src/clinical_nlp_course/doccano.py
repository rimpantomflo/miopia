"""Adaptadores mínimos para preparar y auditar anotaciones de Doccano.

No dependen del servidor ni de su API: permiten probar el contrato con datos
ficticios antes de desplegar la herramienta.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

Label = tuple[int, int, str]


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _as_label(value: Mapping[str, Any] | Sequence[Any]) -> Label:
    if isinstance(value, Mapping):
        return int(value["start"]), int(value["end"]), str(value["label"])
    if len(value) != 3:
        raise ValueError("Una etiqueta Doccano debe contener start, end y label")
    return int(value[0]), int(value[1]), str(value[2])


def validate_doccano_labels(
    text: str, labels: Iterable[Mapping[str, Any] | Sequence[Any]]
) -> list[str]:
    """Comprueba rangos, etiquetas vacías, duplicados y solapamientos."""
    problems: list[str] = []
    parsed: list[Label] = []
    for index, raw_label in enumerate(labels):
        try:
            start, end, label = _as_label(raw_label)
        except (KeyError, TypeError, ValueError) as error:
            problems.append(f"etiqueta {index}: formato inválido ({error})")
            continue
        if not 0 <= start < end <= len(text):
            problems.append(f"etiqueta {index}: offsets fuera del texto")
        if not label.strip():
            problems.append(f"etiqueta {index}: label vacío")
        parsed.append((start, end, label))

    duplicates = [label for label, count in Counter(parsed).items() if count > 1]
    if duplicates:
        problems.append(f"etiquetas duplicadas: {duplicates}")

    ordered = sorted(set(parsed))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            problems.append(f"solapamiento: {previous} / {current}")
    return problems


def dictionary_suggestions(
    text: str,
    concepts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Crea sugerencias conservadoras y no solapadas a partir del diccionario."""
    candidates: list[dict[str, Any]] = []
    folded_text = _fold(text)

    for concept in concepts:
        concept_id = str(concept["concept_id"])
        terms = [str(concept["preferred_term"]), *map(str, concept.get("variants", []))]
        excluded_ranges: list[tuple[int, int]] = []
        for exclusion in map(str, concept.get("exclusions", [])):
            folded_exclusion = _fold(exclusion)
            for match in re.finditer(
                rf"(?<!\w){re.escape(folded_exclusion)}(?!\w)",
                folded_text,
            ):
                excluded_ranges.append(match.span())

        for term in sorted(set(terms), key=len, reverse=True):
            folded_term = _fold(term)
            for match in re.finditer(
                rf"(?<!\w){re.escape(folded_term)}(?!\w)",
                folded_text,
            ):
                start, end = match.span()
                if any(
                    ex_start <= start and end <= ex_end
                    for ex_start, ex_end in excluded_ranges
                ):
                    continue
                candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "label": concept_id,
                        "evidence": text[start:end],
                        "source": "dictionary",
                    }
                )

    # Primero el candidato más largo en cada posición; después elimina
    # duplicados y conflictos para una configuración Doccano sin solapamientos.
    selected: list[dict[str, Any]] = []
    seen: set[Label] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (item["start"], -(item["end"] - item["start"]), item["label"]),
    ):
        key = (candidate["start"], candidate["end"], candidate["label"])
        if key in seen:
            continue
        if any(
            candidate["start"] < prior["end"] and prior["start"] < candidate["end"]
            for prior in selected
        ):
            continue
        selected.append(candidate)
        seen.add(key)
    return sorted(
        selected, key=lambda item: (item["start"], item["end"], item["label"])
    )


def make_doccano_record(
    *,
    document_id: str,
    text: str,
    suggestions: Iterable[Mapping[str, Any]] = (),
    suggestion_version: str | None = None,
) -> dict[str, Any]:
    """Construye JSONL importable conservando procedencia en ``meta``."""
    labels = [
        [int(item["start"]), int(item["end"]), str(item["label"])]
        for item in suggestions
    ]
    problems = validate_doccano_labels(text, labels)
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "text": text,
        "labels": labels,
        "meta": {
            "document_id": str(document_id),
            "preannotated": bool(labels),
            "suggestion_version": suggestion_version,
        },
    }


def annotation_edit_stats(
    suggested: Iterable[Mapping[str, Any] | Sequence[Any]],
    final: Iterable[Mapping[str, Any] | Sequence[Any]],
) -> dict[str, int | float]:
    """Resume aceptación, eliminación y adición con coincidencia exacta."""
    suggested_counter = Counter(_as_label(label) for label in suggested)
    final_counter = Counter(_as_label(label) for label in final)
    accepted = sum((suggested_counter & final_counter).values())
    suggested_total = sum(suggested_counter.values())
    final_total = sum(final_counter.values())
    removed = suggested_total - accepted
    added = final_total - accepted
    return {
        "suggested": suggested_total,
        "final": final_total,
        "accepted": accepted,
        "removed": removed,
        "added": added,
        "acceptance_rate": accepted / suggested_total if suggested_total else 0.0,
    }
