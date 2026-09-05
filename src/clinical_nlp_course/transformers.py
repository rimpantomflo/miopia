"""Preparación y decodificación de NER para tokenizadores de Transformers."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

IGNORE_INDEX = -100


def char_spans_to_bio(
    offsets: Sequence[Sequence[int]],
    spans: Iterable[Mapping[str, Any] | Sequence[Any]],
) -> list[str]:
    """Proyecta spans de caracteres a BIO y rechaza anotaciones ambiguas."""

    normalized: list[tuple[int, int, str]] = []
    for span in spans:
        if isinstance(span, Mapping):
            row = (int(span["start"]), int(span["end"]), str(span["label"]))
        else:
            row = (int(span[0]), int(span[1]), str(span[2]))
        if row[0] < 0 or row[0] >= row[1]:
            raise ValueError(f"span inválido: {row}")
        normalized.append(row)

    labels: list[str] = []
    previous_entity: tuple[int, int, str] | None = None
    for offset in offsets:
        start, end = int(offset[0]), int(offset[1])
        if start == end:
            labels.append("O")
            previous_entity = None
            continue
        covering = [span for span in normalized if start < span[1] and end > span[0]]
        if len(covering) > 1:
            raise ValueError(f"token {(start, end)} cubierto por spans solapados")
        if not covering:
            labels.append("O")
            previous_entity = None
            continue
        entity = covering[0]
        prefix = "I" if entity == previous_entity else "B"
        labels.append(f"{prefix}-{entity[2]}")
        previous_entity = entity
    return labels


def align_word_labels(
    word_ids: Sequence[int | None],
    word_labels: Sequence[int],
    *,
    label_all_subtokens: bool = False,
    b_to_i: Mapping[int, int] | None = None,
) -> list[int]:
    """Alinea una etiqueta por palabra con subtokens y tokens especiales."""

    aligned: list[int] = []
    previous_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE_INDEX)
        elif not 0 <= word_id < len(word_labels):
            raise ValueError(f"word_id fuera de rango: {word_id}")
        elif word_id != previous_word:
            aligned.append(int(word_labels[word_id]))
        elif label_all_subtokens:
            label = int(word_labels[word_id])
            aligned.append(int((b_to_i or {}).get(label, label)))
        else:
            aligned.append(IGNORE_INDEX)
        previous_word = word_id
    return aligned


def bio_to_char_spans(
    labels: Sequence[str],
    offsets: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Reconstruye spans y repara un I huérfano tratándolo como B."""

    if len(labels) != len(offsets):
        raise ValueError("labels y offsets deben tener igual longitud")
    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for label, offset in zip(labels, offsets, strict=True):
        start, end = int(offset[0]), int(offset[1])
        if start == end or label == "O" or label == IGNORE_INDEX:
            if current is not None:
                spans.append(current)
                current = None
            continue
        if not isinstance(label, str) or "-" not in label:
            raise ValueError(f"etiqueta BIO inválida: {label!r}")
        prefix, entity_type = label.split("-", 1)
        if prefix not in {"B", "I"}:
            raise ValueError(f"prefijo BIO inválido: {prefix}")
        continuing = (
            prefix == "I" and current is not None and current["label"] == entity_type
        )
        if continuing:
            current["end"] = end
        else:
            if current is not None:
                spans.append(current)
            current = {"start": start, "end": end, "label": entity_type}
    if current is not None:
        spans.append(current)
    return spans


def assert_no_patient_leakage(rows: Iterable[Mapping[str, Any]]) -> None:
    """Falla si un paciente aparece en más de una partición."""

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_splits[str(row["patient_id"])].add(str(row["split"]))
    leaked = {
        patient: splits for patient, splits in patient_splits.items() if len(splits) > 1
    }
    if leaked:
        raise ValueError(f"fuga de pacientes entre particiones: {leaked}")


def read_token_classification_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Lee el contrato tokens/BIO y comprueba alineación y particiones."""

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise ValueError("el JSONL de token classification está vacío")
    required = {"document_id", "patient_id", "split", "tokens", "ner_tags"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"fila {index}: faltan {sorted(missing)}")
        if len(row["tokens"]) != len(row["ner_tags"]):
            raise ValueError(f"fila {index}: tokens y ner_tags no están alineados")
    assert_no_patient_leakage(rows)
    return rows
