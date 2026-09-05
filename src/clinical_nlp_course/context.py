"""Motor ConText pequeño para español/catalán con alcance auditable."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Trigger:
    phrase: str
    dimension: str
    value: str
    direction: str = "forward"
    max_chars: int = 100
    priority: int = 0


DEFAULT_TRIGGERS = (
    Trigger("no se descarta", "assertion", "possible", priority=20),
    Trigger("no puede descartarse", "assertion", "possible", priority=20),
    Trigger("a descartar", "assertion", "possible", priority=15),
    Trigger("sospecha de", "assertion", "possible", priority=10),
    Trigger("sospita de", "assertion", "possible", priority=10),
    Trigger("posible", "assertion", "possible", priority=10),
    Trigger("probable", "assertion", "possible", priority=10),
    Trigger("sin evidencia de", "assertion", "negated", priority=10),
    Trigger("ausencia de", "assertion", "negated", priority=10),
    Trigger("no presenta", "assertion", "negated", priority=10),
    Trigger("niega", "assertion", "negated", priority=10),
    Trigger("nega", "assertion", "negated", priority=10),
    Trigger("sense", "assertion", "negated", priority=5),
    Trigger("sin", "assertion", "negated", priority=5),
    Trigger("no", "assertion", "negated", priority=1),
    Trigger("antecedentes familiares de", "experiencer", "family", max_chars=140),
    Trigger("antecedents familiars de", "experiencer", "family", max_chars=140),
    Trigger("madre con", "experiencer", "family", max_chars=140),
    Trigger("padre con", "experiencer", "family", max_chars=140),
    Trigger("hermana con", "experiencer", "family", max_chars=140),
    Trigger("hermano con", "experiencer", "family", max_chars=140),
    Trigger("antecedente de", "temporality", "historical", max_chars=140),
    Trigger("antecedents de", "temporality", "historical", max_chars=140),
    Trigger("historia de", "temporality", "historical", max_chars=140),
    Trigger("previamente", "temporality", "historical", max_chars=140),
    Trigger("si aparece", "assertion", "conditional", max_chars=140),
    Trigger("en caso de", "assertion", "conditional", max_chars=140),
    Trigger("riesgo de", "assertion", "conditional", max_chars=140),
)

TERMINATOR_RE = re.compile(
    r"(?:[.;!?\n]|\bpero\b|\bsin embargo\b|\bno obstante\b|\bactualmente\b)",
    flags=re.IGNORECASE,
)


def _fold_with_map(text: str) -> tuple[str, list[int]]:
    """Normaliza conservando un mapa de cada carácter a su offset original."""

    output: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        folded = unicodedata.normalize("NFKD", char.casefold())
        for normalized_char in folded:
            if not unicodedata.combining(normalized_char):
                output.append(normalized_char)
                offsets.append(index)
    return "".join(output), offsets


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    right = len(text)
    for match in TERMINATOR_RE.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return left, right


def _matching_triggers(
    text: str,
    mention_start: int,
    mention_end: int,
    triggers: Iterable[Trigger],
) -> list[dict[str, Any]]:
    clause_start, clause_end = _clause_bounds(text, mention_start, mention_end)
    clause = text[clause_start:clause_end]
    folded, position_map = _fold_with_map(clause)
    matches: list[dict[str, Any]] = []
    for trigger in triggers:
        phrase, _ = _fold_with_map(trigger.phrase)
        for found in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", folded):
            original_start = clause_start + position_map[found.start()]
            original_end = clause_start + position_map[found.end() - 1] + 1
            if trigger.direction == "forward":
                distance = mention_start - original_end
                active = 0 <= distance <= trigger.max_chars
            elif trigger.direction == "backward":
                distance = original_start - mention_end
                active = 0 <= distance <= trigger.max_chars
            else:
                distance = min(
                    abs(mention_start - original_end),
                    abs(original_start - mention_end),
                )
                active = distance <= trigger.max_chars
            if active:
                matches.append(
                    {
                        "phrase": text[original_start:original_end],
                        "start": original_start,
                        "end": original_end,
                        "dimension": trigger.dimension,
                        "value": trigger.value,
                        "priority": trigger.priority,
                        "distance": distance,
                    }
                )
    return matches


def annotate_context(
    text: str,
    mentions: Iterable[Mapping[str, Any]],
    *,
    triggers: Iterable[Trigger] = DEFAULT_TRIGGERS,
) -> list[dict[str, Any]]:
    """Añade aserción, sujeto y temporalidad a spans ya detectados.

    Devuelve los disparadores exactos para que un clínico pueda auditar por qué
    se asignó cada atributo. Las reglas no atraviesan límites de cláusula.
    """

    trigger_list = tuple(triggers)
    output: list[dict[str, Any]] = []
    for mention in mentions:
        start = int(mention["start"])
        end = int(mention["end"])
        if not 0 <= start < end <= len(text):
            raise ValueError(f"span fuera de rango: {(start, end)}")
        active = _matching_triggers(text, start, end, trigger_list)
        values = {
            "assertion": "affirmed",
            "experiencer": "patient",
            "temporality": "current",
        }
        selected: dict[str, dict[str, Any]] = {}
        for candidate in active:
            dimension = candidate["dimension"]
            previous = selected.get(dimension)
            rank = (candidate["priority"], -candidate["distance"])
            previous_rank = (
                (previous["priority"], -previous["distance"])
                if previous is not None
                else (-1, float("-inf"))
            )
            if rank > previous_rank:
                selected[dimension] = candidate
                values[dimension] = candidate["value"]
        output.append(
            {
                **dict(mention),
                **values,
                "evidence": text[start:end],
                "context_triggers": sorted(
                    selected.values(), key=lambda row: (row["start"], row["dimension"])
                ),
            }
        )
    return output
