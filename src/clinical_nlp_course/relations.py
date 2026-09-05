"""Extracción supervisada de relaciones con candidatos y negativos explícitos."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Entity:
    entity_id: str
    start: int
    end: int
    label: str
    text: str = ""


@dataclass(frozen=True)
class RelationCandidate:
    head: Entity
    tail: Entity
    text: str


def _entity(value: Entity | Mapping[str, Any]) -> Entity:
    if isinstance(value, Entity):
        return value
    return Entity(
        entity_id=str(value["entity_id"]),
        start=int(value["start"]),
        end=int(value["end"]),
        label=str(value["label"]),
        text=str(value.get("text", "")),
    )


def relation_candidates(
    text: str,
    entities: Iterable[Entity | Mapping[str, Any]],
    *,
    allowed_type_pairs: set[tuple[str, str]] | None = None,
    max_distance: int = 180,
    directed: bool = True,
) -> list[RelationCandidate]:
    """Genera candidatos reproducibles; los pares ausentes son negativos reales."""

    items = sorted((_entity(item) for item in entities), key=lambda item: item.start)
    candidates: list[RelationCandidate] = []
    for head_index, head in enumerate(items):
        for tail_index, tail in enumerate(items):
            if head_index == tail_index:
                continue
            if not directed and tail_index <= head_index:
                continue
            if (
                allowed_type_pairs
                and (head.label, tail.label) not in allowed_type_pairs
            ):
                continue
            distance = max(0, tail.start - head.end, head.start - tail.end)
            if distance <= max_distance:
                candidates.append(RelationCandidate(head=head, tail=tail, text=text))
    return candidates


def candidate_features(candidate: RelationCandidate) -> dict[str, Any]:
    head, tail = candidate.head, candidate.tail
    left, right = sorted((head, tail), key=lambda item: item.start)
    between = candidate.text[left.end : right.start].casefold()
    tokens = re.findall(r"\b\w+\b", between, flags=re.UNICODE)
    sentence_break = bool(re.search(r"[.!?;\n]", between))
    return {
        "head_type": head.label,
        "tail_type": tail.label,
        "type_pair": f"{head.label}->{tail.label}",
        "head_before_tail": head.start < tail.start,
        "same_sentence": not sentence_break,
        "distance_bucket": min(abs(tail.start - head.end) // 20, 8),
        "between_first": tokens[0] if tokens else "<EMPTY>",
        "between_last": tokens[-1] if tokens else "<EMPTY>",
        "between_has_con": "con" in tokens,
        "between_has_mediante": "mediante" in tokens,
        "between_has_por": "por" in tokens,
        "between_has_debido": "debido" in tokens,
    }


class RelationClassifier:
    """Baseline lineal que aprende sobre todos los pares candidatos."""

    def __init__(self, *, seed: int = 17) -> None:
        self.pipeline = Pipeline(
            [
                # Matriz densa: el baseline usa pocos rasgos y evita que algunas
                # combinaciones SciPy/scikit-learn produzcan índices sparse int64.
                ("features", DictVectorizer(sparse=False)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    def fit(
        self,
        candidates: Sequence[RelationCandidate],
        labels: Sequence[str],
    ) -> "RelationClassifier":
        if len(candidates) != len(labels) or not candidates:
            raise ValueError("candidates y labels deben tener igual longitud")
        if len(set(labels)) < 2:
            raise ValueError("se necesitan ejemplos positivos y negativos")
        self.pipeline.fit([candidate_features(row) for row in candidates], labels)
        return self

    def predict(self, candidates: Sequence[RelationCandidate]) -> list[str]:
        return [
            str(label)
            for label in self.pipeline.predict(
                [candidate_features(row) for row in candidates]
            )
        ]

    def predict_records(
        self,
        candidates: Sequence[RelationCandidate],
    ) -> list[dict[str, Any]]:
        probabilities = self.pipeline.predict_proba(
            [candidate_features(row) for row in candidates]
        )
        classes = [str(label) for label in self.pipeline.classes_]
        predictions = self.pipeline.predict(
            [candidate_features(row) for row in candidates]
        )
        return [
            {
                "head_id": candidate.head.entity_id,
                "tail_id": candidate.tail.entity_id,
                "label": str(prediction),
                "probabilities": {
                    label: float(value)
                    for label, value in zip(classes, row, strict=True)
                },
            }
            for candidate, prediction, row in zip(
                candidates, predictions, probabilities, strict=True
            )
        ]
