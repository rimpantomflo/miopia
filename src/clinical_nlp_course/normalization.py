"""Normalización terminológica en dos etapas con abstención calibrable."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


@dataclass(frozen=True)
class Concept:
    concept_id: str
    preferred_term: str
    variants: tuple[str, ...]
    semantic_type: str = "unknown"


def _concept(value: Concept | Mapping[str, Any]) -> Concept:
    if isinstance(value, Concept):
        return value
    return Concept(
        concept_id=str(value["concept_id"]),
        preferred_term=str(value["preferred_term"]),
        variants=tuple(str(item) for item in value.get("variants", [])),
        semantic_type=str(value.get("semantic_type", "unknown")),
    )


class ConceptNormalizer:
    """Recupera por n-gramas y opcionalmente reordena con ejemplos locales."""

    def __init__(self, concepts: Iterable[Concept | Mapping[str, Any]]) -> None:
        self.concepts = [_concept(value) for value in concepts]
        if not self.concepts:
            raise ValueError("se necesita al menos un concepto")
        ids = [concept.concept_id for concept in self.concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("concept_id duplicado")
        self.terms: list[str] = []
        self.term_concept_indices: list[int] = []
        for index, concept in enumerate(self.concepts):
            for term in (concept.preferred_term, *concept.variants):
                if term.strip():
                    self.terms.append(term)
                    self.term_concept_indices.append(index)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            strip_accents="unicode",
        )
        self.term_matrix = self.vectorizer.fit_transform(self.terms)
        self.reranker: LogisticRegression | None = None

    def _candidates(self, mention: str) -> list[dict[str, Any]]:
        query = self.vectorizer.transform([mention])
        term_scores = cosine_similarity(query, self.term_matrix)[0]
        best_by_concept: dict[int, tuple[float, str]] = {}
        for term_index, score in enumerate(term_scores):
            concept_index = self.term_concept_indices[term_index]
            previous = best_by_concept.get(concept_index)
            candidate = (float(score), self.terms[term_index])
            if previous is None or candidate[0] > previous[0]:
                best_by_concept[concept_index] = candidate
        rows = []
        for concept_index, (score, matched_term) in best_by_concept.items():
            concept = self.concepts[concept_index]
            rows.append(
                {
                    "concept_id": concept.concept_id,
                    "preferred_term": concept.preferred_term,
                    "semantic_type": concept.semantic_type,
                    "matched_term": matched_term,
                    "lexical_score": score,
                    "exact_folded": _fold(mention) == _fold(matched_term),
                    "length_ratio": min(len(mention), len(matched_term))
                    / max(len(mention), len(matched_term), 1),
                }
            )
        return sorted(rows, key=lambda row: (-row["lexical_score"], row["concept_id"]))

    @staticmethod
    def _features(row: Mapping[str, Any]) -> list[float]:
        return [
            float(row["lexical_score"]),
            float(bool(row["exact_folded"])),
            float(row["length_ratio"]),
        ]

    def fit_reranker(
        self,
        training_pairs: Sequence[tuple[str, str]],
        *,
        negatives_per_mention: int = 3,
        seed: int = 17,
    ) -> "ConceptNormalizer":
        """Aprende match/no-match a partir de menciones normalizadas."""

        known_ids = {concept.concept_id for concept in self.concepts}
        features: list[list[float]] = []
        labels: list[int] = []
        for mention, gold_id in training_pairs:
            if gold_id not in known_ids:
                raise ValueError(f"concepto gold desconocido: {gold_id}")
            ranked = self._candidates(mention)
            positives = [row for row in ranked if row["concept_id"] == gold_id]
            if not positives:
                raise RuntimeError("el generador no devolvió el concepto gold")
            features.append(self._features(positives[0]))
            labels.append(1)
            for row in [item for item in ranked if item["concept_id"] != gold_id][
                :negatives_per_mention
            ]:
                features.append(self._features(row))
                labels.append(0)
        if len(set(labels)) < 2:
            raise ValueError("se necesitan conceptos alternativos para crear negativos")
        self.reranker = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=seed,
            solver="liblinear",
        ).fit(features, labels)
        return self

    def rank(self, mention: str, *, k: int = 5) -> list[dict[str, Any]]:
        if not mention.strip():
            return []
        rows = self._candidates(mention)
        for row in rows:
            row["score"] = (
                float(self.reranker.predict_proba([self._features(row)])[0, 1])
                if self.reranker is not None
                else float(row["lexical_score"])
            )
        return sorted(rows, key=lambda row: (-row["score"], row["concept_id"]))[:k]

    def normalize(
        self,
        mention: str,
        *,
        threshold: float = 0.55,
        min_margin: float = 0.05,
    ) -> dict[str, Any]:
        ranked = self.rank(mention, k=2)
        if not ranked:
            return {"mention": mention, "concept_id": None, "status": "abstain_empty"}
        best = ranked[0]
        runner_up = ranked[1]["score"] if len(ranked) > 1 else 0.0
        margin = float(best["score"] - runner_up)
        accepted = best["score"] >= threshold and margin >= min_margin
        return {
            "mention": mention,
            "concept_id": best["concept_id"] if accepted else None,
            "preferred_term": best["preferred_term"] if accepted else None,
            "status": "linked" if accepted else "abstain_low_confidence",
            "score": float(best["score"]),
            "margin": margin,
            "candidates": ranked,
        }
