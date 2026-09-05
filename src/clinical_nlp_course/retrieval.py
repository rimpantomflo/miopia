"""Recuperación híbrida trazable y evaluación para RAG clínico."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", _fold(text), flags=re.UNICODE)


class HybridRetriever:
    """Fusiona BM25 y similitud TF-IDF; admite un encoder denso local opcional."""

    def __init__(
        self,
        *,
        dense_encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.dense_encoder = dense_encoder
        self.rrf_k = rrf_k
        self.documents: list[dict[str, Any]] = []
        self.tokenized: list[list[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.average_length = 0.0
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), lowercase=True, strip_accents="unicode"
        )
        self.sparse_matrix: Any = None
        self.dense_matrix: np.ndarray | None = None

    def fit(self, documents: Iterable[Mapping[str, Any]]) -> "HybridRetriever":
        self.documents = [dict(row) for row in documents]
        if not self.documents:
            raise ValueError("se necesita al menos un documento")
        ids = [str(row["id"]) for row in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("los documentos necesitan id único")
        for row in self.documents:
            row["id"] = str(row["id"])
            row["text"] = str(row["text"])
        self.tokenized = [_tokens(row["text"]) for row in self.documents]
        self.document_frequency = Counter(
            token for tokens in self.tokenized for token in set(tokens)
        )
        self.average_length = sum(map(len, self.tokenized)) / len(self.tokenized)
        self.sparse_matrix = self.vectorizer.fit_transform(
            [row["text"] for row in self.documents]
        )
        if self.dense_encoder is not None:
            self.dense_matrix = np.asarray(
                self.dense_encoder([row["text"] for row in self.documents]),
                dtype=float,
            )
        return self

    def _bm25(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        scores = np.zeros(len(self.documents), dtype=float)
        query_terms = Counter(_tokens(query))
        n_documents = len(self.documents)
        for index, tokens in enumerate(self.tokenized):
            counts = Counter(tokens)
            length = len(tokens)
            for term, query_frequency in query_terms.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(term, 0)
                idf = math.log(
                    1
                    + (n_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * length / max(self.average_length, 1e-12)
                )
                scores[index] += (
                    query_frequency * idf * frequency * (k1 + 1) / denominator
                )
        return scores

    @staticmethod
    def _rank_positions(scores: np.ndarray, allowed: Sequence[int]) -> dict[int, int]:
        ordered = sorted(allowed, key=lambda index: (-float(scores[index]), index))
        return {
            document_index: rank for rank, document_index in enumerate(ordered, start=1)
        }

    def rank(
        self,
        query: str,
        *,
        k: int = 5,
        allowed_scopes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Filtra permisos antes de rankear y devuelve componentes de score."""

        if not self.documents:
            raise RuntimeError("llama a fit antes de rank")
        allowed = [
            index
            for index, row in enumerate(self.documents)
            if allowed_scopes is None
            or str(row.get("access_scope", "public")) in allowed_scopes
        ]
        if not allowed:
            return []
        bm25 = self._bm25(query)
        sparse = cosine_similarity(
            self.vectorizer.transform([query]), self.sparse_matrix
        )[0]
        components = {"bm25": bm25, "sparse": sparse}
        if self.dense_encoder is not None and self.dense_matrix is not None:
            query_dense = np.asarray(self.dense_encoder([query]), dtype=float)
            components["dense"] = cosine_similarity(query_dense, self.dense_matrix)[0]
        ranks = {
            name: self._rank_positions(scores, allowed)
            for name, scores in components.items()
        }
        rows: list[dict[str, Any]] = []
        for index in allowed:
            rrf = sum(
                1 / (self.rrf_k + positions[index]) for positions in ranks.values()
            )
            rows.append(
                {
                    **self.documents[index],
                    "score": float(rrf),
                    "scores": {
                        name: float(scores[index])
                        for name, scores in components.items()
                    },
                    "ranks": {
                        name: positions[index] for name, positions in ranks.items()
                    },
                }
            )
        return sorted(rows, key=lambda row: (-row["score"], row["id"]))[:k]


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, set[str]],
    *,
    k: int = 5,
) -> dict[str, float]:
    """Calcula Recall@k, MRR@k y nDCG@k sobre queries con gold."""

    query_ids = sorted(set(rankings) & set(relevant))
    if not query_ids:
        raise ValueError("no hay queries evaluables")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query_id in query_ids:
        gold = set(relevant[query_id])
        if not gold:
            continue
        predicted = list(rankings[query_id])[:k]
        hits = [1 if item in gold else 0 for item in predicted]
        recalls.append(len(set(predicted) & gold) / len(gold))
        first = next((index for index, hit in enumerate(hits, start=1) if hit), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
        dcg = sum(hit / math.log2(index + 1) for index, hit in enumerate(hits, start=1))
        ideal_hits = min(len(gold), k)
        ideal = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)
    if not recalls:
        raise ValueError("todas las queries tienen gold vacío")
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"mrr@{k}": float(np.mean(reciprocal_ranks)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "n_queries": float(len(recalls)),
    }


def sentence_transformer_encoder(
    model_name: str,
) -> Callable[[Sequence[str]], np.ndarray]:
    """Carga embeddings solo cuando se instala el extra ``embeddings``."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "instala embeddings con: uv sync --extra embeddings"
        ) from error
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True)
