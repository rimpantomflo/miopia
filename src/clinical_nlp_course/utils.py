"""Funciones pequeñas, legibles y probadas para los notebooks avanzados."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import random
import re
import unicodedata
from typing import Any, Callable, Iterable, Mapping, Sequence


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        _safe_divide(2 * precision * recall, precision + recall)
        if not (math.isnan(precision) or math.isnan(recall))
        else math.nan
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _span_tuple(span: Mapping[str, Any] | Sequence[Any]) -> tuple[int, int, str]:
    if isinstance(span, Mapping):
        return (int(span["start"]), int(span["end"]), str(span["label"]))
    if len(span) != 3:
        raise ValueError("Cada span debe contener start, end y label")
    return (int(span[0]), int(span[1]), str(span[2]))


def exact_span_metrics(
    gold_by_doc: Mapping[str, Iterable[Mapping[str, Any] | Sequence[Any]]],
    pred_by_doc: Mapping[str, Iterable[Mapping[str, Any] | Sequence[Any]]],
) -> dict[str, float | int]:
    """Micro P/R/F1: etiqueta y ambos offsets deben coincidir."""
    tp = fp = fn = 0
    for doc_id in set(gold_by_doc) | set(pred_by_doc):
        gold = Counter(_span_tuple(span) for span in gold_by_doc.get(doc_id, []))
        pred = Counter(_span_tuple(span) for span in pred_by_doc.get(doc_id, []))
        matches = gold & pred
        n_matches = sum(matches.values())
        tp += n_matches
        fp += sum(pred.values()) - n_matches
        fn += sum(gold.values()) - n_matches
    return _prf(tp, fp, fn)


def _iou(first: tuple[int, int, str], second: tuple[int, int, str]) -> float:
    if first[2] != second[2]:
        return 0.0
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def overlap_span_metrics(
    gold_by_doc: Mapping[str, Iterable[Mapping[str, Any] | Sequence[Any]]],
    pred_by_doc: Mapping[str, Iterable[Mapping[str, Any] | Sequence[Any]]],
    *,
    min_iou: float = 0.01,
) -> dict[str, float | int]:
    """Micro P/R/F1 con emparejamiento uno-a-uno por solapamiento e igual etiqueta."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou debe estar en (0, 1]")

    tp = fp = fn = 0
    for doc_id in set(gold_by_doc) | set(pred_by_doc):
        gold = [_span_tuple(span) for span in gold_by_doc.get(doc_id, [])]
        pred = [_span_tuple(span) for span in pred_by_doc.get(doc_id, [])]
        candidates = sorted(
            (
                (_iou(gold_span, pred_span), gold_index, pred_index)
                for gold_index, gold_span in enumerate(gold)
                for pred_index, pred_span in enumerate(pred)
            ),
            reverse=True,
        )
        used_gold: set[int] = set()
        used_pred: set[int] = set()
        for score, gold_index, pred_index in candidates:
            if score < min_iou:
                break
            if gold_index in used_gold or pred_index in used_pred:
                continue
            used_gold.add(gold_index)
            used_pred.add(pred_index)
        tp += len(used_gold)
        fp += len(pred) - len(used_pred)
        fn += len(gold) - len(used_gold)
    return _prf(tp, fp, fn)


def patient_hash_split(
    patient_id: str,
    *,
    seed: str = "clinical-nlp-course-v1",
    train_pct: int = 70,
    development_pct: int = 15,
) -> str:
    """Partición estable que mantiene todos los documentos del paciente juntos."""
    if train_pct <= 0 or development_pct < 0 or train_pct + development_pct >= 100:
        raise ValueError("Porcentajes de partición inválidos")
    digest = hashlib.sha256(f"{seed}|{patient_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + development_pct:
        return "development"
    return "test"


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(
        "".join(char for char in token if not unicodedata.combining(char))
        for token in decomposed.split()
    )


def validate_concept_dictionary(concepts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Devuelve problemas reproducibles de un diccionario clínico."""
    issues: list[str] = []
    required = {"concept_id", "preferred_term", "variants", "semantic_type", "version"}
    seen_ids: set[str] = set()
    variant_owner: dict[str, str] = {}

    for index, concept in enumerate(concepts):
        missing = required - set(concept)
        if missing:
            issues.append(f"concept[{index}] carece de: {sorted(missing)}")
            continue
        concept_id = str(concept["concept_id"]).strip()
        if not concept_id:
            issues.append(f"concept[{index}] tiene concept_id vacío")
        elif concept_id in seen_ids:
            issues.append(f"concept_id duplicado: {concept_id}")
        seen_ids.add(concept_id)

        variants = concept["variants"]
        if not isinstance(variants, list) or not variants:
            issues.append(f"{concept_id}: variants debe ser una lista no vacía")
            continue
        all_terms = [str(concept["preferred_term"]), *map(str, variants)]
        for term in all_terms:
            normalized = _fold(term)
            if not normalized:
                issues.append(f"{concept_id}: término vacío")
                continue
            previous = variant_owner.get(normalized)
            if previous and previous != concept_id:
                issues.append(
                    f"variante ambigua {term!r}: pertenece a {previous} y {concept_id}"
                )
            variant_owner[normalized] = concept_id
    return issues


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b[\wáéíóúüñç]+\b", _fold(text), flags=re.UNICODE)


class TfidfRetriever:
    """Recuperador TF-IDF mínimo para aprender el mecanismo de RAG."""

    def __init__(self) -> None:
        self.documents: list[dict[str, str]] = []
        self.idf: dict[str, float] = {}
        self.vectors: list[dict[str, float]] = []

    def fit(self, documents: Sequence[Mapping[str, Any]]) -> "TfidfRetriever":
        self.documents = [
            {"id": str(document["id"]), "text": str(document["text"])}
            for document in documents
        ]
        if not self.documents:
            raise ValueError("Se necesita al menos un documento")
        document_frequency: Counter[str] = Counter()
        tokenized: list[list[str]] = []
        for document in self.documents:
            tokens = _tokens(document["text"])
            tokenized.append(tokens)
            document_frequency.update(set(tokens))
        n_documents = len(self.documents)
        self.idf = {
            term: math.log((1 + n_documents) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(tokens) for tokens in tokenized]
        return self

    def _vector(self, tokens: Sequence[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}
        raw = {
            term: (count / len(tokens)) * self.idf.get(term, 0.0)
            for term, count in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(value * value for value in raw.values()))
        return {term: value / norm for term, value in raw.items()} if norm else {}

    @staticmethod
    def _cosine(first: Mapping[str, float], second: Mapping[str, float]) -> float:
        shared = set(first) & set(second)
        return sum(first[term] * second[term] for term in shared)

    def rank(self, query: str, *, k: int = 3) -> list[dict[str, Any]]:
        if not self.documents:
            raise RuntimeError("Llama a fit antes de rank")
        query_vector = self._vector(_tokens(query))
        scored = [
            {
                **document,
                "score": self._cosine(query_vector, vector),
            }
            for document, vector in zip(self.documents, self.vectors)
        ]
        return sorted(scored, key=lambda item: (-item["score"], item["id"]))[:k]


def validate_llm_extraction(
    record: Mapping[str, Any],
    source_text: str,
    *,
    allowed_assertions: Sequence[str] = ("affirmed", "negated", "possible"),
) -> list[str]:
    """Valida estructura y anclaje de una extracción generativa."""
    issues: list[str] = []
    required = {"concept", "assertion", "evidence", "start", "end"}
    missing = required - set(record)
    if missing:
        return [f"faltan campos: {sorted(missing)}"]
    if record["assertion"] not in allowed_assertions:
        issues.append(f"assertion no permitida: {record['assertion']!r}")
    try:
        start = int(record["start"])
        end = int(record["end"])
    except (TypeError, ValueError):
        issues.append("start/end deben ser enteros")
        return issues
    if not (0 <= start < end <= len(source_text)):
        issues.append("offsets fuera del texto")
        return issues
    recovered = source_text[start:end]
    if recovered != str(record["evidence"]):
        issues.append(
            f"evidence no coincide con source_text[start:end]: {recovered!r}"
        )
    if not str(record["concept"]).strip():
        issues.append("concept vacío")
    return issues


def _binary_counts(
    y_true: Sequence[bool | int],
    y_pred: Sequence[bool | int],
) -> tuple[int, int, int, int]:
    tp = sum(bool(t) and bool(p) for t, p in zip(y_true, y_pred))
    tn = sum(not bool(t) and not bool(p) for t, p in zip(y_true, y_pred))
    fp = sum(not bool(t) and bool(p) for t, p in zip(y_true, y_pred))
    fn = sum(bool(t) and not bool(p) for t, p in zip(y_true, y_pred))
    return tp, tn, fp, fn


def _binary_metric(
    y_true: Sequence[bool | int],
    y_pred: Sequence[bool | int],
    metric: str,
) -> float:
    tp, tn, fp, fn = _binary_counts(y_true, y_pred)
    functions = {
        "sensitivity": lambda: _safe_divide(tp, tp + fn),
        "specificity": lambda: _safe_divide(tn, tn + fp),
        "ppv": lambda: _safe_divide(tp, tp + fp),
        "npv": lambda: _safe_divide(tn, tn + fn),
        "accuracy": lambda: _safe_divide(tp + tn, tp + tn + fp + fn),
    }
    if metric not in functions:
        raise ValueError(f"Métrica no soportada: {metric}")
    return functions[metric]()


def cluster_bootstrap_binary_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    truth_key: str,
    prediction_key: str,
    metric: str = "sensitivity",
    n_resamples: int = 1000,
    seed: int = 17,
) -> dict[str, float]:
    """IC percentil re-muestreando pacientes completos, no documentos sueltos."""
    if n_resamples < 100:
        raise ValueError("Usa al menos 100 remuestreos")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row[group_key])].append(row)
    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("Se necesitan al menos dos grupos")

    y_true = [row[truth_key] for row in rows]
    y_pred = [row[prediction_key] for row in rows]
    point = _binary_metric(y_true, y_pred, metric)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        chosen = [rng.choice(groups) for _ in groups]
        sampled_rows = [row for group in chosen for row in by_group[group]]
        value = _binary_metric(
            [row[truth_key] for row in sampled_rows],
            [row[prediction_key] for row in sampled_rows],
            metric,
        )
        if not math.isnan(value):
            samples.append(value)
    if not samples:
        return {"estimate": point, "lower": math.nan, "upper": math.nan}
    samples.sort()

    def percentile(probability: float) -> float:
        index = round((len(samples) - 1) * probability)
        return samples[index]

    return {
        "estimate": point,
        "lower": percentile(0.025),
        "upper": percentile(0.975),
    }


def brier_score(y_true: Sequence[bool | int], probabilities: Sequence[float]) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("Entradas vacías o con longitudes diferentes")
    if any(not 0 <= probability <= 1 for probability in probabilities):
        raise ValueError("Las probabilidades deben estar entre 0 y 1")
    return sum(
        (float(bool(truth)) - probability) ** 2
        for truth, probability in zip(y_true, probabilities)
    ) / len(y_true)


def expected_calibration_error(
    y_true: Sequence[bool | int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 5,
) -> float:
    if len(y_true) != len(probabilities) or not y_true:
        raise ValueError("Entradas vacías o con longitudes diferentes")
    if n_bins < 2:
        raise ValueError("n_bins debe ser >= 2")
    bins: list[list[tuple[bool, float]]] = [[] for _ in range(n_bins)]
    for truth, probability in zip(y_true, probabilities):
        if not 0 <= probability <= 1:
            raise ValueError("Las probabilidades deben estar entre 0 y 1")
        index = min(int(probability * n_bins), n_bins - 1)
        bins[index].append((bool(truth), probability))
    total = len(y_true)
    error = 0.0
    for bin_rows in bins:
        if not bin_rows:
            continue
        observed = sum(truth for truth, _ in bin_rows) / len(bin_rows)
        predicted = sum(probability for _, probability in bin_rows) / len(bin_rows)
        error += (len(bin_rows) / total) * abs(observed - predicted)
    return error
