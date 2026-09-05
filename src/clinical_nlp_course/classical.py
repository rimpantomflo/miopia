"""Baselines supervisados fuertes, rápidos y reproducibles para texto clínico."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_classifier(*, seed: int = 17) -> Pipeline:
    """Crea un baseline de palabras + caracteres resistente a abreviaturas.

    El modelo no se entrena aquí. Separar construcción y ajuste facilita guardar
    exactamente la misma configuración en un manifiesto de experimento.
    """

    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


@dataclass(frozen=True)
class ClassificationResult:
    labels: list[str]
    probabilities: list[dict[str, float]]


def predict_with_probabilities(
    model: Pipeline,
    texts: Iterable[str],
) -> ClassificationResult:
    """Predice etiquetas y conserva la probabilidad de todas las clases."""

    rows = [str(text) for text in texts]
    labels = [str(label) for label in model.predict(rows)]
    probabilities = model.predict_proba(rows)
    classes = [str(label) for label in model.classes_]
    return ClassificationResult(
        labels=labels,
        probabilities=[
            {label: float(value) for label, value in zip(classes, row, strict=True)}
            for row in probabilities
        ],
    )


def evaluate_classifier(
    model: Pipeline,
    texts: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Informe sin redondear, apto para comparar experimentos."""

    if len(texts) != len(labels) or not texts:
        raise ValueError("texts y labels deben tener igual longitud y no estar vacíos")
    predictions = model.predict(list(texts))
    report = classification_report(
        labels,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    return _native_numbers(report)


def top_linear_features(
    model: Pipeline, *, n: int = 10
) -> dict[str, list[tuple[str, float]]]:
    """Expone las señales de un clasificador binario o multiclase lineal."""

    if n < 1:
        raise ValueError("n debe ser positivo")
    features = model.named_steps["features"].get_feature_names_out()
    classifier = model.named_steps["classifier"]
    result: dict[str, list[tuple[str, float]]] = {}
    if len(classifier.classes_) == 2:
        weights_by_label = (
            (str(classifier.classes_[0]), -classifier.coef_[0]),
            (str(classifier.classes_[1]), classifier.coef_[0]),
        )
    else:
        weights_by_label = tuple(
            (str(classifier.classes_[index]), weights)
            for index, weights in enumerate(classifier.coef_)
        )
    for label, weights in weights_by_label:
        ordered = np.argsort(weights)[-n:][::-1]
        result[label] = [(str(features[i]), float(weights[i])) for i in ordered]
    return result


def _native_numbers(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native_numbers(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
