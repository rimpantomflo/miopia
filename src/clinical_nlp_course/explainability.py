"""Explicaciones auditables para baselines lineales y pruebas de estabilidad."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.pipeline import Pipeline


def linear_feature_contributions(
    model: Pipeline,
    text: str,
    *,
    target_label: str | None = None,
    top_n: int = 15,
) -> dict[str, Any]:
    """Descompone el logit en contribuciones ``tfidf × coeficiente``."""

    if top_n < 1:
        raise ValueError("top_n debe ser positivo")
    features = model.named_steps["features"]
    classifier = model.named_steps["classifier"]
    vector = features.transform([text])
    classes = [str(label) for label in classifier.classes_]
    if target_label is None:
        target_label = str(model.predict([text])[0])
    if target_label not in classes:
        raise ValueError(f"target_label desconocida: {target_label}")

    if len(classes) == 2:
        weights = classifier.coef_[0]
        sign = 1.0 if target_label == classes[1] else -1.0
        intercept = float(classifier.intercept_[0]) * sign
        weights = weights * sign
    else:
        class_index = classes.index(target_label)
        weights = classifier.coef_[class_index]
        intercept = float(classifier.intercept_[class_index])

    names = features.get_feature_names_out()
    contributions = vector.multiply(weights).toarray()[0]
    nonzero = np.flatnonzero(contributions)
    positive = sorted(
        (
            (str(names[index]), float(contributions[index]))
            for index in nonzero
            if contributions[index] > 0
        ),
        key=lambda item: -item[1],
    )[:top_n]
    negative = sorted(
        (
            (str(names[index]), float(contributions[index]))
            for index in nonzero
            if contributions[index] < 0
        ),
        key=lambda item: item[1],
    )[:top_n]
    return {
        "target_label": target_label,
        "intercept": intercept,
        "logit": intercept + float(contributions.sum()),
        "positive": positive,
        "negative": negative,
    }


def leave_one_segment_out(
    text: str,
    segments: Sequence[tuple[int, int, str]],
    *,
    score: Callable[[str], float],
) -> list[dict[str, Any]]:
    """Perturba segmentos definidos, sin fingir causalidad clínica."""

    baseline = float(score(text))
    rows: list[dict[str, Any]] = []
    for start, end, name in segments:
        if not 0 <= start < end <= len(text):
            raise ValueError(f"segmento fuera de rango: {(start, end, name)}")
        perturbed = text[:start] + " " * (end - start) + text[end:]
        value = float(score(perturbed))
        rows.append(
            {
                "segment": name,
                "start": start,
                "end": end,
                "evidence": text[start:end],
                "baseline_score": baseline,
                "perturbed_score": value,
                "delta": baseline - value,
            }
        )
    return sorted(rows, key=lambda row: -abs(row["delta"]))
