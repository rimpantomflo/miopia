"""Evaluación clínica avanzada: umbrales, utilidad, subgrupos e incertidumbre."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


def _validate_binary(y_true: Sequence[int | bool], scores: Sequence[float]) -> None:
    if len(y_true) != len(scores) or not y_true:
        raise ValueError("y_true y scores deben tener igual longitud y no estar vacíos")
    if any(score < 0 or score > 1 for score in scores):
        raise ValueError("scores debe estar en [0, 1]")


def threshold_metrics(
    y_true: Sequence[int | bool],
    scores: Sequence[float],
    *,
    threshold: float,
) -> dict[str, float | int]:
    _validate_binary(y_true, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar en [0, 1]")
    predicted = [score >= threshold for score in scores]
    tn, fp, fn, tp = confusion_matrix(
        [bool(value) for value in y_true], predicted, labels=[False, True]
    ).ravel()

    def divide(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else math.nan

    return {
        "threshold": threshold,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": divide(int(tp), int(tp + fn)),
        "specificity": divide(int(tn), int(tn + fp)),
        "ppv": divide(int(tp), int(tp + fp)),
        "npv": divide(int(tn), int(tn + fn)),
    }


def discrimination_report(
    y_true: Sequence[int | bool], scores: Sequence[float]
) -> dict[str, float]:
    _validate_binary(y_true, scores)
    truth = [bool(value) for value in y_true]
    if len(set(truth)) < 2:
        return {"roc_auc": math.nan, "average_precision": math.nan}
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "average_precision": float(average_precision_score(truth, scores)),
    }


def choose_threshold(
    y_true: Sequence[int | bool],
    scores: Sequence[float],
    *,
    minimum_sensitivity: float = 0.90,
) -> dict[str, float | int]:
    """Elige en development la mayor especificidad que cumple sensibilidad."""

    if not 0 <= minimum_sensitivity <= 1:
        raise ValueError("minimum_sensitivity debe estar en [0, 1]")
    candidates = sorted({0.0, 1.0, *map(float, scores)})
    reports = [
        threshold_metrics(y_true, scores, threshold=value) for value in candidates
    ]
    eligible = [
        report
        for report in reports
        if not math.isnan(float(report["sensitivity"]))
        and float(report["sensitivity"]) >= minimum_sensitivity
    ]
    if not eligible:
        raise ValueError("ningún umbral satisface la sensibilidad requerida")
    return max(
        eligible,
        key=lambda row: (float(row["specificity"]), float(row["threshold"])),
    )


def decision_curve(
    y_true: Sequence[int | bool],
    scores: Sequence[float],
    *,
    thresholds: Iterable[float],
) -> list[dict[str, float]]:
    """Net benefit del modelo frente a tratar a todos o a nadie."""

    _validate_binary(y_true, scores)
    n = len(y_true)
    prevalence = sum(bool(value) for value in y_true) / n
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        if not 0 < threshold < 1:
            raise ValueError("decision curve requiere umbrales en (0, 1)")
        report = threshold_metrics(y_true, scores, threshold=threshold)
        odds = threshold / (1 - threshold)
        model = (float(report["tp"]) / n) - (float(report["fp"]) / n) * odds
        treat_all = prevalence - (1 - prevalence) * odds
        rows.append(
            {
                "threshold": float(threshold),
                "model": model,
                "treat_all": treat_all,
                "treat_none": 0.0,
            }
        )
    return rows


def subgroup_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_key: str,
    truth_key: str,
    score_key: str,
    threshold: float,
    minimum_size: int = 2,
) -> list[dict[str, Any]]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row[group_key])].append(row)
    report: list[dict[str, Any]] = []
    for group, group_rows in sorted(by_group.items()):
        result: dict[str, Any] = {"group": group, "n": len(group_rows)}
        if len(group_rows) < minimum_size:
            result["status"] = "insufficient_sample"
        else:
            result.update(
                threshold_metrics(
                    [row[truth_key] for row in group_rows],
                    [float(row[score_key]) for row in group_rows],
                    threshold=threshold,
                )
            )
            result["status"] = "estimated"
        report.append(result)
    return report


def paired_cluster_bootstrap_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    truth_key: str,
    score_a_key: str,
    score_b_key: str,
    threshold: float = 0.5,
    metric: str = "sensitivity",
    n_resamples: int = 1_000,
    seed: int = 17,
) -> dict[str, float]:
    """IC de diferencia B-A re-muestreando pacientes completos y emparejados."""

    if n_resamples < 100:
        raise ValueError("usa al menos 100 remuestreos")
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row[group_key])].append(row)
    groups = sorted(by_group)
    if len(groups) < 2:
        raise ValueError("se necesitan al menos dos pacientes")

    def difference(sample: Sequence[Mapping[str, Any]]) -> float:
        truth = [row[truth_key] for row in sample]
        a = threshold_metrics(
            truth, [float(row[score_a_key]) for row in sample], threshold=threshold
        )
        b = threshold_metrics(
            truth, [float(row[score_b_key]) for row in sample], threshold=threshold
        )
        if metric not in a:
            raise ValueError(f"métrica no soportada: {metric}")
        return float(b[metric]) - float(a[metric])

    point = difference(rows)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        sampled_groups = [rng.choice(groups) for _ in groups]
        sampled_rows = [row for group in sampled_groups for row in by_group[group]]
        value = difference(sampled_rows)
        if not math.isnan(value):
            samples.append(value)
    if not samples:
        return {"estimate": point, "lower": math.nan, "upper": math.nan}
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"estimate": point, "lower": float(lower), "upper": float(upper)}
