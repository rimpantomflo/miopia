from __future__ import annotations

import math

import pytest

from clinical_nlp_course import (
    ConceptNormalizer,
    Entity,
    HybridRetriever,
    RelationClassifier,
    RuleBasedDemoBackend,
    align_word_labels,
    annotate_context,
    bio_to_char_spans,
    build_tfidf_classifier,
    char_spans_to_bio,
    choose_threshold,
    decision_curve,
    discrimination_report,
    evaluate_classifier,
    extract_structured,
    leave_one_segment_out,
    linear_feature_contributions,
    paired_cluster_bootstrap_difference,
    predict_with_probabilities,
    relation_candidates,
    retrieval_metrics,
    subgroup_report,
    top_linear_features,
)


def test_classical_classifier_probabilities_and_explanation() -> None:
    texts = [
        "continúa hemodiálisis por fístula",
        "sesión de HD completada",
        "sin necesidad de diálisis",
        "manejo conservador sin TRS",
        "hemodiálisis crónica",
        "no precisa hemodiálisis",
    ]
    labels = ["HD", "HD", "NO", "NO", "HD", "NO"]
    model = build_tfidf_classifier().fit(texts, labels)
    result = predict_with_probabilities(model, ["continúa HD"])
    assert result.labels == ["HD"]
    assert sum(result.probabilities[0].values()) == pytest.approx(1.0)
    assert evaluate_classifier(model, texts, labels)["accuracy"] >= 0.8
    assert top_linear_features(model, n=3)
    explanation = linear_feature_contributions(model, "continúa hemodiálisis", top_n=3)
    assert explanation["target_label"] == "HD"
    assert explanation["positive"]


def test_context_priorities_scope_and_experiencer() -> None:
    text = (
        "No se descarta miopía. Sin retinopatía; actualmente miopía estable. "
        "Antecedentes familiares de glaucoma."
    )
    terms = ["miopía", "retinopatía", "miopía", "glaucoma"]
    mentions = []
    cursor = 0
    for term in terms:
        start = text.index(term, cursor)
        mentions.append({"start": start, "end": start + len(term), "label": "FINDING"})
        cursor = start + len(term)
    rows = annotate_context(text, mentions)
    assert [row["assertion"] for row in rows] == [
        "possible",
        "negated",
        "affirmed",
        "affirmed",
    ]
    assert rows[-1]["experiencer"] == "family"
    assert rows[2]["context_triggers"] == []


def test_transformer_alignment_and_decoding() -> None:
    offsets = [(0, 8), (9, 14), (15, 20)]
    labels = char_spans_to_bio(offsets, [(0, 14, "PROBLEM")])
    assert labels == ["B-PROBLEM", "I-PROBLEM", "O"]
    assert bio_to_char_spans(labels, offsets) == [
        {"start": 0, "end": 14, "label": "PROBLEM"}
    ]
    assert align_word_labels(
        [None, 0, 0, 1, None], [1, 3], b_to_i={1: 2}, label_all_subtokens=True
    ) == [-100, 1, 2, 3, -100]
    with pytest.raises(ValueError, match="solapados"):
        char_spans_to_bio([(0, 5)], [(0, 5, "A"), (1, 4, "B")])


def test_relation_candidates_and_classifier() -> None:
    positive_texts = [
        "Hemodiálisis mediante FAV.",
        "Diálisis mediante catéter.",
        "Tratamiento con tacrolimus.",
        "Biopsia compatible con nefropatía.",
    ]
    negative_texts = [
        "Hemodiálisis. Ayer se revisó la FAV de otro episodio.",
        "Diálisis; sin embargo el catéter fue retirado.",
        "Tratamiento. No consta relación con tacrolimus.",
        "Biopsia. La nefropatía se descartó después.",
    ]

    def candidate(text: str):
        first_end = text.find(" ")
        second_start = text.rfind(" ") + 1
        entities = [
            Entity("E1", 0, first_end, "TREATMENT", text[:first_end]),
            Entity("E2", second_start, len(text) - 1, "TARGET", text[second_start:-1]),
        ]
        return relation_candidates(
            text,
            entities,
            allowed_type_pairs={("TREATMENT", "TARGET")},
        )[0]

    candidates = [candidate(text) for text in positive_texts + negative_texts]
    classifier = RelationClassifier().fit(
        candidates, ["RELATED"] * 4 + ["NO_RELATION"] * 4
    )
    records = classifier.predict_records(candidates)
    assert len(records) == 8
    assert set(records[0]["probabilities"]) == {"RELATED", "NO_RELATION"}


def test_normalizer_link_and_abstain() -> None:
    normalizer = ConceptNormalizer(
        [
            {
                "concept_id": "HD",
                "preferred_term": "hemodiálisis",
                "variants": ["HD"],
                "semantic_type": "procedure",
            },
            {
                "concept_id": "DP",
                "preferred_term": "diálisis peritoneal",
                "variants": ["DPA"],
                "semantic_type": "procedure",
            },
            {
                "concept_id": "TX",
                "preferred_term": "trasplante renal",
                "variants": ["injerto renal"],
                "semantic_type": "procedure",
            },
        ]
    )
    normalizer.fit_reranker(
        [("hemodialisis", "HD"), ("DPA", "DP"), ("injerto renal", "TX")]
    )
    assert normalizer.normalize("hemodialisis", threshold=0.3)["concept_id"] == "HD"
    assert normalizer.normalize("", threshold=0.3)["status"] == "abstain_empty"
    uncertain = normalizer.normalize("diálisis", threshold=0.99)
    assert uncertain["concept_id"] is None


def test_hybrid_retrieval_permissions_and_metrics() -> None:
    retriever = HybridRetriever().fit(
        [
            {
                "id": "hd",
                "text": "hemodiálisis mediante fístula",
                "access_scope": "renal",
            },
            {
                "id": "dp",
                "text": "diálisis peritoneal mediante Tenckhoff",
                "access_scope": "renal",
            },
            {"id": "eye", "text": "refracción ocular", "access_scope": "ophthalmology"},
        ]
    )
    ranked = retriever.rank("acceso para hemodiálisis", allowed_scopes={"renal"})
    assert ranked[0]["id"] == "hd"
    assert all(row["access_scope"] == "renal" for row in ranked)
    metrics = retrieval_metrics(
        {"q1": [row["id"] for row in ranked]}, {"q1": {"hd"}}, k=2
    )
    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 1.0


def test_structured_llm_contract_offline() -> None:
    text = "No precisa hemodiálisis. Mantiene diálisis peritoneal."
    backend = RuleBasedDemoBackend(
        {"HEMODIALYSIS": ["hemodiálisis"], "PERITONEAL": ["diálisis peritoneal"]}
    )
    batch = extract_structured(
        text,
        backend=backend,
        allowed_concepts=["HEMODIALYSIS", "PERITONEAL"],
    )
    assert [row.assertion for row in batch.extractions] == ["negated", "affirmed"]
    assert all(text[row.start : row.end] == row.evidence for row in batch.extractions)


def test_advanced_evaluation() -> None:
    truth = [1, 1, 1, 0, 0, 0]
    scores = [0.95, 0.8, 0.55, 0.4, 0.2, 0.1]
    assert discrimination_report(truth, scores)["roc_auc"] == 1.0
    selected = choose_threshold(truth, scores, minimum_sensitivity=2 / 3)
    assert selected["specificity"] == 1.0
    assert decision_curve(truth, scores, thresholds=[0.25, 0.5])
    rows = [
        {"site": "A", "truth": truth[index], "score": scores[index]}
        for index in range(3)
    ] + [
        {"site": "B", "truth": truth[index], "score": scores[index]}
        for index in range(3, 6)
    ]
    report = subgroup_report(
        rows,
        group_key="site",
        truth_key="truth",
        score_key="score",
        threshold=0.5,
    )
    assert len(report) == 2


def test_paired_cluster_bootstrap_and_perturbation() -> None:
    rows = [
        {"patient": "A", "truth": 1, "a": 0.4, "b": 0.9},
        {"patient": "B", "truth": 1, "a": 0.8, "b": 0.8},
        {"patient": "C", "truth": 0, "a": 0.2, "b": 0.1},
    ]
    difference = paired_cluster_bootstrap_difference(
        rows,
        group_key="patient",
        truth_key="truth",
        score_a_key="a",
        score_b_key="b",
        n_resamples=200,
    )
    assert difference["estimate"] >= 0
    assert not math.isnan(difference["lower"])
    text = "hemodiálisis estable"
    explanations = leave_one_segment_out(
        text,
        [(0, 13, "treatment")],
        score=lambda value: 1.0 if "hemodiálisis" in value else 0.0,
    )
    assert explanations[0]["delta"] == 1.0
