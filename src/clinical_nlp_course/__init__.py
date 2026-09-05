"""Utilidades genéricas del curso de NLP clínico."""

from .assessment import (
    ASSESSMENT_RUBRIC,
    REQUIRED_CAPSTONE_EVIDENCE,
    rubric_total,
    score_assessment,
    validate_capstone_evidence,
)
from .classical import (
    ClassificationResult,
    build_tfidf_classifier,
    evaluate_classifier,
    predict_with_probabilities,
    top_linear_features,
)
from .context import DEFAULT_TRIGGERS, Trigger, annotate_context
from .doccano import (
    annotation_edit_stats,
    dictionary_suggestions,
    make_doccano_record,
    validate_doccano_labels,
)
from .evaluation import (
    choose_threshold,
    decision_curve,
    discrimination_report,
    paired_cluster_bootstrap_difference,
    subgroup_report,
    threshold_metrics,
)
from .experiment import build_run_manifest, git_revision, sha256_file
from .explainability import leave_one_segment_out, linear_feature_contributions
from .llm import (
    ClinicalExtraction,
    ExtractionBatch,
    OllamaBackend,
    RuleBasedDemoBackend,
    extract_structured,
    validate_batch,
)
from .monitoring import (
    InputContract,
    canonical_sha256,
    operation_key,
    population_stability_index,
    safe_batch_event,
)
from .normalization import Concept, ConceptNormalizer
from .public_data import PUBLIC_CORPORA, CorpusDescriptor, load_brat, load_entity_tsv
from .relations import (
    Entity,
    RelationCandidate,
    RelationClassifier,
    candidate_features,
    relation_candidates,
)
from .retrieval import HybridRetriever, retrieval_metrics, sentence_transformer_encoder
from .synthetic import generate_renal_classification_rows
from .transformers import (
    IGNORE_INDEX,
    align_word_labels,
    assert_no_patient_leakage,
    bio_to_char_spans,
    char_spans_to_bio,
    read_token_classification_jsonl,
)
from .utils import (
    TfidfRetriever,
    brier_score,
    cluster_bootstrap_binary_metric,
    exact_span_metrics,
    expected_calibration_error,
    overlap_span_metrics,
    patient_hash_split,
    validate_concept_dictionary,
    validate_llm_extraction,
)

__all__ = [
    "ASSESSMENT_RUBRIC",
    "DEFAULT_TRIGGERS",
    "IGNORE_INDEX",
    "PUBLIC_CORPORA",
    "ClassificationResult",
    "ClinicalExtraction",
    "Concept",
    "ConceptNormalizer",
    "CorpusDescriptor",
    "Entity",
    "ExtractionBatch",
    "HybridRetriever",
    "InputContract",
    "OllamaBackend",
    "RelationCandidate",
    "RelationClassifier",
    "REQUIRED_CAPSTONE_EVIDENCE",
    "RuleBasedDemoBackend",
    "TfidfRetriever",
    "Trigger",
    "annotation_edit_stats",
    "align_word_labels",
    "annotate_context",
    "assert_no_patient_leakage",
    "bio_to_char_spans",
    "brier_score",
    "build_run_manifest",
    "build_tfidf_classifier",
    "candidate_features",
    "canonical_sha256",
    "char_spans_to_bio",
    "choose_threshold",
    "cluster_bootstrap_binary_metric",
    "decision_curve",
    "dictionary_suggestions",
    "discrimination_report",
    "evaluate_classifier",
    "exact_span_metrics",
    "expected_calibration_error",
    "extract_structured",
    "git_revision",
    "generate_renal_classification_rows",
    "leave_one_segment_out",
    "linear_feature_contributions",
    "load_brat",
    "load_entity_tsv",
    "make_doccano_record",
    "operation_key",
    "overlap_span_metrics",
    "paired_cluster_bootstrap_difference",
    "patient_hash_split",
    "population_stability_index",
    "predict_with_probabilities",
    "relation_candidates",
    "read_token_classification_jsonl",
    "retrieval_metrics",
    "rubric_total",
    "safe_batch_event",
    "sentence_transformer_encoder",
    "score_assessment",
    "sha256_file",
    "subgroup_report",
    "threshold_metrics",
    "top_linear_features",
    "validate_batch",
    "validate_capstone_evidence",
    "validate_concept_dictionary",
    "validate_doccano_labels",
    "validate_llm_extraction",
]
