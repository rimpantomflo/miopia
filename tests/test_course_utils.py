import math
import unittest

from clinical_nlp_course import (
    TfidfRetriever,
    annotation_edit_stats,
    brier_score,
    cluster_bootstrap_binary_metric,
    dictionary_suggestions,
    exact_span_metrics,
    expected_calibration_error,
    make_doccano_record,
    overlap_span_metrics,
    patient_hash_split,
    validate_concept_dictionary,
    validate_doccano_labels,
    validate_llm_extraction,
)


class SpanMetricTests(unittest.TestCase):
    def test_exact_and_overlap_differ(self):
        gold = {"C1": [(0, 20, "DISEASE")]}
        pred = {"C1": [(0, 10, "DISEASE")]}
        self.assertEqual(exact_span_metrics(gold, pred)["tp"], 0)
        self.assertEqual(overlap_span_metrics(gold, pred)["tp"], 1)

    def test_overlap_requires_same_label(self):
        gold = {"C1": [(0, 10, "DISEASE")]}
        pred = {"C1": [(0, 10, "DRUG")]}
        self.assertEqual(overlap_span_metrics(gold, pred)["tp"], 0)

    def test_complete_disagreement_has_zero_f1(self):
        gold = {"C1": [(0, 4, "DISEASE")]}
        pred = {"C1": [(5, 9, "DISEASE")]}
        self.assertEqual(exact_span_metrics(gold, pred)["f1"], 0.0)


class CorpusUtilityTests(unittest.TestCase):
    def test_patient_split_is_stable(self):
        self.assertEqual(patient_hash_split("P001"), patient_hash_split("P001"))

    def test_dictionary_detects_ambiguous_variant(self):
        concepts = [
            {
                "concept_id": "A",
                "preferred_term": "renal",
                "variants": ["riñón"],
                "semantic_type": "X",
                "version": "1",
            },
            {
                "concept_id": "B",
                "preferred_term": "rinon",
                "variants": ["kidney"],
                "semantic_type": "Y",
                "version": "1",
            },
        ]
        self.assertTrue(validate_concept_dictionary(concepts))


class RetrievalAndLLMTests(unittest.TestCase):
    def test_retrieval(self):
        retriever = TfidfRetriever().fit(
            [
                {"id": "1", "text": "hemodiálisis mediante fístula"},
                {"id": "2", "text": "trasplante renal funcionante"},
            ]
        )
        self.assertEqual(retriever.rank("fístula para diálisis", k=1)[0]["id"], "1")

    def test_llm_evidence_must_match_offsets(self):
        text = "Paciente en hemodiálisis."
        good = {
            "concept": "HEMODIALYSIS",
            "assertion": "affirmed",
            "evidence": "hemodiálisis",
            "start": 12,
            "end": 24,
        }
        self.assertEqual(validate_llm_extraction(good, text), [])
        bad = {**good, "start": 0}
        self.assertTrue(validate_llm_extraction(bad, text))


class ValidationTests(unittest.TestCase):
    def test_brier_score(self):
        self.assertAlmostEqual(brier_score([1, 0], [1.0, 0.0]), 0.0)

    def test_calibration_error(self):
        self.assertAlmostEqual(
            expected_calibration_error([1, 0], [1.0, 0.0], n_bins=2),
            0.0,
        )

    def test_cluster_bootstrap(self):
        rows = [
            {"patient": "A", "truth": 1, "pred": 1},
            {"patient": "A", "truth": 1, "pred": 0},
            {"patient": "B", "truth": 0, "pred": 0},
            {"patient": "C", "truth": 1, "pred": 1},
        ]
        result = cluster_bootstrap_binary_metric(
            rows,
            group_key="patient",
            truth_key="truth",
            prediction_key="pred",
            n_resamples=200,
        )
        self.assertFalse(math.isnan(result["estimate"]))
        self.assertLessEqual(result["lower"], result["upper"])


class DoccanoAdapterTests(unittest.TestCase):
    def setUp(self):
        self.concepts = [
            {
                "concept_id": "HEMODIALYSIS",
                "preferred_term": "hemodiálisis",
                "variants": ["HD"],
                "exclusions": ["opciones de hemodiálisis"],
            }
        ]

    def test_dictionary_suggestions_keep_offsets(self):
        text = "Continúa HD."
        suggestions = dictionary_suggestions(text, self.concepts)
        self.assertEqual(suggestions[0]["evidence"], "HD")
        self.assertEqual(text[suggestions[0]["start"] : suggestions[0]["end"]], "HD")

    def test_dictionary_suggestions_respect_exclusions(self):
        text = "Se explican opciones de hemodiálisis."
        self.assertEqual(dictionary_suggestions(text, self.concepts), [])

    def test_record_and_validation(self):
        text = "Continúa en hemodiálisis."
        suggestions = dictionary_suggestions(text, self.concepts)
        record = make_doccano_record(
            document_id="SYN-1",
            text=text,
            suggestions=suggestions,
            suggestion_version="dict-v1",
        )
        self.assertEqual(validate_doccano_labels(text, record["labels"]), [])
        self.assertTrue(record["meta"]["preannotated"])

    def test_validation_detects_overlap(self):
        problems = validate_doccano_labels(
            "hemodiálisis",
            [[0, 12, "A"], [0, 5, "B"]],
        )
        self.assertTrue(any("solapamiento" in problem for problem in problems))

    def test_edit_stats(self):
        stats = annotation_edit_stats(
            [[0, 2, "A"], [4, 6, "B"]],
            [[0, 2, "A"], [7, 9, "C"]],
        )
        self.assertEqual(stats["accepted"], 1)
        self.assertEqual(stats["removed"], 1)
        self.assertEqual(stats["added"], 1)


if __name__ == "__main__":
    unittest.main()
