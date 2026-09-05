import math
import unittest

from miopia_nlp import (
    aggregate_patient,
    binary_metrics,
    extract_mentions,
    parse_refractions,
    phenotype_course,
    pseudonymize_id,
)


class MentionTests(unittest.TestCase):
    def test_negation(self):
        mention = extract_mentions("No presenta miopía.")[0]
        self.assertEqual(mention["assertion"], "negated")

    def test_uncertainty_is_not_negation(self):
        mention = extract_mentions("No se descarta miopía.")[0]
        self.assertEqual(mention["assertion"], "possible")

    def test_family(self):
        mention = extract_mentions("Madre con miopía magna.")[0]
        self.assertEqual(mention["experiencer"], "family")

    def test_family_section_abbreviation(self):
        mention = extract_mentions("AF: miopía magna.")[0]
        self.assertEqual(mention["experiencer"], "family")

    def test_typo_whitelist(self):
        mention = extract_mentions("Antecedente de mipía.")[0]
        self.assertEqual(mention["rule_id"], "MYOPIA_TYPO_WHITELIST")

    def test_educational_context(self):
        mention = extract_mentions(
            "Se explica que la miopía aumenta el riesgo retiniano."
        )[0]
        self.assertEqual(mention["context"], "educational")

    def test_offsets_recover_original_text(self):
        text = "Paciente con alta miopía bilateral."
        mention = extract_mentions(text)[0]
        self.assertEqual(text[mention["start"] : mention["end"]], mention["text"])

    def test_semicolon_stops_family_scope(self):
        mentions = extract_mentions(
            "Niega antecedentes familiares de miopía; paciente con miopía axial."
        )
        self.assertEqual(mentions[0]["experiencer"], "family")
        self.assertEqual(mentions[1]["experiencer"], "patient")

    def test_question_mark_is_possible(self):
        mention = extract_mentions("¿Miopía? Pendiente de refracción.")[0]
        self.assertEqual(mention["assertion"], "possible")


class RefractionTests(unittest.TestCase):
    def test_decimal_comma_and_unicode_minus(self):
        measurements = parse_refractions("OD −8,00 D; OI −7,50 D.")
        self.assertEqual(len(measurements), 2)
        self.assertEqual(measurements[0]["spherical_equivalent_d"], -8.0)
        self.assertEqual(measurements[0]["refractive_class"], "high_myopia_numeric")

    def test_spherical_equivalent(self):
        measurement = parse_refractions("OD esf -3.00 cil -1.00 x 90")[0]
        self.assertEqual(measurement["spherical_equivalent_d"], -3.5)
        self.assertEqual(measurement["axis_deg"], 90)

    def test_requires_signed_value(self):
        self.assertEqual(parse_refractions("AV OD 0.8, OI 1.0"), [])


class PhenotypeTests(unittest.TestCase):
    def test_family_only_is_not_patient_case(self):
        result = phenotype_course("Madre con miopía.")
        self.assertFalse(result["ever_myopia"])

    def test_historical_counts_for_ever_not_current(self):
        result = phenotype_course("Intervenido mediante LASIK por miopía en 2018.")
        self.assertTrue(result["ever_myopia"])
        self.assertEqual(result["current_status"], "unknown")

    def test_longitudinal_latest_current_evidence(self):
        courses = [
            {"fecha": "2024-01-01", "texto": "Paciente miope."},
            {"fecha": "2025-01-01", "texto": "Actualmente no presenta miopía."},
        ]
        result = aggregate_patient(courses)
        self.assertTrue(result["ever_myopia"])
        self.assertEqual(result["current_status"], "negated")


class UtilityTests(unittest.TestCase):
    def test_metrics(self):
        result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(
            (result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1)
        )
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_undefined_metric_is_nan(self):
        result = binary_metrics([0, 0], [0, 0])
        self.assertTrue(math.isnan(result["sensitivity"]))

    def test_complete_disagreement_has_zero_f1(self):
        result = binary_metrics([1, 0], [0, 1])
        self.assertEqual(result["f1"], 0.0)

    def test_hmac_is_deterministic_and_keyed(self):
        first = pseudonymize_id("123", b"a sufficiently long secret")
        second = pseudonymize_id("123", b"a sufficiently long secret")
        other = pseudonymize_id("123", b"a different long secret")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
