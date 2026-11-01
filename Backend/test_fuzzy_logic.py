"""
Unit tests for Fuzzy Logic Module (Mamdani FIS).
Verifies membership bounds, centroid calculations, rule firing, missing input handling,
and boundary safety conditions.
"""

import unittest
import numpy as np

from fuzzy_rules import (
    MEMBERSHIP_PARAMS,
    evaluate_term_membership,
    build_rule_base,
    trimf,
    trapmf,
)
from fuzzy_logic import (
    FuzzyInferenceSystem,
    get_fuzzy_inference_system,
    assess_speech_fuzzy,
)


class TestFuzzyLogic(unittest.TestCase):
    def setUp(self):
        self.fis = FuzzyInferenceSystem()

    def test_membership_functions_range(self):
        """Membership degrees must strictly remain within [0.0, 1.0]."""
        # Test across domains for all variables
        for var_name, var_cfg in MEMBERSHIP_PARAMS.items():
            d_min, d_max = var_cfg["domain"]
            test_points = np.linspace(d_min - 2.0, d_max + 2.0, 50)
            for term_name in var_cfg["terms"].keys():
                for pt in test_points:
                    deg = evaluate_term_membership(var_name, term_name, pt)
                    self.assertGreaterEqual(
                        deg,
                        0.0,
                        f"Membership < 0 for {var_name}:{term_name} at {pt}",
                    )
                    self.assertLessEqual(
                        deg,
                        1.0,
                        f"Membership > 1 for {var_name}:{term_name} at {pt}",
                    )

    def test_missing_and_corrupt_feature_fallback(self):
        """Gracefully handle missing, None, NaN, or infinite inputs without throwing exceptions."""
        corrupted_inputs = {
            "dl_probability": None,
            "avg_pause_duration": float("nan"),
            "intra_sentence_pauses": float("inf"),
            "phonation_rate": -1.0,
            "jitter": None,
            "shimmer": None,
        }
        assessment = self.fis.compute_assessment(corrupted_inputs)
        self.assertIn("fuzzy_score", assessment)
        self.assertGreaterEqual(assessment["fuzzy_score"], 0.0)
        self.assertLessEqual(assessment["fuzzy_score"], 100.0)
        self.assertIn(assessment["fuzzy_category"], ["Low", "Moderate", "High"])

    def test_high_concern_rule_activation(self):
        """High DL probability + frequent pauses + high jitter/shimmer must yield High concern."""
        inputs = {
            "dl_probability": 0.92,
            "avg_pause_duration": 0.85,
            "intra_sentence_pauses": 35.0,
            "phonation_rate": 0.65,
            "jitter": 7.5,
            "shimmer": 0.065,
        }
        res = self.fis.compute_assessment(inputs)
        self.assertEqual(res["fuzzy_category"], "High")
        self.assertGreaterEqual(res["fuzzy_score"], 65.0)
        self.assertTrue(any("Rule 1" in r for r in res["activated_rules"]))
        self.assertTrue(any("Rule 4" in r for r in res["activated_rules"]))

    def test_low_concern_rule_activation(self):
        """Low DL probability + few pauses + high phonation rate must yield Low concern."""
        inputs = {
            "dl_probability": 0.10,
            "avg_pause_duration": 0.15,
            "intra_sentence_pauses": 2.0,
            "phonation_rate": 0.98,
            "jitter": 3.2,
            "shimmer": 0.012,
        }
        res = self.fis.compute_assessment(inputs)
        self.assertEqual(res["fuzzy_category"], "Low")
        self.assertLess(res["fuzzy_score"], 35.0)
        self.assertTrue(any("Rule 3" in r for r in res["activated_rules"]))

    def test_moderate_discordance_handling(self):
        """When DL is low but acoustic biomarkers are severely disrupted, fuzzy system detects discordance."""
        inputs = {
            "dl_probability": 0.10,  # DL says normal
            "avg_pause_duration": 0.90,  # Abnormal pause length
            "intra_sentence_pauses": 40.0,  # Frequent hesitation
            "phonation_rate": 0.70,  # Low phonation
            "jitter": 7.0,  # High jitter
            "shimmer": 0.06,  # High shimmer
        }
        res = self.fis.compute_assessment(inputs)
        self.assertIn(res["fuzzy_category"], ["Moderate", "High"])
        self.assertIn("Discordance", res["agreement_analysis"]["status"])

    def test_all_rules_validity(self):
        """Ensure all 10 rules compile and can be evaluated on standard membership objects."""
        rules = build_rule_base()
        self.assertEqual(len(rules), 10)
        dummy_memberships = {
            "dl_probability": {"Low": 0.3, "Medium": 0.7, "High": 0.2},
            "avg_pause_duration": {"Short": 0.2, "Moderate": 0.6, "Long": 0.2},
            "intra_sentence_pauses": {"Few": 0.4, "Moderate": 0.5, "Frequent": 0.1},
            "phonation_rate": {"Low": 0.1, "Medium": 0.8, "High": 0.1},
            "jitter": {"Low": 0.5, "Medium": 0.4, "High": 0.1},
            "shimmer": {"Low": 0.6, "Medium": 0.3, "High": 0.1},
        }
        for rule in rules:
            strength = rule.evaluate(dummy_memberships)
            self.assertIsInstance(strength, float)
            self.assertGreaterEqual(strength, 0.0)
            self.assertLessEqual(strength, 1.0)

    def test_assess_speech_fuzzy_convenience_function(self):
        """Verify the assess_speech_fuzzy entry point."""
        res = assess_speech_fuzzy(
            dl_probability=0.72,
            avg_pause_duration=0.55,
            intra_sentence_pauses=20.0,
            phonation_rate=0.82,
            jitter=5.5,
            shimmer=0.035,
        )
        self.assertIn("fuzzy_score", res)
        self.assertIn("fuzzy_category", res)
        self.assertIn("feature_analysis", res)
        self.assertIn("activated_rules", res)
        self.assertIn("explanation", res)
        self.assertIn("assessment_type", res)


if __name__ == "__main__":
    unittest.main()
