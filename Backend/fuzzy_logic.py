"""
Mamdani Fuzzy Inference System (FIS) for Speech-Pattern Assessment.
Integrates Deep Learning prediction probabilities with acoustic biomarkers
(pauses, phonation rate, jitter, shimmer) to provide explainable, uncertainty-aware scores.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from fuzzy_rules import (
    MEMBERSHIP_PARAMS,
    evaluate_term_membership,
    build_rule_base,
    FuzzyRule,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FuzzyInferenceSystem:
    """
    Modular Mamdani Fuzzy Inference Engine.
    Executes Fuzzification -> Rule Evaluation -> Max-Min Aggregation -> Centroid Defuzzification.
    """

    def __init__(self, resolution: int = 1001):
        self.resolution = resolution
        # Universe of discourse for output concern score [0, 100]
        self.output_universe = np.linspace(0.0, 100.0, resolution)
        self.rules = build_rule_base()

        # Precompute output term membership values along output universe for speed
        self.output_memberships = {
            "Low": np.array([
                evaluate_term_membership("concern_score", "Low", u)
                for u in self.output_universe
            ]),
            "Moderate": np.array([
                evaluate_term_membership("concern_score", "Moderate", u)
                for u in self.output_universe
            ]),
            "High": np.array([
                evaluate_term_membership("concern_score", "High", u)
                for u in self.output_universe
            ]),
        }

    def validate_inputs(self, raw_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Validates and sanitizes raw input features with safe fallback boundaries."""
        sanitized = {}
        defaults = {
            "dl_probability": 0.5,
            "avg_pause_duration": 0.38,
            "intra_sentence_pauses": 15.0,
            "phonation_rate": 0.86,
            "jitter": 5.25,
            "shimmer": 0.031,
        }

        for var_name, default_val in defaults.items():
            val = raw_inputs.get(var_name)
            if val is None or not np.isfinite(val):
                logger.warning(
                    "Feature '%s' is missing or non-finite (%s). Substituting empirical median %s.",
                    var_name,
                    val,
                    default_val,
                )
                sanitized[var_name] = float(default_val)
            else:
                var_domain = MEMBERSHIP_PARAMS[var_name]["domain"]
                # Clip gently to prevent extreme outlier domain overflow
                val_float = float(val)
                clamped = max(var_domain[0], min(var_domain[1] * 1.5, val_float))
                sanitized[var_name] = clamped

        return sanitized

    def fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Computes membership degrees for all variables across all linguistic terms.
        Returns: { variable: { term: degree ∈ [0, 1] } }
        """
        memberships = {}
        for var_name, var_cfg in MEMBERSHIP_PARAMS.items():
            if var_name == "concern_score":
                continue
            x_val = inputs[var_name]
            memberships[var_name] = {}
            for term_name in var_cfg["terms"].keys():
                deg = evaluate_term_membership(var_name, term_name, x_val)
                memberships[var_name][term_name] = round(float(deg), 4)
        return memberships

    def determine_dominant_categories(
        self, memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, str]:
        """Identifies the dominant linguistic term for each input variable."""
        dominant = {}
        for var_name, term_dict in memberships.items():
            best_term = max(term_dict.items(), key=lambda item: item[1])[0]
            dominant[var_name] = best_term
        return dominant

    def evaluate_rules(
        self, memberships: Dict[str, Dict[str, float]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Evaluates Mamdani rule base.
        Returns:
            - List of activated rule records
            - Aggregated firing strength per consequent term: {'Low': float, 'Moderate': float, 'High': float}
        """
        activated_rules = []
        consequent_strengths = {"Low": 0.0, "Moderate": 0.0, "High": 0.0}

        for rule in self.rules:
            strength = rule.evaluate(memberships)
            if strength > 0.001:
                activated_rules.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "firing_strength": round(strength, 4),
                    "consequent": rule.consequent,
                    "research_context": rule.research_context,
                })
                # Max aggregation for same consequent term
                consequent_strengths[rule.consequent] = max(
                    consequent_strengths[rule.consequent], strength
                )

        # Sort activated rules descending by firing strength
        activated_rules.sort(key=lambda r: r["firing_strength"], reverse=True)
        return activated_rules, consequent_strengths

    def defuzzify_centroid(
        self, consequent_strengths: Dict[str, float]
    ) -> Tuple[float, np.ndarray]:
        """
        Mamdani aggregation (max-min composition) and Centroid defuzzification.
        Returns:
            - continuous fuzzy score in [0.0, 100.0]
            - aggregated fuzzy curve across the universe
        """
        # Aggregate clipped memberships
        aggregated = np.zeros_like(self.output_universe)
        for term, strength in consequent_strengths.items():
            if strength > 0:
                clipped = np.minimum(strength, self.output_memberships[term])
                aggregated = np.maximum(aggregated, clipped)

        sum_aggregated = np.sum(aggregated)
        if sum_aggregated == 0.0:
            # Fallback if no rule fired
            return 50.0, aggregated

        # Centroid formula: Sum(u * mu(u)) / Sum(mu(u))
        centroid = np.sum(self.output_universe * aggregated) / sum_aggregated
        centroid_score = float(np.clip(centroid, 0.0, 100.0))
        return round(centroid_score, 2), aggregated

    def categorize_score(self, score: float) -> str:
        """Categorizes continuous score into Low / Moderate / High concern."""
        if score < 35.0:
            return "Low"
        elif score < 65.0:
            return "Moderate"
        else:
            return "High"

    def analyze_agreement(
        self, dl_prob: float, fuzzy_category: str, fuzzy_score: float
    ) -> Dict[str, Any]:
        """
        Assesses agreement / discordance between Deep Learning and Fuzzy components.
        Provides explainable diagnostic context and uncertainty quantification.
        """
        dl_category = "High" if dl_prob >= 0.65 else ("Moderate" if dl_prob >= 0.35 else "Low")

        if dl_category == fuzzy_category:
            status = "Strong Agreement"
            desc = (
                f"Both Deep Learning (Probability: {dl_prob:.2f}) and acoustic Fuzzy Inference "
                f"({fuzzy_category} concern, Score: {fuzzy_score:.1f}) converge on consistent speech patterns."
            )
            concordance = 1.0
        elif (dl_category in ["High", "Moderate"] and fuzzy_category in ["High", "Moderate"]) or (
            dl_category in ["Low", "Moderate"] and fuzzy_category in ["Low", "Moderate"]
        ):
            status = "Partial Agreement"
            desc = (
                f"Deep Learning ({dl_category} suspicion) and Fuzzy Inference ({fuzzy_category} concern) "
                f"show moderate alignment with slight nuance in acoustic biomarker support."
            )
            concordance = 0.65
        else:
            status = "Discordance Detected"
            desc = (
                f"Discrepancy identified between neural prediction ({dl_category}) and physical prosody "
                f"biomarkers ({fuzzy_category}). The Fuzzy Logic layer moderates the overall confidence "
                f"to prevent false reliance on deep feature extraction alone."
            )
            concordance = 0.30

        return {
            "status": status,
            "dl_category": dl_category,
            "fuzzy_category": fuzzy_category,
            "concordance_factor": concordance,
            "description": desc,
        }

    def generate_explanation(
        self,
        dominant_categories: Dict[str, str],
        activated_rules: List[Dict[str, Any]],
        agreement: Dict[str, Any],
        fuzzy_score: float,
        fuzzy_category: str,
    ) -> str:
        """Synthesizes an interpretable, transparent XAI rationale for the final score."""
        top_rules = activated_rules[:3]
        rule_texts = [f"{r['rule_id']} ({r['name']}, activation: {r['firing_strength']:.2f})" for r in top_rules]
        rules_summary = "; ".join(rule_texts) if rule_texts else "Baseline distribution rules"

        explanation = (
            f"Fuzzy assessment produced a score of {fuzzy_score:.1f}/100 ({fuzzy_category} concern). "
            f"Key acoustic findings: Pause duration is {dominant_categories['avg_pause_duration'].lower()}, "
            f"sentence pauses are {dominant_categories['intra_sentence_pauses'].lower()}, "
            f"and phonation rate is {dominant_categories['phonation_rate'].lower()}. "
            f"Primary active rules: {rules_summary}. "
            f"{agreement['description']} "
            f"This score reflects speech acoustic characteristics only and does not constitute a clinical medical diagnosis."
        )
        return explanation

    def compute_assessment(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes complete fuzzy inference pipeline from raw feature dictionary.
        """
        sanitized = self.validate_inputs(raw_inputs)
        memberships = self.fuzzify(sanitized)
        dominant_categories = self.determine_dominant_categories(memberships)
        activated_rules, consequent_strengths = self.evaluate_rules(memberships)
        fuzzy_score, _ = self.defuzzify_centroid(consequent_strengths)
        fuzzy_category = self.categorize_score(fuzzy_score)

        agreement = self.analyze_agreement(
            sanitized["dl_probability"], fuzzy_category, fuzzy_score
        )
        explanation = self.generate_explanation(
            dominant_categories, activated_rules, agreement, fuzzy_score, fuzzy_category
        )

        # Simplified rule list for lightweight API consumers
        activated_rule_ids = [r["rule_id"] for r in activated_rules]

        return {
            "deep_learning_probability": round(sanitized["dl_probability"], 4),
            "fuzzy_score": fuzzy_score,
            "fuzzy_category": fuzzy_category,
            "feature_analysis": {
                "average_pause_duration": dominant_categories["avg_pause_duration"],
                "intra_sentence_pauses": dominant_categories["intra_sentence_pauses"],
                "phonation_rate": dominant_categories["phonation_rate"],
                "jitter": dominant_categories["jitter"],
                "shimmer": dominant_categories["shimmer"],
            },
            "feature_raw_values": {
                "average_pause_duration": round(sanitized["avg_pause_duration"], 3),
                "intra_sentence_pauses": int(round(sanitized["intra_sentence_pauses"])),
                "phonation_rate": round(sanitized["phonation_rate"], 3),
                "jitter": round(sanitized["jitter"], 3),
                "shimmer": round(sanitized["shimmer"], 4),
            },
            "feature_memberships": memberships,
            "activated_rules": activated_rule_ids,
            "activated_rule_details": activated_rules,
            "agreement_analysis": agreement,
            "consequent_activations": {
                k: round(v, 4) for k, v in consequent_strengths.items()
            },
            "explanation": explanation,
            "assessment_type": "Research-only speech-pattern assessment",
            "non_clinical_disclaimer": (
                "Speech-pattern assessment derived from hybrid neural and fuzzy acoustic feature modeling. "
                "Not evaluated or authorized as a standalone clinical medical diagnostic instrument."
            ),
        }


# Global FIS singleton instance for quick reuse without re-initialization overhead
_default_fis: Optional[FuzzyInferenceSystem] = None


def get_fuzzy_inference_system() -> FuzzyInferenceSystem:
    """Returns the singleton instance of FuzzyInferenceSystem."""
    global _default_fis
    if _default_fis is None:
        _default_fis = FuzzyInferenceSystem()
    return _default_fis


def assess_speech_fuzzy(
    dl_probability: float,
    avg_pause_duration: float,
    intra_sentence_pauses: float,
    phonation_rate: float,
    jitter: float,
    shimmer: float,
) -> Dict[str, Any]:
    """Helper entry point for single evaluation."""
    fis = get_fuzzy_inference_system()
    inputs = {
        "dl_probability": dl_probability,
        "avg_pause_duration": avg_pause_duration,
        "intra_sentence_pauses": intra_sentence_pauses,
        "phonation_rate": phonation_rate,
        "jitter": jitter,
        "shimmer": shimmer,
    }
    return fis.compute_assessment(inputs)
