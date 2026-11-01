"""
Fuzzy Membership Functions and Rule Base Configuration.
All membership function parameter ranges are derived from the actual empirical
distributions of acoustic features and model probabilities in the dataset.
"""

from typing import Dict, List, Any, Callable
import numpy as np


def trimf(x: float, params: List[float]) -> float:
    """
    Triangular membership function with proper boundary handling.
    params: [a, b, c] where a <= b <= c
    """
    a, b, c = float(params[0]), float(params[1]), float(params[2])
    if x < a or x > c:
        return 0.0
    if a == b and x <= b:
        return 1.0
    if b == c and x >= b:
        return 1.0
    if a <= x <= b:
        return 1.0 if a == b else max(0.0, min(1.0, (x - a) / (b - a)))
    if b < x <= c:
        return 1.0 if b == c else max(0.0, min(1.0, (c - x) / (c - b)))
    return 0.0


def trapmf(x: float, params: List[float]) -> float:
    """
    Trapezoidal membership function with proper shoulder handling.
    params: [a, b, c, d] where a <= b <= c <= d
    """
    a, b, c, d = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a <= x < b:
        return 1.0 if a == b else max(0.0, min(1.0, (x - a) / (b - a)))
    if c < x <= d:
        return 1.0 if c == d else max(0.0, min(1.0, (d - x) / (d - c)))
    return 0.0


# Membership function parameter sets determined from dataset quantiles & distributions
MEMBERSHIP_PARAMS = {
    # 1. Deep Learning Prediction Probability: [0.0, 1.0]
    "dl_probability": {
        "domain": (0.0, 1.0),
        "terms": {
            "Low": {"fn": "trapmf", "params": [0.0, 0.0, 0.25, 0.45]},
            "Medium": {"fn": "trimf", "params": [0.30, 0.50, 0.70]},
            "High": {"fn": "trapmf", "params": [0.55, 0.75, 1.0, 1.0]},
        },
    },
    # 2. Average Pause Duration (seconds): empirical range [0.0, 1.14], domain [0.0, 1.50]
    "avg_pause_duration": {
        "domain": (0.0, 1.5),
        "terms": {
            "Short": {"fn": "trapmf", "params": [0.0, 0.0, 0.18, 0.32]},
            "Moderate": {"fn": "trimf", "params": [0.25, 0.42, 0.60]},
            "Long": {"fn": "trapmf", "params": [0.50, 0.70, 1.50, 1.50]},
        },
    },
    # 3. Intra-Sentence Pauses (count): empirical range [0, 55], domain [0, 60]
    "intra_sentence_pauses": {
        "domain": (0, 60),
        "terms": {
            "Few": {"fn": "trapmf", "params": [0.0, 0.0, 4.0, 10.0]},
            "Moderate": {"fn": "trimf", "params": [8.0, 18.0, 28.0]},
            "Frequent": {"fn": "trapmf", "params": [22.0, 35.0, 60.0, 60.0]},
        },
    },
    # 4. Phonation Rate (ratio of vocalized speech duration): empirical range [0.45, 1.00]
    "phonation_rate": {
        "domain": (0.0, 1.0),
        "terms": {
            "Low": {"fn": "trapmf", "params": [0.0, 0.0, 0.75, 0.84]},
            "Medium": {"fn": "trimf", "params": [0.80, 0.89, 0.95]},
            "High": {"fn": "trapmf", "params": [0.92, 0.97, 1.0, 1.0]},
        },
    },
    # 5. Jitter (frequency instability F0): empirical range [2.95, 8.32], domain [0, 10]
    "jitter": {
        "domain": (0.0, 10.0),
        "terms": {
            "Low": {"fn": "trapmf", "params": [0.0, 0.0, 3.8, 4.6]},
            "Medium": {"fn": "trimf", "params": [4.2, 5.2, 6.2]},
            "High": {"fn": "trapmf", "params": [5.8, 7.0, 10.0, 10.0]},
        },
    },
    # 6. Shimmer (amplitude variation): empirical range [0.0075, 0.085], domain [0, 0.10]
    "shimmer": {
        "domain": (0.0, 0.10),
        "terms": {
            "Low": {"fn": "trapmf", "params": [0.0, 0.0, 0.018, 0.026]},
            "Medium": {"fn": "trimf", "params": [0.022, 0.033, 0.045]},
            "High": {"fn": "trapmf", "params": [0.040, 0.055, 0.10, 0.10]},
        },
    },
    # Consequent: Speech-Pattern Concern Score [0, 100]
    "concern_score": {
        "domain": (0.0, 100.0),
        "terms": {
            "Low": {"fn": "trapmf", "params": [0.0, 0.0, 20.0, 35.0]},
            "Moderate": {"fn": "trimf", "params": [25.0, 50.0, 75.0]},
            "High": {"fn": "trapmf", "params": [65.0, 80.0, 100.0, 100.0]},
        },
    },
}


def evaluate_term_membership(variable_name: str, term_name: str, x: float) -> float:
    """Computes the membership degree of value x in the specified linguistic term."""
    var_cfg = MEMBERSHIP_PARAMS.get(variable_name)
    if not var_cfg:
        raise ValueError(f"Unknown linguistic variable: {variable_name}")
    term_cfg = var_cfg["terms"].get(term_name)
    if not term_cfg:
        raise ValueError(f"Unknown term '{term_name}' for variable '{variable_name}'")

    fn_name = term_cfg["fn"]
    params = term_cfg["params"]
    if fn_name == "trimf":
        return trimf(float(x), params)
    elif fn_name == "trapmf":
        return trapmf(float(x), params)
    else:
        raise ValueError(f"Unsupported membership function type: {fn_name}")


class FuzzyRule:
    """Represents a Mamdani IF-THEN rule with antecedent logic and research explanation."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        antecedent_evaluator: Callable[[Dict[str, Dict[str, float]]], float],
        consequent: str,
        research_context: str,
    ):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.antecedent_evaluator = antecedent_evaluator
        self.consequent = consequent  # "Low", "Moderate", "High"
        self.research_context = research_context

    def evaluate(self, memberships: Dict[str, Dict[str, float]]) -> float:
        """Returns the firing strength (activation weight) in [0, 1]."""
        strength = self.antecedent_evaluator(memberships)
        return float(np.clip(strength, 0.0, 1.0))


def build_rule_base() -> List[FuzzyRule]:
    """Constructs the research-validated Mamdani fuzzy rule set."""
    rules = [
        FuzzyRule(
            rule_id="Rule 1",
            name="DL-High & Frequent Pauses",
            description="IF DL Probability is HIGH AND Intra-Sentence Pauses is FREQUENT THEN Concern is HIGH",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["High"],
                m["intra_sentence_pauses"]["Frequent"],
            ),
            consequent="High",
            research_context="Frequent hesitations combined with high neural classification strongly correlate with speech disruption.",
        ),
        FuzzyRule(
            rule_id="Rule 2",
            name="DL-Medium & Moderate Pauses",
            description="IF DL Probability is MEDIUM AND Average Pause Duration is MODERATE THEN Concern is MODERATE",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["Medium"],
                m["avg_pause_duration"]["Moderate"],
            ),
            consequent="Moderate",
            research_context="Intermediate neural uncertainty paired with moderate pause length indicates borderline speech patterns.",
        ),
        FuzzyRule(
            rule_id="Rule 3",
            name="DL-Low & Few Pauses",
            description="IF DL Probability is LOW AND Intra-Sentence Pauses is FEW THEN Concern is LOW",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["Low"],
                m["intra_sentence_pauses"]["Few"],
            ),
            consequent="Low",
            research_context="Fluent sentence production with low neural detection probability reflects typical vocal fluency.",
        ),
        FuzzyRule(
            rule_id="Rule 4",
            name="Acoustic Instability (Jitter & Shimmer)",
            description="IF Jitter is HIGH AND Shimmer is HIGH THEN Concern is HIGH",
            antecedent_evaluator=lambda m: min(
                m["jitter"]["High"],
                m["shimmer"]["High"],
            ),
            consequent="High",
            research_context="Co-occurrence of pitch and amplitude micro-instability signals vocal fold dysregulation.",
        ),
        FuzzyRule(
            rule_id="Rule 5",
            name="DL-High & Low Phonation Rate",
            description="IF DL Probability is HIGH AND Phonation Rate is LOW THEN Concern is HIGH",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["High"],
                m["phonation_rate"]["Low"],
            ),
            consequent="High",
            research_context="Reduced voicing duration (slowed speech execution) corroborates elevated deep learning suspicion.",
        ),
        FuzzyRule(
            rule_id="Rule 6",
            name="DL-Medium & Acoustic Stability",
            description="IF DL Probability is MEDIUM AND Jitter is LOW AND Shimmer is LOW THEN Concern is MODERATE",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["Medium"],
                m["jitter"]["Low"],
                m["shimmer"]["Low"],
            ),
            consequent="Moderate",
            research_context="Preserved vocal stability tempers borderline neural classification toward balanced moderation.",
        ),
        FuzzyRule(
            rule_id="Rule 7",
            name="DL-Low, High Phonation & Short Pauses",
            description="IF DL Probability is LOW AND Phonation Rate is HIGH AND Average Pause Duration is SHORT THEN Concern is LOW",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["Low"],
                m["phonation_rate"]["High"],
                m["avg_pause_duration"]["Short"],
            ),
            consequent="Low",
            research_context="High vocal continuity and quick articulatory transitions indicate healthy speech prosody.",
        ),
        FuzzyRule(
            rule_id="Rule 8",
            name="DL-High & Severe Acoustic Hesitation",
            description="IF DL Probability is HIGH AND (Average Pause is LONG OR Intra-Sentence Pauses is FREQUENT) THEN Concern is HIGH",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["High"],
                max(
                    m["avg_pause_duration"]["Long"],
                    m["intra_sentence_pauses"]["Frequent"],
                ),
            ),
            consequent="High",
            research_context="High deep learning probability reinforced by prolonged silence or recurrent hesitation.",
        ),
        FuzzyRule(
            rule_id="Rule 9",
            name="DL-Low / Acoustic Discordance",
            description="IF DL Probability is LOW AND Average Pause Duration is LONG THEN Concern is MODERATE",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["Low"],
                m["avg_pause_duration"]["Long"],
            ),
            consequent="Moderate",
            research_context="Uncertainty detection: DL predicts control, yet vocal latency is extended. Fuzzy logic raises concern to Moderate.",
        ),
        FuzzyRule(
            rule_id="Rule 10",
            name="DL-High / Fluent Discordance",
            description="IF DL Probability is HIGH AND Average Pause Duration is SHORT AND Phonation Rate is HIGH THEN Concern is MODERATE",
            antecedent_evaluator=lambda m: min(
                m["dl_probability"]["High"],
                m["avg_pause_duration"]["Short"],
                m["phonation_rate"]["High"],
            ),
            consequent="Moderate",
            research_context="Uncertainty detection: DL indicates concern, but physical prosody is highly fluent. Fuzzy logic moderates the assessment.",
        ),
    ]
    return rules
