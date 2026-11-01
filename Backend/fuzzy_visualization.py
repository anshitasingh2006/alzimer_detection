"""
Generates publication-quality visualizations for Fuzzy Logic Membership Functions
and Defuzzification Curves. Saves figures to plots directory.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from fuzzy_rules import MEMBERSHIP_PARAMS, evaluate_term_membership
from fuzzy_logic import get_fuzzy_inference_system

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "Frontend", "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)


def plot_all_membership_functions(save_path: str = None) -> str:
    """Plots all 6 input membership functions and the output concern score."""
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "fuzzy_membership_functions.png")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    colors = {
        "Low": "#10b981",  # emerald
        "Medium": "#f59e0b",  # amber
        "Moderate": "#f59e0b",
        "High": "#ef4444",  # red
        "Short": "#10b981",
        "Long": "#ef4444",
        "Few": "#10b981",
        "Frequent": "#ef4444",
    }

    titles = {
        "dl_probability": "1. Deep Learning Prediction Probability",
        "avg_pause_duration": "2. Average Pause Duration (seconds)",
        "intra_sentence_pauses": "3. Intra-Sentence Pause Count",
        "phonation_rate": "4. Phonation Rate (Voicing Ratio)",
        "jitter": "5. Jitter (F0 Frequency Instability)",
        "shimmer": "6. Shimmer (Amplitude Instability)",
        "concern_score": "7. Output: Speech-Pattern Concern Score (0–100)",
    }

    vars_to_plot = [
        "dl_probability",
        "avg_pause_duration",
        "intra_sentence_pauses",
        "phonation_rate",
        "jitter",
        "shimmer",
        "concern_score",
    ]

    for idx, var_name in enumerate(vars_to_plot):
        ax = axes[idx]
        var_cfg = MEMBERSHIP_PARAMS[var_name]
        d_min, d_max = var_cfg["domain"]
        x_vals = np.linspace(d_min, d_max, 500)

        for term_name in var_cfg["terms"].keys():
            y_vals = [evaluate_term_membership(var_name, term_name, x) for x in x_vals]
            c = colors.get(term_name, "#6366f1")
            ax.plot(x_vals, y_vals, label=term_name, color=c, linewidth=2.2)
            ax.fill_between(x_vals, 0, y_vals, color=c, alpha=0.15)

        ax.set_title(titles[var_name], fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Membership Degree (μ)", fontsize=10)
        ax.set_xlabel("Feature Value", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)

    # Hide unused 8th subplot
    axes[7].axis("off")
    axes[7].text(
        0.1,
        0.5,
        "Mamdani Fuzzy Inference System\n"
        "• Fuzzification: Triangular & Trapezoidal\n"
        "• Conjunction (AND): Minimum (t-norm)\n"
        "• Disjunction (OR): Maximum (s-norm)\n"
        "• Implication & Aggregation: Max-Min\n"
        "• Defuzzification: Centroid Method\n"
        "• Domain: Alzheimer Voice Analysis",
        fontsize=12,
        linespacing=1.6,
        color="#334155",
        fontweight="medium",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"📊 Membership functions plot saved to: {save_path}")

    # Also save to Frontend public directory so React can display it directly
    public_copy = os.path.join(PUBLIC_DIR, "fuzzy_membership_functions.png")
    plt.savefig(public_copy, dpi=200)
    print(f"🖼️ Public asset saved to: {public_copy}")

    plt.close()
    return save_path


def plot_defuzzification_example(
    raw_inputs: dict, save_name: str = "fuzzy_defuzzification_example.png"
) -> str:
    """Plots the aggregated fuzzy curve and centroid marker for a given sample."""
    fis = get_fuzzy_inference_system()
    sanitized = fis.validate_inputs(raw_inputs)
    memberships = fis.fuzzify(sanitized)
    _, consequent_strengths = fis.evaluate_rules(memberships)
    centroid, aggregated = fis.defuzzify_centroid(consequent_strengths)

    fig, ax = plt.subplots(figsize=(10, 5))
    u = fis.output_universe

    ax.plot(u, aggregated, color="#8b5cf6", linewidth=2.5, label="Aggregated Fuzzy Set")
    ax.fill_between(u, 0, aggregated, color="#8b5cf6", alpha=0.25)

    ax.axvline(
        centroid,
        color="#ef4444",
        linestyle="--",
        linewidth=2.5,
        label=f"Centroid Score: {centroid:.1f}",
    )
    ax.scatter([centroid], [np.interp(centroid, u, aggregated)], color="#ef4444", s=100, zorder=5)

    ax.set_title("Mamdani Defuzzification (Centroid Method)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Speech-Pattern Concern Score [0–100]", fontsize=11)
    ax.set_ylabel("Aggregated Membership Degree (μ)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=11)

    save_path = os.path.join(PLOTS_DIR, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


if __name__ == "__main__":
    plot_all_membership_functions()
    sample = {
        "dl_probability": 0.85,
        "avg_pause_duration": 0.75,
        "intra_sentence_pauses": 35,
        "phonation_rate": 0.70,
        "jitter": 6.8,
        "shimmer": 0.05,
    }
    plot_defuzzification_example(sample)
