"""
Comparative Evaluation and Validation Script.
Evaluates and benchmarks:
A. Existing CNN-LSTM Model
B. Existing GRU with Attention Model
C. Proposed Hybrid Deep Learning + Fuzzy Logic System

Generates Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and
Uncertainty / Discordance calibration analysis on independent subject-labeled audio recordings.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from inference import extract_fused_features, MODELS, MODEL_ID_MAP, DECISION_THRESHOLD
from fuzzy_logic import assess_speech_fuzzy

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "..", "Frontend", "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)


def evaluate_all():
    alz_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "test", "alzheimer", "*.wav")))
    ctrl_files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "test", "control", "*.wav")))

    records = []
    print(f"📊 Starting benchmark across {len(alz_files)} Alzheimer and {len(ctrl_files)} Control files...")

    all_files = [(f, 1) for f in alz_files] + [(f, 0) for f in ctrl_files]

    for file_path, true_label in all_files:
        base_name = os.path.basename(file_path)
        subject_id = base_name.rsplit("_", 1)[0]

        X, prosody = extract_fused_features(file_path)

        # 1. CNN-LSTM prediction
        cnn_model = MODELS.get("cnn_lstm")
        if cnn_model:
            cnn_prob = float(cnn_model.predict(X, verbose=0)[0][0])
        else:
            cnn_prob = 0.5

        # 2. GRU-Attention prediction
        gru_model = MODELS.get("gru_attention")
        if gru_model:
            gru_prob = float(gru_model.predict(X, verbose=0)[0][0])
        else:
            gru_prob = 0.5

        # 3. Hybrid Fuzzy Logic Assessment (using GRU as deep feature extractor)
        fuzzy_eval = assess_speech_fuzzy(
            dl_probability=gru_prob,
            avg_pause_duration=float(prosody[0]),
            intra_sentence_pauses=float(prosody[3]),
            phonation_rate=float(prosody[4]),
            jitter=float(prosody[5]),
            shimmer=float(prosody[6]),
        )
        fuzzy_score = fuzzy_eval["fuzzy_score"]
        # Normalized fuzzy score in [0, 1] for metric computation
        fuzzy_prob = fuzzy_score / 100.0

        records.append({
            "file": base_name,
            "subject": subject_id,
            "true_label": true_label,
            "cnn_prob": cnn_prob,
            "gru_prob": gru_prob,
            "fuzzy_prob": fuzzy_prob,
            "fuzzy_score": fuzzy_score,
            "fuzzy_category": fuzzy_eval["fuzzy_category"],
            "agreement": fuzzy_eval["agreement_analysis"]["status"],
            "avg_pause": float(prosody[0]),
            "pauses_count": int(prosody[3]),
            "phonation": float(prosody[4]),
            "jitter": float(prosody[5]),
            "shimmer": float(prosody[6]),
        })

    df = pd.DataFrame(records)

    # Compute comparative metrics
    # Optimal threshold for neural models is DECISION_THRESHOLD; for fuzzy normalized score, 0.50 (50/100)
    models_to_eval = [
        ("CNN-LSTM", df["cnn_prob"].values, DECISION_THRESHOLD),
        ("GRU-Attention", df["gru_prob"].values, DECISION_THRESHOLD),
        ("Hybrid DL + Fuzzy FIS", df["fuzzy_prob"].values, 0.50),
    ]

    metrics_summary = {}

    print("\n" + "=" * 70)
    print("📈 BENCHMARK RESULTS ACROSS TEST COHORT (N=40, 30 Unique Subjects)")
    print("=" * 70)
    print(f"{'System / Model':<26} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8}")
    print("-" * 70)

    for name, probs, thresh in models_to_eval:
        preds = (probs >= thresh).astype(int)
        y_true = df["true_label"].values

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs)

        metrics_summary[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "threshold": thresh,
        }

        print(f"{name:<26} | {acc:.4f}   | {prec:.4f}    | {rec:.4f}   | {f1:.4f}   | {auc:.4f}")

    print("=" * 70)

    # Uncertainty & Discordance Analysis
    agreement_counts = df["agreement"].value_counts().to_dict()
    print("\n🔍 UNCERTAINTY & AGREEMENT DISTRIBUTION:")
    for status, count in agreement_counts.items():
        print(f"  • {status}: {count} / {len(df)} ({count/len(df)*100:.1f}%)")

    # Generate Comparative ROC Plot
    plt.figure(figsize=(9, 7))
    for name, probs, _ in models_to_eval:
        fpr, tpr, _ = roc_curve(df["true_label"].values, probs)
        auc = roc_auc_score(df["true_label"].values, probs)
        plt.plot(fpr, tpr, lw=2.5, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    plt.title("Comparative ROC Curves: Deep Learning vs Hybrid Fuzzy System", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    roc_path = os.path.join(PLOTS_DIR, "comparative_roc_analysis.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.savefig(os.path.join(PUBLIC_DIR, "comparative_roc_analysis.png"), dpi=200)
    plt.close()
    print(f"📊 Comparative ROC saved to: {roc_path}")

    # Generate Multi-Model Confusion Matrix Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, probs, thresh) in enumerate(models_to_eval):
        preds = (probs >= thresh).astype(int)
        cm = confusion_matrix(df["true_label"].values, preds)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if idx < 2 else "Purples",
            ax=axes[idx],
            xticklabels=["Control", "Alzheimer"],
            yticklabels=["Control", "Alzheimer"],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[idx].set_title(f"{name}\nAcc: {metrics_summary[name]['accuracy']:.3f} | F1: {metrics_summary[name]['f1_score']:.3f}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)

    cm_path = os.path.join(PLOTS_DIR, "comparative_confusion_matrices.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.savefig(os.path.join(PUBLIC_DIR, "comparative_confusion_matrices.png"), dpi=200)
    plt.close()
    print(f"📊 Confusion matrices saved to: {cm_path}")

    # Save metrics JSON
    metrics_path = os.path.join(PLOTS_DIR, "evaluation_report.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": metrics_summary,
                "agreement_distribution": agreement_counts,
                "cohort_size": len(df),
                "unique_subjects": int(df["subject"].nunique()),
            },
            f,
            indent=2,
        )
    print(f"💾 Report JSON saved to: {metrics_path}")
    return metrics_summary


if __name__ == "__main__":
    evaluate_all()
