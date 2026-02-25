import os

import matplotlib
matplotlib.use("Agg")  # headless — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _results_path(filename):
    return os.path.join(RESULTS_DIR, filename)


def plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    path = _results_path(filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curve(y_test, y_scores_ae_old, y_scores_ae_new, y_scores_ae_combined, y_scores_baseline):
    fpr_old, tpr_old, _ = roc_curve(y_test, y_scores_ae_old)
    fpr_new, tpr_new, _ = roc_curve(y_test, y_scores_ae_new)
    fpr_comb, tpr_comb, _ = roc_curve(y_test, y_scores_ae_combined)
    fpr_bl,  tpr_bl,  _ = roc_curve(y_test, y_scores_baseline)
    auc_old  = roc_auc_score(y_test, y_scores_ae_old)
    auc_new  = roc_auc_score(y_test, y_scores_ae_new)
    auc_comb = roc_auc_score(y_test, y_scores_ae_combined)
    auc_bl   = roc_auc_score(y_test, y_scores_baseline)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_old,  tpr_old,  label=f"AE+XGB old      (AUC={auc_old:.4f})", linewidth=2)
    ax.plot(fpr_new,  tpr_new,  label=f"AE+XGB new      (AUC={auc_new:.4f})", linewidth=2)
    ax.plot(fpr_comb, tpr_comb, label=f"AE+XGB combined (AUC={auc_comb:.4f})", linewidth=2)
    ax.plot(fpr_bl,   tpr_bl,   label=f"Baseline        (AUC={auc_bl:.4f})",  linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    path = _results_path("roc_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pr_curve(y_test, y_scores_ae_old, y_scores_ae_new, y_scores_ae_combined, y_scores_baseline):
    prec_old,  rec_old,  _ = precision_recall_curve(y_test, y_scores_ae_old)
    prec_new,  rec_new,  _ = precision_recall_curve(y_test, y_scores_ae_new)
    prec_comb, rec_comb, _ = precision_recall_curve(y_test, y_scores_ae_combined)
    prec_bl,   rec_bl,   _ = precision_recall_curve(y_test, y_scores_baseline)
    ap_old  = average_precision_score(y_test, y_scores_ae_old)
    ap_new  = average_precision_score(y_test, y_scores_ae_new)
    ap_comb = average_precision_score(y_test, y_scores_ae_combined)
    ap_bl   = average_precision_score(y_test, y_scores_baseline)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_old,  prec_old,  label=f"AE+XGB old      (AP={ap_old:.4f})", linewidth=2)
    ax.plot(rec_new,  prec_new,  label=f"AE+XGB new      (AP={ap_new:.4f})", linewidth=2)
    ax.plot(rec_comb, prec_comb, label=f"AE+XGB combined (AP={ap_comb:.4f})", linewidth=2)
    ax.plot(rec_bl,   prec_bl,   label=f"Baseline        (AP={ap_bl:.4f})",  linewidth=2, linestyle="--")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    path = _results_path("pr_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reconstruction_error_dist(errors_eval, y_eval_pool):
    # Reduce (n, 30) per-feature errors to scalar MSE per sample for plotting
    scalar_errors = errors_eval.mean(axis=1)
    normal_errors = scalar_errors[y_eval_pool == 0]
    fraud_errors = scalar_errors[y_eval_pool == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal_errors, bins=100, alpha=0.6, color="steelblue", label="Normal", density=True)
    ax.hist(fraud_errors, bins=100, alpha=0.6, color="crimson", label="Fraud", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Reconstruction Error (MSE, log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    fig.tight_layout()
    path = _results_path("reconstruction_error_dist.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_metrics_table(y_test, y_scores_ae_old, y_pred_ae_old, y_pred_ae_new, y_scores_ae_new, y_pred_ae_combined, y_scores_ae_combined, y_pred_baseline, y_scores_baseline):
    def metrics(y_true, y_pred, y_scores):
        return {
            "Accuracy":  accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
            "AUC":       roc_auc_score(y_true, y_scores),
        }

    old_m  = metrics(y_test, y_pred_ae_old,      y_scores_ae_old)
    new_m  = metrics(y_test, y_pred_ae_new,      y_scores_ae_new)
    comb_m = metrics(y_test, y_pred_ae_combined, y_scores_ae_combined)
    bl_m   = metrics(y_test, y_pred_baseline,    y_scores_baseline)

    header = f"{'Metric':<12} {'AE+XGB(old)':>12} {'AE+XGB(new)':>12} {'AE+XGB(comb)':>14} {'Baseline':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for key in old_m:
        print(f"  {key:<10} {old_m[key]:>12.4f} {new_m[key]:>12.4f} {comb_m[key]:>14.4f} {bl_m[key]:>10.4f}")
    print(sep)


def run_evaluation(
    y_test,
    y_pred_ae_old,
    y_scores_ae_old,
    y_pred_ae_new,
    y_scores_ae_new,
    y_pred_ae_combined,
    y_scores_ae_combined,
    y_pred_baseline,
    y_scores_baseline,
    errors_eval,
    y_eval_pool,
):
    """Master orchestrator: generate all plots and print metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n  [Confusion matrices]")
    plot_confusion_matrix(y_test, y_pred_ae_old,      "Confusion Matrix — AE+XGB (old)",      "confusion_matrix_ae.png")
    plot_confusion_matrix(y_test, y_pred_ae_new,      "Confusion Matrix — AE+XGB (new)",      "confusion_matrix_ae_new.png")
    plot_confusion_matrix(y_test, y_pred_ae_combined, "Confusion Matrix — AE+XGB (combined)", "confusion_matrix_ae_combined.png")
    plot_confusion_matrix(y_test, y_pred_baseline,    "Confusion Matrix — Baseline XGBoost",  "confusion_matrix_baseline.png")

    print("\n  [ROC curve]")
    plot_roc_curve(y_test, y_scores_ae_old, y_scores_ae_new, y_scores_ae_combined, y_scores_baseline)

    print("\n  [PR curve]")
    plot_pr_curve(y_test, y_scores_ae_old, y_scores_ae_new, y_scores_ae_combined, y_scores_baseline)

    print("\n  [Reconstruction error distribution]")
    plot_reconstruction_error_dist(errors_eval, y_eval_pool)

    print("\n  [Metrics]")
    print_metrics_table(y_test, y_scores_ae_old, y_pred_ae_old, y_pred_ae_new, y_scores_ae_new, y_pred_ae_combined, y_scores_ae_combined, y_pred_baseline, y_scores_baseline)
