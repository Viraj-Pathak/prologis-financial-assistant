"""
Generate visualization plots for both ML models.

Outputs to ml/plots/:
  regression_metrics.png           - bar chart of RMSE / MAE + R² annotation
  regression_feature_importance.png - horizontal bar chart of RF feature importances
  classification_metrics.png        - accuracy/precision/recall/F1 bars
  classification_confusion_matrix.png - heatmap of TP/TN/FP/FN

Run: python scripts/generate_plots.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
REG_METRICS = ROOT / "ml" / "regression" / "metrics.json"
CLF_METRICS = ROOT / "ml" / "classification" / "metrics.json"
PLOTS_DIR   = ROOT / "ml" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_regression_metrics(metrics: dict):
    rmse_usd = metrics["rmse"] * 100000
    mae_usd  = metrics["mae"]  * 100000
    r2       = metrics["r2"]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(2)
    bars = ax1.bar(x, [rmse_usd, mae_usd], color=["#3b82f6", "#60a5fa"], width=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["RMSE (USD)", "MAE (USD)"])
    ax1.set_ylabel("Error (USD)", fontsize=11)
    ax1.set_ylim(0, max(rmse_usd, mae_usd) * 1.25)
    for bar, v in zip(bars, [rmse_usd, mae_usd]):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1500, f"${v:,.0f}",
                 ha="center", fontsize=10, fontweight="bold")
    ax1.text(
        1.7, max(rmse_usd, mae_usd) * 0.65,
        f"R² Score: {r2:.4f}",
        fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#dcfce7", edgecolor="#16a34a"),
    )
    plt.title("Random Forest Regression — Test Set Performance", fontsize=12, fontweight="bold")
    plt.suptitle("California Housing Dataset", fontsize=10, y=0.98, color="#666")
    plt.tight_layout()
    out = PLOTS_DIR / "regression_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_regression_feature_importance(metrics: dict):
    fi = metrics["feature_importance"]
    pairs = sorted(fi.items(), key=lambda kv: kv[1])
    features    = [p[0] for p in pairs]
    importances = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars = ax.barh(features, importances, color=colors)
    for bar, imp in zip(bars, importances):
        ax.text(imp + 0.005, bar.get_y() + bar.get_height()/2,
                f"{imp:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title("Random Forest — Feature Importances", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(importances) * 1.18)
    plt.tight_layout()
    out = PLOTS_DIR / "regression_feature_importance.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_classification_metrics(metrics: dict):
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    colors = ["#16a34a", "#3b82f6", "#a855f7", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Logistic Regression — Test Set Performance", fontsize=12, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random baseline")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    plt.suptitle("UCI Bank Marketing Dataset", fontsize=10, y=0.98, color="#666")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = PLOTS_DIR / "classification_metrics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_classification_confusion(metrics: dict):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: No", "Predicted: Yes"])
    ax.set_yticklabels(["Actual: No", "Actual: Yes"])
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}", ha="center", va="center",
                    color=color, fontsize=14, fontweight="bold")
    ax.set_title("Logistic Regression — Confusion Matrix", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = PLOTS_DIR / "classification_confusion_matrix.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    if not REG_METRICS.exists():
        raise SystemExit("ERROR: Run `python ml/regression/train.py` first.")
    if not CLF_METRICS.exists():
        raise SystemExit("ERROR: Run `python ml/classification/train.py` first.")

    reg = json.loads(REG_METRICS.read_text())
    clf = json.loads(CLF_METRICS.read_text())

    print("Generating regression plots...")
    plot_regression_metrics(reg)
    plot_regression_feature_importance(reg)
    print("Generating classification plots...")
    plot_classification_metrics(clf)
    plot_classification_confusion(clf)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
