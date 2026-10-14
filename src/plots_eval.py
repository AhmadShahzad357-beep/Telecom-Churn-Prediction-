"""Figures for Step 5 - same style as the rest of the project (no grid, legend above, own colour per series)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc

from src import eda

eda.set_style()
FOLD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d55181"]
TRAIN_C, CV_C, TEST_C = eda.SINGLE["violet"], eda.SINGLE["aqua"], eda.SINGLE["yellow"]


def _legend_above(ax, ncol, **kw):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=ncol, fontsize=9, **kw)


def plot_cv_roc(folds):
    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, 200)
    tprs = []
    for i, (y, p) in enumerate(folds):
        fpr, tpr, _ = roc_curve(y, p)
        tprs.append(np.interp(grid, fpr, tpr))
        ax.plot(fpr, tpr, lw=1.3, color=FOLD_COLORS[i], alpha=0.85, label=f"Fold {i + 1} (AUC {auc(fpr, tpr):.3f})")
    mean = np.mean(tprs, axis=0)
    ax.plot(grid, mean, lw=2.8, color=eda.TEXT, label=f"Mean (AUC {auc(grid, mean):.3f})")
    ax.plot([0, 1], [0, 1], color=eda.MUTED, lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right", fontsize=9)          # empty corner of a ROC plot - no overlap
    ax.set_title("ROC curve for every CV fold (train, out-of-fold)")
    return fig


def plot_cv_pr(folds, base_rate):
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (y, p) in enumerate(folds):
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, lw=1.5, color=FOLD_COLORS[i], label=f"Fold {i + 1} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.axhline(base_rate, color=eda.MUTED, lw=1, label=f"Random guess ({base_rate:.3f})")
    ax.set_xlabel("Recall (share of churners caught)")
    ax.set_ylabel("Precision (share of alerts that are real churners)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)           # empty corner of a PR plot - no overlap
    ax.set_title("Precision-recall curve for every CV fold (train, out-of-fold)")
    return fig


def plot_score_distribution(y, p, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 41)
    for ax, cls, color, name in ((axes[0], 0, eda.NON_CHURN, "Stayed (non-churn)"),
                                 (axes[1], 1, eda.CHURN, "Churned")):
        vals = p[y == cls]
        ax.hist(vals, bins=bins, color=color, label=f"{name}  n={len(vals):,}")
        ax.axvline(np.median(vals), color=eda.TEXT, lw=1.2, label=f"Median score {np.median(vals):.2f}")
        ax.set_xlabel("Predicted churn score")
        ax.set_ylabel("Customers")
        ax.legend(loc="upper center", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_threshold_tradeoff(curve: pd.DataFrame, chosen: dict):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for col, color, label in (("precision", eda.SINGLE["violet"], "Precision"),
                              ("recall", eda.SINGLE["aqua"], "Recall"),
                              ("f1", eda.SINGLE["magenta"], "F1")):
        ax.plot(curve["threshold"], curve[col], lw=2.2, color=color, label=label)
    for key, color, label in (("f1", eda.SINGLE["magenta"], "Best F1"), ("profit", eda.SINGLE["yellow"], "Best profit")):
        ax.axvline(chosen[key], color=color, lw=1.2, ls="--", label=f"{label} threshold {chosen[key]:.2f}")
    ax.set_xlabel("Decision threshold on the model score")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    _legend_above(ax, 5)
    ax.set_title("Threshold trade-off on train (out-of-fold): thresholds are chosen here, never on test", pad=30)
    return fig


def plot_test_curves(y, p, base_rate):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=2.4, color=TEST_C, label=f"Model (ROC-AUC {auc(fpr, tpr):.3f})")
    axes[0].plot([0, 1], [0, 1], color=eda.MUTED, lw=1, label="Random guess")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (recall)")
    axes[0].set_title("ROC curve", fontsize=11, pad=30)
    _legend_above(axes[0], 2)
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=2.4, color=eda.SINGLE["green"], label=f"Model (PR-AUC {average_precision_score(y, p):.3f})")
    axes[1].axhline(base_rate, color=eda.MUTED, lw=1, label=f"Random guess ({base_rate:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Precision-recall curve", fontsize=11, pad=30)
    _legend_above(axes[1], 2)
    fig.suptitle("Final model on the unseen TEST set", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_confusion(cm, threshold):
    labels = np.array([["True negative", "False positive"], ["False negative", "True positive"]])
    colors = np.array([[eda.NON_CHURN, eda.SINGLE["yellow"]], [eda.SINGLE["magenta"], eda.SINGLE["green"]]])
    total = cm.sum()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1 - i), 0.97, 0.97, color=colors[i, j], alpha=0.9))
            ax.text(j + 0.485, 1 - i + 0.56, f"{cm[i, j]:,}", ha="center", va="center", fontsize=20,
                    fontweight="bold", color="white")
            ax.text(j + 0.485, 1 - i + 0.33, f"{labels[i, j]}\n{cm[i, j] / total:.1%} of customers",
                    ha="center", va="center", fontsize=9.5, color="white")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.485, 1.485], ["Predicted: stays", "Predicted: churns"])
    ax.set_yticks([1.485, 0.485], ["Actually stayed", "Actually churned"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.ravel()]
    ax.legend(handles, list(labels.ravel()), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, fontsize=8.5)
    ax.set_title(f"Confusion matrix on TEST at threshold {threshold:.2f}", pad=30)
    return fig


def plot_calibration(y, raw, cal, bins=10):
    def table(p):
        q = pd.qcut(p, bins, duplicates="drop")
        return pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color=eda.MUTED, lw=1, label="Perfect calibration")
    for p, color, label in ((raw, eda.SINGLE["magenta"], "Before calibration"),
                            (cal, eda.SINGLE["green"], "After calibration (isotonic)")):
        t = table(p)
        ax.plot(t["p"], t["y"], marker="o", ms=7, lw=2, color=color, label=label)
    ax.set_xlabel("Predicted churn probability (decile average)")
    ax.set_ylabel("Actual churn rate in that decile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)           # top-left corner is always empty here
    ax.set_title("Calibration on TEST: does '70%' really mean 70%?")
    return fig


def plot_gains_lift(y, p):
    order = np.argsort(-p)
    ys = np.asarray(y)[order]
    frac = np.arange(1, len(ys) + 1) / len(ys)
    gains = np.cumsum(ys) / ys.sum()
    deciles = pd.qcut(pd.Series(-p).rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    rate = pd.Series(np.asarray(y)).groupby(deciles.values, observed=True).mean()
    lift = rate / np.mean(y)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].plot(frac * 100, gains * 100, lw=2.4, color=TEST_C, label="Model")
    axes[0].plot([0, 100], [0, 100], color=eda.MUTED, lw=1, label="Random targeting")
    for pct in (10, 20):
        g = gains[int(len(gains) * pct / 100) - 1] * 100
        axes[0].scatter([pct], [g], color=eda.TEXT, zorder=3, s=30)
        axes[0].annotate(f"Top {pct}% -> {g:.0f}% of churners", (pct, g), xytext=(10, -14),
                         textcoords="offset points", fontsize=9)
    axes[0].set_xlabel("Customers contacted, ranked by risk (%)")
    axes[0].set_ylabel("Churners caught (%)")
    axes[0].set_title("Cumulative gains", fontsize=11, pad=30)
    _legend_above(axes[0], 2)
    colors = eda.ramp(eda.SINGLE["violet"], 10)[::-1]
    bars = axes[1].bar(rate.index.astype(str), lift.values, color=colors, width=0.65)
    for i, v in enumerate(lift.values):
        axes[1].text(i, v + 0.15, f"{v:.1f}x", ha="center", fontsize=8.5, color=eda.TEXT)
    axes[1].axhline(1, color=eda.MUTED, lw=1)
    axes[1].set_xlabel("Risk decile (D1 = highest model score)")
    axes[1].set_ylabel("Lift vs average churn rate")
    axes[1].set_title("Lift by decile", fontsize=11, pad=30)
    axes[1].legend(list(bars), list(rate.index.astype(str)), loc="lower left", bbox_to_anchor=(0, 1.01),
                   ncol=10, fontsize=7.5, handlelength=0.9, columnspacing=0.6)
    fig.suptitle("Targeting power on the unseen TEST set", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def plot_profit(curve: pd.DataFrame, chosen_frac: float, assumptions: str):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(curve["contacted_pct"], curve["net_rs"] / 1e3, lw=2.4, color=eda.SINGLE["green"], label="Net value of campaign")
    ax.axhline(0, color=eda.MUTED, lw=1, label="Break-even")
    best = curve.loc[curve["net_rs"].idxmax()]
    ax.scatter([best["contacted_pct"]], [best["net_rs"] / 1e3], color=eda.TEXT, zorder=3, s=40,
               label=f"Best possible on test: contact {best['contacted_pct']:.1f}%")
    ax.axvline(chosen_frac * 100, color=eda.SINGLE["yellow"], lw=1.4, ls="--",
               label=f"Threshold chosen on train: contact {chosen_frac * 100:.1f}%")
    ax.set_xlabel("Customers contacted, ranked by risk (%)")
    ax.set_ylabel("Net value (Rs thousand)")
    _legend_above(ax, 2)
    ax.set_title("Retention campaign value on TEST", pad=48)
    ax.text(0.99, 0.97, assumptions, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=eda.MUTED)
    return fig


def plot_risk_tiers(tiers: pd.DataFrame):
    colors = eda.ramp(eda.CHURN, len(tiers))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    specs = (("customers", "Customers", "{:,.0f}"), ("actual_churn_rate", "Actual churn rate", "{:.1%}"),
             ("revenue_at_risk_rs", "Monthly revenue at risk (Rs)", "Rs {:,.0f}"))
    for ax, (col, title, fmt) in zip(axes, specs):
        vals = tiers[col].values
        bars = ax.bar(tiers["tier"], vals, color=colors, width=0.62)
        for i, v in enumerate(vals):
            ax.text(i, v + vals.max() * 0.02, fmt.format(v), ha="center", fontsize=8.5, color=eda.TEXT)
        ax.set_ylim(0, vals.max() * 1.18)
        ax.set_title(title, fontsize=11, pad=30)
        ax.legend(list(bars), list(tiers["tier"]), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4,
                  fontsize=8, handlelength=1)
    fig.suptitle("Risk tiers on TEST (calibrated probability) - where the revenue at risk sits",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


def plot_train_cv_test(rows: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = [TRAIN_C, CV_C, TEST_C]
    for ax, metric, title in ((axes[0], "pr_auc", "PR-AUC"), (axes[1], "roc_auc", "ROC-AUC")):
        bars = ax.bar(rows["set"], rows[metric], color=colors, width=0.55)
        for i, v in enumerate(rows[metric]):
            ax.text(i, v / 2, f"{v:.3f}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11, pad=30)
        ax.legend(list(bars), list(rows["set"]), loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=9)
    fig.suptitle("Overfitting check: train vs cross-validation vs unseen test",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig